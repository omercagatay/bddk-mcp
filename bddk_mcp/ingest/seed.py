"""Build the corpus file from seed_data/ JSON, or export it back.

``seed_data/documents.json`` and ``seed_data/decision_cache.json`` are the
reviewed, git-tracked source of the corpus.  ``bddk-mcp build`` turns them into
one SQLite file with sections, chunks and embeddings; ``bddk-mcp export``
writes the file back to the same JSON shape.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path

from bddk_mcp.store.corpus_db import CorpusDB
from bddk_mcp.store.doc_store import DocumentStore, StoredDocument, upsert_document_sync

logger = logging.getLogger(__name__)

SEED_DIR = Path(__file__).resolve().parents[2] / "seed_data"
_DOCUMENT_COLUMNS = (
    "document_id, title, category, decision_date, decision_number, source_url, markdown_content, "
    "content_hash, downloaded_at, extracted_at, extraction_method, total_pages, file_size"
)
_CATALOG_COLUMNS = "document_id, title, content, decision_date, decision_number, category, source_url"


def _load_records(path: Path) -> list[dict]:
    """Read one JSON seed artifact; a missing file is an empty artifact."""

    if not path.exists():
        return []
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Seed artifact is not valid JSON: {path.name}.") from exc
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise RuntimeError(f"Seed artifact must contain object records: {path.name}.")
    return records


def _validate_seed_documents(docs: list[dict]) -> dict[str, str]:
    """Validate canonical seed documents before model or corpus work."""

    if not isinstance(docs, list) or any(not isinstance(item, dict) for item in docs):
        raise RuntimeError("Seed payload contains an invalid record.")
    doc_ids = [item.get("document_id") for item in docs]
    if any(not isinstance(doc_id, str) or not doc_id for doc_id in doc_ids) or len(set(doc_ids)) != len(doc_ids):
        raise RuntimeError("Seed document identities are missing or duplicated.")

    doc_hashes: dict[str, str] = {}
    for document in docs:
        content = document.get("markdown_content", "")
        content_hash = document.get("content_hash", "")
        if not isinstance(content, str) or hashlib.sha256(content.encode()).hexdigest() != content_hash:
            raise RuntimeError("Seed document content hash validation failed.")
        doc_hashes[document["document_id"]] = content_hash

    return doc_hashes


def _strip_docs_dump_header(text: str) -> str:
    """Return the body of a docs_dump-style markdown file.

    docs_dump files follow the shape ``# Title\n- Document ID: ...\n---\n<body>``.
    Split on the first ``\\n---\\n`` so only the article body is retained. Returns
    ``text`` unchanged when no separator is present.
    """
    parts = text.split("\n---\n", 1)
    return parts[1].lstrip() if len(parts) == 2 else text


def _seed_document(record: dict) -> StoredDocument:
    return StoredDocument(
        document_id=record["document_id"],
        title=record.get("title", ""),
        category=record.get("category", ""),
        decision_date=record.get("decision_date", ""),
        decision_number=record.get("decision_number", ""),
        source_url=record.get("source_url", "") or "",
        markdown_content=record.get("markdown_content", ""),
        content_hash=record["content_hash"],
        extraction_method=record.get("extraction_method", "markitdown"),
        total_pages=record.get("total_pages", 1),
        file_size=record.get("file_size", 0),
    )


# -- Export -------------------------------------------------------------------


async def export_corpus(corpus_path: str | None = None, seed_dir: str | None = None) -> tuple[int, int]:
    """Write the corpus documents and catalog to seed JSON. Returns (documents, catalog)."""
    target = Path(seed_dir) if seed_dir else SEED_DIR
    target.mkdir(parents=True, exist_ok=True)
    db = CorpusDB.open(corpus_path, readonly=True)
    try:
        docs = await db.run(
            lambda conn: [
                dict(row)
                for row in conn.execute(f"SELECT {_DOCUMENT_COLUMNS} FROM documents ORDER BY document_id").fetchall()
            ]
        )
        catalog = await db.run(
            lambda conn: [
                dict(row)
                for row in conn.execute(
                    f"SELECT {_CATALOG_COLUMNS} FROM decision_cache ORDER BY document_id"
                ).fetchall()
            ]
        )
    finally:
        db.close()
    (target / "documents.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    (target / "decision_cache.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Exported %d documents and %d catalog entries to %s", len(docs), len(catalog), target)
    return len(docs), len(catalog)


# -- Build --------------------------------------------------------------------


async def build_corpus(
    corpus_path: str | None = None,
    seed_dir: str | None = None,
    *,
    force: bool = False,
    vector_store_factory=None,
) -> dict:
    """Load seed JSON into the corpus file and embed every document that needs it.

    Unchanged documents keep their rows, sections and chunks; changed or new
    documents are rewritten (the previous body is archived as a version) and
    re-embedded.  ``force`` rewrites and re-embeds everything.  ``vector_store_factory``
    (``db -> VectorStore``) is a test seam for a fake embedder.
    """
    from bddk_mcp.store.vector_store import VectorStore

    seed = Path(seed_dir) if seed_dir else SEED_DIR
    docs = _load_records(seed / "documents.json")
    catalog = _load_records(seed / "decision_cache.json")
    seed_hashes = _validate_seed_documents(docs)
    result = {"decision_cache": 0, "documents": 0, "sections": 0, "chunks": 0, "skipped": False}
    if not docs and not catalog:
        logger.info("Seed files are empty; nothing to build")
        return result

    db = CorpusDB.open(corpus_path)
    try:
        store = DocumentStore(db)
        vs = (vector_store_factory or VectorStore)(db)

        def _stored_state(conn: sqlite3.Connection) -> tuple[dict[str, str], set[str], int]:
            hashes = {
                row["document_id"]: row["content_hash"]
                for row in conn.execute("SELECT document_id, content_hash FROM documents").fetchall()
            }
            indexed = {
                row["doc_id"]
                for row in conn.execute(
                    "SELECT DISTINCT c.doc_id FROM chunks AS c JOIN documents AS d "
                    "ON d.document_id = c.doc_id AND d.content_hash = c.content_hash"
                ).fetchall()
            }
            catalog_count = conn.execute("SELECT COUNT(*) FROM decision_cache").fetchone()[0]
            return hashes, indexed, catalog_count

        stored_hashes, indexed, catalog_count = await db.run(_stored_state)
        stale = [doc_id for doc_id, digest in seed_hashes.items() if stored_hashes.get(doc_id) != digest]
        unindexed = [doc_id for doc_id, record in seed_hashes.items() if doc_id not in indexed and record]
        if not force and not stale and not unindexed and catalog_count == len(catalog):
            logger.info("Corpus already matches the seed data; build skipped")
            result["skipped"] = True
            return result

        if catalog:
            result["decision_cache"] = await store.replace_catalog(catalog)

        rewrite = list(seed_hashes) if force else stale
        now = time.time()

        def _write_documents(conn: sqlite3.Connection) -> int:
            sections = 0
            for record in docs:
                if record["document_id"] in rewrite:
                    sections += upsert_document_sync(conn, _seed_document(record), now=now, prepare=False)[1]
            return sections

        result["sections"] = await db.transaction(_write_documents)
        result["documents"] = len(rewrite)

        to_embed = sorted(set(rewrite) | set(unindexed))
        by_id = {record["document_id"]: record for record in docs}
        for doc_id in to_embed:
            record = by_id[doc_id]
            if not record.get("markdown_content"):
                continue
            result["chunks"] += await vs.add_document(
                doc_id=doc_id,
                title=record.get("title", "") or doc_id,
                content=record["markdown_content"],
                category=record.get("category", ""),
                decision_date=record.get("decision_date", ""),
                decision_number=record.get("decision_number", ""),
                source_url=record.get("source_url", "") or "",
            )
        logger.info(
            "Corpus built (%d catalog, %d documents rewritten, %d sections, %d chunks embedded)",
            result["decision_cache"],
            result["documents"],
            result["sections"],
            result["chunks"],
        )
    finally:
        db.close()
    return result
