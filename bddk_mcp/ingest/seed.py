"""Build the corpus file from seed_data/ JSON, or export it back.

``seed_data/documents.json`` and ``seed_data/decision_cache.json`` are the
reviewed, git-tracked source of the corpus.  ``bddk-mcp build`` turns them into
one SQLite file with sections, chunks and embeddings; ``bddk-mcp export``
writes the file back to the same JSON shape and exports the chunk vectors to
``seed_data/embeddings/`` (one float16 ``.npy`` per document plus a
``manifest.json`` naming the model and a digest of each document's chunk
texts).  A later ``build`` reuses those vectors for every document whose chunk
texts and model still match, so an unchanged seed builds without loading the
embedding model.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
import time
from pathlib import Path

import numpy as np

from bddk_mcp.core import config as _config
from bddk_mcp.core.config import CORPUS_PATH, EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME
from bddk_mcp.store import fingerprint as build_fingerprint
from bddk_mcp.store.corpus_db import CorpusDB
from bddk_mcp.store.doc_store import DocumentStore, StoredDocument, upsert_document_sync

logger = logging.getLogger(__name__)

SEED_DIR = Path(__file__).resolve().parents[2] / "seed_data"
EMBEDDINGS_SUBDIR = "embeddings"
_SAFE_DOC_ID = re.compile(r"[A-Za-z0-9_.-]+")
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
        downloaded_at=record.get("downloaded_at"),
        extracted_at=record.get("extracted_at"),
    )


def _write_canonical_json(path: Path, records: list[dict]) -> None:
    """The one seed file format: records sorted by id, indent 2, UTF-8 text, LF, trailing newline."""
    ordered = sorted(records, key=lambda record: record["document_id"])
    path.write_text(json.dumps(ordered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


# -- Export -------------------------------------------------------------------


async def export_corpus(corpus_path: str | None = None, seed_dir: str | None = None) -> tuple[int, int, int]:
    """Write the corpus documents, catalog and chunk vectors to the seed directory.

    Returns (documents, catalog entries, documents with exported embeddings).
    A corpus that is not ready (for example one without chunks after a failed
    build) is refused before anything is read or written, so an export can never
    delete the seed embedding cache or blank its manifest.
    """
    target = Path(seed_dir) if seed_dir else SEED_DIR
    db = CorpusDB.open(corpus_path, readonly=True)
    try:
        await db.assert_ready(require_corpus=True)
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

        def _chunk_vectors(conn: sqlite3.Connection) -> tuple[dict, dict[str, tuple[list[str], list[bytes]]]]:
            meta = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM meta")}
            by_doc: dict[str, tuple[list[str], list[bytes]]] = {}
            for row in conn.execute("SELECT doc_id, chunk_text, embedding FROM chunks ORDER BY doc_id, chunk_index"):
                texts, blobs = by_doc.setdefault(row["doc_id"], ([], []))
                texts.append(row["chunk_text"])
                blobs.append(row["embedding"])
            return meta, by_doc

        meta, by_doc = await db.run(_chunk_vectors)
    finally:
        db.close()
    target.mkdir(parents=True, exist_ok=True)
    _write_canonical_json(target / "documents.json", docs)
    _write_canonical_json(target / "decision_cache.json", catalog)
    embedded = _export_embeddings(target / EMBEDDINGS_SUBDIR, meta, by_doc)
    logger.info(
        "Exported %d documents, %d catalog entries and embeddings for %d documents to %s",
        len(docs),
        len(catalog),
        embedded,
        target,
    )
    return len(docs), len(catalog), embedded


def _export_embeddings(directory: Path, meta: dict, by_doc: dict[str, tuple[list[str], list[bytes]]]) -> int:
    """Write ``<doc_id>.npy`` (float16 chunk vectors) and ``manifest.json``; drop files of vanished documents."""
    from bddk_mcp.store.vector_store import chunk_digest

    directory.mkdir(parents=True, exist_ok=True)
    dimension = int(meta.get("embedding_dimension") or EMBEDDING_DIMENSION)
    digests: dict[str, str] = {}
    for doc_id, (texts, blobs) in sorted(by_doc.items()):
        if not _SAFE_DOC_ID.fullmatch(doc_id):
            logger.warning("Embedding cache skipped for a document whose id is not a safe file name")
            continue
        vectors = np.frombuffer(b"".join(blobs), dtype=np.float16).reshape(len(blobs), dimension)
        np.save(directory / f"{doc_id}.npy", vectors)
        digests[doc_id] = chunk_digest(texts)
    for stale in directory.glob("*.npy"):
        if stale.stem not in digests:
            stale.unlink()
    manifest = {
        "embedding_model": meta.get("embedding_model", ""),
        "embedding_dimension": dimension,
        "documents": digests,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return len(digests)


# -- Build --------------------------------------------------------------------


class _EmbeddingCache:
    """Vectors exported to ``seed_data/embeddings/``, reused when the chunk texts and the model match."""

    def __init__(self, directory: Path, digests: dict[str, str]) -> None:
        self.directory = directory
        self.digests = digests
        self.chunks_used = 0

    @classmethod
    def load(cls, seed: Path) -> _EmbeddingCache:
        directory = seed / EMBEDDINGS_SUBDIR
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls(directory, {})
        except (OSError, UnicodeError, json.JSONDecodeError):
            logger.warning("Embedding cache manifest is unreadable; every document is embedded")
            return cls(directory, {})
        if not isinstance(manifest, dict):
            manifest = {}
        identity = (EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION)
        if (manifest.get("embedding_model"), manifest.get("embedding_dimension")) != identity:
            logger.warning("Embedding cache was exported with another embedding model; every document is embedded")
            return cls(directory, {})
        documents = manifest.get("documents")
        return cls(directory, documents if isinstance(documents, dict) else {})

    def lookup(self, doc_id: str):
        """``cached_embeddings(digest, count)`` for ``VectorStore.add_document``, or None without an entry."""
        expected = self.digests.get(doc_id)
        if not isinstance(expected, str) or not _SAFE_DOC_ID.fullmatch(doc_id):
            return None

        def cached(digest: str, count: int) -> np.ndarray | None:
            if digest != expected:
                return None
            try:
                vectors = np.load(self.directory / f"{doc_id}.npy", allow_pickle=False)
            except (OSError, ValueError, EOFError) as exc:
                logger.warning("Embedding cache file is unreadable (%s); the document is embedded", type(exc).__name__)
                return None
            if vectors.shape != (count, EMBEDDING_DIMENSION):
                return None
            self.chunks_used += count
            return vectors

        return cached


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
    re-embedded, unless ``seed_data/embeddings/`` still holds the vectors exported for
    exactly its chunk texts and model.  ``force`` rewrites and re-embeds everything and
    ignores that cache; it cannot switch a corpus file to another embedding model (delete
    the file and build instead).  The build skips when every content hash, the catalog size
    and the stored build fingerprint (``bddk_mcp.store.fingerprint``) match.  When the
    fingerprint is missing or its seed, parser or chunker part differs, every document is
    rewritten (vectors still come from the seed cache where the chunk texts match); without
    ``force``, a model-part difference (revision, passage prefix, prefix separator or vector
    dtype) is refused.  The fingerprint is written as the
    last step, after every document succeeded.  Documents removed from the seed stay in
    the file and need a fresh file.
    When a document needs chunking, the chunk tokenizer is loaded before the first
    write, so a host without it fails early; a corpus file that did not exist before
    such a failure is removed again.
    ``vector_store_factory`` (``db -> VectorStore``) is a test seam for a fake embedder.
    """
    from bddk_mcp.store.vector_store import VectorStore

    seed = Path(seed_dir) if seed_dir else SEED_DIR
    docs = _load_records(seed / "documents.json")
    catalog = _load_records(seed / "decision_cache.json")
    seed_hashes = _validate_seed_documents(docs)
    result = {"decision_cache": 0, "documents": 0, "sections": 0, "chunks": 0, "chunks_cached": 0, "skipped": False}
    if not docs and not catalog:
        logger.info("Seed files are empty; nothing to build")
        return result

    current_fingerprint = build_fingerprint.build_fingerprint(docs, catalog)
    corpus_file = Path(corpus_path or CORPUS_PATH)
    corpus_existed = corpus_file.exists()
    db = CorpusDB.open(corpus_path)
    closed = False
    try:
        store = DocumentStore(db)
        vs = (vector_store_factory or VectorStore)(db)

        def _stored_state(conn: sqlite3.Connection) -> tuple[dict[str, str], set[str], int, dict[str, str]]:
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
            meta = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM meta").fetchall()}
            return hashes, indexed, catalog_count, meta

        stored_hashes, indexed, catalog_count, meta = await db.run(_stored_state)
        rewrite_all = _fingerprint_rewrite(meta, current_fingerprint, has_documents=bool(stored_hashes), force=force)
        stale = [doc_id for doc_id, digest in seed_hashes.items() if stored_hashes.get(doc_id) != digest]
        by_id = {record["document_id"]: record for record in docs}
        unindexed = [
            doc_id for doc_id in seed_hashes if doc_id not in indexed and by_id[doc_id].get("markdown_content")
        ]
        if not force and not rewrite_all and not stale and not unindexed and catalog_count == len(catalog):
            logger.info("Corpus already matches the seed data; build skipped")
            result["skipped"] = True
            return result

        rewrite = list(seed_hashes) if force or rewrite_all else stale
        to_embed = sorted(set(rewrite) | set(unindexed))
        if any((by_id[doc_id].get("markdown_content") or "").strip() for doc_id in to_embed):
            # Preflight: fail before the first write when the tokenizer cannot load (offline host).
            try:
                await asyncio.to_thread(vs.load_chunk_tokenizer)
            except BaseException:
                db.close()
                closed = True
                if not corpus_existed:
                    _remove_corpus_file(corpus_file)
                raise

        if catalog:
            result["decision_cache"] = await store.replace_catalog(catalog)

        now = time.time()

        def _write_documents(conn: sqlite3.Connection) -> int:
            sections = 0
            for record in docs:
                if record["document_id"] in rewrite:
                    sections += upsert_document_sync(conn, _seed_document(record), now=now, prepare=False)[1]
            return sections

        result["sections"] = await db.transaction(_write_documents)
        result["documents"] = len(rewrite)

        cache = _EmbeddingCache(seed / EMBEDDINGS_SUBDIR, {}) if force else _EmbeddingCache.load(seed)
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
                cached_embeddings=cache.lookup(doc_id),
            )
        result["chunks_cached"] = cache.chunks_used
        # Last step: a build that failed before here leaves the old (or no) fingerprint and is redone.
        encoded = build_fingerprint.encode(current_fingerprint)
        await db.transaction(
            lambda conn: conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                (build_fingerprint.FINGERPRINT_META_KEY, encoded),
            )
        )
        logger.info(
            "Corpus built (%d catalog, %d documents rewritten, %d sections, %d chunks: %d embedded, %d from the seed cache)",
            result["decision_cache"],
            result["documents"],
            result["sections"],
            result["chunks"],
            result["chunks"] - result["chunks_cached"],
            result["chunks_cached"],
        )
    finally:
        if not closed:
            db.close()
    return result


def _fingerprint_rewrite(meta: dict[str, str], current: dict[str, str], *, has_documents: bool, force: bool) -> bool:
    """Whether every document must be rewritten because the file was built from other inputs or by other code.

    Refuses, before any write, a file embedded with another model name or dimension
    (as the vector store would) and, unless ``force``, one whose model part differs
    otherwise (revision, passage prefix, prefix separator or vector dtype): the seed
    embedding cache cannot detect such a change.
    """
    from bddk_mcp.store.vector_store import RetrievalProfileError

    embedded_with = meta.get("embedding_model")
    if embedded_with is not None and (
        embedded_with != EMBEDDING_MODEL_NAME or meta.get("embedding_dimension") != str(EMBEDDING_DIMENSION)
    ):
        raise RetrievalProfileError(
            "The corpus file was built with a different embedding model; delete it and run `bddk-mcp build` again."
        )
    stored = build_fingerprint.decode(meta.get(build_fingerprint.FINGERPRINT_META_KEY))
    if stored is None:
        if has_documents and not force:
            logger.warning("No build fingerprint: file predates fingerprinting; every document is rewritten")
        return has_documents
    unknown = build_fingerprint.unknown_parts(stored, current)
    if unknown:
        logger.warning("Build fingerprint parts not comparable (source text unavailable): %s", ", ".join(unknown))
    differing = build_fingerprint.differing_parts(stored, current)
    if "model" in differing and embedded_with is not None and not force:
        raise RuntimeError(
            "The corpus file was embedded with other embedding settings: another model revision, passage prefix, "
            "prefix separator or vector dtype "
            f"(BDDK_EMBEDDING_MODEL_REVISION is now {_config.EMBEDDING_MODEL_REVISION or 'unset'}); the seed "
            "embedding cache cannot detect such a change. Delete the corpus file and run `bddk-mcp build --force`."
        )
    if differing and not force:
        logger.warning("Build fingerprint differs (%s); every document is rewritten", ", ".join(differing))
    return bool(differing)


def _remove_corpus_file(path: Path) -> None:
    """Remove a corpus file created by a build that failed its preflight, with its side files."""
    for candidate in (path, *(path.with_name(path.name + suffix) for suffix in ("-wal", "-shm", "-journal"))):
        candidate.unlink(missing_ok=True)
