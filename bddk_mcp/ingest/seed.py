"""
Export / import BDDK data as JSON seed files.

Lets you populate PostgreSQL locally, export the data, bake it into the
Docker image, and deploy with zero BDDK requests.

Usage:
    bddk-seed export          # dump DB → seed_data/
    bddk-seed import          # seed_data/ → DB (skips if DB already matches)
    bddk-seed import --force  # overwrite existing DB data
"""

import argparse
import asyncio
import hashlib
import json
import logging
import math
import time
from pathlib import Path

import asyncpg

from bddk_mcp.core.config import require_database_url
from bddk_mcp.corpus_coordination import acquire_corpus_mutation_lock
from bddk_mcp.db_transport import assert_database_transport
from bddk_mcp.store.bulk_write import (
    insert_document_chunk_rows,
    insert_document_version_rows,
    upsert_decision_cache_rows,
    upsert_document_rows,
)
from bddk_mcp.store.section_index import DocumentSection, extract_document_sections

logger = logging.getLogger(__name__)

SEED_DIR = Path(__file__).resolve().parents[2] / "seed_data"


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
    """Validate canonical seed documents before model or database work."""

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


def _embedding_vector(value, *, dimension: int) -> tuple[float, ...] | None:
    """Parse one embedding value under strict numeric rules."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, list | tuple) or len(value) != dimension:
        return None
    parsed: list[float] = []
    for component in value:
        if isinstance(component, bool):
            return None
        try:
            number = float(component)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(number):
            return None
        parsed.append(number)
    if math.fsum(component * component for component in parsed) <= 0.0:
        return None
    return tuple(parsed)


def _generate_seed_chunks(vs, docs: list[dict]) -> tuple[list[dict], dict[str, list[dict]]]:
    """Regenerate chunks from canonical text under the current pinned profile."""

    from bddk_mcp.core.config import PAGE_SIZE
    from bddk_mcp.store.vector_store import _chunk_document

    tokenizer = vs._chunk_tokenizer()
    generated: list[dict] = []
    grouped: dict[str, list[dict]] = {}
    for document in docs:
        doc_id = document["document_id"]
        content = document["markdown_content"]
        chunks = _chunk_document(doc_id, content, tokenizer=tokenizer)
        if not chunks:
            raise RuntimeError("Seed document could not produce a retrieval chunk.")
        total_pages = max(1, math.ceil(len(content) / PAGE_SIZE))
        rows: list[dict] = []
        for index, chunk in enumerate(chunks):
            row = {
                "doc_id": doc_id,
                "chunk_index": index,
                "title": document.get("title", ""),
                "category": document.get("category", ""),
                "decision_date": document.get("decision_date", ""),
                "decision_number": document.get("decision_number", ""),
                "source_url": document.get("source_url", ""),
                "total_chunks": len(chunks),
                "total_pages": total_pages,
                "content_hash": document["content_hash"],
                "chunk_start_char": chunk.start_char,
                "chunk_end_char": chunk.end_char,
                "section_type": chunk.section_type,
                "section_ref": chunk.section_ref,
                "section_start_char": chunk.section_start_char,
                "section_end_char": chunk.section_end_char,
                "section_content_hash": chunk.section_content_hash,
                "chunk_text": chunk.chunk_text,
            }
            rows.append(row)
            generated.append(row)
        grouped[doc_id] = rows
    return generated, grouped


async def _regenerate_seed_embedding_vectors(
    vs,
    chunks: list[dict],
    *,
    batch_size: int = 32,
) -> list[list[float]]:
    """Regenerate and validate every vector under the store's pinned profile."""

    from bddk_mcp.store.vector_store import EMBEDDING_DIMENSION

    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embedded = await vs._embed([chunk["chunk_text"] for chunk in batch], prefix="passage")
        for vector in embedded:
            parsed = _embedding_vector(vector, dimension=EMBEDDING_DIMENSION)
            if parsed is None:
                raise RuntimeError("Seed embedding output was malformed.")
            vectors.append(list(parsed))
    if len(vectors) != len(chunks):
        raise RuntimeError("Seed embedding output was incomplete.")
    return vectors


async def _embed_seed_chunks(vs, chunks: list[dict], *, batch_size: int = 32) -> list[str]:
    """Precompute every seed vector before opening the publication transaction."""

    vectors = await _regenerate_seed_embedding_vectors(vs, chunks, batch_size=batch_size)
    return ["[" + ",".join(str(value) for value in vector) + "]" for vector in vectors]


def _expected_seed_sections(docs: list[dict]) -> dict[str, list[DocumentSection]]:
    """Parse the deterministic section index expected from seed documents."""
    return {
        doc["document_id"]: extract_document_sections(
            doc["document_id"],
            doc.get("markdown_content", ""),
        )
        for doc in docs
        if doc.get("document_id")
    }


async def _section_index_matches(
    conn,
    expected_sections: dict[str, list[DocumentSection]],
) -> bool:
    """Return whether seeded documents have the exact expected section index."""
    return not await _section_index_mismatches(conn, expected_sections)


async def _section_index_mismatches(
    conn,
    expected_sections: dict[str, list[DocumentSection]],
) -> set[str]:
    """Return the seeded document IDs whose stored section index differs from the expected one."""
    if not expected_sections:
        return set()
    rows = await conn.fetch(
        """
        SELECT doc_id, section_type, section_ref, content_hash,
               start_char, end_char
        FROM document_sections
        WHERE doc_id = ANY($1::text[])
        """,
        sorted(expected_sections),
    )
    actual = {
        (
            row["doc_id"],
            row["section_type"],
            row["section_ref"],
            row["content_hash"],
            row["start_char"],
            row["end_char"],
        )
        for row in rows
    }
    expected = {
        (
            doc_id,
            section.section_type,
            section.section_ref,
            section.content_hash,
            section.start_char,
            section.end_char,
        )
        for doc_id, sections in expected_sections.items()
        for section in sections
    }
    return {key[0] for key in actual ^ expected}


def _strip_docs_dump_header(text: str) -> str:
    """Return the body of a docs_dump-style markdown file.

    docs_dump files follow the shape ``# Title\n- Document ID: ...\n---\n<body>``.
    Split on the first ``\\n---\\n`` so only the article body is retained. Returns
    ``text`` unchanged when no separator is present — the caller can pass either
    raw markdown or a dump file without branching.
    """
    parts = text.split("\n---\n", 1)
    return parts[1].lstrip() if len(parts) == 2 else text


# ── Export ───────────────────────────────────────────────────────────────────


async def export_seed(dsn: str | None = None, pool: asyncpg.Pool | None = None) -> None:
    """Export decision_cache and documents tables to seed_data/ as JSON.

    Chunks and embeddings are not exported: ``import`` regenerates them from
    the canonical document text under the current retrieval profile.
    """
    owns_pool = pool is None
    if owns_pool:
        selected_dsn = assert_database_transport(dsn) if dsn else require_database_url("public")
        pool = await asyncpg.create_pool(selected_dsn, min_size=1, max_size=3)
    SEED_DIR.mkdir(exist_ok=True)

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT document_id, title, content, decision_date, "
                "decision_number, category, source_url FROM decision_cache"
            )
            cache_data = [dict(r) for r in rows]
            cache_path = SEED_DIR / "decision_cache.json"
            cache_path.write_text(
                json.dumps(cache_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Exported {len(cache_data)} decision cache entries → {cache_path}")

            # Documents (without pdf_blob to keep size manageable)
            rows = await conn.fetch(
                "SELECT document_id, title, category, decision_date, "
                "decision_number, source_url, markdown_content, content_hash, "
                "downloaded_at, extracted_at, extraction_method, total_pages, "
                "file_size FROM documents"
            )
            docs_data = [dict(r) for r in rows]
            docs_path = SEED_DIR / "documents.json"
            docs_path.write_text(
                json.dumps(docs_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Exported {len(docs_data)} documents → {docs_path}")
    finally:
        if owns_pool:
            await pool.close()

    print(f"\nSeed data written to {SEED_DIR}/")
    print("Commit this directory and rebuild your Docker image.")


# ── Import ───────────────────────────────────────────────────────────────────


async def import_seed(
    dsn: str | None = None,
    force: bool = False,
    pool: asyncpg.Pool | None = None,
    *,
    reindex_existing: bool = False,
) -> dict:
    """Import seed data from seed_data/ into PostgreSQL.

    Returns dict with counts of imported items.
    Skips import if the database already matches the seed (unless force=True).
    """
    result = {
        "decision_cache": 0,
        "documents": 0,
        "sections": 0,
        "chunks": 0,
        "embedded": 0,
        "reindex_scanned": 0,
        "reindex_published": 0,
        "reindex_current": 0,
        "skipped": False,
    }

    if not SEED_DIR.exists():
        logger.info("No seed_data/ directory found — skipping seed import")
        return result

    docs_data = _load_records(SEED_DIR / "documents.json")
    cache_data = _load_records(SEED_DIR / "decision_cache.json")

    owns_pool = pool is None
    if owns_pool:
        selected_dsn = assert_database_transport(dsn) if dsn else require_database_url("ingestion")
        pool = await asyncpg.create_pool(selected_dsn, min_size=1, max_size=3)

    try:
        # Import is a data-loading operation, not a migration path.  Fail
        # before the first DML statement unless an operator has explicitly
        # prepared the complete schema with ``bddk-mcp migrate``.
        from bddk_mcp.db_lifecycle import assert_database_ready

        await assert_database_ready(pool=pool, require_corpus=False)

        from bddk_mcp.store.doc_store import DocumentStore

        store = DocumentStore(pool)

        from bddk_mcp.store.vector_store import VectorStore

        vs = VectorStore(pool)

        seed_hashes = _validate_seed_documents(docs_data)
        expected_sections = _expected_seed_sections(docs_data)
        seed_doc_ids = sorted(seed_hashes)
        seed_cache_ids = sorted(item["document_id"] for item in cache_data)

        if not cache_data and not docs_data and not reindex_existing:
            logger.info("Seed files are empty; no data was imported")
            return result

        async with pool.acquire() as conn:
            db_rows = await conn.fetch(
                "SELECT document_id, content_hash FROM public.documents WHERE document_id = ANY($1::text[])",
                seed_doc_ids,
            )
            db_hashes = {row["document_id"]: row["content_hash"] or "" for row in db_rows}
            changed = [doc_id for doc_id, value in db_hashes.items() if value != seed_hashes[doc_id]]
            if changed and not force:
                raise RuntimeError(
                    "Seed content differs from existing documents; rerun with --force only after reviewing the change."
                )

            cache_count = await conn.fetchval(
                "SELECT COUNT(*) FROM public.decision_cache WHERE document_id = ANY($1::text[])",
                seed_cache_ids,
            )
            null_embedding_count = await conn.fetchval(
                "SELECT COUNT(*) FROM public.document_chunks WHERE doc_id = ANY($1::text[]) AND embedding IS NULL",
                seed_doc_ids,
            )
            publication_count = await conn.fetchval(
                "SELECT COUNT(*) FROM public.document_retrieval_publications "
                "WHERE doc_id = ANY($1::text[]) AND retrieval_profile_hash = $2",
                seed_doc_ids,
                vs.retrieval_profile_hash,
            )
            sections_ok = await _section_index_matches(conn, expected_sections)
            if (
                not force
                and len(db_hashes) == len(docs_data)
                and cache_count == len(cache_data)
                and null_embedding_count == 0
                and publication_count == len(docs_data)
                and sections_ok
            ):
                logger.info("Seed publication is already current; import skipped")
                result["skipped"] = True
                if reindex_existing:
                    result.update(await reindex_existing_documents(pool, vs=vs))
                return result

        # Expensive model work happens before the database transaction.  No
        # canonical or retrieval row can become visible if embedding fails.
        chunks_data, grouped_chunks = _generate_seed_chunks(vs, docs_data)
        vector_values = await _embed_seed_chunks(vs, chunks_data)

        async with pool.acquire() as conn, conn.transaction():
            await acquire_corpus_mutation_lock(conn)
            locked_rows = await conn.fetch(
                "SELECT document_id, content_hash, markdown_content FROM public.documents "
                "WHERE document_id = ANY($1::text[]) FOR UPDATE",
                seed_doc_ids,
            )
            locked_documents = {row["document_id"]: row for row in locked_rows}
            changed = [
                doc_id for doc_id, row in locked_documents.items() if (row["content_hash"] or "") != seed_hashes[doc_id]
            ]
            if changed and not force:
                raise RuntimeError(
                    "Seed content changed during preparation; import refused without a reviewed --force operation."
                )

            now = time.time()
            if cache_data:
                await conn.execute(
                    "DELETE FROM public.decision_cache WHERE document_id = ANY($1::text[])",
                    seed_cache_ids,
                )
                await upsert_decision_cache_rows(
                    conn,
                    [
                        (
                            item["document_id"],
                            item.get("title", ""),
                            item.get("content", ""),
                            item.get("decision_date", ""),
                            item.get("decision_number", ""),
                            item.get("category", ""),
                            item.get("source_url", ""),
                            now,
                        )
                        for item in cache_data
                    ],
                )

            version_sources = {
                doc_id: existing
                for doc_id, existing in locked_documents.items()
                if existing["content_hash"] and existing["content_hash"] != seed_hashes[doc_id]
            }
            max_versions: dict[str, int] = {}
            if version_sources:
                version_rows = await conn.fetch(
                    """
                    SELECT requested.document_id,
                           COALESCE(pg_catalog.max(version.version), 0)::pg_catalog.int4 AS max_version
                    FROM pg_catalog.unnest($1::pg_catalog.text[]) AS requested(document_id)
                    LEFT JOIN public.document_versions AS version
                      ON version.document_id = requested.document_id
                    GROUP BY requested.document_id
                    """,
                    sorted(version_sources),
                )
                max_versions = {str(row["document_id"]): int(row["max_version"]) for row in version_rows}
            await insert_document_version_rows(
                conn,
                [
                    (
                        doc_id,
                        max_versions[doc_id] + 1,
                        version_sources[doc_id]["content_hash"],
                        version_sources[doc_id]["markdown_content"],
                        now,
                    )
                    for doc_id in sorted(version_sources)
                ],
            )
            await upsert_document_rows(
                conn,
                [
                    (
                        document["document_id"],
                        document.get("title", ""),
                        document.get("category", ""),
                        document.get("decision_date", ""),
                        document.get("decision_number", ""),
                        document.get("source_url", ""),
                        document.get("markdown_content", ""),
                        document["content_hash"],
                        document.get("downloaded_at"),
                        document.get("extracted_at"),
                        document.get("extraction_method", "markitdown"),
                        document.get("total_pages", 1),
                        document.get("file_size", 0),
                    )
                    for document in docs_data
                ],
            )
            # Rewrite sections only where they can differ, so unchanged
            # documents keep their section row identities.
            section_doc_ids = (
                set(changed)
                | (set(seed_doc_ids) - set(locked_documents))
                | await _section_index_mismatches(conn, expected_sections)
            )
            indexed_sections = await store._replace_document_sections_many_on_connection(
                conn,
                {doc_id: expected_sections[doc_id] for doc_id in sorted(section_doc_ids)},
                source_content_hashes={doc_id: seed_hashes[doc_id] for doc_id in section_doc_ids},
            )

            if seed_doc_ids:
                await conn.execute(
                    "DELETE FROM public.document_chunks WHERE doc_id = ANY($1::text[])",
                    seed_doc_ids,
                )
            await insert_document_chunk_rows(
                conn,
                [
                    (
                        chunk["doc_id"],
                        chunk["chunk_index"],
                        chunk.get("title", ""),
                        chunk.get("category", ""),
                        chunk.get("decision_date", ""),
                        chunk.get("decision_number", ""),
                        chunk.get("source_url", ""),
                        chunk["total_chunks"],
                        chunk.get("total_pages", 1),
                        chunk["content_hash"],
                        chunk.get("chunk_start_char"),
                        chunk.get("chunk_end_char"),
                        chunk.get("section_type", ""),
                        chunk.get("section_ref", ""),
                        chunk.get("section_start_char"),
                        chunk.get("section_end_char"),
                        chunk.get("section_content_hash", ""),
                        chunk["chunk_text"],
                        vector,
                    )
                    for chunk, vector in zip(chunks_data, vector_values, strict=True)
                ],
            )
            await vs._publish_documents_on_connection(
                conn,
                [(doc_id, seed_hashes[doc_id], len(grouped_chunks[doc_id])) for doc_id in sorted(grouped_chunks)],
            )

            result.update(
                decision_cache=len(cache_data),
                documents=len(docs_data),
                sections=indexed_sections,
                chunks=len(chunks_data),
                embedded=len(chunks_data),
            )
            logger.info(
                "Atomically published seed (%d cache, %d documents, %d sections, %d chunks)",
                len(cache_data),
                len(docs_data),
                indexed_sections,
                len(chunks_data),
            )

        if reindex_existing:
            result.update(await reindex_existing_documents(pool, vs=vs))

    finally:
        if owns_pool:
            await pool.close()

    return result


async def reindex_existing_documents(pool: asyncpg.Pool, *, vs=None) -> dict[str, int]:
    """Resumably publish every canonical document under the current profile."""

    if vs is None:
        from bddk_mcp.store.vector_store import VectorStore

        vs = VectorStore(pool)
    scanned = published = current = 0
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT document_id, title, markdown_content, category, decision_date,
                   decision_number, source_url
            FROM public.documents
            WHERE COALESCE(markdown_content, '') <> ''
            ORDER BY document_id
            """
        )
    # Each publication has its own short transaction-scoped mutation lock.
    for row in rows:
        scanned += 1
        if await vs.has_document(row["document_id"]):
            current += 1
            continue
        await vs.add_document(
            doc_id=row["document_id"],
            title=row["title"] or row["document_id"],
            content=row["markdown_content"],
            category=row["category"] or "",
            decision_date=row["decision_date"] or "",
            decision_number=row["decision_number"] or "",
            source_url=row["source_url"] or "",
        )
        published += 1

    logger.info(
        "Existing-document vector reconciliation complete (scanned=%d, published=%d, current=%d)",
        scanned,
        published,
        current,
    )
    return {
        "reindex_scanned": scanned,
        "reindex_published": published,
        "reindex_current": current,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="BDDK seed data export/import")
    parser.add_argument("--db", help="PostgreSQL DSN", default=None)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("export", help="Export DB -> seed_data/")

    imp = sub.add_parser("import", help="Import seed_data/ -> DB")
    imp.add_argument("--force", action="store_true", help="Overwrite existing data")
    imp.add_argument(
        "--reindex-existing",
        action="store_true",
        help="Also publish every stored document that lacks chunks for the current retrieval profile",
    )

    args = parser.parse_args()

    if args.command == "export":
        asyncio.run(export_seed(args.db))
    elif args.command == "import":
        result = asyncio.run(import_seed(args.db, force=args.force, reindex_existing=args.reindex_existing))
        if result["skipped"]:
            print("Skipped — DB already matches the seed (use --force to overwrite)")
        else:
            embedded = result.get("embedded", 0)
            print(
                f"\nImported: {result['decision_cache']} cache, {result['documents']} docs, "
                f"{result['sections']} sections, {result['chunks']} chunks ({embedded} embedded)"
            )
        if args.reindex_existing:
            print(
                f"Reindex: {result['reindex_scanned']} scanned, {result['reindex_published']} published, "
                f"{result['reindex_current']} already current"
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
