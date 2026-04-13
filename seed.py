"""
Export / import BDDK data as JSON seed files.

Lets you populate PostgreSQL locally, export the data, bake it into the
Docker image, and deploy to Railway with zero BDDK requests.

Usage:
    python seed.py export          # dump DB → seed_data/
    python seed.py import          # seed_data/ → DB (skips if DB already has data)
    python seed.py import --force  # overwrite existing DB data
"""

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import asyncpg

from config import DATABASE_URL

logger = logging.getLogger(__name__)

SEED_DIR = Path(__file__).parent / "seed_data"


# ── Export ───────────────────────────────────────────────────────────────────


async def export_seed(dsn: str | None = None) -> None:
    """Export decision_cache and documents tables to seed_data/ as JSON."""
    pool = await asyncpg.create_pool(dsn or DATABASE_URL, min_size=1, max_size=3)
    SEED_DIR.mkdir(exist_ok=True)

    try:
        async with pool.acquire() as conn:
            # 1. Decision cache
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

            # 2. Documents (without pdf_blob to keep size manageable)
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

            # 3. Document chunks (embeddings as lists for JSON serialization)
            rows = await conn.fetch(
                "SELECT doc_id, chunk_index, title, category, decision_date, "
                "decision_number, source_url, total_chunks, total_pages, "
                "content_hash, chunk_text FROM document_chunks"
            )
            chunks_data = [dict(r) for r in rows]
            chunks_path = SEED_DIR / "chunks.json"
            chunks_path.write_text(
                json.dumps(chunks_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Exported {len(chunks_data)} chunks → {chunks_path}")

    finally:
        await pool.close()

    print(f"\nSeed data written to {SEED_DIR}/")
    print("Commit this directory and rebuild your Docker image.")


# ── Import ───────────────────────────────────────────────────────────────────


async def import_seed(dsn: str | None = None, force: bool = False) -> dict:
    """Import seed data from seed_data/ into PostgreSQL.

    Returns dict with counts of imported items.
    Skips import if tables already have data (unless force=True).
    """
    result = {"decision_cache": 0, "documents": 0, "chunks": 0, "skipped": False}

    if not SEED_DIR.exists():
        logger.info("No seed_data/ directory found — skipping seed import")
        return result

    pool = await asyncpg.create_pool(dsn or DATABASE_URL, min_size=1, max_size=3)

    try:
        # Initialize schemas first
        from doc_store import DocumentStore

        store = DocumentStore(pool)
        await store.initialize()

        from vector_store import VectorStore

        vs = VectorStore(pool)
        await vs.initialize()

        from client import BddkApiClient

        client = BddkApiClient(pool)
        await client.initialize()

        async with pool.acquire() as conn:
            # Check if DB already has data
            if not force:
                cache_count = await conn.fetchval("SELECT COUNT(*) FROM decision_cache")
                doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents")
                if cache_count > 0 and doc_count > 0:
                    logger.info(
                        "DB already has data (%d cache, %d docs) — skipping seed import",
                        cache_count,
                        doc_count,
                    )
                    result["skipped"] = True
                    return result

            # 1. Decision cache
            cache_path = SEED_DIR / "decision_cache.json"
            if cache_path.exists():
                cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
                if cache_data:
                    now = time.time()
                    async with conn.transaction():
                        await conn.execute("DELETE FROM decision_cache")
                        for d in cache_data:
                            await conn.execute(
                                """INSERT INTO decision_cache
                                   (document_id, title, content, decision_date,
                                    decision_number, category, source_url, cached_at)
                                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                                   ON CONFLICT(document_id) DO NOTHING""",
                                d["document_id"],
                                d.get("title", ""),
                                d.get("content", ""),
                                d.get("decision_date", ""),
                                d.get("decision_number", ""),
                                d.get("category", ""),
                                d.get("source_url", ""),
                                now,
                            )
                    result["decision_cache"] = len(cache_data)
                    logger.info("Imported %d decision cache entries", len(cache_data))

            # 2. Documents
            docs_path = SEED_DIR / "documents.json"
            if docs_path.exists():
                docs_data = json.loads(docs_path.read_text(encoding="utf-8"))
                if docs_data:
                    imported = 0
                    for d in docs_data:
                        try:
                            await conn.execute(
                                """INSERT INTO documents
                                   (document_id, title, category, decision_date,
                                    decision_number, source_url, markdown_content,
                                    content_hash, downloaded_at, extracted_at,
                                    extraction_method, total_pages, file_size)
                                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)
                                   ON CONFLICT(document_id) DO NOTHING""",
                                d["document_id"],
                                d.get("title", ""),
                                d.get("category", ""),
                                d.get("decision_date", ""),
                                d.get("decision_number", ""),
                                d.get("source_url", ""),
                                d.get("markdown_content", ""),
                                d.get("content_hash", ""),
                                d.get("downloaded_at"),
                                d.get("extracted_at"),
                                d.get("extraction_method", "markitdown"),
                                d.get("total_pages", 1),
                                d.get("file_size", 0),
                            )
                            imported += 1
                        except Exception as e:
                            logger.warning("Failed to import doc %s: %s", d.get("document_id"), e)
                    result["documents"] = imported
                    logger.info("Imported %d documents", imported)

            # 3. Chunks (text only — embeddings regenerated on first use)
            chunks_path = SEED_DIR / "chunks.json"
            if chunks_path.exists():
                chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
                if chunks_data:
                    imported = 0
                    for c in chunks_data:
                        try:
                            await conn.execute(
                                """INSERT INTO document_chunks
                                   (doc_id, chunk_index, title, category,
                                    decision_date, decision_number, source_url,
                                    total_chunks, total_pages, content_hash,
                                    chunk_text)
                                   VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
                                   ON CONFLICT(doc_id, chunk_index) DO NOTHING""",
                                c["doc_id"],
                                c["chunk_index"],
                                c.get("title", ""),
                                c.get("category", ""),
                                c.get("decision_date", ""),
                                c.get("decision_number", ""),
                                c.get("source_url", ""),
                                c.get("total_chunks", 1),
                                c.get("total_pages", 1),
                                c.get("content_hash", ""),
                                c["chunk_text"],
                            )
                            imported += 1
                        except Exception as e:
                            logger.warning(
                                "Failed to import chunk %s/%d: %s", c.get("doc_id"), c.get("chunk_index", 0), e
                            )
                    result["chunks"] = imported
                    logger.info("Imported %d chunks", imported)

    finally:
        await pool.close()

    return result


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="BDDK seed data export/import")
    parser.add_argument("--db", help="PostgreSQL DSN", default=None)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("export", help="Export DB → seed_data/")

    imp = sub.add_parser("import", help="Import seed_data/ → DB")
    imp.add_argument("--force", action="store_true", help="Overwrite existing data")

    args = parser.parse_args()

    if args.command == "export":
        asyncio.run(export_seed(args.db))
    elif args.command == "import":
        result = asyncio.run(import_seed(args.db, force=args.force))
        if result["skipped"]:
            print("Skipped — DB already has data (use --force to overwrite)")
        else:
            print(
                f"\nImported: {result['decision_cache']} cache, {result['documents']} docs, {result['chunks']} chunks"
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
