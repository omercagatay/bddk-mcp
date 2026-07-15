"""Rebuild document_sections rows from stored document Markdown."""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

import asyncpg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bddk_mcp.core.config import require_database_url  # noqa: E402
from bddk_mcp.db_identity import assert_database_identity  # noqa: E402
from bddk_mcp.db_lifecycle import assert_database_ready  # noqa: E402
from bddk_mcp.store.doc_store import DocumentStore  # noqa: E402
from bddk_mcp.store.section_index import extract_document_sections  # noqa: E402


@dataclass
class DocumentSectionReindexCandidate:
    document_id: str
    markdown_content: str
    content_hash: str


@dataclass
class SectionReindexStats:
    scanned_documents: int = 0
    documents_with_sections: int = 0
    sections_indexed: int = 0


def _resolve_ingestion_database_url(override: str | None) -> str:
    """Resolve only the configured ingestion identity, never an ad-hoc DSN."""
    configured = require_database_url("ingestion")
    if override is not None and override.strip() != configured:
        raise RuntimeError("--database-url must match BDDK_INGESTION_DATABASE_URL")
    return configured


async def reindex_document_rows(
    rows: list[DocumentSectionReindexCandidate],
    *,
    store,
    dry_run: bool,
) -> SectionReindexStats:
    """Parse section rows and optionally replace persisted section indexes."""
    stats = SectionReindexStats()
    for row in rows:
        stats.scanned_documents += 1
        sections = extract_document_sections(row.document_id, row.markdown_content)
        if sections:
            stats.documents_with_sections += 1
            stats.sections_indexed += len(sections)
        if not dry_run:
            await store.replace_document_sections(
                row.document_id,
                sections,
                source_content_hash=row.content_hash,
            )
    return stats


async def load_document_rows(
    pool: asyncpg.Pool,
    *,
    doc_id: str | None = None,
    limit: int | None = None,
) -> list[DocumentSectionReindexCandidate]:
    """Load stored documents that can be section-indexed."""
    where = ["markdown_content != ''"]
    params: list = []
    if doc_id:
        params.append(doc_id)
        where.append(f"document_id = ${len(params)}")

    sql = f"""
        SELECT document_id, markdown_content, content_hash
        FROM public.documents
        WHERE {" AND ".join(where)}
        ORDER BY document_id
    """
    if limit is not None:
        params.append(limit)
        sql += f" LIMIT ${len(params)}"

    rows = await pool.fetch(sql, *params)
    return [
        DocumentSectionReindexCandidate(
            document_id=row["document_id"],
            markdown_content=row["markdown_content"] or "",
            content_hash=row["content_hash"] or "",
        )
        for row in rows
    ]


async def execute_reindex(
    *,
    dsn: str | None = None,
    doc_id: str | None = None,
    limit: int | None = None,
    dry_run: bool,
) -> SectionReindexStats:
    """Load documents from PostgreSQL and rebuild section indexes."""
    pool = await asyncpg.create_pool(_resolve_ingestion_database_url(dsn), min_size=1, max_size=3)
    try:
        await assert_database_ready(pool=pool, require_corpus=False)
        await assert_database_identity(pool, "ingestion")
        store = DocumentStore(pool)
        rows = await load_document_rows(pool, doc_id=doc_id, limit=limit)
        return await reindex_document_rows(rows, store=store, dry_run=dry_run)
    finally:
        await pool.close()


def render_summary(
    *,
    scanned_documents: int,
    documents_with_sections: int,
    sections_indexed: int,
    dry_run: bool,
) -> str:
    lines = [
        f"Documents scanned: {scanned_documents}",
        f"Documents with sections: {documents_with_sections}",
        f"Sections parsed: {sections_indexed}",
    ]
    if dry_run:
        lines.append("Dry run - no changes made. Use --execute to replace document_sections rows.")
    else:
        lines.append("Reindex complete.")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild document_sections from stored Markdown content.")
    parser.add_argument("--database-url", help="Override BDDK_INGESTION_DATABASE_URL")
    parser.add_argument("--doc-id", help="Target one document ID")
    parser.add_argument("--limit", type=int, help="Maximum number of documents to scan")
    parser.add_argument("--execute", action="store_true", help="Actually replace document_sections rows")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    stats = asyncio.run(
        execute_reindex(
            dsn=args.database_url,
            doc_id=args.doc_id,
            limit=args.limit,
            dry_run=not args.execute,
        )
    )
    print(
        render_summary(
            scanned_documents=stats.scanned_documents,
            documents_with_sections=stats.documents_with_sections,
            sections_indexed=stats.sections_indexed,
            dry_run=not args.execute,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
