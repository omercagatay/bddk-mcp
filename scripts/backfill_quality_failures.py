"""Backfill documents that are explicitly marked as quality failures."""

from __future__ import annotations

import argparse
import asyncio
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import asyncpg
import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bddk_mcp.core.config import require_database_url  # noqa: E402
from bddk_mcp.ingest.doc_sync import DocumentSyncer  # noqa: E402
from bddk_mcp.store.doc_store import DocumentStore  # noqa: E402
from scripts.scan_document_quality import load_quality_failures  # noqa: E402


@dataclass
class QualityFailureCandidate:
    document_id: str
    reason: str = ""
    preferred_backfill: str = ""


def load_fail_documents(path: Path) -> list[QualityFailureCandidate]:
    """Load fail documents from config/quality_failures.yml or quality_findings.csv."""
    if path.suffix.lower() == ".csv":
        return _load_fail_documents_from_csv(path)
    return [
        QualityFailureCandidate(
            document_id=item.get("document_id", ""),
            reason=item.get("reason", ""),
            preferred_backfill=item.get("preferred_backfill", ""),
        )
        for item in load_quality_failures(path)
        if item.get("document_id")
    ]


def _load_fail_documents_from_csv(path: Path) -> list[QualityFailureCandidate]:
    candidates: list[QualityFailureCandidate] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("label") != "fail":
                continue
            candidates.append(
                QualityFailureCandidate(
                    document_id=row.get("document_id", ""),
                    reason=row.get("flags", ""),
                    preferred_backfill="quality_scan_followup",
                )
            )
    return [candidate for candidate in candidates if candidate.document_id]


def filter_candidates(
    candidates: list[QualityFailureCandidate],
    *,
    doc_id: str | None = None,
) -> list[QualityFailureCandidate]:
    """Filter candidates for targeted dry-runs or execution."""
    if doc_id:
        return [candidate for candidate in candidates if candidate.document_id == doc_id]
    return candidates


def render_dry_run(candidates: list[QualityFailureCandidate]) -> str:
    """Render dry-run candidate report."""
    lines = [f"Quality failure backfill candidates: {len(candidates)}"]
    if candidates:
        lines.append("")
        for candidate in candidates:
            parts = [candidate.document_id]
            if candidate.reason:
                parts.append(f"reason={candidate.reason}")
            if candidate.preferred_backfill:
                parts.append(f"preferred={candidate.preferred_backfill}")
            lines.append("  " + "  ".join(parts))
    lines.append("")
    lines.append("Dry run — no changes made. Use --execute to re-extract matching documents.")
    return "\n".join(lines)


async def execute_quality_backfill(candidates: list[QualityFailureCandidate], *, dsn: str | None = None) -> int:
    """Execute targeted re-extraction for quality failures."""
    pool = await asyncpg.create_pool(dsn or require_database_url(), min_size=1, max_size=3)
    http = httpx.AsyncClient()
    try:
        store = DocumentStore(pool)
        async with DocumentSyncer(store, http=http) as syncer:
            for index, candidate in enumerate(candidates, 1):
                print(f"[{index}/{len(candidates)}] {candidate.document_id}: re-extracting")
                doc = await store.get_document(candidate.document_id)
                if doc is None:
                    print("  skipped: not found in document store")
                    continue
                result = await syncer.sync_document(
                    doc_id=doc.document_id,
                    title=doc.title,
                    category=doc.category,
                    source_url=doc.source_url,
                    decision_date=doc.decision_date,
                    decision_number=doc.decision_number,
                    force=True,
                )
                if result.success:
                    print(f"  ok: method={result.method} size={result.size_bytes}")
                else:
                    print(f"  failed: {result.error}")
        return 0
    finally:
        await http.aclose()
        await pool.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill documents labeled as quality failures.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "quality_failures.yml")
    parser.add_argument("--doc-id", help="Target one document ID")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute", action="store_true", help="Actually re-extract matching documents")
    parser.add_argument("--database-url", help="Override BDDK_DATABASE_URL")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    candidates = filter_candidates(load_fail_documents(args.config), doc_id=args.doc_id)

    if not args.execute:
        print(render_dry_run(candidates))
        return 0

    return asyncio.run(execute_quality_backfill(candidates, dsn=args.database_url))


if __name__ == "__main__":
    raise SystemExit(main())
