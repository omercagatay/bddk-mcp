"""Scan BDDK document quality from DB or a local Markdown export."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import asyncpg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bddk_mcp.core.config import require_database_url  # noqa: E402
from bddk_mcp.quality.markdown_quality import (  # noqa: E402
    QUALITY_FAILURES_PATH,
    assess_markdown_quality,
    load_quality_failure_registry,
)
from bddk_mcp.quality.quality_scan import (  # noqa: E402
    DocumentFinding,
    MethodBreakdown,
    QualityReport,
    format_report,
    format_report_csv,
    format_report_json,
    scan_quality,
)


def load_quality_failures(path: Path) -> list[dict[str, str]]:
    """Load a validated quality-failure registry for CLI reporting."""
    return [
        {
            "document_id": failure.document_id,
            "reason": failure.reason,
            "preferred_backfill": failure.preferred_backfill,
        }
        for failure in load_quality_failure_registry(path).values()
    ]


def scan_markdown_dir(md_dir: Path, manifest: Path | None = None) -> QualityReport:
    """Scan a local Markdown export directory and return a QualityReport."""
    del manifest  # reserved for manifest char-count checks in the next CLI slice

    findings: list[DocumentFinding] = []
    markdown_files = sorted(p for p in md_dir.rglob("*.md") if p.is_file())
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        document_id = path.stem
        quality = assess_markdown_quality(text, document_id=document_id)
        if quality.label == "clean":
            continue
        findings.append(
            DocumentFinding(
                document_id=document_id,
                label=quality.label,
                flags=quality.flags,
                counts=quality.counts,
                sample=" ".join(text.split())[:160],
            )
        )

    return QualityReport(
        total_documents=len(markdown_files),
        methods=[MethodBreakdown(method="markdown_export", doc_count=len(markdown_files), avg_chars=0)],
        anomalies=_anomalies_from_findings(findings),
        orphan_chunks=0,
        docs_without_chunks=0,
        document_findings=findings,
    )


async def scan_db() -> QualityReport:
    """Run the canonical DB quality scan."""
    pool = await asyncpg.create_pool(require_database_url(), min_size=1, max_size=3)
    try:
        return await scan_quality(pool)
    finally:
        await pool.close()


def write_outputs(report: QualityReport, out_dir: Path) -> None:
    """Write Markdown, CSV, JSON, and snippets outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quality_report.md").write_text(format_report(report) + "\n", encoding="utf-8")
    (out_dir / "quality_findings.csv").write_text(format_report_csv(report), encoding="utf-8")
    (out_dir / "quality_findings.json").write_text(
        json.dumps(format_report_json(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "suspicious_snippets.md").write_text(_format_snippets(report), encoding="utf-8")


def _anomalies_from_findings(findings: list[DocumentFinding]):
    from bddk_mcp.quality.quality_scan import AnomalyCount

    counts: dict[str, list[str]] = {}
    for finding in findings:
        for flag in finding.flags:
            counts.setdefault(flag, []).append(finding.document_id)
    return [
        AnomalyCount(
            name=flag,
            docs_flagged=len(doc_ids),
            description=f"{flag} detected by markdown quality assessment",
            sample_doc_ids=doc_ids[:5],
        )
        for flag, doc_ids in sorted(counts.items())
    ]


def _format_snippets(report: QualityReport) -> str:
    lines = ["# Suspicious Snippets", ""]
    if not report.document_findings:
        lines.append("No suspicious snippets.")
        return "\n".join(lines) + "\n"

    for finding in report.document_findings:
        lines.append(f"## {finding.document_id} ({finding.label})")
        lines.append("")
        lines.append(f"Flags: {', '.join(finding.flags) or 'none'}")
        lines.append("")
        lines.append("```text")
        lines.append(finding.sample)
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _should_fail(report: QualityReport, *, allow_failures: bool, fail_on: set[str]) -> bool:
    if fail_on and any(flag in fail_on for finding in report.document_findings for flag in finding.flags):
        return True
    if allow_failures:
        return False
    return any(finding.label == "fail" for finding in report.document_findings)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan BDDK document Markdown quality.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--db", action="store_true", help="Scan documents from the configured PostgreSQL database")
    source.add_argument("--md-dir", type=Path, help="Scan local Markdown export directory")
    parser.add_argument("--manifest", type=Path, help="Optional Markdown export manifest CSV")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for quality report outputs")
    parser.add_argument("--allow-failures", action="store_true", help="Exit zero even when fail-labeled docs exist")
    parser.add_argument("--fail-on", default="", help="Comma-separated quality flags that should force non-zero exit")
    parser.add_argument("--quality-failures", type=Path, default=QUALITY_FAILURES_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    load_quality_failures(args.quality_failures)

    if args.db:
        report = asyncio.run(scan_db())
    else:
        report = scan_markdown_dir(args.md_dir, manifest=args.manifest)

    write_outputs(report, args.out_dir)
    fail_on = {item.strip() for item in args.fail_on.split(",") if item.strip()}
    return 1 if _should_fail(report, allow_failures=args.allow_failures, fail_on=fail_on) else 0


if __name__ == "__main__":
    raise SystemExit(main())
