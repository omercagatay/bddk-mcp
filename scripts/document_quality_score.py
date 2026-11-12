"""Print a numeric document-quality score for autoresearch.

Composite score:
    100 - fail_rate*70 - warning_rate*25 - anomaly_density*5 - integrity_rate*10

The command reuses the canonical quality scan so the metric tracks the same
signals that MCP users see as document quality labels and warnings.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bddk_mcp.core.config import CORPUS_PATH  # noqa: E402
from bddk_mcp.quality.quality_scan import (  # noqa: E402
    QualityReport,
    format_report_json,
    scan_quality,
)
from bddk_mcp.store.corpus_db import CorpusDB  # noqa: E402
from scripts.scan_document_quality import scan_markdown_dir  # noqa: E402


def quality_score(report: QualityReport) -> dict:
    total = max(report.total_documents, 1)
    fail_count = sum(1 for finding in report.document_findings if finding.label == "fail")
    warning_count = sum(1 for finding in report.document_findings if finding.label == "warning")
    anomaly_total = sum(anomaly.docs_flagged for anomaly in report.anomalies)
    anomaly_denominator = max(total * max(len(report.anomalies), 1), 1)
    integrity_count = report.orphan_chunks + report.docs_without_chunks

    fail_rate = fail_count / total
    warning_rate = warning_count / total
    anomaly_density = anomaly_total / anomaly_denominator
    integrity_rate = integrity_count / total
    penalty = (fail_rate * 70) + (warning_rate * 25) + (anomaly_density * 5) + (integrity_rate * 10)
    score = max(0.0, 100.0 - penalty)

    return {
        "score": round(score, 4),
        "total_documents": report.total_documents,
        "clean_documents": report.total_documents - len(report.document_findings),
        "warning_documents": warning_count,
        "fail_documents": fail_count,
        "anomaly_hits": anomaly_total,
        "orphan_chunks": report.orphan_chunks,
        "docs_without_chunks": report.docs_without_chunks,
    }


async def scan_db(corpus_path: str | None = None) -> QualityReport:
    """Run the canonical quality scan over the corpus file (opened read-only)."""
    db = CorpusDB.open(corpus_path, readonly=True)
    try:
        return await scan_quality(db)
    finally:
        db.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute BDDK document-quality score.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--db", action="store_true", help="Scan documents from the corpus file")
    source.add_argument("--md-dir", type=Path, help="Scan local Markdown export directory")
    parser.add_argument(
        "--corpus", default=CORPUS_PATH, help="Corpus file path (default: BDDK_CORPUS_PATH or corpus/bddk.sqlite)"
    )
    parser.add_argument("--json", action="store_true", help="Print score details as JSON")
    parser.add_argument("--include-report", action="store_true", help="Include full quality report in JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.db:
        report = asyncio.run(scan_db(args.corpus))
    else:
        report = scan_markdown_dir(args.md_dir)
    result = quality_score(report)
    if args.include_report:
        result["report"] = format_report_json(report)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"{result['score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
