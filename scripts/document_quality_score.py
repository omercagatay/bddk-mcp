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
from collections import defaultdict
from pathlib import Path

import asyncpg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bddk_mcp.core.config import require_database_url  # noqa: E402
from bddk_mcp.db_transport import assert_database_transport  # noqa: E402
from bddk_mcp.quality.markdown_quality import assess_markdown_quality  # noqa: E402
from bddk_mcp.quality.quality_scan import (  # noqa: E402
    AnomalyCount,
    DocumentFinding,
    MethodBreakdown,
    QualityReport,
    format_report_json,
    scan_quality,
)
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


async def scan_db(dsn: str | None) -> QualityReport:
    selected_dsn = assert_database_transport(dsn) if dsn else require_database_url()
    pool = await asyncpg.create_pool(selected_dsn, min_size=1, max_size=3)
    try:
        return await scan_quality(pool)
    finally:
        await pool.close()


def scan_seed_dir(seed_dir: Path) -> QualityReport:
    documents_path = seed_dir / "documents.json"
    chunks_path = seed_dir / "chunks.json"
    documents = json.loads(documents_path.read_text(encoding="utf-8"))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8")) if chunks_path.exists() else []

    method_counts: dict[str, list[int]] = defaultdict(list)
    findings: list[DocumentFinding] = []
    for doc in documents:
        markdown = doc.get("markdown_content") or ""
        method = doc.get("extraction_method") or "unknown"
        method_counts[method].append(len(markdown))
        quality = assess_markdown_quality(markdown, document_id=doc.get("document_id") or "")
        if quality.label == "clean":
            continue
        findings.append(
            DocumentFinding(
                document_id=doc.get("document_id") or "",
                label=quality.label,
                flags=quality.flags,
                counts=quality.counts,
                sample=" ".join(markdown.split())[:160],
            )
        )

    methods = [
        MethodBreakdown(
            method=method,
            doc_count=len(lengths),
            avg_chars=round(sum(lengths) / max(len(lengths), 1)),
        )
        for method, lengths in sorted(method_counts.items(), key=lambda item: (-len(item[1]), item[0]))
    ]

    doc_ids = {doc.get("document_id") for doc in documents}
    chunk_doc_ids = {chunk.get("doc_id") for chunk in chunks}
    orphan_chunks = sum(1 for chunk in chunks if chunk.get("doc_id") not in doc_ids)
    docs_without_chunks = sum(
        1
        for doc in documents
        if len(doc.get("markdown_content") or "") > 500 and doc.get("document_id") not in chunk_doc_ids
    )

    flag_docs: dict[str, list[str]] = defaultdict(list)
    for finding in findings:
        for flag in finding.flags:
            flag_docs[flag].append(finding.document_id)
    anomalies = [
        AnomalyCount(
            name=flag,
            docs_flagged=len(doc_ids_for_flag),
            description=f"{flag} detected by markdown quality assessment",
            sample_doc_ids=doc_ids_for_flag[:5],
        )
        for flag, doc_ids_for_flag in sorted(flag_docs.items())
    ]

    return QualityReport(
        total_documents=len(documents),
        methods=methods,
        anomalies=anomalies,
        orphan_chunks=orphan_chunks,
        docs_without_chunks=docs_without_chunks,
        document_findings=findings,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute BDDK document-quality score.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--db", action="store_true", help="Scan documents from PostgreSQL")
    source.add_argument("--md-dir", type=Path, help="Scan local Markdown export directory")
    source.add_argument("--seed-dir", type=Path, help="Scan seed_data documents.json/chunks.json")
    parser.add_argument("--dsn", help="Override BDDK_DATABASE_URL")
    parser.add_argument("--json", action="store_true", help="Print score details as JSON")
    parser.add_argument("--include-report", action="store_true", help="Include full quality report in JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.db:
        report = asyncio.run(scan_db(args.dsn))
    elif args.seed_dir:
        report = scan_seed_dir(args.seed_dir)
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
