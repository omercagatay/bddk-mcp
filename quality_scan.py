"""
Document quality scan engine.

Runs SQL-side regex checks against the `documents` table to flag common
extraction anomalies: missing whitespace (camelCase concatenation),
replacement characters, leaked HTML, truncation, formula references
without formulas, and diacritic-stripping outliers.

The scan is read-only. Counts are cheap (single-pass regex over the
markdown_content column via PostgreSQL).
"""

from __future__ import annotations

import asyncpg
from pydantic import BaseModel, Field


class AnomalyCount(BaseModel):
    name: str
    docs_flagged: int
    description: str
    sample_doc_ids: list[str] = Field(default_factory=list)


class MethodBreakdown(BaseModel):
    method: str
    doc_count: int
    avg_chars: int


class QualityReport(BaseModel):
    total_documents: int
    methods: list[MethodBreakdown]
    anomalies: list[AnomalyCount]
    orphan_chunks: int
    docs_without_chunks: int


_SAMPLE_LIMIT = 5


async def scan_quality(pool: asyncpg.Pool) -> QualityReport:
    """Run all anomaly checks against the documents table."""
    total = await pool.fetchval("SELECT COUNT(*) FROM documents")

    method_rows = await pool.fetch(
        """
        SELECT extraction_method AS method,
               COUNT(*)          AS doc_count,
               AVG(LENGTH(markdown_content))::int AS avg_chars
        FROM documents
        GROUP BY extraction_method
        ORDER BY doc_count DESC
        """
    )
    methods = [
        MethodBreakdown(method=r["method"] or "unknown", doc_count=r["doc_count"], avg_chars=r["avg_chars"] or 0)
        for r in method_rows
    ]

    anomalies: list[AnomalyCount] = []

    checks = [
        (
            "replacement_char",
            "Contains U+FFFD replacement characters (encoding issue)",
            "markdown_content ~ E'\\uFFFD'",
        ),
        (
            "leaked_img_tag",
            "Raw <img> tag leaked into markdown",
            "markdown_content ~* '<img[[:space:]>]'",
        ),
        (
            "leaked_html_block",
            "Raw <div>/<table>/<tr>/<td> tag leaked into markdown",
            "markdown_content ~* '<(div|table|tr|td)[[:space:]>]'",
        ),
        (
            "short_content",
            "Extraction output is <500 chars (likely truncation)",
            "LENGTH(markdown_content) < 500",
        ),
        (
            "long_dot_run",
            "Contains dot-leader runs (>=10 dots) — likely TOC artifact",
            r"markdown_content ~ '\.{10,}'",
        ),
        (
            "camelcase_concat",
            "Adjacent lowercase+uppercase with no separator (html whitespace loss)",
            # Turkish-aware: a lowercase/turkish letter directly followed by
            # an uppercase letter indicates two words mashed together.
            r"markdown_content ~ '[a-zçğıöşü][A-ZÇĞİÖŞÜ]'",
        ),
        (
            "formula_ref_without_formula",
            "References a formula (`formül`) but emits no LaTeX `$$...$$` block",
            "markdown_content ~* 'formül' AND markdown_content !~ E'\\\\$\\\\$'",
        ),
    ]

    for name, description, predicate in checks:
        count = await pool.fetchval(f"SELECT COUNT(*) FROM documents WHERE {predicate}")
        samples = []
        if count:
            sample_rows = await pool.fetch(
                f"SELECT document_id FROM documents WHERE {predicate} ORDER BY document_id LIMIT $1",
                _SAMPLE_LIMIT,
            )
            samples = [r["document_id"] for r in sample_rows]
        anomalies.append(AnomalyCount(name=name, docs_flagged=count, description=description, sample_doc_ids=samples))

    diacritic_count = await pool.fetchval(
        """
        SELECT COUNT(*) FROM documents
        WHERE LENGTH(markdown_content) > 1000
          AND (LENGTH(markdown_content) - LENGTH(REGEXP_REPLACE(markdown_content, '[çğıöşüÇĞİÖŞÜ]', '', 'g')))::float
              / NULLIF(LENGTH(REGEXP_REPLACE(markdown_content, '[^A-Za-zÇĞİÖŞÜçğıöşü]', '', 'g')), 0)
              < 0.06
        """
    )
    diacritic_samples = []
    if diacritic_count:
        sample_rows = await pool.fetch(
            """
            SELECT document_id FROM documents
            WHERE LENGTH(markdown_content) > 1000
              AND (LENGTH(markdown_content) - LENGTH(REGEXP_REPLACE(markdown_content, '[çğıöşüÇĞİÖŞÜ]', '', 'g')))::float
                  / NULLIF(LENGTH(REGEXP_REPLACE(markdown_content, '[^A-Za-zÇĞİÖŞÜçğıöşü]', '', 'g')), 0)
                  < 0.06
            ORDER BY document_id LIMIT $1
            """,
            _SAMPLE_LIMIT,
        )
        diacritic_samples = [r["document_id"] for r in sample_rows]
    anomalies.append(
        AnomalyCount(
            name="diacritic_outlier",
            docs_flagged=diacritic_count,
            description="Turkish diacritic ratio <6% (expected 8-12%) — possible OCR stripping",
            sample_doc_ids=diacritic_samples,
        )
    )

    orphan_chunks = await pool.fetchval(
        """
        SELECT COUNT(*) FROM document_chunks c
        WHERE NOT EXISTS (SELECT 1 FROM documents d WHERE d.document_id = c.doc_id)
        """
    )
    docs_without_chunks = await pool.fetchval(
        """
        SELECT COUNT(*) FROM documents d
        WHERE LENGTH(d.markdown_content) > 500
          AND NOT EXISTS (SELECT 1 FROM document_chunks c WHERE c.doc_id = d.document_id)
        """
    )

    return QualityReport(
        total_documents=total,
        methods=methods,
        anomalies=anomalies,
        orphan_chunks=orphan_chunks or 0,
        docs_without_chunks=docs_without_chunks or 0,
    )


def format_report(report: QualityReport) -> str:
    """Render a QualityReport as a human-readable markdown block."""
    lines = [
        "**BDDK Document Quality Report**",
        "",
        f"Corpus: **{report.total_documents} documents**",
        "",
        "**Extraction method distribution**",
        "",
        f"  {'Method':<28} {'Docs':>6} {'Avg chars':>10}",
        "  " + "-" * 48,
    ]
    for m in report.methods:
        lines.append(f"  {m.method:<28} {m.doc_count:>6} {m.avg_chars:>10,}")

    lines.append("")
    lines.append("**Anomaly scan**")
    lines.append("")
    lines.append(f"  {'Signal':<32} {'Docs':>6}  Description")
    lines.append("  " + "-" * 96)

    for a in report.anomalies:
        marker = "  " if a.docs_flagged == 0 else "! "
        lines.append(f"{marker}{a.name:<32} {a.docs_flagged:>6}  {a.description}")
        if a.sample_doc_ids:
            lines.append(f"    samples: {', '.join(a.sample_doc_ids)}")

    lines.append("")
    lines.append("**Chunk integrity**")
    lines.append(f"  Orphan chunks (no parent doc): {report.orphan_chunks}")
    lines.append(f"  Docs >500 chars missing chunks: {report.docs_without_chunks}")

    flagged_signals = [a for a in report.anomalies if a.docs_flagged > 0]
    if flagged_signals:
        lines.append("")
        lines.append(
            f"**{len(flagged_signals)} anomaly signal(s) firing.** "
            "Inspect samples and trace each to its extraction method. "
            "Counts are indicators, not guarantees of defect."
        )
    else:
        lines.append("")
        lines.append("All anomaly signals are clean.")

    return "\n".join(lines)
