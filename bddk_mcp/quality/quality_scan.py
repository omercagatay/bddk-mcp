"""
Document quality scan engine.

Runs regex checks over every document body to flag common extraction
anomalies: replacement characters, C1 control characters (cp1254 punctuation
decoded as Latin-1), leaked HTML, truncation, and diacritic-stripping
outliers. Missing whitespace (camelCase concatenation) and formula references
without formulas come from the per-document quality assessment, which knows
the reviewed literal identifiers and inline formulas that a bare regex flags.

The scan is read-only and runs in one pass over the corpus file.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Callable

from pydantic import BaseModel, Field

from bddk_mcp.quality.markdown_quality import MOJIBAKE_RE, assess_markdown_quality
from bddk_mcp.store.corpus_db import CorpusDB


class AnomalyCount(BaseModel):
    name: str
    docs_flagged: int
    description: str
    sample_doc_ids: list[str] = Field(default_factory=list)


class MethodBreakdown(BaseModel):
    method: str
    doc_count: int
    avg_chars: int


class DocumentFinding(BaseModel):
    document_id: str
    label: str
    flags: list[str] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    sample: str = ""


class QualityReport(BaseModel):
    total_documents: int
    methods: list[MethodBreakdown]
    anomalies: list[AnomalyCount]
    orphan_chunks: int
    docs_without_chunks: int
    document_findings: list[DocumentFinding] = Field(default_factory=list)
    # Assessment flag -> number of documents carrying it, counting documents whose label is clean.
    signal_docs: dict[str, int] = Field(default_factory=dict)


_SAMPLE_LIMIT = 5
_C1_RE = re.compile(r"[\u0080-\u009f]")
_IMG_RE = re.compile(r"<img[\s>]", re.IGNORECASE)
_HTML_BLOCK_RE = re.compile(r"<(div|table|tr|td)[\s>]", re.IGNORECASE)
_DOT_RUN_RE = re.compile(r"\.{10,}")
_TURKISH_DIACRITIC_RE = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
_LETTER_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü]")


def _diacritic_outlier(markdown: str) -> bool:
    if len(markdown) <= 1000:
        return False
    letters = len(_LETTER_RE.findall(markdown))
    if not letters:
        return False
    return len(_TURKISH_DIACRITIC_RE.findall(markdown)) / letters < 0.06


_CHECKS: tuple[tuple[str, str, Callable[[str, bool], bool]], ...] = (
    ("replacement_char", "Contains U+FFFD replacement characters (encoding issue)", lambda md, _s: "�" in md),
    (
        "c1_control_char",
        "Contains C1 control characters U+0080-U+009F (cp1254 punctuation decoded as Latin-1)",
        lambda md, _s: _C1_RE.search(md) is not None,
    ),
    ("leaked_img_tag", "Raw <img> tag leaked into markdown", lambda md, _s: _IMG_RE.search(md) is not None),
    (
        "data_uri_image",
        "Embedded data:image/... payload leaked into markdown",
        lambda md, _s: "data:image/" in md.lower(),
    ),
    (
        "wmf_data_uri",
        "Embedded data:image/x-wmf payload leaked into markdown",
        lambda md, _s: "data:image/x-wmf" in md.lower(),
    ),
    ("cid_marker", "CID image/formula markers leaked into markdown", lambda md, _s: "cid:" in md.lower()),
    (
        "leaked_html_block",
        "Raw <div>/<table>/<tr>/<td> tag leaked into markdown",
        lambda md, _s: _HTML_BLOCK_RE.search(md) is not None,
    ),
    (
        "mojibake",
        "Double-encoded Turkish text (UTF-8 bytes decoded as cp1252/Latin-1, for example Ã¶ or ÄŸ)",
        lambda md, _s: MOJIBAKE_RE.search(md) is not None,
    ),
    ("short_content", "Extraction output is <500 chars (likely truncation)", lambda md, _s: len(md) < 500),
    (
        "long_dot_run",
        "Contains dot-leader runs (>=10 dots) — likely TOC artifact",
        lambda md, _s: _DOT_RUN_RE.search(md) is not None,
    ),
    (
        "zero_sections_despite_content",
        "Substantial content but zero parsed sections — invisible to section search",
        lambda md, has_sections: len(md) > 1000 and not has_sections,
    ),
    (
        "diacritic_outlier",
        "Turkish diacritic ratio <6% (expected 8-12%) — possible OCR stripping",
        lambda md, _s: _diacritic_outlier(md),
    ),
)


async def scan_quality(db: CorpusDB) -> QualityReport:
    """Run all anomaly checks against the documents in the corpus."""

    def _load(conn):
        documents = conn.execute(
            "SELECT document_id, markdown_content, extraction_method FROM documents ORDER BY document_id"
        ).fetchall()
        sectioned = {row[0] for row in conn.execute("SELECT DISTINCT doc_id FROM document_sections")}
        chunked = {row[0] for row in conn.execute("SELECT DISTINCT doc_id FROM chunks")}
        orphan_chunks = conn.execute(
            "SELECT COUNT(*) FROM chunks AS c WHERE NOT EXISTS "
            "(SELECT 1 FROM documents AS d WHERE d.document_id = c.doc_id)"
        ).fetchone()[0]
        return [dict(row) for row in documents], sectioned, chunked, orphan_chunks

    documents, sectioned, chunked, orphan_chunks = await db.run(_load)

    method_stats: dict[str, list[int]] = {}
    for document in documents:
        method_stats.setdefault(document["extraction_method"] or "unknown", []).append(
            len(document["markdown_content"] or "")
        )
    methods = sorted(
        (
            MethodBreakdown(method=method, doc_count=len(lengths), avg_chars=int(sum(lengths) / len(lengths)))
            for method, lengths in method_stats.items()
        ),
        key=lambda item: -item.doc_count,
    )

    flagged: dict[str, list[str]] = {name: [] for name, _description, _check in _CHECKS}
    document_findings: list[DocumentFinding] = []
    signal_docs: dict[str, int] = {}
    camelcase_ids: list[str] = []
    formula_ids: list[str] = []
    docs_without_chunks = 0
    for document in documents:
        document_id = document["document_id"]
        markdown = document["markdown_content"] or ""
        for name, _description, check in _CHECKS:
            if check(markdown, document_id in sectioned):
                flagged[name].append(document_id)
        if len(markdown) > 500 and document_id not in chunked:
            docs_without_chunks += 1
        if not markdown:
            continue
        quality = assess_markdown_quality(markdown, document_id=document_id)
        if quality.counts["camelcase_concat"]:
            camelcase_ids.append(document_id)
        if quality.counts["formula_ref_without_latex_or_image"]:
            formula_ids.append(document_id)
        for flag in quality.flags:
            signal_docs[flag] = signal_docs.get(flag, 0) + 1
        if quality.label == "clean":
            continue
        document_findings.append(
            DocumentFinding(
                document_id=document_id,
                label=quality.label,
                flags=quality.flags,
                counts=quality.counts,
                sample=_sample_text(markdown),
            )
        )

    anomalies = [
        AnomalyCount(
            name=name,
            docs_flagged=len(flagged[name]),
            description=description,
            sample_doc_ids=flagged[name][:_SAMPLE_LIMIT],
        )
        for name, description, _check in _CHECKS
    ]
    anomalies.extend(
        AnomalyCount(name=name, docs_flagged=len(ids), description=description, sample_doc_ids=ids[:_SAMPLE_LIMIT])
        for name, description, ids in (
            (
                "camelcase_concat",
                "Adjacent lowercase+uppercase with no separator (html whitespace loss), excluding reviewed identifiers",
                camelcase_ids,
            ),
            (
                "formula_ref_without_formula",
                "References a formula (`formül`) but no LaTeX, image or inline formula is extracted near it",
                formula_ids,
            ),
        )
    )

    return QualityReport(
        total_documents=len(documents),
        methods=methods,
        anomalies=anomalies,
        orphan_chunks=orphan_chunks,
        docs_without_chunks=docs_without_chunks,
        document_findings=document_findings,
        signal_docs=dict(sorted(signal_docs.items())),
    )


def known_signal_names() -> frozenset[str]:
    """Every signal name a report can show: scan anomalies, assessment flags and the registry flag."""
    names = {name for name, _description, _check in _CHECKS}
    names.update({"camelcase_concat", "formula_ref_without_formula"})
    names.update(assess_markdown_quality("").counts)
    names.add("configured_quality_failure")
    return frozenset(names)


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

    if report.document_findings:
        lines.append("")
        lines.append("**Document findings**")
        lines.append("")
        lines.append(f"  {'Document ID':<24} {'Label':<8} Flags")
        lines.append("  " + "-" * 88)
        label_rank = {"fail": 0, "warning": 1, "clean": 2}
        findings = sorted(
            report.document_findings,
            key=lambda f: (label_rank.get(f.label, 9), f.document_id),
        )
        for finding in findings[:20]:
            lines.append(f"  {finding.document_id:<24} {finding.label:<8} {', '.join(finding.flags)}")
        if len(findings) > 20:
            lines.append(f"  ... and {len(findings) - 20} more finding(s)")

    if report.signal_docs:
        lines.append("")
        lines.append("**Assessment flags (all documents)**")
        lines.append("")
        lines.append(f"  {'Flag':<40} {'Docs':>6}")
        lines.append("  " + "-" * 48)
        for name, count in sorted(report.signal_docs.items()):
            lines.append(f"  {name:<40} {count:>6}")

    flagged_signals = [a for a in report.anomalies if a.docs_flagged > 0]
    if flagged_signals:
        lines.append("")
        lines.append(
            f"**{len(flagged_signals)} anomaly signal(s) firing.** "
            "Inspect samples and trace each to its extraction method. "
            "Counts are indicators, not guarantees of defect."
        )
    elif report.docs_without_chunks or report.orphan_chunks:
        lines.append("")
        lines.append("No anomaly signal fires, but the chunk integrity counts above are not zero.")
    else:
        lines.append("")
        lines.append("All anomaly signals are clean.")

    return "\n".join(lines)


def format_report_json(report: QualityReport) -> dict:
    """Render a QualityReport as a JSON-serializable dictionary."""
    return report.model_dump()


def format_report_csv(report: QualityReport) -> str:
    """Render document-level quality findings as CSV."""
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["document_id", "label", "flags", "sample"])
    for finding in report.document_findings:
        writer.writerow(
            [
                finding.document_id,
                finding.label,
                "|".join(finding.flags),
                finding.sample,
            ]
        )
    return output.getvalue()


def _sample_text(markdown: str, limit: int = 160) -> str:
    """Return a compact one-line sample for quality reports."""
    return " ".join(markdown.split())[:limit]
