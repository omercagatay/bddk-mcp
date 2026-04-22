"""Reusable backfill engine for degraded mevzuat documents.

Scans the ``documents`` table for rows matching a corruption signature and
re-extracts them through ``DocumentSyncer.sync_document`` with ``force=True``.
When HTML-first routing is active (CPU-only deployments, no formula-capable
OCR backend), the rescue picks up the richer iframe → html_parser path rather
than re-running markitdown on the cached PDF that produced the degraded row
in the first place.

Both the CLI (``scripts/backfill_mevzuat.py``) and the MCP admin tool
(``tools/admin.py::backfill_degraded_documents``) import from here so a
single scan/execute implementation stays authoritative.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg

    from doc_sync import DocumentSyncer

logger = logging.getLogger(__name__)

# ── Scan SQL ─────────────────────────────────────────────────────────────────
# ``degraded_pdf`` = ``extraction_method = 'markitdown_degraded'`` — the only
# signature the current code path produces. The legacy signatures catch
# historical rows stored with replacement characters, leaked ``<img`` tags
# (from an earlier BeautifulSoup extractor), or suspiciously short content.

_SCAN_DEGRADED_ONLY_SQL = """
SELECT d.document_id,
       d.title,
       d.source_url,
       d.category,
       d.decision_date,
       d.decision_number,
       LENGTH(d.markdown_content) AS len,
       'markitdown_degraded'      AS signature
FROM documents d
WHERE d.document_id LIKE 'mevzuat_%'
  AND d.extraction_method = 'markitdown_degraded'
ORDER BY d.document_id
"""

# Extraction-artifact signatures added 2026-04-22 for the SYSTEMIC-1/3/8
# repair pass (see error_reports.md). Unlike the mevzuat-only legacy
# signatures, these can appear in plain BDDK (numeric) doc IDs too — Đ
# garble in particular is a PDF-font decoding artifact common to both
# catalogs — so the outer WHERE deliberately does not gate them on the
# ``mevzuat_%`` prefix.
_SCAN_ALL_CORRUPTION_SQL = """
SELECT d.document_id,
       d.title,
       d.source_url,
       d.category,
       d.decision_date,
       d.decision_number,
       LENGTH(d.markdown_content) AS len,
       CASE
           WHEN d.markdown_content LIKE '%Đ%'                          THEN 'i_garble'
           WHEN d.markdown_content LIKE '%' || chr(12) || '%'          THEN 'form_feeds'
           WHEN d.markdown_content ~ '[-]'                 THEN 'c1_controls'
           WHEN d.markdown_content LIKE '%' || chr(65533) || '%'       THEN 'ufffd'
           WHEN d.markdown_content LIKE '%<img%'                       THEN 'leaked_img'
           WHEN LENGTH(d.markdown_content) < 500                       THEN 'too_short'
           WHEN d.extraction_method = 'markitdown_degraded'            THEN 'markitdown_degraded'
       END AS signature
FROM documents d
WHERE d.markdown_content LIKE '%Đ%'
   OR d.markdown_content LIKE '%' || chr(12) || '%'
   OR d.markdown_content ~ '[-]'
   OR (d.document_id LIKE 'mevzuat_%'
       AND (d.markdown_content LIKE '%' || chr(65533) || '%'
            OR d.markdown_content LIKE '%<img%'
            OR LENGTH(d.markdown_content) < 500
            OR d.extraction_method = 'markitdown_degraded'))
ORDER BY d.document_id
"""


@dataclass
class BackfillCandidate:
    document_id: str
    title: str
    source_url: str
    category: str
    decision_date: str
    decision_number: str
    len: int
    signature: str


@dataclass
class BackfillOutcome:
    """Per-document result of a backfill attempt."""

    document_id: str
    success: bool
    method: str = ""
    size_bytes: int = 0
    error: str = ""
    elapsed_seconds: float = 0.0


@dataclass
class BackfillReport:
    total: int
    ok: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def summary(self) -> str:
        lines = [f"Backfill: {len(self.ok)} ok, {len(self.failed)} failed ({self.elapsed_seconds:.1f}s)"]
        if self.failed:
            lines.append("Failed:")
            for doc_id, reason in self.failed[:20]:
                lines.append(f"  {doc_id}: {reason}")
            if len(self.failed) > 20:
                lines.append(f"  ... and {len(self.failed) - 20} more")
        return "\n".join(lines)


async def scan_candidates(
    pool: asyncpg.Pool,
    *,
    include_legacy_corruption: bool = False,
    limit: int = 0,
) -> list[BackfillCandidate]:
    """Return rows that match the backfill scan signature.

    Args:
        pool: asyncpg pool bound to the target database.
        include_legacy_corruption: When True, also match rows with U+FFFD
            replacement chars, leaked ``<img`` tags, or content shorter than
            500 chars. Default (False) scans only ``markitdown_degraded`` —
            the signature the current extractor emits — so we don't drag
            unrelated legacy corruption into a routine rescue.
        limit: Maximum rows to return (0 = no cap).
    """
    sql = _SCAN_ALL_CORRUPTION_SQL if include_legacy_corruption else _SCAN_DEGRADED_ONLY_SQL
    rows = await pool.fetch(sql)
    candidates = [
        BackfillCandidate(
            document_id=r["document_id"],
            title=r["title"] or "",
            source_url=r["source_url"] or "",
            category=r["category"] or "",
            decision_date=r["decision_date"] or "",
            decision_number=r["decision_number"] or "",
            len=r["len"],
            signature=r["signature"],
        )
        for r in rows
    ]
    if limit > 0:
        candidates = candidates[:limit]
    return candidates


def group_by_signature(candidates: list[BackfillCandidate]) -> dict[str, int]:
    """Count candidates per signature — used for scan-summary reporting."""
    counts: dict[str, int] = {}
    for c in candidates:
        counts[c.signature] = counts.get(c.signature, 0) + 1
    return counts


ProgressCallback = Callable[[int, int, BackfillOutcome], Awaitable[None] | None]


async def execute_backfill(
    syncer: DocumentSyncer,
    candidates: list[BackfillCandidate],
    *,
    inter_request_delay: float = 2.0,
    on_progress: ProgressCallback | None = None,
) -> BackfillReport:
    """Re-extract each candidate serially via ``syncer.sync_document(force=True)``.

    Serial execution + a small inter-request delay is deliberate — we're hitting
    a government site (mevzuat.gov.tr) and parallel bursts have historically
    tripped rate limits. The caller is responsible for constructing the syncer
    with the desired routing flag (``prefer_html_for_mevzuat``).

    ``on_progress(index, total, outcome)`` is invoked after each document.
    """
    report = BackfillReport(total=len(candidates))
    start = time.monotonic()

    for i, cand in enumerate(candidates, 1):
        t0 = time.monotonic()
        try:
            result = await syncer.sync_document(
                doc_id=cand.document_id,
                title=cand.title,
                category=cand.category,
                source_url=cand.source_url,
                decision_date=cand.decision_date,
                decision_number=cand.decision_number,
                force=True,
            )
        except Exception as e:
            outcome = BackfillOutcome(
                document_id=cand.document_id,
                success=False,
                error=f"{type(e).__name__}: {e}",
                elapsed_seconds=time.monotonic() - t0,
            )
            report.failed.append((cand.document_id, outcome.error))
            logger.error("backfill %s failed: %s", cand.document_id, e)
        else:
            outcome = BackfillOutcome(
                document_id=cand.document_id,
                success=bool(result.success),
                method=result.method or "",
                size_bytes=result.size_bytes or 0,
                error=result.error or "",
                elapsed_seconds=time.monotonic() - t0,
            )
            if outcome.success:
                report.ok.append(cand.document_id)
                logger.info(
                    "backfill %s ok in %.1fs (method=%s, size=%dB)",
                    cand.document_id,
                    outcome.elapsed_seconds,
                    outcome.method,
                    outcome.size_bytes,
                )
            else:
                report.failed.append((cand.document_id, outcome.error or "unknown"))
                logger.warning("backfill %s failed: %s", cand.document_id, outcome.error)

        if on_progress is not None:
            cb_result = on_progress(i, len(candidates), outcome)
            if asyncio.iscoroutine(cb_result):
                await cb_result

        if i < len(candidates) and inter_request_delay > 0:
            await asyncio.sleep(inter_request_delay)

    report.elapsed_seconds = time.monotonic() - start
    return report
