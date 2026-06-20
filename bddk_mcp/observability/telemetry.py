"""Optional privacy-safe tool-call telemetry."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

from bddk_mcp.core.config import TELEMETRY_ENABLED, TELEMETRY_MODEL_ID, TELEMETRY_SESSION_ID, TELEMETRY_STORE_TEXT

logger = logging.getLogger(__name__)

_TEXT_ARG_KEYS = {"query", "keywords", "heading", "prompt", "question", "text"}
_SAFE_ARG_KEYS = {
    "active_only",
    "category",
    "column",
    "currency",
    "date_from",
    "date_to",
    "days",
    "document_id",
    "include_neighbors",
    "institution_type",
    "limit",
    "lookback_weeks",
    "metric_id",
    "month",
    "page",
    "page_number",
    "page_size",
    "party_code",
    "period",
    "section_ref",
    "section_type",
    "table_no",
    "year",
}


def elapsed_ms(start: float) -> int:
    """Return elapsed milliseconds since a perf_counter start timestamp."""
    return max(0, int((time.perf_counter() - start) * 1000))


def args_hash(args: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash of the full argument payload."""
    payload = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize_args(args: dict[str, Any], *, store_text: bool = False) -> dict[str, Any]:
    """Build a privacy-safe args summary.

    Text-like user inputs are hashed and length-counted by default. Raw text is
    only included when store_text=True, which is controlled by an explicit env var.
    """
    summary: dict[str, Any] = {}
    for key, value in sorted(args.items()):
        if value is None:
            continue
        if key in _TEXT_ARG_KEYS:
            summary[key] = _text_summary(str(value), include_value=store_text)
        elif key in _SAFE_ARG_KEYS or isinstance(value, (bool, int, float)):
            summary[key] = value
        elif isinstance(value, str):
            summary[key] = _text_summary(value, include_value=store_text) if len(value) > 80 else value
        else:
            summary[key] = {"type": type(value).__name__}
    return summary


def relevance_stats_from_hits(hits: list[dict]) -> dict[str, Any]:
    """Summarize relevance fields without storing snippets or queries."""
    if not hits:
        return {"result_count": 0}
    relevances = [float(hit.get("relevance", 0.0) or 0.0) for hit in hits]
    match_types = sorted({str(hit.get("match_type", "")) for hit in hits if hit.get("match_type")})
    return {
        "result_count": len(hits),
        "max_relevance": round(max(relevances), 4),
        "min_relevance": round(min(relevances), 4),
        "avg_relevance": round(sum(relevances) / len(relevances), 4),
        "match_types": match_types,
    }


def quality_labels_from_hits(hits: list[dict]) -> dict[str, dict[str, Any]]:
    """Collect quality labels/flags keyed by document ID from search hits."""
    labels: dict[str, dict[str, Any]] = {}
    for hit in hits:
        doc_id = hit.get("doc_id") or hit.get("document_id")
        if not doc_id:
            continue
        labels[str(doc_id)] = {
            "label": hit.get("quality_label", "unknown"),
            "flags": hit.get("quality_flags", []),
        }
    return labels


def unique_doc_ids(values: list[str | None]) -> list[str]:
    """Return doc IDs in first-seen order, dropping blanks."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


async def record_tool_call_trace(
    pool,
    *,
    tool_name: str,
    args: dict[str, Any],
    latency_ms: int,
    result_count: int | None = None,
    doc_ids: list[str] | None = None,
    quality_labels: dict[str, Any] | None = None,
    relevance_stats: dict[str, Any] | None = None,
    model_id: str | None = None,
    session_id: str | None = None,
) -> bool:
    """Persist a tool-call trace when telemetry is enabled.

    Telemetry is best-effort and never raises into the public tool path.
    """
    if not TELEMETRY_ENABLED or pool is None:
        return False

    try:
        await pool.execute(
            """
            INSERT INTO tool_call_traces (
                tool_name, args_hash, args_summary, latency_ms, result_count,
                doc_ids, quality_labels, relevance_stats, model_id, session_id
            )
            VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7::jsonb, $8::jsonb, $9, $10)
            """,
            tool_name,
            args_hash(args),
            json.dumps(summarize_args(args, store_text=TELEMETRY_STORE_TEXT), ensure_ascii=False),
            latency_ms,
            result_count,
            doc_ids or [],
            json.dumps(quality_labels or {}, ensure_ascii=False),
            json.dumps(relevance_stats or {}, ensure_ascii=False),
            model_id if model_id is not None else TELEMETRY_MODEL_ID or None,
            session_id if session_id is not None else TELEMETRY_SESSION_ID or None,
        )
        return True
    except Exception as exc:
        logger.debug("tool telemetry write failed for %s: %s", tool_name, exc)
        return False


def _text_summary(value: str, *, include_value: bool = False) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        "chars": len(value),
        "words": len(value.split()),
    }
    if include_value:
        summary["value"] = value
    return summary
