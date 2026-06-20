"""Tests for optional tool-call telemetry."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def test_summarize_args_redacts_query_text_by_default():
    from bddk_mcp.observability.telemetry import summarize_args

    query = "TFRS 9 kredi riskinde önemli artış"
    summary = summarize_args({"query": query, "limit": 5})
    serialized = json.dumps(summary, ensure_ascii=False)

    assert "TFRS 9" not in serialized
    assert summary["query"]["chars"] == len(query)
    assert len(summary["query"]["sha256"]) == 64
    assert summary["limit"] == 5


def test_summarize_args_can_store_text_when_explicitly_enabled():
    from bddk_mcp.observability.telemetry import summarize_args

    summary = summarize_args({"query": "TFRS 9 kredi riski"}, store_text=True)

    assert summary["query"]["value"] == "TFRS 9 kredi riski"
    assert summary["query"]["chars"] == 18


@pytest.mark.asyncio
async def test_record_tool_call_trace_is_disabled_by_default(monkeypatch):
    from bddk_mcp.observability import telemetry

    monkeypatch.setattr(telemetry, "TELEMETRY_ENABLED", False)
    pool = AsyncMock()

    recorded = await telemetry.record_tool_call_trace(
        pool,
        tool_name="search_document_store",
        args={"query": "secret prompt"},
        latency_ms=12,
        result_count=0,
    )

    assert recorded is False
    pool.execute.assert_not_called()


@pytest.mark.asyncio
async def test_record_tool_call_trace_persists_privacy_safe_summary(doc_store, monkeypatch):
    from bddk_mcp.observability import telemetry

    monkeypatch.setattr(telemetry, "TELEMETRY_ENABLED", True)
    monkeypatch.setattr(telemetry, "TELEMETRY_STORE_TEXT", False)

    recorded = await telemetry.record_tool_call_trace(
        doc_store._pool,
        tool_name="search_document_store",
        args={"query": "TFRS 9 kredi riski", "limit": 3},
        latency_ms=25,
        result_count=2,
        doc_ids=["943", "mevzuat_22599"],
        quality_labels={"943": {"label": "clean"}},
        relevance_stats={"max_relevance": 0.91},
        model_id="qwen-test",
        session_id="benchmark-run-1",
    )

    assert recorded is True
    row = await doc_store._pool.fetchrow("SELECT * FROM tool_call_traces WHERE tool_name = $1", "search_document_store")

    assert row["args_hash"]
    assert row["latency_ms"] == 25
    assert row["result_count"] == 2
    assert row["doc_ids"] == ["943", "mevzuat_22599"]
    assert row["model_id"] == "qwen-test"
    assert row["session_id"] == "benchmark-run-1"
    assert _json(row["quality_labels"]) == {"943": {"label": "clean"}}
    assert _json(row["relevance_stats"]) == {"max_relevance": 0.91}

    args_summary = _json(row["args_summary"])
    assert "TFRS 9" not in json.dumps(args_summary, ensure_ascii=False)
    assert args_summary["query"]["chars"] == 18
    assert args_summary["limit"] == 3
