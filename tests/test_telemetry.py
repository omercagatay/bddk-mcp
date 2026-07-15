"""Tests for optional tool-call telemetry."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest


def _json(value):
    return json.loads(value) if isinstance(value, str) else value


def _exact_telemetry_privileges() -> dict[str, bool]:
    return {
        "relation_exists": True,
        "sequence_exists": True,
        "session_is_current": True,
        "identity_hardened": True,
        "membership_isolated": True,
        "database_capabilities_isolated": True,
        "schema_usage": True,
        "application_schemas_isolated": True,
        "other_relations_isolated": True,
        "other_sequences_isolated": True,
        "application_functions_isolated": True,
        "can_insert_required_columns": True,
        "can_insert_managed_columns": False,
        "can_select": False,
        "can_update": False,
        "can_delete": False,
        "can_truncate": False,
        "sequence_usage": True,
        "sequence_select": False,
        "sequence_update": False,
    }


def test_telemetry_identity_inventory_includes_the_denied_legal_version_workspace() -> None:
    from bddk_mcp.observability.telemetry import _TELEMETRY_PRIVILEGES_SQL

    normalized_sql = " ".join(_TELEMETRY_PRIVILEGES_SQL.split())

    for relation in (
        "regulatory_instruments",
        "regulatory_family_imports",
        "regulatory_source_blobs",
        "regulatory_source_artifacts",
        "regulatory_evidence",
        "regulatory_legal_versions",
        "regulatory_legal_version_artifacts",
        "regulatory_legal_events",
        "regulatory_legal_status_assertions",
        "regulatory_provisions",
        "regulatory_legal_version_provisions",
        "regulatory_validated_section_citations",
    ):
        assert f"('public', '{relation}')" in normalized_sql
    for function_name in (
        "corpus_fingerprint_frame",
        "current_corpus_state_sha256",
        "corpus_retrieval_ready",
        "reject_corpus_release_mutation",
        "publish_verified_corpus_release",
        "resolve_regulation_status",
    ):
        assert f"'bddk_meta', '{function_name}'" in normalized_sql
    assert "to_regclass('bddk_meta" not in normalized_sql


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


def test_summarize_args_redacts_unknown_short_strings_fail_closed():
    from bddk_mcp.observability.telemetry import summarize_args

    summary = summarize_args({"future_user_field": "short secret"})

    assert summary["future_user_field"]["chars"] == 12
    assert summary["future_user_field"]["sha256"]
    assert "short secret" not in json.dumps(summary)


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
async def test_telemetry_identity_accepts_exact_insert_only_privileges():
    from bddk_mcp.observability.telemetry import assert_telemetry_writer_ready

    pool = AsyncMock()
    pool.fetchval.return_value = 170000
    pool.fetchrow.return_value = _exact_telemetry_privileges()

    await assert_telemetry_writer_ready(pool)

    query = pool.fetchrow.await_args.args[0]
    assert query.lstrip().startswith("WITH RECURSIVE target AS")
    assert "('public', 'document_retrieval_publications')" in query


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unexpected_privilege",
    [
        "can_select",
        "can_update",
        "can_delete",
        "can_truncate",
        "can_insert_managed_columns",
        "sequence_select",
        "sequence_update",
    ],
)
async def test_telemetry_identity_rejects_excess_table_privilege(unexpected_privilege):
    from bddk_mcp.observability.telemetry import TelemetryIdentityError, assert_telemetry_writer_ready

    privileges = _exact_telemetry_privileges()
    privileges[unexpected_privilege] = True
    pool = AsyncMock()
    pool.fetchval.return_value = 170000
    pool.fetchrow.return_value = privileges

    with pytest.raises(TelemetryIdentityError, match="INSERT-only"):
        await assert_telemetry_writer_ready(pool)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "required_isolation",
    [
        "session_is_current",
        "identity_hardened",
        "membership_isolated",
        "database_capabilities_isolated",
        "application_schemas_isolated",
        "other_relations_isolated",
        "other_sequences_isolated",
        "application_functions_isolated",
    ],
)
async def test_telemetry_identity_rejects_broader_identity_capabilities(required_isolation):
    from bddk_mcp.observability.telemetry import TelemetryIdentityError, assert_telemetry_writer_ready

    privileges = _exact_telemetry_privileges()
    privileges[required_isolation] = False
    pool = AsyncMock()
    pool.fetchval.return_value = 170000
    pool.fetchrow.return_value = privileges

    with pytest.raises(TelemetryIdentityError, match="INSERT-only"):
        await assert_telemetry_writer_ready(pool)


@pytest.mark.asyncio
async def test_telemetry_identity_refuses_unsupported_postgresql_before_privilege_inspection():
    from bddk_mcp.observability.telemetry import TelemetryIdentityError, assert_telemetry_writer_ready

    pool = AsyncMock()
    pool.fetchval.return_value = 160012

    with pytest.raises(TelemetryIdentityError) as exc_info:
        await assert_telemetry_writer_ready(pool)

    assert "requires PostgreSQL 17" in str(exc_info.value)
    assert "160012" not in str(exc_info.value)
    pool.fetchrow.assert_not_awaited()


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
