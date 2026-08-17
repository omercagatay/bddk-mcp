"""Contract tests for the live official-MCP Phase 2 harness."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from mcp.types import CallToolResult, TextContent, Tool

from bddk_mcp.tools.registry import PUBLIC_TOOL_NAMES
from benchmark.graders import ModelGrade
from benchmark.phase2_e2e import (
    BenchmarkProtocolError,
    LiveMcpContract,
    McpEndpoint,
    McpToolInvocationError,
    _audit_artifacts,
    _corpus_identity,
    _dataset_identity,
    _list_all_tools,
    _read_active_corpus_release,
    _run_agent_loop,
    _stdio_subprocess_env,
    _tool_result_record,
    _tool_result_text,
    _validated_corpus_manifest_identity,
    open_mcp_session,
    run_phase2,
)
from benchmark.test_cases import TestCase as BenchmarkTestCase

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
E2E_SUPPORT = Path(__file__).resolve().parent / "e2e_support"


@pytest.fixture(autouse=True)
def _tracked_corpus_trust_key(monkeypatch):
    # The tracked corpus manifest is Ed25519-signed; identity validation needs
    # the repository trust anchor supplied outside the corpus root.
    monkeypatch.setenv(
        "BDDK_CORPUS_TRUSTED_SIGNING_KEY",
        str(REPOSITORY_ROOT / "deploy" / "trust" / "corpus-signing-public-key.pem"),
    )


def _contract() -> LiveMcpContract:
    return LiveMcpContract(
        tools=(
            Tool(
                name="search_document_store",
                description="Search the local corpus.",
                inputSchema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
        ),
        server_name="bddk-mcp",
        server_version="test",
        protocol_version="2025-11-25",
    )


def _response(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload, request=httpx.Request("POST", "http://llm/v1/chat/completions"))


def _active_release_resource(**overrides: str) -> SimpleNamespace:
    manifest = _validated_corpus_manifest_identity()
    payload = {
        "schema_version": "1.0",
        "status": "active",
        "release_id": "corpus_release_sha256_" + "a" * 64,
        "manifest_id": manifest["manifest_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "retrieval_profile_sha256": "b" * 64,
        **overrides,
    }
    return SimpleNamespace(contents=[SimpleNamespace(text=json.dumps(payload))])


def test_harness_has_no_nonstandard_call_tool_http_route():
    from benchmark import phase2_e2e

    assert "/call-tool" not in inspect.getsource(phase2_e2e)


@pytest.mark.asyncio
async def test_agent_loop_executes_official_session_call_and_returns_answer(monkeypatch):
    from benchmark import phase2_e2e

    monkeypatch.setattr(phase2_e2e, "MAX_TOOL_CALLS", 2)
    llm = AsyncMock()
    llm.post = AsyncMock(
        side_effect=[
            _response(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "search_document_store",
                                            "arguments": '{"query":"sermaye yeterliliği"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            ),
            _response({"choices": [{"message": {"role": "assistant", "content": "943 numaralı kaynak."}}]}),
        ]
    )
    session = AsyncMock()
    session.call_tool.return_value = CallToolResult(content=[TextContent(type="text", text="Document ID: 943\nİlke 5")])

    trace = await _run_agent_loop(llm, session, "model", "soru", _contract())

    assert trace["final_answer"] == "943 numaralı kaynak."
    assert trace["tool_calls"] == [{"name": "search_document_store", "args": {"query": "sermaye yeterliliği"}}]
    assert trace["tool_results"][0]["tool_name"] == "search_document_store"
    assert trace["tool_results"][0]["arguments"] == {"query": "sermaye yeterliliği"}
    assert trace["tool_results"][0]["structured_content"] is None
    assert trace["tool_results"][0]["model_content"] == "Document ID: 943\nİlke 5"
    assert len(trace["tool_results"][0]["model_content_sha256"]) == 64
    session.call_tool.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_loop_rejects_malformed_arguments_instead_of_calling_empty_object(monkeypatch):
    from benchmark import phase2_e2e

    monkeypatch.setattr(phase2_e2e, "MAX_TOOL_CALLS", 1)
    llm = AsyncMock()
    llm.post.return_value = _response(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "search_document_store",
                                    "arguments": "{not-json",
                                },
                            }
                        ]
                    }
                }
            ]
        }
    )
    session = AsyncMock()

    with pytest.raises(BenchmarkProtocolError, match="malformed tool arguments"):
        await _run_agent_loop(llm, session, "model", "soru", _contract())

    session.call_tool.assert_not_awaited()


def test_tool_error_never_becomes_model_visible_result_text():
    result = CallToolResult(
        isError=True,
        content=[TextContent(type="text", text="private upstream failure detail")],
    )

    with pytest.raises(McpToolInvocationError, match="isError=true") as exc_info:
        _tool_result_text(result)

    assert "private upstream" not in str(exc_info.value)


def test_structured_tool_result_record_retains_status_and_full_evidence():
    structured = {
        "schema_version": "1.0",
        "status": "ok",
        "text": "Belge 943",
        "evidence": [{"document_id": "943", "section_type": "ilke", "section_ref": "5"}],
    }
    result = CallToolResult(
        content=[TextContent(type="text", text="Belge 943")],
        structuredContent=structured,
    )

    record = _tool_result_record(result, tool_name="get_document_section", arguments={"document_id": "943"})

    assert record["structured_content"] == structured
    assert '"status": "ok"' in record["model_content"]


def test_stdio_child_environment_is_exactly_allowlisted_and_omits_provider_secrets():
    source = {
        "PATH": "/bin",
        "BDDK_DATABASE_URL": "postgresql://runtime:secret@db/bddk",
        "BDDK_MCP_E2E_STUB": "1",
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "OPENAI_API_KEY": "openai-secret",
        "BDDK_GRADER_MODEL": "external-model",
        "BDDK_BENCHMARK_MCP_TOKEN": "mcp-token",
        "UNRELATED_CORPORATE_SECRET": "do-not-copy",
    }

    child = _stdio_subprocess_env(source)

    assert child == {
        "BDDK_DATABASE_URL": "postgresql://runtime:secret@db/bddk",
        "BDDK_MCP_E2E_STUB": "1",
        "PATH": "/bin",
    }


def test_stdio_environment_fails_closed_if_sdk_broadens_its_implicit_defaults(monkeypatch):
    from benchmark import phase2_e2e

    monkeypatch.setattr(
        phase2_e2e,
        "DEFAULT_INHERITED_ENV_VARS",
        [*phase2_e2e.DEFAULT_INHERITED_ENV_VARS, "FUTURE_PROVIDER_TOKEN"],
    )

    with pytest.raises(BenchmarkProtocolError, match="exceeded the reviewed allowlist"):
        _stdio_subprocess_env({"FUTURE_PROVIDER_TOKEN": "secret"})


def test_audit_artifacts_keep_full_structured_output_but_redact_credentials():
    trace = {
        "tool_calls": [{"name": "search_document_store", "args": {"query": "token=secret-value"}}],
        "tool_results": [
            {
                "tool_name": "search_document_store",
                "arguments": {"query": "token=secret-value"},
                "structured_content": {
                    "schema_version": "1.0",
                    "status": "ok",
                    "text": "api_key=sk-secretsecretsecret",
                    "evidence": [{"document_id": "943", "content_hash": "a" * 64}],
                    "results": [{"document_id": "943", "snippet": "tam içerik"}],
                },
                "model_content": "ignored duplicate",
            }
        ],
        "final_answer": "Bearer abcdefghijklmnop ile 943 bulundu.",
    }

    artifacts = _audit_artifacts(trace)
    rendered = repr(artifacts)

    assert "secret-value" not in rendered
    assert "sk-secretsecretsecret" not in rendered
    assert "abcdefghijklmnop" not in rendered
    assert artifacts["tool_evidence"][0]["structured_content"]["results"][0]["snippet"] == "tam içerik"
    assert artifacts["final_answer"].endswith("ile 943 bulundu.")
    assert len(artifacts["tool_trace_sha256"]) == 64
    assert len(artifacts["tool_evidence"][0]["artifact_sha256"]) == 64


def test_dataset_and_corpus_identities_are_stable_and_evidence_based(monkeypatch):
    case = BenchmarkTestCase(
        id="audit-1",
        question="İlke 5 nedir?",
        expected_tool="get_document_section",
        expected_params={"document_id": "943"},
    )
    first = _dataset_identity([case])
    second = _dataset_identity([case])
    monkeypatch.setenv("BDDK_BENCHMARK_CORPUS_ID", "bank-corpus-2026-07")
    monkeypatch.setenv("BDDK_BENCHMARK_CORPUS_SHA256", "b" * 64)
    corpus = _corpus_identity(
        [{"tool_evidence": [{"structured_content": {"evidence": [{"document_id": "943", "content_hash": "a" * 64}]}}]}]
    )

    assert first == second
    assert first["sha256"] != _dataset_identity([BenchmarkTestCase(id="audit-2", question="başka")])["sha256"]
    assert corpus["declared_id"] == "bank-corpus-2026-07"
    assert corpus["declared_sha256"] == "b" * 64
    assert corpus["observed_reference_count"] == 1
    assert len(corpus["observed_evidence_sha256"]) == 64


def test_benchmark_manifest_identity_is_verified_and_path_free():
    identity = _validated_corpus_manifest_identity()

    assert identity["manifest_id"] == "bddk-job-corpus-2026-08-14"
    assert identity["exhaustive"] is False
    assert len(identity["manifest_sha256"]) == 64
    assert len(identity["artifact_set_sha256"]) == 64
    assert identity["artifact_count"] == 3
    assert all("path" not in key for key in identity)


@pytest.mark.asyncio
async def test_phase2_result_retains_auditable_trace_and_separates_retrieval_comparability(monkeypatch):
    from benchmark import phase2_e2e

    case = BenchmarkTestCase(
        id="live-1",
        question="943 belgesini bul",
        expected_tool="search_document_store",
        expected_params={"query": "943"},
        expected_source_tools=["search_document_store"],
        expected_documents=["943"],
    )
    long_answer = "943 belgesi bulundu. " + "kanıt " * 120
    trace = {
        "tool_calls": [{"name": "search_document_store", "args": {"query": "943"}}],
        "tool_results": [
            {
                "tool_name": "search_document_store",
                "arguments": {"query": "943"},
                "structured_content": {
                    "schema_version": "1.0",
                    "status": "ok",
                    "text": "943 belgesi",
                    "evidence": [{"document_id": "943", "content_hash": "a" * 64}],
                    "results": [{"document_id": "943", "snippet": "kanıt"}],
                },
                "model_content": '{"document_id":"943"}',
            }
        ],
        "final_answer": long_answer,
        "steps": 1,
    }

    session = AsyncMock()
    session.read_resource = AsyncMock(return_value=_active_release_resource())

    @asynccontextmanager
    async def fake_session(_endpoint):
        yield session, _contract()

    monkeypatch.setattr(phase2_e2e, "open_mcp_session", fake_session)
    monkeypatch.setattr(phase2_e2e, "_run_agent_loop", AsyncMock(return_value=trace))
    monkeypatch.setattr(
        phase2_e2e,
        "model_grader",
        AsyncMock(return_value=ModelGrade(score=0.9, status="scored", model="grader-test")),
    )
    monkeypatch.setattr(
        phase2_e2e,
        "_git_state",
        lambda: {"commit_sha": "abc", "dirty": True, "status_sha256": "d" * 64, "change_entry_count": 1},
    )

    result = await run_phase2("model-test", cases=[case])
    detail = result["details"][0]

    assert detail["final_answer"] == long_answer
    assert detail["final_answer_sha256"]
    assert detail["tool_evidence"][0]["structured_content"]["evidence"][0]["document_id"] == "943"
    assert detail["retrieval_completion_success"] is True
    assert detail["expected_arguments_success"] is True
    assert detail["model_grader_model"] == "grader-test"
    assert result["retrieval_comparable_cases"] == 1
    assert result["run_metadata"]["git"]["dirty"] is True
    assert result["run_metadata"]["dataset_identity"]["case_ids"] == ["live-1"]
    assert result["run_metadata"]["corpus_identity"]["observed_reference_count"] == 1
    assert result["run_metadata"]["corpus_manifest"]["manifest_id"] == "bddk-job-corpus-2026-08-14"
    assert result["run_metadata"]["active_corpus_release"]["release_id"].startswith("corpus_release_sha256_")
    session.read_resource.assert_awaited()
    assert session.read_resource.await_count == 2


@pytest.mark.asyncio
async def test_external_grader_unavailable_does_not_turn_retrieval_into_transport_error(monkeypatch):
    from benchmark import phase2_e2e

    case = BenchmarkTestCase(
        id="retrieval-only",
        question="belgeyi bul",
        expected_tool="search_document_store",
        expected_source_tools=["search_document_store"],
    )
    trace = {
        "tool_calls": [{"name": "search_document_store", "args": {"query": "belge"}}],
        "tool_results": [
            {
                "tool_name": "search_document_store",
                "arguments": {"query": "belge"},
                "structured_content": {
                    "schema_version": "1.0",
                    "status": "ok",
                    "text": "belge",
                    "evidence": [{"document_id": "943"}],
                    "results": [{"document_id": "943", "snippet": "kanıt"}],
                },
                "model_content": "belge",
            }
        ],
        "final_answer": "Belge bulundu.",
        "steps": 1,
    }

    session = AsyncMock()
    session.read_resource = AsyncMock(return_value=_active_release_resource())

    @asynccontextmanager
    async def fake_session(_endpoint):
        yield session, _contract()

    monkeypatch.setattr(phase2_e2e, "open_mcp_session", fake_session)
    monkeypatch.setattr(phase2_e2e, "_run_agent_loop", AsyncMock(return_value=trace))
    monkeypatch.setattr(
        phase2_e2e,
        "model_grader",
        AsyncMock(
            return_value=ModelGrade(
                score=None,
                status="unavailable",
                model="grader-test",
                reason="external_egress_not_opted_in",
            )
        ),
    )

    result = await run_phase2("model-test", cases=[case])
    detail = result["details"][0]

    assert detail["comparable"] is False
    assert detail["retrieval_comparable"] is True
    assert detail["retrieval_completion_success"] is True
    assert detail["error"] is None
    assert detail["non_comparability_reason"] == "external_egress_not_opted_in"
    assert result["error_count"] == 0
    assert result["retrieval_completion_success_rate"] == 1.0
    assert result["avg_numeric_claim_support"] is None
    assert result["avg_retrieval_source_correctness"] is None


@pytest.mark.asyncio
async def test_active_release_attestation_rejects_unavailable_and_malformed_resources():
    unavailable = AsyncMock()
    unavailable.read_resource.return_value = SimpleNamespace(
        contents=[SimpleNamespace(text='{"schema_version":"1.0","status":"unavailable"}')]
    )
    with pytest.raises(BenchmarkProtocolError, match="no verified active corpus"):
        await _read_active_corpus_release(unavailable)

    malformed = AsyncMock()
    malformed.read_resource.return_value = SimpleNamespace(contents=[SimpleNamespace(text="not-json")])
    with pytest.raises(BenchmarkProtocolError, match="malformed JSON"):
        await _read_active_corpus_release(malformed)


@pytest.mark.asyncio
async def test_phase2_rejects_active_release_change_on_the_same_session(monkeypatch):
    from benchmark import phase2_e2e

    session = AsyncMock()
    session.read_resource = AsyncMock(
        side_effect=[
            _active_release_resource(),
            _active_release_resource(release_id="corpus_release_sha256_" + "c" * 64),
        ]
    )

    @asynccontextmanager
    async def fake_session(_endpoint):
        yield session, _contract()

    monkeypatch.setattr(phase2_e2e, "open_mcp_session", fake_session)

    with pytest.raises(BenchmarkProtocolError, match="changed during Phase 2"):
        await run_phase2("model-test", cases=[])

    assert session.read_resource.await_count == 2


@pytest.mark.asyncio
async def test_phase2_rejects_live_release_for_a_different_manifest(monkeypatch):
    from benchmark import phase2_e2e

    session = AsyncMock()
    session.read_resource = AsyncMock(return_value=_active_release_resource(manifest_sha256="d" * 64))

    @asynccontextmanager
    async def fake_session(_endpoint):
        yield session, _contract()

    monkeypatch.setattr(phase2_e2e, "open_mcp_session", fake_session)

    with pytest.raises(BenchmarkProtocolError, match="does not match"):
        await run_phase2("model-test", cases=[])

    session.read_resource.assert_awaited_once()


def test_live_contract_hash_covers_discovered_schema_and_is_stable():
    contract = _contract()

    assert len(contract.schema_hash) == 64
    assert contract.schema_hash == _contract().schema_hash
    assert contract.openai_tools[0]["function"]["parameters"]["additionalProperties"] is False


@pytest.mark.asyncio
async def test_tools_list_pagination_is_live_and_duplicate_names_fail():
    session = AsyncMock()
    tool = _contract().tools[0]
    session.list_tools = AsyncMock(
        side_effect=[
            SimpleNamespace(tools=[tool], nextCursor="next"),
            SimpleNamespace(tools=[tool], nextCursor=None),
        ]
    )

    with pytest.raises(BenchmarkProtocolError, match="duplicate tool names"):
        await _list_all_tools(session)

    assert session.list_tools.await_args_list[0].args == (None,)
    assert session.list_tools.await_args_list[1].args == ("next",)


@pytest.mark.asyncio
async def test_benchmark_stdio_endpoint_uses_real_initialize_and_tools_list(monkeypatch):
    executable = Path(sys.executable).with_name("bddk-mcp")
    assert executable.is_file()
    monkeypatch.setenv("BDDK_ADMIN_TOOLS", "false")
    monkeypatch.setenv("BDDK_AUTO_SYNC", "false")
    monkeypatch.setenv("BDDK_MCP_E2E_STUB", "1")
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join((str(E2E_SUPPORT), str(REPOSITORY_ROOT))))
    endpoint = McpEndpoint(
        transport="stdio",
        command=str(executable),
        args=("serve", "--transport", "stdio"),
        cwd=REPOSITORY_ROOT,
    )

    async with asyncio.timeout(20):
        async with open_mcp_session(endpoint) as (_session, contract):
            assert contract.server_name == "BDDK"
            assert contract.names == frozenset(PUBLIC_TOOL_NAMES)
            assert len(contract.schema_hash) == 64
