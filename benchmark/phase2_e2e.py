"""Phase 2: live end-to-end evaluation through the official MCP client.

The harness initializes one real stdio or Streamable HTTP session, discovers
the live tool contract, lets an OpenAI-compatible model select tools, executes
those calls through ``ClientSession.call_tool``, and grades only successful
MCP results.  Transport and tool failures are case failures, never synthetic
answer text.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import DEFAULT_INHERITED_ENV_VARS, stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, Tool

from benchmark.audit import canonical_sha256, sanitize_for_audit
from benchmark.config import LLM_BASE_URL, LLM_TEMPERATURE, LLM_TIMEOUT, MAX_TOOL_CALLS
from benchmark.gold_cases import gold_cases_as_test_cases
from benchmark.graders import (
    EXTERNAL_GRADER_OPT_IN_ENV,
    external_grader_opted_in,
    model_grader,
    numeric_claim_support_grader,
)
from benchmark.scoring import audit_grade_metrics
from benchmark.test_cases import TEST_CASES, TestCase

logger = logging.getLogger(__name__)
PHASE2_CASES = [*TEST_CASES, *gold_cases_as_test_cases()]

SYSTEM_PROMPT = (
    "Sen bir Türk bankacılık düzenleme uzmanısın. BDDK mevzuatı ve verileri hakkında "
    "sorulara cevap vermek için sana sağlanan araçları kullan.\n\n"
    "KRİTİK KURALLAR:\n"
    "- SADECE araç sonuçlarından gelen bilgileri kullan\n"
    "- Araç sonuçlarında olmayan bilgi EKLEME\n"
    "- Emin olmadığın konularda 'araç sonuçlarında bu bilgi yok' de\n"
    "- Sayısal verileri araç sonuçlarından aynen aktar"
)

McpTransport = Literal["streamable-http", "stdio"]

# Exact, reviewable environment passed to a benchmark-owned stdio server.
# Provider/grader credentials and benchmark bearer tokens are intentionally
# absent. Runtime database identities are included because they are required
# by the child server, but are never copied into result metadata.
STDIO_ENV_ALLOWLIST = frozenset(
    {
        "PATH",
        "HOME",
        "LOGNAME",
        "SHELL",
        "TERM",
        "USER",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TZ",
        "TMPDIR",
        "VIRTUAL_ENV",
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONUTF8",
        "PYTHONIOENCODING",
        "PYTHONHASHSEED",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HF_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "MCP_TRANSPORT",
        "MCP_HOST",
        "PORT",
        "BDDK_DATABASE_URL",
        "BDDK_OPERATOR_DATABASE_URL",
        "BDDK_TELEMETRY_DATABASE_URL",
        "BDDK_TELEMETRY_ENABLED",
        "BDDK_TELEMETRY_STORE_TEXT",
        "BDDK_TELEMETRY_MODEL_ID",
        "BDDK_TELEMETRY_SESSION_ID",
        "BDDK_TOOL_PROFILE",
        "BDDK_AUTO_SYNC",
        "BDDK_PG_POOL_MIN",
        "BDDK_PG_POOL_MAX",
        "BDDK_EMBEDDING_MODEL_PATH",
        "BDDK_EMBEDDING_MODEL",
        "BDDK_EMBEDDING_DIM",
        "BDDK_LIGHTOCR_MODEL_PATH",
        "BDDK_LIGHTOCR_MODEL",
        "BDDK_LIGHTOCR_DEVICE",
        "BDDK_OCR_MIN_CONTENT_LEN",
        "BDDK_CHANDRA_MODEL",
        "BDDK_PAGE_SIZE",
        "BDDK_EMBEDDING_CHUNK_SIZE",
        "BDDK_EMBEDDING_CHUNK_OVERLAP",
        "BDDK_EMBEDDING_CHUNK_MODE",
        "BDDK_EMBEDDING_CHUNK_TARGET_TOKENS",
        "BDDK_EMBEDDING_CHUNK_TOKEN_OVERLAP",
        "BDDK_CACHE_TTL",
        "BDDK_SEARCH_CACHE_TTL",
        "BDDK_SEARCH_CACHE_MAX",
        "BDDK_STALE_CACHE_FALLBACK",
        "BDDK_SEMANTIC_THRESHOLD",
        "BDDK_FTS_THRESHOLD",
        "BDDK_HYBRID_SEARCH",
        "BDDK_RRF_K",
        "BDDK_RERANKER",
        "BDDK_RERANKER_MODEL",
        "BDDK_RERANKER_MODEL_PATH",
        "BDDK_RERANKER_TOP_N",
        "BDDK_REQUEST_TIMEOUT",
        "BDDK_HTTP_CONNECT_TIMEOUT",
        "BDDK_HTTP_POOL_TIMEOUT",
        "BDDK_MAX_RETRIES",
        "BDDK_SYNC_CONCURRENCY",
        "BDDK_OPERATOR_JOB_DRAIN_TIMEOUT",
        "BDDK_OPERATOR_JOB_HISTORY",
        "BDDK_PREFER_HTML_FOR_MEVZUAT",
        # Test-only official-client subprocess fixture; harmless in production.
        "BDDK_MCP_E2E_STUB",
    }
)


class BenchmarkProtocolError(RuntimeError):
    """A model, MCP, or benchmark contract was malformed."""


class McpToolInvocationError(RuntimeError):
    """A live MCP tool call returned an MCP-level error."""


@dataclass(frozen=True, slots=True)
class McpEndpoint:
    """Connection details that contain no persisted credentials."""

    transport: McpTransport = "streamable-http"
    url: str = "http://127.0.0.1:8000/mcp"
    command: str = "bddk-mcp"
    args: tuple[str, ...] = ("serve", "--profile", "public", "--transport", "stdio")
    cwd: Path | None = None
    bearer_token: str | None = None

    def __post_init__(self) -> None:
        if self.transport not in {"streamable-http", "stdio"}:
            raise ValueError("unsupported MCP benchmark transport")
        if self.transport == "streamable-http" and not self.url:
            raise ValueError("Streamable HTTP benchmark transport requires an MCP URL")
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio benchmark transport requires a command")


@dataclass(frozen=True, slots=True)
class LiveMcpContract:
    tools: tuple[Tool, ...]
    server_name: str
    server_version: str
    protocol_version: str

    @property
    def names(self) -> frozenset[str]:
        return frozenset(tool.name for tool in self.tools)

    @property
    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
            for tool in self.tools
        ]

    @property
    def schema_hash(self) -> str:
        payload = [tool.model_dump(mode="json", by_alias=True, exclude_none=True) for tool in self.tools]
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@asynccontextmanager
async def open_mcp_session(endpoint: McpEndpoint) -> AsyncIterator[tuple[ClientSession, LiveMcpContract]]:
    """Initialize an official MCP session and discover its complete tool list."""

    if endpoint.transport == "stdio":
        parameters = StdioServerParameters(
            command=endpoint.command,
            args=list(endpoint.args),
            env=_stdio_subprocess_env(),
            cwd=endpoint.cwd,
        )
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                initialized = await session.initialize()
                tools = await _list_all_tools(session)
                yield session, _live_contract(initialized, tools)
        return

    headers: dict[str, str] = {}
    if endpoint.bearer_token:
        headers["Authorization"] = f"Bearer {endpoint.bearer_token}"
    async with streamablehttp_client(
        endpoint.url,
        headers=headers,
        timeout=30,
        sse_read_timeout=300,
    ) as (read_stream, write_stream, _session_id):
        async with ClientSession(read_stream, write_stream) as session:
            initialized = await session.initialize()
            tools = await _list_all_tools(session)
            yield session, _live_contract(initialized, tools)


def _stdio_subprocess_env(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy only explicitly reviewed runtime variables into a stdio child."""

    unreviewed_sdk_defaults = set(DEFAULT_INHERITED_ENV_VARS) - STDIO_ENV_ALLOWLIST
    if unreviewed_sdk_defaults:
        raise BenchmarkProtocolError("MCP SDK stdio environment defaults exceeded the reviewed allowlist")
    environment = os.environ if source is None else source
    return {name: environment[name] for name in sorted(STDIO_ENV_ALLOWLIST) if name in environment}


async def _list_all_tools(session: ClientSession) -> tuple[Tool, ...]:
    tools: list[Tool] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    for _page in range(100):
        response = await session.list_tools(cursor)
        tools.extend(response.tools)
        cursor = response.nextCursor
        if cursor is None:
            break
        if cursor in seen_cursors:
            raise BenchmarkProtocolError("MCP tools/list repeated a pagination cursor")
        seen_cursors.add(cursor)
    else:
        raise BenchmarkProtocolError("MCP tools/list exceeded the pagination bound")

    names = [tool.name for tool in tools]
    if not names:
        raise BenchmarkProtocolError("MCP tools/list returned an empty contract")
    if len(names) != len(set(names)):
        raise BenchmarkProtocolError("MCP tools/list returned duplicate tool names")
    return tuple(tools)


def _live_contract(initialized: Any, tools: tuple[Tool, ...]) -> LiveMcpContract:
    server_info = getattr(initialized, "serverInfo", None)
    return LiveMcpContract(
        tools=tools,
        server_name=str(getattr(server_info, "name", "unknown")),
        server_version=str(getattr(server_info, "version", "unknown")),
        protocol_version=str(getattr(initialized, "protocolVersion", "unknown")),
    )


def _tool_result_text(result: CallToolResult) -> str:
    if result.isError:
        raise McpToolInvocationError("MCP tool returned isError=true")
    if result.structuredContent is not None:
        return json.dumps(result.structuredContent, ensure_ascii=False, sort_keys=True)

    rendered: list[str] = []
    for block in result.content:
        if getattr(block, "type", None) == "text":
            rendered.append(str(getattr(block, "text", "")))
        else:
            rendered.append(json.dumps(block.model_dump(mode="json", by_alias=True), ensure_ascii=False))
    text = "\n".join(part for part in rendered if part)
    if not text:
        raise BenchmarkProtocolError("MCP tool returned no usable content")
    return text


def _tool_result_record(
    result: CallToolResult,
    *,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Keep structured MCP evidence and the exact model-visible rendering."""

    model_content = _tool_result_text(result)
    structured = result.structuredContent
    return {
        "tool_name": tool_name,
        "arguments": arguments,
        "structured_content": structured if isinstance(structured, dict) else None,
        "model_content": model_content,
        "model_content_sha256": hashlib.sha256(model_content.encode("utf-8")).hexdigest(),
    }


def _parse_model_message(data: Any) -> dict[str, Any]:
    try:
        choices = data["choices"]
        message = choices[0]["message"]
    except (KeyError, IndexError, TypeError):
        raise BenchmarkProtocolError("LLM response did not contain choices[0].message") from None
    if not isinstance(message, dict):
        raise BenchmarkProtocolError("LLM response message was not an object")
    return message


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raise BenchmarkProtocolError("Model emitted malformed tool arguments JSON") from None
    if not isinstance(raw, dict):
        raise BenchmarkProtocolError("Model tool arguments were not an object")
    return raw


async def _run_agent_loop(
    llm_client: httpx.AsyncClient,
    mcp_session: ClientSession,
    model: str,
    question: str,
    contract: LiveMcpContract,
) -> dict[str, Any]:
    """Run one bounded model/tool loop against an initialized MCP session."""

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_result_records: list[dict[str, Any]] = []
    tool_calls_made: list[dict[str, Any]] = []

    # MAX_TOOL_CALLS is a call budget, not merely a model-turn budget.  One
    # model message can request several tools, so enforce it before execution.
    for step in range(MAX_TOOL_CALLS + 1):
        response = await llm_client.post(
            f"{LLM_BASE_URL}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "tools": contract.openai_tools,
                "stream": False,
                "temperature": LLM_TEMPERATURE,
            },
            timeout=LLM_TIMEOUT,
        )
        response.raise_for_status()
        message = _parse_model_message(response.json())
        requested_calls = message.get("tool_calls") or []

        if not requested_calls:
            answer = message.get("content", "")
            if answer is None:
                answer = ""
            if not isinstance(answer, str):
                raise BenchmarkProtocolError("LLM final answer was not text")
            return {
                "final_answer": answer,
                "tool_calls": tool_calls_made,
                "tool_results": tool_result_records,
                "steps": step,
            }

        if not isinstance(requested_calls, list):
            raise BenchmarkProtocolError("LLM tool_calls was not a list")
        if len(tool_calls_made) + len(requested_calls) > MAX_TOOL_CALLS:
            return {
                "final_answer": "",
                "tool_calls": tool_calls_made,
                "tool_results": tool_result_records,
                "steps": step,
                "truncated": True,
                "failure_code": "TOOL_CALL_BUDGET_EXCEEDED",
            }

        messages.append(message)
        for tool_call in requested_calls:
            try:
                call_id = str(tool_call["id"])
                function = tool_call["function"]
                tool_name = str(function["name"])
                tool_args = _parse_tool_arguments(function.get("arguments", {}))
            except (KeyError, TypeError):
                raise BenchmarkProtocolError("LLM tool call was missing id/function/name") from None
            if not call_id:
                raise BenchmarkProtocolError("LLM tool call ID was empty")
            if tool_name not in contract.names:
                raise BenchmarkProtocolError("Model requested a tool absent from live tools/list")

            tool_calls_made.append({"name": tool_name, "args": tool_args})
            result = await mcp_session.call_tool(
                tool_name,
                tool_args,
                read_timeout_seconds=timedelta(seconds=60),
            )
            result_record = _tool_result_record(
                result,
                tool_name=tool_name,
                arguments=tool_args,
            )
            tool_result_records.append(result_record)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": result_record["model_content"],
                }
            )

    return {
        "final_answer": "",
        "tool_calls": tool_calls_made,
        "tool_results": tool_result_records,
        "steps": MAX_TOOL_CALLS,
        "truncated": True,
        "failure_code": "AGENT_STEP_BUDGET_EXCEEDED",
    }


async def run_phase2(
    model_tag: str,
    mcp_base_url: str = "http://127.0.0.1:8000/mcp",
    *,
    transport: McpTransport = "streamable-http",
    stdio_command: str = "bddk-mcp",
    stdio_args: Sequence[str] = ("serve", "--profile", "public", "--transport", "stdio"),
    cases: Sequence[TestCase] | None = None,
) -> dict[str, Any]:
    """Run Phase 2 through one live, initialized MCP connection."""

    endpoint = McpEndpoint(
        transport=transport,
        url=mcp_base_url,
        command=stdio_command,
        args=tuple(stdio_args),
        bearer_token=os.environ.get("BDDK_BENCHMARK_MCP_TOKEN") or None,
    )
    selected_cases = list(cases if cases is not None else PHASE2_CASES)
    results: list[dict[str, Any]] = []

    async with open_mcp_session(endpoint) as (mcp_session, contract):
        async with httpx.AsyncClient() as llm_client:
            for case in selected_cases:
                missing_tools = sorted(_required_case_tools(case) - contract.names)
                if missing_tools:
                    results.append(_not_comparable_result(case, missing_tools))
                    continue

                logger.info("Phase 2: model=%s case=%s", model_tag, case.id)
                started = time.perf_counter()
                try:
                    trace = await _run_agent_loop(llm_client, mcp_session, model_tag, case.question, contract)
                    combined_tool_results = _combined_tool_evidence(trace)
                    answer = trace["final_answer"]
                    claim_grade = numeric_claim_support_grader(combined_tool_results, answer)
                    failure_code = trace.get("failure_code")
                    if failure_code:
                        model_score = None
                        model_status = "not_run"
                        model_reason = "agent_loop_failed"
                        grader_model = _configured_grader_model()
                    else:
                        model_grade = await model_grader(combined_tool_results, answer)
                        model_score = model_grade.score
                        model_status = model_grade.status
                        model_reason = model_grade.reason
                        grader_model = model_grade.model
                    grader_comparable = model_status == "scored" and failure_code is None
                    audit_metrics = audit_grade_metrics(
                        case,
                        trace,
                        claim_grade.score,
                        model_score,
                    )
                    if not grader_comparable:
                        audit_metrics["grounded_answer_success"] = False
                        audit_metrics["audit_grade_success"] = False
                    artifacts = _audit_artifacts(trace)
                    results.append(
                        {
                            "case_id": case.id,
                            "question": sanitize_for_audit(case.question),
                            "is_multi_tool": case.is_multi_tool,
                            "comparable": grader_comparable,
                            "grounding_comparable": grader_comparable,
                            "retrieval_comparable": True,
                            "tool_calls": artifacts["tool_calls"],
                            "tool_evidence": artifacts["tool_evidence"],
                            "tool_trace_sha256": artifacts["tool_trace_sha256"],
                            "final_answer": artifacts["final_answer"],
                            "final_answer_sha256": artifacts["final_answer_sha256"],
                            "numeric_claim_support_score": claim_grade.score,
                            "numeric_claim_support_status": claim_grade.status,
                            "numeric_claim_support_reason": claim_grade.reason,
                            "numeric_claim_count": claim_grade.answer_claim_count,
                            "supported_numeric_claim_count": claim_grade.supported_claim_count,
                            "unsupported_numeric_claims": list(claim_grade.unsupported_claims),
                            "model_grounding_score": model_score,
                            "model_grader_status": model_status,
                            "model_grader_reason": model_reason,
                            "model_grader_model": grader_model,
                            **audit_metrics,
                            "steps": trace["steps"],
                            "truncated": trace.get("truncated", False),
                            "failure_code": failure_code,
                            "latency_s": time.perf_counter() - started,
                            "error": failure_code,
                            "non_comparability_reason": (
                                None if grader_comparable else model_reason or "MODEL_GRADER_NOT_COMPARABLE"
                            ),
                        }
                    )
                except Exception as error:
                    logger.warning(
                        "Phase 2 case %s failed (error_type=%s)",
                        case.id,
                        type(error).__name__,
                    )
                    audit_metrics = audit_grade_metrics(case, {}, None, None, error=type(error).__name__)
                    results.append(
                        {
                            "case_id": case.id,
                            "question": sanitize_for_audit(case.question),
                            "comparable": False,
                            "grounding_comparable": False,
                            "retrieval_comparable": True,
                            "error": _safe_error_code(error),
                            "error_type": type(error).__name__,
                            "numeric_claim_support_score": None,
                            "numeric_claim_support_status": "not_run",
                            "model_grounding_score": None,
                            "model_grader_status": "not_run",
                            "model_grader_model": _configured_grader_model(),
                            **audit_metrics,
                            "latency_s": time.perf_counter() - started,
                        }
                    )

        return _aggregate_results(model_tag, endpoint, contract, results, selected_cases)


def _required_case_tools(case: TestCase) -> frozenset[str]:
    names = {case.expected_tool} if case.expected_tool else set()
    names.update(case.expected_chain)
    names.update(case.expected_source_tools)
    return frozenset(names)


def _not_comparable_result(case: TestCase, missing_tools: list[str]) -> dict[str, Any]:
    audit_metrics = audit_grade_metrics(case, {}, None, None, error="LIVE_TOOL_UNAVAILABLE")
    return {
        "case_id": case.id,
        "question": sanitize_for_audit(case.question),
        "is_multi_tool": case.is_multi_tool,
        "comparable": False,
        "grounding_comparable": False,
        "retrieval_comparable": False,
        "error": "LIVE_TOOL_UNAVAILABLE",
        "missing_live_tools": missing_tools,
        "numeric_claim_support_score": None,
        "numeric_claim_support_status": "not_run",
        "model_grounding_score": None,
        "model_grader_status": "not_run",
        "model_grader_model": _configured_grader_model(),
        **audit_metrics,
        "latency_s": 0.0,
    }


def _safe_error_code(error: Exception) -> str:
    if isinstance(error, McpToolInvocationError):
        return "MCP_TOOL_ERROR"
    if isinstance(error, BenchmarkProtocolError):
        return "BENCHMARK_PROTOCOL_ERROR"
    if isinstance(error, httpx.HTTPError):
        return "LLM_TRANSPORT_ERROR"
    return "BENCHMARK_CASE_ERROR"


def _aggregate_results(
    model_tag: str,
    endpoint: McpEndpoint,
    contract: LiveMcpContract,
    results: list[dict[str, Any]],
    cases: Sequence[TestCase],
) -> dict[str, Any]:
    comparable = [result for result in results if result.get("comparable") and not result.get("error")]
    live_eligible = [result for result in results if result.get("error") != "LIVE_TOOL_UNAVAILABLE"]
    retrieval_comparable = [result for result in results if result.get("retrieval_comparable")]
    multi = [result for result in live_eligible if result.get("is_multi_tool")]
    numeric_scored = [result for result in live_eligible if result.get("numeric_claim_support_status") == "scored"]
    payload = {
        "phase": "2",
        "model": sanitize_for_audit(model_tag),
        "total_cases": len(results),
        "comparable_cases": len(comparable),
        "not_comparable_cases": len(results) - len(comparable),
        "retrieval_comparable_cases": len(retrieval_comparable),
        "numeric_claim_support_scored_cases": len(numeric_scored),
        "avg_numeric_claim_support": _optional_mean(numeric_scored, "numeric_claim_support_score"),
        "avg_model_grounding": _optional_mean(comparable, "model_grounding_score"),
        "chain_success_rate": (
            sum(1 for result in multi if result.get("chain_complete")) / len(multi) if multi else 0.0
        ),
        "transport_success_rate": _rate(live_eligible, "transport_success"),
        "tool_routing_success_rate": _rate(live_eligible, "tool_routing_success"),
        "expected_arguments_success_rate": _rate(live_eligible, "expected_arguments_success"),
        "retrieval_completion_success_rate": _rate(live_eligible, "retrieval_completion_success"),
        "grounded_answer_success_rate": _rate(comparable, "grounded_answer_success"),
        "audit_grade_success_rate": _rate(comparable, "audit_grade_success"),
        "avg_citation_or_source_trace_score": _mean(live_eligible, "citation_or_source_trace_score"),
        "retrieval_source_correctness_success_rate": _applicable_rate(
            live_eligible, "retrieval_source_correctness_success"
        ),
        "avg_retrieval_source_correctness": _optional_mean(live_eligible, "retrieval_source_correctness_score"),
        "avg_language_stability": _mean(live_eligible, "language_stability"),
        "error_count": sum(1 for result in results if result.get("error")),
        "avg_latency_s": _mean(results, "latency_s"),
        "run_metadata": _run_metadata(model_tag, endpoint, contract, results, cases),
        "details": results,
    }
    return sanitize_for_audit(payload)


def _rate(results: list[dict[str, Any]], key: str) -> float:
    return sum(1 for result in results if result.get(key)) / len(results) if results else 0.0


def _applicable_rate(results: list[dict[str, Any]], key: str) -> float | None:
    applicable = [result for result in results if result.get(key) is not None]
    if not applicable:
        return None
    return sum(1 for result in applicable if result.get(key)) / len(applicable)


def _mean(results: list[dict[str, Any]], key: str) -> float:
    values = [float(result[key]) for result in results if result.get(key) is not None]
    return sum(values) / len(values) if values else 0.0


def _optional_mean(results: list[dict[str, Any]], key: str) -> float | None:
    values = [float(result[key]) for result in results if result.get(key) is not None]
    return sum(values) / len(values) if values else None


def _run_metadata(
    model_id: str,
    endpoint: McpEndpoint,
    contract: LiveMcpContract,
    results: Sequence[dict[str, Any]],
    cases: Sequence[TestCase],
) -> dict[str, Any]:
    tool_names = [tool.name for tool in contract.tools]
    return {
        "model_id": sanitize_for_audit(model_id),
        "git": _git_state(),
        "mcp_transport": endpoint.transport,
        "mcp_endpoint": _safe_endpoint_url(endpoint.url) if endpoint.transport == "streamable-http" else "stdio",
        "live_tool_list": tool_names,
        "live_tool_schema_sha256": contract.schema_hash,
        "deployment_config": {
            "llm_base_url": _safe_endpoint_url(LLM_BASE_URL),
            "max_tool_calls": MAX_TOOL_CALLS,
            "tool_count": len(tool_names),
        },
        "dataset_identity": _dataset_identity(cases),
        "corpus_identity": _corpus_identity(results),
        "external_model_grader": {
            "explicit_opt_in_env": EXTERNAL_GRADER_OPT_IN_ENV,
            "egress_enabled": external_grader_opted_in(),
            "model": _configured_grader_model(),
        },
        "temperature": LLM_TEMPERATURE,
        "mcp_server_name": contract.server_name,
        "mcp_server_version": contract.server_version,
        "mcp_protocol_version": contract.protocol_version,
    }


def _combined_tool_evidence(trace: dict[str, Any]) -> str:
    records = []
    for result in trace.get("tool_results") or []:
        if not isinstance(result, dict):
            continue
        records.append(
            {
                "tool_name": result.get("tool_name"),
                "structured_content": result.get("structured_content"),
                "model_content": result.get("model_content") if result.get("structured_content") is None else None,
            }
        )
    return json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _audit_artifacts(trace: dict[str, Any]) -> dict[str, Any]:
    safe_calls = sanitize_for_audit(trace.get("tool_calls") or [])
    tool_evidence: list[dict[str, Any]] = []
    for result in trace.get("tool_results") or []:
        if not isinstance(result, dict):
            continue
        structured = result.get("structured_content")
        evidence = {
            "tool_name": result.get("tool_name"),
            "arguments": result.get("arguments"),
            "structured_status": structured.get("status") if isinstance(structured, dict) else None,
            "structured_content": structured if isinstance(structured, dict) else None,
            "text_content": result.get("model_content") if not isinstance(structured, dict) else None,
        }
        safe_evidence = sanitize_for_audit(evidence)
        safe_evidence["artifact_sha256"] = canonical_sha256(safe_evidence)
        tool_evidence.append(safe_evidence)

    final_answer = str(sanitize_for_audit(trace.get("final_answer") or ""))
    trace_artifact = {
        "tool_calls": safe_calls,
        "tool_evidence": tool_evidence,
        "final_answer": final_answer,
    }
    return {
        **trace_artifact,
        "final_answer_sha256": hashlib.sha256(final_answer.encode("utf-8")).hexdigest(),
        "tool_trace_sha256": canonical_sha256(trace_artifact),
    }


def _configured_grader_model() -> str:
    return str(sanitize_for_audit(os.environ.get("BDDK_GRADER_MODEL", "claude-opus-4-6")))


def _git_state() -> dict[str, Any]:
    commit_sha = "unknown"
    status = b""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit_sha = commit.stdout.strip() or "unknown"
        working_tree = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z"],
            check=True,
            capture_output=True,
        )
        status = working_tree.stdout
    except (OSError, subprocess.SubprocessError):
        return {
            "commit_sha": commit_sha,
            "dirty": None,
            "status_sha256": None,
            "change_entry_count": None,
        }
    entries = [entry for entry in status.split(b"\0") if entry]
    return {
        "commit_sha": commit_sha,
        "dirty": bool(entries),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "change_entry_count": len(entries),
    }


def _dataset_identity(cases: Sequence[TestCase]) -> dict[str, Any]:
    payload = [
        {
            "id": case.id,
            "question": case.question,
            "expected_tool": case.expected_tool,
            "expected_params": case.expected_params,
            "category": case.category,
            "is_multi_tool": case.is_multi_tool,
            "expected_chain": case.expected_chain,
            "expected_source_tools": case.expected_source_tools,
            "expected_documents": case.expected_documents,
            "expected_sections": case.expected_sections,
            "expected_terms": case.expected_terms,
        }
        for case in cases
    ]
    return {
        "name": "phase2-live-cases",
        "case_count": len(payload),
        "case_ids": [str(case["id"]) for case in payload],
        "sha256": canonical_sha256(payload),
    }


def _corpus_identity(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    references: list[dict[str, str]] = []
    for result in results:
        for artifact in result.get("tool_evidence") or []:
            structured = artifact.get("structured_content") if isinstance(artifact, dict) else None
            if not isinstance(structured, dict):
                continue
            for evidence in structured.get("evidence") or []:
                if not isinstance(evidence, dict):
                    continue
                references.append(
                    {
                        key: str(evidence[key])
                        for key in ("document_id", "content_hash", "section_type", "section_ref")
                        if evidence.get(key) not in (None, "")
                    }
                )
    unique_references = sorted(
        {json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for item in references}
    )
    explicit_id = _safe_identity_label(os.environ.get("BDDK_BENCHMARK_CORPUS_ID"))
    explicit_sha = os.environ.get("BDDK_BENCHMARK_CORPUS_SHA256", "").strip().lower()
    if explicit_sha and not all(character in "0123456789abcdef" for character in explicit_sha):
        explicit_sha = "invalid"
    if explicit_sha and len(explicit_sha) != 64:
        explicit_sha = "invalid"
    return {
        "declared_id": explicit_id,
        "declared_sha256": explicit_sha or None,
        "observed_reference_count": len(unique_references),
        "observed_evidence_sha256": canonical_sha256(unique_references) if unique_references else None,
    }


def _safe_identity_label(value: str | None) -> str | None:
    if not value:
        return None
    candidate = value.strip()
    if not candidate or len(candidate) > 200 or any(ord(character) < 32 for character in candidate):
        return "invalid"
    return str(sanitize_for_audit(candidate))


def _safe_endpoint_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        if not parsed.scheme or not parsed.hostname:
            return "invalid"
        host = parsed.hostname
        if ":" in host:
            host = f"[{host}]"
        port = f":{parsed.port}" if parsed.port is not None else ""
        return urlunsplit((parsed.scheme, f"{host}{port}", parsed.path, "", ""))
    except (TypeError, ValueError):
        return "invalid"
