# benchmark/phase2_e2e.py
"""Phase 2: Live end-to-end evaluation with grounding scoring.

Requires a running bddk-mcp server. The harness:
1. Sends question + tool schemas to model
2. Model generates tool call(s)
3. Harness executes tool calls against live MCP server via stdio
4. Model receives results and generates final answer
5. Answer graded for grounding (code-based + model-based)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time

import httpx

from benchmark.config import LLM_BASE_URL, LLM_TEMPERATURE, LLM_TIMEOUT, MAX_TOOL_CALLS
from benchmark.graders import code_grader, model_grader
from benchmark.scoring import audit_grade_metrics
from benchmark.test_cases import TEST_CASES
from benchmark.tool_schemas import TOOL_SCHEMAS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Sen bir Türk bankacılık düzenleme uzmanısın. BDDK mevzuatı ve verileri hakkında "
    "sorulara cevap vermek için sana sağlanan araçları kullan.\n\n"
    "KRİTİK KURALLAR:\n"
    "- SADECE araç sonuçlarından gelen bilgileri kullan\n"
    "- Araç sonuçlarında olmayan bilgi EKLEME\n"
    "- Emin olmadığın konularda 'araç sonuçlarında bu bilgi yok' de\n"
    "- Sayısal verileri araç sonuçlarından aynen aktar"
)


async def _run_agent_loop(
    client: httpx.AsyncClient,
    mcp_client: httpx.AsyncClient,
    model: str,
    question: str,
    mcp_base_url: str,
) -> dict:
    """Run the agent loop: model calls tools, harness executes, repeat.

    Returns a dict with the conversation trace and final answer.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    tool_results_text = []
    tool_calls_made = []

    for step in range(MAX_TOOL_CALLS):
        # Ask model
        payload = {
            "model": model,
            "messages": messages,
            "tools": TOOL_SCHEMAS,
            "stream": False,
            "temperature": LLM_TEMPERATURE,
        }

        resp = await client.post(
            f"{LLM_BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        message = choice["message"]
        finish_reason = choice.get("finish_reason", "")  # noqa: F841

        # If no tool calls, we have the final answer
        if not message.get("tool_calls"):
            return {
                "final_answer": message.get("content", ""),
                "tool_calls": tool_calls_made,
                "tool_results": tool_results_text,
                "steps": step,
            }

        # Execute tool calls
        messages.append(message)

        for tc in message["tool_calls"]:
            fn = tc["function"]
            tool_name = fn["name"]
            tool_args = fn.get("arguments", "{}")
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except json.JSONDecodeError:
                    tool_args = {}

            tool_calls_made.append({"name": tool_name, "args": tool_args})

            # Execute against MCP server via HTTP
            try:
                mcp_resp = await mcp_client.post(
                    f"{mcp_base_url}/call-tool",
                    json={"name": tool_name, "arguments": tool_args},
                    timeout=60.0,
                )
                mcp_resp.raise_for_status()
                result_data = mcp_resp.json()
                # MCP returns content as list of content blocks
                result_text = ""
                for block in result_data.get("content", []):
                    if block.get("type") == "text":
                        result_text += block.get("text", "")
            except Exception as e:
                result_text = f"Error executing tool: {e}"

            tool_results_text.append(result_text)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result_text,
                }
            )

    # Max steps reached
    return {
        "final_answer": "",
        "tool_calls": tool_calls_made,
        "tool_results": tool_results_text,
        "steps": MAX_TOOL_CALLS,
        "truncated": True,
    }


async def run_phase2(
    model_tag: str,
    mcp_base_url: str = "http://localhost:8000",
) -> dict:
    """Run Phase 2 end-to-end evaluation.

    Args:
        model_tag: Ollama model tag
        mcp_base_url: Base URL of the running bddk-mcp HTTP server
    """
    results = []

    async with httpx.AsyncClient() as client, httpx.AsyncClient() as mcp_client:
        for case in TEST_CASES:
            logger.info("Phase 2: model=%s case=%d", model_tag, case.id)
            start = time.time()

            try:
                trace = await _run_agent_loop(client, mcp_client, model_tag, case.question, mcp_base_url)
                latency = time.time() - start

                # Grade the answer
                combined_tool_results = "\n---\n".join(trace["tool_results"])
                answer = trace["final_answer"]

                code_score = code_grader(combined_tool_results, answer)
                model_score = await model_grader(combined_tool_results, answer)
                audit_metrics = audit_grade_metrics(case, trace, code_score, model_score)

                # Check if multi-tool chain was completed
                chain_complete = True
                if case.is_multi_tool and case.expected_chain:
                    actual_tools = [tc["name"] for tc in trace["tool_calls"]]
                    chain_complete = all(t in actual_tools for t in case.expected_chain)

                results.append(
                    {
                        "case_id": case.id,
                        "question": case.question,
                        "is_multi_tool": case.is_multi_tool,
                        "tool_calls": trace["tool_calls"],
                        "final_answer": answer[:500],
                        "code_grounding_score": code_score,
                        "model_grounding_score": model_score,
                        "chain_complete": chain_complete,
                        **audit_metrics,
                        "steps": trace["steps"],
                        "truncated": trace.get("truncated", False),
                        "latency_s": latency,
                        "error": None,
                    }
                )

            except Exception as e:
                logger.warning("Phase 2 case %d failed: %s", case.id, e)
                audit_metrics = audit_grade_metrics(case, {}, 0.0, 0.0, error=str(e))
                results.append(
                    {
                        "case_id": case.id,
                        "question": case.question,
                        "error": str(e),
                        "code_grounding_score": 0.0,
                        "model_grounding_score": 0.0,
                        "chain_complete": False,
                        **audit_metrics,
                        "latency_s": time.time() - start,
                    }
                )

    # Aggregate
    n = len(results)
    valid = [r for r in results if not r.get("error")]
    multi = [r for r in results if r.get("is_multi_tool")]

    return {
        "phase": "2",
        "model": model_tag,
        "total_cases": n,
        "avg_code_grounding": sum(r["code_grounding_score"] for r in valid) / len(valid) if valid else 0.0,
        "avg_model_grounding": sum(r["model_grounding_score"] for r in valid) / len(valid) if valid else 0.0,
        "chain_success_rate": sum(1 for r in multi if r.get("chain_complete")) / len(multi) if multi else 0.0,
        "transport_success_rate": _rate(results, "transport_success"),
        "tool_routing_success_rate": _rate(results, "tool_routing_success"),
        "retrieval_completion_success_rate": _rate(results, "retrieval_completion_success"),
        "grounded_answer_success_rate": _rate(results, "grounded_answer_success"),
        "audit_grade_success_rate": _rate(results, "audit_grade_success"),
        "avg_citation_or_source_trace_score": _average(results, "citation_or_source_trace_score"),
        "avg_language_stability": _average(results, "language_stability"),
        "error_count": sum(1 for r in results if r.get("error")),
        "avg_latency_s": sum(r["latency_s"] for r in results) / n if n else 0.0,
        "run_metadata": _run_metadata(model_tag, mcp_base_url),
        "details": results,
    }


def _rate(results: list[dict], key: str) -> float:
    return sum(1 for result in results if result.get(key)) / len(results) if results else 0.0


def _average(results: list[dict], key: str) -> float:
    return sum(float(result.get(key, 0.0) or 0.0) for result in results) / len(results) if results else 0.0


def _run_metadata(model_id: str, mcp_base_url: str) -> dict:
    tool_names = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
    return {
        "model_id": model_id,
        "commit_sha": _current_commit_sha(),
        "exposed_tool_list": tool_names,
        "deployment_config": {
            "mcp_base_url": mcp_base_url,
            "llm_base_url": LLM_BASE_URL,
            "max_tool_calls": MAX_TOOL_CALLS,
            "tool_count": len(tool_names),
        },
        "temperature": LLM_TEMPERATURE,
        "mcp_server_version": os.environ.get("BDDK_MCP_VERSION", "unknown"),
    }


def _current_commit_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return completed.stdout.strip() or "unknown"
