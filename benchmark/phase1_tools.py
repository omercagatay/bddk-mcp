# benchmark/phase1_tools.py
"""Phase 1a: Tool-calling accuracy evaluator.

Sends each test case to a model via Ollama's OpenAI-compatible API
with tool schemas. Measures tool selection accuracy and parameter F1.
Runs each case TRIALS_PER_CASE times for pass@k / pass^k.
"""

from __future__ import annotations

import json
import logging
import time

import httpx

from benchmark.config import (  # noqa: F401 (MAX_TOOL_CALLS used in Phase 2)
    MAX_TOOL_CALLS,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    TRIALS_PER_CASE,
)
from benchmark.scoring import parameter_f1, pass_all_k, pass_at_k, tool_selection_accuracy
from benchmark.test_cases import TEST_CASES, TestCase
from benchmark.tool_schemas import TOOL_SCHEMAS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Sen bir Türk bankacılık düzenleme uzmanısın. BDDK (Bankacılık Düzenleme ve Denetleme Kurumu) "
    "mevzuatı ve verileri hakkında sorulara cevap vermek için sana sağlanan araçları kullan. "
    "Soruyu cevaplamak için en uygun aracı seç ve doğru parametrelerle çağır."
)


async def _call_model(
    client: httpx.AsyncClient,
    model: str,
    question: str,
) -> dict:
    """Send a single question to Ollama and return the response."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "tools": TOOL_SCHEMAS,
        "stream": False,
    }

    resp = await client.post(
        f"{OLLAMA_BASE_URL}/v1/chat/completions",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _extract_tool_call(response: dict) -> tuple[str, dict]:
    """Extract the first tool call name and arguments from the response.

    Returns ("", {}) if no tool call found.
    """
    choices = response.get("choices", [])
    if not choices:
        return "", {}

    message = choices[0].get("message", {})
    tool_calls = message.get("tool_calls", [])
    if not tool_calls:
        return "", {}

    first = tool_calls[0]
    fn = first.get("function", {})
    name = fn.get("name", "")
    args_raw = fn.get("arguments", "{}")

    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            args = {}
    else:
        args = args_raw

    return name, args


async def evaluate_single(
    client: httpx.AsyncClient,
    model: str,
    case: TestCase,
) -> dict:
    """Run a single test case TRIALS_PER_CASE times and score it."""
    trial_results = []

    for trial in range(TRIALS_PER_CASE):
        start = time.time()
        try:
            response = await _call_model(client, model, case.question)
            latency = time.time() - start
            tool_name, tool_args = _extract_tool_call(response)

            tool_correct = tool_selection_accuracy(case.expected_tool, tool_name) == 1.0
            param_score = parameter_f1(case.expected_params, tool_args) if tool_correct else 0.0

            trial_results.append(
                {
                    "trial": trial + 1,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "tool_correct": tool_correct,
                    "param_f1": param_score,
                    "latency_s": latency,
                    "error": None,
                }
            )
        except Exception as e:
            latency = time.time() - start
            logger.warning("Case %d trial %d failed: %s", case.id, trial + 1, e)
            trial_results.append(
                {
                    "trial": trial + 1,
                    "tool_name": "",
                    "tool_args": {},
                    "tool_correct": False,
                    "param_f1": 0.0,
                    "latency_s": latency,
                    "error": str(e),
                }
            )

    tool_successes = [t["tool_correct"] for t in trial_results]
    param_scores = [t["param_f1"] for t in trial_results]

    return {
        "case_id": case.id,
        "question": case.question,
        "expected_tool": case.expected_tool,
        "category": case.category,
        "trials": trial_results,
        "pass_at_k": pass_at_k(tool_successes),
        "pass_all_k": pass_all_k(tool_successes),
        "avg_param_f1": sum(param_scores) / len(param_scores) if param_scores else 0.0,
        "avg_latency_s": sum(t["latency_s"] for t in trial_results) / len(trial_results),
    }


async def run_phase1a(model_tag: str) -> dict:
    """Run Phase 1a for a single model. Returns aggregate results."""
    results = []
    async with httpx.AsyncClient() as client:
        for case in TEST_CASES:
            if case.is_multi_tool:
                continue  # Multi-tool cases scored in Phase 2
            logger.info("Phase 1a: model=%s case=%d", model_tag, case.id)
            result = await evaluate_single(client, model_tag, case)
            results.append(result)

    # Aggregate
    n = len(results)
    tool_acc = sum(r["pass_at_k"] for r in results) / n if n else 0.0
    tool_consistency = sum(r["pass_all_k"] for r in results) / n if n else 0.0
    avg_param_f1 = sum(r["avg_param_f1"] for r in results) / n if n else 0.0
    avg_latency = sum(r["avg_latency_s"] for r in results) / n if n else 0.0

    # Per-category breakdown
    categories: dict[str, list[dict]] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r)

    per_category = {}
    for cat, cat_results in categories.items():
        cn = len(cat_results)
        per_category[cat] = {
            "count": cn,
            "tool_accuracy": sum(r["pass_at_k"] for r in cat_results) / cn,
            "avg_param_f1": sum(r["avg_param_f1"] for r in cat_results) / cn,
        }

    return {
        "phase": "1a",
        "model": model_tag,
        "total_cases": n,
        "tool_selection_accuracy": tool_acc,
        "tool_consistency": tool_consistency,
        "avg_parameter_f1": avg_param_f1,
        "avg_latency_s": avg_latency,
        "per_category": per_category,
        "details": results,
    }
