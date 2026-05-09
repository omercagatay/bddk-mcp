"""Shared scoring functions: pass@k, pass^k, Parameter F1, NLI metrics."""

from __future__ import annotations

from typing import Any

GROUNDING_SUCCESS_THRESHOLD = 0.6
SOURCE_RETRIEVAL_TOOLS = {
    "get_bddk_document",
    "get_document_section",
    "search_document_sections",
    "search_document_store",
}
ERROR_MARKERS = (
    "error executing tool",
    "not found",
    "not available",
    "bulunamadı",
    "bulunamadi",
    "hata",
)
TURKISH_MARKERS = (
    "ı",
    "İ",
    "ğ",
    "ü",
    "ş",
    "ö",
    "ç",
    " ve ",
    " için ",
    " olarak ",
    " düzen",
    " banka",
)
ENGLISH_MARKERS = (" the ", " and ", " regulation ", " according ", " document ")


def audit_grade_metrics(
    case: Any, trace: dict, code_score: float, model_score: float, error: str | None = None
) -> dict:
    """Score Phase 2 results beyond transport success.

    This separates "the model/server exchanged messages" from the stricter
    audit-grade behavior: correct tool routing, completed source retrieval,
    grounded answer, visible source trace, and stable Turkish output.
    """
    tool_calls = trace.get("tool_calls") or []
    tool_results = trace.get("tool_results") or []
    answer = trace.get("final_answer") or ""
    actual_tools = [_tool_name(call) for call in tool_calls]

    expected_chain = list(getattr(case, "expected_chain", []) or [])
    expected_tool = getattr(case, "expected_tool", "")
    expected_source_tools = _expected_source_tools(case)

    transport_success = error is None and not trace.get("transport_error", False)
    tool_routing_success = _tool_routing_success(expected_tool, expected_chain, actual_tools)
    retrieval_completion_success = _retrieval_completion_success(expected_source_tools, tool_calls, tool_results)
    grounded_answer_success = (
        bool(answer.strip())
        and code_score >= GROUNDING_SUCCESS_THRESHOLD
        and model_score >= GROUNDING_SUCCESS_THRESHOLD
    )
    citation_or_source_trace_score = _source_trace_score(expected_source_tools, actual_tools, tool_results, case)
    language_stability = _language_stability(answer)

    audit_grade_success = all(
        [
            transport_success,
            tool_routing_success,
            retrieval_completion_success,
            grounded_answer_success,
            citation_or_source_trace_score >= 0.75,
            language_stability >= 0.75,
            not trace.get("truncated", False),
        ]
    )

    return {
        "transport_success": transport_success,
        "tool_routing_success": tool_routing_success,
        "retrieval_completion_success": retrieval_completion_success,
        "grounded_answer_success": grounded_answer_success,
        "audit_grade_success": audit_grade_success,
        "citation_or_source_trace_score": citation_or_source_trace_score,
        "language_stability": language_stability,
    }


def _tool_name(call: dict) -> str:
    return str(call.get("name") or call.get("function", {}).get("name") or "")


def _expected_source_tools(case: Any) -> list[str]:
    explicit = list(getattr(case, "expected_source_tools", []) or [])
    if explicit:
        return explicit
    expected_tool = getattr(case, "expected_tool", "")
    if expected_tool in SOURCE_RETRIEVAL_TOOLS:
        return [expected_tool]
    return [tool for tool in getattr(case, "expected_chain", []) if tool in SOURCE_RETRIEVAL_TOOLS]


def _tool_routing_success(expected_tool: str, expected_chain: list[str], actual_tools: list[str]) -> bool:
    if expected_chain:
        return all(tool in actual_tools for tool in expected_chain)
    return expected_tool in actual_tools if expected_tool else bool(actual_tools)


def _retrieval_completion_success(
    expected_source_tools: list[str], tool_calls: list[dict], tool_results: list[str]
) -> bool:
    actual_tools = [_tool_name(call) for call in tool_calls]
    if not expected_source_tools:
        return bool(actual_tools) and any(_usable_tool_result(result) for result in tool_results)
    return all(
        any(
            tool == _tool_name(call) and _usable_tool_result(_result_for_call(tool_results, index))
            for index, call in enumerate(tool_calls)
        )
        for tool in expected_source_tools
    )


def _usable_tool_result(result: str) -> bool:
    text = (result or "").strip()
    if len(text) < 20:
        return False
    lowered = text.lower()
    return not any(marker in lowered for marker in ERROR_MARKERS)


def _result_for_call(tool_results: list[str], index: int) -> str:
    return tool_results[index] if index < len(tool_results) else ""


def _source_trace_score(
    expected_source_tools: list[str],
    actual_tools: list[str],
    tool_results: list[str],
    case: Any,
) -> float:
    tool_calls = [{"name": tool} for tool in actual_tools]
    if not _retrieval_completion_success(expected_source_tools, tool_calls, tool_results):
        return 0.0
    expected_documents = [str(doc_id) for doc_id in getattr(case, "expected_documents", []) or []]
    expected_sections = list(getattr(case, "expected_sections", []) or [])
    if not expected_documents and not expected_sections:
        return 1.0

    evidence = _normalize_for_matching("\n".join(tool_results))
    checks = 0
    hits = 0
    for doc_id in expected_documents:
        checks += 1
        if doc_id.lower() in evidence:
            hits += 1
    for section in expected_sections:
        section_type = _normalize_for_matching(str(section.get("type", "")))
        section_ref = _normalize_for_matching(str(section.get("ref", "")))
        checks += 1
        if section_type in evidence and section_ref in evidence:
            hits += 1
    return hits / checks if checks else 1.0


def _language_stability(answer: str) -> float:
    text = f" {answer.strip()} "
    if not answer.strip():
        return 0.0
    lowered = text.lower()
    turkish_hits = sum(1 for marker in TURKISH_MARKERS if marker in text or marker in lowered)
    english_hits = sum(1 for marker in ENGLISH_MARKERS if marker in lowered)
    if turkish_hits and english_hits <= 1:
        return 1.0
    if turkish_hits:
        return 0.75
    if english_hits:
        return 0.25
    return 0.5


def _normalize_for_matching(text: str) -> str:
    return (
        text.casefold()
        .replace("ı", "i")
        .replace("i̇", "i")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ş", "s")
        .replace("ö", "o")
        .replace("ç", "c")
    )


def parameter_f1(expected: dict, actual: dict) -> float:
    """Compute F1 over expected vs actual parameter key-value pairs.

    Each (key, str(value)) pair is a token. F1 = 2*P*R / (P+R).
    """
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0

    expected_pairs = {(k, str(v)) for k, v in expected.items()}
    actual_pairs = {(k, str(v)) for k, v in actual.items()}

    true_positives = len(expected_pairs & actual_pairs)
    if true_positives == 0:
        return 0.0

    precision = true_positives / len(actual_pairs)
    recall = true_positives / len(expected_pairs)
    return 2 * precision * recall / (precision + recall)


def pass_at_k(results: list[bool]) -> float:
    """pass@k: 1.0 if at least one trial succeeded, else 0.0."""
    return 1.0 if any(results) else 0.0


def pass_all_k(results: list[bool]) -> float:
    """pass^k: 1.0 if all trials succeeded, else 0.0."""
    return 1.0 if all(results) else 0.0


def tool_selection_accuracy(expected: str, actual: str) -> float:
    """Exact match on tool name. Returns 1.0 or 0.0."""
    return 1.0 if expected == actual else 0.0


def nli_metrics(true_labels: list[str], pred_labels: list[str]) -> dict:
    """Compute NLI evaluation metrics: accuracy, macro-F1, per-class P/R/F1."""
    assert len(true_labels) == len(pred_labels)
    n = len(true_labels)

    correct = sum(1 for t, p in zip(true_labels, pred_labels, strict=True) if t == p)
    accuracy = correct / n if n else 0.0

    classes = ["entailment", "contradiction", "neutral"]
    per_class = {}
    f1_scores = []

    for cls in classes:
        tp = sum(1 for t, p in zip(true_labels, pred_labels, strict=True) if t == cls and p == cls)
        fp = sum(1 for t, p in zip(true_labels, pred_labels, strict=True) if t != cls and p == cls)
        fn = sum(1 for t, p in zip(true_labels, pred_labels, strict=True) if t == cls and p != cls)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1}
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }
