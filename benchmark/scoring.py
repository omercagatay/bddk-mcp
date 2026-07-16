"""Shared scoring functions: pass@k, pass^k, Parameter F1, NLI metrics."""

from __future__ import annotations

from typing import Any

GROUNDING_SUCCESS_THRESHOLD = 0.6
SOURCE_RETRIEVAL_TOOLS = {
    "search_bddk_regulations",
    "get_bddk_document",
    "get_document_history",
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
    case: Any,
    trace: dict,
    claim_support_score: float | None,
    model_score: float | None,
    error: str | None = None,
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
    argument_metrics = expected_argument_metrics(case, trace)
    chain_metrics = ordered_chain_metrics(expected_chain, actual_tools)
    source_metrics = source_correctness_metrics(case, trace)
    numeric_claims_supported = claim_support_score is None or claim_support_score >= GROUNDING_SUCCESS_THRESHOLD
    grounded_answer_success = (
        bool(answer.strip())
        and numeric_claims_supported
        and model_score is not None
        and model_score >= GROUNDING_SUCCESS_THRESHOLD
    )
    source_correctness_score = source_metrics["retrieval_source_correctness_score"]
    trace_score = _source_trace_score(expected_source_tools, actual_tools, tool_results, case)
    citation_or_source_trace_score = min(
        trace_score,
        source_correctness_score if source_correctness_score is not None else trace_score,
    )
    language_stability = _language_stability(answer)

    audit_grade_success = all(
        [
            transport_success,
            tool_routing_success,
            argument_metrics["expected_arguments_success"],
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
        "structured_retrieval_statuses": _retrieval_statuses(expected_source_tools, tool_calls, tool_results),
        "grounded_answer_success": grounded_answer_success,
        "audit_grade_success": audit_grade_success,
        "citation_or_source_trace_score": citation_or_source_trace_score,
        "language_stability": language_stability,
        **argument_metrics,
        **chain_metrics,
        **source_metrics,
    }


def source_correctness_metrics(case: Any, trace: dict) -> dict:
    """Grade expected sources against individual structured evidence items.

    Document and section references must coexist on the same evidence item.
    This deliberately does not search a serialized JSON blob, where a document
    ID in one result and a section reference in another could create a false
    match.
    """
    evidence_units = _structured_evidence_units(trace)
    expected_documents = [str(doc_id) for doc_id in getattr(case, "expected_documents", []) or []]
    expected_sections = list(getattr(case, "expected_sections", []) or [])
    expected_terms = [str(term) for term in getattr(case, "expected_terms", []) or []]

    expected_checks = 0
    matched_checks = 0
    missing_documents: list[str] = []
    missing_sections: list[dict] = []
    missing_terms: list[str] = []

    for doc_id in expected_documents:
        expected_checks += 1
        if any(_same_identifier(unit["document_id"], doc_id) for unit in evidence_units):
            matched_checks += 1
        else:
            missing_documents.append(doc_id)

    for section in expected_sections:
        expected_checks += 1
        section_type = str(section.get("type", ""))
        section_ref = str(section.get("ref", ""))
        explicit_document = str(section.get("document_id", "")).strip()
        associated_documents = (
            {explicit_document} if explicit_document else set(expected_documents) if expected_documents else set()
        )
        if any(
            _same_identifier(unit["section_type"], section_type)
            and _same_identifier(unit["section_ref"], section_ref)
            and (
                not associated_documents
                or any(_same_identifier(unit["document_id"], doc_id) for doc_id in associated_documents)
            )
            for unit in evidence_units
        ):
            matched_checks += 1
        else:
            missing = {"type": section_type, "ref": section_ref}
            if explicit_document:
                missing["document_id"] = explicit_document
            missing_sections.append(missing)

    evidence_text = _normalize_for_matching("\n".join(text for unit in evidence_units for text in unit.get("text", [])))
    for term in expected_terms:
        expected_checks += 1
        if _normalize_for_matching(term) in evidence_text:
            matched_checks += 1
        else:
            missing_terms.append(term)

    score = matched_checks / expected_checks if expected_checks else None
    return {
        "expected_source_checks": expected_checks,
        "matched_source_checks": matched_checks,
        "retrieval_source_correctness_score": score,
        "retrieval_source_correctness_success": score >= 1.0 if score is not None else None,
        "missing_expected_documents": missing_documents,
        "missing_expected_sections": missing_sections,
        "missing_expected_terms": missing_terms,
    }


def _tool_name(call: dict) -> str:
    return str(call.get("name") or call.get("function", {}).get("name") or "")


def _expected_source_tools(case: Any) -> list[str]:
    explicit = list(getattr(case, "expected_source_tools", []) or [])
    if explicit:
        return explicit
    chain_tools = [tool for tool in getattr(case, "expected_chain", []) if tool in SOURCE_RETRIEVAL_TOOLS]
    if chain_tools:
        return chain_tools
    expected_tool = getattr(case, "expected_tool", "")
    if expected_tool in SOURCE_RETRIEVAL_TOOLS:
        return [expected_tool]
    return []


def _tool_routing_success(expected_tool: str, expected_chain: list[str], actual_tools: list[str]) -> bool:
    if expected_chain:
        return _is_ordered_subsequence(expected_chain, actual_tools)
    return expected_tool in actual_tools if expected_tool else bool(actual_tools)


def _retrieval_completion_success(
    expected_source_tools: list[str], tool_calls: list[dict], tool_results: list[Any]
) -> bool:
    actual_tools = [_tool_name(call) for call in tool_calls]
    if not expected_source_tools:
        return bool(actual_tools) and any(_usable_tool_result(result) for result in tool_results)
    return all(
        any(
            tool == _tool_name(call)
            and _usable_tool_result(_result_for_call(tool_results, index), require_structured_status=True)
            for index, call in enumerate(tool_calls)
        )
        for tool in expected_source_tools
    )


def _usable_tool_result(result: Any, *, require_structured_status: bool = False) -> bool:
    structured = _structured_content(result)
    if structured is not None:
        status = str(structured.get("status", "")).strip().lower()
        if status not in {"ok", "partial"}:
            return False
        evidence = structured.get("evidence")
        return isinstance(evidence, list) and bool(evidence)
    if require_structured_status:
        return False

    text = _result_text(result).strip()
    if len(text) < 20:
        return False
    lowered = text.lower()
    return not any(marker in lowered for marker in ERROR_MARKERS)


def _result_for_call(tool_results: list[Any], index: int) -> Any:
    return tool_results[index] if index < len(tool_results) else None


def _source_trace_score(
    expected_source_tools: list[str],
    actual_tools: list[str],
    tool_results: list[Any],
    case: Any,
) -> float:
    tool_calls = [{"name": tool} for tool in actual_tools]
    if not _retrieval_completion_success(expected_source_tools, tool_calls, tool_results):
        return 0.0
    expected_documents = [str(doc_id) for doc_id in getattr(case, "expected_documents", []) or []]
    expected_sections = list(getattr(case, "expected_sections", []) or [])
    if not expected_documents and not expected_sections:
        return 1.0

    source_metrics = source_correctness_metrics(case, {"tool_results": tool_results})
    score = source_metrics["retrieval_source_correctness_score"]
    return score if score is not None else 1.0


def expected_argument_metrics(case: Any, trace: dict) -> dict[str, Any]:
    """Grade expected arguments on the expected tool's first invocation."""

    expected_tool = str(getattr(case, "expected_tool", "") or "")
    expected = dict(getattr(case, "expected_params", {}) or {})
    if not expected_tool:
        return {
            "expected_argument_checks": 0,
            "matched_expected_arguments": 0,
            "expected_arguments_score": 1.0,
            "expected_arguments_success": True,
            "missing_or_mismatched_arguments": [],
        }

    call = next((item for item in trace.get("tool_calls") or [] if _tool_name(item) == expected_tool), None)
    actual = call.get("args", {}) if isinstance(call, dict) else {}
    if not isinstance(actual, dict):
        actual = {}
    missing_or_mismatched = [
        key for key, value in expected.items() if key not in actual or not _argument_values_match(value, actual[key])
    ]
    checks = len(expected)
    matched = checks - len(missing_or_mismatched)
    invocation_present = call is not None
    return {
        "expected_argument_checks": checks,
        "matched_expected_arguments": matched,
        "expected_arguments_score": matched / checks if checks else (1.0 if invocation_present else 0.0),
        "expected_arguments_success": invocation_present and not missing_or_mismatched,
        "missing_or_mismatched_arguments": missing_or_mismatched,
    }


def ordered_chain_metrics(expected_chain: list[str], actual_tools: list[str]) -> dict[str, Any]:
    """Return ordered-subsequence chain metrics, allowing unrelated extra calls."""

    if not expected_chain:
        return {
            "expected_chain_length": 0,
            "matched_chain_length": 0,
            "chain_complete": True,
        }
    cursor = 0
    for tool in actual_tools:
        if cursor < len(expected_chain) and tool == expected_chain[cursor]:
            cursor += 1
    return {
        "expected_chain_length": len(expected_chain),
        "matched_chain_length": cursor,
        "chain_complete": cursor == len(expected_chain),
    }


def _is_ordered_subsequence(expected: list[str], actual: list[str]) -> bool:
    return bool(ordered_chain_metrics(expected, actual)["chain_complete"])


def _argument_values_match(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return _normalize_for_matching(" ".join(expected.split())) == _normalize_for_matching(" ".join(actual.split()))
    return expected == actual


def _retrieval_statuses(
    expected_source_tools: list[str], tool_calls: list[dict], tool_results: list[Any]
) -> list[dict[str, str]]:
    statuses: list[dict[str, str]] = []
    for index, call in enumerate(tool_calls):
        tool = _tool_name(call)
        if tool not in expected_source_tools:
            continue
        structured = _structured_content(_result_for_call(tool_results, index))
        statuses.append(
            {
                "tool": tool,
                "status": str(structured.get("status", "missing")) if structured is not None else "missing",
            }
        )
    return statuses


def _structured_content(result: Any) -> dict[str, Any] | None:
    if not isinstance(result, dict):
        return None
    nested = result.get("structured_content")
    if isinstance(nested, dict):
        return nested
    if "status" in result and ("schema_version" in result or "evidence" in result):
        return result
    return None


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return str(result.get("model_content") or result.get("text_content") or "")
    return ""


def _structured_evidence_units(trace: dict) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for result in trace.get("tool_results") or []:
        structured = _structured_content(result)
        if structured is None:
            continue
        references = structured.get("evidence")
        if not isinstance(references, list):
            continue
        candidates = _structured_content_candidates(structured)
        for reference in references:
            if not isinstance(reference, dict):
                continue
            document_id = str(reference.get("document_id", ""))
            section_type = str(reference.get("section_type", ""))
            section_ref = str(reference.get("section_ref", ""))
            text: list[str] = []
            title = reference.get("title")
            if isinstance(title, str) and title:
                text.append(title)
            for candidate in candidates:
                if candidate["document_id"] and not _same_identifier(candidate["document_id"], document_id):
                    continue
                if (
                    section_type
                    and candidate["section_type"]
                    and not _same_identifier(candidate["section_type"], section_type)
                ):
                    continue
                if (
                    section_ref
                    and candidate["section_ref"]
                    and not _same_identifier(candidate["section_ref"], section_ref)
                ):
                    continue
                text.extend(candidate["text"])
            units.append(
                {
                    "document_id": document_id,
                    "section_type": section_type,
                    "section_ref": section_ref,
                    "text": list(dict.fromkeys(text)),
                }
            )
    return units


def _structured_content_candidates(structured: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    results = structured.get("results")
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                candidates.append(_content_candidate(item))

    pages = structured.get("pages")
    document_id = str(
        structured.get("resolved_document_id")
        or structured.get("document_id")
        or structured.get("requested_document_id")
        or ""
    )
    if isinstance(pages, list):
        for page in pages:
            if isinstance(page, dict):
                candidate = _content_candidate(page)
                candidate["document_id"] = document_id
                candidates.append(candidate)
    return candidates


def _content_candidate(item: dict[str, Any]) -> dict[str, Any]:
    text_fields = ("content", "snippet", "summary", "heading", "title")
    return {
        "document_id": str(item.get("document_id") or item.get("doc_id") or ""),
        "section_type": str(item.get("section_type") or ""),
        "section_ref": str(item.get("section_ref") or ""),
        "text": [str(item[key]) for key in text_fields if isinstance(item.get(key), str) and item[key]],
    }


def _same_identifier(left: str, right: str) -> bool:
    return bool(left and right) and _normalize_for_matching(left.strip()) == _normalize_for_matching(right.strip())


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
