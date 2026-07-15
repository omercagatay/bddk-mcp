"""Tests for benchmark scoring functions."""

from benchmark.scoring import (
    audit_grade_metrics,
    nli_metrics,
    ordered_chain_metrics,
    parameter_f1,
    pass_all_k,
    pass_at_k,
    source_correctness_metrics,
    tool_selection_accuracy,
)
from benchmark.test_cases import TestCase as BenchmarkTestCase


class TestParameterF1:
    def test_exact_match(self):
        expected = {"keywords": "sermaye yeterliliği", "page": 1}
        actual = {"keywords": "sermaye yeterliliği", "page": 1}
        assert parameter_f1(expected, actual) == 1.0

    def test_partial_match(self):
        expected = {"keywords": "takipteki alacak", "category": "Yönetmelik"}
        actual = {"keywords": "takipteki alacak", "page": 1}
        f1 = parameter_f1(expected, actual)
        assert 0.4 < f1 < 0.6

    def test_no_match(self):
        expected = {"keywords": "sermaye"}
        actual = {"query": "mevduat"}
        assert parameter_f1(expected, actual) == 0.0

    def test_empty_expected(self):
        assert parameter_f1({}, {}) == 1.0

    def test_empty_actual(self):
        expected = {"keywords": "test"}
        assert parameter_f1(expected, {}) == 0.0


class TestPassAtK:
    def test_all_pass(self):
        assert pass_at_k([True, True, True]) == 1.0

    def test_one_pass(self):
        assert pass_at_k([False, True, False]) == 1.0

    def test_none_pass(self):
        assert pass_at_k([False, False, False]) == 0.0


class TestPassAllK:
    def test_all_pass(self):
        assert pass_all_k([True, True, True]) == 1.0

    def test_one_fail(self):
        assert pass_all_k([True, False, True]) == 0.0


class TestToolSelectionAccuracy:
    def test_correct(self):
        assert tool_selection_accuracy("search_bddk_regulations", "search_bddk_regulations") == 1.0

    def test_incorrect(self):
        assert tool_selection_accuracy("search_bddk_regulations", "get_bddk_bulletin") == 0.0


class TestNLIMetrics:
    def test_perfect(self):
        true_labels = ["entailment", "contradiction", "neutral"]
        pred_labels = ["entailment", "contradiction", "neutral"]
        m = nli_metrics(true_labels, pred_labels)
        assert m["accuracy"] == 1.0
        assert m["macro_f1"] == 1.0

    def test_all_wrong(self):
        true_labels = ["entailment", "contradiction", "neutral"]
        pred_labels = ["neutral", "entailment", "contradiction"]
        m = nli_metrics(true_labels, pred_labels)
        assert m["accuracy"] == 0.0

    def test_partial(self):
        true_labels = ["entailment", "entailment", "contradiction", "neutral"]
        pred_labels = ["entailment", "neutral", "contradiction", "neutral"]
        m = nli_metrics(true_labels, pred_labels)
        assert m["accuracy"] == 0.75
        assert 0 < m["macro_f1"] <= 1.0
        assert "per_class" in m


class TestAuditGradeMetrics:
    def test_search_only_fails_retrieval_completion_for_audit_case(self):
        case = BenchmarkTestCase(
            id=100,
            question="Kredi kartı taksit düzenlemelerini bul ve ilgili dökümanın tam metnini getir",
            expected_tool="search_bddk_regulations",
            is_multi_tool=True,
            expected_chain=["search_bddk_regulations", "get_bddk_document"],
        )
        trace = {
            "tool_calls": [{"name": "search_bddk_regulations", "args": {"keywords": "kredi kartı taksit"}}],
            "tool_results": [
                _structured_result(
                    status="ok",
                    evidence=[{"document_id": "1291"}],
                    results=[{"document_id": "1291", "summary": "Kredi kartı taksit düzenlemesi"}],
                )
            ],
            "final_answer": "Kredi kartı taksit düzenlemesi bulunmuştur.",
            "truncated": False,
        }

        metrics = audit_grade_metrics(case, trace, claim_support_score=1.0, model_score=1.0)

        assert metrics["transport_success"] is True
        assert metrics["tool_routing_success"] is False
        assert metrics["retrieval_completion_success"] is False
        assert metrics["grounded_answer_success"] is True
        assert metrics["audit_grade_success"] is False

    def test_completed_source_trace_can_pass_audit_grade(self):
        case = BenchmarkTestCase(
            id=101,
            question="943 numaralı rehberde İlke 5 model validasyonu ne diyor?",
            expected_tool="get_document_section",
            expected_source_tools=["get_document_section"],
            expected_documents=["943"],
            expected_sections=[{"type": "ilke", "ref": "5"}],
        )
        trace = {
            "tool_calls": [
                {
                    "name": "get_document_section",
                    "args": {"document_id": "943", "section_type": "ilke", "section_ref": "5"},
                }
            ],
            "tool_results": [
                _structured_result(
                    status="ok",
                    evidence=[{"document_id": "943", "section_type": "ilke", "section_ref": "5"}],
                    results=[
                        {
                            "document_id": "943",
                            "section_type": "ilke",
                            "section_ref": "5",
                            "heading": "İlke 5 - Model validasyonu",
                            "content": "Model validasyonu bağımsız yapılır.",
                        }
                    ],
                )
            ],
            "final_answer": "943 numaralı rehberde İlke 5, model validasyonunun bağımsız yapılmasını düzenler.",
            "truncated": False,
        }

        metrics = audit_grade_metrics(case, trace, claim_support_score=0.8, model_score=0.8)

        assert metrics["transport_success"] is True
        assert metrics["tool_routing_success"] is True
        assert metrics["retrieval_completion_success"] is True
        assert metrics["grounded_answer_success"] is True
        assert metrics["citation_or_source_trace_score"] == 1.0
        assert metrics["language_stability"] == 1.0
        assert metrics["audit_grade_success"] is True

    def test_transport_success_alone_does_not_pass_audit_grade(self):
        case = BenchmarkTestCase(
            id=102,
            question="Şu dökümanın tam metnini incelemek istiyorum: 1291",
            expected_tool="get_bddk_document",
            expected_source_tools=["get_bddk_document"],
            expected_documents=["1291"],
        )
        trace = {
            "tool_calls": [],
            "tool_results": [],
            "final_answer": "Bu konuda genel bilgi verebilirim.",
            "truncated": False,
        }

        metrics = audit_grade_metrics(case, trace, claim_support_score=0.0, model_score=0.0)

        assert metrics["transport_success"] is True
        assert metrics["tool_routing_success"] is False
        assert metrics["retrieval_completion_success"] is False
        assert metrics["grounded_answer_success"] is False
        assert metrics["audit_grade_success"] is False

    def test_failed_source_tool_result_does_not_count_as_retrieval_completion(self):
        case = BenchmarkTestCase(
            id=103,
            question="Kredi kartı taksit düzenlemelerini bul ve ilgili dökümanın tam metnini getir",
            expected_tool="search_bddk_regulations",
            is_multi_tool=True,
            expected_chain=["search_bddk_regulations", "get_bddk_document"],
            expected_source_tools=["get_bddk_document"],
        )
        trace = {
            "tool_calls": [
                {"name": "search_bddk_regulations", "args": {"keywords": "kredi kartı taksit"}},
                {"name": "get_bddk_document", "args": {"document_id": "1291"}},
            ],
            "tool_results": [
                _structured_result(
                    status="ok",
                    evidence=[{"document_id": "1291"}],
                    results=[{"document_id": "1291", "summary": "Kredi kartı taksit düzenlemesi"}],
                ),
                _structured_result(status="unavailable", evidence=[]),
            ],
            "final_answer": "Belge alınamadı.",
            "truncated": False,
        }

        metrics = audit_grade_metrics(case, trace, claim_support_score=0.8, model_score=0.8)

        assert metrics["tool_routing_success"] is True
        assert metrics["retrieval_completion_success"] is False
        assert metrics["citation_or_source_trace_score"] == 0.0
        assert metrics["audit_grade_success"] is False

    def test_structured_no_results_status_overrides_long_human_text(self):
        case = BenchmarkTestCase(
            id=104,
            question="Belgeyi bul",
            expected_tool="search_document_store",
            expected_params={"query": "belge"},
            expected_source_tools=["search_document_store"],
        )
        trace = {
            "tool_calls": [{"name": "search_document_store", "args": {"query": "belge"}}],
            "tool_results": [
                _structured_result(
                    status="no_results",
                    evidence=[],
                    text="NO RESULTS: " + "Bu açıklama uzundur. " * 20,
                )
            ],
            "final_answer": "Araç sonucu bulunamadı.",
        }

        metrics = audit_grade_metrics(case, trace, claim_support_score=None, model_score=1.0)

        assert metrics["retrieval_completion_success"] is False
        assert metrics["structured_retrieval_statuses"] == [{"tool": "search_document_store", "status": "no_results"}]

    def test_partial_structured_result_with_evidence_is_usable_but_status_is_retained(self):
        case = BenchmarkTestCase(
            id=105,
            question="Belgeyi bul",
            expected_tool="search_document_store",
            expected_source_tools=["search_document_store"],
        )
        trace = {
            "tool_calls": [{"name": "search_document_store", "args": {"query": "belge"}}],
            "tool_results": [
                _structured_result(
                    status="partial",
                    evidence=[{"document_id": "943"}],
                    results=[{"document_id": "943", "snippet": "kısmi kanıt"}],
                )
            ],
            "final_answer": "Kısmi kanıt bulundu.",
        }

        metrics = audit_grade_metrics(case, trace, claim_support_score=None, model_score=1.0)

        assert metrics["retrieval_completion_success"] is True
        assert metrics["structured_retrieval_statuses"][0]["status"] == "partial"

    def test_expected_arguments_and_ordered_chain_are_both_graded(self):
        case = BenchmarkTestCase(
            id=106,
            question="Ara ve getir",
            expected_tool="search_bddk_regulations",
            expected_params={"keywords": "kredi kartı taksit"},
            expected_chain=["search_bddk_regulations", "get_bddk_document"],
        )
        trace = {
            "tool_calls": [
                {"name": "get_bddk_document", "args": {"document_id": "1291"}},
                {"name": "search_bddk_regulations", "args": {"keywords": "başka sorgu"}},
            ],
            "tool_results": [
                _structured_result(status="ok", evidence=[{"document_id": "1291"}]),
                _structured_result(status="ok", evidence=[{"document_id": "1291"}]),
            ],
            "final_answer": "Sonuç bulundu.",
        }

        metrics = audit_grade_metrics(case, trace, claim_support_score=None, model_score=1.0)

        assert metrics["tool_routing_success"] is False
        assert metrics["chain_complete"] is False
        assert metrics["matched_chain_length"] == 1
        assert metrics["expected_arguments_score"] == 0.0
        assert metrics["missing_or_mismatched_arguments"] == ["keywords"]


def test_ordered_chain_allows_unrelated_calls_between_expected_steps():
    metrics = ordered_chain_metrics(
        ["search_bddk_regulations", "get_bddk_document"],
        ["search_bddk_regulations", "health_check", "get_bddk_document"],
    )

    assert metrics["chain_complete"] is True
    assert metrics["matched_chain_length"] == 2


def test_document_and_section_must_match_on_the_same_evidence_item():
    case = BenchmarkTestCase(
        id=107,
        question="943 İlke 5",
        expected_documents=["943"],
        expected_sections=[{"type": "ilke", "ref": "5"}],
    )
    trace = {
        "tool_results": [
            _structured_result(
                status="ok",
                evidence=[
                    {"document_id": "943", "section_type": "ilke", "section_ref": "4"},
                    {"document_id": "999", "section_type": "ilke", "section_ref": "5"},
                ],
                results=[
                    {"document_id": "943", "section_type": "ilke", "section_ref": "4", "content": "A"},
                    {"document_id": "999", "section_type": "ilke", "section_ref": "5", "content": "B"},
                ],
            )
        ]
    }

    metrics = source_correctness_metrics(case, trace)

    assert metrics["matched_source_checks"] == 1
    assert metrics["retrieval_source_correctness_score"] == 0.5
    assert metrics["missing_expected_sections"] == [{"type": "ilke", "ref": "5"}]


def _structured_result(
    *,
    status: str,
    evidence: list[dict],
    results: list[dict] | None = None,
    text: str = "structured retrieval result",
) -> dict:
    return {
        "structured_content": {
            "schema_version": "1.0",
            "status": status,
            "text": text,
            "evidence": evidence,
            "results": results or [],
        },
        "model_content": text,
    }
