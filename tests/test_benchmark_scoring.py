"""Tests for benchmark scoring functions."""

from benchmark.scoring import (
    audit_grade_metrics,
    nli_metrics,
    parameter_f1,
    pass_all_k,
    pass_at_k,
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
            "tool_results": ["1291 - Kredi kartı taksit düzenlemesi"],
            "final_answer": "Kredi kartı taksit düzenlemesi bulunmuştur.",
            "truncated": False,
        }

        metrics = audit_grade_metrics(case, trace, code_score=1.0, model_score=1.0)

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
            "tool_results": ["Document 943\nİlke 5 - Model validasyonu\nModel validasyonu bağımsız yapılır."],
            "final_answer": "943 numaralı rehberde İlke 5, model validasyonunun bağımsız yapılmasını düzenler.",
            "truncated": False,
        }

        metrics = audit_grade_metrics(case, trace, code_score=0.8, model_score=0.8)

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

        metrics = audit_grade_metrics(case, trace, code_score=0.0, model_score=0.0)

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
                "1291 - Kredi kartı taksit düzenlemesi",
                "Error executing tool: document not available",
            ],
            "final_answer": "Belge alınamadı.",
            "truncated": False,
        }

        metrics = audit_grade_metrics(case, trace, code_score=0.8, model_score=0.8)

        assert metrics["tool_routing_success"] is True
        assert metrics["retrieval_completion_success"] is False
        assert metrics["citation_or_source_trace_score"] == 0.0
        assert metrics["audit_grade_success"] is False
