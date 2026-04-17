"""Tests for benchmark scoring functions."""

from benchmark.scoring import (
    nli_metrics,
    parameter_f1,
    pass_all_k,
    pass_at_k,
    tool_selection_accuracy,
)


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
        assert tool_selection_accuracy("search_bddk_decisions", "search_bddk_decisions") == 1.0

    def test_incorrect(self):
        assert tool_selection_accuracy("search_bddk_decisions", "get_bddk_bulletin") == 0.0


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
