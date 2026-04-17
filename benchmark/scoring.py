"""Shared scoring functions: pass@k, pass^k, Parameter F1, NLI metrics."""

from __future__ import annotations


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
