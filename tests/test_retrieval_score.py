import json

from scripts.retrieval_score import _json_safe, composite_score


def test_composite_score_weights_core_retrieval_metrics():
    score = composite_score({"mrr": 0.5, "hit_at_1": 0.25, "f1_at_3": 0.75})

    assert score == 45


def test_json_safe_converts_nested_sets_to_sorted_lists():
    payload = {"expected": {"b", "a"}, "nested": [{"ids": {"2", "1"}}]}

    safe = _json_safe(payload)

    assert safe == {"expected": ["a", "b"], "nested": [{"ids": ["1", "2"]}]}
    json.dumps(safe)
