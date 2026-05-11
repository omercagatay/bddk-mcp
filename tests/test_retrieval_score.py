from scripts.retrieval_score import composite_score


def test_composite_score_weights_core_retrieval_metrics():
    score = composite_score({"mrr": 0.5, "hit_at_1": 0.25, "f1_at_3": 0.75})

    assert score == 45
