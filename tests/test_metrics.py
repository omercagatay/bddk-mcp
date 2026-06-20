from bddk_mcp.observability.metrics import Metrics


def test_record_weak_match_hit_preserves_existing_summary_key():
    metrics = Metrics()

    metrics.record_weak_match_hit()

    assert metrics.summary()["low_confidence_hits"] == 1
