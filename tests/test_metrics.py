from concurrent.futures import ThreadPoolExecutor

from bddk_mcp.observability.metrics import Metrics


def test_record_weak_match_hit_preserves_existing_summary_key():
    metrics = Metrics()

    metrics.record_weak_match_hit()

    assert metrics.summary()["low_confidence_hits"] == 1


def test_metrics_count_attempts_errors_and_latency_consistently():
    metrics = Metrics()

    metrics.record_request("search", 10)
    metrics.record_request("search", 20)
    metrics.record_error("search")

    summary = metrics.summary()
    assert summary["scope"] == "process"
    assert summary["total_requests"] == 2
    assert summary["total_errors"] == 1
    assert summary["tools"] == [
        {
            "tool": "search",
            "requests": 2,
            "successes": 1,
            "errors": 1,
            "avg_latency_ms": 15.0,
        }
    ]


def test_metrics_updates_are_thread_safe():
    metrics = Metrics()
    workers = 8
    updates_per_worker = 1_000

    def update() -> None:
        for _ in range(updates_per_worker):
            metrics.record_request("concurrent_tool", 1)
            metrics.record_cache_hit()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(lambda _index: update(), range(workers)))

    summary = metrics.summary()
    assert summary["total_requests"] == workers * updates_per_worker
    assert summary["cache_hits"] == workers * updates_per_worker
