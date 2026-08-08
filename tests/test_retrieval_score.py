import hashlib
import json

import pytest

from scripts.retrieval_score import (
    _assert_test_database_identity,
    _json_safe,
    _validate_test_database_guard,
    composite_score,
)


def test_composite_score_weights_core_retrieval_metrics():
    score = composite_score({"mrr": 0.5, "hit_at_1": 0.25, "f1_at_3": 0.75})

    assert score == 45


def test_json_safe_converts_nested_sets_to_sorted_lists():
    payload = {"expected": {"b", "a"}, "nested": [{"ids": {"2", "1"}}]}

    safe = _json_safe(payload)

    assert safe == {"expected": ["a", "b"], "nested": [{"ids": ["1", "2"]}]}
    json.dumps(safe)


def test_score_database_identity_rejects_public_runtime_dsn(monkeypatch: pytest.MonkeyPatch):
    dsn = "postgresql://public-runtime.invalid/bddk"
    monkeypatch.setenv("BDDK_DATABASE_URL", dsn)

    with pytest.raises(RuntimeError, match="BDDK_DATABASE_URL"):
        _assert_test_database_identity(dsn)


def test_score_database_identity_accepts_distinct_explicit_test_dsn(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BDDK_DATABASE_URL", "postgresql://public-runtime.invalid/bddk")

    assert _assert_test_database_identity("postgresql://test.invalid/bddk") == "postgresql://test.invalid/bddk"


def test_score_database_guard_requires_the_separately_provisioned_hash():
    token = "test-only-guard-token-with-at-least-32-characters"
    row = {
        "database_name": "bddk_benchmark_test",
        "guard_hash": hashlib.sha256(token.encode()).hexdigest(),
    }

    _validate_test_database_guard(row, token)

    with pytest.raises(RuntimeError, match="approved disposable test database"):
        _validate_test_database_guard(row, token + "-wrong")
