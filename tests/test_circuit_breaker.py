"""Tests for circuit breaker logic in tools/sync.py."""

import time
from deps import Dependencies


def test_circuit_opens_after_threshold():
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    from tools.sync import _record_sync_failure, CIRCUIT_BREAKER_THRESHOLD

    for _ in range(CIRCUIT_BREAKER_THRESHOLD):
        _record_sync_failure(deps, "test error")

    assert deps.sync_circuit_open is True
    assert deps.sync_consecutive_failures == CIRCUIT_BREAKER_THRESHOLD
    assert deps.last_sync_error == "test error"


def test_circuit_resets_on_success():
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    deps.sync_consecutive_failures = 5
    deps.sync_circuit_open = True
    from tools.sync import _record_sync_success

    _record_sync_success(deps)

    assert deps.sync_consecutive_failures == 0
    assert deps.sync_circuit_open is False
    assert deps.last_sync_time is not None
    assert deps.last_sync_error is None


def test_circuit_does_not_open_below_threshold():
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    from tools.sync import _record_sync_failure, CIRCUIT_BREAKER_THRESHOLD

    for _ in range(CIRCUIT_BREAKER_THRESHOLD - 1):
        _record_sync_failure(deps, "test error")

    assert deps.sync_circuit_open is False
    assert deps.sync_consecutive_failures == CIRCUIT_BREAKER_THRESHOLD - 1


def test_record_sync_success_sets_last_sync_time():
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    from tools.sync import _record_sync_success

    before = time.time()
    _record_sync_success(deps)
    after = time.time()

    assert deps.last_sync_time is not None
    assert before <= deps.last_sync_time <= after
