"""Tests for the Dependencies container."""

import time

from deps import Dependencies


def test_dependencies_defaults():
    """Dependencies initializes with correct defaults."""
    deps = Dependencies(
        pool=None,
        doc_store=None,
        client=None,
        http=None,
    )
    assert deps.vector_store is None
    assert deps.sync_task is None
    assert deps.vector_init_task is None
    assert deps.last_sync_time is None
    assert deps.last_sync_error is None
    assert deps.sync_consecutive_failures == 0
    assert deps.sync_circuit_open is False
    assert isinstance(deps.server_start_time, float)
    assert deps.server_start_time <= time.time()
