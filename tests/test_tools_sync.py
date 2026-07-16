"""Tests for tools/sync.py — circuit breaker state machine and tool registration.

The circuit breaker helpers (_record_sync_failure / _record_sync_success) own
mission-critical state that silences the auto-sync loop. Before this file
there were zero tests covering them; a regression would mean document sync
stops without anyone noticing.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.tools.sync import (
    CIRCUIT_BREAKER_THRESHOLD,
    _record_sync_failure,
    _record_sync_success,
    register,
)


def _fresh_deps() -> Dependencies:
    """Dependencies with pristine circuit-breaker state."""
    return Dependencies(pool=None, doc_store=None, client=None, http=None)


def _registered_tool_names(mcp: MagicMock) -> set[str]:
    """Extract the __name__ of each function registered via @mcp.tool()."""
    return {call.args[0].__name__ for call in mcp.tool.return_value.call_args_list}


# -- _record_sync_failure ----------------------------------------------------


class TestRecordSyncFailure:
    def test_first_failure_increments_counter(self):
        deps = _fresh_deps()
        _record_sync_failure(deps, "boom")
        assert deps.sync_consecutive_failures == 1
        assert deps.last_sync_error == "boom"
        assert deps.sync_circuit_open is False

    def test_consecutive_failures_accumulate(self):
        deps = _fresh_deps()
        for i in range(3):
            _record_sync_failure(deps, f"err {i}")
        assert deps.sync_consecutive_failures == 3
        assert deps.last_sync_error == "err 2"
        assert deps.sync_circuit_open is False

    def test_circuit_stays_closed_below_threshold(self):
        deps = _fresh_deps()
        for _ in range(CIRCUIT_BREAKER_THRESHOLD - 1):
            _record_sync_failure(deps, "err")
        assert deps.sync_circuit_open is False

    def test_circuit_opens_exactly_at_threshold(self):
        deps = _fresh_deps()
        for _ in range(CIRCUIT_BREAKER_THRESHOLD):
            _record_sync_failure(deps, "err")
        assert deps.sync_circuit_open is True
        assert deps.sync_consecutive_failures == CIRCUIT_BREAKER_THRESHOLD

    def test_failures_past_threshold_keep_circuit_open(self):
        deps = _fresh_deps()
        for _ in range(CIRCUIT_BREAKER_THRESHOLD + 5):
            _record_sync_failure(deps, "err")
        assert deps.sync_circuit_open is True
        assert deps.sync_consecutive_failures == CIRCUIT_BREAKER_THRESHOLD + 5


# -- _record_sync_success ----------------------------------------------------


class TestRecordSyncSuccess:
    def test_resets_failure_counter(self):
        deps = _fresh_deps()
        deps.sync_consecutive_failures = 7
        _record_sync_success(deps)
        assert deps.sync_consecutive_failures == 0

    def test_closes_open_circuit(self):
        deps = _fresh_deps()
        deps.sync_circuit_open = True
        deps.sync_consecutive_failures = CIRCUIT_BREAKER_THRESHOLD
        _record_sync_success(deps)
        assert deps.sync_circuit_open is False
        assert deps.sync_consecutive_failures == 0

    def test_clears_last_error(self):
        deps = _fresh_deps()
        deps.last_sync_error = "previous failure"
        _record_sync_success(deps)
        assert deps.last_sync_error is None

    def test_sets_last_sync_time(self):
        deps = _fresh_deps()
        before = time.time()
        _record_sync_success(deps)
        after = time.time()
        assert deps.last_sync_time is not None
        assert before <= deps.last_sync_time <= after


# -- Full cycle --------------------------------------------------------------


def test_failure_cycle_then_success_fully_recovers():
    deps = _fresh_deps()
    for _ in range(CIRCUIT_BREAKER_THRESHOLD):
        _record_sync_failure(deps, "err")
    assert deps.sync_circuit_open is True

    _record_sync_success(deps)

    assert deps.sync_circuit_open is False
    assert deps.sync_consecutive_failures == 0
    assert deps.last_sync_error is None


# -- Tool registration -------------------------------------------------------


def test_register_adds_sync_and_generic_job_tools():
    """register() exposes starts, status/cancel, and document health."""
    mcp = MagicMock()
    register(mcp, _fresh_deps())
    assert _registered_tool_names(mcp) == {
        "refresh_bddk_cache",
        "sync_bddk_documents",
        "trigger_startup_sync",
        "get_operator_job",
        "list_operator_jobs",
        "cancel_operator_job",
        "document_health",
    }


@pytest.mark.asyncio
async def test_document_health_without_pool_reports_available_checks_instead_of_crashing():
    store = MagicMock()
    store.stats = AsyncMock(return_value=MagicMock(total_documents=0))
    store.get_sync_failures = AsyncMock(return_value=[])
    client = MagicMock()
    client.cache_size.return_value = 0
    deps = Dependencies(
        pool=None,
        doc_store=store,
        client=client,
        http=None,
        vector_store=None,
    )
    server = FastMCP("document-health-no-pool-test")
    register(server, deps)

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("document_health", {})

    assert result.isError is False
    assert "Decision cache: 0" in result.content[0].text
    assert "No sync failures recorded" in result.content[0].text
    assert "All documents healthy" in result.content[0].text
