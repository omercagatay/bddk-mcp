"""Official-client contract tests for tracked MCP operator jobs."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.ingest.backfill import BackfillCandidate, BackfillOutcome, BackfillReport
from bddk_mcp.jobs import InMemoryJobRepository, JobOutcome, JobState, OperatorJobManager
from bddk_mcp.tools import admin, sync


def _deps(*, client=None, pool=None, doc_store=None, with_manager: bool = True) -> Dependencies:
    manager = OperatorJobManager(InMemoryJobRepository(), lease_retry_seconds=0.001) if with_manager else None
    return Dependencies(
        pool=pool,
        doc_store=doc_store,
        client=client,
        http=MagicMock(),
        vector_store=MagicMock(),
        job_manager=manager,
    )


async def _wait_for_state(
    manager: OperatorJobManager,
    job_id: UUID,
    states: set[JobState],
) -> None:
    async with asyncio.timeout(1):
        while True:
            job = await manager.get(job_id)
            assert job is not None
            if job.state in states:
                return
            await asyncio.sleep(0)


def _job_id(result) -> UUID:
    assert result.isError is False
    assert result.structuredContent is not None
    content = result.structuredContent.get("result", result.structuredContent)
    return UUID(content["job_id"])


@pytest.mark.asyncio
async def test_refresh_returns_job_id_and_generic_status_and_list_are_live():
    client = MagicMock()
    client.refresh_cache = AsyncMock(return_value=7)
    deps = _deps(client=client)
    server = FastMCP("operator-refresh-test")
    sync.register(server, deps)

    async with create_connected_server_and_client_session(server) as session:
        receipt = await session.call_tool("refresh_bddk_cache", {"idempotency_key": "refresh-1"})
        job_id = _job_id(receipt)
        await _wait_for_state(deps.job_manager, job_id, {JobState.SUCCEEDED})  # type: ignore[arg-type]
        status = await session.call_tool("get_operator_job", {"job_id": str(job_id)})
        listing = await session.call_tool("list_operator_jobs", {"limit": 10, "state": "succeeded"})

    assert status.structuredContent["state"] == "succeeded"
    assert status.structuredContent["result_metrics"] == {"cache_items": 7}
    assert listing.structuredContent["count"] == 1
    assert listing.structuredContent["jobs"][0]["job_id"] == str(job_id)
    client.refresh_cache.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_returns_immediately_and_finishes_with_numeric_outcome():
    decision = MagicMock()
    decision.model_dump.return_value = {"document_id": "943"}
    client = MagicMock()
    client.ensure_cache = AsyncMock()
    client.get_cache_items.return_value = [decision]
    store = MagicMock()
    deps = _deps(client=client, doc_store=store)
    server = FastMCP("operator-sync-test")
    sync.register(server, deps)

    syncer = MagicMock()
    syncer.__aenter__ = AsyncMock(return_value=syncer)
    syncer.__aexit__ = AsyncMock(return_value=None)
    syncer.sync_all = AsyncMock(return_value=SimpleNamespace(total=1, downloaded=1, skipped=0, failed=0))
    vector_metrics = {
        "vector_documents": 1,
        "vector_chunks": 2,
        "vector_migrated": 1,
        "vector_complete": True,
    }

    with (
        patch("bddk_mcp.ingest.doc_sync.DocumentSyncer", return_value=syncer),
        patch.object(sync, "_migrate_to_pgvector", new=AsyncMock(return_value=vector_metrics)),
    ):
        async with create_connected_server_and_client_session(server) as session:
            receipt = await session.call_tool(
                "sync_bddk_documents",
                {"force": False, "concurrency": 4, "idempotency_key": "sync-1"},
            )
            job_id = _job_id(receipt)
            assert receipt.structuredContent["state"] in {"queued", "running"}
            await _wait_for_state(deps.job_manager, job_id, {JobState.SUCCEEDED})  # type: ignore[arg-type]
            status = await session.call_tool("get_operator_job", {"job_id": str(job_id)})

    assert status.structuredContent["progress"] == {
        "total": 1,
        "completed": 1,
        "succeeded": 1,
        "failed": 0,
    }
    assert status.structuredContent["result_metrics"]["downloaded"] == 1
    syncer.sync_all.assert_awaited_once()


@pytest.mark.asyncio
async def test_trigger_job_invokes_startup_sync_not_only_vector_migration():
    deps = _deps(client=MagicMock(), doc_store=MagicMock())
    deps.sync_circuit_open = True
    deps.sync_consecutive_failures = 10
    server = FastMCP("operator-trigger-test")
    sync.register(server, deps)
    called = asyncio.Event()

    async def fake_startup(actual_deps, context=None):
        assert actual_deps is deps
        assert context is not None
        called.set()
        return JobOutcome.from_metrics({"downloaded": 2})

    with patch.object(sync, "startup_sync", side_effect=fake_startup) as startup:
        async with create_connected_server_and_client_session(server) as session:
            receipt = await session.call_tool("trigger_startup_sync", {"idempotency_key": "startup-1"})
            job_id = _job_id(receipt)
            await called.wait()
            await _wait_for_state(deps.job_manager, job_id, {JobState.SUCCEEDED})  # type: ignore[arg-type]

    startup.assert_awaited_once()
    assert deps.sync_circuit_open is False
    assert deps.sync_consecutive_failures == 0


@pytest.mark.asyncio
async def test_destructive_backfill_submits_job_and_tracks_safe_progress():
    pool = MagicMock()
    store = MagicMock()
    deps = _deps(client=MagicMock(), pool=pool, doc_store=store)
    server = FastMCP("operator-backfill-test")
    admin.register(server, deps)
    candidate = BackfillCandidate(
        document_id="mevzuat_22599",
        title="title",
        source_url="https://example.invalid",
        category="regulation",
        decision_date="",
        decision_number="",
        len=100,
        signature="markitdown_degraded",
    )

    async def execute(_syncer, candidates, *, on_progress):
        outcome = BackfillOutcome(document_id=candidates[0].document_id, success=True)
        await on_progress(1, 1, outcome)
        return BackfillReport(total=1, ok=[candidates[0].document_id], elapsed_seconds=0.1)

    syncer = MagicMock()
    syncer.__aenter__ = AsyncMock(return_value=syncer)
    syncer.__aexit__ = AsyncMock(return_value=None)
    scan_started = asyncio.Event()
    release_scan = asyncio.Event()

    async def delayed_scan(*_args, **_kwargs):
        scan_started.set()
        await release_scan.wait()
        return [candidate]

    with (
        patch.object(admin, "scan_candidates", new=AsyncMock(side_effect=delayed_scan)) as scan,
        patch.object(admin, "execute_backfill", side_effect=execute),
        patch("bddk_mcp.ingest.doc_sync.DocumentSyncer", return_value=syncer),
    ):
        async with create_connected_server_and_client_session(server) as session:
            receipt = await session.call_tool(
                "backfill_degraded_documents",
                {"dry_run": False, "limit": 1, "idempotency_key": "backfill-1"},
            )
            job_id = _job_id(receipt)
            # Submission returns a receipt even while candidate discovery is
            # still blocked inside the independently tracked runner.
            await scan_started.wait()
            running = await deps.job_manager.get(job_id)  # type: ignore[union-attr]
            assert running is not None and running.state is JobState.RUNNING
            release_scan.set()
            await _wait_for_state(deps.job_manager, job_id, {JobState.SUCCEEDED})  # type: ignore[arg-type]

    scan.assert_awaited_once_with(pool, include_legacy_corruption=False, limit=1)
    job = await deps.job_manager.get(job_id)  # type: ignore[union-attr]
    assert job is not None
    assert job.progress.completed == 1
    assert dict(job.result_metrics) == {"failed": 0, "succeeded": 1, "total": 1}


@pytest.mark.asyncio
async def test_backfill_dry_run_remains_read_only_and_does_not_create_job():
    pool = MagicMock()
    deps = _deps(client=MagicMock(), pool=pool, doc_store=MagicMock())
    server = FastMCP("operator-backfill-dry-run-test")
    admin.register(server, deps)

    with patch.object(admin, "scan_candidates", new=AsyncMock(return_value=[])) as scan:
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool("backfill_degraded_documents", {"dry_run": True, "limit": 100})

    assert result.isError is False
    assert "Nothing to backfill" in result.content[0].text
    assert await deps.job_manager.active_task_count() == 0  # type: ignore[union-attr]
    assert await deps.job_manager.list(limit=10) == []  # type: ignore[union-attr]
    scan.assert_awaited_once_with(pool, include_legacy_corruption=False, limit=100)


@pytest.mark.asyncio
async def test_cancel_operator_job_cancels_tracked_runner_and_returns_terminal_state():
    started = asyncio.Event()

    async def blocking_refresh():
        started.set()
        await asyncio.Event().wait()

    client = MagicMock()
    client.refresh_cache = AsyncMock(side_effect=blocking_refresh)
    deps = _deps(client=client)
    server = FastMCP("operator-cancel-test")
    sync.register(server, deps)

    async with create_connected_server_and_client_session(server) as session:
        receipt = await session.call_tool("refresh_bddk_cache", {})
        job_id = _job_id(receipt)
        await started.wait()
        cancelled = await session.call_tool("cancel_operator_job", {"job_id": str(job_id)})

    assert cancelled.isError is False
    assert cancelled.structuredContent["job_id"] == str(job_id)
    assert cancelled.structuredContent["state"] == "cancelled"
    assert await deps.job_manager.active_task_count() == 0  # type: ignore[union-attr]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("sync_bddk_documents", {"concurrency": 0}),
        ("sync_bddk_documents", {"concurrency": 21}),
        ("sync_bddk_documents", {"force": "false"}),
        ("sync_bddk_documents", {"document_id": "../private"}),
        ("backfill_degraded_documents", {"limit": -1}),
        ("backfill_degraded_documents", {"limit": 0}),
        ("backfill_degraded_documents", {"limit": 1001}),
        ("backfill_degraded_documents", {"dry_run": "false"}),
        ("list_operator_jobs", {"limit": 0}),
        ("list_operator_jobs", {"limit": 101}),
        ("document_health", {"retryable_only": "false"}),
    ],
)
async def test_official_client_rejects_out_of_bounds_operator_inputs(tool_name, arguments):
    deps = _deps(client=MagicMock(), pool=MagicMock(), doc_store=MagicMock())
    server = FastMCP("operator-bound-test")
    sync.register(server, deps)
    admin.register(server, deps)

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(tool_name, arguments)

    assert result.isError is True
    assert await deps.job_manager.active_task_count() == 0  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_operator_job_tools_fail_safely_without_manager():
    deps = _deps(client=MagicMock(), with_manager=False)
    server = FastMCP("operator-no-manager-test")
    sync.register(server, deps)

    async with create_connected_server_and_client_session(server) as session:
        start = await session.call_tool("refresh_bddk_cache", {})
        lookup = await session.call_tool("get_operator_job", {"job_id": str(UUID(int=0))})

    assert start.isError is True
    assert lookup.isError is True
    assert "[ERROR:JOB_MANAGER_UNAVAILABLE]" in start.content[0].text
    assert "[ERROR:JOB_MANAGER_UNAVAILABLE]" in lookup.content[0].text


@pytest.mark.asyncio
async def test_failed_job_status_contains_safe_code_not_upstream_exception_text():
    sentinel = "RAW-UPSTREAM-FAILURE-SENTINEL"
    client = MagicMock()
    client.refresh_cache = AsyncMock(side_effect=RuntimeError(sentinel))
    deps = _deps(client=client)
    server = FastMCP("operator-safe-error-test")
    sync.register(server, deps)

    async with create_connected_server_and_client_session(server) as session:
        receipt = await session.call_tool("refresh_bddk_cache", {})
        job_id = _job_id(receipt)
        await _wait_for_state(deps.job_manager, job_id, {JobState.FAILED})  # type: ignore[arg-type]
        status = await session.call_tool("get_operator_job", {"job_id": str(job_id)})

    assert status.structuredContent["error_code"] == "cache_refresh_failed"
    assert sentinel not in status.content[0].text
