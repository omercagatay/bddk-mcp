"""Focused lifecycle and privacy tests for the standalone operator job core."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from uuid import UUID

import pytest

from bddk_mcp.jobs import (
    IdempotencyConflictError,
    InMemoryJobRepository,
    JobExecutionError,
    JobKind,
    JobManagerDrainingError,
    JobOutcome,
    JobProgress,
    JobState,
    OperatorJob,
    OperatorJobManager,
    fingerprint_arguments,
)
from bddk_mcp.jobs.models import digest_idempotency_key


async def _wait_for_state(
    manager: OperatorJobManager,
    job_id: UUID,
    states: set[JobState],
    *,
    timeout: float = 1.0,
) -> OperatorJob:
    async with asyncio.timeout(timeout):
        while True:
            job = await manager.get(job_id)
            assert job is not None
            if job.state in states:
                return job
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_job_records_use_uuid_and_retain_only_input_digests():
    repository = InMemoryJobRepository()
    manager = OperatorJobManager(repository, lease_retry_seconds=0.001)
    sentinel = "RAW-SECRET-ARGUMENT"

    job = await manager.submit(
        kind=JobKind.DOCUMENT_SYNC,
        arguments={"document_id": sentinel, "force": True},
        idempotency_key="RAW-IDEMPOTENCY-KEY",
        runner=lambda _context: asyncio.sleep(0, result=JobOutcome()),
    )
    await _wait_for_state(manager, job.job_id, {JobState.SUCCEEDED})
    stored = await manager.get(job.job_id)

    assert isinstance(job.job_id, UUID)
    assert stored is not None
    assert stored.kind is JobKind.DOCUMENT_SYNC
    assert len(stored.args_fingerprint) == 64
    assert stored.idempotency_digest == digest_idempotency_key("RAW-IDEMPOTENCY-KEY")
    assert sentinel not in repr(stored)
    assert "RAW-IDEMPOTENCY-KEY" not in repr(stored)


def test_argument_fingerprint_is_stable_type_aware_and_kind_aware():
    left = fingerprint_arguments(JobKind.DOCUMENT_SYNC, {"b": [1, True], "a": "x"})
    reordered = fingerprint_arguments(JobKind.DOCUMENT_SYNC, {"a": "x", "b": [1, True]})
    different_type = fingerprint_arguments(JobKind.DOCUMENT_SYNC, {"a": "x", "b": [1, 1]})
    different_kind = fingerprint_arguments(JobKind.BACKFILL, {"a": "x", "b": [1, True]})

    assert left == reordered
    assert left != different_type
    assert left != different_kind


@pytest.mark.asyncio
async def test_same_idempotency_key_and_fingerprint_reuses_one_job_and_runner():
    manager = OperatorJobManager(InMemoryJobRepository(), lease_retry_seconds=0.001)
    release = asyncio.Event()
    calls = 0

    async def runner(_context):
        nonlocal calls
        calls += 1
        await release.wait()
        return JobOutcome()

    first = await manager.submit(
        kind=JobKind.CORPUS_RECONCILE,
        arguments={"force": False},
        idempotency_key="same-key",
        runner=runner,
    )
    second = await manager.submit(
        kind=JobKind.CORPUS_RECONCILE,
        arguments={"force": False},
        idempotency_key="same-key",
        runner=runner,
    )

    assert first.job_id == second.job_id
    release.set()
    await _wait_for_state(manager, first.job_id, {JobState.SUCCEEDED})
    assert calls == 1


@pytest.mark.asyncio
async def test_concurrent_same_key_submissions_are_atomic():
    manager = OperatorJobManager(InMemoryJobRepository(), lease_retry_seconds=0.001)
    release = asyncio.Event()
    calls = 0

    async def runner(_context):
        nonlocal calls
        calls += 1
        await release.wait()

    submissions = await asyncio.gather(
        *(
            manager.submit(
                kind=JobKind.DOCUMENT_SYNC,
                arguments={"force": False},
                idempotency_key="concurrent-key",
                runner=runner,
            )
            for _ in range(25)
        )
    )
    assert len({job.job_id for job in submissions}) == 1
    release.set()
    await _wait_for_state(manager, submissions[0].job_id, {JobState.SUCCEEDED})
    assert calls == 1


@pytest.mark.asyncio
async def test_idempotency_key_mismatch_is_a_fixed_conflict_without_key_leakage():
    manager = OperatorJobManager(InMemoryJobRepository(), lease_retry_seconds=0.001)
    gate = asyncio.Event()

    async def runner(_context):
        await gate.wait()

    await manager.submit(
        kind=JobKind.DOCUMENT_SYNC,
        arguments={"force": False},
        idempotency_key="sensitive-key-value",
        runner=runner,
    )
    with pytest.raises(IdempotencyConflictError) as exc_info:
        await manager.submit(
            kind=JobKind.DOCUMENT_SYNC,
            arguments={"force": True},
            idempotency_key="sensitive-key-value",
            runner=runner,
        )
    assert "sensitive-key-value" not in str(exc_info.value)
    gate.set()
    await manager.drain(timeout=1)


@pytest.mark.asyncio
async def test_process_local_admission_allows_only_one_active_corpus_mutation():
    manager = OperatorJobManager(InMemoryJobRepository(), lease_retry_seconds=0.001)
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    active = 0
    maximum_active = 0

    async def first_runner(_context):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        first_started.set()
        await release_first.wait()
        active -= 1

    async def second_runner(_context):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        second_started.set()
        active -= 1

    first = await manager.submit(kind=JobKind.CACHE_REFRESH, arguments={}, runner=first_runner)
    second = await manager.submit(kind=JobKind.BACKFILL, arguments={}, runner=second_runner)
    await first_started.wait()
    await asyncio.sleep(0)

    second_queued = await manager.get(second.job_id)
    assert second_queued is not None and second_queued.state is JobState.QUEUED
    assert not second_started.is_set()

    release_first.set()
    await second_started.wait()
    await _wait_for_state(manager, second.job_id, {JobState.SUCCEEDED})
    assert (await manager.get(first.job_id)).state is JobState.SUCCEEDED  # type: ignore[union-attr]
    assert maximum_active == 1


@pytest.mark.asyncio
async def test_repository_lease_extends_single_flight_across_two_managers():
    repository = InMemoryJobRepository()
    first_manager = OperatorJobManager(repository, lease_retry_seconds=0.001)
    second_manager = OperatorJobManager(repository, lease_retry_seconds=0.001)
    release = asyncio.Event()
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    active = 0
    maximum_active = 0

    async def first_runner(_context):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        first_started.set()
        await release.wait()
        active -= 1

    async def second_runner(_context):
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        second_started.set()
        active -= 1

    first = await first_manager.submit(kind=JobKind.DOCUMENT_SYNC, arguments={"n": 1}, runner=first_runner)
    second = await second_manager.submit(kind=JobKind.VECTOR_RECONCILE, arguments={"n": 2}, runner=second_runner)
    await first_started.wait()
    await asyncio.sleep(0.01)
    assert not second_started.is_set()

    release.set()
    await second_started.wait()
    await _wait_for_state(first_manager, first.job_id, {JobState.SUCCEEDED})
    await _wait_for_state(second_manager, second.job_id, {JobState.SUCCEEDED})
    assert maximum_active == 1


@pytest.mark.asyncio
async def test_success_and_completed_with_errors_preserve_only_numeric_metrics():
    manager = OperatorJobManager(InMemoryJobRepository(), lease_retry_seconds=0.001)
    success = await manager.submit(
        kind=JobKind.DOCUMENT_SYNC,
        arguments={},
        runner=lambda _context: asyncio.sleep(
            0,
            result=JobOutcome.from_metrics({"downloaded": 4, "complete": True}),
        ),
    )
    partial = await manager.submit(
        kind=JobKind.BACKFILL,
        arguments={},
        runner=lambda _context: asyncio.sleep(
            0,
            result=JobOutcome.from_metrics({"succeeded": 3, "failed": 1}, completed_with_errors=True),
        ),
    )

    success_job = await _wait_for_state(manager, success.job_id, {JobState.SUCCEEDED})
    partial_job = await _wait_for_state(manager, partial.job_id, {JobState.COMPLETED_WITH_ERRORS})
    assert dict(success_job.result_metrics) == {"complete": True, "downloaded": 4}
    assert dict(partial_job.result_metrics) == {"failed": 1, "succeeded": 3}


@pytest.mark.asyncio
async def test_runner_exception_stores_only_generic_or_explicit_safe_error_code():
    manager = OperatorJobManager(InMemoryJobRepository(), lease_retry_seconds=0.001)
    sentinel = "RAW-UPSTREAM-ERROR-TEXT"

    async def generic_failure(_context):
        raise RuntimeError(sentinel)

    async def coded_failure(_context):
        raise JobExecutionError("upstream_unavailable")

    generic = await manager.submit(kind=JobKind.CACHE_REFRESH, arguments={}, runner=generic_failure)
    coded = await manager.submit(kind=JobKind.CACHE_REFRESH, arguments={"second": True}, runner=coded_failure)
    generic_job = await _wait_for_state(manager, generic.job_id, {JobState.FAILED})
    coded_job = await _wait_for_state(manager, coded.job_id, {JobState.FAILED})

    assert generic_job.error_code == "job_failed"
    assert coded_job.error_code == "upstream_unavailable"
    assert sentinel not in repr(generic_job)


@pytest.mark.asyncio
async def test_cancel_running_job_tracks_request_and_finishes_cancelled():
    manager = OperatorJobManager(InMemoryJobRepository(), lease_retry_seconds=0.001)
    started = asyncio.Event()

    async def runner(_context):
        started.set()
        await asyncio.Event().wait()

    job = await manager.submit(kind=JobKind.DOCUMENT_SYNC, arguments={}, runner=runner)
    await started.wait()
    cancelled = await manager.cancel(job.job_id)

    assert cancelled is not None and cancelled.state is JobState.CANCELLED
    assert cancelled.finished_at is not None
    assert await manager.active_task_count() == 0


@pytest.mark.asyncio
async def test_runner_cancelled_error_is_explicitly_persisted_as_cancelled():
    manager = OperatorJobManager(InMemoryJobRepository(), lease_retry_seconds=0.001)

    async def runner(_context):
        raise asyncio.CancelledError

    job = await manager.submit(kind=JobKind.BACKFILL, arguments={}, runner=runner)
    cancelled = await _wait_for_state(manager, job.job_id, {JobState.CANCELLED})
    assert cancelled.error_code is None


@pytest.mark.asyncio
async def test_cancel_queued_job_never_starts_its_runner():
    manager = OperatorJobManager(InMemoryJobRepository(), lease_retry_seconds=0.001)
    first_started = asyncio.Event()
    release = asyncio.Event()
    second_started = False

    async def first_runner(_context):
        first_started.set()
        await release.wait()

    async def second_runner(_context):
        nonlocal second_started
        second_started = True

    first = await manager.submit(kind=JobKind.DOCUMENT_SYNC, arguments={"n": 1}, runner=first_runner)
    second = await manager.submit(kind=JobKind.DOCUMENT_SYNC, arguments={"n": 2}, runner=second_runner)
    await first_started.wait()
    cancelled = await manager.cancel(second.job_id)

    assert cancelled is not None and cancelled.state is JobState.CANCELLED
    assert second_started is False
    release.set()
    await _wait_for_state(manager, first.job_id, {JobState.SUCCEEDED})


@pytest.mark.asyncio
async def test_progress_is_numeric_validated_and_monotonic_state_safe():
    manager = OperatorJobManager(InMemoryJobRepository(), lease_retry_seconds=0.001)

    async def runner(context):
        await context.update_progress(total=5, completed=2, succeeded=1, failed=1)
        return JobOutcome()

    job = await manager.submit(kind=JobKind.DOCUMENT_SYNC, arguments={}, runner=runner)
    completed = await _wait_for_state(manager, job.job_id, {JobState.SUCCEEDED})
    assert completed.progress == JobProgress(total=5, completed=2, succeeded=1, failed=1)

    with pytest.raises(ValueError):
        JobProgress(total=1, completed=2)


@pytest.mark.asyncio
async def test_progress_regression_fails_job_without_storing_runner_text():
    manager = OperatorJobManager(InMemoryJobRepository(), lease_retry_seconds=0.001)

    async def runner(context):
        await context.update_progress(total=5, completed=3, succeeded=3, failed=0)
        await context.update_progress(total=5, completed=2, succeeded=2, failed=0)

    job = await manager.submit(kind=JobKind.DOCUMENT_SYNC, arguments={}, runner=runner)
    failed = await _wait_for_state(manager, job.job_id, {JobState.FAILED})
    assert failed.error_code == "job_failed"


def test_outcome_rejects_free_text_metrics_before_persistence():
    with pytest.raises(ValueError, match="finite numbers or booleans"):
        JobOutcome(metrics=(("message", "RAW-RESULT-TEXT"),))  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_drain_closes_admission_cancels_after_timeout_and_awaits_tasks():
    manager = OperatorJobManager(InMemoryJobRepository(), lease_retry_seconds=0.001)
    started = asyncio.Event()

    async def runner(_context):
        started.set()
        await asyncio.Event().wait()

    first = await manager.submit(kind=JobKind.DOCUMENT_SYNC, arguments={"n": 1}, runner=runner)
    second = await manager.submit(kind=JobKind.BACKFILL, arguments={"n": 2}, runner=runner)
    await started.wait()
    report = await manager.drain(timeout=0)

    assert report.observed == 2
    assert report.cancelled == 2
    assert report.still_running == 0
    assert (await manager.get(first.job_id)).state is JobState.CANCELLED  # type: ignore[union-attr]
    assert (await manager.get(second.job_id)).state is JobState.CANCELLED  # type: ignore[union-attr]
    assert await manager.active_task_count() == 0
    assert manager.accepting is False
    with pytest.raises(JobManagerDrainingError):
        await manager.submit(kind=JobKind.CACHE_REFRESH, arguments={}, runner=runner)


@pytest.mark.asyncio
async def test_recovery_skips_queued_and_resolves_only_proven_abandoned_records():
    repository = InMemoryJobRepository()
    manager = OperatorJobManager(repository, lease_retry_seconds=0.001)
    now = datetime.now(UTC)
    jobs: list[OperatorJob] = []
    for state in (JobState.QUEUED, JobState.RUNNING, JobState.CANCEL_REQUESTED):
        job = OperatorJob.create(
            kind=JobKind.DOCUMENT_SYNC,
            args_fingerprint=fingerprint_arguments(JobKind.DOCUMENT_SYNC, {"state": state.value}),
            idempotency_digest=None,
            now=now,
        )
        created, is_new = await repository.create_or_reuse(job)
        assert is_new
        if state is not JobState.QUEUED:
            replacement = replace(
                created,
                state=state,
                started_at=now if state is JobState.RUNNING else None,
                updated_at=now,
                revision=1,
            )
            assert await repository.compare_and_set(replacement, expected_revision=0)
            job = replacement
        jobs.append(job)

    assert await manager.recover_interrupted() == 2
    queued, running, cancel_requested = [await manager.get(job.job_id) for job in jobs]
    assert queued is not None and queued.state is JobState.QUEUED and queued.error_code is None
    assert running is not None and running.state is JobState.INTERRUPTED
    assert running.error_code == "job_interrupted"
    assert cancel_requested is not None and cancel_requested.state is JobState.CANCELLED
    assert cancel_requested.error_code is None


@pytest.mark.asyncio
async def test_idempotent_retry_resumes_a_durable_queued_record():
    repository = InMemoryJobRepository()
    key = "resume-after-crash"
    arguments = {"force": False}
    queued = OperatorJob.create(
        kind=JobKind.DOCUMENT_SYNC,
        args_fingerprint=fingerprint_arguments(JobKind.DOCUMENT_SYNC, arguments),
        idempotency_digest=digest_idempotency_key(key),
    )
    stored, created = await repository.create_or_reuse(queued)
    assert created

    manager = OperatorJobManager(repository, lease_retry_seconds=0.001)
    resumed = await manager.submit(
        kind=JobKind.DOCUMENT_SYNC,
        arguments=arguments,
        idempotency_key=key,
        runner=lambda _context: asyncio.sleep(0, result=JobOutcome()),
    )

    assert resumed.job_id == stored.job_id
    completed = await _wait_for_state(manager, stored.job_id, {JobState.SUCCEEDED})
    assert completed.state is JobState.SUCCEEDED


@pytest.mark.asyncio
async def test_recovery_does_not_interrupt_job_holding_repository_lease():
    repository = InMemoryJobRepository()
    owner = OperatorJobManager(repository, lease_retry_seconds=0.001)
    recovering_manager = OperatorJobManager(repository, lease_retry_seconds=0.001)
    started = asyncio.Event()
    release = asyncio.Event()

    async def runner(_context):
        started.set()
        await release.wait()

    job = await owner.submit(kind=JobKind.CORPUS_RECONCILE, arguments={}, runner=runner)
    await started.wait()
    assert await recovering_manager.recover_interrupted() == 0
    running = await owner.get(job.job_id)
    assert running is not None and running.state is JobState.RUNNING

    release.set()
    await _wait_for_state(owner, job.job_id, {JobState.SUCCEEDED})


@pytest.mark.asyncio
async def test_terminal_history_is_bounded_and_list_is_newest_first():
    manager = OperatorJobManager(
        InMemoryJobRepository(),
        retained_history=3,
        lease_retry_seconds=0.001,
    )
    submitted: list[OperatorJob] = []
    for index in range(6):
        submitted.append(
            await manager.submit(
                kind=JobKind.DOCUMENT_SYNC,
                arguments={"index": index},
                runner=lambda _context: asyncio.sleep(0, result=JobOutcome()),
            )
        )

    await _wait_for_state(manager, submitted[-1].job_id, {JobState.SUCCEEDED})
    jobs = await manager.list(limit=3)
    assert len(jobs) == 3
    assert all(job.state.terminal for job in jobs)
    assert jobs == sorted(jobs, key=lambda job: (job.created_at, job.job_id.hex), reverse=True)
    assert await manager.get(submitted[0].job_id) is None
    with pytest.raises(ValueError):
        await manager.list(limit=4)
