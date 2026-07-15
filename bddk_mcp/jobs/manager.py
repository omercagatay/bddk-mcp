"""Single-flight execution and lifecycle management for operator jobs."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Collection, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from bddk_mcp.jobs.models import (
    DrainReport,
    JobKind,
    JobOutcome,
    JobProgress,
    JobState,
    OperatorJob,
    can_transition,
    digest_idempotency_key,
    fingerprint_arguments,
)
from bddk_mcp.jobs.repository import JobExecutionLease, JobRepository

type JobRunner = Callable[["JobContext"], Awaitable[JobOutcome | None]]
_SAFE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class IdempotencyConflictError(RuntimeError):
    """Raised when a key is reused for a different kind or argument set."""

    def __init__(self) -> None:
        super().__init__("idempotency key is already associated with a different job request")


class JobManagerDrainingError(RuntimeError):
    """Raised when new work is submitted after shutdown admission closes."""

    def __init__(self) -> None:
        super().__init__("operator job manager is draining and does not accept new jobs")


class JobExecutionError(RuntimeError):
    """Runner failure with an explicitly safe, machine-readable code."""

    def __init__(self, code: str) -> None:
        if not _SAFE_CODE.fullmatch(code):
            raise ValueError("job error code must be a lowercase identifier")
        self.code = code
        # Do not place upstream text in the exception message either.
        super().__init__(code)


class JobCancellationRequested(asyncio.CancelledError):
    """Internal cooperative-cancellation signal."""


class JobContext:
    """Narrow runner interface for numeric progress and cancellation checks."""

    def __init__(self, manager: OperatorJobManager, job_id: UUID) -> None:
        self._manager = manager
        self.job_id = job_id

    async def checkpoint(self) -> None:
        """Raise cancellation at a safe runner boundary when requested."""

        job = await self._manager.get(self.job_id)
        if job is None or job.state in {JobState.CANCEL_REQUESTED, JobState.CANCELLED}:
            raise JobCancellationRequested()

    async def update_progress(
        self,
        *,
        total: int,
        completed: int,
        succeeded: int,
        failed: int,
    ) -> OperatorJob | None:
        """Persist validated numeric progress without free-text fields."""

        progress = JobProgress(
            total=total,
            completed=completed,
            succeeded=succeeded,
            failed=failed,
        )
        return await self._manager._set_progress(self.job_id, progress)


class OperatorJobManager:
    """Submit, serialize, observe, cancel, and drain operator mutations.

    The local lock guarantees single-flight within one process.  Every runner
    also acquires a repository lease, allowing a PostgreSQL implementation to
    extend the same invariant across replicas with an advisory lock.
    """

    def __init__(
        self,
        repository: JobRepository,
        *,
        retained_history: int = 1_000,
        lease_retry_seconds: float = 0.05,
    ) -> None:
        if retained_history < 1:
            raise ValueError("retained_history must be positive")
        if lease_retry_seconds <= 0:
            raise ValueError("lease_retry_seconds must be positive")
        self._repository = repository
        self._retained_history = retained_history
        self._lease_retry_seconds = lease_retry_seconds
        self._submission_lock = asyncio.Lock()
        self._corpus_admission = asyncio.Lock()
        self._tasks: dict[UUID, asyncio.Task[None]] = {}
        self._accepting = True

    @property
    def accepting(self) -> bool:
        return self._accepting

    async def active_task_count(self) -> int:
        async with self._submission_lock:
            return sum(not task.done() for task in self._tasks.values())

    async def submit(
        self,
        *,
        kind: JobKind,
        arguments: Mapping[str, Any] | None,
        runner: JobRunner,
        idempotency_key: str | None = None,
    ) -> OperatorJob:
        """Atomically create/reuse a job and schedule exactly one local runner."""

        if not isinstance(kind, JobKind):
            kind = JobKind(kind)
        fingerprint = fingerprint_arguments(kind, arguments)
        idempotency_digest = digest_idempotency_key(idempotency_key)
        candidate = OperatorJob.create(
            kind=kind,
            args_fingerprint=fingerprint,
            idempotency_digest=idempotency_digest,
        )

        async with self._submission_lock:
            if not self._accepting:
                raise JobManagerDrainingError()
            job, created = await self._repository.create_or_reuse(candidate)
            if not created:
                if job.kind is not kind or job.args_fingerprint != fingerprint:
                    raise IdempotencyConflictError()
                # A process may have crashed after persisting QUEUED but before
                # its local task obtained the execution lease. A retry with the
                # same idempotency key may safely resume that record. Competing
                # replicas can schedule it too: only one obtains the repository
                # lease, and the others observe the resulting terminal state.
                if job.state is JobState.QUEUED and job.job_id not in self._tasks:
                    self._schedule(job.job_id, runner)
                return job

            self._schedule(job.job_id, runner)
            return job

    def _schedule(self, job_id: UUID, runner: JobRunner) -> None:
        """Create one local task while the submission lock is held."""

        task = asyncio.create_task(
            self._execute(job_id, runner),
            name=f"bddk-operator-job-{job_id}",
        )
        self._tasks[job_id] = task
        task.add_done_callback(lambda completed: self._on_task_done(job_id, completed))

    async def get(self, job_id: UUID) -> OperatorJob | None:
        return await self._repository.get(job_id)

    async def list(
        self,
        *,
        limit: int = 100,
        states: Collection[JobState] | None = None,
    ) -> list[OperatorJob]:
        if not 1 <= limit <= self._retained_history:
            raise ValueError(f"limit must be between 1 and {self._retained_history}")
        return await self._repository.list(limit=limit, states=states)

    async def cancel(self, job_id: UUID) -> OperatorJob | None:
        """Request cancellation and await cleanup for a locally owned task."""

        job = await self._transition(job_id, {JobState.QUEUED, JobState.RUNNING}, JobState.CANCEL_REQUESTED)
        if job is None:
            return None
        if job.state.terminal:
            return job

        async with self._submission_lock:
            task = self._tasks.get(job_id)
            if task is not None and not task.done():
                task.cancel()

        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
            job = await self._transition(
                job_id,
                {JobState.QUEUED, JobState.RUNNING, JobState.CANCEL_REQUESTED},
                JobState.CANCELLED,
            )
            await self._repository.prune_terminal(keep=self._retained_history)
        return job

    async def drain(self, *, timeout: float) -> DrainReport:
        """Close admission, drain until timeout, then cancel and await survivors."""

        if timeout < 0:
            raise ValueError("drain timeout cannot be negative")
        async with self._submission_lock:
            self._accepting = False
            tasks = dict(self._tasks)

        if not tasks:
            return DrainReport(observed=0, completed=0, cancelled=0, still_running=0)

        done, pending = await asyncio.wait(tasks.values(), timeout=timeout)
        pending_ids = {job_id for job_id, task in tasks.items() if task in pending}

        for job_id in pending_ids:
            await self._transition(job_id, {JobState.QUEUED, JobState.RUNNING}, JobState.CANCEL_REQUESTED)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for job_id in pending_ids:
            await self._transition(
                job_id,
                {JobState.QUEUED, JobState.RUNNING, JobState.CANCEL_REQUESTED},
                JobState.CANCELLED,
            )

        await self._repository.prune_terminal(keep=self._retained_history)
        still_running = sum(not task.done() for task in tasks.values())
        return DrainReport(
            observed=len(tasks),
            completed=len(done),
            cancelled=len(pending_ids),
            still_running=still_running,
        )

    async def recover_interrupted(self) -> int:
        """Resolve abandoned durable records without stealing live queued work.

        RUNNING records acquired the repository lease before that transition,
        so obtaining it here proves no live worker still owns the mutation.
        QUEUED records are deliberately not guessed stale: another replica may
        be between persistence and lease acquisition. They can be resumed by an
        idempotent resubmission or cancelled explicitly.
        """

        async with self._submission_lock:
            if any(not task.done() for task in self._tasks.values()):
                raise RuntimeError("cannot recover interrupted jobs while local jobs are active")
        unfinished = await self._repository.list_unfinished()
        recovered = 0
        for job in unfinished:
            if job.state is JobState.QUEUED:
                continue
            lease = await self._repository.try_acquire_execution_lease(
                job_id=job.job_id,
                resource=job.kind.execution_resource,
            )
            if lease is None:
                continue
            try:
                if job.state is JobState.CANCEL_REQUESTED:
                    target = JobState.CANCELLED
                    error_code = None
                else:
                    target = JobState.INTERRUPTED
                    error_code = "job_interrupted"
                updated = await self._transition(
                    job.job_id,
                    {job.state},
                    target,
                    error_code=error_code,
                )
                if updated is not None and updated.state is target:
                    recovered += 1
            finally:
                await lease.release()
        await self._repository.prune_terminal(keep=self._retained_history)
        return recovered

    async def _execute(self, job_id: UUID, runner: JobRunner) -> None:
        lease: JobExecutionLease | None = None
        context = JobContext(self, job_id)
        try:
            async with self._corpus_admission:
                while lease is None:
                    await context.checkpoint()
                    job = await self.get(job_id)
                    if job is None:
                        return
                    lease = await self._repository.try_acquire_execution_lease(
                        job_id=job_id,
                        resource=job.kind.execution_resource,
                    )
                    if lease is None:
                        await asyncio.sleep(self._lease_retry_seconds)

                job = await self._transition(job_id, {JobState.QUEUED}, JobState.RUNNING)
                if job is None or job.state is not JobState.RUNNING:
                    if job is not None and job.state is JobState.CANCEL_REQUESTED:
                        await self._transition(job_id, {JobState.CANCEL_REQUESTED}, JobState.CANCELLED)
                    return

                outcome = await runner(context)
                await context.checkpoint()
                if outcome is None:
                    outcome = JobOutcome()
                if not isinstance(outcome, JobOutcome):
                    raise TypeError("job runner must return JobOutcome or None")
                target = JobState.COMPLETED_WITH_ERRORS if outcome.completed_with_errors else JobState.SUCCEEDED
                completed = await self._transition(
                    job_id,
                    {JobState.RUNNING},
                    target,
                    result_metrics=outcome.metrics,
                )
                if completed is not None and completed.state is JobState.CANCEL_REQUESTED:
                    await self._transition(job_id, {JobState.CANCEL_REQUESTED}, JobState.CANCELLED)
        except asyncio.CancelledError:
            await self._transition(
                job_id,
                {JobState.QUEUED, JobState.RUNNING, JobState.CANCEL_REQUESTED},
                JobState.CANCELLED,
            )
        except Exception as exc:
            code = exc.code if isinstance(exc, JobExecutionError) else "job_failed"
            current = await self.get(job_id)
            if current is not None and current.state is JobState.CANCEL_REQUESTED:
                await self._transition(job_id, {JobState.CANCEL_REQUESTED}, JobState.CANCELLED)
            else:
                await self._transition(
                    job_id,
                    {JobState.QUEUED, JobState.RUNNING},
                    JobState.FAILED,
                    error_code=code,
                )
        except BaseException:
            await self._transition(
                job_id,
                {JobState.QUEUED, JobState.RUNNING, JobState.CANCEL_REQUESTED},
                JobState.INTERRUPTED,
                error_code="job_interrupted",
            )
            raise
        finally:
            if lease is not None:
                await lease.release()
            await self._repository.prune_terminal(keep=self._retained_history)

    async def _set_progress(self, job_id: UUID, progress: JobProgress) -> OperatorJob | None:
        current = await self.get(job_id)
        if current is None or current.state is not JobState.RUNNING:
            return current
        previous = current.progress
        if (
            progress.completed < previous.completed
            or progress.succeeded < previous.succeeded
            or progress.failed < previous.failed
            or (previous.total and progress.total < previous.total)
        ):
            raise ValueError("job progress cannot move backwards")
        return await self._transition(job_id, {JobState.RUNNING}, JobState.RUNNING, progress=progress)

    async def _transition(
        self,
        job_id: UUID,
        expected: Collection[JobState],
        target: JobState,
        *,
        progress: JobProgress | None = None,
        result_metrics: tuple[tuple[str, int | float | bool], ...] | None = None,
        error_code: str | None = None,
    ) -> OperatorJob | None:
        expected_states = set(expected)
        while True:
            current = await self._repository.get(job_id)
            if current is None:
                return None
            if current.state not in expected_states:
                return current
            if target is not current.state and not can_transition(current.state, target):
                raise ValueError(f"invalid job transition {current.state.value} -> {target.value}")

            now = datetime.now(UTC)
            started_at = current.started_at
            finished_at = current.finished_at
            if target is JobState.RUNNING and started_at is None:
                started_at = now
            if target.terminal:
                finished_at = now

            replacement = replace(
                current,
                state=target,
                updated_at=now,
                revision=current.revision + 1,
                started_at=started_at,
                finished_at=finished_at,
                progress=progress or current.progress,
                result_metrics=result_metrics if result_metrics is not None else current.result_metrics,
                error_code=error_code,
            )
            if await self._repository.compare_and_set(replacement, expected_revision=current.revision):
                return replacement

    def _on_task_done(self, job_id: UUID, task: asyncio.Task[None]) -> None:
        self._tasks.pop(job_id, None)
        try:
            task.exception()
        except asyncio.CancelledError:
            pass
