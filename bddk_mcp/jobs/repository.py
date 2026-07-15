"""Repository contracts for durable operator jobs.

The protocol is intentionally compatible with a future PostgreSQL adapter:
creation/idempotency and compare-and-set updates are atomic, while execution
leases can be backed by a connection-scoped advisory lock.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection
from typing import Protocol, runtime_checkable
from uuid import UUID

from bddk_mcp.jobs.models import JobState, OperatorJob


@runtime_checkable
class JobExecutionLease(Protocol):
    """Exclusive execution lease, e.g. a PostgreSQL advisory-lock session."""

    async def release(self) -> None:
        """Release the lease. Implementations must be idempotent."""


@runtime_checkable
class JobRepository(Protocol):
    """Atomic persistence operations required by :class:`OperatorJobManager`."""

    async def create_or_reuse(self, job: OperatorJob) -> tuple[OperatorJob, bool]:
        """Create a job, or return the job owning its idempotency digest."""

    async def get(self, job_id: UUID) -> OperatorJob | None:
        """Return one job by ID."""

    async def list(self, *, limit: int, states: Collection[JobState] | None = None) -> list[OperatorJob]:
        """Return newest-first jobs, optionally filtered by state."""

    async def list_unfinished(self) -> list[OperatorJob]:
        """Return every non-terminal record for crash recovery."""

    async def compare_and_set(self, replacement: OperatorJob, *, expected_revision: int) -> bool:
        """Atomically replace a record only at the expected revision."""

    async def try_acquire_execution_lease(
        self,
        *,
        job_id: UUID,
        resource: str,
    ) -> JobExecutionLease | None:
        """Try to exclusively lease a conflict resource for one job."""

    async def prune_terminal(self, *, keep: int) -> int:
        """Delete oldest terminal history while retaining active jobs."""


class _InMemoryExecutionLease:
    def __init__(self, repository: InMemoryJobRepository, resource: str, job_id: UUID) -> None:
        self._repository = repository
        self._resource = resource
        self._job_id = job_id
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        async with self._repository._lock:
            if self._repository._leases.get(self._resource) == self._job_id:
                del self._repository._leases[self._resource]


class InMemoryJobRepository:
    """Concurrency-safe repository for tests and single-process local use."""

    def __init__(self) -> None:
        self._jobs: dict[UUID, OperatorJob] = {}
        self._idempotency: dict[str, UUID] = {}
        self._leases: dict[str, UUID] = {}
        self._lock = asyncio.Lock()

    async def create_or_reuse(self, job: OperatorJob) -> tuple[OperatorJob, bool]:
        async with self._lock:
            if job.idempotency_digest is not None:
                existing_id = self._idempotency.get(job.idempotency_digest)
                if existing_id is not None:
                    return self._jobs[existing_id], False
            if job.job_id in self._jobs:
                raise ValueError("job ID already exists")
            self._jobs[job.job_id] = job
            if job.idempotency_digest is not None:
                self._idempotency[job.idempotency_digest] = job.job_id
            return job, True

    async def get(self, job_id: UUID) -> OperatorJob | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def list(self, *, limit: int, states: Collection[JobState] | None = None) -> list[OperatorJob]:
        if limit < 1:
            raise ValueError("limit must be positive")
        wanted = set(states) if states is not None else None
        async with self._lock:
            jobs = [job for job in self._jobs.values() if wanted is None or job.state in wanted]
            jobs.sort(key=lambda job: (job.created_at, job.job_id.hex), reverse=True)
            return jobs[:limit]

    async def list_unfinished(self) -> list[OperatorJob]:
        async with self._lock:
            jobs = [job for job in self._jobs.values() if not job.state.terminal]
            jobs.sort(key=lambda job: (job.created_at, job.job_id.hex))
            return jobs

    async def compare_and_set(self, replacement: OperatorJob, *, expected_revision: int) -> bool:
        async with self._lock:
            current = self._jobs.get(replacement.job_id)
            if current is None or current.revision != expected_revision:
                return False
            if replacement.revision != expected_revision + 1:
                raise ValueError("replacement revision must advance by one")
            immutable_current = (
                current.job_id,
                current.kind,
                current.args_fingerprint,
                current.idempotency_digest,
                current.created_at,
            )
            immutable_replacement = (
                replacement.job_id,
                replacement.kind,
                replacement.args_fingerprint,
                replacement.idempotency_digest,
                replacement.created_at,
            )
            if immutable_replacement != immutable_current:
                raise ValueError("job identity fields are immutable")
            self._jobs[replacement.job_id] = replacement
            return True

    async def try_acquire_execution_lease(
        self,
        *,
        job_id: UUID,
        resource: str,
    ) -> JobExecutionLease | None:
        async with self._lock:
            owner = self._leases.get(resource)
            if owner is not None:
                return None
            self._leases[resource] = job_id
            return _InMemoryExecutionLease(self, resource, job_id)

    async def prune_terminal(self, *, keep: int) -> int:
        if keep < 0:
            raise ValueError("keep cannot be negative")
        async with self._lock:
            terminal = sorted(
                (job for job in self._jobs.values() if job.state.terminal),
                key=lambda job: (job.finished_at or job.updated_at, job.job_id.hex),
                reverse=True,
            )
            remove = terminal[keep:]
            for job in remove:
                del self._jobs[job.job_id]
                if job.idempotency_digest is not None:
                    self._idempotency.pop(job.idempotency_digest, None)
            return len(remove)
