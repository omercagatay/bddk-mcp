"""Persistence contract for operator jobs. Production uses PostgresJobRepository."""

from __future__ import annotations

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
        """Return newest-first jobs, optionally filtered by states."""

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
