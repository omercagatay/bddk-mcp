"""Durable operator job primitives. Production persistence is PostgreSQL."""

from bddk_mcp.jobs.manager import (
    IdempotencyConflictError,
    JobContext,
    JobExecutionError,
    JobManagerDrainingError,
    OperatorJobManager,
)
from bddk_mcp.jobs.models import (
    DrainReport,
    JobKind,
    JobOutcome,
    JobProgress,
    JobState,
    OperatorJob,
    fingerprint_arguments,
)
from bddk_mcp.jobs.postgres import (
    MIN_OPERATOR_JOB_POOL_SIZE,
    OPERATOR_JOB_MIGRATION_VERSION,
    OPERATOR_JOB_SCHEMA,
    OperatorJobSchemaError,
    OperatorJobSchemaNotReadyError,
    OperatorJobSchemaReadiness,
    OperatorJobStorageError,
    PostgresJobRepository,
    assert_operator_job_schema_ready,
    inspect_operator_job_schema,
)
from bddk_mcp.jobs.repository import JobExecutionLease, JobRepository

__all__ = (
    "DrainReport",
    "IdempotencyConflictError",
    "JobContext",
    "JobExecutionError",
    "JobExecutionLease",
    "JobKind",
    "JobManagerDrainingError",
    "JobOutcome",
    "JobProgress",
    "JobRepository",
    "JobState",
    "MIN_OPERATOR_JOB_POOL_SIZE",
    "OPERATOR_JOB_SCHEMA",
    "OPERATOR_JOB_MIGRATION_VERSION",
    "OperatorJob",
    "OperatorJobManager",
    "OperatorJobSchemaError",
    "OperatorJobSchemaNotReadyError",
    "OperatorJobSchemaReadiness",
    "OperatorJobStorageError",
    "PostgresJobRepository",
    "assert_operator_job_schema_ready",
    "fingerprint_arguments",
    "inspect_operator_job_schema",
)
