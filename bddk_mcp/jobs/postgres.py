"""PostgreSQL persistence and SELECT-only readiness for operator jobs.

Schema DDL belongs exclusively to the global :mod:`bddk_mcp.migrations`
framework. This module persists only digests and bounded machine-readable
fields and never mutates database structure.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Collection, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import asyncpg

from bddk_mcp.corpus_coordination import CORPUS_JOB_EXECUTION_ADVISORY_KEY
from bddk_mcp.jobs.models import JobKind, JobProgress, JobState, OperatorJob, normalize_metrics
from bddk_mcp.jobs.repository import JobExecutionLease
from bddk_mcp.migrations import MigrationError, inspect_migration_state_connection

OPERATOR_JOB_MIGRATION_VERSION = 2
MIN_OPERATOR_JOB_POOL_SIZE = 2
OPERATOR_JOB_SCHEMA = "bddk_operator"

_JOBS_TABLE = "operator_jobs"
_JOBS_RELATION = f"{OPERATOR_JOB_SCHEMA}.{_JOBS_TABLE}"
_TERMINAL_STATES = tuple(state.value for state in JobState if state.terminal)
_UNFINISHED_STATES = tuple(state.value for state in JobState if not state.terminal)

_REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    _JOBS_TABLE: frozenset(
        {
            "job_id",
            "kind",
            "state",
            "args_fingerprint",
            "idempotency_digest",
            "created_at",
            "updated_at",
            "revision",
            "started_at",
            "finished_at",
            "progress_total",
            "progress_completed",
            "progress_succeeded",
            "progress_failed",
            "result_metrics",
            "error_code",
        }
    ),
}
_REQUIRED_COLUMN_SPECS: dict[str, dict[str, tuple[str, bool]]] = {
    _JOBS_TABLE: {
        "job_id": ("uuid", True),
        "kind": ("text", True),
        "state": ("text", True),
        "args_fingerprint": ("text", True),
        "idempotency_digest": ("text", False),
        "created_at": ("timestamptz", True),
        "updated_at": ("timestamptz", True),
        "revision": ("int8", True),
        "started_at": ("timestamptz", False),
        "finished_at": ("timestamptz", False),
        "progress_total": ("int8", True),
        "progress_completed": ("int8", True),
        "progress_succeeded": ("int8", True),
        "progress_failed": ("int8", True),
        "result_metrics": ("jsonb", True),
        "error_code": ("text", False),
    },
}
_REQUIRED_INDEXES = frozenset(
    {
        "operator_jobs_pkey",
        "operator_jobs_idempotency_digest_uq",
        "operator_jobs_list_idx",
        "operator_jobs_terminal_idx",
        "operator_jobs_unfinished_idx",
    }
)
_CONSTRAINT_REQUIREMENTS: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {
    _JOBS_TABLE: {
        "operator_jobs_pkey": ("p", ("primary key", "job_id")),
        "operator_jobs_kind_check": (
            "c",
            tuple(kind.value for kind in JobKind),
        ),
        "operator_jobs_state_check": (
            "c",
            tuple(state.value for state in JobState),
        ),
        "operator_jobs_args_fingerprint_check": ("c", ("args_fingerprint", "^[0-9a-f]{64}$")),
        "operator_jobs_idempotency_digest_check": ("c", ("idempotency_digest", "^[0-9a-f]{64}$")),
        "operator_jobs_revision_check": ("c", ("revision >= 0",)),
        "operator_jobs_progress_total_check": ("c", ("progress_total >= 0",)),
        "operator_jobs_progress_completed_check": ("c", ("progress_completed >= 0",)),
        "operator_jobs_progress_succeeded_check": ("c", ("progress_succeeded >= 0",)),
        "operator_jobs_progress_failed_check": ("c", ("progress_failed >= 0",)),
        "operator_jobs_result_metrics_check": ("c", ("jsonb_typeof", "result_metrics", "object")),
        "operator_jobs_error_code_check": ("c", ("error_code", "^[a-z][a-z0-9_]{0,63}$")),
        "operator_jobs_progress_within_total": (
            "c",
            ("progress_total", "progress_completed", "progress_succeeded", "progress_failed"),
        ),
        "operator_jobs_terminal_timestamp": (
            "c",
            ("finished_at", "succeeded", "completed_with_errors", "failed", "cancelled", "interrupted"),
        ),
    },
}
_INDEX_REQUIREMENTS: dict[str, tuple[bool, tuple[str, ...], tuple[str, ...]]] = {
    "operator_jobs_pkey": (True, ("(job_id)",), ()),
    "operator_jobs_idempotency_digest_uq": (
        True,
        ("(idempotency_digest)",),
        ("idempotency_digest", "is not null"),
    ),
    "operator_jobs_list_idx": (False, ("(created_at desc, job_id desc)",), ()),
    "operator_jobs_unfinished_idx": (
        False,
        ("(created_at, job_id)",),
        ("queued", "running", "cancel_requested"),
    ),
    "operator_jobs_terminal_idx": (
        False,
        ("(finished_at desc, job_id desc)",),
        ("succeeded", "completed_with_errors", "failed", "cancelled", "interrupted"),
    ),
}

_RELATIONS_SQL = """
SELECT requested.relation_name,
       relation.relkind
FROM unnest($2::text[]) AS requested(relation_name)
LEFT JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.nspname = $1
LEFT JOIN pg_catalog.pg_class AS relation
  ON relation.relnamespace = namespace.oid
 AND relation.relname = requested.relation_name
"""

_COLUMNS_SQL = """
SELECT requested.relation_name AS table_name,
       attribute.attname AS column_name,
       data_type.typname AS data_type,
       attribute.attnotnull AS not_null
FROM unnest($2::text[]) AS requested(relation_name)
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.nspname = $1
JOIN pg_catalog.pg_class AS relation
  ON relation.relnamespace = namespace.oid
 AND relation.relname = requested.relation_name
JOIN pg_catalog.pg_attribute AS attribute
  ON attribute.attrelid = relation.oid
JOIN pg_catalog.pg_type AS data_type
  ON data_type.oid = attribute.atttypid
WHERE attribute.attnum > 0
  AND NOT attribute.attisdropped
"""

_INDEXES_SQL = """
SELECT index_relation.relname AS index_name,
       table_relation.relname AS table_name,
       index_metadata.indisunique AS is_unique,
       index_metadata.indisvalid AS is_valid,
       index_metadata.indisready AS is_ready,
       pg_get_indexdef(index_relation.oid) AS definition,
       pg_get_expr(index_metadata.indpred, index_metadata.indrelid) AS predicate
FROM pg_catalog.pg_namespace AS namespace
JOIN pg_catalog.pg_class AS index_relation
  ON index_relation.relnamespace = namespace.oid
JOIN pg_catalog.pg_index AS index_metadata
  ON index_metadata.indexrelid = index_relation.oid
JOIN pg_catalog.pg_class AS table_relation
  ON table_relation.oid = index_metadata.indrelid
WHERE namespace.nspname = $1
  AND index_relation.relname = ANY($2::text[])
"""

_CONSTRAINTS_SQL = """
SELECT relation.relname AS table_name,
       constraint_record.conname AS constraint_name,
       constraint_record.contype AS constraint_type,
       constraint_record.convalidated AS is_valid,
       pg_get_constraintdef(constraint_record.oid, true) AS definition
FROM pg_catalog.pg_namespace AS namespace
JOIN pg_catalog.pg_class AS relation
  ON relation.relnamespace = namespace.oid
JOIN pg_catalog.pg_constraint AS constraint_record
  ON constraint_record.conrelid = relation.oid
WHERE namespace.nspname = $1
  AND relation.relname = ANY($2::text[])
"""

_RETURNING_COLUMNS = """
job_id, kind, state, args_fingerprint, idempotency_digest,
created_at, updated_at, revision, started_at, finished_at,
progress_total, progress_completed, progress_succeeded, progress_failed,
result_metrics, error_code
"""

_INSERT_SQL = f"""
INSERT INTO bddk_operator.operator_jobs (
    job_id, kind, state, args_fingerprint, idempotency_digest,
    created_at, updated_at, revision, started_at, finished_at,
    progress_total, progress_completed, progress_succeeded, progress_failed,
    result_metrics, error_code
) VALUES (
    $1, $2, $3, $4, $5,
    $6, $7, $8, $9, $10,
    $11, $12, $13, $14,
    $15::jsonb, $16
)
RETURNING {_RETURNING_COLUMNS}
"""

_GET_SQL = f"SELECT {_RETURNING_COLUMNS} FROM bddk_operator.operator_jobs WHERE job_id = $1"
_GET_FOR_UPDATE_SQL = _GET_SQL + " FOR UPDATE"
_GET_BY_IDEMPOTENCY_SQL = f"""
SELECT {_RETURNING_COLUMNS}
FROM bddk_operator.operator_jobs
WHERE idempotency_digest = $1
FOR UPDATE
"""
_LIST_SQL = f"""
SELECT {_RETURNING_COLUMNS}
FROM bddk_operator.operator_jobs
ORDER BY created_at DESC, job_id DESC
LIMIT $1
"""
_LIST_STATES_SQL = f"""
SELECT {_RETURNING_COLUMNS}
FROM bddk_operator.operator_jobs
WHERE state = ANY($1::text[])
ORDER BY created_at DESC, job_id DESC
LIMIT $2
"""
_LIST_UNFINISHED_SQL = f"""
SELECT {_RETURNING_COLUMNS}
FROM bddk_operator.operator_jobs
WHERE state = ANY($1::text[])
ORDER BY created_at, job_id
"""
_UPDATE_SQL = f"""
UPDATE bddk_operator.operator_jobs
SET state = $2,
    updated_at = $3,
    revision = $4,
    started_at = $5,
    finished_at = $6,
    progress_total = $7,
    progress_completed = $8,
    progress_succeeded = $9,
    progress_failed = $10,
    result_metrics = $11::jsonb,
    error_code = $12
WHERE job_id = $1 AND revision = $13
RETURNING {_RETURNING_COLUMNS}
"""
_PRUNE_SQL = """
WITH expired AS (
    SELECT job_id
    FROM bddk_operator.operator_jobs
    WHERE state = ANY($1::text[])
    ORDER BY finished_at DESC, job_id DESC
    OFFSET $2
)
DELETE FROM bddk_operator.operator_jobs AS jobs
USING expired
WHERE jobs.job_id = expired.job_id
RETURNING jobs.job_id
"""


class OperatorJobStorageError(RuntimeError):
    """Sanitized database-operation failure."""


class OperatorJobSchemaError(RuntimeError):
    """Sanitized schema migration or compatibility failure."""


class OperatorJobSchemaNotReadyError(OperatorJobSchemaError):
    """Raised when the operator job schema is absent or incompatible."""


@dataclass(frozen=True, slots=True)
class OperatorJobSchemaReadiness:
    """Read-only assessment of the durable job schema."""

    migration_version: int | None = None
    required_migration_version: int | None = None
    missing_relations: tuple[str, ...] = ()
    invalid_relations: tuple[str, ...] = ()
    missing_columns: tuple[str, ...] = ()
    invalid_columns: tuple[str, ...] = ()
    unexpected_columns: tuple[str, ...] = ()
    missing_constraints: tuple[str, ...] = ()
    invalid_constraints: tuple[str, ...] = ()
    missing_indexes: tuple[str, ...] = ()
    invalid_indexes: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return (
            (
                self.required_migration_version is None
                or (self.migration_version is not None and self.migration_version >= self.required_migration_version)
            )
            and not self.missing_relations
            and not self.invalid_relations
            and not self.missing_columns
            and not self.invalid_columns
            and not self.unexpected_columns
            and not self.missing_constraints
            and not self.invalid_constraints
            and not self.missing_indexes
            and not self.invalid_indexes
        )

    def summary(self) -> str:
        parts: list[str] = []
        if self.required_migration_version is not None and (
            self.migration_version is None or self.migration_version < self.required_migration_version
        ):
            shown = "missing" if self.migration_version is None else str(self.migration_version)
            parts.append(f"global migration version is {shown}; expected at least {self.required_migration_version}")
        if self.missing_relations:
            parts.append("missing tables: " + ", ".join(self.missing_relations))
        if self.invalid_relations:
            parts.append("invalid relations: " + ", ".join(self.invalid_relations))
        if self.missing_columns:
            parts.append("missing columns: " + ", ".join(self.missing_columns))
        if self.invalid_columns:
            parts.append("invalid columns: " + ", ".join(self.invalid_columns))
        if self.unexpected_columns:
            parts.append("unexpected columns: " + ", ".join(self.unexpected_columns))
        if self.missing_constraints:
            parts.append("missing constraints: " + ", ".join(self.missing_constraints))
        if self.invalid_constraints:
            parts.append("invalid constraints: " + ", ".join(self.invalid_constraints))
        if self.missing_indexes:
            parts.append("missing indexes: " + ", ".join(self.missing_indexes))
        if self.invalid_indexes:
            parts.append("invalid indexes: " + ", ".join(self.invalid_indexes))
        return "; ".join(parts) if parts else "ready"


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def _normalize_sql(value: str) -> str:
    return " ".join(value.replace('"', "").lower().split())


def _catalog_code(value: Any) -> str:
    return value.decode("ascii") if isinstance(value, bytes) else str(value)


async def _inspect_operator_job_schema_connection(
    connection: Any,
    *,
    migration_version: int | None = None,
    required_migration_version: int | None = None,
) -> OperatorJobSchemaReadiness:
    table_names = sorted(_REQUIRED_COLUMNS)
    relation_rows = await connection.fetch(_RELATIONS_SQL, OPERATOR_JOB_SCHEMA, table_names)
    relation_kinds = {
        str(_row_value(row, "relation_name", "")): _catalog_code(_row_value(row, "relkind", ""))
        if _row_value(row, "relkind") is not None
        else None
        for row in relation_rows
    }
    missing_relations = tuple(
        f"{OPERATOR_JOB_SCHEMA}.{table_name}" for table_name in table_names if relation_kinds.get(table_name) is None
    )
    invalid_relations = tuple(
        f"{OPERATOR_JOB_SCHEMA}.{table_name}"
        for table_name in table_names
        if relation_kinds.get(table_name) is not None and relation_kinds[table_name] not in {"r", "p"}
    )

    column_rows = await connection.fetch(_COLUMNS_SQL, OPERATOR_JOB_SCHEMA, table_names)
    actual_columns: dict[str, dict[str, tuple[str, bool]]] = {table_name: {} for table_name in table_names}
    for row in column_rows:
        table_name = str(_row_value(row, "table_name", ""))
        column_name = str(_row_value(row, "column_name", ""))
        if table_name in actual_columns and column_name:
            actual_columns[table_name][column_name] = (
                str(_row_value(row, "data_type", "")),
                bool(_row_value(row, "not_null", False)),
            )
    missing_columns = tuple(
        sorted(
            f"{OPERATOR_JOB_SCHEMA}.{table_name}.{column_name}"
            for table_name, required in _REQUIRED_COLUMNS.items()
            if relation_kinds.get(table_name) in {"r", "p"}
            for column_name in required - set(actual_columns[table_name])
        )
    )
    invalid_columns = tuple(
        sorted(
            f"{OPERATOR_JOB_SCHEMA}.{table_name}.{column_name}"
            for table_name, required in _REQUIRED_COLUMN_SPECS.items()
            for column_name, expected in required.items()
            if column_name in actual_columns[table_name] and actual_columns[table_name][column_name] != expected
        )
    )
    unexpected_columns = tuple(
        sorted(
            f"{OPERATOR_JOB_SCHEMA}.{table_name}.{column_name}"
            for table_name, actual in actual_columns.items()
            for column_name in set(actual) - _REQUIRED_COLUMNS[table_name]
        )
    )

    constraint_rows = await connection.fetch(_CONSTRAINTS_SQL, OPERATOR_JOB_SCHEMA, table_names)
    actual_constraints = {
        (str(_row_value(row, "table_name", "")), str(_row_value(row, "constraint_name", ""))): row
        for row in constraint_rows
    }
    missing_constraints_list: list[str] = []
    invalid_constraints_list: list[str] = []
    for table_name, requirements in _CONSTRAINT_REQUIREMENTS.items():
        for constraint_name, (expected_type, fragments) in requirements.items():
            row = actual_constraints.get((table_name, constraint_name))
            qualified_name = f"{OPERATOR_JOB_SCHEMA}.{table_name}.{constraint_name}"
            if row is None:
                missing_constraints_list.append(qualified_name)
                continue
            definition = _normalize_sql(str(_row_value(row, "definition", "")))
            if (
                _catalog_code(_row_value(row, "constraint_type", "")) != expected_type
                or not bool(_row_value(row, "is_valid", False))
                or any(_normalize_sql(fragment) not in definition for fragment in fragments)
            ):
                invalid_constraints_list.append(qualified_name)

    index_rows = await connection.fetch(_INDEXES_SQL, OPERATOR_JOB_SCHEMA, sorted(_REQUIRED_INDEXES))
    actual_indexes = {str(_row_value(row, "index_name", "")): row for row in index_rows}
    missing_indexes = tuple(sorted(f"{OPERATOR_JOB_SCHEMA}.{name}" for name in _REQUIRED_INDEXES - set(actual_indexes)))
    invalid_indexes_list: list[str] = []
    for index_name, (expected_unique, definition_fragments, predicate_fragments) in _INDEX_REQUIREMENTS.items():
        row = actual_indexes.get(index_name)
        if row is None:
            continue
        definition = _normalize_sql(str(_row_value(row, "definition", "")))
        predicate_value = _row_value(row, "predicate")
        predicate = _normalize_sql(str(predicate_value)) if predicate_value is not None else ""
        if (
            str(_row_value(row, "table_name", "")) != _JOBS_TABLE
            or bool(_row_value(row, "is_unique", False)) != expected_unique
            or not bool(_row_value(row, "is_valid", False))
            or not bool(_row_value(row, "is_ready", False))
            or any(_normalize_sql(fragment) not in definition for fragment in definition_fragments)
            or any(_normalize_sql(fragment) not in predicate for fragment in predicate_fragments)
            or (not predicate_fragments and predicate)
        ):
            invalid_indexes_list.append(f"{OPERATOR_JOB_SCHEMA}.{index_name}")

    return OperatorJobSchemaReadiness(
        migration_version=migration_version,
        required_migration_version=required_migration_version,
        missing_relations=missing_relations,
        invalid_relations=invalid_relations,
        missing_columns=missing_columns,
        invalid_columns=invalid_columns,
        unexpected_columns=unexpected_columns,
        missing_constraints=tuple(sorted(missing_constraints_list)),
        invalid_constraints=tuple(sorted(invalid_constraints_list)),
        missing_indexes=missing_indexes,
        invalid_indexes=tuple(sorted(invalid_indexes_list)),
    )


async def inspect_operator_job_schema(
    pool: asyncpg.Pool,
    *,
    require_global_migration: bool = True,
) -> OperatorJobSchemaReadiness:
    """Inspect schema compatibility using SELECT statements only.

    Global migration awareness is enabled for fail-closed runtime startup and
    can be disabled for isolated structural diagnostics.
    """

    try:
        async with pool.acquire() as connection:
            migration_version: int | None = None
            required_version: int | None = None
            if require_global_migration:
                state = await inspect_migration_state_connection(connection)
                migration_version = state.current_version
                required_version = OPERATOR_JOB_MIGRATION_VERSION
            return await _inspect_operator_job_schema_connection(
                connection,
                migration_version=migration_version,
                required_migration_version=required_version,
            )
    except (MigrationError, asyncpg.PostgresError, OSError, TypeError, ValueError):
        raise OperatorJobSchemaError(
            "Operator job schema readiness could not be verified. Ensure the database is reachable and the "
            "operator role has catalog and operator-job table access."
        ) from None


async def assert_operator_job_schema_ready(pool: asyncpg.Pool) -> OperatorJobSchemaReadiness:
    """Fail closed unless the exact supported operator-job schema is ready."""

    readiness = await inspect_operator_job_schema(pool)
    if not readiness.ready:
        raise OperatorJobSchemaNotReadyError(
            "Operator job schema is not ready ("
            + readiness.summary()
            + "). Run the explicit database migration with schema-owner credentials."
        )
    return readiness


def _advisory_key(namespace: str, resource: str) -> int:
    material = f"bddk_mcp:{namespace}:v1:{resource}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=True)


def _metrics_json(job: OperatorJob) -> str:
    return json.dumps(dict(job.result_metrics), sort_keys=True, separators=(",", ":"))


def _job_parameters(job: OperatorJob) -> tuple[Any, ...]:
    return (
        job.job_id,
        job.kind.value,
        job.state.value,
        job.args_fingerprint,
        job.idempotency_digest,
        job.created_at,
        job.updated_at,
        job.revision,
        job.started_at,
        job.finished_at,
        job.progress.total,
        job.progress.completed,
        job.progress.succeeded,
        job.progress.failed,
        _metrics_json(job),
        job.error_code,
    )


def _decode_metrics(value: Any) -> Mapping[str, int | float | bool]:
    decoded = json.loads(value) if isinstance(value, str | bytes | bytearray) else value
    if not isinstance(decoded, Mapping):
        raise ValueError("stored job result_metrics must be a JSON object")
    return decoded


def _job_from_row(row: Any) -> OperatorJob:
    try:
        metrics = normalize_metrics(_decode_metrics(_row_value(row, "result_metrics", {})))
        return OperatorJob(
            job_id=UUID(str(_row_value(row, "job_id"))),
            kind=JobKind(str(_row_value(row, "kind"))),
            state=JobState(str(_row_value(row, "state"))),
            args_fingerprint=str(_row_value(row, "args_fingerprint")),
            idempotency_digest=(
                str(_row_value(row, "idempotency_digest"))
                if _row_value(row, "idempotency_digest") is not None
                else None
            ),
            created_at=_row_value(row, "created_at"),
            updated_at=_row_value(row, "updated_at"),
            revision=int(_row_value(row, "revision")),
            started_at=_row_value(row, "started_at"),
            finished_at=_row_value(row, "finished_at"),
            progress=JobProgress(
                total=int(_row_value(row, "progress_total")),
                completed=int(_row_value(row, "progress_completed")),
                succeeded=int(_row_value(row, "progress_succeeded")),
                failed=int(_row_value(row, "progress_failed")),
            ),
            result_metrics=metrics,
            error_code=(str(_row_value(row, "error_code")) if _row_value(row, "error_code") is not None else None),
        )
    except (TypeError, ValueError):
        raise OperatorJobStorageError("Stored operator job record is invalid.") from None


def _immutable_identity(job: OperatorJob) -> tuple[Any, ...]:
    return (
        job.job_id,
        job.kind,
        job.args_fingerprint,
        job.idempotency_digest,
        job.created_at,
    )


async def _finish_cleanup(cleanup: Awaitable[bool]) -> tuple[bool, asyncio.CancelledError | None]:
    """Finish cleanup even when the awaiting caller is cancelled."""

    task = asyncio.ensure_future(cleanup)
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = exc
    return task.result(), cancellation


async def _return_connection(
    pool: asyncpg.Pool,
    connection: asyncpg.pool.PoolConnectionProxy,
    *,
    advisory_key: int | None = None,
) -> bool:
    """Best-effort unlock and pool return; report failure without leaking details."""

    failed = False
    if advisory_key is not None:
        try:
            await connection.fetchval("SELECT pg_advisory_unlock($1::bigint)", advisory_key)
        except Exception:
            failed = True
    try:
        await pool.release(connection)
    except Exception:
        failed = True
    return failed


class _PostgresExecutionLease:
    def __init__(
        self,
        pool: asyncpg.Pool,
        connection: asyncpg.pool.PoolConnectionProxy,
        *,
        key: int,
        job_id: UUID,
    ) -> None:
        self._pool = pool
        self._connection = connection
        self._key = key
        self.job_id = job_id
        self._release_task: asyncio.Task[bool] | None = None

    async def release(self) -> None:
        if self._release_task is None:
            self._release_task = asyncio.create_task(
                _return_connection(self._pool, self._connection, advisory_key=self._key)
            )
        failed, cancellation = await _finish_cleanup(self._release_task)
        if cancellation is not None:
            raise cancellation
        if failed:
            raise OperatorJobStorageError("Operator job execution lease could not be released safely.")


class PostgresJobRepository:
    """Atomic PostgreSQL implementation of the operator job repository contract.

    The pool must have at least two connections: an executing job pins one
    session for its advisory lease while state/progress operations use another.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        get_max_size = getattr(pool, "get_max_size", None)
        if callable(get_max_size) and get_max_size() < MIN_OPERATOR_JOB_POOL_SIZE:
            raise ValueError(f"operator job pool must have at least {MIN_OPERATOR_JOB_POOL_SIZE} connections")
        self._pool = pool

    async def create_or_reuse(self, job: OperatorJob) -> tuple[OperatorJob, bool]:
        try:
            async with self._pool.acquire() as connection, connection.transaction():
                if job.idempotency_digest is None:
                    row = await connection.fetchrow(_INSERT_SQL, *_job_parameters(job))
                    created = True
                else:
                    lock_key = _advisory_key("idempotency", job.idempotency_digest)
                    await connection.fetchval("SELECT pg_advisory_xact_lock($1::bigint)", lock_key)
                    existing_row = await connection.fetchrow(_GET_BY_IDEMPOTENCY_SQL, job.idempotency_digest)
                    if existing_row is not None:
                        return _job_from_row(existing_row), False
                    row = await connection.fetchrow(_INSERT_SQL, *_job_parameters(job))
                    created = True
            if row is None:
                raise OperatorJobStorageError("Operator job creation returned no record.")
            return _job_from_row(row), created
        except asyncpg.UniqueViolationError:
            raise ValueError("job ID already exists") from None
        except OperatorJobStorageError:
            raise
        except (asyncpg.PostgresError, OSError):
            raise OperatorJobStorageError("Operator job could not be created or reused.") from None

    async def get(self, job_id: UUID) -> OperatorJob | None:
        try:
            async with self._pool.acquire() as connection:
                row = await connection.fetchrow(_GET_SQL, job_id)
            return _job_from_row(row) if row is not None else None
        except (asyncpg.PostgresError, OSError):
            raise OperatorJobStorageError("Operator job could not be read.") from None

    async def list(self, *, limit: int, states: Collection[JobState] | None = None) -> list[OperatorJob]:
        if limit < 1:
            raise ValueError("limit must be positive")
        try:
            async with self._pool.acquire() as connection:
                if states is None:
                    rows = await connection.fetch(_LIST_SQL, limit)
                else:
                    state_values = [JobState(state).value for state in states]
                    rows = await connection.fetch(_LIST_STATES_SQL, state_values, limit)
            return [_job_from_row(row) for row in rows]
        except (asyncpg.PostgresError, OSError):
            raise OperatorJobStorageError("Operator jobs could not be listed.") from None

    async def list_unfinished(self) -> list[OperatorJob]:
        try:
            async with self._pool.acquire() as connection:
                rows = await connection.fetch(_LIST_UNFINISHED_SQL, list(_UNFINISHED_STATES))
            return [_job_from_row(row) for row in rows]
        except (asyncpg.PostgresError, OSError):
            raise OperatorJobStorageError("Unfinished operator jobs could not be listed.") from None

    async def compare_and_set(self, replacement: OperatorJob, *, expected_revision: int) -> bool:
        if replacement.revision != expected_revision + 1:
            raise ValueError("replacement revision must advance by one")
        try:
            async with self._pool.acquire() as connection, connection.transaction():
                current_row = await connection.fetchrow(_GET_FOR_UPDATE_SQL, replacement.job_id)
                if current_row is None:
                    return False
                current = _job_from_row(current_row)
                if current.revision != expected_revision:
                    return False
                if _immutable_identity(current) != _immutable_identity(replacement):
                    raise ValueError("job identity fields are immutable")
                updated_row = await connection.fetchrow(
                    _UPDATE_SQL,
                    replacement.job_id,
                    replacement.state.value,
                    replacement.updated_at,
                    replacement.revision,
                    replacement.started_at,
                    replacement.finished_at,
                    replacement.progress.total,
                    replacement.progress.completed,
                    replacement.progress.succeeded,
                    replacement.progress.failed,
                    _metrics_json(replacement),
                    replacement.error_code,
                    expected_revision,
                )
                if updated_row is None:
                    return False
                _job_from_row(updated_row)
                return True
        except (asyncpg.PostgresError, OSError):
            raise OperatorJobStorageError("Operator job state could not be updated.") from None

    async def try_acquire_execution_lease(
        self,
        *,
        job_id: UUID,
        resource: str,
    ) -> JobExecutionLease | None:
        if not isinstance(resource, str) or not resource or len(resource) > 256:
            raise ValueError("lease resource must contain 1 to 256 characters")
        connection: asyncpg.pool.PoolConnectionProxy | None = None
        try:
            connection = await self._pool.acquire()
            key = (
                CORPUS_JOB_EXECUTION_ADVISORY_KEY
                if resource == "corpus_mutation"
                else _advisory_key("execution", resource)
            )
            acquired = await connection.fetchval("SELECT pg_try_advisory_lock($1::bigint)", key)
            if not acquired:
                returning = connection
                connection = None
                failed, cancellation = await _finish_cleanup(_return_connection(self._pool, returning))
                if cancellation is not None:
                    raise cancellation
                if failed:
                    raise OperatorJobStorageError("Operator job execution lease connection could not be returned.")
                return None
            lease_connection = connection
            connection = None
            return _PostgresExecutionLease(self._pool, lease_connection, key=key, job_id=job_id)
        except BaseException as exc:
            if connection is not None:
                await _finish_cleanup(_return_connection(self._pool, connection, advisory_key=key))
            if isinstance(exc, asyncio.CancelledError):
                raise
            if isinstance(exc, OperatorJobStorageError):
                raise
            if isinstance(exc, (asyncpg.PostgresError, OSError)):
                raise OperatorJobStorageError("Operator job execution lease could not be acquired.") from None
            raise

    async def prune_terminal(self, *, keep: int) -> int:
        if keep < 0:
            raise ValueError("keep cannot be negative")
        try:
            async with self._pool.acquire() as connection:
                rows = await connection.fetch(_PRUNE_SQL, list(_TERMINAL_STATES), keep)
            return len(rows)
        except (asyncpg.PostgresError, OSError):
            raise OperatorJobStorageError("Operator job history could not be pruned.") from None


__all__ = (
    "MIN_OPERATOR_JOB_POOL_SIZE",
    "OPERATOR_JOB_MIGRATION_VERSION",
    "OPERATOR_JOB_SCHEMA",
    "OperatorJobSchemaError",
    "OperatorJobSchemaNotReadyError",
    "OperatorJobSchemaReadiness",
    "OperatorJobStorageError",
    "PostgresJobRepository",
    "assert_operator_job_schema_ready",
    "inspect_operator_job_schema",
)
