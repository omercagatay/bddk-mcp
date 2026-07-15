"""Contract tests for the durable PostgreSQL operator-job repository."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import asyncpg
import pytest

from bddk_mcp.jobs import (
    MIN_OPERATOR_JOB_POOL_SIZE,
    OPERATOR_JOB_MIGRATION_VERSION,
    OPERATOR_JOB_SCHEMA,
    JobKind,
    JobProgress,
    JobState,
    OperatorJob,
    OperatorJobSchemaReadiness,
    OperatorJobStorageError,
    PostgresJobRepository,
    fingerprint_arguments,
    inspect_operator_job_schema,
)
from bddk_mcp.jobs.models import digest_idempotency_key
from bddk_mcp.jobs.postgres import (
    _CONSTRAINT_REQUIREMENTS,
    _INDEX_REQUIREMENTS,
    _REQUIRED_COLUMN_SPECS,
    _REQUIRED_COLUMNS,
)
from bddk_mcp.migrations import LATEST_SCHEMA_VERSION, MIGRATIONS, migrate
from bddk_mcp.migrations.v0002_operator_jobs import V0002_OPERATOR_JOBS


class _Transaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class _Acquire:
    """Small asyncpg PoolAcquireContext-compatible test double."""

    def __init__(self, pool, connection) -> None:
        self._pool = pool
        self._connection = connection

    async def _get(self):
        return self._connection

    def __await__(self):
        return self._get().__await__()

    async def __aenter__(self):
        return self._connection

    async def __aexit__(self, exc_type, exc, traceback):
        await self._pool.release(self._connection)
        return False


class _Pool:
    def __init__(self, connection) -> None:
        self.connection = connection
        self.released: list[object] = []

    def acquire(self):
        return _Acquire(self, self.connection)

    async def release(self, connection):
        self.released.append(connection)


def _connection(**overrides):
    values = {
        "fetch": AsyncMock(return_value=[]),
        "fetchrow": AsyncMock(return_value=None),
        "fetchval": AsyncMock(return_value=None),
        "execute": AsyncMock(return_value="OK"),
        "transaction": lambda: _Transaction(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _job(*, idempotency_key: str | None = None, marker: int = 1) -> OperatorJob:
    return OperatorJob.create(
        kind=JobKind.DOCUMENT_SYNC,
        args_fingerprint=fingerprint_arguments(JobKind.DOCUMENT_SYNC, {"marker": marker}),
        idempotency_digest=digest_idempotency_key(idempotency_key),
    )


def _row(job: OperatorJob, **extra) -> dict:
    row = {
        "job_id": job.job_id,
        "kind": job.kind.value,
        "state": job.state.value,
        "args_fingerprint": job.args_fingerprint,
        "idempotency_digest": job.idempotency_digest,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "revision": job.revision,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "progress_total": job.progress.total,
        "progress_completed": job.progress.completed,
        "progress_succeeded": job.progress.succeeded,
        "progress_failed": job.progress.failed,
        "result_metrics": json.dumps(dict(job.result_metrics)),
        "error_code": job.error_code,
    }
    row.update(extra)
    return row


def test_operator_job_migration_has_no_raw_argument_or_exception_text_columns():
    columns = _REQUIRED_COLUMNS["operator_jobs"]
    forbidden = {
        "arguments",
        "args",
        "request",
        "payload",
        "exception",
        "exception_text",
        "error_message",
        "traceback",
    }

    assert forbidden.isdisjoint(columns)
    assert {"args_fingerprint", "idempotency_digest", "error_code"} <= columns
    assert OPERATOR_JOB_MIGRATION_VERSION == 2
    ddl = "\n".join(V0002_OPERATOR_JOBS.statements).lower()
    assert "args_fingerprint" in ddl
    assert "idempotency_digest" in ddl
    assert "exception_text" not in ddl
    assert "traceback" not in ddl


@pytest.mark.asyncio
async def test_idempotent_upsert_returns_existing_record_atomically():
    candidate = _job(idempotency_key="caller-secret", marker=1)
    existing = replace(_job(idempotency_key="caller-secret", marker=2), args_fingerprint=candidate.args_fingerprint)
    connection = _connection(
        fetchrow=AsyncMock(return_value=_row(existing)),
        fetchval=AsyncMock(return_value=None),
    )
    pool = _Pool(connection)

    stored, created = await PostgresJobRepository(pool).create_or_reuse(candidate)  # type: ignore[arg-type]

    assert stored == existing
    assert created is False
    query = connection.fetchrow.await_args.args[0]
    parameters = connection.fetchrow.await_args.args[1:]
    assert "WHERE idempotency_digest = $1" in query
    assert "FOR UPDATE" in query
    assert "pg_advisory_xact_lock" in connection.fetchval.await_args.args[0]
    assert "caller-secret" not in repr(parameters)
    assert candidate.idempotency_digest in parameters


@pytest.mark.asyncio
async def test_compare_and_set_locks_row_and_preserves_identity_fields():
    current = _job(idempotency_key="immutable-key")
    now = datetime.now(UTC)
    replacement = replace(
        current,
        state=JobState.RUNNING,
        revision=1,
        updated_at=now,
        started_at=now,
        progress=JobProgress(total=3, completed=1, succeeded=1),
    )
    connection = _connection(fetchrow=AsyncMock(side_effect=[_row(current), _row(replacement)]))
    repository = PostgresJobRepository(_Pool(connection))  # type: ignore[arg-type]

    assert await repository.compare_and_set(replacement, expected_revision=0)
    assert "FOR UPDATE" in connection.fetchrow.await_args_list[0].args[0]
    update_arguments = connection.fetchrow.await_args_list[1].args
    assert "UPDATE bddk_operator.operator_jobs" in update_arguments[0]
    assert current.args_fingerprint not in update_arguments[1:]
    assert current.idempotency_digest not in update_arguments[1:]


@pytest.mark.asyncio
async def test_compare_and_set_rejects_identity_changes_without_an_update():
    current = _job()
    replacement = replace(
        current,
        kind=JobKind.BACKFILL,
        revision=1,
        updated_at=datetime.now(UTC),
    )
    connection = _connection(fetchrow=AsyncMock(return_value=_row(current)))
    repository = PostgresJobRepository(_Pool(connection))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="identity fields are immutable"):
        await repository.compare_and_set(replacement, expected_revision=0)

    assert connection.fetchrow.await_count == 1


@pytest.mark.asyncio
async def test_execution_lease_keeps_connection_pinned_until_idempotent_release():
    connection = _connection(fetchval=AsyncMock(side_effect=[True, True]))
    pool = _Pool(connection)
    repository = PostgresJobRepository(pool)  # type: ignore[arg-type]

    lease = await repository.try_acquire_execution_lease(job_id=uuid4(), resource="corpus_mutation")

    assert lease is not None
    assert pool.released == []
    assert "pg_try_advisory_lock" in connection.fetchval.await_args_list[0].args[0]
    await lease.release()
    await lease.release()
    assert pool.released == [connection]
    assert "pg_advisory_unlock" in connection.fetchval.await_args_list[1].args[0]


@pytest.mark.asyncio
async def test_execution_lease_release_finishes_cleanup_before_propagating_cancellation():
    unlock_started = asyncio.Event()
    finish_unlock = asyncio.Event()

    async def fetchval(query, *_args):
        if "pg_try_advisory_lock" in query:
            return True
        unlock_started.set()
        await finish_unlock.wait()
        return True

    connection = _connection(fetchval=AsyncMock(side_effect=fetchval))
    pool = _Pool(connection)
    lease = await PostgresJobRepository(pool).try_acquire_execution_lease(  # type: ignore[arg-type]
        job_id=uuid4(),
        resource="corpus_mutation",
    )
    assert lease is not None

    releasing = asyncio.create_task(lease.release())
    await unlock_started.wait()
    releasing.cancel()
    await asyncio.sleep(0)
    assert not releasing.done()
    assert pool.released == []

    finish_unlock.set()
    with pytest.raises(asyncio.CancelledError):
        await releasing
    assert pool.released == [connection]


@pytest.mark.asyncio
async def test_cancelled_lease_acquisition_returns_connection_and_unlocks_unknown_outcome():
    acquisition_started = asyncio.Event()

    async def fetchval(query, *_args):
        if "pg_try_advisory_lock" in query:
            acquisition_started.set()
            await asyncio.Event().wait()
        assert "pg_advisory_unlock" in query
        return False

    connection = _connection(fetchval=AsyncMock(side_effect=fetchval))
    pool = _Pool(connection)
    acquiring = asyncio.create_task(
        PostgresJobRepository(pool).try_acquire_execution_lease(  # type: ignore[arg-type]
            job_id=uuid4(),
            resource="corpus_mutation",
        )
    )
    await acquisition_started.wait()
    acquiring.cancel()

    with pytest.raises(asyncio.CancelledError):
        await acquiring
    assert pool.released == [connection]
    assert "pg_advisory_unlock" in connection.fetchval.await_args_list[1].args[0]


@pytest.mark.asyncio
async def test_failed_execution_lease_is_returned_to_pool_immediately():
    connection = _connection(fetchval=AsyncMock(return_value=False))
    pool = _Pool(connection)

    lease = await PostgresJobRepository(pool).try_acquire_execution_lease(  # type: ignore[arg-type]
        job_id=uuid4(),
        resource="corpus_mutation",
    )

    assert lease is None
    assert pool.released == [connection]


def test_repository_rejects_pool_that_cannot_service_state_updates_while_leased():
    pool = _Pool(_connection())
    pool.get_max_size = lambda: MIN_OPERATOR_JOB_POOL_SIZE - 1  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="at least 2 connections"):
        PostgresJobRepository(pool)  # type: ignore[arg-type]


def _catalog_rows(query: str, *args, missing_column: str | None = None):
    if "bddk_meta.schema_migrations" in query:
        return [{"version": item.version, "name": item.name, "checksum": item.checksum} for item in MIGRATIONS]
    schema, names = args
    assert schema == OPERATOR_JOB_SCHEMA
    normalized = query.strip()
    assert normalized.upper().startswith("SELECT")
    if "relation.relkind" in query:
        return [{"relation_name": name, "relkind": "r"} for name in names]
    if "pg_attribute" in query:
        return [
            {
                "table_name": table,
                "column_name": column,
                "data_type": data_type,
                "not_null": not_null,
            }
            for table in names
            for column, (data_type, not_null) in _REQUIRED_COLUMN_SPECS[table].items()
            if column != missing_column
        ]
    if "pg_constraint" in query:
        return [
            {
                "table_name": table,
                "constraint_name": name,
                "constraint_type": constraint_type,
                "is_valid": True,
                "definition": " ".join(fragments),
            }
            for table in names
            for name, (constraint_type, fragments) in _CONSTRAINT_REQUIREMENTS[table].items()
        ]
    if "pg_index" in query:
        return [
            {
                "index_name": name,
                "table_name": "operator_jobs",
                "is_unique": unique,
                "is_valid": True,
                "is_ready": True,
                "definition": " ".join(definition_fragments),
                "predicate": " ".join(predicate_fragments) if predicate_fragments else None,
            }
            for name, (unique, definition_fragments, predicate_fragments) in _INDEX_REQUIREMENTS.items()
            if name in names
        ]
    raise AssertionError("unexpected query")


@pytest.mark.asyncio
async def test_readiness_is_select_only_and_accepts_compatible_schema():
    connection = _connection(
        fetch=AsyncMock(side_effect=_catalog_rows),
        fetchval=AsyncMock(return_value="bddk_meta.schema_migrations"),
    )
    report = await inspect_operator_job_schema(_Pool(connection))  # type: ignore[arg-type]

    assert report == OperatorJobSchemaReadiness(
        migration_version=LATEST_SCHEMA_VERSION,
        required_migration_version=OPERATOR_JOB_MIGRATION_VERSION,
    )
    assert report.ready
    assert connection.execute.await_count == 0
    assert all(call.args[0].strip().upper().startswith("SELECT") for call in connection.fetch.await_args_list)


@pytest.mark.asyncio
async def test_structural_readiness_can_be_inspected_without_global_ledger_access():
    connection = _connection(fetch=AsyncMock(side_effect=_catalog_rows))

    report = await inspect_operator_job_schema(  # type: ignore[arg-type]
        _Pool(connection),
        require_global_migration=False,
    )

    assert report == OperatorJobSchemaReadiness()
    assert report.ready
    assert connection.fetchval.await_count == 0


@pytest.mark.asyncio
async def test_repository_errors_are_sanitized():
    sentinel = "postgresql://private:password@internal-bank/operator"
    connection = _connection(fetchrow=AsyncMock(side_effect=asyncpg.PostgresError(sentinel)))

    with pytest.raises(OperatorJobStorageError) as exc_info:
        await PostgresJobRepository(_Pool(connection)).get(uuid4())  # type: ignore[arg-type]

    assert sentinel not in str(exc_info.value)
    assert "password" not in str(exc_info.value)


@pytest.fixture
async def operator_job_pool(pg_pool):
    await migrate(pg_pool)
    await pg_pool.execute("TRUNCATE TABLE bddk_operator.operator_jobs")
    try:
        yield pg_pool
    finally:
        await pg_pool.execute("TRUNCATE TABLE bddk_operator.operator_jobs")


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_postgres_schema_migration_is_versioned_idempotent_and_private(operator_job_pool):
    first = await migrate(operator_job_pool)  # type: ignore[arg-type]
    second = await migrate(operator_job_pool)  # type: ignore[arg-type]

    assert first.current and second.current
    assert first.current_version == LATEST_SCHEMA_VERSION
    async with operator_job_pool.acquire() as connection:
        columns = {
            row["column_name"]
            for row in await connection.fetch(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = $1 AND table_name = 'operator_jobs'
                """,
                OPERATOR_JOB_SCHEMA,
            )
        }
        migration_record = await connection.fetchrow(
            """
            SELECT version, name, checksum
            FROM bddk_meta.schema_migrations
            WHERE version = $1
            """,
            OPERATOR_JOB_MIGRATION_VERSION,
        )
    assert columns == set(_REQUIRED_COLUMNS["operator_jobs"])
    assert {"arguments", "payload", "exception_text", "traceback"}.isdisjoint(columns)
    assert dict(migration_record) == {
        "version": V0002_OPERATOR_JOBS.version,
        "name": V0002_OPERATOR_JOBS.name,
        "checksum": V0002_OPERATOR_JOBS.checksum,
    }


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_postgres_repository_and_readiness_ignore_hostile_search_path(operator_job_pool):
    candidate = _job(marker=99)

    async with operator_job_pool.acquire() as connection:
        await connection.execute("SET search_path TO pg_catalog")
        pinned_pool = _Pool(connection)
        try:
            report = await inspect_operator_job_schema(pinned_pool)  # type: ignore[arg-type]
            repository = PostgresJobRepository(pinned_pool)  # type: ignore[arg-type]
            created, is_new = await repository.create_or_reuse(candidate)
            stored = await repository.get(candidate.job_id)
        finally:
            await connection.execute("RESET search_path")

    assert report.ready
    assert is_new
    assert created == candidate
    assert stored == candidate


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_postgres_repository_roundtrip_cas_list_and_prune(operator_job_pool):
    repository = PostgresJobRepository(operator_job_pool)  # type: ignore[arg-type]
    candidate = _job(idempotency_key="persisted-as-digest", marker=1)

    created, is_new = await repository.create_or_reuse(candidate)
    reused, is_reused_new = await repository.create_or_reuse(
        replace(_job(idempotency_key="persisted-as-digest", marker=2), args_fingerprint=candidate.args_fingerprint)
    )
    assert is_new is True
    assert is_reused_new is False
    assert reused.job_id == created.job_id

    now = datetime.now(UTC)
    running = replace(created, state=JobState.RUNNING, started_at=now, updated_at=now, revision=1)
    assert await repository.compare_and_set(running, expected_revision=0)
    finished_at = datetime.now(UTC)
    succeeded = replace(
        running,
        state=JobState.SUCCEEDED,
        updated_at=finished_at,
        finished_at=finished_at,
        revision=2,
        progress=JobProgress(total=2, completed=2, succeeded=2),
        result_metrics=(("documents", 2), ("finished", True)),
    )
    assert await repository.compare_and_set(succeeded, expected_revision=1)
    assert not await repository.compare_and_set(succeeded, expected_revision=1)

    stored = await repository.get(candidate.job_id)
    assert stored == succeeded
    assert await repository.list(limit=10, states={JobState.SUCCEEDED}) == [succeeded]
    assert await repository.list_unfinished() == []
    assert await repository.prune_terminal(keep=0) == 1
    assert await repository.get(candidate.job_id) is None


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_postgres_idempotent_create_is_atomic_under_concurrency(operator_job_pool):
    repository = PostgresJobRepository(operator_job_pool)  # type: ignore[arg-type]
    candidates = [_job(idempotency_key="one-concurrent-request", marker=1) for _ in range(20)]

    results = await asyncio.gather(*(repository.create_or_reuse(candidate) for candidate in candidates))

    assert sum(created for _job_record, created in results) == 1
    assert len({job.job_id for job, _created in results}) == 1
    assert len(await repository.list(limit=100)) == 1


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_postgres_advisory_lease_is_exclusive_until_connection_release(operator_job_pool):
    first_repository = PostgresJobRepository(operator_job_pool)  # type: ignore[arg-type]
    second_repository = PostgresJobRepository(operator_job_pool)  # type: ignore[arg-type]

    first = await first_repository.try_acquire_execution_lease(
        job_id=uuid4(),
        resource="corpus_mutation",
    )
    blocked = await second_repository.try_acquire_execution_lease(
        job_id=uuid4(),
        resource="corpus_mutation",
    )
    assert first is not None
    assert blocked is None

    await first.release()
    await first.release()
    second = await second_repository.try_acquire_execution_lease(
        job_id=uuid4(),
        resource="corpus_mutation",
    )
    assert second is not None
    await second.release()


def test_job_rows_remain_uuid_based():
    job = _job()
    assert isinstance(job.job_id, UUID)
