"""Tests for the immutable global PostgreSQL migration framework."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
from unittest.mock import patch

import asyncpg
import pytest

from bddk_mcp.migrations import (
    LATEST_SCHEMA_VERSION,
    MIGRATION_LOCK_TIMEOUT,
    MIGRATION_STATEMENT_TIMEOUT,
    MIGRATIONS,
    MigrationCompatibilityError,
    MigrationError,
    MigrationHistoryError,
    MigrationLockTimeoutError,
    MigrationPrerequisiteError,
    MigrationScaleError,
    MigrationState,
    MigrationStatementTimeoutError,
    inspect_migration_state,
    migrate,
    validate_migration_history,
)


def _history_rows(*, migrations=MIGRATIONS):
    return [{"version": item.version, "name": item.name, "checksum": item.checksum} for item in migrations]


def test_registry_is_sequential_named_and_sha256_versioned():
    assert [item.version for item in MIGRATIONS] == list(range(1, LATEST_SCHEMA_VERSION + 1))
    assert len({item.name for item in MIGRATIONS}) == len(MIGRATIONS)
    assert all(len(item.checksum) == 64 for item in MIGRATIONS)
    assert all(item.checksum == replace(item).checksum for item in MIGRATIONS)
    assert replace(MIGRATIONS[0], name="changed_name").checksum != MIGRATIONS[0].checksum


def test_migrations_never_install_dba_managed_extensions_and_qualify_created_relations():
    ddl = "\n".join(statement for item in MIGRATIONS for statement in item.statements).lower()

    assert "create extension" not in ddl
    assert "create table documents" not in ddl
    assert "create table document_chunks" not in ddl
    assert "create table operator_jobs" not in ddl
    assert "create table public.documents" in ddl
    assert "create table public.document_chunks" in ddl
    assert "create table bddk_operator.operator_jobs" in ddl


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                _history_rows()[0],
                {**_history_rows()[1], "checksum": "0" * 64},
            ],
            "checksum",
        ),
        ([{**_history_rows()[0], "name": "renamed"}], "name"),
        ([_history_rows()[1]], "gap"),
        (
            _history_rows() + [{"version": LATEST_SCHEMA_VERSION + 1, "name": "future", "checksum": "f" * 64}],
            "newer",
        ),
    ],
)
def test_history_validation_fails_closed_for_tampering_gaps_and_newer_databases(rows, message):
    with pytest.raises(MigrationHistoryError, match=message):
        validate_migration_history(rows)


def test_history_validation_reports_pending_versions_only_after_valid_prefix():
    empty = validate_migration_history([])
    first = validate_migration_history(_history_rows()[:1])
    current = validate_migration_history(_history_rows())

    assert empty == MigrationState(current_version=0)
    assert empty.pending_versions == tuple(range(1, LATEST_SCHEMA_VERSION + 1))
    assert first.pending_versions == tuple(range(2, LATEST_SCHEMA_VERSION + 1))
    assert current.current
    assert current.pending_versions == ()


class _Transaction:
    def __init__(self) -> None:
        self.entered = False
        self.rolled_back = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.rolled_back = exc_type is not None
        return False


class _FakeMigrationConnection:
    def __init__(
        self,
        *,
        extensions: dict[str, str] | None = None,
        fail_statement: str | None = None,
        fail_exception: Exception | None = None,
        retrieval_tables_populated: bool = False,
        server_version_num: int = 170000,
    ) -> None:
        self.extensions = extensions if extensions is not None else {"unaccent": "public", "vector": "public"}
        self.fail_statement = fail_statement
        self.fail_exception = fail_exception
        self.retrieval_tables_populated = retrieval_tables_populated
        self.server_version_num = server_version_num
        self.history_exists = False
        self.history: list[dict[str, object]] = []
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self.transaction_record = _Transaction()

    def transaction(self):
        return self.transaction_record

    async def fetchval(self, query: str, *args):
        if "server_version_num" in query:
            return self.server_version_num
        if "pg_advisory_xact_lock" in query:
            return None
        if "to_regclass" in query:
            return "bddk_meta.schema_migrations" if self.history_exists else None
        raise AssertionError(f"unexpected fetchval: {query}")

    async def fetch(self, query: str, *args):
        if "pg_extension" in query:
            return [{"extname": name, "extension_schema": schema} for name, schema in sorted(self.extensions.items())]
        if "bddk_meta.schema_migrations" in query:
            return list(self.history)
        raise AssertionError(f"unexpected fetch: {query}")

    async def fetchrow(self, query: str, *args):
        if "has_documents" in query:
            return {
                "has_documents": self.retrieval_tables_populated,
                "has_sections": self.retrieval_tables_populated,
                "has_chunks": self.retrieval_tables_populated,
            }
        assert "to_regprocedure" in query
        return {"has_unaccent": True, "has_vector": True}

    async def execute(self, query: str, *args):
        normalized = " ".join(query.split())
        self.executed.append((normalized, args))
        if self.fail_statement and self.fail_statement in normalized:
            raise self.fail_exception or asyncpg.PostgresError("private database details")
        if normalized.startswith("CREATE TABLE IF NOT EXISTS bddk_meta.schema_migrations"):
            self.history_exists = True
        if normalized.startswith("INSERT INTO bddk_meta.schema_migrations"):
            self.history.append({"version": args[0], "name": args[1], "checksum": args[2]})
        return "OK"


class _FakePool:
    def __init__(self, connection: _FakeMigrationConnection) -> None:
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self.connection

    async def fetchval(self, query: str, *args):
        return await self.connection.fetchval(query, *args)

    async def fetch(self, query: str, *args):
        return await self.connection.fetch(query, *args)


class _PinnedPool:
    """Expose one real connection as the pool interface used by migrations."""

    def __init__(self, connection) -> None:
        self.connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self.connection


async def _downgrade_current_schema_to_v2(connection) -> None:
    """Remove unreleased v4/v3 artifacts inside a rollback-only test transaction."""

    await connection.execute(
        """
        DROP TABLE IF EXISTS
            public.regulatory_legal_version_provisions,
            public.regulatory_legal_status_assertions,
            public.regulatory_legal_events,
            public.regulatory_legal_version_artifacts,
            public.regulatory_provisions,
            public.regulatory_legal_versions,
            public.regulatory_evidence,
            public.regulatory_source_artifacts,
            public.regulatory_source_blobs,
            public.regulatory_family_imports,
            public.regulatory_instruments
        CASCADE
        """
    )
    await connection.execute("DELETE FROM bddk_meta.schema_migrations WHERE version = 4")

    await connection.execute(
        "DROP TRIGGER IF EXISTS invalidate_retrieval_publication_on_chunk_change ON public.document_chunks"
    )
    await connection.execute("DROP FUNCTION IF EXISTS public.invalidate_retrieval_publication()")
    await connection.execute("DROP TABLE IF EXISTS public.document_retrieval_publications")
    await connection.execute("ALTER TABLE public.document_chunks DROP CONSTRAINT IF EXISTS document_chunks_document_fk")
    await connection.execute(
        "ALTER TABLE public.document_sections DROP CONSTRAINT IF EXISTS document_sections_document_fk"
    )
    await connection.execute("ALTER TABLE public.document_sections DROP COLUMN IF EXISTS source_content_hash")
    await connection.execute("DROP TABLE IF EXISTS bddk_meta.legacy_schema_adoptions")
    await connection.execute("DELETE FROM bddk_meta.schema_migrations WHERE version = 3")


@pytest.mark.asyncio
async def test_migrate_serializes_sets_fixed_timeouts_and_records_every_checksum():
    connection = _FakeMigrationConnection()
    state = await migrate(_FakePool(connection))  # type: ignore[arg-type]

    statements = [query for query, _args in connection.executed]
    assert state.current
    assert statements[0] == f"SET LOCAL lock_timeout = '{MIGRATION_LOCK_TIMEOUT}'"
    assert statements[1] == f"SET LOCAL statement_timeout = '{MIGRATION_STATEMENT_TIMEOUT}'"
    assert connection.history == _history_rows()
    assert connection.transaction_record.rolled_back is False


@pytest.mark.asyncio
async def test_migrate_refuses_unsupported_postgresql_before_transaction_or_mutation():
    connection = _FakeMigrationConnection(server_version_num=160012)

    with pytest.raises(MigrationCompatibilityError) as exc_info:
        await migrate(_FakePool(connection))  # type: ignore[arg-type]

    assert "requires PostgreSQL 17" in str(exc_info.value)
    assert "160012" not in str(exc_info.value)
    assert not connection.transaction_record.entered
    assert connection.executed == []
    assert connection.history == []
    assert not connection.history_exists


@pytest.mark.asyncio
async def test_populated_v2_refuses_v3_before_schema_changes_without_narrow_approval():
    connection = _FakeMigrationConnection(retrieval_tables_populated=True)
    connection.history_exists = True
    connection.history = _history_rows()[:2]

    with pytest.raises(MigrationScaleError) as exc_info:
        await migrate(_FakePool(connection))  # type: ignore[arg-type]

    message = str(exc_info.value)
    statements = [query for query, _args in connection.executed]
    assert connection.transaction_record.rolled_back
    assert connection.history == _history_rows()[:2]
    assert not any("ADD COLUMN source_content_hash" in statement for statement in statements)
    assert "--allow-retrieval-publication-backfill" in message
    assert "restorable backup" in message
    assert "postgresql://" not in message


@pytest.mark.asyncio
async def test_populated_v2_requires_explicit_approval_and_suppresses_only_section_fts_during_backfill():
    connection = _FakeMigrationConnection(retrieval_tables_populated=True)
    connection.history_exists = True
    connection.history = _history_rows()[:2]

    state = await migrate(
        _FakePool(connection),  # type: ignore[arg-type]
        allow_retrieval_publication_backfill=True,
    )

    statements = [query for query, _args in connection.executed]
    disable = statements.index("ALTER TABLE public.document_sections DISABLE TRIGGER trg_document_sections_tsv")
    backfill = next(
        index for index, statement in enumerate(statements) if statement.startswith("UPDATE public.document_sections")
    )
    enable = statements.index("ALTER TABLE public.document_sections ENABLE TRIGGER trg_document_sections_tsv")
    assert state.current
    assert disable < backfill < enable
    assert all(
        "DISABLE TRIGGER" not in statement or "trg_document_sections_tsv" in statement for statement in statements
    )


@pytest.mark.parametrize(
    ("driver_error", "expected_error", "message_fragment"),
    [
        (
            asyncpg.LockNotAvailableError("postgresql://private:password@secret-host/bddk"),
            MigrationLockTimeoutError,
            "required lock within",
        ),
        (
            asyncpg.QueryCanceledError("postgresql://private:password@secret-host/bddk"),
            MigrationStatementTimeoutError,
            "canceled or exceeded",
        ),
    ],
)
@pytest.mark.asyncio
async def test_migration_lock_and_statement_timeouts_are_actionable_and_sanitized(
    driver_error,
    expected_error,
    message_fragment,
):
    connection = _FakeMigrationConnection(
        fail_statement="CREATE TABLE public.documents",
        fail_exception=driver_error,
    )

    with pytest.raises(expected_error) as exc_info:
        await migrate(_FakePool(connection))  # type: ignore[arg-type]

    message = str(exc_info.value)
    assert connection.transaction_record.rolled_back
    assert message_fragment in message
    assert "postgresql://" not in message
    assert exc_info.value.__cause__ is None


@pytest.mark.asyncio
async def test_migrate_is_idempotent_after_valid_history():
    connection = _FakeMigrationConnection()
    pool = _FakePool(connection)
    await migrate(pool)  # type: ignore[arg-type]
    first_count = len(connection.executed)

    state = await migrate(pool)  # type: ignore[arg-type]

    assert state.current
    assert len(connection.history) == LATEST_SCHEMA_VERSION
    assert len(connection.executed) == first_count + 4


@pytest.mark.asyncio
async def test_migrate_rolls_back_and_sanitizes_statement_failures():
    connection = _FakeMigrationConnection(fail_statement="CREATE TABLE public.documents")

    with pytest.raises(MigrationError) as exc_info:
        await migrate(_FakePool(connection))  # type: ignore[arg-type]

    assert connection.transaction_record.rolled_back
    assert "private database details" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_migrate_rejects_missing_or_non_public_extensions_without_installing_them():
    missing = _FakeMigrationConnection(extensions={"unaccent": "public"})
    misplaced = _FakeMigrationConnection(extensions={"unaccent": "public", "vector": "extensions"})

    with pytest.raises(MigrationPrerequisiteError, match="vector"):
        await migrate(_FakePool(missing))  # type: ignore[arg-type]
    with pytest.raises(MigrationPrerequisiteError, match="public schema"):
        await migrate(_FakePool(misplaced))  # type: ignore[arg-type]

    assert all("CREATE EXTENSION" not in query for query, _args in missing.executed + misplaced.executed)


@pytest.mark.asyncio
async def test_inspection_is_select_only_and_reports_an_absent_ledger():
    connection = _FakeMigrationConnection()
    state = await inspect_migration_state(_FakePool(connection))  # type: ignore[arg-type]

    assert state == MigrationState(current_version=0)
    assert connection.executed == []


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_global_migrations_run_idempotently_against_postgresql(pg_pool):
    first = await migrate(pg_pool)
    second = await migrate(pg_pool)

    rows = await pg_pool.fetch(
        """
        SELECT version, name, checksum
        FROM bddk_meta.schema_migrations
        ORDER BY version
        """
    )
    assert first.current and second.current
    assert [dict(row) for row in rows] == _history_rows()
    assert await pg_pool.fetchval("SELECT pg_catalog.to_regclass('public.documents')") is not None
    assert await pg_pool.fetchval("SELECT pg_catalog.to_regclass('public.document_chunks')") is not None
    assert await pg_pool.fetchval("SELECT pg_catalog.to_regclass('bddk_operator.operator_jobs')") is not None


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_populated_postgres_v2_requires_approval_and_v3_backfill_is_complete_and_trigger_safe(pg_pool):
    connection = await pg_pool.acquire()
    transaction = connection.transaction()
    await transaction.start()
    try:
        await _downgrade_current_schema_to_v2(connection)
        content_hash = "a" * 64
        await connection.execute(
            """
            INSERT INTO public.documents (document_id, title, markdown_content, content_hash)
            VALUES ('v3-upgrade-proof', 'Upgrade proof', 'Proof body', $1)
            """,
            content_hash,
        )
        await connection.execute(
            """
            INSERT INTO public.document_sections (
                doc_id, section_type, section_ref, start_char, end_char, content, content_hash
            ) VALUES ('v3-upgrade-proof', 'article', '1', 0, 10, 'Proof body', $1)
            """,
            "b" * 64,
        )
        await connection.execute(
            """
            INSERT INTO public.document_chunks (doc_id, chunk_index, content_hash, chunk_text)
            VALUES ('v3-upgrade-proof', 0, $1, 'Proof body')
            """,
            content_hash,
        )
        tsv_before = await connection.fetchval(
            "SELECT tsv::pg_catalog.text FROM public.document_sections WHERE doc_id = 'v3-upgrade-proof'"
        )

        with pytest.raises(MigrationScaleError, match="--allow-retrieval-publication-backfill"):
            await migrate(_PinnedPool(connection))  # type: ignore[arg-type]

        assert await connection.fetchval("SELECT max(version) FROM bddk_meta.schema_migrations") == 2
        assert not await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_catalog.pg_attribute
                WHERE attrelid = 'public.document_sections'::pg_catalog.regclass
                  AND attname = 'source_content_hash'
                  AND NOT attisdropped
            )
            """
        )

        state = await migrate(
            _PinnedPool(connection),  # type: ignore[arg-type]
            allow_retrieval_publication_backfill=True,
        )

        constraints = await connection.fetch(
            """
            SELECT conname, convalidated
            FROM pg_catalog.pg_constraint
            WHERE conname = ANY($1::pg_catalog.text[])
            ORDER BY conname
            """,
            ["document_chunks_document_fk", "document_sections_document_fk"],
        )
        trigger_state = await connection.fetchval(
            """
            SELECT tgenabled::pg_catalog.text
            FROM pg_catalog.pg_trigger
            WHERE tgrelid = 'public.document_sections'::pg_catalog.regclass
              AND tgname = 'trg_document_sections_tsv'
              AND NOT tgisinternal
            """
        )
        source_hash = await connection.fetchval(
            "SELECT source_content_hash FROM public.document_sections WHERE doc_id = 'v3-upgrade-proof'"
        )
        tsv_after = await connection.fetchval(
            "SELECT tsv::pg_catalog.text FROM public.document_sections WHERE doc_id = 'v3-upgrade-proof'"
        )

        assert state.current
        assert source_hash == content_hash
        assert tsv_after == tsv_before
        assert trigger_state == "O"
        assert [(row["conname"], row["convalidated"]) for row in constraints] == [
            ("document_chunks_document_fk", True),
            ("document_sections_document_fk", True),
        ]
        assert (
            await connection.fetchval("SELECT pg_catalog.to_regclass('public.document_retrieval_publications')")
            is not None
        )
        assert (await migrate(_PinnedPool(connection))).current  # type: ignore[arg-type]
    finally:
        await transaction.rollback()
        await pg_pool.release(connection)


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_postgres_v3_failure_after_disabling_section_fts_rolls_back_trigger_and_schema(pg_pool):
    connection = await pg_pool.acquire()
    transaction = connection.transaction()
    await transaction.start()
    try:
        await _downgrade_current_schema_to_v2(connection)
        await connection.execute(
            """
            INSERT INTO public.documents (document_id, title, markdown_content, content_hash)
            VALUES ('v3-rollback-proof', 'Rollback proof', 'Proof body', $1)
            """,
            "c" * 64,
        )
        failing_v3 = replace(
            MIGRATIONS[2],
            statements=MIGRATIONS[2].statements[:3] + ("SELECT 1 / 0",),
        )

        with (
            patch("bddk_mcp.migrations.runner.MIGRATIONS", MIGRATIONS[:2] + (failing_v3,)),
            pytest.raises(MigrationError, match="rolled back"),
        ):
            await migrate(
                _PinnedPool(connection),  # type: ignore[arg-type]
                allow_retrieval_publication_backfill=True,
            )

        trigger_state = await connection.fetchval(
            """
            SELECT tgenabled::pg_catalog.text
            FROM pg_catalog.pg_trigger
            WHERE tgrelid = 'public.document_sections'::pg_catalog.regclass
              AND tgname = 'trg_document_sections_tsv'
              AND NOT tgisinternal
            """
        )
        source_column_exists = await connection.fetchval(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_catalog.pg_attribute
                WHERE attrelid = 'public.document_sections'::pg_catalog.regclass
                  AND attname = 'source_content_hash'
                  AND NOT attisdropped
            )
            """
        )

        assert trigger_state == "O"
        assert not source_column_exists
        assert await connection.fetchval("SELECT max(version) FROM bddk_meta.schema_migrations") == 2
    finally:
        await transaction.rollback()
        await pg_pool.release(connection)


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_postgres_refuses_unmanaged_schema_and_rolls_back_the_entire_invocation(pg_pool):
    await pg_pool.execute("DROP SCHEMA IF EXISTS bddk_operator CASCADE")
    await pg_pool.execute("DROP SCHEMA IF EXISTS bddk_meta CASCADE")
    await pg_pool.execute(
        """
        DROP TABLE IF EXISTS
            public.regulatory_legal_version_provisions,
            public.regulatory_legal_status_assertions,
            public.regulatory_legal_events,
            public.regulatory_legal_version_artifacts,
            public.regulatory_provisions,
            public.regulatory_legal_versions,
            public.regulatory_evidence,
            public.regulatory_source_artifacts,
            public.regulatory_source_blobs,
            public.regulatory_family_imports,
            public.regulatory_instruments,
            public.document_retrieval_publications,
            public.document_chunks,
            public.decision_cache,
            public.sync_failures,
            public.sync_metadata,
            public.tool_call_traces,
            public.document_versions,
            public.document_sections,
            public.documents
        CASCADE
        """
    )
    await pg_pool.execute("DROP FUNCTION IF EXISTS public.chunks_tsv_trigger() CASCADE")
    await pg_pool.execute("DROP FUNCTION IF EXISTS public.invalidate_retrieval_publication() CASCADE")
    await pg_pool.execute("DROP FUNCTION IF EXISTS public.document_sections_tsv_trigger() CASCADE")
    await pg_pool.execute("DROP FUNCTION IF EXISTS public.documents_tsv_trigger() CASCADE")
    await pg_pool.execute("DROP FUNCTION IF EXISTS public.immutable_unaccent(pg_catalog.text) CASCADE")
    await pg_pool.execute("CREATE TABLE public.documents (unmanaged pg_catalog.text)")

    try:
        with pytest.raises(MigrationError, match="rolled back"):
            await migrate(pg_pool)

        assert await pg_pool.fetchval("SELECT pg_catalog.to_regclass('bddk_meta.schema_migrations')") is None
        assert await pg_pool.fetchval("SELECT pg_catalog.to_regclass('bddk_operator.operator_jobs')") is None
        assert (
            await pg_pool.fetchval("SELECT pg_catalog.to_regprocedure('public.immutable_unaccent(pg_catalog.text)')")
            is None
        )
    finally:
        await pg_pool.execute("DROP TABLE IF EXISTS public.documents CASCADE")
        connection = await pg_pool.acquire()
        try:
            await connection.execute("SET search_path TO pg_catalog")
            restored = await migrate(_PinnedPool(connection))  # type: ignore[arg-type]
            assert restored.current
        finally:
            await connection.execute("RESET search_path")
            await pg_pool.release(connection)
