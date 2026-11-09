"""Tests for explicit DB migration and SELECT-only serving readiness."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from bddk_mcp.core.exceptions import BddkStorageError
from bddk_mcp.db_lifecycle import (
    DatabaseLifecycleError,
    DatabaseNotReadyError,
    DatabaseReadiness,
    assert_database_ready,
    inspect_database_readiness,
    migrate_database,
)
from bddk_mcp.ingest.client import BddkApiClient
from bddk_mcp.migrations import LATEST_SCHEMA_VERSION, MIGRATIONS, MigrationScaleError


def _ready_corpus(**overrides) -> dict[str, bool]:
    values = {
        "has_decision_cache": True,
        "has_documents": True,
        "has_sections": True,
        "has_chunks": True,
        "documents_missing_tsv": False,
        "sections_missing_tsv": False,
        "chunks_missing_tsv": False,
        "chunks_missing_embedding": False,
        "chunk_hash_mismatch": False,
        "orphan_chunks": False,
        "documents_without_chunks": False,
        "invalid_document_hash": False,
        "invalid_section_publication": False,
        "missing_current_publication": False,
        "invalid_chunk_publication": False,
    }
    values.update(overrides)
    return values


def _history_rows(count: int = len(MIGRATIONS)) -> list[dict]:
    return [{"version": item.version, "name": item.name, "checksum": item.checksum} for item in MIGRATIONS[:count]]


class ReadOnlyReadinessPool:
    """Answer the readiness SELECTs from canned rows and refuse any mutation."""

    def __init__(
        self,
        *,
        corpus: dict[str, bool] | None = None,
        history: list[dict] | None = None,
        server_version_num: int = 170004,
    ) -> None:
        self.corpus = corpus or _ready_corpus()
        self.history = _history_rows() if history is None else history
        self.server_version_num = server_version_num
        self.statements: list[str] = []

    async def fetchval(self, query: str, *args):
        self.statements.append(query.strip())
        if "server_version_num" in query:
            return self.server_version_num
        if "to_regclass" in query:
            return "bddk_meta.schema_migrations"
        raise AssertionError(f"unexpected fetchval: {query}")

    async def fetch(self, query: str, *args):
        self.statements.append(query.strip())
        if "schema_migrations" in query:
            return list(self.history)
        raise AssertionError(f"unexpected fetch: {query}")

    async def fetchrow(self, query: str, *args):
        self.statements.append(query.strip())
        if "has_decision_cache" in query:
            return dict(self.corpus)
        raise AssertionError(f"unexpected fetchrow: {query}")

    async def execute(self, *args, **kwargs):
        raise AssertionError("readiness must never call execute")

    async def executemany(self, *args, **kwargs):
        raise AssertionError("readiness must never call executemany")


@pytest.mark.asyncio
async def test_readiness_accepts_current_schema_and_uses_only_selects():
    pool = ReadOnlyReadinessPool()

    report = await inspect_database_readiness(pool)  # type: ignore[arg-type]

    assert report == DatabaseReadiness()
    assert report.ready
    assert len(pool.statements) == 4
    assert all(statement.upper().startswith(("SELECT", "WITH")) for statement in pool.statements)


@pytest.mark.asyncio
async def test_schema_only_readiness_skips_the_corpus_query():
    pool = ReadOnlyReadinessPool(corpus=_ready_corpus(has_documents=False))

    report = await inspect_database_readiness(pool, require_corpus=False)  # type: ignore[arg-type]

    assert report.ready
    assert len(pool.statements) == 3


@pytest.mark.asyncio
async def test_readiness_reports_pending_migration_without_querying_corpus():
    pool = ReadOnlyReadinessPool(history=_history_rows(LATEST_SCHEMA_VERSION - 1))

    report = await inspect_database_readiness(pool)  # type: ignore[arg-type]

    assert report.migration_version == LATEST_SCHEMA_VERSION - 1
    assert not report.ready
    assert f"expected {LATEST_SCHEMA_VERSION}" in report.summary()
    assert report.corpus_issues == ()
    assert len(pool.statements) == 3


@pytest.mark.asyncio
async def test_readiness_reports_incomplete_corpus_and_indexes():
    pool = ReadOnlyReadinessPool(
        corpus=_ready_corpus(
            has_decision_cache=False,
            has_sections=False,
            chunks_missing_embedding=True,
            chunk_hash_mismatch=True,
            documents_without_chunks=True,
        )
    )

    report = await inspect_database_readiness(pool)  # type: ignore[arg-type]

    assert report.corpus_issues == (
        "decision cache is empty",
        "section index is empty",
        "chunks require embedding backfill",
        "document and chunk hashes are inconsistent",
        "stored documents are missing vector chunks",
    )
    assert not report.ready


@pytest.mark.asyncio
async def test_assert_ready_error_is_actionable_and_contains_no_connection_details():
    pool = ReadOnlyReadinessPool(corpus=_ready_corpus(has_documents=False))

    with pytest.raises(DatabaseNotReadyError) as exc_info:
        await assert_database_ready(pool=pool)  # type: ignore[arg-type]

    message = str(exc_info.value)
    assert "document corpus is empty" in message
    assert "bddk-mcp migrate" in message
    assert "bddk-seed import" in message
    assert "postgresql://" not in message
    assert "SELECT" not in message


@pytest.mark.asyncio
async def test_readiness_driver_error_is_sanitized():
    sentinel = "postgresql://private:password@secret-host/bddk"

    class FailingPool:
        async def fetchval(self, *args, **kwargs):
            raise asyncpg.PostgresError(sentinel)

        async def fetch(self, *args, **kwargs):
            raise asyncpg.PostgresError(sentinel)

    with pytest.raises(DatabaseLifecycleError) as exc_info:
        await inspect_database_readiness(FailingPool())  # type: ignore[arg-type]

    assert sentinel not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert "compatibility could not be verified" in str(exc_info.value)


@pytest.mark.asyncio
async def test_readiness_refuses_unsupported_postgresql_before_ledger_or_corpus_inspection():
    pool = ReadOnlyReadinessPool(server_version_num=160012)

    with pytest.raises(DatabaseLifecycleError) as exc_info:
        await inspect_database_readiness(pool)  # type: ignore[arg-type]

    assert "requires PostgreSQL 17" in str(exc_info.value)
    assert "160012" not in str(exc_info.value)
    assert len(pool.statements) == 1


@pytest.mark.asyncio
async def test_migrate_orchestrates_every_schema_then_validates_schema_only():
    events: list[str] = []
    pool = MagicMock()

    async def record_migrations(_pool):
        events.append("migrations")

    ready = DatabaseReadiness()

    with (
        patch("bddk_mcp.db_lifecycle.apply_migrations", new=AsyncMock(side_effect=record_migrations)) as migrate,
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=AsyncMock(return_value=ready)) as validate,
    ):
        result = await migrate_database(pool=pool)

    assert result is ready
    assert events == ["migrations"]
    migrate.assert_awaited_once_with(pool)
    validate.assert_awaited_once_with(pool=pool, require_corpus=False)


@pytest.mark.asyncio
async def test_migrate_forwards_retrieval_backfill_approval_only_when_explicit():
    pool = MagicMock()
    ready = DatabaseReadiness()

    with (
        patch("bddk_mcp.db_lifecycle.apply_migrations", new=AsyncMock()) as migrate,
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=AsyncMock(return_value=ready)),
    ):
        result = await migrate_database(pool=pool, allow_retrieval_publication_backfill=True)

    assert result is ready
    migrate.assert_awaited_once_with(pool, allow_retrieval_publication_backfill=True)


@pytest.mark.asyncio
async def test_migrate_preserves_actionable_sanitized_scale_refusal():
    pool = MagicMock()
    refusal = MigrationScaleError(
        "Migration refused before its blocking backfill. Use --allow-retrieval-publication-backfill after rehearsal."
    )

    with patch("bddk_mcp.db_lifecycle.apply_migrations", new=AsyncMock(side_effect=refusal)):
        with pytest.raises(DatabaseLifecycleError, match="after rehearsal") as exc_info:
            await migrate_database(pool=pool)

    assert exc_info.value.__cause__ is None


@pytest.mark.asyncio
async def test_migrate_uses_schema_owner_profile_when_no_dsn_or_pool_is_supplied():
    pool = MagicMock()
    pool.close = AsyncMock()
    ready = DatabaseReadiness()

    with (
        patch(
            "bddk_mcp.db_lifecycle.require_database_url",
            return_value="postgresql://schema-owner",
        ) as require_url,
        patch("bddk_mcp.db_lifecycle.asyncpg.create_pool", new=AsyncMock(return_value=pool)) as create_pool,
        patch("bddk_mcp.db_lifecycle.apply_migrations", new=AsyncMock()),
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=AsyncMock(return_value=ready)),
    ):
        result = await migrate_database()

    assert result is ready
    require_url.assert_called_once_with("schema-owner")
    create_pool.assert_awaited_once_with("postgresql://schema-owner", min_size=1, max_size=3)
    pool.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_readiness_queries_execute_against_postgresql(pg_pool):
    """Validate the ledger and corpus SELECT syntax against real PostgreSQL."""
    schema_report = await inspect_database_readiness(pg_pool, require_corpus=False)
    corpus_report = await inspect_database_readiness(pg_pool, require_corpus=True)

    assert schema_report.ready
    assert isinstance(corpus_report, DatabaseReadiness)


class CachePool:
    """SELECT-only stand-in for the decision-cache load."""

    def __init__(self, rows: list[dict] | None = None, error: Exception | None = None) -> None:
        self.rows = rows or []
        self.error = error
        self.statements: list[str] = []

    async def fetch(self, query: str, *args):
        self.statements.append(query)
        if self.error is not None:
            raise self.error
        return self.rows

    async def execute(self, *args, **kwargs):
        raise AssertionError("read-only cache load must never call execute")


def _cache_row() -> dict:
    return {
        "document_id": "943",
        "title": "Model Riski Yönetimi Rehberi",
        "content": "İlke 5",
        "decision_date": "01.01.2024",
        "decision_number": "1234",
        "category": "Rehber",
        "source_url": "https://www.bddk.org.tr/example",
        "cached_at": 123.0,
    }


@pytest.mark.asyncio
async def test_strict_cache_load_is_select_only():
    pool = CachePool([_cache_row()])
    client = BddkApiClient(pool=pool, http=MagicMock())  # type: ignore[arg-type]

    count = await client.load_cache_read_only()

    assert count == 1
    assert client.cache_size() == 1
    assert len(pool.statements) == 1


@pytest.mark.asyncio
async def test_strict_cache_load_rejects_empty_cache_with_import_instruction():
    client = BddkApiClient(pool=CachePool(), http=MagicMock())  # type: ignore[arg-type]

    with pytest.raises(BddkStorageError, match="bddk-seed import"):
        await client.load_cache_read_only()

    assert client.cache_size() == 0


@pytest.mark.asyncio
async def test_strict_cache_load_sanitizes_database_errors():
    sentinel = "SELECT secret FROM private_table at postgresql://private-host"
    pool = CachePool(error=asyncpg.PostgresError(sentinel))
    client = BddkApiClient(pool=pool, http=MagicMock())  # type: ignore[arg-type]

    with pytest.raises(BddkStorageError) as exc_info:
        await client.load_cache_read_only()

    assert sentinel not in str(exc_info.value)
    assert "private_table" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
