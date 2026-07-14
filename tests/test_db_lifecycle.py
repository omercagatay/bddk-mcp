"""Tests for explicit DB migration and SELECT-only serving readiness."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from bddk_mcp.core.exceptions import BddkStorageError
from bddk_mcp.db_lifecycle import (
    _REQUIRED_COLUMNS,
    DatabaseLifecycleError,
    DatabaseNotReadyError,
    DatabaseReadiness,
    assert_database_ready,
    inspect_database_readiness,
    migrate_database,
)
from bddk_mcp.ingest.client import BddkApiClient


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
    }
    values.update(overrides)
    return values


class ReadOnlyReadinessPool:
    """Catalog-shaped pool that rejects every mutating SQL API."""

    def __init__(
        self,
        *,
        extensions: set[str] | None = None,
        relations: set[str] | None = None,
        columns: dict[str, set[str]] | None = None,
        corpus: dict[str, bool] | None = None,
    ) -> None:
        self.extensions = extensions if extensions is not None else {"unaccent", "vector"}
        self.relations = relations if relations is not None else set(_REQUIRED_COLUMNS)
        self.columns = (
            columns
            if columns is not None
            else {table_name: set(required) for table_name, required in _REQUIRED_COLUMNS.items()}
        )
        self.corpus = corpus if corpus is not None else _ready_corpus()
        self.statements: list[str] = []

    def _record_select(self, query: str) -> None:
        normalized = query.strip()
        assert normalized.upper().startswith("SELECT"), normalized
        self.statements.append(normalized)

    async def fetch(self, query: str, *args):
        self._record_select(query)
        if "FROM pg_extension" in query:
            return [{"extname": name} for name in sorted(self.extensions)]
        if "resolved_relation" in query:
            return [
                {
                    "relation_name": table_name,
                    "resolved_relation": table_name if table_name in self.relations else None,
                }
                for table_name in args[0]
            ]
        if "pg_attribute" in query:
            return [
                {"table_name": table_name, "column_name": column_name}
                for table_name in args[0]
                if table_name in self.relations
                for column_name in sorted(self.columns.get(table_name, set()))
            ]
        raise AssertionError(f"unexpected readiness query: {query}")

    async def fetchrow(self, query: str, *args):
        self._record_select(query)
        assert "has_decision_cache" in query
        return self.corpus

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
    assert all(statement.upper().startswith("SELECT") for statement in pool.statements)


@pytest.mark.asyncio
async def test_schema_only_readiness_skips_all_corpus_queries():
    pool = ReadOnlyReadinessPool(corpus=_ready_corpus(has_documents=False))

    report = await inspect_database_readiness(pool, require_corpus=False)  # type: ignore[arg-type]

    assert report.ready
    assert len(pool.statements) == 3


@pytest.mark.asyncio
async def test_readiness_reports_missing_schema_artifacts_without_querying_corpus():
    columns = {table_name: set(required) for table_name, required in _REQUIRED_COLUMNS.items()}
    columns["document_chunks"].remove("embedding")
    pool = ReadOnlyReadinessPool(
        extensions={"unaccent"},
        relations=set(_REQUIRED_COLUMNS) - {"document_versions"},
        columns=columns,
    )

    report = await inspect_database_readiness(pool)  # type: ignore[arg-type]

    assert report.missing_extensions == ("vector",)
    assert report.missing_relations == ("document_versions",)
    assert report.missing_columns == ("document_chunks.embedding",)
    assert report.corpus_issues == ()
    assert len(pool.statements) == 3


@pytest.mark.asyncio
async def test_readiness_accepts_schema_superset():
    columns = {table_name: set(required) | {"future_column"} for table_name, required in _REQUIRED_COLUMNS.items()}
    pool = ReadOnlyReadinessPool(
        extensions={"unaccent", "vector", "future_extension"},
        relations=set(_REQUIRED_COLUMNS) | {"future_table"},
        columns=columns,
    )

    report = await inspect_database_readiness(pool)  # type: ignore[arg-type]

    assert report.ready


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
    pool = ReadOnlyReadinessPool(relations=set())

    with pytest.raises(DatabaseNotReadyError) as exc_info:
        await assert_database_ready(pool=pool)  # type: ignore[arg-type]

    message = str(exc_info.value)
    assert "bddk-mcp migrate" in message
    assert "bddk-mcp bootstrap" in message
    assert "postgresql://" not in message
    assert "SELECT" not in message


@pytest.mark.asyncio
async def test_readiness_driver_error_is_sanitized():
    sentinel = "postgresql://private:password@secret-host/bddk"

    class FailingPool:
        async def fetch(self, *args, **kwargs):
            raise asyncpg.PostgresError(sentinel)

    with pytest.raises(DatabaseLifecycleError) as exc_info:
        await inspect_database_readiness(FailingPool())  # type: ignore[arg-type]

    assert sentinel not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert "SELECT access" in str(exc_info.value)


@pytest.mark.asyncio
async def test_migrate_orchestrates_every_schema_then_validates_schema_only():
    events: list[str] = []
    pool = MagicMock()

    async def record_documents():
        events.append("documents")

    async def record_cache(_pool):
        events.append("cache")

    async def record_vectors():
        events.append("vectors")

    document_store = MagicMock()
    document_store.initialize = AsyncMock(side_effect=record_documents)
    vector_store = MagicMock()
    vector_store.initialize = AsyncMock(side_effect=record_vectors)
    ready = DatabaseReadiness()

    with (
        patch("bddk_mcp.store.doc_store.DocumentStore", return_value=document_store),
        patch("bddk_mcp.ingest.client.initialize_cache_schema", new=AsyncMock(side_effect=record_cache)),
        patch("bddk_mcp.store.vector_store.VectorStore", return_value=vector_store),
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=AsyncMock(return_value=ready)) as validate,
    ):
        result = await migrate_database(pool=pool)

    assert result is ready
    assert events == ["documents", "cache", "vectors"]
    validate.assert_awaited_once_with(pool=pool, require_corpus=False)


@pytest.mark.asyncio
async def test_readiness_queries_execute_against_postgresql(pg_pool):
    """Validate the catalog and corpus SELECT syntax against real PostgreSQL."""
    await migrate_database(pool=pg_pool)

    schema_report = await inspect_database_readiness(pg_pool, require_corpus=False)
    corpus_report = await inspect_database_readiness(pg_pool, require_corpus=True)

    assert schema_report.ready
    assert isinstance(corpus_report, DatabaseReadiness)


class CachePool:
    def __init__(self, rows=None, error: Exception | None = None) -> None:
        self.rows = rows or []
        self.error = error
        self.statements: list[str] = []

    async def fetch(self, query: str, *args):
        normalized = query.strip()
        assert normalized.upper().startswith("SELECT")
        self.statements.append(normalized)
        if self.error:
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
async def test_strict_cache_load_rejects_empty_cache_with_bootstrap_instruction():
    client = BddkApiClient(pool=CachePool(), http=MagicMock())  # type: ignore[arg-type]

    with pytest.raises(BddkStorageError, match="bddk-mcp bootstrap"):
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
