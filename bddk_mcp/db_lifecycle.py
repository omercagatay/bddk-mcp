"""Explicit PostgreSQL migration and read-only serving-readiness checks.

Schema creation and data repair belong to operator commands.  The serving
process consumes :func:`assert_database_ready`, which deliberately uses only
``SELECT`` statements and accepts compatible schema supersets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import asyncpg

from bddk_mcp.core.config import require_database_url

_REQUIRED_EXTENSIONS = frozenset({"unaccent", "vector"})
_REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "decision_cache": frozenset(
        {
            "document_id",
            "title",
            "content",
            "decision_date",
            "decision_number",
            "category",
            "source_url",
            "cached_at",
        }
    ),
    "documents": frozenset(
        {
            "document_id",
            "title",
            "category",
            "decision_date",
            "decision_number",
            "source_url",
            "pdf_blob",
            "markdown_content",
            "content_hash",
            "downloaded_at",
            "extracted_at",
            "extraction_method",
            "total_pages",
            "file_size",
            "tsv",
        }
    ),
    "document_sections": frozenset(
        {
            "id",
            "doc_id",
            "section_type",
            "section_ref",
            "heading",
            "start_char",
            "end_char",
            "content",
            "content_hash",
            "page_start",
            "page_end",
            "tsv",
        }
    ),
    "document_versions": frozenset({"id", "document_id", "version", "content_hash", "markdown_content", "synced_at"}),
    "document_chunks": frozenset(
        {
            "id",
            "doc_id",
            "chunk_index",
            "title",
            "category",
            "decision_date",
            "decision_number",
            "source_url",
            "total_chunks",
            "total_pages",
            "content_hash",
            "chunk_start_char",
            "chunk_end_char",
            "section_type",
            "section_ref",
            "section_start_char",
            "section_end_char",
            "section_content_hash",
            "chunk_text",
            "embedding",
            "tsv",
        }
    ),
    "tool_call_traces": frozenset(
        {
            "id",
            "created_at",
            "tool_name",
            "args_hash",
            "args_summary",
            "latency_ms",
            "result_count",
            "doc_ids",
            "quality_labels",
            "relevance_stats",
            "model_id",
            "session_id",
        }
    ),
    "sync_metadata": frozenset({"document_id", "etag", "last_modified", "last_sync_at", "sync_count"}),
    "sync_failures": frozenset(
        {
            "document_id",
            "error",
            "error_category",
            "source_url",
            "retryable",
            "attempts",
            "first_failed_at",
            "last_failed_at",
        }
    ),
}

_EXTENSIONS_SQL = """
SELECT extname
FROM pg_extension
WHERE extname = ANY($1::text[])
"""

_RELATIONS_SQL = """
SELECT requested.relation_name,
       to_regclass(requested.relation_name)::text AS resolved_relation
FROM unnest($1::text[]) AS requested(relation_name)
"""

_COLUMNS_SQL = """
SELECT requested.relation_name AS table_name,
       attribute.attname AS column_name
FROM unnest($1::text[]) AS requested(relation_name)
JOIN pg_attribute AS attribute
  ON attribute.attrelid = to_regclass(requested.relation_name)
WHERE attribute.attnum > 0
  AND NOT attribute.attisdropped
"""

_CORPUS_READINESS_SQL = """
SELECT
    EXISTS (SELECT 1 FROM decision_cache LIMIT 1) AS has_decision_cache,
    EXISTS (SELECT 1 FROM documents LIMIT 1) AS has_documents,
    EXISTS (SELECT 1 FROM document_sections LIMIT 1) AS has_sections,
    EXISTS (SELECT 1 FROM document_chunks LIMIT 1) AS has_chunks,
    EXISTS (
        SELECT 1 FROM documents
        WHERE COALESCE(markdown_content, '') <> '' AND tsv IS NULL
        LIMIT 1
    ) AS documents_missing_tsv,
    EXISTS (
        SELECT 1 FROM document_sections WHERE tsv IS NULL LIMIT 1
    ) AS sections_missing_tsv,
    EXISTS (
        SELECT 1 FROM document_chunks WHERE tsv IS NULL LIMIT 1
    ) AS chunks_missing_tsv,
    EXISTS (
        SELECT 1 FROM document_chunks WHERE embedding IS NULL LIMIT 1
    ) AS chunks_missing_embedding,
    EXISTS (
        SELECT 1
        FROM document_chunks AS chunk
        JOIN documents AS document ON document.document_id = chunk.doc_id
        WHERE chunk.content_hash IS DISTINCT FROM document.content_hash
        LIMIT 1
    ) AS chunk_hash_mismatch,
    EXISTS (
        SELECT 1
        FROM document_chunks AS chunk
        LEFT JOIN documents AS document ON document.document_id = chunk.doc_id
        WHERE document.document_id IS NULL
        LIMIT 1
    ) AS orphan_chunks,
    EXISTS (
        SELECT 1
        FROM documents AS document
        WHERE COALESCE(document.markdown_content, '') <> ''
          AND NOT EXISTS (
              SELECT 1 FROM document_chunks AS chunk
              WHERE chunk.doc_id = document.document_id
          )
        LIMIT 1
    ) AS documents_without_chunks
"""


class DatabaseLifecycleError(RuntimeError):
    """Sanitized operator-facing database lifecycle error."""


class DatabaseNotReadyError(DatabaseLifecycleError):
    """Raised when serving prerequisites have not been prepared explicitly."""


@dataclass(frozen=True)
class DatabaseReadiness:
    """Read-only assessment of schema and reviewed-corpus readiness."""

    missing_extensions: tuple[str, ...] = ()
    missing_relations: tuple[str, ...] = ()
    missing_columns: tuple[str, ...] = ()
    corpus_issues: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return not (self.missing_extensions or self.missing_relations or self.missing_columns or self.corpus_issues)

    def summary(self) -> str:
        """Return a bounded description containing no connection or query data."""
        parts: list[str] = []
        if self.missing_extensions:
            parts.append("missing extensions: " + ", ".join(self.missing_extensions))
        if self.missing_relations:
            parts.append("missing tables: " + ", ".join(self.missing_relations))
        if self.missing_columns:
            shown = self.missing_columns[:12]
            suffix = (
                f" (+{len(self.missing_columns) - len(shown)} more)" if len(shown) < len(self.missing_columns) else ""
            )
            parts.append("missing columns: " + ", ".join(shown) + suffix)
        parts.extend(self.corpus_issues)
        return "; ".join(parts) if parts else "ready"


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


async def inspect_database_readiness(pool: asyncpg.Pool, *, require_corpus: bool = True) -> DatabaseReadiness:
    """Inspect readiness using SELECT statements only.

    Compatible schema supersets are accepted.  Corpus checks run only after all
    required relations and columns exist, preventing secondary undefined-table
    errors from obscuring the actionable schema diagnosis.
    """
    table_names = sorted(_REQUIRED_COLUMNS)
    try:
        extension_rows = await pool.fetch(_EXTENSIONS_SQL, sorted(_REQUIRED_EXTENSIONS))
        installed_extensions = {str(_row_value(row, "extname", "")) for row in extension_rows}

        relation_rows = await pool.fetch(_RELATIONS_SQL, table_names)
        resolved_relations = {
            str(_row_value(row, "relation_name", ""))
            for row in relation_rows
            if _row_value(row, "resolved_relation") is not None
        }
        missing_relations = tuple(sorted(set(table_names) - resolved_relations))

        column_rows = await pool.fetch(_COLUMNS_SQL, table_names)
        actual_columns: dict[str, set[str]] = {table_name: set() for table_name in table_names}
        for row in column_rows:
            table_name = str(_row_value(row, "table_name", ""))
            column_name = str(_row_value(row, "column_name", ""))
            if table_name in actual_columns and column_name:
                actual_columns[table_name].add(column_name)
        missing_columns = tuple(
            sorted(
                f"{table_name}.{column_name}"
                for table_name, required in _REQUIRED_COLUMNS.items()
                if table_name not in missing_relations
                for column_name in required - actual_columns[table_name]
            )
        )

        corpus_issues: list[str] = []
        if require_corpus and not missing_relations and not missing_columns:
            corpus = await pool.fetchrow(_CORPUS_READINESS_SQL)
            if not _row_value(corpus, "has_decision_cache", False):
                corpus_issues.append("decision cache is empty")
            if not _row_value(corpus, "has_documents", False):
                corpus_issues.append("document corpus is empty")
            if not _row_value(corpus, "has_sections", False):
                corpus_issues.append("section index is empty")
            if not _row_value(corpus, "has_chunks", False):
                corpus_issues.append("vector chunk index is empty")
            if _row_value(corpus, "documents_missing_tsv", False):
                corpus_issues.append("documents require full-text backfill")
            if _row_value(corpus, "sections_missing_tsv", False):
                corpus_issues.append("sections require full-text backfill")
            if _row_value(corpus, "chunks_missing_tsv", False):
                corpus_issues.append("chunks require full-text backfill")
            if _row_value(corpus, "chunks_missing_embedding", False):
                corpus_issues.append("chunks require embedding backfill")
            if _row_value(corpus, "chunk_hash_mismatch", False):
                corpus_issues.append("document and chunk hashes are inconsistent")
            if _row_value(corpus, "orphan_chunks", False):
                corpus_issues.append("orphan vector chunks exist")
            if _row_value(corpus, "documents_without_chunks", False):
                corpus_issues.append("stored documents are missing vector chunks")

        return DatabaseReadiness(
            missing_extensions=tuple(sorted(_REQUIRED_EXTENSIONS - installed_extensions)),
            missing_relations=missing_relations,
            missing_columns=missing_columns,
            corpus_issues=tuple(corpus_issues),
        )
    except (asyncpg.PostgresError, OSError):
        raise DatabaseLifecycleError(
            "Database readiness could not be verified. Ensure the database is reachable and the serving role "
            "has catalog and SELECT access; run `bddk-mcp migrate` and `bddk-mcp bootstrap` with operator credentials."
        ) from None


async def assert_database_ready(
    dsn: str | None = None,
    *,
    pool: asyncpg.Pool | None = None,
    require_corpus: bool = True,
) -> DatabaseReadiness:
    """Raise an actionable error unless the database is ready for serving."""
    owns_pool = pool is None
    active_pool = pool
    try:
        if active_pool is None:
            active_pool = await asyncpg.create_pool(dsn or require_database_url(), min_size=1, max_size=3)
        readiness = await inspect_database_readiness(active_pool, require_corpus=require_corpus)
        if not readiness.ready:
            raise DatabaseNotReadyError(
                "Database is not ready for serving ("
                + readiness.summary()
                + "). Run `bddk-mcp migrate` and then `bddk-mcp bootstrap` with operator credentials."
            )
        return readiness
    except DatabaseLifecycleError:
        raise
    except Exception:
        raise DatabaseLifecycleError(
            "Database readiness could not be verified. Ensure the database is reachable and rerun "
            "`bddk-mcp migrate` and `bddk-mcp bootstrap` with operator credentials."
        ) from None
    finally:
        if owns_pool and active_pool is not None:
            await active_pool.close()


async def migrate_database(dsn: str | None = None, *, pool: asyncpg.Pool | None = None) -> DatabaseReadiness:
    """Explicitly create/upgrade all current PostgreSQL schema components.

    This is the only lifecycle API in this module that intentionally performs
    DDL or a migration backfill.  It must be run with operator/schema-owner
    credentials, never from the serving lifespan.
    """
    from bddk_mcp.ingest.client import initialize_cache_schema
    from bddk_mcp.store.doc_store import DocumentStore
    from bddk_mcp.store.vector_store import VectorStore

    owns_pool = pool is None
    active_pool = pool
    try:
        if active_pool is None:
            active_pool = await asyncpg.create_pool(dsn or require_database_url(), min_size=1, max_size=3)
        await DocumentStore(active_pool).initialize()
        await initialize_cache_schema(active_pool)
        await VectorStore(active_pool).initialize()
        return await assert_database_ready(pool=active_pool, require_corpus=False)
    except DatabaseLifecycleError:
        raise
    except Exception:
        raise DatabaseLifecycleError(
            "Database migration failed. Run `bddk-mcp migrate` with schema-owner credentials and verify that "
            "the PostgreSQL vector and unaccent extensions are available."
        ) from None
    finally:
        if owns_pool and active_pool is not None:
            await active_pool.close()
