"""Explicit PostgreSQL migration and read-only serving-readiness checks.

Schema creation and data import belong to operator commands.  The serving
process consumes :func:`assert_database_ready`, which deliberately uses only
``SELECT`` statements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import asyncpg

from bddk_mcp.core.config import require_database_url
from bddk_mcp.db_compatibility import PostgreSQLCompatibilityError, assert_supported_postgresql
from bddk_mcp.db_transport import assert_database_transport
from bddk_mcp.migrations import LATEST_SCHEMA_VERSION, MigrationError, inspect_migration_state
from bddk_mcp.migrations import migrate as apply_migrations

_CORPUS_READINESS_SQL = """
WITH chunk_integrity AS (
    SELECT publication.doc_id,
           publication.content_hash,
           publication.expected_chunks,
           COUNT(chunk.id)::pg_catalog.int4 AS actual_chunks,
           MIN(chunk.chunk_index)::pg_catalog.int4 AS first_index,
           MAX(chunk.chunk_index)::pg_catalog.int4 AS last_index,
           pg_catalog.bool_and(chunk.content_hash = publication.content_hash) AS hashes_match,
           pg_catalog.bool_and(chunk.total_chunks = publication.expected_chunks) AS totals_match,
           pg_catalog.bool_and(chunk.embedding IS NOT NULL) AS embeddings_complete
    FROM public.document_retrieval_publications AS publication
    LEFT JOIN public.document_chunks AS chunk
      ON chunk.doc_id = publication.doc_id
    GROUP BY publication.doc_id, publication.content_hash, publication.expected_chunks
)
SELECT
    EXISTS (SELECT 1 FROM public.decision_cache LIMIT 1) AS has_decision_cache,
    EXISTS (SELECT 1 FROM public.documents LIMIT 1) AS has_documents,
    EXISTS (SELECT 1 FROM public.document_sections LIMIT 1) AS has_sections,
    EXISTS (SELECT 1 FROM public.document_chunks LIMIT 1) AS has_chunks,
    EXISTS (
        SELECT 1 FROM public.documents
        WHERE COALESCE(markdown_content, '') <> '' AND tsv IS NULL
        LIMIT 1
    ) AS documents_missing_tsv,
    EXISTS (
        SELECT 1 FROM public.document_sections WHERE tsv IS NULL LIMIT 1
    ) AS sections_missing_tsv,
    EXISTS (
        SELECT 1 FROM public.document_chunks WHERE tsv IS NULL LIMIT 1
    ) AS chunks_missing_tsv,
    EXISTS (
        SELECT 1 FROM public.document_chunks WHERE embedding IS NULL LIMIT 1
    ) AS chunks_missing_embedding,
    EXISTS (
        SELECT 1
        FROM public.document_chunks AS chunk
        JOIN public.documents AS document ON document.document_id = chunk.doc_id
        WHERE chunk.content_hash IS DISTINCT FROM document.content_hash
        LIMIT 1
    ) AS chunk_hash_mismatch,
    EXISTS (
        SELECT 1
        FROM public.document_chunks AS chunk
        LEFT JOIN public.documents AS document ON document.document_id = chunk.doc_id
        WHERE document.document_id IS NULL
        LIMIT 1
    ) AS orphan_chunks,
    EXISTS (
        SELECT 1
        FROM public.documents AS document
        WHERE COALESCE(document.markdown_content, '') <> ''
          AND NOT EXISTS (
              SELECT 1 FROM public.document_chunks AS chunk
              WHERE chunk.doc_id = document.document_id
          )
        LIMIT 1
    ) AS documents_without_chunks
    , EXISTS (
        SELECT 1 FROM public.documents
        WHERE COALESCE(markdown_content, '') <> ''
          AND content_hash !~ '^[0-9a-f]{64}$'
        LIMIT 1
    ) AS invalid_document_hash,
    EXISTS (
        SELECT 1
        FROM public.document_sections AS section
        LEFT JOIN public.documents AS document ON document.document_id = section.doc_id
        WHERE document.document_id IS NULL
           OR section.source_content_hash IS DISTINCT FROM document.content_hash
        LIMIT 1
    ) AS invalid_section_publication,
    EXISTS (
        SELECT 1
        FROM public.documents AS document
        WHERE COALESCE(document.markdown_content, '') <> ''
          AND NOT EXISTS (
              SELECT 1
              FROM public.document_retrieval_publications AS publication
              WHERE publication.doc_id = document.document_id
                AND publication.content_hash = document.content_hash
                AND publication.retrieval_profile_hash = $1
          )
        LIMIT 1
    ) AS missing_current_publication,
    EXISTS (
        SELECT 1
        FROM chunk_integrity
        WHERE actual_chunks <> expected_chunks
           OR first_index <> 0
           OR last_index <> expected_chunks - 1
           OR NOT COALESCE(hashes_match, false)
           OR NOT COALESCE(totals_match, false)
           OR NOT COALESCE(embeddings_complete, false)
        LIMIT 1
    ) AS invalid_chunk_publication
"""

_CORPUS_ISSUE_MESSAGES: tuple[tuple[str, bool, str], ...] = (
    # (column, value that signals a problem, message)
    ("has_decision_cache", False, "decision cache is empty"),
    ("has_documents", False, "document corpus is empty"),
    ("has_sections", False, "section index is empty"),
    ("has_chunks", False, "vector chunk index is empty"),
    ("documents_missing_tsv", True, "documents require full-text backfill"),
    ("sections_missing_tsv", True, "sections require full-text backfill"),
    ("chunks_missing_tsv", True, "chunks require full-text backfill"),
    ("chunks_missing_embedding", True, "chunks require embedding backfill"),
    ("chunk_hash_mismatch", True, "document and chunk hashes are inconsistent"),
    ("orphan_chunks", True, "orphan vector chunks exist"),
    ("documents_without_chunks", True, "stored documents are missing vector chunks"),
    ("invalid_document_hash", True, "stored document hashes are invalid"),
    ("invalid_section_publication", True, "section index is stale or orphaned"),
    ("missing_current_publication", True, "documents require publication for the current retrieval profile"),
    ("invalid_chunk_publication", True, "retrieval publication is incomplete"),
)

_NOT_READY_HINT = "Run `bddk-mcp migrate`, then `bddk-seed import` (or `bddk-mcp sync`) to prepare the database."


class DatabaseLifecycleError(RuntimeError):
    """Sanitized operator-facing database lifecycle error."""


class DatabaseNotReadyError(DatabaseLifecycleError):
    """Raised when serving prerequisites have not been prepared explicitly."""


@dataclass(frozen=True)
class DatabaseReadiness:
    """Read-only assessment of schema and corpus readiness."""

    migration_version: int = LATEST_SCHEMA_VERSION
    corpus_issues: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.migration_version == LATEST_SCHEMA_VERSION and not self.corpus_issues

    def summary(self) -> str:
        """Return a bounded description containing no connection or query data."""
        parts: list[str] = []
        if self.migration_version != LATEST_SCHEMA_VERSION:
            parts.append(f"schema migration version is {self.migration_version}; expected {LATEST_SCHEMA_VERSION}")
        parts.extend(self.corpus_issues)
        return "; ".join(parts) if parts else "ready"


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


async def inspect_database_readiness(pool: asyncpg.Pool, *, require_corpus: bool = True) -> DatabaseReadiness:
    """Inspect readiness using SELECT statements only.

    Corpus checks run only when the migration ledger is current, so a missing
    migration is reported before any secondary undefined-table error.
    """
    try:
        await assert_supported_postgresql(pool)
        migration_state = await inspect_migration_state(pool)

        corpus_issues: list[str] = []
        if require_corpus and migration_state.current:
            from bddk_mcp.store.vector_store import retrieval_profile_hash

            corpus = await pool.fetchrow(_CORPUS_READINESS_SQL, retrieval_profile_hash())
            for column, problem_value, message in _CORPUS_ISSUE_MESSAGES:
                if bool(_row_value(corpus, column, not problem_value)) is problem_value:
                    corpus_issues.append(message)

        return DatabaseReadiness(
            migration_version=migration_state.current_version,
            corpus_issues=tuple(corpus_issues),
        )
    except PostgreSQLCompatibilityError as exc:
        raise DatabaseLifecycleError(str(exc)) from None
    except (MigrationError, asyncpg.PostgresError, OSError):
        raise DatabaseLifecycleError(
            "Database readiness could not be verified. Ensure the database is reachable and the serving role "
            f"has SELECT access. {_NOT_READY_HINT}"
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
            selected_dsn = assert_database_transport(dsn) if dsn else require_database_url()
            active_pool = await asyncpg.create_pool(selected_dsn, min_size=1, max_size=3)
        readiness = await inspect_database_readiness(active_pool, require_corpus=require_corpus)
        if not readiness.ready:
            raise DatabaseNotReadyError(f"Database is not ready for serving ({readiness.summary()}). {_NOT_READY_HINT}")
        return readiness
    except DatabaseLifecycleError:
        raise
    except Exception:
        raise DatabaseLifecycleError(
            f"Database readiness could not be verified. Ensure the database is reachable. {_NOT_READY_HINT}"
        ) from None
    finally:
        if owns_pool and active_pool is not None:
            await active_pool.close()


async def migrate_database(
    dsn: str | None = None,
    *,
    pool: asyncpg.Pool | None = None,
    allow_retrieval_publication_backfill: bool = False,
) -> DatabaseReadiness:
    """Explicitly create/upgrade all current PostgreSQL schema components.

    This is the only lifecycle API in this module that intentionally performs
    DDL or a migration backfill.  It is never run from the serving lifespan.
    """
    owns_pool = pool is None
    active_pool = pool
    try:
        if active_pool is None:
            selected_dsn = assert_database_transport(dsn) if dsn else require_database_url("schema-owner")
            active_pool = await asyncpg.create_pool(selected_dsn, min_size=1, max_size=3)
        migration_options: dict[str, bool] = {}
        if allow_retrieval_publication_backfill:
            migration_options["allow_retrieval_publication_backfill"] = True
        await apply_migrations(active_pool, **migration_options)
        return await assert_database_ready(pool=active_pool, require_corpus=False)
    except MigrationError as exc:
        # Migration exceptions contain bounded, credential-free remediation.
        # Preserve scale, lock, and timeout refusals for operators
        # instead of replacing them with an unactionable generic error.
        raise DatabaseLifecycleError(str(exc)) from None
    except DatabaseLifecycleError:
        raise
    except Exception:
        raise DatabaseLifecycleError(
            "Database migration failed. Verify that the database is reachable and that the PostgreSQL vector "
            "and unaccent extensions are available."
        ) from None
    finally:
        if owns_pool and active_pool is not None:
            await active_pool.close()
