"""Explicit PostgreSQL migration and read-only serving-readiness checks.

Schema creation and data repair belong to operator commands.  The serving
process consumes :func:`assert_database_ready`, which deliberately uses only
``SELECT`` statements and accepts compatible schema supersets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import asyncpg

from bddk_mcp.catalog_integrity import inspect_catalog_integrity
from bddk_mcp.core.config import require_database_url, require_expected_database_name
from bddk_mcp.db_transport import assert_database_transport
from bddk_mcp.migrations import LATEST_SCHEMA_VERSION, LegacyAdoptionError, MigrationError, inspect_migration_state
from bddk_mcp.migrations import migrate as apply_migrations

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
            "source_content_hash",
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
    "document_retrieval_publications": frozenset(
        {
            "doc_id",
            "content_hash",
            "retrieval_profile_hash",
            "expected_chunks",
            "published_at",
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
FROM pg_catalog.pg_extension
WHERE extname = ANY($1::text[])
"""

_RELATIONS_SQL = """
SELECT requested.relation_name,
       relation.relname AS resolved_relation
FROM unnest($1::text[]) AS requested(relation_name)
LEFT JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.nspname = 'public'
LEFT JOIN pg_catalog.pg_class AS relation
  ON relation.relnamespace = namespace.oid
 AND relation.relname = requested.relation_name
 AND relation.relkind IN ('r', 'p')
"""

_COLUMNS_SQL = """
SELECT requested.relation_name AS table_name,
       attribute.attname AS column_name
FROM unnest($1::text[]) AS requested(relation_name)
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.nspname = 'public'
JOIN pg_catalog.pg_class AS relation
  ON relation.relnamespace = namespace.oid
 AND relation.relname = requested.relation_name
 AND relation.relkind IN ('r', 'p')
JOIN pg_catalog.pg_attribute AS attribute
  ON attribute.attrelid = relation.oid
WHERE attribute.attnum > 0
  AND NOT attribute.attisdropped
"""

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

_SCHEMA_OWNER_IDENTITY_SQL = """
WITH RECURSIVE session_role AS (
    SELECT role.oid,
           role.rolname,
           role.rolcanlogin,
           role.rolinherit,
           role.rolsuper,
           role.rolcreaterole,
           role.rolcreatedb,
           role.rolreplication,
           role.rolbypassrls
    FROM pg_catalog.pg_roles AS role
    WHERE role.rolname = session_user
), current_role_record AS (
    SELECT role.oid,
           role.rolname,
           role.rolcanlogin,
           role.rolsuper,
           role.rolcreaterole,
           role.rolcreatedb,
           role.rolreplication,
           role.rolbypassrls
    FROM pg_catalog.pg_roles AS role
    WHERE role.rolname = current_user
), role_closure AS (
    SELECT session_role.oid,
           session_role.rolname,
           session_role.rolcanlogin,
           session_role.rolsuper,
           session_role.rolcreaterole,
           session_role.rolcreatedb,
           session_role.rolreplication,
           session_role.rolbypassrls
    FROM session_role
    UNION
    SELECT inherited.oid,
           inherited.rolname,
           inherited.rolcanlogin,
           inherited.rolsuper,
           inherited.rolcreaterole,
           inherited.rolcreatedb,
           inherited.rolreplication,
           inherited.rolbypassrls
    FROM role_closure AS member_role
    JOIN pg_catalog.pg_auth_members AS membership
      ON membership.member = member_role.oid
    JOIN pg_catalog.pg_roles AS inherited
      ON inherited.oid = membership.roleid
), direct_memberships AS (
    SELECT inherited.rolname,
           membership.admin_option
    FROM session_role
    JOIN pg_catalog.pg_auth_members AS membership
      ON membership.member = session_role.oid
    JOIN pg_catalog.pg_roles AS inherited
      ON inherited.oid = membership.roleid
)
SELECT current_database()::pg_catalog.text AS database_name,
       current_user::pg_catalog.text AS current_user_name,
       session_user::pg_catalog.text AS session_user_name,
       COALESCE((
           SELECT NOT current_role_record.rolcanlogin
              AND NOT current_role_record.rolsuper
              AND NOT current_role_record.rolcreaterole
              AND NOT current_role_record.rolcreatedb
              AND NOT current_role_record.rolreplication
              AND NOT current_role_record.rolbypassrls
           FROM current_role_record
       ), false) AS current_role_is_restricted_owner,
       COALESCE((
           SELECT session_role.rolcanlogin
              AND session_role.rolinherit
              AND NOT session_role.rolsuper
              AND NOT session_role.rolcreaterole
              AND NOT session_role.rolcreatedb
              AND NOT session_role.rolreplication
              AND NOT session_role.rolbypassrls
           FROM session_role
       ), false) AS session_role_is_restricted_login,
       COALESCE(
           ARRAY(SELECT rolname FROM direct_memberships ORDER BY rolname),
           ARRAY[]::pg_catalog.text[]
       ) AS direct_memberships,
       EXISTS (SELECT 1 FROM direct_memberships WHERE admin_option) AS membership_admin,
       EXISTS (
           SELECT 1
           FROM role_closure
           WHERE rolsuper
              OR rolcreaterole
              OR rolcreatedb
              OR rolreplication
              OR rolbypassrls
              OR (rolname <> session_user AND rolcanlogin)
       ) AS unsafe_inherited_role,
       EXISTS (
           SELECT 1
           FROM pg_catalog.pg_database AS database_record
           JOIN session_role ON session_role.oid = database_record.datdba
           WHERE database_record.datname = current_database()
       ) AS session_owns_database,
       pg_catalog.has_database_privilege(
           current_user, current_database(), 'CONNECT'
       ) AS owner_can_connect,
       pg_catalog.has_database_privilege(
           current_user, current_database(), 'CREATE'
       ) AS owner_can_create,
       pg_catalog.has_database_privilege(
           current_user, current_database(), 'TEMPORARY'
       ) AS owner_can_create_temporary,
       EXISTS (
           SELECT 1
           FROM pg_catalog.pg_namespace AS namespace
           JOIN current_role_record ON current_role_record.oid = namespace.nspowner
           WHERE namespace.nspname = 'public'
       ) AS owner_owns_public_schema,
       pg_catalog.has_schema_privilege(current_user, 'public', 'USAGE') AS owner_has_public_usage,
       pg_catalog.has_schema_privilege(current_user, 'public', 'CREATE') AS owner_has_public_create
"""


class DatabaseLifecycleError(RuntimeError):
    """Sanitized operator-facing database lifecycle error."""


class DatabaseNotReadyError(DatabaseLifecycleError):
    """Raised when serving prerequisites have not been prepared explicitly."""


class SchemaOwnerIdentityError(DatabaseLifecycleError):
    """Raised when lifecycle credentials do not match the migration contract."""


@dataclass(frozen=True)
class DatabaseReadiness:
    """Read-only assessment of schema and reviewed-corpus readiness."""

    migration_version: int = LATEST_SCHEMA_VERSION
    missing_extensions: tuple[str, ...] = ()
    missing_relations: tuple[str, ...] = ()
    missing_columns: tuple[str, ...] = ()
    catalog_issues: tuple[str, ...] = ()
    corpus_issues: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.migration_version == LATEST_SCHEMA_VERSION and not (
            self.missing_extensions
            or self.missing_relations
            or self.missing_columns
            or self.catalog_issues
            or self.corpus_issues
        )

    def summary(self) -> str:
        """Return a bounded description containing no connection or query data."""
        parts: list[str] = []
        if self.migration_version != LATEST_SCHEMA_VERSION:
            parts.append(f"schema migration version is {self.migration_version}; expected {LATEST_SCHEMA_VERSION}")
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
        if self.catalog_issues:
            shown_catalog = self.catalog_issues[:8]
            suffix = (
                f" (+{len(self.catalog_issues) - len(shown_catalog)} more)"
                if len(shown_catalog) < len(self.catalog_issues)
                else ""
            )
            parts.append("managed catalog integrity failures: " + ", ".join(shown_catalog) + suffix)
        parts.extend(self.corpus_issues)
        return "; ".join(parts) if parts else "ready"


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


async def assert_schema_owner_identity(pool: asyncpg.Pool, expected_database: str) -> None:
    """Fail closed unless migration uses the exact restricted owner boundary."""

    try:
        row = await pool.fetchrow(_SCHEMA_OWNER_IDENTITY_SQL)
        valid = (
            row is not None
            and str(_row_value(row, "database_name", "")) == expected_database
            and str(_row_value(row, "current_user_name", "")) == "bddk_schema_owner"
            and str(_row_value(row, "session_user_name", "")) != "bddk_schema_owner"
            and bool(_row_value(row, "current_role_is_restricted_owner", False))
            and bool(_row_value(row, "session_role_is_restricted_login", False))
            and tuple(_row_value(row, "direct_memberships", ()) or ()) == ("bddk_schema_owner",)
            and not bool(_row_value(row, "membership_admin", True))
            and not bool(_row_value(row, "unsafe_inherited_role", True))
            and not bool(_row_value(row, "session_owns_database", True))
            and bool(_row_value(row, "owner_can_connect", False))
            and bool(_row_value(row, "owner_can_create", False))
            and not bool(_row_value(row, "owner_can_create_temporary", True))
            and bool(_row_value(row, "owner_owns_public_schema", False))
            and bool(_row_value(row, "owner_has_public_usage", False))
            and bool(_row_value(row, "owner_has_public_create", False))
        )
        if not valid:
            raise SchemaOwnerIdentityError(
                "The migration database identity or target does not satisfy the exact schema-owner contract."
            )
    except SchemaOwnerIdentityError:
        raise
    except Exception:
        raise SchemaOwnerIdentityError(
            "The migration database identity and target could not be verified safely."
        ) from None


async def inspect_database_readiness(pool: asyncpg.Pool, *, require_corpus: bool = True) -> DatabaseReadiness:
    """Inspect readiness using SELECT statements only.

    Compatible schema supersets are accepted.  Corpus checks run only after all
    required relations and columns exist, preventing secondary undefined-table
    errors from obscuring the actionable schema diagnosis.
    """
    table_names = sorted(_REQUIRED_COLUMNS)
    try:
        migration_state = await inspect_migration_state(pool)
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

        catalog_issues: tuple[str, ...] = ()
        if migration_state.current and not missing_relations and not missing_columns:
            catalog_issues = (await inspect_catalog_integrity(pool)).failures

        corpus_issues: list[str] = []
        if require_corpus and not missing_relations and not missing_columns and not catalog_issues:
            from bddk_mcp.store.vector_store import retrieval_profile_hash

            corpus = await pool.fetchrow(_CORPUS_READINESS_SQL, retrieval_profile_hash())
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
            if _row_value(corpus, "invalid_document_hash", False):
                corpus_issues.append("stored document hashes are invalid")
            if _row_value(corpus, "invalid_section_publication", False):
                corpus_issues.append("section index is stale or orphaned")
            if _row_value(corpus, "missing_current_publication", False):
                corpus_issues.append("documents require publication for the current retrieval profile")
            if _row_value(corpus, "invalid_chunk_publication", False):
                corpus_issues.append("retrieval publication is incomplete")

        return DatabaseReadiness(
            migration_version=migration_state.current_version,
            missing_extensions=tuple(sorted(_REQUIRED_EXTENSIONS - installed_extensions)),
            missing_relations=missing_relations,
            missing_columns=missing_columns,
            catalog_issues=catalog_issues,
            corpus_issues=tuple(corpus_issues),
        )
    except (MigrationError, asyncpg.PostgresError, OSError):
        raise DatabaseLifecycleError(
            "Database readiness could not be verified. Ensure the database is reachable and the serving role "
            "has catalog and SELECT access; run `bddk-mcp migrate` with schema-owner credentials and "
            "`bddk-mcp bootstrap` with ingestion credentials."
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
            raise DatabaseNotReadyError(
                "Database is not ready for serving ("
                + readiness.summary()
                + "). Run `bddk-mcp migrate` with schema-owner credentials and then `bddk-mcp bootstrap` "
                "with ingestion credentials."
            )
        return readiness
    except DatabaseLifecycleError:
        raise
    except Exception:
        raise DatabaseLifecycleError(
            "Database readiness could not be verified. Ensure the database is reachable and rerun "
            "`bddk-mcp migrate` with schema-owner credentials and `bddk-mcp bootstrap` with ingestion credentials."
        ) from None
    finally:
        if owns_pool and active_pool is not None:
            await active_pool.close()


async def migrate_database(
    dsn: str | None = None,
    *,
    pool: asyncpg.Pool | None = None,
    adopt_legacy: bool = False,
    allow_retrieval_publication_backfill: bool = False,
    expected_database: str | None = None,
) -> DatabaseReadiness:
    """Explicitly create/upgrade all current PostgreSQL schema components.

    This is the only lifecycle API in this module that intentionally performs
    DDL or a migration backfill. It must be run with schema-owner credentials,
    never from the serving lifespan.
    """
    owns_pool = pool is None
    active_pool = pool
    try:
        if active_pool is None:
            selected_dsn = assert_database_transport(dsn) if dsn else require_database_url("schema-owner")
            active_pool = await asyncpg.create_pool(
                selected_dsn,
                min_size=1,
                max_size=3,
            )
        await assert_schema_owner_identity(
            active_pool,
            expected_database or require_expected_database_name(),
        )
        migration_options: dict[str, bool] = {}
        if adopt_legacy:
            migration_options["adopt_legacy"] = True
        if allow_retrieval_publication_backfill:
            migration_options["allow_retrieval_publication_backfill"] = True
        await apply_migrations(active_pool, **migration_options)
        return await assert_database_ready(pool=active_pool, require_corpus=False)
    except (LegacyAdoptionError, MigrationError) as exc:
        # Migration exceptions contain bounded, credential-free remediation.
        # Preserve scale, lock, timeout, and adoption refusals for operators
        # instead of replacing them with an unactionable generic error.
        raise DatabaseLifecycleError(str(exc)) from None
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
