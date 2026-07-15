"""Transactional, checksum-verified PostgreSQL migration runner."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Final

import asyncpg

from bddk_mcp.db_compatibility import PostgreSQLCompatibilityError, assert_supported_postgresql
from bddk_mcp.migrations.legacy import (
    LEGACY_SOURCE_KIND,
    LEGACY_VERIFIER_VERSION,
    LegacyAdoptionError,
    inspect_legacy_v1,
    lock_legacy_v1_tables,
    normalize_legacy_v1,
)
from bddk_mcp.migrations.model import Migration
from bddk_mcp.migrations.v0001_core import V0001_CORE
from bddk_mcp.migrations.v0002_operator_jobs import V0002_OPERATOR_JOBS
from bddk_mcp.migrations.v0003_retrieval_publication import V0003_RETRIEVAL_PUBLICATION
from bddk_mcp.migrations.v0004_canonical_legal_versions import V0004_CANONICAL_LEGAL_VERSIONS
from bddk_mcp.migrations.v0005_corpus_release_publication import V0005_CORPUS_RELEASE_PUBLICATION
from bddk_mcp.migrations.v0006_legal_status_resolver import V0006_LEGAL_STATUS_RESOLVER

MIGRATIONS: Final[tuple[Migration, ...]] = (
    V0001_CORE,
    V0002_OPERATOR_JOBS,
    V0003_RETRIEVAL_PUBLICATION,
    V0004_CANONICAL_LEGAL_VERSIONS,
    V0005_CORPUS_RELEASE_PUBLICATION,
    V0006_LEGAL_STATUS_RESOLVER,
)
LATEST_SCHEMA_VERSION: Final[int] = MIGRATIONS[-1].version
MIGRATION_LOCK_TIMEOUT: Final[str] = "5s"
MIGRATION_STATEMENT_TIMEOUT: Final[str] = "120s"

_MIGRATION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,95}$")
_CHECKSUM_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_EXTENSIONS = frozenset({"unaccent", "vector"})
_HISTORY_RELATION = "bddk_meta.schema_migrations"
_HISTORY_LOCK_KEY = int.from_bytes(
    hashlib.sha256(b"bddk_mcp:global_schema_migrations:v1").digest()[:8],
    "big",
    signed=True,
)

_EXTENSIONS_SQL = """
SELECT extension.extname,
       namespace.nspname AS extension_schema
FROM pg_catalog.pg_extension AS extension
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = extension.extnamespace
WHERE extension.extname = ANY($1::pg_catalog.text[])
"""
_PUBLIC_EXTENSION_OBJECTS_SQL = """
SELECT pg_catalog.to_regprocedure('public.unaccent(pg_catalog.text)') IS NOT NULL AS has_unaccent,
       pg_catalog.to_regtype('public.vector') IS NOT NULL AS has_vector
"""
_HISTORY_EXISTS_SQL = "SELECT pg_catalog.to_regclass($1)::pg_catalog.text"
_HISTORY_SQL = """
SELECT version, name, checksum
FROM bddk_meta.schema_migrations
ORDER BY version
"""
_RETRIEVAL_PUBLICATION_BACKFILL_SQL = """
SELECT EXISTS(
           SELECT 1 FROM public.documents
       ) AS has_documents,
       EXISTS(
           SELECT 1 FROM public.document_sections
       ) AS has_sections,
       EXISTS(
           SELECT 1 FROM public.document_chunks
       ) AS has_chunks
"""
_CREATE_META_SCHEMA_SQL = "CREATE SCHEMA IF NOT EXISTS bddk_meta"
_CREATE_HISTORY_SQL = """
CREATE TABLE IF NOT EXISTS bddk_meta.schema_migrations (
    version pg_catalog.int4 NOT NULL,
    name pg_catalog.text NOT NULL,
    checksum pg_catalog.text NOT NULL,
    applied_at pg_catalog.timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT schema_migrations_pkey PRIMARY KEY (version),
    CONSTRAINT schema_migrations_name_uq UNIQUE (name),
    CONSTRAINT schema_migrations_version_check CHECK (version > 0),
    CONSTRAINT schema_migrations_name_check CHECK (name ~ '^[a-z][a-z0-9_]{0,95}$'),
    CONSTRAINT schema_migrations_checksum_check CHECK (checksum ~ '^[0-9a-f]{64}$')
)
"""
_INSERT_HISTORY_SQL = """
INSERT INTO bddk_meta.schema_migrations (version, name, checksum)
VALUES ($1, $2, $3)
"""
_INSERT_ADOPTION_AUDIT_SQL = """
INSERT INTO bddk_meta.legacy_schema_adoptions (
    migration_version,
    source_kind,
    verifier_version,
    target_checksum,
    pre_normalization_fingerprint,
    post_normalization_fingerprint,
    normalizations,
    adopted_by,
    adopted_session_user
)
VALUES (
    1, $1, $2, $3, $4, $5, $6::pg_catalog.text[], CURRENT_USER, SESSION_USER
)
"""


class MigrationError(RuntimeError):
    """Sanitized database migration failure."""


class MigrationPrerequisiteError(MigrationError):
    """Raised when DBA-managed PostgreSQL prerequisites are unavailable."""


class MigrationCompatibilityError(MigrationError):
    """Raised when the PostgreSQL major-version contract is not satisfied."""


class MigrationHistoryError(MigrationError):
    """Raised when persisted migration history is incompatible or altered."""


class MigrationNotReadyError(MigrationError):
    """Raised when the database has not reached this release's schema version."""


class MigrationScaleError(MigrationError):
    """Raised when a populated data backfill lacks explicit maintenance approval."""


class MigrationLockTimeoutError(MigrationError):
    """Raised when a migration cannot acquire a required lock in time."""


class MigrationStatementTimeoutError(MigrationError):
    """Raised when PostgreSQL cancels a bounded migration statement."""


@dataclass(frozen=True, slots=True)
class MigrationState:
    """Validated state of the single global migration ledger."""

    current_version: int
    latest_supported_version: int = LATEST_SCHEMA_VERSION

    @property
    def current(self) -> bool:
        return self.current_version == self.latest_supported_version

    @property
    def pending_versions(self) -> tuple[int, ...]:
        return tuple(range(self.current_version + 1, self.latest_supported_version + 1))


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def _validate_registry() -> None:
    versions = tuple(migration.version for migration in MIGRATIONS)
    if versions != tuple(range(1, len(MIGRATIONS) + 1)):
        raise RuntimeError("migration registry versions must be sequential from one")
    if len({migration.name for migration in MIGRATIONS}) != len(MIGRATIONS):
        raise RuntimeError("migration registry names must be unique")
    for migration in MIGRATIONS:
        if not _MIGRATION_NAME_RE.fullmatch(migration.name):
            raise RuntimeError(f"invalid migration name at version {migration.version}")
        if not migration.statements or any(not statement.strip() for statement in migration.statements):
            raise RuntimeError(f"migration {migration.version} has no executable statements")
        if not _CHECKSUM_RE.fullmatch(migration.checksum):
            raise RuntimeError(f"invalid migration checksum at version {migration.version}")


_validate_registry()


def validate_migration_history(rows: list[Any]) -> MigrationState:
    """Validate ordering and immutable identities of persisted migration rows."""

    parsed: list[tuple[int, str, str]] = []
    try:
        for row in rows:
            parsed.append(
                (
                    int(_row_value(row, "version")),
                    str(_row_value(row, "name")),
                    str(_row_value(row, "checksum")),
                )
            )
    except (TypeError, ValueError):
        raise MigrationHistoryError("Migration history contains an invalid record; migration refused.") from None

    versions = [version for version, _name, _checksum in parsed]
    if versions != list(range(1, len(parsed) + 1)):
        raise MigrationHistoryError("Migration history has a version gap or invalid ordering; migration refused.")
    if versions and versions[-1] > LATEST_SCHEMA_VERSION:
        raise MigrationHistoryError(
            "Database schema is newer than this server supports; upgrade the server before startup."
        )

    for version, stored_name, stored_checksum in parsed:
        expected = MIGRATIONS[version - 1]
        if stored_name != expected.name:
            raise MigrationHistoryError(f"Migration {version} name does not match this release; migration refused.")
        if stored_checksum != expected.checksum:
            raise MigrationHistoryError(f"Migration {version} checksum does not match this release; migration refused.")

    return MigrationState(current_version=versions[-1] if versions else 0)


async def inspect_migration_state_connection(connection: Any) -> MigrationState:
    """Inspect and validate migration history using SELECT statements only."""

    relation = await connection.fetchval(_HISTORY_EXISTS_SQL, _HISTORY_RELATION)
    if relation is None:
        return MigrationState(current_version=0)
    rows = await connection.fetch(_HISTORY_SQL)
    return validate_migration_history(list(rows))


async def inspect_migration_state(pool: asyncpg.Pool) -> MigrationState:
    """Inspect the global migration ledger without mutating database state."""

    try:
        return await inspect_migration_state_connection(pool)
    except MigrationError:
        raise
    except (asyncpg.PostgresError, OSError, TypeError, ValueError):
        raise MigrationError(
            "Migration history could not be verified. Ensure the database is reachable and the role can read "
            "bddk_meta.schema_migrations."
        ) from None


async def assert_migrations_current(pool: asyncpg.Pool) -> MigrationState:
    """Fail closed unless every migration shipped by this release is applied."""

    state = await inspect_migration_state(pool)
    if not state.current:
        raise MigrationNotReadyError(
            f"Database schema is at version {state.current_version}; version {LATEST_SCHEMA_VERSION} is required. "
            "Run `bddk-mcp migrate` with schema-owner credentials."
        )
    return state


async def _verify_prerequisites(connection: Any) -> None:
    rows = await connection.fetch(_EXTENSIONS_SQL, sorted(_REQUIRED_EXTENSIONS))
    installed = {str(_row_value(row, "extname", "")): str(_row_value(row, "extension_schema", "")) for row in rows}
    missing = sorted(_REQUIRED_EXTENSIONS - set(installed))
    if missing:
        raise MigrationPrerequisiteError(
            "Required PostgreSQL extensions are not installed: "
            + ", ".join(missing)
            + ". A database administrator must install them before migration."
        )
    misplaced = sorted(name for name, schema in installed.items() if schema != "public")
    if misplaced:
        raise MigrationPrerequisiteError(
            "Required PostgreSQL extensions must be installed in the public schema: " + ", ".join(misplaced) + "."
        )
    objects = await connection.fetchrow(_PUBLIC_EXTENSION_OBJECTS_SQL)
    if not bool(_row_value(objects, "has_unaccent", False)) or not bool(_row_value(objects, "has_vector", False)):
        raise MigrationPrerequisiteError(
            "Required PostgreSQL extension objects are unavailable in the public schema; migration refused."
        )


async def _require_retrieval_publication_backfill_approval(
    connection: Any,
    *,
    allow_retrieval_publication_backfill: bool,
) -> None:
    """Refuse the blocking v3 data migration on a populated v2 by default."""

    population = await connection.fetchrow(_RETRIEVAL_PUBLICATION_BACKFILL_SQL)
    populated = any(
        bool(_row_value(population, field, False)) for field in ("has_documents", "has_sections", "has_chunks")
    )
    if populated and not allow_retrieval_publication_backfill:
        raise MigrationScaleError(
            "Migration 3 found an existing regulatory corpus and was refused before its blocking retrieval-"
            "publication backfill. Stop serving and ingestion workloads, prove a restorable backup, and rehearse "
            "the upgrade against a size-matched restore. Then rerun with "
            "`--allow-retrieval-publication-backfill`. Do not use this flag for a clean database."
        )


async def migrate(
    pool: asyncpg.Pool,
    *,
    adopt_legacy: bool = False,
    allow_retrieval_publication_backfill: bool = False,
) -> MigrationState:
    """Apply all pending migrations atomically under one transaction lock.

    The application verifies, but never installs, the ``unaccent`` and
    ``vector`` extensions. Any error rolls back schema changes and ledger rows
    from this invocation together. ``adopt_legacy`` is deliberately false by
    default and accepts only the fully verified final pre-ledger v0001 shape;
    it is not a repair or best-effort migration mode. A populated database
    requires a separate, narrowly scoped acknowledgement before the blocking
    v3 retrieval-publication backfill can begin.
    """

    try:
        async with pool.acquire() as connection:
            # Refuse an untested backend before opening the migration
            # transaction, taking a lock, or executing any mutating statement.
            await assert_supported_postgresql(connection)
            async with connection.transaction():
                await connection.execute(f"SET LOCAL lock_timeout = '{MIGRATION_LOCK_TIMEOUT}'")
                await connection.execute(f"SET LOCAL statement_timeout = '{MIGRATION_STATEMENT_TIMEOUT}'")
                await connection.fetchval(
                    "SELECT pg_catalog.pg_advisory_xact_lock($1::pg_catalog.int8)",
                    _HISTORY_LOCK_KEY,
                )
                await _verify_prerequisites(connection)
                history_relation = await connection.fetchval(_HISTORY_EXISTS_SQL, _HISTORY_RELATION)

                pending_adoption: tuple[str, str, tuple[str, ...]] | None = None
                if adopt_legacy and history_relation is None:
                    # Inspection is deliberately complete and SELECT-only. The
                    # catalog is mutated only after it proves the single
                    # supported pre-ledger schema shape.
                    await inspect_legacy_v1(connection, allow_known_legacy=True)
                    await lock_legacy_v1_tables(connection)
                    # Repeat under table locks so the fingerprint, maximum IDs,
                    # and sequence state cannot race an uncooperative legacy
                    # writer that does not take the migration advisory lock.
                    legacy_before = await inspect_legacy_v1(connection, allow_known_legacy=True)
                    await normalize_legacy_v1(connection, legacy_before)
                    legacy_after = await inspect_legacy_v1(connection, allow_known_legacy=False)

                    await connection.execute(_CREATE_META_SCHEMA_SQL)
                    await connection.execute(_CREATE_HISTORY_SQL)
                    await connection.execute(
                        _INSERT_HISTORY_SQL,
                        V0001_CORE.version,
                        V0001_CORE.name,
                        V0001_CORE.checksum,
                    )
                    pending_adoption = (
                        legacy_before.fingerprint,
                        legacy_after.fingerprint,
                        legacy_before.normalizations,
                    )
                    state = MigrationState(current_version=V0001_CORE.version)
                else:
                    await connection.execute(_CREATE_META_SCHEMA_SQL)
                    await connection.execute(_CREATE_HISTORY_SQL)
                    state = await inspect_migration_state_connection(connection)

                for version in state.pending_versions:
                    migration = MIGRATIONS[version - 1]
                    if migration.version == V0003_RETRIEVAL_PUBLICATION.version:
                        await _require_retrieval_publication_backfill_approval(
                            connection,
                            allow_retrieval_publication_backfill=allow_retrieval_publication_backfill,
                        )
                    for statement in migration.statements:
                        await connection.execute(statement)
                    await connection.execute(
                        _INSERT_HISTORY_SQL,
                        migration.version,
                        migration.name,
                        migration.checksum,
                    )

                if pending_adoption is not None:
                    pre_fingerprint, post_fingerprint, normalizations = pending_adoption
                    await connection.execute(
                        _INSERT_ADOPTION_AUDIT_SQL,
                        LEGACY_SOURCE_KIND,
                        LEGACY_VERIFIER_VERSION,
                        V0001_CORE.checksum,
                        pre_fingerprint,
                        post_fingerprint,
                        list(normalizations),
                    )

                final_state = await inspect_migration_state_connection(connection)
                if not final_state.current:
                    raise MigrationHistoryError(
                        "Migration history did not reach the required version; changes rolled back."
                    )
                return final_state
    except PostgreSQLCompatibilityError as exc:
        raise MigrationCompatibilityError(str(exc)) from None
    except (MigrationError, LegacyAdoptionError):
        raise
    except asyncpg.LockNotAvailableError:
        raise MigrationLockTimeoutError(
            f"Database migration could not acquire a required lock within {MIGRATION_LOCK_TIMEOUT} and was rolled "
            "back. Stop serving, ingestion, and other schema activity before retrying the reviewed migration."
        ) from None
    except asyncpg.QueryCanceledError:
        raise MigrationStatementTimeoutError(
            f"A database migration statement was canceled or exceeded {MIGRATION_STATEMENT_TIMEOUT}; the migration "
            "was rolled back. Rehearse against a size-matched restore and review capacity before retrying; do not "
            "raise the timeout without measured evidence."
        ) from None
    except (asyncpg.PostgresError, OSError, TypeError, ValueError):
        raise MigrationError(
            "Database migration failed and was rolled back. Verify schema-owner DDL permissions, an empty managed "
            "schema for first installation, and the PostgreSQL vector/unaccent prerequisites."
        ) from None
