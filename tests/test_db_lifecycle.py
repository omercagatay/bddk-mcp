"""Tests for explicit DB migration and SELECT-only serving readiness."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from bddk_mcp.catalog_integrity import (
    _ACTIVE_CORPUS_RELEASE_DEPENDENCIES,
    _ACTIVE_CORPUS_RELEASE_REQUIRED_DEFINITION,
    _CANONICAL_FINGERPRINT_CONFIGURATION,
    _CITATION_VIEW_COLUMNS,
    _CITATION_VIEW_DEPENDENCIES,
    _CITATION_VIEW_REQUIRED_DEFINITION,
    _CORPUS_RELEASE_CONSTRAINTS,
    _CORPUS_RELEASE_RELATIONS,
    _CORPUS_RELEASE_ROUTINES,
    _CORPUS_RELEASE_TRIGGERS,
    _EXPECTED_CONSTRAINTS,
    _EXPECTED_INDEXES,
    _EXPECTED_ROUTINES,
    _EXPECTED_TRIGGERS,
    _EXPECTED_V4_CONSTRAINT_CATALOG_SHA256,
    _EXPECTED_V4_CONSTRAINT_COUNT,
    _EXPECTED_V4_INDEX_CATALOG_SHA256,
    _EXPECTED_V4_INDEX_COUNT,
    _EXPECTED_V7_CATALOG_OBJECT_COUNT,
    _EXPECTED_V7_CATALOG_SHA256,
    _EXPECTED_V8_DEPLOYED_ROUTINE_ACL,
    _LEGAL_STATUS_RESULT_TYPE,
    _V8_CORPUS_RELEASE_CONSTRAINTS,
    _V8_CORPUS_RELEASE_RELATIONS,
    _V8_CORPUS_RELEASE_ROUTINES,
    _V8_CORPUS_RELEASE_TRIGGERS,
    _v6_legal_status_function_source,
)
from bddk_mcp.core.exceptions import BddkStorageError
from bddk_mcp.db_lifecycle import (
    _REQUIRED_COLUMNS,
    DatabaseLifecycleError,
    DatabaseNotReadyError,
    DatabaseReadiness,
    SchemaOwnerIdentityError,
    assert_database_ready,
    assert_schema_owner_identity,
    inspect_database_readiness,
    migrate_database,
)
from bddk_mcp.ingest.client import BddkApiClient
from bddk_mcp.migrations import MIGRATIONS, LegacyAdoptionError, MigrationScaleError


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


def test_readiness_requires_every_canonical_legal_version_column() -> None:
    expected = {
        "regulatory_instruments": {
            "instrument_id",
            "jurisdiction",
            "authority_code",
            "identity_key",
            "canonical_title",
            "instrument_type",
            "created_at",
        },
        "regulatory_family_imports": {
            "bundle_id",
            "bundle_sha256",
            "instrument_id",
            "schema_version",
            "fixture_only",
            "imported_by",
            "imported_current_user",
            "imported_session_user",
            "predecessor_bundle_sha256",
            "member_manifest",
            "imported_at",
        },
        "regulatory_source_blobs": {"blob_id", "content_sha256"},
        "regulatory_source_artifacts": {
            "artifact_id",
            "blob_id",
            "canonical_uri",
            "source_authority",
            "media_type",
            "retrieved_at",
            "repository_document_id",
            "fixture_only",
        },
        "regulatory_evidence": {
            "evidence_id",
            "artifact_id",
            "locator",
            "statement_sha256",
            "authority_level",
        },
        "regulatory_legal_versions": {
            "legal_version_id",
            "instrument_id",
            "version_key",
            "legal_text_sha256",
            "predecessor_version_id",
            "consolidation_state",
            "validation_state",
            "validated_by",
            "validated_at",
            "validation_method",
            "review_record_sha256",
            "created_at",
        },
        "regulatory_legal_version_artifacts": {"legal_version_id", "artifact_id", "source_role"},
        "regulatory_legal_events": {
            "event_id",
            "legal_version_id",
            "event_type",
            "event_date",
            "evidence_id",
            "target_legal_version_id",
            "validation_state",
            "validated_by",
            "validated_at",
            "validation_method",
            "review_record_sha256",
        },
        "regulatory_legal_status_assertions": {
            "assertion_id",
            "legal_version_id",
            "legal_status",
            "valid_from",
            "valid_through",
            "evidence_id",
            "validation_state",
            "validated_by",
            "validated_at",
            "validation_method",
            "review_record_sha256",
        },
        "regulatory_provisions": {"provision_id", "instrument_id", "provision_kind", "canonical_path"},
        "regulatory_legal_version_provisions": {
            "legal_version_id",
            "provision_id",
            "provision_text_sha256",
            "document_section_id",
            "evidence_id",
            "validation_state",
            "validated_by",
            "validated_at",
            "validation_method",
            "review_record_sha256",
        },
        "regulatory_validated_section_citations": {
            "document_section_id",
            "source_document_id",
            "normalized_document_sha256",
            "normalized_section_sha256",
            "instrument_id",
            "instrument_jurisdiction",
            "instrument_authority_code",
            "instrument_identity_key",
            "legal_version_id",
            "legal_version_key",
            "legal_text_sha256",
            "review_record_sha256",
            "provision_review_record_sha256",
            "artifact_id",
            "artifact_blob_id",
            "artifact_sha256",
            "source_url",
            "artifact_retrieved_at",
            "evidence_id",
            "evidence_locator",
            "evidence_statement_sha256",
            "provision_id",
            "provision_kind",
            "provision_path",
            "provision_text_sha256",
        },
    }

    assert {name: set(_REQUIRED_COLUMNS[name]) for name in expected} == expected


class ReadOnlyReadinessPool:
    """Catalog-shaped pool that rejects every mutating SQL API."""

    def __init__(
        self,
        *,
        extensions: set[str] | None = None,
        relations: set[str] | None = None,
        columns: dict[str, set[str]] | None = None,
        corpus: dict[str, bool] | None = None,
        server_version_num: int = 170000,
    ) -> None:
        self.extensions = extensions if extensions is not None else {"unaccent", "vector"}
        self.relations = relations if relations is not None else set(_REQUIRED_COLUMNS)
        self.columns = (
            columns
            if columns is not None
            else {table_name: set(required) for table_name, required in _REQUIRED_COLUMNS.items()}
        )
        self.corpus = corpus if corpus is not None else _ready_corpus()
        self.server_version_num = server_version_num
        self.statements: list[str] = []

    def _record_select(self, query: str) -> None:
        normalized = query.strip()
        assert normalized.upper().startswith(("SELECT", "WITH")), normalized
        self.statements.append(normalized)

    async def fetch(self, query: str, *args):
        self._record_select(query)
        if "SELECT version, name, checksum" in query:
            return [{"version": item.version, "name": item.name, "checksum": item.checksum} for item in MIGRATIONS]
        if "pg_extension" in query:
            return [{"extname": name} for name in sorted(self.extensions)]
        if "resolved_relation" in query:
            return [
                {
                    "relation_name": table_name,
                    "resolved_relation": table_name if table_name in self.relations else None,
                }
                for table_name in args[0]
            ]
        if "ledger_owner.rolname AS ledger_owner_name" in query and "relation.relkind" in query:
            release_relations = {**_CORPUS_RELEASE_RELATIONS, **_V8_CORPUS_RELEASE_RELATIONS}
            return [
                {
                    "relname": name,
                    "relkind": relation_kind,
                    "owner_name": "bddk_schema_owner",
                    "ledger_owner_name": "bddk_schema_owner",
                    "options": options,
                    "columns": columns,
                }
                for name, (relation_kind, columns, options) in release_relations.items()
            ]
        if "pg_attribute" in query:
            return [
                {"table_name": table_name, "column_name": column_name}
                for table_name in args[0]
                if table_name in self.relations
                for column_name in sorted(self.columns.get(table_name, set()))
            ]
        if "pg_constraint" in query:
            if "namespace.nspname = 'bddk_meta'" in query:
                release_constraints = {**_CORPUS_RELEASE_CONSTRAINTS, **_V8_CORPUS_RELEASE_CONSTRAINTS}
                return [
                    {
                        "relname": table,
                        "conname": name,
                        "contype": constraint_type,
                        "convalidated": True,
                        "definition": definition,
                    }
                    for (table, name), (constraint_type, definition) in release_constraints.items()
                ]
            return [
                {
                    "table_name": table,
                    "conname": name,
                    "contype": constraint_type,
                    "convalidated": True,
                    "definition": definition,
                }
                for (table, name), (constraint_type, definition) in _EXPECTED_CONSTRAINTS.items()
            ]
        if "pg_trigger" in query:
            if "namespace.nspname = 'bddk_meta'" in query:
                release_triggers = {**_CORPUS_RELEASE_TRIGGERS, **_V8_CORPUS_RELEASE_TRIGGERS}
                return [
                    {
                        "relname": table,
                        "tgname": name,
                        "tgenabled": "O",
                        "tgtype": trigger_type,
                        "function_identity": function_identity,
                    }
                    for (table, name), (function_identity, trigger_type) in release_triggers.items()
                ]
            return [
                {
                    "table_name": table,
                    "tgname": name,
                    "tgenabled": "O",
                    "tgtype": trigger_type,
                    "tgoldtable": old_table,
                    "tgnewtable": new_table,
                    "function_identity": function_identity,
                }
                for (table, name), (
                    function_identity,
                    trigger_type,
                    old_table,
                    new_table,
                ) in _EXPECTED_TRIGGERS.items()
            ]
        if "pg_index" in query:
            return [
                {
                    "index_name": name,
                    "method": method,
                    "indisunique": False,
                    "indisprimary": False,
                    "indisvalid": True,
                    "indisready": True,
                    "keys": keys,
                    "opclasses": opclasses,
                    "options": options,
                }
                for name, (method, keys, opclasses, options) in _EXPECTED_INDEXES.items()
            ]
        if "pg_proc" in query:
            if "object_identity" in query and "acl_items" in query:
                return [
                    {
                        "object_identity": object_identity,
                        "grantee_name": grantee_name,
                        "privilege_type": privilege_type,
                        "is_grantable": is_grantable,
                    }
                    for object_identity, grantee_name, privilege_type, is_grantable in (
                        _EXPECTED_V8_DEPLOYED_ROUTINE_ACL
                    )
                ]
            if "namespace.nspname = 'bddk_meta'" in query:
                release_routines = {**_CORPUS_RELEASE_ROUTINES, **_V8_CORPUS_RELEASE_ROUTINES}
                return [
                    {
                        "function_identity": identity,
                        "language": language,
                        "provolatile": volatility,
                        "proparallel": parallel,
                        "prosecdef": security_definer,
                        "proleakproof": False,
                        "configuration": (
                            list(_CANONICAL_FINGERPRINT_CONFIGURATION)
                            if identity == "current_corpus_state_sha256(text)"
                            else ["search_path=pg_catalog"]
                        ),
                        "source": source,
                        "owner_name": "bddk_schema_owner",
                        "ledger_owner_name": "bddk_schema_owner",
                        "public_can_execute": False,
                    }
                    for identity, (
                        language,
                        volatility,
                        parallel,
                        security_definer,
                        source,
                    ) in release_routines.items()
                ]
            return [
                {
                    "function_identity": identity,
                    "language": language,
                    "provolatile": volatility,
                    "proparallel": parallel,
                    "prosecdef": False,
                    "proleakproof": False,
                    "configuration": ["search_path=pg_catalog, public"],
                    "source": source,
                }
                for identity, (language, volatility, parallel, source) in _EXPECTED_ROUTINES.items()
            ]
        raise AssertionError(f"unexpected readiness query: {query}")

    async def fetchval(self, query: str, *args):
        self._record_select(query)
        if "server_version_num" in query:
            return self.server_version_num
        if "to_regclass" in query:
            return "bddk_meta.schema_migrations"
        raise AssertionError(f"unexpected readiness query: {query}")

    async def fetchrow(self, query: str, *args):
        self._record_select(query)
        if "resolve_regulation_status" in query and "pg_proc" in query:
            return {
                "function_identity": "resolve_regulation_status(text, date)",
                "language": "sql",
                "provolatile": "s",
                "proparallel": "s",
                "prosecdef": True,
                "proleakproof": False,
                "proisstrict": True,
                "proretset": True,
                "result_type": _LEGAL_STATUS_RESULT_TYPE,
                "configuration": ["search_path=pg_catalog"],
                "source": _v6_legal_status_function_source(),
                "owner_name": "bddk_schema_owner",
                "ledger_owner_name": "bddk_schema_owner",
                "public_can_execute": False,
            }
        if "v4_constraint_catalog_sha256" in query:
            return {
                "object_count": _EXPECTED_V4_CONSTRAINT_COUNT,
                "v4_constraint_catalog_sha256": _EXPECTED_V4_CONSTRAINT_CATALOG_SHA256,
            }
        if "v4_index_catalog_sha256" in query:
            return {
                "object_count": _EXPECTED_V4_INDEX_COUNT,
                "v4_index_catalog_sha256": _EXPECTED_V4_INDEX_CATALOG_SHA256,
            }
        if "AS catalog_sha256" in query and "bddk_retained" in query:
            return {
                "object_count": _EXPECTED_V7_CATALOG_OBJECT_COUNT,
                "catalog_sha256": _EXPECTED_V7_CATALOG_SHA256,
            }
        if "pg_get_viewdef" in query:
            if "active_corpus_release" in query:
                return {
                    "definition": " ".join(_ACTIVE_CORPUS_RELEASE_REQUIRED_DEFINITION),
                    "dependencies": list(_ACTIVE_CORPUS_RELEASE_DEPENDENCIES),
                }
            return {
                "relkind": "v",
                "owner_name": "bddk_schema_owner",
                "ledger_owner_name": "bddk_schema_owner",
                "options": ["security_barrier=true", "security_invoker=false"],
                "definition": " ".join(_CITATION_VIEW_REQUIRED_DEFINITION),
                "columns": list(_CITATION_VIEW_COLUMNS),
                "dependencies": list(_CITATION_VIEW_DEPENDENCIES),
            }
        if "FROM bddk_meta.active_corpus_release" in query:
            return None
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
    assert len(pool.statements) == 24
    assert all(statement.upper().startswith(("SELECT", "WITH")) for statement in pool.statements)


@pytest.mark.asyncio
async def test_schema_only_readiness_skips_all_corpus_queries():
    pool = ReadOnlyReadinessPool(corpus=_ready_corpus(has_documents=False))

    report = await inspect_database_readiness(pool, require_corpus=False)  # type: ignore[arg-type]

    assert report.ready
    assert len(pool.statements) == 22


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
    assert len(pool.statements) == 6


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
async def test_readiness_refuses_unsupported_postgresql_before_catalog_or_corpus_inspection():
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
        patch("bddk_mcp.db_lifecycle.assert_schema_owner_identity", new=AsyncMock()) as owner_check,
        patch("bddk_mcp.db_lifecycle.apply_migrations", new=AsyncMock(side_effect=record_migrations)) as migrate,
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=AsyncMock(return_value=ready)) as validate,
    ):
        result = await migrate_database(pool=pool, expected_database="bddk_test")

    assert result is ready
    assert events == ["migrations"]
    owner_check.assert_awaited_once_with(pool, "bddk_test")
    migrate.assert_awaited_once_with(pool)
    validate.assert_awaited_once_with(pool=pool, require_corpus=False)


@pytest.mark.asyncio
async def test_migrate_forwards_explicit_legacy_adoption_only_when_requested():
    pool = MagicMock()
    ready = DatabaseReadiness()

    with (
        patch("bddk_mcp.db_lifecycle.assert_schema_owner_identity", new=AsyncMock()),
        patch("bddk_mcp.db_lifecycle.apply_migrations", new=AsyncMock()) as migrate,
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=AsyncMock(return_value=ready)),
    ):
        result = await migrate_database(pool=pool, adopt_legacy=True, expected_database="bddk_test")

    assert result is ready
    migrate.assert_awaited_once_with(pool, adopt_legacy=True)


@pytest.mark.asyncio
async def test_migrate_forwards_retrieval_backfill_approval_only_when_explicit():
    pool = MagicMock()
    ready = DatabaseReadiness()

    with (
        patch("bddk_mcp.db_lifecycle.assert_schema_owner_identity", new=AsyncMock()),
        patch("bddk_mcp.db_lifecycle.apply_migrations", new=AsyncMock()) as migrate,
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=AsyncMock(return_value=ready)),
    ):
        result = await migrate_database(
            pool=pool,
            allow_retrieval_publication_backfill=True,
            expected_database="bddk_test",
        )

    assert result is ready
    migrate.assert_awaited_once_with(pool, allow_retrieval_publication_backfill=True)


@pytest.mark.asyncio
async def test_migrate_preserves_bounded_actionable_adoption_refusal():
    pool = MagicMock()
    refusal = LegacyAdoptionError(
        "Legacy adoption refused. No corpus rows were changed. Follow the blue-green runbook."
    )

    with (
        patch("bddk_mcp.db_lifecycle.assert_schema_owner_identity", new=AsyncMock()),
        patch("bddk_mcp.db_lifecycle.apply_migrations", new=AsyncMock(side_effect=refusal)),
    ):
        with pytest.raises(DatabaseLifecycleError, match="No corpus rows were changed") as exc_info:
            await migrate_database(pool=pool, adopt_legacy=True, expected_database="bddk_test")

    assert exc_info.value.__cause__ is None


@pytest.mark.asyncio
async def test_migrate_preserves_actionable_sanitized_scale_refusal():
    pool = MagicMock()
    refusal = MigrationScaleError(
        "Migration refused before its blocking backfill. Use --allow-retrieval-publication-backfill after rehearsal."
    )

    with (
        patch("bddk_mcp.db_lifecycle.assert_schema_owner_identity", new=AsyncMock()),
        patch("bddk_mcp.db_lifecycle.apply_migrations", new=AsyncMock(side_effect=refusal)),
    ):
        with pytest.raises(DatabaseLifecycleError, match="after rehearsal") as exc_info:
            await migrate_database(pool=pool, expected_database="bddk_test")

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
        patch("bddk_mcp.db_lifecycle.require_expected_database_name", return_value="bddk") as expected_name,
        patch("bddk_mcp.db_lifecycle.assert_schema_owner_identity", new=AsyncMock()) as owner_check,
        patch("bddk_mcp.db_lifecycle.apply_migrations", new=AsyncMock()),
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=AsyncMock(return_value=ready)),
    ):
        result = await migrate_database()

    assert result is ready
    require_url.assert_called_once_with("schema-owner")
    expected_name.assert_called_once_with()
    owner_check.assert_awaited_once_with(pool, "bddk")
    create_pool.assert_awaited_once_with("postgresql://schema-owner", min_size=1, max_size=3)
    pool.close.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_readiness_queries_execute_against_postgresql(pg_pool):
    """Validate the catalog and corpus SELECT syntax against real PostgreSQL."""
    schema_report = await inspect_database_readiness(pg_pool, require_corpus=False)
    corpus_report = await inspect_database_readiness(pg_pool, require_corpus=True)

    assert schema_report.ready
    assert isinstance(corpus_report, DatabaseReadiness)


def _valid_schema_owner_identity(**overrides):
    values = {
        "database_name": "bddk",
        "current_user_name": "bddk_schema_owner",
        "session_user_name": "bddk_migrator",
        "current_role_is_restricted_owner": True,
        "session_role_is_restricted_login": True,
        "direct_memberships": ["bddk_schema_owner"],
        "membership_admin": False,
        "unsafe_inherited_role": False,
        "session_owns_database": False,
        "owner_can_connect": True,
        "owner_can_create": True,
        "owner_can_create_temporary": False,
        "owner_owns_public_schema": True,
        "owner_has_public_usage": True,
        "owner_has_public_create": True,
    }
    values.update(overrides)
    return values


@pytest.mark.asyncio
async def test_schema_owner_identity_accepts_exact_restricted_boundary():
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=170000)
    pool.fetchrow = AsyncMock(return_value=_valid_schema_owner_identity())

    await assert_schema_owner_identity(pool, "bddk")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("override", "value"),
    [
        ("database_name", "wrong_database"),
        ("current_user_name", "bddk_migrator"),
        ("session_role_is_restricted_login", False),
        ("direct_memberships", ["bddk_schema_owner", "unrelated_role"]),
        ("membership_admin", True),
        ("unsafe_inherited_role", True),
        ("session_owns_database", True),
        ("owner_can_create_temporary", True),
        ("owner_owns_public_schema", False),
    ],
)
async def test_schema_owner_identity_rejects_wrong_target_or_privilege(override, value):
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=170000)
    pool.fetchrow = AsyncMock(return_value=_valid_schema_owner_identity(**{override: value}))

    with pytest.raises(SchemaOwnerIdentityError, match="exact schema-owner contract"):
        await assert_schema_owner_identity(pool, "bddk")


@pytest.mark.asyncio
async def test_schema_owner_identity_refuses_unsupported_postgresql_before_role_inspection():
    pool = MagicMock()
    pool.fetchval = AsyncMock(return_value=160012)
    pool.fetchrow = AsyncMock(return_value=_valid_schema_owner_identity())

    with pytest.raises(SchemaOwnerIdentityError) as exc_info:
        await assert_schema_owner_identity(pool, "bddk")

    assert "requires PostgreSQL 17" in str(exc_info.value)
    assert "160012" not in str(exc_info.value)
    pool.fetchrow.assert_not_awaited()


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
