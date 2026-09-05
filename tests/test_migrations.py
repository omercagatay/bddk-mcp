"""Tests for the immutable global PostgreSQL migration framework."""

from __future__ import annotations

import asyncio
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest

from bddk_mcp import db_identity
from bddk_mcp.catalog_integrity import inspect_catalog_integrity
from bddk_mcp.corpus_coordination import (
    CORPUS_MUTATION_ADVISORY_KEY,
    SCHEMA_MIGRATION_ADVISORY_KEY,
)
from bddk_mcp.corpus_publication import assert_release_publication_ready, publish_strict_corpus_release
from bddk_mcp.db_identity import (
    assert_release_publication_connection_identity,
    assert_release_publication_identity,
)
from bddk_mcp.db_lifecycle import inspect_database_readiness
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
from bddk_mcp.migrations.v0005_corpus_release_publication import (
    CORPUS_EPOCH_TRACKED_TABLES,
    V0005_CORPUS_RELEASE_PUBLICATION,
)
from bddk_mcp.migrations.v0007_retained_corpus_generations import (
    NONCANONICAL_FINGERPRINT_UPGRADE_SQLSTATE,
    RETAINED_CORPUS_RELATIONS,
    V0007_RETAINED_CORPUS_GENERATIONS,
)
from bddk_mcp.migrations.v0008_staged_corpus_releases import (
    MAX_RELEASE_REQUEST_TTL_SECONDS,
    MIN_RELEASE_REQUEST_TTL_SECONDS,
    RELEASE_REQUEST_ID_PREFIX,
    V0008_STAGED_CORPUS_RELEASES,
)
from bddk_mcp.migrations.v0010_corpus_release_freshness_policy import (
    MEASURED_FRESHNESS_POLICY_RESULT,
    UNMEASURED_FRESHNESS_POLICY_RESULT,
)
from tests.test_corpus_publication import (
    _PROFILE_SHA256,
    _ensure_release_publisher_role,
    _insert_canonical_legal_state,
    _insert_ready_corpus,
    _publish,
    _rollback_savepoint,
)


def _history_rows(*, migrations=MIGRATIONS):
    return [{"version": item.version, "name": item.name, "checksum": item.checksum} for item in migrations]


async def _ensure_v8_release_roles(connection: asyncpg.Connection) -> None:
    await _ensure_release_publisher_role(connection)
    await connection.execute(
        """
        DO $roles$
        BEGIN
            IF pg_catalog.to_regrole('bddk_release_verifier') IS NULL THEN
                CREATE ROLE bddk_release_verifier NOLOGIN NOSUPERUSER NOCREATEDB
                    NOCREATEROLE NOREPLICATION NOBYPASSRLS;
            END IF;
        END
        $roles$
        """
    )
    await connection.execute("REVOKE bddk_release_publisher FROM bddk_release_verifier")
    await connection.execute("REVOKE bddk_release_verifier FROM bddk_release_publisher")
    await connection.execute("GRANT USAGE ON SCHEMA bddk_meta TO bddk_release_verifier, bddk_release_publisher")
    await connection.execute(
        "GRANT EXECUTE ON FUNCTION bddk_meta.stage_verified_corpus_release("
        "pg_catalog.text, pg_catalog.text, pg_catalog.text, pg_catalog.text, pg_catalog.text, "
        "pg_catalog.text, pg_catalog.int4, pg_catalog.int4, pg_catalog.int4, pg_catalog.text, "
        "pg_catalog.text, pg_catalog.text, pg_catalog.int4) TO bddk_release_verifier"
    )
    await connection.execute(
        "GRANT EXECUTE ON FUNCTION bddk_meta.activate_staged_corpus_release(pg_catalog.text) TO bddk_release_publisher"
    )


@asynccontextmanager
async def _session_authorization(connection: asyncpg.Connection, role_name: str):
    if role_name not in {
        "bddk_legacy_direct_grantee",
        "bddk_release_verifier",
        "bddk_release_publisher",
        "bddk_v8_dual_role",
    }:
        raise AssertionError("unexpected test authorization role")
    await connection.execute(f"SET SESSION AUTHORIZATION {role_name}")
    try:
        yield
    finally:
        await connection.execute("RESET SESSION AUTHORIZATION")


async def _stage_v8_release(
    connection: asyncpg.Connection,
    *,
    manifest_id: str = "staged-release-001",
    verification_evidence_sha256: str = "5" * 64,
    retrieval_profile_sha256: str = _PROFILE_SHA256,
    freshness_policy_result: str = MEASURED_FRESHNESS_POLICY_RESULT,
):
    return await connection.fetchrow(
        """
        SELECT *
        FROM bddk_meta.stage_verified_corpus_release(
            $1, $2, $3, $4, $5, $6, 60, 120, 3600, $7, $8, $9, 900
        )
        """,
        manifest_id,
        "1" * 64,
        "2" * 64,
        "3" * 64,
        verification_evidence_sha256,
        freshness_policy_result,
        retrieval_profile_sha256,
        "6" * 64,
        "sha256:" + "7" * 64,
    )


def test_registry_is_sequential_named_and_sha256_versioned():
    assert [item.version for item in MIGRATIONS] == list(range(1, LATEST_SCHEMA_VERSION + 1))
    assert len({item.name for item in MIGRATIONS}) == len(MIGRATIONS)
    assert all(len(item.checksum) == 64 for item in MIGRATIONS)
    assert all(item.checksum == replace(item).checksum for item in MIGRATIONS)
    assert replace(MIGRATIONS[0], name="changed_name").checksum != MIGRATIONS[0].checksum


def test_v5_epoch_and_set_based_chunk_invalidation_contract_is_complete() -> None:
    statements = tuple(" ".join(statement.split()) for statement in V0005_CORPUS_RELEASE_PUBLICATION.statements)
    ddl = "\n".join(statements)

    assert len(CORPUS_EPOCH_TRACKED_TABLES) == 17
    for table_name in CORPUS_EPOCH_TRACKED_TABLES:
        assert (
            "CREATE TRIGGER bump_corpus_state_epoch_on_change "
            f"AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON public.{table_name} "
            "FOR EACH STATEMENT EXECUTE FUNCTION bddk_meta.bump_corpus_state_epoch()"
        ) in ddl

    assert "REFERENCING NEW TABLE AS changed_chunks" in ddl
    assert "REFERENCING OLD TABLE AS changed_chunks" in ddl
    assert "REFERENCING OLD TABLE AS old_chunks NEW TABLE AS new_chunks" in ddl
    assert "FOR EACH ROW EXECUTE FUNCTION public.invalidate_retrieval_publication()" not in ddl

    publisher = next(statement for statement in statements if "publish_verified_corpus_release" in statement)
    assert f"pg_advisory_xact_lock( {CORPUS_MUTATION_ADVISORY_KEY}::pg_catalog.int8 )" in publisher
    assert publisher.index("pg_advisory_xact_lock") < publisher.index("LOCK TABLE")

    active_view = next(
        statement for statement in statements if statement.startswith("CREATE VIEW bddk_meta.active_corpus_release")
    )
    assert "activation.corpus_epoch = epoch.epoch" in active_view
    assert "current_corpus_state_sha256" not in active_view
    assert "corpus_retrieval_ready" not in active_view


def test_v8_separates_verification_claims_from_request_only_activation() -> None:
    statements = tuple(" ".join(statement.split()) for statement in V0008_STAGED_CORPUS_RELEASES.statements)
    ddl = "\n".join(statements)
    stage = next(
        statement for statement in statements if statement.startswith("CREATE FUNCTION bddk_meta.stage_verified")
    )
    compact_stage = stage.replace("( ", "(")
    activate = next(
        statement for statement in statements if statement.startswith("CREATE FUNCTION bddk_meta.activate_staged")
    )

    assert "CREATE TABLE bddk_meta.corpus_release_requests" in ddl
    assert "CREATE TABLE bddk_meta.corpus_release_request_activations" in ddl
    assert "reject_corpus_release_request_update_delete" in ddl
    assert "reject_corpus_release_request_activation_update_delete" in ddl
    assert f"BETWEEN {MIN_RELEASE_REQUEST_TTL_SECONDS} AND {MAX_RELEASE_REQUEST_TTL_SECONDS}" in stage
    assert "pg_has_role( SESSION_USER, pg_catalog.to_regrole('bddk_release_verifier'), 'MEMBER' )" in stage
    assert stage.index("pg_advisory_xact_lock") < stage.index("LOCK TABLE")
    for material_claim in (
        "selected_release_id",
        "requested_manifest_id",
        "requested_manifest_sha256",
        "requested_signature_sha256",
        "requested_signer_key_sha256",
        "requested_verification_evidence_sha256",
        "requested_source_detection_slo_seconds",
        "requested_publication_slo_seconds",
        "requested_max_manifest_age_seconds",
        "requested_retrieval_profile_sha256",
        "selected_state_sha256",
        "selected_corpus_epoch",
        "requested_verifier_revision_sha256",
        "requested_verifier_image_digest",
        "requested_valid_for_seconds",
        "selected_actor_fingerprint",
    ):
        assert f"corpus_fingerprint_frame({material_claim}" in compact_stage
    assert "ON CONFLICT ON CONSTRAINT corpus_release_requests_pkey DO NOTHING" in stage

    assert "requested_request_id pg_catalog.text" in activate
    assert "requested_manifest" not in activate.split("RETURNS TABLE", 1)[0]
    assert "pg_has_role( SESSION_USER, pg_catalog.to_regrole('bddk_release_publisher'), 'MEMBER' )" in activate
    assert "verification_expires_at" in activate
    assert "selected_live_epoch IS DISTINCT FROM selected_request.corpus_epoch" in activate
    assert "selected_live_state_sha256 IS DISTINCT FROM selected_request.corpus_state_sha256" in activate
    assert "staged corpus release request was already activated" in activate
    assert "INSERT INTO bddk_meta.corpus_release_request_activations" in activate
    assert "pg_catalog.aclexplode" in ddl
    assert "acl.grantee <> routine.proowner" in ddl
    assert "REVOKE EXECUTE ON FUNCTION %s FROM %I CASCADE" in ddl


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_v8_staged_release_is_duplicate_suppressed_short_lived_and_single_use(pg_pool) -> None:
    connection = await pg_pool.acquire()
    transaction = connection.transaction()
    await transaction.start()
    try:
        await _ensure_v8_release_roles(connection)
        content_hash = await _insert_ready_corpus(connection, "v8-staged-release")
        await _insert_canonical_legal_state(
            connection,
            document_id="v8-staged-release",
            content_hash=content_hash,
        )

        async with _session_authorization(connection, "bddk_release_verifier"):
            staged = await _stage_v8_release(connection)
            duplicate = await _stage_v8_release(connection)
        assert staged is not None and duplicate is not None
        assert staged["request_id"] == duplicate["request_id"]
        assert staged["release_id"] == duplicate["release_id"]
        assert staged["staged_at"] == duplicate["staged_at"]
        assert staged["verification_expires_at"] == duplicate["verification_expires_at"]
        assert str(staged["request_id"]).startswith(RELEASE_REQUEST_ID_PREFIX)
        assert await connection.fetchval("SELECT count(*) FROM bddk_meta.corpus_release_requests") == 1

        async with _session_authorization(connection, "bddk_release_publisher"):
            activated = await connection.fetchrow(
                "SELECT * FROM bddk_meta.activate_staged_corpus_release($1)",
                staged["request_id"],
            )
        assert activated is not None
        assert activated["request_id"] == staged["request_id"]
        assert activated["release_id"] == staged["release_id"]
        assert (
            await connection.fetchval(
                "SELECT count(*) FROM bddk_meta.corpus_release_request_activations WHERE request_id = $1",
                staged["request_id"],
            )
            == 1
        )
        assert (
            await connection.fetchval("SELECT release_id FROM bddk_meta.active_corpus_release") == staged["release_id"]
        )

        async with _session_authorization(connection, "bddk_release_publisher"):
            async with _rollback_savepoint(connection):
                with pytest.raises(asyncpg.ObjectNotInPrerequisiteStateError, match="already activated"):
                    await connection.fetchrow(
                        "SELECT * FROM bddk_meta.activate_staged_corpus_release($1)",
                        staged["request_id"],
                    )

        async with _rollback_savepoint(connection):
            with pytest.raises(asyncpg.ObjectNotInPrerequisiteStateError, match="append-only"):
                await connection.execute(
                    "UPDATE bddk_meta.corpus_release_requests SET valid_for_seconds = 60 WHERE request_id = $1",
                    staged["request_id"],
                )

        async with _session_authorization(connection, "bddk_release_verifier"):
            expiring = await _stage_v8_release(
                connection,
                manifest_id="staged-release-expired",
                verification_evidence_sha256="8" * 64,
            )
        assert expiring is not None
        await connection.execute(
            "ALTER TABLE bddk_meta.corpus_release_requests DISABLE TRIGGER reject_corpus_release_request_update_delete"
        )
        await connection.execute(
            """
            WITH expired AS (
                SELECT pg_catalog.clock_timestamp() - pg_catalog.interval '2 hours' AS staged_at
            )
            UPDATE bddk_meta.corpus_release_requests AS request
            SET staged_at = expired.staged_at,
                verification_expires_at = expired.staged_at
                    + request.valid_for_seconds * pg_catalog.interval '1 second'
            FROM expired
            WHERE request.request_id = $1
            """,
            expiring["request_id"],
        )
        await connection.execute(
            "ALTER TABLE bddk_meta.corpus_release_requests ENABLE TRIGGER reject_corpus_release_request_update_delete"
        )
        async with _session_authorization(connection, "bddk_release_publisher"):
            async with _rollback_savepoint(connection):
                with pytest.raises(asyncpg.ObjectNotInPrerequisiteStateError, match="expired"):
                    await connection.fetchrow(
                        "SELECT * FROM bddk_meta.activate_staged_corpus_release($1)",
                        expiring["request_id"],
                    )

        async with _session_authorization(connection, "bddk_release_verifier"):
            changed = await _stage_v8_release(
                connection,
                manifest_id="staged-release-changed",
                verification_evidence_sha256="9" * 64,
            )
        assert changed is not None
        async with _rollback_savepoint(connection):
            await connection.execute(
                "UPDATE public.decision_cache SET cached_at = cached_at + 1 WHERE document_id = $1",
                "v8-staged-release",
            )
            async with _session_authorization(connection, "bddk_release_publisher"):
                async with _rollback_savepoint(connection):
                    with pytest.raises(asyncpg.ObjectNotInPrerequisiteStateError, match="epoch has changed"):
                        await connection.fetchrow(
                            "SELECT * FROM bddk_meta.activate_staged_corpus_release($1)",
                            changed["request_id"],
                        )

        async with _session_authorization(connection, "bddk_release_verifier"):
            async with _rollback_savepoint(connection):
                with pytest.raises(asyncpg.ObjectNotInPrerequisiteStateError, match="not retrieval-ready"):
                    await _stage_v8_release(
                        connection,
                        manifest_id="staged-release-wrong-profile",
                        verification_evidence_sha256="a" * 64,
                        retrieval_profile_sha256="f" * 64,
                    )

        await connection.execute("CREATE ROLE bddk_v8_dual_role NOLOGIN")
        await connection.execute("GRANT bddk_release_verifier, bddk_release_publisher TO bddk_v8_dual_role")
        async with _session_authorization(connection, "bddk_v8_dual_role"):
            async with _rollback_savepoint(connection):
                with pytest.raises(asyncpg.InsufficientPrivilegeError, match="not authorized"):
                    await _stage_v8_release(
                        connection,
                        manifest_id="staged-release-dual-role",
                        verification_evidence_sha256="b" * 64,
                    )

        assert not await connection.fetchval(
            "SELECT pg_catalog.has_function_privilege("
            "'bddk_release_publisher', "
            "'bddk_meta.publish_verified_corpus_release(text,text,text,integer,integer,integer,text)', "
            "'EXECUTE')"
        )
    finally:
        await transaction.rollback()
        await pg_pool.release(connection)


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_v8_migration_retires_every_nonowner_legacy_publication_grant(pg_pool) -> None:
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await _downgrade_current_schema_to_v7(connection)
            await connection.execute("CREATE ROLE bddk_legacy_direct_grantee NOLOGIN")
            await connection.execute("CREATE ROLE bddk_legacy_downstream_grantee NOLOGIN")
            await connection.execute("GRANT USAGE ON SCHEMA bddk_meta TO bddk_legacy_direct_grantee")
            legacy_identity = "bddk_meta.publish_verified_corpus_release(text,text,text,integer,integer,integer,text)"
            await connection.execute(
                "GRANT EXECUTE ON FUNCTION " + legacy_identity + " TO bddk_legacy_direct_grantee WITH GRANT OPTION"
            )
            async with _session_authorization(connection, "bddk_legacy_direct_grantee"):
                await connection.execute(
                    "GRANT EXECUTE ON FUNCTION " + legacy_identity + " TO bddk_legacy_downstream_grantee"
                )
            assert await connection.fetchval(
                "SELECT pg_catalog.has_function_privilege($1, $2, 'EXECUTE')",
                "bddk_legacy_direct_grantee",
                legacy_identity,
            )

            migrated = await migrate(_PinnedPool(connection))  # type: ignore[arg-type]

            assert migrated.current
            assert not await connection.fetchval(
                "SELECT pg_catalog.has_function_privilege($1, $2, 'EXECUTE')",
                "bddk_legacy_direct_grantee",
                legacy_identity,
            )
            assert not await connection.fetchval(
                "SELECT pg_catalog.has_function_privilege($1, $2, 'EXECUTE')",
                "bddk_legacy_downstream_grantee",
                legacy_identity,
            )
        finally:
            await transaction.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_v8_concurrent_publishers_can_bind_one_staged_request_only_once(pg_pool) -> None:
    token = secrets.token_hex(32)
    document_id = "v8-concurrent-" + token[:12]
    request_id: str | None = None
    release_id: str | None = None

    async def activate_once() -> tuple[str, object]:
        connection = await pg_pool.acquire()
        try:
            try:
                async with _session_authorization(connection, "bddk_release_publisher"):
                    async with connection.transaction():
                        row = await connection.fetchrow(
                            "SELECT * FROM bddk_meta.activate_staged_corpus_release($1)",
                            request_id,
                        )
                return "activated", row
            except asyncpg.ObjectNotInPrerequisiteStateError as exc:
                return "rejected", exc
        finally:
            await pg_pool.release(connection)

    try:
        async with pg_pool.acquire() as setup, setup.transaction():
            await _ensure_v8_release_roles(setup)
            await _insert_ready_corpus(setup, document_id)
            async with _session_authorization(setup, "bddk_release_verifier"):
                staged = await _stage_v8_release(
                    setup,
                    manifest_id="staged-concurrent-" + token[:12],
                    verification_evidence_sha256=token,
                )
            assert staged is not None
            request_id = str(staged["request_id"])
            release_id = str(staged["release_id"])

        outcomes = await asyncio.gather(activate_once(), activate_once())

        assert sorted(status for status, _result in outcomes) == ["activated", "rejected"]
        rejection = next(result for status, result in outcomes if status == "rejected")
        assert "already activated" in str(rejection)
        assert (
            await pg_pool.fetchval(
                "SELECT count(*) FROM bddk_meta.corpus_release_request_activations WHERE request_id = $1",
                request_id,
            )
            == 1
        )
        assert (
            await pg_pool.fetchval(
                "SELECT count(*) FROM bddk_meta.corpus_release_activations WHERE release_id = $1",
                release_id,
            )
            == 1
        )
    finally:
        async with pg_pool.acquire() as cleanup, cleanup.transaction():
            await cleanup.execute(
                "ALTER TABLE bddk_meta.corpus_release_request_activations "
                "DISABLE TRIGGER reject_corpus_release_request_activation_update_delete"
            )
            await cleanup.execute(
                "DELETE FROM bddk_meta.corpus_release_request_activations WHERE request_id = $1",
                request_id,
            )
            await cleanup.execute(
                "ALTER TABLE bddk_meta.corpus_release_request_activations "
                "ENABLE TRIGGER reject_corpus_release_request_activation_update_delete"
            )
            await cleanup.execute(
                "ALTER TABLE bddk_meta.corpus_release_activations "
                "DISABLE TRIGGER reject_corpus_release_activation_update_delete"
            )
            await cleanup.execute(
                "DELETE FROM bddk_meta.corpus_release_activations WHERE release_id = $1",
                release_id,
            )
            await cleanup.execute(
                "ALTER TABLE bddk_meta.corpus_release_activations "
                "ENABLE TRIGGER reject_corpus_release_activation_update_delete"
            )
            await cleanup.execute(
                "ALTER TABLE bddk_meta.corpus_releases DISABLE TRIGGER reject_corpus_release_update_delete"
            )
            await cleanup.execute("DELETE FROM bddk_meta.corpus_releases WHERE release_id = $1", release_id)
            await cleanup.execute(
                "ALTER TABLE bddk_meta.corpus_releases ENABLE TRIGGER reject_corpus_release_update_delete"
            )
            await cleanup.execute(
                "ALTER TABLE bddk_meta.corpus_release_requests "
                "DISABLE TRIGGER reject_corpus_release_request_update_delete"
            )
            await cleanup.execute(
                "DELETE FROM bddk_meta.corpus_release_requests WHERE request_id = $1",
                request_id,
            )
            await cleanup.execute(
                "ALTER TABLE bddk_meta.corpus_release_requests "
                "ENABLE TRIGGER reject_corpus_release_request_update_delete"
            )
            await cleanup.execute("DELETE FROM public.documents WHERE document_id = $1", document_id)
            await cleanup.execute("DELETE FROM public.decision_cache WHERE document_id = $1", document_id)


def test_v7_retains_exactly_the_v5_corpus_as_typed_generation_members() -> None:
    statements = tuple(" ".join(statement.split()) for statement in V0007_RETAINED_CORPUS_GENERATIONS.statements)
    retained_table_ddl = {
        statement.split("CREATE TABLE bddk_retained.", 1)[1].split(" ", 1)[0]: statement
        for statement in statements
        if statement.startswith("CREATE TABLE bddk_retained.")
    }

    assert RETAINED_CORPUS_RELATIONS == CORPUS_EPOCH_TRACKED_TABLES
    assert len(RETAINED_CORPUS_RELATIONS) == 17
    assert set(retained_table_ddl) == set(RETAINED_CORPUS_RELATIONS)
    for position, relation in enumerate(RETAINED_CORPUS_RELATIONS, start=1):
        ddl = retained_table_ddl[relation]
        assert "generation_id pg_catalog.text NOT NULL" in ddl
        assert f"LIKE public.{relation} INCLUDING STORAGE INCLUDING COMPRESSION" in ddl
        assert f"CONSTRAINT rt_{position:02d}_generation_fk FOREIGN KEY (generation_id)" in ddl
        assert "REFERENCES bddk_meta.corpus_generations(generation_id)" in ddl
        assert f"CONSTRAINT rt_{position:02d}_pkey PRIMARY KEY (generation_id," in ddl


def test_v7_canonicalizes_state_and_member_hashes_across_session_gucs() -> None:
    statements = tuple(" ".join(statement.split()) for statement in V0007_RETAINED_CORPUS_GENERATIONS.statements)
    ddl = "\n".join(statements)
    upgrade_guard = statements[0]
    current_state = next(
        statement
        for statement in statements
        if statement.startswith("CREATE OR REPLACE FUNCTION bddk_meta.current_corpus_state_sha256")
    )
    retained_state = next(
        statement
        for statement in statements
        if statement.startswith("CREATE FUNCTION bddk_meta.retained_corpus_state_sha256")
    )
    row_hash = next(
        statement for statement in statements if statement.startswith("CREATE FUNCTION bddk_meta.retained_row_sha256")
    )

    assert upgrade_guard.startswith("DO $canonical_fingerprint_upgrade_guard$")
    corpus_lock = f"pg_advisory_xact_lock( {CORPUS_MUTATION_ADVISORY_KEY}::pg_catalog.int8 )"
    assert corpus_lock in upgrade_guard
    for setting in (
        "set_config('TimeZone', 'UTC', true)",
        "set_config('DateStyle', 'ISO, YMD', true)",
        "set_config('IntervalStyle', 'postgres', true)",
        "set_config('bytea_output', 'hex', true)",
        "set_config('extra_float_digits', '3', true)",
    ):
        assert setting in upgrade_guard
    assert f"USING ERRCODE = '{NONCANONICAL_FINGERPRINT_UPGRADE_SQLSTATE}'" in upgrade_guard
    assert (
        upgrade_guard.index(corpus_lock)
        < upgrade_guard.index("set_config('TimeZone'")
        < upgrade_guard.index("FROM bddk_meta.active_corpus_release")
        < upgrade_guard.index("current_corpus_state_sha256")
        < upgrade_guard.index("RAISE EXCEPTION")
    )

    for function in (current_state, retained_state, row_hash):
        assert "SET TimeZone = 'UTC'" in function
        assert "SET DateStyle = 'ISO, YMD'" in function
        assert "SET IntervalStyle = 'postgres'" in function
        assert "SET bytea_output = 'hex'" in function
        assert "SET extra_float_digits = 3" in function
    assert "bddk_meta.retained_row_sha256(member, true)" in ddl
    assert "pg_catalog.to_jsonb(member) - 'generation_id'" in row_hash


def test_v7_retention_lock_order_and_additive_serving_boundary_are_explicit() -> None:
    statements = tuple(" ".join(statement.split()) for statement in V0007_RETAINED_CORPUS_GENERATIONS.statements)
    ddl = "\n".join(statements)
    retain = next(statement for statement in statements if "retain_active_corpus_generation" in statement)

    schema_lock = f"pg_advisory_xact_lock( {SCHEMA_MIGRATION_ADVISORY_KEY}::pg_catalog.int8 )"
    corpus_lock = f"pg_advisory_xact_lock( {CORPUS_MUTATION_ADVISORY_KEY}::pg_catalog.int8 )"
    metadata_lock = "LOCK TABLE bddk_meta.corpus_release_activations, bddk_meta.corpus_generations"
    source_lock = "LOCK TABLE " + ", ".join(f"public.{relation}" for relation in RETAINED_CORPUS_RELATIONS)

    assert schema_lock in retain
    assert corpus_lock in retain
    assert metadata_lock in retain
    assert source_lock in retain
    assert (
        retain.index(schema_lock) < retain.index(corpus_lock) < retain.index(metadata_lock) < retain.index(source_lock)
    )

    # V7 is retention-only: it reads the active release but cannot alter the
    # v5 activation relation, serving view, or any live corpus member.
    assert "CREATE VIEW bddk_meta.active_corpus_release" not in ddl
    assert "CREATE OR REPLACE VIEW bddk_meta.active_corpus_release" not in ddl
    assert "INSERT INTO bddk_meta.corpus_release_activations" not in retain
    assert "UPDATE bddk_meta.corpus_release_activations" not in retain
    assert "DELETE FROM bddk_meta.corpus_release_activations" not in retain
    for relation in RETAINED_CORPUS_RELATIONS:
        assert f"INSERT INTO public.{relation}" not in retain
        assert f"UPDATE public.{relation}" not in retain
        assert f"DELETE FROM public.{relation}" not in retain


def test_v7_adds_no_retained_serving_or_vector_search_indexes() -> None:
    statements = tuple(" ".join(statement.split()) for statement in V0007_RETAINED_CORPUS_GENERATIONS.statements)
    retained_ddl = "\n".join(statement for statement in statements if "bddk_retained." in statement)

    assert "CREATE INDEX" not in retained_ddl
    assert " USING gin " not in f" {retained_ddl.lower()} "
    assert " USING hnsw " not in f" {retained_ddl.lower()} "
    assert " USING ivfflat " not in f" {retained_ddl.lower()} "


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
        self.advisory_keys: list[object] = []
        self.transaction_record = _Transaction()

    def transaction(self):
        return self.transaction_record

    async def fetchval(self, query: str, *args):
        if "server_version_num" in query:
            return self.server_version_num
        if "pg_advisory_xact_lock" in query:
            self.advisory_keys.append(args[0])
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


def _sibling_database_dsn(base_dsn: str, database_name: str) -> str:
    parsed = urlsplit(base_dsn)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.netloc:
        raise AssertionError("PostgreSQL integration test DSN must be a URL")
    return urlunsplit((parsed.scheme, parsed.netloc, f"/{database_name}", parsed.query, ""))


def _advisory_lock_parts(key: int) -> tuple[int, int]:
    unsigned = key % (1 << 64)
    return unsigned >> 32, unsigned & 0xFFFFFFFF


async def _provision_exact_v5_publisher_login(
    connection: asyncpg.Connection,
    *,
    database_name: str,
    login_name: str,
    password: str,
) -> None:
    """Provision one disposable actual LOGIN matching the reviewed v5 contract."""

    if not database_name.replace("_", "").isalnum() or not login_name.replace("_", "").isalnum():
        raise AssertionError("disposable PostgreSQL identifiers must be alphanumeric")
    await _ensure_release_publisher_role(connection)
    await connection.execute(
        "ALTER ROLE bddk_release_publisher "
        "NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS INHERIT"
    )
    quoted_password = await connection.fetchval(
        "SELECT pg_catalog.quote_literal($1::pg_catalog.text)",
        password,
    )
    await connection.execute(
        f"CREATE ROLE {login_name} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE "
        f"NOREPLICATION NOBYPASSRLS INHERIT PASSWORD {quoted_password}"
    )
    await connection.execute(f"GRANT bddk_release_publisher TO {login_name}")
    await connection.execute(
        f"REVOKE ALL PRIVILEGES ON DATABASE {database_name} FROM PUBLIC, bddk_release_publisher, {login_name}"
    )
    await connection.execute(f"GRANT CONNECT ON DATABASE {database_name} TO bddk_release_publisher")
    for schema_name in ("public", "bddk_meta", "bddk_operator"):
        await connection.execute(
            f"REVOKE ALL PRIVILEGES ON SCHEMA {schema_name} FROM PUBLIC, bddk_release_publisher, {login_name}"
        )
    await connection.execute("GRANT USAGE ON SCHEMA public, bddk_meta TO bddk_release_publisher")
    for schema_name in ("public", "bddk_meta", "bddk_operator"):
        await connection.execute(
            f"REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA {schema_name} "
            f"FROM PUBLIC, bddk_release_publisher, {login_name}"
        )
        await connection.execute(
            f"REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA {schema_name} "
            f"FROM PUBLIC, bddk_release_publisher, {login_name}"
        )
        await connection.execute(
            f"REVOKE ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA {schema_name} "
            f"FROM PUBLIC, bddk_release_publisher, {login_name}"
        )

    contract = db_identity._V5_RELEASE_PUBLISHER_CONTRACT
    selected_tables = tuple(name for name, privileges in contract.tables.items() if privileges == {"SELECT"})
    executable_routines = tuple(name for name, privileges in contract.routines.items() if privileges == {"EXECUTE"})
    await connection.execute("GRANT SELECT ON TABLE " + ", ".join(selected_tables) + " TO bddk_release_publisher")
    await connection.execute(
        "GRANT EXECUTE ON FUNCTION " + ", ".join(executable_routines) + " TO bddk_release_publisher"
    )


async def _downgrade_current_schema_to_v10(connection) -> None:
    """Restore the exact pre-graph-retention catalog in a rollback-only test."""
    from bddk_mcp.migrations import v0007_retained_corpus_generations as v7
    from bddk_mcp.migrations.v0010_corpus_release_freshness_policy import V0010_CORPUS_RELEASE_FRESHNESS_POLICY

    await connection.execute("DROP TRIGGER bump_corpus_state_epoch_on_change ON public.regulatory_relations")
    await connection.execute("DROP TABLE bddk_retained.regulatory_relations")
    for table, constraint, check in (
        ("corpus_generations", "corpus_generations_schema_check", "generation_schema_version = 1"),
        ("corpus_generation_seals", "corpus_generation_seals_relation_count_check", "relation_count = 17"),
        (
            "corpus_generation_relation_inventory",
            "corpus_generation_relation_inventory_name_check",
            f"relation_name IN ({v7._RELATION_CHECK})",
        ),
    ):
        await connection.execute(
            f"ALTER TABLE bddk_meta.{table} DROP CONSTRAINT {constraint}, ADD CONSTRAINT {constraint} CHECK ({check})"
        )
    for migration, names in (
        (
            V0007_RETAINED_CORPUS_GENERATIONS,
            {
                "current_corpus_state_sha256",
                "retained_corpus_state_sha256",
                "retain_active_corpus_generation",
                "inspect_retained_generation_storage",
            },
        ),
        (V0008_STAGED_CORPUS_RELEASES, {"activate_staged_corpus_release"}),
        (V0010_CORPUS_RELEASE_FRESHNESS_POLICY, {"stage_verified_corpus_release"}),
    ):
        for statement in migration.statements:
            if any(
                statement.strip().startswith(
                    (f"CREATE FUNCTION bddk_meta.{name}(", f"CREATE OR REPLACE FUNCTION bddk_meta.{name}(")
                )
                for name in names
            ):
                await connection.execute(statement.replace("CREATE FUNCTION", "CREATE OR REPLACE FUNCTION", 1))
    await connection.execute("DELETE FROM bddk_meta.schema_migrations WHERE version = 11")


async def _downgrade_current_schema_to_v9(connection) -> None:
    """Restore the measured-only v8 policy surface inside a test transaction."""

    await _downgrade_current_schema_to_v10(connection)

    await connection.execute(
        "DROP FUNCTION IF EXISTS bddk_meta.stage_verified_corpus_release("
        "pg_catalog.text, pg_catalog.text, pg_catalog.text, pg_catalog.text, pg_catalog.text, "
        "pg_catalog.text, pg_catalog.int4, pg_catalog.int4, pg_catalog.int4, pg_catalog.text, "
        "pg_catalog.text, pg_catalog.text, pg_catalog.int4)"
    )
    for statement in V0008_STAGED_CORPUS_RELEASES.statements:
        if statement.strip().startswith("CREATE FUNCTION bddk_meta.stage_verified_corpus_release("):
            await connection.execute(statement)
    for relation in ("corpus_releases", "corpus_release_requests"):
        await connection.execute(f"ALTER TABLE bddk_meta.{relation} DROP CONSTRAINT {relation}_policy_result_check")
        await connection.execute(
            f"ALTER TABLE bddk_meta.{relation} ADD CONSTRAINT {relation}_policy_result_check "
            f"CHECK (freshness_policy_result = '{MEASURED_FRESHNESS_POLICY_RESULT}')"
        )
    await connection.execute("DELETE FROM bddk_meta.schema_migrations WHERE version = 10")


async def _downgrade_current_schema_to_v8(connection) -> None:
    """Remove v10 and v9 inside a rollback-only PostgreSQL test transaction."""

    await _downgrade_current_schema_to_v9(connection)
    await connection.execute("DROP VIEW IF EXISTS public.regulatory_validated_relations")
    await connection.execute("DROP VIEW IF EXISTS public.regulatory_validated_legal_versions")
    await connection.execute("DROP VIEW IF EXISTS public.regulatory_validated_legal_events")
    await connection.execute("DROP TABLE IF EXISTS public.regulatory_relations CASCADE")
    await connection.execute("DELETE FROM bddk_meta.schema_migrations WHERE version = 9")


async def _downgrade_current_schema_to_v7(connection) -> None:
    """Remove v9 and v8 inside a rollback-only PostgreSQL test transaction."""

    await _downgrade_current_schema_to_v8(connection)
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.activate_staged_corpus_release(pg_catalog.text)")
    await connection.execute(
        "DROP FUNCTION IF EXISTS bddk_meta.stage_verified_corpus_release("
        "pg_catalog.text, pg_catalog.text, pg_catalog.text, pg_catalog.text, pg_catalog.text, "
        "pg_catalog.int4, pg_catalog.int4, pg_catalog.int4, pg_catalog.text, pg_catalog.text, "
        "pg_catalog.text, pg_catalog.int4)"
    )
    await connection.execute(
        "DROP TABLE IF EXISTS bddk_meta.corpus_release_request_activations, bddk_meta.corpus_release_requests CASCADE"
    )
    await connection.execute("DELETE FROM bddk_meta.schema_migrations WHERE version = 8")


async def _downgrade_current_schema_to_v6(connection) -> None:
    """Remove v8 and v7 inside a rollback-only PostgreSQL test transaction."""

    await _downgrade_current_schema_to_v7(connection)

    await connection.execute("DROP SCHEMA IF EXISTS bddk_retained CASCADE")
    await connection.execute("DROP VIEW IF EXISTS bddk_meta.corpus_release_retention_status")
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.inspect_retained_generation_storage(pg_catalog.text)")
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.retain_active_corpus_generation(pg_catalog.text)")
    await connection.execute(
        "DROP FUNCTION IF EXISTS bddk_meta.retained_corpus_state_sha256(pg_catalog.text, pg_catalog.text)"
    )
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.retained_row_sha256(anyelement, pg_catalog.bool)")
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.guard_retained_generation_member() CASCADE")
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.reject_retained_generation_mutation() CASCADE")
    await connection.execute(
        "DROP TABLE IF EXISTS bddk_meta.corpus_retained_releases, "
        "bddk_meta.corpus_generation_seals, "
        "bddk_meta.corpus_generation_relation_inventory, "
        "bddk_meta.corpus_generations CASCADE"
    )
    await connection.execute(
        "ALTER TABLE bddk_meta.corpus_releases DROP CONSTRAINT IF EXISTS corpus_releases_retention_identity_uq"
    )
    await connection.execute(
        "ALTER TABLE bddk_meta.corpus_release_activations "
        "DROP CONSTRAINT IF EXISTS corpus_release_activations_retention_identity_uq"
    )
    await connection.execute("DELETE FROM bddk_meta.schema_migrations WHERE version = 7")

    v5_state_function = next(
        statement
        for statement in V0005_CORPUS_RELEASE_PUBLICATION.statements
        if "CREATE FUNCTION bddk_meta.current_corpus_state_sha256" in statement
    )
    await connection.execute(v5_state_function.replace("CREATE FUNCTION", "CREATE OR REPLACE FUNCTION", 1))


async def _downgrade_current_schema_to_v5(connection) -> None:
    """Remove only v7 and v6 for the guarded pre-v7 publication rehearsal."""

    await _downgrade_current_schema_to_v6(connection)
    await connection.execute(
        "DROP FUNCTION IF EXISTS bddk_meta.resolve_regulation_status(pg_catalog.text, pg_catalog.date)"
    )
    await connection.execute("DELETE FROM bddk_meta.schema_migrations WHERE version = 6")


async def _downgrade_current_schema_to_v2(connection) -> None:
    """Remove v7 through v3 inside a rollback-only test transaction."""

    await _downgrade_current_schema_to_v6(connection)

    await connection.execute(
        "DROP FUNCTION IF EXISTS bddk_meta.resolve_regulation_status(pg_catalog.text, pg_catalog.date)"
    )
    await connection.execute("DELETE FROM bddk_meta.schema_migrations WHERE version = 6")
    await connection.execute("DROP VIEW IF EXISTS bddk_meta.active_corpus_release")
    for table_name in CORPUS_EPOCH_TRACKED_TABLES:
        await connection.execute(f"DROP TRIGGER IF EXISTS bump_corpus_state_epoch_on_change ON public.{table_name}")
    await connection.execute(
        "DROP TABLE IF EXISTS bddk_meta.corpus_release_activations, "
        "bddk_meta.corpus_releases, bddk_meta.corpus_state_epoch CASCADE"
    )
    await connection.execute(
        "DROP FUNCTION IF EXISTS bddk_meta.publish_verified_corpus_release("
        "pg_catalog.text, pg_catalog.text, pg_catalog.text, pg_catalog.int4, "
        "pg_catalog.int4, pg_catalog.int4, pg_catalog.text)"
    )
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.corpus_retrieval_ready(pg_catalog.text)")
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.current_corpus_state_sha256(pg_catalog.text)")
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.corpus_fingerprint_frame(pg_catalog.text)")
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.bump_corpus_state_epoch()")
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.reject_corpus_release_mutation()")
    await connection.execute("DELETE FROM bddk_meta.schema_migrations WHERE version = 5")

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

    for trigger_name in (
        "invalidate_retrieval_publication_on_chunk_insert",
        "invalidate_retrieval_publication_on_chunk_delete",
        "invalidate_retrieval_publication_on_chunk_update",
    ):
        await connection.execute(f"DROP TRIGGER IF EXISTS {trigger_name} ON public.document_chunks")
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
    assert connection.advisory_keys == [SCHEMA_MIGRATION_ADVISORY_KEY]
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
async def test_noncanonical_v5_fingerprint_guard_is_actionable_and_sanitized() -> None:
    driver_error = asyncpg.PostgresError("private release and corpus identifiers")
    driver_error.sqlstate = NONCANONICAL_FINGERPRINT_UPGRADE_SQLSTATE
    connection = _FakeMigrationConnection(
        fail_statement="DO $canonical_fingerprint_upgrade_guard$",
        fail_exception=driver_error,
    )
    connection.history_exists = True
    connection.history = _history_rows()[:6]

    with pytest.raises(MigrationPrerequisiteError) as exc_info:
        await migrate(_FakePool(connection))  # type: ignore[arg-type]

    message = str(exc_info.value)
    statements = [query for query, _args in connection.executed]
    assert connection.transaction_record.rolled_back
    assert connection.history == _history_rows()[:6]
    assert "Migration 7 refused" in message
    assert "pre-v7 schema (v5 or v6)" in message
    assert "publish and activate a new release" in message
    assert "docs/LEGACY_DATABASE_UPGRADE.md" in message
    assert "Do not update or backfill the prior release row" in message
    assert "private release" not in message
    assert exc_info.value.__cause__ is None
    assert not any("CREATE SCHEMA bddk_retained" in statement for statement in statements)


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
async def test_v5_publication_migration_failure_rolls_back_without_recording_version() -> None:
    connection = _FakeMigrationConnection(fail_statement="CREATE TABLE bddk_meta.corpus_release_activations")
    connection.history_exists = True
    connection.history = _history_rows()[:4]

    with pytest.raises(MigrationError) as exc_info:
        await migrate(_FakePool(connection))  # type: ignore[arg-type]

    assert connection.transaction_record.rolled_back
    assert connection.history == _history_rows()[:4]
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
async def test_postgres_v7_refuses_noncanonical_v5_release_then_accepts_reviewed_republication(pg_pool) -> None:
    connection = await pg_pool.acquire()
    transaction = connection.transaction()
    await transaction.start()
    document_id = "v7-upgrade-guc-proof"
    try:
        await _downgrade_current_schema_to_v6(connection)
        await _ensure_release_publisher_role(connection)
        content_hash = await _insert_ready_corpus(connection, document_id)
        await _insert_canonical_legal_state(
            connection,
            document_id=document_id,
            content_hash=content_hash,
        )
        await connection.execute("SET LOCAL DateStyle = 'German'")
        noncanonical_release = await _publish(
            connection,
            manifest_id="v7-upgrade-noncanonical",
        )
        assert (
            await connection.fetchval(
                "SELECT bddk_meta.current_corpus_state_sha256($1)",
                _PROFILE_SHA256,
            )
            == noncanonical_release["corpus_state_sha256"]
        )

        with pytest.raises(MigrationPrerequisiteError) as exc_info:
            await migrate(_PinnedPool(connection))  # type: ignore[arg-type]

        message = str(exc_info.value)
        assert "publish and activate a new release" in message
        assert "docs/LEGACY_DATABASE_UPGRADE.md" in message
        assert document_id not in message
        assert str(noncanonical_release["release_id"]) not in message
        assert exc_info.value.__cause__ is None
        assert await connection.fetchval("SELECT max(version) FROM bddk_meta.schema_migrations") == 6
        assert await connection.fetchval("SELECT pg_catalog.to_regnamespace('bddk_retained')") is None
        assert await connection.fetchval("SELECT pg_catalog.to_regclass('bddk_meta.corpus_generations')") is None
        assert (
            await connection.fetchval(
                "SELECT pg_catalog.to_regprocedure('bddk_meta.retained_row_sha256(anyelement,boolean)')"
            )
            is None
        )
        assert (
            await connection.fetchval(
                """
                SELECT pg_catalog.count(*)
                FROM pg_catalog.pg_constraint
                WHERE conname = ANY($1::pg_catalog.text[])
                """,
                [
                    "corpus_releases_retention_identity_uq",
                    "corpus_release_activations_retention_identity_uq",
                ],
            )
            == 0
        )
        assert await connection.fetchval(
            """
            SELECT proconfig
            FROM pg_catalog.pg_proc
            WHERE oid = 'bddk_meta.current_corpus_state_sha256(pg_catalog.text)'::pg_catalog.regprocedure
            """
        ) == ["search_path=pg_catalog"]

        await connection.execute(
            """
            SELECT pg_catalog.set_config('TimeZone', 'Pacific/Auckland', true),
                   pg_catalog.set_config('DateStyle', 'SQL, DMY', true),
                   pg_catalog.set_config('IntervalStyle', 'postgres_verbose', true),
                   pg_catalog.set_config('bytea_output', 'escape', true),
                   pg_catalog.set_config('extra_float_digits', '0', true)
            """
        )
        canonical_release = await publish_strict_corpus_release(
            connection,
            SimpleNamespace(
                manifest_sha256="8" * 64,
                manifest=SimpleNamespace(
                    manifest_id="v7-upgrade-canonical",
                    freshness=SimpleNamespace(
                        source_detection_slo_seconds=60,
                        publication_slo_seconds=120,
                        max_manifest_age_seconds=3600,
                        slo_evidence_status="measured",
                    ),
                    integrity=SimpleNamespace(
                        signature_status="verified",
                        signature_public_key_sha256="9" * 64,
                    ),
                ),
            ),
            retrieval_profile_sha256=_PROFILE_SHA256,
            require_quantified_freshness=True,
            require_measured_freshness=True,
            require_verified_signature=True,
        )
        assert canonical_release.release_id != noncanonical_release["release_id"]
        assert canonical_release.corpus_state_sha256 != noncanonical_release["corpus_state_sha256"]
        assert await connection.fetchval("SELECT current_setting('DateStyle')") == "ISO, YMD"

        ledger_owner = await connection.fetchval(
            "SELECT pg_catalog.pg_get_userbyid(relation.relowner) "
            "FROM pg_catalog.pg_class AS relation "
            "WHERE relation.oid = 'bddk_meta.schema_migrations'::pg_catalog.regclass"
        )
        current_user = await connection.fetchval("SELECT CURRENT_USER")
        if ledger_owner != current_user:
            quoted_owner = await connection.fetchval(
                "SELECT pg_catalog.quote_ident($1::pg_catalog.text)",
                ledger_owner,
            )
            await connection.execute(f"SET LOCAL ROLE {quoted_owner}")
        migrated = await migrate(_PinnedPool(connection))  # type: ignore[arg-type]
        catalog = await inspect_catalog_integrity(connection)
        with patch("bddk_mcp.store.vector_store.retrieval_profile_hash", return_value=_PROFILE_SHA256):
            readiness = await inspect_database_readiness(
                connection,
                require_corpus=True,
                require_active_release=True,
            )

        assert migrated.current
        assert (
            await connection.fetchval("SELECT max(version) FROM bddk_meta.schema_migrations") == LATEST_SCHEMA_VERSION
        )
        assert catalog.valid
        # V11 introduces graph-aware fingerprints and expires earlier activations.
        assert readiness.corpus_issues == ("no verified corpus release is active",)
        assert readiness.active_corpus_release is None
    finally:
        await transaction.rollback()
        await pg_pool.release(connection)


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_actual_v5_publisher_login_can_use_supported_canonical_republication_path(pg_pool) -> None:
    """Exercise the supported pre-v7 remediation boundary with a real LOGIN."""

    base_dsn = os.environ.get(
        "BDDK_TEST_DATABASE_URL",
        "postgresql://bddk:bddk@localhost:5432/bddk_test",
    )
    suffix = f"{os.getpid()}_{secrets.token_hex(3)}"
    database_name = f"bddk_v5_publication_{suffix}"
    login_name = f"bddk_v5_publisher_{suffix}"
    password = "V5publisher_" + secrets.token_hex(12)
    isolated_dsn = _sibling_database_dsn(base_dsn, database_name)
    isolated_pool: asyncpg.Pool | None = None
    publisher_pool: asyncpg.Pool | None = None
    database_created = False

    admin = await asyncpg.connect(base_dsn)
    try:
        await admin.execute(f"CREATE DATABASE {database_name}")
        database_created = True
    finally:
        await admin.close()

    try:
        isolated_pool = await asyncpg.create_pool(isolated_dsn, min_size=1, max_size=4)
        await isolated_pool.execute("CREATE EXTENSION vector")
        await isolated_pool.execute("CREATE EXTENSION unaccent")
        assert (await migrate(isolated_pool)).current

        async with isolated_pool.acquire() as owner:
            await _downgrade_current_schema_to_v5(owner)
            document_id = "v5-actual-login-canonical-publication"
            content_hash = await _insert_ready_corpus(owner, document_id)
            await _insert_canonical_legal_state(
                owner,
                document_id=document_id,
                content_hash=content_hash,
            )
            await _provision_exact_v5_publisher_login(
                owner,
                database_name=database_name,
                login_name=login_name,
                password=password,
            )

        publisher_pool = await asyncpg.create_pool(
            isolated_dsn,
            user=login_name,
            password=password,
            min_size=1,
            max_size=2,
            init=assert_release_publication_connection_identity,
        )
        await assert_release_publication_identity(publisher_pool)
        assert (
            await assert_release_publication_ready(
                publisher_pool,
                retrieval_profile_sha256=_PROFILE_SHA256,
                require_active_release=False,
            )
            is None
        )
        async with publisher_pool.acquire() as publisher, publisher.transaction():
            await publisher.execute("SET LOCAL DateStyle = 'German'")
            noncanonical_release = await _publish(
                publisher,
                manifest_id="v5-actual-login-noncanonical",
            )

        with pytest.raises(MigrationPrerequisiteError):
            await migrate(isolated_pool)
        assert await isolated_pool.fetchval("SELECT max(version) FROM bddk_meta.schema_migrations") == 5

        await assert_release_publication_identity(publisher_pool)
        assert (
            await assert_release_publication_ready(
                publisher_pool,
                retrieval_profile_sha256=_PROFILE_SHA256,
                require_active_release=False,
            )
        ) is not None

        validation = SimpleNamespace(
            manifest_sha256="8" * 64,
            manifest=SimpleNamespace(
                manifest_id="v5-actual-login-canonical",
                freshness=SimpleNamespace(
                    source_detection_slo_seconds=60,
                    publication_slo_seconds=120,
                    max_manifest_age_seconds=3600,
                    slo_evidence_status="measured",
                ),
                integrity=SimpleNamespace(
                    signature_status="verified",
                    signature_public_key_sha256="9" * 64,
                ),
            ),
        )
        async with publisher_pool.acquire() as publisher, publisher.transaction():
            await publisher.execute(
                """
                SELECT pg_catalog.set_config('TimeZone', 'Pacific/Auckland', true),
                       pg_catalog.set_config('DateStyle', 'SQL, DMY', true),
                       pg_catalog.set_config('IntervalStyle', 'postgres_verbose', true),
                       pg_catalog.set_config('bytea_output', 'escape', true),
                       pg_catalog.set_config('extra_float_digits', '0', true)
                """
            )
            canonical_release = await publish_strict_corpus_release(
                publisher,
                validation,
                retrieval_profile_sha256=_PROFILE_SHA256,
                require_quantified_freshness=True,
                require_measured_freshness=True,
                require_verified_signature=True,
            )

        active = await assert_release_publication_ready(
            publisher_pool,
            retrieval_profile_sha256=_PROFILE_SHA256,
            require_active_release=True,
        )
        assert active is not None
        assert active.release_id == canonical_release.release_id
        assert canonical_release.release_id != noncanonical_release["release_id"]
        assert canonical_release.corpus_state_sha256 != noncanonical_release["corpus_state_sha256"]

        await publisher_pool.close()
        publisher_pool = None
        assert (await migrate(isolated_pool)).current
        assert (
            await isolated_pool.fetchval("SELECT max(version) FROM bddk_meta.schema_migrations")
            == LATEST_SCHEMA_VERSION
        )
    finally:
        if publisher_pool is not None:
            await publisher_pool.close()
        if isolated_pool is not None:
            await isolated_pool.close()
        admin = await asyncpg.connect(base_dsn)
        try:
            if database_created:
                await admin.execute(f"DROP DATABASE {database_name} WITH (FORCE)")
            await admin.execute(f"DROP ROLE IF EXISTS {login_name}")
        finally:
            await admin.close()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_postgres_v7_guard_waits_for_inflight_v6_publisher_and_evaluates_its_release(pg_pool) -> None:
    assert await pg_pool.fetchval("SELECT current_setting('server_version_num')::pg_catalog.int4") >= 170000
    base_dsn = os.environ.get(
        "BDDK_TEST_DATABASE_URL",
        "postgresql://bddk:bddk@localhost:5432/bddk_test",
    )
    database_name = f"bddk_v7_guard_concurrency_{os.getpid()}"
    isolated_dsn = _sibling_database_dsn(base_dsn, database_name)
    isolated_pool: asyncpg.Pool | None = None
    publisher: asyncpg.Connection | None = None
    observer: asyncpg.Connection | None = None
    publisher_transaction = None
    migration_task: asyncio.Task[MigrationState] | None = None
    database_created = False

    admin = await asyncpg.connect(base_dsn)
    try:
        await admin.execute(f"CREATE DATABASE {database_name}")
        database_created = True
    finally:
        await admin.close()

    try:
        isolated_pool = await asyncpg.create_pool(isolated_dsn, min_size=1, max_size=4)
        await isolated_pool.execute("CREATE EXTENSION vector")
        await isolated_pool.execute("CREATE EXTENSION unaccent")
        assert (await migrate(isolated_pool)).current

        async with isolated_pool.acquire() as setup:
            await _downgrade_current_schema_to_v6(setup)
            await _ensure_release_publisher_role(setup)
            document_id = "v7-concurrent-publisher-proof"
            content_hash = await _insert_ready_corpus(setup, document_id)
            await _insert_canonical_legal_state(
                setup,
                document_id=document_id,
                content_hash=content_hash,
            )
            assert await setup.fetchval("SELECT count(*) FROM bddk_meta.active_corpus_release") == 0

        publisher = await isolated_pool.acquire()
        observer = await isolated_pool.acquire()
        publisher_transaction = publisher.transaction()
        await publisher_transaction.start()
        await publisher.execute("SET LOCAL DateStyle = 'German'")
        await publisher.fetchval(
            "SELECT pg_catalog.pg_advisory_xact_lock($1::pg_catalog.int8)",
            CORPUS_MUTATION_ADVISORY_KEY,
        )
        publisher_pid = await publisher.fetchval("SELECT pg_catalog.pg_backend_pid()")

        migration_task = asyncio.create_task(migrate(isolated_pool))
        corpus_classid, corpus_objid = _advisory_lock_parts(CORPUS_MUTATION_ADVISORY_KEY)
        schema_classid, schema_objid = _advisory_lock_parts(SCHEMA_MIGRATION_ADVISORY_KEY)
        waiting_pid = None
        deadline = asyncio.get_running_loop().time() + 2.0
        while waiting_pid is None and asyncio.get_running_loop().time() < deadline:
            waiting_pid = await observer.fetchval(
                """
                WITH waiting_corpus AS (
                    SELECT lock_record.pid
                    FROM pg_catalog.pg_locks AS lock_record
                    WHERE lock_record.locktype = 'advisory'
                      AND lock_record.classid = ($1::pg_catalog.int8)::pg_catalog.oid
                      AND lock_record.objid = ($2::pg_catalog.int8)::pg_catalog.oid
                      AND lock_record.objsubid = 1
                      AND NOT lock_record.granted
                ), held_schema AS (
                    SELECT lock_record.pid
                    FROM pg_catalog.pg_locks AS lock_record
                    WHERE lock_record.locktype = 'advisory'
                      AND lock_record.classid = ($3::pg_catalog.int8)::pg_catalog.oid
                      AND lock_record.objid = ($4::pg_catalog.int8)::pg_catalog.oid
                      AND lock_record.objsubid = 1
                      AND lock_record.granted
                )
                SELECT waiting.pid
                FROM waiting_corpus AS waiting
                JOIN held_schema AS held USING (pid)
                LIMIT 1
                """,
                corpus_classid,
                corpus_objid,
                schema_classid,
                schema_objid,
            )
            if waiting_pid is None:
                if migration_task.done():
                    outcome = (await asyncio.gather(migration_task, return_exceptions=True))[0]
                    pytest.fail(
                        "v7 migration completed before waiting for the publisher corpus lock "
                        f"({type(outcome).__name__})",
                        pytrace=False,
                    )
                await asyncio.sleep(0.01)

        assert waiting_pid is not None
        assert waiting_pid != publisher_pid
        assert not migration_task.done()
        assert await observer.fetchval("SELECT count(*) FROM bddk_meta.active_corpus_release") == 0

        published = await _publish(
            publisher,
            manifest_id="v7-concurrent-noncanonical",
        )
        await publisher_transaction.commit()
        publisher_transaction = None

        with pytest.raises(MigrationPrerequisiteError) as exc_info:
            await asyncio.wait_for(migration_task, timeout=4.0)
        migration_task = None

        assert "pre-v7 schema (v5 or v6)" in str(exc_info.value)
        assert await observer.fetchval("SELECT max(version) FROM bddk_meta.schema_migrations") == 6
        assert await observer.fetchval("SELECT pg_catalog.to_regnamespace('bddk_retained')") is None
        assert (
            await observer.fetchval("SELECT release_id FROM bddk_meta.active_corpus_release") == published["release_id"]
        )
    finally:
        if migration_task is not None:
            if not migration_task.done():
                migration_task.cancel()
            await asyncio.gather(migration_task, return_exceptions=True)
        if publisher_transaction is not None and publisher is not None and publisher.is_in_transaction():
            await publisher_transaction.rollback()
        if isolated_pool is not None:
            if observer is not None:
                await isolated_pool.release(observer)
            if publisher is not None:
                await isolated_pool.release(publisher)
            await isolated_pool.close()
        if database_created:
            admin = await asyncpg.connect(base_dsn)
            try:
                await admin.execute(f"DROP DATABASE {database_name} WITH (FORCE)")
            finally:
                await admin.close()


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
    await pg_pool.execute("DROP SCHEMA IF EXISTS bddk_retained CASCADE")
    await pg_pool.execute("DROP SCHEMA IF EXISTS bddk_meta CASCADE")
    await pg_pool.execute(
        """
        DROP TABLE IF EXISTS
            public.regulatory_relations,
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


@pytest.mark.asyncio
async def test_v10_activates_an_explicitly_unmeasured_release_under_its_own_identity(pg_pool) -> None:
    connection = await pg_pool.acquire()
    transaction = connection.transaction()
    await transaction.start()
    try:
        await _ensure_v8_release_roles(connection)
        content_hash = await _insert_ready_corpus(connection, "v10-unmeasured-release")
        await _insert_canonical_legal_state(
            connection,
            document_id="v10-unmeasured-release",
            content_hash=content_hash,
        )

        async with _session_authorization(connection, "bddk_release_verifier"):
            unmeasured = await _stage_v8_release(
                connection,
                freshness_policy_result=UNMEASURED_FRESHNESS_POLICY_RESULT,
            )
        assert unmeasured is not None

        async with _session_authorization(connection, "bddk_release_publisher"):
            activated = await connection.fetchrow(
                "SELECT * FROM bddk_meta.activate_staged_corpus_release($1)",
                unmeasured["request_id"],
            )
        assert activated is not None
        assert activated["freshness_policy_result"] == UNMEASURED_FRESHNESS_POLICY_RESULT
        assert (
            await connection.fetchval("SELECT freshness_policy_result FROM bddk_meta.active_corpus_release")
            == UNMEASURED_FRESHNESS_POLICY_RESULT
        )

        # The policy level is fingerprinted into both identities, so the same
        # corpus state cannot yield one release that is readable as either level.
        async with _session_authorization(connection, "bddk_release_verifier"):
            measured = await _stage_v8_release(
                connection,
                freshness_policy_result=MEASURED_FRESHNESS_POLICY_RESULT,
            )
        assert measured is not None
        assert measured["release_id"] != unmeasured["release_id"]
        assert measured["request_id"] != unmeasured["request_id"]
    finally:
        await transaction.rollback()
        await pg_pool.release(connection)


@pytest.mark.asyncio
async def test_v10_refuses_a_freshness_policy_outside_the_closed_set(pg_pool) -> None:
    connection = await pg_pool.acquire()
    transaction = connection.transaction()
    await transaction.start()
    try:
        await _ensure_v8_release_roles(connection)
        content_hash = await _insert_ready_corpus(connection, "v10-policy-refusal")
        await _insert_canonical_legal_state(
            connection,
            document_id="v10-policy-refusal",
            content_hash=content_hash,
        )

        # Each refusal aborts its statement, so keep them in separate savepoints.
        async with _session_authorization(connection, "bddk_release_verifier"):
            savepoint = connection.transaction()
            await savepoint.start()
            with pytest.raises(asyncpg.PostgresError) as staging_error:
                await _stage_v8_release(
                    connection,
                    freshness_policy_result="unquantified_unsigned_pass",
                )
            await savepoint.rollback()
        assert staging_error.value.sqlstate == "22023"

        savepoint = connection.transaction()
        await savepoint.start()
        with pytest.raises(asyncpg.PostgresError):
            await connection.execute(
                """
                INSERT INTO bddk_meta.corpus_releases (
                    release_id, manifest_id, manifest_sha256, signer_key_sha256,
                    freshness_policy_result, source_detection_slo_seconds,
                    publication_slo_seconds, max_manifest_age_seconds,
                    retrieval_profile_sha256, corpus_state_sha256
                ) VALUES (
                    'corpus_release_sha256_' || repeat('a', 64), 'policy-refusal', repeat('b', 64),
                    repeat('c', 64), 'unquantified_unsigned_pass', 60, 120, 3600,
                    repeat('d', 64), repeat('e', 64)
                )
                """
            )
        await savepoint.rollback()
    finally:
        await transaction.rollback()
        await pg_pool.release(connection)


@pytest.mark.asyncio
async def test_v10_removes_the_measured_only_staging_signature(pg_pool) -> None:
    connection = await pg_pool.acquire()
    try:
        signatures = await connection.fetch(
            """
            SELECT pg_catalog.pg_get_function_identity_arguments(routine.oid) AS identity_arguments
            FROM pg_catalog.pg_proc AS routine
            JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = routine.pronamespace
            WHERE namespace.nspname = 'bddk_meta'
              AND routine.proname = 'stage_verified_corpus_release'
            """
        )
        identities = {str(row["identity_arguments"]) for row in signatures}
        assert len(identities) == 1
        assert "requested_freshness_policy_result text" in identities.pop()
    finally:
        await pg_pool.release(connection)
