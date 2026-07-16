"""Static and opt-in live checks for PostgreSQL least-privilege assets."""

from __future__ import annotations

import os
import re
import secrets
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from urllib.parse import urlsplit

import asyncpg
import pytest

from bddk_mcp.db_identity import (
    DatabaseIdentityError,
    assert_database_connection_identity,
    assert_database_identity,
)
from bddk_mcp.db_lifecycle import _SCHEMA_OWNER_IDENTITY_SQL, assert_schema_owner_identity
from bddk_mcp.migrations import LATEST_SCHEMA_VERSION, migrate
from bddk_mcp.observability.telemetry import assert_telemetry_writer_ready

ROOT = Path(__file__).resolve().parents[1]
ROLES_SQL = (ROOT / "deploy/postgres/01_roles.sql").read_text(encoding="utf-8")
GRANTS_SQL = (ROOT / "deploy/postgres/02_grants.sql").read_text(encoding="utf-8")
README = (ROOT / "deploy/postgres/README.md").read_text(encoding="utf-8")
OPENSHIFT_SECRETS = (ROOT / "deploy/openshift/secrets.example.yaml").read_text(encoding="utf-8")
CI_WORKFLOW = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")

GROUP_ROLES = {
    "bddk_schema_owner",
    "bddk_public_reader",
    "bddk_ingestion",
    "bddk_release_verifier",
    "bddk_release_publisher",
    "bddk_operator_runtime",
    "bddk_telemetry_writer",
}
PUBLIC_CORPUS_TABLES = {
    "public.decision_cache",
    "public.documents",
    "public.document_sections",
    "public.document_versions",
    "public.document_chunks",
    "public.document_retrieval_publications",
}
PUBLIC_READ_RELATIONS = PUBLIC_CORPUS_TABLES | {
    "bddk_meta.active_corpus_release",
    "public.regulatory_validated_section_citations",
}
INGESTION_TABLES = PUBLIC_CORPUS_TABLES | {
    "public.sync_metadata",
    "public.sync_failures",
}
INGESTION_SEQUENCES = {
    "public.document_sections_id_seq",
    "public.document_versions_id_seq",
    "public.document_chunks_id_seq",
}
REGULATORY_VERSION_TABLES = {
    "public.regulatory_evidence",
    "public.regulatory_family_imports",
    "public.regulatory_instruments",
    "public.regulatory_legal_events",
    "public.regulatory_legal_status_assertions",
    "public.regulatory_legal_version_artifacts",
    "public.regulatory_legal_version_provisions",
    "public.regulatory_legal_versions",
    "public.regulatory_provisions",
    "public.regulatory_source_blobs",
    "public.regulatory_source_artifacts",
}
VERIFIER_READ_RELATIONS = PUBLIC_CORPUS_TABLES | REGULATORY_VERSION_TABLES | {"bddk_meta.schema_migrations"}
PUBLISHER_READ_RELATIONS = {
    "bddk_meta.schema_migrations",
    "bddk_meta.active_corpus_release",
    "bddk_meta.corpus_release_retention_status",
}
STAGE_ROUTINE = (
    "bddk_meta.stage_verified_corpus_release( pg_catalog.text, pg_catalog.text, pg_catalog.text, "
    "pg_catalog.text, pg_catalog.text, pg_catalog.int4, pg_catalog.int4, pg_catalog.int4, "
    "pg_catalog.text, pg_catalog.text, pg_catalog.text, pg_catalog.int4 )"
)
RETAINED_CORPUS_TABLES = {
    "bddk_retained.decision_cache",
    "bddk_retained.documents",
    "bddk_retained.document_sections",
    "bddk_retained.document_versions",
    "bddk_retained.document_chunks",
    "bddk_retained.document_retrieval_publications",
    "bddk_retained.regulatory_instruments",
    "bddk_retained.regulatory_family_imports",
    "bddk_retained.regulatory_source_blobs",
    "bddk_retained.regulatory_source_artifacts",
    "bddk_retained.regulatory_evidence",
    "bddk_retained.regulatory_legal_versions",
    "bddk_retained.regulatory_legal_version_artifacts",
    "bddk_retained.regulatory_legal_events",
    "bddk_retained.regulatory_legal_status_assertions",
    "bddk_retained.regulatory_provisions",
    "bddk_retained.regulatory_legal_version_provisions",
}


def _normalized_sql(value: str) -> str:
    without_comments = re.sub(r"--[^\n]*", "", value)
    return " ".join(without_comments.lower().split())


def _grant_object_list(sql: str, privileges: str, role: str, *, kind: str = "table") -> set[str]:
    normalized = _normalized_sql(sql)
    pattern = rf"grant {re.escape(privileges)} on {kind} ([^;]*?) to {re.escape(role)};"
    match = re.search(pattern, normalized)
    assert match is not None, f"missing {privileges} {kind} grant for {role}"
    return {item.strip() for item in match.group(1).split(",")}


def test_group_roles_are_idempotent_non_login_capability_roles() -> None:
    normalized = _normalized_sql(ROLES_SQL)

    assert "password" not in normalized
    assert "create role bddk_operator" not in normalized
    for role in GROUP_ROLES:
        assert f"'{role}'" in ROLES_SQL
        assert re.search(
            rf"alter role {role}\s+nologin nosuperuser nocreatedb nocreaterole noreplication nobypassrls;", normalized
        )


def test_dba_assets_refuse_an_absent_or_wrong_independent_database_target() -> None:
    for sql in (ROLES_SQL, GRANTS_SQL):
        normalized = _normalized_sql(sql)
        guard = normalized.index("current_setting('bddk.expected_database', true)")
        first_mutation = min(
            position
            for token in ("create role", "alter schema", "alter table", "revoke ")
            if (position := normalized.find(token)) >= 0
        )
        assert guard < first_mutation
        assert "current_database() <> expected_database" in normalized


def test_public_defaults_and_existing_objects_are_revoked() -> None:
    normalized = _normalized_sql(ROLES_SQL)

    assert "revoke connect, create, temporary on database" in normalized
    assert "grant connect, create on database" in normalized
    assert "revoke usage, create on schema public from public;" in normalized
    assert "create schema" not in normalized
    assert "revoke all privileges on all tables in schema public from public;" in normalized
    assert "revoke all privileges on all sequences in schema public from public;" in normalized
    for object_kind in ("tables", "sequences", "functions"):
        assert (
            "alter default privileges for role bddk_schema_owner in schema public "
            f"revoke all privileges on {object_kind} from public;"
        ) in normalized


def test_post_migration_schemas_and_ledger_are_hardened() -> None:
    normalized = _normalized_sql(GRANTS_SQL)

    for schema in ("bddk_meta", "bddk_operator"):
        assert f"alter schema {schema} owner to bddk_schema_owner;" in normalized
        assert f"revoke all privileges on all tables in schema {schema} from public;" in normalized
        assert f"revoke all privileges on all sequences in schema {schema} from public;" in normalized
        for object_kind in ("tables", "sequences", "functions"):
            assert (
                f"alter default privileges for role bddk_schema_owner in schema {schema} "
                f"revoke all privileges on {object_kind} from public;"
            ) in normalized
    for role in (
        "bddk_public_reader",
        "bddk_ingestion",
        "bddk_operator_runtime",
    ):
        assert f"grant usage on schema bddk_meta to {role};" in normalized
        assert f"grant select on table bddk_meta.schema_migrations to {role};" in normalized
    assert "grant usage on schema bddk_meta to bddk_release_publisher;" in normalized
    assert "grant usage on schema bddk_meta to bddk_release_verifier;" in normalized
    assert "bddk_meta.schema_migrations" in _grant_object_list(GRANTS_SQL, "select", "bddk_release_verifier")
    assert "bddk_meta.schema_migrations" in _grant_object_list(GRANTS_SQL, "select", "bddk_release_publisher")


def test_retained_generation_assets_are_owner_only_except_publisher_facade() -> None:
    normalized = _normalized_sql(GRANTS_SQL)

    assert "alter schema bddk_retained owner to bddk_schema_owner;" in normalized
    assert "revoke all privileges on schema bddk_retained from public;" in normalized
    assert "revoke all privileges on all tables in schema bddk_retained from public;" in normalized
    for object_kind in ("tables", "sequences", "functions"):
        assert (
            "alter default privileges for role bddk_schema_owner in schema bddk_retained "
            f"revoke all privileges on {object_kind} from public;"
        ) in normalized

    for relation in RETAINED_CORPUS_TABLES:
        assert f"alter table {relation} owner to bddk_schema_owner;" in normalized
        assert not any(
            re.search(rf"\b{re.escape(relation)}\b", statement)
            for statement in normalized.split(";")
            if statement.strip().startswith("grant ")
        )

    view = "bddk_meta.corpus_release_retention_status"
    assert f"alter view {view} owner to bddk_schema_owner;" in normalized
    assert f"{view} to bddk_release_publisher;" in normalized
    assert "grant usage on schema bddk_retained" not in normalized


def test_positive_runtime_grants_are_explicit_not_default_wildcards() -> None:
    normalized = _normalized_sql(GRANTS_SQL)

    assert not re.search(r"(?:^|;) grant [^;]* on all (tables|sequences)", normalized)
    assert _grant_object_list(GRANTS_SQL, "select", "bddk_public_reader") == PUBLIC_READ_RELATIONS
    assert _grant_object_list(GRANTS_SQL, "select, insert, update, delete", "bddk_ingestion") == INGESTION_TABLES
    assert _grant_object_list(GRANTS_SQL, "usage", "bddk_ingestion", kind="sequence") == INGESTION_SEQUENCES
    assert _grant_object_list(GRANTS_SQL, "select", "bddk_release_publisher") == PUBLISHER_READ_RELATIONS
    assert _grant_object_list(GRANTS_SQL, "select", "bddk_release_verifier") == VERIFIER_READ_RELATIONS


def test_legal_version_workspace_is_owned_but_denied_to_every_runtime_role() -> None:
    normalized = _normalized_sql(GRANTS_SQL)

    for relation in REGULATORY_VERSION_TABLES:
        assert f"alter table {relation} owner to bddk_schema_owner;" in normalized
        grants = [
            statement
            for statement in normalized.split(";")
            if statement.strip().startswith("grant ") and re.search(rf"\b{re.escape(relation)}\b", statement)
        ]
        assert grants == [
            next(
                statement
                for statement in normalized.split(";")
                if statement.strip().startswith("grant select on table")
                and statement.strip().endswith("to bddk_release_verifier")
            )
        ]

    revoke = re.search(
        r"revoke all privileges on table (.*?) from public, bddk_public_reader, bddk_ingestion, "
        r"bddk_release_verifier, bddk_release_publisher, bddk_operator_runtime, bddk_telemetry_writer;",
        normalized,
    )
    assert revoke is not None
    assert {item.strip() for item in revoke.group(1).split(",")} == REGULATORY_VERSION_TABLES

    view = "public.regulatory_validated_section_citations"
    assert f"alter view {view} owner to bddk_schema_owner;" in normalized
    assert (
        f"revoke all privileges on table {view} from public, bddk_public_reader, bddk_ingestion, "
        "bddk_release_verifier, bddk_release_publisher, bddk_operator_runtime, bddk_telemetry_writer;"
    ) in normalized


def test_operator_and_telemetry_grants_are_narrow() -> None:
    normalized = _normalized_sql(GRANTS_SQL)

    assert (
        "grant select, insert, update, delete on table bddk_operator.operator_jobs to bddk_operator_runtime;"
    ) in normalized
    assert "operator_job_schema_versions" not in normalized
    assert "on table public.tool_call_traces to bddk_telemetry_writer;" in normalized
    assert "grant usage on sequence public.tool_call_traces_id_seq to bddk_telemetry_writer;" in normalized
    telemetry_grants = {
        statement.strip()
        for statement in normalized.split(";")
        if statement.strip().startswith("grant ") and statement.strip().endswith("to bddk_telemetry_writer")
    }
    assert len(telemetry_grants) == 3  # schema USAGE, column INSERT, sequence USAGE
    assert not any(
        statement.startswith(("grant select ", "grant update ", "grant delete ", "grant all "))
        for statement in telemetry_grants
    )
    insert_columns = normalized.split("on table public.tool_call_traces to bddk_telemetry_writer;")[0].rsplit(
        "grant insert (", 1
    )[1]
    assert {column.strip() for column in insert_columns.rstrip(") ").split(",")} == {
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


def test_application_function_ownership_and_execute_are_exact() -> None:
    normalized = _normalized_sql(GRANTS_SQL)

    functions = {
        "public.immutable_unaccent(pg_catalog.text)",
        "public.documents_tsv_trigger()",
        "public.document_sections_tsv_trigger()",
        "public.chunks_tsv_trigger()",
        "public.invalidate_retrieval_publication()",
    }
    for function in functions:
        assert f"alter function {function} owner to bddk_schema_owner;" in normalized
        assert f"revoke all privileges on function {function} from public;" in normalized
    assert "alter function bddk_meta.bump_corpus_state_epoch() owner to bddk_schema_owner;" in normalized
    assert f"alter function {STAGE_ROUTINE} owner to bddk_schema_owner;" in normalized
    assert (
        "alter function bddk_meta.activate_staged_corpus_release(pg_catalog.text) owner to bddk_schema_owner;"
        in normalized
    )
    for table in (
        "bddk_meta.corpus_release_requests",
        "bddk_meta.corpus_release_request_activations",
    ):
        assert f"alter table {table} owner to bddk_schema_owner;" in normalized
    assert "revoke all privileges on function bddk_meta.bump_corpus_state_epoch()" in normalized
    assert "grant execute on function public.immutable_unaccent(pg_catalog.text) to bddk_public_reader;" in normalized
    assert "grant execute on function public.immutable_unaccent(pg_catalog.text) to bddk_ingestion;" in normalized
    for role in ("bddk_public_reader", "bddk_ingestion", "bddk_operator_runtime"):
        assert (
            f"grant execute on function bddk_meta.current_corpus_state_sha256(pg_catalog.text) to {role};"
        ) not in normalized
        assert (
            f"grant execute on function bddk_meta.corpus_retrieval_ready(pg_catalog.text) to {role};"
        ) not in normalized
    assert "grant execute on function public.documents_tsv_trigger()" not in normalized
    legacy = (
        "bddk_meta.publish_verified_corpus_release( pg_catalog.text, pg_catalog.text, "
        "pg_catalog.text, pg_catalog.int4, pg_catalog.int4, pg_catalog.int4, pg_catalog.text )"
    )
    assert f"grant execute on function {STAGE_ROUTINE} to bddk_release_verifier;" in normalized
    assert "grant execute on function bddk_meta.activate_staged_corpus_release(pg_catalog.text) " in normalized
    for role in GROUP_ROLES - {"bddk_schema_owner"}:
        assert f"grant execute on function {legacy} to {role};" not in normalized
        if role != "bddk_release_verifier":
            assert f"grant execute on function {STAGE_ROUTINE} to {role};" not in normalized
        if role != "bddk_release_publisher":
            assert (
                f"grant execute on function bddk_meta.activate_staged_corpus_release(pg_catalog.text) to {role};"
            ) not in normalized
    assert "cross join lateral pg_catalog.aclexplode(" in normalized
    assert "acl.grantee <> routine.proowner" in normalized
    assert "revoke execute on function %s from %i cascade" in normalized
    resolver = "bddk_meta.resolve_regulation_status( pg_catalog.text, pg_catalog.date )"
    assert f"alter function {resolver} owner to bddk_schema_owner;" in normalized
    assert f"grant execute on function {resolver} to bddk_public_reader;" in normalized
    assert f"grant execute on function {resolver} to bddk_ingestion;" not in normalized
    assert f"grant execute on function {resolver} to bddk_operator_runtime;" not in normalized
    assert "grant usage on schema bddk_meta to bddk_release_publisher;" in normalized
    assert "grant usage on schema public to bddk_release_publisher;" not in normalized
    publisher_reads = _grant_object_list(GRANTS_SQL, "select", "bddk_release_publisher")
    assert "bddk_meta.active_corpus_release" in publisher_reads
    assert "bddk_meta.corpus_release_retention_status" in publisher_reads
    assert "alter table bddk_meta.corpus_state_epoch owner to bddk_schema_owner;" in normalized
    retain = "bddk_meta.retain_active_corpus_generation(pg_catalog.text)"
    storage = "bddk_meta.inspect_retained_generation_storage(pg_catalog.text)"
    for function in (retain, storage):
        assert f"alter function {function} owner to bddk_schema_owner;" in normalized
        assert f"grant execute on function {function} to bddk_release_publisher;" in normalized
        assert f"grant execute on function {function} to bddk_ingestion;" not in normalized
        assert f"grant execute on function {function} to bddk_operator_runtime;" not in normalized
    row_hash = "bddk_meta.retained_row_sha256(anyelement, pg_catalog.bool)"
    assert f"alter function {row_hash} owner to bddk_schema_owner;" in normalized
    assert "revoke all privileges on function bddk_meta.retained_row_sha256(" in normalized
    assert f"grant execute on function {row_hash}" not in normalized


def test_deployment_documentation_maps_separate_workload_identities() -> None:
    for secret in (
        "bddk-mcp-schema-owner-db",
        "bddk-mcp-ingestion-db",
        "bddk-mcp-release-publisher-db",
        "bddk-mcp-public-db",
        "bddk-mcp-operator-db",
    ):
        assert secret in README
    assert "SET ROLE bddk_schema_owner" in README
    assert "Do not apply these assets to a\nshared database" in README
    assert "does **not** create a role named `bddk_operator`" in README
    assert "BDDK_SCHEMA_OWNER_DATABASE_URL" in OPENSHIFT_SECRETS
    assert "BDDK_INGESTION_DATABASE_URL" in OPENSHIFT_SECRETS
    assert "bddk-mcp-release-verifier-db" in OPENSHIFT_SECRETS
    assert "BDDK_RELEASE_VERIFIER_DATABASE_URL" in OPENSHIFT_SECRETS
    assert "LOGIN membership: bddk_ingestion" in OPENSHIFT_SECRETS
    assert "LOGIN membership: bddk_release_verifier" in OPENSHIFT_SECRETS
    assert "LOGIN membership: bddk_release_publisher" in OPENSHIFT_SECRETS
    assert "operator LOGIN must never" in OPENSHIFT_SECRETS
    assert "options=-c%20role%3Dbddk_schema_owner" in OPENSHIFT_SECRETS
    assert "cluster-global" in README
    assert "dedicated PostgreSQL cluster/service" in README


def test_ci_requires_disposable_real_login_and_acl_contracts() -> None:
    assert "postgres-role-contract:" in CI_WORKFLOW
    assert 'BDDK_ALLOW_ROLE_PROVISIONING_TEST: "1"' in CI_WORKFLOW
    assert "BDDK_ALLOW_DISPOSABLE_IDENTITY_TEST: I_UNDERSTAND_THIS_MUTATES_A_DISPOSABLE_CLUSTER" in CI_WORKFLOW
    assert "tests/test_postgres_role_assets.py" in CI_WORKFLOW
    assert "-m postgres" in CI_WORKFLOW


_ROLE_TEST_DSN = os.environ.get("BDDK_ROLE_TEST_DATABASE_URL", "")
_RUN_ROLE_TEST = os.environ.get("BDDK_ALLOW_ROLE_PROVISIONING_TEST", "").lower() in {"1", "true", "yes"}
_RUN_DISPOSABLE_LOGIN_TEST = os.environ.get("BDDK_ALLOW_DISPOSABLE_IDENTITY_TEST", "") == (
    "I_UNDERSTAND_THIS_MUTATES_A_DISPOSABLE_CLUSTER"
)


@pytest.mark.postgres
@pytest.mark.skipif(
    not (_ROLE_TEST_DSN and _RUN_ROLE_TEST),
    reason="requires an explicitly approved, dedicated BDDK_ROLE_TEST_DATABASE_URL",
)
async def test_live_role_allow_and_deny_matrix_is_transactional() -> None:
    """Run real migrations and privileges, then roll everything back."""

    connection = await asyncpg.connect(_ROLE_TEST_DSN, timeout=5)
    transaction = connection.transaction()
    await transaction.start()
    try:
        database_name = await connection.fetchval("SELECT current_database()")
        assert database_name not in {"postgres", "template0", "template1"}
        is_superuser = await connection.fetchval("SELECT rolsuper FROM pg_roles WHERE rolname = current_user")
        if not is_superuser:
            pytest.skip("role provisioning test requires a disposable-database superuser")

        existing = await connection.fetchval(
            "SELECT count(*) FROM pg_roles WHERE rolname = ANY($1::text[])",
            sorted(GROUP_ROLES),
        )
        if existing:
            pytest.skip("dedicated role test requires the BDDK group-role names to be absent")

        user_tables = await connection.fetchval(
            """
            SELECT count(*)
            FROM pg_catalog.pg_class AS relation
            JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname NOT IN ('pg_catalog', 'information_schema')
              AND namespace.nspname NOT LIKE 'pg_toast%'
              AND relation.relkind IN ('r', 'p')
            """
        )
        if user_tables:
            pytest.skip("dedicated role test database must not contain user tables")

        await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await connection.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
        await connection.fetchval("SELECT pg_catalog.set_config('bddk.expected_database', current_database(), true)")
        await connection.execute(ROLES_SQL)
        await connection.execute("SET LOCAL ROLE bddk_schema_owner")
        await migrate(_SingleConnectionPool(connection))
        await connection.execute("RESET ROLE")
        await connection.execute("CREATE ROLE bddk_role_test_unprivileged NOLOGIN")
        await connection.execute("INSERT INTO public.documents (document_id, title) VALUES ('existing', 'Title')")
        await connection.execute(GRANTS_SQL)
        sensitive_facades = (
            "bddk_meta.publish_verified_corpus_release(text,text,text,integer,integer,integer,text)",
            "bddk_meta.stage_verified_corpus_release("
            "text,text,text,text,text,integer,integer,integer,text,text,text,integer)",
            "bddk_meta.activate_staged_corpus_release(text)",
        )
        for routine in sensitive_facades:
            await connection.execute(f"GRANT EXECUTE ON FUNCTION {routine} TO bddk_role_test_unprivileged")
        await connection.execute(GRANTS_SQL)
        for routine in sensitive_facades:
            assert not await connection.fetchval(
                "SELECT pg_catalog.has_function_privilege('bddk_role_test_unprivileged', $1, 'EXECUTE')",
                routine,
            )

        roles_with_login = await connection.fetchval(
            "SELECT count(*) FROM pg_roles WHERE rolname = ANY($1::text[]) AND rolcanlogin",
            sorted(GROUP_ROLES),
        )
        assert roles_with_login == 0

        await connection.execute("SET LOCAL ROLE bddk_public_reader")
        assert await connection.fetchval("SELECT document_id FROM public.documents") == "existing"
        assert await connection.fetchval("SELECT public.immutable_unaccent('İSEDES')") == "ISEDES"
        assert await connection.fetchval("SELECT count(*) FROM bddk_meta.active_corpus_release") == 0
        assert (
            await connection.fetchval(
                "SELECT reason FROM bddk_meta.resolve_regulation_status($1, DATE '2024-01-01')",
                "inst_sha256_" + "0" * 64,
            )
            == "instrument_not_found"
        )
        assert await connection.fetchval("SELECT count(*) FROM bddk_meta.schema_migrations") == LATEST_SCHEMA_VERSION
        await _assert_permission_denied(
            connection,
            "INSERT INTO public.documents VALUES ('reader-write')",
        )
        await _assert_permission_denied(connection, "SELECT * FROM public.sync_failures")
        await _assert_corpus_epoch_admin_denied(connection)

        await connection.execute("SET LOCAL ROLE bddk_ingestion")
        await connection.execute(
            """
            INSERT INTO public.document_sections (
                doc_id, section_type, section_ref, start_char, end_char, content,
                content_hash, source_content_hash
            ) VALUES ('existing', 'article', '1', 0, 4, 'test', 'digest', '')
            """
        )
        await _assert_permission_denied(connection, "SELECT * FROM public.tool_call_traces")
        await _assert_permission_denied(connection, "CREATE TABLE public.ingestion_ddl (id integer)")
        await _assert_corpus_epoch_admin_denied(connection)
        assert not await connection.fetchval(
            "SELECT pg_catalog.has_function_privilege("
            "current_user, 'bddk_meta.publish_verified_corpus_release(text,text,text,integer,integer,integer,text)', "
            "'EXECUTE')"
        )

        await connection.execute("SET LOCAL ROLE bddk_release_verifier")
        assert await connection.fetchval(
            "SELECT pg_catalog.has_function_privilege("
            "current_user, 'bddk_meta.stage_verified_corpus_release("
            "text,text,text,text,text,integer,integer,integer,text,text,text,integer)', "
            "'EXECUTE')"
        )
        assert await connection.fetchval("SELECT count(*) FROM public.documents") == 1
        assert await connection.fetchval("SELECT count(*) FROM public.regulatory_instruments") == 0
        await _assert_permission_denied(connection, "INSERT INTO public.documents VALUES ('verifier-write')")
        await _assert_permission_denied(connection, "SELECT * FROM bddk_meta.corpus_release_requests")
        assert not await connection.fetchval(
            "SELECT pg_catalog.has_function_privilege("
            "current_user, 'bddk_meta.activate_staged_corpus_release(text)', 'EXECUTE')"
        )

        await connection.execute("SET LOCAL ROLE bddk_release_publisher")
        assert await connection.fetchval(
            "SELECT pg_catalog.has_function_privilege("
            "current_user, 'bddk_meta.activate_staged_corpus_release(text)', 'EXECUTE')"
        )
        assert not await connection.fetchval(
            "SELECT pg_catalog.has_function_privilege("
            "current_user, 'bddk_meta.publish_verified_corpus_release(text,text,text,integer,integer,integer,text)', "
            "'EXECUTE')"
        )
        assert not await connection.fetchval(
            "SELECT pg_catalog.has_function_privilege("
            "current_user, 'bddk_meta.stage_verified_corpus_release("
            "text,text,text,text,text,integer,integer,integer,text,text,text,integer)', "
            "'EXECUTE')"
        )
        assert await connection.fetchval("SELECT count(*) FROM bddk_meta.active_corpus_release") == 0
        await _assert_permission_denied(connection, "SELECT * FROM public.documents")
        await _assert_permission_denied(connection, "SELECT * FROM public.regulatory_instruments")
        await _assert_permission_denied(connection, "SELECT * FROM bddk_meta.corpus_releases")
        await _assert_permission_denied(connection, "SELECT * FROM bddk_meta.corpus_release_requests")
        await _assert_permission_denied(connection, "SELECT * FROM bddk_meta.corpus_state_epoch")
        assert not await connection.fetchval(
            "SELECT pg_catalog.has_function_privilege(current_user, 'bddk_meta.bump_corpus_state_epoch()', 'EXECUTE')"
        )

        await connection.execute("SET LOCAL ROLE bddk_operator_runtime")
        await connection.execute(
            """
            INSERT INTO bddk_operator.operator_jobs (
                job_id, kind, state, args_fingerprint, created_at, updated_at
            ) VALUES (
                gen_random_uuid(), 'cache_refresh', 'queued', repeat('a', 64), now(), now()
            )
            """
        )
        await _assert_permission_denied(connection, "SELECT * FROM public.documents")
        await _assert_corpus_epoch_admin_denied(connection)
        assert not await connection.fetchval(
            "SELECT pg_catalog.has_function_privilege("
            "current_user, 'bddk_meta.publish_verified_corpus_release(text,text,text,integer,integer,integer,text)', "
            "'EXECUTE')"
        )

        await connection.execute("SET LOCAL ROLE bddk_telemetry_writer")
        await assert_telemetry_writer_ready(  # type: ignore[arg-type]
            _SingleConnectionPool(connection),
            require_session_identity=False,
        )
        await connection.execute(
            "INSERT INTO public.tool_call_traces (tool_name, args_hash) VALUES ('search', 'digest')"
        )
        await _assert_permission_denied(connection, "SELECT * FROM public.tool_call_traces")
        await _assert_corpus_epoch_admin_denied(connection)
        await _assert_permission_denied(
            connection,
            "INSERT INTO public.tool_call_traces (id, tool_name, args_hash) VALUES (99, 'search', 'digest')",
        )

        await connection.execute("SET LOCAL ROLE bddk_role_test_unprivileged")
        await _assert_permission_denied(connection, "SELECT * FROM public.documents")
    finally:
        await transaction.rollback()
        await connection.close()


@pytest.mark.postgres
@pytest.mark.skipif(
    not (_ROLE_TEST_DSN and _RUN_DISPOSABLE_LOGIN_TEST),
    reason="requires explicit approval for a dedicated disposable PostgreSQL cluster",
)
async def test_live_actual_login_identity_and_acl_provenance_contracts() -> None:
    """Provision real LOGINs on a disposable cluster and test pool admission.

    This test intentionally leaves cluster-global roles behind and is therefore
    guarded by an exact acknowledgement string. CI runs it only in a dedicated
    one-job PostgreSQL service that is destroyed immediately afterwards.
    """

    parsed = urlsplit(_ROLE_TEST_DSN)
    database_name = parsed.path.lstrip("/")
    assert parsed.scheme in {"postgres", "postgresql"}
    assert parsed.hostname
    assert database_name == "bddk_role_contract"

    connection_options = {
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "database": database_name,
        "timeout": 5,
    }
    admin = await asyncpg.connect(_ROLE_TEST_DSN, timeout=5)
    pools: list[asyncpg.Pool] = []
    try:
        assert await admin.fetchval("SELECT rolsuper FROM pg_roles WHERE rolname = current_user")
        existing = await admin.fetchval(
            "SELECT count(*) FROM pg_roles WHERE rolname = ANY($1::text[])",
            sorted(GROUP_ROLES),
        )
        assert existing == 0, "disposable identity cluster is not clean"
        user_tables = await admin.fetchval(
            """
            SELECT count(*)
            FROM pg_catalog.pg_class AS relation
            JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname NOT IN ('pg_catalog', 'information_schema')
              AND namespace.nspname NOT LIKE 'pg_toast%'
              AND relation.relkind IN ('r', 'p')
            """
        )
        assert user_tables == 0, "disposable identity database is not empty"

        await admin.execute("CREATE EXTENSION vector")
        await admin.execute("CREATE EXTENSION unaccent")
        await admin.fetchval("SELECT pg_catalog.set_config('bddk.expected_database', current_database(), false)")
        await admin.execute(ROLES_SQL)

        login_roles = {
            "schema": "bddk_identity_schema_login",
            "public": "bddk_identity_public_login",
            "ingestion": "bddk_identity_ingestion_login",
            "release-verifier": "bddk_identity_release_verifier_login",
            "release-publisher": "bddk_identity_release_publisher_login",
            "operator": "bddk_identity_operator_login",
            "telemetry": "bddk_identity_telemetry_login",
            "unprivileged": "bddk_identity_unprivileged_login",
        }
        passwords = {name: secrets.token_urlsafe(32) for name in login_roles}
        for name, role_name in login_roles.items():
            quoted_password = await admin.fetchval(
                "SELECT pg_catalog.quote_literal($1::pg_catalog.text)",
                passwords[name],
            )
            await admin.execute(
                f"CREATE ROLE {role_name} LOGIN INHERIT NOSUPERUSER NOCREATEDB "
                f"NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD {quoted_password}"
            )

        await admin.execute(f"GRANT bddk_schema_owner TO {login_roles['schema']}")
        await admin.execute(f"GRANT bddk_public_reader TO {login_roles['public']}")
        await admin.execute(f"GRANT bddk_ingestion TO {login_roles['ingestion']}")
        await admin.execute(f"GRANT bddk_release_verifier TO {login_roles['release-verifier']}")
        await admin.execute(f"GRANT bddk_release_publisher TO {login_roles['release-publisher']}")
        await admin.execute(
            f"GRANT bddk_public_reader, bddk_ingestion, bddk_operator_runtime TO {login_roles['operator']}"
        )
        await admin.execute(f"GRANT bddk_telemetry_writer TO {login_roles['telemetry']}")

        await admin.execute("SET ROLE bddk_schema_owner")
        try:
            await migrate(_SingleConnectionPool(admin))
        finally:
            await admin.execute("RESET ROLE")
        await admin.execute(GRANTS_SQL)

        async def workload_pool(profile: str) -> asyncpg.Pool:
            role_name = login_roles[profile]
            pool = await asyncpg.create_pool(
                **connection_options,
                user=role_name,
                password=passwords[profile],
                min_size=2,
                max_size=2,
                init=partial(assert_database_connection_identity, profile=profile),
            )
            pools.append(pool)
            return pool

        workload_pools: dict[str, asyncpg.Pool] = {}
        profiles = {"public", "ingestion", "release-verifier", "release-publisher", "operator"}
        for profile in sorted(profiles):
            pool = await workload_pool(profile)
            workload_pools[profile] = pool
            await assert_database_identity(pool, profile)  # type: ignore[arg-type]
            async with pool.acquire() as first, pool.acquire() as second:
                first_pid, second_pid = (
                    await first.fetchval("SELECT pg_backend_pid()"),
                    await second.fetchval("SELECT pg_backend_pid()"),
                )
                assert first_pid != second_pid
                assert await first.fetchval("SELECT session_user") == login_roles[profile]
                assert await second.fetchval("SELECT session_user") == login_roles[profile]
            for wrong_profile in profiles - {profile}:
                with pytest.raises(DatabaseIdentityError):
                    await assert_database_identity(pool, wrong_profile)  # type: ignore[arg-type]

        schema_pool = await asyncpg.create_pool(
            **connection_options,
            user=login_roles["schema"],
            password=passwords["schema"],
            min_size=1,
            max_size=1,
            server_settings={"role": "bddk_schema_owner"},
        )
        pools.append(schema_pool)
        schema_identity = await schema_pool.fetchrow(_SCHEMA_OWNER_IDENTITY_SQL)
        assert schema_identity["current_user_name"] == "bddk_schema_owner"
        assert schema_identity["session_user_name"] == login_roles["schema"]
        await assert_schema_owner_identity(schema_pool, database_name)

        telemetry_pool = await asyncpg.create_pool(
            **connection_options,
            user=login_roles["telemetry"],
            password=passwords["telemetry"],
            min_size=2,
            max_size=2,
            init=assert_telemetry_writer_ready,
        )
        pools.append(telemetry_pool)
        await assert_telemetry_writer_ready(telemetry_pool)

        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await asyncpg.connect(
                **connection_options,
                user=login_roles["unprivileged"],
                password=passwords["unprivileged"],
            )

        elevated_pool = await asyncpg.create_pool(_ROLE_TEST_DSN, min_size=1, max_size=1)
        pools.append(elevated_pool)
        for profile in sorted(profiles):
            with pytest.raises(DatabaseIdentityError):
                await assert_database_identity(elevated_pool, profile)  # type: ignore[arg-type]

        public_pool = workload_pools["public"]
        await admin.execute(f"GRANT SELECT ON public.documents TO {login_roles['public']}")
        try:
            with pytest.raises(DatabaseIdentityError):
                await assert_database_identity(public_pool, "public")  # type: ignore[arg-type]
        finally:
            await admin.execute(f"REVOKE SELECT ON public.documents FROM {login_roles['public']}")
        await assert_database_identity(public_pool, "public")  # type: ignore[arg-type]

        await admin.execute("GRANT SELECT ON public.documents TO PUBLIC")
        try:
            with pytest.raises(DatabaseIdentityError):
                await assert_database_identity(public_pool, "public")  # type: ignore[arg-type]
        finally:
            await admin.execute("REVOKE SELECT ON public.documents FROM PUBLIC")
        await assert_database_identity(public_pool, "public")  # type: ignore[arg-type]

        await admin.execute(f"GRANT SELECT ON public.document_retrieval_publications TO {login_roles['telemetry']}")
        try:
            with pytest.raises(RuntimeError, match="INSERT-only"):
                await assert_telemetry_writer_ready(telemetry_pool)
        finally:
            await admin.execute(
                f"REVOKE SELECT ON public.document_retrieval_publications FROM {login_roles['telemetry']}"
            )
        await assert_telemetry_writer_ready(telemetry_pool)
    finally:
        for pool in reversed(pools):
            await pool.close()
        await admin.close()


class _SingleConnectionPool:
    """Pool-shaped adapter that keeps migration work inside the test rollback."""

    def __init__(self, connection: asyncpg.Connection) -> None:
        self._connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self._connection

    async def fetchrow(self, query: str, *args):
        return await self._connection.fetchrow(query, *args)

    async def fetchval(self, query: str, *args):
        return await self._connection.fetchval(query, *args)


async def _assert_corpus_epoch_admin_denied(connection: asyncpg.Connection) -> None:
    await _assert_permission_denied(connection, "SELECT * FROM bddk_meta.corpus_state_epoch")
    for function_name in (
        "bump_corpus_state_epoch",
        "current_corpus_state_sha256",
        "corpus_retrieval_ready",
    ):
        assert not await connection.fetchval(
            """
            SELECT pg_catalog.has_function_privilege(current_user, routine.oid, 'EXECUTE')
            FROM pg_catalog.pg_proc AS routine
            JOIN pg_catalog.pg_namespace AS namespace
              ON namespace.oid = routine.pronamespace
            WHERE namespace.nspname = 'bddk_meta'
              AND routine.proname = $1
            """,
            function_name,
        )


async def _assert_permission_denied(connection: asyncpg.Connection, statement: str) -> None:
    """Keep the outer role test usable after one expected SQLSTATE 42501."""

    savepoint = connection.transaction()
    await savepoint.start()
    try:
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await connection.execute(statement)
    finally:
        await savepoint.rollback()
