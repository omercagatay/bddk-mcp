"""Exact PostgreSQL workload-identity contract tests."""

from __future__ import annotations

import os
from dataclasses import replace
from types import MappingProxyType, SimpleNamespace
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from bddk_mcp import db_identity
from bddk_mcp.db_identity import (
    DatabaseIdentityError,
    DatabaseIdentityInspection,
    assert_database_connection_identity,
    assert_database_identity,
    assert_release_publication_connection_identity,
    identity_contract_failures,
    release_publication_identity_contract_failures,
)


def _contract_inspection(contract, *, login: str = "bank_workload_login") -> DatabaseIdentityInspection:
    return DatabaseIdentityInspection(
        current_user=login,
        session_user=login,
        session_can_login=True,
        session_inherits=True,
        direct_memberships=contract.memberships,
        inherited_memberships=contract.memberships,
        unsafe_roles=frozenset(),
        membership_admin=False,
        public_acl_leakage=False,
        direct_login_acl=False,
        database_privileges=frozenset({"CONNECT"}),
        schemas=contract.schemas,
        tables=contract.tables,
        sequences=contract.sequences,
        routines=contract.routines,
    )


def _valid_inspection(profile: str, *, login: str = "bank_workload_login") -> DatabaseIdentityInspection:
    return _contract_inspection(db_identity._CONTRACTS[profile], login=login)


@pytest.mark.parametrize(
    "profile",
    ["public", "ingestion", "release-verifier", "release-publisher", "operator"],
)
def test_reviewed_identity_contracts_are_exact_and_satisfiable(profile: str) -> None:
    assert identity_contract_failures(_valid_inspection(profile), profile) == ()


def test_canonical_legal_version_workspace_is_inventoried_with_zero_runtime_rights() -> None:
    expected = {
        "public.regulatory_evidence",
        "public.regulatory_family_imports",
        "public.regulatory_instruments",
        "public.regulatory_legal_events",
        "public.regulatory_legal_status_assertions",
        "public.regulatory_legal_version_artifacts",
        "public.regulatory_legal_version_provisions",
        "public.regulatory_legal_versions",
        "public.regulatory_provisions",
        "public.regulatory_relations",
        "public.regulatory_source_blobs",
        "public.regulatory_source_artifacts",
    }

    assert db_identity._REGULATORY_VERSION_TABLES == expected
    assert not expected.intersection(db_identity._CORPUS_TABLES)
    assert not expected.intersection(db_identity._INGESTION_TABLES)
    for profile in ("public", "ingestion", "operator"):
        contract = db_identity._CONTRACTS[profile]
        assert {table: contract.tables[table] for table in expected} == {table: frozenset() for table in expected}
    verifier = db_identity._CONTRACTS["release-verifier"]
    assert {table: verifier.tables[table] for table in expected} == {table: frozenset({"SELECT"}) for table in expected}
    publisher = db_identity._CONTRACTS["release-publisher"]
    assert {table: publisher.tables[table] for table in expected} == {table: frozenset() for table in expected}

    view = "public.regulatory_validated_section_citations"
    assert db_identity._REGULATORY_PUBLIC_VIEWS == {view}
    assert db_identity._CONTRACTS["public"].tables[view] == frozenset({"SELECT"})
    assert db_identity._CONTRACTS["ingestion"].tables[view] == frozenset()
    assert db_identity._CONTRACTS["release-verifier"].tables[view] == frozenset()
    assert db_identity._CONTRACTS["release-publisher"].tables[view] == frozenset()
    assert db_identity._CONTRACTS["operator"].tables[view] == frozenset({"SELECT"})


def test_retained_generation_store_is_exactly_inventoried_and_runtime_denied() -> None:
    retained_tables = {f"bddk_retained.{relation}" for relation in db_identity.RETAINED_CORPUS_RELATIONS}

    assert len(retained_tables) == 17
    assert db_identity._RETAINED_CORPUS_TABLES == retained_tables
    for profile in ("public", "ingestion", "release-verifier", "release-publisher", "operator"):
        contract = db_identity._CONTRACTS[profile]
        assert contract.schemas["bddk_retained"] == frozenset()
        assert {table: contract.tables[table] for table in retained_tables} == {
            table: frozenset() for table in retained_tables
        }

    retention_view = "bddk_meta.corpus_release_retention_status"
    retain_routine = "bddk_meta.retain_active_corpus_generation(text)"
    storage_routine = "bddk_meta.inspect_retained_generation_storage(text)"
    for profile in ("public", "ingestion", "release-verifier", "operator"):
        contract = db_identity._CONTRACTS[profile]
        assert contract.tables[retention_view] == frozenset()
        assert contract.routines[retain_routine] == frozenset()
        assert contract.routines[storage_routine] == frozenset()

    publisher = db_identity._CONTRACTS["release-publisher"]
    assert publisher.tables[retention_view] == frozenset({"SELECT"})
    assert publisher.routines[retain_routine] == frozenset({"EXECUTE"})
    assert publisher.routines[storage_routine] == frozenset({"EXECUTE"})
    assert publisher.routines["bddk_meta.retained_corpus_state_sha256(text, text)"] == frozenset()
    assert publisher.routines["bddk_meta.retained_row_sha256(anyelement, boolean)"] == frozenset()
    assert publisher.routines["bddk_meta.guard_retained_generation_member()"] == frozenset()
    assert publisher.routines["bddk_meta.reject_retained_generation_mutation()"] == frozenset()


def test_release_verifier_and_publisher_capabilities_are_mutually_separated() -> None:
    stage = (
        "bddk_meta.stage_verified_corpus_release(text, text, text, text, text, text, integer, integer, "
        "integer, text, text, text, integer)"
    )
    activate = "bddk_meta.activate_staged_corpus_release(text)"
    legacy = "bddk_meta.publish_verified_corpus_release(text, text, text, integer, integer, integer, text)"
    verifier = db_identity._CONTRACTS["release-verifier"]
    publisher = db_identity._CONTRACTS["release-publisher"]

    assert verifier.routines[stage] == frozenset({"EXECUTE"})
    assert verifier.routines[activate] == frozenset()
    assert publisher.routines[stage] == frozenset()
    assert publisher.routines[activate] == frozenset({"EXECUTE"})
    assert verifier.routines[legacy] == frozenset()
    assert publisher.routines[legacy] == frozenset()
    for table in db_identity._CORPUS_TABLES | db_identity._REGULATORY_VERSION_TABLES:
        assert verifier.tables[table] == frozenset({"SELECT"})
        assert publisher.tables[table] == frozenset()
    for table in (
        "bddk_meta.corpus_release_requests",
        "bddk_meta.corpus_release_request_activations",
    ):
        assert verifier.tables[table] == frozenset()
        assert publisher.tables[table] == frozenset()


@pytest.mark.parametrize(
    ("schema_version", "contract"),
    (
        (5, db_identity._V5_RELEASE_PUBLISHER_CONTRACT),
        (6, db_identity._V6_RELEASE_PUBLISHER_CONTRACT),
        (7, db_identity._V7_RELEASE_PUBLISHER_CONTRACT),
    ),
)
def test_publication_only_identity_contracts_are_exact_by_schema_version(schema_version: int, contract) -> None:
    inspection = _contract_inspection(contract)

    assert (
        release_publication_identity_contract_failures(
            inspection,
            schema_version=schema_version,
        )
        == ()
    )
    assert release_publication_identity_contract_failures(
        inspection,
        schema_version=4,
    ) == ("unsupported_schema_version",)

    if schema_version < 7:
        assert "bddk_retained" not in contract.schemas
        assert not db_identity._V7_ONLY_TABLES.intersection(contract.tables)
        assert not db_identity._V7_ONLY_ROUTINES.intersection(contract.routines)
    if schema_version == 5:
        assert "bddk_meta.resolve_regulation_status(text, date)" not in contract.routines


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("schema_version", "contract"),
    (
        (5, db_identity._V5_RELEASE_PUBLISHER_CONTRACT),
        (6, db_identity._V6_RELEASE_PUBLISHER_CONTRACT),
        (7, db_identity._V7_RELEASE_PUBLISHER_CONTRACT),
    ),
)
async def test_publication_connection_identity_selects_only_the_exact_compatible_contract(
    monkeypatch,
    schema_version: int,
    contract,
) -> None:
    monkeypatch.setattr(
        db_identity,
        "inspect_migration_state",
        AsyncMock(return_value=SimpleNamespace(current_version=schema_version)),
    )
    monkeypatch.setattr(
        db_identity,
        "inspect_database_connection_identity",
        AsyncMock(return_value=_contract_inspection(contract)),
    )

    await assert_release_publication_connection_identity(SimpleNamespace())


@pytest.mark.asyncio
async def test_publication_connection_identity_rejects_other_schema_versions(monkeypatch) -> None:
    inspection = AsyncMock()
    monkeypatch.setattr(
        db_identity,
        "inspect_migration_state",
        AsyncMock(return_value=SimpleNamespace(current_version=4)),
    )
    monkeypatch.setattr(db_identity, "inspect_database_connection_identity", inspection)

    with pytest.raises(DatabaseIdentityError, match="supported schema version"):
        await assert_release_publication_connection_identity(SimpleNamespace())

    inspection.assert_not_awaited()


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("current_user", "set_role_identity", "session_role_changed"),
        ("session_can_login", False, "login_attributes"),
        ("session_inherits", False, "login_attributes"),
        ("unsafe_roles", frozenset({"bank_workload_login"}), "unsafe_role_attributes"),
        ("membership_admin", True, "membership_admin"),
        ("public_acl_leakage", True, "public_acl_leakage"),
        ("direct_login_acl", True, "direct_login_acl"),
        ("direct_memberships", frozenset({"bddk_public_reader", "pg_read_all_data"}), "direct_memberships"),
        (
            "inherited_memberships",
            frozenset({"bddk_public_reader", "pg_read_all_data"}),
            "inherited_memberships",
        ),
        ("database_privileges", frozenset({"CONNECT", "TEMPORARY"}), "database_privileges"),
    ],
)
def test_identity_and_membership_escalations_are_rejected(field: str, value: object, failure: str) -> None:
    inspection = replace(_valid_inspection("public"), **{field: value})

    assert failure in identity_contract_failures(inspection, "public")


@pytest.mark.parametrize("inventory", ["schemas", "tables", "sequences", "routines"])
def test_unexpected_application_objects_are_rejected(inventory: str) -> None:
    inspection = _valid_inspection("public")
    changed = dict(getattr(inspection, inventory))
    changed[f"public.unreviewed_{inventory}"] = frozenset()

    failures = identity_contract_failures(
        replace(inspection, **{inventory: MappingProxyType(changed)}),
        "public",
    )

    assert f"{inventory.removesuffix('s')}_privileges" in failures


def test_missing_and_overbroad_object_rights_are_rejected() -> None:
    inspection = _valid_inspection("public")
    missing = dict(inspection.tables)
    missing["public.documents"] = frozenset()
    overbroad = dict(inspection.tables)
    overbroad["public.documents"] = frozenset({"SELECT", "INSERT"})
    column_only = dict(inspection.tables)
    column_only["public.tool_call_traces"] = frozenset({"INSERT_COLUMNS"})

    assert "table_privileges" in identity_contract_failures(
        replace(inspection, tables=MappingProxyType(missing)),
        "public",
    )
    assert "table_privileges" in identity_contract_failures(
        replace(inspection, tables=MappingProxyType(overbroad)),
        "public",
    )
    assert "table_privileges" in identity_contract_failures(
        replace(inspection, tables=MappingProxyType(column_only)),
        "public",
    )


def test_operator_login_cannot_pass_as_public_even_with_a_different_dsn_string() -> None:
    operator = _valid_inspection("operator", login="same_actual_login")

    failures = identity_contract_failures(operator, "public")

    assert "direct_memberships" in failures
    assert "table_privileges" in failures
    assert "schema_privileges" in failures


@pytest.mark.asyncio
async def test_assertion_fails_closed_without_leaking_login_or_acl_details() -> None:
    inspection = replace(
        _valid_inspection("public", login="private_login_name"),
        database_privileges=frozenset({"CONNECT", "CREATE"}),
    )

    with (
        patch.object(db_identity, "inspect_database_identity", new=AsyncMock(return_value=inspection)),
        pytest.raises(DatabaseIdentityError) as exc_info,
    ):
        await assert_database_identity(AsyncMock(), "public")

    message = str(exc_info.value)
    assert "private_login_name" not in message
    assert "CREATE" not in message
    assert "least-privilege contract" in message


@pytest.mark.asyncio
async def test_assertion_sanitizes_catalog_inspection_failures() -> None:
    with (
        patch.object(
            db_identity,
            "inspect_database_identity",
            new=AsyncMock(side_effect=asyncpg.ConnectionDoesNotExistError("private database detail")),
        ),
        pytest.raises(DatabaseIdentityError) as exc_info,
    ):
        await assert_database_identity(AsyncMock(), "ingestion")

    assert "private database detail" not in str(exc_info.value)
    assert "could not be verified" in str(exc_info.value)


@pytest.mark.asyncio
async def test_physical_connection_assertion_accepts_exact_profile() -> None:
    inspection = _valid_inspection("operator")
    connection = AsyncMock()

    with patch.object(
        db_identity,
        "inspect_database_connection_identity",
        new=AsyncMock(return_value=inspection),
    ) as inspect:
        await assert_database_connection_identity(connection, profile="operator")

    inspect.assert_awaited_once_with(connection)


@pytest.mark.asyncio
async def test_physical_connection_assertion_rejects_and_sanitizes_drift() -> None:
    inspection = replace(
        _valid_inspection("public", login="private_remapped_login"),
        database_privileges=frozenset({"CONNECT", "CREATE"}),
    )

    with (
        patch.object(
            db_identity,
            "inspect_database_connection_identity",
            new=AsyncMock(return_value=inspection),
        ),
        pytest.raises(DatabaseIdentityError) as exc_info,
    ):
        await assert_database_connection_identity(AsyncMock(), profile="public")

    assert "private_remapped_login" not in str(exc_info.value)
    assert "CREATE" not in str(exc_info.value)
    assert "least-privilege contract" in str(exc_info.value)


@pytest.mark.asyncio
async def test_physical_connection_refuses_unsupported_postgresql_before_acl_inspection() -> None:
    connection = AsyncMock()
    connection.fetchval.return_value = 160012

    with pytest.raises(DatabaseIdentityError) as exc_info:
        await assert_database_connection_identity(connection, profile="public")

    assert "requires PostgreSQL 17" in str(exc_info.value)
    assert "160012" not in str(exc_info.value)
    connection.fetchrow.assert_not_awaited()
    connection.fetch.assert_not_awaited()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_acl_provenance_inspection_executes_against_postgresql(pg_pool) -> None:
    """The catalog query must detect unreviewed ACL sources, not just effective rights."""

    async with pg_pool.acquire() as connection:
        row = await connection.fetchrow(db_identity._ACL_PROVENANCE_SQL)

    assert isinstance(row["public_acl_leakage"], bool)
    assert isinstance(row["direct_login_acl"], bool)
    # The generic test database intentionally uses its owner and PostgreSQL's
    # defaults instead of the production role assets, so both signals should
    # remain visible here. Dedicated workload-login tests prove the inverse.
    assert row["public_acl_leakage"]
    assert row["direct_login_acl"]


_RUN_LIVE_IDENTITY_TEST = os.environ.get("BDDK_ALLOW_IDENTITY_LOGIN_TEST", "").lower() in {"1", "true", "yes"}
_LIVE_IDENTITY_DSNS = {
    "public": os.environ.get("BDDK_PUBLIC_IDENTITY_TEST_DATABASE_URL", ""),
    "ingestion": os.environ.get("BDDK_INGESTION_IDENTITY_TEST_DATABASE_URL", ""),
    "release-verifier": os.environ.get("BDDK_RELEASE_VERIFIER_IDENTITY_TEST_DATABASE_URL", ""),
    "release-publisher": os.environ.get("BDDK_RELEASE_PUBLISHER_IDENTITY_TEST_DATABASE_URL", ""),
    "operator": os.environ.get("BDDK_OPERATOR_IDENTITY_TEST_DATABASE_URL", ""),
}


@pytest.mark.postgres
@pytest.mark.skipif(
    not (_RUN_LIVE_IDENTITY_TEST and all(_LIVE_IDENTITY_DSNS.values())),
    reason="requires explicitly approved DSNs for all five real workload LOGINs",
)
@pytest.mark.asyncio
async def test_live_workload_login_contracts() -> None:
    """Prove contracts with actual LOGINs; SET ROLE is intentionally insufficient."""

    for profile, dsn in _LIVE_IDENTITY_DSNS.items():
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=1, timeout=5)
        try:
            await assert_database_identity(pool, profile)  # type: ignore[arg-type]
            for other_profile in set(_LIVE_IDENTITY_DSNS) - {profile}:
                with pytest.raises(DatabaseIdentityError):
                    await assert_database_identity(pool, other_profile)  # type: ignore[arg-type]
        finally:
            await pool.close()


_REJECTED_IDENTITY_DSN = os.environ.get("BDDK_REJECTED_IDENTITY_TEST_DATABASE_URL", "")


@pytest.mark.postgres
@pytest.mark.skipif(
    not (_RUN_LIVE_IDENTITY_TEST and _REJECTED_IDENTITY_DSN),
    reason="requires an explicitly approved elevated or cross-profile LOGIN DSN",
)
@pytest.mark.asyncio
async def test_live_wrong_or_elevated_login_is_rejected_for_every_profile() -> None:
    pool = await asyncpg.create_pool(_REJECTED_IDENTITY_DSN, min_size=1, max_size=1, timeout=5)
    try:
        for profile in ("public", "ingestion", "release-verifier", "release-publisher", "operator"):
            with pytest.raises(DatabaseIdentityError):
                await assert_database_identity(pool, profile)
    finally:
        await pool.close()
