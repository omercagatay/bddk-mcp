"""Exact PostgreSQL workload-identity contract tests."""

from __future__ import annotations

import os
from dataclasses import replace
from types import MappingProxyType
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from bddk_mcp import db_identity
from bddk_mcp.db_identity import (
    DatabaseIdentityError,
    DatabaseIdentityInspection,
    assert_database_connection_identity,
    assert_database_identity,
    identity_contract_failures,
)


def _valid_inspection(profile: str, *, login: str = "bank_workload_login") -> DatabaseIdentityInspection:
    contract = db_identity._CONTRACTS[profile]
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


@pytest.mark.parametrize("profile", ["public", "ingestion", "operator"])
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
        "public.regulatory_source_blobs",
        "public.regulatory_source_artifacts",
    }

    assert db_identity._REGULATORY_VERSION_TABLES == expected
    assert not expected.intersection(db_identity._CORPUS_TABLES)
    assert not expected.intersection(db_identity._INGESTION_TABLES)
    for profile in ("public", "ingestion", "operator"):
        contract = db_identity._CONTRACTS[profile]
        assert {table: contract.tables[table] for table in expected} == {table: frozenset() for table in expected}

    view = "public.regulatory_validated_section_citations"
    assert db_identity._REGULATORY_PUBLIC_VIEWS == {view}
    assert db_identity._CONTRACTS["public"].tables[view] == frozenset({"SELECT"})
    assert db_identity._CONTRACTS["ingestion"].tables[view] == frozenset()
    assert db_identity._CONTRACTS["operator"].tables[view] == frozenset({"SELECT"})


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
    "operator": os.environ.get("BDDK_OPERATOR_IDENTITY_TEST_DATABASE_URL", ""),
}


@pytest.mark.postgres
@pytest.mark.skipif(
    not (_RUN_LIVE_IDENTITY_TEST and all(_LIVE_IDENTITY_DSNS.values())),
    reason="requires explicitly approved DSNs for all three real workload LOGINs",
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
        for profile in ("public", "ingestion", "operator"):
            with pytest.raises(DatabaseIdentityError):
                await assert_database_identity(pool, profile)
    finally:
        await pool.close()
