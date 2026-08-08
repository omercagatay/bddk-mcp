"""PostgreSQL major-version admission contract tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

import asyncpg
import pytest

from bddk_mcp.db_compatibility import (
    SUPPORTED_POSTGRESQL_MAJOR_VERSIONS,
    PostgreSQLCompatibility,
    PostgreSQLCompatibilityError,
    assert_supported_postgresql,
    inspect_postgresql_compatibility,
)


def test_supported_postgresql_matrix_is_explicitly_postgresql_17_only() -> None:
    assert SUPPORTED_POSTGRESQL_MAJOR_VERSIONS == frozenset({17})


@pytest.mark.asyncio
async def test_inspection_parses_numeric_postgresql_17_version_with_one_select() -> None:
    connection = AsyncMock()
    connection.fetchval.return_value = "170006"

    result = await inspect_postgresql_compatibility(connection)

    assert result == PostgreSQLCompatibility(server_version_num=170006, major_version=17)
    connection.fetchval.assert_awaited_once()
    query = connection.fetchval.await_args.args[0]
    assert query == "SELECT pg_catalog.current_setting('server_version_num')::pg_catalog.int4"


@pytest.mark.asyncio
@pytest.mark.parametrize("server_version_num", [160012, 180000])
async def test_unsupported_major_versions_are_rejected_without_disclosing_actual_version(
    server_version_num: int,
) -> None:
    connection = AsyncMock()
    connection.fetchval.return_value = server_version_num

    with pytest.raises(PostgreSQLCompatibilityError) as exc_info:
        await assert_supported_postgresql(connection)

    message = str(exc_info.value)
    assert "requires PostgreSQL 17" in message
    assert str(server_version_num) not in message
    assert str(server_version_num // 10_000) not in message


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_version", [None, True, "private-backend-version"])
async def test_unverifiable_versions_and_driver_details_are_sanitized(invalid_version: object) -> None:
    connection = AsyncMock()
    connection.fetchval.return_value = invalid_version

    with pytest.raises(PostgreSQLCompatibilityError) as exc_info:
        await assert_supported_postgresql(connection)

    message = str(exc_info.value)
    assert str(invalid_version) not in message
    assert exc_info.value.__cause__ is None
    assert "could not be verified" in message


@pytest.mark.asyncio
async def test_driver_failure_is_sanitized() -> None:
    sentinel = "postgresql://private:password@secret-bank-host/corpus"
    connection = AsyncMock()
    connection.fetchval.side_effect = asyncpg.PostgresError(sentinel)

    with pytest.raises(PostgreSQLCompatibilityError) as exc_info:
        await assert_supported_postgresql(connection)

    assert sentinel not in str(exc_info.value)
    assert "password" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_live_required_database_is_postgresql_17(pg_pool) -> None:
    compatibility = await assert_supported_postgresql(pg_pool)

    assert compatibility.major_version == 17
    assert 170000 <= compatibility.server_version_num < 180000
