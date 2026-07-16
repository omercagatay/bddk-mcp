"""Fail-closed PostgreSQL server-version compatibility contract.

PostgreSQL behavior, catalog shape, privileges, and extension packaging vary
between major releases.  The application must therefore admit only major
versions exercised by its required database test matrix.  Keep this check at
each physical connection boundary instead of trusting a DSN, proxy target, or
one startup connection to represent every backend a pool may receive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

SUPPORTED_POSTGRESQL_MAJOR_VERSIONS: Final[frozenset[int]] = frozenset({17})

_SERVER_VERSION_NUM_SQL: Final[str] = "SELECT pg_catalog.current_setting('server_version_num')::pg_catalog.int4"


class PostgreSQLCompatibilityError(RuntimeError):
    """The backend version is unsupported or could not be proved safely."""


@dataclass(frozen=True, slots=True)
class PostgreSQLCompatibility:
    """Validated, non-secret PostgreSQL version metadata."""

    server_version_num: int
    major_version: int


def _required_major_text() -> str:
    majors = sorted(SUPPORTED_POSTGRESQL_MAJOR_VERSIONS)
    if len(majors) == 1:
        return str(majors[0])
    return ", ".join(str(major) for major in majors[:-1]) + f", or {majors[-1]}"


async def inspect_postgresql_compatibility(connection: Any) -> PostgreSQLCompatibility:
    """Read and validate the backend's numeric version using one SELECT."""

    try:
        raw_version = await connection.fetchval(_SERVER_VERSION_NUM_SQL)
        if isinstance(raw_version, bool):
            raise ValueError("boolean is not a PostgreSQL version number")
        version_number = int(raw_version)
        major_version = version_number // 10_000
        if version_number < 100_000 or major_version < 10:
            raise ValueError("invalid PostgreSQL version number")
    except Exception:
        raise PostgreSQLCompatibilityError(
            "PostgreSQL server compatibility could not be verified; database operation refused."
        ) from None

    return PostgreSQLCompatibility(
        server_version_num=version_number,
        major_version=major_version,
    )


async def assert_supported_postgresql(connection: Any) -> PostgreSQLCompatibility:
    """Fail closed unless the connected backend belongs to the tested matrix."""

    compatibility = await inspect_postgresql_compatibility(connection)
    if compatibility.major_version not in SUPPORTED_POSTGRESQL_MAJOR_VERSIONS:
        raise PostgreSQLCompatibilityError(
            "Unsupported PostgreSQL major version; this BDDK MCP release requires PostgreSQL "
            + _required_major_text()
            + "."
        )
    return compatibility
