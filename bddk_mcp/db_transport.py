"""Fail-closed PostgreSQL transport validation for non-test entry points."""

from __future__ import annotations

import os
from pathlib import PurePosixPath
from urllib.parse import parse_qs, urlsplit


class DatabaseTransportError(RuntimeError):
    """The configured PostgreSQL DSN does not authenticate its server."""


def insecure_database_transport_allowed() -> bool:
    """Return the explicit local-development escape hatch."""

    return os.environ.get("BDDK_ALLOW_INSECURE_DATABASE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def assert_database_transport(dsn: str) -> str:
    """Require verified TLS unless the local-only escape hatch is explicit."""

    if insecure_database_transport_allowed():
        return dsn
    try:
        parsed = urlsplit(dsn)
        query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=False)
    except (TypeError, ValueError):
        parsed = None
        query = {}
    ssl_modes = query.get("sslmode", [])
    roots = query.get("sslrootcert", [])
    root = roots[0] if len(roots) == 1 else ""
    valid = (
        parsed is not None
        and parsed.scheme in {"postgres", "postgresql"}
        and bool(parsed.hostname)
        and ssl_modes == ["verify-full"]
        and bool(root)
        and PurePosixPath(root).is_absolute()
    )
    if not valid:
        raise DatabaseTransportError(
            "PostgreSQL requires sslmode=verify-full and an absolute sslrootcert path. "
            "BDDK_ALLOW_INSECURE_DATABASE=true is permitted only for isolated local development."
        )
    return dsn
