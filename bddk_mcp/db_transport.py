"""Fail-closed PostgreSQL transport validation for non-test entry points."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
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


def materialize_postgres_ca_from_env() -> None:
    """Write BDDK_POSTGRES_CA_CERT to an absolute file before verify-full connects.

    Railway has no separate CA mount. The PEM stays in a secret variable; the DSN
    still names an absolute sslrootcert path. Unset means the operator mounted the
    file themselves.
    """

    pem = os.environ.get("BDDK_POSTGRES_CA_CERT")
    if not pem or not pem.strip():
        return
    path = os.environ.get("BDDK_POSTGRES_CA_PATH", "/tmp/bddk-db-ca.pem")
    if not PurePosixPath(path).is_absolute():
        raise DatabaseTransportError("BDDK_POSTGRES_CA_PATH must be absolute.")
    data = pem.replace("\\n", "\n").encode()
    if b"BEGIN CERTIFICATE" not in data:
        raise DatabaseTransportError("BDDK_POSTGRES_CA_CERT is not a PEM certificate.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data if data.endswith(b"\n") else data + b"\n")
    finally:
        os.close(fd)


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
