"""Fail-closed PostgreSQL transport validation for non-test entry points."""

from __future__ import annotations

import os
import ssl
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, urlsplit

_ENV_CA_PATH = Path("/tmp/bddk-db-ca.pem")
_MAX_ENV_CA_BYTES = 64 * 1024


class DatabaseTransportError(RuntimeError):
    """The configured PostgreSQL DSN does not authenticate its server."""


def insecure_database_transport_allowed() -> bool:
    """Return the explicit local-development escape hatch."""

    return os.environ.get("BDDK_ALLOW_INSECURE_DATABASE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _materialize_environment_ca() -> None:
    """Prepare an explicitly supplied public CA for ephemeral deployment processes.

    All callers retain verify-full and hostname checks. Atomic replacement never
    follows an existing destination symlink or overwrites an unrelated CA path.
    """
    pem = os.environ.get("BDDK_DATABASE_CA_PEM")
    if pem is None:
        return
    temporary = None
    try:
        encoded = pem.encode("ascii")
        if not 1 <= len(encoded) <= _MAX_ENV_CA_BYTES or "PRIVATE KEY" in pem:
            raise ValueError("invalid CA material")
        ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT).load_verify_locations(cadata=pem)
        with tempfile.NamedTemporaryFile(prefix=".bddk-db-ca-", dir=_ENV_CA_PATH.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, _ENV_CA_PATH)
    except (OSError, ValueError):
        raise DatabaseTransportError("BDDK_DATABASE_CA_PEM could not be validated or prepared safely.") from None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


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
    if root == str(_ENV_CA_PATH):
        _materialize_environment_ca()
    return dsn
