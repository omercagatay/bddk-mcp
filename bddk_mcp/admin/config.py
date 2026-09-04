"""Configuration for the admin console, resolved once at startup."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from bddk_mcp.db_transport import DatabaseTransportError, assert_database_transport
from bddk_mcp.http_security import is_loopback_host

DEFAULT_ADMIN_PORT = 8100


class AdminConfigError(RuntimeError):
    """The admin console cannot start under the supplied configuration."""


@dataclass(frozen=True, slots=True)
class AdminConfig:
    """Resolved, immutable admin console settings."""

    bind_host: str
    port: int
    database_url: str
    loopback_only: bool

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AdminConfig:
        source = os.environ if env is None else env

        database_url = (source.get("BDDK_DATABASE_URL") or "").strip()
        if not database_url:
            raise AdminConfigError("BDDK_DATABASE_URL must be set to run the admin console.")
        try:
            database_url = assert_database_transport(database_url)
        except DatabaseTransportError as exc:
            raise AdminConfigError(str(exc)) from None

        bind_host = (source.get("BDDK_ADMIN_HOST") or "127.0.0.1").strip()
        loopback_only = is_loopback_host(bind_host)

        raw_port = (source.get("BDDK_ADMIN_PORT") or "").strip()
        try:
            port = int(raw_port) if raw_port else DEFAULT_ADMIN_PORT
        except ValueError:
            raise AdminConfigError("BDDK_ADMIN_PORT must be an integer.") from None
        if not 1 <= port <= 65535:
            raise AdminConfigError("BDDK_ADMIN_PORT must be between 1 and 65535.")

        if not loopback_only:
            # Administration can sign a corpus, so it is authenticated or
            # loopback-only by construction. Remote exposure arrives with the
            # deployment slice, never as an accidental default.
            raise AdminConfigError(
                "The admin console must be authenticated or loopback-only; "
                "a non-loopback bind is not supported in this release."
            )

        return cls(bind_host=bind_host, port=port, database_url=database_url, loopback_only=loopback_only)
