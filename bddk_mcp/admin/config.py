"""Configuration for the admin console, resolved once at startup."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from bddk_mcp.db_transport import DatabaseTransportError, assert_database_transport
from bddk_mcp.http_security import (
    HttpSecurityConfig,
    HttpSecurityConfigError,
    is_loopback_host,
    load_http_security_config,
)

DEFAULT_ADMIN_PORT = 8100
_REMOTE_OPT_IN = frozenset({"1", "true", "yes"})


class AdminConfigError(RuntimeError):
    """The admin console cannot start under the supplied configuration."""


@dataclass(frozen=True, slots=True)
class AdminConfig:
    """Resolved, immutable admin console settings."""

    bind_host: str
    port: int
    database_url: str
    loopback_only: bool
    http_security: HttpSecurityConfig | None = None

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

        raw_port = (source.get("BDDK_ADMIN_PORT") or source.get("PORT") or "").strip()
        try:
            port = int(raw_port) if raw_port else DEFAULT_ADMIN_PORT
        except ValueError:
            raise AdminConfigError("BDDK_ADMIN_PORT must be an integer.") from None
        if not 1 <= port <= 65535:
            raise AdminConfigError("BDDK_ADMIN_PORT must be between 1 and 65535.")

        http_security = None if loopback_only else _remote_http_security(source, bind_host=bind_host, port=port)
        return cls(
            bind_host=bind_host,
            port=port,
            database_url=database_url,
            loopback_only=loopback_only,
            http_security=http_security,
        )


def _flag(source: Mapping[str, str], name: str) -> bool:
    return source.get(name, "").strip().lower() in _REMOTE_OPT_IN


def _remote_http_security(source: Mapping[str, str], *, bind_host: str, port: int) -> HttpSecurityConfig:
    if not _flag(source, "BDDK_ADMIN_REMOTE_ENABLED"):
        raise AdminConfigError(
            "The admin console must be authenticated or loopback-only; "
            "a non-loopback bind requires BDDK_ADMIN_REMOTE_ENABLED=true."
        )
    if _flag(source, "BDDK_HTTP_ALLOW_UNAUTHENTICATED"):
        raise AdminConfigError("The admin console cannot run unauthenticated on a non-loopback bind.")
    overlay = dict(source)
    overlay["MCP_HOST"] = bind_host
    overlay["PORT"] = str(port)
    try:
        http_security = load_http_security_config(overlay)
    except HttpSecurityConfigError as exc:
        raise AdminConfigError(str(exc)) from exc
    if "bddk.operator" not in http_security.jwt_required_scopes:
        raise AdminConfigError("Remote admin HTTP requires JWT scope 'bddk.operator'.")
    return http_security
