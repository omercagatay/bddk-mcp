"""Configuration for the admin console, resolved once at startup."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

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
    password: str = ""
    allowed_hosts: tuple[str, ...] = ()
    allowed_origins: tuple[str, ...] = ()
    draft_db: Path | None = None
    signing_key: Path | None = None
    signing_public_key: Path | None = None

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

        password = _admin_password(source)
        if password and not loopback_only:
            _reject_password_conflicts(source)
            allowed_hosts, allowed_origins = _password_allowlists(source)
            http_security = None
        else:
            password = ""
            allowed_hosts, allowed_origins = (), ()
            http_security = None if loopback_only else _remote_http_security(source, bind_host=bind_host, port=port)
        from bddk_mcp.admin.services.governance import resolve_governance_paths

        seed_dir, _ = resolve_governance_paths(source)
        paths = []
        for name in ("BDDK_ADMIN_DRAFT_DB", "BDDK_ADMIN_SIGNING_KEY", "BDDK_ADMIN_SIGNING_PUBLIC_KEY"):
            raw = source.get(name, "").strip()
            path = Path(raw).absolute() if raw else None
            if path and (path.is_relative_to(seed_dir) or path.resolve().is_relative_to(seed_dir.resolve())):
                raise AdminConfigError("Admin draft storage and signing keys must be outside the corpus directory.")
            paths.append(path)
        draft_db, signing_key, signing_public_key = paths
        if bool(signing_key) != bool(signing_public_key) or (signing_key and not draft_db):
            raise AdminConfigError("Admin signing requires a draft database and both separately configured keys.")
        if draft_db and signing_key:
            # Keep keys outside the entire sidecar storage directory, not just the .sqlite file.
            if any(
                p.is_relative_to(draft_db.parent) or p.resolve().is_relative_to(draft_db.resolve().parent)
                for p in (signing_key, signing_public_key)
            ):
                raise AdminConfigError("Admin signing keys must be outside the draft storage directory.")
            if signing_key.resolve() == signing_public_key.resolve():
                raise AdminConfigError("Admin private and trusted public keys must be separate files.")
        return cls(
            bind_host=bind_host,
            port=port,
            database_url=database_url,
            loopback_only=loopback_only,
            http_security=http_security,
            password=password,
            allowed_hosts=allowed_hosts,
            allowed_origins=allowed_origins,
            draft_db=draft_db,
            signing_key=signing_key,
            signing_public_key=signing_public_key,
        )


def _flag(source: Mapping[str, str], name: str) -> bool:
    return source.get(name, "").strip().lower() in _REMOTE_OPT_IN


def _admin_password(source: Mapping[str, str]) -> str:
    password = source.get("BDDK_ADMIN_PASSWORD", "").strip()
    if password and len(password) < 20:
        raise AdminConfigError("BDDK_ADMIN_PASSWORD must be at least 20 characters.")
    return password


def _reject_password_conflicts(source: Mapping[str, str]) -> None:
    if not _flag(source, "BDDK_ADMIN_REMOTE_ENABLED"):
        raise AdminConfigError("A remote admin password requires BDDK_ADMIN_REMOTE_ENABLED=true.")
    if _flag(source, "BDDK_HTTP_ALLOW_UNAUTHENTICATED"):
        raise AdminConfigError("The admin console cannot run unauthenticated on a non-loopback bind.")
    configured_jwt = sorted(name for name in source if name.startswith("BDDK_JWT_") and source[name].strip())
    if configured_jwt:
        raise AdminConfigError(
            "BDDK_ADMIN_PASSWORD cannot be combined with BDDK_JWT_* settings: " + ", ".join(configured_jwt)
        )


def _password_allowlists(source: Mapping[str, str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    from bddk_mcp.http_security import _normalize_allowed_host, _normalize_origin, _parse_normalized_list

    try:
        hosts = _parse_normalized_list(
            source.get("BDDK_HTTP_ALLOWED_HOSTS"),
            name="BDDK_HTTP_ALLOWED_HOSTS",
            normalizer=_normalize_allowed_host,
        )
        origins = _parse_normalized_list(
            source.get("BDDK_HTTP_ALLOWED_ORIGINS"),
            name="BDDK_HTTP_ALLOWED_ORIGINS",
            normalizer=_normalize_origin,
        )
    except HttpSecurityConfigError as exc:
        raise AdminConfigError(str(exc)) from exc
    if not hosts or not origins or any(not origin.startswith("https://") for origin in origins):
        raise AdminConfigError("Remote admin password requires explicit HTTPS hosts and origins.")
    return hosts, origins


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
