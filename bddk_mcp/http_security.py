"""Fail-closed HTTP transport and bearer-token security primitives.

The server integration deliberately lives elsewhere.  This module keeps policy
parsing, request-header/body validation, and JWT verification independently
testable so a future Streamable HTTP entry point can compose them *outside* its
authentication middleware.
"""

from __future__ import annotations

import asyncio
import ipaddress
import math
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

import jwt
from mcp.server.auth.provider import AccessToken
from mcp.server.transport_security import TransportSecurityMiddleware, TransportSecuritySettings
from starlette.requests import HTTPConnection
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

_ASYMMETRIC_JWT_ALGORITHMS = frozenset(
    {
        "RS256",
        "RS384",
        "RS512",
        "PS256",
        "PS384",
        "PS512",
        "ES256",
        "ES384",
        "ES512",
    }
)
_DEFAULT_JWT_ALGORITHMS = ("RS256", "PS256", "ES256")
_DEFAULT_JWT_ACCESS_TOKEN_TYPES = ("at+jwt",)
_SUPPORTED_JWT_ACCESS_TOKEN_TYPES = frozenset(
    {
        "at+jwt",
        "application/at+jwt",
        # Keycloak commonly emits the generic JOSE JWT type for access tokens.
        # It is safe here only because the verifier also requires the dedicated
        # MCP resource-server audience configured by the deployment.
        "jwt",
        "application/jwt",
    }
)
_HOST_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$", re.IGNORECASE)
_SCOPE_TOKEN = re.compile(r"^[!#-\[\]-~]+$")
_RATE_WINDOW_SECONDS = 60.0
_RATE_STATE_IDLE_SECONDS = 120.0
_RATE_STATE_MIN_CLIENTS = 256
_RATE_STATE_MAX_CLIENTS = 16_384
_HEALTH_PATHS = frozenset({"/health/live", "/health/ready"})
_MAX_CONTENT_LENGTH_DIGITS = 20


class HttpSecurityConfigError(ValueError):
    """Raised when HTTP security environment configuration is unsafe."""


class _JwksClient(Protocol):
    def get_signing_key_from_jwt(self, token: str) -> Any:
        """Return the PyJWT signing-key wrapper selected by the token header."""


@dataclass(frozen=True, slots=True)
class HttpSecurityConfig:
    """Immutable, validated HTTP and JWT security policy."""

    bind_host: str
    port: int
    loopback_only: bool
    allowed_hosts: tuple[str, ...]
    allowed_origins: tuple[str, ...]
    jwt_issuer: str | None
    jwt_resource: str | None
    jwt_jwks_url: str | None
    jwt_audience: str | None
    jwt_required_scopes: frozenset[str]
    jwt_algorithms: tuple[str, ...]
    jwt_access_token_types: tuple[str, ...]
    jwt_max_token_length: int
    max_body_bytes: int
    body_first_byte_timeout_seconds: float
    body_chunk_timeout_seconds: float
    body_total_timeout_seconds: float
    max_concurrency: int
    rate_limit_per_minute: int
    allow_unauthenticated: bool = False
    trusted_proxy_hops: int = 0

    def transport_security_settings(self) -> TransportSecuritySettings:
        """Return a fresh SDK settings object for FastMCP integration."""
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=list(self.allowed_hosts),
            allowed_origins=list(self.allowed_origins),
        )


def _parse_positive_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    maximum: int,
) -> int:
    raw = env.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise HttpSecurityConfigError(f"{name} must be an integer") from exc
    if not 1 <= value <= maximum:
        raise HttpSecurityConfigError(f"{name} must be between 1 and {maximum}")
    return value


def _parse_non_negative_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    maximum: int,
) -> int:
    raw = env.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise HttpSecurityConfigError(f"{name} must be an integer") from exc
    if not 0 <= value <= maximum:
        raise HttpSecurityConfigError(f"{name} must be between 0 and {maximum}")
    return value


def _parse_positive_float(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    maximum: float,
) -> float:
    raw = env.get(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise HttpSecurityConfigError(f"{name} must be a number") from exc
    if not math.isfinite(value) or not 0 < value <= maximum:
        raise HttpSecurityConfigError(f"{name} must be greater than zero and at most {maximum:g}")
    return value


def _parse_bool(env: Mapping[str, str], name: str) -> bool:
    """Parse a strict opt-in boolean; anything unrecognised is a configuration error."""
    raw = env.get(name, "").strip().lower()
    if not raw:
        return False
    if raw in {"1", "true", "yes"}:
        return True
    if raw in {"0", "false", "no"}:
        return False
    raise HttpSecurityConfigError(f"{name} must be a boolean value")


def _validate_port(raw: str, *, name: str) -> int:
    if not raw.isascii() or not raw.isdigit():
        raise HttpSecurityConfigError(f"{name} contains an invalid port")
    port = int(raw)
    if not 1 <= port <= 65535:
        raise HttpSecurityConfigError(f"{name} port must be between 1 and 65535")
    return port


def _normalize_hostname(raw: str, *, name: str) -> str:
    host = raw.strip().rstrip(".").lower()
    if not host or any(character.isspace() for character in host):
        raise HttpSecurityConfigError(f"{name} contains an invalid host")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError as exc:
            raise HttpSecurityConfigError(f"{name} contains an invalid host") from exc
        if len(host) > 253 or any(not _HOST_LABEL.fullmatch(label) for label in host.split(".")):
            raise HttpSecurityConfigError(f"{name} contains an invalid host") from None
        return host
    return address.compressed


def _normalize_bind_host(raw: str) -> str:
    value = raw.strip()
    if not value:
        raise HttpSecurityConfigError("MCP_HOST must not be empty")
    if "://" in value or any(character in value for character in "/?#@*"):
        raise HttpSecurityConfigError("MCP_HOST must be a host without a scheme, path, or wildcard")
    if value.startswith("["):
        if not value.endswith("]"):
            raise HttpSecurityConfigError("MCP_HOST contains an invalid bracketed IPv6 address")
        value = value[1:-1]
    return _normalize_hostname(value, name="MCP_HOST")


def is_loopback_host(value: str) -> bool:
    """Return whether a host (optionally with a port) is unambiguously loopback."""
    candidate = value.strip()
    if not candidate:
        return False
    try:
        if candidate.startswith("["):
            end = candidate.find("]")
            if end < 0:
                return False
            host = candidate[1:end]
            remainder = candidate[end + 1 :]
            if remainder:
                if not remainder.startswith(":"):
                    return False
                _validate_port(remainder[1:], name="host")
        else:
            try:
                address = ipaddress.ip_address(candidate)
            except ValueError:
                if candidate.count(":") == 1:
                    host, port_text = candidate.rsplit(":", 1)
                    _validate_port(port_text, name="host")
                else:
                    host = candidate
            else:
                return address.is_loopback
        normalized = _normalize_hostname(host, name="host")
        try:
            return ipaddress.ip_address(normalized).is_loopback
        except ValueError:
            return normalized == "localhost"
    except HttpSecurityConfigError:
        return False


def _format_authority(host: str, port: int | None = None) -> str:
    try:
        is_v6 = ipaddress.ip_address(host).version == 6
    except ValueError:
        is_v6 = False
    authority = f"[{host}]" if is_v6 else host
    return f"{authority}:{port}" if port is not None else authority


def _normalize_allowed_host(raw: str) -> str:
    value = raw.strip()
    if not value or "*" in value or "://" in value or any(character in value for character in "/?#@"):
        raise HttpSecurityConfigError("allowed Hosts must be exact host or host:port values without wildcards")

    port: int | None = None
    if value.startswith("["):
        end = value.find("]")
        if end < 0:
            raise HttpSecurityConfigError("allowed Host contains invalid bracketed IPv6")
        host = value[1:end]
        remainder = value[end + 1 :]
        if remainder:
            if not remainder.startswith(":"):
                raise HttpSecurityConfigError("allowed Host contains trailing data")
            port = _validate_port(remainder[1:], name="allowed Host")
    elif value.count(":") > 1:
        raise HttpSecurityConfigError("IPv6 allowed Hosts must use brackets")
    elif ":" in value:
        host, port_text = value.rsplit(":", 1)
        port = _validate_port(port_text, name="allowed Host")
    else:
        host = value

    return _format_authority(_normalize_hostname(host, name="allowed Host"), port)


def _normalize_origin(raw: str) -> str:
    value = raw.strip()
    if not value or "*" in value or value == "null":
        raise HttpSecurityConfigError("allowed Origins must be exact HTTP(S) origins without wildcards")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise HttpSecurityConfigError("allowed Origin contains an invalid port") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise HttpSecurityConfigError("allowed Origin must be an absolute HTTP(S) origin")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise HttpSecurityConfigError("allowed Origin must not contain credentials, a path, query, or fragment")
    if port is not None and not 1 <= port <= 65535:
        raise HttpSecurityConfigError("allowed Origin port must be between 1 and 65535")
    host = _normalize_hostname(parsed.hostname, name="allowed Origin")
    return f"{parsed.scheme.lower()}://{_format_authority(host, port)}"


def _parse_normalized_list(
    raw: str | None,
    *,
    name: str,
    normalizer,
) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return ()
    pieces = raw.split(",")
    if any(not piece.strip() for piece in pieces):
        raise HttpSecurityConfigError(f"{name} contains an empty list entry")
    normalized = tuple(normalizer(piece) for piece in pieces)
    if len(set(normalized)) != len(normalized):
        raise HttpSecurityConfigError(f"{name} contains duplicate entries")
    return normalized


def _validate_https_url(name: str, raw: str, *, allow_query: bool = False) -> str:
    value = raw.strip()
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise HttpSecurityConfigError(f"{name} contains an invalid port") from exc
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise HttpSecurityConfigError(f"{name} must be an absolute HTTPS URL")
    if parsed.username or parsed.password or parsed.fragment or (parsed.query and not allow_query):
        raise HttpSecurityConfigError(f"{name} must not contain credentials or unsupported URL components")
    if port is not None and not 1 <= port <= 65535:
        raise HttpSecurityConfigError(f"{name} port must be between 1 and 65535")
    _normalize_hostname(parsed.hostname, name=name)
    return value


def _validate_identifier(name: str, raw: str) -> str:
    value = raw.strip()
    if not value or len(value) > 512 or any(ord(character) < 0x21 or ord(character) == 0x7F for character in value):
        raise HttpSecurityConfigError(f"{name} must be a non-empty visible identifier")
    return value


def _validate_scope(scope: str, *, name: str) -> str:
    if len(scope) > 128 or not _SCOPE_TOKEN.fullmatch(scope):
        raise HttpSecurityConfigError(f"{name} contains an invalid OAuth scope")
    return scope


def _parse_scopes(raw: str | None) -> frozenset[str]:
    if raw is None or not raw.strip():
        return frozenset()
    pieces = raw.replace(",", " ").split()
    scopes = frozenset(_validate_scope(piece, name="BDDK_JWT_REQUIRED_SCOPES") for piece in pieces)
    if len(scopes) != len(pieces):
        raise HttpSecurityConfigError("BDDK_JWT_REQUIRED_SCOPES contains duplicate scopes")
    return scopes


def _parse_algorithms(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return _DEFAULT_JWT_ALGORITHMS
    algorithms = tuple(piece.strip().upper() for piece in raw.split(","))
    if any(not algorithm for algorithm in algorithms) or len(set(algorithms)) != len(algorithms):
        raise HttpSecurityConfigError("BDDK_JWT_ALGORITHMS contains empty or duplicate entries")
    unsupported = set(algorithms) - _ASYMMETRIC_JWT_ALGORITHMS
    if unsupported:
        raise HttpSecurityConfigError("BDDK_JWT_ALGORITHMS permits a non-approved asymmetric algorithm")
    return algorithms


def _parse_access_token_types(raw: str | None) -> tuple[str, ...]:
    """Return approved JOSE ``typ`` values in a comparison-safe form.

    RFC 9068 access tokens use ``at+jwt``, which is the fail-closed default.
    Keycloak deployments commonly use the generic ``JWT`` value, so that value
    is available only as an explicit compatibility opt-in while exact audience
    validation provides the token-class/resource binding.  Known ID-token
    labels and arbitrary proprietary types are deliberately unsupported.
    """

    if raw is None or not raw.strip():
        return _DEFAULT_JWT_ACCESS_TOKEN_TYPES
    token_types = tuple(piece.strip().lower() for piece in raw.split(","))
    if any(not token_type for token_type in token_types) or len(set(token_types)) != len(token_types):
        raise HttpSecurityConfigError("BDDK_JWT_ACCESS_TOKEN_TYPES contains empty or duplicate entries")
    unsupported = set(token_types) - _SUPPORTED_JWT_ACCESS_TOKEN_TYPES
    if unsupported:
        raise HttpSecurityConfigError("BDDK_JWT_ACCESS_TOKEN_TYPES permits a non-approved access-token type")
    return token_types


def _local_hosts(port: int, bind_host: str) -> tuple[str, ...]:
    hosts = {"127.0.0.1", f"127.0.0.1:{port}", "localhost", f"localhost:{port}", "[::1]", f"[::1]:{port}"}
    hosts.add(_format_authority(bind_host))
    hosts.add(_format_authority(bind_host, port))
    return tuple(sorted(hosts))


def _local_origins(port: int, bind_host: str) -> tuple[str, ...]:
    hosts = {"127.0.0.1", "localhost", "::1", bind_host}
    return tuple(sorted(f"http://{_format_authority(host, port)}" for host in hosts))


def load_http_security_config(env: Mapping[str, str] | None = None) -> HttpSecurityConfig:
    """Parse environment variables into an immutable fail-closed policy."""
    source = os.environ if env is None else env
    bind_host = _normalize_bind_host(source.get("MCP_HOST", "127.0.0.1"))
    port = _parse_positive_int(source, "PORT", 8000, maximum=65535)
    loopback_only = is_loopback_host(bind_host)

    allowed_hosts = _parse_normalized_list(
        source.get("BDDK_HTTP_ALLOWED_HOSTS"),
        name="BDDK_HTTP_ALLOWED_HOSTS",
        normalizer=_normalize_allowed_host,
    )
    allowed_origins = _parse_normalized_list(
        source.get("BDDK_HTTP_ALLOWED_ORIGINS"),
        name="BDDK_HTTP_ALLOWED_ORIGINS",
        normalizer=_normalize_origin,
    )
    if loopback_only:
        allowed_hosts = allowed_hosts or _local_hosts(port, bind_host)
        allowed_origins = allowed_origins or _local_origins(port, bind_host)
    elif not allowed_hosts or not allowed_origins:
        raise HttpSecurityConfigError(
            "Non-loopback HTTP requires explicit BDDK_HTTP_ALLOWED_HOSTS and BDDK_HTTP_ALLOWED_ORIGINS"
        )
    elif any(not origin.startswith("https://") for origin in allowed_origins):
        raise HttpSecurityConfigError("Non-loopback HTTP requires HTTPS allowed Origins")

    auth_names = (
        "BDDK_JWT_ISSUER",
        "BDDK_JWT_RESOURCE",
        "BDDK_JWT_JWKS_URL",
        "BDDK_JWT_AUDIENCE",
    )
    auth_values = tuple(source.get(name, "").strip() for name in auth_names)
    required_scopes = _parse_scopes(source.get("BDDK_JWT_REQUIRED_SCOPES"))
    allow_unauthenticated = _parse_bool(source, "BDDK_HTTP_ALLOW_UNAUTHENTICATED")
    if allow_unauthenticated:
        # Scan by prefix rather than checking the four discovery values: settings
        # such as BDDK_JWT_ALGORITHMS are not part of auth_values, and silently
        # ignoring them would leave a half-migrated deployment looking healthy.
        configured_jwt = sorted(name for name in source if name.startswith("BDDK_JWT_") and source[name].strip())
        if configured_jwt:
            raise HttpSecurityConfigError(
                "BDDK_HTTP_ALLOW_UNAUTHENTICATED cannot be combined with any BDDK_JWT_* setting; "
                f"remove {', '.join(configured_jwt)}"
            )
    if any(auth_values) and not all(auth_values):
        raise HttpSecurityConfigError("JWT authentication settings must be configured as a complete set")
    if not loopback_only and not all(auth_values) and not allow_unauthenticated:
        raise HttpSecurityConfigError("Non-loopback HTTP requires issuer, resource, JWKS URL, and audience")

    if required_scopes and not all(auth_values):
        raise HttpSecurityConfigError("JWT scopes require complete issuer, resource, JWKS URL, and audience settings")
    if all(auth_values) and not required_scopes:
        raise HttpSecurityConfigError("JWT authentication requires at least one BDDK_JWT_REQUIRED_SCOPES value")
    if not loopback_only and not required_scopes and not allow_unauthenticated:
        raise HttpSecurityConfigError("Non-loopback HTTP requires at least one JWT scope")

    issuer = _validate_https_url(auth_names[0], auth_values[0]) if auth_values[0] else None
    resource = _validate_https_url(auth_names[1], auth_values[1]) if auth_values[1] else None
    jwks_url = _validate_https_url(auth_names[2], auth_values[2], allow_query=True) if auth_values[2] else None
    audience = _validate_identifier(auth_names[3], auth_values[3]) if auth_values[3] else None

    return HttpSecurityConfig(
        bind_host=bind_host,
        port=port,
        loopback_only=loopback_only,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
        jwt_issuer=issuer,
        jwt_resource=resource,
        jwt_jwks_url=jwks_url,
        jwt_audience=audience,
        jwt_required_scopes=required_scopes,
        jwt_algorithms=_parse_algorithms(source.get("BDDK_JWT_ALGORITHMS")),
        jwt_access_token_types=_parse_access_token_types(source.get("BDDK_JWT_ACCESS_TOKEN_TYPES")),
        jwt_max_token_length=_parse_positive_int(source, "BDDK_JWT_MAX_TOKEN_LENGTH", 16384, maximum=65536),
        max_body_bytes=_parse_positive_int(source, "BDDK_HTTP_MAX_BODY_BYTES", 1_048_576, maximum=16_777_216),
        body_first_byte_timeout_seconds=_parse_positive_float(
            source,
            "BDDK_HTTP_BODY_FIRST_BYTE_TIMEOUT_SECONDS",
            5.0,
            maximum=120.0,
        ),
        body_chunk_timeout_seconds=_parse_positive_float(
            source,
            "BDDK_HTTP_BODY_CHUNK_TIMEOUT_SECONDS",
            5.0,
            maximum=120.0,
        ),
        body_total_timeout_seconds=_parse_positive_float(
            source,
            "BDDK_HTTP_BODY_TOTAL_TIMEOUT_SECONDS",
            30.0,
            maximum=300.0,
        ),
        max_concurrency=_parse_positive_int(source, "BDDK_HTTP_MAX_CONCURRENCY", 32, maximum=1024),
        rate_limit_per_minute=_parse_positive_int(
            source,
            "BDDK_HTTP_RATE_LIMIT_PER_MINUTE",
            120,
            maximum=100_000,
        ),
        allow_unauthenticated=allow_unauthenticated,
        trusted_proxy_hops=_parse_non_negative_int(source, "BDDK_HTTP_TRUSTED_PROXY_HOPS", 0, maximum=8),
    )


def _extract_scopes(claims: Mapping[str, Any]) -> frozenset[str]:
    scopes: set[str] = set()
    for claim_name in ("scope", "scp"):
        claim = claims.get(claim_name)
        if claim is None:
            continue
        if isinstance(claim, str):
            values = claim.split()
        elif isinstance(claim, list) and all(isinstance(item, str) for item in claim):
            values = claim
        else:
            raise ValueError("scope claim has an invalid type")
        scopes.update(_validate_scope(value, name=claim_name) for value in values)
    return frozenset(scopes)


class JwtTokenVerifier:
    """MCP TokenVerifier using asymmetric JWT signatures and a JWKS key set."""

    def __init__(self, config: HttpSecurityConfig, *, jwks_client: _JwksClient | None = None) -> None:
        if not all((config.jwt_issuer, config.jwt_resource, config.jwt_jwks_url, config.jwt_audience)):
            raise HttpSecurityConfigError("JwtTokenVerifier requires complete JWT settings")
        if not config.jwt_required_scopes:
            raise HttpSecurityConfigError("JwtTokenVerifier requires at least one scope")
        self._config = config
        self._jwks_client = jwks_client or jwt.PyJWKClient(
            config.jwt_jwks_url,
            cache_keys=True,
            max_cached_keys=16,
            cache_jwk_set=True,
            lifespan=300,
            timeout=5,
        )

    async def verify_token(self, token: str) -> AccessToken | None:
        """Return access information for a valid token, otherwise fail closed."""
        if not isinstance(token, str) or not token or len(token) > self._config.jwt_max_token_length:
            return None
        try:
            return await asyncio.to_thread(self._verify_token_sync, token)
        except Exception:
            # Authentication failures intentionally have one silent result.  Do
            # not log tokens, unverified claims, key IDs, or verifier exceptions.
            return None

    def _verify_token_sync(self, token: str) -> AccessToken:
        header = jwt.get_unverified_header(token)
        algorithm = header.get("alg")
        key_id = header.get("kid")
        token_type = header.get("typ")
        if (
            algorithm not in self._config.jwt_algorithms
            or not isinstance(key_id, str)
            or not key_id
            or not isinstance(token_type, str)
            or token_type.lower() not in self._config.jwt_access_token_types
        ):
            raise jwt.InvalidTokenError("unapproved JWT header")

        signing_key = self._jwks_client.get_signing_key_from_jwt(token)
        key_algorithm = getattr(signing_key, "algorithm_name", None)
        if key_algorithm is not None and key_algorithm != algorithm:
            raise jwt.InvalidTokenError("JWT key algorithm mismatch")
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=list(self._config.jwt_algorithms),
            audience=self._config.jwt_audience,
            issuer=self._config.jwt_issuer,
            # PyJWT validates ``nbf`` whenever it is present.  It is optional
            # because Keycloak and other conforming issuers do not always emit
            # it.  Resource-server binding is the required ``aud`` mapping,
            # rather than a non-standard token ``resource`` claim.
            options={"require": ["iss", "aud", "exp", "sub"]},
        )
        scopes = _extract_scopes(claims)

        subject = claims.get("sub")
        client_id = claims.get("client_id") or claims.get("azp") or subject
        if not isinstance(subject, str) or not subject or not isinstance(client_id, str) or not client_id:
            raise jwt.InvalidTokenError("invalid subject or client identifier")
        expires_at = claims.get("exp")
        if not isinstance(expires_at, int | float):
            raise jwt.InvalidTokenError("invalid expiry")

        return AccessToken(
            token=token,
            client_id=client_id,
            scopes=sorted(scopes),
            expires_at=int(expires_at),
            resource=self._config.jwt_resource,
            subject=subject,
            claims=None,
        )


def _header_values(scope: Scope, header_name: bytes) -> list[bytes]:
    return [value for name, value in scope.get("headers", []) if name.lower() == header_name]


def _client_rate_key(scope: Scope, trusted_proxy_hops: int = 0) -> str:
    """Return a coarse client key, trusting forwarding headers only when configured.

    At the default of zero hops the key is the ASGI socket peer and
    client-controlled ``X-Forwarded-For`` values are ignored entirely, which is
    correct for a directly exposed bind.  A deployment behind ``n`` trusted
    reverse proxies sets ``BDDK_HTTP_TRUSTED_PROXY_HOPS=n``; the key is then the
    ``n``-th entry from the right of the combined forwarded list, which is the
    last hop the operator asserts is trustworthy.  Anything unusable degrades to
    ``"unknown"`` rather than to a spoofable value.
    """
    if trusted_proxy_hops > 0:
        forwarded: list[str] = []
        for raw in _header_values(scope, b"x-forwarded-for"):
            forwarded.extend(part.strip() for part in raw.decode("latin-1").split(","))
        forwarded = [entry for entry in forwarded if entry]
        if len(forwarded) < trusted_proxy_hops:
            return "unknown"
        try:
            return ipaddress.ip_address(forwarded[-trusted_proxy_hops]).compressed
        except ValueError:
            return "unknown"

    client = scope.get("client")
    if not isinstance(client, tuple) or not client or not isinstance(client[0], str):
        return "unknown"
    try:
        return ipaddress.ip_address(client[0]).compressed
    except ValueError:
        return "unknown"


@dataclass(slots=True)
class _RateWindow:
    started_at: float
    count: int
    last_seen_at: float


class _ConcurrencyLimiter:
    """Small process-local limiter with immediate admission or rejection."""

    def __init__(self, maximum: int) -> None:
        self._maximum = maximum
        self._in_flight = 0
        self._lock = threading.Lock()

    def try_acquire(self) -> bool:
        with self._lock:
            if self._in_flight >= self._maximum:
                return False
            self._in_flight += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._in_flight <= 0:  # pragma: no cover - internal invariant guard
                raise RuntimeError("HTTP concurrency limiter released without admission")
            self._in_flight -= 1


class HttpSecurityMiddleware:
    """Outer ASGI request validator with process-local overload controls."""

    def __init__(
        self,
        app: ASGIApp,
        config: HttpSecurityConfig,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._app = app
        self._max_body_bytes = config.max_body_bytes
        self._body_first_byte_timeout_seconds = config.body_first_byte_timeout_seconds
        self._body_chunk_timeout_seconds = config.body_chunk_timeout_seconds
        self._body_total_timeout_seconds = config.body_total_timeout_seconds
        self._transport = TransportSecurityMiddleware(config.transport_security_settings())
        self._clock = clock or time.monotonic
        self._rate_limit = config.rate_limit_per_minute
        self._trusted_proxy_hops = config.trusted_proxy_hops
        self._rate_state_capacity = min(
            _RATE_STATE_MAX_CLIENTS,
            max(_RATE_STATE_MIN_CLIENTS, config.max_concurrency * 16),
        )
        self._rate_windows: OrderedDict[str, _RateWindow] = OrderedDict()
        self._rate_lock = threading.Lock()
        self._concurrency = _ConcurrencyLimiter(config.max_concurrency)

    @property
    def tracked_rate_clients(self) -> int:
        """Return only the bucket count, never client identifiers."""
        with self._rate_lock:
            return len(self._rate_windows)

    def _rate_retry_after(self, scope: Scope) -> int | None:
        now = self._clock()
        client_key = _client_rate_key(scope, self._trusted_proxy_hops)
        with self._rate_lock:
            while self._rate_windows:
                oldest_key = next(iter(self._rate_windows))
                oldest = self._rate_windows[oldest_key]
                if now - oldest.last_seen_at < _RATE_STATE_IDLE_SECONDS:
                    break
                self._rate_windows.popitem(last=False)

            window = self._rate_windows.get(client_key)
            if window is None:
                if len(self._rate_windows) >= self._rate_state_capacity:
                    self._rate_windows.popitem(last=False)
                self._rate_windows[client_key] = _RateWindow(now, 1, now)
                return None

            elapsed = max(0.0, now - window.started_at)
            if elapsed >= _RATE_WINDOW_SECONDS:
                window.started_at = now
                window.count = 1
                window.last_seen_at = now
                self._rate_windows.move_to_end(client_key)
                return None

            window.last_seen_at = now
            self._rate_windows.move_to_end(client_key)
            if window.count >= self._rate_limit:
                return max(1, math.ceil(_RATE_WINDOW_SECONDS - elapsed))
            window.count += 1
            return None

    async def _try_admit(self, scope: Scope, receive: Receive, send: Send) -> bool:
        """Apply rate/concurrency admission before downstream or body streaming."""
        retry_after = self._rate_retry_after(scope)
        if retry_after is not None:
            await PlainTextResponse(
                "Rate limit exceeded",
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )(scope, receive, send)
            return False

        if not self._concurrency.try_acquire():
            await PlainTextResponse(
                "Server concurrency limit reached",
                status_code=503,
                headers={"Retry-After": "1"},
            )(scope, receive, send)
            return False
        return True

    async def _call_with_overload_controls(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not await self._try_admit(scope, receive, send):
            return
        try:
            await self._app(scope, receive, send)
        finally:
            self._concurrency.release()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        # Kubernetes/OpenShift probes commonly use a pod-IP Host value that is
        # not known when the exact MCP allowlist is configured. These fixed,
        # content-free GET routes are outside the MCP protocol/auth surface,
        # but still consume the same bounded rate/concurrency admission.
        if scope.get("method", "GET").upper() == "GET" and scope.get("path") in _HEALTH_PATHS:
            await self._call_with_overload_controls(scope, receive, send)
            return

        host_values = _header_values(scope, b"host")
        if len(host_values) != 1:
            await PlainTextResponse("Invalid Host header", status_code=421)(scope, receive, send)
            return
        if len(_header_values(scope, b"origin")) > 1:
            await PlainTextResponse("Invalid Origin header", status_code=403)(scope, receive, send)
            return

        method = scope.get("method", "GET").upper()
        validation_error = await self._transport.validate_request(HTTPConnection(scope), is_post=method == "POST")
        if validation_error is not None:
            await validation_error(scope, receive, send)
            return

        if method != "POST":
            await self._call_with_overload_controls(scope, receive, send)
            return

        content_lengths = _header_values(scope, b"content-length")
        if len(content_lengths) > 1:
            await PlainTextResponse("Invalid Content-Length header", status_code=400)(scope, receive, send)
            return
        if content_lengths:
            try:
                content_length_text = content_lengths[0].decode("ascii")
            except UnicodeDecodeError:
                content_length_text = ""
            if not content_length_text.isdigit() or len(content_length_text) > _MAX_CONTENT_LENGTH_DIGITS:
                await PlainTextResponse("Invalid Content-Length header", status_code=400)(scope, receive, send)
                return
            if int(content_length_text) > self._max_body_bytes:
                await PlainTextResponse("Request body too large", status_code=413)(scope, receive, send)
                return

        # Admit before consuming a chunked/slow body. Otherwise an attacker can
        # hold arbitrarily many body readers outside the configured concurrency
        # bound even though each individual body is size-limited.
        if not await self._try_admit(scope, receive, send):
            return
        try:
            body = bytearray()
            first_request_message = True
            deadline = asyncio.get_running_loop().time() + self._body_total_timeout_seconds
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                per_message_timeout = (
                    self._body_first_byte_timeout_seconds if first_request_message else self._body_chunk_timeout_seconds
                )
                timeout = min(remaining, per_message_timeout)
                if timeout <= 0:
                    await PlainTextResponse("Request body timeout", status_code=408)(scope, receive, send)
                    return
                try:
                    message = await asyncio.wait_for(receive(), timeout=timeout)
                except TimeoutError:
                    await PlainTextResponse("Request body timeout", status_code=408)(scope, receive, send)
                    return
                if message["type"] == "http.disconnect":
                    return
                if message["type"] != "http.request":
                    continue
                chunk = message.get("body", b"")
                if len(chunk) > self._max_body_bytes - len(body):
                    await PlainTextResponse("Request body too large", status_code=413)(scope, receive, send)
                    return
                body.extend(chunk)
                if chunk:
                    first_request_message = False
                if not message.get("more_body", False):
                    break

            body_replayed = False

            async def replay_receive() -> Message:
                nonlocal body_replayed
                if body_replayed:
                    return {"type": "http.request", "body": b"", "more_body": False}
                body_replayed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}

            await self._app(scope, replay_receive, send)
        finally:
            self._concurrency.release()
