"""Fail-closed HTTP configuration, header/body, and JWT verification tests."""

from __future__ import annotations

import asyncio
import base64
import time
from dataclasses import FrozenInstanceError, replace
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
from starlette.middleware.authentication import AuthenticationMiddleware

from bddk_mcp.http_security import (
    HttpSecurityConfigError,
    HttpSecurityMiddleware,
    JwtTokenVerifier,
    is_loopback_host,
    load_http_security_config,
)


def _remote_env() -> dict[str, str]:
    return {
        "MCP_HOST": "0.0.0.0",
        "PORT": "8443",
        "BDDK_HTTP_ALLOWED_HOSTS": "mcp.bank.example:8443",
        "BDDK_HTTP_ALLOWED_ORIGINS": "https://audit.bank.example",
        "BDDK_JWT_ISSUER": "https://id.bank.example/realms/bddk",
        "BDDK_JWT_RESOURCE": "https://mcp.bank.example/mcp",
        "BDDK_JWT_JWKS_URL": "https://id.bank.example/realms/bddk/jwks",
        "BDDK_JWT_AUDIENCE": "bddk-mcp",
        "BDDK_JWT_REQUIRED_SCOPES": "bddk.read bddk.audit",
        "BDDK_JWT_ALGORITHMS": "RS256",
        "BDDK_JWT_ACCESS_TOKEN_TYPES": "at+jwt,JWT",
    }


def test_local_defaults_are_loopback_only_and_dns_rebinding_protected():
    config = load_http_security_config({})

    assert config.loopback_only is True
    assert config.bind_host == "127.0.0.1"
    assert "127.0.0.1:8000" in config.allowed_hosts
    assert "localhost:8000" in config.allowed_hosts
    assert "http://localhost:8000" in config.allowed_origins
    assert config.jwt_issuer is None
    assert config.jwt_access_token_types == ("at+jwt",)
    settings = config.transport_security_settings()
    assert settings.enable_dns_rebinding_protection is True
    assert "*" not in "".join(settings.allowed_hosts + settings.allowed_origins)

    with pytest.raises(FrozenInstanceError):
        config.port = 9999  # type: ignore[misc]


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("localhost", True),
        ("localhost:8000", True),
        ("127.42.1.9", True),
        ("127.0.0.1:9000", True),
        ("::1", True),
        ("[::1]:8000", True),
        ("0.0.0.0", False),
        ("::", False),
        ("10.0.0.1", False),
        ("bank.example", False),
        ("bad host", False),
    ],
)
def test_loopback_detection_is_address_aware(host, expected):
    assert is_loopback_host(host) is expected


def test_remote_bind_refuses_incomplete_security_configuration():
    with pytest.raises(HttpSecurityConfigError, match="explicit BDDK_HTTP_ALLOWED_HOSTS"):
        load_http_security_config({"MCP_HOST": "0.0.0.0"})

    incomplete = _remote_env()
    del incomplete["BDDK_JWT_JWKS_URL"]
    with pytest.raises(HttpSecurityConfigError, match="complete set"):
        load_http_security_config(incomplete)


def test_remote_bind_accepts_complete_exact_policy():
    config = load_http_security_config(_remote_env())

    assert config.loopback_only is False
    assert config.allowed_hosts == ("mcp.bank.example:8443",)
    assert config.allowed_origins == ("https://audit.bank.example",)
    assert config.jwt_required_scopes == frozenset({"bddk.read", "bddk.audit"})
    assert config.jwt_algorithms == ("RS256",)
    assert config.jwt_access_token_types == ("at+jwt", "jwt")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("PORT", "0"),
        ("PORT", "65536"),
        ("BDDK_HTTP_MAX_BODY_BYTES", "0"),
        ("BDDK_HTTP_BODY_FIRST_BYTE_TIMEOUT_SECONDS", "0"),
        ("BDDK_HTTP_BODY_CHUNK_TIMEOUT_SECONDS", "nan"),
        ("BDDK_HTTP_BODY_TOTAL_TIMEOUT_SECONDS", "301"),
        ("BDDK_HTTP_MAX_CONCURRENCY", "0"),
        ("BDDK_HTTP_RATE_LIMIT_PER_MINUTE", "100001"),
        ("BDDK_JWT_MAX_TOKEN_LENGTH", "65537"),
        ("BDDK_JWT_ALGORITHMS", "HS256"),
        ("BDDK_JWT_ACCESS_TOKEN_TYPES", "id+jwt"),
        ("BDDK_JWT_ACCESS_TOKEN_TYPES", "JWT,jwt"),
        ("BDDK_HTTP_ALLOWED_HOSTS", "*.bank.example"),
        ("BDDK_HTTP_ALLOWED_HOSTS", "mcp.bank.example:99999"),
        ("BDDK_HTTP_ALLOWED_ORIGINS", "https://*.bank.example"),
        ("BDDK_HTTP_ALLOWED_ORIGINS", "https://audit.bank.example/path"),
        ("BDDK_HTTP_ALLOWED_ORIGINS", "http://audit.bank.example"),
        ("BDDK_JWT_ISSUER", "http://id.bank.example"),
        ("BDDK_JWT_RESOURCE", "https://user:secret@mcp.bank.example/mcp"),
        ("BDDK_JWT_JWKS_URL", "https://id.bank.example:0/jwks"),
    ],
)
def test_remote_policy_rejects_invalid_values(key, value):
    env = _remote_env()
    env[key] = value

    with pytest.raises(HttpSecurityConfigError):
        load_http_security_config(env)


def test_local_scope_configuration_cannot_silently_bypass_missing_authentication():
    with pytest.raises(HttpSecurityConfigError, match="scopes require complete"):
        load_http_security_config({"BDDK_JWT_REQUIRED_SCOPES": "bddk.read"})


class _RejectingApp:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, scope, receive, send) -> None:
        self.calls += 1
        response = httpx.Response(401, text="downstream auth rejected")
        headers = [(key.lower().encode(), value.encode()) for key, value in response.headers.items()]
        await send({"type": "http.response.start", "status": response.status_code, "headers": headers})
        await send({"type": "http.response.body", "body": response.content})


class _BlockingApp:
    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, scope, receive, send) -> None:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})


@pytest.mark.asyncio
async def test_invalid_origin_is_rejected_before_downstream_auth():
    downstream = _RejectingApp()
    app = HttpSecurityMiddleware(downstream, load_http_security_config({}))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8000") as client:
        response = await client.post(
            "/mcp",
            content=b"{}",
            headers={"content-type": "application/json", "origin": "https://evil.example"},
        )

    assert response.status_code == 403
    assert downstream.calls == 0


@pytest.mark.asyncio
async def test_invalid_host_returns_421_before_downstream_auth():
    downstream = _RejectingApp()
    app = HttpSecurityMiddleware(downstream, load_http_security_config({}))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://evil.example") as client:
        response = await client.post("/mcp", content=b"{}", headers={"content-type": "application/json"})

    assert response.status_code == 421
    assert downstream.calls == 0


@pytest.mark.asyncio
async def test_oversized_post_returns_413_before_downstream_auth():
    downstream = _RejectingApp()
    config = replace(load_http_security_config({}), max_body_bytes=4)
    app = HttpSecurityMiddleware(downstream, config)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8000") as client:
        response = await client.post("/mcp", content=b"12345", headers={"content-type": "application/json"})

    assert response.status_code == 413
    assert downstream.calls == 0


@pytest.mark.asyncio
async def test_unbounded_numeric_content_length_is_rejected_without_integer_conversion():
    downstream = _RejectingApp()
    app = HttpSecurityMiddleware(downstream, load_http_security_config({}))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8000") as client:
        response = await client.post(
            "/mcp",
            content=b"{}",
            headers={"content-type": "application/json", "content-length": "9" * 5000},
        )

    assert response.status_code == 400
    assert downstream.calls == 0


@pytest.mark.asyncio
async def test_streamed_post_without_content_length_is_still_size_limited():
    downstream = _RejectingApp()
    config = replace(load_http_security_config({}), max_body_bytes=4)
    app = HttpSecurityMiddleware(downstream, config)

    async def chunks():
        yield b"123"
        yield b"45"

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8000") as client:
        response = await client.post("/mcp", content=chunks(), headers={"content-type": "application/json"})

    assert response.status_code == 413
    assert downstream.calls == 0


@pytest.mark.asyncio
async def test_single_oversized_asgi_chunk_is_rejected_before_buffer_growth():
    downstream = _RejectingApp()
    config = replace(load_http_security_config({}), max_body_bytes=4)
    app = HttpSecurityMiddleware(downstream, config)
    oversized_chunk = b"12345"
    messages = iter(
        [
            {"type": "http.request", "body": oversized_chunk, "more_body": False},
        ]
    )
    sent: list[dict[str, Any]] = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    await app(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/mcp",
            "raw_path": b"/mcp",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"localhost:8000"), (b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8000),
        },
        receive,
        send,
    )

    assert sent[0]["status"] == 413
    assert downstream.calls == 0


@pytest.mark.asyncio
async def test_slow_streamed_body_times_out_and_releases_concurrency_admission():
    downstream = _RejectingApp()
    body_started = asyncio.Event()
    release_body = asyncio.Event()
    config = replace(
        load_http_security_config({}),
        max_concurrency=1,
        rate_limit_per_minute=100,
        body_chunk_timeout_seconds=0.02,
        body_total_timeout_seconds=0.05,
    )
    app = HttpSecurityMiddleware(downstream, config, clock=lambda: 100.0)

    async def slow_body():
        body_started.set()
        yield b"{"
        await release_body.wait()
        yield b"}"

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8000") as client:
        first_task = asyncio.create_task(
            client.post("/mcp", content=slow_body(), headers={"content-type": "application/json"})
        )
        await asyncio.wait_for(body_started.wait(), timeout=1)
        rejected = await client.post("/mcp", content=b"{}", headers={"content-type": "application/json"})
        first = await asyncio.wait_for(first_task, timeout=1)
        admitted_after_timeout = await client.post(
            "/mcp",
            content=b"{}",
            headers={"content-type": "application/json"},
        )
        release_body.set()

    assert rejected.status_code == 503
    assert first.status_code == 408
    assert admitted_after_timeout.status_code == 401
    assert downstream.calls == 1


@pytest.mark.asyncio
async def test_empty_request_event_does_not_satisfy_first_byte_deadline():
    downstream = _RejectingApp()
    config = replace(
        load_http_security_config({}),
        body_first_byte_timeout_seconds=0.01,
        body_chunk_timeout_seconds=1.0,
        body_total_timeout_seconds=1.0,
    )
    app = HttpSecurityMiddleware(downstream, config)
    messages = 0
    sent: list[dict[str, Any]] = []

    async def receive():
        nonlocal messages
        messages += 1
        if messages == 1:
            return {"type": "http.request", "body": b"", "more_body": True}
        await asyncio.sleep(1)
        return {"type": "http.request", "body": b"{}", "more_body": False}

    async def send(message):
        sent.append(message)

    await app(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/mcp",
            "raw_path": b"/mcp",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"host", b"localhost:8000"), (b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8000),
        },
        receive,
        send,
    )

    assert sent[0]["status"] == 408
    assert downstream.calls == 0


@pytest.mark.asyncio
async def test_continuous_body_drip_cannot_exceed_total_deadline():
    downstream = _RejectingApp()
    config = replace(
        load_http_security_config({}),
        body_first_byte_timeout_seconds=0.05,
        body_chunk_timeout_seconds=0.05,
        body_total_timeout_seconds=0.03,
    )
    app = HttpSecurityMiddleware(downstream, config)

    async def chunks():
        for _ in range(10):
            yield b"x"
            await asyncio.sleep(0.008)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8000") as client:
        response = await client.post("/mcp", content=chunks(), headers={"content-type": "application/json"})

    assert response.status_code == 408
    assert downstream.calls == 0


@pytest.mark.asyncio
async def test_rate_limit_has_deterministic_retry_after_and_resets_on_injected_clock():
    downstream = _RejectingApp()
    now = [100.0]
    config = replace(load_http_security_config({}), rate_limit_per_minute=2)
    app = HttpSecurityMiddleware(downstream, config, clock=lambda: now[0])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8000") as client:
        first = await client.post("/mcp", content=b"{}", headers={"content-type": "application/json"})
        second = await client.post("/mcp", content=b"{}", headers={"content-type": "application/json"})
        limited = await client.post("/mcp", content=b"{}", headers={"content-type": "application/json"})
        now[0] = 159.2
        nearly_reset = await client.post("/mcp", content=b"{}", headers={"content-type": "application/json"})
        now[0] = 160.0
        reset = await client.post("/mcp", content=b"{}", headers={"content-type": "application/json"})

    assert [first.status_code, second.status_code, limited.status_code] == [401, 401, 429]
    assert limited.headers["retry-after"] == "60"
    assert nearly_reset.status_code == 429
    assert nearly_reset.headers["retry-after"] == "1"
    assert reset.status_code == 401
    assert downstream.calls == 3


@pytest.mark.asyncio
async def test_unauthenticated_health_routes_still_use_overload_admission():
    downstream = _RejectingApp()
    config = replace(load_http_security_config({}), rate_limit_per_minute=1)
    app = HttpSecurityMiddleware(downstream, config, clock=lambda: 100.0)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://pod-ip") as client:
        admitted = await client.get("/health/ready")
        limited = await client.get("/health/ready")

    assert admitted.status_code == 401
    assert limited.status_code == 429
    assert limited.headers["retry-after"] == "60"
    assert downstream.calls == 1


@pytest.mark.asyncio
async def test_invalid_origin_is_rejected_before_rate_admission():
    downstream = _RejectingApp()
    config = replace(load_http_security_config({}), rate_limit_per_minute=1)
    app = HttpSecurityMiddleware(downstream, config, clock=lambda: 100.0)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8000") as client:
        invalid = await client.post(
            "/mcp",
            content=b"{}",
            headers={"content-type": "application/json", "origin": "https://evil.example"},
        )
        valid = await client.post("/mcp", content=b"{}", headers={"content-type": "application/json"})
        limited = await client.post("/mcp", content=b"{}", headers={"content-type": "application/json"})

    assert invalid.status_code == 403
    assert valid.status_code == 401
    assert limited.status_code == 429
    assert downstream.calls == 1


@pytest.mark.asyncio
async def test_rate_limit_does_not_trust_client_supplied_forwarding_headers():
    downstream = _RejectingApp()
    config = replace(load_http_security_config({}), rate_limit_per_minute=1)
    app = HttpSecurityMiddleware(downstream, config, clock=lambda: 100.0)
    transport = httpx.ASGITransport(app=app, client=("192.0.2.10", 12345))
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8000") as client:
        first = await client.post(
            "/mcp",
            content=b"{}",
            headers={"content-type": "application/json", "x-forwarded-for": "198.51.100.1"},
        )
        second = await client.post(
            "/mcp",
            content=b"{}",
            headers={"content-type": "application/json", "x-forwarded-for": "203.0.113.1"},
        )

    assert first.status_code == 401
    assert second.status_code == 429
    assert downstream.calls == 1


@pytest.mark.asyncio
async def test_rate_state_prunes_idle_client_buckets():
    downstream = _RejectingApp()
    now = [100.0]
    config = replace(load_http_security_config({}), rate_limit_per_minute=1)
    app = HttpSecurityMiddleware(downstream, config, clock=lambda: now[0])

    for client_address in (("192.0.2.1", 1), ("192.0.2.2", 2)):
        transport = httpx.ASGITransport(app=app, client=client_address)
        async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8000") as client:
            await client.post("/mcp", content=b"{}", headers={"content-type": "application/json"})
    assert app.tracked_rate_clients == 2

    now[0] = 220.0
    transport = httpx.ASGITransport(app=app, client=("192.0.2.3", 3))
    async with httpx.AsyncClient(transport=transport, base_url="http://localhost:8000") as client:
        await client.post("/mcp", content=b"{}", headers={"content-type": "application/json"})

    assert app.tracked_rate_clients == 1


@pytest.mark.asyncio
async def test_concurrency_limit_rejects_immediately_and_releases_capacity():
    downstream = _BlockingApp()
    config = replace(load_http_security_config({}), max_concurrency=1, rate_limit_per_minute=100)
    app = HttpSecurityMiddleware(downstream, config, clock=lambda: 100.0)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8000") as client:
        first_task = asyncio.create_task(
            client.post("/mcp", content=b"{}", headers={"content-type": "application/json"})
        )
        await asyncio.wait_for(downstream.started.wait(), timeout=1)
        rejected = await client.post("/mcp", content=b"{}", headers={"content-type": "application/json"})
        downstream.release.set()
        first = await asyncio.wait_for(first_task, timeout=1)
        admitted_after_release = await client.post(
            "/mcp",
            content=b"{}",
            headers={"content-type": "application/json"},
        )

    assert first.status_code == 204
    assert rejected.status_code == 503
    assert rejected.headers["retry-after"] == "1"
    assert admitted_after_release.status_code == 204
    assert downstream.calls == 2


def _b64_uint(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


class _StaticJwksClient:
    def __init__(self, jwks: dict[str, Any]) -> None:
        self._keys = {item["kid"]: jwt.PyJWK.from_dict(item) for item in jwks["keys"]}

    def get_signing_key_from_jwt(self, token: str):
        key_id = jwt.get_unverified_header(token)["kid"]
        return self._keys[key_id]


@pytest.fixture(scope="module")
def jwt_material():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_numbers = private_key.public_key().public_numbers()
    jwks = {
        "keys": [
            {
                "kty": "RSA",
                "use": "sig",
                "kid": "test-rsa",
                "alg": "RS256",
                "n": _b64_uint(public_numbers.n),
                "e": _b64_uint(public_numbers.e),
            }
        ]
    }
    return private_key, jwks


def _claims(**overrides):
    now = int(time.time())
    claims = {
        "iss": "https://id.bank.example/realms/bddk",
        "aud": "bddk-mcp",
        "exp": now + 300,
        "sub": "service-account-42",
        # Keycloak service-account access tokens commonly expose the client as
        # ``azp`` and do not require custom ``resource`` or ``nbf`` claims.
        "azp": "audit-client",
        "scope": "bddk.read bddk.audit",
    }
    claims.update(overrides)
    return claims


def _encode(private_key, claims=None, *, token_type: str | None = "JWT") -> str:
    return jwt.encode(
        claims or _claims(),
        private_key,
        algorithm="RS256",
        headers={"kid": "test-rsa", "typ": token_type},
    )


@pytest.mark.asyncio
async def test_jwt_verifier_accepts_keycloak_style_token_without_nbf_or_resource(jwt_material):
    private_key, jwks = jwt_material
    config = load_http_security_config(_remote_env())
    verifier = JwtTokenVerifier(config, jwks_client=_StaticJwksClient(jwks))

    access = await verifier.verify_token(_encode(private_key))

    assert access is not None
    assert access.client_id == "audit-client"
    assert access.subject == "service-account-42"
    assert access.resource == "https://mcp.bank.example/mcp"
    assert set(access.scopes) == {"bddk.read", "bddk.audit"}
    assert access.claims is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claims",
    [
        _claims(iss="https://wrong.example"),
        _claims(aud="wrong-audience"),
        _claims(exp=int(time.time()) - 60),
        _claims(nbf=int(time.time()) + 60),
    ],
)
async def test_jwt_verifier_fails_closed_for_invalid_claims(jwt_material, claims):
    private_key, jwks = jwt_material
    verifier = JwtTokenVerifier(load_http_security_config(_remote_env()), jwks_client=_StaticJwksClient(jwks))

    assert await verifier.verify_token(_encode(private_key, claims)) is None


@pytest.mark.asyncio
async def test_jwt_verifier_accepts_optional_valid_nbf_and_rfc_9068_type(jwt_material):
    private_key, jwks = jwt_material
    verifier = JwtTokenVerifier(load_http_security_config(_remote_env()), jwks_client=_StaticJwksClient(jwks))

    access = await verifier.verify_token(_encode(private_key, _claims(nbf=int(time.time()) - 1), token_type="at+jwt"))

    assert access is not None


@pytest.mark.asyncio
async def test_jwt_verifier_rejects_id_token_types_and_keycloak_id_token_audience(jwt_material):
    private_key, jwks = jwt_material
    verifier = JwtTokenVerifier(load_http_security_config(_remote_env()), jwks_client=_StaticJwksClient(jwks))

    explicitly_typed_id_token = _encode(private_key, token_type="id+jwt")
    untyped_token = _encode(private_key, token_type=None)
    keycloak_style_id_token = _encode(
        private_key,
        _claims(aud="audit-client", nonce="oidc-nonce", auth_time=int(time.time()) - 10),
        token_type="JWT",
    )

    assert await verifier.verify_token(explicitly_typed_id_token) is None
    assert await verifier.verify_token(untyped_token) is None
    assert await verifier.verify_token(keycloak_style_id_token) is None


@pytest.mark.asyncio
async def test_jwt_access_token_type_policy_defaults_to_rfc_9068_only(jwt_material):
    private_key, jwks = jwt_material
    env = _remote_env()
    del env["BDDK_JWT_ACCESS_TOKEN_TYPES"]
    verifier = JwtTokenVerifier(load_http_security_config(env), jwks_client=_StaticJwksClient(jwks))

    assert await verifier.verify_token(_encode(private_key, token_type="JWT")) is None
    assert await verifier.verify_token(_encode(private_key, token_type="at+jwt")) is not None


@pytest.mark.asyncio
async def test_jwt_verifier_authenticates_under_scoped_token_for_authorization_layer(jwt_material):
    private_key, jwks = jwt_material
    verifier = JwtTokenVerifier(load_http_security_config(_remote_env()), jwks_client=_StaticJwksClient(jwks))

    access = await verifier.verify_token(_encode(private_key, _claims(scope="bddk.read")))

    assert access is not None
    assert access.scopes == ["bddk.read"]


@pytest.mark.asyncio
async def test_sdk_authorization_middleware_returns_403_for_authenticated_under_scoped_token(jwt_material):
    private_key, jwks = jwt_material
    config = load_http_security_config(_remote_env())
    verifier = JwtTokenVerifier(config, jwks_client=_StaticJwksClient(jwks))
    downstream = _RejectingApp()
    authorized = RequireAuthMiddleware(downstream, required_scopes=sorted(config.jwt_required_scopes))
    authenticated = AuthenticationMiddleware(authorized, backend=BearerAuthBackend(verifier))
    app = HttpSecurityMiddleware(authenticated, config)
    token = _encode(private_key, _claims(scope="bddk.read"))

    transport = httpx.ASGITransport(app=app, client=("192.0.2.10", 12345))
    async with httpx.AsyncClient(transport=transport, base_url="https://mcp.bank.example:8443") as client:
        response = await client.post(
            "/mcp",
            content=b"{}",
            headers={
                "authorization": f"Bearer {token}",
                "content-type": "application/json",
                "origin": "https://audit.bank.example",
            },
        )

    assert response.status_code == 403
    assert response.json()["error"] == "insufficient_scope"
    assert downstream.calls == 0


@pytest.mark.asyncio
async def test_jwt_verifier_rejects_wrong_signature_disallowed_algorithm_and_oversized_token(jwt_material):
    _private_key, jwks = jwt_material
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    config = load_http_security_config(_remote_env())
    verifier = JwtTokenVerifier(config, jwks_client=_StaticJwksClient(jwks))

    wrong_signature = _encode(other_key)
    symmetric = jwt.encode(
        _claims(),
        "not-a-real-shared-secret-with-32-bytes",
        algorithm="HS256",
        headers={"kid": "test-rsa"},
    )

    assert await verifier.verify_token(wrong_signature) is None
    assert await verifier.verify_token(symmetric) is None
    assert await verifier.verify_token("x" * (config.jwt_max_token_length + 1)) is None
