"""Official-client coverage for the secured Streamable HTTP composition."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.auth.provider import AccessToken

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.http_security import HttpSecurityMiddleware, load_http_security_config
from bddk_mcp.tools.registry import PUBLIC_TOOL_NAMES, ToolProfile


@pytest.mark.asyncio
async def test_official_client_initializes_lists_and_calls_over_secured_local_http():
    from bddk_mcp.server import create_mcp

    doc_store = MagicMock()
    doc_store.get_document_history = AsyncMock(return_value=[])
    deps = Dependencies(pool=None, doc_store=doc_store, client=MagicMock(), http=None)
    security = load_http_security_config({"MCP_HOST": "127.0.0.1", "PORT": "8123"})
    server = create_mcp(deps, profile=ToolProfile.PUBLIC, http_security=security)
    raw_app = server.streamable_http_app()
    app = HttpSecurityMiddleware(raw_app, security)

    transport = httpx.ASGITransport(app=app)
    async with raw_app.router.lifespan_context(raw_app):
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8123") as client:
            async with streamable_http_client(
                "http://127.0.0.1:8123/mcp",
                http_client=client,
            ) as (read_stream, write_stream, _session_id):
                async with ClientSession(read_stream, write_stream) as session:
                    initialized = await session.initialize()
                    listed = await session.list_tools()
                    result = await session.call_tool("get_document_history", {"document_id": "943"})

    assert initialized.serverInfo.name == "BDDK"
    assert {tool.name for tool in listed.tools} == set(PUBLIC_TOOL_NAMES)
    assert result.isError is False
    assert result.content[0].text == "No version history found for document 943."


class _TestTokenVerifier:
    async def verify_token(self, token: str) -> AccessToken | None:
        if token == "invalid":
            return None
        scopes = ["bddk.read"] if token == "valid" else ["unrelated"]
        return AccessToken(
            token=token,
            client_id="test-client",
            scopes=scopes,
            expires_at=None,
            resource="https://mcp.bank.example/mcp",
            subject="test-subject",
            claims=None,
        )


def _remote_security(scopes: str = "bddk.read"):
    return load_http_security_config(
        {
            "MCP_HOST": "0.0.0.0",
            "PORT": "8443",
            "BDDK_HTTP_ALLOWED_HOSTS": "mcp.bank.example",
            "BDDK_HTTP_ALLOWED_ORIGINS": "https://client.bank.example",
            "BDDK_JWT_ISSUER": "https://id.bank.example/realms/bddk",
            "BDDK_JWT_RESOURCE": "https://mcp.bank.example/mcp",
            "BDDK_JWT_JWKS_URL": "https://id.bank.example/realms/bddk/jwks",
            "BDDK_JWT_AUDIENCE": "bddk-mcp",
            "BDDK_JWT_REQUIRED_SCOPES": scopes,
            "BDDK_JWT_ALGORITHMS": "RS256",
        }
    )


@pytest.mark.asyncio
async def test_http_auth_scope_and_origin_statuses_are_fail_closed():
    from bddk_mcp.server import create_mcp

    security = _remote_security()
    deps = Dependencies(pool=None, doc_store=MagicMock(), client=MagicMock(), http=None)
    with patch("bddk_mcp.server.JwtTokenVerifier", return_value=_TestTokenVerifier()):
        server = create_mcp(deps, profile=ToolProfile.PUBLIC, http_security=security)
    raw_app = server.streamable_http_app()
    app = HttpSecurityMiddleware(raw_app, security)

    transport = httpx.ASGITransport(app=app)
    async with raw_app.router.lifespan_context(raw_app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="https://mcp.bank.example",
            headers={"origin": "https://client.bank.example"},
        ) as client:
            missing = await client.post("/mcp", json={})
            invalid = await client.post("/mcp", json={}, headers={"authorization": "Bearer invalid"})
            under_scoped = await client.post("/mcp", json={}, headers={"authorization": "Bearer under-scoped"})
            bad_origin = await client.post("/mcp", json={}, headers={"origin": "https://evil.example"})

    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert under_scoped.status_code == 403
    assert bad_origin.status_code == 403
    assert "resource_metadata" in missing.headers["www-authenticate"]


@pytest.mark.asyncio
async def test_content_free_health_routes_support_orchestrator_host_headers():
    from bddk_mcp.server import create_mcp

    pool = MagicMock()
    deps = Dependencies(pool=pool, doc_store=MagicMock(), client=MagicMock(), http=None)
    security = load_http_security_config({"MCP_HOST": "127.0.0.1", "PORT": "8123"})
    server = create_mcp(deps, http_security=security)
    raw_app = server.streamable_http_app()
    app = HttpSecurityMiddleware(raw_app, security)
    schema_ready = AsyncMock()
    identity_ready = AsyncMock()

    with (
        patch("bddk_mcp.server.assert_database_ready", new=schema_ready),
        patch("bddk_mcp.server.assert_database_identity", new=identity_ready),
    ):
        async with raw_app.router.lifespan_context(raw_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://10.42.0.17",
            ) as client:
                live = await client.get("/health/live")
                ready = await client.get("/health/ready")
                cached_ready = await client.get("/health/ready")

    assert live.status_code == 200
    assert live.json() == {"status": "alive"}
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}
    assert cached_ready.status_code == 200
    schema_ready.assert_awaited_once_with(pool=pool)
    identity_ready.assert_awaited_once_with(pool, "public")


@pytest.mark.asyncio
async def test_readiness_fails_closed_when_periodic_identity_attestation_fails():
    from bddk_mcp.server import create_mcp

    pool = MagicMock()
    deps = Dependencies(pool=pool, doc_store=MagicMock(), client=MagicMock(), http=None)
    security = load_http_security_config({"MCP_HOST": "127.0.0.1", "PORT": "8123"})
    server = create_mcp(deps, http_security=security)
    raw_app = server.streamable_http_app()

    with (
        patch("bddk_mcp.server.assert_database_ready", new=AsyncMock()),
        patch(
            "bddk_mcp.server.assert_database_identity",
            new=AsyncMock(side_effect=RuntimeError("private ACL detail")),
        ),
    ):
        async with raw_app.router.lifespan_context(raw_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=raw_app),
                base_url="http://127.0.0.1:8123",
            ) as client:
                response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}
    assert "private ACL detail" not in response.text


def test_remote_operator_profile_requires_explicit_private_enablement_and_scope(monkeypatch):
    from bddk_mcp.http_security import HttpSecurityConfigError
    from bddk_mcp.server import create_mcp

    deps = Dependencies(pool=None, doc_store=MagicMock(), client=MagicMock(), http=None)
    monkeypatch.delenv("BDDK_OPERATOR_REMOTE_ENABLED", raising=False)
    with pytest.raises(HttpSecurityConfigError, match="disabled by default"):
        create_mcp(deps, profile=ToolProfile.OPERATOR, http_security=_remote_security("bddk.operator"))

    monkeypatch.setenv("BDDK_OPERATOR_REMOTE_ENABLED", "true")
    with pytest.raises(HttpSecurityConfigError, match="bddk.operator"):
        create_mcp(deps, profile=ToolProfile.OPERATOR, http_security=_remote_security("bddk.read"))
