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
from bddk_mcp.store.corpus_db import CorpusReadiness
from bddk_mcp.tools.registry import PUBLIC_TOOL_NAMES
from bddk_mcp.tools.structured_outputs import CORPUS_SCOPE_WARNING, SOURCE_DATA_BEGIN, SOURCE_DATA_END


def _healthy_vector_store() -> MagicMock:
    vector_store = MagicMock()
    vector_store.semantic_search_ready = True
    return vector_store


@pytest.mark.asyncio
async def test_official_client_initializes_lists_and_calls_over_secured_local_http():
    from bddk_mcp.server import create_mcp

    doc_store = MagicMock()
    doc_store.get_document_history = AsyncMock(return_value=[])
    deps = Dependencies(db=None, doc_store=doc_store, client=MagicMock(), http=None)
    security = load_http_security_config({"MCP_HOST": "127.0.0.1", "PORT": "8123"})
    server = create_mcp(deps, http_security=security)
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
    text = result.content[0].text
    assert SOURCE_DATA_BEGIN in text
    assert "No version history found for document 943." in text
    assert SOURCE_DATA_END in text
    assert CORPUS_SCOPE_WARNING in text


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


def _unauthenticated_remote_security():
    return load_http_security_config(
        {
            "MCP_HOST": "0.0.0.0",
            "PORT": "8443",
            "BDDK_HTTP_ALLOWED_HOSTS": "mcp.bank.example",
            "BDDK_HTTP_ALLOWED_ORIGINS": "https://client.bank.example",
            "BDDK_HTTP_ALLOW_UNAUTHENTICATED": "true",
        }
    )


@pytest.mark.asyncio
async def test_http_auth_scope_and_origin_statuses_are_fail_closed():
    from bddk_mcp.server import create_mcp

    security = _remote_security()
    deps = Dependencies(db=None, doc_store=MagicMock(), client=MagicMock(), http=None)
    with patch("bddk_mcp.server.JwtTokenVerifier", return_value=_TestTokenVerifier()):
        server = create_mcp(deps, http_security=security)
    raw_app = server.streamable_http_app()
    app = HttpSecurityMiddleware(raw_app, security)

    transport = httpx.ASGITransport(app=app)
    async with raw_app.router.lifespan_context(raw_app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="https://mcp.bank.example",
            headers={"origin": "https://client.bank.example"},
        ) as client:
            protected_resource = await client.get("/.well-known/oauth-protected-resource/mcp")
            missing = await client.post("/mcp", json={})
            invalid = await client.post("/mcp", json={}, headers={"authorization": "Bearer invalid"})
            under_scoped = await client.post("/mcp", json={}, headers={"authorization": "Bearer under-scoped"})
            bad_origin = await client.post("/mcp", json={}, headers={"origin": "https://evil.example"})

    assert protected_resource.status_code == 200
    assert protected_resource.json() == {
        "resource": "https://mcp.bank.example/mcp",
        "authorization_servers": ["https://id.bank.example/realms/bddk"],
        "scopes_supported": ["bddk.read"],
        "bearer_methods_supported": ["header"],
    }
    assert missing.status_code == 401
    assert invalid.status_code == 401
    assert under_scoped.status_code == 403
    assert bad_origin.status_code == 403
    assert (
        'resource_metadata="https://mcp.bank.example/.well-known/oauth-protected-resource/mcp"'
        in missing.headers["www-authenticate"]
    )


@pytest.mark.asyncio
async def test_content_free_health_routes_support_orchestrator_host_headers():
    from bddk_mcp.server import create_mcp

    db = MagicMock()
    deps = Dependencies(
        db=db,
        doc_store=MagicMock(),
        client=MagicMock(),
        http=None,
        vector_store=_healthy_vector_store(),
    )
    security = load_http_security_config({"MCP_HOST": "127.0.0.1", "PORT": "8123"})
    server = create_mcp(deps, http_security=security)
    raw_app = server.streamable_http_app()
    app = HttpSecurityMiddleware(raw_app, security)
    schema_ready = AsyncMock(return_value=CorpusReadiness())

    with patch.object(db, "assert_ready", new=schema_ready, create=True):
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
    schema_ready.assert_awaited_once_with(require_corpus=True)


@pytest.mark.asyncio
async def test_public_readiness_fails_closed_when_semantic_search_is_unhealthy():
    from bddk_mcp.server import create_mcp

    db = MagicMock()
    vector_store = MagicMock()
    vector_store.semantic_search_ready = False
    deps = Dependencies(
        db=db,
        doc_store=MagicMock(),
        client=MagicMock(),
        http=None,
        vector_store=vector_store,
    )
    security = load_http_security_config({"MCP_HOST": "127.0.0.1", "PORT": "8123"})
    server = create_mcp(deps, http_security=security)
    raw_app = server.streamable_http_app()

    async with raw_app.router.lifespan_context(raw_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=raw_app),
            base_url="http://127.0.0.1:8123",
        ) as client:
            response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_public_profile_accepts_unauthenticated_remote_exposure():
    from bddk_mcp.server import create_mcp

    deps = Dependencies(db=None, doc_store=MagicMock(), client=MagicMock(), http=None)
    server = create_mcp(deps, http_security=_unauthenticated_remote_security())

    assert server.settings.auth is None


@pytest.mark.asyncio
async def test_unauthenticated_public_server_advertises_no_oauth():
    from bddk_mcp.server import create_mcp

    security = _unauthenticated_remote_security()
    assert security.jwt_issuer is None

    deps = Dependencies(db=None, doc_store=MagicMock(), client=MagicMock(), http=None)
    server = create_mcp(deps, http_security=security)
    raw_app = server.streamable_http_app()
    app = HttpSecurityMiddleware(raw_app, security)

    transport = httpx.ASGITransport(app=app)
    async with raw_app.router.lifespan_context(raw_app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="https://mcp.bank.example",
            headers={"origin": "https://client.bank.example"},
        ) as client:
            protected_resource = await client.get("/.well-known/oauth-protected-resource/mcp")
            authorization_server = await client.get("/.well-known/oauth-authorization-server")
            unauthenticated = await client.post("/mcp", json={})
            bad_origin = await client.post("/mcp", json={}, headers={"origin": "https://evil.example"})

    # No OAuth is advertised, so no connector will start a flow against the
    # decommissioned issuer.
    assert protected_resource.status_code == 404
    assert authorization_server.status_code == 404
    assert "www-authenticate" not in unauthenticated.headers
    assert unauthenticated.status_code != 401

    # Transport checks still apply with authentication removed.
    assert bad_origin.status_code == 403


@pytest.mark.asyncio
async def test_official_client_full_round_trip_on_unauthenticated_remote_public_profile():
    """Exercise the delivered bank configuration end to end.

    The bank topology serves the PUBLIC profile non-loopback with the explicit
    unauthenticated opt-in and exact Host/Origin allowlists; access control is
    the network boundary. This test drives initialize -> tools/list ->
    tools/call through the official client against exactly that composition,
    so the shipped configuration's happy path is exercised, not inferred.
    """
    from bddk_mcp.server import create_mcp

    security = _unauthenticated_remote_security()
    doc_store = MagicMock()
    doc_store.get_document_history = AsyncMock(return_value=[])
    deps = Dependencies(db=None, doc_store=doc_store, client=MagicMock(), http=None)
    server = create_mcp(deps, http_security=security)
    raw_app = server.streamable_http_app()
    app = HttpSecurityMiddleware(raw_app, security)

    transport = httpx.ASGITransport(app=app)
    async with raw_app.router.lifespan_context(raw_app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="https://mcp.bank.example",
            headers={"origin": "https://client.bank.example"},
        ) as client:
            async with streamable_http_client(
                "https://mcp.bank.example/mcp",
                http_client=client,
            ) as (read_stream, write_stream, _session_id):
                async with ClientSession(read_stream, write_stream) as session:
                    initialized = await session.initialize()
                    listed = await session.list_tools()
                    result = await session.call_tool("get_document_history", {"document_id": "943"})

    assert initialized.serverInfo.name == "BDDK"
    assert {tool.name for tool in listed.tools} == set(PUBLIC_TOOL_NAMES)
    assert result.content, "tools/call must return content without any bearer token"
    assert not result.isError, "the delivered unauthenticated profile must serve tool calls, not error them"
