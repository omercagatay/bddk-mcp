"""Official-client coverage for the secured Streamable HTTP composition."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.auth.provider import AccessToken

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.corpus_manifest import CORPUS_SCOPE_WARNING
from bddk_mcp.corpus_publication import CorpusReleaseIdentity
from bddk_mcp.db_lifecycle import DatabaseReadiness
from bddk_mcp.http_security import HttpSecurityMiddleware, load_http_security_config
from bddk_mcp.resources import ACTIVE_CORPUS_RELEASE_RESOURCE_URI
from bddk_mcp.tools.registry import PUBLIC_TOOL_NAMES, ToolProfile
from bddk_mcp.tools.structured_outputs import SOURCE_DATA_BEGIN, SOURCE_DATA_END


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
                    listed_resources = await session.list_resources()
                    release_resource = await session.read_resource(ACTIVE_CORPUS_RELEASE_RESOURCE_URI)
                    result = await session.call_tool("get_document_history", {"document_id": "943"})

    assert initialized.serverInfo.name == "BDDK"
    assert {tool.name for tool in listed.tools} == set(PUBLIC_TOOL_NAMES)
    assert [str(resource.uri) for resource in listed_resources.resources] == [ACTIVE_CORPUS_RELEASE_RESOURCE_URI]
    assert json.loads(release_resource.contents[0].text) == {
        "schema_version": "1.0",
        "status": "unavailable",
    }
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

    pool = MagicMock()
    deps = Dependencies(pool=pool, doc_store=MagicMock(), client=MagicMock(), http=None)
    security = load_http_security_config({"MCP_HOST": "127.0.0.1", "PORT": "8123"})
    server = create_mcp(deps, http_security=security)
    raw_app = server.streamable_http_app()
    app = HttpSecurityMiddleware(raw_app, security)
    schema_ready = AsyncMock(return_value=DatabaseReadiness())
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
    schema_ready.assert_awaited_once_with(
        pool=pool,
        require_corpus=True,
        require_active_release=False,
    )
    identity_ready.assert_awaited_once_with(pool, "public")


def _active_release() -> CorpusReleaseIdentity:
    return CorpusReleaseIdentity(
        release_id="corpus_release_sha256_" + "1" * 64,
        manifest_id="release-test-001",
        manifest_sha256="2" * 64,
        signer_key_sha256="3" * 64,
        freshness_policy_result="quantified_measured_signature_verified_pass",
        source_detection_slo_seconds=60,
        publication_slo_seconds=120,
        max_manifest_age_seconds=3600,
        retrieval_profile_sha256="4" * 64,
        corpus_state_sha256="5" * 64,
        completed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_remote_release_resource_requires_auth_and_omits_operator_only_evidence():
    from bddk_mcp.server import create_mcp

    release = _active_release()
    pool = MagicMock()
    deps = Dependencies(pool=pool, doc_store=MagicMock(), client=MagicMock(), http=None)
    security = _remote_security()
    with patch("bddk_mcp.server.JwtTokenVerifier", return_value=_TestTokenVerifier()):
        server = create_mcp(deps, profile=ToolProfile.PUBLIC, http_security=security)
    raw_app = server.streamable_http_app()
    app = HttpSecurityMiddleware(raw_app, security)

    with patch("bddk_mcp.resources.inspect_active_corpus_release", new=AsyncMock(return_value=release)):
        async with raw_app.router.lifespan_context(raw_app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="https://mcp.bank.example",
                headers={
                    "origin": "https://client.bank.example",
                    "authorization": "Bearer valid",
                },
            ) as client:
                async with streamable_http_client(
                    "https://mcp.bank.example/mcp",
                    http_client=client,
                ) as (read_stream, write_stream, _session_id):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        listed = await session.list_resources()
                        resource = await session.read_resource(ACTIVE_CORPUS_RELEASE_RESOURCE_URI)

            async with httpx.AsyncClient(
                transport=transport,
                base_url="https://mcp.bank.example",
                headers={"origin": "https://client.bank.example"},
            ) as unauthenticated:
                denied = await unauthenticated.post(
                    "/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "resources/read",
                        "params": {"uri": ACTIVE_CORPUS_RELEASE_RESOURCE_URI},
                    },
                )

    assert [str(item.uri) for item in listed.resources] == [ACTIVE_CORPUS_RELEASE_RESOURCE_URI]
    payload = json.loads(resource.contents[0].text)
    assert payload == {
        "schema_version": "1.0",
        "status": "active",
        "release_id": release.release_id,
        "manifest_id": release.manifest_id,
        "manifest_sha256": release.manifest_sha256,
        "retrieval_profile_sha256": release.retrieval_profile_sha256,
    }
    assert release.signer_key_sha256 not in resource.contents[0].text
    assert release.corpus_state_sha256 not in resource.contents[0].text
    assert denied.status_code == 401


@pytest.mark.asyncio
async def test_strict_health_rechecks_active_release_and_exposes_only_opaque_id():
    from bddk_mcp import server as server_module

    release = _active_release()
    pool = MagicMock()
    deps = Dependencies(pool=pool, doc_store=MagicMock(), client=MagicMock(), http=None)
    security = load_http_security_config({"MCP_HOST": "127.0.0.1", "PORT": "8123"})
    with patch.object(server_module, "REQUIRE_ACTIVE_CORPUS_RELEASE", True):
        server = server_module.create_mcp(deps, http_security=security)
    raw_app = server.streamable_http_app()
    schema_ready = AsyncMock(return_value=DatabaseReadiness(active_corpus_release=release))
    current_release = AsyncMock(side_effect=(release, None))

    with (
        patch.object(server_module, "REQUIRE_ACTIVE_CORPUS_RELEASE", True),
        patch.object(server_module, "assert_database_ready", new=schema_ready),
        patch.object(server_module, "assert_database_identity", new=AsyncMock()),
        patch.object(server_module, "inspect_active_corpus_release", new=current_release),
    ):
        async with raw_app.router.lifespan_context(raw_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=raw_app),
                base_url="http://127.0.0.1:8123",
            ) as client:
                live = await client.get("/health/live")
                ready = await client.get("/health/ready")
                invalidated = await client.get("/health/ready")

    assert live.json() == {"status": "alive"}
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "active_corpus_release_id": release.release_id}
    assert invalidated.status_code == 503
    assert invalidated.json() == {"status": "not_ready"}
    assert schema_ready.await_count == 1
    assert current_release.await_count == 2
    sensitive_values = {
        release.manifest_id,
        release.manifest_sha256,
        release.signer_key_sha256,
        release.retrieval_profile_sha256,
        release.corpus_state_sha256,
        str(release.source_detection_slo_seconds),
        str(release.publication_slo_seconds),
        str(release.max_manifest_age_seconds),
    }
    public_health_text = live.text + ready.text + invalidated.text
    assert all(value not in public_health_text for value in sensitive_values)


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


def test_operator_profile_refuses_unauthenticated_remote_exposure(monkeypatch):
    from bddk_mcp.http_security import HttpSecurityConfigError
    from bddk_mcp.server import create_mcp

    deps = Dependencies(pool=None, doc_store=MagicMock(), client=MagicMock(), http=None)

    monkeypatch.delenv("BDDK_OPERATOR_REMOTE_ENABLED", raising=False)
    with pytest.raises(HttpSecurityConfigError, match="never be exposed unauthenticated"):
        create_mcp(deps, profile=ToolProfile.OPERATOR, http_security=_unauthenticated_remote_security())

    # The two opt-ins must not compose: enabling remote operator exposure does not
    # unlock unauthenticated operator exposure.
    monkeypatch.setenv("BDDK_OPERATOR_REMOTE_ENABLED", "true")
    with pytest.raises(HttpSecurityConfigError, match="never be exposed unauthenticated"):
        create_mcp(deps, profile=ToolProfile.OPERATOR, http_security=_unauthenticated_remote_security())


def test_public_profile_accepts_unauthenticated_remote_exposure():
    from bddk_mcp.server import create_mcp

    deps = Dependencies(pool=None, doc_store=MagicMock(), client=MagicMock(), http=None)
    server = create_mcp(deps, profile=ToolProfile.PUBLIC, http_security=_unauthenticated_remote_security())

    assert server.settings.auth is None


@pytest.mark.asyncio
async def test_unauthenticated_public_server_advertises_no_oauth():
    from bddk_mcp.server import create_mcp

    security = _unauthenticated_remote_security()
    assert security.jwt_issuer is None

    deps = Dependencies(pool=None, doc_store=MagicMock(), client=MagicMock(), http=None)
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
