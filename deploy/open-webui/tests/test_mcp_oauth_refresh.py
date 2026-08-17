import asyncio
import time
from functools import wraps
from types import SimpleNamespace

import httpx
import pytest
from mcp.server.fastmcp import FastMCP
from open_webui.utils import middleware as middleware_module
from open_webui.utils import oauth as oauth_module
from open_webui.utils.mcp import client as mcp_client_module
from open_webui.utils.mcp.client import RefreshingBearerAuth


def async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


@async_test
async def test_auth_uses_current_token_without_retry():
    token_calls = []
    requests = []

    async def get_token(force_refresh, invalid_access_token):
        token_calls.append((force_refresh, invalid_access_token))
        return "current"

    async def handler(request):
        requests.append((request.headers["Authorization"], await request.aread()))
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=RefreshingBearerAuth(get_token),
    ) as client:
        response = await client.post("https://mcp.example/mcp", json={"id": 1})

    assert response.status_code == 200
    assert token_calls == [(False, None)]
    assert [authorization for authorization, _ in requests] == ["Bearer current"]


@async_test
async def test_auth_refreshes_and_replays_buffered_request_once_on_401():
    token_calls = []
    requests = []

    async def get_token(force_refresh, invalid_access_token):
        token_calls.append((force_refresh, invalid_access_token))
        return "new" if force_refresh else "old"

    async def handler(request):
        body = await request.aread()
        authorization = request.headers["Authorization"]
        requests.append((authorization, body))
        status = 401 if authorization == "Bearer old" else 200
        return httpx.Response(status, json={"status": status})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=RefreshingBearerAuth(get_token),
    ) as client:
        response = await client.post("https://mcp.example/mcp", json={"method": "tools/call"})

    assert response.status_code == 200
    assert token_calls == [(False, None), (True, "old")]
    assert [authorization for authorization, _ in requests] == ["Bearer old", "Bearer new"]
    assert requests[0][1] == requests[1][1]


@async_test
async def test_auth_stops_after_one_retry_and_overwrites_stale_header():
    sends = 0

    async def get_token(force_refresh, invalid_access_token):
        return "new" if force_refresh else "old"

    async def handler(request):
        nonlocal sends
        sends += 1
        assert request.headers["Authorization"] in {"Bearer old", "Bearer new"}
        return httpx.Response(401)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=RefreshingBearerAuth(get_token),
        headers={"Authorization": "Bearer stale"},
    ) as client:
        response = await client.post("https://mcp.example/mcp", content=b"{}")

    assert response.status_code == 401
    assert sends == 2


@async_test
async def test_auth_rejects_missing_access_token_without_sending_request():
    sends = 0

    async def get_token(force_refresh, invalid_access_token):
        return None

    async def handler(request):
        nonlocal sends
        sends += 1
        return httpx.Response(200)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        auth=RefreshingBearerAuth(get_token),
    ) as client:
        with pytest.raises(RuntimeError, match="authorization is required"):
            await client.post("https://mcp.example/mcp", content=b"{}")

    assert sends == 0


@async_test
async def test_real_mcp_transport_refreshes_after_connection_is_established(monkeypatch):
    server = FastMCP("oauth-refresh-test", stateless_http=True, json_response=True)

    @server.tool()
    def ping():
        return "pong"

    server_app = server.streamable_http_app()
    state = {"access_token": "old", "reject_old": False}
    token_calls = []

    async def get_token(force_refresh, invalid_access_token):
        token_calls.append((force_refresh, invalid_access_token))
        if force_refresh:
            state["access_token"] = "new"
        return state["access_token"]

    class RejectExpiredBearer:
        async def __call__(self, scope, receive, send):
            headers = dict(scope.get("headers", []))
            if scope["type"] == "http" and state["reject_old"] and headers.get(b"authorization") != b"Bearer new":
                await send(
                    {
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send({"type": "http.response.body", "body": b'{"error":"expired"}'})
                return
            await server_app(scope, receive, send)

    transport = httpx.ASGITransport(app=RejectExpiredBearer())

    def httpx_client_factory(headers=None, timeout=None, auth=None):
        return httpx.AsyncClient(
            transport=transport,
            headers=headers,
            timeout=timeout,
            auth=auth,
        )

    monkeypatch.setattr(mcp_client_module, "create_httpx_client", httpx_client_factory)
    monkeypatch.setattr(mcp_client_module, "create_insecure_httpx_client", httpx_client_factory)

    client = mcp_client_module.MCPClient()
    async with server_app.router.lifespan_context(server_app):
        await client.connect(
            "http://localhost:8000/mcp",
            auth=RefreshingBearerAuth(get_token),
        )
        state["reject_old"] = True
        specs = await client.list_tool_specs()
        result = await client.call_tool("ping", {})
        await client.disconnect()

    assert [spec["name"] for spec in specs] == ["ping"]
    assert result[0]["text"] == "pong"
    assert token_calls.count((True, "old")) == 1


class SessionStore:
    def __init__(self, session):
        self.session = session
        self.deleted_ids = []

    async def get_session_by_provider_and_user_id(self, client_id, user_id):
        return self.session

    async def delete_session_by_id(self, session_id):
        self.deleted_ids.append(session_id)
        self.session = None


class MultiSessionStore:
    def __init__(self, sessions):
        self.sessions = sessions

    async def get_session_by_provider_and_user_id(self, client_id, user_id):
        return self.sessions[(client_id, user_id)]


class FreshReadSessionStore:
    def __init__(self, session):
        self.data = {
            "id": session.id,
            "provider": session.provider,
            "token": dict(session.token),
            "expires_at": session.expires_at,
        }

    def fresh(self):
        return SimpleNamespace(
            id=self.data["id"],
            provider=self.data["provider"],
            token=dict(self.data["token"]),
            expires_at=self.data["expires_at"],
        )

    async def get_session_by_provider_and_user_id(self, client_id, user_id):
        return self.fresh()

    async def update_session_by_id(self, session_id, token):
        assert session_id == self.data["id"]
        self.data["token"] = dict(token)
        self.data["expires_at"] = token["expires_at"]
        return self.fresh()


def oauth_manager():
    manager = oauth_module.OAuthClientManager.__new__(oauth_module.OAuthClientManager)
    manager._refresh_locks = {}
    manager._refresh_failures = {}
    return manager


@async_test
async def test_five_minute_token_is_not_refreshed_immediately(monkeypatch):
    session = SimpleNamespace(
        id="session",
        provider="mcp:bddk",
        token={"access_token": "current"},
        expires_at=int(time.time()) + 300,
    )
    monkeypatch.setattr(oauth_module, "OAuthSessions", SessionStore(session))
    manager = oauth_manager()
    refreshes = 0

    async def refresh(_session):
        nonlocal refreshes
        refreshes += 1
        return {"access_token": "unexpected"}

    manager._refresh_token = refresh
    token = await manager.get_oauth_token("user", "mcp:bddk")

    assert token["access_token"] == "current"
    assert refreshes == 0


@async_test
async def test_concurrent_expiry_refreshes_are_coalesced(monkeypatch):
    session = SimpleNamespace(
        id="session",
        provider="mcp:bddk",
        token={"access_token": "old"},
        expires_at=int(time.time()) + 1,
    )
    monkeypatch.setattr(oauth_module, "OAuthSessions", SessionStore(session))
    manager = oauth_manager()
    refreshes = 0

    async def refresh(_session):
        nonlocal refreshes
        refreshes += 1
        await asyncio.sleep(0)
        session.token = {"access_token": "new"}
        session.expires_at = int(time.time()) + 300
        return session.token

    manager._refresh_token = refresh
    first, second = await asyncio.gather(
        manager.get_oauth_token("user", "mcp:bddk"),
        manager.get_oauth_token("user", "mcp:bddk"),
    )

    assert first["access_token"] == second["access_token"] == "new"
    assert refreshes == 1


@async_test
async def test_concurrent_401_refreshes_reuse_replacement_token(monkeypatch):
    session = SimpleNamespace(
        id="session",
        provider="mcp:bddk",
        token={"access_token": "rejected"},
        expires_at=int(time.time()) + 300,
    )
    monkeypatch.setattr(oauth_module, "OAuthSessions", SessionStore(session))
    manager = oauth_manager()
    refreshes = 0

    async def refresh(_session):
        nonlocal refreshes
        refreshes += 1
        await asyncio.sleep(0)
        session.token = {"access_token": "replacement"}
        session.expires_at = int(time.time()) + 300
        return session.token

    manager._refresh_token = refresh
    first, second = await asyncio.gather(
        manager.get_oauth_token("user", "mcp:bddk", force_refresh=True, invalid_access_token="rejected"),
        manager.get_oauth_token("user", "mcp:bddk", force_refresh=True, invalid_access_token="rejected"),
    )

    assert first["access_token"] == second["access_token"] == "replacement"
    assert refreshes == 1


@async_test
async def test_users_refresh_independently(monkeypatch):
    sessions = {
        ("mcp:bddk", "user-1"): SimpleNamespace(
            id="session-1",
            provider="mcp:bddk",
            token={"access_token": "old-1"},
            expires_at=int(time.time()) + 1,
        ),
        ("mcp:bddk", "user-2"): SimpleNamespace(
            id="session-2",
            provider="mcp:bddk",
            token={"access_token": "old-2"},
            expires_at=int(time.time()) + 1,
        ),
    }
    monkeypatch.setattr(oauth_module, "OAuthSessions", MultiSessionStore(sessions))
    manager = oauth_manager()
    refreshed_sessions = []

    async def refresh(session):
        refreshed_sessions.append(session.id)
        session.token = {"access_token": f"new-{session.id[-1]}"}
        session.expires_at = int(time.time()) + 300
        return session.token

    manager._refresh_token = refresh
    first, second = await asyncio.gather(
        manager.get_oauth_token("user-1", "mcp:bddk"),
        manager.get_oauth_token("user-2", "mcp:bddk"),
    )

    assert first["access_token"] == "new-1"
    assert second["access_token"] == "new-2"
    assert set(refreshed_sessions) == {"session-1", "session-2"}


@async_test
async def test_rotated_tokens_are_persisted_for_fresh_reads(monkeypatch):
    session = SimpleNamespace(
        id="session",
        provider="mcp:bddk",
        token={"access_token": "old", "refresh_token": "old-refresh"},
        expires_at=int(time.time()) - 1,
    )
    store = FreshReadSessionStore(session)
    monkeypatch.setattr(oauth_module, "OAuthSessions", store)
    manager = oauth_manager()
    replacement = {
        "access_token": "new",
        "refresh_token": "rotated-refresh",
        "expires_at": int(time.time()) + 300,
    }

    async def perform_refresh(_session):
        return replacement

    manager._perform_token_refresh = perform_refresh
    refreshed = await manager._refresh_token(store.fresh())
    next_read = await oauth_manager().get_oauth_token("user", "mcp:bddk")

    assert refreshed == replacement
    assert next_read == replacement
    assert store.data["token"]["refresh_token"] == "rotated-refresh"


@async_test
async def test_transient_proactive_refresh_failure_preserves_valid_token(monkeypatch):
    session = SimpleNamespace(
        id="session",
        provider="mcp:bddk",
        token={"access_token": "still-valid", "refresh_token": "keep-me"},
        expires_at=int(time.time()) + 10,
    )
    store = SessionStore(session)
    monkeypatch.setattr(oauth_module, "OAuthSessions", store)
    manager = oauth_manager()

    async def failed_refresh(_session):
        return None

    manager._refresh_token = failed_refresh
    token = await manager.get_oauth_token("user", "mcp:bddk")

    assert token == session.token
    assert store.session.token["refresh_token"] == "keep-me"


@async_test
async def test_failed_refresh_does_not_return_token_that_expired_while_waiting(monkeypatch):
    clock = {"now": 1_000.0}
    session = SimpleNamespace(
        id="session",
        provider="mcp:bddk",
        token={"access_token": "old", "refresh_token": "keep-me"},
        expires_at=1_001,
    )
    store = SessionStore(session)
    monkeypatch.setattr(oauth_module, "OAuthSessions", store)
    monkeypatch.setattr(oauth_module.time, "time", lambda: clock["now"])
    manager = oauth_manager()

    async def failed_refresh(_session):
        clock["now"] = 1_002.0
        return None

    manager._refresh_token = failed_refresh
    token = await manager.get_oauth_token("user", "mcp:bddk")

    assert token is None
    assert store.session.token["refresh_token"] == "keep-me"


@async_test
async def test_concurrent_failed_refreshes_use_one_idp_attempt(monkeypatch):
    session = SimpleNamespace(
        id="session",
        provider="mcp:bddk",
        token={"access_token": "old", "refresh_token": "keep-me"},
        expires_at=int(time.time()) + 10,
    )
    monkeypatch.setattr(oauth_module, "OAuthSessions", SessionStore(session))
    manager = oauth_manager()
    refreshes = 0

    async def failed_refresh(_session):
        nonlocal refreshes
        refreshes += 1
        await asyncio.sleep(0)
        return None

    manager._refresh_token = failed_refresh
    tokens = await asyncio.gather(*(manager.get_oauth_token("user", "mcp:bddk") for _ in range(5)))

    assert all(token == session.token for token in tokens)
    assert refreshes == 1


@async_test
async def test_concurrent_failed_forced_refreshes_fail_closed_after_one_attempt(monkeypatch):
    session = SimpleNamespace(
        id="session",
        provider="mcp:bddk",
        token={"access_token": "rejected", "refresh_token": "keep-me"},
        expires_at=int(time.time()) + 300,
    )
    monkeypatch.setattr(oauth_module, "OAuthSessions", SessionStore(session))
    manager = oauth_manager()
    refreshes = 0

    async def failed_refresh(_session):
        nonlocal refreshes
        refreshes += 1
        await asyncio.sleep(0)
        return None

    manager._refresh_token = failed_refresh
    tokens = await asyncio.gather(
        *(
            manager.get_oauth_token(
                "user",
                "mcp:bddk",
                force_refresh=True,
                invalid_access_token="rejected",
            )
            for _ in range(5)
        )
    )

    assert tokens == [None] * 5
    assert refreshes == 1


@async_test
async def test_permanently_rejected_refresh_credential_requires_reconnect(monkeypatch):
    session = SimpleNamespace(
        id="session",
        provider="mcp:bddk",
        token={"access_token": "rejected", "refresh_token": "revoked"},
        expires_at=int(time.time()) - 1,
    )
    store = SessionStore(session)
    monkeypatch.setattr(oauth_module, "OAuthSessions", store)
    manager = oauth_manager()

    async def permanently_failed_refresh(_session):
        raise oauth_module.PermanentOAuthRefreshError("invalid_grant")

    manager._refresh_token = permanently_failed_refresh
    token = await manager.get_oauth_token("user", "mcp:bddk")

    assert token is None
    assert store.deleted_ids == ["session"]


@async_test
async def test_cancelled_refresh_releases_lock_for_next_request(monkeypatch):
    session = SimpleNamespace(
        id="session",
        provider="mcp:bddk",
        token={"access_token": "old", "refresh_token": "keep-me"},
        expires_at=int(time.time()) - 1,
    )
    monkeypatch.setattr(oauth_module, "OAuthSessions", SessionStore(session))
    manager = oauth_manager()
    refresh_started = asyncio.Event()
    never_finishes = asyncio.Event()

    async def cancelled_refresh(_session):
        refresh_started.set()
        await never_finishes.wait()

    manager._refresh_token = cancelled_refresh
    first = asyncio.create_task(manager.get_oauth_token("user", "mcp:bddk"))
    await refresh_started.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    async def successful_refresh(_session):
        session.token = {"access_token": "new", "refresh_token": "keep-me"}
        session.expires_at = int(time.time()) + 300
        return session.token

    manager._refresh_token = successful_refresh
    token = await asyncio.wait_for(manager.get_oauth_token("user", "mcp:bddk"), timeout=1)

    assert token["access_token"] == "new"


class FakeMCPClient:
    instances = []

    def __init__(self):
        self.connected = None
        self.__class__.instances.append(self)

    async def connect(self, url, headers=None, auth=None):
        self.connected = {"url": url, "headers": headers, "auth": auth}

    async def list_tool_specs(self):
        return []


@async_test
async def test_middleware_binds_dynamic_auth_to_user_and_mcp_connection(monkeypatch):
    connection = {
        "type": "mcp",
        "url": "https://mcp.example/mcp",
        "auth_type": "oauth_2.1_static",
        "info": {"id": "bddk"},
        "config": {},
    }
    header_auth_types = []
    token_calls = []

    async def config_get(key, default=None):
        return [connection]

    async def access_allowed(user, candidate):
        return True

    async def build_headers(candidate, *args, **kwargs):
        header_auth_types.append(candidate["auth_type"])
        return {"X-Custom": "retained"}, {}

    class TokenManager:
        async def get_oauth_token(self, user_id, client_id, **kwargs):
            token_calls.append((user_id, client_id, kwargs))
            return {"access_token": "current"}

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(oauth_client_manager=TokenManager())))
    user = SimpleNamespace(id="user-1")
    FakeMCPClient.instances = []
    monkeypatch.setattr(middleware_module.Config, "get", config_get)
    monkeypatch.setattr(middleware_module, "has_connection_access", access_allowed)
    monkeypatch.setattr(middleware_module, "build_tool_server_headers", build_headers)
    monkeypatch.setattr(middleware_module, "MCPClient", FakeMCPClient)

    client, specs = await middleware_module.connect_mcp_server(request, "bddk", user, {}, {})

    assert specs == []
    assert header_auth_types == ["none"]
    assert client.connected["headers"] == {"X-Custom": "retained"}
    assert isinstance(client.connected["auth"], RefreshingBearerAuth)
    assert await client.connected["auth"]._token_getter(False, None) == "current"
    assert token_calls == [
        (
            "user-1",
            "mcp:bddk",
            {"force_refresh": False, "invalid_access_token": None},
        )
    ]


@async_test
async def test_middleware_keeps_non_oauth_mcp_auth_unchanged(monkeypatch):
    connection = {
        "type": "mcp",
        "url": "https://mcp.example/mcp",
        "auth_type": "bearer",
        "key": "static-key",
        "info": {"id": "bddk"},
        "config": {},
    }

    async def config_get(key, default=None):
        return [connection]

    async def access_allowed(user, candidate):
        return True

    async def build_headers(candidate, *args, **kwargs):
        assert candidate is connection
        return {"Authorization": "Bearer static-key"}, {}

    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    user = SimpleNamespace(id="user-1")
    FakeMCPClient.instances = []
    monkeypatch.setattr(middleware_module.Config, "get", config_get)
    monkeypatch.setattr(middleware_module, "has_connection_access", access_allowed)
    monkeypatch.setattr(middleware_module, "build_tool_server_headers", build_headers)
    monkeypatch.setattr(middleware_module, "MCPClient", FakeMCPClient)

    client, _ = await middleware_module.connect_mcp_server(request, "bddk", user, {}, {})

    assert client.connected["headers"] == {"Authorization": "Bearer static-key"}
    assert client.connected["auth"] is None
