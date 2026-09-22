from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from bddk_mcp.admin.config import AdminConfig, AdminConfigError
from bddk_mcp.admin.password_auth import PasswordAuthMiddleware, PasswordSessions

_PASSWORD = "correct-horse-battery-staple"
_ENV = {
    "BDDK_DATABASE_URL": "postgresql://reader@db.internal/bddk?sslmode=verify-full&sslrootcert=%2Fca.pem",
    "BDDK_ADMIN_HOST": "0.0.0.0",
    "BDDK_ADMIN_REMOTE_ENABLED": "true",
    "BDDK_HTTP_ALLOWED_HOSTS": "admin.example",
    "BDDK_HTTP_ALLOWED_ORIGINS": "https://admin.example",
    "BDDK_ADMIN_PASSWORD": _PASSWORD,
}


def test_remote_password_rejects_jwt_and_short_secrets() -> None:
    with pytest.raises(AdminConfigError):
        AdminConfig.from_env({**_ENV, "BDDK_JWT_ISSUER": "https://idp.example"})
    with pytest.raises(AdminConfigError):
        AdminConfig.from_env({**_ENV, "BDDK_ADMIN_PASSWORD": "too-short"})


def test_password_gate_accepts_only_the_configured_secret() -> None:
    config = AdminConfig.from_env(_ENV)
    sessions = PasswordSessions(config.password)

    async def page(_request):
        return PlainTextResponse("ok")

    app = PasswordAuthMiddleware(Starlette(routes=[Route("/documents", page)]), config, sessions)
    client = TestClient(app)
    denied = client.get("/documents", headers={"accept": "text/html", "host": "admin.example"}, follow_redirects=False)
    assert denied.status_code == 303 and denied.headers["location"] == "/login"
    assert not sessions.accepts("wrong-password-wrong-password")
    assert sessions.accepts(_PASSWORD)
