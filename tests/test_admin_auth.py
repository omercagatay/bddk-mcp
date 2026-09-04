from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from bddk_mcp.admin.app import create_app
from bddk_mcp.admin.config import AdminConfig
from bddk_mcp.admin.services.documents import DocumentService
from bddk_mcp.store.doc_store import StoreStats


@pytest.fixture(autouse=True)
def _allow_insecure_database(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BDDK_ALLOW_INSECURE_DATABASE", "true")


class _Store:
    async def list_documents(self, category=None, limit=100, offset=0):
        return [{"document_id": "mevzuat_1", "title": "Bankacilik Kanunu", "category": "mevzuat", "total_pages": 1}]

    async def stats(self):
        return StoreStats(categories={"mevzuat": 1}, total_documents=1)


class _Governance:
    async def status(self):
        raise AssertionError("governance status must not be fetched by auth tests")


class _Verifier:
    async def verify_token(self, token: str):
        return object() if token == "good" else None


def _remote_config() -> AdminConfig:
    return AdminConfig.from_env(
        {
            "BDDK_DATABASE_URL": "postgresql://u:p@localhost:5432/bddk",
            "BDDK_ADMIN_HOST": "0.0.0.0",
            "BDDK_ADMIN_PORT": "8443",
            "BDDK_ADMIN_REMOTE_ENABLED": "true",
            "BDDK_HTTP_ALLOWED_HOSTS": "admin.bank.example:8443",
            "BDDK_HTTP_ALLOWED_ORIGINS": "https://admin.bank.example",
            "BDDK_JWT_ISSUER": "https://id.bank.example/realms/bddk",
            "BDDK_JWT_RESOURCE": "https://mcp.bank.example/mcp",
            "BDDK_JWT_JWKS_URL": "https://id.bank.example/realms/bddk/jwks",
            "BDDK_JWT_AUDIENCE": "bddk-mcp",
            "BDDK_JWT_REQUIRED_SCOPES": "bddk.operator",
        }
    )


HOST = {"host": "admin.bank.example:8443"}


def _client() -> TestClient:
    return TestClient(
        create_app(
            _remote_config(),
            DocumentService(_Store()),
            _Governance(),
            token_verifier=_Verifier(),
        ),
        base_url="https://admin.bank.example:8443",
    )


def test_remote_documents_without_token_are_unauthorized() -> None:
    response = _client().get("/documents", headers=HOST)
    assert response.status_code == 401


def test_remote_html_get_without_token_redirects_to_login() -> None:
    response = _client().get("/documents", headers={**HOST, "accept": "text/html"}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_remote_wrong_host_is_rejected() -> None:
    response = _client().get("/documents", headers={"host": "evil.example"})
    assert response.status_code == 421


def test_remote_bearer_token_opens_documents() -> None:
    response = _client().get("/documents", headers={**HOST, "authorization": "Bearer good"})
    assert response.status_code == 200
    assert "Bankacilik Kanunu" in response.text


def test_login_sets_cookie_and_opens_documents() -> None:
    client = _client()
    response = client.post("/login", headers=HOST, data={"token": "good"}, follow_redirects=False)
    assert response.status_code == 303
    assert client.cookies.get("bddk_admin") == "good"
    listed = client.get("/documents", headers=HOST)
    assert listed.status_code == 200
    assert "Bankacilik Kanunu" in listed.text


def test_login_rejects_bad_token() -> None:
    response = _client().post("/login", headers=HOST, data={"token": "bad"})
    assert response.status_code == 401
    assert "Gecersiz" in response.text


def test_health_is_open() -> None:
    response = _client().get("/health/live")
    assert response.status_code == 200
    assert response.text == "ok"
