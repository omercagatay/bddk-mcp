from __future__ import annotations

import asyncio

import pytest

from bddk_mcp.admin.config import AdminConfigError
from bddk_mcp.admin.runtime import build_app_from_env


def test_build_app_requires_configuration() -> None:
    with pytest.raises(AdminConfigError):
        asyncio.run(build_app_from_env({}))


class _FakePool:
    """Stands in for the asyncpg pool without touching a real database."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_build_app_closes_pool_when_store_initialize_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale or missing migration must not leak the already-open pool."""

    fake_pool = _FakePool()

    async def fake_create_pool(*_args, **_kwargs):
        return fake_pool

    async def failing_initialize(self) -> None:
        raise RuntimeError('relation "documents" does not exist')

    monkeypatch.setattr("bddk_mcp.admin.runtime.asyncpg.create_pool", fake_create_pool)
    monkeypatch.setattr("bddk_mcp.admin.runtime.DocumentStore.initialize", failing_initialize)
    monkeypatch.setenv("BDDK_ALLOW_INSECURE_DATABASE", "true")

    env = {"BDDK_DATABASE_URL": "postgresql://x", "BDDK_ADMIN_HOST": "127.0.0.1"}

    with pytest.raises(RuntimeError, match="does not exist"):
        asyncio.run(build_app_from_env(env))

    assert fake_pool.closed is True


def test_runtime_uses_public_profile_and_persistent_sidecar(tmp_path, monkeypatch):
    from starlette.testclient import TestClient

    from tests.test_admin_drafts import ReadOnlyStore, form_fields

    canonical = ReadOnlyStore().doc

    class Pool(_FakePool):
        async def fetchrow(self, query, *args):
            assert query.startswith("SELECT")
            assert args == ("doc",)
            row = canonical.model_dump()
            row["pdf_blob"] = row.pop("pdf_bytes")
            return row

    opened = []

    async def create_pool(*args, **kwargs):
        assert kwargs["init"].keywords == {"profile": "public"}
        pool = Pool()
        opened.append(pool)
        return pool

    async def ready(self):
        pass  # PostgreSQL schema/identity admission has its own live LOGIN lane.

    monkeypatch.setattr("bddk_mcp.admin.runtime.asyncpg.create_pool", create_pool)
    monkeypatch.setattr("bddk_mcp.admin.runtime.DocumentStore.initialize", ready)
    monkeypatch.setenv("BDDK_ALLOW_INSECURE_DATABASE", "true")
    env = {"BDDK_DATABASE_URL": "postgresql://x", "BDDK_ADMIN_DRAFT_DB": str(tmp_path / "draft.sqlite")}
    for revision in (0, 1):
        app, _, shutdown = asyncio.run(build_app_from_env(env))
        try:
            client = TestClient(app, base_url="http://127.0.0.1")
            fields = form_fields(client.get("/documents/doc/edit"))
            assert fields["revision"] == str(revision)
            fields["title"] = "Runtime draft"
            response = client.post(
                "/documents/doc/edit", data=fields, headers={"origin": "http://127.0.0.1"}, follow_redirects=False
            )
            assert response.status_code == 303
            assert "Runtime draft" in client.get("/documents/doc").text
        finally:
            asyncio.run(shutdown())
    assert all(pool.closed for pool in opened)
    assert canonical.title == "Original"
