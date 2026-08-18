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

    env = {"BDDK_DATABASE_URL": "postgresql://x", "BDDK_ADMIN_HOST": "127.0.0.1"}

    with pytest.raises(RuntimeError, match="does not exist"):
        asyncio.run(build_app_from_env(env))

    assert fake_pool.closed is True
