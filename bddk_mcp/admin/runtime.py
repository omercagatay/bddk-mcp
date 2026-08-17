"""Wire the admin console to a real database pool."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from functools import partial

import asyncpg
from starlette.applications import Starlette

from bddk_mcp.admin.app import create_app
from bddk_mcp.admin.config import AdminConfig
from bddk_mcp.admin.services.documents import DocumentService
from bddk_mcp.db_identity import assert_database_connection_identity
from bddk_mcp.store.doc_store import DocumentStore


async def build_app_from_env(env: Mapping[str, str] | None = None) -> tuple[Starlette, Callable[[], Awaitable[None]]]:
    """Resolve configuration, open a least-privilege pool, and build the app."""

    config = AdminConfig.from_env(env)
    pool = await asyncpg.create_pool(
        config.database_url,
        min_size=1,
        max_size=4,
        command_timeout=30,
        timeout=10,
        init=partial(assert_database_connection_identity, profile="public"),
    )
    store = DocumentStore(pool)
    await store.initialize()
    app = create_app(config, DocumentService(store))

    async def shutdown() -> None:
        await pool.close()

    return app, shutdown
