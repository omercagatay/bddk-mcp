"""Wire the admin console to a real database pool."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from functools import partial

import asyncpg
from starlette.types import ASGIApp

from bddk_mcp.admin.app import create_app
from bddk_mcp.admin.config import AdminConfig
from bddk_mcp.admin.services.documents import DocumentService
from bddk_mcp.admin.services.governance import GovernanceService, resolve_governance_paths
from bddk_mcp.db_identity import assert_database_connection_identity
from bddk_mcp.http_security import JwtTokenVerifier
from bddk_mcp.store.doc_store import DocumentStore


async def build_app_from_env(
    env: Mapping[str, str] | None = None,
) -> tuple[ASGIApp, AdminConfig, Callable[[], Awaitable[None]]]:
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
    try:
        store = DocumentStore(pool)
        await store.initialize()
    except Exception:
        # The pool was already opened; a stale or missing migration must not
        # leak the connections it holds.
        await pool.close()
        raise
    seed_dir, trusted_signing_key = resolve_governance_paths(env)
    governance = GovernanceService(pool, seed_dir=seed_dir, trusted_signing_key=trusted_signing_key)
    verifier = None if config.http_security is None else JwtTokenVerifier(config.http_security)
    app = create_app(config, DocumentService(store), governance, token_verifier=verifier)

    async def shutdown() -> None:
        await pool.close()

    return app, config, shutdown
