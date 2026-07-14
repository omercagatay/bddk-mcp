"""MCP protocol regressions for the exported production server."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from bddk_mcp import __version__
from bddk_mcp.core.deps import Dependencies
from bddk_mcp.tools.registry import PUBLIC_TOOL_NAMES


@pytest.mark.asyncio
async def test_root_shim_export_lists_public_tools_with_official_client():
    """The object imported by ``mcp run server.py`` must not expose zero tools."""
    import server as root_shim

    with (
        patch("bddk_mcp.server.startup", new=AsyncMock(return_value=MagicMock())),
        patch("bddk_mcp.server.shutdown", new=AsyncMock()),
    ):
        async with create_connected_server_and_client_session(root_shim.mcp) as session:
            result = await session.list_tools()

    assert {tool.name for tool in result.tools} == set(PUBLIC_TOOL_NAMES)


@pytest.mark.asyncio
async def test_factory_supports_protocol_tool_call_without_database():
    """Injected dependencies keep MCP contract tests independent of PostgreSQL."""
    from bddk_mcp.server import create_mcp

    doc_store = MagicMock()
    doc_store.get_document_history = AsyncMock(return_value=[])
    deps = Dependencies(pool=None, doc_store=doc_store, client=MagicMock(), http=None)
    test_server = create_mcp(deps)

    async with create_connected_server_and_client_session(test_server) as session:
        result = await session.call_tool("get_document_history", {"document_id": "943"})

    assert result.isError is False
    assert result.content[0].text == "No version history found for document 943."
    doc_store.get_document_history.assert_awaited_once_with("943")
    assert test_server._mcp_server.version == __version__


def test_runtime_defaults_to_loopback_and_rejects_unknown_transport(monkeypatch):
    import bddk_mcp.server as server_module

    monkeypatch.delenv("MCP_HOST", raising=False)
    assert server_module._runtime_host() == "127.0.0.1"

    monkeypatch.setenv("MCP_TRANSPORT", "legacy-sse")
    with pytest.raises(RuntimeError, match="Invalid MCP_TRANSPORT"):
        server_module._runtime_transport()


@pytest.mark.asyncio
async def test_nested_lifespans_share_one_dependency_runtime():
    """HTTP's outer lease prevents per-request dependency reinitialization."""
    import bddk_mcp.server as server_module

    created = Dependencies(pool=MagicMock(), doc_store=MagicMock(), client=MagicMock(), http=MagicMock())
    server_module._runtime_leases = 0
    server_module._runtime_lock = None
    server_module._empty_runtime_deps()

    with (
        patch.object(server_module, "create_deps", new=AsyncMock(return_value=created)) as create_deps,
        patch.object(server_module, "teardown_deps", new=AsyncMock()) as teardown,
    ):
        async with server_module.server_lifespan(server_module.mcp) as outer_deps:
            async with server_module.server_lifespan(server_module.mcp) as inner_deps:
                assert inner_deps is outer_deps
                assert create_deps.await_count == 1
            teardown.assert_not_awaited()

        teardown.assert_awaited_once_with(outer_deps)

    assert server_module._runtime_leases == 0
    assert server_module._runtime_deps.pool is None
    assert server_module._runtime_lock is None
    assert server_module._runtime_loop is None


@pytest.mark.asyncio
async def test_create_deps_closes_http_when_pool_creation_fails():
    """Partially-created resources must not leak when startup fails."""
    import bddk_mcp.server as server_module

    http = MagicMock()
    http.aclose = AsyncMock()
    with (
        patch.object(server_module, "require_database_url", return_value="postgresql://test"),
        patch.object(server_module.httpx, "AsyncClient", return_value=http),
        patch.object(
            server_module.asyncpg,
            "create_pool",
            new=AsyncMock(side_effect=RuntimeError("database unavailable")),
        ),
        pytest.raises(RuntimeError, match="database unavailable"),
    ):
        await server_module.create_deps()

    http.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_create_deps_closes_pool_and_http_when_readiness_fails():
    """A readiness failure after pool creation must unwind every resource."""
    import bddk_mcp.server as server_module

    http = MagicMock()
    http.aclose = AsyncMock()
    pool = MagicMock()
    pool.close = AsyncMock()
    with (
        patch.object(server_module, "require_database_url", return_value="postgresql://test"),
        patch.object(server_module.httpx, "AsyncClient", return_value=http),
        patch.object(server_module.asyncpg, "create_pool", new=AsyncMock(return_value=pool)),
        patch.object(
            server_module,
            "assert_database_ready",
            new=AsyncMock(side_effect=RuntimeError("schema unavailable")),
        ),
        pytest.raises(RuntimeError, match="schema unavailable"),
    ):
        await server_module.create_deps()

    pool.close.assert_awaited_once()
    http.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_teardown_attempts_every_close_and_aggregates_failures():
    """One broken closer must not prevent the remaining resources from closing."""
    import bddk_mcp.server as server_module

    client = MagicMock()
    client.close = AsyncMock(side_effect=ValueError("client close"))
    http = MagicMock()
    http.aclose = AsyncMock(side_effect=RuntimeError("http close"))
    pool = MagicMock()
    pool.close = AsyncMock(side_effect=OSError("pool close"))
    deps = Dependencies(pool=pool, doc_store=MagicMock(), client=client, http=http)

    with pytest.raises(ExceptionGroup) as exc_info:
        await server_module.teardown_deps(deps)

    assert len(exc_info.value.exceptions) == 3
    client.close.assert_awaited_once()
    http.aclose.assert_awaited_once()
    pool.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_readiness_rolls_back_unpublished_dependencies():
    """Cancellation during read-only readiness must close partial resources."""
    import bddk_mcp.server as server_module

    entered = asyncio.Event()

    async def blocking_readiness(*, pool):
        entered.set()
        await asyncio.Event().wait()

    client = MagicMock()
    client.close = AsyncMock()
    http = MagicMock()
    http.aclose = AsyncMock()
    pool = MagicMock()
    pool.close = AsyncMock()
    server_module._runtime_leases = 0
    server_module._runtime_lock = None
    server_module._runtime_loop = None
    server_module._runtime_lock_users = 0
    server_module._empty_runtime_deps()

    with (
        patch.object(server_module, "require_database_url", return_value="postgresql://test"),
        patch.object(server_module.httpx, "AsyncClient", return_value=http),
        patch.object(server_module.asyncpg, "create_pool", new=AsyncMock(return_value=pool)),
        patch.object(server_module, "assert_database_ready", side_effect=blocking_readiness),
    ):
        startup_task = asyncio.create_task(server_module.create_deps())
        await entered.wait()
        startup_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await startup_task

    client.close.assert_not_awaited()
    http.aclose.assert_awaited_once()
    pool.close.assert_awaited_once()
    assert server_module._runtime_leases == 0
    assert server_module._runtime_deps.pool is None
    assert server_module._runtime_lock is None
    assert server_module._runtime_loop is None


@pytest.mark.asyncio
async def test_create_deps_uses_only_read_only_runtime_initialization():
    """Serving must not invoke schema, seed, sync, or vector backfill APIs."""
    import bddk_mcp.server as server_module

    http = MagicMock()
    http.aclose = AsyncMock()
    pool = MagicMock()
    pool.close = AsyncMock()
    doc_store = MagicMock()
    client = MagicMock()
    client.load_cache_read_only = AsyncMock(return_value=1)
    vector_store = MagicMock()
    readiness = AsyncMock()

    with (
        patch.object(server_module, "AUTO_SYNC", False),
        patch.object(server_module, "require_database_url", return_value="postgresql://test"),
        patch.object(server_module.httpx, "AsyncClient", return_value=http),
        patch.object(server_module.asyncpg, "create_pool", new=AsyncMock(return_value=pool)),
        patch.object(server_module, "assert_database_ready", new=readiness),
        patch.object(server_module, "DocumentStore", return_value=doc_store),
        patch.object(server_module, "BddkApiClient", return_value=client) as client_type,
        patch.object(server_module, "VectorStore", return_value=vector_store),
    ):
        deps = await server_module.create_deps()

    readiness.assert_awaited_once_with(pool=pool)
    client.load_cache_read_only.assert_awaited_once_with()
    client_type.assert_called_once_with(
        pool=pool,
        doc_store=doc_store,
        http=http,
        allow_live_population=False,
    )
    assert deps.vector_store is vector_store
    assert not hasattr(doc_store, "initialize") or doc_store.initialize.call_count == 0


@pytest.mark.asyncio
async def test_create_deps_rejects_legacy_auto_sync_before_opening_resources():
    import bddk_mcp.server as server_module

    with (
        patch.object(server_module, "AUTO_SYNC", True),
        patch.object(server_module.httpx, "AsyncClient") as http_type,
        pytest.raises(RuntimeError, match="not allowed in serving mode"),
    ):
        await server_module.create_deps()

    http_type.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_shutdown_finishes_closing_before_propagating():
    """Transport cancellation must not interrupt dependency cleanup."""
    import bddk_mcp.server as server_module

    close_started = asyncio.Event()
    allow_close = asyncio.Event()

    async def blocking_pool_close():
        close_started.set()
        await allow_close.wait()

    client = MagicMock()
    client.close = AsyncMock()
    http = MagicMock()
    http.aclose = AsyncMock()
    pool = MagicMock()
    pool.close = AsyncMock(side_effect=blocking_pool_close)
    deps = Dependencies(pool=pool, doc_store=MagicMock(), client=client, http=http)
    server_module._runtime_leases = 1
    server_module._runtime_lock = None
    server_module._runtime_loop = None
    server_module._runtime_lock_users = 0
    server_module._copy_deps(deps)

    shutdown_task = asyncio.create_task(server_module.shutdown())
    await close_started.wait()
    shutdown_task.cancel()
    await asyncio.sleep(0)
    assert not shutdown_task.done()

    allow_close.set()
    with pytest.raises(asyncio.CancelledError):
        await shutdown_task

    client.close.assert_awaited_once()
    http.aclose.assert_awaited_once()
    pool.close.assert_awaited_once()
    assert server_module._runtime_leases == 0
    assert server_module._runtime_deps.pool is None
    assert server_module._runtime_lock is None
    assert server_module._runtime_loop is None
