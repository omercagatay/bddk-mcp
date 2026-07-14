"""MCP server exposing BDDK decision search, document retrieval, and data tools."""

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import fields

import anyio
import asyncpg
import httpx
from mcp.server.fastmcp import FastMCP

from bddk_mcp import __version__
from bddk_mcp.core.config import (
    ADMIN_TOOLS,
    AUTO_SYNC,
    DATABASE_URL,
    HTTP_CONNECT_TIMEOUT,
    HTTP_POOL_TIMEOUT,
    PG_POOL_MAX,
    PG_POOL_MIN,
    REQUEST_TIMEOUT,
    require_database_url,
)
from bddk_mcp.core.deps import Dependencies
from bddk_mcp.core.logging_config import configure_logging
from bddk_mcp.db_lifecycle import assert_database_ready
from bddk_mcp.ingest.client import BddkApiClient
from bddk_mcp.store.doc_store import DocumentStore
from bddk_mcp.store.vector_store import VectorStore
from bddk_mcp.tools.registry import ToolProfile, assert_tool_profile, register_tool_profile

configure_logging()
logger = logging.getLogger(__name__)

MCP_INSTRUCTIONS = """\
Search and retrieve BDDK (Turkish Banking Regulation) decisions, regulations, and statistical data.

GROUNDING RULES — follow these strictly:
1. ONLY use information returned by tool calls. Never supplement with your own knowledge about BDDK decisions.
2. If a search returns no results, say so explicitly. Do NOT guess or invent decisions.
3. Always include document_id, decision_date, and decision_number in your response when available.
4. If document content is paginated, do NOT speculate about content on pages you have not retrieved.
5. Never fabricate karar numarası (decision numbers), tarih (dates), or legal conclusions.
6. When quoting from a document, quote only text that appears verbatim in the tool output.
7. If relevance scores are below 50%, flag this to the user and recommend refining the query.
8. Distinguish clearly between: (a) information from BDDK tools, and (b) your general knowledge.

RESPONSE STYLE AND TOOL-USE DISCIPLINE:
9. Treat tool discovery, tool schemas, function schemas, Request/Response transcripts, and intermediate tool traces as hidden implementation details. Do not paste them unless the user explicitly asks for raw tool output or debug logs.
10. Do not narrate internal reasoning, private planning, or step-by-step tool orchestration. Avoid phrases like "the user wants", "let me load tools", "now I will", or standalone "done" status lines in the final answer.
11. Answer in the user's language. If the user writes in Turkish, answer in Turkish unless they request another language.
12. Cite each regulatory claim with available document_id and section/page reference, such as "943 Ilke 5", "mevzuat_22599 Madde 9", or "page 3". Prefer exact sections over whole-document summaries for audit or compliance questions.
13. If a tool result contains a quality warning or formula/image extraction warning, surface that caveat and recommend verifying critical figures, formulas, and images against the source PDF.
14. Search results and snippets are leads, not final authority. Retrieve exact sections or pages before making detailed legal or audit conclusions.
"""

# Tool functions close over this stable object.  Runtime resources are copied
# into it when the FastMCP lifespan starts, which lets import-based launchers
# discover the complete tool surface before a database connection is opened.
_runtime_deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
_runtime_lock: asyncio.Lock | None = None
_runtime_loop: asyncio.AbstractEventLoop | None = None
_runtime_lock_users = 0
_runtime_leases = 0

_TRANSPORTS = frozenset({"stdio", "streamable-http"})


def _runtime_host() -> str:
    """Return the configured bind host, defaulting to loopback."""
    return os.environ.get("MCP_HOST", "127.0.0.1").strip() or "127.0.0.1"


def _runtime_transport() -> str:
    """Return a validated MCP transport name."""
    transport = os.environ.get("MCP_TRANSPORT", "stdio").strip().lower()
    if transport not in _TRANSPORTS:
        allowed = ", ".join(sorted(_TRANSPORTS))
        raise RuntimeError(f"Invalid MCP_TRANSPORT {transport!r}; expected one of: {allowed}")
    return transport


async def create_deps() -> Dependencies:
    """Create serving dependencies without schema, seed, or index writes."""
    if AUTO_SYNC:
        raise RuntimeError(
            "BDDK_AUTO_SYNC is not allowed in serving mode. Use an explicit operator workflow, "
            "then start `bddk-mcp serve` with BDDK_AUTO_SYNC=false."
        )
    dsn = require_database_url()

    http: httpx.AsyncClient | None = None
    pool: asyncpg.Pool | None = None
    doc_store: DocumentStore | None = None
    client: BddkApiClient | None = None
    try:
        http = httpx.AsyncClient(
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            },
            timeout=httpx.Timeout(
                REQUEST_TIMEOUT,
                connect=HTTP_CONNECT_TIMEOUT,
                pool=HTTP_POOL_TIMEOUT,
            ),
            follow_redirects=True,
        )

        pool = await asyncpg.create_pool(
            dsn,
            min_size=PG_POOL_MIN,
            max_size=PG_POOL_MAX,
            command_timeout=30,
            timeout=10,
        )
        logger.info("PostgreSQL pool created")

        await assert_database_ready(pool=pool)

        doc_store = DocumentStore(pool)
        client = BddkApiClient(
            pool=pool,
            doc_store=doc_store,
            http=http,
            allow_live_population=False,
        )
        await client.load_cache_read_only()
        vector_store = VectorStore(pool)

        return Dependencies(
            pool=pool,
            doc_store=doc_store,
            client=client,
            http=http,
            vector_store=vector_store,
            server_start_time=time.time(),
        )
    except BaseException as startup_error:
        partial = Dependencies(pool=pool, doc_store=doc_store, client=client, http=http)
        try:
            await _await_cleanup_shielded(teardown_deps(partial))
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "Dependency creation and rollback both failed",
                [startup_error, cleanup_error],
            ) from None
        raise


def configured_tool_profile() -> ToolProfile:
    """Map the compatibility admin flag to one reviewed registry profile."""
    return ToolProfile.OPERATOR if ADMIN_TOOLS else ToolProfile.PUBLIC


def register_tools(
    server: FastMCP,
    deps: Dependencies,
    *,
    profile: ToolProfile | None = None,
) -> None:
    """Register and verify one canonical MCP tool profile."""
    selected_profile = profile or configured_tool_profile()
    register_tool_profile(server, deps, selected_profile)
    assert_tool_profile(server, selected_profile)


async def teardown_deps(deps: Dependencies) -> None:
    """Attempt to close every dependency and aggregate independent failures."""
    logger.info("Graceful shutdown initiated...")
    errors: list[BaseException] = []

    def record_failure(resource: str, error: BaseException) -> None:
        error.add_note(f"while closing BDDK dependency: {resource}")
        errors.append(error)

    for task_attr in ("vector_init_task", "sync_task"):
        task = getattr(deps, task_attr)
        if task:
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError as error:
                # Cancellation is the expected result of stopping a background task.
                current_task = asyncio.current_task()
                if not task.cancelled() or (current_task is not None and current_task.cancelling()):
                    record_failure(task_attr, error)
            except BaseException as error:
                record_failure(task_attr, error)
    if deps.client:
        try:
            await deps.client.close()
        except BaseException as error:
            record_failure("client", error)
    if deps.http:
        try:
            await deps.http.aclose()
        except BaseException as error:
            record_failure("http", error)
    if deps.pool:
        try:
            await deps.pool.close()
            logger.info("PostgreSQL pool closed")
        except BaseException as error:
            record_failure("pool", error)

    if errors:
        logger.error("Graceful shutdown completed with %d error(s)", len(errors))
        raise BaseExceptionGroup("One or more BDDK dependencies failed to close", errors)

    logger.info("Graceful shutdown complete")


async def _await_cleanup_shielded(awaitable):
    """Finish cleanup before propagating cancellation to the caller.

    AnyIO cancellation is level-triggered while raw asyncio task cancellation
    is edge-triggered.  The cancel scope covers the former; the shielded task
    loop handles the latter without abandoning live resource handles.
    """
    cleanup_task = asyncio.create_task(awaitable)
    pending_cancellation: asyncio.CancelledError | None = None

    with anyio.CancelScope(shield=True):
        while True:
            try:
                result = await asyncio.shield(cleanup_task)
                break
            except asyncio.CancelledError as error:
                if cleanup_task.cancelled():
                    raise
                pending_cancellation = pending_cancellation or error
                if cleanup_task.done():
                    try:
                        result = cleanup_task.result()
                    except BaseException as cleanup_error:
                        raise BaseExceptionGroup(
                            "Cleanup failed while its caller was cancelled",
                            [pending_cancellation, cleanup_error],
                        ) from None
                    break
            except BaseException as cleanup_error:
                if pending_cancellation is not None:
                    raise BaseExceptionGroup(
                        "Cleanup failed while its caller was cancelled",
                        [pending_cancellation, cleanup_error],
                    ) from None
                raise

    if pending_cancellation is not None:
        raise pending_cancellation
    return result


def _copy_deps(source: Dependencies, target: Dependencies = _runtime_deps) -> Dependencies:
    """Copy dependency state while retaining the object captured by tools."""
    for field_info in fields(Dependencies):
        setattr(target, field_info.name, getattr(source, field_info.name))
    return target


def _empty_runtime_deps() -> None:
    """Remove references to closed resources after the final runtime lease."""
    _copy_deps(Dependencies(pool=None, doc_store=None, client=None, http=None))


def _get_runtime_lock() -> asyncio.Lock:
    global _runtime_lock, _runtime_loop
    loop = asyncio.get_running_loop()
    if _runtime_lock is None:
        _runtime_lock = asyncio.Lock()
        _runtime_loop = loop
    elif _runtime_loop is not loop:
        raise RuntimeError("The BDDK dependency runtime cannot be shared across event loops")
    return _runtime_lock


@asynccontextmanager
async def _runtime_guard() -> AsyncIterator[None]:
    """Serialize runtime state and retire an idle loop-bound lock safely."""
    global _runtime_lock, _runtime_lock_users, _runtime_loop

    lock = _get_runtime_lock()
    _runtime_lock_users += 1
    try:
        async with lock:
            yield
    finally:
        _runtime_lock_users -= 1
        if _runtime_lock_users == 0 and _runtime_leases == 0:
            _runtime_lock = None
            _runtime_loop = None


async def startup() -> Dependencies:
    """Acquire the process-wide dependency runtime.

    FastMCP invokes its low-level lifespan once per session (and once per
    request in stateless HTTP mode).  Reference counting keeps the dependencies
    alive when the HTTP process holds an outer lease while still supporting the
    import-based stdio launcher used by ``mcp run server.py``.
    """
    global _runtime_leases

    async with _runtime_guard():
        if _runtime_leases == 0:
            deps = await create_deps()
            _copy_deps(deps)
        _runtime_leases += 1
        return _runtime_deps


async def _shutdown_unshielded() -> None:
    """Release one runtime lease; callers must protect this from cancellation."""
    global _runtime_leases

    async with _runtime_guard():
        if _runtime_leases == 0:
            logger.warning("Runtime shutdown requested without an active lease")
            return

        _runtime_leases -= 1
        if _runtime_leases == 0:
            try:
                await teardown_deps(_runtime_deps)
            finally:
                _empty_runtime_deps()


async def shutdown() -> None:
    """Release a runtime lease without abandoning cleanup on cancellation."""
    await _await_cleanup_shielded(_shutdown_unshielded())


@asynccontextmanager
async def server_lifespan(_server: FastMCP) -> AsyncIterator[Dependencies]:
    """FastMCP lifecycle hook shared by SDK and direct launch paths."""
    deps = await startup()
    try:
        yield deps
    except BaseException as runtime_error:
        try:
            await shutdown()
        except BaseException as cleanup_error:
            raise BaseExceptionGroup(
                "MCP runtime and dependency cleanup both failed",
                [runtime_error, cleanup_error],
            ) from None
        raise
    else:
        await shutdown()


def create_mcp(
    deps: Dependencies,
    *,
    lifespan=None,
    profile: ToolProfile | None = None,
) -> FastMCP:
    """Construct a fully registered BDDK FastMCP server.

    Supplying dependencies separately makes the protocol surface testable with
    the official MCP client without opening a real database connection.
    """
    server = FastMCP(
        "BDDK",
        instructions=MCP_INSTRUCTIONS,
        host=_runtime_host(),
        port=int(os.environ.get("PORT", 8000)),
        stateless_http=True,
        lifespan=lifespan,
    )
    # FastMCP 1.27 does not expose the low-level version in its constructor.
    # Set the SDK Server field so initialize returns this project's version,
    # rather than the installed `mcp` library version.
    server._mcp_server.version = __version__
    register_tools(server, deps, profile=profile)
    return server


# This object is intentionally complete at import time.  The MCP CLI imports
# ``server.py:mcp`` and calls ``mcp.run()``; registering tools only in a custom
# main() startup path therefore exposed an empty server to that supported path.
mcp = create_mcp(_runtime_deps, lifespan=server_lifespan)


def main() -> None:
    """Entry point — selects transport and runs the MCP server."""
    try:
        import uvloop

        uvloop.install()
        logger.info("uvloop installed")
    except ImportError:
        pass

    _transport = _runtime_transport()
    logger.info("Transport: %s", _transport)
    logger.info("BDDK_AUTO_SYNC=%s", os.environ.get("BDDK_AUTO_SYNC", "(not set)"))
    logger.info("Database configuration present: %s", bool(DATABASE_URL))

    if _transport == "streamable-http":
        import uvicorn

        app = mcp.streamable_http_app()
        port = int(os.environ.get("PORT", 8000))

        async def _run_server():
            config = uvicorn.Config(app, host=_runtime_host(), port=port)
            server = uvicorn.Server(config)

            async with server_lifespan(mcp):
                await server.serve()

        asyncio.run(_run_server())
    else:
        # Default transport: stdio
        import anyio

        async def _run_stdio():
            # FastMCP enters server_lifespan on this event loop.
            await mcp.run_stdio_async()

        anyio.run(_run_stdio)


if __name__ == "__main__":
    main()
