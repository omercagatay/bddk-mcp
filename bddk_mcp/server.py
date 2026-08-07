"""MCP server exposing BDDK decision search, document retrieval, and data tools."""

import asyncio
import logging
import os
import time

import asyncpg
import httpx
from mcp.server.fastmcp import FastMCP

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
from bddk_mcp.ingest.client import BddkApiClient
from bddk_mcp.store.doc_store import DocumentStore
from bddk_mcp.tools import admin, analytics, bulletin, documents, graph, search, sections, sync

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

mcp = FastMCP(
    "BDDK",
    instructions=MCP_INSTRUCTIONS,
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 8000)),
    stateless_http=True,
)


async def create_deps() -> Dependencies:
    """Create all dependencies eagerly. Fails fast if DB is unreachable."""
    dsn = require_database_url()

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
    logger.info("PostgreSQL pool created: %s", dsn.split("@")[-1])

    doc_store = DocumentStore(pool)
    await doc_store.initialize()

    client = BddkApiClient(pool=pool, doc_store=doc_store, http=http)
    await client.initialize()

    return Dependencies(
        pool=pool,
        doc_store=doc_store,
        client=client,
        http=http,
        vector_store=None,
        server_start_time=time.time(),
    )


async def init_vector_store(deps: Dependencies) -> None:
    """Background task: load embedding model and initialize VectorStore."""
    try:
        from bddk_mcp.store.vector_store import VectorStore

        vs = VectorStore(deps.pool)
        await vs.initialize()
        deps.vector_store = vs
        logger.info("VectorStore initialized (background)")
    except Exception as e:
        logger.error("VectorStore init failed: %s", e)


def register_tools(deps: Dependencies) -> None:
    """Register all tool modules on the shared FastMCP instance."""
    search.register(mcp, deps)
    documents.register(mcp, deps)
    sections.register(mcp, deps)
    graph.register(mcp, deps)
    bulletin.register(mcp, deps)
    analytics.register(mcp, deps)
    if ADMIN_TOOLS:
        sync.register(mcp, deps)
        admin.register(mcp, deps)


async def import_seed_data(deps: Dependencies) -> None:
    """Seed the DB from bundled seed_data/ when present (non-fatal on failure)."""
    try:
        from bddk_mcp.ingest.seed import SEED_DIR, import_seed

        if SEED_DIR.exists():
            result = await import_seed(pool=deps.pool)
            if not result["skipped"]:
                logger.info(
                    "Seed: %d cache, %d docs, %d chunks",
                    result["decision_cache"],
                    result["documents"],
                    result["chunks"],
                )
            else:
                logger.info("DB populated — seed skipped")
    except Exception as e:
        logger.warning("Seed failed (non-fatal): %s", e)


def start_background_tasks(deps: Dependencies) -> None:
    """Kick off vector-store init and (optionally) auto-sync on the running loop."""
    deps.vector_init_task = asyncio.create_task(init_vector_store(deps))

    if AUTO_SYNC:

        async def _sync_after_vector_init():
            if deps.vector_init_task:
                await deps.vector_init_task
            await sync.startup_sync(deps)

        deps.sync_task = asyncio.create_task(_sync_after_vector_init())
        logger.info("[STARTUP] background sync scheduled")


async def startup() -> Dependencies:
    """Shared startup sequence for both transports."""
    deps = await create_deps()
    register_tools(deps)
    await import_seed_data(deps)
    start_background_tasks(deps)
    return deps


async def teardown_deps(deps: Dependencies) -> None:
    """Shut down in correct order: tasks first, then connections."""
    logger.info("Graceful shutdown initiated...")
    for task_attr in ("vector_init_task", "sync_task"):
        task = getattr(deps, task_attr)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    if deps.client:
        await deps.client.close()
    if deps.http:
        await deps.http.aclose()
    if deps.pool:
        await deps.pool.close()
        logger.info("PostgreSQL pool closed")
    logger.info("Graceful shutdown complete")


def main() -> None:
    """Entry point — selects transport and runs the MCP server."""
    try:
        import uvloop

        uvloop.install()
        logger.info("uvloop installed")
    except ImportError:
        pass

    _transport = os.environ.get("MCP_TRANSPORT", "stdio").strip().lower()
    # Guard against misconfigured env vars like VALUE="MCP_TRANSPORT=streamable-http"
    if "=" in _transport:
        _transport = _transport.split("=", 1)[-1].strip()
    logger.info("Transport: %s", _transport)
    logger.info("BDDK_AUTO_SYNC=%s", os.environ.get("BDDK_AUTO_SYNC", "(not set)"))
    logger.info("DATABASE_URL=%s", DATABASE_URL.split("@")[-1])

    if _transport == "streamable-http":
        import uvicorn

        app = mcp.streamable_http_app()
        port = int(os.environ.get("PORT", 8000))

        async def _run_server():
            config = uvicorn.Config(app, host="0.0.0.0", port=port)
            server = uvicorn.Server(config)

            deps = await startup()
            try:
                await server.serve()
            finally:
                await teardown_deps(deps)

        asyncio.run(_run_server())
    else:
        # Default transport: stdio
        import anyio

        async def _run_stdio():
            # startup() schedules background tasks on this same event loop so they actually run
            deps = await startup()
            try:
                await mcp.run_stdio_async()
            finally:
                await teardown_deps(deps)

        anyio.run(_run_stdio)


if __name__ == "__main__":
    main()
