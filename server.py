"""MCP server exposing BDDK decision search, document retrieval, and data tools."""

import asyncio
import logging
import os
import time

import asyncpg
import httpx
from mcp.server.fastmcp import FastMCP

from client import BddkApiClient
from config import AUTO_SYNC, DATABASE_URL, PG_POOL_MAX, PG_POOL_MIN, REQUEST_TIMEOUT
from deps import Dependencies
from doc_store import DocumentStore
from logging_config import configure_logging
from tools import admin, analytics, bulletin, documents, search, sync

configure_logging()
logger = logging.getLogger(__name__)

mcp = FastMCP(
    "BDDK",
    instructions="""\
Search and retrieve BDDK (Turkish Banking Regulation) decisions, regulations, and statistical data.

GROUNDING RULES — follow these strictly:
1. ONLY use information returned by tool calls. Never supplement with your own knowledge about BDDK decisions.
2. If a search returns no results, say so explicitly. Do NOT guess or invent decisions.
3. Always include document_id, decision_date, and decision_number in your response when available.
4. If document content is paginated, do NOT speculate about content on pages you have not retrieved.
5. Never fabricate karar numarası (decision numbers), tarih (dates), or legal conclusions.
6. When quoting from a document, quote only text that appears verbatim in the tool output.
7. If relevance scores are below 50%, flag this to the user and recommend refining the query.
8. Distinguish clearly between: (a) information from BDDK tools, and (b) your general knowledge.\
""",
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 8000)),
    stateless_http=True,
)


async def create_deps() -> Dependencies:
    """Create all dependencies eagerly. Fails fast if DB is unreachable."""
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
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        follow_redirects=True,
    )

    pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=PG_POOL_MIN,
        max_size=PG_POOL_MAX,
        command_timeout=30,
        timeout=10,
    )
    logger.info("PostgreSQL pool created: %s", DATABASE_URL.split("@")[-1])

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
        from vector_store import VectorStore

        vs = VectorStore(deps.pool)
        await vs.initialize()
        deps.vector_store = vs
        logger.info("VectorStore initialized (background)")
    except Exception as e:
        logger.error("VectorStore init failed: %s", e)


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


if __name__ == "__main__":
    try:
        import uvloop

        uvloop.install()
        logger.info("uvloop installed")
    except ImportError:
        pass

    _transport = os.environ.get("MCP_TRANSPORT", "stdio")
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

            deps = await create_deps()

            # Register all tool modules
            search.register(mcp, deps)
            documents.register(mcp, deps)
            bulletin.register(mcp, deps)
            analytics.register(mcp, deps)
            sync.register(mcp, deps)
            admin.register(mcp, deps)

            # Seed DB
            try:
                from seed import SEED_DIR, import_seed

                if SEED_DIR.exists():
                    result = await import_seed()
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

            # Background: vector store init
            deps.vector_init_task = asyncio.create_task(init_vector_store(deps))

            # Background: auto-sync
            if AUTO_SYNC:

                async def _sync_after_vector_init():
                    if deps.vector_init_task:
                        await deps.vector_init_task
                    await sync.startup_sync(deps)

                deps.sync_task = asyncio.create_task(_sync_after_vector_init())
                logger.info("[STARTUP] background sync scheduled")

            await server.serve()
            await teardown_deps(deps)

        asyncio.run(_run_server())
    else:

        async def _run_stdio():
            deps = await create_deps()

            search.register(mcp, deps)
            documents.register(mcp, deps)
            bulletin.register(mcp, deps)
            analytics.register(mcp, deps)
            sync.register(mcp, deps)
            admin.register(mcp, deps)

            deps.vector_init_task = asyncio.create_task(init_vector_store(deps))

            if AUTO_SYNC:

                async def _sync_after_vector_init():
                    if deps.vector_init_task:
                        await deps.vector_init_task
                    await sync.startup_sync(deps)

                deps.sync_task = asyncio.create_task(_sync_after_vector_init())

        asyncio.get_event_loop().run_until_complete(_run_stdio())
        mcp.run(transport=_transport)
