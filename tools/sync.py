"""Sync tools for BDDK MCP Server.

Provides document sync, pgvector migration, and circuit breaker helpers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from config import PREFER_NOUGAT
from exceptions import BddkError

if TYPE_CHECKING:
    from deps import Dependencies

logger = logging.getLogger(__name__)

# -- Circuit breaker constants ------------------------------------------------

CIRCUIT_BREAKER_THRESHOLD = 10
STARTUP_SYNC_TIMEOUT = 300  # 5 minutes
MIGRATION_TIMEOUT = 600  # 10 minutes


# -- Circuit breaker helpers --------------------------------------------------


def _record_sync_failure(deps: Dependencies, error: str) -> None:
    """Record a sync failure and open circuit if threshold reached."""
    deps.sync_consecutive_failures += 1
    deps.last_sync_error = error
    if deps.sync_consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
        deps.sync_circuit_open = True


def _record_sync_success(deps: Dependencies) -> None:
    """Record a successful sync and reset circuit."""
    deps.sync_consecutive_failures = 0
    deps.sync_circuit_open = False
    deps.last_sync_time = time.time()
    deps.last_sync_error = None


# -- Migration helper ---------------------------------------------------------


async def _migrate_to_pgvector(deps: Dependencies) -> str:
    """Migrate documents from document store to pgvector if needed.

    Uses a batch existence check instead of per-document has_document() calls.
    Aborts after MIGRATION_TIMEOUT seconds.

    Returns a status string describing what happened.
    """
    vs = deps.vector_store
    store = deps.doc_store
    pool = deps.pool

    if vs is None:
        return "pgvector unavailable (not initialized)"
    if store is None:
        return "doc_store unavailable"

    try:
        vs_stats = await vs.stats()
        sqlite_stats = await store.stats()

        if vs_stats["total_documents"] >= sqlite_stats.total_documents * 0.9:
            logger.info(
                "pgvector has %d/%d documents, skipping migration",
                vs_stats["total_documents"],
                sqlite_stats.total_documents,
            )
            return f"pgvector up-to-date: {vs_stats['total_documents']}/{sqlite_stats.total_documents} documents"

        logger.info(
            "pgvector incomplete (%d/%d) — migrating...",
            vs_stats["total_documents"],
            sqlite_stats.total_documents,
        )

        start = time.time()
        docs = await store.list_documents(limit=2000)
        migrated = 0
        total_chunks = 0

        # Batch existence check: one query instead of N has_document() calls
        doc_ids = [meta["document_id"] for meta in docs]
        existing_ids: set[str] = set()
        batch_succeeded = False
        if pool is not None and doc_ids:
            try:
                rows = await pool.fetch(
                    "SELECT DISTINCT doc_id FROM document_chunks WHERE doc_id = ANY($1)",
                    doc_ids,
                )
                existing_ids = {r["doc_id"] for r in rows}
                batch_succeeded = True
            except Exception as e:
                logger.warning("Batch existence check failed, falling back to per-doc: %s", e)

        deadline = start + MIGRATION_TIMEOUT

        for i, meta in enumerate(docs):
            if time.time() > deadline:
                logger.warning("pgvector migration timed out after %ds", MIGRATION_TIMEOUT)
                break

            doc_id = meta["document_id"]

            # Use batch result if available, otherwise fall back to per-doc check
            if batch_succeeded:
                if doc_id in existing_ids:
                    continue
            else:
                if await vs.has_document(doc_id):
                    continue

            doc = await store.get_document(doc_id)
            if not doc or not doc.markdown_content:
                continue

            chunks = await vs.add_document(
                doc_id=doc.document_id,
                title=doc.title,
                content=doc.markdown_content,
                category=doc.category,
                decision_date=doc.decision_date,
                decision_number=doc.decision_number,
                source_url=doc.source_url,
            )
            total_chunks += chunks
            migrated += 1

            if (i + 1) % 100 == 0:
                logger.info("pgvector migration: %d/%d docs", i + 1, len(docs))

        elapsed = time.time() - start
        logger.info(
            "pgvector migration complete: %d docs, %d chunks, %.1fs",
            migrated,
            total_chunks,
            elapsed,
        )
        return f"Migrated {migrated} documents, {total_chunks} chunks in {elapsed:.1f}s"

    except (BddkError, RuntimeError, OSError) as e:
        logger.error("pgvector migration failed: %s", e)
        return f"Migration failed: {e}"


# -- Startup sync (module-level, called from server.py) -----------------------


async def startup_sync(deps: Dependencies) -> None:
    """Auto-sync documents on startup: download missing + embed to pgvector.

    Uses existing PostgreSQL cache — does NOT scrape BDDK for the decision list.
    Only downloads document content that is missing from the document store.
    Wrapped in asyncio.timeout(STARTUP_SYNC_TIMEOUT) to prevent hanging.
    """
    if deps.sync_circuit_open:
        logger.warning(
            "Startup sync skipped: circuit breaker open (%d consecutive failures, last: %s)",
            deps.sync_consecutive_failures,
            deps.last_sync_error,
        )
        return

    logger.info("Startup sync started...")
    try:
        async with asyncio.timeout(STARTUP_SYNC_TIMEOUT):
            from doc_sync import DocumentSyncer

            store = deps.doc_store
            client = deps.client

            logger.info("Using existing cache: %d documents", client.cache_size())
            if not client.cache_size():
                logger.warning("Cache is empty — skipping startup sync (run refresh_bddk_cache first)")
                _record_sync_failure(deps, "Cache is empty")
                return

            st = await store.stats()
            cache_size = client.cache_size()

            # Phase 1: Download missing documents
            if st.total_documents < cache_size * 0.9:
                logger.info(
                    "Document store incomplete (%d/%d) — downloading...",
                    st.total_documents,
                    cache_size,
                )
                items = [d.model_dump() for d in client.get_cache_items()]
                async with DocumentSyncer(store, prefer_nougat=PREFER_NOUGAT, http=deps.http) as syncer:
                    report = await syncer.sync_all(items, concurrency=10, force=False)
                logger.info(
                    "Document sync: %d downloaded, %d failed, %.1fs",
                    report.downloaded,
                    report.failed,
                    report.elapsed_seconds,
                )
            else:
                logger.info("Document store has %d/%d documents, OK", st.total_documents, cache_size)

            # Phase 2: Migrate to pgvector
            await _migrate_to_pgvector(deps)

            _record_sync_success(deps)

    except TimeoutError:
        msg = f"Startup sync timed out after {STARTUP_SYNC_TIMEOUT}s"
        logger.error(msg)
        _record_sync_failure(deps, msg)
    except (BddkError, RuntimeError, OSError) as e:
        msg = str(e)
        logger.error("Startup sync failed: %s", msg)
        _record_sync_failure(deps, msg)


# -- Tool registration --------------------------------------------------------


def register(mcp, deps: Dependencies) -> None:
    """Register sync tools on the given MCP instance."""

    @mcp.tool()
    async def refresh_bddk_cache() -> str:
        """
        Force re-scrape BDDK website and update the PostgreSQL decision cache.

        Use this when you need the latest regulations/decisions from BDDK.
        Normally the server serves from PostgreSQL without hitting BDDK.
        This tool explicitly fetches fresh data from bddk.org.tr.
        """
        count = await deps.client.refresh_cache()
        return f"BDDK cache refreshed: {count} decisions/regulations scraped and saved to PostgreSQL."

    @mcp.tool()
    async def sync_bddk_documents(
        force: bool = False,
        document_id: str | None = None,
        concurrency: int = 5,
    ) -> str:
        """
        Sync BDDK documents to local storage.

        Downloads documents from BDDK and mevzuat.gov.tr, extracts content to
        Markdown, and stores in PostgreSQL database for fast offline access.

        Args:
            force: Re-download all documents even if already cached
            document_id: Sync a single document by ID (e.g. "1291" or "mevzuat_42628")
            concurrency: Number of parallel downloads (default 5)
        """
        from doc_sync import DocumentSyncer

        store = deps.doc_store
        client = deps.client
        await client.ensure_cache()

        single_report = None
        sync_report = None

        async with DocumentSyncer(store, prefer_nougat=PREFER_NOUGAT, http=deps.http) as syncer:
            if document_id:
                source_url = ""
                title = document_id
                category = ""
                found = client.find_by_id(document_id)
                if found:
                    source_url = found.source_url
                    title = found.title
                    category = found.category

                result = await syncer.sync_document(
                    doc_id=document_id,
                    title=title,
                    category=category,
                    source_url=source_url,
                    force=force,
                )
                status = "OK" if result.success else "FAIL"
                single_report = f"[{status}] {result.document_id}: {result.method or result.error}"
            else:
                items = [d.model_dump() for d in client.get_cache_items()]
                report = await syncer.sync_all(items, concurrency=concurrency, force=force)
                sync_report = (
                    f"**Sync Report**\n"
                    f"  Total: {report.total}\n"
                    f"  Downloaded: {report.downloaded}\n"
                    f"  Skipped: {report.skipped}\n"
                    f"  Failed: {report.failed}\n"
                    f"  Time: {report.elapsed_seconds}s"
                )

        # Migrate documents to pgvector for semantic search
        embed_report = ""
        try:
            migration_status = await _migrate_to_pgvector(deps)
            if deps.vector_store is not None:
                vs_stats = await deps.vector_store.stats()
                embed_report = (
                    f"\n\n**Embedding Report**\n"
                    f"  {migration_status}\n"
                    f"  Documents: {vs_stats['total_documents']}\n"
                    f"  Chunks: {vs_stats['total_chunks']}"
                )
            else:
                embed_report = f"\n\n**Embedding:** {migration_status}"
        except Exception as e:
            embed_report = f"\n\n**Embedding:** failed ({e})"

        if single_report:
            return single_report + embed_report
        return sync_report + embed_report

    @mcp.tool()
    async def trigger_startup_sync() -> str:
        """
        Manually trigger document sync if auto-sync is still running or was skipped.
        Returns current sync status.
        """
        if deps.sync_task and not deps.sync_task.done():
            return "Sync is already running in background."

        # Reset circuit breaker to allow re-triggering even after failures
        deps.sync_circuit_open = False
        deps.sync_consecutive_failures = 0

        store = deps.doc_store
        st = await store.stats()

        # Run pgvector migration if documents exist but embeddings are missing
        embed_report = ""
        try:
            migration_status = await _migrate_to_pgvector(deps)
            if deps.vector_store is not None:
                vs_stats = await deps.vector_store.stats()
                embed_report = (
                    f"\n  {migration_status}"
                    f"\n  Vector documents: {vs_stats['total_documents']}"
                    f"\n  Vector chunks: {vs_stats['total_chunks']}"
                )
            else:
                embed_report = f"\n  {migration_status}"
        except Exception as e:
            embed_report = f"\n  Embedding migration failed: {e}"

        return f"Store has {st.total_documents} documents.{embed_report}"

    @mcp.tool()
    async def document_health(retryable_only: bool = False) -> str:
        """
        Check document completeness and show any sync failures.

        Reports:
        - Total documents vs decision cache size
        - Documents missing content
        - Persistent sync failures with error categories
        - Vector store coverage

        Args:
            retryable_only: Only show failures that can be retried (e.g. timeouts)
        """
        store = deps.doc_store
        client = deps.client

        # Document completeness
        st = await store.stats()
        cache_size = client.cache_size()

        lines = ["**Document Health Report**\n"]
        lines.append(f"Decision cache: {cache_size}")
        lines.append(f"Documents with content: {st.total_documents}")

        if cache_size > 0:
            pct = st.total_documents / cache_size * 100
            lines.append(f"Coverage: {pct:.1f}%")

        # Sync failures
        failures = await store.get_sync_failures(retryable_only=retryable_only)
        if failures:
            lines.append(f"\n**Sync Failures: {len(failures)}**")

            # Group by category
            by_cat: dict[str, list[dict]] = {}
            for f in failures:
                cat = f["error_category"]
                by_cat.setdefault(cat, []).append(f)

            for cat, items in sorted(by_cat.items()):
                retryable_count = sum(1 for i in items if i["retryable"])
                lines.append(f"\n  [{cat}] {len(items)} failures ({retryable_count} retryable)")
                for item in items[:5]:
                    lines.append(f"    - {item['document_id']}: {item['error'][:80]} (attempts: {item['attempts']})")
                if len(items) > 5:
                    lines.append(f"    ... and {len(items) - 5} more")

            lines.append("\nTo retry failed documents, run sync_bddk_documents with force=True")
        else:
            lines.append("\nNo sync failures recorded.")

        # Vector store
        if deps.vector_store is not None:
            try:
                vs_stats = await deps.vector_store.stats()
                lines.append("\n**Vector Store**")
                lines.append(f"  Documents: {vs_stats['total_documents']}")
                lines.append(f"  Chunks: {vs_stats['total_chunks']}")
            except Exception:
                lines.append("\n**Vector Store:** unavailable")

        return "\n".join(lines)
