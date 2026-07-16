"""Sync tools for BDDK MCP Server.

Provides document sync, pgvector migration, and circuit breaker helpers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Annotated, Any, Never
from uuid import UUID

from mcp.server.fastmcp.exceptions import ToolError
from pydantic import BeforeValidator, Field

from bddk_mcp.core.exceptions import BddkError
from bddk_mcp.jobs import (
    IdempotencyConflictError,
    JobContext,
    JobExecutionError,
    JobKind,
    JobManagerDrainingError,
    JobOutcome,
    JobState,
    OperatorJob,
    OperatorJobManager,
)
from bddk_mcp.tools.contract_types import OptionalDocumentId
from bddk_mcp.tools.tool_logging import logged_tool

if TYPE_CHECKING:
    from bddk_mcp.core.deps import Dependencies

logger = logging.getLogger(__name__)

CIRCUIT_BREAKER_THRESHOLD = 10
STARTUP_SYNC_TIMEOUT = 300  # 5 minutes
MIGRATION_TIMEOUT = 600  # 10 minutes
MAX_SYNC_CONCURRENCY = 20

SyncConcurrency = Annotated[
    int,
    Field(ge=1, le=MAX_SYNC_CONCURRENCY, description="Parallel document workers (1-20)."),
]
IdempotencyKey = Annotated[
    str,
    Field(
        min_length=1,
        max_length=256,
        description="Optional retry key; reuse is valid only with identical arguments.",
    ),
]
JobListLimit = Annotated[int, Field(ge=1, le=100, description="Maximum jobs to return (1-100).")]


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        _tool_error(f"invalid_{name}")
    return value


OptionalIdempotencyKey = Annotated[
    IdempotencyKey | None,
    Field(description="Optional retry key; reuse is valid only with identical arguments."),
]
ForceSync = Annotated[
    bool,
    Field(description="When true, replace cached document content and derived indexes."),
    BeforeValidator(lambda value: _strict_bool(value, name="force")),
]
OperatorJobId = Annotated[UUID, Field(description="Operator job UUID returned by a mutating tool.")]
JobStateFilter = Annotated[
    JobState | None,
    Field(description="Optional exact lifecycle-state filter."),
]
RetryableOnly = Annotated[
    bool,
    Field(description="When true, report only failures marked retryable."),
    BeforeValidator(lambda value: _strict_bool(value, name="retryable_only")),
]

JobView = dict[str, object]
JobRunner = Callable[[JobContext], Awaitable[JobOutcome | None]]


def _job_manager(deps: Dependencies) -> OperatorJobManager | None:
    """Return the injected manager without assuming older dependency shapes."""

    manager = getattr(deps, "job_manager", None)
    return manager if isinstance(manager, OperatorJobManager) else None


def _tool_error(code: str, *, retryable: bool = False) -> Never:
    """Raise a bounded MCP error without exception or argument text."""

    normalized = code.upper()
    raise ToolError(f"[ERROR:{normalized}] retryable={str(retryable).lower()}\nOperator request failed safely.")


def _job_receipt(job: OperatorJob) -> JobView:
    return {
        "job_id": str(job.job_id),
        "kind": job.kind.value,
        "state": job.state.value,
    }


def _job_view(job: OperatorJob) -> JobView:
    """Render only reviewed, privacy-safe job metadata."""

    return {
        **_job_receipt(job),
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "progress": {
            "total": job.progress.total,
            "completed": job.progress.completed,
            "succeeded": job.progress.succeeded,
            "failed": job.progress.failed,
        },
        "result_metrics": dict(job.result_metrics),
        "error_code": job.error_code,
    }


async def _submit_tracked_job(
    deps: Dependencies,
    *,
    kind: JobKind,
    arguments: Mapping[str, Any],
    idempotency_key: str | None,
    runner: JobRunner,
) -> JobView:
    """Submit one job and map repository failures to stable safe codes."""

    manager = _job_manager(deps)
    if manager is None:
        return _tool_error("job_manager_unavailable")
    try:
        job = await manager.submit(
            kind=kind,
            arguments=arguments,
            idempotency_key=idempotency_key,
            runner=runner,
        )
    except IdempotencyConflictError:
        return _tool_error("idempotency_conflict")
    except JobManagerDrainingError:
        return _tool_error("job_manager_draining", retryable=True)
    except (RuntimeError, ValueError, OSError) as exc:
        logger.error("Operator job submission failed", extra={"error_type": type(exc).__name__})
        return _tool_error("job_submission_failed", retryable=True)
    return _job_receipt(job)


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


async def _migrate_to_pgvector(deps: Dependencies) -> dict[str, int | bool]:
    """Migrate documents from document store to pgvector if needed.

    Uses a batch existence check instead of per-document has_document() calls.
    Aborts after MIGRATION_TIMEOUT seconds.

    Returns text-free numeric metrics and raises only safe job error codes.
    """
    vs = deps.vector_store
    store = deps.doc_store
    pool = deps.pool

    if vs is None:
        raise JobExecutionError("vector_store_unavailable")
    if store is None:
        raise JobExecutionError("document_store_unavailable")

    try:
        vs_stats = await vs.stats()
        store_stats = await store.stats()
        have, want = int(vs_stats["total_documents"]), int(store_stats.total_documents)

        if have >= want:
            logger.info("pgvector has %d/%d documents, skipping migration", have, want)
            return {
                "vector_documents": have,
                "vector_chunks": int(vs_stats["total_chunks"]),
                "vector_migrated": 0,
                "vector_complete": True,
            }

        logger.info("pgvector incomplete (%d/%d) — migrating...", have, want)

        start = time.monotonic()
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
                    "SELECT DISTINCT chunk.doc_id FROM public.document_chunks AS chunk "
                    "JOIN public.documents AS document "
                    "ON document.document_id = chunk.doc_id AND document.content_hash = chunk.content_hash "
                    "WHERE chunk.doc_id = ANY($1)",
                    doc_ids,
                )
                existing_ids = {r["doc_id"] for r in rows}
                batch_succeeded = True
            except Exception as e:
                logger.warning(
                    "Batch existence check failed; using per-document checks",
                    extra={"error_type": type(e).__name__},
                )

        deadline = start + MIGRATION_TIMEOUT

        for i, meta in enumerate(docs):
            if time.monotonic() > deadline:
                logger.warning("pgvector migration timed out after %ds", MIGRATION_TIMEOUT)
                raise JobExecutionError("vector_reconcile_timeout")

            doc_id = meta["document_id"]
            if (doc_id in existing_ids) if batch_succeeded else await vs.has_document(doc_id):
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

        elapsed = time.monotonic() - start
        logger.info("pgvector migration complete: %d docs, %d chunks, %.1fs", migrated, total_chunks, elapsed)
        final_stats = await vs.stats()
        return {
            "vector_documents": int(final_stats["total_documents"]),
            "vector_chunks": int(final_stats["total_chunks"]),
            "vector_migrated": migrated,
            "vector_complete": int(final_stats["total_documents"]) >= want,
        }

    except JobExecutionError:
        raise
    except (BddkError, RuntimeError, OSError, ValueError, TypeError) as exc:
        logger.error("pgvector migration failed", extra={"error_type": type(exc).__name__})
        raise JobExecutionError("vector_reconcile_failed") from None


async def startup_sync(deps: Dependencies, context: JobContext | None = None) -> JobOutcome:
    """Auto-sync documents on startup: download missing + embed to pgvector.

    Uses existing PostgreSQL cache — does NOT scrape BDDK for the decision list.
    Only downloads document content that is missing from the document store.
    Wrapped in asyncio.timeout(STARTUP_SYNC_TIMEOUT) to prevent hanging.
    """
    if deps.sync_circuit_open:
        logger.warning(
            "Startup sync skipped: circuit breaker open (%d consecutive failures)",
            deps.sync_consecutive_failures,
        )
        raise JobExecutionError("sync_circuit_open")

    logger.info("Startup sync started...")
    try:
        async with asyncio.timeout(STARTUP_SYNC_TIMEOUT):
            from bddk_mcp.ingest.doc_sync import DocumentSyncer

            store = deps.doc_store
            client = deps.client

            logger.info("Using existing cache: %d documents", client.cache_size())
            if not client.cache_size():
                logger.warning("Cache is empty — skipping startup sync (run refresh_bddk_cache first)")
                raise JobExecutionError("decision_cache_empty")

            st = await store.stats()
            cache_size = client.cache_size()
            downloaded = 0
            skipped = 0
            failed = 0

            # Phase 1: Download missing documents
            if st.total_documents < cache_size:
                logger.info("Document store incomplete (%d/%d) — downloading...", st.total_documents, cache_size)
                items = [d.model_dump() for d in client.get_cache_items()]
                async with DocumentSyncer(store, http=deps.http, vector_store=deps.vector_store) as syncer:
                    report = await syncer.sync_all(items, concurrency=10, force=False)
                downloaded = report.downloaded
                skipped = report.skipped
                failed = report.failed
                logger.info(
                    "Document sync: %d downloaded, %d failed, %.1fs",
                    report.downloaded,
                    report.failed,
                    report.elapsed_seconds,
                )
                if context is not None:
                    await context.update_progress(
                        total=report.total,
                        completed=report.downloaded + report.skipped + report.failed,
                        succeeded=report.downloaded + report.skipped,
                        failed=report.failed,
                    )
            else:
                logger.info("Document store has %d/%d documents, OK", st.total_documents, cache_size)

            # Phase 2: Migrate to pgvector
            vector_metrics = await _migrate_to_pgvector(deps)

            incomplete = failed > 0 or not bool(vector_metrics["vector_complete"])
            if incomplete:
                _record_sync_failure(deps, "document_sync_partial")
            else:
                _record_sync_success(deps)
            return JobOutcome.from_metrics(
                {
                    "cache_items": cache_size,
                    "downloaded": downloaded,
                    "skipped": skipped,
                    "failed": failed,
                    **vector_metrics,
                },
                completed_with_errors=incomplete,
            )

    except TimeoutError:
        logger.error("Startup sync timed out")
        _record_sync_failure(deps, "startup_sync_timeout")
        raise JobExecutionError("startup_sync_timeout") from None
    except JobExecutionError as exc:
        _record_sync_failure(deps, exc.code)
        raise
    except (BddkError, RuntimeError, OSError, AttributeError, TypeError, ValueError) as exc:
        logger.error("Startup sync failed", extra={"error_type": type(exc).__name__})
        _record_sync_failure(deps, "startup_sync_failed")
        raise JobExecutionError("startup_sync_failed") from None


def register(mcp, deps: Dependencies) -> None:
    """Register sync tools on the given MCP instance."""

    @mcp.tool()
    @logged_tool(logger)
    async def refresh_bddk_cache(idempotency_key: OptionalIdempotencyKey = None) -> JobView:
        """
        Submit a tracked job to re-scrape and replace the BDDK decision cache.

        Returns immediately with a job UUID and state. Use get_operator_job to
        inspect completion. Reusing an idempotency key is allowed only when all
        arguments are identical.
        """

        async def runner(context: JobContext) -> JobOutcome:
            try:
                await context.checkpoint()
                count = int(await deps.client.refresh_cache())
                await context.update_progress(total=1, completed=1, succeeded=1, failed=0)
                return JobOutcome.from_metrics({"cache_items": count})
            except JobExecutionError:
                raise
            except (BddkError, RuntimeError, OSError, AttributeError, TypeError, ValueError):
                raise JobExecutionError("cache_refresh_failed") from None

        return await _submit_tracked_job(
            deps,
            kind=JobKind.CACHE_REFRESH,
            arguments={},
            idempotency_key=idempotency_key,
            runner=runner,
        )

    @mcp.tool()
    @logged_tool(logger)
    async def sync_bddk_documents(
        force: ForceSync = False,
        document_id: OptionalDocumentId = None,
        concurrency: SyncConcurrency = 5,
        idempotency_key: OptionalIdempotencyKey = None,
    ) -> JobView:
        """
        Submit a tracked BDDK document synchronization job.

        Downloads documents from BDDK and mevzuat.gov.tr, extracts content to
        Markdown, stores it in PostgreSQL, and reconciles vector chunks. Returns
        immediately with a job UUID and state.

        Args:
            force: Re-download all documents even if already cached
            document_id: Sync a single document by ID (e.g. "1291" or "mevzuat_42628")
            concurrency: Number of parallel downloads, from 1 through 20.
            idempotency_key: Optional retry key for an identical request.
        """
        if isinstance(concurrency, bool) or not 1 <= concurrency <= MAX_SYNC_CONCURRENCY:
            return _tool_error("invalid_concurrency")

        async def runner(context: JobContext) -> JobOutcome:
            from bddk_mcp.ingest.doc_sync import DocumentSyncer

            try:
                store = deps.doc_store
                client = deps.client
                if store is None or client is None:
                    raise JobExecutionError("sync_dependencies_unavailable")
                await context.checkpoint()
                await client.ensure_cache()

                async with DocumentSyncer(store, http=deps.http, vector_store=deps.vector_store) as syncer:
                    if document_id:
                        found = client.find_by_id(document_id)
                        title, source_url, category = (
                            (found.title, found.source_url, found.category) if found else (document_id, "", "")
                        )
                        result = await syncer.sync_document(
                            doc_id=document_id,
                            title=title,
                            category=category,
                            source_url=source_url,
                            force=force,
                        )
                        total = 1
                        downloaded = int(result.success and result.method != "cached")
                        skipped = int(result.success and result.method == "cached")
                        failed = int(not result.success)
                    else:
                        items = [decision.model_dump() for decision in client.get_cache_items()]
                        report = await syncer.sync_all(items, concurrency=concurrency, force=force)
                        total = report.total
                        downloaded = report.downloaded
                        skipped = report.skipped
                        failed = report.failed

                completed = downloaded + skipped + failed
                await context.update_progress(
                    total=total,
                    completed=completed,
                    succeeded=downloaded + skipped,
                    failed=failed,
                )
                await context.checkpoint()
                vector_metrics = await _migrate_to_pgvector(deps)
                incomplete = failed > 0 or not bool(vector_metrics["vector_complete"])
                if incomplete:
                    _record_sync_failure(deps, "document_sync_partial")
                else:
                    _record_sync_success(deps)
                return JobOutcome.from_metrics(
                    {
                        "total": total,
                        "downloaded": downloaded,
                        "skipped": skipped,
                        "failed": failed,
                        **vector_metrics,
                    },
                    completed_with_errors=incomplete,
                )
            except JobExecutionError as exc:
                _record_sync_failure(deps, exc.code)
                raise
            except (BddkError, RuntimeError, OSError, AttributeError, TypeError, ValueError):
                _record_sync_failure(deps, "document_sync_failed")
                raise JobExecutionError("document_sync_failed") from None

        return await _submit_tracked_job(
            deps,
            kind=JobKind.DOCUMENT_SYNC,
            arguments={
                "force": force,
                "document_id": document_id,
                "concurrency": concurrency,
            },
            idempotency_key=idempotency_key,
            runner=runner,
        )

    @mcp.tool()
    @logged_tool(logger)
    async def trigger_startup_sync(idempotency_key: OptionalIdempotencyKey = None) -> JobView:
        """
        Submit the startup reconciliation routine as a tracked job.

        This invokes startup_sync: missing document content is downloaded and
        vector coverage is reconciled. Returns a job UUID immediately.
        """

        async def runner(context: JobContext) -> JobOutcome:
            # A manual operator action is the explicit half-open circuit probe.
            deps.sync_circuit_open = False
            deps.sync_consecutive_failures = 0
            return await startup_sync(deps, context=context)

        return await _submit_tracked_job(
            deps,
            kind=JobKind.CORPUS_RECONCILE,
            arguments={},
            idempotency_key=idempotency_key,
            runner=runner,
        )

    @mcp.tool()
    @logged_tool(logger)
    async def get_operator_job(job_id: OperatorJobId) -> JobView:
        """Return privacy-safe status for one operator job UUID."""

        manager = _job_manager(deps)
        if manager is None:
            return _tool_error("job_manager_unavailable")
        try:
            job = await manager.get(job_id)
        except (RuntimeError, OSError) as exc:
            logger.error("Operator job lookup failed", extra={"error_type": type(exc).__name__})
            return _tool_error("job_lookup_failed", retryable=True)
        return _job_view(job) if job is not None else _tool_error("job_not_found")

    @mcp.tool()
    @logged_tool(logger)
    async def list_operator_jobs(
        limit: JobListLimit = 20,
        state: JobStateFilter = None,
    ) -> JobView:
        """List up to 100 newest operator jobs, optionally filtered by state."""

        if isinstance(limit, bool) or not 1 <= limit <= 100:
            return _tool_error("invalid_limit")
        manager = _job_manager(deps)
        if manager is None:
            return _tool_error("job_manager_unavailable")
        try:
            states = {state} if state is not None else None
            jobs = await manager.list(limit=limit, states=states)
        except (RuntimeError, ValueError, OSError) as exc:
            logger.error("Operator job listing failed", extra={"error_type": type(exc).__name__})
            return _tool_error("job_list_failed", retryable=True)
        return {"count": len(jobs), "jobs": [_job_view(job) for job in jobs]}

    @mcp.tool()
    @logged_tool(logger)
    async def cancel_operator_job(job_id: OperatorJobId) -> JobView:
        """Request cancellation of one queued or running operator job UUID."""

        manager = _job_manager(deps)
        if manager is None:
            return _tool_error("job_manager_unavailable")
        try:
            job = await manager.cancel(job_id)
        except (RuntimeError, OSError) as exc:
            logger.error("Operator job cancellation failed", extra={"error_type": type(exc).__name__})
            return _tool_error("job_cancel_failed", retryable=True)
        return _job_view(job) if job is not None else _tool_error("job_not_found")

    @mcp.tool()
    @logged_tool(logger)
    async def document_health(retryable_only: RetryableOnly = False) -> str:
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
        from bddk_mcp.ingest.doc_sync import _is_error_page

        store = deps.doc_store
        client = deps.client
        pool = deps.pool

        # Document completeness
        st = await store.stats()
        cache_size = client.cache_size()

        lines = [
            f"**Document Health Report**\n\nDecision cache: {cache_size}\nDocuments with content: {st.total_documents}"
        ]
        if cache_size > 0:
            lines.append(f"Coverage: {st.total_documents / cache_size * 100:.1f}%")

        corrupted: list[dict] = []
        too_short: list[dict] = []
        missing_chunks: list[dict] = []
        if pool is not None:
            rows = await pool.fetch(
                "SELECT document_id, title, length(markdown_content) as content_len, "
                "left(markdown_content, 500) as preview "
                "FROM documents WHERE markdown_content IS NOT NULL"
            )
            for r in rows:
                if r["content_len"] < 100:
                    too_short.append(dict(r))
                elif _is_error_page(r["preview"]):
                    corrupted.append(dict(r))

        if corrupted:
            lines.append(f"\n**Corrupted Content: {len(corrupted)}** (error/404 pages stored as documents)")
            for doc in corrupted[:10]:
                lines.append(f"  - {doc['document_id']}: {doc['title'][:60]} ({doc['content_len']} bytes)")
            if len(corrupted) > 10:
                lines.append(f"  ... and {len(corrupted) - 10} more")
            lines.append("\nFix: run sync_bddk_documents with force=True for these document IDs")

        if too_short:
            lines.append(f"\n**Suspiciously Short: {len(too_short)}** (< 100 bytes)")
            for doc in too_short[:10]:
                lines.append(f"  - {doc['document_id']}: {doc['title'][:60]} ({doc['content_len']} bytes)")

        if pool is not None:
            missing_chunks = await pool.fetch(
                "SELECT d.document_id, d.title, length(d.markdown_content) as content_len "
                "FROM documents d "
                "LEFT JOIN (SELECT doc_id, count(*) as cnt FROM document_chunks GROUP BY doc_id) c "
                "ON c.doc_id = d.document_id "
                "WHERE length(d.markdown_content) > 1000 AND COALESCE(c.cnt, 0) <= 1"
            )
            if missing_chunks:
                lines.append(f"\n**Missing Chunks: {len(missing_chunks)}** (content exists but not chunked)")
                for doc in missing_chunks[:10]:
                    lines.append(f"  - {doc['document_id']}: {doc['title'][:60]} ({doc['content_len']} bytes)")
                lines.append("\nFix: run trigger_startup_sync to generate chunks")

        # Sync failures
        failures = await store.get_sync_failures(retryable_only=retryable_only)
        if failures:
            lines.append(f"\n**Sync Failures: {len(failures)}**")

            by_cat: dict[str, list[dict]] = {}
            for f in failures:
                by_cat.setdefault(f["error_category"], []).append(f)

            for cat, items in sorted(by_cat.items()):
                retryable_count = sum(1 for i in items if i["retryable"])
                lines.append(f"\n  [{cat}] {len(items)} failures ({retryable_count} retryable)")
                for item in items[:5]:
                    lines.append(f"    - {item['document_id']}: details withheld (attempts: {item['attempts']})")
                if len(items) > 5:
                    lines.append(f"    ... and {len(items) - 5} more")

            lines.append("\nTo retry failed documents, run sync_bddk_documents with force=True")
        else:
            lines.append("\nNo sync failures recorded.")

        if deps.vector_store is not None:
            try:
                vs = await deps.vector_store.stats()
                lines.append(
                    f"\n**Vector Store**\n  Documents: {vs['total_documents']}\n  Chunks: {vs['total_chunks']}"
                )
            except Exception:
                lines.append("\n**Vector Store:** unavailable")

        if not corrupted and not too_short and not missing_chunks and not failures:
            lines.append("\nAll documents healthy.")

        return "\n".join(lines)
