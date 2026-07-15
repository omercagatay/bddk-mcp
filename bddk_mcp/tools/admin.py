"""Admin tools: health_check, bddk_metrics, document_quality_report, backfill_degraded_documents."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Annotated

from pydantic import BeforeValidator, Field

from bddk_mcp.core.exceptions import BddkError, BddkStorageError
from bddk_mcp.corpus_publication import CorpusPublicationError, inspect_active_corpus_release
from bddk_mcp.ingest.backfill import BackfillOutcome, execute_backfill, group_by_signature, scan_candidates
from bddk_mcp.jobs import JobContext, JobExecutionError, JobKind, JobOutcome
from bddk_mcp.observability.metrics import metrics
from bddk_mcp.quality.quality_scan import format_report, scan_quality
from bddk_mcp.tools.sync import (
    JobView,
    OptionalIdempotencyKey,
    _job_manager,
    _submit_tracked_job,
    _tool_error,
)
from bddk_mcp.tools.tool_logging import logged_tool

if TYPE_CHECKING:
    from bddk_mcp.core.deps import Dependencies

logger = logging.getLogger(__name__)

BackfillLimit = Annotated[
    int,
    Field(ge=1, le=1000, description="Maximum candidates to process (1-1000)."),
]
BackfillDryRun = Annotated[
    bool,
    Field(description="When true, scan and report without changing stored documents."),
    BeforeValidator(lambda value: _strict_bool(value, name="dry_run")),
]
IncludeLegacyCorruption = Annotated[
    bool,
    Field(description="When true, include reviewed historical extraction-corruption signatures."),
    BeforeValidator(lambda value: _strict_bool(value, name="include_legacy_corruption")),
]


def _strict_bool(value: object, *, name: str) -> bool:
    if type(value) is not bool:
        _tool_error(f"invalid_{name}")
    return value


def register(mcp, deps: Dependencies) -> None:
    """Register admin tools on the given MCP instance."""

    @mcp.tool()
    @logged_tool(logger)
    async def health_check() -> str:
        """
        Check server health status.

        Returns uptime, cache status, store stats, and last sync time.
        """
        uptime_s = int(time.time() - deps.server_start_time)
        hours, remainder = divmod(uptime_s, 3600)
        minutes, seconds = divmod(remainder, 60)

        if deps.sync_circuit_open:
            status = "  Status: DEGRADED (sync circuit open after 10 consecutive failures)"
        elif deps.vector_store is None:
            status = "  Status: INITIALIZING (vector store loading)"
        else:
            status = "  Status: OK"
        last_sync = (
            f"  Last sync: {int(time.time() - deps.last_sync_time)}s ago"
            if deps.last_sync_time
            else "  Last sync: never"
        )
        lines = [
            f"**BDDK MCP Server Health**\n\n{status}\n  Uptime: {hours}h {minutes}m {seconds}s\n  Backend: PostgreSQL + pgvector",
            last_sync,
        ]
        if deps.last_sync_error:
            lines.append("  Last sync error: recorded (details withheld from tool output)")

        try:
            cs = deps.client.cache_status()
            lines.append(f"  Cache items: {cs['total_items']}\n  Cache valid: {cs['cache_valid']}")
        except (RuntimeError, BddkError, AttributeError):
            lines.append("  Cache: unavailable")

        try:
            st = await deps.doc_store.stats()
            lines.append(f"  Documents: {st.total_documents}")
        except (RuntimeError, BddkStorageError, AttributeError):
            lines.append("  Documents: unavailable")

        try:
            pool = deps.pool
            lines.append(f"  Pool: {pool.get_size()}/{pool.get_max_size()} connections ({pool.get_idle_size()} idle)")
        except (RuntimeError, AttributeError):
            lines.append("  Pool: unavailable")

        manager = _job_manager(deps)
        active_jobs = await manager.active_task_count() if manager is not None else 0
        lines.append(f"  Operator jobs active: {active_jobs}")

        try:
            release = await inspect_active_corpus_release(deps.pool) if deps.pool is not None else None
        except CorpusPublicationError:
            lines.append("  Active corpus release: unavailable")
        else:
            if release is None:
                lines.append("  Active corpus release: none")
            else:
                lines.extend(
                    (
                        f"  Active corpus release: {release.release_id}",
                        f"  Corpus manifest: id={release.manifest_id} sha256={release.manifest_sha256}",
                        f"  Corpus signer key sha256: {release.signer_key_sha256}",
                        f"  Corpus freshness policy: {release.freshness_policy_result}",
                        "  Corpus freshness SLOs: "
                        f"detection={release.source_detection_slo_seconds}s "
                        f"publication={release.publication_slo_seconds}s "
                        f"max_age={release.max_manifest_age_seconds}s",
                        f"  Retrieval profile sha256: {release.retrieval_profile_sha256}",
                        f"  Corpus state sha256: {release.corpus_state_sha256}",
                        f"  Corpus release completed at: {release.completed_at.isoformat()}",
                    )
                )

        return "\n".join(lines)

    @mcp.tool()
    @logged_tool(logger)
    async def bddk_metrics() -> str:
        """
        Show server performance metrics.

        Includes request counts, average latency per tool, error rates, and cache statistics.
        """
        m = metrics.summary()

        lines = [
            "**BDDK MCP Server Metrics**\n",
            f"  Uptime: {m['uptime_seconds']}s",
            f"  Total requests: {m['total_requests']}",
            f"  Total errors: {m['total_errors']}",
            f"  Cache hit rate: {m['cache_hit_rate']}%",
            f"  Cache hits/misses: {m['cache_hits']}/{m['cache_misses']}",
        ]

        if m["tools"]:
            lines.append(
                f"\n**Per-Tool Metrics:**\n  {'Tool':<35} {'Requests':>10} {'Errors':>8} {'Avg ms':>10}\n  " + "-" * 65
            )
            for t in m["tools"]:
                lines.append(f"  {t['tool']:<35} {t['requests']:>10} {t['errors']:>8} {t['avg_latency_ms']:>10.1f}")

        return "\n".join(lines)

    @mcp.tool()
    @logged_tool(logger)
    async def backfill_degraded_documents(
        dry_run: BackfillDryRun = True,
        limit: BackfillLimit = 100,
        include_legacy_corruption: IncludeLegacyCorruption = False,
        idempotency_key: OptionalIdempotencyKey = None,
    ) -> str | JobView:
        """
        Scan for degraded mevzuat documents and (optionally) re-extract them.

        Defaults to dry_run=True: reports candidates without modifying anything.
        Set dry_run=False to submit a tracked rescue job and receive its UUID;
        poll ``get_operator_job`` or ``list_operator_jobs`` for progress.

        The rescue path relies on HTML-first routing (``BDDK_PREFER_HTML_FOR_MEVZUAT``).
        On CPU-only deployments the default ``auto`` flips to True, and the
        mevzuat iframe HTML → html_parser extraction replaces the degraded
        markitdown-on-PDF output. On GPU deployments the flag stays False and
        this tool is largely a no-op — prefer running a fresh ``sync`` with
        LightOCR instead.

        Args:
            dry_run: If True (default), only scan and report. If False, execute
                the re-extraction in a background task.
            limit: Cap candidates processed (1-1000; default 100).
            include_legacy_corruption: Also match historical corruption signatures
                (U+FFFD chars, leaked ``<img`` tags, <500-char content) **and
                the three extraction-artifact signatures** detectable in
                stored markdown: ``i_garble`` (Đ where İ belongs — markitdown
                PDF font decode quirk, BDDK + mevzuat), ``form_feeds`` (PDF
                page-break 0x0C bytes left inline), ``c1_controls`` (Windows-1252
                punctuation bytes mis-decoded as C1 controls — shows as tofu
                boxes in readers). Default scans only
                ``extraction_method='markitdown_degraded'``.

        Returns a human-readable report. Destructive only when dry_run=False.
        """
        if isinstance(limit, bool) or not 1 <= limit <= 1000:
            return _tool_error("invalid_limit")
        if deps.pool is None:
            return "Backfill unavailable: DB pool not initialized."
        if deps.doc_store is None:
            return "Backfill unavailable: document store not initialized."

        if dry_run:
            try:
                candidates = await scan_candidates(
                    deps.pool,
                    include_legacy_corruption=include_legacy_corruption,
                    limit=limit,
                )
            except (BddkError, BddkStorageError, RuntimeError, OSError) as exc:
                logger.warning("Backfill candidate scan failed", extra={"error_type": type(exc).__name__})
                return _tool_error("backfill_scan_failed", retryable=True)

            by_sig = group_by_signature(candidates)
            lines = [f"**Backfill candidates: {len(candidates)}**"]
            for sig, count in sorted(by_sig.items()):
                lines.append(f"  {sig}: {count}")
            if candidates:
                lines.append("\n**First 10:**")
                for candidate in candidates[:10]:
                    lines.append(f"  {candidate.document_id}  len={candidate.len:>6}  sig={candidate.signature}")
            if len(candidates) > 10:
                lines.append(f"  ... and {len(candidates) - 10} more")
            if not candidates:
                lines.append("\nNothing to backfill.")
                return "\n".join(lines)
            lines.append("\nDry run — no changes made. Call with dry_run=False to execute.")
            return "\n".join(lines)

        async def runner(context: JobContext) -> JobOutcome:
            from bddk_mcp.ingest.doc_sync import DocumentSyncer

            try:
                await context.checkpoint()
                candidates = await scan_candidates(
                    deps.pool,
                    include_legacy_corruption=include_legacy_corruption,
                    limit=limit,
                )
                succeeded = 0
                failed = 0

                async def on_progress(index: int, total: int, outcome: BackfillOutcome) -> None:
                    nonlocal succeeded, failed
                    succeeded += int(outcome.success)
                    failed += int(not outcome.success)
                    await context.update_progress(
                        total=total,
                        completed=index,
                        succeeded=succeeded,
                        failed=failed,
                    )

                async with DocumentSyncer(deps.doc_store, http=deps.http, vector_store=deps.vector_store) as syncer:
                    report = await execute_backfill(syncer, candidates, on_progress=on_progress)
                return JobOutcome.from_metrics(
                    {
                        "total": report.total,
                        "succeeded": len(report.ok),
                        "failed": len(report.failed),
                    },
                    completed_with_errors=bool(report.failed),
                )
            except JobExecutionError:
                raise
            except (BddkError, BddkStorageError, RuntimeError, OSError, AttributeError, TypeError, ValueError):
                raise JobExecutionError("backfill_failed") from None

        return await _submit_tracked_job(
            deps,
            kind=JobKind.BACKFILL,
            arguments={
                "limit": limit,
                "include_legacy_corruption": include_legacy_corruption,
            },
            idempotency_key=idempotency_key,
            runner=runner,
        )

    @mcp.tool()
    @logged_tool(logger)
    async def document_quality_report() -> str:
        """
        Scan the document corpus for extraction anomalies.

        Reports extraction-method distribution and counts for each of:
        replacement characters, leaked HTML, short content, dot-leader runs,
        word-concatenation (html whitespace loss), formula-references-without-
        formulas, Turkish-diacritic outliers, orphan chunks, and docs missing
        chunks. Returns sample document IDs for each firing signal so issues
        can be traced to their source.

        Read-only. No network calls. Safe to run against a live server.
        """
        if deps.pool is None:
            return "Quality scan unavailable: DB pool not initialized."
        try:
            report = await scan_quality(deps.pool)
        except (BddkError, BddkStorageError, RuntimeError) as exc:
            logger.warning("Document quality scan failed", extra={"error_type": type(exc).__name__})
            return _tool_error("quality_scan_failed", retryable=True)
        return format_report(report)
