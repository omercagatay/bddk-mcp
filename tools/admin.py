"""Admin tools: health_check, bddk_metrics, document_quality_report, backfill_degraded_documents."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from backfill import BackfillOutcome, execute_backfill, group_by_signature, scan_candidates
from exceptions import BddkError, BddkStorageError
from metrics import metrics
from quality_scan import format_report, scan_quality

if TYPE_CHECKING:
    from deps import Dependencies

logger = logging.getLogger(__name__)


def register(mcp, deps: Dependencies) -> None:
    """Register admin tools on the given MCP instance."""

    @mcp.tool()
    async def health_check() -> str:
        """
        Check server health status.

        Returns uptime, cache status, store stats, and last sync time.
        """
        uptime_s = int(time.time() - deps.server_start_time)
        hours, remainder = divmod(uptime_s, 3600)
        minutes, seconds = divmod(remainder, 60)

        lines = ["**BDDK MCP Server Health**\n"]

        if deps.sync_circuit_open:
            lines.append("  Status: DEGRADED (sync circuit open after 10 consecutive failures)")
        elif deps.vector_store is None:
            lines.append("  Status: INITIALIZING (vector store loading)")
        else:
            lines.append("  Status: OK")

        lines.append(f"  Uptime: {hours}h {minutes}m {seconds}s")
        lines.append("  Backend: PostgreSQL + pgvector")

        if deps.last_sync_time:
            ago = int(time.time() - deps.last_sync_time)
            lines.append(f"  Last sync: {ago}s ago")
        else:
            lines.append("  Last sync: never")

        if deps.last_sync_error:
            lines.append(f"  Last sync error: {deps.last_sync_error}")

        # Cache status
        try:
            status = deps.client.cache_status()
            lines.append(f"  Cache items: {status['total_items']}")
            lines.append(f"  Cache valid: {status['cache_valid']}")
        except (RuntimeError, BddkError, AttributeError):
            lines.append("  Cache: unavailable")

        # Store status
        try:
            st = await deps.doc_store.stats()
            lines.append(f"  Documents: {st.total_documents}")
        except (RuntimeError, BddkStorageError, AttributeError):
            lines.append("  Documents: unavailable")

        # Pool utilization
        try:
            size = deps.pool.get_size()
            max_size = deps.pool.get_max_size()
            idle = deps.pool.get_idle_size()
            lines.append(f"  Pool: {size}/{max_size} connections ({idle} idle)")
        except (RuntimeError, AttributeError):
            lines.append("  Pool: unavailable")

        sync_status = "running" if (deps.sync_task and not deps.sync_task.done()) else "idle"
        lines.append(f"  Sync status: {sync_status}")

        return "\n".join(lines)

    @mcp.tool()
    async def bddk_metrics() -> str:
        """
        Show server performance metrics.

        Includes request counts, average latency per tool, error rates, and cache statistics.
        """
        m = metrics.summary()

        lines = ["**BDDK MCP Server Metrics**\n"]
        lines.append(f"  Uptime: {m['uptime_seconds']}s")
        lines.append(f"  Total requests: {m['total_requests']}")
        lines.append(f"  Total errors: {m['total_errors']}")
        lines.append(f"  Cache hit rate: {m['cache_hit_rate']}%")
        lines.append(f"  Cache hits/misses: {m['cache_hits']}/{m['cache_misses']}")

        if m["tools"]:
            lines.append("\n**Per-Tool Metrics:**")
            lines.append(f"  {'Tool':<35} {'Requests':>10} {'Errors':>8} {'Avg ms':>10}")
            lines.append("  " + "-" * 65)
            for t in m["tools"]:
                lines.append(f"  {t['tool']:<35} {t['requests']:>10} {t['errors']:>8} {t['avg_latency_ms']:>10.1f}")

        return "\n".join(lines)

    @mcp.tool()
    async def backfill_degraded_documents(
        dry_run: bool = True,
        limit: int = 0,
        include_legacy_corruption: bool = False,
    ) -> str:
        """
        Scan for degraded mevzuat documents and (optionally) re-extract them.

        Defaults to dry_run=True: reports candidates without modifying anything.
        Set dry_run=False to kick off the rescue in a background task; poll
        ``backfill_status`` to watch it, or ``document_quality_report`` to
        confirm the ``markitdown_degraded`` count has dropped to zero.

        The rescue path relies on HTML-first routing (``BDDK_PREFER_HTML_FOR_MEVZUAT``).
        On CPU-only deployments the default ``auto`` flips to True, and the
        mevzuat iframe HTML → html_parser extraction replaces the degraded
        markitdown-on-PDF output. On GPU deployments the flag stays False and
        this tool is largely a no-op — prefer running a fresh ``sync`` with
        LightOCR instead.

        Args:
            dry_run: If True (default), only scan and report. If False, execute
                the re-extraction in a background task.
            limit: Cap candidates processed (0 = no cap).
            include_legacy_corruption: Also match historical corruption signatures
                (U+FFFD chars, leaked ``<img`` tags, <500-char content). Default
                scans only ``extraction_method='markitdown_degraded'``.

        Returns a human-readable report. Destructive only when dry_run=False.
        """
        if deps.pool is None:
            return "Backfill unavailable: DB pool not initialized."
        if deps.doc_store is None:
            return "Backfill unavailable: document store not initialized."

        if not dry_run and deps.backfill_task and not deps.backfill_task.done():
            return "Backfill already running. Call `backfill_status` to see progress."

        try:
            candidates = await scan_candidates(
                deps.pool,
                include_legacy_corruption=include_legacy_corruption,
                limit=limit,
            )
        except (BddkError, BddkStorageError, RuntimeError) as exc:
            logger.warning("backfill_degraded_documents scan failed: %s", exc)
            return f"Scan failed: {exc}"

        by_sig = group_by_signature(candidates)
        lines = [f"**Backfill candidates: {len(candidates)}**"]
        for sig, count in sorted(by_sig.items()):
            lines.append(f"  {sig}: {count}")
        preview = candidates[:10]
        if preview:
            lines.append("\n**First 10:**")
            for c in preview:
                lines.append(f"  {c.document_id}  len={c.len:>6}  sig={c.signature}")
        if len(candidates) > 10:
            lines.append(f"  ... and {len(candidates) - 10} more")

        if not candidates:
            lines.append("\nNothing to backfill.")
            return "\n".join(lines)

        if dry_run:
            lines.append("\nDry run — no changes made. Call with dry_run=False to execute.")
            return "\n".join(lines)

        # Kick off the background task
        deps.backfill_progress = {
            "total": len(candidates),
            "processed": 0,
            "succeeded": 0,
            "failed": 0,
            "current": "",
            "state": "running",
            "signatures": by_sig,
        }
        deps.backfill_started_at = time.time()

        async def _run_backfill() -> None:
            from doc_sync import DocumentSyncer

            store = deps.doc_store
            http = deps.http
            vector_store = deps.vector_store

            async def on_progress(index: int, total: int, outcome: BackfillOutcome) -> None:
                deps.backfill_progress["processed"] = index
                if outcome.success:
                    deps.backfill_progress["succeeded"] += 1
                else:
                    deps.backfill_progress["failed"] += 1
                deps.backfill_progress["current"] = outcome.document_id

            try:
                async with DocumentSyncer(store, http=http, vector_store=vector_store) as syncer:
                    report = await execute_backfill(syncer, candidates, on_progress=on_progress)
                deps.backfill_progress["state"] = "done"
                deps.backfill_progress["elapsed_seconds"] = report.elapsed_seconds
                deps.backfill_progress["ok"] = len(report.ok)
                deps.backfill_progress["failures"] = report.failed
            except Exception as exc:
                logger.exception("Backfill task crashed")
                deps.backfill_progress["state"] = "error"
                deps.backfill_progress["error"] = f"{type(exc).__name__}: {exc}"

        deps.backfill_task = asyncio.create_task(_run_backfill())

        lines.append(f"\nStarted backfill of {len(candidates)} documents in background.")
        lines.append("Call `backfill_status` for progress (≈13s/doc on CPU-only deployments).")
        return "\n".join(lines)

    @mcp.tool()
    async def backfill_status() -> str:
        """
        Report progress of the most recent ``backfill_degraded_documents`` run.

        Shows running / done / error state, processed-vs-total counts, the
        current document ID (while running), elapsed time, and the list of
        failed IDs (when finished). Safe to poll.
        """
        if not deps.backfill_progress:
            return "No backfill has been triggered yet."

        p = deps.backfill_progress
        state = p.get("state", "unknown")
        total = p.get("total", 0)
        processed = p.get("processed", 0)
        succeeded = p.get("succeeded", 0)
        failed = p.get("failed", 0)

        lines = [f"**Backfill: {state}**"]
        lines.append(f"  Processed: {processed}/{total}")
        lines.append(f"  Succeeded: {succeeded}")
        lines.append(f"  Failed: {failed}")

        if deps.backfill_started_at:
            elapsed = time.time() - deps.backfill_started_at
            lines.append(f"  Elapsed: {elapsed:.1f}s")

        if state == "running":
            current = p.get("current", "")
            if current:
                lines.append(f"  Current: {current}")
        elif state == "done":
            lines.append(f"  Total time: {p.get('elapsed_seconds', 0):.1f}s")
            failures = p.get("failures", [])
            if failures:
                lines.append(f"\n**Failed IDs ({len(failures)}):**")
                for doc_id, reason in failures[:20]:
                    lines.append(f"  {doc_id}: {reason}")
                if len(failures) > 20:
                    lines.append(f"  ... and {len(failures) - 20} more")
        elif state == "error":
            lines.append(f"  Error: {p.get('error', 'unknown')}")

        return "\n".join(lines)

    @mcp.tool()
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
            logger.warning("document_quality_report: scan failed: %s", exc)
            return f"Quality scan failed: {exc}"
        return format_report(report)
