"""Admin tools: health_check, bddk_metrics, document_quality_report."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

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
