"""Analytics tools: analyze_bulletin_trends, get_regulatory_digest, compare_bulletin_metrics, check_bddk_updates."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bddk_mcp.core.config import ANNOUNCEMENT_CATEGORY_IDS, validate_column, validate_currency, validate_metric_id
from bddk_mcp.core.exceptions import BddkUpstreamError
from bddk_mcp.ingest.data_sources import fetch_announcements
from bddk_mcp.observability.analytics import analyze_trends, build_digest, check_updates, compare_metrics
from bddk_mcp.tools.contract_types import (
    BulletinColumn,
    HistoryDays,
    LookbackWeeks,
    MetricId,
    MetricIdList,
    RegulatoryPeriod,
    WeeklyCurrency,
    parse_metric_ids,
)
from bddk_mcp.tools.errors import INVALID_INPUT, UPSTREAM_FETCH_FAILED, tool_error
from bddk_mcp.tools.structured_outputs import frame_untrusted_source
from bddk_mcp.tools.tool_logging import logged_tool

if TYPE_CHECKING:
    from bddk_mcp.core.deps import Dependencies

logger = logging.getLogger(__name__)


def register(mcp, deps: Dependencies, *, include_operator: bool = False) -> None:
    """Register public analytics and, when requested, stateful monitoring."""

    @mcp.tool()
    @logged_tool(logger)
    async def analyze_bulletin_trends(
        metric_id: MetricId = "1.0.1",
        currency: WeeklyCurrency = "TRY",
        column: BulletinColumn = "1",
        lookback_weeks: LookbackWeeks = 12,
    ) -> str:
        """
        Analyze trends in BDDK weekly bulletin data with week-over-week changes.

        Returns current value, WoW change %, trend direction, min/max over the
        lookback period, and a Turkish-language narrative summary.

        Args:
            metric_id: Metric ID (e.g. 1.0.1=Toplam Krediler, 1.0.2=Tüketici Kredileri,
                1.0.4=Konut, 1.0.8=Bireysel Kredi Kartları, 1.0.10=Ticari Krediler)
            currency: TRY or USD
            column: 1=TP (TL), 2=YP (Foreign Currency), 3=Toplam
            lookback_weeks: Number of weeks to analyze (default 12)
        """
        try:
            validate_metric_id(metric_id)
            validate_currency(currency, "weekly")
            validate_column(column)
        except ValueError:
            return tool_error(INVALID_INPUT, "One or more trend parameters are invalid.", retryable=False)

        result = await analyze_trends(deps.http, metric_id, currency, column, lookback_weeks)

        if "error" in result:
            return tool_error(
                UPSTREAM_FETCH_FAILED,
                "BDDK trend data could not be analyzed because the upstream request failed.",
                retryable=True,
                hint="BDDK upstream may be temporarily unavailable; retry later.",
            )

        lines = [f"**Trend Analizi: {result['title']}**\n"]
        lines.append(result["narrative"])
        lines.append("")
        lines.append(f"  Güncel ({result['current_date']}): {result['current']:,.2f}")
        lines.append(f"  Önceki ({result['previous_date']}): {result['previous']:,.2f}")
        lines.append(f"  Haftalık değişim: {result['wow_change']:+,.2f} (%{result['wow_pct']:+.2f})")
        lines.append(f"  Dönem ortalaması: {result['avg']:,.2f}")
        lines.append(f"  Dönem min: {result['min']:,.2f} ({result['min_date']})")
        lines.append(f"  Dönem max: {result['max']:,.2f} ({result['max_date']})")
        lines.append(f"  Trend: {result['trend_direction']}")
        lines.append(f"  Veri noktası: {result['data_points']}")
        return frame_untrusted_source("\n".join(lines))

    @mcp.tool()
    @logged_tool(logger)
    async def get_regulatory_digest(
        period: RegulatoryPeriod = "month",
    ) -> str:
        """
        Get a digest of recent BDDK regulatory changes.

        Combines: new decisions, announcements, and bulletin data into
        an executive summary.

        Args:
            period: Time period -- day (last 24 hours), week (7 days), month (30 days), quarter (90 days)
        """
        period_map = {"day": 1, "week": 7, "month": 30, "quarter": 90}
        if period not in period_map:
            return tool_error(INVALID_INPUT, "Unsupported digest period.", retryable=False)
        days = period_map[period]

        await deps.client.ensure_cache()

        digest = await build_digest(deps.http, deps.client.get_cache_items(), days)

        lines = [f"**BDDK Düzenleyici Özet — Son {days} Gün**\n", digest["narrative"], ""]

        if digest["decisions_by_category"]:
            lines.append("**Kararlar (kategoriye göre):**")
            for cat, count in sorted(digest["decisions_by_category"].items(), key=lambda x: -x[1]):
                lines.append(f"  {cat}: {count}")
            lines.append("")

        if digest["new_decisions"]:
            lines.append("**Son Kararlar:**")
            for d in digest["new_decisions"][:10]:
                lines.append(f"  - {d['title']} ({d.get('decision_date', '')}) [{d.get('category', '')}]")
            lines.append("")

        if not digest.get("announcements_available", True):
            lines.append("**Duyurular:** veri alınamadı (BDDK üst kaynağına erişilemedi); duyuru bilgisi bu özette eksiktir.")
            lines.append("")
        elif digest["announcements"]:
            lines.append(f"**Duyurular ({len(digest['announcements'])}):**")
            for a in digest["announcements"][:10]:
                lines.append(f"  - {a['title']} ({a.get('date', '')})")
            lines.append("")

        if not digest.get("bulletin_snapshot_available", True):
            lines.append("**Bülten Özet:** veri alınamadı (BDDK üst kaynağına erişilemedi).")
        elif digest["bulletin_snapshot"]:
            lines.append("**Bülten Özet (ilk 5 metrik):**")
            for r in digest["bulletin_snapshot"]:
                lines.append(f"  {r['name']}: TP={r['tp']}, YP={r['yp']}")

        return frame_untrusted_source("\n".join(lines))

    @mcp.tool()
    @logged_tool(logger)
    async def compare_bulletin_metrics(
        metric_ids: MetricIdList = "1.0.1,1.0.2",
        currency: WeeklyCurrency = "TRY",
        column: BulletinColumn = "1",
        days: HistoryDays = 90,
    ) -> str:
        """
        Compare multiple BDDK bulletin metrics side-by-side.

        Args:
            metric_ids: Comma-separated metric IDs (e.g. "1.0.1,1.0.2,1.0.4")
                Common: 1.0.1=Toplam Krediler, 1.0.2=Tüketici, 1.0.4=Konut,
                1.0.8=Kredi Kartları, 1.0.10=Ticari Krediler
            currency: TRY or USD
            column: 1=TP, 2=YP, 3=Toplam
            days: Days of history (default 90)
        """
        ids = parse_metric_ids(metric_ids)

        try:
            for mid in ids:
                validate_metric_id(mid)
            validate_currency(currency, "weekly")
            validate_column(column)
        except ValueError:
            return tool_error(INVALID_INPUT, "One or more comparison parameters are invalid.", retryable=False)

        result = await compare_metrics(deps.http, ids, currency, column, days)

        col_label = {"1": "TP", "2": "YP", "3": "Toplam"}.get(column, column)
        lines = [
            f"**Metrik Karşılaştırması** ({currency}, {col_label})\n",
            f"{'Metrik':<55} {'Güncel':>15} {'Haftalık %':>12}",
            "-" * 85,
        ]
        for m in result["metrics"]:
            if "error" in m:
                lines.append(f"{m['metric_id']:<55} {'HATA':>15} {'-':>12}")
            else:
                lines.append(f"{m['title'][:55]:<55} {m['current']:>15,.2f} {m['wow_pct']:>+11.2f}%")

        return frame_untrusted_source("\n".join(lines))

    if not include_operator:
        return

    @mcp.tool()
    @logged_tool(logger)
    async def check_bddk_updates() -> str:
        """
        Check for new BDDK announcements since last check.

        Compares current announcements with cached state to detect new items.
        Useful for monitoring regulatory changes.
        """
        known_urls = deps.client.known_announcements
        if not known_urls:
            baseline: set[str] = set()
            try:
                for cat_id in ANNOUNCEMENT_CATEGORY_IDS:
                    baseline.update(a["url"] for a in await fetch_announcements(deps.http, cat_id) if a.get("url"))
            except BddkUpstreamError:
                # Never store a partial baseline: it would report every missed
                # announcement as "new" on the next call.
                return tool_error(
                    UPSTREAM_FETCH_FAILED,
                    "BDDK announcements could not be fetched, so no update baseline was created. "
                    "This is NOT evidence that there are no announcements.",
                    retryable=True,
                    hint="Retry later. In restricted networks, verify egress to www.bddk.org.tr is permitted.",
                )
            known_urls.update(baseline)
            deps.client.known_announcements = known_urls
            return f"Baseline oluşturuldu: {len(known_urls)} duyuru biliniyor. Bir sonraki çağrıda yeni duyurular tespit edilecek."

        try:
            result = await check_updates(deps.http, deps.client.get_cache_items(), known_urls)
        except BddkUpstreamError:
            return tool_error(
                UPSTREAM_FETCH_FAILED,
                "BDDK announcements could not be fetched; update status is unknown. "
                "This is NOT evidence that there are no new announcements.",
                retryable=True,
                hint="Retry later. In restricted networks, verify egress to www.bddk.org.tr is permitted.",
            )

        for a in result.get("new_announcements", []):
            if a.get("url"):
                known_urls.add(a["url"])

        if not result["new_announcements"]:
            return "Yeni duyuru yok. Her şey güncel."

        lines = [f"**{result['new_announcements_count']} Yeni Duyuru Tespit Edildi!**\n"]
        for a in result["new_announcements"]:
            date = a.get("date", "")
            lines.append(f"  - {a['title']} ({date})")
            if a.get("url"):
                lines.append(f"    {a['url']}")
        return frame_untrusted_source("\n".join(lines))
