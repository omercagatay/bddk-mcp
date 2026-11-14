"""Analytics tools: analyze_bulletin_trends, get_regulatory_digest, compare_bulletin_metrics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from bddk_mcp.core.config import validate_column, validate_currency, validate_metric_id
from bddk_mcp.ingest.data_sources import (
    UNKNOWN_METRIC,
    WEEKLY_NUMBER_FORMAT_NOTE,
    WEEKLY_SERIES_LIMIT_NOTE,
    format_bulletin_number,
    format_snapshot_cell,
    weekly_metric_guide,
)
from bddk_mcp.observability.analytics import (
    PCT_UNAVAILABLE_TEXT,
    REASON_INSUFFICIENT_DATA,
    analyze_trends,
    build_digest,
    compare_metrics,
)
from bddk_mcp.tools.bulletin import unknown_metric_error, weekly_coverage_lines, with_doc
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
from bddk_mcp.tools.errors import INSUFFICIENT_DATA, INVALID_INPUT, UPSTREAM_FETCH_FAILED, tool_error
from bddk_mcp.tools.structured_outputs import frame_untrusted_source
from bddk_mcp.tools.tool_logging import logged_tool

if TYPE_CHECKING:
    from bddk_mcp.core.deps import Dependencies

logger = logging.getLogger(__name__)

_TREND_DOC = f"""
        Analyze trends in BDDK weekly bulletin data with week-over-week changes.

        Returns current value, WoW change %, trend direction, min/max over the
        lookback period, and a Turkish-language narrative summary.

        Args:
            metric_id: Metric ID, with BDDK's own row label:
{weekly_metric_guide(" " * 16)}
            currency: TRY or USD
            column: 1=TP (TL), 2=YP (Foreign Currency), 3=Toplam
            lookback_weeks: Number of weeks to analyze (default 12), counted back
                from the latest observation. BDDK serves only about the last 13 weeks
                of a series, so a longer lookback cannot be met; the output states
                the period actually covered and says so.
        """

_COMPARE_DOC = f"""
        Compare multiple BDDK bulletin metrics side-by-side.

        Args:
            metric_ids: Up to 4 comma-separated metric IDs (e.g. "1.0.1,1.0.3,1.0.12").
                Metric IDs with BDDK's own row label:
{weekly_metric_guide(" " * 16)}
            currency: TRY or USD
            column: 1=TP, 2=YP, 3=Toplam
            days: Days of history (default 90), counted back from the latest
                observation. BDDK serves only about the last 13 weeks of a series,
                so a longer window cannot be met; the output says so.
        """


def register(mcp, deps: Dependencies) -> None:
    """Register the public analytics tools."""

    @mcp.tool()
    @logged_tool(logger)
    @with_doc(_TREND_DOC)
    async def analyze_bulletin_trends(
        metric_id: MetricId = "1.0.1",
        currency: WeeklyCurrency = "TRY",
        column: BulletinColumn = "1",
        lookback_weeks: LookbackWeeks = 12,
    ) -> str:
        try:
            validate_metric_id(metric_id)
            validate_currency(currency, "weekly")
            validate_column(column)
        except ValueError:
            return tool_error(INVALID_INPUT, "One or more trend parameters are invalid.", retryable=False)

        result = await analyze_trends(deps.http, metric_id, currency, column, lookback_weeks)

        if result.get("reason") == UNKNOWN_METRIC:
            return unknown_metric_error(metric_id)
        if result.get("reason") == REASON_INSUFFICIENT_DATA:
            if result.get("trimmed_observations"):
                hint = (
                    f"The lookback window left out {result['trimmed_observations']} observation(s) BDDK returned; "
                    "increase lookback_weeks or choose another metric. Retrying the same request will not help."
                )
            else:
                hint = (
                    "Choose another metric; a longer lookback_weeks will not help, because BDDK serves only about "
                    "the last 13 weeks. Retrying the same request will not help."
                )
            return tool_error(
                INSUFFICIENT_DATA,
                f"Trend analysis needs at least 2 observations in the lookback window. {result['error']}",
                retryable=False,
                hint=hint,
            )
        if "error" in result:
            return tool_error(
                UPSTREAM_FETCH_FAILED,
                "BDDK trend data could not be analyzed because the upstream request failed.",
                retryable=True,
                hint="BDDK upstream may be temporarily unavailable; retry later.",
            )

        fmt = format_bulletin_number
        change = ("+" if result["wow_change"] >= 0 else "") + fmt(result["wow_change"])
        lines = [f"**Trend Analizi: {result['title']}**\n"]
        lines.extend(
            weekly_coverage_lines(
                first_date=result.get("first_date", ""),
                last_date=result["current_date"],
                observations=result["data_points"] if result.get("first_date") else 0,
                requested=f"son {lookback_weeks} hafta ({lookback_weeks * 7} gün)",
                unit=result.get("unit"),
                window_met=result.get("window_met"),
                returned_first_date=result.get("returned_first_date", ""),
            )
        )
        lines.append(WEEKLY_NUMBER_FORMAT_NOTE)
        lines.append("")
        lines.append(result["narrative"])
        lines.append("")
        lines.append(f"  Güncel ({result['current_date']}): {fmt(result['current'])}")
        lines.append(f"  Önceki ({result['previous_date']}): {fmt(result['previous'])}")
        if result["wow_pct"] is None:
            lines.append(f"  Haftalık değişim: {change} ({PCT_UNAVAILABLE_TEXT})")
        else:
            lines.append(f"  Haftalık değişim: {change} (%{result['wow_pct']:+.2f})")
        lines.append(f"  Dönem ortalaması: {fmt(result['avg'])}")
        lines.append(f"  Dönem min: {fmt(result['min'])} ({result['min_date']})")
        lines.append(f"  Dönem max: {fmt(result['max'])} ({result['max_date']})")
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
            period: Time period in calendar days, today included -- day (1), week (7), month (30), quarter (90).
                Decisions come from the local corpus catalog, not a live feed.
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
            if digest["total_decisions"] > 10:
                lines.append(f"  ... ve {digest['total_decisions'] - 10} daha")
            lines.append("")

        if not digest.get("announcements_available", True):
            lines.append(
                "**Duyurular:** veri alınamadı (BDDK üst kaynağına erişilemedi); duyuru bilgisi bu özette eksiktir."
            )
            lines.append("")
        elif digest["announcements"]:
            lines.append(f"**Duyurular ({digest['total_announcements']}):**")
            for a in digest["announcements"][:10]:
                lines.append(f"  - {a['title']} ({a.get('date', '')})")
            if digest["total_announcements"] > 10:
                lines.append(f"  ... ve {digest['total_announcements'] - 10} daha")
            lines.append("")

        if not digest.get("bulletin_snapshot_available", True):
            lines.append("**Bülten Özet:** veri alınamadı (BDDK üst kaynağına erişilemedi).")
        elif digest["bulletin_snapshot"]:
            heading = digest.get("bulletin_heading") or "Krediler"
            lines.append(f"**Bülten Özet (haftalık bülten, {heading} tablosu, ilk 5 satır):**")
            lines.append(
                f"  Bülten tarihi: {digest.get('bulletin_date') or 'belirtilmemiş'} | "
                f"Birim: {digest.get('bulletin_unit') or 'belirtilmemiş'} | {WEEKLY_NUMBER_FORMAT_NOTE}"
            )
            for r in digest["bulletin_snapshot"]:
                lines.append(f"  {r['name']}: TP={format_snapshot_cell(r['tp'])}, YP={format_snapshot_cell(r['yp'])}")

        return frame_untrusted_source("\n".join(lines))

    @mcp.tool()
    @logged_tool(logger)
    @with_doc(_COMPARE_DOC)
    async def compare_bulletin_metrics(
        metric_ids: MetricIdList = "1.0.1,1.0.2",
        currency: WeeklyCurrency = "TRY",
        column: BulletinColumn = "1",
        days: HistoryDays = 90,
    ) -> str:
        ids = parse_metric_ids(metric_ids)

        try:
            for mid in ids:
                validate_metric_id(mid)
            validate_currency(currency, "weekly")
            validate_column(column)
        except ValueError:
            return tool_error(INVALID_INPUT, "One or more comparison parameters are invalid.", retryable=False)

        result = await compare_metrics(deps.http, ids, currency, column, days)
        if result["metrics"] and all(m.get("reason") == UNKNOWN_METRIC for m in result["metrics"]):
            return unknown_metric_error(", ".join(m["metric_id"] for m in result["metrics"]))

        col_label = {"1": "TP", "2": "YP", "3": "Toplam"}.get(column, column)
        lines = [
            f"**Metrik Karşılaştırması** ({currency}, {col_label})\n",
            f"{'Metrik':<55} {'Tarih':>11} {'Güncel':>17} {'Haftalık %':>12}",
            "-" * 98,
        ]
        pct_missing = False
        found = [m for m in result["metrics"] if "error" not in m]
        for m in result["metrics"]:
            if m.get("reason") == REASON_INSUFFICIENT_DATA:
                lines.append(f"{m['metric_id']:<55} {'-':>11} {'YETERSİZ VERİ':>17} {'-':>12}  {m['error']}")
            elif m.get("reason") == UNKNOWN_METRIC:
                lines.append(f"{m['metric_id']:<55} {'-':>11} {'BİLİNMEYEN ID':>17} {'-':>12}  {m['error']}")
            elif "error" in m:
                lines.append(f"{m['metric_id']:<55} {'-':>11} {'HATA':>17} {'-':>12}")
            else:
                current = format_bulletin_number(m["current"])
                as_of = m.get("current_date", "")
                if m["wow_pct"] is None:
                    pct_missing = True
                    lines.append(f"{m['title'][:55]:<55} {as_of:>11} {current:>17} {'n/a':>12}")
                else:
                    lines.append(f"{m['title'][:55]:<55} {as_of:>11} {current:>17} {m['wow_pct']:>+11.2f}%")
        if pct_missing:
            lines.append(f"n/a: {PCT_UNAVAILABLE_TEXT}")
        if found:
            lines.append("")
            for m in found:
                if m.get("first_date"):
                    lines.append(
                        f"Kapsanan dönem ({m['metric_id']}): {m['first_date']} – {m.get('current_date', '')} "
                        f"({m['data_points']} gözlem; istenen pencere: son {days} gün)"
                    )
            units = sorted({m.get("unit") or "" for m in found})
            if units == [""]:
                lines.append(
                    "Birim: BDDK bu görünüm için birim belirtmiyor (sayfa birimi yalnızca TL görünümü için yazar)."
                )
            else:
                lines.append(
                    f"Birim: {', '.join(u for u in units if u)} (BDDK'nın haftalık bülten sayfasında belirttiği birim)"
                )
            if any(m.get("window_met") is False for m in found):
                lines.append(f"Not: {WEEKLY_SERIES_LIMIT_NOTE}; istenen pencere karşılanamadı.")
            lines.append(WEEKLY_NUMBER_FORMAT_NOTE)

        return frame_untrusted_source("\n".join(lines))
