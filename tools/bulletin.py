"""Bulletin tools: get_bddk_bulletin, get_bddk_bulletin_snapshot, get_bddk_monthly, bddk_cache_status."""

from __future__ import annotations

from typing import TYPE_CHECKING

from config import (
    validate_column,
    validate_currency,
    validate_metric_id,
    validate_month,
    validate_table_no,
    validate_year,
)
from data_sources import fetch_bulletin_snapshot, fetch_weekly_bulletin

if TYPE_CHECKING:
    from deps import Dependencies


def register(mcp, deps: Dependencies) -> None:
    """Register bulletin tools on the given MCP instance."""

    @mcp.tool()
    async def get_bddk_bulletin(
        metric_id: str = "1.0.1",
        currency: str = "TRY",
        column: str = "1",
        date: str = "",
        days: int = 90,
    ) -> str:
        """
        Get weekly banking sector bulletin time-series data from BDDK.

        Args:
            metric_id: Metric ID. Common IDs:
                1.0.1=Toplam Krediler, 1.0.2=Tüketici Kredileri,
                1.0.4=Konut Kredileri, 1.0.8=Bireysel Kredi Kartları,
                1.0.10=Ticari Krediler. Use get_bddk_bulletin_snapshot for all metrics.
            currency: TRY or USD
            column: 1=TP (TL), 2=YP (Foreign Currency), 3=Toplam
            date: Specific date (DD.MM.YYYY), empty for latest
            days: Number of days of history (default 90)
        """
        try:
            validate_metric_id(metric_id)
            validate_currency(currency, "weekly")
            validate_column(column)
        except ValueError as e:
            return f"Validation error: {e}"

        data = await fetch_weekly_bulletin(
            deps.http,
            metric_id,
            currency,
            days,
            date,
            column,
        )

        if "error" in data:
            return f"Error fetching bulletin: {data['error']}"

        lines = [f"**{data.get('title', 'BDDK Weekly Bulletin')}** ({data['currency']})\n"]

        dates = data.get("dates", [])
        values = data.get("values", [])

        if dates and values:
            for d, v in zip(dates[-10:], values[-10:], strict=False):
                lines.append(f"  {d}: {v}")
        else:
            lines.append("No data returned for the given parameters.")

        return "\n".join(lines)

    @mcp.tool()
    async def get_bddk_bulletin_snapshot() -> str:
        """
        Get the latest weekly bulletin snapshot -- all metrics with current TP/YP values.

        Returns a table of all banking sector metrics (loans, deposits, etc.)
        with their latest TP (TL) and YP (foreign currency) values.
        """
        rows = await fetch_bulletin_snapshot(deps.http)

        if not rows:
            return "No bulletin data available."

        lines = ["**BDDK Weekly Bulletin — Latest Snapshot**\n"]
        lines.append(f"{'#':<4} {'Metric':<50} {'TP':>15} {'YP':>15} {'ID'}")
        lines.append("-" * 100)
        for r in rows:
            lines.append(f"{r['row_number']:<4} {r['name']:<50} {r['tp']:>15} {r['yp']:>15} {r['metric_id']}")
        return "\n".join(lines)

    @mcp.tool()
    async def get_bddk_monthly(
        table_no: int = 1,
        year: int = 2025,
        month: int = 12,
        currency: str = "TL",
        party_code: str = "10001",
    ) -> str:
        """
        Get BDDK monthly banking sector data (more detailed than weekly bulletin).

        Args:
            table_no: Table number (1-17). Key tables:
                1=Aktif Toplamı, 2=Krediler, 3=Menkul Değerler,
                4=Mevduat, 9=Sermaye Yeterliliği, 11=Gelir Tablosu,
                14=Takipteki Alacaklar
            year: Year (e.g. 2025)
            month: Month (1-12)
            currency: TL or USD
            party_code: Bank group code. 10001=Sektör, 10002=Mevduat Bankaları,
                10003=Kalkınma ve Yatırım, 10004=Katılım Bankaları,
                20001=Kamu, 20002=Özel, 20003=Yabancı
        """
        try:
            validate_table_no(table_no)
            validate_year(year)
            validate_month(month)
            validate_currency(currency, "monthly")
        except ValueError as e:
            return f"Validation error: {e}"

        from data_sources import fetch_monthly_bulletin

        result = await fetch_monthly_bulletin(
            deps.http,
            table_no,
            year,
            month,
            currency,
            party_code,
        )

        if "error" in result:
            return f"Error: {result['error']}"

        lines = [f"**{result.get('title', 'BDDK Aylık Bülten')}**\n"]
        lines.append(f"Dönem: {month}/{year} | Para Birimi: {currency}\n")

        rows = result.get("rows", [])
        if not rows:
            lines.append("Bu parametreler için veri bulunamadı.")
        else:
            lines.append(f"{'Kalem':<55} {'TP':>15} {'YP':>15} {'Toplam':>15}")
            lines.append("-" * 105)
            for r in rows:
                lines.append(f"{r['name']:<55} {r.get('tp', ''):>15} {r.get('yp', ''):>15} {r.get('total', ''):>15}")

        return "\n".join(lines)

    @mcp.tool()
    async def bddk_cache_status() -> str:
        """
        Show BDDK cache statistics: total items, age, categories, and any page errors.
        """
        status = deps.client.cache_status()

        lines = ["**BDDK Cache Status**\n"]
        lines.append(f"  Total items: {status['total_items']}")
        lines.append(f"  Cache valid: {status['cache_valid']}")
        if status["cache_age_seconds"] is not None:
            mins = status["cache_age_seconds"] // 60
            lines.append(f"  Cache age: {mins} min ({status['cache_age_seconds']}s)")
        lines.append(f"  TTL: {status['ttl_seconds']}s")

        if status["categories"]:
            lines.append("\n**Categories:**")
            for cat, count in status["categories"].items():
                lines.append(f"  {cat}: {count}")

        if status["page_errors"]:
            lines.append("\n**Page Errors:**")
            for page_id, err in status["page_errors"].items():
                lines.append(f"  Page {page_id}: {err}")

        return "\n".join(lines)
