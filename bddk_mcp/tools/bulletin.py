"""Bulletin tools: get_bddk_bulletin, get_bddk_bulletin_snapshot, get_bddk_monthly."""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING

from bddk_mcp.core.config import (
    validate_column,
    validate_currency,
    validate_metric_id,
    validate_month,
    validate_table_no,
    validate_year,
)
from bddk_mcp.core.exceptions import BddkUpstreamError
from bddk_mcp.ingest.data_sources import (
    MONTHLY_CLASSIC_COLUMNS,
    MONTHLY_REFUSED,
    MONTHLY_SHAPE_ERROR,
    UNKNOWN_METRIC,
    WEEKLY_NUMBER_FORMAT_NOTE,
    WEEKLY_SERIES_LIMIT_NOTE,
    fetch_bulletin_snapshot_table,
    fetch_monthly_bulletin,
    fetch_weekly_bulletin,
    format_bulletin_number,
    format_snapshot_cell,
    recent_complete_months,
    weekly_metric_guide,
)
from bddk_mcp.tools.contract_types import (
    BulletinColumn,
    BulletinDate,
    HistoryDays,
    MetricId,
    MonthlyCurrency,
    MonthlyMonth,
    MonthlyTableNumber,
    MonthlyYear,
    PartyCode,
    WeeklyCurrency,
)
from bddk_mcp.tools.errors import INVALID_INPUT, UPSTREAM_FETCH_FAILED, tool_error
from bddk_mcp.tools.structured_outputs import frame_untrusted_source
from bddk_mcp.tools.tool_logging import logged_tool

if TYPE_CHECKING:
    from bddk_mcp.core.deps import Dependencies

logger = logging.getLogger(__name__)

_MAX_PRINTED_OBSERVATIONS = 60
# BDDK publishes a month several weeks after it ends, so the newest complete
# month is usually not out yet; without a period the tool walks back this far.
_DEFAULT_MONTH_LOOKBACK = 3

# BDDK's own captions of the 17 monthly tables (as returned for TL, period suffix
# removed), checked against captured answers in tests/fixtures/bddk_monthly_tables.json.
MONTHLY_TABLE_CAPTIONS: dict[int, str] = {
    1: "Bilanço (milyon TL)",
    2: "Kar Zarar (milyon TL)",
    3: "Krediler (milyon TL)",
    4: "Tüketici Kredileri (milyon TL)",
    5: "Sektörel Kredi Dağılımı (bin TL)",
    6: "KOBİ Kredileri (milyon TL)",
    7: "Sendikasyon Seküritizasyon Kredileri (milyon TL)",
    8: "Menkul Kıymetler (milyon TL)",
    9: "Mevduat Türler İtibarıyla (milyon TL)",
    10: "Mevduat Vade İtibarıyla (milyon TL)",
    11: "Likidite Durumu (milyon TL)",
    12: "Sermaye Yeterliliği (milyon TL)",
    13: "Yabancı Para Pozisyonu (milyon TL)",
    14: "Bilanço Dışı İşlemler (milyon TL)",
    15: "Rasyolar",
    16: "Diğer Bilgiler",
    17: "Yurt Dışı Şube Rasyoları",
}


def with_doc(doc: str):
    """Set a generated tool description; FastMCP reads the function docstring.

    The text is cleaned like a docstring (inspect.cleandoc): written indented in an f-string, it would
    otherwise start with spaces that Markdown clients render as a code block.
    """

    def apply(func):
        func.__doc__ = inspect.cleandoc(doc)
        return func

    return apply


_MONTHLY_TABLE_LINES = "\n".join(
    f"                {no}={caption}," for no, caption in MONTHLY_TABLE_CAPTIONS.items()
).rstrip(",")

_MONTHLY_DOC = f"""
        Get BDDK monthly banking sector data (more detailed than weekly bulletin).

        The output prints BDDK's caption for the table and one column per value
        column BDDK sends (TP/YP/Toplam, maturity buckets, deposit-size bands,
        a single Toplam/Rasyo/Adet column, ...), under BDDK's own labels.

        Args:
            table_no: Table number (1-17). BDDK's table captions:
{_MONTHLY_TABLE_LINES}
            year: Year (e.g. 2025). Omit year and month together to get the most
                recent month BDDK has published (the last 3 complete months are
                tried, newest first; the output names the period used).
            month: Month (1-12)
            currency: TL or USD
            party_code: Bank group code. 10001=Sektör, 10002=Mevduat Bankaları,
                10003=Kalkınma ve Yatırım, 10004=Katılım Bankaları,
                20001=Kamu, 20002=Özel, 20003=Yabancı
        """

_WEEKLY_DOC = f"""
        Get weekly banking sector bulletin time-series data from BDDK.

        Args:
            metric_id: Metric ID, with BDDK's own row label:
{weekly_metric_guide(" " * 16)}.
                get_bddk_bulletin_snapshot lists every 1.0.x id of the Krediler table.
            currency: TRY or USD
            column: 1=TP (TL), 2=YP (Foreign Currency), 3=Toplam
            date: Specific date (DD.MM.YYYY), empty for latest
            days: Number of days of history (default 90), counted back from the
                latest observation. BDDK serves only about the last 13 weeks of a
                series whatever window is asked for, so a longer window cannot be
                met; the output states the period actually covered and says so.
        """


def unknown_metric_error(metric_id: str):
    """BDDK's placeholder answer for an id it does not know is bad input, not an empty period."""
    return tool_error(
        INVALID_INPUT,
        f"Unknown weekly metric id: {metric_id}. BDDK does not know this id (bilinmeyen metrik kimliği); "
        "this is not a period without data.",
        retryable=False,
        hint="Use get_bddk_bulletin_snapshot for the valid 1.0.x ids of the Krediler table, "
        "or an id listed in this tool's description.",
    )


def weekly_coverage_lines(
    *,
    first_date: str,
    last_date: str,
    observations: int,
    requested: str,
    unit: str | None,
    window_met: bool | None,
    returned_first_date: str = "",
) -> list[str]:
    """State the period a weekly answer really covers, its unit, and an unmet window."""
    lines = []
    if observations:
        lines.append(
            f"Kapsanan dönem: {first_date} – {last_date} ({observations} gözlem; istenen pencere: {requested})"
        )
    if unit:
        lines.append(f"Birim: {unit} (BDDK'nın haftalık bülten sayfasında belirttiği birim)")
    else:
        lines.append("Birim: BDDK bu görünüm için birim belirtmiyor (sayfa birimi yalnızca TL görünümü için yazar).")
    if window_met is False:
        oldest = f" (BDDK'nın döndürdüğü en eski gözlem: {returned_first_date})" if returned_first_date else ""
        lines.append(f"Not: {WEEKLY_SERIES_LIMIT_NOTE}{oldest}; istenen pencere karşılanamadı.")
    return lines


def register(
    mcp,
    deps: Dependencies,
) -> None:
    """Register bulletin tools on the given MCP instance."""

    @mcp.tool()
    @logged_tool(logger)
    @with_doc(_WEEKLY_DOC)
    async def get_bddk_bulletin(
        metric_id: MetricId = "1.0.1",
        currency: WeeklyCurrency = "TRY",
        column: BulletinColumn = "1",
        date: BulletinDate = "",
        days: HistoryDays = 90,
    ) -> str:
        try:
            validate_metric_id(metric_id)
            validate_currency(currency, "weekly")
            validate_column(column)
        except ValueError:
            return tool_error(INVALID_INPUT, "One or more bulletin parameters are invalid.", retryable=False)

        data = await fetch_weekly_bulletin(deps.http, metric_id, currency, days, date, column)

        if data.get("reason") == UNKNOWN_METRIC:
            return unknown_metric_error(metric_id)
        if "error" in data:
            return tool_error(
                UPSTREAM_FETCH_FAILED,
                "BDDK weekly bulletin data could not be retrieved.",
                retryable=True,
                hint="BDDK upstream may be temporarily unavailable; retry later.",
            )

        lines = [f"**{data.get('title', 'BDDK Weekly Bulletin')}** ({data['currency']})\n"]
        dates, values = data.get("dates", []), data.get("values", [])
        lines.extend(
            weekly_coverage_lines(
                first_date=dates[0] if dates else "",
                last_date=dates[-1] if dates else "",
                observations=len(dates),
                requested=f"son {days} gün",
                unit=data.get("unit"),
                window_met=data.get("window_met"),
                returned_first_date=data.get("returned_first_date", ""),
            )
        )
        if dates and values:
            lines.append(WEEKLY_NUMBER_FORMAT_NOTE)
            if len(values) > _MAX_PRINTED_OBSERVATIONS:
                lines.append(f"Son {_MAX_PRINTED_OBSERVATIONS} / {len(values)} gözlem gösteriliyor.")
            for d, v in zip(dates[-_MAX_PRINTED_OBSERVATIONS:], values[-_MAX_PRINTED_OBSERVATIONS:], strict=False):
                lines.append(f"  {d}: {format_bulletin_number(v)}")
        else:
            lines.append("No data returned for the given parameters.")

        return frame_untrusted_source("\n".join(lines))

    @mcp.tool()
    @logged_tool(logger)
    async def get_bddk_bulletin_snapshot() -> str:
        """
        Get the latest weekly bulletin snapshot: the Krediler (loans) table only.

        This is the one table BDDK's weekly bulletin page carries: the sector's
        Krediler rows (metric ids 1.0.x) with their latest TP (TL) and YP
        (foreign currency) values, the bulletin date and the unit BDDK states.
        BDDK's other weekly tables (deposits, NPLs, securities, ...) are not in
        it; their series can still be read with get_bddk_bulletin, for example
        2.0.1 (Takipteki Alacaklar) and 4.0.1 (Mevduat (Katılım Fonu)).
        """
        try:
            snapshot = await fetch_bulletin_snapshot_table(deps.http)
        except BddkUpstreamError:
            return tool_error(
                UPSTREAM_FETCH_FAILED,
                "The BDDK weekly bulletin snapshot could not be retrieved. "
                "This is NOT evidence that no bulletin data exists.",
                retryable=True,
                hint="BDDK upstream may be temporarily unavailable; retry later.",
            )

        rows = snapshot["rows"]
        if not rows:
            return "No bulletin data available."

        heading = snapshot.get("heading") or "Krediler"
        party = f" ({snapshot['party']})" if snapshot.get("party") else ""
        lines = [f"**BDDK Haftalık Bülten — {heading}{party} tablosu**\n"]
        lines.append(
            "Yalnızca haftalık bültenin Krediler tablosudur (BDDK sayfası bu tabloyu gösterir); "
            "mevduat, takipteki alacaklar ve diğer haftalık tablolar burada yoktur."
        )
        lines.append(f"Bülten tarihi: {snapshot.get('bulletin_date') or 'BDDK sayfasında belirtilmemiş'}")
        lines.append(f"Birim: {snapshot.get('unit') or 'BDDK sayfasında belirtilmemiş'}")
        lines.append(WEEKLY_NUMBER_FORMAT_NOTE + "\n")
        lines.append(f"{'#':<4} {'Metric':<50} {'TP':>15} {'YP':>15} {'ID'}")
        lines.append("-" * 100)
        for r in rows:
            tp, yp = format_snapshot_cell(r["tp"]), format_snapshot_cell(r["yp"])
            lines.append(f"{r['row_number']:<4} {r['name']:<50} {tp:>15} {yp:>15} {r['metric_id']}")
        return frame_untrusted_source("\n".join(lines))

    @mcp.tool()
    @logged_tool(logger)
    @with_doc(_MONTHLY_DOC)
    async def get_bddk_monthly(
        table_no: MonthlyTableNumber = 1,
        year: MonthlyYear = None,
        month: MonthlyMonth = None,
        currency: MonthlyCurrency = "TL",
        party_code: PartyCode = "10001",
    ) -> str:
        if (year is None) != (month is None):
            return tool_error(INVALID_INPUT, "Give both year and month, or neither.", retryable=False)
        explicit_period = year is not None and month is not None
        try:
            validate_table_no(table_no)
            if explicit_period:
                validate_year(year)
                validate_month(month)
            validate_currency(currency, "monthly")
        except ValueError:
            return tool_error(INVALID_INPUT, "One or more monthly bulletin parameters are invalid.", retryable=False)

        periods = [(year, month)] if explicit_period else recent_complete_months(_DEFAULT_MONTH_LOOKBACK)
        result: dict = {}
        for year, month in periods:
            result = await fetch_monthly_bulletin(deps.http, table_no, year, month, currency, party_code)
            if "error" in result or result.get("rows"):
                break

        if result.get("reason") == MONTHLY_SHAPE_ERROR:
            return tool_error(
                UPSTREAM_FETCH_FAILED,
                f"BDDK answered for monthly table {table_no}, but the response format was not understood "
                "(its rows do not match its column list). This is NOT evidence that BDDK has no data, "
                "and it is not a temporary outage.",
                retryable=False,
                hint="Retrying the same request will not help; the tool's parser needs to be updated.",
            )
        if result.get("reason") == MONTHLY_REFUSED:
            return tool_error(
                INVALID_INPUT,
                f"BDDK refused this request for monthly table {table_no} ({month}/{year}, {currency}, "
                f'taraf {party_code}). BDDK\'s answer: "{result["notice"]}" '
                "This is BDDK's own answer to these parameters (for example a currency or period the table "
                "does not offer), not an outage.",
                retryable=False,
                hint="Sending the same request again will not help; change the parameters (e.g. currency TL).",
            )
        if "error" in result and result.get("notice"):
            return tool_error(
                UPSTREAM_FETCH_FAILED,
                f"BDDK monthly bulletin data could not be retrieved: BDDK answered with an error of its own: "
                f'"{result["notice"]}"',
                retryable=True,
                hint="BDDK upstream may be temporarily unavailable; retry later.",
            )
        if "error" in result:
            return tool_error(
                UPSTREAM_FETCH_FAILED,
                "BDDK monthly bulletin data could not be retrieved.",
                retryable=True,
                hint="BDDK upstream may be temporarily unavailable; retry later.",
            )

        lines = [f"**Tablo {table_no}: {result.get('title', 'BDDK Aylık Bülten')}**\n"]
        rows = result.get("rows", [])
        notice = [f"BDDK'nın cevabı: {result['notice']}"] if result.get("notice") else []
        if not rows and not explicit_period:
            tried = ", ".join(f"{m}/{y}" for y, m in periods)
            lines.append(f"Denenen dönemler: {tried} | Para Birimi: {currency}\n")
            lines.append("BDDK bu dönemlerin hiçbiri için veri döndürmedi.")
            lines.extend(notice)
            return frame_untrusted_source("\n".join(lines))

        lines.append(f"Dönem: {month}/{year} | Para Birimi: {currency}\n")
        if not explicit_period:
            lines.append("Dönem verilmedi; BDDK'nın veri döndürdüğü en son tamamlanmış ay kullanıldı.\n")
        if not rows:
            lines.append("BDDK bu dönem ve parametreler için veri döndürmedi; dönem henüz yayımlanmamış olabilir.")
            lines.extend(notice)
        else:
            lines.append(
                "Sayı biçimi: BDDK'nın sayı olarak gönderdiği değerler 1,234.50 biçiminde yazılır "
                "(binlik ayırıcı virgül, ondalık nokta); metin olarak gönderdiği değerler olduğu gibi bırakılır; "
                "'-' = BDDK'da değer yok."
            )
            # BDDK's own value-column labels; a TP/YP/Toplam table keeps its old layout.
            columns = result.get("columns") or list(MONTHLY_CLASSIC_COLUMNS)
            widths = [max(15, len(label)) for label in columns]
            header = f"{'Kalem':<55}" + "".join(f" {label:>{w}}" for label, w in zip(columns, widths, strict=True))
            lines.append(header)
            lines.append("-" * max(105, len(header)))
            for r in rows:
                values = r.get("values") or [r.get("tp", "-"), r.get("yp", "-"), r.get("total", "-")]
                cells = "".join(f" {value:>{w}}" for value, w in zip(values, widths, strict=False))
                lines.append(f"{r['name']:<55}{cells}")
            if result.get("footnote"):
                lines.append(f"\nBDDK'nın notu: {result['footnote']}")

        return frame_untrusted_source("\n".join(lines))
