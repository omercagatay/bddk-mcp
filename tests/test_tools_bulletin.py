"""Tests for tools/bulletin.py — tool registration."""

from __future__ import annotations

import copy
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.core.exceptions import BddkUpstreamError
from bddk_mcp.ingest.data_sources import ISTANBUL_TZ
from bddk_mcp.tools.bulletin import register


def test_register_exposes_end_user_tools_only_by_default():
    """Only the three bulletin tools are registered."""
    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=None, http=None)
    register(mcp, deps)

    tool_names = {call.args[0].__name__ for call in mcp.tool.return_value.call_args_list}
    assert tool_names == {
        "get_bddk_bulletin",
        "get_bddk_bulletin_snapshot",
        "get_bddk_monthly",
    }


# -- Upstream failure surfacing ----------------------------------------------


def _registered_tools(mcp: MagicMock) -> dict:
    return {call.args[0].__name__: call.args[0] for call in mcp.tool.return_value.call_args_list}


@pytest.mark.asyncio
async def test_bulletin_snapshot_upstream_failure_is_tool_error():
    """Blocked egress must surface as a retryable error, not 'No bulletin data'."""
    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=None, http=MagicMock())
    register(mcp, deps)
    tool = _registered_tools(mcp)["get_bddk_bulletin_snapshot"]

    with (
        patch(
            "bddk_mcp.tools.bulletin.fetch_bulletin_snapshot_table",
            new=AsyncMock(side_effect=BddkUpstreamError("unreachable")),
        ),
        pytest.raises(ToolError) as excinfo,
    ):
        await tool()

    message = str(excinfo.value)
    assert "[ERROR:UPSTREAM_FETCH_FAILED]" in message
    assert "retryable=true" in message
    assert "NOT evidence" in message


# -- Honest outputs ------------------------------------------------------------


def _frozen_istanbul_now() -> datetime:
    return datetime(2026, 1, 15, 12, 0, tzinfo=ISTANBUL_TZ)


def _bulletin_tool(name: str):
    mcp = MagicMock()
    register(mcp, Dependencies(db=None, doc_store=None, client=None, http=MagicMock()))
    return _registered_tools(mcp)[name]


@pytest.mark.asyncio
async def test_weekly_bulletin_prints_window_up_to_60_and_says_when_cut():
    tool = _bulletin_tool("get_bddk_bulletin")
    data = {
        "title": "T",
        "currency": "TRY",
        "dates": [f"d{i}" for i in range(213)],
        "values": [float(i) for i in range(213)],
    }
    with patch("bddk_mcp.tools.bulletin.fetch_weekly_bulletin", new=AsyncMock(return_value=data)):
        out = await tool()
    assert "Son 60 / 213 gözlem gösteriliyor." in out
    assert "d212: 212.0" in out and "d153: 153.0" in out
    assert "d152:" not in out

    short = {**data, "dates": data["dates"][:13], "values": data["values"][:13]}
    with patch("bddk_mcp.tools.bulletin.fetch_weekly_bulletin", new=AsyncMock(return_value=short)):
        out = await tool()
    assert "gözlem gösteriliyor" not in out
    assert "d0: 0.0" in out and "d12: 12.0" in out


def test_recent_complete_months_are_computed_from_the_istanbul_date(monkeypatch):
    import bddk_mcp.ingest.data_sources as data_sources

    monkeypatch.setattr(data_sources, "istanbul_now", _frozen_istanbul_now)  # frozen today = 2026-01-15
    assert data_sources.recent_complete_months(3) == [(2025, 12), (2025, 11), (2025, 10)]


_MONTHLY_ROWS = [{"name": "Krediler", "tp": "1", "yp": "2", "total": "3"}]


_NOT_PUBLISHED = "2026 yılının en son 9 ayına ait veri bulunmaktadır!"


def _monthly_fetch(published: set[tuple[int, int]]) -> AsyncMock:
    """A fetch double: BDDK returns rows only for the published (year, month) periods."""

    async def fetch(_http, _table_no, year, month, *_rest):
        if (year, month) in published:
            return {"title": "T", "rows": _MONTHLY_ROWS}
        return {"title": "T", "rows": [], "notice": _NOT_PUBLISHED}

    return AsyncMock(side_effect=fetch)


def _periods(fetch: AsyncMock) -> list[tuple[int, int]]:
    return [call.args[2:4] for call in fetch.await_args_list]


@pytest.mark.asyncio
async def test_monthly_without_a_period_walks_back_to_the_newest_published_month(monkeypatch):
    """BDDK publishes a month weeks after it ends: the last complete month is usually not out yet."""
    import bddk_mcp.ingest.data_sources as data_sources

    monkeypatch.setattr(data_sources, "istanbul_now", _frozen_istanbul_now)
    tool = _bulletin_tool("get_bddk_monthly")

    fetch = _monthly_fetch({(2025, 11), (2025, 10)})
    with patch("bddk_mcp.tools.bulletin.fetch_monthly_bulletin", new=fetch):
        out = await tool()
    assert _periods(fetch) == [(2025, 12), (2025, 11)]
    assert "Dönem: 11/2025" in out
    assert "Dönem verilmedi" in out
    assert "Krediler" in out

    fetch = _monthly_fetch(set())
    with patch("bddk_mcp.tools.bulletin.fetch_monthly_bulletin", new=fetch):
        out = await tool()
    assert _periods(fetch) == [(2025, 12), (2025, 11), (2025, 10)]
    assert "Denenen dönemler: 12/2025, 11/2025, 10/2025" in out
    assert "hiçbiri için veri döndürmedi" in out
    assert f"BDDK'nın cevabı: {_NOT_PUBLISHED}" in out


@pytest.mark.asyncio
async def test_monthly_with_an_explicit_period_asks_once_and_says_when_bddk_has_no_data():
    tool = _bulletin_tool("get_bddk_monthly")

    fetch = _monthly_fetch(set())
    with patch("bddk_mcp.tools.bulletin.fetch_monthly_bulletin", new=fetch):
        out = await tool(year=2026, month=10)
    assert _periods(fetch) == [(2026, 10)]
    assert "Dönem: 10/2026" in out
    assert "Dönem verilmedi" not in out
    assert "yayımlanmamış olabilir" in out
    assert f"BDDK'nın cevabı: {_NOT_PUBLISHED}" in out

    with pytest.raises(ToolError, match="INVALID_INPUT"):
        await tool(year=2024)


@pytest.mark.asyncio
async def test_monthly_upstream_failure_stops_the_walk_and_is_a_tool_error(monkeypatch):
    import bddk_mcp.ingest.data_sources as data_sources

    monkeypatch.setattr(data_sources, "istanbul_now", _frozen_istanbul_now)
    tool = _bulletin_tool("get_bddk_monthly")
    fetch = AsyncMock(return_value={"error": "Approved BDDK upstream request failed."})
    with patch("bddk_mcp.tools.bulletin.fetch_monthly_bulletin", new=fetch):
        with pytest.raises(ToolError, match="UPSTREAM_FETCH_FAILED"):
            await tool()
    assert _periods(fetch) == [(2025, 12)]


_MONTHLY_PAGE = '<html><input name="__RequestVerificationToken" value="tok"/></html>'


def _patch_monthly_upstream(monkeypatch, payload):
    import bddk_mcp.ingest.data_sources as data_sources

    request = httpx.Request("POST", "https://www.bddk.org.tr/api")
    monkeypatch.setattr(
        data_sources, "_get", AsyncMock(return_value=httpx.Response(200, text=_MONTHLY_PAGE, request=request))
    )
    monkeypatch.setattr(
        data_sources, "_post", AsyncMock(return_value=httpx.Response(200, json=payload, request=request))
    )


@pytest.mark.asyncio
async def test_fetch_monthly_bulletin_parses_rows(monkeypatch):
    from bddk_mcp.ingest.data_sources import fetch_monthly_bulletin

    payload = {
        "success": True,
        "Json": {
            "caption": "Aktif",
            "data": {
                "rows": [
                    {"cell": [1, 1, "Krediler", 0, 1500, 2.5, "x"]},
                    {"cell": [2, 2, "Menkul", 0, None, "", 1234.5]},
                ]
            },
        },
    }
    _patch_monthly_upstream(monkeypatch, payload)
    result = await fetch_monthly_bulletin(MagicMock(), 1, 2025, 12, "TL", "10001")
    assert result["title"] == "Aktif"
    assert result["columns"] == ["TP", "YP", "Toplam"]  # no colNames: the classic 7-cell layout
    assert result["rows"] == [
        {"name": "Krediler", "values": ["1,500", "2.50", "x"], "tp": "1,500", "yp": "2.50", "total": "x"},
        # missing cell is "-", never "None"
        {"name": "Menkul", "values": ["-", "-", "1,234.50"], "tp": "-", "yp": "-", "total": "1,234.50"},
    ]
    assert result["period"] == "12/2025"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data",
    [
        {},  # renamed or missing rows key
        {"rows": None},
        {"rows": {"cell": []}},  # not a list
        {"rows": [{"cell": [1, 1, "Krediler", 0, 1500, 2.5]}]},  # 6 cells
        {"rows": [{"cell": "not a list"}]},
        {"rows": [{"cell": [1, 1, "K", 0, 1, 2, 3]}, {"cell": [1]}]},  # one bad row poisons the table
        {"rows": ["not an object"]},
    ],
)
async def test_fetch_monthly_bulletin_unexpected_rows_shape_is_an_error(monkeypatch, data):
    from bddk_mcp.ingest.data_sources import fetch_monthly_bulletin

    _patch_monthly_upstream(monkeypatch, {"success": True, "Json": {"caption": "x", "data": data}})
    assert "error" in await fetch_monthly_bulletin(MagicMock(), 1, 2025, 12, "TL", "10001")


@pytest.mark.asyncio
async def test_fetch_monthly_bulletin_empty_rows_list_still_means_no_data(monkeypatch):
    from bddk_mcp.ingest.data_sources import fetch_monthly_bulletin

    _patch_monthly_upstream(monkeypatch, {"success": True, "Json": {"caption": "x", "data": {"rows": []}}})
    result = await fetch_monthly_bulletin(MagicMock(), 1, 2025, 12, "TL", "10001")
    assert "error" not in result and result["rows"] == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "-"),
        ("", "-"),
        (1500, "1,500"),
        (1234.5, "1,234.50"),
        (0, "0"),
        # Text is never re-interpreted: "1.234" and "1.500" do not say whether the
        # dot is a thousands separator or a decimal mark.
        ("1.234,56", "1.234,56"),
        ("1.234", "1.234"),
        ("1.500", "1.500"),
        ("1234.56", "1234.56"),
        ("0.123", "0.123"),
        ("-1.234,5", "-1.234,5"),
        (" 1.234,56 ", "1.234,56"),
        ("n/a", "n/a"),
        ("  ", "-"),
        (float("nan"), "-"),
    ],
)
def test_format_number_formats_json_numbers_and_passes_text_through_unchanged(value, expected):
    from bddk_mcp.ingest.data_sources import _format_number

    assert _format_number(value) == expected


@pytest.mark.parametrize(
    ("utc_now", "expected_today"),
    [
        (datetime(2025, 12, 31, 21, 30, tzinfo=UTC), (2026, 1, 1)),  # 00:30 in Istanbul already next day
        (datetime(2025, 12, 31, 20, 59, tzinfo=UTC), (2025, 12, 31)),
    ],
)
def test_istanbul_today_is_independent_of_host_time_zone(utc_now, expected_today):
    from bddk_mcp.ingest.data_sources import istanbul_today, recent_complete_months

    today = istanbul_today(lambda: utc_now)
    assert (today.year, today.month, today.day) == expected_today
    # The monthly walk-back starts from the Istanbul month.
    first = recent_complete_months(1, lambda: utc_now)[0]
    assert first == ((2025, 12) if expected_today == (2026, 1, 1) else (2025, 11))


@pytest.mark.asyncio
async def test_weekly_bulletin_empty_period_prints_no_data_instead_of_upstream_failure():
    tool = _bulletin_tool("get_bddk_bulletin")
    data = {"title": "T", "currency": "TRY", "dates": [], "values": []}
    with patch("bddk_mcp.tools.bulletin.fetch_weekly_bulletin", new=AsyncMock(return_value=data)):
        out = await tool()
    assert "No data returned for the given parameters." in out


@pytest.mark.asyncio
async def test_monthly_table_states_number_format_and_prints_dash_for_missing_cells():
    tool = _bulletin_tool("get_bddk_monthly")
    result = {"title": "T", "rows": [{"name": "Krediler", "tp": "-", "yp": "2", "total": "3"}]}
    with patch("bddk_mcp.tools.bulletin.fetch_monthly_bulletin", new=AsyncMock(return_value=result)):
        out = await tool(year=2025, month=12)
    assert "None" not in out
    # The format line documents the number format and says that text values are not re-interpreted.
    assert "1,234.50" in out
    assert "olduğu gibi bırakılır" in out


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"success": False, "Json": {"caption": "x", "data": {"rows": []}}},
        {"success": False, "error": "Güvenlik doğrulaması başarısız."},
        {"success": False, "error": None},
        {"success": True, "Json": None},
        {"success": True},
        {"success": True, "Json": {"data": None}},
        ["not", "an", "object"],
    ],
)
async def test_fetch_monthly_bulletin_unexpected_shape_is_an_error_not_no_data(monkeypatch, payload):
    from bddk_mcp.ingest.data_sources import fetch_monthly_bulletin

    _patch_monthly_upstream(monkeypatch, payload)
    assert "error" in await fetch_monthly_bulletin(MagicMock(), 1, 2025, 12, "TL", "10001")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [_NOT_PUBLISHED, "2031 yılına ait veri bulunmamaktadır!"],  # both seen live on 2026-11-12
)
async def test_fetch_monthly_bulletin_reads_bddk_refusal_of_a_period_as_no_rows(monkeypatch, message):
    """Only BDDK's own "no data for this period" answer is no data; it is passed on as the notice."""
    from bddk_mcp.ingest.data_sources import fetch_monthly_bulletin

    _patch_monthly_upstream(monkeypatch, {"success": False, "error": message})
    result = await fetch_monthly_bulletin(MagicMock(), 1, 2026, 10, "TL", "10001")
    assert "error" not in result
    assert result["rows"] == []
    assert result["period"] == "10/2026"
    assert result["notice"] == message


# -- BDDK's own refusal of a monthly request ---------------------------------------

_FIXTURES = Path(__file__).parent / "fixtures"


def _monthly_answers() -> dict:
    return json.loads((_FIXTURES / "bddk_monthly_answers.json").read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_bddk_refusal_of_a_monthly_request_is_a_non_retryable_error_quoting_bddk(monkeypatch):
    """BDDK refusing a currency the table does not offer is its answer; retrying never helps."""
    refusal = _monthly_answers()["refusal_usd"]
    _patch_monthly_upstream(monkeypatch, refusal)
    with pytest.raises(ToolError) as excinfo:
        await _bulletin_tool("get_bddk_monthly")(table_no=17, year=2026, month=9, currency="USD", party_code="20003")
    message = str(excinfo.value)
    assert message.startswith("[ERROR:INVALID_INPUT] retryable=false")
    assert refusal["error"] in message
    assert "BDDK refused" in message
    assert "temporarily unavailable" not in message
    assert "retry" not in message.lower().replace("retryable=false", "")


@pytest.mark.asyncio
async def test_not_offered_match_is_turkish_aware_for_upper_case_and_dotted_i(monkeypatch):
    sentence = "BU TABLOYA İLİŞKİN USD CİNSİNDEN VERİ SUNULMAMAKTADIR."
    _patch_monthly_upstream(monkeypatch, {"success": False, "error": sentence})
    with pytest.raises(ToolError) as excinfo:
        await _bulletin_tool("get_bddk_monthly")(table_no=17, year=2026, month=9, currency="USD")
    message = str(excinfo.value)
    assert message.startswith("[ERROR:INVALID_INPUT] retryable=false")
    assert sentence in message


@pytest.mark.asyncio
async def test_other_bddk_error_message_is_a_retryable_upstream_error_quoting_bddk(monkeypatch):
    """A security/token failure is not the caller's arguments and may be transient."""
    _patch_monthly_upstream(monkeypatch, {"success": False, "error": "Güvenlik doğrulaması başarısız."})
    with pytest.raises(ToolError) as excinfo:
        await _bulletin_tool("get_bddk_monthly")(table_no=1, year=2026, month=9)
    message = str(excinfo.value)
    assert message.startswith("[ERROR:UPSTREAM_FETCH_FAILED] retryable=true")
    assert "Güvenlik doğrulaması başarısız." in message
    assert "error of its own" in message
    assert "INVALID_INPUT" not in message
    assert "refused" not in message


@pytest.mark.asyncio
async def test_bddk_no_data_answer_is_unchanged_by_the_refusal_handling(monkeypatch):
    _patch_monthly_upstream(monkeypatch, {"success": False, "error": _NOT_PUBLISHED})
    out = await _bulletin_tool("get_bddk_monthly")(table_no=1, year=2026, month=10)
    assert "BDDK bu dönem ve parametreler için veri döndürmedi" in out
    assert f"BDDK'nın cevabı: {_NOT_PUBLISHED}" in out


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload", [{"success": False, "error": ""}, {"success": False, "error": None}, {"success": False}]
)
async def test_refusal_without_a_bddk_message_keeps_the_retryable_upstream_error(monkeypatch, payload):
    _patch_monthly_upstream(monkeypatch, payload)
    with pytest.raises(ToolError) as excinfo:
        await _bulletin_tool("get_bddk_monthly")(table_no=1, year=2026, month=9)
    message = str(excinfo.value)
    assert message.startswith("[ERROR:UPSTREAM_FETCH_FAILED] retryable=true")
    assert "temporarily unavailable" in message


# -- Monthly tables in BDDK's real shapes (trimmed captured payloads) -------------


def _monthly_fixture(table_no: int) -> dict:
    tables = json.loads((_FIXTURES / "bddk_monthly_tables.json").read_text(encoding="utf-8"))["tables"]
    return copy.deepcopy(tables[str(table_no)])


_FOOTNOTE_12 = "*Merkezi Karşı Taraf- Garanti Fonu Tutarları Riskine Esas bilgisini de içermektedir."


@pytest.mark.asyncio
async def test_bddk_footnote_of_a_monthly_table_is_printed_once_after_the_table(monkeypatch):
    """Table 12 carries BDDK's footnote in Json.uyari as an HTML string with <br/> breaks."""
    from bddk_mcp.ingest.data_sources import fetch_monthly_bulletin

    payload = _monthly_fixture(12)
    payload["Json"]["uyari"] = _monthly_answers()["footnotes"]["12"]
    _patch_monthly_upstream(monkeypatch, payload)
    result = await fetch_monthly_bulletin(MagicMock(), 12, 2026, 9, "TL", "10001")
    assert result["footnote"] == _FOOTNOTE_12

    out = await _bulletin_tool("get_bddk_monthly")(table_no=12, year=2026, month=9)
    lines = out.splitlines()
    note_lines = [line for line in lines if line.startswith("BDDK'nın notu:")]
    assert note_lines == [f"BDDK'nın notu: {_FOOTNOTE_12}"]
    assert out.count(_FOOTNOTE_12) == 1
    last_row = next(i for i, line in enumerate(lines) if line.startswith("Orana Esas Sermaye Toplamı"))
    assert lines.index(note_lines[0]) > last_row
    assert "<br" not in out


@pytest.mark.asyncio
@pytest.mark.parametrize("uyari", ["absent", "empty"])
async def test_monthly_table_without_a_bddk_footnote_prints_no_note_line(monkeypatch, uyari):
    from bddk_mcp.ingest.data_sources import fetch_monthly_bulletin

    payload = _monthly_fixture(2)
    if uyari == "empty":
        payload["Json"]["uyari"] = _monthly_answers()["footnotes"]["2"]
    _patch_monthly_upstream(monkeypatch, payload)
    result = await fetch_monthly_bulletin(MagicMock(), 2, 2026, 9, "TL", "10001")
    assert "footnote" not in result

    out = await _bulletin_tool("get_bddk_monthly")(table_no=2, year=2026, month=9)
    assert "BDDK'nın notu" not in out


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("table_no", "column", "first_name", "first_value"),
    [
        (12, "Toplam", "Ana Sermaye Toplamı", "5,311,340"),
        (13, "Toplam", None, None),
        (15, "Rasyo", "Takipteki Alacaklar (Brüt) / Toplam Nakdi Krediler (%)", "3.05"),
        (16, "Adet", None, None),
        (17, "Rasyo", None, None),
    ],
)
async def test_single_value_column_tables_are_rows_not_an_outage(
    monkeypatch, table_no, column, first_name, first_value
):
    """Tables 12, 13, 15, 16, 17 send 5-cell rows with one value column; that is a full answer."""
    from bddk_mcp.ingest.data_sources import fetch_monthly_bulletin

    payload = _monthly_fixture(table_no)
    _patch_monthly_upstream(monkeypatch, payload)
    result = await fetch_monthly_bulletin(MagicMock(), table_no, 2026, 9, "TL", "10001")
    assert "error" not in result
    assert result["columns"] == [column]
    assert len(result["rows"]) == len(payload["Json"]["data"]["rows"])
    assert all(len(row["values"]) == 1 and "tp" not in row for row in result["rows"])

    out = await _bulletin_tool("get_bddk_monthly")(table_no=table_no, year=2026, month=9)
    assert f"Tablo {table_no}: {payload['Json']['caption']}" in out
    assert f"{'Kalem':<55} {column:>15}" in out
    if first_name:
        line = next(line for line in out.splitlines() if line.startswith(first_name))
        assert line.split()[-1] == first_value


@pytest.mark.asyncio
async def test_deposit_size_band_table_shows_bddk_headers_and_the_real_total(monkeypatch):
    payload = _monthly_fixture(9)
    _patch_monthly_upstream(monkeypatch, payload)
    out = await _bulletin_tool("get_bddk_monthly")(table_no=9, year=2026, month=9)
    header = next(line for line in out.splitlines() if line.startswith("Kalem"))
    for label in ["On Bin TL'ye Kadar", "On Bin-Elli Bin TL Arası", "Bir Milyon TL Üzeri", "Toplam"]:
        assert label in header
    assert " TP " not in header
    row = next(line for line in out.splitlines() if line.startswith("TP Mevduat / Katılım Fonları - Yurt İçi Yerleşik"))
    assert row.split()[-6:] == ["129,641", "394,986", "867,475", "1,526,373", "16,587,177", "19,505,652"]


@pytest.mark.asyncio
@pytest.mark.parametrize("table_no", [3, 5, 6, 10, 11])
async def test_multi_column_tables_keep_every_value_column(monkeypatch, table_no):
    from bddk_mcp.ingest.data_sources import fetch_monthly_bulletin

    payload = _monthly_fixture(table_no)
    _patch_monthly_upstream(monkeypatch, payload)
    result = await fetch_monthly_bulletin(MagicMock(), table_no, 2026, 9, "TL", "10001")
    expected = [" ".join(name.split()) for name in payload["Json"]["colNames"][4:]]
    assert result["columns"] == expected and len(expected) > 3
    first_cells = payload["Json"]["data"]["rows"][0]["cell"][4:]
    assert len(result["rows"][0]["values"]) == len(first_cells)
    assert "tp" not in result["rows"][0]

    out = await _bulletin_tool("get_bddk_monthly")(table_no=table_no, year=2026, month=9)
    header = next(line for line in out.splitlines() if line.startswith("Kalem"))
    assert all(label in header for label in expected)


@pytest.mark.asyncio
async def test_credit_table_separates_short_term_total_from_overall_total(monkeypatch):
    _patch_monthly_upstream(monkeypatch, _monthly_fixture(3))
    out = await _bulletin_tool("get_bddk_monthly")(table_no=3, year=2026, month=9)
    header = next(line for line in out.splitlines() if line.startswith("Kalem"))
    assert "Kısa Toplam" in header and "Orta Uzun TP" in header and header.rstrip().endswith(" Toplam")
    row = next(line for line in out.splitlines() if line.startswith("İhracat Kredileri"))
    assert row.split()[-9:] == [
        "395,029", "1,386,590", "1,781,619", "352,705", "1,379,627", "1,732,332", "747,734", "2,766,217", "3,513,951",
    ]  # fmt: skip


@pytest.mark.asyncio
async def test_tp_yp_toplam_table_keeps_its_layout_and_keys(monkeypatch):
    from bddk_mcp.ingest.data_sources import fetch_monthly_bulletin

    payload = _monthly_fixture(1)
    _patch_monthly_upstream(monkeypatch, payload)
    result = await fetch_monthly_bulletin(MagicMock(), 1, 2026, 9, "TL", "10001")
    assert result["columns"] == ["TP", "YP", "Toplam"]
    first = result["rows"][0]
    assert [first["tp"], first["yp"], first["total"]] == first["values"]

    out = await _bulletin_tool("get_bddk_monthly")(table_no=1, year=2026, month=9)
    assert f"{'Kalem':<55} {'TP':>15} {'YP':>15} {'Toplam':>15}" in out
    assert "-" * 105 in out


def _shape_error_payloads() -> list[dict]:
    short_row = _monthly_fixture(12)
    short_row["Json"]["data"]["rows"][1]["cell"] = short_row["Json"]["data"]["rows"][1]["cell"][:4]
    long_row = _monthly_fixture(1)
    long_row["Json"]["data"]["rows"][0]["cell"].append(1)
    no_col_names = _monthly_fixture(12)
    del no_col_names["Json"]["colNames"]
    return [short_row, long_row, no_col_names]


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", _shape_error_payloads())
async def test_rows_that_disagree_with_col_names_are_a_shape_error_not_an_outage(monkeypatch, payload):
    from bddk_mcp.ingest.data_sources import MONTHLY_SHAPE_ERROR, fetch_monthly_bulletin

    _patch_monthly_upstream(monkeypatch, payload)
    result = await fetch_monthly_bulletin(MagicMock(), 12, 2026, 9, "TL", "10001")
    assert result.get("reason") == MONTHLY_SHAPE_ERROR
    with pytest.raises(ToolError) as excinfo:
        await _bulletin_tool("get_bddk_monthly")(table_no=12, year=2026, month=9)
    message = str(excinfo.value)
    assert "retryable=false" in message
    assert "not understood" in message
    assert "temporarily unavailable" not in message


@pytest.mark.asyncio
async def test_classic_seven_cell_rows_without_col_names_are_still_read(monkeypatch):
    from bddk_mcp.ingest.data_sources import fetch_monthly_bulletin

    payload = _monthly_fixture(1)
    del payload["Json"]["colNames"]
    _patch_monthly_upstream(monkeypatch, payload)
    result = await fetch_monthly_bulletin(MagicMock(), 1, 2026, 9, "TL", "10001")
    assert result["columns"] == ["TP", "YP", "Toplam"] and result["rows"][0]["tp"] == result["rows"][0]["values"][0]


def _caption_without_period(caption: str) -> str:
    return " ".join(caption.split(", Dönem:")[0].split())


def test_monthly_table_captions_match_what_bddk_returns():
    from bddk_mcp.tools.bulletin import MONTHLY_TABLE_CAPTIONS

    tables = json.loads((_FIXTURES / "bddk_monthly_tables.json").read_text(encoding="utf-8"))["tables"]
    assert MONTHLY_TABLE_CAPTIONS == {
        int(no): _caption_without_period(t["Json"]["caption"]) for no, t in tables.items()
    }


def test_monthly_description_lists_bddk_captions_not_the_old_guesses():
    from bddk_mcp.tools.bulletin import MONTHLY_TABLE_CAPTIONS

    doc = " ".join(_bulletin_tool("get_bddk_monthly").__doc__.split())
    for no, caption in MONTHLY_TABLE_CAPTIONS.items():
        assert f"{no}={caption}" in doc
    assert "12=Sermaye Yeterliliği" in doc and "15=Rasyolar" in doc
    for wrong in ["9=Sermaye Yeterliliği", "14=Takipteki Alacaklar", "11=Gelir Tablosu", "1=Aktif Toplamı"]:
        assert wrong not in doc


def test_weekly_bulletin_description_uses_bddk_metric_labels():
    doc = " ".join(_bulletin_tool("get_bddk_bulletin").__doc__.split())
    assert "1.0.12=Ticari ve Diğer Krediler" in doc
    assert "1.0.10=b) Taksitsiz (Bireysel Kredi Kartları altında)" in doc
    assert "4.0.1=Mevduat (Katılım Fonu)" in doc and "2.0.1=Takipteki Alacaklar" in doc
    assert "1.0.10=Ticari" not in doc and "1.0.2=Tüketici Kredileri," not in doc


# -- Weekly answers say what period and unit they cover (captured BDDK page) -------


def _patch_weekly_upstream(monkeypatch, metric_id: str = "1.0.1"):
    import bddk_mcp.ingest.data_sources as data_sources

    series = json.loads((_FIXTURES / "bddk_weekly_series.json").read_text(encoding="utf-8"))["series"][metric_id]
    page = (_FIXTURES / "bddk_weekly_page.html").read_text(encoding="utf-8")
    request = httpx.Request("GET", "https://www.bddk.org.tr/bultenhaftalik")
    monkeypatch.setattr(data_sources, "_get", AsyncMock(return_value=httpx.Response(200, text=page, request=request)))
    monkeypatch.setattr(
        data_sources, "_post", AsyncMock(return_value=httpx.Response(200, json=series, request=request))
    )


def _date_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if re.match(r"\s+\d{1,2}\.\d{2}\.\d{4}: ", line)]


@pytest.mark.asyncio
async def test_weekly_bulletin_prints_only_the_requested_window_with_period_and_unit(monkeypatch):
    _patch_weekly_upstream(monkeypatch)
    out = await _bulletin_tool("get_bddk_bulletin")(metric_id="1.0.1", days=14)
    assert [line.strip() for line in _date_lines(out)] == [
        "23.10.2026: 18,320,099.58",
        "30.10.2026: 18,535,605.59",
        "6.11.2026: 18,523,337.66",
    ]
    assert "Kapsanan dönem: 23.10.2026 – 6.11.2026 (3 gözlem; istenen pencere: son 14 gün)" in out
    assert "Birim: Milyon TL" in out
    assert "Sayı biçimi: binlik ayırıcı virgül, ondalık nokta (1,234.50)." in out
    assert "karşılanamadı" not in out


@pytest.mark.asyncio
async def test_weekly_bulletin_says_when_bddk_cannot_meet_the_requested_window(monkeypatch):
    _patch_weekly_upstream(monkeypatch)
    out = await _bulletin_tool("get_bddk_bulletin")(metric_id="1.0.1", days=3650)
    assert len(_date_lines(out)) == 13
    assert "Kapsanan dönem: 14.08.2026 – 6.11.2026 (13 gözlem; istenen pencere: son 3650 gün)" in out
    assert "yalnızca son yaklaşık 13 haftayı döndürür" in out and "istenen pencere karşılanamadı" in out


@pytest.mark.asyncio
async def test_unknown_weekly_metric_is_invalid_input_pointing_to_the_snapshot(monkeypatch):
    _patch_weekly_upstream(monkeypatch, "9.9.9")
    with pytest.raises(ToolError) as excinfo:
        await _bulletin_tool("get_bddk_bulletin")(metric_id="9.9.9")
    message = str(excinfo.value)
    assert "[ERROR:INVALID_INPUT] retryable=false" in message
    assert "9.9.9" in message and "get_bddk_bulletin_snapshot" in message
    assert "Grafik Başlığı" not in message and "No data returned" not in message


@pytest.mark.asyncio
async def test_snapshot_names_the_krediler_table_its_date_and_unit(monkeypatch):
    _patch_weekly_upstream(monkeypatch)
    out = await _bulletin_tool("get_bddk_bulletin_snapshot")()
    assert "**BDDK Haftalık Bülten — Krediler (Sektör) tablosu**" in out
    assert "Yalnızca haftalık bültenin Krediler tablosudur" in out
    assert "Bülten tarihi: 6 Kasım 2026 Cuma" in out
    assert "Birim: Milyon TL" in out
    assert "Sayı biçimi: binlik ayırıcı virgül, ondalık nokta (1,234.50)." in out
    row = next(line for line in out.splitlines() if line.startswith("1 "))
    assert row.split()[-3:] == ["18,523,337", "10,517,881", "1.0.1"]


def test_snapshot_description_does_not_promise_deposits():
    doc = " ".join(_bulletin_tool("get_bddk_bulletin_snapshot").__doc__.split())
    assert "Krediler (loans) table only" in doc
    assert "loans, deposits" not in doc and "all banking sector metrics" not in doc


def test_weekly_days_description_states_the_bddk_limit():
    doc = " ".join(_bulletin_tool("get_bddk_bulletin").__doc__.split())
    assert "only about the last 13 weeks" in doc
