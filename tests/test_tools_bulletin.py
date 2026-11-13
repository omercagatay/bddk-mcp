"""Tests for tools/bulletin.py — tool registration."""

from __future__ import annotations

from datetime import UTC, datetime
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
            "bddk_mcp.tools.bulletin.fetch_bulletin_snapshot",
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
    assert result["rows"] == [
        {"name": "Krediler", "tp": "1,500", "yp": "2.50", "total": "x"},
        {"name": "Menkul", "tp": "-", "yp": "-", "total": "1,234.50"},  # missing cell is "-", never "None"
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
