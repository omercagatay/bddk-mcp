"""Tests for tools/bulletin.py — tool registration."""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from mcp.server.fastmcp.exceptions import ToolError

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.core.exceptions import BddkUpstreamError
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


class _FrozenDate(date):
    @classmethod
    def today(cls):
        return cls(2026, 1, 15)


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


def test_last_complete_month_is_computed_from_today(monkeypatch):
    import bddk_mcp.ingest.data_sources as data_sources

    monkeypatch.setattr(data_sources, "date", _FrozenDate)
    assert data_sources.last_complete_month() == (2025, 12)  # frozen today = 2026-01-15


@pytest.mark.asyncio
async def test_monthly_defaults_to_last_complete_month(monkeypatch):
    import bddk_mcp.ingest.data_sources as data_sources

    monkeypatch.setattr(data_sources, "date", _FrozenDate)
    tool = _bulletin_tool("get_bddk_monthly")
    fetch = AsyncMock(return_value={"title": "T", "rows": []})
    with patch("bddk_mcp.tools.bulletin.fetch_monthly_bulletin", new=fetch):
        out = await tool()
    assert fetch.await_args.args[2:4] == (2025, 12)
    assert "Dönem: 12/2025" in out

    with pytest.raises(ToolError, match="INVALID_INPUT"):
        await tool(year=2024)


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
            "data": {"rows": [{"cell": [1, 1, "Krediler", 0, 1500, 2.5, "x"]}, {"cell": [1]}]},
        },
    }
    _patch_monthly_upstream(monkeypatch, payload)
    result = await fetch_monthly_bulletin(MagicMock(), 1, 2025, 12, "TL", "10001")
    assert result["title"] == "Aktif"
    assert result["rows"] == [{"name": "Krediler", "tp": "1,500", "yp": "2.50", "total": "x"}]
    assert result["period"] == "12/2025"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"success": False, "Json": {"caption": "x", "data": {"rows": []}}},
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
