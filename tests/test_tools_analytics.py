"""Tests for tools/analytics.py — tool registration and digest period mapping."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.tools.analytics import register


def _registered_tools(mcp: MagicMock) -> dict:
    """Return {tool_name: tool_fn} for all @mcp.tool()-registered functions."""
    return {call.args[0].__name__: call.args[0] for call in mcp.tool.return_value.call_args_list}


def test_public_register_adds_only_stateless_analytics_tools():
    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=None, http=None)
    register(mcp, deps)

    assert set(_registered_tools(mcp).keys()) == {
        "analyze_bulletin_trends",
        "get_regulatory_digest",
        "compare_bulletin_metrics",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "period,expected_days",
    [("day", 1), ("week", 7), ("month", 30), ("quarter", 90)],
)
async def test_regulatory_digest_period_mapping(period, expected_days):
    """get_regulatory_digest must map 'day' → 1 day and fall back to 30 on unknown."""
    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=MagicMock(), http=MagicMock())
    deps.client.ensure_cache = AsyncMock()
    deps.client.get_cache_items = MagicMock(return_value=[])
    register(mcp, deps)
    get_regulatory_digest = _registered_tools(mcp)["get_regulatory_digest"]

    with patch(
        "bddk_mcp.tools.analytics.build_digest",
        new=AsyncMock(
            return_value={
                "narrative": "ok",
                "decisions_by_category": {},
                "new_decisions": [],
                "announcements": [],
                "bulletin_snapshot": [],
            }
        ),
    ) as mock_digest:
        out = await get_regulatory_digest(period=period)

    mock_digest.assert_awaited_once()
    called_period_days = mock_digest.await_args.args[2]
    assert called_period_days == expected_days
    assert f"Son {expected_days} Gün" in out


@pytest.mark.asyncio
async def test_regulatory_digest_rejects_unknown_period():
    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=MagicMock(), http=MagicMock())
    register(mcp, deps)
    get_regulatory_digest = _registered_tools(mcp)["get_regulatory_digest"]

    with pytest.raises(ToolError, match="INVALID_INPUT"):
        await get_regulatory_digest(period="unknown")


# -- Upstream failure surfacing ----------------------------------------------


@pytest.mark.asyncio
async def test_regulatory_digest_marks_unavailable_upstream_sections():
    """The digest must say announcement/bulletin data was unavailable, not omit it."""
    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=MagicMock(), http=MagicMock())
    deps.client.ensure_cache = AsyncMock()
    deps.client.get_cache_items = MagicMock(return_value=[])
    register(mcp, deps)
    get_regulatory_digest = _registered_tools(mcp)["get_regulatory_digest"]

    with patch(
        "bddk_mcp.tools.analytics.build_digest",
        new=AsyncMock(
            return_value={
                "narrative": "Duyuru verileri alınamadı.",
                "decisions_by_category": {},
                "new_decisions": [],
                "announcements": [],
                "announcements_available": False,
                "bulletin_snapshot": [],
                "bulletin_snapshot_available": False,
            }
        ),
    ):
        out = await get_regulatory_digest(period="week")

    assert "Duyurular:" in out
    assert "veri alınamadı" in out
    assert "Bülten Özet:" in out


@pytest.mark.asyncio
async def test_regulatory_digest_header_shows_true_totals_and_cut_lines():
    """Capped lists must say how many entries were left out; the header prints the true total."""
    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=MagicMock(), http=MagicMock())
    deps.client.ensure_cache = AsyncMock()
    deps.client.get_cache_items = MagicMock(return_value=[])
    register(mcp, deps)
    get_regulatory_digest = _registered_tools(mcp)["get_regulatory_digest"]

    with patch(
        "bddk_mcp.tools.analytics.build_digest",
        new=AsyncMock(
            return_value={
                "narrative": "ok",
                "decisions_by_category": {"Kurul Kararı": 25},
                "new_decisions": [{"title": f"K{i}", "decision_date": "01.01.2026"} for i in range(20)],
                "total_decisions": 25,
                "announcements": [{"title": f"D{i}", "date": "01.01.2026"} for i in range(20)],
                "total_announcements": 57,
                "bulletin_snapshot": [],
            }
        ),
    ):
        out = await get_regulatory_digest(period="month")

    assert "**Duyurular (57):**" in out
    assert "... ve 47 daha" in out
    assert "... ve 15 daha" in out


def _analytics_tool(name: str):
    mcp = MagicMock()
    register(mcp, Dependencies(db=None, doc_store=None, client=MagicMock(), http=MagicMock()))
    return _registered_tools(mcp)[name]


@pytest.mark.asyncio
async def test_trend_with_too_few_points_is_a_non_retryable_error_not_an_upstream_failure():
    tool = _analytics_tool("analyze_bulletin_trends")
    result = {"error": "En az 2 gözlem gerekir (bulunan: 1).", "reason": "insufficient_data"}
    with patch("bddk_mcp.tools.analytics.analyze_trends", new=AsyncMock(return_value=result)):
        with pytest.raises(ToolError) as excinfo:
            await tool()
    message = str(excinfo.value)
    assert "[ERROR:INSUFFICIENT_DATA] retryable=false" in message
    assert "UPSTREAM_FETCH_FAILED" not in message and "En az 2 gözlem" in message


@pytest.mark.asyncio
async def test_trend_upstream_error_stays_retryable():
    tool = _analytics_tool("analyze_bulletin_trends")
    with patch("bddk_mcp.tools.analytics.analyze_trends", new=AsyncMock(return_value={"error": "x"})):
        with pytest.raises(ToolError, match="UPSTREAM_FETCH_FAILED"):
            await tool()


@pytest.mark.asyncio
async def test_compare_table_shows_the_reason_and_the_missing_percentage():
    tool = _analytics_tool("compare_bulletin_metrics")
    result = {
        "metrics": [
            {"metric_id": "1.0.1", "error": "En az 2 gözlem gerekir (bulunan: 1).", "reason": "insufficient_data"},
            {"metric_id": "1.0.2", "error": "Approved BDDK upstream request failed."},
            {"metric_id": "1.0.4", "title": "Konut", "current": 5.0, "wow_pct": None},
        ]
    }
    with patch("bddk_mcp.tools.analytics.compare_metrics", new=AsyncMock(return_value=result)):
        out = await tool(metric_ids="1.0.1,1.0.2,1.0.4")
    assert "En az 2 gözlem gerekir" in out
    assert "HATA" in out
    assert "n/a" in out and "yüzde değişim hesaplanamadı (önceki değer 0)" in out


@pytest.mark.asyncio
async def test_trend_text_for_a_zero_base_does_not_print_a_percentage():
    tool = _analytics_tool("analyze_bulletin_trends")
    result = {
        "title": "T",
        "narrative": "n",
        "current": 50.0,
        "current_date": "b",
        "previous": 0.0,
        "previous_date": "a",
        "wow_change": 50.0,
        "wow_pct": None,
        "avg": 25.0,
        "min": 0.0,
        "min_date": "a",
        "max": 50.0,
        "max_date": "b",
        "trend_direction": "yükseliş",
        "data_points": 2,
    }
    with patch("bddk_mcp.tools.analytics.analyze_trends", new=AsyncMock(return_value=result)):
        out = await tool()
    assert "yüzde değişim hesaplanamadı (önceki değer 0)" in out and "%+" not in out
