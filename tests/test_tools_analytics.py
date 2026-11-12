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
