"""Tests for tools/analytics.py — tool registration and digest period mapping."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deps import Dependencies
from tools.analytics import register


def _registered_tools(mcp: MagicMock) -> dict:
    """Return {tool_name: tool_fn} for all @mcp.tool()-registered functions."""
    return {call.args[0].__name__: call.args[0] for call in mcp.tool.return_value.call_args_list}


def test_register_adds_four_analytics_tools():
    """register() should expose exactly the four documented analytics tools."""
    mcp = MagicMock()
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    register(mcp, deps)

    assert set(_registered_tools(mcp).keys()) == {
        "analyze_bulletin_trends",
        "get_regulatory_digest",
        "compare_bulletin_metrics",
        "check_bddk_updates",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "period,expected_days",
    [("day", 1), ("week", 7), ("month", 30), ("quarter", 90), ("unknown", 30)],
)
async def test_regulatory_digest_period_mapping(period, expected_days):
    """get_regulatory_digest must map 'day' → 1 day and fall back to 30 on unknown."""
    mcp = MagicMock()
    deps = Dependencies(pool=None, doc_store=None, client=MagicMock(), http=MagicMock())
    deps.client.ensure_cache = AsyncMock()
    deps.client.get_cache_items = MagicMock(return_value=[])
    register(mcp, deps)
    get_regulatory_digest = _registered_tools(mcp)["get_regulatory_digest"]

    with patch(
        "tools.analytics.build_digest",
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
