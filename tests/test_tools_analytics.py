"""Tests for tools/analytics.py — tool registration and digest period mapping."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from bddk_mcp.core.config import ANNOUNCEMENT_CATEGORY_IDS
from bddk_mcp.core.deps import Dependencies
from bddk_mcp.tools.analytics import register


def _registered_tools(mcp: MagicMock) -> dict:
    """Return {tool_name: tool_fn} for all @mcp.tool()-registered functions."""
    return {call.args[0].__name__: call.args[0] for call in mcp.tool.return_value.call_args_list}


def test_public_register_adds_only_stateless_analytics_tools():
    mcp = MagicMock()
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    register(mcp, deps)

    assert set(_registered_tools(mcp).keys()) == {
        "analyze_bulletin_trends",
        "get_regulatory_digest",
        "compare_bulletin_metrics",
    }


def test_operator_register_adds_stateful_update_monitor():
    mcp = MagicMock()
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    register(mcp, deps, include_operator=True)

    assert "check_bddk_updates" in _registered_tools(mcp)


@pytest.mark.asyncio
async def test_operator_update_monitor_preserves_cross_request_baseline_state():
    mcp = MagicMock()
    client = MagicMock()
    client.known_announcements = set()
    client.get_cache_items.return_value = []
    deps = Dependencies(pool=None, doc_store=None, client=client, http=MagicMock())
    register(mcp, deps, include_operator=True)
    monitor = _registered_tools(mcp)["check_bddk_updates"]

    with (
        patch(
            "bddk_mcp.tools.analytics.fetch_announcements",
            new=AsyncMock(return_value=[{"url": "https://example.invalid/baseline"}]),
        ) as fetch,
        patch(
            "bddk_mcp.tools.analytics.check_updates",
            new=AsyncMock(
                return_value={
                    "new_announcements": [
                        {
                            "title": "Yeni duyuru",
                            "date": "2026-07-15",
                            "url": "https://example.invalid/new",
                        }
                    ],
                    "new_announcements_count": 1,
                }
            ),
        ) as check,
    ):
        baseline = await monitor()
        update = await monitor()

    assert baseline.startswith("Baseline oluşturuldu:")
    assert "1 Yeni Duyuru" in update
    assert "https://example.invalid/baseline" in client.known_announcements
    assert "https://example.invalid/new" in client.known_announcements
    assert fetch.await_count == len(ANNOUNCEMENT_CATEGORY_IDS)
    check.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "period,expected_days",
    [("day", 1), ("week", 7), ("month", 30), ("quarter", 90)],
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
    deps = Dependencies(pool=None, doc_store=None, client=MagicMock(), http=MagicMock())
    register(mcp, deps)
    get_regulatory_digest = _registered_tools(mcp)["get_regulatory_digest"]

    with pytest.raises(ToolError, match="INVALID_INPUT"):
        await get_regulatory_digest(period="unknown")


# -- Upstream failure surfacing ----------------------------------------------


@pytest.mark.asyncio
async def test_check_updates_baseline_upstream_failure_is_tool_error_and_stores_nothing():
    """A failed baseline fetch must not create an empty (false) baseline."""
    from mcp.server.fastmcp.exceptions import ToolError

    from bddk_mcp.core.exceptions import BddkUpstreamError

    mcp = MagicMock()
    client = MagicMock()
    client.known_announcements = set()
    deps = Dependencies(pool=None, doc_store=None, client=client, http=MagicMock())
    register(mcp, deps, include_operator=True)
    monitor = _registered_tools(mcp)["check_bddk_updates"]

    with (
        patch(
            "bddk_mcp.tools.analytics.fetch_announcements",
            new=AsyncMock(side_effect=BddkUpstreamError("unreachable")),
        ),
        pytest.raises(ToolError) as excinfo,
    ):
        await monitor()

    message = str(excinfo.value)
    assert "[ERROR:UPSTREAM_FETCH_FAILED]" in message
    assert "retryable=true" in message
    assert client.known_announcements == set()


@pytest.mark.asyncio
async def test_regulatory_digest_marks_unavailable_upstream_sections():
    """The digest must say announcement/bulletin data was unavailable, not omit it."""
    mcp = MagicMock()
    deps = Dependencies(pool=None, doc_store=None, client=MagicMock(), http=MagicMock())
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
async def test_check_updates_post_baseline_upstream_failure_is_tool_error():
    """With a baseline present, an upstream failure must not read as 'no news'."""
    from bddk_mcp.core.exceptions import BddkUpstreamError

    mcp = MagicMock()
    client = MagicMock()
    client.known_announcements = {"https://example.invalid/known"}
    client.get_cache_items.return_value = []
    deps = Dependencies(pool=None, doc_store=None, client=client, http=MagicMock())
    register(mcp, deps, include_operator=True)
    monitor = _registered_tools(mcp)["check_bddk_updates"]

    with (
        patch(
            "bddk_mcp.tools.analytics.check_updates",
            new=AsyncMock(side_effect=BddkUpstreamError("upstream down")),
        ),
        pytest.raises(ToolError) as excinfo,
    ):
        await monitor()

    message = str(excinfo.value)
    assert "[ERROR:UPSTREAM_FETCH_FAILED]" in message
    assert "NOT evidence" in message
