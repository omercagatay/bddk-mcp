"""Tests for tools/bulletin.py — tool registration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.core.exceptions import BddkUpstreamError
from bddk_mcp.tools.bulletin import register


def test_register_exposes_end_user_tools_only_by_default():
    """Default (ADMIN_TOOLS=false) hides bddk_cache_status from end users."""
    mcp = MagicMock()
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    register(mcp, deps)

    tool_names = {call.args[0].__name__ for call in mcp.tool.return_value.call_args_list}
    assert tool_names == {
        "get_bddk_bulletin",
        "get_bddk_bulletin_snapshot",
        "get_bddk_monthly",
    }


def test_register_adds_admin_tool_only_when_explicitly_requested():
    """The registry must explicitly request the operator-only cache tool."""
    mcp = MagicMock()
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    register(mcp, deps, include_operator=True)

    tool_names = {call.args[0].__name__ for call in mcp.tool.return_value.call_args_list}
    assert tool_names == {
        "get_bddk_bulletin",
        "get_bddk_bulletin_snapshot",
        "get_bddk_monthly",
        "bddk_cache_status",
    }


# -- Upstream failure surfacing ----------------------------------------------


def _registered_tools(mcp: MagicMock) -> dict:
    return {call.args[0].__name__: call.args[0] for call in mcp.tool.return_value.call_args_list}


@pytest.mark.asyncio
async def test_bulletin_snapshot_upstream_failure_is_tool_error():
    """Blocked egress must surface as a retryable error, not 'No bulletin data'."""
    mcp = MagicMock()
    deps = Dependencies(pool=None, doc_store=None, client=None, http=MagicMock())
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
