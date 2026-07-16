"""Tests for tools/bulletin.py — tool registration."""

from __future__ import annotations

from unittest.mock import MagicMock

from bddk_mcp.core.deps import Dependencies
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
