"""Tests for tools/bulletin.py — tool registration."""

from __future__ import annotations

from unittest.mock import MagicMock

from deps import Dependencies
from tools.bulletin import register


def test_register_adds_four_bulletin_tools():
    """register() should expose exactly the four documented bulletin tools."""
    mcp = MagicMock()
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    register(mcp, deps)

    tool_names = {call.args[0].__name__ for call in mcp.tool.return_value.call_args_list}
    assert tool_names == {
        "get_bddk_bulletin",
        "get_bddk_bulletin_snapshot",
        "get_bddk_monthly",
        "bddk_cache_status",
    }
