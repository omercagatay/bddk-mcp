"""Tests for tools/analytics.py — tool registration."""

from __future__ import annotations

from unittest.mock import MagicMock

from deps import Dependencies
from tools.analytics import register


def test_register_adds_four_analytics_tools():
    """register() should expose exactly the four documented analytics tools."""
    mcp = MagicMock()
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    register(mcp, deps)

    tool_names = {call.args[0].__name__ for call in mcp.tool.return_value.call_args_list}
    assert tool_names == {
        "analyze_bulletin_trends",
        "get_regulatory_digest",
        "compare_bulletin_metrics",
        "check_bddk_updates",
    }
