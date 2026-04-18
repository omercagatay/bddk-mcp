"""Tests for tools/documents.py — tool registration."""

from __future__ import annotations

from unittest.mock import MagicMock

from deps import Dependencies
from tools.documents import register


def test_register_adds_three_document_tools():
    """register() should expose exactly the three documented document tools."""
    mcp = MagicMock()
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    register(mcp, deps)

    tool_names = {call.args[0].__name__ for call in mcp.tool.return_value.call_args_list}
    assert tool_names == {
        "get_bddk_document",
        "get_document_history",
        "document_store_stats",
    }
