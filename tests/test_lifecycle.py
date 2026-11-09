"""Integration tests for server lifecycle wiring."""

from bddk_mcp.core.deps import Dependencies


def test_deps_creation():
    """Dependencies can be created with None values for unit tests."""
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    assert deps.vector_store is None
    assert deps.pool is None


def test_server_module_exports_a_complete_mcp_object():
    """``mcp run server.py`` relies on the import-time server object."""
    from bddk_mcp import server
    from bddk_mcp.tools.registry import PUBLIC_TOOL_NAMES

    assert {tool.name for tool in server.mcp._tool_manager.list_tools()} == set(PUBLIC_TOOL_NAMES)
