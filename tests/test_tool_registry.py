"""Contract tests for canonical MCP tool profiles."""

from mcp.server.fastmcp import FastMCP

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.tools.registry import (
    OPERATOR_TOOL_NAMES,
    PUBLIC_TOOL_NAMES,
    ToolProfile,
    assert_tool_profile,
    expected_tool_names,
    register_tool_profile,
    registered_tool_names,
)


def _profile_server(profile: ToolProfile) -> FastMCP:
    server = FastMCP("contract-test")
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    register_tool_profile(server, deps, profile)
    return server


def test_public_profile_matches_reviewed_runtime_contract():
    server = _profile_server(ToolProfile.PUBLIC)

    assert set(registered_tool_names(server)) == set(PUBLIC_TOOL_NAMES)
    assert len(PUBLIC_TOOL_NAMES) == 15
    assert_tool_profile(server, ToolProfile.PUBLIC)


def test_operator_profile_is_public_plus_reviewed_operator_contract():
    server = _profile_server(ToolProfile.OPERATOR)

    assert set(registered_tool_names(server)) == set(expected_tool_names(ToolProfile.OPERATOR))
    assert len(OPERATOR_TOOL_NAMES) == 11
    assert len(registered_tool_names(server)) == 26
    assert_tool_profile(server, ToolProfile.OPERATOR)
