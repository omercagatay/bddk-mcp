"""OpenAI-compatible schemas exported from the canonical MCP tool registry."""

from __future__ import annotations

from copy import deepcopy

from mcp.server.fastmcp import FastMCP

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.tools.registry import ToolProfile, assert_tool_profile, register_tool_profile


def build_tool_schemas(profile: ToolProfile = ToolProfile.OPERATOR) -> list[dict]:
    """Build benchmark schemas from the handlers used by the runtime server."""
    server = FastMCP("bddk-contract-export")
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    register_tool_profile(server, deps, profile)
    assert_tool_profile(server, profile)

    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": (tool.description or "").strip(),
                "parameters": deepcopy(tool.parameters),
            },
        }
        for tool in server._tool_manager.list_tools()
    ]


# The model-routing benchmark deliberately sees the full operator contract.
# Runs record this profile and should not be compared with a public-only host.
TOOL_SCHEMAS: list[dict] = build_tool_schemas()


def get_tool_names() -> list[str]:
    """Return tool names in the same order as TOOL_SCHEMAS."""
    return [schema["function"]["name"] for schema in TOOL_SCHEMAS]
