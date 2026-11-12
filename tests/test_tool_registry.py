"""Contract tests for the reviewed MCP tool surface."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.tools.registry import (
    PUBLIC_TOOL_NAMES,
    assert_tool_surface,
    register_tools,
    registered_tool_names,
)


def _server() -> FastMCP:
    server = FastMCP("contract-test")
    deps = Dependencies(db=None, doc_store=None, client=None, http=None)
    register_tools(server, deps)
    return server


def test_registered_surface_matches_reviewed_contract():
    server = _server()

    assert set(registered_tool_names(server)) == set(PUBLIC_TOOL_NAMES)
    assert len(PUBLIC_TOOL_NAMES) == 14
    assert_tool_surface(server)


_EXPECTED_ANNOTATIONS = {
    "search_bddk_regulations": (True, False, True, False),
    "search_bddk_institutions": (True, False, True, True),
    "search_bddk_announcements": (True, False, True, True),
    "search_document_store": (True, False, True, False),
    "get_bddk_document": (True, False, True, False),
    "get_document_history": (True, False, True, False),
    "get_document_section": (True, False, True, False),
    "search_document_sections": (True, False, True, False),
    "get_bddk_bulletin": (True, False, True, True),
    "get_bddk_bulletin_snapshot": (True, False, True, True),
    "get_bddk_monthly": (True, False, True, True),
    "analyze_bulletin_trends": (True, False, True, True),
    "get_regulatory_digest": (True, False, True, True),
    "compare_bulletin_metrics": (True, False, True, True),
}


def _annotation_tuple(tool) -> tuple[bool | None, bool | None, bool | None, bool | None]:
    annotations = tool.annotations
    assert annotations is not None
    return (
        annotations.readOnlyHint,
        annotations.destructiveHint,
        annotations.idempotentHint,
        annotations.openWorldHint,
    )


def test_every_tool_has_reviewed_exact_risk_annotations():
    server = _server()

    actual = {tool.name: _annotation_tuple(tool) for tool in server._tool_manager.list_tools()}

    assert actual == _EXPECTED_ANNOTATIONS


def test_every_input_schema_forbids_unexpected_properties():
    server = _server()

    for tool in server._tool_manager.list_tools():
        assert tool.parameters["additionalProperties"] is False, tool.name
        assert tool.fn_metadata.arg_model.model_config["extra"] == "forbid", tool.name


def test_every_input_parameter_has_a_client_visible_description():
    server = _server()

    for tool in server._tool_manager.list_tools():
        for parameter_name, schema in tool.parameters.get("properties", {}).items():
            assert schema.get("description", "").strip(), f"{tool.name}.{parameter_name} has no description"


@pytest.mark.asyncio
async def test_official_client_rejects_extra_tool_argument():
    doc_store = MagicMock()
    doc_store.get_document_history = AsyncMock(return_value=[])
    deps = Dependencies(db=None, doc_store=doc_store, client=MagicMock(), http=None)
    server = FastMCP("strict-contract-test")
    register_tools(server, deps)

    async with create_connected_server_and_client_session(server) as session:
        listed = await session.list_tools()
        history_tool = next(tool for tool in listed.tools if tool.name == "get_document_history")
        result = await session.call_tool(
            "get_document_history",
            {"document_id": "943", "unexpected": "must not be ignored"},
        )

    assert history_tool.inputSchema["additionalProperties"] is False
    assert result.isError is True
    assert "Extra inputs are not permitted" in result.content[0].text
    doc_store.get_document_history.assert_not_awaited()


def test_assert_tool_surface_raises_when_a_reviewed_tool_is_missing():
    server = _server()
    server.remove_tool("get_bddk_monthly")

    with pytest.raises(RuntimeError, match=r"drifted.*missing=\['get_bddk_monthly'\]"):
        assert_tool_surface(server)


def test_assert_tool_surface_raises_when_an_unreviewed_tool_is_registered():
    server = _server()

    @server.tool()
    def unreviewed_tool() -> str:
        return "x"

    with pytest.raises(RuntimeError, match=r"drifted.*unexpected=\['unreviewed_tool'\]"):
        assert_tool_surface(server)


@pytest.mark.parametrize("drift", ["missing", "unexpected"])
def test_building_the_server_fails_when_the_tool_surface_drifts(monkeypatch, drift):
    from bddk_mcp import server as server_module

    def drifted_registration(server, deps):
        register_tools(server, deps)
        if drift == "missing":
            server.remove_tool("get_bddk_monthly")
        else:
            server.tool(name="unreviewed_tool")(lambda: "x")

    monkeypatch.setattr(server_module, "register_tools", drifted_registration)
    deps = Dependencies(db=None, doc_store=None, client=None, http=None)

    with pytest.raises(RuntimeError, match="tool surface drifted"):
        server_module.create_mcp(deps)


def test_building_the_server_with_the_reviewed_surface_succeeds():
    from bddk_mcp import server as server_module

    deps = Dependencies(db=None, doc_store=None, client=None, http=None)

    assert set(registered_tool_names(server_module.create_mcp(deps))) == set(PUBLIC_TOOL_NAMES)
