"""Contract tests for canonical MCP tool profiles."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

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
    assert "check_bddk_updates" not in PUBLIC_TOOL_NAMES
    assert_tool_profile(server, ToolProfile.PUBLIC)


def test_operator_profile_is_public_plus_reviewed_operator_contract():
    server = _profile_server(ToolProfile.OPERATOR)

    assert set(registered_tool_names(server)) == set(expected_tool_names(ToolProfile.OPERATOR))
    assert len(OPERATOR_TOOL_NAMES) == 14
    assert "check_bddk_updates" in OPERATOR_TOOL_NAMES
    assert len(registered_tool_names(server)) == 29
    assert_tool_profile(server, ToolProfile.OPERATOR)


_EXPECTED_ANNOTATIONS = {
    "search_bddk_regulations": (True, False, True, False),
    "search_bddk_institutions": (True, False, True, True),
    "search_bddk_announcements": (True, False, True, True),
    "search_document_store": (True, False, True, False),
    "get_bddk_document": (True, False, True, False),
    "get_document_history": (True, False, True, False),
    "get_document_section": (True, False, True, False),
    "search_document_sections": (True, False, True, False),
    "resolve_regulation_status": (True, False, True, False),
    "get_bddk_bulletin": (True, False, True, True),
    "get_bddk_bulletin_snapshot": (True, False, True, True),
    "get_bddk_monthly": (True, False, True, True),
    "analyze_bulletin_trends": (True, False, True, True),
    "get_regulatory_digest": (True, False, True, True),
    "compare_bulletin_metrics": (True, False, True, True),
    "check_bddk_updates": (False, False, False, True),
    "document_store_stats": (True, False, True, False),
    "bddk_cache_status": (True, False, True, False),
    "refresh_bddk_cache": (False, True, False, True),
    "sync_bddk_documents": (False, True, False, True),
    "trigger_startup_sync": (False, True, False, True),
    "get_operator_job": (True, False, True, False),
    "list_operator_jobs": (True, False, True, False),
    "cancel_operator_job": (False, True, True, False),
    "document_health": (True, False, True, False),
    "health_check": (True, False, True, False),
    "bddk_metrics": (True, False, True, False),
    "backfill_degraded_documents": (False, True, False, True),
    "document_quality_report": (True, False, True, False),
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


def test_every_operator_tool_has_reviewed_exact_risk_annotations():
    server = _profile_server(ToolProfile.OPERATOR)

    actual = {tool.name: _annotation_tuple(tool) for tool in server._tool_manager.list_tools()}

    assert actual == _EXPECTED_ANNOTATIONS


@pytest.mark.parametrize("profile", [ToolProfile.PUBLIC, ToolProfile.OPERATOR])
def test_every_input_schema_forbids_unexpected_properties(profile):
    server = _profile_server(profile)

    for tool in server._tool_manager.list_tools():
        assert tool.parameters["additionalProperties"] is False, tool.name
        assert tool.fn_metadata.arg_model.model_config["extra"] == "forbid", tool.name


@pytest.mark.parametrize("profile", [ToolProfile.PUBLIC, ToolProfile.OPERATOR])
def test_every_input_parameter_has_a_client_visible_description(profile):
    server = _profile_server(profile)

    for tool in server._tool_manager.list_tools():
        for parameter_name, schema in tool.parameters.get("properties", {}).items():
            assert schema.get("description", "").strip(), f"{tool.name}.{parameter_name} has no description"


@pytest.mark.asyncio
async def test_official_client_rejects_extra_tool_argument():
    doc_store = MagicMock()
    doc_store.get_document_history = AsyncMock(return_value=[])
    deps = Dependencies(pool=None, doc_store=doc_store, client=MagicMock(), http=None)
    server = FastMCP("strict-contract-test")
    register_tool_profile(server, deps, ToolProfile.PUBLIC)

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
