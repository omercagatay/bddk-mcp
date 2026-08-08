"""MCP-boundary tests for bounded, descriptive public-tool arguments."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.tools import analytics, bulletin, legal_status, search, sections
from bddk_mcp.tools.contract_types import MAX_METRIC_IDS, MAX_QUERY_LENGTH

_PARAMETERS = {
    "search_bddk_regulations": {"keywords", "page", "page_size", "category", "date_from", "date_to"},
    "search_bddk_institutions": {"keywords", "institution_type", "active_only"},
    "search_bddk_announcements": {"keywords", "category"},
    "search_document_store": {"query", "category", "limit"},
    "get_document_section": {"document_id", "section_type", "section_ref", "heading"},
    "search_document_sections": {"query", "document_id", "section_type", "limit"},
    "resolve_regulation_status": {"instrument_id", "as_of"},
    "get_bddk_bulletin": {"metric_id", "currency", "column", "date", "days"},
    "get_bddk_bulletin_snapshot": set(),
    "get_bddk_monthly": {"table_no", "year", "month", "currency", "party_code"},
    "analyze_bulletin_trends": {"metric_id", "currency", "column", "lookback_weeks"},
    "get_regulatory_digest": {"period"},
    "compare_bulletin_metrics": {"metric_ids", "currency", "column", "days"},
}


def _server(*, include_operator: bool = False) -> tuple[FastMCP, Dependencies]:
    vector_store = MagicMock()
    vector_store.search = AsyncMock(return_value=[])
    client = MagicMock()
    doc_store = MagicMock()
    server = FastMCP("public-contract-tests")
    deps = Dependencies(
        pool=None,
        doc_store=doc_store,
        client=client,
        http=MagicMock(),
        vector_store=vector_store,
    )
    search.register(server, deps)
    sections.register(server, deps)
    legal_status.register(server, deps)
    bulletin.register(server, deps, include_operator=include_operator)
    analytics.register(server, deps, include_operator=include_operator)
    return server, deps


@pytest.mark.asyncio
async def test_stateful_update_monitor_schema_is_operator_only():
    public_server, _ = _server()
    operator_server, _ = _server(include_operator=True)

    async with create_connected_server_and_client_session(public_server) as session:
        public_names = {tool.name for tool in (await session.list_tools()).tools}
    async with create_connected_server_and_client_session(operator_server) as session:
        operator_schemas = {tool.name: tool.inputSchema for tool in (await session.list_tools()).tools}

    assert "check_bddk_updates" not in public_names
    assert operator_schemas["check_bddk_updates"]["properties"] == {}


def _nonnull(schema: dict) -> dict:
    return next((item for item in schema.get("anyOf", []) if item.get("type") != "null"), schema)


@pytest.mark.asyncio
async def test_tools_list_describes_every_public_parameter_and_important_bounds():
    server, _ = _server()

    async with create_connected_server_and_client_session(server) as session:
        listed = await session.list_tools()

    schemas = {tool.name: tool.inputSchema for tool in listed.tools}
    assert set(schemas) == set(_PARAMETERS)
    for tool_name, expected_parameters in _PARAMETERS.items():
        properties = schemas[tool_name]["properties"]
        assert set(properties) == expected_parameters
        for parameter, schema in properties.items():
            assert schema.get("description", "").strip(), f"{tool_name}.{parameter} has no description"

    regulations = schemas["search_bddk_regulations"]["properties"]
    assert regulations["keywords"]["maxLength"] == MAX_QUERY_LENGTH
    assert regulations["page"]["minimum"] == 1
    assert regulations["page_size"]["maximum"] == 50
    assert "Rehber" in _nonnull(regulations["category"])["enum"]
    assert _nonnull(regulations["date_from"])["pattern"] == r"^\d{2}\.\d{2}\.\d{4}$"
    assert regulations["date_from"]["format"] == "date-dd-mm-yyyy"

    institutions = schemas["search_bddk_institutions"]["properties"]
    assert "Banka" in _nonnull(institutions["institution_type"])["enum"]
    assert institutions["active_only"]["type"] == "boolean"

    section = schemas["get_document_section"]["properties"]
    assert section["document_id"]["pattern"] == r"^[A-Za-z0-9_-]+$"
    assert "gecici_madde" in _nonnull(section["section_type"])["enum"]
    assert any(item.get("pattern") for item in section["section_ref"]["anyOf"])

    legal_status_schema = schemas["resolve_regulation_status"]["properties"]
    assert legal_status_schema["instrument_id"]["pattern"] == r"^inst_sha256_[0-9a-f]{64}$"
    assert legal_status_schema["as_of"]["pattern"] == r"^\d{4}-\d{2}-\d{2}$"
    assert legal_status_schema["as_of"]["format"] == "date"

    weekly = schemas["get_bddk_bulletin"]["properties"]
    assert weekly["metric_id"]["pattern"] == r"^\d+\.\d+\.\d+$"
    assert weekly["currency"]["enum"] == ["TRY", "USD"]
    assert weekly["column"]["enum"] == ["1", "2", "3"]
    assert weekly["date"]["format"] == "date-dd-mm-yyyy-or-empty"
    assert weekly["days"]["minimum"] == 1
    assert weekly["days"]["maximum"] == 3650

    monthly = schemas["get_bddk_monthly"]["properties"]
    assert monthly["table_no"]["maximum"] == 17
    assert monthly["month"]["maximum"] == 12
    assert monthly["currency"]["enum"] == ["TL", "USD"]
    assert monthly["party_code"]["enum"] == [
        "10001",
        "10002",
        "10003",
        "10004",
        "20001",
        "20002",
        "20003",
    ]

    compare = schemas["compare_bulletin_metrics"]["properties"]
    assert compare["metric_ids"]["pattern"].startswith(r"^\d+")
    assert schemas["get_regulatory_digest"]["properties"]["period"]["enum"] == [
        "day",
        "week",
        "month",
        "quarter",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments", "private_value"),
    [
        ("search_document_store", {"query": "PRIVATE_QUERY_" + "x" * MAX_QUERY_LENGTH}, "PRIVATE_QUERY_"),
        ("search_bddk_regulations", {"keywords": "kredi", "date_from": "31.02.2025"}, "31.02.2025"),
        (
            "search_bddk_regulations",
            {"keywords": "kredi", "date_from": "02.01.2025", "date_to": "01.01.2025"},
            "02.01.2025",
        ),
        ("search_bddk_announcements", {"category": "PRIVATE_CATEGORY"}, "PRIVATE_CATEGORY"),
        ("search_bddk_institutions", {"institution_type": "PRIVATE_TYPE"}, "PRIVATE_TYPE"),
        ("get_document_section", {"document_id": "../PRIVATE_PATH"}, "PRIVATE_PATH"),
        ("search_document_sections", {"query": "kredi", "section_type": "PRIVATE_SECTION"}, "PRIVATE_SECTION"),
        ("get_bddk_bulletin", {"days": 0}, None),
        ("get_bddk_monthly", {"party_code": "PRIVATE_PARTY"}, "PRIVATE_PARTY"),
        ("get_regulatory_digest", {"period": "PRIVATE_PERIOD"}, "PRIVATE_PERIOD"),
        ("compare_bulletin_metrics", {"metric_ids": "1.0.1,,1.0.2"}, None),
        (
            "compare_bulletin_metrics",
            {"metric_ids": ",".join("1.0.1" for _ in range(MAX_METRIC_IDS + 1))},
            None,
        ),
    ],
)
async def test_invalid_arguments_fail_with_stable_sanitized_error(tool_name, arguments, private_value):
    server, _ = _server()

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool(tool_name, arguments)

    text = result.content[0].text
    assert result.isError is True
    assert "[ERROR:INVALID_INPUT] retryable=false" in text
    if private_value:
        assert private_value not in text
    assert "pydantic.dev" not in text


@pytest.mark.asyncio
async def test_invalid_category_and_metric_list_never_reach_backends():
    server, _ = _server()
    announcement_fetch = AsyncMock()
    metric_compare = AsyncMock()

    with (
        patch("bddk_mcp.tools.search.fetch_announcements", new=announcement_fetch),
        patch("bddk_mcp.tools.analytics.compare_metrics", new=metric_compare),
    ):
        async with create_connected_server_and_client_session(server) as session:
            category_result = await session.call_tool("search_bddk_announcements", {"category": "unsupported"})
            metrics_result = await session.call_tool(
                "compare_bulletin_metrics",
                {"metric_ids": "1.0.1,,1.0.2"},
            )

    assert category_result.isError is True
    assert metrics_result.isError is True
    announcement_fetch.assert_not_awaited()
    metric_compare.assert_not_awaited()


@pytest.mark.asyncio
async def test_valid_boundary_values_are_normalized_and_reach_backends():
    server, deps = _server()
    weekly_fetch = AsyncMock(return_value={"title": "Test", "currency": "TRY", "dates": [], "values": []})
    metric_compare = AsyncMock(return_value={"metrics": []})

    with (
        patch("bddk_mcp.tools.bulletin.fetch_weekly_bulletin", new=weekly_fetch),
        patch("bddk_mcp.tools.analytics.compare_metrics", new=metric_compare),
    ):
        async with create_connected_server_and_client_session(server) as session:
            search_result = await session.call_tool(
                "search_document_store",
                {"query": "x" * MAX_QUERY_LENGTH, "category": "rehber", "limit": 50},
            )
            bulletin_result = await session.call_tool(
                "get_bddk_bulletin",
                {"metric_id": "1.0.1", "currency": "try", "column": "3", "date": "29.02.2024", "days": 1},
            )
            compare_result = await session.call_tool(
                "compare_bulletin_metrics",
                {"metric_ids": ",".join("1.0.1" for _ in range(MAX_METRIC_IDS)), "days": 1},
            )

    assert search_result.isError is False
    assert bulletin_result.isError is False
    assert compare_result.isError is False
    deps.vector_store.search.assert_awaited_once_with("x" * MAX_QUERY_LENGTH, limit=50, category="Rehber")
    weekly_fetch.assert_awaited_once_with(deps.http, "1.0.1", "TRY", 1, "29.02.2024", "3")
    metric_compare.assert_awaited_once()
    assert len(metric_compare.await_args.args[1]) == MAX_METRIC_IDS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "patch_target", "safe_fragment"),
    [
        ("get_bddk_bulletin", "bddk_mcp.tools.bulletin.fetch_weekly_bulletin", "weekly bulletin"),
        ("get_bddk_monthly", "bddk_mcp.tools.bulletin.fetch_monthly_bulletin", "monthly bulletin"),
        ("analyze_bulletin_trends", "bddk_mcp.tools.analytics.analyze_trends", "trend data"),
    ],
)
async def test_upstream_failure_details_are_not_exposed(tool_name, patch_target, safe_fragment):
    server, _ = _server()
    sentinel = "PRIVATE-UPSTREAM-RESPONSE-BODY"

    with patch(patch_target, new=AsyncMock(return_value={"error": sentinel})):
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(tool_name, {})

    text = result.content[0].text
    assert result.isError is True
    assert "[ERROR:UPSTREAM_FETCH_FAILED] retryable=true" in text
    assert safe_fragment in text
    assert sentinel not in text


@pytest.mark.asyncio
async def test_operator_cache_status_withholds_page_error_details():
    server, deps = _server(include_operator=True)
    sentinel = "PRIVATE-UPSTREAM-PAGE-ERROR"
    deps.client.cache_status.return_value = {
        "total_items": 10,
        "cache_valid": True,
        "cache_age_seconds": 5,
        "ttl_seconds": 60,
        "categories": {},
        "page_errors": {39: sentinel, 40: sentinel},
    }

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("bddk_cache_status", {})

    text = result.content[0].text
    assert result.isError is False
    assert "2 page(s) failed; details withheld." in text
    assert sentinel not in text
