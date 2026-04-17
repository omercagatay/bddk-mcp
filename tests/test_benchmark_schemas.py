"""Tests for benchmark tool schemas."""

from benchmark.tool_schemas import TOOL_SCHEMAS, get_tool_names


def test_schema_count():
    """We have 21 tools in bddk-mcp."""
    assert len(TOOL_SCHEMAS) == 21


def test_each_schema_has_required_fields():
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert "name" in fn
        assert "description" in fn
        assert "parameters" in fn
        assert fn["parameters"]["type"] == "object"


def test_get_tool_names():
    names = get_tool_names()
    assert "search_bddk_decisions" in names
    assert "get_bddk_bulletin" in names
    assert "health_check" in names
    assert len(names) == 21


def test_search_bddk_decisions_schema():
    schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "search_bddk_decisions")
    params = schema["function"]["parameters"]["properties"]
    assert "keywords" in params
    assert "category" in params
    assert "page" in params


def test_no_duplicate_names():
    names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    assert len(names) == len(set(names))
