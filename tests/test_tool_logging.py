"""Tests for MCP tool-boundary logging helpers."""

from __future__ import annotations

import logging

import pytest

from bddk_mcp.tools.tool_logging import _summarize_args, _summarize_result, logged_tool


def test_summarize_args_redacts_sensitive_values_and_truncates_long_text():
    args = _summarize_args(
        {
            "keywords": "Kredilerin Sınıflandırılması",
            "query": "kredi " * 80,
            "database_url": "postgresql://user:secret@example/db",
            "api_token": "secret-token",
            "private_key": "pem-data",
            "limit": 10,
        }
    )

    assert args["database_url"] == "<redacted>"
    assert args["api_token"] == "<redacted>"
    assert args["private_key"] == "<redacted>"
    assert args["keywords"] == "Kredilerin Sınıflandırılması"
    assert args["limit"] == 10
    assert args["query"].endswith("...")
    assert "kredi kredi" in args["query"]
    assert len(args["query"]) <= 203


def test_summarize_result_keeps_preview_short_and_reports_size():
    result = _summarize_result("MADDE 1 " + ("çok uzun metin " * 100))

    assert result["result_type"] == "str"
    assert result["result_size"] > result["result_preview_chars"]
    assert result["result_preview"].startswith("MADDE 1")
    assert result["result_preview"].endswith("...")
    assert len(result["result_preview"]) <= 203


@pytest.mark.asyncio
async def test_logged_tool_logs_success_with_args_duration_and_result_summary(caplog):
    logger = logging.getLogger("tests.tool_logging")

    @logged_tool(logger)
    async def sample_tool(query: str, limit: int = 5) -> str:
        return "one\n" + ("two " * 100)

    with caplog.at_level(logging.INFO, logger="tests.tool_logging"):
        out = await sample_tool("teminat", limit=3)

    assert out.startswith("one")
    start = next(record for record in caplog.records if record.message == "MCP tool call started")
    success = next(record for record in caplog.records if record.message == "MCP tool call completed")

    assert start.tool_name == "sample_tool"
    assert start.tool_args == {"query": "teminat", "limit": 3}
    assert success.tool_name == "sample_tool"
    assert success.duration_ms >= 0
    assert success.result_type == "str"
    assert success.result_size == len(out)
    assert success.result_preview.startswith("one")


@pytest.mark.asyncio
async def test_logged_tool_logs_failure_with_exception_metadata(caplog):
    logger = logging.getLogger("tests.tool_logging")

    @logged_tool(logger)
    async def failing_tool(document_id: str) -> str:
        raise RuntimeError(f"boom for {document_id}")

    with caplog.at_level(logging.INFO, logger="tests.tool_logging"):
        with pytest.raises(RuntimeError, match="boom"):
            await failing_tool("mevzuat_1")

    failure = next(record for record in caplog.records if record.message == "MCP tool call failed")
    assert failure.tool_name == "failing_tool"
    assert failure.tool_args == {"document_id": "mevzuat_1"}
    assert failure.error_type == "RuntimeError"
    assert failure.error_message == "boom for mevzuat_1"
    assert failure.duration_ms >= 0
