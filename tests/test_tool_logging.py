"""Tests for privacy-safe MCP tool-boundary logging helpers."""

from __future__ import annotations

import logging

import pytest

import bddk_mcp.tools.tool_logging as tool_logging_module
from bddk_mcp.core.logging_config import JsonFormatter
from bddk_mcp.observability.metrics import Metrics
from bddk_mcp.tools.tool_logging import _summarize_args, _summarize_result, logged_tool

QUERY_SENTINEL = "PRIVATE_QUERY_SENTINEL_7f9d"
RESULT_SENTINEL = "PRIVATE_RESULT_SENTINEL_31ab"
ERROR_SENTINEL = "PRIVATE_ERROR_SENTINEL_920c"


def _render_records(records: list[logging.LogRecord]) -> str:
    formatter = JsonFormatter()
    return "\n".join(formatter.format(record) for record in records)


def test_summarize_args_keeps_metadata_and_redacts_all_text_by_default():
    args = _summarize_args(
        {
            "keywords": QUERY_SENTINEL,
            "query": "kredi " * 80,
            "content": RESULT_SENTINEL,
            "database_url": "postgresql://user:secret@example/db",
            "api_token": "secret-token",
            "document_ids": ["mevzuat_1", "mevzuat_2"],
            "filters": {QUERY_SENTINEL: RESULT_SENTINEL},
            "limit": 10,
            "active_only": True,
        }
    )

    assert args["database_url"] == "<redacted>"
    assert args["api_token"] == "<redacted>"
    assert args["keywords"] == {"value_type": "str", "char_count": len(QUERY_SENTINEL)}
    assert args["query"] == {"value_type": "str", "char_count": len("kredi " * 80)}
    assert args["content"] == {"value_type": "str", "char_count": len(RESULT_SENTINEL)}
    assert args["document_ids"] == {"value_type": "list", "item_count": 2}
    assert args["filters"] == {"value_type": "dict", "item_count": 1}
    assert args["limit"] == 10
    assert args["active_only"] is True
    assert QUERY_SENTINEL not in repr(args)
    assert RESULT_SENTINEL not in repr(args)


def test_summarize_result_reports_size_without_a_preview_by_default():
    output = f"MADDE 1 {RESULT_SENTINEL} " + ("çok uzun metin " * 100)
    result = _summarize_result(output)

    assert result == {"result_type": "str", "result_size": len(output)}
    assert RESULT_SENTINEL not in repr(result)


def test_summaries_expose_truncated_content_only_when_explicitly_requested():
    args = _summarize_args(
        {"query": QUERY_SENTINEL, "api_token": "must-stay-secret"},
        include_content=True,
    )
    result = _summarize_result(f"{RESULT_SENTINEL} " + ("x" * 400), include_content=True)

    assert args["query"] == QUERY_SENTINEL
    assert args["api_token"] == "<redacted>"
    assert result["result_preview"].startswith(RESULT_SENTINEL)
    assert result["result_preview"].endswith("...")
    assert len(result["result_preview"]) <= 203


@pytest.mark.asyncio
async def test_logged_tool_success_is_metadata_only_by_default(caplog, monkeypatch):
    monkeypatch.delenv("BDDK_TOOL_LOG_CONTENT", raising=False)
    collector = Metrics()
    monkeypatch.setattr(tool_logging_module, "metrics", collector)
    logger = logging.getLogger("tests.tool_logging.success")

    @logged_tool(logger)
    async def search_document_store(query: str, limit: int = 5) -> str:
        return f"MADDE 1 {RESULT_SENTINEL}\n" + ("regulatory text " * 20)

    with caplog.at_level(logging.INFO, logger=logger.name):
        output = await search_document_store(QUERY_SENTINEL, limit=3)

    start = next(record for record in caplog.records if record.message == "MCP tool call started")
    success = next(record for record in caplog.records if record.message == "MCP tool call completed")

    assert RESULT_SENTINEL in output
    assert start.tool_name == "search_document_store"
    assert start.tool_status == "started"
    assert start.argument_count == 2
    assert start.tool_args == {
        "query": {"value_type": "str", "char_count": len(QUERY_SENTINEL)},
        "limit": 3,
    }
    assert success.tool_status == "success"
    assert success.duration_ms >= 0
    assert success.result_type == "str"
    assert success.result_size == len(output)
    assert not hasattr(success, "result_preview")

    rendered = _render_records(caplog.records)
    assert QUERY_SENTINEL not in rendered
    assert RESULT_SENTINEL not in rendered
    assert collector.summary()["tools"][0]["requests"] == 1
    assert collector.summary()["tools"][0]["errors"] == 0


@pytest.mark.asyncio
async def test_logged_tool_failure_omits_arguments_message_and_traceback_by_default(caplog, monkeypatch):
    monkeypatch.delenv("BDDK_TOOL_LOG_CONTENT", raising=False)
    collector = Metrics()
    monkeypatch.setattr(tool_logging_module, "metrics", collector)
    logger = logging.getLogger("tests.tool_logging.failure")

    @logged_tool(logger)
    async def search_document_sections(query: str) -> str:
        raise RuntimeError(
            f"{ERROR_SENTINEL}: failed while handling {query}; "
            "dsn=postgresql://private:password@db/bddk; "
            "token=PRIVATE_TOKEN; path=/bank/private/audit.txt; "
            "sql=SELECT private_column FROM private_table"
        )

    with caplog.at_level(logging.INFO, logger=logger.name):
        with pytest.raises(RuntimeError, match=ERROR_SENTINEL):
            await search_document_sections(QUERY_SENTINEL)

    failure = next(record for record in caplog.records if record.message == "MCP tool call failed")
    assert failure.tool_name == "search_document_sections"
    assert failure.tool_status == "error"
    assert failure.argument_count == 1
    assert failure.tool_args == {
        "query": {"value_type": "str", "char_count": len(QUERY_SENTINEL)},
    }
    assert failure.error_type == "RuntimeError"
    assert failure.duration_ms >= 0
    assert not hasattr(failure, "error_message")
    assert failure.exc_info is None

    rendered = _render_records(caplog.records)
    for secret in (
        QUERY_SENTINEL,
        ERROR_SENTINEL,
        "postgresql://private",
        "PRIVATE_TOKEN",
        "/bank/private/audit.txt",
        "private_column",
    ):
        assert secret not in rendered
    assert collector.summary()["tools"][0]["requests"] == 1
    assert collector.summary()["tools"][0]["errors"] == 1


@pytest.mark.asyncio
async def test_logged_tool_content_preview_requires_explicit_environment_opt_in(caplog, monkeypatch):
    monkeypatch.setenv("BDDK_TOOL_LOG_CONTENT", "true")
    logger = logging.getLogger("tests.tool_logging.opt_in")

    @logged_tool(logger)
    async def sample_tool(query: str) -> str:
        return RESULT_SENTINEL

    with caplog.at_level(logging.INFO, logger=logger.name):
        await sample_tool(QUERY_SENTINEL)

    start = next(record for record in caplog.records if record.message == "MCP tool call started")
    success = next(record for record in caplog.records if record.message == "MCP tool call completed")
    assert start.tool_args["query"] == QUERY_SENTINEL
    assert success.result_preview == RESULT_SENTINEL
