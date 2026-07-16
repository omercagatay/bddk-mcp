"""Project FastMCP boundary with stable, privacy-safe execution errors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import ContentBlock
from pydantic import ValidationError

from bddk_mcp.core.logging_config import correlation_scope

if TYPE_CHECKING:
    from bddk_mcp.corpus_serving import ActiveCorpusGuard


def _safe_tool_error(error: ToolError) -> str:
    """Collapse SDK exception chains without returning values or trace details."""
    current: BaseException | None = error
    while current is not None:
        if isinstance(current, ValidationError):
            return "[ERROR:INVALID_INPUT] retryable=false\nTool arguments do not satisfy the published input schema."
        if isinstance(current, ToolError):
            message = str(current)
            if message.startswith("[ERROR:"):
                return message
        current = current.__cause__ or current.__context__

    if str(error).startswith("Unknown tool:"):
        return "[ERROR:TOOL_NOT_FOUND] retryable=false\nThe requested tool is not registered in this profile."
    return (
        "[ERROR:TOOL_EXECUTION_FAILED] retryable=true\n"
        "The tool could not complete. Review server logs using the request correlation metadata."
    )


class BddkFastMCP(FastMCP):
    """FastMCP variant that never returns raw validation or exception text."""

    _bddk_active_corpus_guard: ActiveCorpusGuard | None = None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Sequence[ContentBlock] | dict[str, Any]:
        with correlation_scope():
            try:
                guard = self._bddk_active_corpus_guard
                if guard is None:
                    return await super().call_tool(name, arguments)
                async with guard.tool_call(name):
                    return await super().call_tool(name, arguments)
            except ToolError as error:
                raise ToolError(_safe_tool_error(error)) from None
