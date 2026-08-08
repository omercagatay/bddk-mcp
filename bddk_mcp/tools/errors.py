"""Structured error formatting for MCP tool responses.

Error returns start with a machine-parseable line — `[ERROR:<code>] retryable=<bool>` —
followed by a human-readable message, so the calling LLM can distinguish bad input
(fix the arguments) from transient upstream failures (retrying may help).
"""

from typing import Never

from mcp.server.fastmcp.exceptions import ToolError

INVALID_INPUT = "INVALID_INPUT"
NOT_FOUND = "NOT_FOUND"
UPSTREAM_FETCH_FAILED = "UPSTREAM_FETCH_FAILED"


def tool_error(code: str, message: str, *, retryable: bool, hint: str = "") -> Never:
    """Raise a stable MCP execution error rather than a successful text result."""
    lines = [f"[ERROR:{code}] retryable={str(retryable).lower()}", message]
    if hint:
        lines.append(f"Hint: {hint}")
    raise ToolError("\n".join(lines))
