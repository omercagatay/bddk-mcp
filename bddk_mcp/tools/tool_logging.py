"""Helpers for MCP tool-boundary diagnostic logging."""

from __future__ import annotations

import functools
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar

_MAX_TEXT_CHARS = 200
_SENSITIVE_KEY_EXACT = {"database_url", "dsn"}
_SENSITIVE_KEY_PARTS = (
    "password",
    "secret",
    "token",
    "credential",
    "api_key",
    "access_key",
    "private_key",
    "secret_key",
)

F = TypeVar("F", bound=Callable[..., Awaitable[str]])


def _truncate(value: str, max_chars: int = _MAX_TEXT_CHARS) -> str:
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "..."


def _is_sensitive_key(key: str) -> bool:
    key_lower = key.lower()
    return key_lower in _SENSITIVE_KEY_EXACT or any(part in key_lower for part in _SENSITIVE_KEY_PARTS)


def _summarize_value(value: Any) -> Any:
    if isinstance(value, str):
        return _truncate(value)
    if isinstance(value, int | float | bool) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): "<redacted>" if _is_sensitive_key(str(k)) else _summarize_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        items = list(value)
        summarized = [_summarize_value(item) for item in items[:10]]
        if len(items) > 10:
            summarized.append(f"... {len(items) - 10} more")
        return summarized
    return _truncate(repr(value))


def _summarize_args(args: Mapping[str, Any]) -> dict[str, Any]:
    """Return log-safe argument values."""
    return {
        str(key): "<redacted>" if _is_sensitive_key(str(key)) else _summarize_value(value)
        for key, value in args.items()
    }


def _summarize_result(result: Any) -> dict[str, Any]:
    """Return compact result metadata without dumping full tool output."""
    if isinstance(result, str):
        return {
            "result_type": "str",
            "result_size": len(result),
            "result_preview_chars": min(len(result), _MAX_TEXT_CHARS),
            "result_preview": _truncate(result.replace("\n", "\\n")),
        }
    if isinstance(result, bytes | bytearray):
        return {
            "result_type": type(result).__name__,
            "result_size": len(result),
            "result_preview_chars": 0,
            "result_preview": "",
        }
    preview = _truncate(repr(result).replace("\n", "\\n"))
    return {
        "result_type": type(result).__name__,
        "result_size": len(preview),
        "result_preview_chars": min(len(preview), _MAX_TEXT_CHARS),
        "result_preview": preview,
    }


def _bound_arguments(func: Callable[..., Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
    signature = inspect.signature(func)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return dict(bound.arguments)


def logged_tool(logger: logging.Logger) -> Callable[[F], F]:
    """Decorate an async MCP tool to log start, success, and failure boundaries."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> str:
            started = time.perf_counter()
            tool_args = _summarize_args(_bound_arguments(func, *args, **kwargs))
            logger.info(
                "MCP tool call started",
                extra={
                    "operation": "mcp_tool_call",
                    "tool_name": func.__name__,
                    "tool_args": tool_args,
                },
            )
            try:
                result = await func(*args, **kwargs)
            except Exception as exc:
                duration_ms = round((time.perf_counter() - started) * 1000, 2)
                logger.exception(
                    "MCP tool call failed",
                    extra={
                        "operation": "mcp_tool_call",
                        "tool_name": func.__name__,
                        "tool_args": tool_args,
                        "duration_ms": duration_ms,
                        "error_type": type(exc).__name__,
                        "error_message": _truncate(str(exc)),
                    },
                )
                raise

            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info(
                "MCP tool call completed",
                extra={
                    "operation": "mcp_tool_call",
                    "tool_name": func.__name__,
                    "tool_args": tool_args,
                    "duration_ms": duration_ms,
                    **_summarize_result(result),
                },
            )
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
