"""Helpers for MCP tool-boundary diagnostic logging."""

from __future__ import annotations

import functools
import inspect
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeVar

from bddk_mcp.core.logging_config import tool_content_logging_enabled

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


def _summarize_value(value: Any, *, include_content: bool = False) -> Any:
    if isinstance(value, str):
        if include_content:
            return _truncate(value)
        return {"value_type": "str", "char_count": len(value)}
    if isinstance(value, bytes | bytearray):
        return {"value_type": type(value).__name__, "byte_count": len(value)}
    if isinstance(value, int | float | bool) or value is None:
        return value
    if isinstance(value, Mapping):
        if not include_content:
            return {"value_type": type(value).__name__, "item_count": len(value)}
        return {
            str(k): "<redacted>" if _is_sensitive_key(str(k)) else _summarize_value(v, include_content=include_content)
            for k, v in value.items()
        }
    if isinstance(value, list | tuple | set):
        if not include_content:
            return {"value_type": type(value).__name__, "item_count": len(value)}
        items = list(value)
        summarized = [_summarize_value(item, include_content=True) for item in items[:10]]
        if len(items) > 10:
            summarized.append(f"... {len(items) - 10} more")
        return summarized
    if include_content:
        return _truncate(repr(value))
    return {"value_type": type(value).__name__}


def _summarize_args(args: Mapping[str, Any], *, include_content: bool = False) -> dict[str, Any]:
    """Return argument metadata without user-provided text by default."""
    return {
        str(key): "<redacted>"
        if _is_sensitive_key(str(key))
        else _summarize_value(value, include_content=include_content)
        for key, value in args.items()
    }


def _summarize_result(result: Any, *, include_content: bool = False) -> dict[str, Any]:
    """Return compact result metadata, with previews only by explicit opt-in."""
    if isinstance(result, str):
        summary: dict[str, Any] = {
            "result_type": "str",
            "result_size": len(result),
        }
        if include_content:
            summary.update(
                {
                    "result_preview_chars": min(len(result), _MAX_TEXT_CHARS),
                    "result_preview": _truncate(result.replace("\n", "\\n")),
                }
            )
        return summary
    if isinstance(result, bytes | bytearray):
        return {
            "result_type": type(result).__name__,
            "result_size": len(result),
        }
    summary = {
        "result_type": type(result).__name__,
    }
    try:
        summary["result_size"] = len(result)
    except TypeError:
        pass
    if include_content:
        preview = _truncate(repr(result).replace("\n", "\\n"))
        summary.update(
            {
                "result_preview_chars": min(len(preview), _MAX_TEXT_CHARS),
                "result_preview": preview,
            }
        )
    return summary


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
            bound_arguments = _bound_arguments(func, *args, **kwargs)
            include_content = tool_content_logging_enabled()
            tool_args = _summarize_args(bound_arguments, include_content=include_content)
            logger.info(
                "MCP tool call started",
                extra={
                    "operation": "mcp_tool_call",
                    "tool_name": func.__name__,
                    "tool_status": "started",
                    "argument_count": len(bound_arguments),
                    "tool_args": tool_args,
                },
            )
            try:
                result = await func(*args, **kwargs)
            except Exception as exc:
                duration_ms = round((time.perf_counter() - started) * 1000, 2)
                failure_metadata = {
                    "operation": "mcp_tool_call",
                    "tool_name": func.__name__,
                    "tool_status": "error",
                    "argument_count": len(bound_arguments),
                    "tool_args": tool_args,
                    "duration_ms": duration_ms,
                    "error_type": type(exc).__name__,
                }
                if include_content:
                    failure_metadata["error_message"] = _truncate(str(exc))
                    logger.exception("MCP tool call failed", extra=failure_metadata)
                else:
                    # Exception messages and formatted tracebacks can contain the
                    # caller's query or retrieved content. Keep production logs
                    # metadata-only unless the operator explicitly opts in.
                    logger.error("MCP tool call failed", extra=failure_metadata)
                raise

            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info(
                "MCP tool call completed",
                extra={
                    "operation": "mcp_tool_call",
                    "tool_name": func.__name__,
                    "tool_status": "success",
                    "argument_count": len(bound_arguments),
                    "tool_args": tool_args,
                    "duration_ms": duration_ms,
                    **_summarize_result(result, include_content=include_content),
                },
            )
            return result

        return wrapper  # type: ignore[return-value]

    return decorator
