"""Structured JSON logging configuration for BDDK MCP Server."""

import json
import logging
import os
import re
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

# Context variable for request-level correlation IDs
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
_TOOL_CONTENT_LOG_ENV = "BDDK_TOOL_LOG_CONTENT"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


@contextmanager
def correlation_scope(cid: str | None = None) -> Iterator[str]:
    """Set one bounded correlation ID and restore the parent context afterward."""

    selected = uuid.uuid4().hex[:12] if cid is None else cid
    if not isinstance(selected, str) or re.fullmatch(r"[A-Za-z0-9._:-]{1,64}", selected) is None:
        raise ValueError("correlation ID must be a 1-64 character safe identifier")
    token = _correlation_id.set(selected)
    try:
        yield selected
    finally:
        _correlation_id.reset(token)


def tool_content_logging_enabled() -> bool:
    """Return whether sensitive MCP tool previews were explicitly enabled."""
    return os.environ.get(_TOOL_CONTENT_LOG_ENV, "").strip().lower() in _TRUE_VALUES


# Class names the SDK logs as arguments (request, notification and warning types).
_SAFE_LOG_ARGUMENT_RE = re.compile(r"[A-Z][A-Za-z]{0,60}(?:Request|Notification|Warning|Error)")
_WITHHELD = "<withheld>"


class SdkLogPrivacyFilter(logging.Filter):
    """Keep request content and tracebacks of the MCP SDK out of the logs.

    The SDK logs outside this project's tool boundary: a request that fails its
    schema check is logged with the rejected values (``input_value=...``), an
    unknown tool with the name the caller sent, and transport errors with a full
    traceback. Those values are caller text (a query, a document excerpt) and
    the tracebacks carry absolute paths, so the metadata-only policy of
    ``tools/tool_logging.py`` would not hold for them. Unless
    ``BDDK_TOOL_LOG_CONTENT`` is enabled, a record of the SDK (logger ``mcp.*``,
    or the root logger its session layer writes to) keeps only its fixed message
    text and the exception type.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "root" and record.name != "mcp" and not record.name.startswith("mcp."):
            return True
        if tool_content_logging_enabled():
            return True

        if record.args:
            arguments = record.args if isinstance(record.args, tuple) else (record.args,)
            try:
                message = str(record.msg) % tuple(self._safe_argument(value) for value in arguments)
            except (TypeError, ValueError, KeyError):
                message = str(record.msg)
        else:
            # The SDK builds these messages with f-strings: the fixed text ends at the first colon.
            text = str(record.msg)
            head, separator, _detail = text.partition(":")
            message = f"{head}: {_WITHHELD}" if separator else text
        if record.exc_info and record.exc_info[0] is not None:
            message = f"{message} [{record.exc_info[0].__name__}]"

        record.msg = message[:200]
        record.args = None
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True

    @staticmethod
    def _safe_argument(value: object) -> object:
        if isinstance(value, bool | int | float) or value is None:
            return value
        if isinstance(value, str) and _SAFE_LOG_ARGUMENT_RE.fullmatch(value):
            return value
        return _WITHHELD


class JsonFormatter(logging.Formatter):
    """Structured JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": _correlation_id.get(""),
        }

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Include extra fields from record
        for key in (
            "operation",
            "duration_ms",
            "tool_name",
            "tool_status",
            "argument_count",
            "tool_args",
            "result_type",
            "result_size",
            "result_preview_chars",
            "result_preview",
            "error_type",
            "error_message",
            "attempt",
            "max_retries",
            "max_attempts",
            "http_method",
            "upstream_boundary",
            "status_code",
        ):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        return json.dumps(log_entry, ensure_ascii=False, default=str)


class HumanFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self.FORMAT, datefmt="%H:%M:%S")


def configure_logging(json_output: bool | None = None) -> None:
    """Configure logging for the application.

    Args:
        json_output: Force JSON (True) or human (False) output.
            If None, auto-detect: JSON in production (MCP_TRANSPORT=streamable-http),
            human-readable otherwise.
    """
    if json_output is None:
        json_output = os.environ.get("MCP_TRANSPORT") == "streamable-http"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Remove existing handlers
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_output else HumanFormatter())
    handler.addFilter(SdkLogPrivacyFilter())
    root.addHandler(handler)

    if tool_content_logging_enabled():
        root.warning(
            "BDDK_TOOL_LOG_CONTENT is enabled; MCP logs may contain confidential "
            "queries, document excerpts, and error details"
        )

    # Reduce noise from third-party libraries
    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
