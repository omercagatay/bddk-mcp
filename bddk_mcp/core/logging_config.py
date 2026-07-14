"""Structured JSON logging configuration for BDDK MCP Server."""

import json
import logging
import os
import sys
import uuid
from contextvars import ContextVar

# Context variable for request-level correlation IDs
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")
_TOOL_CONTENT_LOG_ENV = "BDDK_TOOL_LOG_CONTENT"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def get_correlation_id() -> str:
    """Get the current correlation ID, or generate a new one."""
    cid = _correlation_id.get()
    if not cid:
        cid = uuid.uuid4().hex[:12]
        _correlation_id.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current context."""
    _correlation_id.set(cid)


def tool_content_logging_enabled() -> bool:
    """Return whether sensitive MCP tool previews were explicitly enabled."""
    return os.environ.get(_TOOL_CONTENT_LOG_ENV, "").strip().lower() in _TRUE_VALUES


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
            "doc_id",
            "query",
            "result_count",
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
    root.addHandler(handler)

    if tool_content_logging_enabled():
        root.warning(
            "BDDK_TOOL_LOG_CONTENT is enabled; MCP logs may contain confidential "
            "queries, document excerpts, and error details"
        )

    # Reduce noise from third-party libraries
    for noisy in ("httpx", "httpcore", "chromadb", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
