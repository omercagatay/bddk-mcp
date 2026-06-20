"""Tests for tools/errors.py — structured tool error formatting."""

from bddk_mcp.tools.errors import INVALID_INPUT, NOT_FOUND, UPSTREAM_FETCH_FAILED, tool_error


def test_tool_error_first_line_is_machine_parseable():
    out = tool_error(INVALID_INPUT, "bad month", retryable=False)
    assert out.splitlines()[0] == "[ERROR:INVALID_INPUT] retryable=false"
    assert "bad month" in out


def test_tool_error_includes_hint_when_given():
    out = tool_error(UPSTREAM_FETCH_FAILED, "timeout", retryable=True, hint="retry later")
    assert "[ERROR:UPSTREAM_FETCH_FAILED] retryable=true" in out
    assert out.splitlines()[-1] == "Hint: retry later"


def test_tool_error_omits_hint_line_by_default():
    out = tool_error(NOT_FOUND, "missing", retryable=False)
    assert "Hint:" not in out
