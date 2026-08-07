"""Tool-layer formatting for the regulatory graph tools."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bddk_mcp.tools import graph
from tests.test_graph_queries import _seed
from tests.test_regulatory_repository import regulatory_pool  # noqa: F401

pytestmark = pytest.mark.asyncio


def _register_and_capture(deps):
    """Register tools on a stub MCP and return {tool_name: coroutine_fn}."""
    captured = {}

    class _StubMCP:
        def tool(self, *args, **kwargs):
            def decorator(fn):
                captured[fn.__name__] = fn
                return fn

            return decorator

    graph.register(_StubMCP(), deps)
    return captured


def _deps_with_pool(pool):
    deps = MagicMock()
    deps.pool = pool
    return deps


async def test_amendment_chain_tool_formats_chain(regulatory_pool):  # noqa: F811
    await _seed(regulatory_pool)
    tools = _register_and_capture(_deps_with_pool(regulatory_pool))
    output = await tools["get_amendment_chain"](document_id="943")
    assert "ver-1" in output and "ver-2" in output
    assert "supersession" in output
    assert "ev-4" in output  # evidence id surfaced


async def test_amendment_chain_tool_no_coverage_message(regulatory_pool):  # noqa: F811
    tools = _register_and_capture(_deps_with_pool(regulatory_pool))
    output = await tools["get_amendment_chain"](document_id="unmapped-doc")
    assert "kapsam" in output.lower()  # explicit no-coverage marker, not empty text


async def test_cross_references_tool_flags_unvalidated(regulatory_pool):  # noqa: F811
    await _seed(regulatory_pool)
    tools = _register_and_capture(_deps_with_pool(regulatory_pool))
    validated_only = await tools["get_cross_references"](document_id="943")
    assert "cites" not in validated_only
    everything = await tools["get_cross_references"](document_id="943", include_unvalidated=True)
    assert "cites" in everything
    assert "doğrulanmamış" in everything.lower()
