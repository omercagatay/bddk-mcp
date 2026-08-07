"""Tool-layer formatting for the regulatory graph tools."""

from __future__ import annotations

from unittest.mock import MagicMock

import asyncpg
import pytest

from bddk_mcp.regulatory.bridge import ensure_section_provision_map, refresh_section_provision_map
from bddk_mcp.store.doc_store import StoredDocument
from bddk_mcp.tools import graph
from tests.test_graph_queries import _seed, _seed_contested_amends_edges, _seed_incoming_edge
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


async def test_amendment_chain_tool_validated_only_and_never_rejected(regulatory_pool):  # noqa: F811
    await _seed(regulatory_pool)
    await _seed_contested_amends_edges(regulatory_pool)
    tools = _register_and_capture(_deps_with_pool(regulatory_pool))
    default = await tools["get_amendment_chain"](document_id="943")
    assert "ev-edge-1" in default  # human-validated edge kept
    assert "ev-edge-machine" not in default
    assert "ev-edge-rejected" not in default
    everything = await tools["get_amendment_chain"](document_id="943", include_unvalidated=True)
    assert "ev-edge-machine" in everything
    assert "doğrulanmamış" in everything.lower()  # unvalidated edges flagged in output
    assert "ev-edge-rejected" not in everything  # rejected never surfaces


async def test_tools_return_no_coverage_when_schema_missing():
    """Deployments without the regulatory schema get the explicit marker, not a traceback."""

    class _SchemalessPool:
        async def fetch(self, *args, **kwargs):
            raise asyncpg.exceptions.UndefinedTableError('relation "regulatory_relations" does not exist')

        async def fetchval(self, *args, **kwargs):
            raise asyncpg.exceptions.UndefinedTableError('relation "regulatory_legal_versions" does not exist')

        async def execute(self, *args, **kwargs):
            return "SELECT 0"

    tools = _register_and_capture(_deps_with_pool(_SchemalessPool()))
    chain_out = await tools["get_amendment_chain"](document_id="943")
    xref_out = await tools["get_cross_references"](document_id="943")
    assert "kapsamı bulunmuyor" in chain_out
    assert "kapsamı bulunmuyor" in xref_out


async def test_cross_references_tool_normalizes_section_args(regulatory_pool, doc_store_factory):  # noqa: F811
    store = await doc_store_factory(regulatory_pool)
    await store.store_document(
        StoredDocument(
            document_id="943",
            title="TFRS 9 Uygulama Rehberi",
            category="Rehber",
            source_url="https://www.bddk.org.tr/example/943.pdf",
            markdown_content=(
                "# TFRS 9 Uygulama Rehberi\n\n"
                "İlke 5 — Model validasyonu\n"
                "Banka, beklenen kredi zararı modellerini bağımsız olarak valide eder.\n"
            ),
            extraction_method="markitdown",
        )
    )
    await _seed(regulatory_pool)
    await ensure_section_provision_map(regulatory_pool)
    await refresh_section_provision_map(regulatory_pool)
    tools = _register_and_capture(_deps_with_pool(regulatory_pool))
    lower = await tools["get_cross_references"](document_id="943", section_type="ilke", section_ref="5")
    assert "amends" in lower  # load-bearing: the section filter actually resolved
    mixed = await tools["get_cross_references"](document_id="943", section_type="İlke", section_ref="5")
    assert mixed == lower


async def test_cross_references_tool_distinguishes_filtered_from_no_coverage(regulatory_pool):  # noqa: F811
    await _seed(regulatory_pool)
    tools = _register_and_capture(_deps_with_pool(regulatory_pool))
    # Mapped doc, but the incoming-direction filter excludes every (outgoing) edge.
    filtered = await tools["get_cross_references"](document_id="943", direction="incoming")
    assert "eşleşen çapraz referans kenarı bulunamadı" in filtered
    assert "bağlanmamış" not in filtered  # must NOT claim the doc is unmapped
    # Genuinely unmapped doc keeps the explicit no-coverage marker.
    no_coverage = await tools["get_cross_references"](document_id="unmapped-doc")
    assert "bağlanmamış" in no_coverage


async def test_cross_references_tool_renders_incoming_direction(regulatory_pool):  # noqa: F811
    await _seed(regulatory_pool)
    await _seed_incoming_edge(regulatory_pool)
    tools = _register_and_capture(_deps_with_pool(regulatory_pool))
    out = await tools["get_cross_references"](document_id="943", direction="incoming")
    assert "`amends` ← kaynak: inst-other" in out
    assert "→" not in out  # every rendered edge is incoming; no inverted arrows
