"""Tool-shape tests for get_amendment_chain and get_cross_references."""

from __future__ import annotations

import pytest

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.regulatory.relations import import_relations
from bddk_mcp.tools import graph as graph_module
from tests.test_graph_queries import seed_family_for_doc
from tests.test_regulatory_relations import VALIDATED, external_relation

pytestmark = pytest.mark.asyncio


def _capture_tools(deps):
    captured = {}

    class _StubMCP:
        def tool(self, *args, **kwargs):
            def decorator(fn):
                captured[fn.__name__] = fn
                return fn

            return decorator

    graph_module.register(_StubMCP(), deps)
    return captured


def _deps(pool) -> Dependencies:
    return Dependencies(pool=pool, doc_store=None, client=None, http=None)


async def test_amendment_chain_tool_reports_no_coverage_for_unknown_doc(regulatory_pool):
    tools = _capture_tools(_deps(regulatory_pool))
    result = await tools["get_amendment_chain"](document_id="no-such-doc")
    assert "graf kapsamı bulunmuyor" in result.text


async def test_amendment_chain_tool_renders_validated_versions(regulatory_pool):
    await seed_family_for_doc(regulatory_pool, document_id="943")
    tools = _capture_tools(_deps(regulatory_pool))
    result = await tools["get_amendment_chain"](document_id="943")
    assert "synthetic-v1" in result.text
    assert "Yalnızca insan onaylı" in result.text
    # The unvalidated synthetic-v2 successor never renders.
    assert "synthetic-v2" not in result.text
    payload = result.structuredContent
    assert payload["status"] == "ok"
    assert [version["version_key"] for version in payload["versions"]] == ["synthetic-v1"]


async def test_cross_references_tool_renders_validated_edges(regulatory_pool):
    bundle = await seed_family_for_doc(regulatory_pool, document_id="943")
    edge = external_relation(bundle, statement="validated cite", validation=VALIDATED)
    await import_relations(regulatory_pool, [edge], imported_by="test-suite")
    tools = _capture_tools(_deps(regulatory_pool))
    result = await tools["get_cross_references"](document_id="943")
    assert "cites" in result.text
    assert "5411 sayılı Bankacılık Kanunu madde 93" in result.text
    payload = result.structuredContent
    assert payload["status"] == "ok"
    assert payload["edges"][0]["relation_type"] == "cites"
    assert payload["edges"][0]["direction"] == "outgoing"


async def test_cross_references_tool_distinguishes_filtered_from_uncovered(regulatory_pool):
    bundle = await seed_family_for_doc(regulatory_pool, document_id="943")
    # Only an unvalidated candidate exists: the doc is mapped, but no edge is
    # servable, and the message must say so instead of claiming no coverage.
    edge = external_relation(bundle, statement="machine cite")
    await import_relations(regulatory_pool, [edge], imported_by="test-suite")
    tools = _capture_tools(_deps(regulatory_pool))
    result = await tools["get_cross_references"](document_id="943")
    assert "graf kapsamında" in result.text
    assert "eşleşen doğrulanmış çapraz referans kenarı bulunamadı" in result.text

    uncovered = await tools["get_cross_references"](document_id="no-such-doc")
    assert "graf kapsamı bulunmuyor" in uncovered.text


async def test_cross_references_tool_normalizes_section_args(regulatory_pool):
    await seed_family_for_doc(regulatory_pool, document_id="943")
    tools = _capture_tools(_deps(regulatory_pool))
    # "İlke" normalizes to the stored lowercase form; the section still has no
    # validated provision citation, so the mapped-but-filtered message renders.
    result = await tools["get_cross_references"](document_id="943", section_type="ilke", section_ref="5")
    assert "eşleşen doğrulanmış çapraz referans kenarı bulunamadı" in result.text


async def test_graph_tools_survive_missing_pool():
    tools = _capture_tools(_deps(None))
    chain_result = await tools["get_amendment_chain"](document_id="943")
    xref_result = await tools["get_cross_references"](document_id="943")
    assert "graf kapsamı bulunmuyor" in chain_result.text
    assert "graf kapsamı bulunmuyor" in xref_result.text
