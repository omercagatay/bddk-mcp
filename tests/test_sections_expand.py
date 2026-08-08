"""Opt-in one-hop reference expansion in section search.

The SQL behind one_hop_section_refs is exercised against live PostgreSQL in
test_graph_queries; here the engine call is stubbed so the tool contract —
off-by-default byte-identical output, labeled pointers, fail-open degradation —
is tested without building a full multi-document validated citation graph.
"""

from __future__ import annotations

import pytest

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.store.doc_store import StoredDocument
from bddk_mcp.tools import sections as sections_module

pytestmark = pytest.mark.asyncio


def _capture_tools(deps):
    captured = {}

    class _StubMCP:
        def tool(self, *args, **kwargs):
            def decorator(fn):
                captured[fn.__name__] = fn
                return fn

            return decorator

    sections_module.register(_StubMCP(), deps)
    return captured


@pytest.fixture
async def section_deps(regulatory_pool, doc_store_factory):
    store = await doc_store_factory(regulatory_pool)
    await store.store_document(
        StoredDocument(
            document_id="943",
            title="TFRS 9 Uygulama Rehberi",
            markdown_content=(
                "İlke 5 — Model validasyonu\n\n"
                "Model validasyonu bağımsız olarak yürütülür.\n\n"
                "İlke 6 — Model izleme\n\n"
                "Model performansı düzenli izlenir.\n"
            ),
        )
    )
    return Dependencies(pool=regulatory_pool, doc_store=store, client=None, http=None)


async def test_flag_off_output_is_byte_identical(section_deps, monkeypatch):
    tools = _capture_tools(section_deps)
    baseline = await tools["search_document_sections"](query="943 İlke 5 model validasyonu")

    async def _explode(*args, **kwargs):
        raise AssertionError("expansion must not run when the flag is off")

    monkeypatch.setattr(sections_module, "one_hop_section_refs", _explode)
    tools = _capture_tools(section_deps)
    unexpanded = await tools["search_document_sections"](query="943 İlke 5 model validasyonu")
    assert unexpanded.text == baseline.text


async def test_flag_on_appends_labeled_pointers(section_deps, monkeypatch):
    async def _neighbors(pool, *, doc_id, section_type, section_ref, limit):
        if (doc_id, section_type, section_ref) == ("943", "ilke", "5"):
            return [
                {
                    "doc_id": "mevzuat_22599",
                    "section_type": "madde",
                    "section_ref": "9",
                    "relation_type": "cites",
                }
            ]
        return []

    monkeypatch.setattr(sections_module, "one_hop_section_refs", _neighbors)
    tools = _capture_tools(section_deps)
    result = await tools["search_document_sections"](
        query="943 İlke 5 model validasyonu", expand_references=True
    )
    assert "İlişkili bölümler (doğrulanmış kenarlar) — 943 ilke 5" in result.text
    assert "mevzuat_22599 — madde 9 (kenar: cites)" in result.text
    # Pointers only — related section content is never inlined.
    assert "Model izleme" not in result.text.split("İlişkili bölümler", 1)[1]


async def test_expansion_failure_degrades_to_plain_search(section_deps, monkeypatch):
    async def _boom(*args, **kwargs):
        raise RuntimeError("synthetic expansion failure")

    monkeypatch.setattr(sections_module, "one_hop_section_refs", _boom)
    tools = _capture_tools(section_deps)
    result = await tools["search_document_sections"](
        query="943 İlke 5 model validasyonu", expand_references=True
    )
    assert "İlişkili bölümler" not in result.text
    assert "İlke 5" in result.text


async def test_expansion_with_no_neighbors_adds_nothing(section_deps, monkeypatch):
    async def _empty(*args, **kwargs):
        return []

    monkeypatch.setattr(sections_module, "one_hop_section_refs", _empty)
    tools = _capture_tools(section_deps)
    expanded = await tools["search_document_sections"](
        query="943 İlke 5 model validasyonu", expand_references=True
    )
    assert "İlişkili bölümler" not in expanded.text
