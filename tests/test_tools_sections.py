"""Tests for tools/sections.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from deps import Dependencies
from doc_store import StoredDocumentSection
from tools.sections import register


def _capture_tool(deps: Dependencies, name: str):
    mcp = MagicMock()
    register(mcp, deps)
    for call in mcp.tool.return_value.call_args_list:
        fn = call.args[0]
        if fn.__name__ == name:
            return fn
    raise AssertionError(f"{name} not registered")


def _section(
    doc_id: str = "943",
    section_type: str = "ilke",
    section_ref: str = "5",
    content: str = "İlke 5\nBankalar model validasyonunu yapar.",
) -> StoredDocumentSection:
    return StoredDocumentSection(
        doc_id=doc_id,
        section_type=section_type,
        section_ref=section_ref,
        heading="Model validasyonu",
        start_char=10,
        end_char=80,
        content=content,
        content_hash="abc123",
    )


def test_register_exposes_section_tools():
    mcp = MagicMock()
    deps = Dependencies(pool=None, doc_store=MagicMock(), client=None, http=None)
    register(mcp, deps)

    tool_names = {call.args[0].__name__ for call in mcp.tool.return_value.call_args_list}
    assert tool_names == {"get_document_section", "search_document_sections"}


@pytest.mark.asyncio
async def test_get_document_section_returns_exact_match():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[_section()])
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "get_document_section")
    out = await tool("943", section_type="ilke", section_ref="5")

    assert "Document ID: 943" in out
    assert "Section: ilke 5" in out
    assert "Model validasyonu" in out
    assert "Bankalar model validasyonunu yapar." in out
    doc_store.get_document_section.assert_awaited_once_with("943", section_type="ilke", section_ref="5", heading=None)


@pytest.mark.asyncio
async def test_get_document_section_accepts_integer_section_ref():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[_section()])
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "get_document_section")
    out = await tool("943", section_type="ilke", section_ref=5)

    assert "Section: ilke 5" in out
    doc_store.get_document_section.assert_awaited_once_with("943", section_type="ilke", section_ref="5", heading=None)


@pytest.mark.asyncio
async def test_get_document_section_no_match_suggests_search():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[])
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "get_document_section")
    out = await tool("943", section_type="madde", section_ref="99")

    assert "No section found" in out
    assert "search_document_sections" in out
    assert "943 madde 99" in out


@pytest.mark.asyncio
async def test_get_document_section_integer_ref_no_match_suggests_search():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[])
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "get_document_section")
    out = await tool("943", section_type="madde", section_ref=99)

    assert "No section found" in out
    assert "943 madde 99" in out


@pytest.mark.asyncio
async def test_get_document_section_disambiguates_duplicate_matches():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(
        return_value=[
            _section(content="İlke 5\nBirinci eşleşme."),
            _section(content="İlke 5\nİkinci eşleşme."),
        ]
    )
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "get_document_section")
    out = await tool("943", section_type="ilke", section_ref="5")

    assert "Multiple sections matched" in out
    assert "start_char=10" in out
    assert "İkinci eşleşme" in out


@pytest.mark.asyncio
async def test_search_document_sections_outputs_ranked_sections():
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[])
    doc_store.search_document_sections = AsyncMock(
        return_value=[
            _section("943", "ilke", "5", "İlke 5\nModel validasyonu yapılır."),
            _section("mevzuat_22599", "madde", "9", "MADDE 9\nTFRS 9 karşılık ayrılır."),
        ]
    )
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "search_document_sections")
    out = await tool("943 İlke 5 model validasyonu", limit=5)

    assert "Found 2 section result(s)" in out
    assert "943 — ilke 5" in out
    assert "mevzuat_22599 — madde 9" in out
    assert "Model validasyonu yapılır" in out
    doc_store.search_document_sections.assert_awaited_once_with(
        "943 İlke 5 model validasyonu", document_id="943", section_type="ilke", limit=5
    )


@pytest.mark.asyncio
async def test_search_document_sections_boosts_exact_legal_reference():
    exact = _section("943", "ilke", "5", "İlke 5\nExact legal reference match.")
    fts = _section("943", "ilke", "6", "İlke 6\nFTS result.")
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[exact])
    doc_store.search_document_sections = AsyncMock(return_value=[fts, exact])
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "search_document_sections")
    out = await tool("943 İlke 5 model validasyonu", limit=5)

    assert out.index("ilke 5") < out.index("ilke 6")
    assert out.count("943 — ilke 5") == 1
    doc_store.get_document_section.assert_awaited_once_with("943", section_type="ilke", section_ref="5")


@pytest.mark.asyncio
async def test_search_document_sections_uses_loose_fallback_when_strict_misses():
    query = "Bilgi Sistemleri ve İş Süreçleri Bağımsız Denetimi denetim teknikleri dış teyit tetkik gözlem"
    target = _section(
        "mevzuat_39257",
        "madde",
        "31",
        "MADDE 31 - Denetim teknikleri\nDenetçi; tetkik, gözlem ve yeniden uygulama tekniklerini kullanır.",
    )
    doc_store = MagicMock()
    doc_store.get_document_section = AsyncMock(return_value=[])

    async def search_side_effect(search_query, *, document_id=None, section_type=None, limit=10):
        if search_query == query:
            return []
        if search_query in {"denetim", "teknikleri", "tetkik", "gözlem"}:
            return [target]
        return []

    doc_store.search_document_sections = AsyncMock(side_effect=search_side_effect)
    deps = Dependencies(pool=None, doc_store=doc_store, client=None, http=None)

    tool = _capture_tool(deps, "search_document_sections")
    out = await tool(query, limit=5)

    assert "Found 1 section result(s)" in out
    assert "mevzuat_39257 — madde 31" in out
    assert "Denetim teknikleri" in out
    assert doc_store.search_document_sections.await_args_list[0].kwargs == {
        "document_id": None,
        "section_type": None,
        "limit": 5,
    }
