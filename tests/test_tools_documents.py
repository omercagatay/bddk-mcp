"""Tests for tools/documents.py — tool registration and bare-ID resolver."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from deps import Dependencies
from doc_store import DocumentPage
from tools import documents as documents_mod
from tools.documents import _is_formula_aware, register


def test_register_exposes_end_user_tools_only_by_default():
    """Default (ADMIN_TOOLS=false) hides document_store_stats from end users."""
    mcp = MagicMock()
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    register(mcp, deps)

    tool_names = {call.args[0].__name__ for call in mcp.tool.return_value.call_args_list}
    assert tool_names == {
        "get_bddk_document",
        "get_document_history",
    }


def test_register_adds_admin_tool_when_flag_enabled(monkeypatch):
    """With ADMIN_TOOLS=true the operator-only document_store_stats is also exposed."""
    monkeypatch.setattr(documents_mod, "ADMIN_TOOLS", True)
    mcp = MagicMock()
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    register(mcp, deps)

    tool_names = {call.args[0].__name__ for call in mcp.tool.return_value.call_args_list}
    assert tool_names == {
        "get_bddk_document",
        "get_document_history",
        "document_store_stats",
    }


def _capture_get_bddk_document(deps: Dependencies):
    """Register tools on a stub MCP and return the get_bddk_document callable."""
    mcp = MagicMock()
    register(mcp, deps)
    for call in mcp.tool.return_value.call_args_list:
        fn = call.args[0]
        if fn.__name__ == "get_bddk_document":
            return fn
    raise AssertionError("get_bddk_document not registered")


def _make_deps(*, doc_store, client=None, vector_store=None) -> Dependencies:
    deps = Dependencies(pool=None, doc_store=doc_store, client=client or MagicMock(), http=None)
    deps.vector_store = vector_store
    if client is None:
        deps.client.find_by_id = MagicMock(return_value=None)
    return deps


@pytest.mark.asyncio
async def test_bare_numeric_id_resolves_to_mevzuat_prefix():
    """get_bddk_document('21192') should auto-resolve to 'mevzuat_21192' on bare-ID miss."""
    page = DocumentPage(
        document_id="mevzuat_21192",
        title="Sermaye Yeterliliği Yönetmeliği",
        markdown_content="BANKALARIN SERMAYE YETERLİLİĞİ...",
        page_number=1,
        total_pages=28,
    )

    async def fake_get_page(doc_id, page_number):
        return page if doc_id == "mevzuat_21192" else None

    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(side_effect=fake_get_page)
    deps = _make_deps(doc_store=doc_store)

    tool = _capture_get_bddk_document(deps)
    out = await tool("21192", 1)

    assert "BANKALARIN SERMAYE YETERLİLİĞİ" in out
    assert "Document ID: mevzuat_21192" in out
    assert "Resolved from: `21192` -> `mevzuat_21192`" in out
    # Must have probed the bare ID before falling back to the prefixed form
    assert doc_store.get_document_page.await_args_list[0].args == ("21192", 1)
    assert doc_store.get_document_page.await_args_list[1].args == ("mevzuat_21192", 1)


@pytest.mark.asyncio
async def test_bare_id_exact_match_skips_prefix_lookup():
    """If the bare ID exists as-is, the prefix variants must not be probed."""
    page = DocumentPage(
        document_id="1291",
        title="Karar 1291",
        markdown_content="Kurul kararı...",
        page_number=1,
        total_pages=1,
    )
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=page)
    deps = _make_deps(doc_store=doc_store)

    tool = _capture_get_bddk_document(deps)
    out = await tool("1291", 1)

    assert "Resolved from" not in out
    assert "Document ID: 1291" in out
    assert doc_store.get_document_page.await_count == 1


@pytest.mark.asyncio
async def test_prefixed_id_does_not_get_expanded():
    """An already-prefixed ID must be tried only as-is — no '21192' or 'bddk_21192' probes."""
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=None)
    deps = _make_deps(doc_store=doc_store)

    tool = _capture_get_bddk_document(deps)
    out = await tool("mevzuat_21192", 1)

    assert "not available in the local store" in out
    assert "airlocked" in out
    assert doc_store.get_document_page.await_count == 1
    assert doc_store.get_document_page.await_args_list[0].args == ("mevzuat_21192", 1)


@pytest.mark.parametrize(
    "method,expected",
    [
        ("lightocr", True),
        ("chandra2", True),
        ("mevzuat_pdf+lightocr", True),
        ("cached_pdf+chandra2", True),
        ("manual_latex", True),
        ("html_parser+manual_latex", True),
        ("markitdown", False),
        ("markitdown_degraded", False),
        ("manual_pdf+markitdown", False),
        ("html_parser", False),
        ("", False),
        ("glm_ocr", False),
    ],
)
def test_is_formula_aware_classification(method, expected):
    assert _is_formula_aware(method) is expected


@pytest.mark.asyncio
async def test_degraded_extraction_emits_warning_and_method_line():
    """A document extracted by markitdown must surface a formula-loss warning."""
    page = DocumentPage(
        document_id="mevzuat_20029",
        title="Kredi Riski Azaltım Tekniklerine İlişkin Tebliğ",
        markdown_content="MADDE 46 ... aşağıda yer alan formül kullanılarak artırılır.",
        page_number=18,
        total_pages=24,
        extraction_method="manual_pdf+markitdown",
    )
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=page)
    doc_store.get_extraction_method = AsyncMock(return_value="manual_pdf+markitdown")
    deps = _make_deps(doc_store=doc_store)

    tool = _capture_get_bddk_document(deps)
    out = await tool("mevzuat_20029", 18)

    assert "- Extraction: manual_pdf+markitdown (formula-unaware" in out
    assert "⚠" in out
    assert "formülü hafızadan veya standart literatürden yeniden kurma" in out
    # The extraction line must come before the content boundary.
    assert out.index("- Extraction:") < out.index("---")


@pytest.mark.asyncio
async def test_formula_aware_extraction_no_warning():
    """A lightocr-extracted document must NOT emit the formula-loss warning."""
    page = DocumentPage(
        document_id="mevzuat_20029",
        title="Kredi Riski Azaltım Tekniklerine İlişkin Tebliğ",
        markdown_content="formül içeren metin",
        page_number=1,
        total_pages=1,
        extraction_method="mevzuat_pdf+lightocr",
    )
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=page)
    doc_store.get_extraction_method = AsyncMock(return_value="mevzuat_pdf+lightocr")
    deps = _make_deps(doc_store=doc_store)

    tool = _capture_get_bddk_document(deps)
    out = await tool("mevzuat_20029", 1)

    assert "- Extraction: mevzuat_pdf+lightocr" in out
    assert "formula-unaware" not in out
    assert "⚠" not in out


@pytest.mark.asyncio
async def test_pgvector_path_still_looks_up_extraction_method():
    """When served via vector_store, the tool must still fetch extraction_method via doc_store."""
    vector_store = MagicMock()
    vector_store.get_document_page = AsyncMock(
        return_value={
            "doc_id": "mevzuat_20029",
            "title": "Kredi Riski Azaltım Tekniklerine İlişkin Tebliğ",
            "content": "page text",
            "page_number": 1,
            "total_pages": 24,
        }
    )
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=None)
    doc_store.get_extraction_method = AsyncMock(return_value="markitdown")
    deps = _make_deps(doc_store=doc_store, vector_store=vector_store)

    tool = _capture_get_bddk_document(deps)
    out = await tool("mevzuat_20029", 1)

    doc_store.get_extraction_method.assert_awaited_once_with("mevzuat_20029")
    assert "- Extraction: markitdown (formula-unaware" in out
    assert "⚠" in out


@pytest.mark.asyncio
async def test_missing_doc_returns_airlocked_error_for_all_candidates():
    """When no candidate hits, return the airlock error referencing the original ID."""
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(return_value=None)
    deps = _make_deps(doc_store=doc_store)

    tool = _capture_get_bddk_document(deps)
    out = await tool("99999999", 1)

    assert "Document 99999999 is not available" in out
    assert "airlocked" in out
    # All three candidates probed: bare, mevzuat_, bddk_
    assert doc_store.get_document_page.await_count == 3
