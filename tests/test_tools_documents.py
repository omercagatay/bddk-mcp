"""Tests for tools/documents.py — tool registration and bare-ID resolver."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from deps import Dependencies
from doc_store import DocumentPage
from tools.documents import register


def test_register_adds_three_document_tools():
    """register() should expose exactly the three documented document tools."""
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
