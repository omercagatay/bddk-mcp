"""Grounding contract: every corpus tool's text carries what the server instructions ask the model to cite.

The assertions use labelled lines and stable markers, not whole sentences, so that
harmless copy edits do not break them.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.core.models import BddkDecisionSummary, BddkSearchResult
from bddk_mcp.store.doc_store import StoredDocument
from bddk_mcp.tools.search import _search_cache

_DOWNLOADED_AT = 1775667192.0  # 2026-04-08 UTC
_DOWNLOADED_ON = "2026-04-08"
_APPLICABILITY_MARKER = "does not establish current legal applicability"
_SOURCE_DATE_LABEL = "Text obtained from the source on: "

_IN_FORCE = {
    "document_id": "1291",
    "title": "Sermaye Yeterliliği Rehberi",
    "category": "Rehber",
    "decision_date": "15.03.2024",
    "decision_number": "11200",
    "source_url": "https://www.bddk.org.tr/Mevzuat/DokumanGetir/1291",
}
_REPEALED = {
    "document_id": "1230",
    "title": "4389 sayılı Bankalar Kanunu",
    "category": "Mülga Düzenleme",
    "decision_date": "18.06.1999",
    "decision_number": "4389",
    "source_url": "https://www.bddk.org.tr/Mevzuat/DokumanGetir/1230",
}


def _body(meta: dict) -> str:
    return f"MADDE 1 - {meta['title']} kapsamında sermaye yeterliliği oranı hesaplanır.\n"


def _hit(meta: dict) -> dict:
    return {
        "doc_id": meta["document_id"],
        "title": meta["title"],
        "category": meta["category"],
        "decision_date": meta["decision_date"],
        "decision_number": meta["decision_number"],
        "source_url": meta["source_url"],
        "section_type": "madde",
        "section_ref": "1",
        "snippet": _body(meta),
        "relevance": 0.8,
        "match_type": "hybrid",
    }


def _summary(meta: dict) -> BddkDecisionSummary:
    return BddkDecisionSummary(content="Katalog özeti", **meta)


@pytest.fixture
async def grounding_server(doc_store):
    from bddk_mcp.server import create_mcp

    for meta in (_IN_FORCE, _REPEALED):
        await doc_store.store_document(
            StoredDocument(
                markdown_content=_body(meta), downloaded_at=_DOWNLOADED_AT, extraction_method="manual", **meta
            )
        )
    catalog = {meta["document_id"]: _summary(meta) for meta in (_IN_FORCE, _REPEALED)}
    client = MagicMock()
    client.find_by_id = MagicMock(side_effect=catalog.get)
    client.search_decisions = AsyncMock(
        side_effect=lambda request: BddkSearchResult(
            decisions=[catalog[request.keywords]], total_results=1, page=1, page_size=10
        )
    )
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(
        side_effect=lambda query, **_kwargs: [_hit(_IN_FORCE if query.startswith("1291") else _REPEALED)]
    )
    _search_cache._data.clear()
    deps = Dependencies(db=None, doc_store=doc_store, client=client, http=None, vector_store=vector_store)
    return create_mcp(deps)


def _calls(meta: dict) -> dict[str, dict]:
    doc_id = meta["document_id"]
    return {
        "get_bddk_document": {"document_id": doc_id},
        "get_document_section": {"document_id": doc_id, "section_type": "madde", "section_ref": "1"},
        "search_document_sections": {"query": "sermaye yeterliliği oranı", "document_id": doc_id},
        "search_document_store": {"query": f"{doc_id} sermaye yeterliliği"},
        "search_bddk_regulations": {"keywords": doc_id},
    }


# Tools that return stored document text state when that text was downloaded.
# Search-store hits and catalog rows carry no download date, so they are leads only.
_STORED_TEXT_TOOLS = {"get_bddk_document", "get_document_section", "search_document_sections"}


async def _texts(server, meta: dict) -> dict[str, str]:
    async with create_connected_server_and_client_session(server) as session:
        results = {name: await session.call_tool(name, arguments) for name, arguments in _calls(meta).items()}
    for name, result in results.items():
        assert result.isError is False, (name, result.content[0].text)
    return {name: result.content[0].text for name, result in results.items()}


@pytest.mark.asyncio
async def test_every_corpus_tool_carries_title_identifiers_source_and_applicability(grounding_server):
    texts = await _texts(grounding_server, _IN_FORCE)

    for name, text in texts.items():
        assert _IN_FORCE["title"] in text, name
        assert _IN_FORCE["decision_date"] in text, name
        assert _IN_FORCE["decision_number"] in text, name
        assert _IN_FORCE["source_url"] in text, name
        assert _APPLICABILITY_MARKER in text, name
        assert "Document status:" not in text, name
        if name in _STORED_TEXT_TOOLS:
            assert f"{_SOURCE_DATE_LABEL}{_DOWNLOADED_ON} (UTC download date" in text, name
    for name in ("get_bddk_document", "get_document_section", "search_document_sections"):
        assert "Decision date / number: 15.03.2024 / 11200" in texts[name], name


@pytest.mark.asyncio
async def test_every_corpus_tool_marks_a_repealed_document(grounding_server):
    texts = await _texts(grounding_server, _REPEALED)

    for name, text in texts.items():
        assert _REPEALED["title"] in text, name
        assert "Document status: BDDK catalog class: MÜLGA (repealed)" in text, name
        assert _APPLICABILITY_MARKER in text, name
