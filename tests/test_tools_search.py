"""Tests for tools/search.py — LRU cache and search tool registration."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from bddk_mcp.store.vector_store import SemanticSearchUnavailableError
from bddk_mcp.tools.search import _LRUCache, _search_cache
from bddk_mcp.tools.structured_outputs import DocumentSearchItem

# -- LRU cache unit tests ---------------------------------------------------


def test_lru_cache_eviction():
    cache = _LRUCache(max_size=2, ttl=60)
    cache.set("a", "val_a")
    cache.set("b", "val_b")
    cache.set("c", "val_c")  # should evict "a"
    assert cache.get("a") is None
    assert cache.get("b") == "val_b"
    assert cache.get("c") == "val_c"


def test_lru_cache_access_refreshes():
    cache = _LRUCache(max_size=2, ttl=60)
    cache.set("a", "val_a")
    cache.set("b", "val_b")
    cache.get("a")  # refresh "a" — now "b" is oldest
    cache.set("c", "val_c")  # should evict "b", not "a"
    assert cache.get("a") == "val_a"
    assert cache.get("b") is None
    assert cache.get("c") == "val_c"


def test_lru_cache_ttl_expiry():
    """Expired entries should not be returned."""
    cache = _LRUCache(max_size=10, ttl=10)
    cache.set("x", "val_x")

    # Directly backdate the stored timestamp to simulate expiry
    cache._data["x"] = (time.time() - 20, "val_x")
    assert cache.get("x") is None


def test_lru_cache_empty_get():
    cache = _LRUCache(max_size=5, ttl=60)
    assert cache.get("nonexistent") is None


def test_lru_cache_overwrite():
    cache = _LRUCache(max_size=2, ttl=60)
    cache.set("a", "val_a")
    cache.set("a", "val_a_new")
    assert cache.get("a") == "val_a_new"


def test_lru_cache_size_one():
    """Single-slot cache: every new set evicts the previous."""
    cache = _LRUCache(max_size=1, ttl=60)
    cache.set("a", "val_a")
    cache.set("b", "val_b")
    assert cache.get("a") is None
    assert cache.get("b") == "val_b"


# -- Register smoke test ----------------------------------------------------


def test_search_register():
    """search.register() exposes exactly the four documented search tools."""
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import register

    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=None, http=None)
    register(mcp, deps)

    tool_names = {call.args[0].__name__ for call in mcp.tool.return_value.call_args_list}
    assert tool_names == {
        "search_bddk_regulations",
        "search_bddk_institutions",
        "search_bddk_announcements",
        "search_document_store",
    }


def _registered_tools(mcp: MagicMock) -> dict:
    return {call.args[0].__name__: call.args[0] for call in mcp.tool.return_value.call_args_list}


@pytest.mark.asyncio
async def test_search_document_store_uses_match_wording_and_section_guidance():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(
        return_value=[
            {
                "title": "TFRS 9 Uyarınca Beklenen Kredi Zararı Karşılığı Hesaplamasına İlişkin Rehber",
                "category": "Rehber",
                "decision_date": "",
                "doc_id": "943",
                "snippet": "İlke 5 - BKZ model validasyonu",
                "relevance": 0.58,
                "confidence": "medium",
            }
        ]
    )
    deps = Dependencies(db=None, doc_store=None, client=None, http=None, vector_store=vector_store)
    mcp = MagicMock()
    register(mcp, deps)
    search_document_store = _registered_tools(mcp)["search_document_store"]

    out = await search_document_store("TFRS 9 denetimi", limit=1)

    assert "moderate match" in out
    assert "confidence" not in out.lower()
    assert "search_document_sections" in out
    assert "get_document_section" in out


@pytest.mark.asyncio
async def test_search_document_store_registry_overrides_stale_clean_index_metadata(historical_quality_registry):
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(
        return_value=[
            {
                "title": "Configured quality failure",
                "category": "Rehber",
                "decision_date": "",
                "doc_id": "903",
                "snippet": "MADDE 1 - Temiz görünen metin",
                "relevance": 0.81,
                "quality_label": "clean",
                "quality_flags": [],
            }
        ]
    )
    deps = Dependencies(db=None, doc_store=None, client=None, http=None, vector_store=vector_store)
    mcp = MagicMock()
    register(mcp, deps)
    search_document_store = _registered_tools(mcp)["search_document_store"]

    out = await search_document_store("örnek", limit=1)

    assert "Quality: fail" in out
    assert "configured_quality_failure" in out
    assert "listed in the configured quality-failure registry" in out


@pytest.mark.asyncio
async def test_search_document_store_returns_non_retryable_error_when_embedding_runtime_failed():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock(
        side_effect=SemanticSearchUnavailableError("private runtime detail")
    )
    vector_store.search = AsyncMock()
    deps = Dependencies(db=None, doc_store=None, client=None, http=None, vector_store=vector_store)
    mcp = MagicMock()
    register(mcp, deps)
    search_document_store = _registered_tools(mcp)["search_document_store"]

    with pytest.raises(ToolError) as exc_info:
        await search_document_store("sermaye yeterliliği", limit=1)

    message = str(exc_info.value)
    assert message.startswith("[ERROR:SEMANTIC_SEARCH_UNAVAILABLE] retryable=false")
    assert "private runtime detail" not in message
    assert "search_document_sections" in message
    vector_store.search.assert_not_awaited()


# -- get_version_counts integration test (temporary corpus file) ---------------


@pytest.mark.asyncio
async def test_get_version_counts_empty(doc_store):
    """Querying unknown doc IDs should return an empty dict, not raise."""
    result = await doc_store.get_version_counts(["unknown_1", "unknown_2"])
    assert result == {}


@pytest.mark.asyncio
async def test_get_version_counts_no_args(doc_store):
    """Empty list should short-circuit and return {} without a DB query."""
    result = await doc_store.get_version_counts([])
    assert result == {}


# -- Upstream failure surfacing ----------------------------------------------
# Blocked egress must surface as a retryable tool error, never as "no results".


@pytest.mark.asyncio
async def test_search_institutions_upstream_failure_is_tool_error():
    from unittest.mock import patch

    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.core.exceptions import BddkUpstreamError
    from bddk_mcp.tools.search import register

    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=None, http=MagicMock())
    register(mcp, deps)
    tool = _registered_tools(mcp)["search_bddk_institutions"]

    with (
        patch(
            "bddk_mcp.tools.search.fetch_institutions_with_status",
            new=AsyncMock(side_effect=BddkUpstreamError("unreachable")),
        ),
        pytest.raises(ToolError) as excinfo,
    ):
        await tool()

    message = str(excinfo.value)
    assert "[ERROR:UPSTREAM_FETCH_FAILED]" in message
    assert "retryable=true" in message
    assert "NOT evidence" in message


@pytest.mark.asyncio
async def test_search_announcements_total_upstream_failure_is_tool_error():
    from unittest.mock import patch

    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.core.exceptions import BddkUpstreamError
    from bddk_mcp.tools.search import register

    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=None, http=MagicMock())
    register(mcp, deps)
    tool = _registered_tools(mcp)["search_bddk_announcements"]

    fetch = AsyncMock(side_effect=BddkUpstreamError("category failed"))
    with (
        patch("bddk_mcp.tools.search.fetch_announcements", new=fetch),
        pytest.raises(ToolError) as excinfo,
    ):
        await tool(category="tümü")

    message = str(excinfo.value)
    assert "[ERROR:UPSTREAM_FETCH_FAILED]" in message
    assert "retryable=true" in message
    # A per-category failure is not a host failure: every category is still
    # attempted, because the others may well succeed.
    assert fetch.await_count == 5


@pytest.mark.asyncio
async def test_search_announcements_unreachable_host_aborts_remaining_categories():
    """A blocked host must not be retried once per category (serial timeouts)."""
    from unittest.mock import patch

    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.core.exceptions import BddkUpstreamUnreachableError
    from bddk_mcp.tools.search import register

    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=None, http=MagicMock())
    register(mcp, deps)
    tool = _registered_tools(mcp)["search_bddk_announcements"]

    fetch = AsyncMock(side_effect=BddkUpstreamUnreachableError("blocked egress"))
    with (
        patch("bddk_mcp.tools.search.fetch_announcements", new=fetch),
        pytest.raises(ToolError) as excinfo,
    ):
        await tool(category="tümü")

    assert "[ERROR:UPSTREAM_FETCH_FAILED]" in str(excinfo.value)
    assert fetch.await_count == 1


@pytest.mark.asyncio
async def test_search_announcements_partial_failure_appends_warning():
    from unittest.mock import patch

    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.core.exceptions import BddkUpstreamError
    from bddk_mcp.tools.search import register

    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=None, http=MagicMock())
    register(mcp, deps)
    tool = _registered_tools(mcp)["search_bddk_announcements"]

    fetch = AsyncMock(
        side_effect=[
            [{"title": "Basın duyurusu örneği", "date": "01.08.2026", "url": ""}],
            BddkUpstreamError("unreachable"),
            BddkUpstreamError("unreachable"),
            BddkUpstreamError("unreachable"),
            BddkUpstreamError("unreachable"),
        ]
    )
    with patch("bddk_mcp.tools.search.fetch_announcements", new=fetch):
        result = await tool(category="tümü")

    assert "Basın duyurusu örneği" in result
    assert "WARNING" in result
    assert "incomplete" in result


@pytest.mark.asyncio
async def test_search_institutions_partial_directory_is_marked_incomplete():
    """A truncated directory must never be presented as the complete one."""
    from unittest.mock import patch

    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.ingest.data_sources import InstitutionDirectory
    from bddk_mcp.tools.search import register

    mcp = MagicMock()
    deps = Dependencies(db=None, doc_store=None, client=None, http=MagicMock())
    register(mcp, deps)
    tool = _registered_tools(mcp)["search_bddk_institutions"]

    directory = InstitutionDirectory(
        institutions=[{"name": "Örnek Banka", "type": "Banka", "status": "Aktif", "website": ""}],
        failed_pages=2,
        attempted_pages=5,
    )
    with patch(
        "bddk_mcp.tools.search.fetch_institutions_with_status",
        new=AsyncMock(return_value=directory),
    ):
        result = await tool()

    assert "Örnek Banka" in result
    assert "WARNING" in result
    assert "incomplete" in result
    assert "2 of 5" in result


def _hit(doc_id: str, relevance: float, match_type: str) -> dict:
    return {
        "title": f"Belge {doc_id}",
        "category": "Yönetmelik",
        "decision_date": "",
        "doc_id": doc_id,
        "snippet": f"MADDE 5 - metin {doc_id}",
        "relevance": relevance,
        "match_type": match_type,
    }


@pytest.mark.asyncio
async def test_fts_exact_hits_are_labelled_as_keyword_matches_not_weak_matches():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(
        return_value=[
            _hit("good", 0.875, "hybrid"),
            _hit("x1", 0.0, "fts_exact"),
            _hit("x2", 0.0, "fts_exact"),
            _hit("vec", 0.84, "vector"),
        ]
    )
    deps = Dependencies(db=None, doc_store=None, client=None, http=None, vector_store=vector_store)
    mcp = MagicMock()
    register(mcp, deps)

    result = await _registered_tools(mcp)["search_document_store"]("sermaye yeterliliği madde 5", limit=4)

    assert "relevance 0.0%" not in result.text
    assert "weak match" not in result.text
    assert result.text.count("[keyword match on the legal reference in the query; no semantic score]") == 2
    assert "WARNING" not in result.text
    structured = result.structuredContent
    assert not any("weak matches" in warning for warning in structured["warnings"])
    assert [(item["match_type"], item["match_strength"]) for item in structured["results"]] == [
        ("hybrid", "strong"),
        ("fts_exact", "keyword"),
        ("fts_exact", "keyword"),
        ("vector", "strong"),
    ]


@pytest.mark.asyncio
async def test_keyword_only_hits_are_listed_as_keyword_matches_and_not_as_no_results():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(return_value=[_hit("sem", 0.8, "hybrid"), _hit("806", 0.0, "fts")])
    deps = Dependencies(db=None, doc_store=None, client=None, http=None, vector_store=vector_store)
    mcp = MagicMock()
    register(mcp, deps)

    result = await _registered_tools(mcp)["search_document_store"]("pandemi", limit=10)

    assert "No documents found" not in result.text
    label = (
        "[keyword match: the words of the query all occur in this chunk or in its document title as separate "
        "words; this does not say that the phrase or the figure of the query occurs; no semantic score]"
    )
    assert result.text.count(label) == 1
    assert "every query word occurs" not in result.text
    assert "relevance 0.0%" not in result.text
    assert "weak match" not in result.text
    items = result.structuredContent["results"]
    assert [(item["match_type"], item["match_strength"]) for item in items] == [
        ("hybrid", "strong"),
        ("fts", "keyword"),
    ]
    assert (
        "the words of the query all occur in the chunk or in its document title as separate words"
        in DocumentSearchItem.model_fields["match_type"].description
    )
    assert "keyword-only" in DocumentSearchItem.model_fields["relevance"].description


@pytest.mark.asyncio
async def test_no_results_only_when_neither_semantic_nor_keyword_hits_exist():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(return_value=[])
    deps = Dependencies(db=None, doc_store=None, client=None, http=None, vector_store=vector_store)
    mcp = MagicMock()
    register(mcp, deps)

    result = await _registered_tools(mcp)["search_document_store"]("zzqqxx", limit=10)

    assert result.structuredContent["status"] == "no_results"
    assert "no document contains every query word" in result.text


@pytest.mark.asyncio
async def test_weak_semantic_hits_still_count_as_weak_matches():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(return_value=[_hit("low", 0.31, "vector"), _hit("x", 0.0, "fts_exact")])
    deps = Dependencies(db=None, doc_store=None, client=None, http=None, vector_store=vector_store)
    mcp = MagicMock()
    register(mcp, deps)

    result = await _registered_tools(mcp)["search_document_store"]("madde 5 karşılık", limit=2)

    assert "[weak match, relevance 31.0%]" in result.text
    assert "1 result(s) are weak matches" in result.text


@pytest.mark.asyncio
async def test_search_cache_never_serves_a_response_computed_before_a_corpus_write(corpus_db, doc_store):
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.store.doc_store import StoredDocument
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(
        side_effect=[[_hit("old", 0.9, "hybrid")], [_hit("new", 0.9, "hybrid")], [_hit("unexpected", 0.9, "hybrid")]]
    )
    deps = Dependencies(db=corpus_db, doc_store=doc_store, client=None, http=None, vector_store=vector_store)
    mcp = MagicMock()
    register(mcp, deps)
    tool = _registered_tools(mcp)["search_document_store"]

    first = await tool("sermaye yeterliliği", limit=1)
    cached = await tool("sermaye yeterliliği", limit=1)
    assert vector_store.search.await_count == 1
    assert cached.text == first.text

    await doc_store.store_document(
        StoredDocument(document_id="new", title="Yeni", markdown_content="MADDE 1 - Yeni metin.")
    )
    after_write = await tool("sermaye yeterliliği", limit=1)

    assert vector_store.search.await_count == 2
    assert "Document ID: new" in after_write.text


@pytest.mark.asyncio
async def test_search_hits_show_section_source_decision_and_the_exact_section_call():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    cited = {
        **_hit("kurul", 0.81, "hybrid"),
        "decision_date": "01.02.2024",
        "decision_number": "10812",
        "source_url": "https://www.bddk.org.tr/Mevzuat/DokumanGetir/1",
        "section_type": "madde",
        "section_ref": "4",
    }
    remainder = {**_hit("plain", 0.79, "vector"), "section_type": "govde", "section_ref": ""}
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(return_value=[cited, remainder])
    deps = Dependencies(db=None, doc_store=None, client=None, http=None, vector_store=vector_store)
    mcp = MagicMock()
    register(mcp, deps)

    result = await _registered_tools(mcp)["search_document_store"]("sermaye yeterliliği", limit=2)

    assert "  Section: madde 4" in result.text
    assert "  Catalog source (not a validated citation): https://www.bddk.org.tr/Mevzuat/DokumanGetir/1" in result.text
    assert "  Decision date / number: 01.02.2024 / 10812" in result.text
    assert 'get_document_section(document_id="kurul", section_type="madde", section_ref="4")' in result.text
    assert "  Section: govde remainder — not a legal provision identity" in result.text
    assert 'get_document_section(document_id="plain"' not in result.text
    structured = result.structuredContent
    first, second = structured["results"]
    assert (first["section_type"], first["section_ref"], first["source_url"], first["decision_number"]) == (
        "madde",
        "4",
        "https://www.bddk.org.tr/Mevzuat/DokumanGetir/1",
        "10812",
    )
    assert "decision_number" not in second and "source_url" not in second
    evidence = structured["evidence"][0]
    assert (evidence["section_type"], evidence["section_ref"], evidence["decision_number"]) == ("madde", "4", "10812")
    assert evidence["source_url"] == "https://www.bddk.org.tr/Mevzuat/DokumanGetir/1"


@pytest.mark.asyncio
async def test_govde_chunk_of_a_structureless_document_is_a_full_text_window(doc_store):
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.store.doc_store import StoredDocument
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    await doc_store.store_document(
        StoredDocument(
            document_id="779",
            title="Kurul Kararı",
            markdown_content="Kurulun 01.02.2024 tarihli toplantısında gecikme süresi doksan gün olarak belirlendi.",
        )
    )
    await doc_store.store_document(
        StoredDocument(
            document_id="mevzuat_8",
            title="Yönetmelik",
            markdown_content="Dipnot: değişikliklerle okunur.\n\nMADDE 1 - Amaç\nBu yönetmeliğin amacı denetimdir.\n",
        )
    )
    window = {**_hit("779", 0.81, "vector"), "section_type": "govde", "section_ref": "1", "decision_number": "10812"}
    window["decision_date"] = "01.02.2024"
    remainder = {**_hit("mevzuat_8", 0.79, "vector"), "section_type": "govde", "section_ref": "1"}
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(return_value=[window, remainder])
    mcp = MagicMock()
    register(mcp, Dependencies(db=None, doc_store=doc_store, client=None, http=None, vector_store=vector_store))

    result = await _registered_tools(mcp)["search_document_store"]("gecikme süresi", limit=2)

    window_block, remainder_block = result.text.split("**Belge mevzuat_8**")
    assert "  Section: full-text window (this document has no article structure)" in window_block
    assert "Cite this document by its title and decision date/number" in window_block
    assert "  Decision date / number: 01.02.2024 / 10812" in window_block
    assert "govde remainder" not in window_block
    assert "  Section: govde remainder — not a legal provision identity" in remainder_block
    assert await doc_store.structureless_documents(["779", "mevzuat_8", "missing"]) == {"779"}


@pytest.mark.asyncio
async def test_repealed_and_draft_hits_are_marked_in_both_search_tools():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.core.models import BddkDecisionSummary, BddkSearchResult
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(
        return_value=[
            {**_hit("1230", 0.8, "hybrid"), "category": "Mülga Düzenleme"},
            {**_hit("1290", 0.8, "hybrid"), "category": "Düzenleme Taslağı"},
            _hit("22599", 0.8, "hybrid"),
        ]
    )
    client = MagicMock()
    client.search_decisions = AsyncMock(
        return_value=BddkSearchResult(
            decisions=[
                BddkDecisionSummary(title="Eski", document_id="1230", content="", category="Mülga Düzenleme"),
                BddkDecisionSummary(title="Yeni", document_id="22599", content="", category="Yönetmelik"),
            ],
            total_results=2,
            page=1,
            page_size=10,
        )
    )
    doc_store = MagicMock()
    doc_store.get_version_counts = AsyncMock(return_value={})
    deps = Dependencies(db=None, doc_store=doc_store, client=client, http=None, vector_store=vector_store)
    mcp = MagicMock()
    register(mcp, deps)
    tools = _registered_tools(mcp)

    store = await tools["search_document_store"]("faiz riski", limit=3)
    catalog = await tools["search_bddk_regulations"]("faiz riski")

    assert store.text.count("Document status: BDDK catalog class: MÜLGA (repealed)") == 1
    assert store.text.count("Document status: BDDK catalog class: TASLAK (draft)") == 1
    assert any("MÜLGA" in w for w in store.structuredContent["warnings"])
    assert catalog.text.count("Document status: BDDK catalog class: MÜLGA (repealed)") == 1
    assert catalog.text.count("Document status:") == 1
    assert any("MÜLGA" in w for w in catalog.structuredContent["warnings"])


@pytest.mark.asyncio
async def test_version_counts_render_the_replacement_time_in_utc(doc_store, monkeypatch):
    import time as _time

    from bddk_mcp.store.doc_store import StoredDocument

    await doc_store.store_document(StoredDocument(document_id="1", title="T", markdown_content="eski"))
    monkeypatch.setattr("bddk_mcp.store.doc_store.time.time", lambda: 1775691000.0)
    await doc_store.store_document(StoredDocument(document_id="1", title="T", markdown_content="yeni"))
    monkeypatch.undo()
    monkeypatch.setenv("TZ", "Pacific/Kiritimati")
    _time.tzset()
    try:
        counts = await doc_store.get_version_counts(["1"])
    finally:
        monkeypatch.undo()
        _time.tzset()

    assert counts == {"1": (1, "2026-04-08 23:30 UTC")}


@pytest.mark.asyncio
async def test_catalog_version_line_is_a_local_change_count_not_amendment_history():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.core.models import BddkDecisionSummary, BddkSearchResult
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    client = MagicMock()
    client.search_decisions = AsyncMock(
        return_value=BddkSearchResult(
            decisions=[BddkDecisionSummary(title="Y", document_id="22599", content="", category="Yönetmelik")],
            total_results=1,
            page=1,
            page_size=10,
        )
    )
    doc_store = MagicMock()
    doc_store.get_version_counts = AsyncMock(return_value={"22599": (2, "2026-04-08 23:30 UTC")})
    mcp = MagicMock()
    register(mcp, Dependencies(db=None, doc_store=doc_store, client=client, http=None))

    result = await _registered_tools(mcp)["search_bddk_regulations"]("versiyon sayımı")

    assert "Versions:" not in result.text
    assert "Replaced bodies in this corpus file: 2 (latest replaced on: 2026-04-08 23:30 UTC)" in result.text
    assert "not amendment history" in result.text


def _catalog_tools(items):
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.ingest.client import BddkApiClient
    from bddk_mcp.tools.search import register
    from tests.conftest import MockStore

    _search_cache._data.clear()
    client = BddkApiClient(store=MockStore())
    client._cache = list(items)
    client._cache_timestamp = 9999999999.0
    doc_store = MagicMock()
    doc_store.get_version_counts = AsyncMock(return_value={})
    mcp = MagicMock()
    register(mcp, Dependencies(db=None, doc_store=doc_store, client=client, http=None))
    return _registered_tools(mcp)


@pytest.mark.asyncio
async def test_catalog_no_results_names_the_corpus_catalog_size_and_does_not_claim_absence():
    from bddk_mcp.core.models import BddkDecisionSummary

    items = [BddkDecisionSummary(title=f"Rehber {i}", document_id=str(i), category="Rehber") for i in range(7)]
    result = await _catalog_tools(items)["search_bddk_regulations"]("kripto yönetmeliği")

    assert result.structuredContent["status"] == "no_results"
    assert "this corpus catalog of 7 documents" in result.text
    assert "Absence here is NOT evidence that the instrument does not exist" in result.text


@pytest.mark.asyncio
async def test_catalog_summary_line_is_skipped_when_it_repeats_the_title():
    from bddk_mcp.core.models import BddkDecisionSummary

    items = [
        BddkDecisionSummary(title="Kredi Rehberi", document_id="1", content="Kredi Rehberi", category="Rehber"),
        BddkDecisionSummary(
            title="Kredi Genelgesi", document_id="2", content="Kredi hesapları hakkında özet", category="Genelge"
        ),
    ]
    result = await _catalog_tools(items)["search_bddk_regulations"]("kredi")

    assert result.text.count("Catalog summary/excerpt (NOT document text") == 1
    assert "Catalog summary/excerpt (NOT document text; never quote as the provision): Kredi hesapları" in result.text
    assert result.structuredContent["results"][0]["summary"] in {"Kredi Rehberi", "Kredi hesapları hakkında özet"}


@pytest.mark.asyncio
async def test_catalog_browse_mode_needs_a_category_or_date_and_lists_without_keywords():
    from bddk_mcp.core.models import BddkDecisionSummary

    items = [
        BddkDecisionSummary(
            title="A Kurul Kararı", document_id="1", decision_date="01.01.2020", category="Kurul Kararı"
        ),
        BddkDecisionSummary(title="B Rehber", document_id="2", category="Rehber"),
    ]
    tools = _catalog_tools(items)

    with pytest.raises(ToolError, match="INVALID_INPUT"):
        await tools["search_bddk_regulations"]("")
    browse = await tools["search_bddk_regulations"]("", category="Kurul Kararı")
    dated = await tools["search_bddk_regulations"](date_from="01.01.2019")

    assert [r["document_id"] for r in browse.structuredContent["results"]] == ["1"]
    assert [r["document_id"] for r in dated.structuredContent["results"]] == ["1"]
    assert dated.structuredContent["undated_excluded"] == 1
    assert "1 catalog entries have no decision date and were not considered by the date filter." in dated.text
    assert browse.structuredContent["undated_excluded"] == 0
    assert "no decision date" not in browse.text


@pytest.mark.asyncio
async def test_catalog_date_filter_no_results_reports_skipped_undated_entries():
    from bddk_mcp.core.models import BddkDecisionSummary

    items = [
        BddkDecisionSummary(
            title="A Kurul Kararı", document_id="1", decision_date="01.01.2020", category="Kurul Kararı"
        ),
        BddkDecisionSummary(title="B Rehber", document_id="2", category="Rehber"),
    ]
    result = await _catalog_tools(items)["search_bddk_regulations"](category="Rehber", date_from="01.01.2019")

    structured = result.structuredContent
    assert structured["status"] == "no_results"
    assert structured["undated_excluded"] == 1
    note = "1 catalog entries have no decision date and were not considered by the date filter."
    assert note in result.text
    assert note in structured["warnings"]
    assert "may simply not be in this corpus" not in result.text


def test_catalog_tool_description_says_it_searches_the_local_corpus_catalog():
    from bddk_mcp.tools.search import register

    mcp = MagicMock()
    from bddk_mcp.core.deps import Dependencies

    register(mcp, Dependencies(db=None, doc_store=None, client=None, http=None))
    doc = _registered_tools(mcp)["search_bddk_regulations"].__doc__
    assert "local corpus catalog" in doc


@pytest.mark.asyncio
async def test_catalog_page_beyond_the_last_page_says_so_and_gives_the_totals():
    from bddk_mcp.core.models import BddkDecisionSummary

    items = [
        BddkDecisionSummary(
            title=f"Kredi Kurul Kararı {i}", document_id=str(i), decision_date="01.01.2020", category="Kurul Kararı"
        )
        for i in range(23)
    ]
    result = await _catalog_tools(items)["search_bddk_regulations"](category="Kurul Kararı", page=5, page_size=10)

    assert result.structuredContent["status"] == "no_results"
    assert result.structuredContent["total_results"] == 23
    assert "No entry in this corpus catalog" not in result.text
    assert "page 5 is beyond the last page" in result.text
    assert "23 result(s)" in result.text
    assert "3 page(s)" in result.text


@pytest.mark.asyncio
async def test_catalog_empty_section_says_it_holds_no_documents():
    from bddk_mcp.core.models import BddkDecisionSummary

    items = [BddkDecisionSummary(title="Kredi Kartları Rehberi", document_id="1", category="Rehber")]
    result = await _catalog_tools(items)["search_bddk_regulations"](category="Banka Kartları")

    assert result.structuredContent["status"] == "no_results"
    assert "Banka Kartları" in result.text
    assert "holds 0 documents" in result.text
    assert "it may simply not be in this corpus" not in result.text


@pytest.mark.asyncio
async def test_catalog_category_filter_names_the_matches_in_other_sections():
    from bddk_mcp.core.models import BddkDecisionSummary

    items = [
        BddkDecisionSummary(
            title="Kredi Riski Azaltım Tekniklerine İlişkin Tebliğ", document_id="kra", category="Sermaye Yeterliliği"
        ),
        BddkDecisionSummary(title="Bankalarca Risk Yönetimine İlişkin Tebliğ", document_id="risk", category="Tebliğ"),
        BddkDecisionSummary(title="Kredi Riski Rehberi", document_id="rehber", category="Rehber"),
    ]
    tools = _catalog_tools(items)

    empty = await tools["search_bddk_regulations"]("kredi riski", category="Tebliğ")
    assert empty.structuredContent["status"] == "no_results"
    assert "2 match(es)" in empty.text
    assert "Sermaye Yeterliliği (1)" in empty.text
    assert "Rehber (1)" in empty.text
    assert "it may simply not be in this corpus" not in empty.text

    partial = await tools["search_bddk_regulations"]("tebliğ", category="Tebliğ")
    assert [r["document_id"] for r in partial.structuredContent["results"]] == ["risk"]
    assert "1 more match(es)" in partial.text
    assert "Sermaye Yeterliliği (1)" in partial.text

    alone = await tools["search_bddk_regulations"]("risk yönetimine", category="Tebliğ")
    assert "more match(es)" not in alone.text


def test_catalog_category_is_described_as_a_website_section_not_the_instrument_type():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import register

    mcp = MagicMock()
    register(mcp, Dependencies(db=None, doc_store=None, client=None, http=None))
    doc = _registered_tools(mcp)["search_bddk_regulations"].__doc__
    assert "website section" in doc
    assert "not the legal type of the instrument" in doc


@pytest.mark.asyncio
async def test_announcements_are_sorted_newest_first_across_categories_and_the_cut_is_stated():
    from unittest.mock import patch

    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import register

    mcp = MagicMock()
    register(mcp, Dependencies(db=None, doc_store=None, client=None, http=MagicMock()))
    tool = _registered_tools(mcp)["search_bddk_announcements"]

    press = [
        {"title": f"Basın {n}", "date": f"{n + 1:02d}.01.2025", "url": "", "category": "Basın Duyuruları"}
        for n in range(25)
    ]
    regulation = [
        {"title": "Mevzuat yeni", "date": "15.06.2026", "url": "", "category": "Mevzuat Duyuruları"},
        {"title": "Mevzuat tarihsiz", "date": "", "url": "", "category": "Mevzuat Duyuruları"},
        {"title": "Mevzuat bozuk", "date": "31.02.2026", "url": "", "category": "Mevzuat Duyuruları"},
    ]
    data = [{"title": "Veri yayını", "date": "01.03.2026", "url": "", "category": "Veri Yayımlama"}]
    fetch = AsyncMock(side_effect=[press, regulation, [], data, []])
    with patch("bddk_mcp.tools.search.fetch_announcements", new=fetch):
        result = await tool(category="tümü")

    titles = [line.split("**")[1] for line in result.splitlines() if line.startswith("**")]
    assert "Found 29 announcement(s); showing the 20 most recent of 29" in result
    assert titles[:3] == ["Mevzuat yeni", "Veri yayını", "Basın 24"]
    assert len(titles) == 20
    assert "Mevzuat tarihsiz" not in titles and "Basın 0" not in titles
    assert "... and 9 more" in result
    assert "**Mevzuat yeni** (15.06.2026) [Mevzuat Duyuruları]" in result


@pytest.mark.asyncio
async def test_short_announcement_list_has_no_cut_notice_and_keeps_undated_items_last():
    from unittest.mock import patch

    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import _newest_first, register

    items = [
        {"title": "A", "date": "", "category": "Basın Duyuruları"},
        {"title": "B", "date": "01.01.2024", "category": "Basın Duyuruları"},
        {"title": "C", "date": "01.01.2024", "category": "Basın Duyuruları"},
        {"title": "D", "date": "bozuk", "category": "Basın Duyuruları"},
        {"title": "E", "date": "02.01.2024", "category": "Basın Duyuruları"},
    ]
    assert [a["title"] for a in _newest_first(items)] == ["E", "B", "C", "A", "D"]

    mcp = MagicMock()
    register(mcp, Dependencies(db=None, doc_store=None, client=None, http=MagicMock()))
    with patch("bddk_mcp.tools.search.fetch_announcements", new=AsyncMock(return_value=items)):
        result = await _registered_tools(mcp)["search_bddk_announcements"]()

    assert "Found 5 announcement(s):" in result
    assert "most recent" not in result and "more (narrow" not in result


@pytest.mark.asyncio
async def test_search_institutions_marks_digital_banks_without_stray_brackets():
    """Built on the live '(Dijital Banka)' suffix form of BDDK's directory page."""
    from pathlib import Path
    from unittest.mock import patch

    from bs4 import BeautifulSoup

    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.ingest.data_sources import InstitutionDirectory, _parse_card_institutions
    from bddk_mcp.tools.search import register

    html = (Path(__file__).parent / "fixtures" / "bddk_kurulus_77_katilim.html").read_text(encoding="utf-8")
    items = _parse_card_institutions(BeautifulSoup(html, "html.parser"), "Banka")
    mcp = MagicMock()
    register(mcp, Dependencies(db=None, doc_store=None, client=None, http=MagicMock()))
    tool = _registered_tools(mcp)["search_bddk_institutions"]
    directory = InstitutionDirectory(institutions=items, failed_pages=0, attempted_pages=1)
    with patch("bddk_mcp.tools.search.fetch_institutions_with_status", new=AsyncMock(return_value=directory)):
        result = await tool(keywords="katılım", institution_type="Banka")

    assert "**ADİL KATILIM BANKASI A.Ş.** (Dijital Banka) [Banka]" in result
    assert "**HAYAT FİNANS KATILIM BANKASI A.Ş.** (Dijital Banka) [Banka]" in result
    assert "**DÜNYA KATILIM BANKASI A.Ş.** [Banka]" in result
    assert "A.Ş.()" not in result


@pytest.mark.asyncio
async def test_a_hit_on_another_number_of_the_requested_section_kind_says_so():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    other = {**_hit("949", 0.9, "hybrid"), "section_type": "ilke", "section_ref": "22"}
    same = {**_hit("950", 0.9, "hybrid"), "section_type": "ilke", "section_ref": "12"}
    article = {**_hit("951", 0.9, "hybrid"), "section_type": "madde", "section_ref": "4"}
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(return_value=[other, same, article])
    deps = Dependencies(db=None, doc_store=None, client=None, http=None, vector_store=vector_store)
    mcp = MagicMock()
    register(mcp, deps)

    result = await _registered_tools(mcp)["search_document_store"]("Kredi yönetimi rehberi İlke 12", limit=3)

    blocks = result.text.split("**Belge ")
    assert "this hit is about ilke 22, not the ilke 12 requested in the query" in blocks[1]
    assert "requested in the query" not in blocks[2]
    assert "requested in the query" not in blocks[3]
    assert [item["relevance"] for item in result.structuredContent["results"]] == [0.9, 0.9, 0.9]


@pytest.mark.asyncio
async def test_a_query_without_a_section_reference_gets_no_mismatch_note():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(
        return_value=[{**_hit("949", 0.9, "hybrid"), "section_type": "ilke", "section_ref": "22"}]
    )
    deps = Dependencies(db=None, doc_store=None, client=None, http=None, vector_store=vector_store)
    mcp = MagicMock()
    register(mcp, deps)

    result = await _registered_tools(mcp)["search_document_store"]("kredi limitleri", limit=1)

    assert "requested in the query" not in result.text


@pytest.mark.asyncio
async def test_a_normalised_query_is_stated_in_the_text_and_the_structured_result():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.store.vector_store import SearchHits
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    hits = SearchHits([_hit("1224", 0.8, "hybrid")])
    hits.searched_as = "müşteri şikayetleri"
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(return_value=hits)
    deps = Dependencies(db=None, doc_store=None, client=None, http=None, vector_store=vector_store)
    mcp = MagicMock()
    register(mcp, deps)
    tool = _registered_tools(mcp)["search_document_store"]

    result = await tool("musteri sikayetleri", limit=1)
    vector_store.search = AsyncMock(return_value=[_hit("1224", 0.8, "hybrid")])
    plain = await tool("müşteri şikayetleri", limit=1)

    assert "Searched as: 'müşteri şikayetleri'" in result.text
    assert result.structuredContent["searched_as"] == "müşteri şikayetleri"
    assert "Searched as" not in plain.text
    assert "searched_as" not in plain.structuredContent


@pytest.mark.asyncio
async def test_an_expanded_abbreviation_is_stated():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.store.vector_store import SearchHits
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    hits = SearchHits([_hit("806", 0.0, "fts")])
    hits.expansions = [("LKO", "likidite karşılama oranı")]
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(return_value=hits)
    deps = Dependencies(db=None, doc_store=None, client=None, http=None, vector_store=vector_store)
    mcp = MagicMock()
    register(mcp, deps)

    result = await _registered_tools(mcp)["search_document_store"]("LKO", limit=5)

    assert "LKO read as likidite karşılama oranı" in result.text
    assert result.structuredContent["abbreviations_expanded"] == ["LKO read as likidite karşılama oranı"]


_TWO_ARTICLES_WITH_AN_ANNEX = (
    "MADDE 1 – Amaç\n(1) Bu Yönetmeliğin amacı denetimdir.\n\nMADDE 2 – Kapsam\n(1) Birinci fıkra.\n"
    "(2) İkinci fıkra metni.\n\nEK-2\n\nKarşı taraf kredi riski hesaplaması\n\n"
    + " ".join(f"cümle{i} uzun ek metni." for i in range(3000))
    + "\n"
)


async def _store_with_sections(doc_store):
    from bddk_mcp.store.doc_store import StoredDocument

    await doc_store.store_document(
        StoredDocument(
            document_id="m1", title="Yönetmelik", markdown_content=_TWO_ARTICLES_WITH_AN_ANNEX, category="Yönetmelik"
        )
    )
    return {
        (row.section_type, row.section_ref, row.start_char): row for row in await doc_store.get_document_section("m1")
    }


def _section_hit(row, relevance=0.9):
    return {
        **_hit(row.doc_id, relevance, "hybrid"),
        "section_type": row.section_type,
        "section_ref": row.section_ref,
        "section_start_char": row.start_char,
        "section_end_char": row.end_char,
    }


@pytest.mark.asyncio
async def test_a_fikra_hit_names_its_parent_article_and_points_to_a_unique_address(doc_store):
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    rows = await _store_with_sections(doc_store)
    fikra = next(row for (kind, ref, _), row in rows.items() if (kind, ref) == ("fikra", "1") and row.start_char > 54)
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(return_value=[_section_hit(fikra)])
    mcp = MagicMock()
    register(mcp, Dependencies(db=None, doc_store=doc_store, client=None, http=None, vector_store=vector_store))

    result = await _registered_tools(mcp)["search_document_store"]("birinci fıkra", limit=1)

    assert "  Section: fikra 1 — Parent: madde 2" in result.text
    pointer = (
        f'get_document_section(document_id="m1", section_type="fikra", section_ref="1", start_char={fikra.start_char})'
    )
    assert pointer in result.text
    assert 'get_document_section(document_id="m1", section_type="madde", section_ref="2")' in result.text
    assert (
        len(
            await doc_store.get_document_section(
                "m1", section_type="fikra", section_ref="1", start_char=fikra.start_char
            )
        )
        == 1
    )
    item = result.structuredContent["results"][0]
    assert (item["section_start_char"], item["parent_section_type"], item["parent_section_ref"]) == (
        fikra.start_char,
        "madde",
        "2",
    )


@pytest.mark.asyncio
async def test_a_govde_hit_that_continues_a_capped_annex_is_labelled_as_that_continuation(doc_store):
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    rows = await _store_with_sections(doc_store)
    continuation = next(row for (kind, ref, _), row in rows.items() if (kind, ref) == ("govde", "2"))
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(return_value=[_section_hit(continuation)])
    mcp = MagicMock()
    register(mcp, Dependencies(db=None, doc_store=doc_store, client=None, http=None, vector_store=vector_store))

    result = await _registered_tools(mcp)["search_document_store"]("uzun ek metni", limit=1)

    assert "govde remainder" not in result.text
    assert "  Section: continuation of ek 2 beyond its" in result.text
    assert 'get_document_section(document_id="m1", section_type="govde", section_ref="2")' in result.text
    assert 'get_document_section(document_id="m1", section_type="ek", section_ref="2")' in result.text


@pytest.mark.asyncio
async def test_section_refs_are_printed_in_display_form_and_control_characters_are_not_echoed():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    inserted = {**_hit("5411", 0.9, "hybrid"), "section_type": "madde", "section_ref": "66/a"}
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(return_value=[inserted])
    deps = Dependencies(db=None, doc_store=None, client=None, http=None, vector_store=vector_store)
    mcp = MagicMock()
    register(mcp, deps)

    result = await _registered_tools(mcp)["search_document_store"]("ek madde\x07 sermaye", limit=1)

    assert "  Section: madde 66/A" in result.text
    assert 'section_ref="66/a")' in result.text
    assert "\x07" not in result.text
    assert "ek madde\\x07 sermaye" in result.text


@pytest.mark.asyncio
async def test_a_clamped_relevance_is_shown_as_the_calibration_ceiling_and_snippets_have_honest_ellipses():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    clamped = {**_hit("a", 1.0, "hybrid"), "relevance_clamped": True, "semantic_relevance": 0.95}
    clamped["snippet"] = "x" * 250
    plain = {**_hit("b", 0.8, "hybrid"), "relevance_clamped": False, "semantic_relevance": 0.93}
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(return_value=[clamped, plain])
    deps = Dependencies(db=None, doc_store=None, client=None, http=None, vector_store=vector_store)
    mcp = MagicMock()
    register(mcp, deps)

    result = await _registered_tools(mcp)["search_document_store"]("sermaye", limit=2)

    assert "100.0%" not in result.text
    assert (
        "[strong match, relevance at the calibration ceiling (100%); tied hits are ordered by the internal score]"
        in (result.text)
    )
    assert "[strong match, relevance 80.0%]" in result.text
    assert f"\n  {'x' * 200}...\n" in result.text
    assert "\n  MADDE 5 - metin b\n" in result.text
    assert "\n  ..." not in result.text
    first, second = result.structuredContent["results"]
    assert (first["relevance"], first["relevance_clamped"], first["semantic_relevance"]) == (1.0, True, 0.95)
    assert (second["relevance_clamped"], second["semantic_relevance"]) == (False, 0.93)


@pytest.mark.asyncio
async def test_function_word_queries_and_decision_number_matches_are_said_plainly():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.store.vector_store import SearchHits
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    empty = SearchHits([])
    empty.no_content_words = True
    decision = {**_hit("775", 0.0, "fts_exact"), "decision_number": "8948", "decision_number_match": True}
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(side_effect=[empty, [decision]])
    deps = Dependencies(db=None, doc_store=None, client=None, http=None, vector_store=vector_store)
    mcp = MagicMock()
    register(mcp, deps)
    tool = _registered_tools(mcp)["search_document_store"]

    function_words = await tool("ve ile bir bu da de için", limit=10)
    number = await tool("8948", limit=10)

    assert function_words.structuredContent["status"] == "no_results"
    assert "only function words" in function_words.text
    assert "match" not in function_words.text.split("only function words")[0]
    assert "[decision-number match: this document's decision number is 8948; no semantic score]" in number.text
    assert number.structuredContent["results"][0]["decision_number_match"] is True


@pytest.mark.asyncio
async def test_a_reference_that_repeats_in_the_document_gets_start_char_in_its_pointer(doc_store):
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.store.doc_store import StoredDocument
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    content = "BİRİNCİ KISIM\n\nMADDE 1 – Amaç birinci kısım.\n\nMADDE 2 – Kapsam.\n\nİKİNCİ KISIM\n\nMADDE 1 – Amaç ikinci kısım.\n"
    await doc_store.store_document(StoredDocument(document_id="m2", title="Yönetmelik", markdown_content=content))
    rows = {(row.section_ref, row.start_char): row for row in await doc_store.get_document_section("m2")}
    second_madde_1 = max((row for (ref, _), row in rows.items() if ref == "1"), key=lambda row: row.start_char)
    madde_2 = next(row for (ref, _), row in rows.items() if ref == "2")
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(return_value=[_section_hit(second_madde_1), _section_hit(madde_2)])
    mcp = MagicMock()
    register(mcp, Dependencies(db=None, doc_store=doc_store, client=None, http=None, vector_store=vector_store))

    result = await _registered_tools(mcp)["search_document_store"]("amaç", limit=2)

    assert (
        f'get_document_section(document_id="m2", section_type="madde", section_ref="1", '
        f"start_char={second_madde_1.start_char})"
    ) in result.text
    assert 'get_document_section(document_id="m2", section_type="madde", section_ref="2")' in result.text


@pytest.mark.asyncio
async def test_decision_number_hits_have_their_own_match_type_and_the_structured_query_has_no_raw_controls():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    decision = {**_hit("775", 0.0, "decision_number"), "decision_number": "8948", "decision_number_match": True}
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(return_value=[decision])
    deps = Dependencies(db=None, doc_store=None, client=None, http=None, vector_store=vector_store)
    mcp = MagicMock()
    register(mcp, deps)

    result = await _registered_tools(mcp)["search_document_store"]("8948\x07", limit=10)

    item = result.structuredContent["results"][0]
    assert (item["match_type"], item["match_strength"]) == ("decision_number", "keyword")
    assert "decision_number" in DocumentSearchItem.model_fields["match_type"].description
    assert "every query word" not in result.text.split("Document ID: 775")[0]
    assert result.structuredContent["query"] == "8948\\x07"
    assert "WARNING" not in result.text


@pytest.mark.asyncio
async def test_a_decision_text_hit_carries_a_per_hit_label_not_the_plural_warning(doc_store):
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.store.doc_store import StoredDocument
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    content = (
        "Kurulun 01.02.2024 tarihli toplantısında aşağıdaki karar alınmıştır. " * 8 + "\n\nEK-1\n\nEk tablo metni.\n"
    )
    await doc_store.store_document(
        StoredDocument(document_id="k1", title="Kurul Kararı", markdown_content=content, category="Kurul Kararı")
    )
    [decision_text] = await doc_store.get_document_section("k1", section_type="govde")
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(return_value=[_section_hit(decision_text)])
    mcp = MagicMock()
    register(mcp, Dependencies(db=None, doc_store=doc_store, client=None, http=None, vector_store=vector_store))

    result = await _registered_tools(mcp)["search_document_store"]("karar", limit=1)

    assert "  Section: leading decision text of a Kurul Kararı whose only named sections are annexes" in result.text
    assert "One or more hits" not in result.text
    assert "Cite this decision by its title and decision date/number" in result.text


@pytest.mark.asyncio
async def test_catalog_search_names_the_section_reference_it_did_not_search():
    from bddk_mcp.core.models import BddkDecisionSummary

    items = [
        BddkDecisionSummary(
            title="Kredilerin Sınıflandırılması ve Bunlar İçin Ayrılacak Karşılıklara İlişkin Yönetmelik",
            document_id="mevzuat_22599",
            category="Yönetmelik",
        )
    ]
    tools = _catalog_tools(items)

    found = await tools["search_bddk_regulations"]("kredilerin sınıflandırılması yönetmeliği madde 9")
    missing = await tools["search_bddk_regulations"]("kripto yönetmeliği madde 9")
    plain = await tools["search_bddk_regulations"]("kredilerin sınıflandırılması")

    assert found.structuredContent["results"][0]["document_id"] == "mevzuat_22599"
    assert "not searched in the catalog: madde 9" in found.text
    assert any("not searched in the catalog: madde 9" in warning for warning in found.structuredContent["warnings"])
    assert missing.structuredContent["status"] == "no_results"
    assert "not searched in the catalog: madde 9" in missing.text
    assert "not searched in the catalog" not in plain.text


_REPEATED_ARTICLE_TEXT = (
    "MADDE 1 - Amaç\n(1) Birinci metin.\n(2) İkinci metin.\n\nMADDE 2 - Kapsam\n(1) Kapsam.\n\n"
    "MADDE 1 - Değişik\n(1) Üçüncü metin.\n(2) Dördüncü metin kredi.\n"
)


@pytest.mark.asyncio
async def test_the_full_provision_pointer_carries_the_start_of_a_parent_whose_reference_repeats(doc_store):
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.store.doc_store import StoredDocument
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    await doc_store.store_document(
        StoredDocument(document_id="m9", title="Yönetmelik", markdown_content=_REPEATED_ARTICLE_TEXT)
    )
    articles = sorted(
        row.start_char for row in await doc_store.get_document_section("m9", section_type="madde", section_ref="1")
    )
    fikras = await doc_store.get_document_section("m9", section_type="fikra", section_ref="2")
    late = max(fikras, key=lambda row: row.start_char)
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(return_value=[_section_hit(late)])
    mcp = MagicMock()
    register(mcp, Dependencies(db=None, doc_store=doc_store, client=None, http=None, vector_store=vector_store))

    result = await _registered_tools(mcp)["search_document_store"]("dördüncü metin", limit=1)

    assert (
        f'Full provision: get_document_section(document_id="m9", section_type="madde", section_ref="1", '
        f"start_char={articles[1]})" in result.text
    )


@pytest.mark.asyncio
async def test_the_mismatch_note_also_compares_the_parent_provision(doc_store):
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.search import register

    _search_cache._data.clear()
    rows = await _store_with_sections(doc_store)
    fikra = next(row for (kind, ref, _), row in rows.items() if (kind, ref) == ("fikra", "1") and row.start_char > 54)
    vector_store = MagicMock()
    vector_store.assert_semantic_search_ready = AsyncMock()
    vector_store.search = AsyncMock(return_value=[_section_hit(fikra)])
    mcp = MagicMock()
    register(mcp, Dependencies(db=None, doc_store=doc_store, client=None, http=None, vector_store=vector_store))
    tool = _registered_tools(mcp)["search_document_store"]

    other = await tool("madde 12 birinci fıkra", limit=1)
    _search_cache._data.clear()
    same = await tool("madde 2 birinci fıkra", limit=1)

    assert "[note: this hit is part of madde 2, not the madde 12 requested in the query]" in other.text
    assert "[note:" not in same.text
