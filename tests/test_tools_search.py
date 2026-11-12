"""Tests for tools/search.py — LRU cache and search tool registration."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.server.fastmcp.exceptions import ToolError

from bddk_mcp.store.vector_store import SemanticSearchUnavailableError
from bddk_mcp.tools.search import _LRUCache, _search_cache

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
