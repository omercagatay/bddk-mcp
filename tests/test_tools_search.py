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
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
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
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None, vector_store=vector_store)
    mcp = MagicMock()
    register(mcp, deps)
    search_document_store = _registered_tools(mcp)["search_document_store"]

    out = await search_document_store("TFRS 9 denetimi", limit=1)

    assert "moderate match" in out
    assert "confidence" not in out.lower()
    assert "search_document_sections" in out
    assert "get_document_section" in out


@pytest.mark.asyncio
async def test_search_document_store_registry_overrides_stale_clean_index_metadata():
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
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None, vector_store=vector_store)
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
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None, vector_store=vector_store)
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


# -- get_version_counts integration test (requires PostgreSQL) ---------------


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
    deps = Dependencies(pool=None, doc_store=None, client=None, http=MagicMock())
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
    deps = Dependencies(pool=None, doc_store=None, client=None, http=MagicMock())
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
    deps = Dependencies(pool=None, doc_store=None, client=None, http=MagicMock())
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
    deps = Dependencies(pool=None, doc_store=None, client=None, http=MagicMock())
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
    deps = Dependencies(pool=None, doc_store=None, client=None, http=MagicMock())
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
