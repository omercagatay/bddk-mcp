"""Tests for tools/search.py — LRU cache and search tool registration."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from tools.search import _LRUCache


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
    """search.register() adds the four search tools on the MCP instance."""
    from deps import Dependencies
    from tools.search import register

    mcp = MagicMock()
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    register(mcp, deps)
    assert mcp.tool.call_count >= 4


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
