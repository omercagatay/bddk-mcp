# BDDK MCP Server Restructure — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure server.py monolith into modular architecture with dependency injection, fixing startup timeouts, improving search performance, and establishing proper lifecycle management.

**Architecture:** Dependencies dataclass replaces 6 global singletons. Tool handlers move from server.py into `tools/` modules that receive deps via closure capture. Background vector store init prevents Railway startup timeouts. Shared httpx client reduces TLS overhead.

**Tech Stack:** Python 3.12, FastMCP, asyncpg, pgvector, httpx, sentence-transformers

---

### Task 1: Create `deps.py` — Dependencies Container

**Files:**
- Create: `deps.py`
- Test: `tests/test_deps.py`

- [ ] **Step 1: Write the failing test for Dependencies creation**

```python
# tests/test_deps.py
"""Tests for the Dependencies container."""

import time
from deps import Dependencies


def test_dependencies_defaults():
    """Dependencies initializes with correct defaults."""
    deps = Dependencies(
        pool=None,
        doc_store=None,
        client=None,
        http=None,
    )
    assert deps.vector_store is None
    assert deps.sync_task is None
    assert deps.vector_init_task is None
    assert deps.last_sync_time is None
    assert deps.last_sync_error is None
    assert deps.sync_consecutive_failures == 0
    assert deps.sync_circuit_open is False
    assert isinstance(deps.server_start_time, float)
    assert deps.server_start_time <= time.time()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/test_deps.py::test_dependencies_defaults -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'deps'`

- [ ] **Step 3: Write the Dependencies dataclass**

```python
# deps.py
"""Dependency container for BDDK MCP Server."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg
    import httpx

    from client import BddkApiClient
    from doc_store import DocumentStore
    from vector_store import VectorStore


@dataclass
class Dependencies:
    """Shared state for all tool modules.

    Created once at startup, injected into tool modules via register().
    Tools access dependencies through closure capture.
    """

    pool: asyncpg.Pool | None
    doc_store: DocumentStore | None
    client: BddkApiClient | None
    http: httpx.AsyncClient | None
    vector_store: VectorStore | None = None
    sync_task: asyncio.Task | None = None
    vector_init_task: asyncio.Task | None = None

    # Health state
    last_sync_time: float | None = None
    last_sync_error: str | None = None
    sync_consecutive_failures: int = 0
    sync_circuit_open: bool = False
    server_start_time: float = field(default_factory=time.time)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/test_deps.py::test_dependencies_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add deps.py tests/test_deps.py
git commit -m "feat: add Dependencies container (deps.py)"
```

---

### Task 2: Create `tools/__init__.py` and `tools/admin.py`

Start with admin tools (`health_check`, `bddk_metrics`) — simplest module, validates the register pattern.

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/admin.py`
- Test: `tests/test_tools_admin.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools_admin.py
"""Tests for admin tool module registration."""

import time
from unittest.mock import MagicMock

from deps import Dependencies


def test_admin_register():
    """admin.register() adds health_check and bddk_metrics tools."""
    mcp = MagicMock()
    deps = Dependencies(
        pool=None, doc_store=None, client=None, http=None,
        server_start_time=time.time(),
    )
    from tools.admin import register
    register(mcp, deps)
    # Verify mcp.tool() was called (via decorator)
    assert mcp.tool.call_count >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/test_tools_admin.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools'`

- [ ] **Step 3: Create `tools/__init__.py`**

```python
# tools/__init__.py
"""BDDK MCP tool modules."""
```

- [ ] **Step 4: Create `tools/admin.py` — extract `health_check` and `bddk_metrics`**

```python
# tools/admin.py
"""Admin tools: health check and metrics."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from exceptions import BddkError, BddkStorageError
from metrics import metrics

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from deps import Dependencies

logger = logging.getLogger(__name__)


def register(mcp: FastMCP, deps: Dependencies) -> None:
    """Register admin tools with the MCP server."""

    @mcp.tool()
    async def health_check() -> str:
        """
        Check server health status.

        Returns uptime, cache status, store stats, and last sync time.
        """
        uptime_s = int(time.time() - deps.server_start_time)
        hours, remainder = divmod(uptime_s, 3600)
        minutes, seconds = divmod(remainder, 60)

        lines = ["**BDDK MCP Server Health**\n"]

        # Degraded states
        if deps.sync_circuit_open:
            lines.append("  Status: DEGRADED (sync circuit open after 10 consecutive failures)")
        elif deps.vector_store is None:
            lines.append("  Status: INITIALIZING (vector store loading)")
        else:
            lines.append("  Status: OK")

        lines.append(f"  Uptime: {hours}h {minutes}m {seconds}s")
        lines.append("  Backend: PostgreSQL + pgvector")

        if deps.last_sync_time:
            ago = int(time.time() - deps.last_sync_time)
            lines.append(f"  Last sync: {ago}s ago")
        else:
            lines.append("  Last sync: never")

        if deps.last_sync_error:
            lines.append(f"  Last sync error: {deps.last_sync_error}")

        # Cache status
        try:
            if deps.client:
                status = deps.client.cache_status()
                lines.append(f"  Cache items: {status['total_items']}")
                lines.append(f"  Cache valid: {status['cache_valid']}")
        except (RuntimeError, BddkError):
            lines.append("  Cache: unavailable")

        # Store status
        try:
            if deps.doc_store:
                st = await deps.doc_store.stats()
                lines.append(f"  Documents: {st.total_documents}")
        except (RuntimeError, BddkStorageError):
            lines.append("  Documents: unavailable")

        # Pool utilization
        if deps.pool:
            lines.append(f"  Pool: {deps.pool.get_size()}/{deps.pool.get_max_size()} connections ({deps.pool.get_idle_size()} idle)")

        sync_status = "running" if (deps.sync_task and not deps.sync_task.done()) else "idle"
        lines.append(f"  Sync status: {sync_status}")

        return "\n".join(lines)

    @mcp.tool()
    async def bddk_metrics() -> str:
        """
        Show server performance metrics.

        Includes request counts, average latency per tool, error rates, and cache statistics.
        """
        m = metrics.summary()

        lines = ["**BDDK MCP Server Metrics**\n"]
        lines.append(f"  Uptime: {m['uptime_seconds']}s")
        lines.append(f"  Total requests: {m['total_requests']}")
        lines.append(f"  Total errors: {m['total_errors']}")
        lines.append(f"  Cache hit rate: {m['cache_hit_rate']}%")
        lines.append(f"  Cache hits/misses: {m['cache_hits']}/{m['cache_misses']}")

        if m["tools"]:
            lines.append("\n**Per-Tool Metrics:**")
            lines.append(f"  {'Tool':<35} {'Requests':>10} {'Errors':>8} {'Avg ms':>10}")
            lines.append("  " + "-" * 65)
            for t in m["tools"]:
                lines.append(
                    f"  {t['tool']:<35} {t['requests']:>10} {t['errors']:>8} {t['avg_latency_ms']:>10.1f}"
                )

        return "\n".join(lines)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/test_tools_admin.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tools/__init__.py tools/admin.py tests/test_tools_admin.py
git commit -m "feat: extract admin tools (health_check, bddk_metrics) into tools/admin.py"
```

---

### Task 3: Create `tools/search.py` — Search Tools with LRU Cache

Extract `search_bddk_decisions`, `search_bddk_institutions`, `search_bddk_announcements`, and `search_document_store`. Replace O(n) cache eviction with `OrderedDict` LRU. Add batch `get_version_counts`.

**Files:**
- Create: `tools/search.py`
- Modify: `doc_store.py` — add `get_version_counts()` batch method
- Test: `tests/test_tools_search.py`

- [ ] **Step 1: Write failing test for batch `get_version_counts`**

```python
# tests/test_tools_search.py
"""Tests for search tool module."""

import pytest


@pytest.fixture
async def store_with_versions(doc_store, sample_doc):
    """Populate store with documents that have version history."""
    await doc_store.add_document(sample_doc)
    return doc_store


@pytest.mark.asyncio
async def test_get_version_counts_empty(doc_store):
    """Batch version count returns empty dict for unknown IDs."""
    result = await doc_store.get_version_counts(["unknown_1", "unknown_2"])
    assert result == {}


@pytest.mark.asyncio
async def test_get_version_counts_with_data(store_with_versions, sample_doc):
    """Batch version count returns counts for known documents."""
    result = await store_with_versions.get_version_counts([sample_doc.document_id, "unknown"])
    # sample_doc was just added so version_count depends on whether
    # add_document creates a version entry
    assert isinstance(result, dict)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/test_tools_search.py -v`
Expected: FAIL — `AttributeError: 'DocumentStore' object has no attribute 'get_version_counts'`

- [ ] **Step 3: Add `get_version_counts()` to `doc_store.py`**

Add this method after the existing `get_version_count()` method in `doc_store.py`:

```python
async def get_version_counts(self, doc_ids: list[str]) -> dict[str, tuple[int, str | None]]:
    """Batch version count for multiple documents. Returns {doc_id: (count, latest_date)}."""
    if not doc_ids:
        return {}
    rows = await self._pool.fetch(
        "SELECT document_id, COUNT(*) AS cnt, MAX(synced_at) AS latest "
        "FROM document_versions WHERE document_id = ANY($1) "
        "GROUP BY document_id",
        doc_ids,
    )
    result = {}
    for row in rows:
        latest = None
        if row["latest"]:
            latest = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["latest"]))
        result[row["document_id"]] = (row["cnt"], latest)
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/test_tools_search.py -v`
Expected: PASS

- [ ] **Step 5: Write failing test for LRU cache**

Add to `tests/test_tools_search.py`:

```python
from tools.search import _LRUCache


def test_lru_cache_eviction():
    """LRU cache evicts oldest entry when full."""
    cache = _LRUCache(max_size=2, ttl=60)
    cache.set("a", "val_a")
    cache.set("b", "val_b")
    cache.set("c", "val_c")  # should evict "a"
    assert cache.get("a") is None
    assert cache.get("b") == "val_b"
    assert cache.get("c") == "val_c"


def test_lru_cache_access_refreshes():
    """Accessing an entry moves it to end, preventing eviction."""
    cache = _LRUCache(max_size=2, ttl=60)
    cache.set("a", "val_a")
    cache.set("b", "val_b")
    cache.get("a")  # refresh "a"
    cache.set("c", "val_c")  # should evict "b", not "a"
    assert cache.get("a") == "val_a"
    assert cache.get("b") is None
    assert cache.get("c") == "val_c"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/test_tools_search.py::test_lru_cache_eviction -v`
Expected: FAIL — `ImportError: cannot import name '_LRUCache' from 'tools.search'`

- [ ] **Step 7: Create `tools/search.py`**

```python
# tools/search.py
"""Search tools: decisions, institutions, announcements, semantic search."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from typing import TYPE_CHECKING

import asyncpg

from client import _turkish_lower
from config import SEARCH_CACHE_MAX, SEARCH_CACHE_TTL
from data_sources import fetch_announcements, fetch_institutions
from exceptions import BddkError
from metrics import metrics
from models import BddkSearchRequest

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from deps import Dependencies

logger = logging.getLogger(__name__)


class _LRUCache:
    """O(1) LRU cache with TTL expiry."""

    def __init__(self, max_size: int = SEARCH_CACHE_MAX, ttl: float = SEARCH_CACHE_TTL):
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl

    def get(self, key: str) -> str | None:
        entry = self._data.get(key)
        if entry is None:
            return None
        ts, value = entry
        if (time.time() - ts) >= self._ttl:
            del self._data[key]
            return None
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: str) -> None:
        if key in self._data:
            self._data.move_to_end(key)
            self._data[key] = (time.time(), value)
            return
        if len(self._data) >= self._max_size:
            self._data.popitem(last=False)  # evict oldest
        self._data[key] = (time.time(), value)


def register(mcp: FastMCP, deps: Dependencies) -> None:
    """Register search tools with the MCP server."""

    _cache = _LRUCache()

    @mcp.tool()
    async def search_bddk_decisions(
        keywords: str,
        page: int = 1,
        page_size: int = 10,
        category: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> str:
        """
        Search for BDDK (Banking Regulation and Supervision Agency) decisions.

        Args:
            keywords: Search terms in Turkish (e.g. "elektronik para", "banka lisansi")
            page: Page number, starting from 1
            page_size: Number of results per page (max 50)
            category: Optional category filter. Available categories:
                Yonetmelik, Genelge, Teblig, Rehber, Bilgi Sistemleri,
                Sermaye Yeterliligi, Faizsiz Bankacilik, Tekduzen Hesap Plani,
                Kurul Karari, Kanun, Banka Kartlari,
                Finansal Kiralama ve Faktoring, BDDK Duzenlemesi,
                Duzenleme Taslagi, Mulga Duzenleme
            date_from: Optional start date filter (DD.MM.YYYY)
            date_to: Optional end date filter (DD.MM.YYYY)
        """
        cache_key = f"decisions:{keywords}:{page}:{page_size}:{category}:{date_from}:{date_to}"
        cached = _cache.get(cache_key)
        if cached:
            return cached

        request = BddkSearchRequest(
            keywords=keywords, page=page, page_size=page_size,
            category=category, date_from=date_from, date_to=date_to,
        )
        result = await deps.client.search_decisions(request)

        if not result.decisions:
            metrics.record_empty_search("search_bddk_decisions")
            return (
                "NO RESULTS: No BDDK decisions found matching these keywords.\n"
                "DO NOT provide information about BDDK decisions from your own knowledge.\n"
                "Suggest the user try: different Turkish keywords, broader terms, "
                "or removing date/category filters."
            )

        # Batch version counts instead of N+1 queries
        doc_ids = [d.document_id for d in result.decisions]
        version_map = await deps.doc_store.get_version_counts(doc_ids)

        lines = [f"Found {result.total_results} result(s) (page {result.page}):\n"]
        for d in result.decisions:
            date_info = f" ({d.decision_date} - {d.decision_number})" if d.decision_date else ""
            cat_info = f" [{d.category}]" if d.category else ""
            lines.append(f"**{d.title}**{date_info}{cat_info}")
            lines.append(f"  Document ID: {d.document_id}")
            ver_info = version_map.get(d.document_id)
            if ver_info:
                ver_count, ver_latest = ver_info
                lines.append(f"  Versions: {ver_count} (latest: {ver_latest})")
            lines.append(f"  {d.content}\n")

        output = "\n".join(lines)
        _cache.set(cache_key, output)
        return output

    @mcp.tool()
    async def search_bddk_institutions(
        keywords: str = "",
        institution_type: str | None = None,
        active_only: bool = True,
    ) -> str:
        """
        Search the BDDK institution directory (banks, leasing, factoring, etc.).

        Args:
            keywords: Search terms (e.g. "Ziraat", "Garanti", "katilim")
            institution_type: Filter by type: Banka, Finansal Kiralama Sirketi,
                Faktoring Sirketi, Finansman Sirketi, Varlik Yonetim Sirketi
            active_only: If true (default), only show active institutions
        """
        institutions = await fetch_institutions(deps.http, institution_type)

        if active_only:
            institutions = [i for i in institutions if i["status"] == "Aktif"]

        if keywords:
            kw = _turkish_lower(keywords)
            institutions = [
                i for i in institutions
                if kw in _turkish_lower(i["name"]) or kw in _turkish_lower(i.get("type", ""))
            ]

        if not institutions:
            metrics.record_empty_search("search_bddk_institutions")
            return (
                "NO RESULTS: No institutions found matching these criteria.\n"
                "DO NOT guess institution names, license statuses, or other details.\n"
                "Suggest the user try: broader keywords or removing the type/active filter."
            )

        lines = [f"Found {len(institutions)} institution(s):\n"]
        for i in institutions:
            status = f" ({i['status']})" if i["status"] != "Aktif" else ""
            website = f" -- {i['website']}" if i["website"] else ""
            lines.append(f"**{i['name']}**{status} [{i['type']}]{website}")
        return "\n".join(lines)

    @mcp.tool()
    async def search_bddk_announcements(
        keywords: str = "",
        category: str = "basin",
    ) -> str:
        """
        Search BDDK announcements and press releases.

        Args:
            keywords: Search terms in Turkish
            category: Announcement type: basin (press), mevzuat (regulation),
                insan kaynaklari (HR), veri (data publication), kurulus (institution).
                Use "tumu" or "all" to search across all categories.
        """
        cat_lower = _turkish_lower(category)

        cat_map: dict[str, list[int]] = {
            "basin": [39], "press": [39],
            "mevzuat": [40], "regul": [40],
            "insan": [41], "hr": [41],
            "veri": [42], "data": [42],
            "kurulus": [48], "institution": [48],
            "tumu": [39, 40, 41, 42, 48], "all": [39, 40, 41, 42, 48],
        }

        cat_ids = [39]
        for key, ids in cat_map.items():
            if key in cat_lower:
                cat_ids = ids
                break

        announcements: list[dict] = []
        for cat_id in cat_ids:
            announcements.extend(await fetch_announcements(deps.http, cat_id))

        if keywords:
            kw = _turkish_lower(keywords)
            announcements = [a for a in announcements if kw in _turkish_lower(a.get("title", ""))]

        if not announcements:
            metrics.record_empty_search("search_bddk_announcements")
            return (
                "NO RESULTS: No BDDK announcements found matching these criteria.\n"
                "DO NOT fabricate announcements or press releases.\n"
                "Suggest the user try: different keywords or a different category "
                "(basin, mevzuat, insan kaynaklari, veri, kurulus, or tumu for all)."
            )

        lines = [f"Found {len(announcements)} announcement(s):\n"]
        for a in announcements[:20]:
            date_info = f" ({a['date']})" if a.get("date") else ""
            lines.append(f"**{a['title']}**{date_info}")
            if a.get("url"):
                lines.append(f"  URL: {a['url']}")
            lines.append("")
        return "\n".join(lines)

    @mcp.tool()
    async def search_document_store(
        query: str,
        category: str | None = None,
        limit: int = 10,
    ) -> str:
        """
        Semantic search across all BDDK documents using vector embeddings.

        Uses pgvector with multilingual-e5-base model for Turkish legal text.

        Args:
            query: Natural language query in Turkish
            category: Optional category filter
            limit: Maximum results to return (default 10)
        """
        if deps.vector_store is None:
            return (
                "Semantic search is still initializing (loading embedding model). "
                "Please try again in a few seconds, or use search_bddk_decisions for keyword search."
            )

        cache_key = f"semantic:{query}:{category}:{limit}"
        cached = _cache.get(cache_key)
        if cached:
            return cached

        hits = await deps.vector_store.search(query, limit=limit, category=category)

        if not hits:
            metrics.record_empty_search("search_document_store")
            return (
                f"NO RESULTS: No documents found matching '{query}'.\n"
                "DO NOT provide information from your own knowledge about BDDK regulations.\n"
                "Suggest the user try: different Turkish keywords, broader terms, "
                "or removing the category filter."
            )

        lines = [f"Found {len(hits)} result(s) for '{query}':\n"]
        for h in hits:
            date_info = f" ({h['decision_date']})" if h.get("decision_date") else ""
            cat_info = f" [{h['category']}]" if h.get("category") else ""
            confidence = h.get("confidence", "unknown")
            confidence_icon = {"high": "\U0001f7e2", "medium": "\U0001f7e1", "low": "\U0001f534"}.get(confidence, "\u26aa")
            relevance = f" [{confidence_icon} {confidence} confidence, {h['relevance']:.1%}]"
            lines.append(f"**{h['title']}**{date_info}{cat_info}{relevance}")
            lines.append(f"  Document ID: {h['doc_id']}")
            if h.get("snippet"):
                lines.append(f"  ...{h['snippet'][:200]}...")
            lines.append("")

        low_count = sum(1 for h in hits if h.get("confidence") == "low")
        if low_count > 0:
            metrics.record_low_confidence_hit()
            lines.append(
                f"\n\u26a0\ufe0f {low_count} result(s) have low confidence. "
                "These may not be directly relevant. Verify before citing."
            )

        output = "\n".join(lines)
        _cache.set(cache_key, output)
        return output
```

- [ ] **Step 8: Run tests**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/test_tools_search.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add tools/search.py doc_store.py tests/test_tools_search.py
git commit -m "feat: extract search tools into tools/search.py with LRU cache and batch version counts"
```

---

### Task 4: Create `tools/documents.py` — Document Tools

Extract `get_bddk_document`, `get_document_history`, `document_store_stats`.

**Files:**
- Create: `tools/documents.py`

- [ ] **Step 1: Create `tools/documents.py`**

```python
# tools/documents.py
"""Document retrieval tools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from exceptions import BddkError, BddkStorageError
from metrics import metrics

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from deps import Dependencies

logger = logging.getLogger(__name__)


def register(mcp: FastMCP, deps: Dependencies) -> None:
    """Register document tools with the MCP server."""

    @mcp.tool()
    async def get_bddk_document(
        document_id: str,
        page_number: int = 1,
    ) -> str:
        """
        Retrieve a BDDK decision document as Markdown.

        Uses local pgvector store for instant retrieval.
        Falls back to PostgreSQL document store, then live fetch if not found locally.

        Args:
            document_id: The numeric document ID (from search results)
            page_number: Page of the markdown output (documents are split into 5000-char pages)
        """
        meta_title = document_id
        meta_date = ""
        meta_number = ""
        meta_category = ""
        source_url = ""
        for dec in deps.client._cache:
            if dec.document_id == document_id:
                meta_title = dec.title
                meta_date = dec.decision_date
                meta_number = dec.decision_number
                meta_category = dec.category
                source_url = dec.source_url or ""
                break

        def _build_header(page_num: int, total: int) -> str:
            return (
                f"## {meta_title}\n"
                f"- Document ID: {document_id}\n"
                f"- Decision Date: {meta_date or 'N/A'}\n"
                f"- Decision Number: {meta_number or 'N/A'}\n"
                f"- Category: {meta_category or 'N/A'}\n"
                f"- Source: {source_url or 'N/A'}\n"
                f"- Page: {page_num}/{total}\n"
                f"---\n"
                f"Use ONLY the text below. Do not add information not present in this document.\n\n"
            )

        # Try pgvector first (instant)
        if deps.vector_store is not None:
            try:
                page = await deps.vector_store.get_document_page(document_id, page_number)
                if page and page["content"] and "Invalid page" not in page["content"]:
                    return _build_header(page["page_number"], page["total_pages"]) + page["content"]
            except Exception as e:
                logger.debug("pgvector lookup failed for %s: %s", document_id, e)

        # Fallback to document store -> live fetch
        doc = await deps.client.get_document_markdown(document_id, page_number)
        return _build_header(doc.page_number, doc.total_pages) + doc.markdown_content

    @mcp.tool()
    async def get_document_history(document_id: str) -> str:
        """
        Get version history for a BDDK document.

        Args:
            document_id: The document ID (from search results)
        """
        history = await deps.doc_store.get_document_history(document_id)

        if not history:
            return f"No version history found for document {document_id}."

        lines = [f"**Version History for {document_id}** ({len(history)} version(s)):\n"]
        for v in history:
            lines.append(
                f"  v{v['version']} -- {v['synced_at']} "
                f"(hash: {v['content_hash'][:12]}..., {v['content_length']} chars)"
            )
        return "\n".join(lines)

    @mcp.tool()
    async def document_store_stats() -> str:
        """
        Show document store statistics for PostgreSQL and pgvector stores.
        """
        lines = ["**Document Store Statistics**\n"]

        # pgvector stats
        if deps.vector_store is not None:
            try:
                vs_stats = await deps.vector_store.stats()
                lines.append("**pgvector (Vector Store):**")
                lines.append(f"  Documents: {vs_stats['total_documents']}")
                lines.append(f"  Chunks: {vs_stats['total_chunks']}")
                lines.append(f"  Embedding model: {vs_stats['embedding_model']}")
                if vs_stats.get("categories"):
                    lines.append("  Categories:")
                    for cat, count in vs_stats["categories"].items():
                        lines.append(f"    {cat}: {count}")
            except Exception as e:
                lines.append(f"  pgvector: unavailable ({e})")
        else:
            lines.append("  pgvector: initializing")

        # PostgreSQL document stats
        try:
            if deps.doc_store:
                st = await deps.doc_store.stats()
                lines.append("\n**PostgreSQL (Document Store):**")
                lines.append(f"  Documents: {st.total_documents}")
                lines.append(f"  Size: {st.total_size_mb} MB")
        except (RuntimeError, BddkStorageError) as e:
            lines.append(f"  PostgreSQL: unavailable ({e})")

        return "\n".join(lines)
```

- [ ] **Step 2: Verify import works**

Run: `cd /home/cagatay/bddk-mcp && python -c "from tools.documents import register; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tools/documents.py
git commit -m "feat: extract document tools into tools/documents.py"
```

---

### Task 5: Create `tools/bulletin.py` — Bulletin & Data Tools

Extract `get_bddk_bulletin`, `get_bddk_bulletin_snapshot`, `get_bddk_monthly`, `bddk_cache_status`.

**Files:**
- Create: `tools/bulletin.py`

- [ ] **Step 1: Create `tools/bulletin.py`**

```python
# tools/bulletin.py
"""Bulletin and data tools."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from config import (
    validate_column,
    validate_currency,
    validate_metric_id,
    validate_month,
    validate_table_no,
    validate_year,
)
from data_sources import fetch_bulletin_snapshot, fetch_weekly_bulletin

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from deps import Dependencies

logger = logging.getLogger(__name__)


def register(mcp: FastMCP, deps: Dependencies) -> None:
    """Register bulletin tools with the MCP server."""

    @mcp.tool()
    async def get_bddk_bulletin(
        metric_id: str = "1.0.1",
        currency: str = "TRY",
        column: str = "1",
        date: str = "",
        days: int = 90,
    ) -> str:
        """
        Get weekly banking sector bulletin time-series data from BDDK.

        Args:
            metric_id: Metric ID. Common IDs:
                1.0.1=Toplam Krediler, 1.0.2=Tuketici Kredileri,
                1.0.4=Konut Kredileri, 1.0.8=Bireysel Kredi Kartlari,
                1.0.10=Ticari Krediler.
            currency: TRY or USD
            column: 1=TP (TL), 2=YP (Foreign Currency), 3=Toplam
            date: Specific date (DD.MM.YYYY), empty for latest
            days: Number of days of history (default 90)
        """
        try:
            validate_metric_id(metric_id)
            validate_currency(currency, "weekly")
            validate_column(column)
        except ValueError as e:
            return f"Validation error: {e}"

        data = await fetch_weekly_bulletin(deps.http, metric_id, currency, days, date, column)

        if "error" in data:
            return f"Error fetching bulletin: {data['error']}"

        lines = [f"**{data.get('title', 'BDDK Weekly Bulletin')}** ({data['currency']})\n"]
        dates = data.get("dates", [])
        values = data.get("values", [])

        if dates and values:
            for d, v in zip(dates[-10:], values[-10:], strict=False):
                lines.append(f"  {d}: {v}")
        else:
            lines.append("No data returned for the given parameters.")

        return "\n".join(lines)

    @mcp.tool()
    async def get_bddk_bulletin_snapshot() -> str:
        """
        Get the latest weekly bulletin snapshot -- all metrics with current TP/YP values.
        """
        rows = await fetch_bulletin_snapshot(deps.http)

        if not rows:
            return "No bulletin data available."

        lines = ["**BDDK Weekly Bulletin -- Latest Snapshot**\n"]
        lines.append(f"{'#':<4} {'Metric':<50} {'TP':>15} {'YP':>15} {'ID'}")
        lines.append("-" * 100)
        for r in rows:
            lines.append(
                f"{r['row_number']:<4} {r['name']:<50} {r['tp']:>15} {r['yp']:>15} {r['metric_id']}"
            )
        return "\n".join(lines)

    @mcp.tool()
    async def get_bddk_monthly(
        table_no: int = 1,
        year: int = 2025,
        month: int = 12,
        currency: str = "TL",
        party_code: str = "10001",
    ) -> str:
        """
        Get BDDK monthly banking sector data (more detailed than weekly bulletin).

        Args:
            table_no: Table number (1-17)
            year: Year (e.g. 2025)
            month: Month (1-12)
            currency: TL or USD
            party_code: Bank group code. 10001=Sektor, 10002=Mevduat Bankalari
        """
        try:
            validate_table_no(table_no)
            validate_year(year)
            validate_month(month)
            validate_currency(currency, "monthly")
        except ValueError as e:
            return f"Validation error: {e}"

        from data_sources import fetch_monthly_bulletin

        result = await fetch_monthly_bulletin(deps.http, table_no, year, month, currency, party_code)

        if "error" in result:
            return f"Error: {result['error']}"

        lines = [f"**{result.get('title', 'BDDK Aylik Bulten')}**\n"]
        lines.append(f"Donem: {month}/{year} | Para Birimi: {currency}\n")

        rows = result.get("rows", [])
        if not rows:
            lines.append("Bu parametreler icin veri bulunamadi.")
        else:
            lines.append(f"{'Kalem':<55} {'TP':>15} {'YP':>15} {'Toplam':>15}")
            lines.append("-" * 105)
            for r in rows:
                lines.append(
                    f"{r['name']:<55} {r.get('tp', ''):>15} {r.get('yp', ''):>15} {r.get('total', ''):>15}"
                )

        return "\n".join(lines)

    @mcp.tool()
    async def bddk_cache_status() -> str:
        """
        Show BDDK cache statistics: total items, age, categories, and any page errors.
        """
        status = deps.client.cache_status()

        lines = ["**BDDK Cache Status**\n"]
        lines.append(f"  Total items: {status['total_items']}")
        lines.append(f"  Cache valid: {status['cache_valid']}")
        if status["cache_age_seconds"] is not None:
            mins = status["cache_age_seconds"] // 60
            lines.append(f"  Cache age: {mins} min ({status['cache_age_seconds']}s)")
        lines.append(f"  TTL: {status['ttl_seconds']}s")

        if status["categories"]:
            lines.append("\n**Categories:**")
            for cat, count in status["categories"].items():
                lines.append(f"  {cat}: {count}")

        if status["page_errors"]:
            lines.append("\n**Page Errors:**")
            for page_id, err in status["page_errors"].items():
                lines.append(f"  Page {page_id}: {err}")

        return "\n".join(lines)
```

- [ ] **Step 2: Verify import works**

Run: `cd /home/cagatay/bddk-mcp && python -c "from tools.bulletin import register; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tools/bulletin.py
git commit -m "feat: extract bulletin tools into tools/bulletin.py"
```

---

### Task 6: Create `tools/analytics.py` — Analytics Tools

Extract `analyze_bulletin_trends`, `get_regulatory_digest`, `compare_bulletin_metrics`, `check_bddk_updates`.

**Files:**
- Create: `tools/analytics.py`

- [ ] **Step 1: Create `tools/analytics.py`**

```python
# tools/analytics.py
"""Analytics tools: trends, digest, comparison, update checking."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from analytics import analyze_trends, build_digest, check_updates, compare_metrics
from config import validate_column, validate_currency, validate_metric_id

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from deps import Dependencies

logger = logging.getLogger(__name__)


def register(mcp: FastMCP, deps: Dependencies) -> None:
    """Register analytics tools with the MCP server."""

    @mcp.tool()
    async def analyze_bulletin_trends(
        metric_id: str = "1.0.1",
        currency: str = "TRY",
        column: str = "1",
        lookback_weeks: int = 12,
    ) -> str:
        """
        Analyze trends in BDDK weekly bulletin data with week-over-week changes.

        Args:
            metric_id: Metric ID (e.g. 1.0.1=Toplam Krediler)
            currency: TRY or USD
            column: 1=TP (TL), 2=YP (Foreign Currency), 3=Toplam
            lookback_weeks: Number of weeks to analyze (default 12)
        """
        try:
            validate_metric_id(metric_id)
            validate_currency(currency, "weekly")
            validate_column(column)
        except ValueError as e:
            return f"Validation error: {e}"

        result = await analyze_trends(deps.http, metric_id, currency, column, lookback_weeks)

        if "error" in result:
            return f"Error: {result['error']}"

        lines = [f"**Trend Analizi: {result['title']}**\n"]
        lines.append(result["narrative"])
        lines.append("")
        lines.append(f"  Guncel ({result['current_date']}): {result['current']:,.2f}")
        lines.append(f"  Onceki ({result['previous_date']}): {result['previous']:,.2f}")
        lines.append(f"  Haftalik degisim: {result['wow_change']:+,.2f} (%{result['wow_pct']:+.2f})")
        lines.append(f"  Donem ortalamasi: {result['avg']:,.2f}")
        lines.append(f"  Donem min: {result['min']:,.2f} ({result['min_date']})")
        lines.append(f"  Donem max: {result['max']:,.2f} ({result['max_date']})")
        lines.append(f"  Trend: {result['trend_direction']}")
        lines.append(f"  Veri noktasi: {result['data_points']}")
        return "\n".join(lines)

    @mcp.tool()
    async def get_regulatory_digest(period: str = "month") -> str:
        """
        Get a digest of recent BDDK regulatory changes.

        Args:
            period: Time period -- week (7 days), month (30 days), quarter (90 days)
        """
        period_map = {"week": 7, "month": 30, "quarter": 90}
        days = period_map.get(period, 30)

        await deps.client.ensure_cache()
        digest = await build_digest(deps.http, deps.client._cache, days)

        lines = [f"**BDDK Duzenleyici Ozet -- Son {days} Gun**\n"]
        lines.append(digest["narrative"])
        lines.append("")

        if digest["decisions_by_category"]:
            lines.append("**Kararlar (kategoriye gore):**")
            for cat, count in sorted(digest["decisions_by_category"].items(), key=lambda x: -x[1]):
                lines.append(f"  {cat}: {count}")
            lines.append("")

        if digest["new_decisions"]:
            lines.append("**Son Kararlar:**")
            for d in digest["new_decisions"][:10]:
                date = d.get("decision_date", "")
                lines.append(f"  - {d['title']} ({date}) [{d.get('category', '')}]")
            lines.append("")

        if digest["announcements"]:
            lines.append(f"**Duyurular ({len(digest['announcements'])}):**")
            for a in digest["announcements"][:10]:
                lines.append(f"  - {a['title']} ({a.get('date', '')})")
            lines.append("")

        if digest["bulletin_snapshot"]:
            lines.append("**Bulten Ozet (ilk 5 metrik):**")
            for r in digest["bulletin_snapshot"]:
                lines.append(f"  {r['name']}: TP={r['tp']}, YP={r['yp']}")

        return "\n".join(lines)

    @mcp.tool()
    async def compare_bulletin_metrics(
        metric_ids: str = "1.0.1,1.0.2",
        currency: str = "TRY",
        column: str = "1",
        days: int = 90,
    ) -> str:
        """
        Compare multiple BDDK bulletin metrics side-by-side.

        Args:
            metric_ids: Comma-separated metric IDs (e.g. "1.0.1,1.0.2,1.0.4")
            currency: TRY or USD
            column: 1=TP, 2=YP, 3=Toplam
            days: Days of history (default 90)
        """
        ids = [m.strip() for m in metric_ids.split(",") if m.strip()]
        if not ids:
            return "Please provide at least one metric ID."

        try:
            for mid in ids:
                validate_metric_id(mid)
            validate_currency(currency, "weekly")
            validate_column(column)
        except ValueError as e:
            return f"Validation error: {e}"

        result = await compare_metrics(deps.http, ids, currency, column, days)

        col_label = {"1": "TP", "2": "YP", "3": "Toplam"}.get(column, column)
        lines = [f"**Metrik Karsilastirmasi** ({currency}, {col_label})\n"]
        lines.append(f"{'Metrik':<55} {'Guncel':>15} {'Haftalik %':>12}")
        lines.append("-" * 85)

        for m in result["metrics"]:
            if "error" in m:
                lines.append(f"{m['metric_id']:<55} {'HATA':>15} {'-':>12}")
            else:
                title = m["title"][:55]
                lines.append(f"{title:<55} {m['current']:>15,.2f} {m['wow_pct']:>+11.2f}%")

        return "\n".join(lines)

    @mcp.tool()
    async def check_bddk_updates() -> str:
        """
        Check for new BDDK announcements since last check.
        """
        known_urls: set[str] = set()
        if hasattr(deps.client, "_known_announcements"):
            known_urls = deps.client._known_announcements
        else:
            from data_sources import fetch_announcements as _fa

            for cat_id in [39, 40, 41, 42, 48]:
                anns = await _fa(deps.http, cat_id)
                for a in anns:
                    if a.get("url"):
                        known_urls.add(a["url"])
            deps.client._known_announcements = known_urls
            return (
                f"Baseline olusturuldu: {len(known_urls)} duyuru biliniyor. "
                "Bir sonraki cagirida yeni duyurular tespit edilecek."
            )

        result = await check_updates(deps.http, deps.client._cache, known_urls)

        for a in result.get("new_announcements", []):
            if a.get("url"):
                known_urls.add(a["url"])

        if not result["new_announcements"]:
            return "Yeni duyuru yok. Her sey guncel."

        lines = [f"**{result['new_announcements_count']} Yeni Duyuru Tespit Edildi!**\n"]
        for a in result["new_announcements"]:
            date = a.get("date", "")
            lines.append(f"  - {a['title']} ({date})")
            if a.get("url"):
                lines.append(f"    {a['url']}")
        return "\n".join(lines)
```

- [ ] **Step 2: Verify import works**

Run: `cd /home/cagatay/bddk-mcp && python -c "from tools.analytics import register; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tools/analytics.py
git commit -m "feat: extract analytics tools into tools/analytics.py"
```

---

### Task 7: Create `tools/sync.py` — Sync Tools with Circuit Breaker

Extract `refresh_bddk_cache`, `sync_bddk_documents`, `trigger_startup_sync`, and `_migrate_to_pgvector`. Add circuit breaker pattern and sync timeout.

**Files:**
- Create: `tools/sync.py`
- Test: `tests/test_circuit_breaker.py`

- [ ] **Step 1: Write failing test for circuit breaker**

```python
# tests/test_circuit_breaker.py
"""Tests for sync circuit breaker."""

import time
from deps import Dependencies


def test_circuit_opens_after_threshold():
    """Circuit opens after 10 consecutive failures."""
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    from tools.sync import _record_sync_failure, CIRCUIT_BREAKER_THRESHOLD

    for _ in range(CIRCUIT_BREAKER_THRESHOLD):
        _record_sync_failure(deps, "test error")

    assert deps.sync_circuit_open is True
    assert deps.sync_consecutive_failures == CIRCUIT_BREAKER_THRESHOLD
    assert deps.last_sync_error == "test error"


def test_circuit_resets_on_success():
    """Successful sync resets the circuit."""
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    deps.sync_consecutive_failures = 5
    deps.sync_circuit_open = True

    from tools.sync import _record_sync_success

    _record_sync_success(deps)
    assert deps.sync_consecutive_failures == 0
    assert deps.sync_circuit_open is False
    assert deps.last_sync_time is not None
    assert deps.last_sync_error is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/test_circuit_breaker.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Create `tools/sync.py`**

```python
# tools/sync.py
"""Sync tools: cache refresh, document sync, pgvector migration, circuit breaker."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from config import PREFER_NOUGAT
from exceptions import BddkError

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from deps import Dependencies

logger = logging.getLogger(__name__)

CIRCUIT_BREAKER_THRESHOLD = 10
STARTUP_SYNC_TIMEOUT = 300  # 5 minutes
MIGRATION_TIMEOUT = 600  # 10 minutes


def _record_sync_failure(deps: Dependencies, error: str) -> None:
    """Record a sync failure and open circuit if threshold reached."""
    deps.sync_consecutive_failures += 1
    deps.last_sync_error = error
    if deps.sync_consecutive_failures >= CIRCUIT_BREAKER_THRESHOLD:
        deps.sync_circuit_open = True
        logger.error("Sync circuit breaker OPEN after %d consecutive failures", CIRCUIT_BREAKER_THRESHOLD)


def _record_sync_success(deps: Dependencies) -> None:
    """Record a successful sync and reset circuit."""
    deps.sync_consecutive_failures = 0
    deps.sync_circuit_open = False
    deps.last_sync_time = time.time()
    deps.last_sync_error = None


async def _migrate_to_pgvector(deps: Dependencies) -> str:
    """Migrate documents from document store to pgvector if needed.

    Returns a status string for reporting.
    """
    if deps.vector_store is None:
        return "vector store not initialized"

    vs_stats = await deps.vector_store.stats()
    sqlite_stats = await deps.doc_store.stats()

    if vs_stats["total_documents"] >= sqlite_stats.total_documents * 0.9:
        logger.info(
            "pgvector has %d/%d documents, skipping migration",
            vs_stats["total_documents"], sqlite_stats.total_documents,
        )
        return f"pgvector up to date ({vs_stats['total_documents']}/{sqlite_stats.total_documents})"

    logger.info(
        "pgvector incomplete (%d/%d) -- migrating...",
        vs_stats["total_documents"], sqlite_stats.total_documents,
    )

    start = time.time()
    docs = await deps.doc_store.list_documents(limit=2000)

    # Batch existence check instead of per-document queries
    all_doc_ids = [m["document_id"] for m in docs]
    existing_ids = set()
    if all_doc_ids:
        rows = await deps.pool.fetch(
            "SELECT DISTINCT doc_id FROM document_chunks WHERE doc_id = ANY($1)",
            all_doc_ids,
        )
        existing_ids = {r["doc_id"] for r in rows}

    missing = [m for m in docs if m["document_id"] not in existing_ids]
    migrated = 0
    total_chunks = 0

    for i, meta in enumerate(missing):
        if time.time() - start > MIGRATION_TIMEOUT:
            logger.warning("Migration timeout after %d docs", migrated)
            break

        doc = await deps.doc_store.get_document(meta["document_id"])
        if not doc or not doc.markdown_content:
            continue

        chunks = await deps.vector_store.add_document(
            doc_id=doc.document_id, title=doc.title,
            content=doc.markdown_content, category=doc.category,
            decision_date=doc.decision_date, decision_number=doc.decision_number,
            source_url=doc.source_url,
        )
        total_chunks += chunks
        migrated += 1

        if (i + 1) % 100 == 0:
            logger.info("pgvector migration: %d/%d docs", i + 1, len(missing))

    elapsed = time.time() - start
    logger.info("pgvector migration: %d docs, %d chunks, %.1fs", migrated, total_chunks, elapsed)
    return f"migrated {migrated} docs, {total_chunks} chunks in {elapsed:.1f}s"


async def startup_sync(deps: Dependencies) -> None:
    """Auto-sync documents on startup with circuit breaker and timeout."""
    if deps.sync_circuit_open:
        logger.warning("Sync circuit is open, skipping startup sync")
        return

    logger.info("Startup sync started...")
    try:
        async with asyncio.timeout(STARTUP_SYNC_TIMEOUT):
            from doc_sync import DocumentSyncer

            await deps.client.ensure_cache()
            logger.info("Using existing cache: %d documents", len(deps.client._cache))
            if not deps.client._cache:
                logger.warning("Cache is empty -- skipping (run refresh_bddk_cache first)")
                return

            st = await deps.doc_store.stats()
            cache_size = len(deps.client._cache)

            # Phase 1: Download missing documents
            if st.total_documents < cache_size * 0.9:
                logger.info("Document store incomplete (%d/%d) -- downloading...", st.total_documents, cache_size)
                items = [d.model_dump() for d in deps.client._cache]
                async with DocumentSyncer(deps.doc_store, prefer_nougat=PREFER_NOUGAT, http=deps.http) as syncer:
                    report = await syncer.sync_all(items, concurrency=10, force=False)
                logger.info("Sync: %d downloaded, %d failed, %.1fs", report.downloaded, report.failed, report.elapsed_seconds)
            else:
                logger.info("Document store has %d/%d documents, OK", st.total_documents, cache_size)

            # Phase 2: Migrate to pgvector
            await _migrate_to_pgvector(deps)

            _record_sync_success(deps)

    except TimeoutError:
        msg = f"Startup sync timed out after {STARTUP_SYNC_TIMEOUT}s"
        logger.warning(msg)
        _record_sync_failure(deps, msg)
    except (BddkError, RuntimeError, OSError) as e:
        logger.error("Startup sync failed: %s", e)
        _record_sync_failure(deps, str(e))


def register(mcp: FastMCP, deps: Dependencies) -> None:
    """Register sync tools with the MCP server."""

    @mcp.tool()
    async def refresh_bddk_cache() -> str:
        """
        Force re-scrape BDDK website and update the PostgreSQL decision cache.
        """
        count = await deps.client.refresh_cache()
        return f"BDDK cache refreshed: {count} decisions/regulations scraped and saved to PostgreSQL."

    @mcp.tool()
    async def sync_bddk_documents(
        force: bool = False,
        document_id: str | None = None,
        concurrency: int = 5,
    ) -> str:
        """
        Sync BDDK documents to local storage.

        Args:
            force: Re-download all documents even if already cached
            document_id: Sync a single document by ID
            concurrency: Number of parallel downloads (default 5)
        """
        from doc_sync import DocumentSyncer

        await deps.client.ensure_cache()

        single_report = None
        sync_report = None

        async with DocumentSyncer(deps.doc_store, prefer_nougat=PREFER_NOUGAT, http=deps.http) as syncer:
            if document_id:
                source_url = ""
                title = document_id
                category = ""
                for dec in deps.client._cache:
                    if dec.document_id == document_id:
                        source_url = dec.source_url
                        title = dec.title
                        category = dec.category
                        break

                result = await syncer.sync_document(
                    doc_id=document_id, title=title,
                    category=category, source_url=source_url, force=force,
                )
                status = "OK" if result.success else "FAIL"
                single_report = f"[{status}] {result.document_id}: {result.method or result.error}"
            else:
                items = [d.model_dump() for d in deps.client._cache]
                report = await syncer.sync_all(items, concurrency=concurrency, force=force)
                sync_report = (
                    f"**Sync Report**\n"
                    f"  Total: {report.total}\n"
                    f"  Downloaded: {report.downloaded}\n"
                    f"  Skipped: {report.skipped}\n"
                    f"  Failed: {report.failed}\n"
                    f"  Time: {report.elapsed_seconds}s"
                )

        # Migrate to pgvector
        embed_report = ""
        try:
            status = await _migrate_to_pgvector(deps)
            if deps.vector_store:
                vs_stats = await deps.vector_store.stats()
                embed_report = (
                    f"\n\n**Embedding Report**\n"
                    f"  Documents: {vs_stats['total_documents']}\n"
                    f"  Chunks: {vs_stats['total_chunks']}"
                )
        except Exception as e:
            embed_report = f"\n\n**Embedding:** failed ({e})"

        if single_report:
            return single_report + embed_report
        return sync_report + embed_report

    @mcp.tool()
    async def trigger_startup_sync() -> str:
        """
        Manually trigger document sync if auto-sync is still running or was skipped.
        Resets the circuit breaker if it was open.
        """
        if deps.sync_task and not deps.sync_task.done():
            return "Sync is already running in background."

        # Reset circuit breaker on manual trigger
        deps.sync_circuit_open = False
        deps.sync_consecutive_failures = 0

        st = await deps.doc_store.stats()

        embed_report = ""
        try:
            status = await _migrate_to_pgvector(deps)
            if deps.vector_store:
                vs_stats = await deps.vector_store.stats()
                embed_report = (
                    f"\n  Vector documents: {vs_stats['total_documents']}"
                    f"\n  Vector chunks: {vs_stats['total_chunks']}"
                )
        except Exception as e:
            embed_report = f"\n  Embedding migration failed: {e}"

        return f"Store has {st.total_documents} documents.{embed_report}"
```

- [ ] **Step 4: Run tests**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/test_circuit_breaker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/sync.py tests/test_circuit_breaker.py
git commit -m "feat: extract sync tools into tools/sync.py with circuit breaker"
```

---

### Task 8: Shared httpx client in `BddkApiClient`

Modify `BddkApiClient.__init__` to accept an external `httpx.AsyncClient` and track ownership.

**Files:**
- Modify: `client.py:175-194`

- [ ] **Step 1: Write failing test**

```python
# tests/test_shared_http.py
"""Test shared httpx client injection."""

import httpx
import pytest

from conftest import MockPool


@pytest.mark.asyncio
async def test_client_accepts_external_http():
    """BddkApiClient uses injected http client when provided."""
    from client import BddkApiClient

    external_http = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
    client = BddkApiClient(pool=MockPool(), http=external_http)

    assert client._http is external_http
    assert client._owns_http is False

    await external_http.aclose()


@pytest.mark.asyncio
async def test_client_creates_own_http():
    """BddkApiClient creates its own http client when none provided."""
    from client import BddkApiClient

    client = BddkApiClient(pool=MockPool())
    assert client._http is not None
    assert client._owns_http is True
    await client.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/test_shared_http.py -v`
Expected: FAIL — `TypeError: BddkApiClient.__init__() got an unexpected keyword argument 'http'`

- [ ] **Step 3: Modify `BddkApiClient.__init__` in `client.py`**

Change the constructor to accept an optional `http` parameter:

```python
def __init__(
    self,
    pool: asyncpg.Pool,
    request_timeout: float = REQUEST_TIMEOUT,
    doc_store: DocumentStore | None = None,
    http: httpx.AsyncClient | None = None,
) -> None:
    self._pool = pool
    self._owns_http = http is None
    if http is not None:
        self._http = http
    else:
        self._http = httpx.AsyncClient(
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            },
            timeout=httpx.Timeout(request_timeout),
            follow_redirects=True,
        )
    self._md = MarkItDown()
    self._doc_store = doc_store
    self._cache: list[BddkDecisionSummary] = []
    self._cache_timestamp: float = 0.0
    self._page_errors: dict[int, str] = {}
```

Also update `close()`:

```python
async def close(self) -> None:
    """Close the underlying HTTP session (only if we own it)."""
    if self._owns_http:
        await self._http.aclose()
        logger.info("BddkApiClient session closed")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/test_shared_http.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/ -v --timeout=30`
Expected: All existing tests still pass

- [ ] **Step 6: Commit**

```bash
git add client.py tests/test_shared_http.py
git commit -m "feat: BddkApiClient accepts shared httpx client via http parameter"
```

---

### Task 9: Shared httpx client in `DocumentSyncer`

Modify `DocumentSyncer` to accept an external `httpx.AsyncClient`.

**Files:**
- Modify: `doc_sync.py`

- [ ] **Step 1: Read `DocumentSyncer.__init__` to find exact lines**

Run: `cd /home/cagatay/bddk-mcp && grep -n "class DocumentSyncer" doc_sync.py`
and: `cd /home/cagatay/bddk-mcp && sed -n '/class DocumentSyncer/,/def __aenter__/p' doc_sync.py`

- [ ] **Step 2: Modify `DocumentSyncer.__init__`**

Add `http: httpx.AsyncClient | None = None` parameter. If provided, use it and set `self._owns_http = False`. If not, create one as before and set `self._owns_http = True`.

- [ ] **Step 3: Update `__aexit__` to only close if owned**

```python
async def __aexit__(self, *exc) -> None:
    if self._owns_http:
        await self._http.aclose()
```

- [ ] **Step 4: Run existing sync tests**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/ -k sync -v --timeout=30`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add doc_sync.py
git commit -m "feat: DocumentSyncer accepts shared httpx client"
```

---

### Task 10: Cache save without full wipe

Replace `DELETE ALL + N INSERTs` in `_save_cache_to_db` with bulk upsert.

**Files:**
- Modify: `client.py` — `_save_cache_to_db()` method

- [ ] **Step 1: Read current `_save_cache_to_db` implementation**

Run: `cd /home/cagatay/bddk-mcp && grep -n "_save_cache_to_db" client.py`

- [ ] **Step 2: Replace with upsert using `executemany`**

Replace the `DELETE FROM decision_cache` + loop with:

```python
async def _save_cache_to_db(self) -> None:
    """Persist cache to PostgreSQL using upsert (no DELETE ALL)."""
    try:
        async with self._pool.acquire() as conn:
            now = time.time()
            args_list = [
                (d.document_id, d.title, d.content, d.decision_date,
                 d.decision_number, d.category, d.source_url or "", now)
                for d in self._cache
            ]
            await conn.executemany(
                """
                INSERT INTO decision_cache
                    (document_id, title, content, decision_date, decision_number,
                     category, source_url, cached_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT(document_id) DO UPDATE SET
                    title=EXCLUDED.title, content=EXCLUDED.content,
                    decision_date=EXCLUDED.decision_date,
                    decision_number=EXCLUDED.decision_number,
                    category=EXCLUDED.category, source_url=EXCLUDED.source_url,
                    cached_at=EXCLUDED.cached_at
                """,
                args_list,
            )
        self._cache_timestamp = now
        logger.debug("Cache saved to PostgreSQL: %d items", len(self._cache))
    except (asyncpg.PostgresError, OSError) as e:
        logger.error("Failed to save cache to PostgreSQL: %s", e)
```

- [ ] **Step 3: Run existing tests**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/ -v --timeout=30`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add client.py
git commit -m "fix: cache save uses upsert instead of DELETE ALL + N INSERTs"
```

---

### Task 11: Concurrent hybrid search and thread executor

Modify `VectorStore` to run vector and FTS searches concurrently, and run embedding/reranking in thread executor.

**Files:**
- Modify: `vector_store.py` — `_hybrid_search`, `_embed`, `_rerank`

- [ ] **Step 1: Make `_hybrid_search` use `asyncio.gather`**

In `vector_store.py`, change `_hybrid_search` (line ~490):

```python
# BEFORE (sequential):
vector_hits = await self._vector_search(query, limit=50, category=category, fetch_limit=100)
fts_hits = await self._fts_search(query, limit=50, category=category)

# AFTER (concurrent):
vector_hits, fts_hits = await asyncio.gather(
    self._vector_search(query, limit=50, category=category, fetch_limit=100),
    self._fts_search(query, limit=50, category=category),
)
```

Add `import asyncio` to top of file if not already present.

- [ ] **Step 2: Make `_embed` async with thread executor**

```python
async def _embed(self, texts: list[str], prefix: str = "passage") -> list[list[float]]:
    """Generate embeddings in a thread to avoid blocking the event loop."""
    self._ensure_embeddings()
    prefixed = [f"{prefix}: {t}" for t in texts]
    loop = asyncio.get_running_loop()
    embeddings = await loop.run_in_executor(
        None, self._embed_fn.encode, prefixed, # positional args
    )
    return embeddings.tolist()
```

Note: This changes `_embed` from sync to async. All callers must be updated to `await` it.

- [ ] **Step 3: Update all `_embed` callers to use `await`**

In `add_document` (line ~252):

```python
# BEFORE:
embeddings = self._embed(chunks)

# AFTER:
embeddings = await self._embed(chunks)
```

In `_vector_search` (line ~386):

```python
# BEFORE:
query_embedding = self._embed([query], prefix="query")[0]

# AFTER:
query_embedding = (await self._embed([query], prefix="query"))[0]
```

- [ ] **Step 4: Make `_rerank` async with thread executor**

```python
async def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
    """Re-rank candidates using a cross-encoder model in a thread."""
    if not candidates:
        return candidates

    self._ensure_reranker()
    pairs = [(query, c["snippet"]) for c in candidates]

    loop = asyncio.get_running_loop()
    scores = await loop.run_in_executor(None, self._rerank_fn.predict, pairs)

    for candidate, score in zip(candidates, scores):
        candidate["rerank_score"] = float(score)
        import math as _math
        candidate["relevance"] = round(1.0 / (1.0 + _math.exp(-float(score))), 4)

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
```

Update caller in `_hybrid_search`:

```python
# BEFORE:
fused[:top_n] = self._rerank(query, fused[:top_n])

# AFTER:
fused[:top_n] = await self._rerank(query, fused[:top_n])
```

- [ ] **Step 5: Run tests**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/ -v --timeout=60`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add vector_store.py
git commit -m "perf: concurrent hybrid search + thread executor for embedding/reranking"
```

---

### Task 12: Bulk chunk insertion

Replace per-chunk INSERT loop with `executemany`.

**Files:**
- Modify: `vector_store.py` — `add_document()`

- [ ] **Step 1: Replace INSERT loop with `executemany`**

In `add_document` (around line 255-273), replace:

```python
# BEFORE:
for i, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True)):
    vec_str = "[" + ",".join(str(v) for v in emb) + "]"
    await conn.execute(
        """INSERT INTO document_chunks (...) VALUES (...)""",
        doc_id, i, title, ...
    )

# AFTER:
args_list = []
for i, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True)):
    vec_str = "[" + ",".join(str(v) for v in emb) + "]"
    args_list.append((
        doc_id, i, title, category, decision_date,
        decision_number, source_url, len(chunks), total_pages,
        content_hash, chunk, vec_str,
    ))

await conn.executemany(
    """
    INSERT INTO document_chunks (
        doc_id, chunk_index, title, category, decision_date,
        decision_number, source_url, total_chunks, total_pages,
        content_hash, chunk_text, embedding
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::vector)
    """,
    args_list,
)
```

- [ ] **Step 2: Run tests**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/ -v --timeout=60`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add vector_store.py
git commit -m "perf: bulk chunk insertion with executemany"
```

---

### Task 13: Targeted page retrieval

Replace `get_document_page` fetching ALL chunks with targeted chunk fetching.

**Files:**
- Modify: `vector_store.py` — `get_document_page()`

- [ ] **Step 1: Rewrite `get_document_page`**

Replace the current implementation (lines 306-334) that calls `get_document()` (fetching ALL chunks) with targeted retrieval:

```python
async def get_document_page(self, doc_id: str, page: int = 1) -> dict | None:
    """Retrieve a paginated page by fetching only the overlapping chunks."""
    # Get document metadata (total_pages, total_chunks, title)
    meta = await self._pool.fetchrow(
        "SELECT title, total_pages, total_chunks FROM document_chunks "
        "WHERE doc_id = $1 LIMIT 1",
        doc_id,
    )
    if not meta:
        return None

    total_pages = meta["total_pages"] or 1
    if page < 1 or page > total_pages:
        return {
            "doc_id": doc_id,
            "title": meta["title"] or "",
            "content": f"Invalid page {page}. Document has {total_pages} page(s).",
            "page_number": page,
            "total_pages": total_pages,
        }

    # Calculate which chunks overlap with the requested page
    step = EMBEDDING_CHUNK_SIZE - EMBEDDING_CHUNK_OVERLAP
    start_char = (page - 1) * PAGE_SIZE
    end_char = page * PAGE_SIZE
    first_chunk = max(0, start_char // step)
    last_chunk = end_char // step + 1  # +1 for safety margin

    rows = await self._pool.fetch(
        "SELECT chunk_index, chunk_text FROM document_chunks "
        "WHERE doc_id = $1 AND chunk_index >= $2 AND chunk_index <= $3 "
        "ORDER BY chunk_index",
        doc_id, first_chunk, last_chunk,
    )

    if not rows:
        # Fallback: fetch all chunks (shouldn't happen normally)
        doc = await self.get_document(doc_id)
        if not doc:
            return None
        content = doc["content"]
        chunk = content[start_char:end_char]
        return {
            "doc_id": doc_id,
            "title": doc["title"],
            "content": chunk,
            "page_number": page,
            "total_pages": total_pages,
        }

    # Reconstruct just the needed slice
    content = self._reconstruct_content(rows)
    # Adjust for the offset within the reconstructed content
    local_start = start_char - first_chunk * step
    local_start = max(0, local_start)
    chunk = content[local_start:local_start + PAGE_SIZE]

    category_row = await self._pool.fetchval(
        "SELECT category FROM document_chunks WHERE doc_id = $1 LIMIT 1", doc_id,
    )

    return {
        "doc_id": doc_id,
        "title": meta["title"] or "",
        "content": chunk,
        "page_number": page,
        "total_pages": total_pages,
        "category": category_row or "",
    }
```

- [ ] **Step 2: Run tests**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/ -v --timeout=60`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add vector_store.py
git commit -m "perf: targeted page retrieval fetches only overlapping chunks"
```

---

### Task 14: Rewrite `server.py` — Entry Point Only

Strip server.py down to ~80 lines: lifecycle management, FastMCP setup, tool registration.

**Files:**
- Modify: `server.py` (complete rewrite)

- [ ] **Step 1: Rewrite `server.py`**

```python
"""MCP server exposing BDDK decision search, document retrieval, and data tools."""

import asyncio
import logging
import os
import time

import asyncpg
import httpx
from mcp.server.fastmcp import FastMCP

from client import BddkApiClient
from config import AUTO_SYNC, DATABASE_URL, PG_POOL_MAX, PG_POOL_MIN, REQUEST_TIMEOUT
from deps import Dependencies
from doc_store import DocumentStore
from logging_config import configure_logging
from tools import admin, analytics, bulletin, documents, search, sync

configure_logging()
logger = logging.getLogger(__name__)

# -- FastMCP instance ---------------------------------------------------------

mcp = FastMCP(
    "BDDK",
    instructions="""\
Search and retrieve BDDK (Turkish Banking Regulation) decisions, regulations, and statistical data.

GROUNDING RULES -- follow these strictly:
1. ONLY use information returned by tool calls. Never supplement with your own knowledge about BDDK decisions.
2. If a search returns no results, say so explicitly. Do NOT guess or invent decisions.
3. Always include document_id, decision_date, and decision_number in your response when available.
4. If document content is paginated, do NOT speculate about content on pages you have not retrieved.
5. Never fabricate karar numarasi (decision numbers), tarih (dates), or legal conclusions.
6. When quoting from a document, quote only text that appears verbatim in the tool output.
7. If relevance scores are below 50%, flag this to the user and recommend refining the query.
8. Distinguish clearly between: (a) information from BDDK tools, and (b) your general knowledge.\
""",
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 8000)),
    stateless_http=True,
)


async def create_deps() -> Dependencies:
    """Create all dependencies eagerly. Fails fast if DB is unreachable."""
    http = httpx.AsyncClient(
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        },
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        follow_redirects=True,
    )

    pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=PG_POOL_MIN,
        max_size=PG_POOL_MAX,
        command_timeout=30,
        timeout=10,
    )
    logger.info("PostgreSQL pool created: %s", DATABASE_URL.split("@")[-1])

    doc_store = DocumentStore(pool)
    await doc_store.initialize()

    client = BddkApiClient(pool=pool, doc_store=doc_store, http=http)
    await client.initialize()

    return Dependencies(
        pool=pool, doc_store=doc_store, client=client, http=http,
        vector_store=None, server_start_time=time.time(),
    )


async def init_vector_store(deps: Dependencies) -> None:
    """Background task: load embedding model and initialize VectorStore."""
    try:
        from vector_store import VectorStore

        vs = VectorStore(deps.pool)
        await vs.initialize()
        deps.vector_store = vs
        logger.info("VectorStore initialized (background)")
    except Exception as e:
        logger.error("VectorStore init failed: %s", e)


async def teardown_deps(deps: Dependencies) -> None:
    """Shut down in correct order: tasks first, then connections."""
    logger.info("Graceful shutdown initiated...")

    # 1. Cancel background tasks
    for task_attr in ("vector_init_task", "sync_task"):
        task = getattr(deps, task_attr)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # 2. Close client (only closes http if it owns it)
    if deps.client:
        await deps.client.close()

    # 3. Close shared http
    if deps.http:
        await deps.http.aclose()

    # 4. Close pool last
    if deps.pool:
        await deps.pool.close()
        logger.info("PostgreSQL pool closed")

    logger.info("Graceful shutdown complete")


# -- Entry point --------------------------------------------------------------

if __name__ == "__main__":
    try:
        import uvloop
        uvloop.install()
        logger.info("uvloop installed")
    except ImportError:
        pass

    _transport = os.environ.get("MCP_TRANSPORT", "stdio")
    logger.info("Transport: %s", _transport)
    logger.info("BDDK_AUTO_SYNC=%s", os.environ.get("BDDK_AUTO_SYNC", "(not set)"))
    logger.info("DATABASE_URL=%s", DATABASE_URL.split("@")[-1])

    if _transport == "streamable-http":
        import uvicorn

        app = mcp.streamable_http_app()
        port = int(os.environ.get("PORT", 8000))

        async def _run_server():
            config = uvicorn.Config(app, host="0.0.0.0", port=port)
            server = uvicorn.Server(config)

            # Create dependencies eagerly
            deps = await create_deps()

            # Register all tool modules
            search.register(mcp, deps)
            documents.register(mcp, deps)
            bulletin.register(mcp, deps)
            analytics.register(mcp, deps)
            sync.register(mcp, deps)
            admin.register(mcp, deps)

            # Seed DB from bundled JSON if tables are empty
            try:
                from seed import SEED_DIR, import_seed
                if SEED_DIR.exists():
                    result = await import_seed()
                    if not result["skipped"]:
                        logger.info(
                            "Seed import: %d cache, %d docs, %d chunks",
                            result["decision_cache"], result["documents"], result["chunks"],
                        )
                    else:
                        logger.info("DB already populated -- seed import skipped")
            except Exception as e:
                logger.warning("Seed import failed (non-fatal): %s", e)

            # Background: load vector store (embedding model ~1.1GB)
            deps.vector_init_task = asyncio.create_task(init_vector_store(deps))

            # Background: auto-sync if enabled
            if AUTO_SYNC:
                # Wait for vector store before syncing (needs embeddings)
                async def _sync_after_vector_init():
                    if deps.vector_init_task:
                        await deps.vector_init_task
                    await sync.startup_sync(deps)

                deps.sync_task = asyncio.create_task(_sync_after_vector_init())
                logger.info("[STARTUP] background sync scheduled")

            await server.serve()
            await teardown_deps(deps)

        asyncio.run(_run_server())
    else:
        mcp.run(transport=_transport)
```

- [ ] **Step 2: Verify the rewritten server imports correctly**

Run: `cd /home/cagatay/bddk-mcp && python -c "import server; print('OK')"`
Expected: `OK` (or import warnings that are non-fatal)

- [ ] **Step 3: Run full test suite**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/ -v --timeout=60`
Expected: PASS (some tests may need import updates — see Task 15)

- [ ] **Step 4: Commit**

```bash
git add server.py
git commit -m "refactor: strip server.py to entry point with lifecycle management"
```

---

### Task 15: Fix Test Imports and Add Integration Tests

Update any tests that imported directly from `server.py` and add a Dependencies integration test.

**Files:**
- Modify: any test files that import from `server`
- Create: `tests/test_lifecycle.py`

- [ ] **Step 1: Find tests that import from server**

Run: `cd /home/cagatay/bddk-mcp && grep -rn "from server import\|import server" tests/`

- [ ] **Step 2: Update imports**

For each test that imports from `server`, update to import from the new location:
- `_cached_search`, `_store_search` -> `from tools.search import _LRUCache` (or remove if testing internal cache)
- `_get_pool`, `_get_client`, etc. -> no longer exist as standalone functions, use `create_deps()`
- Any tool function references -> import from `tools.<module>`

- [ ] **Step 3: Write lifecycle integration test**

```python
# tests/test_lifecycle.py
"""Integration tests for server lifecycle."""

import asyncio
import time

import pytest

from deps import Dependencies


def test_deps_creation():
    """Dependencies can be created with None values for unit tests."""
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)
    assert deps.server_start_time <= time.time()
    assert deps.vector_store is None
    assert deps.sync_circuit_open is False


def test_deps_health_state_tracking():
    """Dependencies tracks health state correctly."""
    deps = Dependencies(pool=None, doc_store=None, client=None, http=None)

    # Simulate failures
    deps.sync_consecutive_failures = 10
    deps.sync_circuit_open = True
    deps.last_sync_error = "Connection refused"

    assert deps.sync_circuit_open is True
    assert deps.last_sync_error == "Connection refused"

    # Reset
    deps.sync_consecutive_failures = 0
    deps.sync_circuit_open = False
    deps.last_sync_time = time.time()
    deps.last_sync_error = None

    assert deps.sync_circuit_open is False
    assert deps.last_sync_time is not None
```

- [ ] **Step 4: Run full test suite**

Run: `cd /home/cagatay/bddk-mcp && python -m pytest tests/ -v --timeout=60`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test: fix imports for restructured modules, add lifecycle tests"
```
