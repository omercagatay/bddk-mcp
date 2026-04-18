"""Search tools: search_bddk_decisions, search_bddk_institutions,
search_bddk_announcements, and search_document_store.

Uses an OrderedDict-based LRU cache for O(1) eviction instead of the
O(n) min() scan in server.py.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from typing import TYPE_CHECKING

from client import _turkish_lower
from config import ANNOUNCEMENT_CATEGORY_IDS, SEARCH_CACHE_MAX, SEARCH_CACHE_TTL
from data_sources import fetch_announcements, fetch_institutions
from metrics import metrics
from models import BddkSearchRequest

if TYPE_CHECKING:
    from deps import Dependencies


# -- LRU Cache ----------------------------------------------------------------


class _LRUCache:
    """In-memory LRU cache with TTL.

    Uses OrderedDict for O(1) eviction of the least-recently-used entry.
    Each value is stored as (timestamp, payload).
    """

    def __init__(self, max_size: int, ttl: int) -> None:
        self._max_size = max_size
        self._ttl = ttl
        self._data: OrderedDict[str, tuple[float, object]] = OrderedDict()

    def get(self, key: str) -> object | None:
        """Return the cached value if present and not expired, else None.

        On cache hit the entry is moved to the end (most-recently-used).
        """
        entry = self._data.get(key)
        if entry is None:
            return None
        ts, value = entry
        if (time.time() - ts) >= self._ttl:
            del self._data[key]
            return None
        # Move to end — most recently used
        self._data.move_to_end(key)
        return value

    def set(self, key: str, value: object) -> None:
        """Store a value. Evicts the least-recently-used entry when full."""
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = (time.time(), value)
        while len(self._data) > self._max_size:
            # popitem(last=False) removes the front — oldest/least-recently-used
            self._data.popitem(last=False)


# Module-level cache shared across all invocations
_search_cache: _LRUCache = _LRUCache(max_size=SEARCH_CACHE_MAX, ttl=SEARCH_CACHE_TTL)


# -- Tool registration --------------------------------------------------------


def register(mcp, deps: Dependencies) -> None:  # type: ignore[type-arg]
    """Register the four search tools on the given MCP instance."""

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
            keywords: Search terms in Turkish (e.g. "elektronik para", "banka lisansı")
            page: Page number, starting from 1
            page_size: Number of results per page (max 50)
            category: Optional category filter. Available categories:
                Yönetmelik, Genelge, Tebliğ, Rehber, Bilgi Sistemleri,
                Sermaye Yeterliliği, Faizsiz Bankacılık, Tekdüzen Hesap Planı,
                Kurul Kararı, Kanun, Banka Kartları,
                Finansal Kiralama ve Faktoring, BDDK Düzenlemesi,
                Düzenleme Taslağı, Mülga Düzenleme
            date_from: Optional start date filter (DD.MM.YYYY)
            date_to: Optional end date filter (DD.MM.YYYY)
        """
        cache_key = f"decisions:{keywords}:{page}:{page_size}:{category}:{date_from}:{date_to}"
        cached = _search_cache.get(cache_key)
        if cached:
            return cached  # type: ignore[return-value]

        request = BddkSearchRequest(
            keywords=keywords,
            page=page,
            page_size=page_size,
            category=category,
            date_from=date_from,
            date_to=date_to,
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

        # Batch version count lookup — one query instead of N
        doc_ids = [d.document_id for d in result.decisions]
        version_counts = await deps.doc_store.get_version_counts(doc_ids)

        lines = [f"Found {result.total_results} result(s) (page {result.page}):\n"]
        for d in result.decisions:
            date_info = f" ({d.decision_date} - {d.decision_number})" if d.decision_date else ""
            cat_info = f" [{d.category}]" if d.category else ""
            lines.append(f"**{d.title}**{date_info}{cat_info}")
            lines.append(f"  Document ID: {d.document_id}")
            ver_count, ver_latest = version_counts.get(d.document_id, (0, None))
            if ver_count:
                lines.append(f"  Versions: {ver_count} (latest: {ver_latest})")
            lines.append(f"  {d.content}\n")

        output = "\n".join(lines)
        _search_cache.set(cache_key, output)
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
            keywords: Search terms (e.g. "Ziraat", "Garanti", "katılım")
            institution_type: Filter by type: Banka, Finansal Kiralama Şirketi,
                Faktoring Şirketi, Finansman Şirketi, Varlık Yönetim Şirketi
            active_only: If true (default), only show active institutions
        """
        institutions = await fetch_institutions(deps.http, institution_type)

        if active_only:
            institutions = [i for i in institutions if i["status"] == "Aktif"]

        if keywords:
            kw = _turkish_lower(keywords)
            institutions = [
                i for i in institutions if kw in _turkish_lower(i["name"]) or kw in _turkish_lower(i.get("type", ""))
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
            website = f" — {i['website']}" if i["website"] else ""
            lines.append(f"**{i['name']}**{status} [{i['type']}]{website}")
        return "\n".join(lines)

    @mcp.tool()
    async def search_bddk_announcements(
        keywords: str = "",
        category: str = "basın",
    ) -> str:
        """
        Search BDDK announcements and press releases.

        Args:
            keywords: Search terms in Turkish
            category: Announcement type: basın (press), mevzuat (regulation),
                insan kaynakları (HR), veri (data publication), kuruluş (institution).
                Use "tümü" or "all" to search across all categories.
        """
        cat_lower = _turkish_lower(category)

        cat_map: dict[str, list[int]] = {
            "basın": [39],
            "press": [39],
            "mevzuat": [40],
            "regul": [40],
            "insan": [41],
            "hr": [41],
            "veri": [42],
            "data": [42],
            "kuruluş": [48],
            "institution": [48],
            "tümü": list(ANNOUNCEMENT_CATEGORY_IDS),
            "all": list(ANNOUNCEMENT_CATEGORY_IDS),
        }

        cat_ids = [39]  # default
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
                "(basın, mevzuat, insan kaynakları, veri, kuruluş, or tümü for all)."
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
        Understands meaning, not just keywords.

        Args:
            query: Natural language query in Turkish (e.g. "faiz oranı riski nasıl hesaplanır")
            category: Optional category filter (e.g. "Yönetmelik", "Rehber", "Kurul Kararı")
            limit: Maximum results to return (default 10)
        """
        if deps.vector_store is None:
            return "Vector store is still initializing. Please try again in a few moments."

        cache_key = f"semantic:{query}:{category}:{limit}"
        cached = _search_cache.get(cache_key)
        if cached:
            return cached  # type: ignore[return-value]

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
            confidence_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
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
                f"\n⚠️ {low_count} result(s) have low confidence. "
                "These may not be directly relevant. Verify before citing."
            )

        output = "\n".join(lines)
        _search_cache.set(cache_key, output)
        return output
