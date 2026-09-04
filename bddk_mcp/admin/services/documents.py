"""Read-only document queries for the admin console."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_PAGE_SIZE = 200
STORE_FAILURE = "Veri katmani kullanilamiyor."


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """One page of document rows plus the flag a template needs for paging."""

    items: list[dict[str, Any]]
    page: int
    page_size: int
    has_next: bool
    error: str | None = None


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    """Search results, or the reason there are none."""

    query: str
    hits: list[Any]
    error: str | None = None


@dataclass(frozen=True, slots=True)
class DocumentOutcome:
    """A single document lookup, or the reason it could not be loaded.

    Distinct from "not found": a store failure must never be rendered as a
    missing document, so callers branch on `error` before `doc is None`.
    """

    doc: Any | None
    error: str | None = None


class DocumentService:
    """Wraps DocumentStore so views never touch SQL or pagination arithmetic."""

    def __init__(self, store: Any) -> None:
        self._store = store

    async def list_page(self, page: int = 1, page_size: int = 50, category: str | None = None) -> DocumentPage:
        page = max(1, page)
        page_size = max(1, min(page_size, MAX_PAGE_SIZE))
        offset = (page - 1) * page_size
        try:
            # Fetch one extra row: cheaper than a COUNT(*) and enough to know
            # whether a Next control should render.
            rows = await self._store.list_documents(category=category, limit=page_size + 1, offset=offset)
        except Exception:  # never rendered as an empty list or as the exception text
            return DocumentPage(items=[], page=page, page_size=page_size, has_next=False, error=STORE_FAILURE)
        has_next = len(rows) > page_size
        return DocumentPage(items=list(rows[:page_size]), page=page, page_size=page_size, has_next=has_next)

    async def get(self, doc_id: str) -> DocumentOutcome:
        try:
            doc = await self._store.get_document(doc_id)
        except Exception:  # never rendered as "not found" or as the exception text
            return DocumentOutcome(doc=None, error=STORE_FAILURE)
        return DocumentOutcome(doc=doc)

    async def search(self, query: str, limit: int = 20) -> SearchOutcome:
        query = query.strip()
        if not query:
            return SearchOutcome(query="", hits=[])
        try:
            hits = list(await self._store.search_content(query, limit=limit))
        except Exception:  # never rendered as "no results" or as the exception text
            return SearchOutcome(query=query, hits=[], error=STORE_FAILURE)
        return SearchOutcome(query=query, hits=hits)

    async def categories(self) -> dict[str, int]:
        stats = await self._store.stats()
        return dict(stats.categories)
