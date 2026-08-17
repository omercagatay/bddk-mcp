"""Read-only document queries for the admin console."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_PAGE_SIZE = 200


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """One page of document rows plus the flag a template needs for paging."""

    items: list[dict[str, Any]]
    page: int
    page_size: int
    has_next: bool


class DocumentService:
    """Wraps DocumentStore so views never touch SQL or pagination arithmetic."""

    def __init__(self, store: Any) -> None:
        self._store = store

    async def list_page(self, page: int = 1, page_size: int = 50, category: str | None = None) -> DocumentPage:
        page = max(1, page)
        page_size = max(1, min(page_size, MAX_PAGE_SIZE))
        offset = (page - 1) * page_size
        # Fetch one extra row: cheaper than a COUNT(*) and enough to know
        # whether a Next control should render.
        rows = await self._store.list_documents(category=category, limit=page_size + 1, offset=offset)
        has_next = len(rows) > page_size
        return DocumentPage(items=list(rows[:page_size]), page=page, page_size=page_size, has_next=has_next)

    async def get(self, doc_id: str) -> Any:
        return await self._store.get_document(doc_id)

    async def search(self, query: str, limit: int = 20) -> list[Any]:
        query = query.strip()
        if not query:
            return []
        return list(await self._store.search_content(query, limit=limit))

    async def categories(self) -> dict[str, int]:
        stats = await self._store.stats()
        return dict(stats.categories)
