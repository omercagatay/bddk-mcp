from __future__ import annotations

import asyncio
from types import SimpleNamespace

from bddk_mcp.admin.services.documents import DocumentService


class FakeStore:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.calls: list[tuple] = []

    async def list_documents(self, category=None, limit=100, offset=0):
        self.calls.append((category, limit, offset))
        selected = [r for r in self.rows if category is None or r["category"] == category]
        return selected[offset : offset + limit]

    async def stats(self):
        return SimpleNamespace(categories={"mevzuat": 2}, total_documents=len(self.rows))


def _rows(count: int) -> list[dict]:
    return [{"document_id": f"doc-{i}", "title": f"Belge {i}", "category": "mevzuat"} for i in range(count)]


def test_list_page_requests_one_extra_row_to_detect_a_next_page() -> None:
    store = FakeStore(_rows(5))
    service = DocumentService(store)

    page = asyncio.run(service.list_page(page=1, page_size=2))

    assert [item["document_id"] for item in page.items] == ["doc-0", "doc-1"]
    assert page.has_next is True
    # page_size + 1 is fetched so the template can render a Next control
    assert store.calls == [(None, 3, 0)]


def test_last_page_reports_no_next() -> None:
    store = FakeStore(_rows(4))
    service = DocumentService(store)

    page = asyncio.run(service.list_page(page=2, page_size=2))

    assert [item["document_id"] for item in page.items] == ["doc-2", "doc-3"]
    assert page.has_next is False


def test_page_numbers_below_one_are_clamped() -> None:
    store = FakeStore(_rows(3))
    service = DocumentService(store)

    page = asyncio.run(service.list_page(page=0, page_size=2))

    assert page.page == 1
    assert store.calls == [(None, 3, 0)]


def test_categories_come_from_store_stats() -> None:
    service = DocumentService(FakeStore(_rows(2)))
    assert asyncio.run(service.categories()) == {"mevzuat": 2}
