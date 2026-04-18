"""DocumentSyncer must re-index the vector store after a successful sync."""

from unittest.mock import AsyncMock

import httpx
import pytest

from doc_sync import DocumentSyncer
from ocr_backends import MarkitdownBackend
from tests.conftest import make_http_response


class _DummyStore:
    """Minimal DocumentStore satisfying DocumentSyncer's interface.

    Mirrors the MemStore in scripts/resync_corrupted_mevzuat.py — lets
    this test run without a Postgres fixture.
    """

    def __init__(self) -> None:
        self.has: set[str] = set()
        self.stored: dict = {}

    async def has_document(self, doc_id: str) -> bool:
        return doc_id in self.has

    async def store_document(self, doc) -> None:
        self.stored[doc.document_id] = doc
        self.has.add(doc.document_id)

    async def clear_sync_failure(self, doc_id: str) -> None:
        pass

    async def record_sync_failure(self, *args, **kwargs) -> None:
        pass

    async def get_pdf_bytes(self, doc_id: str):
        return None


@pytest.mark.asyncio
async def test_sync_document_calls_add_document_on_success():
    """After a successful extraction, vector_store.add_document must be called."""
    store = _DummyStore()
    vector_store = AsyncMock()
    vector_store.add_document = AsyncMock(return_value=3)

    html = (
        "<html><body><h1>Test Doc</h1>"
        "<p>Madde 1 - Bu belge bir test dokumanidir ve icerikte yeterli karakter "
        "bulunmaktadir cunku extraction minimum uzunluk esigini gecmesi "
        "gerekmektedir. " * 4 + "</p></body></html>"
    )

    async with DocumentSyncer(
        store,
        ocr_backends=[MarkitdownBackend()],
        vector_store=vector_store,
    ) as syncer:
        syncer._http = AsyncMock(spec=httpx.AsyncClient)
        syncer._http.get = AsyncMock(return_value=make_http_response(text=html, content_type="text/html"))
        result = await syncer.sync_document(
            doc_id="999999",
            title="Test Doc",
            category="karar",
            source_url="https://example.test/999999",
            decision_date="2026-01-01",
            decision_number="999/1",
            force=True,
        )

    assert result.success, f"sync failed: {result.error}"
    assert "999999" in store.stored, "document was not stored"
    vector_store.add_document.assert_awaited_once()
    call_kwargs = vector_store.add_document.await_args.kwargs
    assert call_kwargs["doc_id"] == "999999"
    assert call_kwargs["title"] == "Test Doc"
    assert call_kwargs["category"] == "karar"
    assert call_kwargs["source_url"] == "https://example.test/999999"
    assert call_kwargs["content"]  # non-empty markdown
