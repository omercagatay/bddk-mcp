"""DocumentSyncer must re-index the vector store after a successful sync."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bddk_mcp.ingest.doc_sync import DocumentSyncer
from bddk_mcp.ocr.base import MarkitdownBackend


class _DummyStore:
    """Minimal DocumentStore satisfying DocumentSyncer's interface.

    Mirrors the MemStore in scripts/resync_corrupted_mevzuat.py — lets
    this test run without a Postgres fixture.
    """

    def __init__(self) -> None:
        self.has: set[str] = set()
        self.stored: dict = {}
        self.cleared: list[str] = []
        self.failures: list[tuple] = []

    async def has_document(self, doc_id: str) -> bool:
        return doc_id in self.has

    async def store_document(self, doc) -> None:
        self.stored[doc.document_id] = doc
        self.has.add(doc.document_id)

    async def clear_sync_failure(self, doc_id: str) -> None:
        self.cleared.append(doc_id)

    async def record_sync_failure(self, *args, **kwargs) -> None:
        self.failures.append((args, kwargs))

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
        syncer._fetch_trusted_bddk = AsyncMock(
            return_value=SimpleNamespace(
                status_code=200,
                content=html.encode(),
                headers={"content-type": "text/html"},
            )
        )
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
    assert store.cleared == ["999999"]
    assert store.failures == []


@pytest.mark.asyncio
async def test_sync_document_records_sanitized_reindex_failure_without_clearing_it():
    store = _DummyStore()
    vector_store = AsyncMock()
    vector_store.add_document = AsyncMock(side_effect=RuntimeError("postgresql://secret@db/private"))
    html = "<html><body><h1>Test</h1><p>" + ("Yeterli düzenleyici içerik. " * 30) + "</p></body></html>"

    async with DocumentSyncer(
        store,
        ocr_backends=[MarkitdownBackend()],
        vector_store=vector_store,
    ) as syncer:
        syncer._fetch_trusted_bddk = AsyncMock(
            return_value=SimpleNamespace(
                status_code=200,
                content=html.encode(),
                headers={"content-type": "text/html"},
            )
        )
        result = await syncer.sync_document(doc_id="999998", force=True)

    assert result.success is False
    assert result.error == "reindex_failed"
    assert "secret" not in result.model_dump_json()
    assert store.cleared == []
    assert store.failures == [(("999998", "reindex_failed", "index", "", True), {})]
