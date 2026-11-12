"""DocumentSyncer must re-index the vector store after a successful sync."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bddk_mcp.ingest.doc_sync import DocumentSyncer
from bddk_mcp.ocr.base import MarkitdownBackend
from bddk_mcp.store.doc_store import StoredDocument


class _DummyStore:
    """Minimal DocumentStore satisfying DocumentSyncer's interface.

    Lets this test run without a corpus file.
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

    async def get_document(self, doc_id: str):
        return self.stored.get(doc_id)


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


def _stored_without_chunks(store: _DummyStore, doc_id: str) -> StoredDocument:
    doc = StoredDocument(
        document_id=doc_id,
        title="Stored title",
        category="Yönetmelik",
        decision_date="2020-01-01",
        decision_number="31000",
        source_url="https://www.mevzuat.gov.tr/mevzuat?MevzuatNo=1",
        markdown_content="MADDE 1 – Gözden geçirilmiş tohum metni.",
        extraction_method="lightocr",
    )
    store.stored[doc_id] = doc
    store.has.add(doc_id)
    return doc


def _forbid_downloads(syncer: DocumentSyncer) -> None:
    error = AssertionError("a stored document without chunks must not be downloaded again")
    syncer._fetch_trusted_bddk = AsyncMock(side_effect=error)
    syncer._fetch_trusted_mevzuat = AsyncMock(side_effect=error)
    syncer._download_bddk = AsyncMock(side_effect=error)
    syncer._download_mevzuat = AsyncMock(side_effect=error)


@pytest.mark.asyncio
async def test_plain_resync_reindexes_stored_text_whose_chunks_are_missing():
    """A stored document without current chunks (earlier reindex_failed) is re-indexed from the stored text."""
    store = _DummyStore()
    stored = _stored_without_chunks(store, "999997")
    vector_store = AsyncMock()
    vector_store.has_document = AsyncMock(return_value=False)
    vector_store.add_document = AsyncMock(return_value=3)

    async with DocumentSyncer(store, ocr_backends=[MarkitdownBackend()], vector_store=vector_store) as syncer:
        _forbid_downloads(syncer)
        result = await syncer.sync_document(doc_id="999997", title="Catalog title")
        report = await syncer.sync_all([{"document_id": "999997"}], concurrency=1)

    assert result.success, result.error
    assert result.method == "reindexed"
    assert vector_store.add_document.await_count == 2
    call_kwargs = vector_store.add_document.await_args.kwargs
    assert call_kwargs == {
        "doc_id": "999997",
        "title": "Stored title",
        "content": stored.markdown_content,
        "category": "Yönetmelik",
        "decision_date": "2020-01-01",
        "decision_number": "31000",
        "source_url": stored.source_url,
    }
    assert store.stored["999997"] is stored
    assert store.stored["999997"].extraction_method == "lightocr"
    assert store.cleared == ["999997", "999997"]
    assert (report.reindexed, report.downloaded, report.skipped, report.failed) == (1, 0, 0, 0)


@pytest.mark.asyncio
async def test_plain_resync_records_reindex_failed_without_downloading():
    store = _DummyStore()
    stored = _stored_without_chunks(store, "mevzuat_1")
    vector_store = AsyncMock()
    vector_store.has_document = AsyncMock(return_value=False)
    vector_store.add_document = AsyncMock(side_effect=RuntimeError("tokenizer could not load"))

    async with DocumentSyncer(store, ocr_backends=[MarkitdownBackend()], vector_store=vector_store) as syncer:
        _forbid_downloads(syncer)
        result = await syncer.sync_document(doc_id="mevzuat_1")
        report = await syncer.sync_all([{"document_id": "mevzuat_1"}], concurrency=1)

    assert result.success is False
    assert result.error == "reindex_failed"
    assert result.document_id == "mevzuat_1"
    assert store.stored["mevzuat_1"] is stored
    assert store.cleared == []
    assert store.failures[0] == (("mevzuat_1", "reindex_failed", "index", stored.source_url, True), {})
    assert (report.reindexed, report.failed) == (0, 1)
    assert [error.document_id for error in report.errors] == ["mevzuat_1"]


@pytest.mark.asyncio
async def test_plain_resync_skips_document_with_text_and_current_chunks():
    store = _DummyStore()
    store.has.add("999996")
    vector_store = AsyncMock()
    vector_store.has_document = AsyncMock(return_value=True)

    async with DocumentSyncer(store, ocr_backends=[MarkitdownBackend()], vector_store=vector_store) as syncer:
        result = await syncer.sync_document(doc_id="999996")

    assert result.success
    assert result.method == "cached"
    vector_store.add_document.assert_not_awaited()
