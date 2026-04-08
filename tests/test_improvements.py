"""Tests for improvements: chunk overlap fix, stale cache, extraction errors, etc."""

import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from client import BddkApiClient
from doc_sync import DocumentSyncer, ExtractionResult
from models import BddkDecisionSummary
from tests.conftest import MockPool, make_http_response
from vector_store import _chunk_text

# -- Chunk Overlap Reconstruction Bug Fix ------------------------------------


class TestChunkOverlapFix:

    def test_short_text_no_chunking(self):
        chunks = _chunk_text("Hello world", chunk_size=100, overlap=20)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_empty_text(self):
        assert _chunk_text("") == []

    def test_exact_chunk_size(self):
        text = "A" * 1000
        chunks = _chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) == 1

    def test_overlap_creates_expected_chunks(self):
        text = "A" * 2000
        chunks = _chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) == 3
        assert len(chunks[0]) == 1000
        assert len(chunks[1]) == 1000
        assert len(chunks[2]) == 400

    def test_reconstruction_preserves_content(self):
        """The critical test: reconstruct must produce the original text."""
        from vector_store import VectorStore

        vs = VectorStore.__new__(VectorStore)

        original = "ABCDE" * 400  # 2000 chars
        chunks = _chunk_text(original, chunk_size=1000, overlap=200)

        # Simulate asyncpg.Record-like objects for _reconstruct_content
        class FakeRecord:
            def __init__(self, text):
                self._data = {"chunk_text": text}
            def __getitem__(self, key):
                return self._data[key]

        rows = [FakeRecord(chunk) for chunk in chunks]
        reconstructed = vs._reconstruct_content(rows)
        assert reconstructed == original

    def test_reconstruction_large_document(self):
        from vector_store import VectorStore

        vs = VectorStore.__new__(VectorStore)
        original = "".join(chr(65 + (i % 26)) for i in range(10000))
        chunks = _chunk_text(original, chunk_size=1000, overlap=200)

        class FakeRecord:
            def __init__(self, text):
                self._data = {"chunk_text": text}
            def __getitem__(self, key):
                return self._data[key]

        rows = [FakeRecord(chunk) for chunk in chunks]
        reconstructed = vs._reconstruct_content(rows)
        assert reconstructed == original

    def test_reconstruction_single_chunk(self):
        from vector_store import VectorStore

        vs = VectorStore.__new__(VectorStore)

        class FakeRecord:
            def __init__(self, text):
                self._data = {"chunk_text": text}
            def __getitem__(self, key):
                return self._data[key]

        rows = [FakeRecord("Hello world")]
        assert vs._reconstruct_content(rows) == "Hello world"

    def test_reconstruction_empty(self):
        from vector_store import VectorStore

        vs = VectorStore.__new__(VectorStore)
        assert vs._reconstruct_content([]) == ""


# -- Stale Cache Fallback ---------------------------------------------------


class TestStaleCacheFallback:
    """Test that stale cache from DB is served when BDDK is unreachable."""

    @pytest.mark.asyncio
    async def test_stale_db_cache_loaded(self, doc_store):
        pool = doc_store._pool
        # Save stale cache to DB
        client = BddkApiClient(pool=pool)
        await client.initialize()
        client._cache = [
            BddkDecisionSummary(title="Stale Decision", document_id="stale_999", content="stale", category="Rehber")
        ]
        client._cache_timestamp = 1.0  # very old
        await client._save_cache_to_db()

        # New client loads from DB
        client2 = BddkApiClient(pool=pool)
        loaded = await client2._load_cache_from_db()
        assert loaded
        assert any(d.document_id == "stale_999" for d in client2._cache)

        await client.close()
        await client2.close()

    @pytest.mark.asyncio
    async def test_empty_db_returns_false(self):
        pool = MockPool()
        client = BddkApiClient(pool=pool)
        assert not await client._load_cache_from_db()


# -- Extraction Error Handling -----------------------------------------------


class TestExtractionResult:

    def test_successful_extraction(self):
        result = ExtractionResult(content="# Hello", method="html_parser")
        assert result.content == "# Hello"
        assert result.method == "html_parser"
        assert result.error == ""
        assert not result.retryable

    def test_failed_extraction(self):
        result = ExtractionResult(
            method="failed",
            error="markitdown: failed for .pdf; html_parser: no content",
            retryable=True,
        )
        assert result.content == ""
        assert result.retryable


class TestExtractStructured:

    def test_html_extraction(self):
        syncer = DocumentSyncer.__new__(DocumentSyncer)
        syncer._prefer_nougat = False

        html = b"<html><body><h1>Test</h1><p>Content here</p></body></html>"
        result = syncer._extract_structured(html, ".html")
        assert result.content
        assert result.method in ("html_parser", "markitdown")
        assert result.error == ""

    def test_unknown_extension_fails(self):
        syncer = DocumentSyncer.__new__(DocumentSyncer)
        syncer._prefer_nougat = False

        result = syncer._extract_structured(b"data", ".xyz")
        assert result.method == "failed"
        assert "Unsupported extension" in result.error

    def test_empty_html_tries_fallback(self):
        syncer = DocumentSyncer.__new__(DocumentSyncer)
        syncer._prefer_nougat = False

        result = syncer._extract_structured(b"<html></html>", ".html")
        assert result.method in ("html_parser", "markitdown", "failed")

    def test_retryable_flag_for_small_content(self):
        syncer = DocumentSyncer.__new__(DocumentSyncer)
        syncer._prefer_nougat = False

        result = syncer._extract_structured(b"<h>err</h>", ".html")
        if result.method == "failed":
            assert result.retryable


# -- Unmapped Category Warnings ----------------------------------------------


class TestUnmappedCategoryWarning:

    @pytest.mark.asyncio
    async def test_unmapped_category_logged(self, caplog):
        html = """
        <div class="card">
          <h5>Bilinmeyen Kategori (3)</h5>
          <div class="card-body">
            <a href="/Mevzuat/DokumanGetir/999">Test Doc</a>
          </div>
        </div>
        """
        client = BddkApiClient(pool=MockPool())
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.get = AsyncMock(return_value=make_http_response(html))

        import logging
        with caplog.at_level(logging.WARNING, logger="client"):
            results = await client._fetch_and_parse_accordion_page(50)

        assert any("Unmapped accordion category" in msg for msg in caplog.messages)
        assert len(results) == 1
        assert results[0].category == "Bilinmeyen Kategori"

    @pytest.mark.asyncio
    async def test_known_category_no_warning(self, caplog):
        from tests.conftest import BDDK_ACCORDION_HTML

        client = BddkApiClient(pool=MockPool())
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.get = AsyncMock(return_value=make_http_response(BDDK_ACCORDION_HTML))

        import logging
        with caplog.at_level(logging.WARNING, logger="client"):
            await client._fetch_and_parse_accordion_page(50)

        assert not any("Unmapped accordion category" in msg for msg in caplog.messages)


# -- Sync Progress Visibility ------------------------------------------------


class TestSyncProgress:

    @pytest.mark.asyncio
    async def test_progress_callback_called(self, doc_store):
        progress_calls = []

        def on_progress(doc_id, completed, total):
            progress_calls.append((doc_id, completed, total))

        syncer = DocumentSyncer(doc_store, progress_callback=on_progress)
        syncer._http = AsyncMock(spec=httpx.AsyncClient)
        syncer._http.aclose = AsyncMock()

        async def mock_download(doc_id):
            return b"<h1>Test</h1><p>Content</p>", "mock", ".html"

        syncer._download_bddk = mock_download

        documents = [
            {"document_id": "1", "title": "Doc 1"},
            {"document_id": "2", "title": "Doc 2"},
        ]
        await syncer.sync_all(documents, concurrency=1, force=True)

        assert len(progress_calls) == 2
        assert progress_calls[0][1] == 1
        assert progress_calls[0][2] == 2
        assert progress_calls[1][1] == 2


# -- Document Versioning in Search -------------------------------------------


class TestDocumentVersioning:

    @pytest.mark.asyncio
    async def test_version_created_on_content_change(self, doc_store, sample_doc):
        await doc_store.store_document(sample_doc)

        from doc_store import StoredDocument
        updated = StoredDocument(
            document_id=sample_doc.document_id,
            title=sample_doc.title,
            category=sample_doc.category,
            markdown_content="Updated content here",
            extraction_method="markitdown",
        )
        await doc_store.store_document(updated)

        history = await doc_store.get_document_history(sample_doc.document_id)
        assert len(history) >= 1
        assert history[0]["content_length"] > 0


# -- All Announcement Categories ---------------------------------------------


class TestAllAnnouncementCategories:

    @pytest.mark.asyncio
    async def test_check_updates_covers_all_categories(self):
        from analytics import check_updates

        mock_http = AsyncMock(spec=httpx.AsyncClient)
        called_categories = []

        async def tracking_fetch(http, category_id):
            called_categories.append(category_id)
            return []

        with patch("analytics.fetch_announcements", side_effect=tracking_fetch):
            result = await check_updates(mock_http, [], set())

        assert set(called_categories) == {39, 40, 41, 42, 48}
        assert len(result["checked_categories"]) == 5
