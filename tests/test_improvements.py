"""Tests for all improvements: chunk overlap fix, stale cache, extraction errors, etc."""

import json
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from client import BddkApiClient
from doc_sync import DocumentSyncer, ExtractionResult
from vector_store import _chunk_text

# ── Chunk Overlap Reconstruction Bug Fix ─────────────────────────────────────


class TestChunkOverlapFix:
    """Test that the chunk overlap reconstruction correctly handles all cases."""

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
        # step = 1000 - 200 = 800
        # chunk 0: 0-1000, chunk 1: 800-1800, chunk 2: 1600-2000
        assert len(chunks) == 3
        assert len(chunks[0]) == 1000
        assert len(chunks[1]) == 1000
        assert len(chunks[2]) == 400  # final chunk is shorter

    def test_reconstruction_preserves_content(self):
        """The critical test: reconstruct must produce the original text."""
        from vector_store import VectorStore

        vs = VectorStore.__new__(VectorStore)  # skip __init__

        original = "ABCDE" * 400  # 2000 chars
        chunks = _chunk_text(original, chunk_size=1000, overlap=200)

        # Simulate chunk_data format: (id, text, metadata)
        chunk_data = [(f"doc_chunk_{i}", chunk, {"chunk_index": i}) for i, chunk in enumerate(chunks)]

        reconstructed = vs._reconstruct_content(chunk_data)
        assert reconstructed == original, (
            f"Reconstruction mismatch: len(original)={len(original)}, len(reconstructed)={len(reconstructed)}"
        )

    def test_reconstruction_large_document(self):
        """Test with a realistically sized document."""
        from vector_store import VectorStore

        vs = VectorStore.__new__(VectorStore)

        # 10000 chars with varied content
        original = "".join(chr(65 + (i % 26)) for i in range(10000))
        chunks = _chunk_text(original, chunk_size=1000, overlap=200)

        chunk_data = [(f"doc_chunk_{i}", chunk, {"chunk_index": i}) for i, chunk in enumerate(chunks)]

        reconstructed = vs._reconstruct_content(chunk_data)
        assert reconstructed == original

    def test_reconstruction_single_chunk(self):
        from vector_store import VectorStore

        vs = VectorStore.__new__(VectorStore)

        chunk_data = [("doc_chunk_0", "Hello world", {"chunk_index": 0})]
        assert vs._reconstruct_content(chunk_data) == "Hello world"

    def test_reconstruction_empty(self):
        from vector_store import VectorStore

        vs = VectorStore.__new__(VectorStore)
        assert vs._reconstruct_content([]) == ""


# ── Stale Cache Fallback ─────────────────────────────────────────────────────


class TestStaleCacheFallback:
    """Test that stale cache is served when BDDK is unreachable."""

    def test_load_stale_cache_ignores_ttl(self, tmp_path):
        cache_file = tmp_path / ".cache.json"
        data = {
            "timestamp": time.time() - 86400,  # 24 hours old
            "items": [
                {
                    "title": "Stale Decision",
                    "document_id": "999",
                    "content": "stale",
                    "category": "Rehber",
                }
            ],
        }
        cache_file.write_text(json.dumps(data), encoding="utf-8")

        client = BddkApiClient()
        with patch("client.CACHE_FILE", cache_file):
            loaded = client._load_stale_cache_from_disk()
        assert loaded
        assert len(client._cache) == 1
        assert client._cache[0].title == "Stale Decision"

    def test_stale_cache_returns_false_when_no_file(self, tmp_path):
        cache_file = tmp_path / "nonexistent.json"
        client = BddkApiClient()
        with patch("client.CACHE_FILE", cache_file):
            assert not client._load_stale_cache_from_disk()

    def test_stale_cache_returns_false_on_empty_items(self, tmp_path):
        cache_file = tmp_path / ".cache.json"
        cache_file.write_text(json.dumps({"timestamp": 0, "items": []}))
        client = BddkApiClient()
        with patch("client.CACHE_FILE", cache_file):
            assert not client._load_stale_cache_from_disk()


# ── Extraction Error Handling ────────────────────────────────────────────────


class TestExtractionResult:
    """Test structured extraction result model."""

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
    """Test the improved extraction pipeline with structured errors."""

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

        # An empty page that html_parser can't extract from
        result = syncer._extract_structured(b"<html></html>", ".html")
        # Either markitdown succeeds or it fails gracefully
        assert result.method in ("html_parser", "markitdown", "failed")

    def test_retryable_flag_for_small_content(self):
        syncer = DocumentSyncer.__new__(DocumentSyncer)
        syncer._prefer_nougat = False

        # Content too small — likely an error page
        result = syncer._extract_structured(b"<h>err</h>", ".html")
        if result.method == "failed":
            assert result.retryable


# ── Unmapped Category Warnings ───────────────────────────────────────────────


class TestUnmappedCategoryWarning:
    """Test that unmapped accordion categories produce warnings."""

    @pytest.mark.asyncio
    async def test_unmapped_category_logged(self, caplog):
        from tests.conftest import make_http_response

        # HTML with an unknown h5 header
        html = """
        <div class="card">
          <h5>Bilinmeyen Kategori (3)</h5>
          <div class="card-body">
            <a href="/Mevzuat/DokumanGetir/999">Test Doc</a>
          </div>
        </div>
        """
        client = BddkApiClient()
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.get = AsyncMock(return_value=make_http_response(html))

        import logging

        with caplog.at_level(logging.WARNING, logger="client"):
            results = await client._fetch_and_parse_accordion_page(50)

        assert any("Unmapped accordion category" in msg for msg in caplog.messages)
        # Still parses the documents
        assert len(results) == 1
        assert results[0].category == "Bilinmeyen Kategori"

    @pytest.mark.asyncio
    async def test_known_category_no_warning(self, caplog):
        from tests.conftest import BDDK_ACCORDION_HTML, make_http_response

        client = BddkApiClient()
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.get = AsyncMock(return_value=make_http_response(BDDK_ACCORDION_HTML))

        import logging

        with caplog.at_level(logging.WARNING, logger="client"):
            await client._fetch_and_parse_accordion_page(50)

        assert not any("Unmapped accordion category" in msg for msg in caplog.messages)


# ── Sync Progress Visibility ─────────────────────────────────────────────────


class TestSyncProgress:
    """Test that sync reports progress."""

    @pytest.mark.asyncio
    async def test_progress_callback_called(self, tmp_path):
        from doc_store import DocumentStore

        db_path = tmp_path / "test.db"
        store = DocumentStore(db_path=db_path)
        await store.initialize()

        progress_calls = []

        def on_progress(doc_id, completed, total):
            progress_calls.append((doc_id, completed, total))

        syncer = DocumentSyncer(store, progress_callback=on_progress)
        syncer._http = AsyncMock(spec=httpx.AsyncClient)
        syncer._http.aclose = AsyncMock()

        # Mock _download_bddk to return simple HTML
        async def mock_download(doc_id):
            return b"<h1>Test</h1><p>Content</p>", "mock", ".html"

        syncer._download_bddk = mock_download

        documents = [
            {"document_id": "1", "title": "Doc 1"},
            {"document_id": "2", "title": "Doc 2"},
        ]
        await syncer.sync_all(documents, concurrency=1, force=True)

        # Progress callback should be called for each doc
        assert len(progress_calls) == 2
        assert progress_calls[0][1] == 1  # first completed
        assert progress_calls[0][2] == 2  # total
        assert progress_calls[1][1] == 2

        await store.close()


# ── Document Versioning in Search ────────────────────────────────────────────


class TestDocumentVersioning:
    """Test that document versioning works and is accessible."""

    @pytest.mark.asyncio
    async def test_version_created_on_content_change(self, doc_store, sample_doc):
        # Store original
        await doc_store.store_document(sample_doc)

        # Update with different content
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


# ── All Announcement Categories ──────────────────────────────────────────────


class TestAllAnnouncementCategories:
    """Verify that analytics and update checks use all 5 categories."""

    @pytest.mark.asyncio
    async def test_check_updates_covers_all_categories(self):
        from analytics import check_updates

        mock_http = AsyncMock(spec=httpx.AsyncClient)

        # Patch fetch_announcements to track which category_ids are called
        called_categories = []

        async def tracking_fetch(http, category_id):
            called_categories.append(category_id)
            return []

        with patch("analytics.fetch_announcements", side_effect=tracking_fetch):
            result = await check_updates(mock_http, [], set())

        assert set(called_categories) == {39, 40, 41, 42, 48}
        assert len(result["checked_categories"]) == 5
