"""Integration tests: end-to-end flows across multiple modules."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from client import BddkApiClient
from doc_store import DocumentStore, StoredDocument
from doc_sync import DocumentSyncer
from models import BddkSearchRequest
from tests.conftest import (
    BDDK_ACCORDION_HTML,
    make_http_response,
)


class TestSearchThenRetrieveFlow:
    """Test the full search → retrieve → store flow."""

    @pytest.mark.asyncio
    async def test_search_then_get_document(self, tmp_path):
        """Search for decisions, pick one, retrieve its markdown."""
        db_path = tmp_path / "test.db"
        store = DocumentStore(db_path=db_path)
        await store.initialize()

        client = BddkApiClient(doc_store=store)
        client._http = AsyncMock(spec=httpx.AsyncClient)

        # Mock accordion page response
        client._http.get = AsyncMock(return_value=make_http_response(BDDK_ACCORDION_HTML))

        # Populate cache (bypass disk cache)
        with patch("client.CACHE_FILE", tmp_path / "fake_cache.json"):
            await client.ensure_cache()
        assert len(client._cache) > 0

        # Search — use a term that matches something in the accordion HTML
        request = BddkSearchRequest(keywords="Rehber")
        result = await client.search_decisions(request)
        assert result.total_results > 0

        # Get document (mock the HTTP fetch)
        doc_html = "<html><body><h1>Sermaye Rehberi</h1><p>Bu rehber bankacilik sektorunde...</p></body></html>"
        client._http.get = AsyncMock(return_value=make_http_response(doc_html))

        doc = await client.get_document_markdown(result.decisions[0].document_id, 1)
        assert doc.markdown_content
        assert doc.page_number == 1

        await store.close()

    @pytest.mark.asyncio
    async def test_store_first_strategy(self, tmp_path):
        """Verify documents are served from store if available."""
        db_path = tmp_path / "test.db"
        store = DocumentStore(db_path=db_path)
        await store.initialize()

        # Pre-populate store
        doc = StoredDocument(
            document_id="1291",
            title="Test Doc",
            markdown_content="# Test\n\nThis is a test document with enough content.",
            extraction_method="markitdown",
        )
        await store.store_document(doc)

        client = BddkApiClient(doc_store=store)
        client._http = AsyncMock(spec=httpx.AsyncClient)

        # Should NOT make HTTP call — served from store
        result = await client.get_document_markdown("1291", 1)
        assert "Test" in result.markdown_content
        client._http.get.assert_not_called()

        await store.close()


class TestSyncThenSearchFlow:
    """Test sync → store → search flow."""

    @pytest.mark.asyncio
    async def test_sync_and_fts_search(self, tmp_path):
        """Sync a document, then search for it via FTS5."""
        db_path = tmp_path / "test.db"
        store = DocumentStore(db_path=db_path)
        await store.initialize()

        # Store a document directly
        doc = StoredDocument(
            document_id="42",
            title="Bankacılık Sektörü Sermaye Yeterliliği Rehberi",
            category="Rehber",
            markdown_content=(
                "# Sermaye Yeterliliği\n\n"
                "Bu rehber bankacılık sektöründe sermaye yeterliliği "
                "hesaplamalarını düzenler. Kredi riski için asgari sermaye "
                "oranı yüzde sekiz olarak belirlenmiştir."
            ),
            extraction_method="markitdown",
        )
        await store.store_document(doc)

        # FTS search
        hits = await store.search_content("sermaye yeterliliği")
        assert len(hits) > 0
        assert hits[0].document_id == "42"

        # Page retrieval
        page = await store.get_document_page("42", 1)
        assert page is not None
        assert "sermaye" in page.markdown_content.lower()

        await store.close()


class TestCacheFallbackFlow:
    """Test the cache fallback chain: memory → disk → stale disk."""

    @pytest.mark.asyncio
    async def test_full_cache_fallback_chain(self, tmp_path):
        """When BDDK is unreachable, stale cache should be served."""
        cache_file = tmp_path / ".cache.json"

        # Pre-populate stale cache file
        stale_data = {
            "timestamp": 0,  # Very old
            "items": [
                {
                    "title": "Stale Regulation",
                    "document_id": "old1",
                    "content": "Old content",
                    "category": "Yönetmelik",
                }
            ],
        }
        cache_file.write_text(json.dumps(stale_data), encoding="utf-8")

        client = BddkApiClient()
        client._http = AsyncMock(spec=httpx.AsyncClient)
        # All HTTP calls fail
        client._http.get = AsyncMock(side_effect=httpx.TransportError("Network unreachable"))

        with (
            patch("client.CACHE_FILE", cache_file),
            patch("client.STALE_CACHE_FALLBACK", True),
        ):
            await client._ensure_cache()

        # Should have loaded stale cache
        assert len(client._cache) == 1
        assert client._cache[0].title == "Stale Regulation"


class TestExtractionPipelineFlow:
    """Test the extraction pipeline with various content types."""

    @pytest.mark.asyncio
    async def test_html_extraction_end_to_end(self, tmp_path):
        db_path = tmp_path / "test.db"
        store = DocumentStore(db_path=db_path)
        await store.initialize()

        syncer = DocumentSyncer(store, prefer_nougat=False)
        syncer._http = AsyncMock(spec=httpx.AsyncClient)
        syncer._http.aclose = AsyncMock()

        html_content = (
            b"<html><body>"
            b"<h1>Banking Regulation</h1>"
            b"<p>This regulation sets the rules for the banking sector.</p>"
            b"<h2>ARTICLE 1</h2>"
            b"<p>Purpose and scope</p>"
            b"</body></html>"
        )

        async def mock_download(doc_id):
            return html_content, "bddk_direct", ".html"

        syncer._download_bddk = mock_download

        # Use numeric doc_id (BDDK format)
        result = await syncer.sync_document(
            doc_id="9999",
            title="Test Regulation",
            category="Yonetmelik",
            force=True,
        )

        assert result.success
        assert "html_parser" in result.method or "markitdown" in result.method

        # Verify it's in the store
        stored = await store.get_document("9999")
        assert stored is not None
        assert stored.markdown_content

        await store.close()


class TestConfigIntegration:
    """Test that config values are properly used across modules."""

    def test_page_size_consistent_across_modules(self):
        # client.py, doc_store.py, and vector_store.py should all use PAGE_SIZE
        import client
        import doc_store
        from config import PAGE_SIZE

        # They import PAGE_SIZE from config — verify it's the same value
        assert client.PAGE_SIZE == PAGE_SIZE
        assert doc_store.PAGE_SIZE == PAGE_SIZE
