"""Integration tests: end-to-end flows across multiple modules."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from client import BddkApiClient
from doc_store import StoredDocument
from doc_sync import DocumentSyncer
from models import BddkSearchRequest
from tests.conftest import (
    BDDK_ACCORDION_HTML,
    make_http_response,
)


class TestSearchThenRetrieveFlow:
    """Test the full search -> retrieve -> store flow."""

    @pytest.mark.asyncio
    async def test_search_then_get_document(self, doc_store):
        """Search for decisions, pick one, retrieve its markdown."""
        client = BddkApiClient(pool=doc_store._pool, doc_store=doc_store)
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.get = AsyncMock(return_value=make_http_response(BDDK_ACCORDION_HTML))

        # Populate cache directly (bypass DB cache)
        client._cache = []
        client._cache_timestamp = 0
        await client._ensure_cache()
        assert client.cache_size() > 0

        request = BddkSearchRequest(keywords="Rehber")
        result = await client.search_decisions(request)
        assert result.total_results > 0

        doc_html = "<html><body><h1>Sermaye Rehberi</h1><p>Bu rehber bankacilik sektorunde...</p></body></html>"
        client._http.get = AsyncMock(return_value=make_http_response(doc_html))

        doc = await client.get_document_markdown(result.decisions[0].document_id, 1)
        assert doc.markdown_content
        assert doc.page_number == 1

        await client.close()

    @pytest.mark.asyncio
    async def test_store_first_strategy(self, doc_store):
        """Verify documents are served from store if available."""
        doc = StoredDocument(
            document_id="1291",
            title="Test Doc",
            markdown_content="# Test\n\nThis is a test document with enough content.",
            extraction_method="markitdown",
        )
        await doc_store.store_document(doc)

        client = BddkApiClient(pool=doc_store._pool, doc_store=doc_store)
        client._http = AsyncMock(spec=httpx.AsyncClient)

        result = await client.get_document_markdown("1291", 1)
        assert "Test" in result.markdown_content
        client._http.get.assert_not_called()

        await client.close()


class TestSyncThenSearchFlow:
    """Test sync -> store -> search flow."""

    @pytest.mark.asyncio
    async def test_sync_and_fts_search(self, doc_store):
        """Sync a document, then search for it via FTS."""
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
        await doc_store.store_document(doc)

        hits = await doc_store.search_content("sermaye yeterliliği")
        assert len(hits) > 0
        assert hits[0].document_id == "42"

        page = await doc_store.get_document_page("42", 1)
        assert page is not None
        assert "sermaye" in page.markdown_content.lower()


class TestCacheFallbackFlow:
    """Test the cache fallback chain: memory -> DB -> stale DB."""

    @pytest.mark.asyncio
    async def test_stale_db_cache_used_when_bddk_unreachable(self, doc_store):
        """When BDDK is unreachable, stale DB cache should be served."""
        pool = doc_store._pool  # SingleConnPool from fixture

        # Pre-populate DB cache
        client = BddkApiClient(pool=pool)
        await client.initialize()
        from models import BddkDecisionSummary

        client._cache = [
            BddkDecisionSummary(title="Stale Reg", document_id="stale_test_1", content="Old", category="Yönetmelik")
        ]
        client._cache_timestamp = 1.0  # very old
        await client._save_cache_to_db()

        # New client, all HTTP fails
        client2 = BddkApiClient(pool=pool)
        await client2.initialize()
        client2._http = AsyncMock(spec=httpx.AsyncClient)
        client2._http.get = AsyncMock(side_effect=httpx.TransportError("Network unreachable"))

        with patch("client.STALE_CACHE_FALLBACK", True):
            await client2._ensure_cache()

        # Should have loaded stale cache from DB
        assert client2.cache_size() >= 1
        assert any(d.document_id == "stale_test_1" for d in client2.get_cache_items())

        await client.close()
        await client2.close()


class TestExtractionPipelineFlow:
    """Test the extraction pipeline with various content types."""

    @pytest.mark.asyncio
    async def test_html_extraction_end_to_end(self, doc_store):
        syncer = DocumentSyncer(doc_store, prefer_nougat=False)
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

        result = await syncer.sync_document(
            doc_id="9999",
            title="Test Regulation",
            category="Yonetmelik",
            force=True,
        )

        assert result.success
        assert "html_parser" in result.method or "markitdown" in result.method

        stored = await doc_store.get_document("9999")
        assert stored is not None
        assert stored.markdown_content


class TestConfigIntegration:
    """Test that config values are properly used across modules."""

    def test_page_size_consistent_across_modules(self):
        import client
        import doc_store
        from config import PAGE_SIZE

        assert client.PAGE_SIZE == PAGE_SIZE
        assert doc_store.PAGE_SIZE == PAGE_SIZE
