"""Tests for BddkApiClient: HTTP scraping, caching, and document retrieval."""

import time
from unittest.mock import AsyncMock

import httpx
import pytest

from client import BddkApiClient
from models import BddkDecisionSummary
from tests.conftest import BDDK_ACCORDION_HTML, BDDK_DECISION_HTML, MockPool, make_http_response


def _make_client(**kwargs) -> BddkApiClient:
    """Create a client with a mock pool (no real DB)."""
    c = BddkApiClient(pool=MockPool(), **kwargs)
    return c


class TestFetchWithRetry:
    @pytest.mark.asyncio
    async def test_success_first_try(self):
        client = _make_client()
        client._http = AsyncMock(spec=httpx.AsyncClient)
        resp = make_http_response("OK")
        client._http.get = AsyncMock(return_value=resp)

        result = await client._fetch_with_retry("https://example.com")
        assert result.text == "OK"
        assert client._http.get.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_transport_error(self):
        client = _make_client()
        client._http = AsyncMock(spec=httpx.AsyncClient)
        resp = make_http_response("OK")
        client._http.get = AsyncMock(side_effect=[httpx.TransportError("timeout"), resp])

        result = await client._fetch_with_retry("https://example.com")
        assert result.text == "OK"
        assert client._http.get.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        client = _make_client()
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.get = AsyncMock(side_effect=httpx.TransportError("timeout"))

        with pytest.raises(httpx.TransportError):
            await client._fetch_with_retry("https://example.com")
        assert client._http.get.call_count == 3


class TestCachePersistence:
    """Test cache save/load via PostgreSQL."""

    @pytest.mark.asyncio
    async def test_save_and_load_cache(self, doc_store):
        """Use doc_store fixture which provides a transactional pool wrapper."""

        pool = doc_store._pool  # SingleConnPool wrapping a real connection
        client = BddkApiClient(pool=pool)
        await client.initialize()

        client._cache = [
            BddkDecisionSummary(title="Test", document_id="cache_test_1", content="test", category="Rehber")
        ]
        client._cache_timestamp = time.time()

        await client._save_cache_to_db()

        # Same pool should load cached data
        client2 = BddkApiClient(pool=pool)
        loaded = await client2._load_cache_from_db()
        assert loaded
        assert len(client2._cache) >= 1
        assert any(d.document_id == "cache_test_1" for d in client2._cache)

        await client.close()
        await client2.close()

    @pytest.mark.asyncio
    async def test_load_empty_cache(self):
        """Empty DB returns False."""
        pool = MockPool()
        client = BddkApiClient(pool=pool)
        loaded = await client._load_cache_from_db()
        assert not loaded


class TestAccordionParsing:
    @pytest.mark.asyncio
    async def test_parse_accordion_page(self):
        client = _make_client()
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.get = AsyncMock(return_value=make_http_response(BDDK_ACCORDION_HTML))

        decisions = await client._fetch_and_parse_accordion_page(50)
        assert len(decisions) >= 1
        bddk_docs = [d for d in decisions if d.document_id == "1291"]
        assert len(bddk_docs) == 1

    @pytest.mark.asyncio
    async def test_parse_accordion_empty(self):
        client = _make_client()
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.get = AsyncMock(return_value=make_http_response("<html><body></body></html>"))

        decisions = await client._fetch_and_parse_accordion_page(50)
        assert decisions == []


class TestDecisionParsing:
    @pytest.mark.asyncio
    async def test_parse_decision_page(self):
        client = _make_client()
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.get = AsyncMock(return_value=make_http_response(BDDK_DECISION_HTML))

        decisions = await client._fetch_and_parse_decision_page(55)
        assert len(decisions) == 2
        assert decisions[0].decision_date == "31.10.2024"
        assert decisions[0].decision_number == "11000"
        assert decisions[0].category == "Kurul Kararı"

    @pytest.mark.asyncio
    async def test_parse_decision_page_empty(self):
        client = _make_client()
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.get = AsyncMock(return_value=make_http_response("<html></html>"))

        decisions = await client._fetch_and_parse_decision_page(55)
        assert decisions == []


class TestDocumentUrlResolution:
    def test_numeric_id(self):
        client = _make_client()
        url = client._resolve_document_url("1296")
        assert url == "https://www.bddk.org.tr/Mevzuat/DokumanGetir/1296"

    def test_mevzuat_id_with_cache(self):
        client = _make_client()
        client._cache = [
            BddkDecisionSummary(
                title="Test",
                document_id="mevzuat_42628",
                content="",
                source_url="https://mevzuat.gov.tr/mevzuat?MevzuatNo=42628&MevzuatTur=7&MevzuatTertip=5",
            )
        ]
        url = client._resolve_document_url("mevzuat_42628")
        assert "mevzuat.gov.tr" in url
        assert "42628" in url

    def test_mevzuat_id_fallback(self):
        client = _make_client()
        client._cache = []
        url = client._resolve_document_url("mevzuat_99999")
        assert "mevzuat.gov.tr" in url
        assert "99999" in url


class TestCacheValidity:
    def test_empty_cache_invalid(self):
        client = _make_client()
        assert not client._is_cache_valid()

    def test_fresh_cache_valid(self):
        client = _make_client()
        client._cache = [BddkDecisionSummary(title="T", document_id="1", content="")]
        client._cache_timestamp = time.time()
        assert client._is_cache_valid()

    def test_expired_cache_invalid(self):
        client = _make_client()
        client._cache = [BddkDecisionSummary(title="T", document_id="1", content="")]
        client._cache_timestamp = time.time() - 7200
        assert not client._is_cache_valid()


class TestCacheStatus:
    def test_cache_status_with_data(self):
        client = _make_client()
        client._cache = [
            BddkDecisionSummary(title="A", document_id="1", content="", category="Rehber"),
            BddkDecisionSummary(title="B", document_id="2", content="", category="Genelge"),
            BddkDecisionSummary(title="C", document_id="3", content="", category="Rehber"),
        ]
        client._cache_timestamp = time.time()

        status = client.cache_status()
        assert status["total_items"] == 3
        assert status["cache_valid"] is True
        assert status["categories"]["Rehber"] == 2
        assert status["categories"]["Genelge"] == 1

    def test_cache_status_empty(self):
        client = _make_client()
        status = client.cache_status()
        assert status["total_items"] == 0
        assert status["cache_valid"] is False


class TestPublicCacheAPI:
    """Tests for BddkApiClient public cache API methods."""

    def _client_with_cache(self) -> BddkApiClient:
        client = _make_client()
        client._cache = [
            BddkDecisionSummary(title="Rehber A", document_id="doc-1", content="", category="Rehber"),
            BddkDecisionSummary(title="Genelge B", document_id="doc-2", content="", category="Genelge"),
        ]
        return client

    def test_get_cache_items_returns_copy(self):
        client = self._client_with_cache()
        items = client.get_cache_items()
        assert len(items) == 2
        # Mutating the returned list must not affect internal cache
        items.clear()
        assert client.cache_size() == 2

    def test_get_cache_items_empty(self):
        client = _make_client()
        assert client.get_cache_items() == []

    def test_find_by_id_found(self):
        client = self._client_with_cache()
        result = client.find_by_id("doc-1")
        assert result is not None
        assert result.title == "Rehber A"

    def test_find_by_id_not_found(self):
        client = self._client_with_cache()
        assert client.find_by_id("nonexistent") is None

    def test_cache_size(self):
        client = self._client_with_cache()
        assert client.cache_size() == 2

    def test_cache_size_empty(self):
        client = _make_client()
        assert client.cache_size() == 0
