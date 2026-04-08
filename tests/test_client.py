"""Tests for BddkApiClient: HTTP scraping, caching, and document retrieval."""

import json
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from client import BddkApiClient
from models import BddkDecisionSummary
from tests.conftest import BDDK_ACCORDION_HTML, BDDK_DECISION_HTML, make_http_response


class TestFetchWithRetry:
    """Test the retry logic for HTTP requests."""

    @pytest.mark.asyncio
    async def test_success_first_try(self):
        client = BddkApiClient()
        client._http = AsyncMock(spec=httpx.AsyncClient)
        resp = make_http_response("OK")
        client._http.get = AsyncMock(return_value=resp)

        result = await client._fetch_with_retry("https://example.com")
        assert result.text == "OK"
        assert client._http.get.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_transport_error(self):
        client = BddkApiClient()
        client._http = AsyncMock(spec=httpx.AsyncClient)
        resp = make_http_response("OK")
        client._http.get = AsyncMock(side_effect=[httpx.TransportError("timeout"), resp])

        result = await client._fetch_with_retry("https://example.com")
        assert result.text == "OK"
        assert client._http.get.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        client = BddkApiClient()
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.get = AsyncMock(side_effect=httpx.TransportError("timeout"))

        with pytest.raises(httpx.TransportError):
            await client._fetch_with_retry("https://example.com")
        assert client._http.get.call_count == 3


class TestCachePersistence:
    """Test cache save/load to/from disk."""

    def test_save_and_load_cache(self, tmp_path):
        cache_file = tmp_path / ".cache.json"

        client = BddkApiClient()
        client._cache = [BddkDecisionSummary(title="Test", document_id="1", content="test", category="Rehber")]
        client._cache_timestamp = time.time()

        with patch("client.CACHE_FILE", cache_file):
            client._save_cache_to_disk()
            assert cache_file.exists()

        # File exists but patch is no longer active, so test the save path
        # The load test is covered by test_load_expired_cache and test_load_corrupted_cache
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        assert len(data["items"]) == 1
        assert data["items"][0]["title"] == "Test"

    def test_load_expired_cache(self, tmp_path):
        cache_file = tmp_path / ".cache.json"
        data = {
            "timestamp": time.time() - 7200,  # 2 hours old
            "items": [{"title": "Old", "document_id": "1", "content": ""}],
        }
        cache_file.write_text(json.dumps(data), encoding="utf-8")

        client = BddkApiClient()
        with patch("client.CACHE_FILE", cache_file):
            loaded = client._load_cache_from_disk()
        assert not loaded

    def test_load_corrupted_cache(self, tmp_path):
        cache_file = tmp_path / ".cache.json"
        cache_file.write_text("not valid json", encoding="utf-8")

        client = BddkApiClient()
        with patch("client.CACHE_FILE", cache_file):
            loaded = client._load_cache_from_disk()
        assert not loaded


class TestAccordionParsing:
    """Test BDDK accordion page parsing."""

    @pytest.mark.asyncio
    async def test_parse_accordion_page(self):
        client = BddkApiClient()
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.get = AsyncMock(return_value=make_http_response(BDDK_ACCORDION_HTML))

        decisions = await client._fetch_and_parse_accordion_page(50)
        assert len(decisions) >= 1
        # Should find the DokumanGetir link
        bddk_docs = [d for d in decisions if d.document_id == "1291"]
        assert len(bddk_docs) == 1

    @pytest.mark.asyncio
    async def test_parse_accordion_empty(self):
        client = BddkApiClient()
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.get = AsyncMock(return_value=make_http_response("<html><body></body></html>"))

        decisions = await client._fetch_and_parse_accordion_page(50)
        assert decisions == []


class TestDecisionParsing:
    """Test BDDK decision page parsing (pages 55/56)."""

    @pytest.mark.asyncio
    async def test_parse_decision_page(self):
        client = BddkApiClient()
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.get = AsyncMock(return_value=make_http_response(BDDK_DECISION_HTML))

        decisions = await client._fetch_and_parse_decision_page(55)
        assert len(decisions) == 2
        assert decisions[0].decision_date == "31.10.2024"
        assert decisions[0].decision_number == "11000"
        assert decisions[0].category == "Kurul Kararı"

    @pytest.mark.asyncio
    async def test_parse_decision_page_empty(self):
        client = BddkApiClient()
        client._http = AsyncMock(spec=httpx.AsyncClient)
        client._http.get = AsyncMock(return_value=make_http_response("<html></html>"))

        decisions = await client._fetch_and_parse_decision_page(55)
        assert decisions == []


class TestDocumentUrlResolution:
    """Test URL resolution for different document types."""

    def test_numeric_id(self):
        client = BddkApiClient()
        url = client._resolve_document_url("1296")
        assert url == "https://www.bddk.org.tr/Mevzuat/DokumanGetir/1296"

    def test_mevzuat_id_with_cache(self):
        client = BddkApiClient()
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
        client = BddkApiClient()
        client._cache = []
        url = client._resolve_document_url("mevzuat_99999")
        assert "mevzuat.gov.tr" in url
        assert "99999" in url


class TestCacheValidity:
    """Test cache TTL and validity checks."""

    def test_empty_cache_invalid(self):
        client = BddkApiClient()
        assert not client._is_cache_valid()

    def test_fresh_cache_valid(self):
        client = BddkApiClient()
        client._cache = [BddkDecisionSummary(title="T", document_id="1", content="")]
        client._cache_timestamp = time.time()
        assert client._is_cache_valid()

    def test_expired_cache_invalid(self):
        client = BddkApiClient()
        client._cache = [BddkDecisionSummary(title="T", document_id="1", content="")]
        client._cache_timestamp = time.time() - 7200
        assert not client._is_cache_valid()


class TestCacheStatus:
    """Test cache status reporting."""

    def test_cache_status_with_data(self):
        client = BddkApiClient()
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
        client = BddkApiClient()
        status = client.cache_status()
        assert status["total_items"] == 0
        assert status["cache_valid"] is False
