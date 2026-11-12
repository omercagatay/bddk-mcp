"""Tests for BddkApiClient: HTTP scraping, caching, and document retrieval."""

import logging
import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from bddk_mcp.core.exceptions import BddkStorageError, BddkUpstreamError
from bddk_mcp.core.models import BddkDecisionSummary, BddkSearchRequest
from bddk_mcp.ingest.client import (
    _ALL_PAGE_IDS,
    _DECISION_PAGE_IDS,
    _EXCLUDED_CATEGORIES,
    _EXCLUDED_TITLE_SUBSTRINGS,
    _FLAT_PAGE_IDS,
    BddkApiClient,
    _is_in_scope,
)
from tests.conftest import BDDK_ACCORDION_HTML, BDDK_DECISION_HTML, MockStore, make_http_response


def _make_client(**kwargs) -> BddkApiClient:
    """Create a client with a store double (no corpus file)."""
    c = BddkApiClient(store=MockStore(), **kwargs)
    return c


@pytest.mark.asyncio
async def test_search_operational_log_omits_raw_regulatory_query(caplog):
    """Internal research terms must not cross the production log boundary."""

    sentinel = "PRIVATE_AUDIT_SEARCH_TERM_7f31"
    client = _make_client()
    client._cache = [
        BddkDecisionSummary(
            title=sentinel,
            document_id="privacy-log-proof",
            category="Düzenleme",
        )
    ]
    client._cache_timestamp = time.time()

    with caplog.at_level(logging.INFO, logger="bddk_mcp.ingest.client"):
        result = await client.search_decisions(BddkSearchRequest(keywords=sentinel))

    assert result.total_results == 1
    assert sentinel not in caplog.text
    assert f"keyword_chars={len(sentinel)}" in caplog.text
    assert "keyword_terms=1 matches=1 page=1 returned=1" in caplog.text


class TestFetchWithRetry:
    @pytest.mark.asyncio
    async def test_success_first_try(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, text="OK", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = _make_client(http=http)
            client._assert_public_resolution = AsyncMock()
            result = await client._fetch_with_retry("https://www.bddk.org.tr/Mevzuat/Liste/50")

        assert result.text == "OK"
        assert calls == 1

    @pytest.mark.asyncio
    async def test_retry_on_transport_error(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ConnectError("timeout", request=request)
            return httpx.Response(200, text="OK", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = _make_client(http=http)
            client._assert_public_resolution = AsyncMock()
            with patch("bddk_mcp.core.outbound_http.asyncio.sleep", new=AsyncMock()):
                result = await client._fetch_with_retry("https://www.bddk.org.tr/Mevzuat/Liste/50")

        assert result.text == "OK"
        assert calls == 2

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ConnectError("timeout", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = _make_client(http=http)
            client._assert_public_resolution = AsyncMock()
            with (
                patch("bddk_mcp.core.outbound_http.asyncio.sleep", new=AsyncMock()),
                pytest.raises(httpx.TransportError),
            ):
                await client._fetch_with_retry("https://www.bddk.org.tr/Mevzuat/Liste/50")

        assert calls == 3


class TestCachePersistence:
    """Test cache save/load through the corpus file."""

    @pytest.mark.asyncio
    async def test_save_and_load_cache(self, doc_store):
        """Use the doc_store fixture, which is backed by a temporary corpus file."""

        client = BddkApiClient(store=doc_store)

        client._cache = [
            BddkDecisionSummary(title="Test", document_id="cache_test_1", content="test", category="Rehber")
        ]
        client._cache_timestamp = time.time()

        await client._save_cache_to_db()

        client2 = BddkApiClient(store=doc_store)
        loaded = await client2._load_cache_from_db()
        assert loaded
        assert len(client2._cache) >= 1
        assert any(d.document_id == "cache_test_1" for d in client2._cache)

        await client.close()
        await client2.close()

    @pytest.mark.asyncio
    async def test_complete_catalog_publication_retires_removed_upstream_rows(self, doc_store):
        """A successful full refresh must publish exact membership, not append forever."""

        client = BddkApiClient(store=doc_store)
        client._cache = [
            BddkDecisionSummary(title="Retained", document_id="catalog-retained", category="Rehber"),
            BddkDecisionSummary(title="Removed", document_id="catalog-removed", category="Rehber"),
        ]
        await client._save_cache_to_db()

        client._cache = [
            BddkDecisionSummary(title="Retained", document_id="catalog-retained", category="Rehber"),
        ]
        await client._save_cache_to_db()

        rows = await doc_store.load_catalog()
        assert [row["document_id"] for row in rows] == ["catalog-retained"]

    @pytest.mark.asyncio
    async def test_empty_catalog_cannot_erase_last_known_good_publication(self):
        client = _make_client()
        client._cache = []

        with pytest.raises(BddkStorageError, match="empty decision catalog"):
            await client._save_cache_to_db()

    @pytest.mark.asyncio
    async def test_load_empty_cache(self):
        """An empty catalog loads as False."""
        client = BddkApiClient(store=MockStore())
        loaded = await client._load_cache_from_db()
        assert not loaded

    @pytest.mark.asyncio
    async def test_context_entry_only_reads_the_catalog(self):
        store = MockStore()
        client = BddkApiClient(store=store, http=AsyncMock(spec=httpx.AsyncClient))

        async with client as entered:
            assert entered is client

        store.load_catalog.assert_awaited_once_with()
        store.replace_catalog.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_serving_mode_never_populates_an_empty_cache_from_network(self):
        client = BddkApiClient(store=MockStore(), allow_live_population=False)
        client._scrape_bddk = AsyncMock()

        with pytest.raises(BddkStorageError, match="will not populate it from the network"):
            await client._ensure_cache()

        client._scrape_bddk.assert_not_awaited()


class TestAccordionParsing:
    @pytest.mark.asyncio
    async def test_parse_accordion_page(self):
        client = _make_client()
        client._fetch_with_retry = AsyncMock(return_value=make_http_response(BDDK_ACCORDION_HTML))

        decisions = await client._fetch_and_parse_accordion_page(50)
        assert len(decisions) >= 1
        bddk_docs = [d for d in decisions if d.document_id == "1291"]
        assert len(bddk_docs) == 1

    @pytest.mark.asyncio
    async def test_parse_accordion_empty(self):
        client = _make_client()
        client._fetch_with_retry = AsyncMock(return_value=make_http_response("<html><body></body></html>"))

        decisions = await client._fetch_and_parse_accordion_page(50)
        assert decisions == []


class TestDecisionParsing:
    @pytest.mark.asyncio
    async def test_parse_decision_page(self):
        client = _make_client()
        client._fetch_with_retry = AsyncMock(return_value=make_http_response(BDDK_DECISION_HTML))

        decisions = await client._fetch_and_parse_decision_page(55)
        assert len(decisions) == 2
        assert decisions[0].decision_date == "31.10.2024"
        assert decisions[0].decision_number == "11000"
        assert decisions[0].category == "Kurul Kararı"

    @pytest.mark.asyncio
    async def test_parse_decision_page_empty(self):
        client = _make_client()
        client._fetch_with_retry = AsyncMock(return_value=make_http_response("<html></html>"))

        decisions = await client._fetch_and_parse_decision_page(55)
        assert decisions == []


class TestCacheRefreshIntegrity:
    """A refresh must never publish a partial or unpersisted catalog."""

    @staticmethod
    def _decision(document_id: str) -> BddkDecisionSummary:
        return BddkDecisionSummary(title=f"Document {document_id}", document_id=document_id, content="")

    def _mock_all_pages(self, client: BddkApiClient) -> None:
        client._fetch_and_parse_accordion_page = AsyncMock(side_effect=lambda page_id: [self._decision(str(page_id))])
        client._fetch_and_parse_decision_page = AsyncMock(side_effect=lambda page_id: [self._decision(str(page_id))])
        client._fetch_and_parse_flat_page = AsyncMock(side_effect=lambda page_id: [self._decision(str(page_id))])
        client._save_cache_to_db = AsyncMock()

    @pytest.mark.asyncio
    async def test_complete_refresh_publishes_and_persists_new_catalog(self):
        client = _make_client()
        self._mock_all_pages(client)

        count = await client.refresh_cache()

        assert count == len(_ALL_PAGE_IDS)
        assert {item.document_id for item in client.get_cache_items()} == {str(page_id) for page_id in _ALL_PAGE_IDS}
        client._save_cache_to_db.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_exact_duplicate_catalog_links_are_published_once(self):
        client = _make_client()
        duplicate = self._decision("shared-document")
        client._fetch_and_parse_accordion_page = AsyncMock(return_value=[duplicate])
        client._fetch_and_parse_decision_page = AsyncMock(return_value=[duplicate])
        client._fetch_and_parse_flat_page = AsyncMock(return_value=[duplicate])
        client._save_cache_to_db = AsyncMock()

        count = await client.refresh_cache()

        assert count == 1
        assert [item.document_id for item in client.get_cache_items()] == ["shared-document"]
        client._save_cache_to_db.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_conflicting_duplicate_catalog_links_retain_last_known_good(self):
        client = _make_client()
        previous = self._decision("last-known-good")
        client._cache = [previous]
        client._cache_timestamp = 789.0
        left = self._decision("shared-document")
        right = left.model_copy(update={"title": "Conflicting upstream title"})
        client._fetch_and_parse_accordion_page = AsyncMock(return_value=[left])
        client._fetch_and_parse_decision_page = AsyncMock(return_value=[right])
        client._fetch_and_parse_flat_page = AsyncMock(return_value=[left])
        client._save_cache_to_db = AsyncMock()

        with pytest.raises(BddkUpstreamError, match="conflicting records"):
            await client.refresh_cache()

        assert client.get_cache_items() == [previous]
        assert client._cache_timestamp == 789.0
        client._save_cache_to_db.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_partial_fetch_failure_retains_last_known_good_catalog(self):
        client = _make_client()
        previous = self._decision("last-known-good")
        client._cache = [previous]
        client._cache_timestamp = 123.0
        self._mock_all_pages(client)
        sentinel = "sensitive-upstream-detail"

        async def flat_page(page_id: int):
            if page_id == _FLAT_PAGE_IDS[0]:
                raise httpx.TransportError(sentinel)
            return [self._decision(str(page_id))]

        client._fetch_and_parse_flat_page = AsyncMock(side_effect=flat_page)

        with pytest.raises(BddkUpstreamError, match="incomplete"):
            await client.refresh_cache()

        assert client.get_cache_items() == [previous]
        assert client._cache_timestamp == 123.0
        assert sentinel not in repr(client._page_errors)
        client._save_cache_to_db.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_parsed_page_rejects_refresh(self):
        client = _make_client()
        previous = self._decision("last-known-good")
        client._cache = [previous]
        self._mock_all_pages(client)
        client._fetch_and_parse_decision_page = AsyncMock(return_value=[])

        with pytest.raises(BddkUpstreamError, match="incomplete"):
            await client.refresh_cache()

        assert client.get_cache_items() == [previous]
        client._save_cache_to_db.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_persistence_failure_rolls_back_in_memory_catalog(self):
        client = _make_client()
        previous = self._decision("last-known-good")
        client._cache = [previous]
        client._cache_timestamp = 456.0
        self._mock_all_pages(client)
        client._save_cache_to_db = AsyncMock(side_effect=BddkStorageError("safe"))

        with pytest.raises(BddkStorageError):
            await client.refresh_cache()

        assert client.get_cache_items() == [previous]
        assert client._cache_timestamp == 456.0


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
        assert len(client.get_cache_items()) == 2

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
        assert len(client.get_cache_items()) == 2

    def test_cache_size_empty(self):
        client = _make_client()
        assert len(client.get_cache_items()) == 0


class TestScopeFilter:
    """Tests for _is_in_scope: excludes items not relevant to a conventional bank."""

    def _dec(self, title: str = "T", category: str = "") -> BddkDecisionSummary:
        return BddkDecisionSummary(title=title, document_id="x", content="", category=category)

    def test_keeps_in_scope_item(self):
        assert _is_in_scope(self._dec(title="Kredi Riski Azaltım Tebliği", category="Tebliğ")) is True

    def test_drops_faizsiz_bankacilik_category(self):
        assert _is_in_scope(self._dec(title="Herhangi bir başlık", category="Faizsiz Bankacılık")) is False

    def test_drops_6361_sayili_title(self):
        assert _is_in_scope(self._dec(title="6361 sayılı Kanun değişiklik", category="Kanun")) is False

    def test_keeps_kurul_karari(self):
        """_is_in_scope is generic — page-55 firm pattern is enforced at scrape time, not here."""
        assert _is_in_scope(self._dec(title="27.04.2023 #10585", category="Kurul Kararı")) is True

    def test_page_52_not_scraped(self):
        """Page 52 (Finansal Kiralama/Faktoring) is removed from scrape loop."""
        assert 52 not in _FLAT_PAGE_IDS

    def test_page_55_not_scraped(self):
        """Page 55 (firm-specific Kurul Kararları) is removed from decision page loop."""
        assert 55 not in _DECISION_PAGE_IDS

    def test_exclusion_constants_populated(self):
        assert "Faizsiz Bankacılık" in _EXCLUDED_CATEGORIES
        assert any("6361" in s for s in _EXCLUDED_TITLE_SUBSTRINGS)


def test_client_defaults_to_no_live_population():
    """The airlock must be the default: live population is an explicit opt-in."""
    client = BddkApiClient(store=MockStore())
    assert client._allow_live_population is False
