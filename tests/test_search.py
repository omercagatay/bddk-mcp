"""Unit tests for search logic in BddkApiClient."""

import pytest

from client import BddkApiClient
from models import BddkDecisionSummary, BddkSearchRequest
from tests.conftest import MockPool


def _make_client_with_cache(items: list[BddkDecisionSummary]) -> BddkApiClient:
    """Create a client with a pre-populated cache (no HTTP needed)."""
    client = BddkApiClient(pool=MockPool())
    client._cache = items
    client._cache_timestamp = 9999999999.0  # far future so cache stays valid
    return client


SAMPLE_ITEMS = [
    BddkDecisionSummary(
        title="Bankaların Kredi İşlemlerine İlişkin Yönetmelik",
        document_id="mevzuat_40520",
        content="Bankaların Kredi İşlemlerine İlişkin Yönetmelik",
        category="Yönetmelik",
        source_url="https://mevzuat.gov.tr/mevzuat?MevzuatNo=40520",
    ),
    BddkDecisionSummary(
        title="2025/1 Sayılı Genelge: Bankacılık Hesapları",
        document_id="1296",
        content="Bankacılık hesapları hakkında genelge",
        category="Genelge",
        source_url="https://www.bddk.org.tr/Mevzuat/DokumanGetir/1296",
    ),
    BddkDecisionSummary(
        title="Ziraat Bankası faaliyet izni",
        document_id="1280",
        content="Ziraat Bankası faaliyet izni",
        decision_date="31.10.2024",
        decision_number="11000",
        category="Kurul Kararı",
        source_url="https://www.bddk.org.tr/Mevzuat/DokumanGetir/1280",
    ),
    BddkDecisionSummary(
        title="Katılım bankası kuruluş izni",
        document_id="1281",
        content="Katılım bankası kuruluş izni",
        decision_date="15.06.2023",
        decision_number="10800",
        category="Kurul Kararı",
        source_url="https://www.bddk.org.tr/Mevzuat/DokumanGetir/1281",
    ),
    BddkDecisionSummary(
        title="5411 Sayılı Bankacılık Kanunu",
        document_id="mevzuat_5411",
        content="Bankacılık Kanunu",
        category="Kanun",
        source_url="https://mevzuat.gov.tr/mevzuat?MevzuatNo=5411&MevzuatTur=1",
    ),
]


@pytest.mark.asyncio
async def test_basic_keyword_search():
    client = _make_client_with_cache(SAMPLE_ITEMS)
    req = BddkSearchRequest(keywords="banka")
    result = await client.search_decisions(req)
    # Should match items containing "banka" in title/content/category
    assert result.total_results >= 3


@pytest.mark.asyncio
async def test_category_filter():
    client = _make_client_with_cache(SAMPLE_ITEMS)
    req = BddkSearchRequest(keywords="banka", category="Yönetmelik")
    result = await client.search_decisions(req)
    assert result.total_results >= 1
    assert all(d.category == "Yönetmelik" for d in result.decisions)


@pytest.mark.asyncio
async def test_date_from_filter():
    client = _make_client_with_cache(SAMPLE_ITEMS)
    req = BddkSearchRequest(keywords="banka", date_from="01.01.2024")
    result = await client.search_decisions(req)
    # Only the 2024 Kurul Kararı should match (2023 one excluded, no-date items excluded)
    assert result.total_results == 1
    assert result.decisions[0].decision_date == "31.10.2024"


@pytest.mark.asyncio
async def test_date_range_filter():
    client = _make_client_with_cache(SAMPLE_ITEMS)
    req = BddkSearchRequest(keywords="banka", date_from="01.01.2023", date_to="31.12.2023")
    result = await client.search_decisions(req)
    assert result.total_results == 1
    assert result.decisions[0].decision_date == "15.06.2023"


@pytest.mark.asyncio
async def test_date_filter_excludes_no_date_items():
    client = _make_client_with_cache(SAMPLE_ITEMS)
    req = BddkSearchRequest(keywords="banka", date_from="01.01.2020")
    result = await client.search_decisions(req)
    # Only items WITH dates should appear
    assert all(d.decision_date for d in result.decisions)


@pytest.mark.asyncio
async def test_pagination():
    client = _make_client_with_cache(SAMPLE_ITEMS)
    req = BddkSearchRequest(keywords="banka", page=1, page_size=2)
    result = await client.search_decisions(req)
    assert len(result.decisions) == 2
    assert result.total_results >= 3

    req2 = BddkSearchRequest(keywords="banka", page=2, page_size=2)
    result2 = await client.search_decisions(req2)
    assert len(result2.decisions) >= 1


@pytest.mark.asyncio
async def test_relevance_ranking():
    client = _make_client_with_cache(SAMPLE_ITEMS)
    req = BddkSearchRequest(keywords="kredi")
    result = await client.search_decisions(req)
    # "Kredi" appears directly in title of first item, should be ranked higher
    assert result.total_results >= 1
    assert "Kredi" in result.decisions[0].title


@pytest.mark.asyncio
async def test_turkish_stemming():
    client = _make_client_with_cache(SAMPLE_ITEMS)
    # "Bankaların" should match "Banka" via stemming
    req = BddkSearchRequest(keywords="bankaların")
    result = await client.search_decisions(req)
    assert result.total_results >= 1


@pytest.mark.asyncio
async def test_no_results():
    client = _make_client_with_cache(SAMPLE_ITEMS)
    req = BddkSearchRequest(keywords="xyznomatch")
    result = await client.search_decisions(req)
    assert result.total_results == 0
    assert result.decisions == []


@pytest.mark.asyncio
async def test_cache_status():
    client = _make_client_with_cache(SAMPLE_ITEMS)
    status = client.cache_status()
    assert status["total_items"] == len(SAMPLE_ITEMS)
    assert status["cache_valid"] is True
    assert "Yönetmelik" in status["categories"]


@pytest.mark.asyncio
async def test_resolve_document_url_numeric():
    client = _make_client_with_cache(SAMPLE_ITEMS)
    url = client._resolve_document_url("1296")
    assert url == "https://www.bddk.org.tr/Mevzuat/DokumanGetir/1296"


@pytest.mark.asyncio
async def test_resolve_document_url_mevzuat():
    client = _make_client_with_cache(SAMPLE_ITEMS)
    url = client._resolve_document_url("mevzuat_40520")
    assert "mevzuat.gov.tr" in url
    assert "40520" in url


@pytest.mark.asyncio
async def test_page_validation():
    client = _make_client_with_cache(SAMPLE_ITEMS)
    # We can't test get_document_markdown without HTTP, but we can test _resolve
    url = client._resolve_document_url("999999")
    assert "DokumanGetir/999999" in url
