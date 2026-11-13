"""Unit tests for search logic in BddkApiClient."""

import pytest

from bddk_mcp.core.models import BddkDecisionSummary, BddkSearchRequest
from bddk_mcp.ingest.client import BddkApiClient
from tests.conftest import MockStore


def _make_client_with_cache(items: list[BddkDecisionSummary]) -> BddkApiClient:
    """Create a client with a pre-populated cache (no HTTP needed)."""
    client = BddkApiClient(store=MockStore())
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
async def test_date_filter_counts_the_undated_entries_it_could_not_consider():
    items = [
        BddkDecisionSummary(title="Kredi Yönetmeliği", document_id="y1", category="Yönetmelik"),
        BddkDecisionSummary(title="Kredi Rehberi", document_id="r1", category="Rehber"),
        BddkDecisionSummary(title="Faktoring Rehberi", document_id="r2", category="Rehber"),
        BddkDecisionSummary(
            title="Kredi kararı", document_id="k1", decision_date="01.02.2021", category="Kurul Kararı"
        ),
        BddkDecisionSummary(
            title="Eski kredi kararı", document_id="k2", decision_date="01.02.2001", category="Kurul Kararı"
        ),
    ]
    client = _make_client_with_cache(items)

    keyword = await client.search_decisions(BddkSearchRequest(keywords="kredi", date_from="01.01.2020"))
    browse = await client.search_decisions(BddkSearchRequest(keywords="", category="Rehber", date_from="01.01.2020"))
    undated_filter_absent = await client.search_decisions(BddkSearchRequest(keywords="kredi"))

    assert [d.document_id for d in keyword.decisions] == ["k1"]
    assert keyword.undated_excluded == 2
    assert browse.decisions == []
    assert browse.undated_excluded == 2
    assert undated_filter_absent.undated_excluded == 0


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


# -- Natural Turkish catalog queries (titles as in the BDDK catalog) ---------

NATURAL_CATALOG = [
    BddkDecisionSummary(
        title="Kredilerin Sınıflandırılması ve Bunlar İçin Ayrılacak Karşılıklara İlişkin Usul ve Esaslar Hakkında "
        "Yönetmelik",
        document_id="mevzuat_22599",
        category="Yönetmelik",
    ),
    BddkDecisionSummary(
        title="Kredilerin Donuk Alacak Olarak Sınıflandırılması için Öngörülen Gecikme Süresinin Değiştirilmesi ve "
        "Yeniden Yapılandırma hk. Kurul Kararı",
        document_id="775",
        decision_date="17.03.2020",
        decision_number="8948",
        category="Kurul Kararı",
    ),
    BddkDecisionSummary(
        title="Kredilerin İkinci Grupta Sınıflandırılması için Öngörülen Gecikme Süresinin Değiştirilmesi hk. Kurul "
        "Kararı",
        document_id="779",
        decision_date="27.03.2020",
        decision_number="8970",
        category="Kurul Kararı",
    ),
    BddkDecisionSummary(
        title="Bankaların Bilgi Sistemleri ve Elektronik Bankacılık Hizmetleri Hakkında Yönetmelik",
        document_id="mevzuat_34360",
        category="Bilgi Sistemleri",
    ),
    BddkDecisionSummary(
        title="Bilgi Sistemlerine İlişkin Sızma Testleri Hakkında Genelge",
        document_id="915",
        category="Bilgi Sistemleri",
    ),
    BddkDecisionSummary(
        title="Bankaların Sermaye Yeterliliğinin Ölçülmesine ve Değerlendirilmesine İlişkin Yönetmelik",
        document_id="mevzuat_21192",
        category="Yönetmelik",
    ),
    BddkDecisionSummary(
        title="Kredi Riski Azaltım Tekniklerine İlişkin Tebliğ",
        document_id="mevzuat_20029",
        category="Sermaye Yeterliliği",
    ),
    BddkDecisionSummary(
        title="On İli Etkileyen Deprem Felaketi Hk. Kurul Kararı (Sermaye Yeterliliği ve Özkaynak Hesaplamasına "
        "İlişkin)",
        document_id="1162",
        decision_date="14.02.2023",
        decision_number="10508",
        category="Kurul Kararı",
    ),
    BddkDecisionSummary(
        title="Bankaların Sermaye Yeterliliğinin Ölçülmesine ve Değerlendirilmesine İlişkin Yönetmelik Taslağı",
        document_id="1290",
        category="Düzenleme Taslağı",
    ),
]


async def _ids(keywords: str, **kwargs) -> list[str]:
    client = _make_client_with_cache(list(NATURAL_CATALOG))
    result = await client.search_decisions(BddkSearchRequest(keywords=keywords, **kwargs))
    return [d.document_id for d in result.decisions]


@pytest.mark.asyncio
async def test_inflected_document_type_word_is_not_required_literally_and_ranks_its_type_first():
    ids = await _ids("kredilerin sınıflandırılması yönetmeliği")
    assert ids[0] == "mevzuat_22599"
    # The Kurul Kararı titles still match the content words; they only rank lower.
    assert {"775", "779"} <= set(ids)


@pytest.mark.asyncio
async def test_bilgi_sistemleri_yonetmeligi_finds_the_regulation_first():
    ids = await _ids("bilgi sistemleri yönetmeliği")
    assert ids[0] == "mevzuat_34360"
    assert "915" in ids


@pytest.mark.asyncio
async def test_teblig_word_is_satisfied_by_title_and_content_words_by_the_category():
    ids = await _ids("sermaye yeterliliği tebliği")
    # mevzuat_20029 matches the content words only through its category; it is still found, but the
    # documents whose titles contain both content words rank first (exact title matches first).
    assert "mevzuat_20029" in ids
    assert ids.index("mevzuat_21192") < ids.index("mevzuat_20029")
    assert set(ids[: ids.index("mevzuat_20029")]) == {"mevzuat_21192", "1162", "1290"}


@pytest.mark.asyncio
async def test_exact_title_match_of_another_type_outranks_a_category_only_match_of_the_requested_type():
    client = _make_client_with_cache(
        [
            BddkDecisionSummary(
                title="Kredi Riski Azaltım Tekniklerine İlişkin Tebliğ",
                document_id="category_only",
                category="Sermaye Yeterliliği",
            ),
            BddkDecisionSummary(
                title="Bankaların Sermaye Yeterliliğinin Ölçülmesine İlişkin Yönetmelik",
                document_id="exact_title",
                category="Yönetmelik",
            ),
        ]
    )
    result = await client.search_decisions(BddkSearchRequest(keywords="sermaye yeterliliği tebliği"))
    assert [d.document_id for d in result.decisions] == ["exact_title", "category_only"]


@pytest.mark.asyncio
async def test_type_word_alone_ranks_documents_whose_category_also_names_that_type_first():
    client = _make_client_with_cache(
        [
            BddkDecisionSummary(
                title="Bilgi Sistemleri Yönetmeliğinin Uygulanmasına İlişkin Genelge",
                document_id="genelge",
                category="Genelge",
            ),
            BddkDecisionSummary(
                title="Kurul Kararı (Yönetmelik Değişikliği)", document_id="karar", category="Kurul Kararı"
            ),
            BddkDecisionSummary(
                title="Bankaların Kuruluş Yönetmeliği", document_id="regulation", category="Yönetmelik"
            ),
        ]
    )
    result = await client.search_decisions(BddkSearchRequest(keywords="yönetmelik"))
    assert [d.document_id for d in result.decisions][0] == "regulation"
    assert {d.document_id for d in result.decisions} == {"genelge", "karar", "regulation"}


@pytest.mark.asyncio
async def test_derived_words_are_content_words_not_document_type_inflections():
    """kanuni, kararname and rehberlik are not inflections of kanun, karar, rehber: a catalog without
    those words returns nothing for them."""
    client = _make_client_with_cache(
        [
            BddkDecisionSummary(
                title="Bağımsız Denetim Takip Sistemi Hakkında Kurul Kararı",
                document_id="bades",
                category="Kurul Kararı",
            ),
            BddkDecisionSummary(title="Bankacılık Kanunu", document_id="kanun", category="Kanun"),
            BddkDecisionSummary(title="Kredi Kartları Rehberi", document_id="rehber", category="Rehber"),
        ]
    )
    for keywords in ("kanuni takip", "kanuni", "kanunen", "kararname", "kararlaştırılan", "rehberlik"):
        result = await client.search_decisions(BddkSearchRequest(keywords=keywords))
        assert result.decisions == [], keywords
    # The real inflections still work as type words.
    kanun = await client.search_decisions(BddkSearchRequest(keywords="bankacılık kanununun"))
    assert [d.document_id for d in kanun.decisions] == ["kanun"]
    karar = await client.search_decisions(BddkSearchRequest(keywords="takip kararları"))
    assert [d.document_id for d in karar.decisions] == ["bades"]


@pytest.mark.asyncio
async def test_punctuation_only_keywords_have_no_tokens():
    from bddk_mcp.ingest.client import catalog_keyword_tokens

    assert catalog_keyword_tokens("...") == []
    assert catalog_keyword_tokens("( )") == []
    assert catalog_keyword_tokens("8970 sayılı") == ["8970", "sayılı"]


@pytest.mark.asyncio
async def test_decision_number_with_function_word_finds_that_decision_first():
    ids = await _ids("8970 sayılı Kurul kararı")
    assert ids == ["779"]


@pytest.mark.asyncio
async def test_decision_number_ranks_its_decision_above_other_matches():
    ids = await _ids("8970 gecikme süresi")
    assert ids[0] == "779"


@pytest.mark.asyncio
async def test_draft_word_with_kg_alternation_ranks_the_draft_first():
    ids = await _ids("sermaye yeterliliği yönetmelik taslağı")
    assert ids[0] == "1290"


@pytest.mark.asyncio
async def test_content_tokens_match_by_a_shared_stem_with_exact_title_matches_first():
    ids = await _ids("sınıflandırma")
    assert {"mevzuat_22599", "775", "779"} == set(ids)
    exact = await _ids("sınıflandırılması")
    assert set(exact) == set(ids)


@pytest.mark.asyncio
async def test_unaccented_type_word_is_recognised():
    ids = await _ids("bilgi sistemleri yonetmeligi")
    assert ids[0] == "mevzuat_34360"


@pytest.mark.asyncio
async def test_content_word_that_matches_nothing_still_returns_nothing():
    assert await _ids("kripto yönetmeliği") == []
    assert await _ids("8971 sayılı kurul kararı") == []
    assert await _ids("kredilerin xyzabc") == []


@pytest.mark.asyncio
async def test_stem_matching_keeps_a_five_character_floor():
    # "geçici" and "gecikme" share fewer than five leading characters; "faiz" is too short to stem-match.
    assert await _ids("geçici") == []
    assert await _ids("gecici") == []
    assert await _ids("finansallar") == []


@pytest.mark.asyncio
async def test_type_word_alone_lists_documents_of_that_type():
    ids = await _ids("tebliğ")
    assert ids == ["mevzuat_20029"]


@pytest.mark.asyncio
async def test_empty_keywords_with_category_browse_newest_first():
    ids = await _ids("", category="Kurul Kararı")
    assert ids == ["1162", "779", "775"]


@pytest.mark.asyncio
async def test_empty_keywords_with_date_range_browse_is_paginated():
    client = _make_client_with_cache(list(NATURAL_CATALOG))
    result = await client.search_decisions(BddkSearchRequest(keywords="", date_from="01.01.2020", page_size=2))
    assert result.total_results == 3
    assert [d.document_id for d in result.decisions] == ["1162", "779"]
    page2 = await client.search_decisions(BddkSearchRequest(keywords="", date_from="01.01.2020", page=2, page_size=2))
    assert [d.document_id for d in page2.decisions] == ["775"]


@pytest.mark.asyncio
async def test_result_reports_the_catalog_size():
    client = _make_client_with_cache(list(NATURAL_CATALOG))
    result = await client.search_decisions(BddkSearchRequest(keywords="xyznomatch"))
    assert result.catalog_size == len(NATURAL_CATALOG)


@pytest.mark.asyncio
async def test_exact_title_match_ranks_above_a_stem_match():
    client = _make_client_with_cache(
        [
            BddkDecisionSummary(title="Kredilerin Sınıflandırılması Hakkında Yönetmelik", document_id="stem"),
            BddkDecisionSummary(title="Kredi Sınıflandırma Rehberi", document_id="exact"),
        ]
    )
    result = await client.search_decisions(BddkSearchRequest(keywords="sınıflandırma"))
    assert [d.document_id for d in result.decisions] == ["exact", "stem"]
