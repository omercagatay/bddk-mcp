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
    # Tokens are folded (normalize_search_text): "sayılı" is matched as "sayili".
    assert catalog_keyword_tokens("8970 sayılı") == ["8970", "sayili"]


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


# -- Folding, whole-word matching and the stem rule on real catalog titles ----------------------------------

REAL_CATALOG = [
    BddkDecisionSummary(title=title, document_id=doc_id, decision_date=date, decision_number=number, category=cat)
    for doc_id, title, date, number, cat in (
        (
            "mevzuat_22599",
            "Kredilerin Sınıflandırılması ve Bunlar İçin Ayrılacak Karşılıklara İlişkin Usul ve Esaslar Hakkında "
            "Yönetmelik",
            "",
            "",
            "Yönetmelik",
        ),
        (
            "mevzuat_19864",
            "Bankaların İç Sistemleri ve İçsel Sermaye Yeterliliği Değerlendirme Süreci Hakkında Yönetmelik",
            "",
            "",
            "Yönetmelik",
        ),
        (
            "mevzuat_19498",
            "Bankaların Likidite Karşılama Oranı Hesaplamasına İlişkin Yönetmelik",
            "",
            "",
            "Yönetmelik",
        ),
        ("mevzuat_20029", "Kredi Riski Azaltım Tekniklerine İlişkin Tebliğ", "", "", "Sermaye Yeterliliği"),
        (
            "1312",
            "Kredi Riski Azaltım Tekniklerine İlişkin Tebliğ Taslağı ve Genel Gerekçe",
            "",
            "",
            "Düzenleme Taslağı",
        ),
        ("795", "Kredi Vade Sınırları Hk Kurul Kararı", "04.09.2020", "9131", "Kurul Kararı"),
        (
            "780",
            "Taşıt ve Tüketici Kredilerinde Yapılacak Ötelemelerin Kredi Vade Sınırlarında Dikkate Alınmamasına "
            "İlişkin Kurul Kararı",
            "27.03.2020",
            "8971",
            "Kurul Kararı",
        ),
        (
            "1259",
            "Cep Telefonu Alımlarında Kredi Vade Sınırlarında Değişiklik Yapılması Hk. Kurul Kararı",
            "05.12.2024",
            "11022",
            "Kurul Kararı",
        ),
        ("mevzuat_5411", "5411 sayılı Bankacılık Kanunu", "", "", "Kanun"),
        (
            "105",
            "Societe Generale S.A. İstanbul Merkez Şubesinin 5411 sayılı Kanunun 4 üncü maddesinin birinci "
            "fıkrasının (u) bendi çerçevesinde faaliyet genişlemesi talebine izin verilmesine ilişkin Kurul Kararı",
            "08.11.2007",
            "2374",
            "Kurul Kararı",
        ),
        (
            "1087",
            "JPMorgan Chase Bank N.A. Merkezi Columbus Ohio İstanbul Türkiye Şubesi'ne 5411 sayılı Bankacılık "
            "Kanununun 4 üncü maddesinin birinci fıkrasının (v) bendi ile Geçici 3 üncü maddesi uyarınca faaliyet "
            "genişlemesi izni verilmesine ilişkin Kurul Kararı",
            "21.12.2021",
            "9989",
            "Kurul Kararı",
        ),
        ("mevzuat_5464", "5464 sayılı Banka Kartları ve Kredi Kartları Kanunu", "", "", "Kanun"),
        (
            "450",
            "5464 sayılı Banka Kartları ve Kredi Kartları Kanununun 8 inci maddesinin birinci fıkrası uyarınca genel "
            "müdürlük veya şube haricinde kredi kartı talebi toplanabilecek yerlerin belirlenmesi",
            "20.04.2006",
            "1855",
            "Kurul Kararı",
        ),
        (
            "1254",
            "Kredi Kartı Borçlarının ve İhtiyaç Kredilerinin Yeniden Yapılandırılması Hk. Kurul Kararı",
            "26.09.2024",
            "10972",
            "Kurul Kararı",
        ),
        (
            "1325",
            "Kredi Kartı Borçlarının ve İhtiyaç Kredilerinin Yeniden Yapılandırılması Hk. Kurul Kararı",
            "29.01.2026",
            "11366",
            "Kurul Kararı",
        ),
        (
            "mevzuat_42628",
            "Bankacılık Hesaplarından Kaynaklanan Faiz Oranı Riskinin Standart Yaklaşımla Ölçülmesine ve "
            "Değerlendirilmesine İlişkin Yönetmelik",
            "",
            "",
            "Yönetmelik",
        ),
        ("1323", "Faizsiz Bankacılık İlke ve Standartlarına Uyuma İlişkin Tebliğ", "", "", "Mülga Düzenleme"),
        (
            "703",
            "Döngüsel sermaye tamponu uygulaması ile bankalarca yapılacak kâr dağıtımına ilişkin usul ve esaslar "
            "hakkında Kurul Kararı",
            "18.12.2015",
            "6602",
            "Kurul Kararı",
        ),
        (
            "1028",
            "Şirketlerin Derecelendirme Zorunluluğuna İlişkin Kurul Kararı hk.",
            "27.05.2021",
            "9590",
            "Kurul Kararı",
        ),
        (
            "mevzuat_40519",
            "Risk Gruplarının Belirlenmesine ve Kredi Sınırlarına İlişkin Yönetmelik",
            "",
            "",
            "Yönetmelik",
        ),
        (
            "473",
            "Tasfiye halinde bulunan 5 adet şirketin faaliyet izinlerinin iptaline ilişkin Kurul Kararı",
            "28.02.2008",
            "2496",
            "Kurul Kararı",
        ),
        (
            "238",
            "Intesa Sanpaolo S.p.A. İtalya İstanbul Merkez Şubesine faaliyet izni verilmesine ilişkin Kurul Kararı",
            "29.05.2014",
            "5868",
            "Kurul Kararı",
        ),
        (
            "mevzuat_24788",
            "Finansal Sektöre Olan Borçların Yeniden Yapılandırılması Hakkında Yönetmelik",
            "",
            "",
            "Yönetmelik",
        ),
        (
            "mevzuat_21278",
            "Bankalarca Yapılacak Repo Ve Ters Repo İşlemlerine İlişkin Esaslar Hakkında Yönetmelik",
            "",
            "",
            "Yönetmelik",
        ),
        ("915", "Bilgi Sistemlerine İlişkin Sızma Testleri Hakkında Genelge", "", "", "Bilgi Sistemleri"),
        (
            "779",
            "Kredilerin İkinci Grupta Sınıflandırılması için Öngörülen Gecikme Süresinin Değiştirilmesi hk. Kurul "
            "Kararı",
            "27.03.2020",
            "8970",
            "Kurul Kararı",
        ),
        (
            "886",
            "Kredili Mevduat Hesabı (KMH) Limitinin Belirlenmesi Hk. Kurul Kararı",
            "13.05.2021",
            "9566",
            "Kurul Kararı",
        ),
        (
            "81",
            "TMSF'ye devredilen bankaların hisselerinin TMSF tarafından devralınması halinde işlerlik kazanan, kayıtlı "
            "bilanço dışı yükümlülükler de dahil olmak üzere Türkiye'de kurulmuş mevduat bankalarının ve bu bankaların "
            "bilançolarında hesapları konsolide edilen yurt dışı şubelerinin yükümlülüklerinin tamamının garanti "
            "kapsamında olması uygulamasının 5/7/2004 tarihinden geçerli olmak üzere kaldırılması",
            "03.07.2003",
            "1084",
            "Kurul Kararı",
        ),
        (
            "mevzuat_10731",
            "Yabancı Para Net Genel Pozisyon/Özkaynak Standart Oranının Bankalarca Konsolide ve Konsolide Olmayan "
            "Bazda Hesaplanması ve Uygulanması Hakkında Yönetmelik",
            "",
            "",
            "Yönetmelik",
        ),
        (
            "mevzuat_21193",
            "Bankalarca Risk Yönetimine İlişkin Kamuya Yapılacak Açıklamalar Hakkında Tebliğ(RG-23/10/2015-29511)",
            "",
            "",
            "Tebliğ",
        ),
    )
]


async def _real_ids(keywords: str, *, reverse: bool = False, **kwargs) -> list[str]:
    catalog = list(reversed(REAL_CATALOG)) if reverse else list(REAL_CATALOG)
    client = _make_client_with_cache(catalog)
    result = await client.search_decisions(BddkSearchRequest(keywords=keywords, page_size=50, **kwargs))
    return [d.document_id for d in result.decisions]


@pytest.mark.asyncio
async def test_ascii_upper_case_and_combining_dot_queries_find_what_the_turkish_spelling_finds():
    written = await _real_ids("kredilerin sınıflandırılması")
    assert "mevzuat_22599" in written
    for variant in ("kredilerin siniflandirilmasi", "KREDILERIN SINIFLANDIRILMASI", "KREDİLERİN SINIFLANDIRILMASI"):
        assert await _real_ids(variant) == written, variant
    ic = await _real_ids("bankaların iç sistemleri")
    assert ic[0] == "mevzuat_19864"
    for variant in ("BANKALARIN IÇ SISTEMLERI", "bankalarin ic sistemleri", "bankaların i̇ç sistemleri"):
        assert await _real_ids(variant) == ic, variant
    assert (await _real_ids("likidite karsilama orani"))[0] == "mevzuat_19498"


@pytest.mark.asyncio
async def test_unaccented_function_and_type_words_are_recognised_after_folding():
    # "SAYILI" and "HAKKINDA" are function words, "YONETMELIGI" a type word, whatever the spelling.
    assert (await _real_ids("5411 SAYILI"))[0] == "mevzuat_5411"
    assert (await _real_ids("repo HAKKINDA YONETMELIGI"))[0] == "mevzuat_21278"
    # sayısı and tarihi are function words too.
    assert await _real_ids("8970 sayısı") == ["779"]
    assert await _real_ids("27.03.2020 tarihi") == await _real_ids("27.03.2020")


@pytest.mark.asyncio
async def test_slash_date_matches_like_a_dotted_date():
    dotted = await _real_ids("27.03.2020")
    assert set(dotted) == {"779", "780"}
    assert await _real_ids("27/03/2020") == dotted


@pytest.mark.asyncio
async def test_a_content_word_matches_whole_title_words_not_substrings():
    faiz = await _real_ids("faiz")
    assert "mevzuat_42628" in faiz
    assert "1323" not in faiz  # Faizsiz
    kar = await _real_ids("kar")
    assert kar == ["703"]  # kâr, not every "Kararı"
    assert "1028" not in kar
    # A number matches a whole number only.
    assert await _real_ids("897") == []
    assert await _real_ids("8970") == ["779"]
    # The same for numbers in titles: "541" is not the "5411" of "5411 sayılı Bankacılık Kanunu" or of the
    # decisions citing it.
    assert {"mevzuat_5411", "105", "1087"} <= set(await _real_ids("5411"))
    assert await _real_ids("541") == []
    # "Morgan" is a substring of "JPMorgan", not a word of the title.
    assert await _real_ids("morgan") == []


@pytest.mark.asyncio
async def test_interior_apostrophe_is_cut_before_matching():
    assert "mevzuat_5411" in (await _real_ids("Bankacılık Kanunu'nun"))[:5]
    assert (await _real_ids("Bankacılık Kanunu'nun"))[0] == "mevzuat_5411"
    # The title side is cut as well: "Şubesi'ne" is the word "şubesi".
    assert await _real_ids("JPMorgan'ın şubesi") == ["1087"]
    # Cutting keeps the stem only; it never turns a substring into a word.
    assert await _real_ids("Morgan'ın") == []


@pytest.mark.asyncio
async def test_short_roots_match_their_possessive_and_plural_forms():
    assert "915" in await _real_ids("sızma testi")
    assert "mevzuat_5464" in (await _real_ids("kredi kartı kanunu"))[:5]
    assert (await _real_ids("kredi kartı kanunu"))[0] == "mevzuat_5464"
    # A 4-character prefix needs an inflectional remainder on both sides: "-siz" is a derivation.
    assert "1323" not in await _real_ids("faiz")


@pytest.mark.asyncio
async def test_consonant_softening_is_an_inflection_of_a_short_root():
    assert "mevzuat_40519" in (await _real_ids("risk grubu"))[:5]
    hesabi = await _real_ids("mevduat hesabı")
    assert {"886", "81"} <= set(hesabi)  # Hesabı, hesapları
    # Softening only joins inflections: "Hesaplanması" is a derived word, not a form of "hesabı".
    assert "mevzuat_10731" not in await _real_ids("hesabı")


@pytest.mark.asyncio
async def test_elided_roots_match_their_full_form():
    izni = await _real_ids("faaliyet izni")
    assert {"473", "238", "1087"} <= set(izni)  # izinlerinin, izni, izni
    assert {"473", "238", "105"} <= set(await _real_ids("faaliyet izin"))
    # The elision map restores a root only before an inflectional ending: "devredilen" is not "devir".
    assert await _real_ids("devir") == []


@pytest.mark.asyncio
async def test_a_long_token_does_not_match_a_shorter_unrelated_word_through_a_short_prefix():
    ids = await _real_ids("yapılandırma")
    assert "mevzuat_24788" in ids  # Yapılandırılması
    assert "mevzuat_21278" not in ids  # Yapılacak
    assert "1259" not in ids  # Yapılması
    assert "703" not in ids  # yapılacak


ORAN_CATALOG = [
    BddkDecisionSummary(
        title="Türkiye Vakıflar Bankası T.A.O.'nun 31.12.2001 itibariyle % 5 olan konsolide sermaye yeterliliği "
        "standart oranını % 9'a ulaştıracak sermaye benzeri kredinin Bankaya tahsis edilmesi",
        document_id="73",
        decision_date="08.08.2002",
        decision_number="774",
        category="Kurul Kararı",
    ),
    BddkDecisionSummary(
        title="Derecelendirme Kuruluşlarının Yetkilendirilmesine ve Faaliyetlerine İlişkin Esaslar Hakkında Yönetmelik",
        document_id="mevzuat_16071",
        category="Yönetmelik",
    ),
]


@pytest.mark.asyncio
async def test_a_short_root_matches_its_accusative_after_a_possessive():
    client = _make_client_with_cache(list(ORAN_CATALOG))
    for keywords in ("sermaye yeterliliği standart oran", "oran"):
        result = await client.search_decisions(BddkSearchRequest(keywords=keywords))
        assert [d.document_id for d in result.decisions] == ["73"], keywords
    # The accusative follows a possessive only: "Kuruluşlarının" is a derived word, not a form of "kurul".
    result = await client.search_decisions(BddkSearchRequest(keywords="kurul"))
    assert "mevzuat_16071" not in [d.document_id for d in result.decisions]


DAIRE_CATALOG = [
    BddkDecisionSummary(
        title="Ana hizmet birimi kapsamında Veri Yönetimi ve Analiz Teknikleri Dairesi Başkanlığının kurulması",
        document_id="77",
        decision_date="05.03.2003",
        decision_number="1004",
        category="Kurul Kararı",
    ),
    BddkDecisionSummary(
        title="Bankacılık Hizmetlerinin Erişilebilirliğine Dair Yönetmelik",
        document_id="mevzuat_38873",
        category="Yönetmelik",
    ),
]


@pytest.mark.asyncio
async def test_a_content_word_does_not_reach_a_function_word_of_the_title_through_the_stem_rule():
    client = _make_client_with_cache(list(DAIRE_CATALOG))

    async def ids(keywords: str) -> list[str]:
        result = await client.search_decisions(BddkSearchRequest(keywords=keywords))
        return [d.document_id for d in result.decisions]

    # "daire" (department) is not an inflection of the function word "Dair".
    assert await ids("daire") == ["77"]
    # A query of function words only still matches them literally.
    assert await ids("dair") == ["mevzuat_38873"]


@pytest.mark.asyncio
async def test_two_vowel_endings_of_a_short_root_are_one_word():
    client = _make_client_with_cache(
        [
            BddkDecisionSummary(
                title="Operasyonel Riske Esas Tutarın İleri Ölçüm Yaklaşımı İle Hesaplanmasına İlişkin Tebliğ",
                document_id="mevzuat_20031",
                category="Sermaye Yeterliliği",
            ),
            BddkDecisionSummary(
                title="Bankaların İzne Tabi İşlemleri ile Dolaylı Pay Sahipliğine İlişkin Yönetmelik",
                document_id="mevzuat_10746",
                category="Yönetmelik",
            ),
        ]
    )
    # Dative "Riske"/"İzne" and accusative or possessive "riski"/"izni" are endings of one root.
    for keywords, expected in (("operasyonel riski", "mevzuat_20031"), ("izni tabi işlemler", "mevzuat_10746")):
        result = await client.search_decisions(BddkSearchRequest(keywords=keywords))
        assert [d.document_id for d in result.decisions] == [expected], keywords


def test_stem_rule_examples_from_real_titles():
    from bddk_mcp.ingest.client import _shares_stem

    positive = [
        ("grubu", "gruplarinin"),
        ("karti", "kartlari"),
        ("testi", "testleri"),
        ("izni", "izinlerinin"),
        ("izin", "izni"),
        ("hesabi", "hesaplari"),
        ("kanunu", "kanunu"),
        ("siniflandirma", "siniflandirilmasi"),
        ("yapilandirma", "yapilandirilmasi"),
        ("kredilerin", "kredi"),
        ("banka", "bankacilik"),
        ("oran", "oranini"),
        ("hisse", "hisselerini"),
        ("fonlara", "fonlarini"),
        ("riski", "riske"),
        ("izni", "izne"),
        ("kredilere", "kredileri"),
    ]
    negative = [
        ("faiz", "faizsiz"),
        ("yapilandirma", "yapilmasi"),
        ("yapilandirma", "yapilacak"),
        ("yapilandirma", "yapilan"),
        ("hesabi", "hesaplanmasi"),
        ("devir", "devredilen"),
        ("devir", "devralinmasi"),
        ("gecici", "gecikme"),
        ("kurul", "kuruluslarinin"),
    ]
    for token, word in positive:
        assert _shares_stem(token, word), (token, word)
    for token, word in negative:
        assert not _shares_stem(token, word), (token, word)


# One ending class per row: a query root and a real catalog title word carrying that ending (positive), and the
# same class of ending after a derivation or on another word (negative). Title words are folded, as the search
# compares them.
INFLECTION_CLASS_EXAMPLES = [
    # class, positive (query, title word), negative (query, title word)
    ("plural", ("esas", "esaslar"), ("faal", "faaliyetleri")),  # Usul ve Esaslar / gayri faal, Faaliyetleri (-iyet)
    ("possessive", ("fon", "fonu"), ("kar", "karsi")),  # Sigorta Fonu / Karşı: -sı follows vowels only
    ("possessive short root", ("dis", "disi"), ("kur", "kurumu")),  # Banka Dışı / Kurumu (-um)
    ("accusative", ("fon", "fonlarini"), ("bir", "birimi")),  # Fonlarını / Birimi (-im)
    ("dative", ("fon", "fonuna"), ("turk", "turkiye")),  # Sigorta Fonuna / Türkiye
    ("locative", ("alim", "alimlarinda"), ("kurul", "kurulmasinda")),  # Alımlarında / Kurulmasında (-ma)
    ("ablative", ("hesap", "hesaplarindan"), ("ortak", "ortakliklarindan")),  # Hesaplarından / Ortaklıklarından
    ("genitive", ("fon", "fonunun"), ("bir", "birliginin")),  # Fonunun / Birliğinin (-lik)
    ("instrumental", ("yerlesik", "yerlesiklerle"), ("net", "nedeniyle")),  # Yerleşiklerle / Nedeniyle
    ("equative", ("fon", "fonunca"), ("kurul", "kuruluslarinca")),  # Sigorta Fonunca / Kuruluşlarınca
    ("equative after plural", ("banka", "bankalarca"), ("kurulu", "kuruluslarca")),  # Bankalarca / Kuruluşlarca
    ("locative + relative -ki", ("alim", "alimlarindaki"), ("bank", "bankalardaki")),  # Alımlarındaki / Bank+a
    ("possessive + locative", ("ili", "ilinde"), ("ili", "iliskin")),  # İli / İlinde; İlişkin
]


@pytest.mark.parametrize(("ending", "positive", "negative"), INFLECTION_CLASS_EXAMPLES)
def test_inflection_rule_accepts_each_nominal_ending_and_rejects_the_derivation(ending, positive, negative):
    from bddk_mcp.ingest.client import _stem_grade

    assert _stem_grade(*positive) == 2, (ending, positive)
    assert _stem_grade(*negative) == 0, (ending, negative)


@pytest.mark.parametrize(
    ("token", "word"),
    [
        ("kanunu", "kanunun"),  # Kanunu / Kanunun: tails "" and "n" at "kanunu"; root "kanun"
        ("kararin", "karariyla"),  # Kararın / Kararıyla: tails "n" and "yla" at "karari"; root "karar"
        ("bankalarda", "bankalardaki"),  # Bankalarda / Bankalardaki: root "bankalar"
    ],
)
def test_inflection_rule_backs_off_to_a_shorter_root_when_the_full_common_prefix_leaves_no_ending(token, word):
    from bddk_mcp.ingest.client import _stem_grade

    # At the full common prefix one tail is not an ending; only a shorter root makes both tails
    # inflectional. Without that back-off the pair falls to the general prefix rule (grade 1).
    assert _stem_grade(token, word) == 2
    assert _stem_grade(word, token) == 2


FON_CATALOG = [
    BddkDecisionSummary(title=title, document_id=doc_id, decision_date=date, decision_number=number, category=cat)
    for doc_id, title, date, number, cat in (
        (
            "445",
            "Sigortaya Tabi Tasarruf Mevduatı ile Tasarruf Mevduatı Sigorta Fonunca Tahsil Olunacak Primler Hakkında "
            "Esaslar",
            "31.10.2003",
            "1143",
            "Kurul Kararı",
        ),
        (
            "449",
            "Sigortaya Tabi Tasarruf Mevduatı ile Tasarruf Mevduatı Sigorta Fonunca Tahsil Olunacak Primlere İlişkin "
            "Esasların 1, 2 ve 3 üncü maddelerinin birinci fıkralarının uygulamasının değiştirilmesi",
            "23.02.2005",
            "1584",
            "Kurul Kararı",
        ),
        (
            "40",
            "Bankalarda Zamanaşımına Uğrayan Mevduat, Emanet ve Alacakların Tasarruf Mevduatı Sigorta Fonuna Gelir "
            "Kaydedilmesine İlişkin Esas ve Usuller",
            "17.10.2000",
            "78",
            "Kurul Kararı",
        ),
        (
            "mevzuat_10735",
            "Mevduat ve Katılım Fonunun Kabulüne, Çekilmesine ve Zamanaşımına Uğrayan Mevduat, Katılım Fonu, Emanet ve "
            "Alacaklara İlişkin Usul ve Esaslar Hakkında Yönetmelik",
            "",
            "",
            "Yönetmelik",
        ),
        (
            "782",
            "Banka ve Banka Dışı Finansal Kuruluşların Yapacakları Raporlamalar ve Bildirimler Hk Kurul Kararı",
            "02.04.2020",
            "8976",
            "Kurul Kararı",
        ),
        (
            "1125",
            "Yurt Dışı Yerleşiklerle Türev İşlem Gerçekleştirilmesi Halinde Uygulanacak Risk Ağırlıkları hk. Kurul "
            "Kararı",
            "23.06.2022",
            "10248",
            "Kurul Kararı",
        ),
        (
            "mevzuat_19864",
            "Bankaların İç Sistemleri ve İçsel Sermaye Yeterliliği Değerlendirme Süreci Hakkında Yönetmelik",
            "",
            "",
            "Yönetmelik",
        ),
        (
            "944",
            "Kurum Tarafından Yapılan Denetimlerde Dikkate Alınan Değerlendirme Ölçütleri Hakkında Rehber",
            "",
            "",
            "Rehber",
        ),
        (
            "mevzuat_11085",
            "Bankacılık Düzenleme ve Denetleme Kurumu İnsan Kaynakları Yönetmeliği",
            "",
            "",
            "BDDK Düzenlemesi",
        ),
        (
            "fixture_kur",
            "Döviz Kuru Riski Hakkında Rehber",
            "",
            "",
            "Rehber",
        ),
    )
]


async def _fon_ids(keywords: str, *, reverse: bool = False) -> list[str]:
    catalog = list(reversed(FON_CATALOG)) if reverse else list(FON_CATALOG)
    client = _make_client_with_cache(catalog)
    result = await client.search_decisions(BddkSearchRequest(keywords=keywords, page_size=50))
    return [d.document_id for d in result.decisions]


@pytest.mark.asyncio
async def test_sigorta_fonu_primleri_finds_the_titles_saying_sigorta_fonunca():
    # "Fonunca" is the equative after the possessive (fon + u + nca); "Primlere" the dative plural of "prim".
    assert set(await _fon_ids("sigorta fonu primleri")) == {"445", "449"}


@pytest.mark.asyncio
async def test_three_letter_roots_reach_their_inflected_title_forms():
    # fon: Fonunca, Fonuna, Fonunun, Fonu.
    assert set(await _fon_ids("fon")) == {"445", "449", "40", "mevzuat_10735"}
    # dış: Dışı.
    assert set(await _fon_ids("dış")) == {"782", "1125"}
    # iç has two letters: only the whole word "İç" matches, not "İçsel".
    assert await _fon_ids("iç") == ["mevzuat_19864"]


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_words_joined_by_a_slash_or_a_dot_are_separate_title_words(reverse):
    catalog = [
        BddkDecisionSummary(
            title="Bankalarda uygulama birliğinin sağlanmasını teminen; destek/danışmanlık hizmetinin bankalar "
            "tarafından gerçekleştirilebilecek faaliyet konuları arasında sayılması",
            document_id="93",
            decision_date="03.08.2006",
            decision_number="1947",
            category="Kurul Kararı",
        ),
        BddkDecisionSummary(
            title="T.Vakıflar Bankası T.A.O.'ya 5411 sayılı Bankacılık Kanununun 4 üncü maddesinin birinci fıkrasının "
            "(v) bendi ile geçici 3 üncü maddesi uyarınca faaliyet genişlemesi izni verilmesine ilişkin Kurul Kararı",
            document_id="285",
            decision_date="22.10.2015",
            decision_number="6505",
            category="Kurul Kararı",
        ),
    ]
    client = _make_client_with_cache(list(reversed(catalog)) if reverse else catalog)

    async def ids(keywords: str) -> list[str]:
        return [d.document_id for d in (await client.search_decisions(BddkSearchRequest(keywords=keywords))).decisions]

    assert await ids("destek hizmeti") == ["93"]
    assert await ids("danışmanlık") == ["93"]
    assert await ids("vakıflar bankası") == ["285"]
    assert (await _real_ids("net genel pozisyon özkaynak standart oranı"))[0] == "mevzuat_10731"


def test_abbreviations_dates_and_numbers_stay_one_token():
    from bddk_mcp.ingest.client import _catalog_tokens

    assert _catalog_tokens("T.A.O.'ya A.Ş. S.A.L 2000/682 27/03/2020 Pozisyon/Özkaynak") == [
        "t.a.o",
        "a.s",
        "s.a.l",
        "2000/682",
        "27.03.2020",
        "pozisyon",
        "ozkaynak",
    ]


@pytest.mark.asyncio
async def test_kur_does_not_reach_kurul_or_kurum_titles():
    # "Kuru" (kur + u) is an inflection; "Kurul", "Kurum", "Kurumu" and the category "Kurul Kararı" are other words.
    assert await _fon_ids("kur") == ["fixture_kur"]


# -- Numbers that identify an instrument and the ranking ------------------------------------------------------
# Every ranking test runs on the catalog in both orders: catalog order must never decide the ranking.


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_a_law_number_ranks_the_law_first_above_decisions_citing_it(reverse):
    ids = await _real_ids("5411", reverse=reverse)
    assert ids[0] == "mevzuat_5411"
    assert {"105", "1087"} <= set(ids)
    assert (await _real_ids("5464", reverse=reverse))[0] == "mevzuat_5464"


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_a_mevzuat_number_or_document_id_finds_that_document(reverse):
    assert await _real_ids("22599", reverse=reverse) == ["mevzuat_22599"]
    assert await _real_ids("mevzuat_22599", reverse=reverse) == ["mevzuat_22599"]
    # A decision number still ranks its decision first.
    assert (await _real_ids("8970 gecikme", reverse=reverse))[0] == "779"


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_the_title_equal_to_the_query_ranks_first(reverse):
    ids = await _real_ids("Kredi Riski Azaltım Tekniklerine İlişkin Tebliğ", reverse=reverse)
    assert ids[:2] == ["mevzuat_20029", "1312"]
    assert (await _real_ids("Kredi Vade Sınırları Hk Kurul Kararı", reverse=reverse))[0] == "795"


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_in_force_text_ranks_before_its_draft_unless_the_query_asks_for_the_draft(reverse):
    assert (await _real_ids("kredi riski azaltım teknikleri", reverse=reverse))[:2] == ["mevzuat_20029", "1312"]
    assert (await _real_ids("kredi riski azaltım teknikleri taslağı", reverse=reverse))[0] == "1312"


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_a_title_holding_every_query_word_ranks_before_one_found_partly_through_the_category(reverse):
    catalog = [
        # "devralınma" is a title word, "kurul" only its category: half of the query is in the title.
        BddkDecisionSummary(
            title="Banka Kuruluşunda veya Mevcut Bankaların Hisselerinin Devralınma-sında ya da Tasarruf Mevduatı "
            "Sigorta Fonu Bünyesinde Bulunan Bankaların Hisselerini Satın Almak Üzere Başvuracaklarda Aranacak "
            "Koşullara İlişkin Karar",
            document_id="43",
            decision_date="03.11.2000",
            decision_number="98",
            category="Kurul Kararı",
        ),
        # Both words are in the title, by shared stems only ("Devralınacak", "Kuruluş").
        BddkDecisionSummary(
            title="Varlık Yönetim Şirketlerinin Kuruluş ve Faaliyet Esasları ile Devralınacak Alacaklara İlişkin "
            "İşlemler Hakkında Yönetmelik",
            document_id="mevzuat_38797",
            category="Yönetmelik",
        ),
    ]
    client = _make_client_with_cache(list(reversed(catalog)) if reverse else catalog)
    result = await client.search_decisions(BddkSearchRequest(keywords="devralınma kurul"))
    assert [d.document_id for d in result.decisions] == ["mevzuat_38797", "43"]


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_a_number_leading_the_title_as_n_sayili_ranks_before_titles_citing_it(reverse):
    catalog = [
        BddkDecisionSummary(
            title="5464 sayılı Banka Kartları ve Kredi Kartları Kanununun 8 inci maddesinin birinci fıkrası uyarınca "
            "genel müdürlük veya şube haricinde kredi kartı talebi toplanabilecek yerlerin belirlenmesi",
            document_id="450",
            decision_date="20.04.2006",
            decision_number="1855",
            category="Kurul Kararı",
        ),
        # Fixture: a newer decision that cites the law inside its title; the newest-date tie-break alone would
        # put it first.
        BddkDecisionSummary(
            title="Banka Kartları ve Kredi Kartları Hakkında 5464 sayılı Kanun Kapsamında Asgari Ödeme Tutarı",
            document_id="fixture_cites_5464",
            decision_date="01.06.2024",
            decision_number="10900",
            category="Kurul Kararı",
        ),
    ]
    client = _make_client_with_cache(list(reversed(catalog)) if reverse else catalog)
    result = await client.search_decisions(BddkSearchRequest(keywords="5464"))
    assert [d.document_id for d in result.decisions] == ["450", "fixture_cites_5464"]


IN_FORCE_CATALOG = [
    BddkDecisionSummary(
        title="Bankacılık Hesaplarından Kaynaklanan Faiz Oranı Riskinin Yönetimine İlişkin Rehber",
        document_id="1291",
        category="Rehber",
    ),
    # Repealed, with a shorter title: the shorter-title tie-break alone would put it first.
    BddkDecisionSummary(
        title="Faiz Oranın Riskinin Yönetimine İlişkin Rehber",
        document_id="1311",
        category="Mülga Düzenleme",
    ),
    BddkDecisionSummary(
        title="Kredi Riski Azaltım Tekniklerine İlişkin Tebliğ",
        document_id="mevzuat_20029",
        category="Sermaye Yeterliliği",
    ),
    # Fixture: the draft is given a date, so the newest-date tie-break alone would put it first.
    BddkDecisionSummary(
        title="Kredi Riski Azaltım Tekniklerine İlişkin Tebliğ Taslağı ve Genel Gerekçe",
        document_id="1312",
        decision_date="01.06.2025",
        category="Düzenleme Taslağı",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_in_force_text_ranks_before_a_newer_draft_and_a_shorter_repealed_text(reverse):
    catalog = list(reversed(IN_FORCE_CATALOG)) if reverse else list(IN_FORCE_CATALOG)
    client = _make_client_with_cache(catalog)

    async def ids(keywords: str) -> list[str]:
        result = await client.search_decisions(BddkSearchRequest(keywords=keywords))
        return [d.document_id for d in result.decisions]

    assert await ids("faiz oranı riskinin yönetimi") == ["1291", "1311"]
    assert await ids("mülga faiz oranı riskinin yönetimi") == ["1311"]
    assert await ids("kredi riski azaltım teknikleri") == ["mevzuat_20029", "1312"]
    assert (await ids("kredi riski azaltım teknikleri taslağı"))[0] == "1312"


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_equal_matches_are_ordered_newest_decision_first_not_by_document_id(reverse):
    # 1254 and 1325 have the same title; the 2026 decision comes first.
    ids = await _real_ids("kredi kartı borçlarının yeniden yapılandırılması", reverse=reverse)
    assert ids[:2] == ["1325", "1254"]


@pytest.mark.asyncio
async def test_browse_mode_keeps_newest_first_order():
    client = _make_client_with_cache(list(REAL_CATALOG))
    result = await client.search_decisions(BddkSearchRequest(keywords="", category="Kurul Kararı", page_size=3))
    assert [d.document_id for d in result.decisions] == ["1325", "1259", "1254"]


# -- The instrument asked for: phrase hit and title-side coverage ----------------------------------------------

INSTRUMENT_CATALOG = [
    BddkDecisionSummary(title=title, document_id=doc_id, decision_date=date, decision_number=number, category=cat)
    for doc_id, title, date, number, cat in (
        (
            "mevzuat_22599",
            "Kredilerin Sınıflandırılması ve Bunlar İçin Ayrılacak Karşılıklara İlişkin Usul ve Esaslar Hakkında "
            "Yönetmelik",
            "",
            "",
            "Yönetmelik",
        ),
        (
            "775",
            "Kredilerin Donuk Alacak Olarak Sınıflandırılması için Öngörülen Gecikme Süresinin Değiştirilmesi ve "
            "Yeniden Yapılandırma hk. Kurul Kararı",
            "17.03.2020",
            "8948",
            "Kurul Kararı",
        ),
        (
            "777",
            "Banka Dışı Finansal Kuruluşlarda Kredilerin, Tasfiye Olunacak Alacaklar Hesaplarında Sınıflandırılması "
            "için Öngörülen Gecikme Süresinin Değiştirilmesi hk. Kurul Kararı",
            "19.03.2020",
            "8950",
            "Kurul Kararı",
        ),
        (
            "779",
            "Kredilerin İkinci Grupta Sınıflandırılması için Öngörülen Gecikme Süresinin Değiştirilmesi hk. Kurul "
            "Kararı",
            "27.03.2020",
            "8970",
            "Kurul Kararı",
        ),
        (
            "mevzuat_21192",
            "Bankaların Sermaye Yeterliliğinin Ölçülmesine ve Değerlendirilmesine İlişkin Yönetmelik",
            "",
            "",
            "Yönetmelik",
        ),
        (
            "1290",
            "Bankaların Sermaye Yeterliliğinin Ölçülmesine ve Değerlendirilmesine İlişkin Yönetmelik Taslağı ve Genel "
            "Gerekçe",
            "",
            "",
            "Düzenleme Taslağı",
        ),
        (
            "1249",
            "Sermaye Yeterliliği Hesaplamasında % 0 Risk Ağırlığı Uygulanacak Çok Taraflı Kalkınma Bankalarının "
            "Kapsamının Basel Standartları Çerçevesinde Güncellenmesi Hk. Kurul Kararı",
            "15.08.2024",
            "10953",
            "Kurul Kararı",
        ),
        (
            "1162",
            "On İli Etkileyen Deprem Felaketi Hk. Kurul Kararı (Sermaye Yeterliliği ve Özkaynak Hesaplamasına İlişkin)",
            "14.02.2023",
            "10508",
            "Kurul Kararı",
        ),
        (
            "759",
            "Bankaların Sermaye Yeterliliğinin Ölçülmesine ve Değerlendirilmesine İlişkin Yönetmelik kapsamında "
            "kullanılacak KOBİ tanımı ve perakende kredi limitine ilişkin Kurul Kararı",
            "18.01.2019",
            "8199",
            "Kurul Kararı",
        ),
        (
            "744",
            "Bankaların Sermaye Yeterliliğinin Ölçülmesine ve Değerlendirilmesine İlişkin Yönetmelik kapsamında "
            "kullanılacak KOBİ tanımı ve perakende kredi limitine ilişkin Kurul Kararı",
            "07.06.2018",
            "7841",
            "Kurul Kararı",
        ),
        (
            "73",
            "Türkiye Vakıflar Bankası T.A.O.'nun 31.12.2001 itibariyle % 5 olan konsolide sermaye yeterliliği standart "
            "oranını % 9'a ulaştıracak sermaye benzeri kredinin Bankaya tahsis edilmesi",
            "08.08.2002",
            "774",
            "Kurul Kararı",
        ),
        (
            "mevzuat_19864",
            "Bankaların İç Sistemleri ve İçsel Sermaye Yeterliliği Değerlendirme Süreci Hakkında Yönetmelik",
            "",
            "",
            "Yönetmelik",
        ),
    )
]


async def _ranked(catalog: list[BddkDecisionSummary], keywords: str, *, reverse: bool) -> list[str]:
    client = _make_client_with_cache(list(reversed(catalog)) if reverse else list(catalog))
    result = await client.search_decisions(BddkSearchRequest(keywords=keywords, page_size=50))
    return [d.document_id for d in result.decisions]


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_the_regulation_named_by_the_query_ranks_before_dated_decisions_mentioning_its_words(reverse):
    # Every candidate holds both words; the undated regulation lost the newest-date tie-break before.
    ids = await _ranked(INSTRUMENT_CATALOG, "kredilerin sınıflandırılması", reverse=reverse)
    assert ids[0] == "mevzuat_22599"
    assert set(ids) == {"mevzuat_22599", "775", "777", "779"}


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_sermaye_yeterliligi_lists_the_capital_adequacy_regulation_first_and_before_its_draft(reverse):
    ids = await _ranked(INSTRUMENT_CATALOG, "sermaye yeterliliği", reverse=reverse)
    assert ids[0] == "mevzuat_21192"
    assert ids.index("mevzuat_21192") < ids.index("1290")


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_adjacent_query_words_rank_by_phrase_position_when_title_coverage_ties(reverse):
    # Fixtures: every title has three content words, two of them the query's, so title-side coverage ties at 2/3;
    # the newer the decision, the weaker its phrase, so the date alone would reverse the order.
    catalog = [
        BddkDecisionSummary(title="Kredi Kartı Ücretleri", document_id="leading", decision_date="01.01.2020"),
        # A leading document-type word does not move the phrase from the start of the title.
        BddkDecisionSummary(
            title="Rehber: Kredi Kartı Ücretleri", document_id="leading_after_type", decision_date="01.01.2021"
        ),
        BddkDecisionSummary(title="Ücretler ve Kredi Kartı", document_id="later", decision_date="01.01.2022"),
        BddkDecisionSummary(title="Kredi Ücretleri ve Kartı", document_id="apart", decision_date="01.01.2023"),
    ]
    assert await _ranked(catalog, "kredi kartı", reverse=reverse) == [
        "leading_after_type",
        "leading",
        "later",
        "apart",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_title_side_coverage_decides_between_titles_with_the_same_phrase(reverse):
    # Fixtures: both titles start with the query phrase; the older one has fewer other words (2/3 against 2/4)
    # and ranks first although the newest-date tie-break would put the other first.
    catalog = [
        BddkDecisionSummary(title="Faiz Oranı Riski Hakkında Rehber", document_id="short", decision_date="01.01.2010"),
        BddkDecisionSummary(
            title="Faiz Oranı Riski Ölçümü Hakkında Rehber", document_id="long", decision_date="01.01.2024"
        ),
    ]
    assert await _ranked(catalog, "faiz oranı", reverse=reverse) == ["short", "long"]
    # Exact fractions: 1/3 and 2/6 tie, so the newest decision comes first.
    catalog = [
        BddkDecisionSummary(title="Kur Riski Bilgisi", document_id="third", decision_date="01.01.2010"),
        BddkDecisionSummary(
            title="Kur Riski Bilgisi Kur Riski Ölçümü", document_id="two_sixths", decision_date="01.01.2024"
        ),
    ]
    assert await _ranked(catalog, "kur", reverse=reverse) == ["two_sixths", "third"]


# -- Boilerplate in title-side coverage: "Kurul" before a type word, a leading "<n> sayılı" -------------------

BOILERPLATE_CATALOG = [
    BddkDecisionSummary(title=title, document_id=doc_id, decision_date=date, decision_number=number, category=cat)
    for doc_id, title, date, number, cat in (
        ("795", "Kredi Vade Sınırları Hk Kurul Kararı", "04.09.2020", "9131", "Kurul Kararı"),
        ("949", "Bankaların Kredi Yönetimine İlişkin Rehber", "", "", "Rehber"),
        ("mevzuat_6362", "6362 sayılı Sermaye Piyasası Kanunu", "", "", "Kanun"),
        ("956", "Piyasa Riski Yönetimine İlişkin Rehber", "", "", "Rehber"),
        (
            "758",
            "Banka Kartları ve Kredi Kartları Hakkında Yönetmeliğe İlişkin Kurul Kararı",
            "11.01.2019",
            "8198",
            "Kurul Kararı",
        ),
        ("mevzuat_5464", "5464 sayılı Banka Kartları ve Kredi Kartları Kanunu", "", "", "Kanun"),
    )
]


def test_title_side_coverage_leaves_out_kurul_before_a_type_word_and_a_leading_citation_number():
    from fractions import Fraction

    from bddk_mcp.ingest.client import _catalog_tokens, _phrase_and_title_coverage

    def coverage(query: str, title: str) -> Fraction:
        return _phrase_and_title_coverage(_catalog_tokens(query), _catalog_tokens(title))[1]

    # Kredi, Vade, Sınırları: "Kurul" before "Kararı" is not a content word of the title.
    assert coverage("kredi", "Kredi Vade Sınırları Hk Kurul Kararı") == Fraction(1, 3)
    # Sermaye, Piyasası: the number of the leading "6362 sayılı" is not a content word of the title.
    assert coverage("piyasa", "6362 sayılı Sermaye Piyasası Kanunu") == Fraction(1, 2)
    # Banka, Kartları, Kredi, Kartları on both sides.
    assert coverage("kredi kartları", "Banka Kartları ve Kredi Kartları Hakkında Yönetmeliğe İlişkin Kurul Kararı") == (
        Fraction(3, 4)
    )
    assert coverage("kredi kartları", "5464 sayılı Banka Kartları ve Kredi Kartları Kanunu") == Fraction(3, 4)
    # "Kurul" elsewhere in a title and a number that does not lead it stay content words.
    assert coverage("kurul", "Bankacılık Düzenleme ve Denetleme Kurulunun Çalışma Usul ve Esasları") == Fraction(1, 7)
    assert coverage("kanun", "Banka Kartları Hakkında 5464 sayılı Kanun Kapsamında Asgari Ödeme") == Fraction(0, 6)


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_boilerplate_does_not_count_against_a_title_in_title_side_coverage(reverse):
    # Without the boilerplate both pairs tie on title-side coverage or turn the other way; the phrase position or
    # the coverage then puts the title whose words are the query's subject first.
    assert (await _ranked(BOILERPLATE_CATALOG[:2], "kredi", reverse=reverse)) == ["795", "949"]
    assert (await _ranked(BOILERPLATE_CATALOG[2:4], "piyasa", reverse=reverse)) == ["mevzuat_6362", "956"]
    # The number and "kurul" still match as query words.
    assert (await _ranked(BOILERPLATE_CATALOG, "6362", reverse=reverse)) == ["mevzuat_6362"]
    assert set(await _ranked(BOILERPLATE_CATALOG, "kurul", reverse=reverse)) == {"795", "758"}


# -- Lexicalised words that are not inflections of a shorter root ----------------------------------------------

LEXICALISED_CATALOG = [
    BddkDecisionSummary(title=title, document_id=doc_id, decision_date=date, decision_number=number, category=cat)
    for doc_id, title, date, number, cat in (
        (
            "1240",
            "Operasyonel Riske Esas Tutarın Hesaplanmasına İlişkin Tebliğ Taslağı ve Genel Gerekçe",
            "",
            "",
            "Düzenleme Taslağı",
        ),
        ("1286", "Yapısal Pozisyon Tebliğ Taslağı ve Genel Gerekçe", "", "", "Düzenleme Taslağı"),
        (
            "83",
            "İnsan Kaynakları Dairesi ile İdari ve Mali İşler Dairesinin birleştirilerek, İnsan Kaynakları ve Mali İşler "
            "Dairesinin kurulması",
            "22.01.2004",
            "1187",
            "Kurul Kararı",
        ),
        (
            "mevzuat_10737",
            "Bankaların Üst Yönetimine Atanacakların Bildirimi, Yemin ve Mal Beyanında Bulunulması ve Karar "
            "Defterlerinin Tutulmasına İlişkin Usul ve Esaslar Hakkında Yönetmelik",
            "",
            "",
            "Yönetmelik",
        ),
        (
            "445",
            "Sigortaya Tabi Tasarruf Mevduatı ile Tasarruf Mevduatı Sigorta Fonunca Tahsil Olunacak Primler Hakkında "
            "Esaslar",
            "31.10.2003",
            "1143",
            "Kurul Kararı",
        ),
        (
            "776",
            "Değeri Beş Yüz Bin TL ve Altındaki Konutlar için Kullandırılacak Konut Teminatlı Kredilerde, Kredi "
            "Tutarının Konut Değerine Oranının Değiştirilmesi hk. Kurul Kararı",
            "19.03.2020",
            "8949",
            "Kurul Kararı",
        ),
    )
]


def test_lexicalised_words_are_never_an_inflection_of_a_shorter_root():
    from bddk_mcp.ingest.client import _stem_grade

    # gerekçe (justification) is not gerek + equative -ce; malî (financial) is not mal + -i. Both directions.
    for short, lexicalised in (("geregi", "gerekce"), ("mal", "mali")):
        assert _stem_grade(short, lexicalised) == 0
        assert _stem_grade(lexicalised, short) == 0
    # The words still match themselves and their own inflections.
    assert _stem_grade("gerekce", "gerekcesi") == 2
    assert _stem_grade("mali", "malisi") == 2
    # Ordinary -ce/-ca and -i inflections after a consonant are unaffected.
    assert _stem_grade("fonu", "fonunca") == 2
    assert _stem_grade("deger", "degeri") == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_gerekce_and_mali_titles_are_found_by_their_own_words_only(reverse):
    drafts = {"1240", "1286"}
    assert not drafts & set(await _ranked(LEXICALISED_CATALOG, "gereği", reverse=reverse))
    assert not drafts & set(await _ranked(LEXICALISED_CATALOG, "kanun gereği", reverse=reverse))
    assert set(await _ranked(LEXICALISED_CATALOG, "gerekçe", reverse=reverse)) == drafts
    assert await _ranked(LEXICALISED_CATALOG, "mal", reverse=reverse) == ["mevzuat_10737"]
    assert await _ranked(LEXICALISED_CATALOG, "mali", reverse=reverse) == ["83"]
    # Real-title positives of the same endings: Fonunca (fon + u + nca), Değeri (değer + i).
    assert await _ranked(LEXICALISED_CATALOG, "fonu", reverse=reverse) == ["445"]
    assert await _ranked(LEXICALISED_CATALOG, "değer", reverse=reverse) == ["776"]
