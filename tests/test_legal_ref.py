"""Tests for Turkish legal-reference parsing."""

from __future__ import annotations

from bddk_mcp.store.legal_ref import (
    canonical_section_ref,
    document_id_candidates,
    parse_legal_refs,
    turkish_casefold,
)


def test_parse_document_ids_numeric_and_prefixed():
    refs = parse_legal_refs("943 ve mevzuat_22599 dokümanlarını karşılaştır")

    assert "943" in refs.document_ids
    assert "mevzuat_22599" in refs.document_ids


def test_document_id_candidates_resolve_numeric_and_prefixed_aliases():
    assert document_id_candidates("22599") == ["22599", "mevzuat_22599", "bddk_22599"]
    assert document_id_candidates("MEVZUAT_22599") == ["mevzuat_22599"]


def test_parse_madde_references_variants():
    refs = parse_legal_refs("Madde 9, 76. madde ve m. 12 hükümlerini bul")

    assert ("madde", "9") in refs.sections
    assert ("madde", "76") in refs.sections
    assert ("madde", "12") in refs.sections
    assert refs.document_ids == []


def test_paragraph_reference_numbers_do_not_become_document_filters():
    refs = parse_legal_refs("943 paragraf 43; paragraf 76")
    assert refs.document_ids == ["943"]
    assert refs.sections == [("paragraf", "43"), ("paragraf", "76")]
    assert parse_legal_refs("paragraf 134").document_ids == []
    assert parse_legal_refs("1135 paragraf 2.1").sections == [("paragraf", "2.1")]


def test_parse_ilke_references_with_turkish_case():
    refs = parse_legal_refs("İlke 5 ile ILKE 7 model validasyonu")

    assert ("ilke", "5") in refs.sections
    assert ("ilke", "7") in refs.sections


def test_parse_decision_numbers_dates_and_categories():
    refs = parse_legal_refs("08.10.2015 tarihli 6478 sayılı Kurul Kararı ve Yönetmelik")

    assert "6478" in refs.decision_numbers
    assert "6478" not in refs.document_ids
    assert "08.10.2015" in refs.dates
    assert "Kurul Kararı" in refs.categories
    assert "Yönetmelik" in refs.categories


def test_date_fragments_are_not_document_ids():
    refs = parse_legal_refs("2015/3 ve 08.10.2015 tarihli kararlar")

    assert "2015/3" in refs.dates
    assert "08.10.2015" in refs.dates
    assert "2015" not in refs.document_ids


def test_parse_category_hints():
    refs = parse_legal_refs("Rehber, Tebliğ, Genelge ve Kurul Kararı ara")

    assert {"Rehber", "Tebliğ", "Genelge", "Kurul Kararı"} <= set(refs.categories)


def test_turkish_casefold_handles_dotted_and_dotless_i():
    assert turkish_casefold("İLKE I ı i") == "ilke ı ı i"
    refs = parse_legal_refs("ilke 5")
    assert ("ilke", "5") in refs.sections


def test_inserted_articles_parse_to_the_stored_ref_and_bare_articles_stay_bare():
    for query in ("mevzuat_5411 madde 66/A", "mevzuat_5411 madde 66A", "mevzuat_5411 Madde 66/a'ya göre"):
        refs = parse_legal_refs(query)
        assert refs.document_ids == ["mevzuat_5411"]
        assert refs.sections == [("madde", "66/a")]
    assert parse_legal_refs("m. 13/Ç").sections == [("madde", "13/ç")]
    assert parse_legal_refs("66/A. madde") == parse_legal_refs("madde 66/A")
    assert parse_legal_refs("mevzuat_5411 madde 66").sections == [("madde", "66")]
    assert parse_legal_refs("madde 4/1-a").sections == [("madde", "4")]
    assert parse_legal_refs("madde 5 a) bendi").sections == [("madde", "5")]


def test_gecici_and_ek_madde_map_to_their_own_section_types():
    assert parse_legal_refs("mevzuat_10749 GEÇİCİ MADDE 1").sections == [("gecici_madde", "1")]
    assert parse_legal_refs("gecici madde 2").sections == [("gecici_madde", "2")]
    assert parse_legal_refs("5411 sayılı Kanun ek madde 1").sections == [("ek_madde", "1")]
    assert parse_legal_refs("EK-1 madde 5").sections == [("madde", "5")]


def test_canonical_section_ref_is_the_stored_form():
    cases = {"66/A": "66/a", "66A": "66/a", " 13/Ç ": "13/ç", "a": "a", "Ç": "ç", "I": "i", "İ": "i", "2.1": "2.1"}
    assert {value: canonical_section_ref(value) for value in cases} == cases


def test_standard_and_framework_numbers_are_not_document_ids():
    for query in (
        "TFRS 16 kiralama işlemleri",
        "TMS 39 finansal araçlar",
        "TFRS-15 hasılat",
        "IFRS 17 sigorta sözleşmeleri",
        "IAS 37 karşılıklar",
        "UFRS 10 ve UMS 28",
        "BDS 701 kilit denetim konuları",
        "BOBİ FRS 12 uygulaması",
        "Basel 3 sermaye tamponları",
        "Basel III 2017 reformları",
        "COVID-19 kredi erteleme",
    ):
        assert parse_legal_refs(query).document_ids == [], query


def test_years_followed_by_yil_are_not_document_ids():
    assert parse_legal_refs("2024 yılı kredi kartı asgari ödeme").document_ids == []
    assert parse_legal_refs("2019 yılında yürürlüğe giren düzenleme").document_ids == []
    assert parse_legal_refs("1999 YILI bütçesi").document_ids == []


def test_bare_ids_next_to_standards_and_years_are_still_inferred():
    assert parse_legal_refs("943 İlke 5 TFRS 9 model validasyonu").document_ids == ["943"]
    assert parse_legal_refs("943 TFRS 16 kiralama").document_ids == ["943"]
    assert parse_legal_refs("mevzuat_22599 2024 yılı karşılıklar").document_ids == ["mevzuat_22599"]
    assert parse_legal_refs("1028 kredi kartı").document_ids == ["1028"]


def test_ordinal_article_references_parse_as_madde_and_stay_out_of_document_ids():
    refs = parse_legal_refs("5411 sayılı Kanunun 10 uncu maddesi")
    assert refs.document_ids == ["5411"]
    assert refs.sections == [("madde", "10")]
    for query in ("9 uncu madde", "9. madde", "9 uncu maddesi", "9'uncu maddesinde", "9. MADDE"):
        assert parse_legal_refs(query).sections == [("madde", "9")], query
    assert parse_legal_refs("3 üncü madde").sections == [("madde", "3")]
    assert parse_legal_refs("geçici 2 nci madde").sections == [("gecici_madde", "2")]
    assert parse_legal_refs("ek 1 inci maddesine göre").sections == [("ek_madde", "1")]
