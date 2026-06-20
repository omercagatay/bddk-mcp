"""Tests for Turkish legal-reference parsing."""

from __future__ import annotations

from bddk_mcp.store.legal_ref import parse_legal_refs, turkish_casefold


def test_parse_document_ids_numeric_and_prefixed():
    refs = parse_legal_refs("943 ve mevzuat_22599 dokümanlarını karşılaştır")

    assert "943" in refs.document_ids
    assert "mevzuat_22599" in refs.document_ids


def test_parse_madde_references_variants():
    refs = parse_legal_refs("Madde 9, 76. madde ve m. 12 hükümlerini bul")

    assert ("madde", "9") in refs.sections
    assert ("madde", "76") in refs.sections
    assert ("madde", "12") in refs.sections


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
