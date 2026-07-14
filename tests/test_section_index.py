"""Tests for legal/regulatory section extraction."""

from __future__ import annotations

from bddk_mcp.store.section_index import extract_document_sections


def test_extracts_madde_sections_with_offsets_and_content_hash():
    text = "Başlangıç\n\nMadde 9 - Karşılık ayrılması\n(1) Birinci fıkra.\n\nMADDE 10\n(1) İkinci madde."

    sections = extract_document_sections("mevzuat_22599", text)

    madde9 = next(s for s in sections if s.section_type == "madde" and s.section_ref == "9")
    assert madde9.doc_id == "mevzuat_22599"
    assert madde9.heading == "Karşılık ayrılması"
    assert "(1) Birinci fıkra." in madde9.content
    assert "MADDE 10" not in madde9.content
    assert madde9.start_char < madde9.end_char
    assert len(madde9.content_hash) == 64


def test_extracts_ilke_and_paragraf_sections():
    text = "İlke 5: Model validasyonu\nBankalar modeli doğrular.\n\nParagraf 76\nÖnemli artış kriterleri."

    sections = extract_document_sections("943", text)

    assert ("ilke", "5") in {(s.section_type, s.section_ref) for s in sections}
    assert ("paragraf", "76") in {(s.section_type, s.section_ref) for s in sections}
    ilke5 = next(s for s in sections if s.section_type == "ilke")
    assert "Model validasyonu" in ilke5.heading
    assert "Bankalar modeli doğrular." in ilke5.content


def test_extracts_turkish_letter_subsections():
    text = "MADDE 1\n(1) Birinci fıkra.\n(a) A bendi.\n(ç) Ç bendi.\n\nMadde 2\nSonraki madde."

    sections = extract_document_sections("doc", text)

    refs = {(s.section_type, s.section_ref) for s in sections}
    assert ("fikra", "1") in refs
    assert ("bent", "a") in refs
    assert ("bent", "ç") in refs


def test_extracts_gecici_madde_and_ek_headings():
    text = "Geçici Madde 3 - Uyum süreci\nHüküm.\n\nEk-1\nHesaplama Tablosu\n\nEK 2\nAçıklama"

    sections = extract_document_sections("doc", text)

    refs = {(s.section_type, s.section_ref) for s in sections}
    assert ("gecici_madde", "3") in refs
    assert ("ek", "1") in refs
    assert ("ek", "2") in refs


def test_no_sections_for_plain_text():
    assert extract_document_sections("doc", "Sadece normal paragraf.") == []


def test_benchmark_critical_refs_are_identified():
    tfrs9_text = "İlke 5 - Model Validasyonu\nModel validasyonu bağımsız yapılır."
    karsilik_text = "MADDE 9 - TFRS 9 kapsamında karşılık ayrılması\nBankalar karşılık ayırır."

    tfrs9_sections = extract_document_sections("943", tfrs9_text)
    karsilik_sections = extract_document_sections("mevzuat_22599", karsilik_text)

    assert any(s.section_type == "ilke" and s.section_ref == "5" for s in tfrs9_sections)
    assert any(s.section_type == "madde" and s.section_ref == "9" for s in karsilik_sections)


def test_cp1252_endash_headings_parse():
    """mevzuat_10736 regression: cp1252 \x96 en-dash in headings must not defeat parsing."""
    text = "MADDE 1 \x96 (1) Bu Yönetmeliğin amacı.\n\nMADDE 2 \x96 (1) Kapsam hükmü.\n"

    sections = extract_document_sections("mevzuat_10736", text)

    refs = {(s.section_type, s.section_ref) for s in sections}
    assert ("madde", "1") in refs
    assert ("madde", "2") in refs


def test_bold_markdown_headings_parse():
    text = "**Madde 5 - Başlık**\nHüküm içeriği.\n"

    sections = extract_document_sections("doc", text)

    assert ("madde", "5") in {(s.section_type, s.section_ref) for s in sections}


def test_jumbo_section_span_is_capped():
    from bddk_mcp.store.section_index import MAX_SECTION_CHARS

    filler = "EK TABLO satırı içerik " * 2000  # ~46k chars of trailing annex junk
    text = "Madde 39 - Yürürlük\nHüküm.\n" + filler

    sections = extract_document_sections("doc", text)

    madde39 = next(s for s in sections if s.section_ref == "39")
    assert madde39.end_char - madde39.start_char <= MAX_SECTION_CHARS
    # content may exceed the cap only by the appended truncation marker
    assert len(madde39.content) <= MAX_SECTION_CHARS + 200


def test_zero_sections_warning_logged(caplog):
    import logging

    sentinel = "PRIVATE_AUDIT_SUBJECT_123"
    with caplog.at_level(logging.WARNING, logger="section_index"):
        result = extract_document_sections("doc", (sentinel + " uzun ama yapısız metin. ") * 100)

    assert result == []
    assert any("no section headings matched" in r.message for r in caplog.records)
    assert sentinel not in caplog.text


def test_bold_closed_heading_with_endash_parses():
    """mevzuat_15481 corpus shape: closing ** after the ref, before the dash."""
    text = "**MADDE 1** – (1) Bu Yönetmeliğin amacı.\n\n**MADDE 2** \x96 (1) Kapsam.\n"

    sections = extract_document_sections("mevzuat_15481", text)

    refs = {(s.section_type, s.section_ref) for s in sections}
    assert ("madde", "1") in refs
    assert ("madde", "2") in refs


def test_single_asterisk_bullet_is_not_a_heading():
    """Amendment lists ('* Madde 5 – ... değiştirilmiştir.') must not index."""
    text = "MADDE 1 - Gerçek hüküm.\nİçerik.\n* Madde 5 – ibare değiştirilmiştir.\n* Madde 7 - yürürlükten kaldırılmıştır.\n"

    sections = extract_document_sections("doc", text)

    level1 = {(s.section_type, s.section_ref) for s in sections if s.section_type == "madde"}
    assert level1 == {("madde", "1")}


def test_bold_heading_title_has_no_trailing_asterisks():
    sections = extract_document_sections("doc", "**Madde 5 - Başlık**\nHüküm.\n")

    madde5 = next(s for s in sections if s.section_ref == "5")
    assert madde5.heading == "Başlık"


def test_capped_section_carries_visible_truncation_marker():
    filler = "EK TABLO satırı içerik " * 2000
    text = "Madde 39 - Yürürlük\nHüküm.\n" + filler

    sections = extract_document_sections("doc", text)

    madde39 = next(s for s in sections if s.section_ref == "39")
    assert "[BÖLÜM KESİLDİ" in madde39.content
    assert "get_bddk_document" in madde39.content


def test_no_subsection_rows_beyond_capped_parent():
    annex = ("satır içerik dolgu metni " * 1200) + "\n(2) Annex içindeki sahte fıkra.\n"
    text = "Madde 39 - Yürürlük\n(1) Gerçek fıkra.\n" + annex

    sections = extract_document_sections("doc", text)

    fikra_starts = [s.start_char for s in sections if s.section_type == "fikra"]
    madde39 = next(s for s in sections if s.section_ref == "39")
    assert all(start < madde39.end_char for start in fikra_starts)
