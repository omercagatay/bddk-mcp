"""Tests for legal/regulatory section extraction."""

from __future__ import annotations

import json
import re
from pathlib import Path

from bddk_mcp.store.section_index import _find_section_starts, extract_document_sections


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


def test_annex_headings_with_printed_dash_or_markdown_marker_end_the_last_article():
    text = (
        "MADDE 12 – (1) Bu Tebliğ hükümlerini Kurum Başkanı yürütür.\n\n"
        "EK–1\n\nRİSK AĞIRLIKLI TUTAR HESAPLAMASI\n\n## EK-2\n\nKarşı taraf kredi riski\n"
    )

    sections = {(s.section_type, s.section_ref): s for s in extract_document_sections("doc", text)}

    assert ("ek", "1") in sections
    assert ("ek", "2") in sections
    assert "RİSK AĞIRLIKLI" not in sections[("madde", "12")].content


def test_heading_with_marker_after_the_number():
    text = (
        "MADDE 7$^{(1)}$ – (Değişik:RG-14/10/2012-28441)\nHüküm.\n\nGeçici Madde 32$^{[71]}$ – (Ek:17/7/2019)\nHüküm."
    )

    refs = {(s.section_type, s.section_ref) for s in extract_document_sections("doc", text)}

    assert ("madde", "7") in refs
    assert ("gecici_madde", "32") in refs


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
    document_sentinel = "PRIVATE_DOCUMENT_ID_456"
    with caplog.at_level(logging.WARNING, logger="bddk_mcp.store.section_index"):
        result = extract_document_sections(document_sentinel, (sentinel + " uzun ama yapısız metin. ") * 100)

    assert result
    assert {s.section_type for s in result} == {"govde"}
    assert any("no section headings matched" in r.message for r in caplog.records)
    assert any("govde" in r.message for r in caplog.records)
    assert sentinel not in caplog.text
    assert document_sentinel not in caplog.text


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


def test_repeated_identical_article_yields_one_section():
    """Consolidated tebliğ repeat an identical closing article per amendment.

    A section's identity is (doc_id, section_type, section_ref, content_hash),
    so emitting a row per occurrence yields rows no lookup can tell apart.
    Keep the first.
    """
    article = "MADDE 2 – Bu Tebliğ 1/1/2021 tarihinde yürürlüğe girer.\n\n"
    text = article * 3

    # Precondition: the raw spans really do collide on the schema's key.
    raw_starts = [s for s in _find_section_starts(text) if s.get("section_ref") == "2"]
    assert len(raw_starts) == 3

    sections = extract_document_sections("802", text)

    madde2 = [s for s in sections if s.section_type == "madde" and s.section_ref == "2"]
    assert len(madde2) == 1
    assert madde2[0].start_char == min(s["start_char"] for s in raw_starts)

    keys = [(s.section_type, s.section_ref, s.content_hash) for s in sections]
    assert len(keys) == len(set(keys))


def test_same_ref_with_different_content_keeps_both_sections():
    """Only exact content duplicates collapse; genuinely distinct spans stay."""
    text = "MADDE 2 – Birinci yürürlük hükmü.\n\nEK-1 Ayrım\n\nMADDE 2 – Farklı bir ikinci hüküm.\n"

    sections = extract_document_sections("doc", text)

    madde2 = [s for s in sections if s.section_type == "madde" and s.section_ref == "2"]
    assert len(madde2) == 2
    assert madde2[0].content_hash != madde2[1].content_hash


def test_numbered_paragraphs_index_when_no_classic_heading_matches():
    """Rehber corpus shape (954, 948): plain sequential numbered paragraphs."""
    text = (
        "1.  Bu Rehber likidite riskine ilişkindir.\n"
        "2.  Yönetim kurulu sorumludur.\n"
        "3.  Üst düzey yönetim uygular.\n"
        "4.  Bilgi sistemleri desteği sağlanır.\n"
    )

    sections = extract_document_sections("954", text)

    refs = {(s.section_type, s.section_ref) for s in sections}
    assert refs == {("paragraf", "1"), ("paragraf", "2"), ("paragraf", "3"), ("paragraf", "4")}
    p2 = next(s for s in sections if s.section_ref == "2")
    assert p2.heading == "Yönetim kurulu sorumludur."
    assert "3.  Üst düzey" not in p2.content


def test_numbered_fallback_requires_minimum_sections():
    text = "1.  Birinci husus uzun açıklama.\n2.  İkinci husus uzun açıklama.\n"

    assert extract_document_sections("doc", text) == []


def test_numbered_fallback_tolerates_a_lost_paragraph_number():
    """Rehber 950 corpus shape: paragraph 3 was lost by extraction; 4..N must still index."""
    text = "1.  Amaç hükmü.\n2.  Kapsam hükmü.\n4.  Tanımlar hükmü.\n5.  Sorumluluk hükmü.\n"

    sections = extract_document_sections("950", text)

    refs = {s.section_ref for s in sections}
    assert refs == {"1", "2", "4", "5"}


def test_numbered_fallback_does_not_match_footnotes_or_cross_references():
    """Bare numbers without a trailing dot (footnotes, '5 inci maddesinin') never index."""
    text = (
        "1.  Amaç hükmü.\n"
        "2.  Kapsam hükmü.\n"
        "3 Orta vadeli fonlama rasyosu gibi.\n"
        "3.  Tanımlar hükmü.\n"
        "5  inci  maddesinin  dokuzuncu  fıkrası  uyarınca paylaşılabilir.\n"
        "4.  Sorumluluk hükmü.\n"
    )

    sections = extract_document_sections("doc", text)

    assert {s.section_ref for s in sections} == {"1", "2", "3", "4"}


def test_numbered_fallback_skips_list_items_restarting_below_the_sequence():
    headings = [f"{n}.  Gerçek bölüm başlığı numara {n}.\n" for n in range(1, 11)]
    embedded = "1.  Kurumun uygun görüşü alınmış liste kalemi.\n2.  İkinci liste kalemi.\n"
    text = "".join(headings[:6]) + embedded + "".join(headings[6:])

    sections = extract_document_sections("doc", text)

    assert {s.section_ref for s in sections} == {str(n) for n in range(1, 11)}
    p6 = next(s for s in sections if s.section_ref == "6")
    assert "liste kalemi" in p6.content


def test_numbered_fallback_dotted_child_evicts_impersonating_list_item():
    """Genelge 1135 corpus shape: an embedded list's '3.' lands exactly on the
    next expected top-level number; the following authoritative '2.2.' child
    must evict it so the real '3.' still indexes."""
    text = (
        "1.  Sır saklama yükümlülüğüne ilişkin açıklamalar.\n"
        "1.1.  Banka çalışanlarına ait veriler.\n"
        "2.  İstisna tutulan haller.\n"
        "2.1.  Uyum riski kapsamındaki paylaşımlar.\n"
        "1.  Kurumun uygun görüşü kaydıyla kredi kalemi.\n"
        "2.  Kurumun uygun görüşü kaydıyla karşı taraf kalemi.\n"
        "3.  Kurumun uygun görüşüne gerek olmayan kalem.\n"
        "2.2.  Suç gelirlerinin aklanmasının önlenmesi.\n"
        "2.3.  Yönetim kurulu onayı ile paylaşım.\n"
        "3.  Genel ilkeler hakkında açıklamalar.\n"
        "3.1.  Ölçülülük ilkesi.\n"
        "3.2.  Amaçla sınırlılık ilkesi.\n"
        "4.  Yürürlük açıklaması.\n"
        "4.1.  Uygulama tarihi.\n"
        "5.  Son hükümler.\n"
    )

    sections = extract_document_sections("1135", text)

    refs = {s.section_ref for s in sections}
    assert refs == {"1", "1.1", "2", "2.1", "2.2", "2.3", "3", "3.1", "3.2", "4", "4.1", "5"}
    p3 = next(s for s in sections if s.section_ref == "3")
    assert p3.heading == "Genel ilkeler hakkında açıklamalar."


def test_numbered_fallback_refuses_ambiguous_restarting_numbering():
    """Genelge 905 corpus shape: numbering restarts per BÖLÜM, so any indexed
    subset would attach wrong content to paragraf refs; remainder is govde."""
    text = (
        "1.  Muadil ülkeler açıklaması.\n"
        "2.  Yönetmelik açıklaması.\n"
        "1.  Gerçek kişiler prensibi.\n"
        "2.  Tüzel kişiler prensibi.\n"
        "1.  İdari kamu kurumları.\n"
        "2.  Düzenleyici kurumlar.\n"
        "3.  Sosyal kamu kurumları.\n"
    )

    sections = extract_document_sections("905", text)
    assert not any(s.section_type == "paragraf" for s in sections)
    assert {s.section_type for s in sections} == {"govde"}
    assert any("Muadil ülkeler" in s.content for s in sections)


def test_numbered_fallback_refuses_a_trailing_annex_list():
    """When the body's numbering uses a style this grammar does not recognize
    (here "N)"), the only well-formed run is an annex template at the very end.
    Indexing it would bind low refs to annex fragments, so the document must be
    refused entirely rather than publish a guessed subset."""
    body = "\n\n".join(f"{n})  Bu Rehberin {n}. paragrafı. " + ("Metin " * 40) for n in range(1, 41))
    annex = "\n\nEK-2 STRES TESTİ RAPORU\n\n1.  Özkaynak etkisi,\n2.  Tahmini süre,\n3.  Bağımlılıklar,\n"

    sections = extract_document_sections("946", body + annex)
    assert not any(s.section_type == "paragraf" for s in sections)
    # The inline-titled annex line is an annex heading of its own; no paragraf row exists.
    assert {s.section_type for s in sections} == {"govde", "ek"}
    assert [(s.section_ref, s.heading) for s in sections if s.section_type == "ek"] == [("2", "STRES TESTİ RAPORU")]
    assert any("STRES TESTİ" in s.content or "paragrafı" in s.content for s in sections)


def test_numbered_fallback_accepts_a_late_but_in_range_start():
    """Genuine documents can open with a preamble (doc 43 starts at 34%); only a
    run starting past the guard ratio is treated as an annex artifact."""
    preamble = "Giriş bölümü metni. " * 22
    headings = "".join(f"{n}.  Gerçek bölüm {n} içeriği. " + ("Metin " * 20) + "\n" for n in range(1, 8))
    text = preamble + "\n" + headings
    assert 0.2 < len(preamble) / len(text) < 0.4  # a real preamble, still inside the guard

    sections = extract_document_sections("43", text)

    assert {s.section_ref for s in sections} == {str(n) for n in range(1, 8)}


def _dash_body(count: int, *, start: int = 1) -> str:
    return "".join(
        f"{n}-  Bu Rehberin {n} numaralı paragrafı. " + ("Metin " * 20) + "\n\n" for n in range(start, start + count)
    )


def test_dash_form_paragraphs_index_when_no_classic_heading_matches():
    """Rehber 946/1167 corpus shape: paragraphs numbered "1-  ..." not "1.  ..."."""
    sections = extract_document_sections("946", _dash_body(25))

    assert {s.section_type for s in sections} == {"paragraf"}
    assert {s.section_ref for s in sections} == {str(n) for n in range(1, 26)}
    first = next(s for s in sections if s.section_ref == "1")
    assert first.heading.startswith("Bu Rehberin 1 numaralı paragrafı.")


def test_dash_form_requires_more_evidence_than_dot_form():
    """ "N-" is also how this corpus enumerates short lists (audit-firm names in
    1138/803), so a run shorter than the dash minimum must be refused even
    though the same length in dot form would be accepted."""
    short_dash = _dash_body(8)
    short_dot = "".join(f"{n}.  Gerçek bölüm {n} içeriği. " + ("Metin " * 20) + "\n\n" for n in range(1, 9))

    dash_sections = extract_document_sections("1138", short_dash)
    assert not any(s.section_type == "paragraf" for s in dash_sections)
    assert {s.section_type for s in dash_sections} == {"govde"}
    assert len(extract_document_sections("doc", short_dot)) == 8


def test_short_dot_body_survives_a_long_leading_dash_list():
    """The winning style is the one that spans the document, not the one with
    the most items: a long leading definitions list must not displace a shorter
    genuine paragraph body and silently rebind every ref."""
    dash_list = "".join(f"{n}-  Kısaltma {n} açıklaması.\n\n" for n in range(1, 27))
    dot_body = "".join(f"{n}.  Gerçek paragraf {n}. " + ("Metin " * 20) + "\n\n" for n in range(1, 26))

    sections = extract_document_sections("903", dash_list + dot_body)

    assert {s.section_ref for s in sections} == {str(n) for n in range(1, 26)}
    assert all("Kısaltma" not in s.heading for s in sections)


def test_dash_form_rejects_amounts_and_ranges():
    """Sub-numbering is not accepted in dash form, so "1.500- TL" is not a heading."""
    body = _dash_body(25)
    noise = "1.500- TL tutarındaki tavan.\n\n2.750- TL tutarındaki taban.\n\n"

    sections = extract_document_sections("doc", body + noise)

    assert all("." not in s.section_ref for s in sections)
    assert {s.section_ref for s in sections} == {str(n) for n in range(1, 26)}


def _tms_body(numbers: list[str]) -> str:
    return "".join(f"{n} Bu Standardın {n} numaralı paragrafı{' metni' * 12}.\n" for n in numbers)


def test_bare_numbered_paragraphs_index_a_kgk_standard():
    """kgk_tms_12 corpus shape: a paragraph number is printed bare, an inserted
    paragraph carries a letter ("29A", stored as 29/a) and a deleted run is one
    range line.  Example items restart at 1 inside their paragraph."""
    text = (
        "TMS 99\nÖRNEK VERGİLER\n\nAmaç\n\nBu Standardın amacı gelir vergilerini düzenlemektir.\n\nKapsam\n\n"
        + _tms_body(["1", "2", "3", "4", "4A", "5", "6", "7"])
        + "\nÖrnekler\n\n1 Bir makinenin maliyeti 100 TL’dir.\n2 Faiz alacaklarının defter değeri 100 TL’dir.\n\n"
        + _tms_body([str(n) for n in range(8, 30)] + ["29A", "29B"])
        + "\nSunum\n\n30–31 [Silinmiştir]\n\nMahsup\n\n"
        + _tms_body(["32", "33"])
    )

    sections = extract_document_sections("kgk_tms_99", text)

    paragraphs = {s.section_ref: s for s in sections if s.section_type == "paragraf"}
    expected = [str(n) for n in range(1, 30)] + ["4/a", "29/a", "29/b", "32", "33"]
    assert set(paragraphs) == set(expected)
    assert paragraphs["29/a"].content.startswith("29A Bu Standardın 29A numaralı paragrafı")
    assert paragraphs["29/a"].heading.startswith("Bu Standardın 29A numaralı paragrafı")
    assert "1 Bir makinenin maliyeti" in paragraphs["7"].content
    # The range closes 29B without an identity of its own; the heading before it is trimmed.
    assert paragraphs["29/b"].content.endswith("metni.")
    assert all(text[s.start_char : s.end_char].strip() == s.content for s in sections)
    assert any("Bu Standardın amacı" in s.content for s in sections if s.section_type == "govde")


def test_bare_numbered_fallback_requires_a_standard_length_run():
    """A Kurul Kararı can number a few clauses bare; only a run as long as a
    standard's paragraph body indexes."""
    short = extract_document_sections("doc", _tms_body([str(n) for n in range(1, 20)]))
    long = extract_document_sections("doc", _tms_body([str(n) for n in range(1, 21)]))

    assert {s.section_type for s in short} == {"govde"}
    assert {s.section_ref for s in long} == {str(n) for n in range(1, 21)}


def test_bare_numbered_wrapped_numbers_and_cited_paragraphs_are_prose():
    """A wrapped standard name ("bakınız: TMS" + "21 Kur …") and an example
    heading citing a paragraph ("22A paragrafını açıklayan örnek") both land on
    the next expected number; taking them would bind 21 and 22A to wrong text."""
    text = (
        _tms_body([str(n) for n in range(1, 20)])
        + "20 Parasal kalemler geçerli para birimiyle ölçülür (bakınız: TMS\n"
        + "21 Kur Değişiminin Etkileri). Kur farkları ertelenmiş vergiyi etkiler.\n"
        + _tms_body(["21", "22"])
        + "\n22A paragrafını açıklayan örnek\n\nBir gayrimenkulün maliyeti 100 TL’dir.\n\n"
        + _tms_body(["22A", "23"])
    )

    paragraphs = {
        s.section_ref: s for s in extract_document_sections("kgk_tms_99", text) if s.section_type == "paragraf"
    }

    assert "21 Kur Değişiminin Etkileri" in paragraphs["20"].content
    assert paragraphs["21"].content.startswith("21 Bu Standardın 21 numaralı paragrafı")
    assert "22A paragrafını açıklayan örnek" in paragraphs["22"].content
    assert paragraphs["22/a"].content.startswith("22A Bu Standardın 22A numaralı paragrafı")


def test_bare_paragraph_keeps_its_closing_lines_while_standalone_headings_are_trimmed():
    """kgk_tms_12 51A/85 shape: a wrapped closing line and the last table rows
    belong to their paragraph; only a subheading set apart by blank lines before
    the next paragraph is trimmed."""
    text = (
        _tms_body([str(n) for n in range(1, 21)])
        + "21 Yeniden değerleme ertelenmiş vergi doğurur.\n(not: 61A paragrafı uyarınca, ilave borç doğrudan\n"
        + "özkaynaklar ile ilişkilendirilir.)\n\n"
        + "22 Vergi gideri aşağıdaki gibi uzlaştırılır.\nMuhasebe kârı 2,500 3,000\nVergi gideri 760 780\n\n"
        + "Sunum\n\nVergi varlıkları ve vergi borçları\n\n"
        + _tms_body(["23", "24"])
    )

    paragraphs = {
        s.section_ref: s for s in extract_document_sections("kgk_tms_99", text) if s.section_type == "paragraf"
    }

    assert paragraphs["21"].content.endswith("özkaynaklar ile ilişkilendirilir.)")
    assert paragraphs["22"].content.endswith("Muhasebe kârı 2,500 3,000\nVergi gideri 760 780")


def test_dominant_marker_style_wins_over_embedded_other_style():
    """A document numbers its paragraphs one way; the losing style is the
    enumerated lists inside it."""
    body = _dash_body(25)
    embedded_dot_list = "1.  Birinci liste kalemi.\n2.  İkinci liste kalemi.\n3.  Üçüncü liste kalemi.\n\n"

    sections = extract_document_sections("doc", body + embedded_dot_list)

    assert len(sections) == 25
    assert {s.section_type for s in sections} == {"paragraf"}


def test_numbered_body_indexes_behind_a_late_annex_heading():
    """Rehber 1040/945 corpus shape: the only classic headings are annexes near
    the end, so the numbered body would otherwise stay invisible. Both must be
    kept, and the body run must stop at the first classic heading."""
    body = _dash_body(25)
    # The annex numbering CONTINUES the body sequence, so the sequence gate
    # cannot do the region bound's job: only region_end can exclude these.
    annex = "Ek 1: Örnek Kriterler\n\n26-  Annex içindeki sahte paragraf.\n27-  İkinci sahte paragraf.\n"
    text = body + annex

    sections = extract_document_sections("1040", text)

    paragraf = [s for s in sections if s.section_type == "paragraf"]
    ek = [s for s in sections if s.section_type == "ek"]
    assert len(paragraf) == 25
    assert {s.section_ref for s in paragraf} == {str(n) for n in range(1, 26)}
    assert [s.section_ref for s in ek] == ["1"]
    # The body run stops at the annex; annex-internal numbering is not swept in.
    assert all(s.start_char < text.index("Ek 1:") for s in paragraf)


def test_early_classic_heading_does_not_trigger_numbered_body_merge():
    """A document whose classic headings already cover it must be untouched.

    The heading lands at ~18% of the text, so the merge region is non-empty and
    holds the whole dash run: only the 40% gate suppresses the merge. Placing
    the heading at char 0 would make the region empty and the test would pass
    even with the gate removed.
    """
    text = _dash_body(25) + "MADDE 1 - Amaç\nHüküm metni. " + ("Dolgu " * 3000)

    sections = extract_document_sections("doc", text)

    assert all(s.section_type != "paragraf" for s in sections)
    assert ("madde", "1") in {(s.section_type, s.section_ref) for s in sections}


def test_restarting_numbering_in_the_body_region_is_refused():
    """A body that renumbers from 1 per section (1041 does not: it numbers its
    paragraphs globally 1..261) cannot be trusted in any subset, so only the
    classic annex headings survive."""
    restarting = "".join(
        f"{n}.  Bölüm içi madde {n}. " + ("Metin " * 20) + "\n\n" for _ in range(8) for n in range(1, 5)
    )
    text = restarting + "Ek 1: Ekler\n\nEk içeriği.\n"

    sections = extract_document_sections("1041", text)

    assert all(s.section_type != "paragraf" for s in sections)
    assert ("ek", "1") in {(s.section_type, s.section_ref) for s in sections}


def test_classic_headings_suppress_numbered_fallback():
    text = "MADDE 1 - Amaç\nHüküm.\n1.  Numaralı liste kalemi.\n2.  İkinci kalem.\n3.  Üçüncü kalem.\n4.  Dördüncü.\n"

    sections = extract_document_sections("doc", text)

    assert all(s.section_type != "paragraf" for s in sections)
    assert ("madde", "1") in {(s.section_type, s.section_ref) for s in sections}


def test_numbered_dotted_children_survive_a_capped_parent():
    from bddk_mcp.store.section_index import MAX_SECTION_CHARS

    filler = "dolgu metin satırı uzun içerik\n" * ((MAX_SECTION_CHARS // 30) + 50)
    text = (
        "1.  Giriş açıklaması.\n"
        "1.1.  Alt açıklama.\n"
        "2.  Uzun bölüm başlangıcı.\n" + filler + "2.1.  Kapasitenin ötesindeki gerçek alt bölüm.\n"
        "3.  Sonraki bölüm.\n"
    )

    sections = extract_document_sections("doc", text)

    refs = {s.section_ref for s in sections}
    assert "2.1" in refs
    p2 = next(s for s in sections if s.section_ref == "2")
    assert p2.end_char - p2.start_char <= MAX_SECTION_CHARS


def test_trailing_next_article_title_is_not_stored_on_the_previous_madde():
    text = (
        "Madde 69 - İyileştirici önlemler\n"
        "(1) Tedbirlerin alınmasını ister.\n\n"
        "Kısıtlayıcı önlemler\n\n"
        "Madde 70 - Kısıtlayıcı\n"
        "(1) Kredi kullandırımını yasaklar.\n"
    )

    sections = extract_document_sections("mevzuat_5411", text)

    madde69 = next(s for s in sections if s.section_type == "madde" and s.section_ref == "69")
    assert "Tedbirlerin alınmasını ister." in madde69.content
    assert "Kısıtlayıcı önlemler" not in madde69.content
    assert "Madde 70" not in madde69.content


def test_pdf_wrap_debris_after_last_fikra_is_trimmed():
    text = (
        "MADDE 4- (1) Rasyo, ekonomik değer değişimi risk tutarının ana sermayeye "
        "bölünmesi suretiyle hesaplanır.\n\n"
        "(2) Konsolide ve konsolide olmayan rasyo %15’i aşamaz.\n\n"
        "(3) Katılma hesabı kaynaklı olanlar Kurulca belirlenecek oranda dikkate alınır.\n\n"
        "faiz  oranı\n\n"
        "standart\n\n"
        "riski\n\n"
        "Standart yaklaşım uyarınca ekonomik değer değişimi hesaplamasına ilişkin genel hükümler\n\n"
        "MADDE 5- (1) Faize duyarlı pozisyonlar dikkate alınır.\n"
    )

    sections = extract_document_sections("mevzuat_42628", text)

    madde4 = next(s for s in sections if s.section_type == "madde" and s.section_ref == "4")
    assert "%15" in madde4.content
    assert "dikkate alınır." in madde4.content
    assert "Standart yaklaşım" not in madde4.content
    assert "riski" not in madde4.content.split("alınır.")[-1]


def test_closing_formula_stays_in_its_section_while_the_next_title_is_trimmed():
    text = (
        "MADDE 39- (1) Volatilite ayarlamaları kullanılır.\n\n"
        "(2) “H”, volatilite ayarlamasını\nifade etmek üzere aşağıdaki formül yoluyla belirlenir.\n\n"
        "$$\nH = \\sum_{i} a_{i} H_{i}\n$$\n\n"
        "Standart Volatilite Ayarlaması Yaklaşımı\n\n"
        "MADDE 40- (1) Ayarlamalar günlük yapılır.\n\n"
        "(2) Kur uyumsuzluğu için teminatın değeri aşağıdaki formül yoluyla hesaplanır:\n\n"
        "G* = G x (1 – HFX)\n\n"
        "(3) Vade uyumsuzluğu dikkate alınır.\n"
    )

    sections = extract_document_sections("1312", text)
    madde40_start = text.index("MADDE 40")

    def section(section_type: str, ref: str, after: int = 0):
        return next(
            s for s in sections if s.section_type == section_type and s.section_ref == ref and s.start_char >= after
        )

    for closing in (section("madde", "39"), section("fikra", "2")):
        assert closing.content.endswith("$$\nH = \\sum_{i} a_{i} H_{i}\n$$")
    assert "Standart Volatilite" not in section("madde", "39").content
    assert section("fikra", "2", madde40_start).content.endswith("G* = G x (1 – HFX)")


def test_unstructured_body_is_indexed_as_govde_not_paragraf():
    text = (
        "Kurul tarafından 16.11.2006 tarih ve 2026 sayılı Karar ile belirlenen hedef oran. "
        "8 Söz konusu tutar Banka’nın RAV tutarının aynı 2026 sayılı Karar ile belirlenir. "
    ) * 40

    sections = extract_document_sections("951", text)

    assert sections
    assert {s.section_type for s in sections} == {"govde"}
    assert any("2026 sayılı" in s.content for s in sections)
    assert all(s.section_ref.isdigit() for s in sections)


def test_numbered_refusal_still_keeps_footnote_text_in_govde():
    text = (
        "1.  Birinci husus uzun açıklama.\n"
        "2.  İkinci husus uzun açıklama.\n"
        "8 Söz konusu tutar Kurul tarafından 16.11.2006 tarih ve 2026 sayılı Karar ile belirlenir.\n" * 3
    )

    sections = extract_document_sections("951", text)

    assert not any(s.section_type == "paragraf" for s in sections)
    assert {s.section_type for s in sections} == {"govde"}
    assert any("2026 sayılı" in s.content for s in sections)


def test_capped_span_remainder_is_govde():
    from bddk_mcp.store.section_index import MAX_SECTION_CHARS

    filler = "EK TABLO satırı içerik 2026 sayılı " * 2000
    text = "Madde 39 - Yürürlük\nHüküm.\n" + filler

    sections = extract_document_sections("doc", text)

    madde39 = next(s for s in sections if s.section_type == "madde" and s.section_ref == "39")
    assert madde39.end_char - madde39.start_char <= MAX_SECTION_CHARS
    govde = [s for s in sections if s.section_type == "govde"]
    assert govde
    assert any("2026 sayılı" in s.content for s in govde)


def test_govde_windows_end_at_a_paragraph_line_or_word_break():
    """A window cut at exactly the cap split kgk_tms_12's "dönemlerde" into
    "döne" and "mlerde", so section search could find neither half."""
    from bddk_mcp.store.section_index import MAX_SECTION_CHARS

    lines = "Vergilendirilebilir kâr gelecek dönemlerde oluşur.\n" * 1500
    words = "vergilendirilebilir dönemlerde " * 2500
    unbroken = "x" * 30_000

    for text, separator in ((lines, "\n"), (words, " "), (unbroken, None)):
        windows = extract_document_sections("doc", text)

        assert {s.section_type for s in windows} == {"govde"}
        assert "".join(text[s.start_char : s.end_char] for s in windows) == text
        for window in windows[:-1]:
            assert MAX_SECTION_CHARS // 2 < window.end_char - window.start_char <= MAX_SECTION_CHARS
            if separator is None:
                assert window.end_char - window.start_char == MAX_SECTION_CHARS
            else:
                assert text[window.end_char - 1] == separator


def test_guideline_paragraph_inventory_refuses_restarting_numbering():
    text = "İlke 1 - Başlık\n" + "".join(f"{n}. Açıklama metni.\n" for n in range(1, 25))
    text += "İlke 2 - İkinci başlık\n1. Yeniden başlayan numara.\n2. Bir başka kalem.\n"
    sections = extract_document_sections("doc", text)
    assert not any(s.section_type == "paragraf" for s in sections)


def test_guideline_paragraph_inventory_preserves_adjacent_transposition_and_bents():
    numbers = list(range(1, 25))
    numbers[20:22] = [22, 21]
    text = "İlke 1 - Başlık\n" + "".join(f"{n}. Açıklama metni.\n(a) Alt bent metni.\n" for n in numbers)
    sections = extract_document_sections("doc", text)
    paragraphs = [s for s in sections if s.section_type == "paragraf"]
    assert [int(s.section_ref) for s in paragraphs] == numbers
    assert all("(a) Alt bent metni." in s.content for s in paragraphs)
    assert all(text[s.start_char : s.end_char].strip() == s.content for s in paragraphs)


def test_inserted_additional_and_misprinted_articles_are_their_own_sections():
    text = (
        "Madde 66 — (1) Asıl madde.\n\n"
        "Madde 66/A – (Ek:20/2/2020-7222/6 md.)\n(1) Araya eklenen madde.\n\n"
        "MADDE 13/Ç- (Ek:RG-19/8/2015-29450)\n(1) Harfli madde.\n\n"
        "MADDE 14. - 1. Noktalı numara.\n\n"
        "GEÇİÇİ MADDE 1 – (1) Yanlış yazılmış geçici madde.\n\n"
        "GEÇİCİMADDE 2 – Bitişik yazılmış geçici madde.\n\n"
        "MADDE 16 (1) Tiresiz ilk fıkra.\n\n"
        "Ek Madde 1 — (Ek: 13/2/2011-6111/149 md.)\n(1) Ek madde.\n"
    )

    sections = {(s.section_type, s.section_ref): s for s in extract_document_sections("doc", text)}

    assert {
        ("madde", "66"),
        ("madde", "66/a"),
        ("madde", "13/ç"),
        ("madde", "14"),
        ("madde", "16"),
        ("gecici_madde", "1"),
        ("gecici_madde", "2"),
        ("ek_madde", "1"),
    } <= set(sections)
    assert ("madde", "1") not in sections
    assert "Araya eklenen" not in sections[("madde", "66")].content
    assert sections[("madde", "16")].heading == "(1) Tiresiz ilk fıkra."


def test_article_range_closes_the_previous_article_without_an_identity():
    text = (
        "MADDE 145 – (1) Önceki madde.\n\n"
        "MADDE 146 ila 149 – (660 sayılı KHK ile ilgili olup yerine işlenmiştir.)\n\n"
        "MADDE 150 – (1) Bu Kanun yayımı tarihinde yürürlüğe girer.\n"
    )

    maddeler = {s.section_ref: s for s in extract_document_sections("doc", text) if s.section_type == "madde"}

    assert set(maddeler) == {"145", "150"}
    assert "ila 149" not in maddeler["145"].content


def test_cross_references_and_undotted_principles_stay_prose():
    for line in (
        "Madde 4, Fıkra 2, (b) bendinde listelenen süreçler",
        "Geçici Madde 2’ye paralel olarak tamamlanır.",
        "Madde 5 (1) inci fıkrası uyarınca bildirilir.",
        "İlke 6. Bankalar, hangi tür faaliyetlerin yoğunlaşma doğurduğunu belirler.",
    ):
        assert _find_section_starts(line + "\n") == []


def test_dot_principles_index_only_once_a_classic_principle_exists():
    text = (
        "İlke 1. Yoğunlaşma riski yönetimi.\nMetin.\n\n"
        "İlke 2: Bankalar yoğunlaşmayı izler.\nMetin.\n\n"
        "İlke 3. Bankalar acil fonlama planı hazırlar.\nMetin.\n\n"
        "EK 1. YOĞUNLAŞMA RİSKİ ÖRNEKLERİ\nTablo.\n"
    )

    sections = {(s.section_type, s.section_ref): s for s in extract_document_sections("947", text)}

    assert {("ilke", "1"), ("ilke", "2"), ("ilke", "3"), ("ek", "1")} <= set(sections)
    assert "ÖRNEKLERİ" not in sections[("ilke", "3")].content
    # 946 shape: without a classic principle the line stays inside the numbered body.
    body = extract_document_sections("946", "İlke 1. Bankalar yoğunlaşmayı izler.\n" + _dash_body(25))
    assert [s.section_type for s in body].count("paragraf") == 25
    assert not any(s.section_type == "ilke" for s in body)


def test_wrapped_cross_references_and_reference_sentences_are_not_subsections():
    text = (
        "MADDE 7 – (1) Bankaların ödenmiş sermayesi asgari tutardır.\n\n"
        "(2) Ödenmiş sermaye tutarı 7 nci maddenin\n"
        "(2) numaralı fıkrasının (d) bendinde belirtilen tutarın üçte ikisinden az olamaz.\n\n"
        "(3) Tabloda;\n\n"
        "(a) ile (e) arasındaki sütunlar risk ağırlıklarını gösterir.\n\n"
        "(a) Birinci bent.\n"
    )

    subsections = [
        (s.section_type, s.section_ref, s.start_char)
        for s in extract_document_sections("1230", text)
        if s.section_type in ("fikra", "bent")
    ]

    assert [(kind, ref) for kind, ref, _ in subsections] == [("fikra", "2"), ("fikra", "3"), ("bent", "a")]
    assert subsections[-1][2] == text.index("(a) Birinci bent.")


def test_attached_law_lists_close_the_last_article_without_becoming_subsections():
    text = (
        "Madde 171 – Bu Kanun hükümlerini Cumhurbaşkanı yürütür.\n\n"
        "(I) SAYILI CETVEL\n\nKURUMU: BANKACILIK DÜZENLEME VE DENETLEME KURUMU\n\n"
        "(2) SAYILI LİSTE\n\n| SINIF | UNVAN |\n|---|---|\n| GİH | Kurul Üyesi |\n"
    )

    sections = extract_document_sections("mevzuat_5411", text)
    last = next(s for s in sections if (s.section_type, s.section_ref) == ("madde", "171"))

    assert last.content == "Madde 171 – Bu Kanun hükümlerini Cumhurbaşkanı yürütür."
    assert not [s for s in sections if s.section_type in ("fikra", "bent")]
    assert any(s.section_type == "govde" and "SAYILI LİSTE" in s.content for s in sections)


def test_article_quoted_by_an_amending_law_stays_in_the_quoting_article():
    text = (
        "MADDE 24. – (1) Önceki madde.\n\n"
        "MADDE 25. – 1211 sayılı Kanunun 44 üncü maddesi aşağıdaki şekilde değiştirilmiştir;\n\n"
        "Madde 44- Banka, faaliyette bulunan bankalar hakkında bilgi toplar.\n\n"
        "MADDE 26. – (1) Bu Kanun yayımı tarihinde yürürlüğe girer.\n"
    )

    maddeler = {s.section_ref: s for s in extract_document_sections("1230", text) if s.section_type == "madde"}

    assert set(maddeler) == {"24", "25", "26"}
    assert "Madde 44- Banka" in maddeler["25"].content


def test_wrapped_last_sentence_stays_with_its_article_while_the_next_title_is_trimmed():
    text = (
        "MADDE 20. – (1) Çek hesabı açılır.\n\n"
        "(5) Kanun kapsamındaki çeklerin bu Kanun\n"
        "ve diğer mevzuatın çeke ilişkin hükümlerine tâbi olup olmayacağı Bakanlar\n"
        "Kurulunca düzenlenir. Düzenleme dışı kalan hususlarda 6762 sayılı Türk Ticaret\n"
        "Kanunu ve ilgili diğer kanun hükümleri uygulanır\n\n"
        "İdari suç ve cezalar\n\n"
        "MADDE 21. – (1) Cezalar uygulanır.\n"
    )

    madde20 = next(s for s in extract_document_sections("1230", text) if s.section_ref == "20")

    assert madde20.content.endswith("Kanunu ve ilgili diğer kanun hükümleri uygulanır")
    assert "İdari suç" not in madde20.content


def test_reference_and_legend_lines_and_roman_sublists_are_not_bents():
    text = (
        "MADDE 9 – (1) Tabloda;\n\n"
        "(c) bentleri, validasyon sürecinde dikkate alınır.\n\n"
        "(g) sütunu, risk ağırlığını gösterir.\n\n"
        "(c) - (e) sütunları toplamı verir.\n\n"
        "(2) Hesaplamada;\n\n"
        "(i) Mevduat bankaları için:\n\n"
        "(ii) Katılım bankaları için:\n\n"
        "(3) Bentler şunlardır:\n\n"
        "(ı) birinci bent,\n\n"
        "(i) ikinci bent,\n\n"
        "(j) üçüncü bent.\n"
    )

    subsections = [(s.section_type, s.section_ref) for s in extract_document_sections("doc", text)]

    assert [ref for kind, ref in subsections if kind == "bent"] == ["ı", "i", "j"]
    assert [ref for kind, ref in subsections if kind == "fikra"] == ["2", "3"]


def test_draft_rationale_repealed_one_liners_and_annex_tables_keep_their_own_text():
    text = (
        "MADDE 71 — (1) Önceki madde.\n\n"
        "Madde 72 — (Mülga: 6/12/2012-6362/145 md.)\n\n"
        "YEDİNCİ KISIM\n\nKanunî Yükümlülükler\n\nSırların saklanması\n\n"
        "Madde 73 — (1) Kurul üyeleri sırları açıklayamaz.\n\n"
        "| Yönetmeliğin Yayımlandığı Resmî Gazete | Sayısı |\n|---|---|\n| 1 | 29314 |\n\n"
        "EK-1\n\n| 20 GETS İçin (%) | 10 GETS İçin (%) |\n|---|---|\n| 11,314 | 8 |\n\n"
        "EK-2\n\nKISALTMALAR\n\nBK : Beklenen Kayıp\n\nTO : Temerrüt olasılığı\n\n"
        "GENEL GEREKÇE\n\nBilindiği üzere; bu Taslak ile düzenleme yapılması amaçlanmaktadır.\n"
    )

    sections = {(s.section_type, s.section_ref): s for s in extract_document_sections("doc", text)}

    assert sections[("madde", "72")].content == "Madde 72 — (Mülga: 6/12/2012-6362/145 md.)"
    assert "| 1 | 29314 |" not in sections[("madde", "73")].content
    assert sections[("ek", "1")].content.endswith("| 11,314 | 8 |")
    assert sections[("ek", "2")].content.endswith("TO : Temerrüt olasılığı")
    assert any(s.section_type == "govde" and "Bilindiği üzere" in s.content for s in sections.values())


_ARTICLE_TYPES = ("madde", "gecici_madde", "ek_madde")
_SECTION_FIELDS = ("section_type", "section_ref", "start_char", "end_char", "content", "content_hash")


def _field_rows(sections):
    return [tuple(getattr(s, field) for field in _SECTION_FIELDS) for s in sections]


def _madde(sections, ref):
    return next(s for s in sections if s.section_type == "madde" and s.section_ref == ref)


def test_mevzuat_title_line_above_madde_becomes_its_heading():
    text = "Dayanak\n\nMADDE 2 – (1) X.\n\nTanımlar\n\nMADDE 3 – (1) Y;"

    sections = extract_document_sections("mevzuat_22599", text)

    madde3 = _madde(sections, "3")
    assert madde3.heading == "Tanımlar"
    # The title is part of the section and is its heading.
    assert madde3.start_char == text.index("Tanımlar")
    assert madde3.content == "Tanımlar\n\nMADDE 3 – (1) Y;"
    assert "Tanımlar" not in _madde(sections, "2").content
    # The first article's title opens the document: it is the same kind of title line.
    assert _madde(sections, "2").heading == "Dayanak"
    assert _madde(sections, "2").content.startswith("Dayanak\n\nMADDE 2 –")


def test_article_title_opens_its_section_and_is_the_heading():
    with_titles = "Giriş metni.\n\nMADDE 2 – (1) X.\n\nTanımlar\n\nMADDE 3 – (1) Y;\n\nYürürlük\nMADDE 4 – (1) Z."
    # The same text with each title line blanked out character for character.
    without_titles = with_titles.replace("Tanımlar", " " * 8).replace("Yürürlük", " " * 8)
    assert len(with_titles) == len(without_titles)

    titled = extract_document_sections("doc", with_titles)
    untitled = extract_document_sections("doc", without_titles)

    # The title is now part of the section and is the heading: a titled article
    # starts at its title line, so its content and hash include the title.
    assert all(section.content == with_titles[section.start_char : section.end_char].strip() for section in titled)
    assert _field_rows(titled)[0] == _field_rows(untitled)[0]
    madde3, madde4 = _madde(titled, "3"), _madde(titled, "4")
    assert (madde3.start_char, madde3.content) == (with_titles.index("Tanımlar"), "Tanımlar\n\nMADDE 3 – (1) Y;")
    assert (madde4.start_char, madde4.content) == (with_titles.index("Yürürlük"), "Yürürlük\nMADDE 4 – (1) Z.")
    assert madde3.end_char == _madde(untitled, "3").end_char
    assert madde3.content_hash != _madde(untitled, "3").content_hash
    assert [s.heading for s in titled if s.section_type == "madde"] == ["(1) X.", "Tanımlar", "Yürürlük"]
    assert [s.heading for s in untitled if s.section_type == "madde"] == ["(1) X.", "(1) Y;", "(1) Z."]


def test_part_caption_above_article_title_is_not_the_heading():
    text = (
        "MADDE 3 – (1) Önceki madde.\n\n"
        "İKİNCİ BÖLÜM\n\nGenel Hükümler\n\nKurulun teşekkülü\n\n"
        "MADDE 4 – (1) Kurul beş üyeden oluşur.\n\n"
        "ÜÇÜNCÜ BÖLÜM\nÇeşitli ve Son Hükümler\n\n"
        "MADDE 5 – (1) Başlıksız madde.\n"
    )

    sections = extract_document_sections("mevzuat_10494", text)

    assert _madde(sections, "4").heading == "Kurulun teşekkülü"
    # No article title: the part caption under the BÖLÜM line is not taken.
    assert _madde(sections, "5").heading == "(1) Başlıksız madde."


def test_article_title_keeps_its_amendment_note():
    text = (
        "MADDE 13 – (1) Önceki madde.\n\n"
        "Önlem planı tedbirlerinin etkileri (Değişik başlık:RG-29/3/2024-32504)\n\n"
        "MADDE 14 – (1) (Değişik:RG-29/3/2024-32504) Önlem planı tedbirlerinin etkileri;\n"
    )

    madde14 = _madde(extract_document_sections("mevzuat_38416", text), "14")

    assert madde14.heading == "Önlem planı tedbirlerinin etkileri (Değişik başlık:RG-29/3/2024-32504)"
    # The title is part of the section and is the heading.
    assert madde14.content.startswith(
        "Önlem planı tedbirlerinin etkileri (Değişik başlık:RG-29/3/2024-32504)\n\nMADDE 14 –"
    )


def test_kanun_title_above_a_first_sentence_heads_the_article_and_untitled_first_article_is_unchanged():
    kanun = "Amaç\n\nMadde 1 – Bu Kanunun amacı, güveni sağlamaktır.\n\nKapsam\n\nMadde 2 – Kapsam hükmü.\n"
    sections = extract_document_sections("mevzuat_5411", kanun)
    # The title is part of the section and is the heading, also over a Kanun's first sentence.
    assert _madde(sections, "1").heading == "Amaç"
    assert _madde(sections, "1").content.startswith("Amaç\n\nMadde 1 –")
    assert _madde(sections, "2").heading == "Kapsam"
    assert _madde(sections, "2").content == "Kapsam\n\nMadde 2 – Kapsam hükmü."

    untitled = "BANKALARIN DENETİMİ HAKKINDA YÖNETMELİK\n\nMADDE 1 – (1) Bu Yönetmeliğin amacı.\n"
    assert _madde(extract_document_sections("doc", untitled), "1").heading == "(1) Bu Yönetmeliğin amacı."
    assert _madde(extract_document_sections("doc", "MADDE 1 – (1) İlk."), "1").heading == "(1) İlk."


def test_wrapped_sentence_above_article_is_not_its_title():
    text = "MADDE 1 – (1) Bu hüküm Kurulca\nbelirlenir\n\nMADDE 2 – (1) İkinci.\n"

    assert _madde(extract_document_sections("doc", text), "2").heading == "(1) İkinci."


def test_half_of_a_wrapped_title_is_not_the_heading_but_a_title_under_a_part_caption_is():
    wrapped = (
        "MADDE 60 – (1) Önceki madde.\n\n"
        "Nitelikli Merkezi Karşı Taraf Kapsamındaki İşlemlere İlişkin Riske Esas Tutara\n"
        "Dair Üst Sınır\nMADDE 61 – (1) Üst sınır uygulanır.\n"
    )
    assert _madde(extract_document_sections("1313", wrapped), "61").heading == "(1) Üst sınır uygulanır."

    captioned = (
        "MADDE 3 – (1) Önceki madde.\n\n"
        "İKİNCİ BÖLÜM\nBağış ve Yardım Sınırlarına İlişkin Uygulama\nHesaplama dönemi\n\n"
        "MADDE 4 – (1) Bankalar hesaplar.\n"
    )
    assert _madde(extract_document_sections("mevzuat_10739", captioned), "4").heading == "Hesaplama dönemi"


_ARTICLE_LINE_RE = re.compile(
    r"^\s*(?:#{1,6}\s+)?(?:\*{2,3}\s*)?(?:geçi[cç]i\s*madde|ek\s+madde|madde)\s*\d", re.IGNORECASE
)


def test_seed_article_titles_open_their_article_and_are_its_heading():
    seed = Path(__file__).resolve().parents[1] / "seed_data" / "documents.json"
    titled = 0
    for doc in json.loads(seed.read_text(encoding="utf-8")):
        text = doc["markdown_content"] or ""
        sections = extract_document_sections(doc["document_id"], text)
        named = [s for s in sections if s.section_type != "govde"]
        for section in sections:
            if section.section_type not in _ARTICLE_TYPES or _ARTICLE_LINE_RE.match(section.content):
                continue
            # The title is part of the section and is the heading: the section starts at
            # the start of the title line, the title is one line (or a title wrapped over
            # two lines), and only blank lines separate it from the MADDE line.
            titled += 1
            key = (doc["document_id"], section.section_type, section.section_ref)
            assert section.start_char == 0 or text[section.start_char - 1] == "\n", key
            lines = section.content.split("\n")
            article_line = next(i for i, line in enumerate(lines) if _ARTICLE_LINE_RE.match(line))
            title_lines = [line for line in lines[:article_line] if line.strip()]
            assert 1 <= len(title_lines) <= 2, key
            assert lines[: len(title_lines)] == title_lines, key
            assert (
                re.sub(r"[*#]", " ", " ".join(title_lines)).split() == re.sub(r"[*#]", " ", section.heading).split()
            ), key
            # No title stays inside an earlier provision.
            assert not any(
                other.start_char < section.start_char < other.end_char for other in named if other is not section
            ), key
    assert titled > 2000


def test_amendment_note_bold_and_markdown_article_lines_take_the_title_above():
    text = (
        "MADDE 5 – (1) Önceki madde.\n\n"
        "Birinci kalite likit varlıklar\n\n"
        "MADDE 6 – (Değişik:RG-20/8/2015-29451) 1\n\n(1) Birinci kalite varlıklar.\n\n"
        "**Dayanak**\n\n**MADDE 7 –** (1) Bu Tebliğ dayanır.\n\n"
        "## Yürürlük\n\nMADDE 8 – (1) Yürürlüğe girer.\n"
    )

    sections = extract_document_sections("mevzuat_19498", text)

    assert _madde(sections, "6").heading == "Birinci kalite likit varlıklar"
    assert _madde(sections, "6").content.startswith("Birinci kalite likit varlıklar\n\nMADDE 6 –")
    assert _madde(sections, "7").heading == "Dayanak"
    assert _madde(sections, "7").content.startswith("**Dayanak**\n\n**MADDE 7 –**")
    assert _madde(sections, "8").heading == "Yürürlük"
    assert _madde(sections, "8").content.startswith("## Yürürlük\n\nMADDE 8 –")
    assert _madde(sections, "5").content == "MADDE 5 – (1) Önceki madde."


def test_date_led_long_and_wrapped_titles_leave_the_previous_article():
    long_title = "Faiz oranı riski sermaye yükümlülüğü hesaplamasında faiz oranı türev ve bilanço dışı araçlardan kaynaklı pozisyonlar"
    text = (
        "MADDE 22 – (1) Önceki yönetmelik yürürlükten kaldırılmıştır.\n\n"
        "1/1/2018 tarihi itibarıyla TFRS 9’a hazır olmayan bankalar\n\n"
        "GEÇİCİ MADDE 1 – (1) Geçici hüküm.\n\n"
        f"{long_title}\n\n"
        "MADDE 23 – (1) Pozisyonlar.\n\n"
        "Kredi türevleriyle koruma sağlanan pozisyonlar için spesifik risk sermaye\nyükümlülüğü\n\n"
        "MADDE 24 – (1) Yükümlülük.\n"
    )

    sections = extract_document_sections("mevzuat_22599", text)
    gecici = next(s for s in sections if s.section_type == "gecici_madde")

    assert _madde(sections, "22").content == "MADDE 22 – (1) Önceki yönetmelik yürürlükten kaldırılmıştır."
    assert gecici.heading == "1/1/2018 tarihi itibarıyla TFRS 9’a hazır olmayan bankalar"
    assert gecici.content.startswith("1/1/2018 tarihi itibarıyla")
    assert gecici.content.endswith("(1) Geçici hüküm.")
    assert _madde(sections, "23").heading == long_title
    assert _madde(sections, "24").heading == (
        "Kredi türevleriyle koruma sağlanan pozisyonlar için spesifik risk sermaye yükümlülüğü"
    )
    assert _madde(sections, "24").content.startswith("Kredi türevleriyle koruma sağlanan pozisyonlar")
    assert _madde(sections, "23").content.endswith("(1) Pozisyonlar.")


def test_text_that_is_not_a_title_stays_where_it_is():
    # A date-led line that is provision text (it ends a sentence), a line that ends a
    # clause, and a title printed on the MADDE line itself take nothing from above.
    text = (
        "MADDE 1 – (1) Bankalar uygular:\n\n"
        "2018 yılı için oran yüzde birdir.\n\n"
        "MADDE 2 – (1) Kurul belirler;\n\n"
        "Bankalar aşağıdaki hususlara uyar:\n\n"
        "MADDE 3 – (1) Uyulur.\n\n"
        "Başlangıç\n\n"
        "Madde 4 - Karşılık ayrılması\n(1) Karşılık ayrılır.\n"
    )

    sections = extract_document_sections("doc", text)

    assert _madde(sections, "1").content.endswith("2018 yılı için oran yüzde birdir.")
    assert _madde(sections, "3").heading == "(1) Uyulur."
    assert _madde(sections, "3").content.startswith("MADDE 3 –")
    assert _madde(sections, "4").heading == "Karşılık ayrılması"
    assert _madde(sections, "4").content.startswith("Madde 4 -")


def test_title_under_a_caption_that_carries_its_name_or_wraps_in_title_case():
    text = (
        "MADDE 29 – (1) Önceki madde.\n\n"
        "## BEŞİNCİ BÖLÜM Standart Yaklaşım Yönteminde Korelasyonlar\n\n"
        "### Faiz oranı riskine ilişkin korelasyonlar\n\n"
        "MADDE 30 – (1) Korelasyonlar.\n\n"
        "ÜÇÜNCÜ BÖLÜM\nSistemik Önemli Bankalara Ait Yükümlülükler, Uyumsuzluk,\nHesaplama ve Bildirim Dönemi\n"
        "Sistemik önemli banka tamponu\n\n"
        "MADDE 31 – (1) Tampon bulundurulur.\n\n"
        "DÖRDÜNCÜ BÖLÜM\nÇeşitli ve Son Hükümler\n\n"
        "MADDE 32 – (1) Başlıksız madde.\n"
    )

    sections = extract_document_sections("mevzuat_21425", text)

    assert _madde(sections, "30").heading == "Faiz oranı riskine ilişkin korelasyonlar"
    assert _madde(sections, "31").heading == "Sistemik önemli banka tamponu"
    assert _madde(sections, "31").content.startswith("Sistemik önemli banka tamponu\n\nMADDE 31")
    # The part's own name under a caption is never an article title.
    assert _madde(sections, "32").heading == "(1) Başlıksız madde."
    assert _madde(sections, "32").content.startswith("MADDE 32 –")


def _seed_sections(doc_id):
    seed = Path(__file__).resolve().parents[1] / "seed_data" / "documents.json"
    text = next(
        d["markdown_content"] for d in json.loads(seed.read_text(encoding="utf-8")) if d["document_id"] == doc_id
    )
    return text, {(s.section_type, s.section_ref): s for s in extract_document_sections(doc_id, text)}


def test_seed_article_titles_head_bankacilik_kanunu_and_regulations():
    _, bankacilik = _seed_sections("mevzuat_5411")
    for ref, title in (("54", "Kredi sınırları"), ("10", "Faaliyet izni"), ("73", "Sırların saklanması")):
        assert bankacilik[("madde", ref)].heading == title
        assert bankacilik[("madde", ref)].content.startswith(f"{title}\n")
    assert bankacilik[("madde", "54")].content.split("\n\n", 1)[1].startswith("Madde 54 —")

    _, ic_sistemler = _seed_sections("mevzuat_19864")
    assert ic_sistemler[("madde", "3")].heading == "Tanımlar"
    assert ic_sistemler[("madde", "3")].content.startswith("Tanımlar\n")

    # MADDE lines that carry only an amendment note, and a bold title line.
    _, likidite = _seed_sections("mevzuat_19498")
    assert likidite[("madde", "6")].heading == "Birinci kalite likit varlıklar"
    _, sermaye = _seed_sections("mevzuat_21192")
    assert sermaye[("gecici_madde", "1")].heading == "TFRS 9 uygulamaya başlayan bankalara ilişkin uygulama"
    assert "araçlardan kaynaklı pozisyonlar" not in sermaye[("madde", "18")].content
    _, kra = _seed_sections("mevzuat_20029")
    assert kra[("madde", "2")].heading == "Dayanak"
    assert kra[("madde", "42")].heading == "Standart Volatilite Ayarlaması Yaklaşımı"
    assert kra[("madde", "42")].content.startswith("**Standart Volatilite Ayarlaması Yaklaşımı**\n")

    # A date-led title no longer ends the previous article.
    _, karsilik = _seed_sections("mevzuat_22599")
    assert "TFRS 9’a hazır olmayan bankalar" not in karsilik[("madde", "22")].content
    assert karsilik[("gecici_madde", "1")].heading == "1/1/2018 tarihi itibarıyla TFRS 9’a hazır olmayan bankalar"
    assert karsilik[("madde", "23")].heading == "Yürürlük"


def test_no_seed_section_boundary_falls_inside_an_article_title():
    seed = Path(__file__).resolve().parents[1] / "seed_data" / "documents.json"
    for doc in json.loads(seed.read_text(encoding="utf-8")):
        text = doc["markdown_content"] or ""
        sections = extract_document_sections(doc["document_id"], text)
        boundaries = {s.start_char for s in sections} | {s.end_char for s in sections}
        for section in sections:
            if section.section_type not in _ARTICLE_TYPES or _ARTICLE_LINE_RE.match(section.content):
                continue
            title_end = section.start_char + len(section.content.split("\n", 1)[0])
            inside = [b for b in boundaries if section.start_char < b < title_end]
            assert not inside, (doc["document_id"], section.section_type, section.section_ref, inside)


def test_capped_section_window_ends_at_a_break_and_states_its_true_lengths():
    from bddk_mcp.store.section_index import (
        MAX_SECTION_CHARS,
        split_section_truncation_notice,
        truncation_notice_original_length,
    )

    paragraph = "Bankalar dönemlerde karşılık ayırır ve raporlar " * 30
    body = "\n\n".join(paragraph.strip() for _ in range(20))
    text = f"MADDE 1 – (1) {body}\n\nMADDE 2 – (1) Son.\n"
    madde2 = text.index("MADDE 2")

    sections = extract_document_sections("doc", text)
    madde1 = _madde(sections, "1")
    legal, notice = split_section_truncation_notice(madde1.content)

    assert madde1.end_char - madde1.start_char <= MAX_SECTION_CHARS
    assert text[madde1.end_char - 2 : madde1.end_char] == "\n\n"
    assert legal == text[madde1.start_char : madde1.end_char].strip()
    # The notice records the uncapped span, which runs to the blank line before MADDE 2.
    assert madde1.start_char + truncation_notice_original_length(notice) == madde2 - 1
    assert f"karakterden {madde1.end_char - madde1.start_char} karaktere" in notice
    # The continuation starts exactly where the window ends.
    assert any(s.section_type == "govde" and s.start_char == madde1.end_char for s in sections)


def test_no_seed_capped_section_ends_inside_a_word_and_its_continuation_follows():
    from bddk_mcp.store.section_index import split_section_truncation_notice, truncation_notice_original_length

    seed = Path(__file__).resolve().parents[1] / "seed_data" / "documents.json"
    capped = 0
    for doc in json.loads(seed.read_text(encoding="utf-8")):
        text = doc["markdown_content"] or ""
        sections = extract_document_sections(doc["document_id"], text)
        for section in sections:
            notice = split_section_truncation_notice(section.content)[1]
            if notice is None:
                continue
            capped += 1
            key = (doc["document_id"], section.section_type, section.section_ref)
            end = section.end_char
            assert not (text[end - 1].isalnum() and text[end].isalnum()), key
            length = end - section.start_char
            assert f"karakterden {length} karaktere" in notice, key
            assert truncation_notice_original_length(notice) > length, key
            # The next text is covered: a nested row runs on, or a govde window starts exactly here.
            following = end + len(text[end:]) - len(text[end:].lstrip())
            assert any(s.start_char == end and s.section_type == "govde" for s in sections) or any(
                s is not section and s.start_char <= following < s.end_char for s in sections
            ), key
    # 1171 madde 13 and its fikra 2 were quoted articles (item B2); 1285 paragraf 2.2.4 ends at its
    # undotted sibling 2.2.5 and is no longer capped (item B5d).
    assert capped == 23


def test_table_rows_at_the_end_of_any_section_are_its_content():
    text = (
        "GEÇİCİ MADDE 1 – (1) Oranlar aşağıdaki gibidir.\n\n"
        "| Yıl | SKT Oranı (%) |\n|---|---|\n| 2017 | 1,25 |\n| 2018 | 1,875 |\n\n"
        "Yürürlük\n\nMADDE 2 – (1) Yürürlüğe girer.\n"
    )

    sections = extract_document_sections("mevzuat_18991", text)
    gecici = next(s for s in sections if s.section_type == "gecici_madde")

    assert gecici.content.endswith("| 2018 | 1,875 |")
    assert _madde(sections, "2").heading == "Yürürlük"


_AMENDMENT_TABLE = (
    "|  | Yönetmeliğin Yayımlandığı Resmî Gazete’nin | Yönetmeliğin Yayımlandığı Resmî Gazete’nin |\n"
    "|---|---|---|\n|  | Tarihi | Sayısı |\n|  | 1/11/2006 | 26333 |\n"
)


def test_amendment_history_table_after_the_last_article_is_not_part_of_it():
    text = (
        "MADDE 8 – (1) Önceki madde.\n\nYürütme\n\nMADDE 9 – (1) Bu Yönetmelik hükümlerini Başkan yürütür.\n\n"
        f"{_AMENDMENT_TABLE}\n(1) Bu değişiklik 1/1/2012 tarihinde yürürlüğe girer.\n\n"
        "EK-1\n\nBİLDİRİM CETVELİ\n\n| A | B |\n|---|---|\n| 1 | 2 |\n"
    )

    sections = extract_document_sections("mevzuat_10731", text)
    madde9 = _madde(sections, "9")
    assert madde9.content.endswith("Başkan yürütür.")
    assert "Yayımlandığı" not in madde9.content
    govde = next(s for s in sections if s.section_type == "govde")
    assert govde.content.startswith("|  | Yönetmeliğin Yayımlandığı")
    assert govde.start_char == madde9.end_char
    # The amendment footnote after the table is no fıkra of the last article; the annex keeps its row.
    assert not [s for s in sections if s.section_type == "fikra"]
    assert next(s for s in sections if s.section_type == "ek").content.endswith("| 1 | 2 |")


def test_a_table_inside_a_provision_is_never_split_off():
    text = (
        f"MADDE 1 – (1) Tablo:\n\n{_AMENDMENT_TABLE}\n(2) Devam eden hüküm.\n\n"
        "MADDE 2 – (1) Son madde.\n\nEK-1\n\n"
        f"{_AMENDMENT_TABLE}"
    )

    sections = extract_document_sections("doc", text)

    assert "1/11/2006" in _madde(sections, "1").content
    assert next(s for s in sections if s.section_type == "ek").content.endswith("| 1/11/2006 | 26333 |")


def test_seed_table_rows_stay_in_their_provision_and_amendment_tables_leave_the_last_article():
    _, ozkaynak = _seed_sections("mevzuat_18991")
    assert ozkaynak[("gecici_madde", "1")].content.endswith("| 2018 | 1,875 |")
    text, _ = _seed_sections("1240")
    fikra = [s for s in extract_document_sections("1240", text) if (s.section_type, s.section_ref) == ("fikra", "3")]
    assert fikra[0].content.endswith("| 3 | > 900 | 18 |")
    _, sinirlar = _seed_sections("mevzuat_10731")
    assert "Yayımlandığı Resmî Gazete" not in sinirlar[("madde", "9")].content

    seed = Path(__file__).resolve().parents[1] / "seed_data" / "documents.json"
    header = re.compile(r"^[ \t]*\|.*Yayımlandığı\s+Resm[iî]\s+Gazete", re.MULTILINE)
    documents = 0
    for doc in json.loads(seed.read_text(encoding="utf-8")):
        text = doc["markdown_content"] or ""
        match = header.search(text)
        if match is None:
            continue
        documents += 1
        sections = extract_document_sections(doc["document_id"], text)
        position = match.start()
        while position < len(text) and text[position:].lstrip(" \t").startswith("|"):
            covering = [s.section_type for s in sections if s.start_char <= position < s.end_char]
            assert covering and not set(covering) & {*_ARTICLE_TYPES, "fikra", "bent"}, (doc["document_id"], position)
            line_end = text.find("\n", position)
            position = len(text) if line_end == -1 else line_end + 1
    assert documents == 44


def test_colon_form_annexes_end_the_last_article_and_take_their_title():
    text = (
        "Yürütme\n\nMADDE 10 – (1) Bu Yönetmelik hükümlerini Kurum Başkanı yürütür.\n\n"
        "EK: 1\n\nKALDIRAÇ ORANI BİLDİRİM TABLOSU\n\n| A | B |\n|---|---|\n| 1 | 2 |\n"
        "EK: 2\n\nKONSOLİDE KALDIRAÇ ORANI BİLDİRİM TABLOSU\n\n| A | B |\n|---|---|\n| 3 | 4 |\n"
    )

    sections = extract_document_sections("mevzuat_18990", text)

    assert _madde(sections, "10").content.endswith("Kurum Başkanı yürütür.")
    annexes = [(s.section_ref, s.heading, s.content.splitlines()[0]) for s in sections if s.section_type == "ek"]
    assert annexes == [
        ("1", "KALDIRAÇ ORANI BİLDİRİM TABLOSU", "EK: 1"),
        ("2", "KONSOLİDE KALDIRAÇ ORANI BİLDİRİM TABLOSU", "EK: 2"),
    ]


def test_inline_titled_annexes_close_a_dash_paragraph_body_and_lowercase_references_stay_prose():
    body = "".join(f"{n}-  Bu Rehberin {n} numaralı paragrafı. " + ("Metin " * 20) + "\n\n" for n in range(1, 31))
    annexes = (
        "EK-1 TANIMLAR\n\nAd hoc stres testi\n\nTanım metni.\n\nEK-2 STRES TESTİ RAPORU\n\n1. Risklerin Tanımlanması\n"
    )
    text = body + annexes

    sections = extract_document_sections("946", text)

    paragraphs = [s for s in sections if s.section_type == "paragraf"]
    assert [s.section_ref for s in paragraphs] == [str(n) for n in range(1, 31)]
    assert "EK-1" not in paragraphs[-1].content
    assert [(s.section_ref, s.heading) for s in sections if s.section_type == "ek"] == [
        ("1", "TANIMLAR"),
        ("2", "STRES TESTİ RAPORU"),
    ]
    prose = "3. Paragraf metni:\n\nEK-1 bir ilâ sekizinci fıkralarda belirtilen hükümler esas alınacaktır.\n"
    assert not [s for s in extract_document_sections("903", "MADDE 1 – (1) X.\n\n" + prose) if s.section_type == "ek"]


def test_colon_form_echoes_and_page_captions_create_no_annex_rows():
    echo = (
        "MADDE 1 – (1) Hüküm.\n\nEK-1\n\n(Değişik:RG-4/2/2022-31740)\n\nEk:1\nBİN TL\n\nCETVEL\n\n"
        "EK-2\n\n(Değişik:RG-4/2/2022-31740)\n\nEk: 2\nBİN TL\n\nCETVEL\n"
    )
    annexes = [s.section_ref for s in extract_document_sections("mevzuat_19498", echo) if s.section_type == "ek"]
    assert annexes == ["1", "2"]

    captions = "İlke 1 – Başlık\n\nMetin.\n\n" + "".join(
        f"Ek:{n}\n\nSayfa metni {n}.\n\n" for n in (1, 1, 2, 2, 3, 3, 3)
    )
    assert not [s for s in extract_document_sections("934", captions) if s.section_type == "ek"]
    # Without an article before it, a colon-form line is a sub-annex caption of an explanatory annex.
    genelge = "Açıklamalar.\n\n" + "Metin satırı.\n" * 40 + "EK: 1\nÖRNEKLER\n\nListe.\n"
    assert not [s for s in extract_document_sections("905", genelge) if s.section_type == "ek"]


def test_annex_heading_comes_from_the_title_under_the_annex_line_and_spans_are_unchanged():
    text = (
        "MADDE 1 – (1) Hüküm.\n\n## EK-2\n\n## (Değişik:RG-4/2/2022-31740)$^{(1)}$\n\n## KARŞI TARAF KREDİ RİSKİ\n\n"
        "Metin.\n\nEK–3\n\nİHTİSAS KREDİLERİNE İLİŞKİN SINIFLANDIRMA KRİTERLERİ\n\nMetin.\n\n"
        "EK-4\n\n| A | B |\n|---|---|\n| 1 | 2 |\n"
    )

    sections = extract_document_sections("mevzuat_21192", text)

    annexes = {s.section_ref: s for s in sections if s.section_type == "ek"}
    assert annexes["2"].heading == "KARŞI TARAF KREDİ RİSKİ"
    assert annexes["3"].heading == "İHTİSAS KREDİLERİNE İLİŞKİN SINIFLANDIRMA KRİTERLERİ"
    assert annexes["4"].heading == ""
    for annex in annexes.values():
        assert annex.content == text[annex.start_char : annex.end_char].strip()
        assert text[annex.start_char :].lstrip("# ").startswith("EK")


def test_a_form_addressee_under_the_annex_number_is_no_annex_title():
    text = (
        "MADDE 1 – (1) Hüküm.\n\nEK-3\n\nTAAHHÜTNAME\n\nMetin.\n\n"
        "EK-4\n\nTASARRUF MEVDUATI SİGORTA FONU KURULU BAŞKANLIĞINA\n\nBankacılık Düzenleme ve Denetleme Kurumuna verilmek üzere.\n"
    )

    annexes = {s.section_ref: s for s in extract_document_sections("mevzuat_16071", text) if s.section_type == "ek"}

    assert annexes["3"].heading == "TAAHHÜTNAME"
    assert annexes["4"].heading == ""
    _, seed = _seed_sections("mevzuat_16071")
    assert seed[("ek", "4")].heading == ""


def test_seed_annexes_after_the_last_article_or_paragraph_are_their_own_rows():
    _, kaldirac = _seed_sections("mevzuat_18990")
    assert kaldirac[("ek", "1")].heading == "KALDIRAÇ ORANI BİLDİRİM TABLOSU"
    assert "KALDIRAÇ ORANI BİLDİRİM TABLOSU" not in kaldirac[("madde", "10")].content
    assert kaldirac[("ek", "2")].heading == "KONSOLİDE KALDIRAÇ ORANI BİLDİRİM TABLOSU"
    _, stres = _seed_sections("946")
    assert "EK-1 TANIMLAR" not in stres[("paragraf", "163")].content
    assert stres[("ek", "1")].heading == "TANIMLAR"
    assert stres[("ek", "2")].heading == "STRES TESTİ RAPORU"
    assert len([key for key in stres if key[0] == "paragraf"]) == 163
    _, sermaye = _seed_sections("mevzuat_21192")
    assert sermaye[("ek", "2")].heading == "KARŞI TARAF KREDİ RİSKİ"
    _, likidite = _seed_sections("mevzuat_19498")
    assert sorted(key[1] for key in likidite if key[0] == "ek") == ["1", "2", "3"]
    _, idd = _seed_sections("934")
    assert not [key for key in idd if key[0] == "ek"]


def test_genelge_that_only_quotes_articles_has_no_article_rows():
    text = (
        "Konu: Elektronik bankacılık\n\nYönetmeliğin 12 nci ve 13 üncü maddelerinde de aşağıdaki hükümlere yer verilmiştir:\n\n"
        "MADDE 12 – (1) Alıntılanan hüküm.\n\n(2) İkinci fıkra.\n\nKimlik tespiti\n\n"
        "MADDE 13 – (1) Alıntılanan başka hüküm.\n\n(2) Bu Yönetmelikte yer alan şartlar.\n\n"
        "Bilgilerinize rica olunur. Tebliğ olunur.\n\nBaşkan\n\nEk: Açıklamalar\n\n"
        + "Açıklama metni ve gereklilikler. " * 20
        + "\n\n(2) Ekteki bir madde işareti.\n"
    )

    sections = extract_document_sections("1171", text)

    assert {s.section_type for s in sections} == {"govde"}
    assert any("Ek: Açıklamalar" in s.content for s in sections)


def test_unnumbered_annex_line_closes_the_last_article():
    for annex in ("Ek: Eşleştirme Tablosu", "EK- İSEDES RAPORUNUN ŞEKLİ VE İÇERİĞİ"):
        text = (
            "MADDE 1 – (1) Birinci madde.\n\nMADDE 2 – (1) Son madde hükmü.\n\n(2) İkinci fıkra.\n\n"
            f"{annex}\n\n| Not | Derece |\n|---|---|\n| AAA | 1 |\n\n(3) Tablo dipnotu.\n"
        )

        sections = extract_document_sections("696", text)

        assert _madde(sections, "2").content.endswith("İkinci fıkra.")
        assert [s.section_ref for s in sections if s.section_type == "fikra"] == ["2"]
        assert not [s for s in sections if s.section_type == "ek"]
        assert next(s for s in sections if s.section_type == "govde").content.startswith(annex)


def test_seed_quoted_articles_and_unnumbered_annexes():
    text, genelge = _seed_sections("1171")
    assert {key[0] for key in genelge} == {"govde"}
    _, ulke = _seed_sections("955")
    assert "ÜLKE RİSKİNİ ETKİLEYEN FAKTÖRLER" not in ulke[("paragraf", "48")].content
    assert len([key for key in ulke if key[0] == "paragraf"]) == 48
    _, sir = _seed_sections("1135")
    assert len([key for key in sir if key[0] == "paragraf"]) == 28


def test_nested_row_ends_with_its_capped_parent_window():
    from bddk_mcp.store.section_index import MAX_SECTION_CHARS, split_section_truncation_notice

    filler = "".join(f"Ek metni satırı {n} ve açıklaması.\n\n" for n in range(520))
    paragraphs = "".join(f"{n}. Numaralı paragraf {n} metni. " + "Hüküm " * 30 + "\n\n" for n in range(8, 60))
    text = (
        f"MADDE 1 – (1) Hüküm.\n\nEK-2\n\nKARŞI TARAF KREDİ RİSKİ\n\n{filler}(2) Netleştirme fıkrası.\n\n{paragraphs}"
    )
    assert MAX_SECTION_CHARS - 2_000 < text.index("(2) Netleştirme") - text.index("EK-2") < MAX_SECTION_CHARS

    sections = extract_document_sections("mevzuat_21192", text)

    annex = next(s for s in sections if s.section_type == "ek")
    fikra = next(s for s in sections if s.section_type == "fikra" and s.section_ref == "2")
    assert fikra.end_char == annex.end_char
    # The fıkra's next marker lies past the window and is no reliable end, so no longer span is claimed.
    legal, notice = split_section_truncation_notice(fikra.content)
    assert legal == text[fikra.start_char : fikra.end_char].strip() and notice is None
    assert "40. Numaralı paragraf" not in fikra.content
    assert any(s.section_type == "govde" and s.start_char == annex.end_char for s in sections)
    # An article that is not capped keeps its fıkra ends.
    short = extract_document_sections("doc", "MADDE 1 – (1) Bir.\n\n(2) İki.\n\n(3) Üç.\n\nMADDE 2 – (1) Dört.\n")
    assert [s.content for s in short if s.section_type == "fikra"] == ["(2) İki.", "(3) Üç."]


def test_seed_netting_fikra_does_not_run_past_its_annex_window():
    text, sermaye = _seed_sections("mevzuat_21192")
    annex = sermaye[("ek", "2")]
    sections = extract_document_sections("mevzuat_21192", text)
    netting = next(s for s in sections if (s.section_type, s.section_ref, s.start_char) == ("fikra", "2", 191866))
    assert netting.end_char == annex.end_char
    assert "YM = max (CPD – CTD, ET + ATT – NBTT, 0)" not in netting.content
    assert any(s.section_type == "govde" and s.start_char == annex.end_char for s in sections)


def test_amendment_footnotes_are_not_fikra_rows_of_the_last_article():
    text = (
        "Yürürlük\n\nMADDE 5 – (1) Bu Yönetmelik yayımı tarihinde yürürlüğe girer.\n\n"
        "Yürütme\n\nMADDE 6 – (1) Bu Yönetmelik hükümlerini Kurum Başkanı yürütür.\n\n"
        "(1) 23/10/2015 tarihli ve 29511 sayılı Resmi Gazete’de yayımlanan Yönetmelik değişikliği 31/3/2016 tarihinde "
        "yürürlüğe girer.\n\n(2) Bu değişiklik 1/1/2017 tarihinde yürürlüğe girer.\n\n"
    )

    sections = extract_document_sections("mevzuat_10750", text)

    assert _madde(sections, "6").content.endswith("Kurum Başkanı yürütür.")
    assert not [s for s in sections if s.section_type == "fikra"]
    govde = next(s for s in sections if s.section_type == "govde")
    assert govde.content.startswith("(1) 23/10/2015") and govde.content.endswith("tarihinde yürürlüğe girer.")


def test_non_advancing_marker_is_no_fikra_and_genuine_fikras_are_unchanged():
    text = (
        "MADDE 1 – (1) Birinci fıkra.\n\n(1) Dipnot satırı, 1/1/2012 tarihli değişiklik.\n\n(2) İkinci fıkra.\n\n"
        "MADDE 2 –\n\n(1) Bir.\n\n(2) İki.\n\n(3) Üç.\n\nMADDE 3 – (1) Son.\n"
    )

    sections = extract_document_sections("doc", text)

    fikras = [(s.section_ref, s.content) for s in sections if s.section_type == "fikra"]
    assert fikras == [
        ("2", "(2) İkinci fıkra."),
        ("1", "(1) Bir."),
        ("2", "(2) İki."),
        ("3", "(3) Üç."),
    ]
    assert "(1) Dipnot satırı" in _madde(sections, "1").content


def test_seed_amendment_footnotes_have_no_fikra_rows():
    _, kurumsal = _seed_sections("mevzuat_10750")
    assert ("fikra", "1") not in kurumsal
    assert kurumsal[("madde", "6")].content.endswith("Kurumu Başkanı yürütür.")
    seed = Path(__file__).resolve().parents[1] / "seed_data" / "documents.json"
    footnote = re.compile(r"^Bu değişiklik\b|tarihli ve \d+ sayılı Resm[iî] Gazete’de yayımlanan (?:\w+ )?değişikli")
    rows = [
        (doc["document_id"], s.start_char)
        for doc in json.loads(seed.read_text(encoding="utf-8"))
        for s in extract_document_sections(doc["document_id"], doc["markdown_content"] or "")
        if s.section_type == "fikra" and footnote.search(s.heading)
    ]
    assert rows == []


def test_outline_headings_far_behind_a_global_paragraph_sequence_are_dividers():
    def paragraphs(numbers):
        return "".join(f"{n}. Rehber paragrafı {n}. " + ("Metin " * 20) + "\n\n" for n in numbers)

    text = (
        paragraphs(range(1, 41))
        + "2.3.4. Kredibiliteyi Değerlendirmede Kullanılan Modeller\n\n"
        + paragraphs(range(41, 61))
        + "Ek 1 – Kredi tahsis kriterleri\n\nEk içeriği.\n"
    )

    sections = extract_document_sections("1041", text)

    paragraf = [s for s in sections if s.section_type == "paragraf"]
    assert [s.section_ref for s in paragraf] == [str(n) for n in range(1, 61)]
    assert all(s.content.startswith(f"{s.section_ref}. ") for s in paragraf)
    assert not any("2.3.4." in s.content for s in paragraf)
    assert ("ek", "1") in {(s.section_type, s.section_ref) for s in sections}


def test_seed_credit_guide_body_is_its_numbered_paragraphs():
    text, rehber = _seed_sections("1041")
    paragraf = [s for s in extract_document_sections("1041", text) if s.section_type == "paragraf"]
    assert [s.section_ref for s in paragraf] == [str(n) for n in range(1, 262)]
    assert all(text[s.start_char : s.end_char].startswith(f"{s.section_ref}. ") for s in paragraf)
    assert rehber[("paragraf", "54")].content.startswith("54. Bankalarda kredi karar verme")
    # The 1-6 list inside a definition stays in paragraph 7.
    assert rehber[("paragraf", "7")].start_char < 7258 < rehber[("paragraf", "7")].end_char
    assert {key[1] for key in rehber if key[0] == "ek"} == {"1", "2", "3"}


def _guide_paragraphs(numbers, marker="."):
    return "".join(f"{n}{marker} Rehber paragrafı {n} metni. " + ("Metin " * 12) + "\n\n" for n in numbers)


def test_dash_numbered_paragraphs_under_principles_are_rows_and_a_repeated_number_binds_to_neither_text():
    text = (
        _guide_paragraphs(range(1, 4), "-")
        + "İlke 1- Bankalar kredi riski stratejisi oluşturur.\n\n"
        + _guide_paragraphs([*range(4, 57), 58, 59, 60], "-")
        + "İlke 2- Kredi tahsis süreci.\n\n(a) Bent metni.\n\n"
        + _guide_paragraphs(range(60, 80), "-")
    )

    sections = extract_document_sections("949", text)

    paragraf = [s for s in sections if s.section_type == "paragraf"]
    assert [s.section_ref for s in paragraf] == [str(n) for n in [*range(1, 57), 58, 59, *range(61, 80)]]
    assert all(s.content.startswith(f"{s.section_ref}- ") for s in paragraf)
    assert "60- Rehber" not in next(s for s in paragraf if s.section_ref == "59").content
    assert [s.section_ref for s in sections if s.section_type == "ilke"] == ["1", "2"]
    assert ("bent", "a") in {(s.section_type, s.section_ref) for s in sections}


def test_principle_guide_with_annexes_indexes_its_preamble_paragraphs_but_not_the_annex_list():
    text = (
        _guide_paragraphs(range(1, 5))
        + "5. Bu rehberde yer alan;\n\na) Acil ve beklenmedik durum planı: Tanım.\n\n"
        + _guide_paragraphs(range(6, 20))
        + "İlke 1: Yoğunlaşma riski yönetimi.\n\n"
        + _guide_paragraphs(range(20, 25))
        + "İlke 2. Yoğunlaşma riski limitleri.\n\n"
        + _guide_paragraphs(range(25, 30))
        + "EK 1. YOĞUNLAŞMA RİSKİ ÖRNEKLERİ\n\n"
        + _guide_paragraphs(range(1, 7))
    )

    sections = extract_document_sections("947", text)

    paragraf = {s.section_ref: s for s in sections if s.section_type == "paragraf"}
    assert sorted(paragraf, key=int) == [str(n) for n in range(1, 30)]
    assert paragraf["5"].content.startswith("5. Bu rehberde yer alan;")
    assert all(s.end_char <= text.index("EK 1.") for s in paragraf.values())
    assert {(s.section_type, s.section_ref) for s in sections} >= {("ek", "1"), ("ilke", "1"), ("ilke", "2")}


def test_body_numbered_before_the_principles_stands_alone_when_the_principles_restart_lists():
    text = (
        _guide_paragraphs(range(1, 31))
        + "İlke 1: Birinci ilke.\n\nİlke 2: İkinci ilke.\n\nEk:1\n\n"
        + _guide_paragraphs(range(1, 5))
        + "Ek:2\n\n"
        + _guide_paragraphs(range(1, 4))
    )

    sections = extract_document_sections("934", text)

    paragraf = [s for s in sections if s.section_type == "paragraf"]
    assert [s.section_ref for s in paragraf] == [str(n) for n in range(1, 31)]
    assert paragraf[-1].end_char <= text.index("İlke 1:")
    assert not any("Ek:" in s.content for s in paragraf)


def test_seed_principle_guides_have_paragraph_rows():
    for doc_id, last, missing in (("947", 101, set()), ("949", 190, {57, 60}), ("934", 102, set())):
        text, rows = _seed_sections(doc_id)
        paragraf = [s for s in extract_document_sections(doc_id, text) if s.section_type == "paragraf"]
        assert [int(s.section_ref) for s in paragraf] == [n for n in range(1, last + 1) if n not in missing]
        assert all(re.match(rf"{s.section_ref}[.-]\s", text[s.start_char : s.end_char]) for s in paragraf)
    _, kredi = _seed_sections("949")
    assert kredi[("paragraf", "182")].content.startswith("182- Bankalar donuk alacakların yönetimine")
    _, yogunlasma = _seed_sections("947")
    assert yogunlasma[("paragraf", "5")].content.startswith("5. Bu rehberde yer alan;")


def test_numbered_body_ends_at_an_unnumbered_annex_whose_outline_restarts():
    body = "".join(f"{n}. Rehber paragrafı {n}. " + ("Metin " * 15) + "\n\n" for n in range(1, 12))
    annex = "EK- İSEDES RAPORUNUN ŞEKLİ VE İÇERİĞİ\n\n" + "".join(
        f"{n}. RAPOR BÖLÜMÜ {n}\n\nBölüm açıklaması.\n\n{n}.1. Alt bölüm\n\nMetin.\n\n"
        for n in (1, 2, 1, 2, 3, 1, 2, 3, 4)
    )

    sections = extract_document_sections("951", body + annex)

    paragraf = [s for s in sections if s.section_type == "paragraf"]
    assert [s.section_ref for s in paragraf] == [str(n) for n in range(1, 12)]
    assert paragraf[-1].end_char <= len(body)
    assert any(s.section_type == "govde" and s.content.startswith("EK- İSEDES") for s in sections)


def test_seed_isedes_guide_paragraphs_end_before_its_annex():
    text, rehber = _seed_sections("951")
    assert sorted(int(key[1]) for key in rehber if key[0] == "paragraf") == list(range(1, 12))
    assert rehber[("paragraf", "4")].content.startswith("4. İSEDES Raporu’nda kullanılan verilerin doğruluğu")
    assert rehber[("paragraf", "11")].end_char <= text.index("EK- İSEDES RAPORUNUN")


def test_undotted_upper_case_sub_headings_and_lettered_parts_close_the_outline_entry():
    def entry(ref, title):
        return f"{ref} {title}\n\n" + "Kriter metni. " * 20 + "\n\n"

    text = (
        entry("1.", "TÜKETİCİ FİNANSMANI")
        + entry("2.", "TİCARİ FİNANSAL VARLIKLAR")
        + entry("2.1.", "GENEL KRİTERLER")
        + entry("2.2.", "SEKTÖREL KRİTERLER")
        + entry("2.2.4.", "İMALAT SEKTÖRÜ")
        + "2.5 milyon TL tutarındaki krediler için ek kriter.\n\n"
        + entry("2.2.5", "LOJİSTİK VE ULAŞTIRMA SEKTÖRÜ")
        + entry("2.2.6", "İNŞAAT VE GAYRİMENKUL SEKTÖRÜ")
        + "B - ÇEVREYE ÖNEMLİ ZARAR VERMEME KRİTERLERİ\n\n"
        + "Ölçüt metni. " * 20
    )

    sections = {s.section_ref: s for s in extract_document_sections("1285", text) if s.section_type == "paragraf"}

    assert sorted(sections) == ["1", "2", "2.1", "2.2", "2.2.4", "2.2.5", "2.2.6"]
    assert sections["2.2.4"].end_char <= text.index("2.2.5 LOJİSTİK")
    assert "2.5 milyon TL" in sections["2.2.4"].content
    assert sections["2.2.6"].end_char <= text.index("B - ÇEVREYE")
    assert sections["2"].end_char <= text.index("B - ÇEVREYE")


def test_seed_green_asset_decision_sector_headings():
    text, karar = _seed_sections("1285")
    assert karar[("paragraf", "2.2.5")].heading == "LOJİSTİK VE ULAŞTIRMA SEKTÖRÜ"
    assert karar[("paragraf", "2.2.6")].heading == "İNŞAAT VE GAYRİMENKUL SEKTÖRÜ"
    assert "LOJİSTİK VE ULAŞTIRMA SEKTÖRÜ" not in karar[("paragraf", "2.2.4")].content
    assert karar[("paragraf", "2.2.6")].end_char <= text.index("B - ÇEVREYE ÖNEMLİ ZARAR VERMEME")


def test_dotless_outline_headings_far_behind_a_global_paragraph_sequence_are_dividers():
    def paragraphs(numbers):
        return "".join(f"{n}. Rehber paragrafı {n}. " + ("Metin " * 20) + "\n\n" for n in numbers)

    text = (
        paragraphs(range(1, 41))
        + "2.1 Kredi Riski Yönetimi ve Kültürü\n\n2.1.1 Üst Yönetimin Sorumlulukları\n\n"
        + paragraphs(range(41, 61))
    )

    paragraf = [s for s in extract_document_sections("1041", text) if s.section_type == "paragraf"]

    assert [s.section_ref for s in paragraf] == [str(n) for n in range(1, 61)]
    assert not any("Kredi Riski Yönetimi" in s.content or "Üst Yönetimin" in s.content for s in paragraf)


def test_seed_credit_guide_paragraphs_end_before_their_dotless_chapter_headings():
    text, rehber = _seed_sections("1041")
    heading = re.compile(r"^\d{1,3}(?:\.\d{1,3})+\.?\s+[A-ZÇĞİÖŞÜ]", re.MULTILINE)
    carrying = [key[1] for key, s in rehber.items() if key[0] == "paragraf" and heading.search(s.content)]
    assert carrying == []
    assert [key[1] for key in rehber if key[0] == "paragraf"] == [str(n) for n in range(1, 262)]
