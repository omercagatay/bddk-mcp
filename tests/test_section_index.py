"""Tests for legal/regulatory section extraction."""

from __future__ import annotations

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

    ``document_sections_identity_uq`` declares a section's identity to be
    (doc_id, section_type, section_ref, content_hash), so emitting a row per
    occurrence produces rows the schema cannot store.  Keep the first.
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
    assert {s.section_type for s in sections} == {"govde"}
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
    """Rehber 1041/1230/934 corpus shape: the body renumbers from 1 per section,
    so no subset can be trusted and only the classic annex headings survive."""
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
