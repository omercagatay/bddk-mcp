"""Tests for Markdown sanitization and quality assessment."""

from __future__ import annotations

from markdown_quality import (
    assess_markdown_quality,
    sanitize_markdown_for_context,
    sanitize_markdown_for_storage,
)


def test_storage_sanitizer_normalizes_control_and_layout_artifacts():
    raw = "Başlık\f\n\n\n****\n** **\n**\nMadde 1\n" + "_" * 80 + "\nA\u00a0B\u200bC"
    out = sanitize_markdown_for_storage(raw)

    assert "\f" not in out
    assert "\u00a0" not in out
    assert "\u200b" not in out
    assert "****" not in out
    assert "** **" not in out
    assert "\n**\n" not in out
    assert "_" * 40 not in out
    assert "\n\n\n" not in out
    assert "Madde 1" in out


def test_storage_sanitizer_preserves_markdown_tables_and_legal_numbering():
    raw = "| Sıra | Değer |\n|---|---|\n| 1 | Madde 9 - Kapsam |\n(1) Birinci fıkra.\n"
    out = sanitize_markdown_for_storage(raw)

    assert "| Sıra | Değer |" in out
    assert "|---|---|" in out
    assert "Madde 9 - Kapsam" in out
    assert "(1) Birinci fıkra." in out


def test_storage_sanitizer_repairs_bddk_circular_header_spacing():
    raw = "Mehmet Ali AKBENBaşkanSayı :24049440-010.06[4/5]-E.109/03/2017Konu:Nakit Kredi  GENELGE2017/1"
    out = sanitize_markdown_for_storage(raw)

    assert "AKBEN\nBaşkan\nSayı: 24049440-010.06[4/5]-E.109/03/2017" in out
    assert "Konu: Nakit Kredi" in out
    assert "GENELGE 2017/1" in out


def test_storage_sanitizer_repairs_common_pdf_spacing_loss_phrases():
    raw = (
        "Esaslar HakkındaYönetmeliğin hükümleri ve Ölçülmesine ilişkinYönetmelik "
        "Regulatory ConsistencyAssessment Programme Kredi Riski StandartYaklaşımı"
    )
    out = sanitize_markdown_for_storage(raw)

    assert "Hakkında Yönetmeliğin" in out
    assert "ilişkin Yönetmelik" in out
    assert "Consistency Assessment Programme" in out
    assert "Standart Yaklaşımı" in out


def test_storage_sanitizer_repairs_ocr_doubled_uppercase_headings():
    raw = "BBANKACILIK DDÜZENLEME VE DDENETLEME KKURUMU"
    out = sanitize_markdown_for_storage(raw)

    assert out == "BANKACILIK DÜZENLEME VE DENETLEME KURUMU"


def test_context_sanitizer_removes_unsafe_embedded_blobs_and_raw_html():
    raw = (
        "<div><table><tr><td>Madde 9</td></tr></table>"
        '<img src="data:image/x-wmf;base64,AAAAcid:12">'
        '<span style="x">metin</span><font>son</font></div>'
    )
    out = sanitize_markdown_for_context(raw)

    assert "data:image/" not in out
    assert "base64" not in out
    assert "cid:" not in out
    for tag in ("<img", "<div", "<table", "<tr", "<td", "<span", "<font"):
        assert tag not in out.lower()
    assert "[removed embedded image/formula artifact]" in out
    assert "Madde 9" in out


def test_storage_sanitizer_removes_cid_image_references():
    raw = "MADDE 1 - Metin öncesi cid:image001.png@01D12345 sonrası hukuki metin."
    out = sanitize_markdown_for_storage(raw)

    assert "cid:" not in out
    assert "MADDE 1" in out
    assert "sonrası hukuki metin" in out


def test_storage_sanitizer_removes_markdown_data_image_blobs():
    raw = (
        "Aşağıdaki denklem uygulanır.\n\n"
        "![](data:image/x-wmf;base64,AQAJAAADcgIAAAIAHAAAAAAABQAAAA==) = maksimum (teminat, 0)\n\n"
        "Bu denklemde teminat tutarı dikkate alınır."
    )
    out = sanitize_markdown_for_storage(raw)

    assert "data:image/" not in out
    assert "base64" not in out
    assert "maksimum (teminat, 0)" in out
    assert "Bu denklemde teminat" in out


def test_context_sanitizer_caps_pathological_line_lengths():
    raw = "A" * 3500
    out = sanitize_markdown_for_context(raw, max_line_length=1000)

    assert max(len(line) for line in out.splitlines()) <= 1000


def test_quality_assessment_marks_known_hard_fail_signals():
    result = assess_markdown_quality(
        "<img src='data:image/x-wmf;base64,AAA'> " + "cid:12 " * 20,
        document_id="mevzuat_21192",
    )

    assert result.label == "fail"
    assert "data_uri_image" in result.flags
    assert "wmf_data_uri" in result.flags
    assert "cid_marker" in result.flags
    assert result.warning


def test_quality_assessment_does_not_fail_cleaned_document_by_legacy_id():
    result = assess_markdown_quality("MADDE 1 - Temiz mevzuat metni.", document_id="mevzuat_21192")

    assert result.label == "clean"


def test_quality_assessment_does_not_fail_cleaned_legacy_ids():
    for document_id in ("903", "905", "907", "1334", "1314", "1313", "1305"):
        result = assess_markdown_quality("MADDE 1 - Temiz mevzuat metni.", document_id=document_id)

        assert result.label == "clean"


def test_quality_assessment_marks_warning_for_formula_reference_without_formula():
    result = assess_markdown_quality("Bu metinde aşağıdaki formül kullanılır.", document_id="x")

    assert result.label == "warning"
    assert "formula_ref_without_latex_or_image" in result.flags


def test_quality_assessment_accepts_inline_extracted_formula():
    result = assess_markdown_quality(
        "Her bir münferit opsiyon için aşağıdaki formül ile bulunur: Gama Etkisi = 1/2 x Gama x (FD)2",
        document_id="x",
    )

    assert result.label == "clean"
    assert "formula_ref_without_latex_or_image" not in result.flags


def test_quality_assessment_accepts_inline_ratio_formula():
    result = assess_markdown_quality(
        "Süreklilik Yüzdesi: MTBF/(MTBF+MTTR) formülü ile bulunacak yüzdesel değeri ifade eder.",
        document_id="x",
    )

    assert result.label == "clean"
    assert "formula_ref_without_latex_or_image" not in result.flags


def test_quality_assessment_accepts_lettered_annex_formulas():
    result = assess_markdown_quality(
        "Ek-3'te yer alan formül uyarınca hesaplanır.\n\n"
        "Yüksek kaliteli likit varlık stokunun hesaplamasında aşağıdaki formüller kullanılır.\n"
        "(a) 2B Kalite Likit Varlıklar için %15 Üst Sınır Aşım Tutarı =\n"
        "Maksimum [Düzeltilmiş 2B Kalite Likit Varlıklar - 15/85 x Düzeltilmiş Birinci Kalite, 0]",
        document_id="x",
    )

    assert result.label == "clean"
    assert "formula_ref_without_latex_or_image" not in result.flags


def test_quality_assessment_still_warns_when_formula_reference_has_only_prose():
    result = assess_markdown_quality(
        "Bu durumda öncelikle aşağıdaki formülü kullanarak hesaplama yapılır. "
        "Birinci bankanın ipotek hakkı ve ikinci bankanın alacağı daha sonra değerlendirilir.",
        document_id="x",
    )

    assert result.label == "warning"
    assert "formula_ref_without_latex_or_image" in result.flags


def test_quality_assessment_ignores_generic_formula_mentions():
    result = assess_markdown_quality(
        "Referans değer, periyodik ya da düzenli olarak bir formül yoluyla belirlenen tutardır.",
        document_id="x",
    )

    assert result.label == "clean"
    assert "formula_ref_without_latex_or_image" not in result.flags


def test_quality_assessment_counts_but_does_not_warn_for_one_duplicate_paragraph():
    paragraph = "Bu paragraf mevzuat içinde iki kez geçen uzun ve anlamlı bir listedir."
    result = assess_markdown_quality(f"{paragraph}\n\n{paragraph}", document_id="x")

    assert result.counts["duplicate_paragraphs"] == 1
    assert result.label == "clean"


def test_quality_assessment_warns_for_repeated_paragraph_blocks():
    paragraph = (
        "Bu paragrafın üç kez tekrarlanması çıkarım kalitesi açısından şüphelidir ve "
        "kaynak metinde tekrar eden bozuk blok bulunduğunu gösterebilir. Aynı uzun "
        "bloğun sayfa sayfa tekrarlanması kullanıcıya sunulan mevzuat metninin "
        "çıkarım kalitesini doğrudan etkiler."
    )
    result = assess_markdown_quality(f"{paragraph}\n\n{paragraph}\n\n{paragraph}", document_id="x")

    assert result.counts["repeated_para_blocks_gt2"] == 1
    assert result.label == "warning"


def test_quality_assessment_does_not_warn_for_short_repeated_boilerplate():
    paragraph = "Atatürk Bulvarı No:191 Kavaklıdere 06680 ANKARA Tel: (312) 455 67 80"
    result = assess_markdown_quality(f"{paragraph}\n\n{paragraph}\n\n{paragraph}", document_id="x")

    assert result.counts["duplicate_paragraphs"] == 2
    assert result.counts["repeated_para_blocks_gt2"] == 0
    assert result.label == "clean"


def test_quality_assessment_warns_for_true_missing_space_camelcase():
    result = assess_markdown_quality("Mehmet Ali AKBENBaşkanSayı ve HakkındaYönetmeliğin metni.", document_id="x")

    assert result.counts["camelcase_concat"] == 2
    assert result.label == "warning"


def test_quality_assessment_ignores_known_mixed_case_terms():
    result = assess_markdown_quality(
        "iOS cihazlar, mTLS bağlantısı, RmD modeli, nSEB koşulu, 250kW değeri ve HashCalc aracı.",
        document_id="x",
    )

    assert result.counts["camelcase_concat"] == 0
    assert result.label == "clean"


def test_quality_assessment_ignores_camelcase_inside_urls():
    result = assess_markdown_quality(
        "Kaynak http://www.bddk.org.tr/WebSitesi/turkce/Mevzuat adresinde yayımlandı.",
        document_id="x",
    )

    assert result.counts["camelcase_concat"] == 0
    assert result.label == "clean"


def test_quality_assessment_ignores_camelcase_inside_xml_tags():
    result = assess_markdown_quality(
        "<GuncellemeTarihi>2010-12-22T12:45:00</GuncellemeTarihi> <HizmetAdi>Bireysel Kredi</HizmetAdi>",
        document_id="x",
    )

    assert result.counts["camelcase_concat"] == 0
    assert result.label == "clean"
