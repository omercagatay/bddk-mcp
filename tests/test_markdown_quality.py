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


def test_quality_assessment_still_warns_when_formula_reference_has_only_prose():
    result = assess_markdown_quality(
        "Bu durumda öncelikle aşağıdaki formülü kullanarak hesaplama yapılır. "
        "Birinci bankanın ipotek hakkı ve ikinci bankanın alacağı daha sonra değerlendirilir.",
        document_id="x",
    )

    assert result.label == "warning"
    assert "formula_ref_without_latex_or_image" in result.flags


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
