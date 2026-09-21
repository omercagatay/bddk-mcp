"""Source-reviewed patches must not silently alter their signed baseline or prose."""

import gzip
import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from bddk_mcp.quality.markdown_quality import assess_markdown_quality

ROOT = Path(__file__).resolve().parents[1]
BASELINE_BYTES = gzip.decompress(
    (ROOT / "docs/evidence/document-repairs/signed-baseline-documents.json.gz").read_bytes()
)
BASELINE_DOCUMENTS = json.loads(BASELINE_BYTES)
pytestmark = pytest.mark.usefixtures("historical_quality_registry")


def reconstruct_candidate(document_id):
    repair = json.loads((ROOT / f"docs/evidence/document-repairs/{document_id}.json").read_text())
    original = next(doc["markdown_content"] for doc in BASELINE_DOCUMENTS if doc["document_id"] == document_id)
    assert hashlib.sha256(original.encode()).hexdigest() == repair["original_markdown_sha256"]
    assert repair["status"] == "source_reviewed_candidate_awaiting_owner_approval"

    end = 0
    pieces = []
    for patch in repair["replacements"]:
        assert end <= patch["start"] <= patch["end"] <= len(original)
        assert original[patch["start"] : patch["end"]] == patch["old"]
        pieces.extend((original[end : patch["start"]], patch["new"]))
        end = patch["end"]
    pieces.append(original[end:])
    candidate = "".join(pieces)
    assert hashlib.sha256(candidate.encode()).hexdigest() == repair["candidate_markdown_sha256"]
    return repair, original, candidate


@pytest.mark.parametrize(
    "document_id",
    ["903", "905", "907", "1043", "1045", "1305", "1313", "1314", "1334", "mevzuat_16290", "mevzuat_21192"],
)
def test_candidate_keeps_registry_protection(document_id):
    _, _, candidate = reconstruct_candidate(document_id)
    assert assess_markdown_quality(candidate, document_id=document_id).label == "fail"


def test_signed_repair_corpus_matches_all_reviewed_candidates(monkeypatch):
    from bddk_mcp.quality import markdown_quality

    assert (
        hashlib.sha256(BASELINE_BYTES).hexdigest() == "4a469fc11bd9848d2acd540296d794126ab9c506dc68d929fb72f580ebe5dba6"
    )
    registry = markdown_quality.load_quality_failure_registry()
    assert registry == {}
    monkeypatch.setattr(markdown_quality, "_QUALITY_FAILURES", registry)
    current = json.loads((ROOT / "seed_data/documents.json").read_text())
    assert len(current) == 318
    repaired = 0
    for doc in current:
        did = doc["document_id"]
        patch_path = ROOT / f"docs/evidence/document-repairs/{did}.json"
        if patch_path.exists():
            _, _, expected = reconstruct_candidate(did)
            native = ROOT / f"docs/evidence/document-repairs/{did}-native.json"
            if native.exists():
                expected = json.loads(native.read_text())["replacements"][0]["new"]
            assert doc["markdown_content"] == expected
            repaired += 1
        assert hashlib.sha256(doc["markdown_content"].encode()).hexdigest() == doc["content_hash"]
        assert assess_markdown_quality(doc["markdown_content"], did).label == "clean"
    assert repaired == 50


def test_document_903_source_reviewed_candidate():
    repair, _, candidate = reconstruct_candidate("903")
    assert len(repair["replacements"]) == repair["display_formulas"] == 11
    for patch in repair["replacements"]:
        assert any(unicodedata.category(char) == "Co" for char in patch["old"])
        assert patch["new"].startswith("$$\n") and patch["new"].endswith("\n$$")
    assert not any(unicodedata.category(char) == "Co" for char in candidate)
    assert candidate.count("$$") == 22
    # Preserve the source's inconsistencies, not a silently corrected calculation.
    assert candidate.count(r"\sum_{1}^{2}") == 2
    assert r"= -3{,}5\,TL" in candidate
    assert "40 TL tutarında ikinci derece ipoteği olduğundan" in candidate


def test_document_907_reading_order_and_literal_xml():
    repair, original, candidate = reconstruct_candidate("907")
    assert original.count("Đ") == repair["dotted_i_repairs"] == 21
    assert "Đ" not in candidate and "()" not in candidate
    assert "- Konut Kredileri\n- Taşıt Kredileri\n- İhtiyaç Kredileri" in candidate
    assert "tahsil edilecek azami tutar ve oranlar açıklanmalı" in candidate
    assert "01.07.2011 tarihinden itibaren" in candidate
    assert "\ntarihinden\n" not in candidate
    blocks = re.findall(r"```xml\n(.*?)\n```", candidate, re.DOTALL)
    assert len(blocks) == repair["xml_code_blocks"] == 3
    assert all("Atatürk Bulvarı" not in block for block in blocks)
    example = "\n".join(blocks[:2])
    assert ET.fromstring(example).tag == "Banka"
    assert ET.fromstring(blocks[2]).tag == "{http://www.w3.org/2001/XMLSchema}schema"
    # Framing may change; XML/XSD semantics (including source mistakes) may not.
    normalized = original.replace("Đ", "İ")
    footer = (
        "Atatürk Bulvarı No:191 Kavaklıdere 06680 ANKARA\n"
        "Tel.: (312) 455 65 80    Faks: (312) 424 17 47\n"
        "İnternet adresi: www.bddk.org.tr"
    )
    source_xml = normalized.split("Örnek XML Dokümanı\n\n", 1)[1].split("</Banka>", 1)[0] + "</Banka>"
    source_xml = source_xml.replace(footer, "")
    source_xsd = normalized.split("XSD Dokümanı\n\n", 1)[1].split("</xsd:schema>", 1)[0] + "</xsd:schema>"
    assert ET.canonicalize(source_xml, strip_text=True) == ET.canonicalize(example, strip_text=True)
    assert ET.canonicalize(source_xsd, strip_text=True) == ET.canonicalize(blocks[2], strip_text=True)
    assert "`<deger>`" in candidate and '<xsd:field xpath="@Tur"/>' in blocks[2]
    assert candidate.count("Atatürk Bulvarı") == original.count("Atatürk Bulvarı")


def test_document_1314_formulas_tables_and_page_coverage():
    repair, _, candidate = reconstruct_candidate("1314")
    progress = json.loads((ROOT / "docs/evidence/document-repairs/1314-review-progress.json").read_text())
    assert repair["source_pages_visually_reviewed"] == progress["pages_visually_reviewed"] == list(range(1, 27))
    assert progress["assembled_markdown_sha256"] == repair["candidate_markdown_sha256"]
    assert candidate.count("$$") == 2 * repair["display_formulas"] == 40
    assert re.findall(r"\*\*MADDE (\d+)", candidate) == [str(n) for n in range(1, 38)]
    assert re.findall(r"\*\*Tablo (\d+):", candidate) == [str(n) for n in range(1, 15)]
    assert not any(unicodedata.category(char) == "Co" for char in candidate)
    quality = assess_markdown_quality(candidate)
    assert quality.counts["malformed_table_rows"] == quality.counts["raw_html_tag"] == 0
    # Keep source inconsistencies; do not silently normalize equations or draft status.
    for text in (r"\frac{1}{a}", r"\gamma_{bc} s_b s_c", "$KDA_S$", "ihtiyarlarına", "dışından", "değişiklerle"):
        assert text in candidate
    assert r"\*\*/\*\*/2026" in candidate
    # OCR wrongly assigned these YYD rows to YRD; check the cross-page table repair.
    table4 = candidate.split("**Tablo 4:", 1)[1].split("(2) Referans kredi", 1)[0]
    for row in range(1, 8):
        assert f"| {row} | YYD |" in table4
    for row in range(8, 15):
        assert f"| {row} | YRD veya DNB |" in table4
    assert "| 8 | YRD veya DNB |" in table4
    assert "faiz oranı veya risk" not in candidate and "kur veya risk" not in candidate
    assert "vega risk faktörleri" in candidate


def test_document_905_reviewed_stages_and_source_coverage():
    progress = json.loads((ROOT / "docs/evidence/document-repairs/905-review-progress.json").read_text())
    assert progress["status"] == "source_reviewed_candidate_awaiting_owner_approval"
    original = next(doc["markdown_content"] for doc in BASELINE_DOCUMENTS if doc["document_id"] == "905")
    assert hashlib.sha256(original.encode()).hexdigest() == progress["original_markdown_sha256"]
    mapping = {item["old"]: item["new"] for item in progress["character_repairs"]}
    assert mapping == {"Ģ": "ş", "Ġ": "İ", "ġ": "Ş", "\uf0b7": "•"}
    assert all(original.count(item["old"]) == item["count"] for item in progress["character_repairs"])
    candidate = original.translate(str.maketrans(mapping))
    assert hashlib.sha256(candidate.encode()).hexdigest() == progress["partial_markdown_sha256"]
    assert len(candidate) == len(original)
    coverage = json.loads((ROOT / "docs/evidence/document-repairs/905-textual-coverage.json").read_text())
    assert coverage["source_pdf_sha256"] == progress["source_pdf_sha256"]
    assert coverage["baseline_encoding_only_sha256"] == progress["partial_markdown_sha256"]
    assert [page["physical_page"] for page in coverage["pages"]] == list(range(1, 43))
    assert sum(page["tokens_equal"] for page in coverage["pages"]) == 35
    end = 0
    for page in coverage["pages"]:
        assert page["baseline_start"] == end
        end = page["baseline_end"]
        tokens = re.findall(r"\w+|[^\w\s]", candidate[page["baseline_start"] : end])
        assert hashlib.sha256("\n".join(tokens).encode()).hexdigest() == page["baseline_tokens_sha256"]
        if not page["requires_visual_table_comparison"]:
            assert page["tokens_equal"]
            assert page["baseline_tokens_sha256"] == page["source_tokens_sha256"]
    assert not candidate[end:].strip()
    assert assess_markdown_quality(candidate, document_id="905").label == "fail"
    table = json.loads((ROOT / "docs/evidence/document-repairs/905-bank-table.json").read_text())
    assert table["status"] == "partial_table_repair_not_release_ready"
    assert table["source_pages"] == [20, 21, 22]
    assert table["input_markdown_sha256"] == progress["partial_markdown_sha256"]
    (patch,) = table["replacements"]
    assert candidate[patch["start"] : patch["end"]] == patch["old"]
    candidate = candidate[: patch["start"]] + patch["new"] + candidate[patch["end"] :]
    assert hashlib.sha256(candidate.encode()).hexdigest() == table["output_markdown_sha256"]
    assert patch["new"].count("| Türkiye’de bulunan bankalar |") == 2
    assert patch["new"].count("| Yurtdışında Bulunan Bankalar |") == 3
    assert "Derecelendirme notu varsa⁷" in patch["new"]
    assert "(Fıkra 28 uyarınca) %50 den büyük" in patch["new"]
    assert "(Fıkra 30 uyarınca) %20 den büyük" in patch["new"]
    assert assess_markdown_quality(candidate).counts["malformed_table_rows"] == 0
    assert assess_markdown_quality(candidate, document_id="905").label == "fail"
    numeric = json.loads((ROOT / "docs/evidence/document-repairs/905-numeric-tables.json").read_text())
    assert numeric["status"] == "partial_numerical_repair_not_release_ready"
    assert numeric["input_markdown_sha256"] == table["output_markdown_sha256"]
    patches = numeric["replacements"]
    assert all(a["end"] <= b["start"] for a, b in zip(patches, patches[1:], strict=False))
    for patch in reversed(patches):
        assert candidate[patch["start"] : patch["end"]] == patch["old"]
        candidate = candidate[: patch["start"]] + patch["new"] + candidate[patch["end"] :]
    assert hashlib.sha256(candidate.encode()).hexdigest() == numeric["output_markdown_sha256"]
    assert "| 40.000.000 | 1,37 | 0,75 | 1,03 | 31.111.111,11 |" in candidate
    assert "411.827.746,45" in candidate
    assert "1,95¹¹" in candidate and "1,46¹³" in candidate and "51.118.421,05¹⁴" in candidate
    assert "%350⁸" in candidate and "%1.250⁹" in candidate
    assert "%3508" not in candidate and "%1.2509" not in candidate
    assert "| | 8.000.000 | %20 | 8.000.000*%20= | 1.600.000 |" in candidate
    assert assess_markdown_quality(candidate).counts["malformed_table_rows"] == 0
    assert assess_markdown_quality(candidate, document_id="905").label == "fail"
    repair, _, reviewed = reconstruct_candidate("905")
    assert candidate == reviewed
    assert sorted(
        repair["source_pages_individually_inspected"] + repair["source_pages_layout_contact_sheet_reviewed"]
    ) == list(range(1, 43))
    assert len(re.findall(r"^\| ---", reviewed, re.MULTILINE)) == 6
    assert "Türkiye Sermaye Piyasası Aracı Kuruluşları Birliği" in reviewed


def test_document_1313_formula_repair_preserves_source_and_registry():
    evidence = json.loads((ROOT / "docs/evidence/document-repairs/1313-review-progress.json").read_text())
    assert evidence["status"] == "source_reviewed_candidate_awaiting_owner_approval"
    original = next(d["markdown_content"] for d in BASELINE_DOCUMENTS if d["document_id"] == "1313")
    assert hashlib.sha256(original.encode()).hexdigest() == evidence["original_markdown_sha256"]
    pieces, end = [], 0
    for patch in evidence["replacements"]:
        assert end <= patch["start"] <= patch["end"] <= len(original)
        assert original[patch["start"] : patch["end"]] == patch["old"]
        pieces.extend((original[end : patch["start"]], patch["new"]))
        end = patch["end"]
    candidate = "".join([*pieces, original[end:]])
    assert hashlib.sha256(candidate.encode()).hexdigest() == evidence["partial_markdown_sha256"]
    assert candidate.count("$$") == 2 * evidence["display_formulas"] == 88
    assert evidence["tables"] == 4
    assert evidence["reading_order_repairs"] == 12
    for patch in evidence["replacements"]:
        if patch["kind"] == "reading_order":
            assert Counter(re.findall(r"\w+|[^\w\s]", patch["old"])) == Counter(
                re.findall(r"\w+|[^\w\s]", patch["new"])
            )
    repair, _, reviewed = reconstruct_candidate("1313")
    assert reviewed == candidate and repair == evidence
    coverage = json.loads((ROOT / "docs/evidence/document-repairs/1313-textual-coverage.json").read_text())
    assert coverage["status"] == "token_differences_accounted_for"
    assert coverage["unresolved_differences"] == []
    assert coverage["source_pages"] == 54
    assert (
        coverage["review_progress_sha256"]
        == hashlib.sha256((ROOT / "docs/evidence/document-repairs/1313-review-progress.json").read_bytes()).hexdigest()
    )
    assert all(item["reviewed_patch_indices"] for item in coverage["differences"])
    assert sorted(evidence["pages_individually_reviewed"] + evidence["pages_layout_contact_sheet_reviewed"]) == list(
        range(1, 55)
    )
    assert re.findall(r"\bMADDE\s+(\d+)", candidate) == [str(n) for n in range(1, 70)]
    assert "GENEL GEREKÇE" in candidate and "**/**/2026" in candidate
    assert r"\frac{\sum_t T_t}{\sum_s K_s}" in candidate
    assert r"$\mathrm{BRT}_k$ : üncü maddesinde" in candidate
    assert r"\sum_a \text{Eklenti}^{a}" in candidate
    assert r"/ \sigma_i * \mathrm{VO}_i^{1/2}" in candidate
    assert candidate.count(r"\text{ф}") == 4
    assert "1 yıl vadeli 3 aylık Eurodollar future | 1 yıl | 1 yıl | 1.25 yıl" in candidate
    assert r"^{0{,}5}" in candidate and r"^{0.5}" in candidate
    assert not any(unicodedata.category(char) == "Co" for char in candidate)
    assert r"\sum_{k=1}^{\min(1\text{ yıl};\text{vade})}" in candidate
    assert r"\mathrm{EBRT}_{t_k} * \Delta t_k" in candidate
    assert r"\mathrm{NG}\ \text{€}\ \mathrm{MS}" in candidate
    assert r"T_K^{J\,(E)} = D_k^{(E)}" in candidate
    assert "| Döviz kuru | | % 4 | | % 15 |" in candidate
    assert "| | Tarım | % 18 | % 40 | % 70 |" in candidate
    assert assess_markdown_quality(candidate).counts["malformed_table_rows"] == 0
    assert assess_markdown_quality(candidate, document_id="1313").label == "fail"


def test_document_1045_catalog_inventory_keeps_leading_zeros_and_review_gate():
    repair, original, candidate = reconstruct_candidate("1045")
    receipt = json.loads((ROOT / "docs/evidence/document-repairs/1045-review-progress.json").read_text())
    explanation = json.loads((ROOT / "docs/evidence/document-repairs/1045-explanation-progress.json").read_text())
    start = re.search(r"I\.\s+Tekdüzen Hesap Planı", original).end()
    end = re.search(r"II\.\s+Tekdüzen Hesap Planı İzahnamesi", original).start()
    codes = re.findall(r"(?m)^\s*(\d{1,7})(?=\s|$)", original[start:end])
    catalog_md = candidate.split("II.", 1)[0]
    table_codes = re.findall(r"(?m)^(?:## |\| )(\d{1,7}) ", catalog_md)
    assert codes == table_codes
    assert len(codes) == len(set(codes)) == repair["account_code_count"] == 2708
    assert codes[:3] == ["0", "010", "011"]
    assert hashlib.sha256("\n".join(codes).encode()).hexdigest() == receipt["ordered_code_inventory_sha256"]
    assert repair["continuation_line_count"] == receipt["continuation_contexts_visually_reviewed"] == 42
    assert repair["uncoded_heading_count"] == receipt["uncoded_heading_contexts_visually_reviewed"] == 8
    assert repair["geometry_verified_explanation_headings"] == explanation["geometry_verified_heading_count"] == 200
    assert candidate.count("## 0 DÖNEN DEĞERLER") == 2
    assert "| 944 |" in candidate and "T.P." in candidate.split("| 944 |", 1)[1].split("\n", 1)[0]
    assert "CAYILABİLİR TAAHHÜTLER" in candidate
    quality = assess_markdown_quality(candidate)
    assert quality.counts["repeated_para_blocks_gt2"] == 0
    assert quality.counts["duplicate_paragraphs"] == 10
    assert assess_markdown_quality(original, document_id="1045").label == "fail"
    assert receipt["status"] == explanation["status"] == "source_reviewed_candidate_awaiting_owner_approval"
    assert repair["source_pages_visually_reviewed"] == list(range(1, 77))


def test_document_1043_catalog_inventory_keeps_leading_zeros_and_review_gate():
    repair, original, candidate = reconstruct_candidate("1043")
    receipt = json.loads((ROOT / "docs/evidence/document-repairs/1043-review-progress.json").read_text())
    explanation = json.loads((ROOT / "docs/evidence/document-repairs/1043-explanation-progress.json").read_text())
    start = re.search(r"I\.\s*\n\s*Tekdüzen Hesap Planı", original).end()
    end = re.search(r"II\.\s*\n\s*Tekdüzen hesap planı izahnamesi", original).start()
    orig_codes = re.findall(r"(?m)^\s*(\d{1,8})(?=\s|$)", original[start:end])
    catalog_md = candidate.split("II.", 1)[0]
    table_codes = re.findall(r"(?m)^(?:## |\| )(\d{1,8}) ", catalog_md)
    assert orig_codes[:4] == ["0", "010", "011", "012"]
    assert table_codes[:4] == ["0", "010", "011", "012"]
    assert len(table_codes) == len(set(table_codes)) == repair["account_code_count"] == 6776
    assert hashlib.sha256("\n".join(table_codes).encode()).hexdigest() == receipt["ordered_code_inventory_sha256"]
    assert repair["uncoded_heading_count"] == 78
    assert repair["continuation_line_count"] == 64
    assert repair["geometry_verified_explanation_headings"] == explanation["geometry_verified_heading_count"] == 281
    assert candidate.count("## 0 DÖNEN DEĞERLER") == 2
    assert "ALICI (LEHDAR) TARAF=" in candidate
    quality = assess_markdown_quality(candidate)
    assert quality.counts["repeated_para_blocks_gt2"] == 0
    assert quality.counts["duplicate_paragraphs"] == 10
    assert assess_markdown_quality(original, document_id="1043").label == "fail"
    assert receipt["status"] == explanation["status"] == "source_reviewed_candidate_awaiting_owner_approval"
    assert repair["source_pages_visually_reviewed"] == list(range(1, 178))


def test_document_1334_catalog_inventory_keeps_review_gate():
    repair, original, candidate = reconstruct_candidate("1334")
    receipt = json.loads((ROOT / "docs/evidence/document-repairs/1334-review-progress.json").read_text())
    explanation = json.loads((ROOT / "docs/evidence/document-repairs/1334-explanation-progress.json").read_text())
    catalog_md = candidate.split("II.", 1)[0]
    table_codes = re.findall(r"(?m)^(?:## |\| )(\d{1,8}) ", catalog_md)
    assert table_codes[:4] == ["0", "010", "011", "012"]
    assert len(table_codes) == repair["account_code_count"] == 9381
    assert len(set(table_codes)) == 9380
    assert table_codes.count("138114") == 2
    assert hashlib.sha256("\n".join(table_codes).encode()).hexdigest() == receipt["ordered_code_inventory_sha256"]
    assert repair["uncoded_heading_count"] == 51
    assert repair["continuation_line_count"] == 150
    assert repair["geometry_verified_explanation_headings"] == explanation["geometry_verified_heading_count"] == 273
    assert candidate.count("## 0 DÖNEN DEĞERLER") == 2
    assert "SATICI (KEŞİDECİ) TARAF" in candidate
    quality = assess_markdown_quality(candidate)
    assert quality.counts["repeated_para_blocks_gt2"] == 0
    assert assess_markdown_quality(original, document_id="1334").label == "fail"
    assert receipt["status"] == explanation["status"] == "source_reviewed_candidate_awaiting_owner_approval"
    assert repair["source_pages_visually_reviewed"] == list(range(1, 299))


def test_document_16290_table_draft_keeps_review_gate():
    repair, original, candidate = reconstruct_candidate("mevzuat_16290")
    assert original.count("|") == 0
    assert repair["table_count"] == 152
    assert candidate.count("| ---") >= 152
    assert assess_markdown_quality(original).counts["repeated_para_blocks_gt2"] == 1
    assert assess_markdown_quality(candidate).counts["repeated_para_blocks_gt2"] == 0
    assert "MADDE 29" in candidate
    assert assess_markdown_quality(original, document_id="mevzuat_16290").label == "fail"


def test_document_21192_annex_formulas_are_not_tables_and_keep_review_gate():
    repair, original, candidate = reconstruct_candidate("mevzuat_21192")
    assert repair["main_pdf_page_count"] == 36
    assert repair["stored_total_pages"] == 70
    assert repair["display_formulas"] == 2
    assert assess_markdown_quality(original).counts["malformed_table_rows"] == 2
    assert assess_markdown_quality(candidate).counts["malformed_table_rows"] == 0
    assert candidate.count("$$") == 4
    assert r"\lvert Dj1\rvert" in candidate
    assert "DOVFj(DK)" in candidate
    assert original.count("Dj = | Dj1 |") == 1
    assert assess_markdown_quality(original, document_id="mevzuat_21192").label == "fail"


def test_historical_publication_handoff_preserves_candidate_state():
    handoff = json.loads((ROOT / "docs/evidence/document-repairs/publication-handoff.json").read_text())
    assert handoff["record_kind"] == "historical_pre_signature_candidate_handoff"
    assert handoff["superseded_by"] == "local-release-v5.json"
    assert handoff["signed_seed_untouched"] is True
    assert handoff["failure_registry_untouched"] is True
    assert handoff["private_signing_key_present"] is False
    assert handoff["live_local_db_markdown_matches_signed_seed"] is True
    assert handoff["live_local_db_serves_reviewed_candidates"] is False
    assert handoff["documents_applied"] == 50
    for row in handoff["documents"]:
        repair = json.loads((ROOT / f"docs/evidence/document-repairs/{row['document_id']}.json").read_text())
        assert row["candidate_markdown_sha256"] == repair["candidate_markdown_sha256"]
        assert repair["status"] == "source_reviewed_candidate_awaiting_owner_approval"


def test_document_1312_formulas_are_restored_from_source_pdf():
    repair, original, candidate = reconstruct_candidate("1312")
    assert repair["display_formulas"] == 3
    assert assess_markdown_quality(original).counts["malformed_table_rows"] == 3
    assert assess_markdown_quality(candidate).counts["malformed_table_rows"] == 0
    assert r"\sum_{i} K_{i}" in candidate
    assert r"H = \sum_{i} a_{i} H_{i}" in candidate
    assert r"H = H_{10}" in candidate
    assert "K* = maksimum{0, [(ΣK – ΣT) + (RMD)]}" in candidate
    assert "()()" not in candidate
    assert assess_markdown_quality(candidate, document_id="1312").label == "clean"


F0B7_BULLET_DOCUMENTS = [
    "786",
    "808",
    "915",
    "944",
    "946",
    "947",
    "950",
    "951",
    "952",
    "954",
    "955",
    "1030",
    "1085",
    "1086",
    "1099",
    "1135",
    "1155",
    "1157",
    "1171",
    "1175",
    "1218",
    "1311",
]


WINGDINGS_DOCUMENTS = ["787", "789", "793", "798", "800", "804", "806", "934", "948", "1038", "1131", "1191"]


@pytest.mark.parametrize("document_id", F0B7_BULLET_DOCUMENTS)
def test_f0b7_private_use_bullets_are_replaced(document_id):
    repair, original, candidate = reconstruct_candidate(document_id)
    assert repair["f0b7_bullet_repairs"] == original.count("\uf0b7") > 0
    assert "\uf0b7" not in candidate
    assert not any(unicodedata.category(char) == "Co" for char in candidate)


@pytest.mark.parametrize("document_id", WINGDINGS_DOCUMENTS)
def test_reviewed_wingdings_private_use_glyphs_are_replaced(document_id):
    repair, original, candidate = reconstruct_candidate(document_id)
    assert repair["glyph_repairs"] > 0
    assert not any(unicodedata.category(char) == "Co" for char in candidate)
    assert any(unicodedata.category(char) == "Co" for char in original)


def test_mevzuat_20029_abs_value_latex_is_not_a_table_row():
    repair, original, candidate = reconstruct_candidate("mevzuat_20029")
    assert assess_markdown_quality(original).counts["malformed_table_rows"] == 1
    assert assess_markdown_quality(candidate).counts["malformed_table_rows"] == 0
    assert r"\lvert K_{MK}\rvert" in candidate


def test_mevzuat_21425_junk_grid_is_unwrapped():
    _, original, candidate = reconstruct_candidate("mevzuat_21425")
    assert assess_markdown_quality(original).counts["excessive_pipe_density"] > 0
    assert assess_markdown_quality(candidate).counts["excessive_pipe_density"] == 0
    assert assess_markdown_quality(candidate).counts["malformed_table_rows"] == 0
    assert "**MADDE 1 –**" in candidate
    assert "| 4. Grup (Boş) | %3 |" in candidate


def test_rg_32202_header_pipe_is_not_a_table_row():
    _, original, candidate = reconstruct_candidate("rg_32202_20230526_6")
    assert assess_markdown_quality(original).counts["malformed_table_rows"] == 1
    assert assess_markdown_quality(candidate).counts["malformed_table_rows"] == 0
    assert "26.05.2023 — Resmi Gazete Sayısı: 32202" in candidate


def test_document_1296_shock_tables_are_restored_from_source_pdf():
    repair, original, candidate = reconstruct_candidate("1296")
    assert repair["shock_tables"] == 4
    assert original.count("()()") == 40
    assert "()()" not in candidate
    assert r"\bar{R}_{paralel}" in candidate
    assert "| $\\pm \\bar{R}_{uzun}$ | 300 baz puan |" in candidate
    assert "Para Birimi p" in candidate
    assert assess_markdown_quality(candidate, document_id="1296").label == "clean"


@pytest.mark.parametrize("document_id,table_count", [("mevzuat_16290", 112), ("mevzuat_21192", 23)])
def test_native_source_repairs_preserve_cells_spans_and_protection(document_id, table_count):
    evidence_root = ROOT / "docs/evidence/document-repairs"
    repair = json.loads((evidence_root / f"{document_id}-native.json").read_text())
    _, _, previous = reconstruct_candidate(document_id)
    (patch,) = repair["replacements"]
    assert patch["start"] == 0 and patch["end"] == len(previous)
    assert patch["old"] == previous
    candidate = patch["new"]
    assert hashlib.sha256(candidate.encode()).hexdigest() == repair["candidate_markdown_sha256"]
    assert assess_markdown_quality(candidate).label == "clean"
    assert (
        assess_markdown_quality(candidate, document_id).label == "fail"
    )  # Historical registry still overrides heuristics.

    for source in repair["source_inputs"].values():
        raw = gzip.decompress((ROOT / source["retained_path"]).read_bytes())
        assert len(raw) == source["bytes"]
        assert hashlib.sha256(raw).hexdigest() == source["sha256"]
    proofs = json.loads(gzip.decompress((evidence_root / repair["coverage_file"]).read_bytes()))
    if isinstance(proofs, dict):
        proofs = [proofs]
    grids = []
    for proof in proofs:
        assert proof["unconsumed_source_text_nodes"] == 0
        source_name = Path(proof["source_path"]).name
        key = "16290.doc" if document_id == "mevzuat_16290" else f"html/{source_name}"
        source = repair["source_inputs"][key]
        raw = gzip.decompress((ROOT / source["retained_path"]).read_bytes())
        soup = BeautifulSoup(raw.decode("utf-16" if key == "16290.doc" else "utf-8"), "html.parser")
        native_tables = soup.find_all("table")
        assert len(native_tables) == len(proof["tables"])
        for native, table in zip(native_tables, proof["tables"], strict=True):
            assert not native.find("img")
            source_cells = [c for row in native.find_all("tr") for c in row.find_all(["td", "th"], recursive=False)]
            assert len(source_cells) == len(table["cells"])
            slots = {}
            for cell, stored in zip(source_cells, table["cells"], strict=True):
                assert int(cell.get("rowspan", 1)) == stored["rowspan"]
                assert int(cell.get("colspan", 1)) == stored["colspan"]
                plain = re.sub(r"\$[_^]\{([^{}]*)\}\$", r"\1", stored["text"])
                # Inline sub/sup formatting may move whitespace, never source characters.
                assert re.sub(r"\s+", "", cell.get_text()) == re.sub(r"\s+", "", plain)
                for r in range(stored["row"], stored["row"] + stored["rowspan"]):
                    for c in range(stored["column"], stored["column"] + stored["colspan"]):
                        assert (r, c) not in slots
                        slots[r, c] = stored["text"]
            expected = [[slots.get((r, c), "") for c in range(table["columns"])] for r in range(table["rows"])]
            assert expected == table["grid"]
            grids.append([[re.sub(r"(?<=\})\$\$(?=[_^])", "", cell) for cell in row] for row in expected])
    blocks = re.findall(r"(?m)^\|[^\n]*(?:\n\|[^\n]*)*", candidate)
    assert len(blocks) == len(grids) == table_count
    for block, grid in zip(blocks, grids, strict=True):
        lines = block.splitlines()
        actual = [
            [c.strip().replace(r"\|", "|") for c in re.split(r"(?<!\\)\|", line[1:-1])]
            for line in [lines[0], *lines[2:]]
        ]
        assert actual == grid
    if document_id == "mevzuat_21192":
        assert sum(len(p["image_replacements"]) for p in proofs) == 33
        assert sum(len(p["non_displayed_quote_fields"]) for p in proofs) == 2
        assert r"BRT_{i+1}\cdot D_{i+1}" in candidate
        assert r"\sum_{k=1}^{\min(1\text{ yıl};\text{vade})}" in candidate
        assert r"TF_i+\mu\cdot TF_i^t" in candidate
        assert "EK-1" in candidate and "EK-2" in candidate and "EK-3" in candidate and "EK-4" in candidate
        assert not any(ord(c) < 32 and c not in "\t\n\r" for c in candidate)
        assert not re.search(r"!\[.*?\]\(|<img|data:image", candidate)


def test_document_1305_missing_image_equations_are_restored_without_publication():
    evidence = json.loads((ROOT / "docs/evidence/document-repairs/1305-review-progress.json").read_text())
    assert evidence["status"] == "partial_source_review_not_release_ready"
    original = next(d["markdown_content"] for d in BASELINE_DOCUMENTS if d["document_id"] == "1305")
    assert hashlib.sha256(original.encode()).hexdigest() == evidence["original_markdown_sha256"]
    pieces, end = [], 0
    for patch in evidence["replacements"]:
        assert end <= patch["start"] <= patch["end"] <= len(original)
        assert original[patch["start"] : patch["end"]] == patch["old"]
        pieces.extend((original[end : patch["start"]], patch["new"]))
        end = patch["end"]
    candidate = "".join([*pieces, original[end:]])
    assert hashlib.sha256(candidate.encode()).hexdigest() == evidence["partial_markdown_sha256"]
    assert sum(p["kind"] == "restored_image_formula" for p in evidence["replacements"]) == 11
    assert candidate.count("$$") == 2 * evidence["display_formulas"] == 34
    assert evidence["tables"] == 5
    assert "| Diğer İhtisas Kredileri | <2,5 | %50 | %70 | %115 | %250 | %0 |" in candidate
    assert "| Diğer İhtisas Kredileri | ≥2,5 | %0,4 | %0,8 | %2,8 | %8 | %50 |" in candidate
    assert "| Nitelikli Rotatif Perakende Alacaklar | %50 | - |" in candidate
    assert r"(V-2{,}5)\times b" in candidate
    assert r"\ln(\mathrm{TO})" in candidate and r"\delta" not in candidate
    assert r"0{,}11852-0{,}05478" in candidate
    assert r"0{,}30\times" in candidate and r"0{,}16\times" in candidate
    assert r"\min\{\max\{(C/10),S\};C\}" in candidate
    assert r"\sum_t t\times\mathrm{NA}_t" in candidate
    assert "(THKS)" in candidate and r"\mathrm{THK}_T" in candidate
    assert "| Diğer fiziksel teminat | %25 | %40 |" in candidate
    assert assess_markdown_quality(candidate, document_id="1305").label == "fail"
    annex = json.loads((ROOT / "docs/evidence/document-repairs/1305-annex-project.json").read_text())
    assert annex["status"] == "partial_annex_source_review_not_release_ready"
    assert annex["source_pages"] == list(range(46, 54))
    assert annex["input_markdown_sha256"] == evidence["partial_markdown_sha256"]
    (patch,) = annex["replacements"]
    assert candidate[patch["start"] : patch["end"]] == patch["old"]
    candidate = candidate[: patch["start"]] + patch["new"] + candidate[patch["end"] :]
    assert hashlib.sha256(candidate.encode()).hexdigest() == annex["output_markdown_sha256"]
    source_tables = [t for t in annex["source_tables"] if len(t["rows"][0]) == 5]
    assert len(source_tables) == 8
    rows = annex["project_rows"]
    assert len(rows) == 36
    for c in range(5):
        before = re.sub(r"\s", "", "".join((row[c] or "") for t in source_tables for row in t["rows"][1:]))
        after = re.sub(r"\s", "", "".join(row[c] for row in rows))
        assert before == after
        assert hashlib.sha256(before.encode()).hexdigest() == annex["project_column_nonwhitespace_sha256"][c]
    activity = next(row for row in rows if row[0].startswith("Faaliyet Riski"))
    assert activity[1].endswith("için ayrılan karşılık hesapları.")
    operator = next(row for row in rows if row[0].startswith("İşletmecinin uzmanlığı"))
    assert operator[1].startswith("Çok güçlü")
    stress = next(row for row in rows if row[0] == "Stres analizi")
    assert "yükümlülüklerini yerine getirebilir" in stress[2]
    remaining = json.loads((ROOT / "docs/evidence/document-repairs/1305-annex-remaining.json").read_text())
    assert remaining["input_markdown_sha256"] == annex["output_markdown_sha256"]
    assert remaining["pages_visually_reviewed"] == list(range(54, 68))
    (patch,) = remaining["replacements"]
    assert candidate[patch["start"] : patch["end"]] == patch["old"]
    candidate = candidate[: patch["start"]] + patch["new"] + candidate[patch["end"] :]
    assert hashlib.sha256(candidate.encode()).hexdigest() == remaining["output_markdown_sha256"]
    assert [len(group["rows"]) for group in remaining["groups"]] == [19, 25, 15]
    for group in remaining["groups"]:
        for c in range(5):
            before = re.sub(r"\s", "", "".join((row[c] or "") for t in group["source_tables"] for row in t["rows"][1:]))
            after = re.sub(r"\s", "", "".join(row[c] for row in group["rows"]))
            assert before == after
            assert hashlib.sha256(before.encode()).hexdigest() == group["column_nonwhitespace_sha256"][c]
    design = next(row for row in remaining["groups"][0]["rows"] if row[0] == "Tasarımı ve bakımlılığı")
    assert "Yeni binalarla yüksek rekabet gücüne sahiptir." in design[1]
    assert "özellikleri yeni binalarla rekabet edebilecek seviyededir." in design[2]
    assert sum(row[0].startswith("İşletmecinin finansal gücü,") for row in remaining["groups"][1]["rows"]) == 2
    repair, _, reviewed = reconstruct_candidate("1305")
    assert reviewed == candidate
    assert repair["tables"] == len(re.findall(r"^\| ---", candidate, re.MULTILINE)) == 10
    assert repair["reading_order_repairs"] == evidence["reading_order_repairs"] == 10
    for p in evidence["replacements"]:
        if p["kind"] == "reading_order":
            assert Counter(re.findall(r"\w+|[^\w\s]", p["old"])) == Counter(re.findall(r"\w+|[^\w\s]", p["new"]))
    coverage = json.loads((ROOT / "docs/evidence/document-repairs/1305-textual-coverage.json").read_text())
    assert coverage["status"] == "token_differences_accounted_for"
    assert coverage["unresolved_differences"] == []
    assert coverage["embedded_image_count"] == repair["restored_image_formulas"] == 11
    for name, sha in coverage["review_evidence_sha256"].items():
        assert hashlib.sha256((ROOT / "docs/evidence/document-repairs" / name).read_bytes()).hexdigest() == sha
    assert sorted(repair["pages_individually_reviewed"] + repair["pages_layout_contact_sheet_reviewed"]) == list(
        range(1, 69)
    )
    assert re.findall(r"\bMADDE\s+(\d+)", candidate) == [str(n) for n in range(1, 12)]
    assert "GENEL GEREKÇE" in candidate and "**/**/2026" in candidate
    assert assess_markdown_quality(candidate).counts["malformed_table_rows"] == 0
    assert assess_markdown_quality(candidate, document_id="1305").label == "fail"
