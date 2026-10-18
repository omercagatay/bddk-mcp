"""Write one sha-only receipt per changed document into the quality-pass evidence directory.

usage: build_receipts.py   (reads the baseline, seed_data/documents.json, specs/ and state/; no document text is copied)
"""

import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
E = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bddk_mcp.store.section_index import extract_document_sections  # noqa: E402

base = {d["document_id"]: d for d in json.loads(gzip.decompress((E / "baseline-documents.json.gz").read_bytes()))}
cur = {d["document_id"]: d for d in json.loads((ROOT / "seed_data/documents.json").read_text())}
state = {
    name: json.loads((E / "state" / f"{name}.json").read_text())
    for name in ("phase_a_log", "methods", "viewed", "gate", "pdf_hashes", "edit_logs", "respace_log")
}
phase_a, methods, viewed, gate, pdfs, logs = (
    state[k] for k in ("phase_a_log", "methods", "viewed", "gate", "pdf_hashes", "edit_logs")
)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def sections(did, text):
    return [
        f"{s.section_type} {s.section_ref}" for s in extract_document_sections(did, text) if s.section_type != "govde"
    ]


GLYPH = {
    "'\\x96'": "U+0096 (cp1254 en dash) -> –",
    "'\\x92'": "U+0092 (cp1254 right quote) -> ’",
    "'\\x93'": "U+0093 (cp1254 left double quote) -> “",
    "'\\x94'": "U+0094 (cp1254 right double quote) -> ”",
    "'Đ'": "Đ (PDF text-layer mapping error; printed glyph is İ) -> İ",
    "';'": "U+037E Greek question mark -> ;",
    "'ﬁ'": "ﬁ ligature -> fi",
    "'ﬂ'": "ﬂ ligature -> fl",
}
RELAY_LAYOUT = [
    "bold, italic and underline emphasis; font sizes; page breaks",
    "justified-line double spaces collapse to single spaces; physical line breaks inside a paragraph are kept",
    "bare page numbers and the BDDK letterhead footer (address/phone/web lines) are page furniture and are dropped",
    "list bullets come from the PDF text layer (private-use bullet glyphs mapped to • ▪ ➢ ✓ → —)",
    "multi-line capitalised titles are one paragraph; paragraph breaks follow PDF indentation, short closing lines "
    "and vertical spacing",
    "table cells merged across rows or columns repeat their label in each cell; line breaks inside a cell are <br/>",
]
SOURCE_ANOMALIES = {
    "mevzuat_20029": ["The official PDF itself prints '£' (Symbol-font ≤) in Tablo 1; kept as printed."],
    "rg_32202_20230526_6": [
        "Kept as printed: 'Dikkate Anma Oranı' (row 12), '20/1/(b)' (row 20), 'oranınave' (MADDE 2)."
    ],
}
SECTION_NOTES = {
    "mevzuat_19498": "gained EK-1 and EK-2 (added from the official .xlsx annexes) and an EK-3 section (the annex text "
    "used to sit after MADDE 34 without one)",
    "mevzuat_21194": "the lost bents were row labels of the flattened EK-3 slotting tables ('(a) Al ya da öde …', "
    "'(c) İnşaat halindeki gayrimenkul'), now table cells",
    "935": "gained numbered paragraphs that the scrambled stored text had hidden mid-paragraph",
    "950": "gained numbered paragraphs that the scrambled stored text had hidden mid-paragraph",
    "1167": "gained numbered paragraphs that the scrambled stored text had hidden mid-paragraph",
    "1230": "the lost bent was built from a scrambled fragment ('(d) bendinde belirtilen'); the text now reads as printed",
    "1288": "the lost fıkra was a scrambled duplicate of 'MADDE 6 – (1) Maruz kalınan riskler', now inline as printed",
    "1240": "EK-1 notes (1)-(11) sit inside the EK-1 table as printed, so they are no longer separate paragraphs",
    "rg_32202_20230526_6": "the lost bents came from OCR garbage of the table images; the tables are now transcribed",
    "mevzuat_10735": "gained fıkras that the stored text had run into one paragraph; the footnote bodies '(1)', '(2)' "
    "also parse as fıkra 1 and 2",
    "mevzuat_21425": "gained MADDE 10, 11 and 12, missing from the stored extraction, and fıkras the stored text had "
    "run into one paragraph",
    "mevzuat_21193": "the two lost '(i)' bents are row labels of a template table ('(i) Tezgahüstü türev işlemler'), "
    "now table rows; the lost '(b)' bent was a line break inside Geçici Madde 1 ('… (b) bendinde yer alan …')",
}
STATIC_MEVZUAT_PDF = {"mevzuat_10735", "mevzuat_15481", "mevzuat_21425"}
DOC_NOTES = {
    "905": [
        "the PDF text layer maps three Turkish glyphs to Ġ/ġ/Ģ; mapped back (İ/Ş/ş) after alignment with the stored text",
        "the 6 tables are the reviewed stored table blocks, placed at their page areas",
        "12 glued footnote markers and the 15 footnote bodies written as Unicode superscripts (the reviewed form "
        "of this document's tables); 3 paragraphs split by a page-bottom footnote block rejoined (post905.py)",
    ],
    "943": [
        "running header 'Bankacılık Düzenleme ve Denetleme Kurulu' (21 pages) dropped as page furniture",
        "glued footnote markers 'tanımlamaktadır1', 'kapsamalıdır2' written as $^{1}$/$^{2}$ with their bodies",
    ],
    "kgk_tms_12": [
        "all 6 joins are page-break splits: the first part ends a page (followed by the page number) and the "
        "sentence continues on the next page; positions were located in a later KGK download whose SHA-256 differs "
        "from the file pinned in scripts/add_kgk_tms_12.py, so the pinned source PDF itself was not re-read",
    ],
    "1313": [
        "formula legend symbols written as $\\mathrm{TF}_i$ and $\\mathrm{TF}_i^t$ (the second was split "
        "'TFi' / 't' over two paragraphs), matching the display formula"
    ],
    "mevzuat_21193": [
        "every template table rebuilt from the ruled cells of its page (regions2.py), with reviewed fixes "
        "(specs/fix_mevzuat_21193.json): table halves split by a page break merged, rows the GeneratePdf output "
        "prints twice on top of each other (overprint) taken from the stored text",
        "a glyph the PDF font lacks (printed as an empty box) written as '•' in lists; EK-1 marks written as '✓', "
        "placed from the template frames and checked against the printed column totals (20/20/4/22/14)",
        "3 character-order glitches from overprinted glyphs corrected as printed",
        "2 text-layer fragments of overprinted table rows dropped (spec 'drop': 'Tablolarstoğundaki' p18, "
        "'65 Toplam' p10); the rows themselves are in the rebuilt tables",
    ],
    "mevzuat_21194": [
        "tables rebuilt from the ruled cells of their pages (regions2.py) with reviewed fixes "
        "(specs/fix_mevzuat_21194.json): Tablo 1 risk weights (5 categories x 2 maturities), the closing Resmî Gazete "
        "table, the EK-3 slotting tables merged across page breaks, overprinted continuation rows taken from the "
        "stored text, and two blocks of unruled sub-rows split by line position (splitrows.py)",
        "the GeneratePdf output prints every formula as an empty box (no glyphs, no images): the stored LaTeX was "
        "grafted at the box positions checked on the page images (post21194.py); the stored inline legend of the "
        "Ek-1 maturity formula replaces the text-layer legend whose symbols were boxes. The 12 display formulas and "
        "that legend were compared with the formula images of the original Tebliğ (resmigazete.gov.tr "
        "eskiler/2015/10/20151023-15.pdf, SHA-256 0b0a97af26fe29cb2c00bdac1ccfc36a7f297edabe21f85f2798b648b0854cf2, "
        "pp. 9-14 and 20-21): all match",
        "the KOBİ correlation formula (paragraph marked '(Değişik:RG-14/3/2018-30360)') was transcribed from the "
        "amending Tebliğ's own image (resmigazete.gov.tr 20180314-8_dosyalar/image002.gif, SHA-256 "
        "c5a66fc2f894db2ee7de1bc93ee9f6e2749633e6e468aecc3aa20ce25ba430eb). The stored formula (max[S, 5], -5, /36) "
        "was a mis-transcription of the 2015 original (max{4, S}, -4, /36 on p. 10 of the RG PDF above), and RG 30360 "
        "replaced the whole paragraph with the C-form. The consolidated text also kept the 2016 legend under "
        "it (RG 20/1/2016-29599, byte-identical: net satış hâsılatı, no 'C'); RG 30360 MADDE 3 replaced the whole "
        "paragraph, so its legend is taken verbatim from the amending Tebliğ's page (legend21194.py). Every other "
        "provision amended by RG 29599, 30360 and 30742 was checked present in the consolidated text",
        "'GEÇİCİ MADDE 1 - (1)' restored: the bold '-' is overprinted on '(1)' in the text layer",
        "4 text-layer fragments of the EK-3 header row reprinted over continuation rows (pp. 20-22) dropped "
        "(spec 'drop'); the header is the table's first row",
    ],
}
GATE_NOTE = (
    "Pre-review gate on the unreviewed relay draft (token multiset, sections, content-stream order, punctuation); "
    "'review'/'fail' sent the draft to the review recorded in gate_resolution before anything was applied."
)


receipts = 0
for did, d in cur.items():
    b = base[did]
    if d["markdown_content"] == b["markdown_content"]:
        continue
    log = logs.get(did, {})
    steps = log.get("manual_steps", [])
    m = methods.get(did)
    g = gate.get(did)
    pages_viewed = viewed.get(did, [])
    classes = []
    if did in phase_a:
        classes += [f"{GLYPH.get(k, k)} ×{n}" for k, n in phase_a[did].items()]
    if log.get("applied_windows"):
        classes.append(
            f"scrambled word order: {log['applied_windows']} windows restored from PDF order "
            "(pdftotext -layout, confirmed by -raw content-stream order)"
        )
    if m:
        classes.append(
            "full rebuild from the official source text layer with relay.py (paragraphs, page furniture, "
            "scramble-free order)"
        )
    if g and g.get("reordered"):
        classes.append(
            "stored word order differed from the PDF; rebuilt order matches the PDF content stream (pdftotext -raw)"
        )
    spec = E / "specs" / f"{did}.json"
    if spec.exists():
        sp = json.loads(spec.read_text())
        n_tab = sum(1 for t in sp.get("tables", []) if t.get("md", "").lstrip().startswith("|"))
        n_tex = sum(1 for t in sp.get("tables", []) if t.get("md", "").lstrip().startswith("$$"))
        seen = "reviewed against the page image" if pages_viewed else "built from the PDF text layer"
        if n_tab:
            classes.append(f"{n_tab} table block(s) written as Markdown tables, {seen}")
        if n_tex:
            classes.append(f"{n_tex} display formula(s) as LaTeX, {seen}")
        if sp.get("regions") and did != "905":  # 905 carries its reviewed stored tables (DOC_NOTES)
            n_reg = sum(1 for r in sp["regions"] if r["md"])
            classes.append(f"{n_reg} table block(s) placed by page area (spec 'regions'), {seen}")
        if sp.get("replace"):
            classes.append(
                f"{len(sp['replace'])} reviewed spot replacement(s) in the rebuilt text: "
                + "; ".join(f"{a.strip()!r} -> {b.strip()!r}" for a, b, _ in sp["replace"])
            )
    if state["respace_log"].get(did):
        classes.append(
            f"{state['respace_log'][did]} word boundaries taken from the stored text where the official "
            "PDF text layer glued words (e.g. 'MADDE 1 –(1)')"
        )
    if any("superscripts" in s for s in steps):
        classes.append("superscript footnote markers written as $^{n}$ (with their footnote bodies)")
    if any("footnote_rejoin" in s for s in steps):
        classes.append("paragraph split by a page-bottom footnote block rejoined; footnotes follow the paragraph")
    if any("rg_table" in s for s in steps):
        classes.append("flattened closing Resmî Gazete list written as the reviewed table layout")
    n_join = sum(1 for s in steps if "joins" in s)
    if n_join:
        classes.append(
            "paragraphs broken mid-sentence (page break, hanging indent or justified-line split) rejoined; "
            "each break was checked against the page image or the relay layout before joining"
        )
    if any("page_numbers_removed" in s for s in steps):
        classes.append("bare printed page numbers removed (page furniture)")
    if any("page_furniture_removed" in s for s in steps):
        classes.append("BDDK letterhead footer blocks removed (page furniture)")
    n_scr = log.get("scripts_count", 0)
    if any("scripts" in s for s in steps):
        classes.append(
            f"{n_scr} glued formula-variable sub/superscripts written as LaTeX ($_{{…}}$ / $^{{…}}$), each token "
            "checked against the PDF glyph size and baseline; occurrences the PDF prints plain stay plain (scriptfix.py)"
        )
    if any("annex_xlsx" in s for s in steps):
        classes.append(
            "EK-1 and EK-2 (the LCR reporting forms with the dikkate alınma / çıkış / giriş rates Madde 9 refers to) "
            "added from the official annex archive mevzuat.gov.tr/MevzuatMetin/yonetmelik/7.5.19498-ek.zip (SHA-256 "
            "d0c1cea72e3d1048e854f9e73b7cc95f77e2a13b9d028b5e1e8f195f5e318585): the ingest annex reader handled .docx "
            "only, so the two .xlsx annexes were missing. One Markdown table per sheet block; every one of the 628 and "
            "634 cells checked present in sheet order; rates written from Excel's percent format as %N; EK-3's three "
            "formulas (text in the docx) written as LaTeX; annex headings as plain 'EK-n' lines so the section index "
            "ends MADDE 34 before them (annex19498.py)"
        )
    if any("text_formulas" in s for s in steps):
        classes.append(
            "formulas printed as plain text written as LaTeX with the printed symbols kept, each compared with the "
            "official source line (textformulas.py / annex19498.py for EK-3)"
        )
    if any("whitespace" in s for s in steps):
        classes.append(
            "whitespace only: justified-text runs of spaces collapsed and paragraph indentation removed (4+ leading "
            "spaces rendered paragraphs as code blocks); non-whitespace characters unchanged (wsclean.py)"
        )
    if any("flat_tables" in s for s in steps):
        classes.append(
            "flattened tables (one cell per paragraph) written as Markdown tables, reviewed against the page image"
        )
    last = max((i for i, s in enumerate(steps) if "full_rebuild" in s), default=-1)
    n_fix = sum(1 for s in steps[last + 1 :] if not s)
    if n_fix and did != "1296":
        seen = "reviewed against the page image" if pages_viewed else "checked in the text diff"
        classes.append(f"{n_fix} manual correction step(s){' after the rebuild' if last >= 0 else ''}, {seen}")
    if did == "1296":
        classes = [
            "letterhead line order restored (T.C before the BDDK name), as printed",
            "3 paragraphs/titles broken mid-sentence by the extractor rejoined, as printed",
            "flattened 'Çekirdek mevduat' limits table written as a Markdown table, reviewed against the page image "
            "(stored values kept; they equal the current official PDF)",
            "2 section headings split over two paragraphs rejoined",
            "whitespace only: justified-text runs of spaces collapsed and paragraph indentation removed (4+ leading "
            "spaces rendered paragraphs as code blocks); non-whitespace characters unchanged (wsclean.py)",
            "brought to the current official PDF at the same source URL (created 31/8/2026; token diff against its "
            "text layer had exactly three content changes, page 2 compared visually): item 1 gains "
            "'(Değişik: Kurul Kararı-20/8/2026-11559)' and katılma hesabı kaynaklı pozisyonlar %100 -> %80; the "
            "Kurul Kararı date is printed '29/05/2025' (update1296.py). downloaded_at, extracted_at and file_size "
            "describe that PDF; the title's double space is collapsed",
            "quality engine: with single spaces, 'Yönetmelik EK-2’de yer alan formüllerde' (formulas of another "
            "regulation, mevzuat_42628) met the formula-reference rule although no formula of this text is missing; "
            "with the owner's approval, policy markdown-quality-assessment-v6 lists it as a reviewed external "
            "reference, and the document is clean",
        ]
    classes += DOC_NOTES.get(did, [])
    if did == "rg_32202_20230526_6":
        classes.append(
            "full rebuild: prose from the RG .htm, tables visually transcribed from RG issue PDF pp.49-56 "
            "(■ = cell printed black)"
        )
    pdf = pdfs.get(did)
    if m or did == "rg_32202_20230526_6":
        repair_class = "full_rebuild_from_official_source"
    elif did in phase_a and not log.get("applied_windows") and not steps:
        repair_class = "glyph_repair"
    else:
        repair_class = "local_edit"
    receipt = {
        "document_id": did,
        "status": "source_reviewed_candidate_awaiting_owner_approval",
        "review_scope": "Assistant technical source review; not independent human/legal approval.",
        "repair_class": repair_class,
        "input_markdown_sha256": sha(b["markdown_content"]),
        "output_markdown_sha256": sha(d["markdown_content"]),
        "extraction_method_before": b["extraction_method"],
        "extraction_method_after": d["extraction_method"],
        "change_classes": classes,
        "pages_visually_reviewed": pages_viewed,
        "source_url": d["source_url"],
    }
    before, after = Counter(sections(did, b["markdown_content"])), Counter(sections(did, d["markdown_content"]))
    if before != after:
        receipt["section_changes"] = {
            "lost": sorted((before - after).elements()),
            "gained": sorted((after - before).elements()),
            "note": SECTION_NOTES[did],
        }
    if pdf:
        receipt["source_pdf_sha256"] = pdf["sha256"]
        receipt["source_pdf_bytes"] = pdf["bytes"]
        receipt["source_pdf_matches_stored_file_size"] = pdf["matches_stored_file_size"]
        if did in STATIC_MEVZUAT_PDF and not pdf["matches_stored_file_size"]:
            receipt["source_pdf_note"] = (
                "static mevzuat.gov.tr MevzuatMetin PDF of the current consolidated text; the stored text came from "
                "the mevzuat HTML page, so the stored file size describes that page"
            )
        elif did.startswith("mevzuat_") and not pdf["matches_stored_file_size"]:
            receipt["source_pdf_note"] = (
                "mevzuat.gov.tr File/GeneratePdf output is regenerated per request (not byte-reproducible)"
            )
    if g:
        receipt["pre_review_gate"] = {
            "note": GATE_NOTE,
            "status": g["status"],
            "reasons": g["reasons"],
            "paragraph_break_spots": g["n_susp"],
        }
        resolution = []
        if g["status"] == "pass":
            resolution.append("relay output word tokens equal the stored text except dropped page furniture")
        markup = any(r.startswith(("stored has", "ruled tables")) for r in g["reasons"])
        if any(r.startswith("punctuation") for r in g["reasons"]):
            resolution.append(
                "punctuation difference = table/LaTeX markup written differently and dropped page-furniture lines"
                if markup
                else "punctuation difference = dropped page-furniture lines (letterhead footer, page numbers)"
            )
        if any(r.startswith("section") for r in g["reasons"]):
            resolution.append(
                "section list change listed in section_changes"
                if "section_changes" in receipt
                else "the draft's section list difference is gone in the final text (same sections as stored)"
            )
        if g["n_susp"]:
            resolution.append(
                f"{g['n_susp']} paragraph-break difference(s) from the stored text; rebuilt breaks follow the "
                "PDF layout (indentation and vertical spacing)"
            )
        if any(r.startswith(("rebuilt-only", "stored-only", "stored has", "ruled tables")) for r in g["reasons"]):
            resolution.append(
                "token, table and formula differences resolved by the reviewed specs/ and manual steps above"
                + (", checked against the viewed page images" if pages_viewed else "")
            )
        receipt["gate_resolution"] = resolution
    if m:
        receipt["layout_not_carried"] = RELAY_LAYOUT
    if did in SOURCE_ANOMALIES:
        receipt["source_anomalies_preserved"] = SOURCE_ANOMALIES[did]
    if did == "rg_32202_20230526_6":
        receipt["source_note"] = (
            "file_size/downloaded_at still describe the original .htm download; the tables are images in "
            "both the .htm and RG issue PDF https://www.resmigazete.gov.tr/eskiler/2023/05/20230526.pdf (pp.49-56)"
        )
    (E / f"{did}.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=1) + "\n")
    receipts += 1
print("receipts", receipts)
