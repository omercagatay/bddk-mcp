"""Source-verified rebuilds of markitdown-scrambled mevzuat.gov.tr regulations from their official PDFs.

markitdown split widely spaced justified lines and pushed their words out of order (for example
mevzuat_22599 Madde 6/5 lost "donuk alacakların sınıf değişikliğine" to the end of the fıkra).
Each document listed in SPECS is rebuilt from the PDF the stored text was extracted from:

- characters come from the PDF text layer (`pdftotext -bbox-layout`); physical lines are rebuilt
  from word coordinates, because the text layer itself splits some justified lines;
- a line indented past the body margin starts a paragraph, flush-left lines continue it, also
  across page breaks; 7 pt words are superscript footnote markers, written $^{(n)}$ like the
  16290/21192 rebuilds, and listed footnotes get the same label;
- tables and forms are reviewed Markdown blocks whose cell words must equal the PDF lines they
  replace; image-only content has no text layer and is transcribed visually;
- every PDF page was compared visually. Outside blocks, the candidate's non-whitespace character
  stream must equal the PDF's.

Usage: uv run python scripts/rebuild_mevzuat_from_pdf.py <document_id> <official.pdf> [--apply]
Not a corpus signature or a publish.
"""

from __future__ import annotations

import difflib
import gzip
import hashlib
import html
import json
import math
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DOCS_PATH = ROOT / "seed_data" / "documents.json"
REPAIRS = ROOT / "docs" / "evidence" / "document-repairs"
BASELINE_PATH = REPAIRS / "signed-baseline-documents.json.gz"
NEW_METHOD = "manual_pdf"
PAGE_SIZE = 5000  # bddk_mcp.core.config default; total_pages = ceil(len / PAGE_SIZE) like the rest of the seed
WORD_RE = re.compile(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="[\d.]+" yMax="([\d.]+)">([^<]*)</word>')
# The printed footnote rule is a run of underscores; >=10 trips the long_underscore_run heuristic
# meant for fill-in blanks. Its length is layout, so it is written like 16290's rule.
RULE = "_" * 7


@dataclass(frozen=True)
class Block:
    """PDF lines from the one starting with `start` through the one starting with `end`, as reviewed Markdown."""

    start: str
    end: str | None  # None: through the end of the document
    markdown: str
    rewraps: tuple[tuple[str, str], ...] = ()  # a word the PDF wraps inside a narrow cell without a hyphen


@dataclass(frozen=True)
class ImageTable:
    """Image-only content (no text layer), transcribed visually and placed after the paragraph `after`."""

    after: str
    markdown: str
    page: int
    pixels: str
    sha256: str  # of `pdfimages -png` output


@dataclass(frozen=True)
class Spec:
    pdf_sha256: str
    pages: int
    baseline_sha256: str
    title_lines: int
    restored: tuple[tuple[str, str, str], ...]  # (section_type, section_ref, text the rebuild must contain)
    joins: tuple[str, ...] = ()  # line prefixes whose next line continues the same heading
    footnotes: tuple[str, ...] = ()  # paragraph prefixes "(n) …" that are footnotes
    blocks: tuple[Block, ...] = ()
    images: tuple[ImageTable, ...] = ()
    layout_not_carried: tuple[str, ...] = ()
    anomalies: tuple[str, ...] = ()


RG_HEAD = "Yönetmeliğin Yayımlandığı Resmî Gazete’nin"
AMENDING_REGULATIONS = "Yönetmelikte Değişiklik Yapan Yönetmeliklerin Yayımlandığı Resmî Gazetelerin"


def rg_table(original: tuple[str, str], amended_head: str, rows: list[tuple[str, str, str]]) -> str:
    """Closing Resmî Gazete list; merged cells repeat their label, as in the 16290/21192 rebuilds."""
    lines = [f"|  | {RG_HEAD} | {RG_HEAD} |", "|---|---|---|", "|  | Tarihi | Sayısı |"]
    lines += [f"|  | {original[0]} | {original[1]} |", f"|  | {amended_head} | {amended_head} |"]
    lines += ["|  | Tarihi | Sayısı |"] + [f"| {n} | {date} | {number} |" for n, date, number in rows]
    return "\n".join(lines)


SPECS: dict[str, Spec] = {
    "mevzuat_22599": Spec(
        pdf_sha256="a5b278ddd0e286da3427b7a5f6e60abdeaa820115c8189914385102769627e2c",
        pages=13,
        baseline_sha256="cf1af936f1ba010a0748d09f79dd1a0d0750c30603b545fe2a65266753a4fb58",
        title_lines=3,
        restored=(
            (
                "madde",
                "6",
                "(5) (Ek:RG-6/7/2021-31533) Bankalar, donuk alacakların sınıf değişikliğine ilişkin kriterlerini",
            ),
            ("madde", "15", "| Beşinci Grup Teminatlar | Yüzde yirmi (%20) |"),
        ),
        footnotes=("(1) Bu değişiklik 1/1/2017", "(2) Bu değişiklik 1/1/2018", "(3) Bu değişiklik 19/7/2019"),
        blocks=(
            Block(
                RG_HEAD,
                None,
                rg_table(
                    ("22/6/2016", "29750"),
                    AMENDING_REGULATIONS,
                    [
                        ("1", "14/12/2016", "29918"),
                        ("2", "27/3/2018", "30373"),
                        ("3", "15/8/2018", "30510"),
                        ("4", "18/10/2018", "30569"),
                        ("5", "27/11/2019", "30961"),
                        ("6", "6/7/2021", "31533"),
                        ("7", "24/5/2022", "31845"),
                    ],
                ),
            ),
        ),
        images=(
            ImageTable(
                after="a) Özel karşılığa tabi alacak tutarı, teminatların 14 üncü",
                markdown="\n".join(
                    [
                        "| Birinci Grup Teminatlar | Yüzde yüz (%100) |",
                        "|---|---|",
                        "| İkinci Grup Teminatlar | Yüzde seksen (%80) |",
                        "| Üçüncü Grup Teminatlar | Yüzde altmış (%60) |",
                        "| Dördüncü Grup Teminatlar | Yüzde kırk (%40) |",
                        "| Beşinci Grup Teminatlar | Yüzde yirmi (%20) |",
                    ]
                ),
                page=10,
                pixels="465x93",
                sha256="e94c85b15bc15a35b83d0b33023abc16f1d022b2d9a08cf18f87c0c50953fc75",
            ),
        ),
        layout_not_carried=("bold article/amendment labels and the underline on Madde 18/9 'bir milyon'",),
        anomalies=("Madde 6/5 has no closing period in the official text",),
    ),
    "mevzuat_39171": Spec(
        pdf_sha256="f58d2edcc841dc5c67907b19d4a1ac8e95578af8df936f6352ee75c09f4b0575",
        pages=6,
        baseline_sha256="0a11034901921bc386800af7e08e5e2c23fe9ba710302efb2ac16f2447d7c911",
        title_lines=2,
        restored=(),
        footnotes=("(1) Bu değişiklik 1/7/2022",),
        blocks=(
            Block(
                RG_HEAD,
                "2.",
                rg_table(("4/6/2021", "31501"), AMENDING_REGULATIONS, [("1.", "24/12/2021", "31699"), ("2.", "", "")]),
            ),
        ),
        layout_not_carried=("bold article/amendment labels",),
        anomalies=("The amendment list prints an empty second row",),
    ),
    "mevzuat_20645": Spec(
        pdf_sha256="80c62e79e2e8ac214f20d5c5bafc5364ad5e36a2a34275eb21491560a0435b7c",
        pages=9,
        baseline_sha256="6af982097080c1316283e271ff19ac619bf113e4783e50d278c1a5cef697bc41",
        title_lines=1,
        restored=(),
        joins=("Bağımsız Denetim Kuruluşlarının Listeye Alınmasına, Ortaklarına,",),
        blocks=(
            Block(
                RG_HEAD,
                "2.",
                rg_table(("2/4/2015", "29314"), AMENDING_REGULATIONS, [("1.", "25/5/2022", "31846"), ("2.", "", "")]),
            ),
        ),
        layout_not_carried=("bold article/amendment labels and the underlined amended phrases in 18/3 and 19/2",),
        anomalies=(
            "The amendment list prints an empty second row",
            "'Ekleri için tıklayınız.' is printed in the PDF; the annexes are not part of it",
        ),
    ),
    "mevzuat_10522": Spec(
        pdf_sha256="1b7d3cac3b68c42507c1e2218bedbb38b446af41c85f36c01dec60fcbf3decdd",
        pages=8,
        baseline_sha256="65a1fb680973e4a3c9f86b711b13f7a4b0673d3b66309ac0335dafc618d58fe3",
        title_lines=2,
        restored=(),
        blocks=(
            Block(
                RG_HEAD,
                None,
                rg_table(
                    ("22/7/2006", "26236"),
                    "Yönetmelikte Değişiklik Yapan Yönetmeliğin Yayımlandığı Resmî Gazetelerin",
                    [
                        ("1.", "10/3/2007", "26458"),
                        ("2.", "24/2/2013", "28569"),
                        ("3.", "11/7/2014", "29057"),
                        ("4.", "23/10/2015", "29511"),
                        ("5.", "21/4/2022", "31816"),
                    ],
                ),
            ),
        ),
        layout_not_carried=("bold article/amendment labels",),
    ),
    "mevzuat_40203": Spec(
        pdf_sha256="e34218286fdbd676c1e28eb97cbd77b1f4e520140150bdf09b672108a3a76261",
        pages=9,
        baseline_sha256="71db4d62064c81ecb626c936d0c6f9c07ab4db303f1c8cfaef075068ddea668d",
        title_lines=2,
        restored=(),
        joins=("Net İstikrarlı Fonlama Oranına İlişkin Raporlama ve",),
        layout_not_carried=("bold article labels",),
    ),
    "mevzuat_40519": Spec(
        pdf_sha256="ab7d776e69bd911972d1306c4e30d85f5aba6bb8f395825fa7829e7d7b0b7f85",
        pages=10,
        baseline_sha256="28e5f092bc78d48f1b3180ef27e4600d6d2816f576afa738e2c655de3a6b0c0f",
        title_lines=2,
        restored=(),
        joins=("Kredi Sınırları, Kredi Sınırları Hesaplamasında Risk Tutarı, Kredi Sınırlarına Tabi Olmayan",),
        blocks=(
            Block(
                "Risk Tutarı",
                "Ortaklık payları",
                "\n".join(
                    [
                        "|  | Risk Tutarı |",
                        "|---|---|",
                        "| Ticari riskler | Diğer tüm riskler gibi |",
                        "| Serbest olmayan başlangıç teminatı (iflastan ifraz edilmiş olan) | 0 (sıfır) |",
                        "| Serbest başlangıç teminatı (iflastan ifraz edilmiş olmayan) | Başlangıç teminatının nominal tutarı |",
                        "| Fonlanmamış garanti fonu taahhüdü | 0 (sıfır) |",
                        "| Fonlanmış garanti fonu tutarı | Fonlanmış garanti fonuna yapılan katkının nominal tutarı |",
                        "| Ortaklık payları | Ortaklık paylarının nominal tutarı |",
                    ]
                ),
            ),
        ),
        layout_not_carried=("bold article labels",),
    ),
    "mevzuat_16071": Spec(
        pdf_sha256="38ca2d4f63ef790c4b4f5f77fa674c94535d73ca2fb66c6d8e48e8104e4719e8",
        pages=29,
        baseline_sha256="5fb95544797ab3daa838e3c149407de73e6b9c7026eb624f1f4aaf68ec633a23",
        title_lines=2,
        restored=(),
        joins=(
            "ORTAKLARIN/ KALİTE GÜVENCESİ SİSTEMİNDEN SORUMLU YÖNETİM KURULU",
            "ÜYELERİNİN/ DERECELENDİRME KOMİTESİ ÜYELERİNİN/DERECELENDİRME UZMANLARININ",
        ),
        blocks=(
            Block(
                "ADI-SOYADI :",
                "Yılı Davacı Dava Konusu İlgili Mahkeme",
                "\n".join(
                    [
                        "| ADI-SOYADI | : |",
                        "|---|---|",
                        "| İKAMETGAH ADRESİ | : |",
                        "| ÖĞRENİM DURUMU (Ayrıntılı) | : |",
                        "| HALEN ÇALISTIĞI İSYERİNİN UNVANI VE ADRESİ | : |",
                        "| MESLEĞİ VE GÖREV UNVANI | : |",
                        "| TC KİMLİK/PASAPORT NUMARASI | : |",
                        "| VERGİ KİMLİK NUMARASI$^{1}$ | : |",
                        "| ORTAKLIK PAYI | : |",
                        "",
                        "DAHA ÖNCE ÇALIŞTIĞI İŞYERLERİ",
                        "",
                        "|  | Kuruluşun Unvanı | Giriş-Ayrılış Tarihi | Görev Unvanı |",
                        "|---|---|---|---|",
                        "| 1- |  |  |  |",
                        "| 2- |  |  |  |",
                        "",
                        "SON BEŞ YILDA DERECELENDİRİLEN İŞLETMELERE İLİŞKİN BİLGİLER",
                        "",
                        "| İşletmenin Unvanı | Denetim Süresi | Görev Unvanı |",
                        "|---|---|---|",
                        "|  |  |  |",
                        "|  |  |  |",
                        "",
                        "DERECELENDİRMEYLE İLGİLİ ALINAN EĞİTİMLER ve SERTİFİKALAR",
                        "",
                        "| Yılı | Eğitimin Süresi | Eğitimin Adı | Sertifika |",
                        "|---|---|---|---|",
                        "|  |  |  |  |",
                        "|  |  |  |  |",
                        "",
                        "ÇALIŞAN, YÖNETİCİ, ORTAK VEYA KURUCU KONUMUNDA BULUNDUĞU KURULUŞLAR",
                        "",
                        "| Başlangıç Yıl/Ay | Şirket Adı/Yeri | Faaliyet Konusu | Pozisyon İlişki Türü | % Ortaklık |",
                        "|---|---|---|---|---|",
                        "|  |  |  |  |  |",
                        "|  |  |  |  |  |",
                        "",
                        "DAVALARA İLİŞKİN BİLGİLER",
                        "",
                        "| Yılı | Davacı | Dava Konusu | İlgili Mahkeme |",
                        "|---|---|---|---|",
                        "|  |  |  |  |",
                        "|  |  |  |  |",
                    ]
                ),
                rewraps=(("Başlan", "gıç"),),
            ),
            Block(
                RG_HEAD,
                "6 1/9/2023",
                rg_table(
                    ("17/4/2012", "28267"),
                    AMENDING_REGULATIONS,
                    [
                        ("1", "2/4/2015", "29314"),
                        ("2", "20/1/2016", "29599"),
                        ("3", "6/7/2018", "30470"),
                        ("4", "3/12/2019", "30967"),
                        ("5", "20/9/2022", "31959"),
                        ("6", "1/9/2023", "32296"),
                    ],
                ),
            ),
        ),
        layout_not_carried=(
            "bold article/amendment labels",
            "EK-1 is a bordered form; its field rows and five sub-tables are Markdown tables",
            "page 24 is blank",
        ),
        anomalies=(
            "EK-1 prints 'HALEN ÇALISTIĞI İSYERİNİN' without the Turkish Ş",
            "Madde 25/6 prints '(MülgaRG-20/1/2016-29599)' without a colon",
        ),
    ),
}


def pdf_lines(pdf_path: Path) -> list[list[tuple[float, list[tuple[str, bool]]]]]:
    """Per page, physical lines as (xMin, [(word, is_superscript)]) ordered top-down, left-right."""
    bbox = subprocess.check_output(["pdftotext", "-enc", "UTF-8", "-bbox-layout", str(pdf_path), "-"], text=True)
    pages = []
    for page in bbox.split("<page ")[1:]:
        words = [(float(x), float(y0), float(y1), html.unescape(w)) for x, y0, y1, w in WORD_RE.findall(page)]
        lines: list[list] = []  # [y0, y1, [(x, word, superscript)]]
        for x, y0, y1, word in sorted((w for w in words if w[2] - w[1] >= 9), key=lambda w: (w[1], w[0])):
            line = next((ln for ln in lines if abs((ln[0] + ln[1]) / 2 - (y0 + y1) / 2) < 3), None)
            if line is None:
                lines.append([y0, y1, [(x, word, False)]])
            else:
                line[2].append((x, word, False))
        for x, _y0, y1, word in (w for w in words if w[2] - w[1] < 9):
            # A raised marker belongs to the line whose box holds its baseline.
            line = min(lines, key=lambda ln: abs((ln[0] + ln[1]) / 2 - y1) + (0 if ln[0] <= y1 <= ln[1] else 100))
            line[2].append((x, word, True))
        pages.append([(min(w[0] for w in ws), [(w[1], w[2]) for w in sorted(ws)]) for _, _, ws in sorted(lines)])
    return pages


def line_text(words: list[tuple[str, bool]]) -> str:
    text = ""
    for word, superscript in words:
        if superscript:
            if not re.fullmatch(r"\(?\d\)?", word):
                raise SystemExit(f"unexpected superscript {word!r}")
            text += "$^{" + word + "}$"
        else:
            text += (" " if text else "") + word
    return text


def markdown_words(markdown: str) -> list[str]:
    """Words a reviewed block asserts are in the PDF: merged (left-repeated) and empty cells count once."""
    words = []
    for row in markdown.replace("$^{", " ").replace("}$", " ").splitlines():
        if re.fullmatch(r"\|(?:-{3,}\|)+", row):
            continue
        if not row.startswith("|"):
            words += row.split()
            continue
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        words += [w for i, cell in enumerate(cells) if cell and (i == 0 or cell != cells[i - 1]) for w in cell.split()]
    return words


def build_candidate(spec: Spec, pages) -> tuple[str, list[str]]:
    """Return (markdown, PDF words outside blocks in candidate order); blocks are verified as they are placed."""
    lines = [line for page in pages for line in page]
    margin = Counter(round(x) for x, _ in lines).most_common(1)[0][0]
    paras: list[str] = []
    prose_words: list[str] = []
    joining = after_block = False
    index = 0
    while index < len(lines):
        x0, words = lines[index]
        text = line_text(words)
        block = next((b for b in spec.blocks if text.startswith(b.start)), None)
        if block is not None:
            end = len(lines) - 1
            if block.end is not None:
                end = next(i for i in range(index, len(lines)) if line_text(lines[i][1]).startswith(block.end))
            pdf_words = [w for _, ws in lines[index : end + 1] for w, _ in ws]
            for head, tail in block.rewraps:
                pdf_words.remove(head)
                pdf_words.remove(tail)
                pdf_words.append(head + tail)
            if Counter(markdown_words(block.markdown)) != Counter(pdf_words):
                raise SystemExit(f"block {block.start!r} differs from its PDF lines")
            paras.append(block.markdown)
            index, joining, after_block = end + 1, False, True
            continue
        prose_words += [w for w, _ in words]
        if re.fullmatch(r"_{10,}", text):
            text = RULE
        if joining:
            paras[-1] += " " + text
        elif x0 > margin + 10 or x0 < margin - 5 or not paras or after_block:
            paras.append(text)
        else:
            paras[-1] += ("" if paras[-1].endswith("-") else " ") + text
        joining, after_block = any(text.startswith(j) for j in spec.joins), False
        index += 1
    paras[: spec.title_lines] = [" ".join(paras[: spec.title_lines])]
    for i, para in enumerate(paras):
        if any(para.startswith(f) for f in spec.footnotes):
            paras[i] = re.sub(r"^(\(\d+\)) ", r"$^{\1}$ ", para)
    for image in spec.images:
        (anchor,) = [i for i, para in enumerate(paras) if para.startswith(image.after)]
        paras.insert(anchor + 1, image.markdown)
    return "\n\n".join(paras) + "\n", prose_words


def verify_prose(spec: Spec, candidate: str, prose_words: list[str]) -> None:
    prose = candidate
    for item in (*spec.blocks, *spec.images):
        prose = prose.replace(item.markdown, "")
    prose = prose.replace("$^{", "").replace("}$", "")
    if re.sub(r"\s", "", prose) != re.sub(r"_{10,}", RULE, "".join(prose_words)):
        raise SystemExit("candidate prose differs from the PDF text layer")


def changed_spans(original: str, candidate: str) -> list[dict[str, str]]:
    """Word spans that moved or changed between the stored text and the rebuild."""

    def words(text: str) -> list[str]:
        return re.sub(r"\|(?:-{3,}\|)+|\|", " ", text.replace("$^{", "").replace("}$", "")).split()

    old, new = words(original), words(candidate)
    return [
        {"op": op, "stored": " ".join(old[a1:a2]), "rebuilt": " ".join(new[b1:b2])}
        for op, a1, a2, b1, b2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
        if op != "equal"
    ]


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] not in SPECS:
        raise SystemExit(__doc__)
    doc_id, pdf_path, apply = sys.argv[1], Path(sys.argv[2]), "--apply" in sys.argv
    spec = SPECS[doc_id]
    pdf_bytes = pdf_path.read_bytes()
    if hashlib.sha256(pdf_bytes).hexdigest() != spec.pdf_sha256:
        raise SystemExit("PDF sha256 differs from the reviewed source; re-review before rebuilding")
    pages = pdf_lines(pdf_path)
    if len(pages) != spec.pages:
        raise SystemExit(f"PDF has {len(pages)} pages, expected {spec.pages}")
    candidate, prose_words = build_candidate(spec, pages)
    verify_prose(spec, candidate, prose_words)

    from bddk_mcp.quality.markdown_quality import assess_markdown_quality
    from bddk_mcp.store.section_index import extract_document_sections

    baseline = {d["document_id"]: d for d in json.loads(gzip.decompress(BASELINE_PATH.read_bytes()))}[doc_id]
    original = baseline["markdown_content"]
    original_sha = hashlib.sha256(original.encode()).hexdigest()
    if original_sha != spec.baseline_sha256:
        raise SystemExit("signed baseline text differs from the reviewed baseline")
    articles = {(s.section_type, s.section_ref) for s in extract_document_sections(doc_id, candidate)}
    lost = {(s.section_type, s.section_ref) for s in extract_document_sections(doc_id, original)} - articles
    if any(kind in {"madde", "gecici_madde", "ek_madde"} for kind, _ in lost):
        raise SystemExit(f"rebuild lost articles: {sorted(lost)}")
    sections = {(s.section_type, s.section_ref): s.content for s in extract_document_sections(doc_id, candidate)}
    for kind, ref, text in spec.restored:
        if text not in sections.get((kind, ref), ""):
            raise SystemExit(f"{kind} {ref} does not contain {text!r}")
    label = assess_markdown_quality(candidate, document_id=doc_id).label
    if label != "clean":
        raise SystemExit(f"candidate quality label is {label}")

    raw = DOCS_PATH.read_text(encoding="utf-8")
    docs = json.loads(raw)
    if json.dumps(docs, ensure_ascii=False) + "\n" != raw:
        raise SystemExit("documents.json round-trip identity failed")
    document = next(item for item in docs if item["document_id"] == doc_id)
    if document["file_size"] != len(pdf_bytes):
        raise SystemExit("PDF size differs from the stored download")
    stored_sha = hashlib.sha256(document["markdown_content"].encode()).hexdigest()
    candidate_sha = hashlib.sha256(candidate.encode()).hexdigest()
    if stored_sha not in {original_sha, candidate_sha}:
        raise SystemExit(f"stored {doc_id} is neither the reviewed baseline nor the rebuild; refusing to overwrite")
    spans = changed_spans(original, candidate)
    print(f"{doc_id} {original_sha[:12]} -> {candidate_sha[:12]}; {len(spans)} changed spans vs stored text")
    for span in spans:
        print(f"  {span['op']}: {span['stored'][:90]!r} -> {span['rebuilt'][:90]!r}")
    if not apply:
        print("dry-run only; pass --apply to write seed")
        return 0

    if stored_sha != candidate_sha:
        document["markdown_content"] = candidate
        document["content_hash"] = candidate_sha
        document["extraction_method"] = NEW_METHOD
        document["total_pages"] = math.ceil(len(candidate) / PAGE_SIZE)
        tmp = DOCS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(docs, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(DOCS_PATH)
    number = re.sub(r"\D", "", doc_id)
    evidence = {
        "document_id": doc_id,
        "status": "source_reviewed_candidate_awaiting_owner_approval",
        "repair_class": "full_rebuild_from_official_pdf",
        "review_scope": "Assistant technical source review; not independent human/legal approval.",
        "source_url": document["source_url"],
        "source_pdf_url": f"https://www.mevzuat.gov.tr/MevzuatMetin/yonetmelik/7.5.{number}.pdf",
        "source_pdf_sha256": spec.pdf_sha256,
        "source_pdf_bytes": len(pdf_bytes),
        "source_pdf_pages": spec.pages,
        "source_pdf_matches_stored_file_size": True,
        "pages_visually_compared": f"1-{spec.pages}",
        "rebuilt_with": "scripts/rebuild_mevzuat_from_pdf.py",
        "text_layer_check": "Outside reviewed blocks the non-whitespace character stream equals pdftotext "
        "-bbox-layout; each block's words equal the PDF lines it replaces",
        "image_only_content": [
            {"page": i.page, "pixels": i.pixels, "sha256": i.sha256, "transcribed_visually": i.markdown}
            for i in spec.images
        ],
        "original_markdown_sha256": original_sha,
        "candidate_markdown_sha256": candidate_sha,
        "extraction_method_after": NEW_METHOD,
        "layout_not_carried": [
            *spec.layout_not_carried,
            "footnote rules of 10 or more underscores are written with 7: their length is layout, and >=10 trips the "
            "long_underscore_run heuristic for fill-in blanks (same as the 16290 rebuild)",
        ],
        "source_anomalies_preserved": list(spec.anomalies),
        "changed_spans_vs_stored_text": spans,
        "replacements": [{"start": 0, "end": len(original), "old": original, "new": candidate}],
    }
    (REPAIRS / f"{doc_id}.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {DOCS_PATH} and {REPAIRS / f'{doc_id}.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
