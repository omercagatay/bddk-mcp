"""Source-verified rebuilds of markitdown-scrambled mevzuat.gov.tr regulations from their official PDFs.

markitdown split widely spaced justified lines and pushed their words out of order (for example
mevzuat_22599 Madde 6/5 lost "donuk alacakların sınıf değişikliğine" to the end of the fıkra).
Each document listed in SPECS is rebuilt from the PDF the stored text was extracted from:

- characters come from the PDF text layer (`pdftotext -bbox-layout`); physical lines are rebuilt
  from word coordinates, because the text layer itself splits some justified lines;
- a line indented past the body margin starts a paragraph, flush-left lines continue it, also
  across page breaks; 7 pt words are superscript footnote markers, written $^{(n)}$ like the
  16290/21192 rebuilds, and listed footnotes get the same label; a 7 pt word sitting on its
  line's baseline is a subscript, written $_{0,p}$;
- formulas printed as images are transcribed visually into LaTeX (inline $...$, or display
  $$...$$ blocks together with the text-layer words on their line);
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
import statistics
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
WORD_RE = re.compile(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>')
FORMULA_TOKEN = re.compile(r"\[\[F(\d+)\]\]")  # an image formula's place among the text-layer words
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
    # Words the source prints twice: a table header repeated at the top of continuation pages, which
    # File/GeneratePdf overprints on the first rows. They are page furniture, not table content.
    overprinted: tuple[str, ...] = ()
    merged_cells: bool = True  # False: equal adjacent cells are separate values (e.g. DERECE 2, ADET 2)
    words: tuple[str, ...] | None = None  # display formula: the text-layer words and [[Fn]] images it replaces


@dataclass(frozen=True)
class ImageTable:
    """Image-only content (no text layer), transcribed visually and placed after the paragraph `after`."""

    after: str
    markdown: str
    page: int
    pixels: str
    sha256: str  # of `pdfimages -png` output


@dataclass(frozen=True)
class Formula:
    """A formula printed as an image: its place on the page (points, from the top) and its LaTeX."""

    page: int
    bbox: tuple[float, float, float, float]
    latex: str


@dataclass(frozen=True)
class Spec:
    pdf_sha256: str
    pages: int
    baseline_sha256: str
    title_lines: int
    restored: tuple[tuple[str, str, str], ...]  # (section_type, section_ref, text the rebuild must contain)
    joins: tuple[str, ...] = ()  # line prefixes whose next line continues the same heading
    breaks: tuple[str, ...] = ()  # flush-left line prefixes that still start a paragraph
    footnotes: tuple[str, ...] = ()  # paragraph prefixes "(n) …" that are footnotes
    blocks: tuple[Block, ...] = ()
    images: tuple[ImageTable, ...] = ()
    formulas: tuple[Formula, ...] = ()  # in page order; [[Fn]] is formulas[n - 1]
    layout_not_carried: tuple[str, ...] = ()
    anomalies: tuple[str, ...] = ()
    # "static": MevzuatMetin PDF, byte-identical to the stored download. "generated": File/GeneratePdf output,
    # which the server regenerates per request, so only its word count can be matched to the stored text.
    source_kind: str = "static"
    pdf_url: str | None = None


RG_HEAD = "Yönetmeliğin Yayımlandığı Resmî Gazete’nin"


def reviewed_table(doc_id: str, name: str) -> str:
    """A table built from PDF word geometry and reviewed row by row against the page image."""
    return json.loads((REPAIRS / "native-sources" / f"{doc_id}-tables.json").read_text())["tables"][name]


def reviewed_formulas(doc_id: str) -> tuple[Formula, ...]:
    """Image formulas, each transcribed from a 400 dpi crop of its page."""
    images = json.loads((REPAIRS / "native-sources" / f"{doc_id}-formulas.json").read_text())["images"]
    return tuple(Formula(i["page"], tuple(i["bbox"]), i["latex"]) for i in images)


def reviewed_display(doc_id: str, name: str) -> str:
    """A display formula line: its image parts and text-layer words written as one LaTeX expression."""
    latex = json.loads((REPAIRS / "native-sources" / f"{doc_id}-formulas.json").read_text())["displays"][name]
    return f"$$\n{latex}\n$$"


AMENDING_REGULATIONS = "Yönetmelikte Değişiklik Yapan Yönetmeliklerin Yayımlandığı Resmî Gazetelerin"


def rg_table(
    original: tuple[str, str], amended_head: str, rows: list[tuple[str, str, str]], head: str = RG_HEAD
) -> str:
    """Closing Resmî Gazete list; merged cells repeat their label, as in the 16290/21192 rebuilds."""
    lines = [f"|  | {head} | {head} |", "|---|---|---|", "|  | Tarihi | Sayısı |"]
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
    "mevzuat_38416": Spec(
        pdf_sha256="a624d17ee445a16d1bb396ef2193cef6f152b66f6ed1c6c043a846c813d76500",
        pages=9,
        baseline_sha256="5c884300d192fa794ecc5e0e11543d9c0a4abdec581e05184219ad58e8744b5d",
        title_lines=2,
        restored=(),
        blocks=(
            Block(
                RG_HEAD,
                "2.",
                rg_table(("16/3/2021", "31425"), AMENDING_REGULATIONS, [("1.", "29/3/2024", "32504"), ("2.", "", "")]),
            ),
        ),
        layout_not_carried=("bold article/amendment labels and the underlined amended phrases in 14/6, 19/1 and 29/1",),
        anomalies=(
            "The amendment list prints an empty second row",
            "'Eki için tıklayınız' is printed in the PDF; the annex is not part of it",
        ),
    ),
    "mevzuat_10731": Spec(
        pdf_sha256="1c57ca1515bad3fd6cefbb64a686e18ca630a14812887a5aefb1e80d21d7dec6",
        pages=4,
        baseline_sha256="b6824cf8fc906853f81fb9c3878aa9cfa5728d6ae6db6069562581a26aedf9a8",
        title_lines=2,
        restored=(
            (
                "madde",
                "3",
                "Gazete’de yayımlanan Bankaların Özkaynaklarına İlişkin Yönetmeliğe göre hesaplanan konsolide",
            ),
        ),
        joins=("Konsolide Olmayan ve Konsolide Yabancı Para Net Genel Pozisyon/Özkaynak",),
        footnotes=("(1) Bu değişiklik 9/1/2023",),
        blocks=(
            Block(
                RG_HEAD,
                "3.",
                rg_table(
                    ("1/11/2006", "26333"),
                    AMENDING_REGULATIONS,
                    [("1.", "21/3/2014", "28948"), ("2.", "10/12/2022", "32039"), ("3.", "", "")],
                ),
            ),
        ),
        layout_not_carried=(
            "bold article/amendment labels and the underlined amended phrases in 3(d), (e), (g), (i), (l) and 6/4",
        ),
        anomalies=(
            "Madde 3(n) dates the 29511 Sermaye Yeterliliği Yönetmeliği 23/11/2015 (Resmî Gazete 29511 is 23/10/2015)",
            "The amendment list prints an empty third row",
            "'Ekleri için tıklayınız' is printed in the PDF; the annexes are not part of it",
        ),
    ),
    "mevzuat_42359": Spec(
        pdf_sha256="68c95a4e40a2d4d217d9810704a17fcc3de7d30f367511c84f081e07d24066c0",
        pages=3,
        baseline_sha256="c21c18d9b0ee0ff1dacb4bd8c031d7ea4faa592eede300f43c3a9f26a402308e",
        title_lines=4,
        restored=(
            (
                "madde",
                "3",
                "ğ) Özkaynak: 5/9/2013 tarihli ve 28756 sayılı Resmî Gazete’de yayımlanan Bankaların Özkaynaklarına",
            ),
        ),
        joins=("Kredi Müşterilerinden, Ortaklıklardan ve Ortaklardan Sağlanacak Fonlar ve Bunların",),
        layout_not_carried=("bold article labels",),
    ),
    "mevzuat_38568": Spec(
        pdf_sha256="bf55878ede1f7594fbcb4129c2b934ed879c87eba90c4d18a3808f7a062d2e2e",
        pages=7,
        baseline_sha256="d90ede935e70b43b0cfb8cfddee2365b87bec12a8803e62605826cccb0f06e53",
        title_lines=3,
        restored=(),
        joins=("Yurt Dışında Yerleşik Vatandaşlar Mevduat ve Katılım Sistemi (YUVAM) Kapsamında",),
        footnotes=("(1) 6/4/2022 tarihli", "(2) Bu değişiklik 1/6/2023", "(3) 25/5/2023 tarihli"),
        blocks=(
            Block(
                RG_HEAD,
                "2. 25/5/2023",
                rg_table(
                    ("1/4/2021", "31441"),
                    AMENDING_REGULATIONS,
                    [("1.", "6/4/2022", "31801"), ("2.", "25/5/2023", "32201")],
                ),
            ),
        ),
        layout_not_carried=(
            "bold article/amendment labels and the underlined amended phrases in 1/2, 3(ı), 6/1, 11/1 and 12/3(b)",
            "footnote text is italic in the PDF",
        ),
    ),
    "mevzuat_23228": Spec(
        pdf_sha256="d6ca15bd513d41b7ddfb426379baf7d7be866a7762904a4897a7dcb3be279d67",
        pages=11,
        baseline_sha256="bedf3a3319e9d293c3446af9bba2e9ee3c4abfe34257251a7b9df0fa58c1b720",
        title_lines=3,
        restored=(),
        joins=("Mesleki Sorumluluk Sigortası, Gözetim ve Denetim,",),
        footnotes=("(1) 9/1/2020 tarihli", "(2) 17/9/2022 tarihli", "(3) Bu değişiklik yayımı"),
        blocks=(
            Block(
                RG_HEAD,
                "3 17/9/2022",
                rg_table(
                    ("12/1/2017", "29946"),
                    AMENDING_REGULATIONS,
                    [("1", "12/8/2017", "30152"), ("2", "9/1/2020", "31003"), ("3", "17/9/2022", "31956")],
                ),
            ),
        ),
        layout_not_carried=("bold article/amendment labels", "footnote text is italic in the PDF"),
        anomalies=(
            "Amendment labels print 'RG-12/8/2017- 30152' with a space after the hyphen",
            "Madde 21/1 prints 'fıkraları,12 nci' without a space",
            "'Ekleri için tıklayınız.' is printed in the PDF; the annexes are not part of it",
        ),
    ),
    "mevzuat_11203": Spec(
        pdf_sha256="b3371e77abd12a8c8100e3cff686a7bf09417e29a8a71783f4c7471229e9912e",
        pages=14,
        baseline_sha256="e27e195eebae95e56097461ae4d081a6e597623eb0d6a61d506630f47ef4e46b",
        title_lines=2,
        restored=(),
        footnotes=("(1) 14/9/2013 tarihli", "(2) 14/9/2013 tarihli"),
        blocks=(
            Block(
                RG_HEAD,
                "6. 30/10/2022",
                rg_table(
                    ("23/3/2007", "26471"),
                    AMENDING_REGULATIONS,
                    [
                        ("1.", "14/11/2010", "27759"),
                        ("2.", "18/6/2011", "27968"),
                        ("3.", "19/10/2011", "28089"),
                        ("4.", "14/10/2012", "28441"),
                        ("5.", "14/9/2013", "28765"),
                        ("6.", "30/10/2022", "31998"),
                    ],
                ),
            ),
        ),
        layout_not_carried=(
            "bold article/amendment labels and underlined amended phrases",
            "footnote text is italic in the PDF",
        ),
        anomalies=(
            "Madde 8/3 and 14/1 print '©' (apparently a word processor's autocorrection of a bent label); kept",
            "Madde 33/1(d) prints '(c) ve (ç) bentleri bendi'; Madde 36/2(ç) prints '((Değişik ibare'",
            "The footnote separator is a run of hyphens, kept as printed",
        ),
    ),
    "mevzuat_10746": Spec(
        pdf_sha256="6de69c4038d939177c6769296f047ef18ed8a90ff7b70ccb66c71adaa203ba57",
        pages=12,
        baseline_sha256="9bf3581826793316bb86a7a591c86f3bb219646817329be5e0fcb412b0107a3a",
        title_lines=1,
        restored=(
            (
                "madde",
                "8",
                "Gazete’de yayımlanan Bankaların Kredi İşlemlerine İlişkin Yönetmeliğin 19 uncu maddesi",
            ),
        ),
        footnotes=("(1) 5/4/2008 tarihli", "(2) Bu değişiklik 1/1/2020"),
        blocks=(
            Block(
                RG_HEAD,
                "9. 13/2/2019",
                rg_table(
                    ("01/11/2006", "26333"),
                    AMENDING_REGULATIONS,
                    [
                        ("1.", "24/07/2007", "26592"),
                        ("2.", "5/4/2008", "26838"),
                        ("3.", "5/8/2008", "26958"),
                        ("4.", "23/1/2009", "27119"),
                        ("5.", "8/10/2010", "27723"),
                        ("6.", "4/2/2011", "27836"),
                        ("7.", "9/6/2017", "30091"),
                        ("8.", "14/8/2018", "30509"),
                        ("9.", "13/2/2019", "30685"),
                    ],
                ),
            ),
        ),
        layout_not_carried=(
            "bold article/amendment labels, the underlined amended phrase in 8/7, italic footnote text",
        ),
        anomalies=(
            "Madde 3(a) has no closing comma; Madde 8/7 restarts with a capital 'Alım' after a comma",
            "Amendment labels print 'RG-9/6/2017- 30091' with a space after the hyphen",
            "'Yönetmeliğin eklerini görmek için tıklayınız' is printed in the PDF; the annexes are not part of it",
        ),
    ),
    "mevzuat_38797": Spec(
        pdf_sha256="80d4e8b80ad76fd6a944ad41295f96570da3fe9f636b01b2746ad69e797b49e2",
        pages=14,
        baseline_sha256="6101613aa54130888782763119db34a191286bc2a2d9f9d6fd1f04db963721b1",
        title_lines=3,
        restored=(
            (
                "madde",
                "6",
                "esasına göre faaliyette bulunacak şirketlerin ise ticaret ünvanlarında katılım ibaresinin",
            ),
        ),
        joins=("$^{[1]}$ 27/8/2025 tarihli",),
        blocks=(
            # A flush-left link line; as a one-line block it stays its own paragraph.
            Block("Ekleri için tıklayınız.", "Ekleri için tıklayınız.", "Ekleri için tıklayınız."),
            Block(
                RG_HEAD,
                "1 27/8/2025",
                rg_table(("14/7/2021", "31541"), AMENDING_REGULATIONS, [("1", "27/8/2025", "32999")]),
            ),
        ),
        layout_not_carried=("bold article/amendment labels, underlined amended phrases and italic footnote text",),
        anomalies=(
            "Madde 13/5 prints 'üzere; genel esaslar dâhilinde ortaklardan fon sağlayabilir,', the 2025 "
            "amendment phrase of Madde 14/1(d); kept as printed",
            "Madde 14/5 prints 'zorunludur.Yapılacak' and Madde 21/1 '(1)Varlık' without a space",
            "The footnote [1] is printed after the Resmî Gazete table",
            "'Ekleri için tıklayınız.' is printed in the PDF; the annexes are not part of it",
        ),
    ),
    "mevzuat_5411": Spec(
        pdf_sha256="73cc9c5eb135ddc8c9b7ef738eb93df575aed6e157b37d511055884c9aa15876",
        pages=54,
        baseline_sha256="11dd3b9563f28f01619c5172ce1af969b6e13719d9cd524f1072cbd2a109c721",
        title_lines=1,
        restored=(),
        source_kind="generated",
        pdf_url="https://www.mevzuat.gov.tr/File/GeneratePdf?mevzuatNo=5411&mevzuatTur=Kanun&mevzuatTertip=5",
        joins=(
            "5411 SAYILI KANUNA EK VE DEĞİŞİKLİK GETİREN MEVZUATIN VEYA",
            "ANAYASA MAHKEMESİ TARAFINDAN İPTAL EDİLEN HÜKÜMLERİN",
            # Geçici Madde 3/1 restarts mid-sentence on an indented line ("... beyannameyi Kuruma / verirler.").
            "girmesinden itibaren, iki ay içinde faaliyet gösterdikleri alanları bildiren beyannameyi Kuruma",
        ),
        blocks=(
            Block(
                "SINIF UNVAN DERECE ADET",
                "GENEL TOPLAM 867",
                reviewed_table("mevzuat_5411", "cetvel_1"),
                overprinted=("SINIF", "UNVAN", "DERECE", "ADET") * 2,
                merged_cells=False,
            ),
            Block(
                "SINIFI UNVANI DERECESİ ADET",
                "GENEL TOPLAM 415",
                reviewed_table("mevzuat_5411", "cetvel_2"),
                merged_cells=False,
            ),
            Block(
                "POZİSYON UNVANI ADEDİ",
                "TOPLAM 155",
                reviewed_table("mevzuat_5411", "cetvel_3"),
                merged_cells=False,
            ),
            Block(
                "Değiştiren Kanunun/",
                "10765 Geçici Madde 32 25/12/2025",
                reviewed_table("mevzuat_5411", "degisiklik"),
                overprinted=tuple(
                    "Değiştiren Kanunun/ KHK’nin/İptal Eden 5411 Sayılı Kanunun Değişen veya Yürürlüğe Giriş "
                    "Tarihi Anayasa Mahkemesi İptal Edilen Maddeleri Kararının Numarası".split()
                ),
                merged_cells=False,
            ),
        ),
        layout_not_carried=(
            "bold headings, blue endnote links and italic endnote text",
            "the table header GeneratePdf repeats (overprinted) at the top of continuation pages",
        ),
        anomalies=(
            "Geçici Madde 3/1 restarts 'verirler.' on an indented line mid-sentence; joined into one paragraph",
            "Madde 68 and 69 carry a second, plain superscript after the endnote ref ('ister.11', "
            "'gönderilmesi,12'), and Madde 143 prints '(…)61' after '(…)[62]'; kept as printed",
        ),
    ),
    "mevzuat_6362": Spec(
        pdf_sha256="c0f5f3d3cd3ff92039aec6d899cd0a0272524c6930f57e2c8b6ca0c1dc72312f",
        pages=49,
        baseline_sha256="6e4ad927061ea02f3b5c4d326e3c15539500b97fb2467eeb7a4beae322ad788e",
        title_lines=1,
        restored=(),
        source_kind="generated",
        pdf_url="https://www.mevzuat.gov.tr/File/GeneratePdf?mevzuatNo=6362&mevzuatTur=Kanun&mevzuatTertip=5",
        joins=(
            "Sermaye Piyasası Araçlarının İhracı, Kamunun Aydınlatılması ve",
            "Sermaye Piyasalarında Borsalar,",
            "Türkiye Sermaye Piyasaları Birliği, Türkiye Değerleme Uzmanları Birliği,",
            "Merkezî Takas Kuruluşları, Merkezî Saklama Kuruluşları ve",
            "SERMAYE PİYASASI KURULU",
            "HAZİNEYE DEVREDİLEREK BORSA İSTANBUL",
            "ANONİM ŞİRKETİNİN KULLANIMINA",
            "6362 SAYILI KANUNA EK VE DEĞİŞİKLİK GETİREN MEVZUATIN VEYA",
            "ANAYASA MAHKEMESİ TARAFINDAN VERİLEN İPTAL KARARLARININ",
        ),
        blocks=(
            Block(
                "SINIF UNVAN DERECE ADET",
                "TOPLAM 920",
                reviewed_table("mevzuat_6362", "kadro_1"),
                merged_cells=False,
            ),
            Block(
                "İLİ İLÇESİ MAHALLESİ PAFTA ADA PARSEL",
                "Ankara Çankaya/2 Karakusunlar - 13911 4",
                reviewed_table("mevzuat_6362", "tasinmaz_2"),
                merged_cells=False,
            ),
            Block(
                "İLİ İLÇESİ MAHALLESİ PAFTA ADA PARSEL",
                "İstanbul Sarıyer İstinye 50 380 18",
                reviewed_table("mevzuat_6362", "tasinmaz_3"),
                merged_cells=False,
            ),
            Block(
                "İLİ İLÇESİ MAHALLESİ PAFTA ADA PARSEL",
                "İstanbul Sarıyer Mirgün 48 154 120",
                reviewed_table("mevzuat_6362", "tasinmaz_4"),
                merged_cells=False,
            ),
            Block(
                "SINIF UNVAN DERECE ADET",
                "TOPLAM 7",
                reviewed_table("mevzuat_6362", "kadro_5"),
                merged_cells=False,
            ),
            Block(
                "Değiştiren Kanunun/",
                "Madde 11",
                reviewed_table("mevzuat_6362", "degisiklik"),
                merged_cells=False,
            ),
        ),
        layout_not_carried=("bold headings, blue endnote links and italic endnote text",),
        anomalies=(
            "Madde 101/1, 107/2 and 130/5 carry a plain superscript besides or instead of the endnote ref "
            "('olarak;12', 'cezalandırılırlar.15', 'md.)22'); kept as printed",
            "Madde 78/3 opens with '3)' without the opening parenthesis; kept as printed",
        ),
    ),
    "mevzuat_42628": Spec(
        pdf_sha256="bfed75fef1a4e0d085a53fd36c4076201e3a5686b477aed5dcc03446a631533a",
        pages=8,
        baseline_sha256="1753bd25e591fa54270ee9ba4af95ec0d975e62a8459cc95f008870ec81e21b3",
        title_lines=2,
        restored=(
            ("madde", "7", r"KEO^{kp}_{i,p} = \min(1, s_i \ast KEO^{kp}_{0,p})"),
            ("madde", "8", r"OR_{i,p} = \sum_{so=1}^{n} \Delta OD^{so}_{i,p}"),
            ("madde", "9", r"\text{EDD risk tutarı} = \max_{i \in \{1,2,\ldots,6\}}"),
        ),
        joins=("Bankacılık Hesaplarından Kaynaklanan Faiz Oranı Riski Standart",),
        # EK-2 sets its numbered and lettered paragraphs flush left.
        breaks=(
            "1. Her para birimi için",
            "a) Para birimi (p) bazında",
            "b) Para birimi bazında kısa",
            "c) Artan ve azalan eğim",
            "ç) Para birimi bazında artan",
        ),
        blocks=(
            Block(
                "Mevduat sınıfları bazında",
                "Diğer kişi 50 4",
                reviewed_table("mevzuat_42628", "cekirdek_mevduat"),
            ),
            Block(
                "[[F2]] = min (1,",
                "[[F2]] = min (1,",
                reviewed_display("mevzuat_42628", "keo"),
                words=("[[F2]]", "=", "min", "(1,", "[[F3]]", ")"),
            ),
            Block(
                "Faiz oranı şoku senaryoları altında koşullu",
                "6 Kısa Vadeler Aşağı 1,2",
                reviewed_table("mevzuat_42628", "erken_odeme"),
            ),
            Block(
                "[[F4]] [[F5]]",
                "[[F4]] [[F5]]",
                reviewed_display("mevzuat_42628", "na_kp"),
                words=("[[F4]]", "[[F5]]"),
            ),
            Block(
                "[[F10]] = min(1,",
                "[[F10]] = min(1,",
                reviewed_display("mevzuat_42628", "eko"),
                words=("[[F10]]", "=", "min(1,", "[[F11]]", ")"),
            ),
            Block(
                "Faiz oranı şok senaryoları altında erken",
                "6 Kısa Vadeli Aşağı 0,8",
                reviewed_table("mevzuat_42628", "erken_kapanma"),
            ),
            Block(
                "[[F12]] [[F13]]",
                "[[F12]] [[F13]]",
                reviewed_display("mevzuat_42628", "na_mp"),
                words=("[[F12]]", "[[F13]]"),
            ),
            Block(
                "OR$_{i,p}$= [[F18]]",
                "OR$_{i,p}$= [[F18]]",
                reviewed_display("mevzuat_42628", "or"),
                words=("OR", "i,p", "=", "[[F18]]"),
            ),
            Block(
                "İO$_{0,p}$(t$_{o}$) = exp(",
                "İO$_{0,p}$(t$_{o}$) = exp(",
                reviewed_display("mevzuat_42628", "io_0"),
                words=("İO", "0,p", "(t", "o", ")", "=", "exp(-F", "0,p", "(t", "o", ")", "[[F19]]", "t", "o", ")"),
            ),
            Block(
                "İO$_{i,p}$(t$_{o}$) = exp(",
                "İO$_{i,p}$(t$_{o}$) = exp(",
                reviewed_display("mevzuat_42628", "io_i"),
                words=("İO", "i,p", "(t", "o", ")", "=", "exp(-F", "i,p", "(t", "o", ")", "[[F20]]", "t", "o", ")"),
            ),
            Block("[[F22]]", "[[F22]]", reviewed_display("mevzuat_42628", "ed"), words=("[[F22]]",)),
            Block(
                "EDD$_{i,p}$ = [[F23]]",
                "EDD$_{i,p}$ = [[F23]]",
                reviewed_display("mevzuat_42628", "edd"),
                words=("EDD", "i,p", "=", "[[F23]]", "OR", "i,p"),
            ),
            Block("[[F26]]", "[[F26]]", reviewed_display("mevzuat_42628", "edd_risk"), words=("[[F26]]",)),
            Block(
                "Nakit akışlarının yerleştirileceği",
                "(7,5 yıl) (8,5 yıl)",
                reviewed_table("mevzuat_42628", "ek1_vade_dilimleri"),
            ),
            Block("[[F27]]", "[[F27]]", reviewed_display("mevzuat_42628", "r_paralel"), words=("[[F27]]",)),
            Block("[[F29]]", "[[F29]]", reviewed_display("mevzuat_42628", "r_kisa"), words=("[[F29]]",)),
            Block("[[F32]]", "[[F32]]", reviewed_display("mevzuat_42628", "r_uzun"), words=("[[F32]]",)),
            Block("[[F33]]", "[[F33]]", reviewed_display("mevzuat_42628", "r_artan"), words=("[[F33]]",)),
            Block("[[F34]]", "[[F34]]", reviewed_display("mevzuat_42628", "r_azalan"), words=("[[F34]]",)),
        ),
        formulas=reviewed_formulas("mevzuat_42628"),
        layout_not_carried=(
            "bold headings and table borders",
            "formula fonts (bold, upright or italic letters); the formulas' symbols and notation are carried",
        ),
        anomalies=(
            "Madde 7/4(b) and 7/5(b) print the scenario multiplier as 's' with the subscript 'i;' (the semicolon "
            "is set inside the subscript); kept as printed",
            "Madde 9/1(c) sets a blank 3x18 px image between the two closing parentheses of '(F_{i,p}(t_o) )'",
            "Madde 9/1(e) prints one closing parenthesis more than it opens ('EDD_{i,p}))}') and the summation "
            "index as 'p:EDD_{i,p}'; kept as printed",
            "EK-1 prints the first vade orta noktası of the 1-month bucket as '(0,041 7 yıl)'; kept as printed",
            "EK-2 writes decimals with a point (-0.65, 0.9, 0.8, 0.6) and 1(a) writes '(to)' without a subscript; "
            "kept as printed",
        ),
    ),
    "mevzuat_11201": Spec(
        pdf_sha256="67c766ce53c9a23f0ff1bd36b1beb197cbaa2223e404f885653bfe07ae35a537",
        pages=2,
        baseline_sha256="35c572c37c2172242b732c8a65f8b88dcf800e01546a29ffedea268ecf417b2e",
        title_lines=1,
        restored=(),
        source_kind="generated",
        pdf_url="https://www.mevzuat.gov.tr/File/GeneratePdf?mevzuatNo=11201&mevzuatTur=Teblig&mevzuatTertip=5",
        joins=("BANKACILIK VERİ TRANSFER SİSTEMİ KAPSAMINDA YAPILAN RAPORLAMALARDA UYGULANACAK İDARİ PARA",),
        blocks=(
            Block(
                "Kurum Bankacılık Düzenleme ve Denetleme Kurumu",
                "Resmî Gazete Sayısı 26470",
                reviewed_table("mevzuat_11201", "kunye"),
            ),
            Block(
                "Tebliğin Yayımlandığı Resmî Gazete’nin",
                "2.",
                rg_table(
                    ("22/3/2007", "26470"),
                    "Tebliğde Değişiklik Yapan Tebliğlerin Yayımlandığı Resmî Gazetelerin",
                    [("1.", "25/3/2014", "28952"), ("2.", "", "")],
                    head="Tebliğin Yayımlandığı Resmî Gazete’nin",
                ),
            ),
        ),
        layout_not_carried=("bold headings and the boxed identification table's borders",),
        anomalies=(
            "The identification table dates the Resmî Gazete '22/03/2007' and the closing list '22/3/2007'; "
            "both kept as printed",
            "The amendment list prints an empty second row",
        ),
    ),
    "mevzuat_5464": Spec(
        pdf_sha256="3c5248ca4f8d2480d89ca6b2521983d7ef216f530528118c550d724eeb79f92b",
        pages=9,
        baseline_sha256="eeb6dece757f293e953502a284bd19ca53b995aacdb352c507eadc70defb2104",
        title_lines=1,
        restored=(),
        source_kind="generated",
        pdf_url="https://www.mevzuat.gov.tr/File/GeneratePdf?mevzuatNo=5464&mevzuatTur=Kanun&mevzuatTertip=5",
        joins=(
            "5464 SAYILI KANUNA EK VE DEĞİŞİKLİK GETİREN MEVZUATIN VEYA",
            "ANAYASA MAHKEMESİ TARAFINDAN İPTAL EDİLEN HÜKÜMLERİN",
        ),
        blocks=(
            Block(
                "Değiştiren Kanunun/",
                "7247 24 26/6/2020",
                reviewed_table("mevzuat_5464", "degisiklik"),
            ),
        ),
        layout_not_carried=("bold headings, blue endnote links and italic endnote text",),
    ),
}


def pdf_lines(pdf_path: Path, formulas: tuple[Formula, ...] = ()) -> list[list[tuple[float, list[tuple[str, str]]]]]:
    """Per page, physical lines as (xMin, [(word, kind)]) ordered top-down, left-right.

    kind: "" body word, "^" superscript, "_" subscript, "+" body word glued to the subscript before it.
    Image formulas join the lines as [[Fn]] words at their page position.
    """
    bbox = subprocess.check_output(["pdftotext", "-enc", "UTF-8", "-bbox-layout", str(pdf_path), "-"], text=True)
    pages = []
    for number, page in enumerate(bbox.split("<page ")[1:], 1):
        words = [
            (float(x), float(x1), float(y0), float(y1), html.unescape(w)) for x, y0, x1, y1, w in WORD_RE.findall(page)
        ]
        if not words:
            pages.append([])
            continue
        # Superscripts are clearly smaller than the page's body glyphs (7 pt vs 11 pt in mevzuat PDFs). Law
        # PDFs from File/GeneratePdf raise "[n]" endnote references above the line instead of shrinking them.
        body = statistics.median(w[3] - w[2] for w in words)
        cut = body * 0.75

        def raised(w, cut=cut):
            return w[3] - w[2] < cut or re.fullmatch(r"(?:\[\d+\])+", w[4]) is not None

        words += [(*f.bbox[::2], *f.bbox[1::2], f"[[F{n}]]") for n, f in enumerate(formulas, 1) if f.page == number]
        lines: list[list] = []  # [y0, y1, [(x, word, kind, x1, y1)]]
        for x, x1, y0, y1, word in sorted((w for w in words if not raised(w)), key=lambda w: (w[2], w[0])):
            # The text layer sometimes glues a raised marker to the word before it ("(…)61", "Yardımcısı[72]");
            # the merged box is then taller than a body word.
            glued = y1 - y0 > body * 1.15 and re.fullmatch(r"(.+?)((?:\[\d+\])+|\d+)", word)
            parts = [(x, glued[1], "", x1, y1), (x + 0.01, glued[2], "^", x1, y1)] if glued else [(x, word, "", x1, y1)]
            near = (y1 - y0) * 0.3
            line = next((ln for ln in lines if abs((ln[0] + ln[1]) / 2 - (y0 + y1) / 2) < near), None)
            if line is None:
                lines.append([y0, y1, parts])
            else:
                line[2].extend(parts)
        for lone in [ln for ln in lines if len(ln[2]) == 1 and re.fullmatch(r"\d+", ln[2][0][1])]:
            # A body-size number set above the line holding its baseline is an unbracketed ref ("md.)22").
            host = next((ln for ln in lines if ln is not lone and ln[0] < lone[1] < ln[1] - body * 0.15), None)
            if host is not None:
                lines.remove(lone)
                host[2].append((*lone[2][0][:2], "^", *lone[2][0][3:]))
        for x, x1, _y0, y1, word in (w for w in words if raised(w)):
            # A raised marker belongs to the line whose box holds its baseline; one sharing the line's
            # text baseline is a subscript ("İO" + "0,p").
            line = min(lines, key=lambda ln: abs((ln[0] + ln[1]) / 2 - y1) + (0 if ln[0] <= y1 <= ln[1] else 100))
            baseline = max((w[4] for w in line[2] if not w[2] and not FORMULA_TOKEN.fullmatch(w[1])), default=line[1])
            line[2].append((x, word, "_" if abs(y1 - baseline) < 1 else "^", x1, y1))
        rows = []
        for _, _, ws in sorted(lines):
            ws = sorted(ws)
            # A body word set flush against a subscript or an image formula before it ("İO$_{0,p}$(t", "...$,")
            # is written without a space.
            kinds = [
                "+"
                if not w[2]
                and i
                and (ws[i - 1][2] == "_" or FORMULA_TOKEN.fullmatch(ws[i - 1][1]))
                and w[0] - ws[i - 1][3] < 1
                else w[2]
                for i, w in enumerate(ws)
            ]
            rows.append((min(w[0] for w in ws), [(w[1], kind) for w, kind in zip(ws, kinds, strict=True)]))
        pages.append(rows)
    return pages


def line_text(words: list[tuple[str, str]], check: bool = True) -> str:
    text = ""
    for word, kind in words:
        if kind == "^":
            if check and not re.fullmatch(r"(?:\(\d+\))+|(?:\[\d+\])+|\d+", word):
                raise SystemExit(f"unexpected superscript {word!r}")
            text += "$^{" + word + "}$"
        elif kind == "_":
            if check and not re.fullmatch(r"[0io](?:,p)?;?", word):
                raise SystemExit(f"unexpected subscript {word!r}")
            text += "$_{" + word + "}$"
        elif kind == "+":
            text += word
        else:
            text += (" " if text else "") + word
    return text


def markdown_words(markdown: str, merged_cells: bool = True) -> list[str]:
    """Words a reviewed block asserts are in the PDF: merged (left-repeated) and empty cells count once."""
    words = []
    for row in markdown.replace("$^{", " ").replace("$_{", " ").replace("}$", " ").splitlines():
        if re.fullmatch(r"\|(?:-{3,}\|)+", row):
            continue
        if not row.startswith("|"):
            words += row.split()
            continue
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        keep = [cell for i, cell in enumerate(cells) if cell and (not merged_cells or i == 0 or cell != cells[i - 1])]
        words += [w for cell in keep for w in cell.split()]
    return words


def build_candidate(spec: Spec, pages) -> tuple[str, list[str]]:
    """Return (markdown, PDF words outside blocks in candidate order); blocks are verified as they are placed."""
    lines = [line for page in pages for line in page]
    # Body margin: the leftmost line start shared by at least a tenth of all lines (short-item regulations
    # can have more indented paragraph starts than flush-left continuation lines).
    starts = Counter(round(x) for x, _ in lines)
    margin = min(x for x, count in starts.items() if count >= len(lines) / 10)
    paras: list[str] = []
    prose_words: list[str] = []
    joining = after_block = False
    pending = list(spec.blocks)  # in document order; blocks may share a header line ("İLİ İLÇESİ ...")
    index = 0
    while index < len(lines):
        x0, words = lines[index]
        text = line_text(words, check=False)
        block = next((b for b in pending if text.startswith(b.start)), None)
        if block is not None:
            pending.remove(block)
            end = len(lines) - 1
            if block.end is not None:
                end = next(
                    i for i in range(index, len(lines)) if line_text(lines[i][1], check=False).startswith(block.end)
                )
            pdf_words = [w for _, ws in lines[index : end + 1] for w, _ in ws]
            for word in block.overprinted:
                pdf_words.remove(word)
            for head, tail in block.rewraps:
                pdf_words.remove(head)
                pdf_words.remove(tail)
                pdf_words.append(head + tail)
            expected = block.words if block.words is not None else markdown_words(block.markdown, block.merged_cells)
            if Counter(expected) != Counter(pdf_words):
                raise SystemExit(f"block {block.start!r} differs from its PDF lines")
            paras.append(block.markdown)
            index, joining, after_block = end + 1, False, True
            continue
        text = line_text(words)  # prose: every raised word must be a known marker
        prose_words += [w for w, _ in words if not FORMULA_TOKEN.fullmatch(w)]
        if re.fullmatch(r"_{10,}", text):
            text = RULE
        if joining:
            paras[-1] += " " + text
        elif (
            x0 > margin + 10
            or x0 < margin - 5
            or not paras
            or after_block
            or words[0][1]
            or any(text.startswith(b) for b in spec.breaks)
        ):
            # words[0][1]: a line opening with a raised label is an endnote ("[1] 2/7/2018 tarihli ...").
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
    # Each image formula sits on exactly one line: display lines are blocks, the rest are inline in prose.
    placed = Counter(w for _, ws in lines for w, _ in ws if FORMULA_TOKEN.fullmatch(w))
    if placed != Counter(f"[[F{n}]]" for n in range(1, len(spec.formulas) + 1)):
        raise SystemExit("image formulas were not each placed once")

    def inline(match: re.Match) -> str:
        latex = spec.formulas[int(match[1]) - 1].latex
        return f" ${latex}$" if latex else ""  # an empty latex is a blank spacer image

    paras = [re.sub(r" ?\[\[F(\d+)\]\]", inline, para).lstrip(" ") for para in paras]
    return "\n\n".join(paras) + "\n", prose_words


def verify_prose(spec: Spec, candidate: str, prose_words: list[str]) -> None:
    prose = candidate
    for item in (*spec.blocks, *spec.images):
        prose = prose.replace(item.markdown, "")
    for formula in spec.formulas:
        if formula.latex:
            prose = prose.replace(f"${formula.latex}$", "")
    prose = prose.replace("$^{", "").replace("$_{", "").replace("}$", "")
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
    pages = pdf_lines(pdf_path, spec.formulas)
    if spec.formulas:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            found = [
                (page.page_number, *(round(image[k], 1) for k in ("x0", "top", "x1", "bottom")))
                for page in pdf.pages
                for image in page.images
            ]
        if found != [(f.page, *f.bbox) for f in spec.formulas]:
            raise SystemExit("PDF images differ from the reviewed formula list")
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
    if spec.source_kind == "static" and document["file_size"] != len(pdf_bytes):
        raise SystemExit("PDF size differs from the stored download")
    if spec.source_kind == "generated":
        pdf_count = sum(len(ws) for page in pages for _, ws in page)
        stored_count = len(re.sub(r"\|(?:-{3,}\|)+|\|", " ", original).split())
        if abs(pdf_count - stored_count) > max(10, stored_count // 500):
            raise SystemExit(f"generated PDF has {pdf_count} words, stored text {stored_count}; not the same version")
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
        document["extraction_method"] = NEW_METHOD + ("+manual_latex" if spec.formulas else "")
        document["total_pages"] = math.ceil(len(candidate) / PAGE_SIZE)
        tmp = DOCS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(docs, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(DOCS_PATH)
    number = re.sub(r"\D", "", doc_id)
    pdf_url = spec.pdf_url or f"https://www.mevzuat.gov.tr/MevzuatMetin/yonetmelik/7.5.{number}.pdf"
    evidence = {
        "document_id": doc_id,
        "status": "source_reviewed_candidate_awaiting_owner_approval",
        "repair_class": "full_rebuild_from_official_pdf",
        "review_scope": "Assistant technical source review; not independent human/legal approval.",
        "source_url": document["source_url"],
        "source_pdf_url": pdf_url,
        "source_pdf_kind": spec.source_kind,
        "source_pdf_sha256": spec.pdf_sha256,
        "source_pdf_bytes": len(pdf_bytes),
        "source_pdf_pages": spec.pages,
        "source_pdf_matches_stored_file_size": spec.source_kind == "static",
        **(
            {
                "source_pdf_note": "mevzuat.gov.tr regenerates this PDF on every request, so it is not byte-reproducible; "
                f"the reviewed copy is retained as native-sources/{number}-generated.pdf.gz and its word count "
                "matches the stored text"
            }
            if spec.source_kind == "generated"
            else {}
        ),
        "pages_visually_compared": f"1-{spec.pages}",
        "rebuilt_with": "scripts/rebuild_mevzuat_from_pdf.py",
        "text_layer_check": "Outside reviewed blocks the non-whitespace character stream equals pdftotext "
        "-bbox-layout; each block's words equal the PDF lines it replaces",
        "image_only_content": [
            {"page": i.page, "pixels": i.pixels, "sha256": i.sha256, "transcribed_visually": i.markdown}
            for i in spec.images
        ],
        **(
            {
                "image_formulas": len(spec.formulas),
                "image_formula_transcriptions": f"native-sources/{doc_id}-formulas.json",
                "image_formula_check": "Each PDF image is matched by page and bbox and placed once: inline as $...$ "
                "or inside a display $$...$$ block whose text-layer words are declared. The LaTeX is a visual "
                "reading of each image; images have no text layer to compare it with.",
            }
            if spec.formulas
            else {}
        ),
        "original_markdown_sha256": original_sha,
        "candidate_markdown_sha256": candidate_sha,
        "extraction_method_after": NEW_METHOD + ("+manual_latex" if spec.formulas else ""),
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
