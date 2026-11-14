"""Structural parser for Turkish legal/regulatory Markdown."""

from __future__ import annotations

import hashlib
import logging
import re

from pydantic import BaseModel

from bddk_mcp.store.legal_ref import canonical_section_ref

logger = logging.getLogger(__name__)

# Hard upper bound for a single section's span. Legitimate maddeler are a few
# thousand chars; spans beyond this are parser artifacts (typically trailing EK
# annexes swallowed by the last matched heading) and poison section search.
MAX_SECTION_CHARS = 20_000
# The stored section content of a capped span carries a system truncation notice.
# The notice is storage metadata, never part of the legal text: verbatim serving
# strips it and reports truncation separately.
TRUNCATION_NOTICE_PREFIX = "[BÖLÜM KESİLDİ:"
_TRUNCATION_NOTICE_RE = re.compile(r"\s*\[BÖLÜM KESİLDİ:[^\]]*\]\s*$")


def split_section_truncation_notice(content: str) -> tuple[str, str | None]:
    """Split stored section content into exact legal characters and any notice."""
    if not content:
        return content or "", None
    match = _TRUNCATION_NOTICE_RE.search(content)
    if match is None:
        return content, None
    return content[: match.start()], content[match.start() :].strip()


_NOTICE_ORIGINAL_LENGTH_RE = re.compile(r"içerik (\d+) karakterden")


def truncation_notice_original_length(notice: str | None) -> int | None:
    """Return the uncapped span length recorded in a truncation notice, if it states one."""
    if not notice:
        return None
    match = _NOTICE_ORIGINAL_LENGTH_RE.search(notice)
    return int(match.group(1)) if match else None


# Uncovered remainder (preamble, refused numbered bodies, footnotes, text
# past a capped span) is indexed as this type so section FTS can see it
# without inventing madde/paragraf identities.
GOVDE_SECTION_TYPE = "govde"
GOVDE_HEADING = "yapısal başlık yok — gövde/dipnot kalanı"
_MIN_GOVDE_CHARS = 80

# Dash class includes \x96/\x97: Windows-1252 en/em dashes that survive in
# documents extracted from cp1252 source HTML (e.g. mevzuat.gov.tr), where
# every heading reads "MADDE 1 \x96 (1) ...". Bold markers require 2-3
# asterisks ("**MADDE 1** – ..."): a single "* " is a markdown bullet, and
# amendment lists ("* Madde 5 – ... değiştirilmiştir.") must not index as
# headings.  A Markdown heading marker ("## EK-1") and the printed en/em dash
# between EK and its number ("EK–1") are accepted too; without them the last
# article runs on over the annexes until the span cap.
#
# An article inserted by amendment keeps its printed letter ("Madde 66/A –" is
# madde "66/a", never madde 66), and "EK MADDE 1" is its own series (ek_madde),
# as GEÇİCİ MADDE is.  Printed variants: "MADDE 1. –" (1230; the dot counts only
# before a dash, so the "İlke 6." lines of numbered-paragraph Rehber stay
# prose), "GEÇİÇİ MADDE" (19864), "GEÇİCİMADDE" (10735) and "MADDE 16 (1) Her
# ..." with no dash before its first fıkra (1313).
_ARTICLE_REF = r"\d+(?:/?[A-Za-zÇĞİÖŞÜçğıöşü])?"
_HEADING_RE = re.compile(
    r"^(?P<prefix>\s*(?:#{1,6}\s+)?(?:\*{2,3}\s*)?)"
    rf"(?:(?P<gecici>geçi[cç]i\s*madde)\s+(?P<gecici_ref>{_ARTICLE_REF})"
    rf"|(?P<ek_madde>ek\s+madde)\s+(?P<ek_madde_ref>{_ARTICLE_REF})"
    rf"|(?P<madde>madde)\s+(?P<madde_ref>{_ARTICLE_REF})"
    r"|(?P<ilke>ilke)\s+(?P<ilke_ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?)"
    r"|(?P<paragraf>paragraf)\s+(?P<paragraf_ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?)"
    r"|(?P<ek>ek)(?P<ek_separator>[-\s:–—\x96\x97]*)(?P<ek_ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?))"
    r"(?:\.(?=\s*[-–—\x96\x97]))?"
    # A footnote/endnote marker printed right after the number ("MADDE 7$^{(1)}$ –").
    r"(?:\$\^\{[^}]*\}\$)*"
    r"(?:\s*\*{1,3})?"
    r"\s*(?:(?:[-:–—\x96\x97]\s*|(?=\(1\)\s+(?-i:[A-ZÇĞİÖŞÜ])))(?P<title>.*?))?"
    r"\s*\*{0,3}\s*$",
    re.IGNORECASE,
)
_SUBSECTION_RE = re.compile(r"^\s*\((?P<ref>\d+|[A-Za-zÇĞİÖŞÜçğıöşü])\)\s+(?P<title>.*)$")
# A text line that stops mid-sentence: the next line continues it, even when that
# line opens with a "(N)"/"(x)" marker.
_WRAPPED_LINE_RE = re.compile(r"[^.:;)\]*\s]\s*$")
_SENTENCE_BOUNDARY_RE = re.compile(r"[a-zçğıöşü]\.\s+[A-ZÇĞİÖŞÜ]")
# A marker that opens a reference or a table legend rather than an item:
# "(a) ile (e) arasındaki", "(c) – (e)", "(c) bentleri", "(2) numaralı fıkra", "(g) sütunu".
# Lowercase only: a fıkra may open with a capitalised "Bentler …".
_SUBSECTION_REFERENCE_RE = re.compile(
    r"^\((?:\d+|[a-zçğıöşü])\)\s*(?:(?:ile|ilâ|ila|ve|veya|[-–])\s*\(|(?:bent|bend|fıkra|numaralı|alt\s+bent|sütun))"
)
# A draft's "GENEL GEREKÇE" (rationale) closes its last article or annex; the
# rationale is not provision text and stays searchable as govde.
_RATIONALE_RE = re.compile(r"^\**\s*GENEL\s+GEREKÇE\s*\**$")
# "(i)" followed by "(ii)" opens a roman-numeral sub-list inside a fıkra or bent, not bent ı/i.
# The header row of the amendment-history table that ends a consolidated regulation.
_AMENDMENT_TABLE_RE = re.compile(r"^[ \t]*\|.*Yayımlandığı\s+Resm[iî]\s+Gazete", re.MULTILINE)
_ROMAN_SUBITEM_RE = re.compile(r"^\s*\(ii\)\s", re.IGNORECASE)
# "MADDE 146 ila 149 – (… yerine işlenmiştir.)" (6362) names no single article:
# it closes the previous madde, and its note stays searchable as govde.
_ARTICLE_RANGE_RE = re.compile(r"^\s*madde\s+\d+\s+ila\s+\d+\s*[-–—\x96\x97]", re.IGNORECASE)
# A law's attached lists ("(1) SAYILI LİSTE" in 6362, "(I) SAYILI CETVEL" in 5411)
# close its last article; they are not fıkra/bent rows and stay searchable as govde.
_ANNEX_LIST_RE = re.compile(
    r"^\s*\((?:\d+|[IVXLC]+)\)\s+SAYILI\s+(?:LİSTE|CETVEL)\s*$|^\s*\((?:\d+|[ivxlc]+)\)\s+sayılı\s+(?:liste|cetvel)\s*$",
    re.IGNORECASE,
)
# 947 prints most principles and both annexes with a dot ("İlke 1.", "EK 1.")
# but two principles with a colon ("İlke 4:").  The dot form counts only once a
# classic ilke heading proves the document is principle-structured, and only
# before a capitalised title ("Ek 2. bölümünde" stays prose).  Elsewhere (945,
# 946, 948, 950, 954-956, 1311) these lines sit inside a numbered paragraph body
# that only the numbered fallback indexes; parsing them there deletes it.
_DOT_HEADING_RE = re.compile(r"^\s*(?:(?P<ilke>(?i:ilke))|(?i:ek))\s+(?P<ref>\d+)\.\s+(?P<title>[A-ZÇĞİÖŞÜ].*)$")
# An annex number followed directly by its upper-case title ("EK-1 TANIMLAR",
# "EK-2 STRES TESTİ RAPORU" in 946).  The title must be upper case: "EK-1 bir
# ilâ sekizinci fıkralarda belirtilen hükümler …" (902/903) is a cross-reference
# inside a paragraph.
_ANNEX_INLINE_TITLE_RE = re.compile(
    r"^\s*(?:#{1,6}\s+)?(?:\*{2,3}\s*)?(?i:ek)[-\s–—\x96\x97]*(?P<ref>\d+)\s+(?P<title>[^\s(].*?)\s*\*{0,3}\s*$"
)

# An annex printed without a number ("Ek: Açıklamalar", "Ek: Eşleştirme Tablosu"
# in the Genelgeler, "EK- İSEDES RAPORUNUN ŞEKLİ VE İÇERİĞİ" in 951) closes the
# provision before it; it has no identity of its own and stays searchable as
# govde.  An amendment note "Ek:RG-…" is no annex.
_UNNUMBERED_ANNEX_RE = re.compile(
    r"^\s*(?:#{1,6}\s+)?(?:\*{2,3}\s*)?(?i:ek)\s*(?::\s*(?!RG\b|RG-)[A-ZÇĞİÖŞÜ]|[-–—\x96\x97]\s*[A-ZÇĞİÖŞÜ]{2})"
)


def _is_annex_inline_title(title: str) -> bool:
    letters = [ch for ch in title if ch.isalpha()]
    return len(letters) >= 2 and title[:1].isalpha() and title.upper() == title


# Fallback grammar for documents without MADDE/İlke-style headings: Rehber
# number their paragraphs "1. ... 206." and Genelge use decimal outlines
# ("2.", "2.1.").  The trailing dot is required — footnotes ("3 Orta vadeli
# fonlama...") and cross-references ("5 inci maddesinin...") have none.
_NUMBERED_HEADING_RE = re.compile(r"^\s*(?:\*{2,3}\s*)?(?P<top>\d{1,3})(?P<sub>(?:\.\d{1,3})*)\.\s+(?P<title>\S.*)$")
# A sub-heading whose trailing dot the source dropped ("2.2.5 LOJİSTİK VE
# ULAŞTIRMA SEKTÖRÜ" after "2.2.4. İMALAT SEKTÖRÜ" in 1285).  It needs a
# sub-number and an upper-case title, so amounts ("2.5 milyon TL") and
# footnotes stay prose; the sequence check still applies.
_NUMBERED_UNDOTTED_SUBHEADING_RE = re.compile(
    r"^\s*(?:\*{2,3}\s*)?(?P<top>\d{1,3})(?P<sub>(?:\.\d{1,3})+)\s+(?P<title>[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ ,'’-]{3,})$"
)
# A lettered part heading of a decimal outline ("B - ÇEVREYE ÖNEMLİ ZARAR
# VERMEME KRİTERLERİ" in 1285) closes the outline entry before it; it has no
# identity of its own.
# A chapter heading of a decimal outline without its trailing dot ("2.1 Kredi
# Riski Yönetimi ve Kültürü" in 1041).  It only ever divides: it is honoured
# solely far behind a globally numbered body, like the dotted outline headings.
_NUMBERED_DOTLESS_OUTLINE_RE = re.compile(r"^\s*(?:\*{2,3}\s*)?(?P<top>\d{1,3})(?:\.\d{1,3})+\s+[A-ZÇĞİÖŞÜ][^.:;]*$")
_LETTER_PART_HEADING_RE = re.compile(r"^\s*[A-ZÇĞİÖŞÜ]\s+[-–—]\s+[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ ,'’-]{3,}$")
# A second family of Rehber numbers the same paragraphs with a dash ("1-  Bu
# rehberin amacı...").  Sub-numbering is deliberately not accepted here: no
# document in the corpus writes "2.1-", while "1.500- TL" style amounts would
# match if it were allowed.
_NUMBERED_DASH_HEADING_RE = re.compile(r"^\s*(?:\*{2,3}\s*)?(?P<top>\d{1,3})(?P<sub>)-\s+(?P<title>\S.*)$")
# A top-level candidate must advance the sequence by at most this much: real
# Rehber paragraphs run 1..N with at most a hole from a lost OCR line (950
# skips "3."), while list items restarting at 1 inside a section fall behind
# the sequence and are skipped.
_NUMBERED_MAX_GAP = 2
# A document whose candidates mostly violate the sequence numbers its lists
# the same way as its headings (905 restarts numbering per BÖLÜM); indexing a
# guessed subset would attach wrong content to refs, so refuse the document.
_NUMBERED_MIN_SECTIONS = 3
_NUMBERED_MAX_SKIP_RATIO = 0.2
# "N-" is how this corpus enumerates short lists as well as how a few Rehber
# number their paragraphs, so it needs more evidence that the run really is the
# document's backbone.  The measured split is wide: the genuine dash bodies run
# 31-213 paragraphs, while every dash list (audit-firm names in 1138/803,
# effective-date clauses in the Kurul Kararı) stops at 9.
_NUMBERED_DASH_MIN_SECTIONS = 20
_NUMBERED_STYLES = (
    (_NUMBERED_HEADING_RE, _NUMBERED_MIN_SECTIONS),
    (_NUMBERED_DASH_HEADING_RE, _NUMBERED_DASH_MIN_SECTIONS),
)
# The accepted run must also start near the top of the document.  A nested
# sub-list or an annex template (946's "EK-2 STRES TESTİ RAPORU", 1167's
# "101- (b)(ii)" bullets) is internally well-formed, so the sequence and skip
# gates both pass, but it sits at the very end of a body whose real paragraphs
# use a style this grammar does not recognize.  Indexing it would bind
# "946 paragraf 4" to an annex fragment.  Measured over the seed corpus the
# split is unambiguous: every genuine fallback document starts by 34%, the two
# artifacts start at 93% and 97%.
_NUMBERED_MAX_FIRST_OFFSET_RATIO = 0.4

# KGK's TMS/TFRS translations (kgk_tms_12) print the paragraph number bare:
# "1 Bu Standart …", an inserted paragraph with its letter ("29A Bazı …", stored
# as 29/a) and a deleted run as one range ("69–70 [Silinmiştir]").  A bare number
# also opens example items ("1 Bir makinenin maliyeti …"), dates, amounts and
# wrapped lines, so this grammar is tried only when the dot and dash forms find
# nothing, and only lines that follow a finished line can start a paragraph.
_NUMBERED_BARE_HEADING_RE = re.compile(
    r"^\s*(?P<top>\d{1,3})(?P<letter>[A-Z]?)"
    r"(?:\s*[-–—]\s*(?P<last>\d{1,3})(?P<last_letter>[A-Z]?))?"
    r"\s+(?P<title>\S.*)$"
)
# A line ending in a word, number or comma runs on: "… (bakınız: TMS" followed by
# "21 Kur Değişiminin Etkileri)" is a wrapped standard name, not paragraph 21.
# Every paragraph and range line of kgk_tms_12 follows a blank line, ".", "]" or "”".
_MID_SENTENCE_END_RE = re.compile(r"[\w,]$")
# "51C paragrafını açıklayan örnek" and "52A ve 57A paragraflarını açıklayan
# örnek" head an example that cites a paragraph; no paragraph opens that way.
_PARAGRAPH_CITATION_TITLE_RE = re.compile(r"^(?:(?:ve|ile|ila|veya)\s+\d{1,3}[A-Z]?\s+)?paragraf", re.IGNORECASE)
# A standard runs to dozens of paragraphs (TMS 12 has 140, 43 of them
# inserted); no other seed document that reaches this grammar has even 5 bare
# numbers in sequence.
_NUMBERED_BARE_MIN_SECTIONS = 20

# Rehber can interleave principles and a globally numbered paragraph body.
# Decimal outlines / KISIM headings close a principle but are not themselves
# numbered legal paragraphs. Keep them as span boundaries, not citation IDs.
_GUIDELINE_PARAGRAPH_RE = re.compile(r"^\s*(?:\*{2,3}\s*)?(?P<ref>\d{1,3})\.\s+(?P<title>\S.*)$")
_GUIDELINE_DASH_PARAGRAPH_RE = re.compile(r"^\s*(?:\*{2,3}\s*)?(?P<ref>\d{1,3})-\s+(?P<title>\S.*)$")
_GUIDELINE_DIVIDER_RE = re.compile(r"^(?:[A-ZÇĞİÖŞÜ]+\s+(?:KISIM|BÖLÜM)|\d+(?:\.\d+)+\.?\s+[A-ZÇĞİÖŞÜ].*)$")


class DocumentSection(BaseModel):
    """A section extracted from a legal/regulatory Markdown document."""

    doc_id: str
    section_type: str
    section_ref: str
    heading: str = ""
    start_char: int
    end_char: int
    content: str
    content_hash: str
    page_start: int | None = None
    page_end: int | None = None


def extract_document_sections(doc_id: str, text: str) -> list[DocumentSection]:
    """Extract legal sections from Markdown text."""
    if not text:
        return []

    matches = _find_section_starts(text)
    if any(start["section_type"] == "ilke" for start in matches):
        matches = _find_section_starts(text, dot_headings=True)
    if all(start.get("boundary_only") for start in matches):
        # Only boundaries (an unnumbered "Ek: Açıklamalar" line): the document has
        # no classic structure, and the boundaries close the numbered body's rows.
        boundaries = matches
        matches = _find_numbered_paragraph_starts(text) or _find_bare_paragraph_starts(text)
        annex = next((start["start_char"] for start in boundaries if start.get("unnumbered_annex")), None)
        if not matches and annex is not None:
            # The body may end where an unnumbered annex starts whose own outline
            # restarts the numbering (951's "EK- İSEDES RAPORUNUN ŞEKLİ VE İÇERİĞİ").
            matches = _find_numbered_paragraph_starts(text, region_end=annex)
        if matches:
            matches = sorted(matches + boundaries, key=lambda start: start["start_char"])
    elif _is_principle_structured(matches):
        matches = _merge_guideline_paragraphs(text, matches)
    else:
        # A document can carry classic headings for its annexes only (1040's
        # "Ek 1:" starts at 80%, 1041's at 90%), leaving a numbered body that
        # the classic grammar cannot see.  Parse that body and keep both; the
        # body run is bounded by the first classic heading so annex-internal
        # numbering is never swept in.  No seed document pairs a numbered body
        # with a classic "Paragraf N" heading, the one shape that could repeat
        # a (type, ref) pair here, and content_hash keeps the identity key
        # unique even then — so a repeat would read as ambiguous, not collide.
        classic_start = min(start["start_char"] for start in matches)
        if classic_start > len(text) * _NUMBERED_MAX_FIRST_OFFSET_RATIO:
            matches = _find_numbered_paragraph_starts(text, region_end=classic_start) + matches
    if not matches and len(text) > 1000:
        logger.warning(
            "extract_document_sections: no section headings matched (%d chars); "
            "indexing uncovered text as govde windows",
            len(text),
        )
    sections: list[DocumentSection] = []
    # A section's identity is (doc_id, section_type, section_ref, content_hash):
    # lookups key on it and chunks bind to sections through section_content_hash.
    # Consolidated tebliğ repeat an identical closing article once per amendment,
    # so emitting a row per occurrence yields rows no lookup can tell apart.
    # Keep the first.
    seen_identities: set[tuple[str, str, str]] = set()
    level1_capped_end: int | None = None
    # The stored window end of each open provision level that was capped.
    capped_window_ends: dict[int, int] = {}
    for index, start in enumerate(matches):
        if start.get("boundary_only"):
            continue
        if (
            start["level"] == 2
            and not start.get("sequence_validated")
            and level1_capped_end is not None
            and start["start_char"] >= level1_capped_end
        ):
            # Subsection markers found beyond a capped parent span are annex
            # artifacts (tables/lists swallowed by the last heading), not real
            # fıkra/bent rows; indexing them collides with genuine refs.
            # Numbered-outline children are exempt: they were accepted by the
            # sequence check, and a top-level outline span routinely exceeds
            # the cap while its dotted children remain genuine.
            continue
        end_char = _section_end_char(matches, index, len(text))
        if start.get("isolated_headings"):
            end_char = _trim_trailing_heading_blocks(text, start["start_char"], end_char)
        else:
            end_char = _trim_trailing_heading_leak(
                text,
                start["start_char"],
                end_char,
                before_article=_ends_at_article(matches, index),
            )
        start_char, heading = start["start_char"], start["heading"]
        if start["section_type"] in _ARTICLE_TITLE_TYPES and not _carries_inline_title(text, start_char, heading):
            # The title line stays outside every earlier section (the trimming
            # above left it to none of them).
            floor = max((section.end_char for section in sections), default=0)
            title = _article_title_above(text, min(floor, start_char), start_char)
            if title is not None:
                heading, start_char = title
        if start["section_type"] == "ek" and not heading:
            heading = _annex_title_below(text, start_char, end_char)
        truncated_from = None
        if start["section_type"] in {"fikra", "bent"}:
            # A nested row ends with its parent's stored window: the text past a
            # capped provision's window is served by the govde rows that continue
            # it, never as part of a fıkra that started inside the window
            # (21192 EK-2's netting fıkra (2) ran 8,500 characters on over the
            # annex's numbered paragraphs).  A row longer than the cap keeps its
            # truncation notice, as before; a shorter one claims no longer span,
            # because its next marker past the window is no reliable end (the
            # annex's "N." paragraphs are no markers).
            for level, window_end in capped_window_ends.items():
                if level < start["level"] and start_char < window_end < end_char:
                    if end_char - start_char > MAX_SECTION_CHARS:
                        truncated_from = truncated_from or end_char - start_char
                    end_char = window_end
        if truncated_from is None and end_char - start_char > MAX_SECTION_CHARS:
            truncated_from = end_char - start_char
            logger.warning(
                "extract_document_sections: capping oversized section span %d-%d (%d chars) to %d",
                start_char,
                end_char,
                truncated_from,
                MAX_SECTION_CHARS,
            )
            # The window ends at a paragraph, line or word break at or before the
            # cap, never inside a word; its continuation starts exactly there.
            end_char = _govde_window_end(text, start_char, end_char)
        if start["level"] == 1:
            level1_capped_end = end_char if truncated_from else None
        capped_window_ends = {level: end for level, end in capped_window_ends.items() if level < start["level"]}
        if truncated_from:
            capped_window_ends[start["level"]] = end_char
        content = text[start_char:end_char].strip()
        if not content:
            continue
        if truncated_from is not None:
            # The marker travels inside the stored content so every consumer
            # (get_document_section, search previews) sees the truncation.
            content += (
                f"\n\n{TRUNCATION_NOTICE_PREFIX} içerik {truncated_from} karakterden "
                f"{end_char - start_char} karaktere kısaltıldı — tam metin için get_bddk_document kullanın]"
            )
        content_hash = _content_hash(content)
        identity = (start["section_type"], start["section_ref"], content_hash)
        if identity in seen_identities:
            logger.info(
                "extract_document_sections: dropping repeated %s %s at %d (identical to an earlier span)",
                start["section_type"],
                start["section_ref"],
                start["start_char"],
            )
            continue
        seen_identities.add(identity)
        sections.append(
            DocumentSection(
                doc_id=doc_id,
                section_type=start["section_type"],
                section_ref=start["section_ref"],
                heading=heading,
                start_char=start_char,
                end_char=end_char,
                content=content,
                content_hash=content_hash,
            )
        )
    sections.extend(_remainder_govde_sections(doc_id, text, sections, seen_identities))
    return sections


# Mevzuat prints an article's title on its own line above "MADDE N –".  The
# title is the article's heading, whatever the MADDE line carries after the
# dash (its first fıkra "(1) Bu …", an amendment note "(Değişik:RG-…)", or a
# Kanun's first sentence), and the section starts at the title line.
_ARTICLE_TITLE_TYPES = frozenset({"madde", "gecici_madde", "ek_madde"})
_PART_CAPTION_RE = re.compile(r"BÖLÜM|KISIM|\bBölüm\b|\bKısım\b")
# Markdown heading and emphasis markup around a title line ("## Yürürlük",
# "**Dayanak**", "### **KDA …**"); it is layout, not part of the title.
_TITLE_MARKUP_RE = re.compile(r"^#{1,6}\s+|^\*{1,3}\s*|\s*\*{1,3}$")
_INNER_EMPHASIS_RE = re.compile(r"\s*\*{2,3}\s*")
# A title line set apart directly above an article heading may run longer than
# a leaked fragment ("Faiz oranı riski sermaye yükümlülüğü hesaplamasında …
# kaynaklı pozisyonlar", 15 words) and may open with a date or a rate ("1/1/2018
# tarihi itibarıyla …", "%0 volatilite …").  The longest seed title has 19 words.
_ARTICLE_TITLE_MAX_WORDS = 24
_CLAUSE_END_RE = re.compile(r"[;:,]$")
# A title printed on the MADDE line is a bare phrase; a first sentence carries punctuation.
_INLINE_TITLE_PUNCTUATION_RE = re.compile(r"[.;:,!?…]")
_CAPTION_WITH_NAME_RE = re.compile(r"(?:BÖLÜM|KISIM)\s+\S")
# A caption block is the part caption, one to three lines of its name, and the title.
_CAPTION_BLOCK_MIN_LINES = 3
_CAPTION_BLOCK_MAX_LINES = 5
# Title-case captions keep short connectives in lower case ("… Grubu ve
# Mensupları ile İlişkiler"); these words do not make a line sentence case.
_SENTENCE_CASE_MIN_WORD = 4
_TITLE_CASE_LOWER_WORDS = frozenset({"için", "veya", "ilişkin", "dair", "olan", "olarak", "üzere", "gibi", "kadar"})


def _strip_title_markup(line: str) -> str:
    previous = None
    while previous != line:
        previous, line = line, _TITLE_MARKUP_RE.sub("", line).strip()
    return line


def _is_article_title_line(line: str) -> bool:
    """Whether a line (markup stripped) reads as an article title rather than provision text."""
    if (
        not (line[:1].isupper() or line[:1].isdigit() or line[:1] == "%")
        or line.upper() == line
        # A line that ends a clause introduces a list ("… hususlara uyar:"); it is no title.
        or _CLAUSE_END_RE.search(line)
    ):
        return False
    if line[:1].isupper() and _is_leaked_trailing_line(line):
        return True
    return not (
        line.startswith("(")
        or _SENTENCE_END_RE.search(line)
        or _FORMULA_LINE_RE.search(line)
        or _GLOSSARY_LINE_RE.match(line)
        or _HEADING_RE.match(line)
        or not any(ch.isalpha() for ch in line)
        or len(line.split()) > _ARTICLE_TITLE_MAX_WORDS
    )


def _is_part_caption(line: str) -> bool:
    """A part caption line ("İKİNCİ BÖLÜM"), not provision text that cites one ("… Üçüncü Bölümde …")."""
    return bool(_PART_CAPTION_RE.search(line)) and _is_leaked_trailing_line(line)


def _article_title_line_count(lines: list[str]) -> int:
    """How many trailing lines (markup stripped, the last non-blank) form an article's title.

    One line, or two when a title wraps: a first half that reads as a title
    and a second half that continues it in lower case ("… sermaye" /
    "yükümlülüğü").  A title set directly under another text line is no title
    (it may be the last half of a wrap, "… Riske Esas Tutara" / "Dair Üst
    Sınır"); only a part caption may sit there.  Returns 0 when there is none.
    """
    title = lines[-1]
    if title[:1].islower():
        if len(lines) < 2 or not lines[-2] or (len(lines) > 2 and lines[-3]):
            return 0
        first, joined = lines[-2], f"{lines[-2]} {title}"
        if (
            _is_article_title_line(first.removesuffix(","))
            and _is_article_title_line(joined)
            and not _PART_CAPTION_RE.search(joined)
        ):
            return 2
        return 0
    above = [line for line in lines[:-1] if line]
    # A structural caption ("İKİNCİ BÖLÜM", "GENEL HÜKÜMLER", a document title) is not the article's.
    if not _is_article_title_line(title) or _PART_CAPTION_RE.search(title):
        return 0
    if above and _is_part_caption(above[-1]):
        # The caption of a part ("Genel Hükümler" under "BİRİNCİ BÖLÜM") is not the
        # article's either, but a caption that carries its name on its own line
        # ("BEŞİNCİ BÖLÜM Standart Yaklaşım Yönteminde Korelasyonlar") leaves the
        # sentence-case line under it to the article.
        return 1 if _CAPTION_WITH_NAME_RE.search(above[-1]) and _is_sentence_case(title) else 0
    if lines[-2:-1] and lines[-2]:
        if len(above) > 1 and _is_part_caption(above[-2]):
            return 1
        # A sentence-case title set directly under a caption block whose name wraps
        # in title case ("ÜÇÜNCÜ BÖLÜM" / "Sistemik Önemli Bankalara Ait …," /
        # "Hesaplama ve Bildirim Dönemi" / "Sistemik önemli banka tamponu"), or under
        # an upper-case document title.
        block = _trailing_block(lines)
        if (
            _CAPTION_BLOCK_MIN_LINES <= len(block) <= _CAPTION_BLOCK_MAX_LINES
            and (_is_part_caption(block[0]) or block[0].upper() == block[0])
            and not any(_is_sentence_case(line) for line in block[1:-1])
            and _is_sentence_case(title)
        ):
            return 1
        return 0
    return 1


def _carries_inline_title(text: str, start_char: int, heading: str) -> bool:
    """Whether the MADDE line prints the article's own title ("Madde 9 - Karşılık ayrılması").

    Such a line is complete: a blank line or the first fıkra follows it.  A
    first sentence that wraps ("Madde 26. – Bu Kanunun 14 üncü maddesi ve …")
    or an amendment note is no title, and the title line above then heads the article.
    """
    words = heading.split()
    if (
        not heading[:1].isupper()
        or len(words) > _LEAKED_TITLE_MAX_WORDS
        or _INLINE_TITLE_PUNCTUATION_RE.search(heading)
        or not _is_article_title_line(heading)
    ):
        return False
    line_end = text.find("\n", start_char)
    if line_end == -1:
        return True
    next_end = text.find("\n", line_end + 1)
    following = text[line_end + 1 : next_end if next_end != -1 else len(text)].strip()
    return not following or following.startswith("(")


def _trailing_block(lines: list[str]) -> list[str]:
    start = len(lines)
    while start > 0 and lines[start - 1]:
        start -= 1
    return lines[start:]


def _is_sentence_case(line: str) -> bool:
    """Whether a line is written in sentence case ("Kredilerin izlenmesi"), not title case."""
    return any(
        len(word) >= _SENTENCE_CASE_MIN_WORD and word[:1].islower() and word not in _TITLE_CASE_LOWER_WORDS
        for word in re.findall(r"[^\W\d_]+", line)
    )


def _article_title_above(text: str, floor: int, start_char: int) -> tuple[str, int] | None:
    """Return an article's title above its MADDE line and the offset where the title starts.

    The title must lie after the end of every earlier section (``floor``) and
    end on the last non-blank line before the article start.  The returned
    heading is the title without Markdown markup, a wrapped title joined by a space.
    """
    offsets: list[int] = []
    lines: list[str] = []
    position = floor
    for line in text[floor:start_char].splitlines(keepends=True):
        offsets.append(position)
        lines.append(_strip_title_markup(line.strip()))
        position += len(line)
    while lines and not lines[-1]:
        lines.pop()
    if not lines:
        return None
    count = _article_title_line_count(lines)
    if not count:
        return None
    title = " ".join(lines[-count:])
    return _INNER_EMPHASIS_RE.sub(" ", title).strip(), offsets[len(lines) - count]


# An amendment note printed under an annex number ("(Değişik:RG-4/2/2022-31740)$^{(1)}$").
_AMENDMENT_NOTE_RE = re.compile(r"^\(\s*(?:Değişik|Ek|Mülga)\s*:", re.IGNORECASE)


def _annex_title_below(text: str, start_char: int, end_char: int) -> str:
    """Return the title printed under an annex number line ("EK-2" / "(Değişik:…)" / "KARŞI TARAF KREDİ RİSKİ").

    The title is the first non-blank line after the annex line that is not an
    amendment note, when that line reads as a title (upper case is the norm for
    annex titles).  Only the heading comes from it; the section's span is unchanged.
    """
    line_end = text.find("\n", start_char, end_char)
    if line_end == -1:
        return ""
    for raw in text[line_end + 1 : end_char].splitlines():
        line = _strip_title_markup(raw.strip())
        if not line or _AMENDMENT_NOTE_RE.match(line):
            continue
        title = _INNER_EMPHASIS_RE.sub(" ", line).strip()
        if _ADDRESSEE_LINE_RE.search(title):
            # A form's addressee ("… KURULU BAŞKANLIĞINA" under 16071's EK-4) is no title.
            return ""
        return title if _is_annex_title_line(title) else ""
    return ""


# The dative ending of an office or institution an annex form is addressed to.
_ADDRESSEE_LINE_RE = re.compile(r"(?:L[IİUÜ][ĞG][IİUÜ]N[AE]|KURUMUNA|BANKASINA)$")


def _is_annex_title_line(line: str) -> bool:
    """The article-title shape rules, without refusing an upper-case line."""
    return not (
        not (line[:1].isupper() or line[:1].isdigit())
        or _CLAUSE_END_RE.search(line)
        or line.startswith(("(", "|"))
        or _SENTENCE_END_RE.search(line)
        or _FORMULA_LINE_RE.search(line)
        or _GLOSSARY_LINE_RE.match(line)
        or _HEADING_RE.match(line)
        or not any(ch.isalpha() for ch in line)
        or len(line.split()) > _ARTICLE_TITLE_MAX_WORDS
    )


def _remainder_govde_sections(
    doc_id: str,
    text: str,
    sections: list[DocumentSection],
    seen_identities: set[tuple[str, str, str]],
) -> list[DocumentSection]:
    """Index uncovered spans as govde windows. Refs are window numbers, not print numbers."""
    remainder: list[DocumentSection] = []
    ref = 0
    for start_char, end_char in _uncovered_spans(len(text), sections):
        cursor = start_char
        while cursor < end_char:
            window_end = _govde_window_end(text, cursor, end_char)
            content = text[cursor:window_end].strip()
            if len(content) < _MIN_GOVDE_CHARS:
                cursor = window_end
                continue
            ref += 1
            content_hash = _content_hash(content)
            identity = (GOVDE_SECTION_TYPE, str(ref), content_hash)
            if identity in seen_identities:
                cursor = window_end
                continue
            seen_identities.add(identity)
            remainder.append(
                DocumentSection(
                    doc_id=doc_id,
                    section_type=GOVDE_SECTION_TYPE,
                    section_ref=str(ref),
                    heading=GOVDE_HEADING,
                    start_char=cursor,
                    end_char=window_end,
                    content=content,
                    content_hash=content_hash,
                )
            )
            cursor = window_end
    return remainder


def _govde_window_end(text: str, start_char: int, end_char: int) -> int:
    """End a capped window at a paragraph, line or word break, in that order.

    A cut at exactly MAX_SECTION_CHARS splits a word ("döne" + "mlerde"), and
    then neither window contains it.  The break is looked for only in the
    window's second half, so no window shrinks below half the cap; a span with
    no break there keeps the exact cut.
    """
    limit = start_char + MAX_SECTION_CHARS
    if limit >= end_char:
        return end_char
    floor = start_char + MAX_SECTION_CHARS // 2
    for separator in ("\n\n", "\n", " "):
        cut = text.rfind(separator, floor, limit)
        if cut != -1:
            return cut + len(separator)
    return limit


def _uncovered_spans(text_len: int, sections: list[DocumentSection]) -> list[tuple[int, int]]:
    if text_len <= 0:
        return []
    if not sections:
        return [(0, text_len)]
    merged: list[tuple[int, int]] = []
    for start, end in sorted((section.start_char, section.end_char) for section in sections):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < text_len:
        gaps.append((cursor, text_len))
    return gaps


def _find_section_starts(text: str, *, dot_headings: bool = False) -> list[dict]:
    starts: list[dict] = []
    char_pos = 0
    current_major: str | None = None
    previous = ""
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        stripped = line.strip()
        major = _parse_major_heading(stripped, dot_headings=dot_headings)
        if major is not None:
            current_major = None if major.get("boundary_only") else major["section_type"]
            starts.append({**major, "start_char": char_pos, "level": 1})
        elif current_major is not None:
            subsection = _parse_subsection(stripped)
            # A wrapped cross-reference ("… 7 nci maddenin\n(2) numaralı fıkrası") or a sentence
            # opening with one ("(a) ile (e) arasındaki") is prose, not a fıkra or bent.
            wrapped = _WRAPPED_LINE_RE.search(previous) is not None
            if (
                subsection is not None
                and not wrapped
                and not _SUBSECTION_REFERENCE_RE.match(stripped)
                and not (subsection["section_ref"] == "i" and _opens_roman_sublist(lines, index, dot_headings))
            ):
                starts.append({**subsection, "start_char": char_pos, "level": 2})
        previous = "" if major is not None else stripped
        char_pos += len(line)
    starts = _without_quoted_only_articles(text, _without_annex_captions(starts))
    return _without_amendment_footnotes(_with_amendment_table_boundary(text, _without_quoted_articles(starts)))


# An amendment footnote printed as a numbered marker: "(1) Bu değişiklik 1/1/2012 tarihinde yürürlüğe girer."
_AMENDMENT_FOOTNOTE_RE = re.compile(r"^Bu\s+değişiklik\b.*\byürürlüğe\s+girer\.?$")
_INLINE_FIKRA_RE = re.compile(r"^\((\d+)\)\s")


def _without_amendment_footnotes(starts: list[dict]) -> list[dict]:
    """Keep amendment footnotes out of the fıkra rows.

    A consolidated regulation prints its amendment footnotes with the same
    "(N)" markers as fıkras ("MADDE 6 – (1) Bu Yönetmelik hükümlerini …
    yürütür." / "(1) Bu değişiklik 1/1/2012 tarihinde yürürlüğe girer.").  A
    marker that does not advance its article's fıkra numbering (an inline
    "(1)" on the MADDE line counts as fıkra 1), and a "(N) Bu değişiklik …
    yürürlüğe girer." line, is no fıkra.  After the last article, or the last
    provision of the document, the first such footnote closes the provision,
    and the footnotes after it stay searchable as govde.
    """
    footnotes: set[int] = set()
    last_fikra = 0
    in_article = False
    for index, start in enumerate(starts):
        if start["level"] == 1:
            in_article = start["section_type"] in _ARTICLE_TITLE_TYPES and not start.get("boundary_only")
            inline = _INLINE_FIKRA_RE.match(start.get("heading") or "")
            last_fikra = int(inline.group(1)) if inline else 0
            continue
        if start["section_type"] != "fikra":
            continue
        number = int(start["section_ref"])
        if _AMENDMENT_FOOTNOTE_RE.match(start["heading"]) or (in_article and number <= last_fikra):
            footnotes.add(index)
        else:
            last_fikra = number
    if not footnotes:
        return starts
    closing = [
        index
        for index, start in enumerate(starts)
        if start["level"] == 1 and not start.get("boundary_only") and start["section_type"] in _ARTICLE_TITLE_TYPES
    ][-1:]
    closing += [index for index, start in enumerate(starts) if start["level"] == 1 and not start.get("boundary_only")][
        -1:
    ]
    boundaries: set[int] = set()
    for parent in closing:
        for index in range(parent + 1, len(starts)):
            if starts[index]["level"] == 1:
                break
            if index in footnotes:
                boundaries.add(index)
                break
    kept: list[dict] = []
    closed = False
    for index, start in enumerate(starts):
        if start["level"] == 1:
            closed = False
        if index in boundaries:
            kept.append({"section_type": "ek", "boundary_only": True, "start_char": start["start_char"], "level": 1})
            closed = True
            continue
        if index in footnotes or (closed and start["level"] > 1):
            continue
        kept.append(start)
    return kept


def _without_quoted_only_articles(text: str, starts: list[dict]) -> list[dict]:
    """Drop the articles of a document whose article run has no first article.

    An instrument's own articles start at madde 1.  A Genelge that quotes
    another regulation ("… 12 nci ve 13 üncü maddelerinde de aşağıdaki
    hükümlere yer verilmiştir: MADDE 12 – … MADDE 13 – …", 1171) has no article
    of its own: the quoted MADDE lines and their fıkra/bent markers are not its
    provisions, and the document is indexed as text windows.  A document with a
    geçici or ek madde keeps its articles, and so does an excerpt that does not
    open with a sentence introducing the quotation (its last line before the
    first article, or before that article's title, ends with a colon).
    """
    articles = [s for s in starts if s["level"] == 1 and not s.get("boundary_only") and s["section_type"] == "madde"]
    if (
        not articles
        or any(_article_number(s["section_ref"]) == 1 for s in articles)
        or any(s["level"] == 1 and s["section_type"] in {"gecici_madde", "ek_madde"} for s in starts)
        or not _introduces_quotation(text[: articles[0]["start_char"]])
    ):
        return starts
    kept: list[dict] = []
    dropping = False
    for start in starts:
        if start["level"] == 1:
            dropping = start["section_type"] == "madde" and not start.get("boundary_only")
        if not dropping:
            kept.append(start)
    return kept


def _without_annex_captions(starts: list[dict]) -> list[dict]:
    """Keep a colon-form annex heading ("EK: 1") only where it opens the next annex.

    Mevzuat_18990 heads its annexes "EK: 1" and "EK: 2".  The same form also
    echoes an annex just opened ("EK-1" / "(Değişik:RG-…)" / "Ek:1" in 19498)
    and serves as a page caption repeated over an annex's pages (934 prints
    "Ek:1", "Ek:1", "Ek:2", "Ek:2", "Ek:2", "Ek:3" …), and a Genelge's
    explanatory annex numbers its own sub-annexes that way (905).  An echo is
    dropped; a colon-form line that does not advance the annex numbering, or
    that no article heading precedes, marks the whole document's colon-form
    lines as captions, and none of them is a heading.
    The text under a dropped line stays in the section that contains it.
    """
    if not any(start.get("colon_form") for start in starts):
        return starts
    echoes: set[int] = set()
    captions = False
    highest = 0
    after_article = False
    previous_major: dict | None = None
    for index, start in enumerate(starts):
        if start["level"] != 1:
            continue
        is_annex = start["section_type"] == "ek" and not start.get("boundary_only")
        number = _article_number(start["section_ref"]) if is_annex else None
        if start.get("colon_form"):
            if (
                previous_major is not None
                and previous_major["section_type"] == "ek"
                and not previous_major.get("colon_form")
                and previous_major.get("section_ref") == start["section_ref"]
            ):
                echoes.add(index)
                continue
            if number is None or number <= highest or not after_article:
                captions = True
        after_article = after_article or (
            start["section_type"] in _ARTICLE_TITLE_TYPES and not start.get("boundary_only")
        )
        if number is not None:
            highest = max(highest, number)
        previous_major = start
    return [
        start
        for index, start in enumerate(starts)
        if index not in echoes and not (captions and start.get("colon_form"))
    ]


def _introduces_quotation(preamble: str) -> bool:
    lines = [_strip_title_markup(line.strip()) for line in preamble.splitlines() if line.strip()]
    if lines and _is_article_title_line(lines[-1]):
        lines.pop()
    return bool(lines) and lines[-1].endswith(":")


def _with_amendment_table_boundary(text: str, starts: list[dict]) -> list[dict]:
    """Close the last article at the amendment-history table that follows it.

    A consolidated regulation prints, after its last article (and before any
    annexes), a table of its publication and amendments ("| | Yönetmeliğin
    Yayımlandığı Resmî Gazete’nin | …").  The table is not part of the last
    article ("Yürütme"): its header row becomes a boundary, so the table is
    indexed as govde.  Only a table after the last article heading and before
    any annex heading counts, so a table inside a provision or an annex stays
    where it is.  Subsection markers between the table and the next heading
    (amendment footnotes "(1) Bu değişiklik …") are not the article's fıkra rows.
    """
    articles = [
        index
        for index, start in enumerate(starts)
        if start["level"] == 1 and start["section_type"] in _ARTICLE_TITLE_TYPES and not start.get("boundary_only")
    ]
    if not articles:
        return starts
    last = articles[-1]
    region_end = next((s["start_char"] for s in starts[last + 1 :] if s["level"] == 1), len(text))
    line_end = text.find("\n", starts[last]["start_char"])
    match = _AMENDMENT_TABLE_RE.search(text, line_end, region_end) if line_end != -1 else None
    if match is None:
        return starts
    boundary = {"section_type": "ek", "boundary_only": True, "start_char": match.start(), "level": 1}
    before = [s for s in starts if s["start_char"] < match.start()]
    after = [
        s for s in starts if s["start_char"] >= match.start() and (s["level"] == 1 or s["start_char"] >= region_end)
    ]
    return [*before, boundary, *after]


def _opens_roman_sublist(lines: list[str], index: int, dot_headings: bool) -> bool:
    for line in lines[index + 1 :]:
        stripped = line.strip()
        if _ROMAN_SUBITEM_RE.match(stripped):
            return True
        if _parse_major_heading(stripped, dot_headings=dot_headings) is not None or _SUBSECTION_RE.match(stripped):
            return False
    return False


def _article_number(ref: str) -> int | None:
    digits = re.match(r"\d+", ref)
    return int(digits.group()) if digits else None


def _without_quoted_articles(starts: list[dict]) -> list[dict]:
    """Drop an article heading quoted inside another article.

    An amending law quotes the new text of another law's article ("… 44 üncü
    maddesi aşağıdaki şekilde değiştirilmiştir; Madde 44- …"): its number jumps
    past the running sequence (25 → 44) while the next real article resumes it
    (26).  The quoted text then stays in the quoting article, with no rows of its own.
    """
    maddes = [
        i
        for i, s in enumerate(starts)
        if s["level"] == 1 and s["section_type"] == "madde" and not s.get("boundary_only")
    ]
    quoted: set[int] = set()
    for before, current, after in zip(maddes, maddes[1:], maddes[2:], strict=False):
        numbers = [_article_number(starts[i]["section_ref"]) for i in (before, current, after)]
        if None not in numbers and numbers[1] > numbers[0] + 2 and numbers[0] < numbers[2] <= numbers[0] + 2:
            quoted.add(current)
    if not quoted:
        return starts
    kept: list[dict] = []
    dropping = False
    for i, start in enumerate(starts):
        if start["level"] == 1:
            dropping = i in quoted
            if dropping:
                continue
        elif dropping:
            continue
        kept.append(start)
    return kept


def _is_principle_structured(matches: list[dict]) -> bool:
    """Whether the classic headings are principles, optionally followed by annexes (947's "EK 1.")."""
    types = {s["section_type"] for s in matches if s["level"] == 1 and not s.get("boundary_only")}
    if types == {"ilke"}:
        return {s["section_type"] for s in matches if s["level"] == 1} == {"ilke"}
    return types == {"ilke", "ek"}


def _merge_guideline_paragraphs(text: str, starts: list[dict]) -> list[dict]:
    """Accept a unique, near-contiguous paragraph inventory, never restarting lists.

    A long run is required before interpreting bare numbers inside principles.
    Physical order is retained (943's extracted 88/87 are transposed); uniqueness
    and the sorted sequence validate identities without moving or rewriting text.
    Paragraphs are numbered "N." or, in 949, "N-"; one style per document.  The
    inventory ends at the first annex heading (947's "EK 1." restarts its own
    list), and a body numbered before the first principle stands on its own
    when the principles carry restarting lists of their own (934's annex
    tables under İlke 4).  A number printed twice in a row (949's "60-" before
    and after "İlke 10") binds to neither text: both lines only end the
    paragraph before them.
    """
    boundaries: list[dict] = []
    char_pos = 0
    for line in text.splitlines(keepends=True):
        if _GUIDELINE_DIVIDER_RE.match(line.strip()):
            boundaries.append({"start_char": char_pos, "level": 1, "boundary_only": True})
        char_pos += len(line)
    region_end = min(
        (
            s["start_char"]
            for s in starts
            if s["level"] == 1 and s["section_type"] == "ek" and not s.get("boundary_only")
        ),
        default=len(text),
    )
    first_principle = min(s["start_char"] for s in starts if s["level"] == 1 and s["section_type"] == "ilke")
    for pattern in (_GUIDELINE_PARAGRAPH_RE, _GUIDELINE_DASH_PARAGRAPH_RE):
        paragraphs = _guideline_paragraph_candidates(text, pattern, region_end)
        inventory = _validated_guideline_inventory(text, paragraphs)
        if inventory is None:
            inventory = _validated_guideline_inventory(
                text, [p for p in paragraphs if p["start_char"] < first_principle]
            )
        if inventory is not None:
            # Fıkra/bent markers are children of the numbered paragraph, not its peers.
            starts = [{**s, "level": 3} if s["level"] == 2 else s for s in starts]
            return sorted(starts + inventory + boundaries, key=lambda s: s["start_char"])
    return sorted(starts + boundaries, key=lambda s: s["start_char"])


def _guideline_paragraph_candidates(text: str, pattern: re.Pattern[str], region_end: int) -> list[dict]:
    paragraphs: list[dict] = []
    char_pos = 0
    for line in text.splitlines(keepends=True):
        if char_pos >= region_end:
            break
        match = pattern.match(line.strip())
        if match:
            paragraphs.append(
                {
                    "section_type": "paragraf",
                    "section_ref": match.group("ref"),
                    "heading": match.group("title").strip(),
                    "start_char": char_pos,
                    "level": 2,
                    "sequence_validated": True,
                }
            )
        char_pos += len(line)
    return paragraphs


def _validated_guideline_inventory(text: str, paragraphs: list[dict]) -> list[dict] | None:
    """The paragraph rows (with repeated numbers as plain boundaries), or None when the inventory fails."""
    repeated = {
        i
        for i in range(1, len(paragraphs))
        if paragraphs[i]["section_ref"] == paragraphs[i - 1]["section_ref"]
        for i in (i - 1, i)
    }
    kept = [p for i, p in enumerate(paragraphs) if i not in repeated]
    refs = sorted(int(p["section_ref"]) for p in kept)
    if (
        len(refs) < _NUMBERED_DASH_MIN_SECTIONS
        or refs[0] > _NUMBERED_MAX_GAP
        or any(not 0 < right - left <= _NUMBERED_MAX_GAP for left, right in zip(refs, refs[1:], strict=False))
        or kept[0]["start_char"] > len(text) * _NUMBERED_MAX_FIRST_OFFSET_RATIO
    ):
        return None
    ends = [
        {"section_type": "paragraf", "boundary_only": True, "start_char": p["start_char"], "level": 2}
        for i, p in enumerate(paragraphs)
        if i in repeated
    ]
    return kept + ends


def _find_numbered_paragraph_starts(text: str, region_end: int | None = None) -> list[dict]:
    """Parse a numbered paragraph body, choosing the dominant marker style.

    ``region_end`` bounds the scan to the text before a classic heading, so an
    annex's own numbering is never mistaken for the body's.  Both marker styles
    are scanned and the run spanning the most text wins — not the one with the
    most items.  A backbone runs the length of the document while an enumerated
    list is local, so counting items lets a long definitions list displace a
    shorter but genuine paragraph run and silently rebind every ref.
    """
    best: list[dict] = []
    best_span = -1
    for pattern, min_sections in _NUMBERED_STYLES:
        candidate = _scan_numbered_style(text, pattern, min_sections, region_end)
        if not candidate:
            continue
        numbered = [start for start in candidate if not start.get("boundary_only")]
        span = numbered[-1]["start_char"] - numbered[0]["start_char"]
        if span > best_span:
            best, best_span = candidate, span
    return best


def _scan_numbered_style(
    text: str,
    pattern: re.Pattern[str],
    min_sections: int,
    region_end: int | None,
) -> list[dict]:
    """Accept one marker style's run, or return [] if the evidence is too weak.

    Accepts a bare "N." only while it advances the top-level sequence
    (previous < N <= previous + _NUMBERED_MAX_GAP) and a dotted "N.M." only
    under the current top-level N; everything else is a list item and is
    skipped.  A dotted heading is authoritative — list items are never dotted
    in this corpus — so one arriving under an earlier top-level evicts the
    bare accepts after that parent (an embedded list can land exactly on
    previous + 1 and impersonate the next heading).
    """
    limit = len(text) if region_end is None else region_end
    accepted: list[dict] = []
    # Bare accepts since the last dotted confirmation, still open to eviction.
    tentative: list[int] = []
    dividers: list[dict] = []
    skipped = 0
    current_top = 0
    char_pos = 0
    for line in text.splitlines(keepends=True):
        match = pattern.match(line.strip())
        char_start, char_pos = char_pos, char_pos + len(line)
        if char_start >= limit:
            break
        if match is None and pattern is _NUMBERED_HEADING_RE:
            if _LETTER_PART_HEADING_RE.match(line.strip()):
                dividers.append(
                    {"section_type": "paragraf", "boundary_only": True, "start_char": char_start, "level": 1}
                )
                continue
            match = _NUMBERED_UNDOTTED_SUBHEADING_RE.match(line.strip())
            outline = None if match else _NUMBERED_DOTLESS_OUTLINE_RE.match(line.strip())
            if outline and int(outline.group("top")) < current_top - _NUMBERED_MAX_GAP:
                dividers.append(
                    {"section_type": "paragraf", "boundary_only": True, "start_char": char_start, "level": 1}
                )
                continue
        if not match:
            continue
        top = int(match.group("top"))
        sub = match.group("sub")
        candidate = {
            "section_type": "paragraf",
            "section_ref": f"{top}{sub}",
            "heading": match.group("title").strip(),
            "start_char": char_start,
            "level": 2 if sub else 1,
            "sequence_validated": True,
        }
        if sub and top < current_top - _NUMBERED_MAX_GAP:
            # A chapter outline heading far behind a globally numbered body
            # ("2.3.4. Kredibiliteyi Değerlendirmede …" after paragraph 37 in
            # 1041) is no child of the running paragraph: it closes that
            # paragraph and has no identity of its own.
            dividers.append({"section_type": "paragraf", "boundary_only": True, "start_char": char_start, "level": 1})
            continue
        if sub:
            evicted = [i for i in tentative if int(accepted[i]["section_ref"]) > top]
            if top == current_top:
                accepted.append(candidate)
                tentative = []
            elif top < current_top and evicted:
                for i in sorted(evicted, reverse=True):
                    accepted.pop(i)
                skipped += len(evicted)
                current_top = top
                accepted.append(candidate)
                tentative = []
            else:
                skipped += 1
        elif current_top < top <= current_top + _NUMBERED_MAX_GAP:
            current_top = top
            tentative.append(len(accepted))
            accepted.append(candidate)
        else:
            skipped += 1
    total = len(accepted) + skipped
    if len(accepted) < min_sections or skipped > total * _NUMBERED_MAX_SKIP_RATIO:
        return []
    if accepted[0]["start_char"] > limit * _NUMBERED_MAX_FIRST_OFFSET_RATIO:
        return []
    return sorted(accepted + dividers, key=lambda start: start["start_char"])


def _find_bare_paragraph_starts(text: str) -> list[dict]:
    """Parse a TMS/TFRS body whose paragraph numbers carry no dot or dash.

    The sequence is ordered by number, then letter (4 < 4A < 4B < 5), and a
    candidate is accepted only while it advances that order by at most
    _NUMBERED_MAX_GAP numbers or letters.  A range ("69–70 [Silinmiştir]") moves
    the sequence to its last number and closes the previous paragraph without an
    identity of its own, as "MADDE 146 ila 149" does.  A wrapped line or a cited
    paragraph is prose, not a candidate; the skip and offset gates of the other
    numbered styles apply to the rest.
    """
    accepted: list[dict] = []
    skipped = 0
    current = (0, 0)
    previous = ""
    char_pos = 0
    for line in text.splitlines(keepends=True):
        char_start, char_pos = char_pos, char_pos + len(line)
        stripped = line.strip()
        wrapped = _MID_SENTENCE_END_RE.search(previous) is not None
        previous = stripped
        match = _NUMBERED_BARE_HEADING_RE.match(stripped)
        if match is None or wrapped or _PARAGRAPH_CITATION_TITLE_RE.match(match.group("title")):
            continue
        first = _bare_paragraph_order(match.group("top"), match.group("letter"))
        last = first
        if match.group("last") is not None:
            last = _bare_paragraph_order(match.group("last"), match.group("last_letter"))
        if not _advances_bare_sequence(current, first) or (match.group("last") is not None and last <= first):
            skipped += 1
            continue
        current = last
        if match.group("last") is not None:
            accepted.append({"section_type": "paragraf", "boundary_only": True, "start_char": char_start, "level": 1})
            continue
        accepted.append(
            {
                "section_type": "paragraf",
                "section_ref": _normalize_ref(match.group("top") + match.group("letter")),
                "heading": match.group("title").strip(),
                "start_char": char_start,
                "level": 1,
                "sequence_validated": True,
                "isolated_headings": True,
            }
        )
    total = len(accepted) + skipped
    if len(accepted) < _NUMBERED_BARE_MIN_SECTIONS or skipped > total * _NUMBERED_MAX_SKIP_RATIO:
        return []
    if accepted[0]["start_char"] > len(text) * _NUMBERED_MAX_FIRST_OFFSET_RATIO:
        return []
    return accepted


def _bare_paragraph_order(number: str, letter: str) -> tuple[int, int]:
    return int(number), ord(letter) - ord("A") + 1 if letter else 0


def _advances_bare_sequence(current: tuple[int, int], candidate: tuple[int, int]) -> bool:
    (top, letter), (current_top, current_letter) = candidate, current
    if top == current_top:
        return current_letter < letter <= current_letter + _NUMBERED_MAX_GAP
    return current_top < top <= current_top + _NUMBERED_MAX_GAP and letter <= _NUMBERED_MAX_GAP


def _section_end_char(matches: list[dict], index: int, text_len: int) -> int:
    current_level = matches[index]["level"]
    for later in matches[index + 1 :]:
        if later["level"] <= current_level:
            return later["start_char"]
    return text_len


# Next-article titles and PDF wrap debris sit after the last sentence and before
# the following MADDE heading, so they would otherwise be stored as this section.
_LEAKED_TITLE_MAX_WORDS = 12
_LEAKED_FRAGMENT_MAX_WORDS = 3
_SENTENCE_END_RE = re.compile(r"[.!?…]$")
# Display-math delimiters, LaTeX commands and equations end a section's own
# formula ("… aşağıdaki formüle göre hesaplanır." followed by the formula); they
# are never a leaked next-article title.
_FORMULA_LINE_RE = re.compile(r"\$\$|\\[A-Za-z]|\s=\s")
# An abbreviation list entry ("BK : Beklenen Kayıp") is annex content, not a title.
_GLOSSARY_LINE_RE = re.compile(r"^[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜa-zçğıöşü0-9/&-]{0,11}\s*:\s*\S")


def _is_leaked_trailing_line(line: str) -> bool:
    # A Markdown table row ("| 2018 | 1,875 |", "|---|---|") is provision or annex
    # content in every section type, never a leaked heading.
    if (
        line.startswith(("(", "|"))
        or _SENTENCE_END_RE.search(line)
        or _FORMULA_LINE_RE.search(line)
        or _GLOSSARY_LINE_RE.match(line)
    ):
        return False
    words = line.split()
    if not words or not any(ch.isalpha() for ch in line):
        return True
    if len(words) <= _LEAKED_FRAGMENT_MAX_WORDS:
        return True
    return len(words) <= _LEAKED_TITLE_MAX_WORDS and words[0][:1].isupper()


def _ends_at_article(matches: list[dict], index: int) -> bool:
    """Whether the section at ``index`` ends where an article heading (MADDE line) starts."""
    current_level = matches[index]["level"]
    for later in matches[index + 1 :]:
        if later["level"] <= current_level:
            return later.get("section_type") in _ARTICLE_TITLE_TYPES and not later.get("boundary_only")
    return False


def _trim_trailing_heading_leak(text: str, start_char: int, end_char: int, *, before_article: bool = False) -> int:
    lines = text[start_char:end_char].splitlines(keepends=True)
    last_keep = len(lines)
    # The first line is the section's own heading ("Madde 72 — (Mülga: …)"), never a leak;
    # table rows ("| 11,314 | 8 | 5,657 |") are content (see _is_leaked_trailing_line).
    if before_article:
        # The next article's title, set apart by a blank line, belongs to that
        # article even when it runs long, opens with a date or wraps.
        end = last_keep
        while end > 1 and not lines[end - 1].strip():
            end -= 1
        stripped_lines = [_strip_title_markup(line.strip()) for line in lines[:end]]
        count = _article_title_line_count(stripped_lines)
        if count and end - count >= 2 and not stripped_lines[end - count - 1]:
            last_keep = end - count
    while last_keep > 1:
        stripped = lines[last_keep - 1].strip()
        if not stripped or _is_leaked_trailing_line(stripped):
            last_keep -= 1
            continue
        break
    if last_keep <= 0 or last_keep == len(lines):
        return end_char
    # A last sentence wrapped over short capitalised lines ("… olmayacağı Bakanlar" /
    # "Kurulunca düzenlenir. Düzenleme …" / "Kanunu … uygulanır") is provision text,
    # not a leaked title: a sentence boundary inside the continuation proves it.
    continuation: list[str] = []
    for line in lines[last_keep:]:
        if not line.strip():
            break
        continuation.append(line)
    if (
        continuation
        and _WRAPPED_LINE_RE.search(lines[last_keep - 1].strip())
        and _SENTENCE_BOUNDARY_RE.search("".join(continuation))
    ):
        last_keep += len(continuation)
    return start_char + sum(len(line) for line in lines[:last_keep])


def _trim_trailing_heading_blocks(text: str, start_char: int, end_char: int) -> int:
    """Trim whole trailing blocks of heading lines, never a line out of a block.

    KGK's layout sets every subheading ("Tanımlar", "Sunum") apart with blank
    lines, while a paragraph's wrapped last line ("özkaynaklar ile
    ilişkilendirilir.)"), closing table row ("Vergi gideri 760 780") or list item
    sits in its last block; trimming those line by line would cut them off.
    """
    lines = text[start_char:end_char].splitlines(keepends=True)
    keep = len(lines)
    while True:
        end = keep
        while end > 1 and not lines[end - 1].strip():
            end -= 1
        block_start = end
        while block_start > 0 and lines[block_start - 1].strip():
            block_start -= 1
        if block_start == 0 or not all(_is_leaked_trailing_line(line.strip()) for line in lines[block_start:end]):
            return start_char + sum(len(line) for line in lines[:end])
        keep = block_start


def _parse_major_heading(line: str, *, dot_headings: bool = False) -> dict | None:
    match = _HEADING_RE.match(line)
    if not match:
        titled = _ANNEX_INLINE_TITLE_RE.match(line)
        if titled is not None and _is_annex_inline_title(titled.group("title")):
            return {
                "section_type": "ek",
                "section_ref": _normalize_ref(titled.group("ref")),
                "heading": titled.group("title").strip(),
            }
        if _ARTICLE_RANGE_RE.match(line):
            return {"section_type": "madde", "boundary_only": True}
        if _UNNUMBERED_ANNEX_RE.match(line):
            return {"section_type": "ek", "boundary_only": True, "unnumbered_annex": True}
        if _ANNEX_LIST_RE.match(line) or _RATIONALE_RE.match(line):
            return {"section_type": "ek", "boundary_only": True}
        dot = _DOT_HEADING_RE.match(line) if dot_headings else None
        if dot is None:
            return None
        return {
            "section_type": "ilke" if dot.group("ilke") else "ek",
            "section_ref": dot.group("ref"),
            "heading": dot.group("title").strip(),
        }

    if match.group("gecici"):
        section_type, section_ref = "gecici_madde", match.group("gecici_ref")
    elif match.group("ek_madde"):
        section_type, section_ref = "ek_madde", match.group("ek_madde_ref")
    elif match.group("madde"):
        section_type, section_ref = "madde", match.group("madde_ref")
    elif match.group("ilke"):
        section_type, section_ref = "ilke", match.group("ilke_ref")
    elif match.group("paragraf"):
        section_type, section_ref = "paragraf", match.group("paragraf_ref")
    else:
        section_type, section_ref = "ek", match.group("ek_ref")

    parsed = {
        "section_type": section_type,
        "section_ref": _normalize_ref(section_ref),
        "heading": (match.group("title") or "").strip(),
    }
    if section_type == "ek" and ":" in match.group("ek_separator"):
        parsed["colon_form"] = True
    return parsed


def _parse_subsection(line: str) -> dict | None:
    match = _SUBSECTION_RE.match(line)
    if not match:
        return None
    ref = _normalize_ref(match.group("ref"))
    section_type = "fikra" if ref.isdigit() else "bent"
    return {
        "section_type": section_type,
        "section_ref": ref,
        "heading": match.group("title").strip(),
    }


def _normalize_ref(ref: str) -> str:
    return canonical_section_ref(ref)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
