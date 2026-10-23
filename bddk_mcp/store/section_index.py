"""Structural parser for Turkish legal/regulatory Markdown."""

from __future__ import annotations

import hashlib
import logging
import re

from pydantic import BaseModel

from bddk_mcp.store.legal_ref import canonical_section_ref

logger = logging.getLogger(__name__)

# Bump whenever heading recognition, span construction, subsection handling,
# truncation, or section-content hashing changes.  Persisted chunk metadata is
# bound to this value by the vector retrieval profile.
SECTION_PARSER_PROFILE_VERSION = "turkish-regulatory-sections-v11"
SECTION_SEARCH_PROFILE_VERSION = "document-section-simple-fts-length-normalized-v6"

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
    r"|(?P<ek>ek)[-\s–—\x96\x97]*(?P<ek_ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?))"
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

# Fallback grammar for documents without MADDE/İlke-style headings: Rehber
# number their paragraphs "1. ... 206." and Genelge use decimal outlines
# ("2.", "2.1.").  The trailing dot is required — footnotes ("3 Orta vadeli
# fonlama...") and cross-references ("5 inci maddesinin...") have none.
_NUMBERED_HEADING_RE = re.compile(r"^\s*(?:\*{2,3}\s*)?(?P<top>\d{1,3})(?P<sub>(?:\.\d{1,3})*)\.\s+(?P<title>\S.*)$")
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
    if not matches:
        matches = _find_numbered_paragraph_starts(text) or _find_bare_paragraph_starts(text)
    elif {s["section_type"] for s in matches if s["level"] == 1} == {"ilke"}:
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
    # document_sections_identity_uq declares a section's identity to be
    # (doc_id, section_type, section_ref, content_hash), and chunks bind to
    # sections through section_content_hash.  Consolidated tebliğ repeat an
    # identical closing article once per amendment, so emitting a row per
    # occurrence produces rows the schema cannot store.  Keep the first.
    seen_identities: set[tuple[str, str, str]] = set()
    level1_capped_end: int | None = None
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
                text, start["start_char"], end_char, annex=start["section_type"] == "ek"
            )
        truncated_from = None
        if end_char - start["start_char"] > MAX_SECTION_CHARS:
            truncated_from = end_char - start["start_char"]
            logger.warning(
                "extract_document_sections: capping oversized section span %d-%d (%d chars) to %d",
                start["start_char"],
                end_char,
                truncated_from,
                MAX_SECTION_CHARS,
            )
            end_char = start["start_char"] + MAX_SECTION_CHARS
        if start["level"] == 1:
            level1_capped_end = end_char if truncated_from else None
        content = text[start["start_char"] : end_char].strip()
        if not content:
            continue
        if truncated_from is not None:
            # The marker travels inside the stored content so every consumer
            # (get_document_section, search previews) sees the truncation.
            content += (
                f"\n\n{TRUNCATION_NOTICE_PREFIX} içerik {truncated_from} karakterden "
                f"{MAX_SECTION_CHARS} karaktere kısaltıldı — tam metin için get_bddk_document kullanın]"
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
                heading=start["heading"],
                start_char=start["start_char"],
                end_char=end_char,
                content=content,
                content_hash=content_hash,
            )
        )
    sections.extend(_remainder_govde_sections(doc_id, text, sections, seen_identities))
    return sections


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
    return _without_quoted_articles(starts)


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


def _merge_guideline_paragraphs(text: str, starts: list[dict]) -> list[dict]:
    """Accept a unique, near-contiguous paragraph inventory, never restarting lists.

    A long run is required before interpreting bare numbers inside principles.
    Physical order is retained (943's extracted 88/87 are transposed); uniqueness
    and the sorted sequence validate identities without moving or rewriting text.
    """
    paragraphs: list[dict] = []
    boundaries: list[dict] = []
    char_pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        match = _GUIDELINE_PARAGRAPH_RE.match(stripped)
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
        elif _GUIDELINE_DIVIDER_RE.match(stripped):
            boundaries.append({"start_char": char_pos, "level": 1, "boundary_only": True})
        char_pos += len(line)
    refs = sorted(int(p["section_ref"]) for p in paragraphs)
    if (
        len(refs) < _NUMBERED_DASH_MIN_SECTIONS
        or refs[0] > _NUMBERED_MAX_GAP
        or any(not 0 < right - left <= _NUMBERED_MAX_GAP for left, right in zip(refs, refs[1:], strict=False))
        or paragraphs[0]["start_char"] > len(text) * _NUMBERED_MAX_FIRST_OFFSET_RATIO
    ):
        return sorted(starts + boundaries, key=lambda s: s["start_char"])
    # Fıkra/bent markers are children of the numbered paragraph, not its peers.
    starts = [{**s, "level": 3} if s["level"] == 2 else s for s in starts]
    return sorted(starts + paragraphs + boundaries, key=lambda s: s["start_char"])


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
        span = candidate[-1]["start_char"] - candidate[0]["start_char"]
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
    skipped = 0
    current_top = 0
    char_pos = 0
    for line in text.splitlines(keepends=True):
        match = pattern.match(line.strip())
        char_start, char_pos = char_pos, char_pos + len(line)
        if char_start >= limit:
            break
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
    return accepted


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
    if (
        line.startswith("(")
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


def _trim_trailing_heading_leak(text: str, start_char: int, end_char: int, *, annex: bool = False) -> int:
    lines = text[start_char:end_char].splitlines(keepends=True)
    last_keep = len(lines)
    # The first line is the section's own heading ("Madde 72 — (Mülga: …)"), never a leak;
    # an annex's table rows ("| 11,314 | 8 | 5,657 |") are its content.
    while last_keep > 1:
        stripped = lines[last_keep - 1].strip()
        if not stripped or (_is_leaked_trailing_line(stripped) and not (annex and stripped.startswith("|"))):
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
        if _ARTICLE_RANGE_RE.match(line):
            return {"section_type": "madde", "boundary_only": True}
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

    return {
        "section_type": section_type,
        "section_ref": _normalize_ref(section_ref),
        "heading": (match.group("title") or "").strip(),
    }


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
