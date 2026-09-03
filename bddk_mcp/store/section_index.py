"""Structural parser for Turkish legal/regulatory Markdown."""

from __future__ import annotations

import hashlib
import logging
import re

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Bump whenever heading recognition, span construction, subsection handling,
# truncation, or section-content hashing changes.  Persisted chunk metadata is
# bound to this value by the vector retrieval profile.
SECTION_PARSER_PROFILE_VERSION = "turkish-regulatory-sections-v7"
SECTION_SEARCH_PROFILE_VERSION = "document-section-simple-fts-length-normalized-v3"

# Hard upper bound for a single section's span. Legitimate maddeler are a few
# thousand chars; spans beyond this are parser artifacts (typically trailing EK
# annexes swallowed by the last matched heading) and poison section search.
MAX_SECTION_CHARS = 20_000
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
# headings.
_HEADING_RE = re.compile(
    r"^(?P<prefix>\s*(?:\*{2,3}\s*)?)"
    r"(?:(?P<gecici>geçici\s+madde)\s+(?P<gecici_ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?)"
    r"|(?P<madde>madde)\s+(?P<madde_ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?)"
    r"|(?P<ilke>ilke)\s+(?P<ilke_ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?)"
    r"|(?P<paragraf>paragraf)\s+(?P<paragraf_ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?)"
    r"|(?P<ek>ek)[-\s]*(?P<ek_ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?))"
    r"(?:\s*\*{1,3})?"
    r"\s*(?:[-:–—\x96\x97]\s*(?P<title>.*?))?"
    r"\s*\*{0,3}\s*$",
    re.IGNORECASE,
)
_SUBSECTION_RE = re.compile(r"^\s*\((?P<ref>\d+|[A-Za-zÇĞİÖŞÜçğıöşü])\)\s+(?P<title>.*)$")

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
    if not matches:
        matches = _find_numbered_paragraph_starts(text)
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
        end_char = _trim_trailing_heading_leak(text, start["start_char"], _section_end_char(matches, index, len(text)))
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
                f"\n\n[BÖLÜM KESİLDİ: içerik {truncated_from} karakterden "
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
            window_end = min(end_char, cursor + MAX_SECTION_CHARS)
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


def _find_section_starts(text: str) -> list[dict]:
    starts: list[dict] = []
    char_pos = 0
    current_major: str | None = None
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        major = _parse_major_heading(stripped)
        if major is not None:
            current_major = major["section_type"]
            starts.append({**major, "start_char": char_pos, "level": 1})
        elif current_major is not None:
            subsection = _parse_subsection(stripped)
            if subsection is not None:
                starts.append({**subsection, "start_char": char_pos, "level": 2})
        char_pos += len(line)
    return starts


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


def _is_leaked_trailing_line(line: str) -> bool:
    if line.startswith("(") or _SENTENCE_END_RE.search(line):
        return False
    words = line.split()
    if not words or not any(ch.isalpha() for ch in line):
        return True
    if len(words) <= _LEAKED_FRAGMENT_MAX_WORDS:
        return True
    return len(words) <= _LEAKED_TITLE_MAX_WORDS and words[0][:1].isupper()


def _trim_trailing_heading_leak(text: str, start_char: int, end_char: int) -> int:
    lines = text[start_char:end_char].splitlines(keepends=True)
    last_keep = len(lines)
    while last_keep > 0:
        stripped = lines[last_keep - 1].strip()
        if not stripped or _is_leaked_trailing_line(stripped):
            last_keep -= 1
            continue
        break
    if last_keep <= 0 or last_keep == len(lines):
        return end_char
    return start_char + sum(len(line) for line in lines[:last_keep])


def _parse_major_heading(line: str) -> dict | None:
    match = _HEADING_RE.match(line)
    if not match:
        return None

    if match.group("gecici"):
        section_type, section_ref = "gecici_madde", match.group("gecici_ref")
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
    return ref.strip().lower()


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
