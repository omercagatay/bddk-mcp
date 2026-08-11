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
SECTION_PARSER_PROFILE_VERSION = "turkish-regulatory-sections-v3"
SECTION_SEARCH_PROFILE_VERSION = "document-section-simple-fts-length-normalized-v2"

# Hard upper bound for a single section's span. Legitimate maddeler are a few
# thousand chars; spans beyond this are parser artifacts (typically trailing EK
# annexes swallowed by the last matched heading) and poison section search.
MAX_SECTION_CHARS = 20_000

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
    if not matches and len(text) > 1000:
        logger.warning(
            "extract_document_sections: no section headings matched (%d chars); "
            "document will be invisible to section search",
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
        if start["level"] == 2 and level1_capped_end is not None and start["start_char"] >= level1_capped_end:
            # Subsection markers found beyond a capped parent span are annex
            # artifacts (tables/lists swallowed by the last heading), not real
            # fıkra/bent rows; indexing them collides with genuine refs.
            continue
        end_char = _section_end_char(matches, index, len(text))
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
    return sections


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


def _section_end_char(matches: list[dict], index: int, text_len: int) -> int:
    current_level = matches[index]["level"]
    for later in matches[index + 1 :]:
        if later["level"] <= current_level:
            return later["start_char"]
    return text_len


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
