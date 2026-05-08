"""Structural parser for Turkish legal/regulatory Markdown."""

from __future__ import annotations

import hashlib
import re

from pydantic import BaseModel

_HEADING_RE = re.compile(
    r"^(?P<prefix>\s*)"
    r"(?:(?P<gecici>geçici\s+madde)\s+(?P<gecici_ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?)"
    r"|(?P<madde>madde)\s+(?P<madde_ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?)"
    r"|(?P<ilke>ilke)\s+(?P<ilke_ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?)"
    r"|(?P<paragraf>paragraf)\s+(?P<paragraf_ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?)"
    r"|(?P<ek>ek)[-\s]*(?P<ek_ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?))"
    r"\s*(?:[-:–—]\s*(?P<title>.*))?$",
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
    sections: list[DocumentSection] = []
    for index, start in enumerate(matches):
        end_char = _section_end_char(matches, index, len(text))
        content = text[start["start_char"] : end_char].strip()
        if not content:
            continue
        sections.append(
            DocumentSection(
                doc_id=doc_id,
                section_type=start["section_type"],
                section_ref=start["section_ref"],
                heading=start["heading"],
                start_char=start["start_char"],
                end_char=end_char,
                content=content,
                content_hash=_content_hash(content),
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
