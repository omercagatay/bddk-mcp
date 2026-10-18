"""Turkish legal-reference parsing helpers."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

_DOCUMENT_ID_RE = re.compile(r"\b(?:mevzuat_|bddk_)?\d{2,8}\b", re.IGNORECASE)
# "66/A" (or "66A") is the article inserted after 66, never article 66 itself;
# "ek madde N" and "geçici madde N" are their own series.
_ARTICLE_REF = r"\d+(?:/?[A-Za-zÇĞİÖŞÜçğıöşü])?"
_MADDE_PREFIX_RE = re.compile(
    rf"\b(?:(?P<kind>ek|ge[cç]i[cç]i)\s+)?(?:madde|m)\.?\s*(?P<ref>{_ARTICLE_REF})\b", re.IGNORECASE
)
_MADDE_SUFFIX_RE = re.compile(rf"\b(?P<ref>{_ARTICLE_REF})\.\s*madde\b", re.IGNORECASE)
_ILKE_RE = re.compile(r"\b(?:ilke|ılke)\s*(?P<ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?)\b", re.IGNORECASE)
_PARAGRAPH_RE = re.compile(r"\bparagraf\s*(?P<ref>\d+(?:\.\d+)*)\b", re.IGNORECASE)
_LETTERED_REF_RE = re.compile(r"^(\d+)/?([a-zçğıöşü])$")
_DECISION_RE = re.compile(r"\b(\d{3,6})\s+sayılı\s+kurul\s+kararı\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b|\b\d{4}/\d+\b")

_CATEGORY_ALIASES = {
    "yönetmelik": "Yönetmelik",
    "yonetmelik": "Yönetmelik",
    "rehber": "Rehber",
    "tebliğ": "Tebliğ",
    "teblig": "Tebliğ",
    "genelge": "Genelge",
    "kurul kararı": "Kurul Kararı",
    "kurul karari": "Kurul Kararı",
    "kanun": "Kanun",
}


class LegalRefs(BaseModel):
    """Parsed legal references from a query."""

    document_ids: list[str] = Field(default_factory=list)
    sections: list[tuple[str, str]] = Field(default_factory=list)
    decision_numbers: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)


def document_id_candidates(document_id: str) -> list[str]:
    """Return stable local-ID aliases in lookup priority order."""
    normalized = _normalize_doc_id(document_id.strip())
    if not normalized:
        return []
    if normalized.isdigit():
        return [normalized, f"mevzuat_{normalized}", f"bddk_{normalized}"]
    if normalized.startswith(("mevzuat_", "bddk_")):
        return [normalized]
    return [normalized]


def turkish_casefold(text: str) -> str:
    """Lowercase with Turkish dotted/dotless-I behavior."""
    return text.translate(str.maketrans({"I": "ı", "İ": "i"})).lower()


def canonical_section_ref(ref: str) -> str:
    """Return the stored form of a section ref; the section index stores exactly this.

    Python lowercase, except that "İ" becomes "i" rather than "i" + U+0307; an
    inserted article's letter follows a slash ("66/A", "66A" -> "66/a").
    """
    return _LETTERED_REF_RE.sub(r"\1/\2", ref.strip().replace("İ", "i").lower())


def parse_legal_refs(query: str) -> LegalRefs:
    """Parse document IDs, article/principle refs, decisions, dates, and category hints."""
    folded = turkish_casefold(query)
    excluded_doc_spans = _excluded_document_id_spans(query, folded)

    return LegalRefs(
        document_ids=_unique(
            _normalize_doc_id(match.group(0))
            for match in _DOCUMENT_ID_RE.finditer(query)
            if not _span_overlaps(match.span(), excluded_doc_spans)
        ),
        sections=_parse_sections(query, folded),
        decision_numbers=_unique(match.group(1) for match in _DECISION_RE.finditer(folded)),
        dates=_unique(match.group(0) for match in _DATE_RE.finditer(query)),
        categories=_parse_categories(folded),
    )


def _parse_sections(query: str, folded: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    for match in _MADDE_PREFIX_RE.finditer(query):
        kind = (match.group("kind") or "").lower()
        section_type = "ek_madde" if kind == "ek" else "gecici_madde" if kind else "madde"
        sections.append((section_type, canonical_section_ref(match.group("ref"))))
    sections.extend(("madde", canonical_section_ref(match.group("ref"))) for match in _MADDE_SUFFIX_RE.finditer(query))
    sections.extend(("ilke", canonical_section_ref(match.group("ref"))) for match in _ILKE_RE.finditer(folded))
    sections.extend(("paragraf", match.group("ref")) for match in _PARAGRAPH_RE.finditer(folded))
    return _unique(sections)


def _parse_categories(folded: str) -> list[str]:
    categories: list[str] = []
    for alias, canonical in _CATEGORY_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", folded):
            categories.append(canonical)
    return _unique(categories)


def _normalize_doc_id(document_id: str) -> str:
    if document_id.lower().startswith("mevzuat_"):
        return "mevzuat_" + document_id.split("_", 1)[1]
    if document_id.lower().startswith("bddk_"):
        return "bddk_" + document_id.split("_", 1)[1]
    return document_id


def _excluded_document_id_spans(query: str, folded: str) -> list[tuple[int, int]]:
    spans = [match.span() for match in _DATE_RE.finditer(query)]
    spans.extend(match.span(1) for match in _DECISION_RE.finditer(folded))
    for pattern in (_MADDE_PREFIX_RE, _MADDE_SUFFIX_RE, _ILKE_RE, _PARAGRAPH_RE):
        spans.extend(match.span("ref") for match in pattern.finditer(folded))
    return spans


def _span_overlaps(span: tuple[int, int], excluded_spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < excluded_end and end > excluded_start for excluded_start, excluded_end in excluded_spans)


def _unique[T](items: list[T] | tuple[T, ...] | object) -> list[T]:
    return list(dict.fromkeys(items))
