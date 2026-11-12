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
# "9. madde", "9 uncu madde", "10'uncu maddesi": the ordinal form of an article reference.
# Matched on the Turkish-casefolded query; a preceding "ek"/"geçici" names that series.
_MADDE_SUFFIX_RE = re.compile(
    rf"\b(?P<ref>{_ARTICLE_REF})(?:\.\s*|(?:\s+|\s*['’]\s*)(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü)\s+)"
    r"madde(?:si\w*)?\b"
)
_SERIES_BEFORE_RE = re.compile(r"\b(?P<kind>ek|ge[cç]i[cç]i)\s+$")
_ILKE_RE = re.compile(r"\b(?:ilke|ılke)\s*(?P<ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?)\b", re.IGNORECASE)
_PARAGRAPH_RE = re.compile(r"\bparagraf\s*(?P<ref>\d+(?:\.\d+)*)\b", re.IGNORECASE)
_LETTERED_REF_RE = re.compile(r"^(\d+)/?([a-zçğıöşü])$")
_DECISION_RE = re.compile(r"\b(\d{3,6})\s+sayılı\s+kurul\s+kararı\b", re.IGNORECASE)
_DATE_RE = re.compile(r"\b\d{2}\.\d{2}\.\d{4}\b|\b\d{4}/\d+\b")
# A number right after an accounting/audit standard or framework name is that
# standard's number ("TFRS 16", "TMS 39", "BDS 701", "Basel 3"), and "COVID-19"
# names a disease; neither is a local document ID. Matched on the Turkish-casefolded query, where "I" becomes "ı".
_STANDARD_NUMBER_RE = re.compile(
    r"\b(?:tfrs|tms|ufrs|ums|[iı]frs|[iı]as|bds|tfryk|ufryk|[iı]fr[iı]c|bob[iı]\s*frs|basel(?:\s+[iı]{1,3})?|cov[iı]d)"
    r"\s*-?\s*(?P<ref>\d+)\b"
)
# "2024 yılı", "2019 yılında": a calendar year, never a local document ID.
_YEAR_RE = re.compile(r"\b(?P<ref>(?:19|20)\d{2})\s+y[ıi]l")

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
    for match in _MADDE_SUFFIX_RE.finditer(folded):
        series = _SERIES_BEFORE_RE.search(folded[: match.start()])
        kind = series.group("kind") if series else ""
        section_type = "ek_madde" if kind == "ek" else "gecici_madde" if kind else "madde"
        sections.append((section_type, canonical_section_ref(match.group("ref"))))
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
    for pattern in (_MADDE_PREFIX_RE, _MADDE_SUFFIX_RE, _ILKE_RE, _PARAGRAPH_RE, _STANDARD_NUMBER_RE, _YEAR_RE):
        spans.extend(match.span("ref") for match in pattern.finditer(folded))
    return spans


def _span_overlaps(span: tuple[int, int], excluded_spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < excluded_end and end > excluded_start for excluded_start, excluded_end in excluded_spans)


def _unique[T](items: list[T] | tuple[T, ...] | object) -> list[T]:
    return list(dict.fromkeys(items))
