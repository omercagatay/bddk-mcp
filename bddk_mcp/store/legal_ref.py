"""Turkish legal-reference parsing helpers."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

# A local document ID: a 2-8 digit number with an optional mevzuat_/bddk_ prefix, or a stored ID with the
# kgk_ or rg_ prefix ("kgk_tms_12", "rg_32202_20230526_6"); other snake-case words ("covid_19", "basel_3")
# are not IDs. A candidate is only a preference: the tool uses it only when a document with that ID is stored.
_DOCUMENT_ID_RE = re.compile(r"\b(?:(?:mevzuat_|bddk_)?\d{2,8}|(?:kgk|rg)_[a-z0-9_]*\d[a-z0-9_]*)\b", re.IGNORECASE)
# "66/A" (or "66A") is the article inserted after 66, never article 66 itself;
# "ek madde N" and "geçici madde N" are their own series.
_ARTICLE_REF = r"\d+(?:/?[A-Za-zÇĞİÖŞÜçğıöşü])?"
_MADDE_PREFIX_RE = re.compile(
    rf"\b(?:(?P<kind>ek|ge[cç]i[cç]i)\s+)?(?:madde|md|m)\.?\s*(?P<ref>{_ARTICLE_REF})\b", re.IGNORECASE
)
# The ordinal marker of "9. madde", "9 uncu madde", "10'uncu maddesi".
_ORDINAL = r"(?:\.\s*|(?:\s+|\s*['’]\s*)(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü)\s+)"
# "9. madde", "9 uncu madde", "10'uncu maddesi", "43 üncü maddenin", "13. maddedeki": the ordinal form of
# an article reference, in any case form. Matched on the Turkish-casefolded query; a preceding
# "ek"/"geçici" names that series.
_MADDE_SUFFIX_RE = re.compile(rf"\b(?P<ref>{_ARTICLE_REF}){_ORDINAL}madde\w*\b")
_SERIES_BEFORE_RE = re.compile(r"\b(?P<kind>ek|ge[cç]i[cç]i)\s+$")
_ILKE_RE = re.compile(r"\b(?:ilke|ılke)\s*(?P<ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?)\b", re.IGNORECASE)
# "5. ilke", "5 inci ilkesi": the ordinal form of a principle reference (casefolded query).
_ILKE_SUFFIX_RE = re.compile(r"\b(?P<ref>\d+[a-zçğıöşü]?)" + _ORDINAL + r"ilke\w*\b")
# "paragraf 15", "paragraf 2.1", "paragraf 21A" (an inserted KGK paragraph, stored as "21/a").
_PARAGRAPH_RE = re.compile(
    r"\bparagraf\s*(?P<ref>\d+(?:\.\d+)*(?:[A-Za-zÇĞİÖŞÜçğıöşü](?![\w.]))?)(?![\w])", re.IGNORECASE
)
# "15. paragraf", "39'uncu paragrafı": the ordinal form of a paragraph reference (casefolded query).
_PARAGRAPH_SUFFIX_RE = re.compile(r"\b(?P<ref>\d+[a-zçğıöşü]?)" + _ORDINAL + r"paragraf\w*\b")
# "EK-2", "Ek 2", "ek–3": an annex, unless the number opens an "ek madde" reference ("ek 1 inci maddesi",
# "EK-1 madde 5"). "2 numaralı ek" is the same annex. Matched on the casefolded query.
_ANNEX_PREFIX_RE = re.compile(
    r"\bek\s*[-–—]?\s*(?P<ref>\d+)\b(?![.\s'’]*(?:(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü)\s+)?madde)"
)
_ANNEX_SUFFIX_RE = re.compile(r"\b(?P<ref>\d+)\s+numaral[ıi]\s+ek\w*\b")
# Digits that are part of an amount, rate or percentage ("5.000", "0,03", "%15", "yüzde 0,90"): never a
# document ID. Matched on the casefolded query.
_FIGURE_RE = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)+|%\s*\d+(?:[.,]\d+)*|\by[uü]zde\s+\d+(?:[.,]\d+)*")
_LETTERED_REF_RE = re.compile(r"^(\d+)/?([a-zçğıöşü])$")
# What follows a decision number: "<n> sayılı karar", "kararı", "Kurul kararı" and their case forms (kararın,
# kararında, kararıyla ...). Matched on a Turkish-casefolded or a search-normalised (ı -> i) text; a number
# followed by it is a decision number, never a document ID. VectorStore.search uses the same pattern.
DECISION_NUMBER_SUFFIX = r"\s+say[ıi]l[ıi]\s+(?:kurul\s+)?karar(?!name)\w*"
_DECISION_RE = re.compile(rf"\b(\d{{3,6}}){DECISION_NUMBER_SUFFIX}")
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
    document_id_spans: list[tuple[int, int]] = Field(default_factory=list)
    """Character span in the query of the first occurrence of each entry of ``document_ids``."""
    sections: list[tuple[str, str]] = Field(default_factory=list)
    section_spans: list[tuple[int, int, str, str]] = Field(default_factory=list)
    """Character span in the query of every occurrence of a section reference, in query order, with the
    (section type, reference) of ``sections`` it names; an ordinal article form includes its "ek"/"geçici"."""
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
    doc_spans: dict[str, tuple[int, int]] = {}
    for match in _DOCUMENT_ID_RE.finditer(query):
        if not _span_overlaps(match.span(), excluded_doc_spans):
            doc_spans.setdefault(_normalize_doc_id(match.group(0)), match.span())

    mentions = _section_mentions(query, folded)
    return LegalRefs(
        document_ids=list(doc_spans),
        document_id_spans=list(doc_spans.values()),
        sections=_unique((kind, ref) for _, _, kind, ref in mentions),
        section_spans=sorted(mentions),
        decision_numbers=_unique(match.group(1) for match in _DECISION_RE.finditer(folded)),
        dates=_unique(match.group(0) for match in _DATE_RE.finditer(query)),
        categories=_parse_categories(folded),
    )


def _section_mentions(query: str, folded: str) -> list[tuple[int, int, str, str]]:
    """Every section reference of the query as (start, end, section type, reference), in pattern order."""
    mentions: list[tuple[int, int, str, str]] = []
    for match in _MADDE_PREFIX_RE.finditer(query):
        kind = (match.group("kind") or "").lower()
        section_type = "ek_madde" if kind == "ek" else "gecici_madde" if kind else "madde"
        mentions.append((*match.span(), section_type, canonical_section_ref(match.group("ref"))))
    for match in _MADDE_SUFFIX_RE.finditer(folded):
        series = _SERIES_BEFORE_RE.search(folded[: match.start()])
        kind = series.group("kind") if series else ""
        section_type = "ek_madde" if kind == "ek" else "gecici_madde" if kind else "madde"
        start = series.start() if series else match.start()
        mentions.append((start, match.end(), section_type, canonical_section_ref(match.group("ref"))))
    for pattern in (_ILKE_RE, _ILKE_SUFFIX_RE):
        mentions.extend(
            (*match.span(), "ilke", canonical_section_ref(match.group("ref"))) for match in pattern.finditer(folded)
        )
    for pattern in (_PARAGRAPH_RE, _PARAGRAPH_SUFFIX_RE):
        # The letter of an inserted KGK paragraph ("98I") is Latin: take it from the unfolded query.
        mentions.extend(
            (*match.span(), "paragraf", canonical_section_ref(query[slice(*match.span("ref"))]))
            for match in pattern.finditer(folded)
        )
    for pattern in (_ANNEX_PREFIX_RE, _ANNEX_SUFFIX_RE):
        mentions.extend((*match.span(), "ek", match.group("ref")) for match in pattern.finditer(folded))
    return mentions


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
    if not document_id.isdigit() and "_" in document_id:
        # Letter-prefixed stored IDs are lowercase ("kgk_tms_12").
        return document_id.lower()
    return document_id


def _excluded_document_id_spans(query: str, folded: str) -> list[tuple[int, int]]:
    spans = [match.span() for match in _DATE_RE.finditer(query)]
    spans.extend(match.span(1) for match in _DECISION_RE.finditer(folded))
    spans.extend(match.span() for match in _FIGURE_RE.finditer(folded))
    for pattern in (
        _MADDE_PREFIX_RE,
        _MADDE_SUFFIX_RE,
        _ILKE_RE,
        _ILKE_SUFFIX_RE,
        _PARAGRAPH_RE,
        _PARAGRAPH_SUFFIX_RE,
        _ANNEX_PREFIX_RE,
        _ANNEX_SUFFIX_RE,
        _STANDARD_NUMBER_RE,
        _YEAR_RE,
    ):
        spans.extend(match.span("ref") for match in pattern.finditer(folded))
    return spans


def _span_overlaps(span: tuple[int, int], excluded_spans: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < excluded_end and end > excluded_start for excluded_start, excluded_end in excluded_spans)


def _unique[T](items: list[T] | tuple[T, ...] | object) -> list[T]:
    return list(dict.fromkeys(items))
