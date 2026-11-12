"""Section-level document retrieval and search tools."""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import TYPE_CHECKING

from bddk_mcp.quality.markdown_quality import (
    FORMULA_EXTRACTION_WARNING,
    QualityAssessment,
    assess_markdown_quality,
    is_formula_aware,
    mentions_formula,
    unsafe_verbatim_reason,
)
from bddk_mcp.quotations import ExactQuotationCheck, check_exact_quotation
from bddk_mcp.store.doc_store import fts_prefix_stem_query
from bddk_mcp.store.legal_ref import document_id_candidates, parse_legal_refs, turkish_casefold
from bddk_mcp.store.section_index import (
    GOVDE_HEADING,
    MAX_SECTION_CHARS,
    split_section_truncation_notice,
    truncation_notice_original_length,
)
from bddk_mcp.tools.contract_types import (
    DocumentId,
    HeadingFilter,
    OptionalDocumentId,
    OptionalQuotation,
    OptionalSectionStartChar,
    SectionQuery,
    SectionRef,
    SectionResultLimit,
    SectionType,
)
from bddk_mcp.tools.errors import tool_error
from bddk_mcp.tools.structured_outputs import (
    LEGAL_APPLICABILITY_WARNING,
    STRUCTURELESS_GOVDE_LABEL,
    UNTRUSTED_SOURCE_WARNING,
    DocumentSectionResponse,
    DocumentSectionToolResult,
    EvidenceReference,
    QualityMetadata,
    SectionAnswerAssessment,
    SectionItem,
    SectionSearchResponse,
    SectionSearchToolResult,
    decision_identifier_text,
    document_status_notice,
    source_date_text,
    structured_tool_result,
    utc_date,
)
from bddk_mcp.tools.tool_logging import logged_tool

if TYPE_CHECKING:
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.store.doc_store import StoredDocumentSection

logger = logging.getLogger(__name__)


_MAX_EXACT_SECTION_CHARS = 30_000
_MAX_SEARCH_EXCERPT_CHARS = 2_000
_MAX_DISAMBIGUATION_RESULTS = 10
_SECTION_TRUNCATION_WARNING = (
    "One or more section bodies were returned as bounded excerpts. Use an exact document/section reference "
    "or paginated full-document retrieval before relying on omitted text."
)
_EXACT_REFERENCE_MISSING_WARNING = (
    "The requested exact provision was not found. Returned lexical matches are other sections, "
    "not evidence for that requested provision. Check the reference or retrieve the full document."
)
_LEGAL_APPLICABILITY_WARNING = LEGAL_APPLICABILITY_WARNING
_NESTED_ROW_HINT = (
    "fikra and bent rows are search-oriented and are not unique addresses (many rows can share one reference); "
    'where a stored parent article contains a row it is shown as "Parent:"; retrieve the parent madde '
    "(section_type=madde, or gecici_madde/ek_madde) for the full provision."
)
_CAPPED_QUOTATION_NOTE = (
    "This reference is a capped stored window of the provision: the quotation was compared only with that "
    "window, and text beyond it was NOT compared. The rest of the provision is in full in get_bddk_document; "
    "parts of it are also stored as later rows of the same document (nested fikra/bent/paragraf rows and/or "
    "govde windows), which do not necessarily cover every character."
)
_Parents = dict[tuple[str, str, int], tuple[str, str]]
# Value: (type, ref, true end code point, row runs past that end) of the capped named section a govde row continues.
_Continuations = dict[tuple[str, str, int], tuple[str, str, int, bool]]
_STRUCTURELESS_LABEL = STRUCTURELESS_GOVDE_LABEL
_STRUCTURELESS_WARNING = (
    "One or more hits are full-text windows of a document that has no article structure (for example a Kurul "
    "Kararı). Cite such a document by its title and decision date/number, not as an article (madde) or as a "
    "govde window."
)
_GOVDE_WARNING = (
    "One or more hits are govde remainder (unparsed body/footnote text), not madde/ilke/paragraf identities. "
    "Do not cite them as a legal provision."
)

_LOOSE_SECTION_SEARCH_STOPWORDS = {
    "acaba",
    "altına",
    "aşağıdaki",
    "aşağıdakilerden",
    "bir",
    "biri",
    "değildir",
    "göre",
    "hangi",
    "hangisi",
    "hakkında",
    "için",
    "ilişkin",
    "olan",
    "olarak",
    "ve",
    "veya",
    "yanlıştır",
    "yer",
}
# Standalone FTS on these floods IPC / generic-ratio articles and crowds out
# the provision that actually contains the distinctive tokens.
_LOOSE_FTS_SKIP = {
    "asgari",
    "azami",
    "aşım",
    "ceza",
    "cezası",
    "idari",
    "oran",
    "oranı",
    "para",
    "yaptırım",
    "yüzde",
}


def _section_quality(section: StoredDocumentSection) -> QualityAssessment:
    local = assess_markdown_quality(section.content, document_id=section.doc_id)
    document = section.document_quality
    severity = {"clean": 0, "warning": 1, "unknown": 2, "fail": 3}
    quality = local
    if document is not None:
        worst = max((document, local), key=lambda item: severity[item.label])
        quality = worst.model_copy(update={"flags": list(dict.fromkeys([*document.flags, *local.flags]))})
    if (
        section.document_extraction_method
        and not is_formula_aware(section.document_extraction_method)
        and mentions_formula(section.content)
    ):
        quality = quality.model_copy(
            update={
                "label": "warning" if quality.label == "clean" else quality.label,
                "flags": [*quality.flags, "formula_unaware_extraction"],
                "warning": " ".join(filter(None, (quality.warning, FORMULA_EXTRACTION_WARNING))),
            }
        )
    return quality


def _quality_lines(quality: QualityAssessment, *, prefix: str) -> list[str]:
    """Return consistent user-visible quality metadata for a non-clean section."""
    if quality.label not in {"warning", "fail", "unknown"}:
        return []
    flags = ", ".join(quality.flags) if quality.flags else "none"
    lines = [f"{prefix}Quality: {quality.label}", f"{prefix}Quality flags: {flags}"]
    if quality.warning:
        lines.append(f"{prefix}⚠ Quality warning: {quality.warning}")
    return lines


def _quality_metadata(quality: QualityAssessment) -> QualityMetadata:
    return QualityMetadata(
        label=quality.label,  # type: ignore[arg-type]
        flags=list(quality.flags),
        warning=quality.warning or None,
    )


def _section_excerpt(
    section: StoredDocumentSection,
    *,
    max_chars: int,
    query: str | None = None,
) -> tuple[str, bool, int, int]:
    """Return a bounded excerpt and absolute offsets into normalized content."""

    # Stored truncation notices are storage metadata, never legal characters.
    raw_content, truncation_notice = split_section_truncation_notice(section.content or "")
    if len(raw_content) <= max_chars:
        local_start = 0
        local_end = len(raw_content)
    else:
        match_offset = 0
        if query:
            # Search the original string so Turkish İ / Unicode case changes
            # cannot shift normalized code-point offsets. Prefer the complete
            # phrase (including PDF line wraps) over its earliest generic word.
            phrase = re.search(r"\s+".join(re.escape(word) for word in query.split()), raw_content, re.IGNORECASE)
            if phrase:
                match_offset = phrase.start()
            else:
                matches = [
                    re.search(re.escape(term), raw_content, re.IGNORECASE) for term in _loose_search_terms(query)
                ]
                match_offset = min((match.start() for match in matches if match), default=0)
        local_start = max(0, min(match_offset - max_chars // 4, len(raw_content) - max_chars))
        local_end = min(len(raw_content), local_start + max_chars)

    # Verbatim contract: the excerpt is an exact stored substring, never a
    # sanitized/summarized rendering of it.
    excerpt = raw_content[local_start:local_end]
    unsafe_reason = unsafe_verbatim_reason(excerpt)
    if unsafe_reason is not None:
        tool_error(
            "VERBATIM_UNAVAILABLE",
            f"Section {section.doc_id}/{section.section_type} {section.section_ref} cannot be returned verbatim: "
            f"its stored text contains an unsafe extraction artifact ({unsafe_reason}). This server does not "
            "rewrite or summarize section text; repair and re-publish the source instead.",
            retryable=False,
        )
    source_range = section.normalized_source_range
    leading_space = len(source_range) - len(source_range.lstrip()) if source_range.strip() == raw_content else 0
    return (
        excerpt,
        local_start > 0 or local_end < len(raw_content) or truncation_notice is not None,
        section.start_char + leading_space + local_start,
        min(section.end_char, section.start_char + leading_space + local_end),
    )


def _capped_span(section: StoredDocumentSection) -> tuple[bool, int | None]:
    """Return whether the stored section is a capped window, and the uncapped span end if recorded."""
    _legal_text, notice = split_section_truncation_notice(section.content or "")
    if notice is None:
        return False, None
    original_length = truncation_notice_original_length(notice)
    return True, None if original_length is None else section.start_char + original_length


def _capped_span_line(section: StoredDocumentSection) -> str | None:
    capped, original_end = _capped_span(section)
    if not capped:
        return None
    if original_end is None:
        return (
            f"- Stored window capped at {MAX_SECTION_CHARS:,} characters; the provision continues past code point "
            f"{section.end_char}. {_CAPPED_REMAINDER_TEXT}"
        )
    return (
        f"- Stored window capped at {MAX_SECTION_CHARS:,} of {original_end - section.start_char:,} characters; "
        f"the provision continues to code point {original_end} (uncapped span [{section.start_char}, "
        f"{original_end})). {_CAPPED_REMAINDER_TEXT}"
    )


_CAPPED_REMAINDER_TEXT = (
    "The rest of the provision is in full in get_bddk_document; parts of it are also stored as later rows of this "
    "document (nested fikra/bent/paragraf rows and/or govde windows), which do not necessarily cover every character."
)


def _continuation_of(
    section: StoredDocumentSection, continuations: _Continuations
) -> tuple[str, str, int, bool] | None:
    """Return (type, ref) of the capped named section a govde row continues, if any."""
    if section.section_type != "govde":
        return None
    return continuations.get((section.doc_id, section.source_content_hash, section.start_char))


def _continuation_label(parent: tuple[str, str, int, bool]) -> str:
    return f"continuation of {parent[0]} {parent[1]} beyond its {MAX_SECTION_CHARS:,}-character stored window"


def _continuation_overrun(parent: tuple[str, str, int, bool]) -> str | None:
    """State the out-of-provision tail when the govde window has text past the capped provision's true end."""
    if not parent[3]:
        return None
    return f"the characters from code point {parent[2]} on are outside {parent[0]} {parent[1]}"


def _govde_label(section: StoredDocumentSection, continuations: _Continuations) -> str | None:
    """Label a govde row as a capped-provision continuation, a structureless full text, or None."""
    if parent := _continuation_of(section, continuations):
        label = _continuation_label(parent)
        if overrun := _continuation_overrun(parent):
            label = f"{label} up to code point {parent[2]}; {overrun}"
        return label
    if section.section_type == "govde" and section.document_structureless:
        return _STRUCTURELESS_LABEL
    return None


def _heading_suffix(section: StoredDocumentSection, continuations: _Continuations) -> str:
    """The heading part of a title line.

    The parser stores one generic heading on every govde row ("no structural heading; body/footnote
    remainder").  For a row that is a full-text window or the continuation of a capped provision that
    wording contradicts the kind label printed below it, so the title line leaves it out.
    """
    if not section.heading or (section.heading == GOVDE_HEADING and _govde_label(section, continuations)):
        return ""
    return f" — {section.heading}"


def _continuation_warning(section: StoredDocumentSection, parent: tuple[str, str, int, bool]) -> str:
    belongs = f"{parent[0]} {parent[1]}"
    if overrun := _continuation_overrun(parent):
        belongs = f"{belongs} up to code point {parent[2]}; {overrun}"
    return (
        f"{section.doc_id} govde {section.section_ref} is the {_continuation_label(parent)}: this text belongs to "
        f"{belongs}. Cite that provision and verify the full text with get_bddk_document."
    )


async def _find_continuations(deps: Dependencies, sections: list[StoredDocumentSection]) -> _Continuations:
    if not any(section.section_type == "govde" for section in sections):
        return {}
    return await deps.doc_store.find_capped_continuations(sections)


async def _find_parents(deps: Dependencies, sections: list[StoredDocumentSection]) -> _Parents:
    if not any(section.section_type in {"fikra", "bent"} for section in sections):
        return {}
    return await deps.doc_store.find_parent_articles(sections)


def _parent_of(section: StoredDocumentSection, parents: _Parents) -> tuple[str, str] | None:
    return parents.get((section.doc_id, section.source_content_hash, section.start_char))


def _parent_line(section: StoredDocumentSection, parents: _Parents, *, prefix: str) -> list[str]:
    parent = _parent_of(section, parents)
    return [f"{prefix}Parent: {parent[0]} {parent[1]}"] if parent else []


def _with_parent(item: SectionItem, section: StoredDocumentSection, parents: _Parents) -> SectionItem:
    parent = _parent_of(section, parents)
    if parent is None:
        return item
    return item.model_copy(update={"parent_section_type": parent[0], "parent_section_ref": parent[1]})


def _section_item(
    section: StoredDocumentSection,
    *,
    max_chars: int = _MAX_SEARCH_EXCERPT_CHARS,
    query: str | None = None,
    continuations: _Continuations | None = None,
) -> SectionItem:
    content, truncated, excerpt_start, excerpt_end = _section_excerpt(
        section,
        max_chars=max_chars,
        query=query,
    )
    capped, original_end = _capped_span(section)
    parent = _continuation_of(section, continuations or {})
    return SectionItem(
        document_id=section.doc_id,
        section_type=section.section_type,
        section_ref=section.section_ref,
        # Same rule as the text title line: a generic govde heading that contradicts the kind label is left out.
        heading=_heading_suffix(section, continuations or {}).removeprefix(" — "),
        start_char=section.start_char,
        end_char=section.end_char,
        page_start=section.page_start,
        page_end=section.page_end,
        content=content,
        content_truncated=truncated,
        excerpt_start_char=excerpt_start,
        excerpt_end_char=excerpt_end,
        content_hash=section.content_hash,
        rank=section.rank,
        quality=_quality_metadata(_section_quality(section)),
        stored_span_capped=capped,
        original_end_char=original_end,
        continuation_of_section_type=parent[0] if parent else None,
        continuation_of_section_ref=parent[1] if parent else None,
        continuation_of_section_end_char=parent[2] if parent else None,
        document_structureless=section.document_structureless,
    )


def _section_evidence(section: StoredDocumentSection) -> EvidenceReference:
    return EvidenceReference(
        document_id=section.doc_id,
        title=section.document_title or None,
        source_url=section.document_source_url or None,
        decision_date=section.document_decision_date or None,
        decision_number=section.document_decision_number or None,
        source_downloaded_on=utc_date(section.document_downloaded_at),
        category=section.document_category or None,
        extraction_method=section.document_extraction_method or None,
        retrieval_source="section_index",
        page_start=section.page_start,
        page_end=section.page_end,
        section_type=section.section_type,
        section_ref=section.section_ref,
        start_char=section.start_char,
        end_char=section.end_char,
        content_hash=section.content_hash,
        quality=_quality_metadata(_section_quality(section)),
    )


def _assess_quotation(sections: list[StoredDocumentSection], quotation: str) -> SectionAnswerAssessment:
    """Check a proposed quotation character-for-character against one resolved section."""

    if len(sections) != 1:
        return SectionAnswerAssessment(
            basis="insufficient",
            quotation_status="unavailable",
            gaps=["exact_section_not_resolved"],
        )
    section = sections[0]
    # The stored hash binds the stored content, truncation notice included. Verify that
    # binding first, then compare only the legal text: the notice is storage metadata,
    # never legal text. It is trailing, so offsets into the legal text stay valid.
    if hashlib.sha256(section.content.encode("utf-8")).hexdigest() != section.content_hash:
        check = ExactQuotationCheck(status="unavailable", reason="reference_hash_mismatch")
    else:
        legal_text, notice = split_section_truncation_notice(section.content)
        check = check_exact_quotation(
            legal_text,
            quotation,
            expected_reference_sha256=hashlib.sha256(legal_text.encode("utf-8")).hexdigest(),
        )
        if notice is not None:
            # A capped section stores only a window of the provision. A quotation absent
            # from that window may lie in the part that was not compared, so the check is
            # inconclusive, and a match of the whole window is still only an excerpt.
            if check.status == "mismatch":
                check = check.model_copy(update={"status": "unavailable", "reason": "reference_truncated"})
            elif check.extent == "whole_reference":
                check = check.model_copy(update={"extent": "excerpt"})
    if check.status == "exact_reference_match":
        status = "exact_reference_match"
    elif check.status == "mismatch":
        status = "not_found"
    else:
        status = "unavailable"
    return SectionAnswerAssessment(
        basis="local_text",
        quotation_status=status,  # type: ignore[arg-type]
        quotation_check=check,
        gaps=[] if status == "exact_reference_match" else [f"quotation_{check.reason}"],
    )


def _section_warnings(
    sections: list[StoredDocumentSection],
    *,
    content_truncated: bool = False,
    continuations: _Continuations | None = None,
) -> list[str]:
    quality_warnings = list(
        dict.fromkeys(quality.warning for section in sections if (quality := _section_quality(section)).warning)
    )
    warnings = (
        [UNTRUSTED_SOURCE_WARNING, _LEGAL_APPLICABILITY_WARNING, *quality_warnings] if sections else quality_warnings
    )
    warnings.extend(
        dict.fromkeys(notice for section in sections if (notice := document_status_notice(section.document_category)))
    )
    if content_truncated:
        warnings.append(_SECTION_TRUNCATION_WARNING)
    remainder = structureless = False
    for section in sections:
        if section.section_type != "govde":
            continue
        if parent := _continuation_of(section, continuations or {}):
            warnings.append(_continuation_warning(section, parent))
        elif section.document_structureless:
            structureless = True
        else:
            remainder = True
    if structureless:
        warnings.append(_STRUCTURELESS_WARNING)
    if remainder:
        warnings.append(_GOVDE_WARNING)
    return warnings


def _provenance_lines(section: StoredDocumentSection, *, prefix: str) -> list[str]:
    """Decision identifiers and the source download date of the section's document."""
    lines = []
    if decision := decision_identifier_text(section.document_decision_date, section.document_decision_number):
        lines.append(f"{prefix}Decision date / number: {decision}")
    lines.append(f"{prefix}Text obtained from the source on: {source_date_text(section.document_downloaded_at)}")
    return lines


def _format_section(
    section: StoredDocumentSection,
    *,
    include_content: bool = True,
    quality: QualityAssessment | None = None,
    continuations: _Continuations | None = None,
    parents: _Parents | None = None,
) -> str:
    heading = _heading_suffix(section, continuations or {})
    lines = [
        f"### {section.doc_id} — {section.section_type} {section.section_ref}{heading}",
        f"- Document ID: {section.doc_id}",
        f"- Section: {section.section_type} {section.section_ref}",
        (f"- Normalized Markdown code-point range: [{section.start_char}, {section.end_char}); not source PDF pages"),
    ]
    lines.extend(_parent_line(section, parents or {}, prefix="- "))
    if capped_line := _capped_span_line(section):
        lines.append(capped_line)
    if label := _govde_label(section, continuations or {}):
        lines.append(f"- Section kind: {label}")
    if section.document_title:
        lines.append(f"- Document title: {section.document_title}")
    if status_notice := document_status_notice(section.document_category):
        lines.append(f"- Document status: {status_notice}")
    lines.extend(_provenance_lines(section, prefix="- "))
    if section.document_source_url:
        lines.append(f"- Catalog source (not a validated citation): {section.document_source_url}")
    if section.document_extraction_method:
        lines.append(f"- Extraction method: {section.document_extraction_method}")
    if section.page_start is not None:
        page_end = section.page_end if section.page_end is not None else section.page_start
        lines.append(f"- Normalized page window: {section.page_start}-{page_end} (not verified source PDF pages)")
    quality = quality or _section_quality(section)
    lines.extend(_quality_lines(quality, prefix="- "))
    if include_content:
        excerpt, truncated, excerpt_start, excerpt_end = _section_excerpt(
            section,
            max_chars=_MAX_EXACT_SECTION_CHARS,
        )
        if truncated:
            lines.append(
                f"- Returned normalized excerpt range: [{excerpt_start}, {excerpt_end}) (section body truncated)"
            )
        lines.extend(["", excerpt])
    return "\n".join(lines)


def _section_preview(section: StoredDocumentSection, *, length: int = 220) -> str:
    text = " ".join(section.content.split())
    if len(text) <= length:
        return text
    return text[:length].rsplit(" ", 1)[0]


def _section_key(section: StoredDocumentSection) -> tuple[str, str, str, str]:
    return (section.doc_id, section.section_type, section.section_ref, section.content_hash)


def _normalize_optional(value: str | int | None) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value.lower() if value else None


def _loose_search_text(text: str) -> str:
    # Search-only acronym equivalence; never rewrite returned source evidence.
    return re.sub(r"\blgd\b", "thk", turkish_casefold(text))


def _loose_search_terms(query: str, *, limit: int = 16) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in re.findall(r"\w+", query, flags=re.UNICODE):
        term = _loose_search_text(raw_term)
        if len(term) < 3 or term in _LOOSE_SECTION_SEARCH_STOPWORDS or term in seen:
            continue
        terms.append(term)
        seen.add(term)
        if len(terms) >= limit:
            break
    return terms


def _loose_section_score(section: StoredDocumentSection, terms: list[str], weights: dict[str, float]) -> float:
    text = _loose_search_text(f"{section.heading} {section.content}")
    score = sum(weights[term] for term in terms if term in text)
    # Penalize large cross-topic spans without rewarding tiny fragments:
    # short clauses (up to 100 words) compete on coverage, not brevity.
    score /= 1 + math.log(max(1, len(text.split()) / 100))
    if section.section_type == "gecici_madde" and "geçici" not in terms and "gecici" not in terms:
        score -= 10_000
    return score


async def _any_document_stored(deps: Dependencies, document_id: str) -> bool:
    for candidate in document_id_candidates(document_id):
        if await deps.doc_store.has_document(candidate):
            return True
    return False


async def _search_sections_strict_with_stem_top_up(
    deps: Dependencies,
    query: str,
    *,
    candidate_doc_ids: list[str | None],
    section_type: str | None,
    limit: int,
) -> tuple[list[StoredDocumentSection], set[tuple[str, str, str, str]]]:
    """Strict FTS pass, topped up with the prefix-stem pass when it returns fewer than ``limit`` hits.

    The top-up uses the same document scope (the candidate that produced the strict hits, or
    each candidate in turn when there were none) and section filter. Strict hits come first;
    top-up hits are de-duplicated against them and carry no rank. Returns the hits and the
    keys of the top-up hits.
    """
    hits: list[StoredDocumentSection] = []
    used_doc_id: str | None = None
    for candidate_doc_id in candidate_doc_ids:
        hits = await deps.doc_store.search_document_sections(
            query,
            document_id=candidate_doc_id,
            section_type=section_type,
            limit=limit,
        )
        if hits:
            used_doc_id = candidate_doc_id
            break
    if len(hits) >= limit or fts_prefix_stem_query(query) is None:
        return hits, set()
    seen = {_section_key(section) for section in hits}
    stem_keys: set[tuple[str, str, str, str]] = set()
    scope = [used_doc_id] if hits else candidate_doc_ids
    top_up: list[StoredDocumentSection] = []
    for candidate_doc_id in scope:
        stem_hits = await deps.doc_store.search_document_sections(
            query,
            document_id=candidate_doc_id,
            section_type=section_type,
            limit=limit + len(hits),
            prefix_stem=True,
        )
        top_up = [section for section in stem_hits if _section_key(section) not in seen]
        if top_up:
            break
    top_up = [section.model_copy(update={"rank": None}) for section in top_up[: limit - len(hits)]]
    stem_keys = {_section_key(section) for section in top_up}
    return hits + top_up, stem_keys


async def _search_sections_loose(
    deps: Dependencies,
    query: str,
    *,
    document_id: str | None,
    section_type: str | None,
    limit: int,
) -> list[StoredDocumentSection]:
    terms = _loose_search_terms(query)
    if not terms:
        return []
    fts_terms = [term for term in terms if (len(term) >= 5 or term == "thk") and term not in _LOOSE_FTS_SKIP]
    if not fts_terms:
        fts_terms = terms
    if "thk" in fts_terms:
        fts_terms = [*fts_terms, "lgd"]

    merged: dict[tuple[str, str, str, str], StoredDocumentSection] = {}
    # Retrieve a wider pool before cross-term ranking, rather than letting
    # single-word top-k lists discard the multi-concept match permanently.
    candidate_limit = min(100, max(40, limit * 4))
    for term in fts_terms:
        term_hits = await deps.doc_store.search_document_sections(
            term,
            document_id=document_id,
            section_type=section_type,
            limit=candidate_limit,
            include_source_range=False,
        )
        for section in term_hits:
            # Per-term ranks come from different FTS queries and are not
            # comparable; surfacing them would mislead rank-gating clients.
            merged[_section_key(section)] = section.model_copy(update={"rank": None})

    texts = [_loose_search_text(f"{s.heading} {s.content}") for s in merged.values()]
    # Rarity already captures specificity. Squared character length favored
    # long title words over short substantive terms such as test, veri and THK.
    weights = {
        term: 1 + math.log((1 + len(texts)) / (1 + sum(term in text for text in texts)))
        if term not in _LOOSE_FTS_SKIP
        else 1.0
        for term in terms
    }
    ranked = sorted(
        merged.values(),
        key=lambda section: (-_loose_section_score(section, terms, weights), section.start_char, _section_key(section)),
    )
    top = ranked[:limit]
    # Only the hits that leave this function need their verbatim source range.
    return await deps.doc_store.hydrate_source_ranges(top) if top else []


def register(mcp, deps: Dependencies) -> None:
    """Register section tools on the given MCP instance."""

    @mcp.tool()
    @logged_tool(logger)
    async def get_document_section(
        document_id: DocumentId,
        section_type: SectionType = None,
        section_ref: SectionRef = None,
        heading: HeadingFilter = None,
        quotation: OptionalQuotation = None,
        start_char: OptionalSectionStartChar = None,
    ) -> DocumentSectionToolResult:
        """
        Retrieve exact structural sections from a stored BDDK document.

        Use when the user asks for a specific article/principle/paragraph such as
        `943 İlke 5` or `mevzuat_22599 Madde 9`.

        Args:
            document_id: Stored document ID, e.g. `943` or `mevzuat_22599`
            section_type: Optional exact section type, e.g. madde, gecici_madde, ek_madde, ilke, paragraf, ek
            section_ref: Optional exact section reference, e.g. 9 or 66/A (inserted article)
            heading: Optional heading substring filter
            quotation: Optional proposed quotation; verify it in the exact provision,
                preserving every character, including whitespace. An exact stored-reference
                match does not establish original-source fidelity, a paraphrase or a duty.
            start_char: Optional exact start offset that selects one of several sections
                sharing type, reference and heading; copy it from the disambiguation list.
                It is valid only for the corpus file being served and is not a citation.

        fikra and bent rows are search-oriented, not unique addresses: many rows can
        share one reference. Where a stored parent article contains a row it is shown
        as "Parent:"; fetch that parent article for the full provision.

        Supplying quotation returns answer_assessment with the mechanical match result.
        The quotation is compared with the section's legal text after its stored hash
        is confirmed; a storage truncation notice on a capped section is not legal text
        and is excluded, so reference_sha256 is the hash of exactly the text compared.
        A capped section holds only a 20,000-character window of its provision, so text
        beyond that window is not compared: a quotation absent from the window is
        reported as unavailable (gap quotation_reference_truncated), never not_found,
        and a match is never reported as the whole reference.
        Scope and semantic entailment still require review; never infer them from a
        quotation match.
        """
        sections = []
        for candidate in document_id_candidates(document_id):
            sections = await deps.doc_store.get_document_section(
                candidate,
                section_type=_normalize_optional(section_type),
                section_ref=_normalize_optional(section_ref),
                heading=heading,
                limit=_MAX_DISAMBIGUATION_RESULTS + 1,
                start_char=start_char,
            )
            if sections:
                break
        section_item = quality = None
        continuations = await _find_continuations(deps, sections[:_MAX_DISAMBIGUATION_RESULTS])
        parents = await _find_parents(deps, sections[:_MAX_DISAMBIGUATION_RESULTS])
        if len(sections) == 1:
            quality = _section_quality(sections[0])
            section_item = _with_parent(
                _section_item(sections[0], max_chars=_MAX_EXACT_SECTION_CHARS, continuations=continuations),
                sections[0],
                parents,
            )
        assessment = None
        assessment_text = ""
        if quotation is not None:
            assessment = _assess_quotation(sections, quotation)
            assessment_text = (
                f"\n\nAnswer evidence checks: {assessment.basis}\n"
                f"Quotation in stored reference (no normalization): {assessment.quotation_status}\n"
                "Original-source fidelity is NOT established by this character comparison.\n"
                f"Evidence gaps: {', '.join(assessment.gaps) or 'none in the requested mechanical checks'}\n"
                "Scope and semantic entailment are NOT assessed. A matching quotation does not prove "
                "a broader claim, obligation, frequency, threshold, or applicability to a particular bank."
            )
            if len(sections) == 1 and split_section_truncation_notice(sections[0].content)[1] is not None:
                assessment_text += f"\n{_CAPPED_QUOTATION_NOTE}"
        assessment_incomplete = bool(assessment and assessment.gaps)
        if not sections:
            query = " ".join(
                str(part) for part in (document_id, section_type or "", section_ref or "", heading or "") if part
            )
            output = (
                f"No section found for document {document_id} with the requested filters.\n"
                f"Try search_document_sections with query: {query or document_id}"
            )
            return structured_tool_result(
                DocumentSectionResponse(
                    status="no_results",
                    text=output + assessment_text,
                    answer_assessment=assessment,
                    requested_document_id=document_id,
                    section_type=_normalize_optional(section_type),
                    section_ref=_normalize_optional(section_ref),
                    heading=heading,
                )
            )

        if len(sections) == 1:
            assert section_item is not None and quality is not None
            section = sections[0]
            return structured_tool_result(
                DocumentSectionResponse(
                    status="partial" if section_item.content_truncated or assessment_incomplete else "ok",
                    text=_format_section(section, quality=quality, continuations=continuations, parents=parents)
                    + assessment_text,
                    answer_assessment=assessment,
                    evidence=[_section_evidence(section)],
                    warnings=_section_warnings(
                        [section], content_truncated=section_item.content_truncated, continuations=continuations
                    ),
                    requested_document_id=document_id,
                    section_type=_normalize_optional(section_type),
                    section_ref=_normalize_optional(section_ref),
                    heading=heading,
                    results=[section_item],
                )
            )

        result_sections = sections[:_MAX_DISAMBIGUATION_RESULTS]
        more_results = len(sections) > len(result_sections)
        lines = [
            (
                f"More than {_MAX_DISAMBIGUATION_RESULTS} sections matched. "
                if more_results
                else f"Multiple sections matched ({len(result_sections)}). "
            )
            + "Narrow by section_type, section_ref, or heading, or repeat the call with start_char=<value> "
            "from a line below to select exactly one section (start_char is valid only for this corpus file "
            "and is not a citation):\n"
        ]
        # Every listed section comes from the one resolved document.
        if result_sections[0].document_title:
            lines.append(f"- Document title: {result_sections[0].document_title}")
        lines.extend(_provenance_lines(result_sections[0], prefix="- "))
        for status_notice in dict.fromkeys(
            notice for section in result_sections if (notice := document_status_notice(section.document_category))
        ):
            lines.append(f"- Document status: {status_notice}")
        if any(section.section_type in {"fikra", "bent"} for section in result_sections):
            lines.append(_NESTED_ROW_HINT)
        for section in result_sections:
            heading_text = _heading_suffix(section, continuations)
            lines.append(
                f"- {section.doc_id} {section.section_type} {section.section_ref}{heading_text} "
                f"(start_char={section.start_char}, hash={section.content_hash[:12]})"
            )
            lines.extend(_parent_line(section, parents, prefix="  "))
            if label := _govde_label(section, continuations):
                lines.append(f"  Section kind: {label}")
            lines.extend(_quality_lines(_section_quality(section), prefix="  "))
            lines.append(f"  {_section_preview(section)}")
        if more_results:
            lines.append("... additional matches omitted; add a section reference or heading filter.")
        result_items = [
            _with_parent(_section_item(section, continuations=continuations), section, parents)
            for section in result_sections
        ]
        result_content_truncated = more_results or any(item.content_truncated for item in result_items)
        return structured_tool_result(
            DocumentSectionResponse(
                status="partial" if result_content_truncated or assessment_incomplete else "ok",
                text="\n".join(lines) + assessment_text,
                answer_assessment=assessment,
                evidence=[_section_evidence(section) for section in result_sections],
                warnings=_section_warnings(
                    result_sections,
                    content_truncated=result_content_truncated,
                    continuations=continuations,
                ),
                requested_document_id=document_id,
                section_type=_normalize_optional(section_type),
                section_ref=_normalize_optional(section_ref),
                heading=heading,
                results=result_items,
            )
        )

    @mcp.tool()
    @logged_tool(logger)
    async def search_document_sections(
        query: SectionQuery,
        document_id: OptionalDocumentId = None,
        section_type: SectionType = None,
        limit: SectionResultLimit = 10,
    ) -> SectionSearchToolResult:
        """
        Search section-level content in stored BDDK documents.

        Parses exact legal references from the query and uses them as filters when
        explicit arguments are not provided.

        Args:
            query: Turkish legal query, e.g. `943 İlke 5 model validasyonu`
            document_id: Optional document ID filter
            section_type: Optional section type filter
            limit: Maximum number of section results

        Nested fıkra/bent rows are omitted unless `section_type` requests them.
        `govde` rows are of three kinds: (1) the full-text windows of a document with
        no article structure (for example a Kurul Kararı), which are searched by
        default and cited by the document's title and decision date/number, not as
        an article; (2) the continuation of a named provision beyond its
        20,000-character stored window, which belongs to that provision and is
        labelled with it; (3) genuine remainder text of a structured document
        (preamble, footnotes, unparsed leftovers), which is not a legal provision.
        In a document with article structure, govde rows are omitted unless
        `section_type=govde` requests them.

        Each FTS hit includes a `Match rank` (SQLite FTS5 BM25 over the section
        heading and body, heading weighted 2x, sign-flipped so that higher is
        better): ranks are comparable within one query's results and can be
        used to gate low-confidence retrieval.

        When the exact-word pass returns fewer than `limit` hits, it is topped up
        with an inflection-tolerant pass (word prefixes, e.g. "süresi" also finds
        "süresinin"). Those hits follow the exact-word hits, carry no rank and are
        marked `stem_match`. The loose token fallback runs only when both passes
        find nothing.

        A bare number in the query that equals a stored document ID limits the
        search to that document only when the exact-reference, exact-word or
        inflection-tolerant pass finds something there; the text then says so and
        `inferred_document_filter_applied` is true. Otherwise the search runs over
        every document.
        """
        refs = parse_legal_refs(query)
        # An explicit document_id is a hard filter. A number inferred from the
        # query text is only a preference: it is ignored when no such document
        # is stored, and dropped when the exact-reference, strict and
        # inflection-tolerant passes find nothing in it; it never scopes the
        # loose fallback.
        query_doc_id = None if document_id else (refs.document_ids[0] if refs.document_ids else None)
        filter_drop_reason: str | None = None
        if query_doc_id and not await _any_document_stored(deps, query_doc_id):
            filter_drop_reason = "no stored document has that ID"
            query_doc_id = None
        inferred_doc_id = document_id or query_doc_id
        inferred_section_type = section_type or (refs.sections[0][0] if refs.sections else None)
        inferred_section_ref = refs.sections[0][1] if refs.sections else None
        exact_ref_detected = bool(inferred_doc_id and inferred_section_type and inferred_section_ref)
        candidate_doc_ids: list[str | None] = (
            list(document_id_candidates(inferred_doc_id)) if inferred_doc_id else [None]
        )

        exact_hits = []
        if exact_ref_detected:
            for candidate_doc_id in candidate_doc_ids:
                exact_hits = await deps.doc_store.get_document_section(
                    candidate_doc_id,
                    section_type=_normalize_optional(inferred_section_type),
                    section_ref=_normalize_optional(inferred_section_ref),
                    limit=limit,
                )
                if exact_hits:
                    break
        hits, stem_keys = await _search_sections_strict_with_stem_top_up(
            deps,
            query,
            candidate_doc_ids=candidate_doc_ids,
            section_type=_normalize_optional(inferred_section_type),
            limit=limit,
        )
        loose_fallback_used = False
        if not hits and not exact_hits and query_doc_id:
            # Nothing exact or strict inside the inferred document: a loose match there must not
            # outrank a strict match elsewhere, so repeat strict, then loose, with no document scope.
            filter_drop_reason = f"no section in document {query_doc_id} matched the query"
            inferred_doc_id = None
            # With the inferred document dropped no document is identified, so the section reference
            # was not applied to the unscoped search.
            exact_ref_detected = False
            hits, stem_keys = await _search_sections_strict_with_stem_top_up(
                deps,
                query,
                candidate_doc_ids=[None],
                section_type=_normalize_optional(inferred_section_type),
                limit=limit,
            )
            candidate_doc_ids = [None]
        if not hits and not exact_hits:
            # The in-document loose loop runs only for an explicit document_id argument.
            for candidate_doc_id in candidate_doc_ids:
                hits = await _search_sections_loose(
                    deps,
                    query,
                    document_id=candidate_doc_id,
                    section_type=_normalize_optional(inferred_section_type),
                    limit=limit,
                )
                if hits:
                    break
            loose_fallback_used = bool(hits)
        # A section reference without an identified document cannot be looked up, so it is not applied.
        # Computed after the drop decision so that a dropped inferred document reports it as not applied.
        unapplied_ref_note = (
            f"Note: {refs.sections[0][0]} {inferred_section_ref} in the query was not applied as a filter because "
            "no document was identified; find the document id with search_bddk_regulations, then call "
            f"get_document_section(document_id, section_type='{refs.sections[0][0]}', "
            f"section_ref='{inferred_section_ref}')."
            if inferred_section_ref and not exact_ref_detected
            else None
        )
        reported_section_ref = None if unapplied_ref_note else _normalize_optional(inferred_section_ref)
        filter_dropped = filter_drop_reason is not None
        inferred_filter_applied = bool(query_doc_id and not filter_dropped)
        filter_note = (
            f"Note: the number {refs.document_ids[0]} in the query was not applied as a document filter "
            f"({filter_drop_reason}).\n"
            if filter_dropped
            else (
                f"Note: the search was limited to document {query_doc_id} because of the number {query_doc_id} "
                "in the query; pass document_id or remove the number to search every document.\n"
                if inferred_filter_applied
                else ""
            )
        )
        if exact_hits:
            # Exact-ref lookups carry no FTS rank; inherit it from the FTS
            # duplicate being deduplicated away so the top hit is not the
            # only unscored result.
            fts_ranks = {_section_key(s): s.rank for s in hits if s.rank is not None}
            exact_hits = [
                s.model_copy(update={"rank": fts_ranks.get(_section_key(s))}) if s.rank is None else s
                for s in exact_hits
            ]
            seen = {_section_key(section) for section in exact_hits}
            hits = (exact_hits + [section for section in hits if _section_key(section) not in seen])[:limit]
            # The explicitly requested provision is an exact-reference hit even when the prefix top-up
            # also found it: never label it as an inflection-tolerant match.
            stem_keys -= seen
        if not hits:
            passed_filters = [
                name for name, value in (("document_id", document_id), ("section_type", section_type)) if value
            ]
            remove_hint = f"remove the {'/'.join(passed_filters)} filter, " if passed_filters else ""
            output = (
                f"NO RESULTS: No document sections found matching '{query}'.\n"
                + filter_note
                + (f"{unapplied_ref_note}\n" if unapplied_ref_note else "")
                + f"Try a broader query, {remove_hint}or retrieve the full document."
            )
            return structured_tool_result(
                SectionSearchResponse(
                    status="no_results",
                    text=output,
                    query=query,
                    document_id=inferred_doc_id,
                    section_type=_normalize_optional(inferred_section_type),
                    section_ref=reported_section_ref,
                    parsed_section_ref=_normalize_optional(inferred_section_ref),
                    exact_reference_detected=exact_ref_detected,
                    loose_fallback_used=False,
                    warnings=[unapplied_ref_note] if unapplied_ref_note else [],
                    inferred_document_filter_dropped=filter_dropped,
                )
            )

        continuations = await _find_continuations(deps, hits)
        parents = await _find_parents(deps, hits)
        lines = [f"Found {len(hits)} section result(s) for '{query}':\n"]
        if unapplied_ref_note:
            lines.insert(0, f"{unapplied_ref_note}\n")
        if filter_note:
            lines.insert(0, filter_note)
        has_exact_word_hits = any(_section_key(hit) not in stem_keys for hit in hits)
        for hit in hits:
            heading = _heading_suffix(hit, continuations)
            lines.append(f"**{hit.doc_id} — {hit.section_type} {hit.section_ref}{heading}**")
            lines.append(f"  Document ID: {hit.doc_id}")
            if hit.document_title:
                lines.append(f"  Document title: {hit.document_title}")
            if status_notice := document_status_notice(hit.document_category):
                lines.append(f"  Document status: {status_notice}")
            lines.extend(_provenance_lines(hit, prefix="  "))
            if hit.document_source_url:
                lines.append(f"  Catalog source (not a validated citation): {hit.document_source_url}")
            lines.append(f"  Section: {hit.section_type} {hit.section_ref}")
            lines.extend(_parent_line(hit, parents, prefix="  "))
            if label := _govde_label(hit, continuations):
                lines.append(f"  Note: {label}")
            elif hit.section_type == "govde":
                lines.append("  Note: govde remainder — not a legal provision identity")
            lines.append(
                f"  Normalized Markdown code-point range: [{hit.start_char}, {hit.end_char}); not source PDF pages"
            )
            if hit.rank is not None:
                # "relative" guards against conflation with the percent-scale
                # relevance gate in the server instructions (store search).
                lines.append(f"  Match rank: {hit.rank:.4f} (relative, FTS)")
            if _section_key(hit) in stem_keys:
                lines.append(
                    "  Match: inflection-tolerant match (word prefixes of the query terms, not the exact words); "
                    + ("no rank, listed after the exact-word hits" if has_exact_word_hits else "no rank")
                )
            lines.extend(_quality_lines(_section_quality(hit), prefix="  "))
            excerpt, truncated, _, _ = _section_excerpt(hit, max_chars=_MAX_SEARCH_EXCERPT_CHARS, query=query)
            if excerpt:
                lines.append(excerpt)
                if truncated:
                    lines.append("  [excerpt truncated — call get_document_section for the full section]")
            lines.append("")
        result_items = [
            _with_parent(_section_item(hit, query=query, continuations=continuations), hit, parents).model_copy(
                update={"stem_match": _section_key(hit) in stem_keys}
            )
            for hit in hits
        ]
        result_content_truncated = any(item.content_truncated for item in result_items)
        return structured_tool_result(
            SectionSearchResponse(
                status="partial" if result_content_truncated or (exact_ref_detected and not exact_hits) else "ok",
                text="\n".join(lines),
                evidence=[_section_evidence(hit) for hit in hits],
                warnings=[
                    *_section_warnings(hits, content_truncated=result_content_truncated, continuations=continuations),
                    *([_EXACT_REFERENCE_MISSING_WARNING] if exact_ref_detected and not exact_hits else []),
                    *([unapplied_ref_note] if unapplied_ref_note else []),
                    *(
                        [
                            "Loose token fallback was used: results may match only part of the query. "
                            "Search each substantive issue separately and verify exact provisions; "
                            "this ranking is not a legal-confidence score."
                        ]
                        if loose_fallback_used
                        else []
                    ),
                ],
                query=query,
                document_id=inferred_doc_id,
                section_type=_normalize_optional(inferred_section_type),
                section_ref=reported_section_ref,
                parsed_section_ref=_normalize_optional(inferred_section_ref),
                exact_reference_detected=exact_ref_detected,
                loose_fallback_used=loose_fallback_used,
                inferred_document_filter_dropped=filter_dropped,
                inferred_document_filter_applied=inferred_filter_applied,
                results=result_items,
            )
        )
