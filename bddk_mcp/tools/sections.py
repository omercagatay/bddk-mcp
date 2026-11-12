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
from bddk_mcp.store.legal_ref import document_id_candidates, parse_legal_refs, turkish_casefold
from bddk_mcp.store.section_index import split_section_truncation_notice
from bddk_mcp.tools.contract_types import (
    DocumentId,
    HeadingFilter,
    OptionalDocumentId,
    OptionalQuotation,
    SectionQuery,
    SectionRef,
    SectionResultLimit,
    SectionType,
)
from bddk_mcp.tools.errors import tool_error
from bddk_mcp.tools.structured_outputs import (
    UNTRUSTED_SOURCE_WARNING,
    DocumentSectionResponse,
    DocumentSectionToolResult,
    EvidenceReference,
    QualityMetadata,
    SectionAnswerAssessment,
    SectionItem,
    SectionSearchResponse,
    SectionSearchToolResult,
    structured_tool_result,
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
_LEGAL_APPLICABILITY_WARNING = (
    "Text retrieval does not establish current legal applicability. Verify the instrument's scope and its "
    "publication/amendment status against the official source for the required date; if that evidence is "
    "absent, abstain from a current-law conclusion. Keep TFRS 9/BKZ, problem-loan resolution, and IRB/IDD "
    "requirements distinct."
)
_NESTED_ROW_HINT = (
    "fikra and bent rows are search-oriented and carry no parent madde, so they are not unique addresses; "
    "retrieve the parent madde (section_type=madde) for the full provision."
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


def _section_item(
    section: StoredDocumentSection,
    *,
    max_chars: int = _MAX_SEARCH_EXCERPT_CHARS,
    query: str | None = None,
) -> SectionItem:
    content, truncated, excerpt_start, excerpt_end = _section_excerpt(
        section,
        max_chars=max_chars,
        query=query,
    )
    return SectionItem(
        document_id=section.doc_id,
        section_type=section.section_type,
        section_ref=section.section_ref,
        heading=section.heading,
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
    )


def _section_evidence(section: StoredDocumentSection) -> EvidenceReference:
    return EvidenceReference(
        document_id=section.doc_id,
        title=section.document_title or None,
        source_url=section.document_source_url or None,
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
        legal_text, _notice = split_section_truncation_notice(section.content)
        check = check_exact_quotation(
            legal_text,
            quotation,
            expected_reference_sha256=hashlib.sha256(legal_text.encode("utf-8")).hexdigest(),
        )
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
) -> list[str]:
    quality_warnings = list(
        dict.fromkeys(quality.warning for section in sections if (quality := _section_quality(section)).warning)
    )
    warnings = (
        [UNTRUSTED_SOURCE_WARNING, _LEGAL_APPLICABILITY_WARNING, *quality_warnings] if sections else quality_warnings
    )
    if content_truncated:
        warnings.append(_SECTION_TRUNCATION_WARNING)
    if any(section.section_type == "govde" for section in sections):
        warnings.append(_GOVDE_WARNING)
    return warnings


def _format_section(
    section: StoredDocumentSection,
    *,
    include_content: bool = True,
    quality: QualityAssessment | None = None,
) -> str:
    heading = f" — {section.heading}" if section.heading else ""
    lines = [
        f"### {section.doc_id} — {section.section_type} {section.section_ref}{heading}",
        f"- Document ID: {section.doc_id}",
        f"- Section: {section.section_type} {section.section_ref}",
        (f"- Normalized Markdown code-point range: [{section.start_char}, {section.end_char}); not source PDF pages"),
    ]
    if section.document_title:
        lines.append(f"- Document title: {section.document_title}")
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
            # Per-term ranks come from different tsqueries and are not
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

        fikra and bent rows are search-oriented, not unique addresses: they carry no
        parent madde, so many rows can share one reference. Fetch the parent madde for
        the full provision.

        Supplying quotation returns answer_assessment with the mechanical match result.
        The quotation is compared with the section's legal text after its stored hash
        is confirmed; a storage truncation notice on a capped section is not legal text
        and is excluded, so reference_sha256 is the hash of exactly the text compared.
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
            )
            if sections:
                break
        section_item = quality = None
        if len(sections) == 1:
            quality = _section_quality(sections[0])
            section_item = _section_item(sections[0], max_chars=_MAX_EXACT_SECTION_CHARS)
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
                    text=_format_section(section, quality=quality) + assessment_text,
                    answer_assessment=assessment,
                    evidence=[_section_evidence(section)],
                    warnings=_section_warnings([section], content_truncated=section_item.content_truncated),
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
            + "Narrow by section_type, section_ref, or heading:\n"
        ]
        if any(section.section_type in {"fikra", "bent"} for section in result_sections):
            lines.append(_NESTED_ROW_HINT)
        for section in result_sections:
            heading_text = f" — {section.heading}" if section.heading else ""
            lines.append(
                f"- {section.doc_id} {section.section_type} {section.section_ref}{heading_text} "
                f"(start_char={section.start_char}, hash={section.content_hash[:12]})"
            )
            lines.extend(_quality_lines(_section_quality(section), prefix="  "))
            lines.append(f"  {_section_preview(section)}")
        if more_results:
            lines.append("... additional matches omitted; add a section reference or heading filter.")
        result_items = [_section_item(section) for section in result_sections]
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

        Unparsed `govde` remainder and nested fıkra/bent rows are omitted unless
        `section_type` requests them.

        Each FTS hit includes a `Match rank` (SQLite FTS5 BM25 over the section
        heading and body, heading weighted 2x, sign-flipped so that higher is
        better): ranks are comparable within one query's results and can be
        used to gate low-confidence retrieval.
        """
        refs = parse_legal_refs(query)
        inferred_doc_id = document_id or (refs.document_ids[0] if refs.document_ids else None)
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
        hits = []
        for candidate_doc_id in candidate_doc_ids:
            hits = await deps.doc_store.search_document_sections(
                query,
                document_id=candidate_doc_id,
                section_type=_normalize_optional(inferred_section_type),
                limit=limit,
            )
            if hits:
                break
        loose_fallback_used = False
        if not hits and not exact_hits:
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
        if not hits:
            output = (
                f"NO RESULTS: No document sections found matching '{query}'.\n"
                "Try a broader query, remove the document_id/section_type filter, or retrieve the full document."
            )
            return structured_tool_result(
                SectionSearchResponse(
                    status="no_results",
                    text=output,
                    query=query,
                    document_id=inferred_doc_id,
                    section_type=_normalize_optional(inferred_section_type),
                    section_ref=_normalize_optional(inferred_section_ref),
                    exact_reference_detected=exact_ref_detected,
                    loose_fallback_used=False,
                )
            )

        lines = [f"Found {len(hits)} section result(s) for '{query}':\n"]
        for hit in hits:
            heading = f" — {hit.heading}" if hit.heading else ""
            lines.append(f"**{hit.doc_id} — {hit.section_type} {hit.section_ref}{heading}**")
            lines.append(f"  Document ID: {hit.doc_id}")
            if hit.document_title:
                lines.append(f"  Document title: {hit.document_title}")
            if hit.document_source_url:
                lines.append(f"  Catalog source (not a validated citation): {hit.document_source_url}")
            lines.append(f"  Section: {hit.section_type} {hit.section_ref}")
            if hit.section_type == "govde":
                lines.append("  Note: govde remainder — not a legal provision identity")
            lines.append(
                f"  Normalized Markdown code-point range: [{hit.start_char}, {hit.end_char}); not source PDF pages"
            )
            if hit.rank is not None:
                # "relative" guards against conflation with the percent-scale
                # relevance gate in the server instructions (store search).
                lines.append(f"  Match rank: {hit.rank:.4f} (relative, FTS)")
            lines.extend(_quality_lines(_section_quality(hit), prefix="  "))
            excerpt, truncated, _, _ = _section_excerpt(hit, max_chars=_MAX_SEARCH_EXCERPT_CHARS, query=query)
            if excerpt:
                lines.append(excerpt)
                if truncated:
                    lines.append("  [excerpt truncated — call get_document_section for the full section]")
            lines.append("")
        result_items = [_section_item(hit, query=query) for hit in hits]
        result_content_truncated = any(item.content_truncated for item in result_items)
        return structured_tool_result(
            SectionSearchResponse(
                status="partial" if result_content_truncated or (exact_ref_detected and not exact_hits) else "ok",
                text="\n".join(lines),
                evidence=[_section_evidence(hit) for hit in hits],
                warnings=[
                    *_section_warnings(hits, content_truncated=result_content_truncated),
                    *([_EXACT_REFERENCE_MISSING_WARNING] if exact_ref_detected and not exact_hits else []),
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
                section_ref=_normalize_optional(inferred_section_ref),
                exact_reference_detected=exact_ref_detected,
                loose_fallback_used=loose_fallback_used,
                results=result_items,
            )
        )
