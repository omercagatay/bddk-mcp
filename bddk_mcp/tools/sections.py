"""Section-level document retrieval and search tools."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import TYPE_CHECKING

from pydantic import ValidationError

from bddk_mcp.citations import (
    CitationQuality,
    CitationV1,
    NormalizedTextRange,
    TrustedCitationContext,
    build_normalized_range_citation,
    section_retrieval_profile_sha256,
)
from bddk_mcp.observability.telemetry import elapsed_ms, record_tool_call_trace, unique_doc_ids
from bddk_mcp.quality.markdown_quality import (
    QualityAssessment,
    assess_markdown_quality,
    sanitize_markdown_for_context,
)
from bddk_mcp.regulatory.graph_queries import one_hop_section_refs
from bddk_mcp.store.legal_ref import document_id_candidates, parse_legal_refs, turkish_casefold
from bddk_mcp.tools.contract_types import (
    DocumentId,
    ExpandReferences,
    HeadingFilter,
    OptionalDocumentId,
    SectionQuery,
    SectionRef,
    SectionResultLimit,
    SectionType,
)
from bddk_mcp.tools.structured_outputs import (
    UNTRUSTED_SOURCE_WARNING,
    DocumentSectionResponse,
    DocumentSectionToolResult,
    EvidenceReference,
    QualityMetadata,
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
_GOVDE_WARNING = (
    "One or more hits are govde remainder (unparsed body/footnote text), not madde/ilke/paragraf identities. "
    "Do not cite them as a legal provision."
)
_CITATION_UNAVAILABLE_NO_MAPPING = (
    "[citation_v1_unavailable_no_validated_mapping] Citation v1 was not emitted: this section has no "
    "validated authoritative, non-fixture legal-version occurrence mapping."
)
_CITATION_UNAVAILABLE_TRUNCATED = (
    "[citation_v1_unavailable_truncated] Citation v1 was not emitted because the returned section body is truncated."
)
_CITATION_UNAVAILABLE_QUALITY_FAILURE = (
    "[citation_v1_unavailable_quality_failure] Citation v1 was not emitted because extraction quality is failed."
)
_CITATION_UNAVAILABLE_RECONSTRUCTION = (
    "[citation_v1_unavailable_reconstruction_mismatch] Citation v1 was not emitted because the returned text "
    "could not be reconstructed exactly from the validated normalized-document range."
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
    return assess_markdown_quality(section.content, document_id=section.doc_id)


def _quality_lines(quality: QualityAssessment, *, prefix: str) -> list[str]:
    """Return consistent user-visible quality metadata for a non-clean section."""
    if quality.label not in {"warning", "fail"}:
        return []
    flags = ", ".join(quality.flags) if quality.flags else "none"
    lines = [f"{prefix}Quality: {quality.label}", f"{prefix}Quality flags: {flags}"]
    if quality.warning:
        lines.append(f"{prefix}⚠ Quality warning: {quality.warning}")
    return lines


def _quality_labels(sections: list[StoredDocumentSection]) -> dict[str, dict[str, object]]:
    labels: dict[str, dict[str, object]] = {}
    for section in sections:
        quality = _section_quality(section)
        labels[section.doc_id] = {"label": quality.label, "flags": quality.flags}
    return labels


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

    raw_content = section.content or ""
    if len(raw_content) <= max_chars:
        local_start = 0
        local_end = len(raw_content)
    else:
        match_offset = 0
        if query:
            folded = raw_content.casefold()
            candidates = [query.casefold(), *(_loose_search_terms(query))]
            offsets = [folded.find(candidate) for candidate in candidates if candidate]
            found_offsets = [offset for offset in offsets if offset >= 0]
            if found_offsets:
                match_offset = min(found_offsets)
        local_start = max(0, min(match_offset - max_chars // 4, len(raw_content) - max_chars))
        local_end = min(len(raw_content), local_start + max_chars)

    excerpt = sanitize_markdown_for_context(raw_content[local_start:local_end])
    # Context sanitization can add line wraps. Keep the serialized MCP field
    # within the advertised bound as well as the raw source slice.
    while len(excerpt) > max_chars and local_end > local_start:
        local_end -= min(local_end - local_start, len(excerpt) - max_chars)
        excerpt = sanitize_markdown_for_context(raw_content[local_start:local_end])
    return (
        excerpt,
        local_start > 0 or local_end < len(raw_content),
        section.start_char + local_start,
        section.start_char + local_end,
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


def _section_evidence(
    section: StoredDocumentSection,
    *,
    citation: CitationV1 | None = None,
) -> EvidenceReference:
    return EvidenceReference(
        document_id=section.doc_id,
        source_url=citation.source_url if citation else None,
        retrieval_source="section_index",
        page_start=section.page_start,
        page_end=section.page_end,
        section_type=section.section_type,
        section_ref=section.section_ref,
        start_char=section.start_char,
        end_char=section.end_char,
        content_hash=section.content_hash,
        quality=_quality_metadata(_section_quality(section)),
        citation=citation,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _build_exact_section_citation(
    section: StoredDocumentSection,
    *,
    section_item: SectionItem,
    quality: QualityAssessment,
) -> tuple[CitationV1 | None, str | None]:
    """Build Citation v1 only for an exact, complete, validated occurrence."""

    mapping = section.citation_mapping
    if mapping is None:
        return None, _CITATION_UNAVAILABLE_NO_MAPPING
    if section_item.content_truncated:
        return None, _CITATION_UNAVAILABLE_TRUNCATED
    if quality.label == "fail":
        return None, _CITATION_UNAVAILABLE_QUALITY_FAILURE

    normalized_source_range = section.normalized_source_range
    try:
        trusted = TrustedCitationContext(
            instrument_id=mapping.instrument_id,
            instrument_jurisdiction=mapping.instrument_jurisdiction,
            instrument_authority_code=mapping.instrument_authority_code,
            instrument_identity_key=mapping.instrument_identity_key,
            legal_version_id=mapping.legal_version_id,
            legal_version_key=mapping.legal_version_key,
            legal_validation_record_sha256=mapping.legal_validation_record_sha256,
            provision_validation_record_sha256=mapping.provision_validation_record_sha256,
            artifact_id=mapping.artifact_id,
            artifact_blob_id=mapping.artifact_blob_id,
            artifact_sha256=mapping.artifact_sha256,
            source_url=mapping.source_url,
            artifact_retrieved_at=mapping.artifact_retrieved_at,
            source_document_id=section.doc_id,
            normalized_document_sha256=section.source_content_hash,
            evidence_id=mapping.evidence_id,
            evidence_locator=mapping.evidence_locator,
            evidence_statement_sha256=mapping.evidence_statement_sha256,
            provision_id=mapping.provision_id,
            provision_kind=mapping.provision_kind,
            provision_path=mapping.provision_path,
            provision_text_sha256=section.content_hash,
            locator=NormalizedTextRange(
                start_char=section.start_char,
                end_char=section.end_char,
                normalized_range_sha256=_sha256_text(normalized_source_range),
            ),
            excerpt_sha256=_sha256_text(section_item.content),
            excerpt_length=len(section_item.content),
            retrieval_profile_sha256=section_retrieval_profile_sha256(),
            quality=CitationQuality(
                label=quality.label,
                flags=tuple(sorted(set(quality.flags))),
                warning=quality.warning or None,
            ),
        )
        citation = build_normalized_range_citation(
            trusted=trusted,
            provision_text=section.content,
            normalized_source_range=normalized_source_range,
            rendered_excerpt=section_item.content,
        )
    except (ValidationError, ValueError):
        logger.warning("Citation v1 reconstruction rejected for a retrieved section")
        return None, _CITATION_UNAVAILABLE_RECONSTRUCTION
    return citation, None


def _citation_lines(citation: CitationV1) -> list[str]:
    locator = citation.locator
    return [
        "",
        "#### Citation v1",
        f"- Citation ID: {citation.citation_id}",
        f"- Instrument ID: {citation.instrument_id}",
        f"- Legal version ID: {citation.legal_version_id}",
        f"- Artifact ID: {citation.artifact_id}",
        f"- Evidence ID: {citation.evidence_id}",
        f"- Provision ID: {citation.provision_id}",
        f"- Authoritative source: {citation.source_url}",
        f"- Normalized document SHA-256: {citation.normalized_document_sha256}",
        f"- Provision SHA-256: {citation.provision_text_sha256}",
        (f"- Normalized Markdown code-point range: [{locator.start_char}, {locator.end_char}); not source PDF pages"),
        f"- Normalized range SHA-256: {locator.normalized_range_sha256}",
        f"- Returned excerpt SHA-256: {citation.excerpt_sha256}",
    ]


def _section_warnings(
    sections: list[StoredDocumentSection],
    *,
    content_truncated: bool = False,
) -> list[str]:
    quality_warnings = list(
        dict.fromkeys(quality.warning for section in sections if (quality := _section_quality(section)).warning)
    )
    warnings = [UNTRUSTED_SOURCE_WARNING, *quality_warnings] if sections else quality_warnings
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
    citation: CitationV1 | None = None,
) -> str:
    heading = f" — {section.heading}" if section.heading else ""
    lines = [
        f"### {section.doc_id} — {section.section_type} {section.section_ref}{heading}",
        f"- Document ID: {section.doc_id}",
        f"- Section: {section.section_type} {section.section_ref}",
        (f"- Normalized Markdown code-point range: [{section.start_char}, {section.end_char}); not source PDF pages"),
    ]
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
    if citation is not None:
        lines.extend(_citation_lines(citation))
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


def _loose_search_terms(query: str, *, limit: int = 16) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw_term in query.strip().split():
        term = turkish_casefold(re.sub(r"[^\w]+", "", raw_term, flags=re.UNICODE))
        if len(term) < 3 or term in _LOOSE_SECTION_SEARCH_STOPWORDS or term in seen:
            continue
        terms.append(term)
        seen.add(term)
        if len(terms) >= limit:
            break
    return terms


def _loose_section_score(section: StoredDocumentSection, terms: list[str]) -> int:
    text = turkish_casefold(f"{section.heading} {section.content}")
    score = sum(len(term) ** 2 for term in terms if term in text)
    if "uyumsuzluk" in terms and "uyumsuzluk" in text:
        score += 400
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
    fts_terms = [term for term in terms if len(term) >= 5 and term not in _LOOSE_FTS_SKIP]
    if not fts_terms:
        fts_terms = terms

    merged: dict[tuple[str, str, str, str], StoredDocumentSection] = {}
    for term in fts_terms:
        term_hits = await deps.doc_store.search_document_sections(
            term,
            document_id=document_id,
            section_type=section_type,
            limit=limit,
        )
        for section in term_hits:
            # Per-term ranks come from different tsqueries and are not
            # comparable; surfacing them would mislead rank-gating clients.
            merged[_section_key(section)] = section.model_copy(update={"rank": None})

    ranked = sorted(
        merged.values(),
        key=lambda section: (-_loose_section_score(section, terms), section.start_char),
    )
    return ranked[:limit]


def register(mcp, deps: Dependencies) -> None:
    """Register section tools on the given MCP instance."""

    @mcp.tool()
    @logged_tool(logger)
    async def get_document_section(
        document_id: DocumentId,
        section_type: SectionType = None,
        section_ref: SectionRef = None,
        heading: HeadingFilter = None,
    ) -> DocumentSectionToolResult:
        """
        Retrieve exact structural sections from a stored BDDK document.

        Use when the user asks for a specific article/principle/paragraph such as
        `943 İlke 5` or `mevzuat_22599 Madde 9`.

        Args:
            document_id: Stored document ID, e.g. `943` or `mevzuat_22599`
            section_type: Optional exact section type, e.g. madde, ilke, paragraf, ek
            section_ref: Optional exact section reference, e.g. 9 or 5
            heading: Optional heading substring filter
        """
        start = time.perf_counter()
        args = {
            "document_id": document_id,
            "section_type": section_type,
            "section_ref": section_ref,
            "heading": heading,
        }
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
        if not sections:
            query = " ".join(
                str(part) for part in (document_id, section_type or "", section_ref or "", heading or "") if part
            )
            await record_tool_call_trace(
                getattr(deps, "telemetry_pool", None),
                tool_name="get_document_section",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=0,
                doc_ids=[],
                relevance_stats={"status": "not_found"},
            )
            output = (
                f"No section found for document {document_id} with the requested filters.\n"
                f"Try search_document_sections with query: {query or document_id}"
            )
            return structured_tool_result(
                DocumentSectionResponse(
                    status="no_results",
                    text=output,
                    requested_document_id=document_id,
                    section_type=_normalize_optional(section_type),
                    section_ref=_normalize_optional(section_ref),
                    heading=heading,
                )
            )

        if len(sections) == 1:
            quality = _section_quality(sections[0])
            section_item = _section_item(sections[0], max_chars=_MAX_EXACT_SECTION_CHARS)
            citation, citation_warning = _build_exact_section_citation(
                sections[0],
                section_item=section_item,
                quality=quality,
            )
            await record_tool_call_trace(
                getattr(deps, "telemetry_pool", None),
                tool_name="get_document_section",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=1,
                doc_ids=[sections[0].doc_id],
                quality_labels={sections[0].doc_id: {"label": quality.label, "flags": quality.flags}},
                relevance_stats={"status": "exact_match"},
            )
            section = sections[0]
            return structured_tool_result(
                DocumentSectionResponse(
                    status="partial" if section_item.content_truncated else "ok",
                    text=_format_section(section, quality=quality, citation=citation),
                    evidence=[_section_evidence(section, citation=citation)],
                    warnings=[
                        *_section_warnings(
                            [section],
                            content_truncated=section_item.content_truncated,
                        ),
                        *([citation_warning] if citation_warning else []),
                    ],
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
        await record_tool_call_trace(
            getattr(deps, "telemetry_pool", None),
            tool_name="get_document_section",
            args=args,
            latency_ms=elapsed_ms(start),
            result_count=len(result_sections),
            doc_ids=unique_doc_ids([section.doc_id for section in result_sections]),
            quality_labels=_quality_labels(result_sections),
            relevance_stats={"status": "disambiguation", "additional_matches_omitted": more_results},
        )
        return structured_tool_result(
            DocumentSectionResponse(
                status="partial" if result_content_truncated else "ok",
                text="\n".join(lines),
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
        expand_references: ExpandReferences = False,
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
            expand_references: Doğrulanmış çapraz referans kenarlarını 1 adım
                takip ederek ilişkili bölüm etiketleri ekle (içerik dahil edilmez)

        Unparsed `govde` remainder and nested fıkra/bent rows are omitted unless
        `section_type` requests them.

        Each FTS hit includes a `Match rank` (length-normalized ts_rank_cd):
        higher is better; ranks are comparable within one query's results and
        can be used to gate low-confidence retrieval.
        """
        start = time.perf_counter()
        args = {
            "query": query,
            "document_id": document_id,
            "section_type": section_type,
            "limit": limit,
            "expand_references": expand_references,
        }
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
            await record_tool_call_trace(
                getattr(deps, "telemetry_pool", None),
                tool_name="search_document_sections",
                args=args,
                latency_ms=elapsed_ms(start),
                result_count=0,
                doc_ids=[],
                relevance_stats={"exact_ref_detected": exact_ref_detected, "status": "no_results"},
            )
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
        if expand_references and deps.pool is not None:
            expanded_blocks: list[str] = []
            for hit in hits:
                try:
                    neighbors = await one_hop_section_refs(
                        deps.pool,
                        doc_id=hit.doc_id,
                        section_type=hit.section_type,
                        section_ref=hit.section_ref,
                        limit=3,
                    )
                except Exception:  # noqa: BLE001 — expansion must never break search
                    neighbors = []
                if not neighbors:
                    continue
                block = [
                    f"#### İlişkili bölümler (doğrulanmış kenarlar) — {hit.doc_id} {hit.section_type} {hit.section_ref}"
                ]
                block.extend(
                    f"- {n['doc_id']} — {n['section_type']} {n['section_ref']} (kenar: {n['relation_type']})"
                    for n in neighbors
                )
                expanded_blocks.append("\n".join(block))
            if expanded_blocks:
                lines.append("\n\n".join(expanded_blocks))
        await record_tool_call_trace(
            getattr(deps, "telemetry_pool", None),
            tool_name="search_document_sections",
            args=args,
            latency_ms=elapsed_ms(start),
            result_count=len(hits),
            doc_ids=unique_doc_ids([hit.doc_id for hit in hits]),
            quality_labels=_quality_labels(hits),
            relevance_stats={"exact_ref_detected": exact_ref_detected, "loose_fallback": loose_fallback_used},
        )
        result_items = [_section_item(hit, query=query) for hit in hits]
        result_content_truncated = any(item.content_truncated for item in result_items)
        return structured_tool_result(
            SectionSearchResponse(
                status="partial" if result_content_truncated else "ok",
                text="\n".join(lines),
                evidence=[_section_evidence(hit) for hit in hits],
                warnings=_section_warnings(hits, content_truncated=result_content_truncated),
                query=query,
                document_id=inferred_doc_id,
                section_type=_normalize_optional(inferred_section_type),
                section_ref=_normalize_optional(inferred_section_ref),
                exact_reference_detected=exact_ref_detected,
                loose_fallback_used=loose_fallback_used,
                results=result_items,
            )
        )
