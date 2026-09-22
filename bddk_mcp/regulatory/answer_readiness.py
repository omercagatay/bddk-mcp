"""Link exact source quotations to independently validated dated-version evidence.

No model, heuristic entailment score, or caller-supplied legal approval is used.
A positive quotation check says only that the words occur in this provision.
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import TYPE_CHECKING, Any

from bddk_mcp.quotations import check_exact_quotation
from bddk_mcp.regulatory.status_repository import RegulationStatusRepositoryError, resolve_regulation_status
from bddk_mcp.store.section_index import split_section_truncation_notice
from bddk_mcp.tools.structured_outputs import SectionAnswerAssessment

if TYPE_CHECKING:
    from bddk_mcp.citations import CitationV1
    from bddk_mcp.store.doc_store import StoredDocumentSection
    from bddk_mcp.tools.structured_outputs import SectionItem


async def assess_section_answer(
    pool: Any,
    sections: list[StoredDocumentSection],
    *,
    item: SectionItem | None = None,
    citation: CitationV1 | None = None,
    as_of: date | None = None,
    quotation: str | None = None,
) -> SectionAnswerAssessment:
    """Check server-retrieved evidence; never trust a client's citation/status assertion."""
    result = SectionAnswerAssessment(
        basis="insufficient",
        as_of=as_of,
        quotation_status="unavailable" if quotation is not None else "not_requested",
    )
    if len(sections) != 1 or item is None:
        result.gaps.append("section_not_found" if not sections else "ambiguous_section")
        return result

    section = sections[0]
    if section.section_type == "govde":
        result.gaps.append("unparsed_section")
    if item.content_truncated:
        result.gaps.append("section_truncated")
    if item.quality.label in {"fail", "unknown"}:
        result.gaps.append("extraction_quality_unverified")
    source = section.normalized_source_range
    # The stored truncation notice is storage metadata; the legal characters are
    # the notice-free prefix. Verbatim checks use only those characters.
    verbatim_content, truncation_notice = split_section_truncation_notice(section.content)
    leading_space = len(source) - len(source.lstrip()) if source and source.strip() == verbatim_content else 0
    excerpt_start = item.excerpt_start_char - section.start_char - leading_space
    excerpt_end = item.excerpt_end_char - section.start_char - leading_space
    excerpt_is_exact_slice = (
        0 <= excerpt_start <= excerpt_end <= len(verbatim_content)
        and verbatim_content[excerpt_start:excerpt_end] == item.content
        and len(item.content) == item.excerpt_end_char - item.excerpt_start_char
    )
    if (
        not source
        or source.strip() != section.content
        # Verbatim contract: the returned excerpt must be the exact stored characters.
        # A rewritten/sanitized/summarized excerpt must never verify as source evidence.
        or not excerpt_is_exact_slice
        or (not item.content_truncated and item.content != verbatim_content)
        or (truncation_notice is not None and not item.content_truncated)
        or section.end_char - section.start_char != len(source)
        or hashlib.sha256(section.content.encode("utf-8")).hexdigest() != section.content_hash
    ):
        result.gaps.append("source_range_unverified")
    if result.gaps:
        return result

    result.basis = "local_text"
    if quotation is not None:
        result.quotation_check = check_exact_quotation(
            verbatim_content,
            quotation,
            expected_reference_sha256=hashlib.sha256(verbatim_content.encode("utf-8")).hexdigest(),
        )
        match = result.quotation_check
        if match.status == "exact_reference_match":
            result.quotation_status = "exact_reference_match"
        else:
            result.quotation_status = "not_found" if match.status == "mismatch" else "unavailable"
            result.basis = "insufficient"
            result.gaps.append("quotation_not_found" if match.status == "mismatch" else "quotation_" + match.reason)

    result.citation_available = citation is not None
    if citation is None:
        result.gaps.append("validated_citation_unavailable")
    elif result.basis != "insufficient":
        result.basis = "validated_citation"
    if as_of is None:
        result.gaps.append("as_of_required")
        return result
    if citation is None:
        result.gaps.append("legal_status_not_checked_without_citation")
        return result
    if pool is None:
        result.gaps.append("legal_status_unavailable")
        return result
    try:
        status = await resolve_regulation_status(pool, instrument_id=citation.instrument_id, as_of=as_of)
    except RegulationStatusRepositoryError:
        result.gaps.append("legal_status_unavailable")
        return result
    result.status_reason = status.reason
    if not status.resolved or status.legal_version is None:
        result.gaps.append("legal_status_unresolved")
        return result
    version = status.legal_version
    result.resolved_legal_version_id = version.legal_version_id
    result.legal_evidence = list(status.evidence)
    if (
        version.legal_version_id != citation.legal_version_id
        or version.legal_text_sha256 != citation.normalized_document_sha256
        or version.version_review_record_sha256 != citation.legal_validation_record_sha256
    ):
        result.gaps.append("cited_version_not_validated_for_date")
    elif result.basis != "insufficient":
        result.basis = "dated_version"
    return result
