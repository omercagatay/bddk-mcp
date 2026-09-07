"""Link exact source quotations to independently validated dated-version evidence.

No model, heuristic entailment score, or caller-supplied legal approval is used.
A positive quotation check says only that the words occur in this provision.
"""

from __future__ import annotations

import hashlib
from datetime import date
from typing import TYPE_CHECKING, Any

from bddk_mcp.quality.markdown_quality import sanitize_markdown_for_context
from bddk_mcp.regulatory.status_repository import RegulationStatusRepositoryError, resolve_regulation_status
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
    if (
        not source
        or source.strip() != section.content
        or item.content != sanitize_markdown_for_context(section.content)
        or section.end_char - section.start_char != len(source)
        or hashlib.sha256(section.content.encode("utf-8")).hexdigest() != section.content_hash
    ):
        result.gaps.append("source_range_unverified")
    if result.gaps:
        return result

    result.basis = "local_text"
    if quotation is not None:
        # Preserve case, accents, numbers and punctuation. PDF line wrapping is
        # the only tolerated difference; never stem, casefold or fuzzy-match a quote.
        found = " ".join(quotation.split()) in " ".join(section.content.split())
        result.quotation_status = "verified" if found else "not_found"
        if not found:
            result.basis = "insufficient"
            result.gaps.append("quotation_not_found")

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
