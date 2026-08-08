"""Abstention-first public access to validated canonical legal-status evidence."""

from __future__ import annotations

import logging
from datetime import date

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.regulatory.status_repository import (
    RegulationStatusRepositoryError,
)
from bddk_mcp.regulatory.status_repository import (
    resolve_regulation_status as resolve_status_record,
)
from bddk_mcp.tools.contract_types import AsOfDate, InstrumentId
from bddk_mcp.tools.errors import tool_error
from bddk_mcp.tools.structured_outputs import (
    UNTRUSTED_SOURCE_WARNING,
    RegulationStatusResponse,
    RegulationStatusToolResult,
    structured_tool_result,
)
from bddk_mcp.tools.tool_logging import logged_tool

logger = logging.getLogger(__name__)

_LEGAL_USE_WARNING = (
    "This is a bounded as-of resolution from separately validated canonical claims. "
    "It is not legal advice, does not establish source authenticity by itself, and must not be extrapolated "
    "to another date."
)
_ABSTENTION_MESSAGES = {
    "fixture_only_data": "Only repository-fixture evidence is present; no legal status is returned.",
    "instrument_not_found": "The exact canonical instrument ID was not found.",
    "no_validated_version": "No independently validated legal version is available for this instrument.",
    "status_not_validated_for_date": "No validated authoritative status assertion covers the requested date.",
    "conflicting_status_evidence": "Conflicting or incomplete status signals prevent a legal conclusion.",
    "ambiguous_validated_versions": "More than one validated version qualifies; the resolver refuses to choose.",
}


def register(mcp, deps: Dependencies) -> None:
    """Register the public legal-status resolver."""

    @mcp.tool()
    @logged_tool(logger)
    async def resolve_regulation_status(
        instrument_id: InstrumentId,
        as_of: AsOfDate,
    ) -> RegulationStatusToolResult:
        """Resolve one exact legal instrument for one required date.

        The tool returns a version only when exactly one canonical legal version
        has validated authoritative publication, effective-date, and bounded
        effective-status evidence. It does not infer currentness from document
        recency, a missing repeal, version labels, or the current date.

        Args:
            instrument_id: Exact canonical `inst_sha256_<64 lowercase hex>` identifier.
            as_of: Required inclusive calendar date in ISO `YYYY-MM-DD` format.
        """

        if deps.pool is None:
            tool_error(
                "LEGAL_EVIDENCE_UNAVAILABLE",
                "Validated legal-status evidence is not available in this runtime.",
                retryable=True,
            )
        try:
            record = await resolve_status_record(
                deps.pool,
                instrument_id=instrument_id,
                as_of=date.fromisoformat(as_of),
            )
        except RegulationStatusRepositoryError:
            tool_error(
                "LEGAL_EVIDENCE_UNAVAILABLE",
                "Validated legal-status evidence could not be verified.",
                retryable=True,
            )

        if not record.resolved:
            message = _ABSTENTION_MESSAGES.get(
                record.reason.value,
                "Validated evidence is insufficient for a legal-status conclusion.",
            )
            text = (
                "LEGAL STATUS ABSTAINED\n"
                f"Instrument ID: {record.instrument_id}\n"
                f"As of: {record.as_of.isoformat()}\n"
                f"Reason: {record.reason.value}\n"
                f"{message}\n"
                f"{_LEGAL_USE_WARNING}"
            )
            return structured_tool_result(
                RegulationStatusResponse(
                    status="unavailable",
                    text=text,
                    warnings=[_LEGAL_USE_WARNING],
                    instrument_id=record.instrument_id,
                    as_of=record.as_of,
                    resolved=False,
                    reason=record.reason,
                )
            )

        if record.legal_version is None:
            tool_error(
                "LEGAL_EVIDENCE_UNAVAILABLE",
                "Validated legal-status evidence could not be verified.",
                retryable=True,
            )
        legal_version = record.legal_version
        evidence_lines = [
            f"- {item.role}: claim={item.claim_id}; evidence={item.evidence_id}; artifact={item.artifact_id}"
            for item in record.evidence
        ]
        text = "\n".join(
            [
                "LEGAL STATUS RESOLVED FROM VALIDATED EVIDENCE",
                f"Instrument ID: {record.instrument_id}",
                f"As of: {record.as_of.isoformat()}",
                f"Legal version ID: {legal_version.legal_version_id}",
                "Validated status: effective",
                f"Consolidation state: {legal_version.consolidation_state}",
                f"Amends version ID: {legal_version.amends_version_id or 'not_validated'}",
                "Evidence identities:",
                *evidence_lines,
                _LEGAL_USE_WARNING,
            ]
        )
        return structured_tool_result(
            RegulationStatusResponse(
                status="ok",
                text=text,
                warnings=[UNTRUSTED_SOURCE_WARNING, _LEGAL_USE_WARNING],
                instrument_id=record.instrument_id,
                as_of=record.as_of,
                resolved=True,
                reason=record.reason,
                legal_version=legal_version,
                legal_evidence=list(record.evidence),
            )
        )
