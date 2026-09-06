"""Read-only access to the hardened canonical legal-status resolver."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from typing import Any, Protocol

import asyncpg
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from bddk_mcp.regulatory.legal_versions import (
    ResolutionReason,
    artifact_id_for,
    blob_id_for,
    evidence_id_for,
    event_id_for,
    legal_version_id_for,
    status_assertion_id_for,
)
from bddk_mcp.tools.structured_outputs import LegalClaimEvidence, ResolvedLegalVersion

_RESOLVE_STATUS_SQL = """
SELECT resolved,
       reason,
       instrument_id,
       as_of,
       legal_version_id,
       version_key,
       legal_text_sha256,
       version_review_record_sha256,
       amends_version_id,
       consolidation_state,
       evidence_json
FROM bddk_meta.resolve_regulation_status($1::pg_catalog.text, $2::pg_catalog.date)
"""


class _Pool(Protocol):
    async def fetch(self, query: str, *args: object) -> Any: ...


class RegulationStatusRepositoryError(RuntimeError):
    """A content-free failure at the legal-status persistence boundary."""


class RegulationStatusRecord(BaseModel):
    """Exact database result before MCP rendering."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    resolved: bool
    reason: ResolutionReason
    instrument_id: str = Field(pattern=r"^inst_sha256_[0-9a-f]{64}$")
    as_of: date
    legal_version: ResolvedLegalVersion | None = None
    evidence: tuple[LegalClaimEvidence, ...] = ()

    @model_validator(mode="after")
    def _resolved_and_abstained_shapes_are_distinct(self) -> RegulationStatusRecord:
        if self.resolved:
            if self.reason is not ResolutionReason.RESOLVED or self.legal_version is None or len(self.evidence) < 3:
                raise ValueError("resolved status lacks complete evidence")
            roles = {item.role for item in self.evidence}
            if len(roles) != len(self.evidence) or not {"publication", "effective", "status"}.issubset(roles):
                raise ValueError("resolved status lacks a required evidence role")
        elif self.reason is ResolutionReason.RESOLVED or self.legal_version is not None or self.evidence:
            raise ValueError("abstention contains legal claims")
        return self


def _row_mapping(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    try:
        return dict(row)
    except (TypeError, ValueError):
        raise RegulationStatusRepositoryError("Legal-status resolver returned an invalid record.") from None


def _parse_evidence(value: Any) -> tuple[LegalClaimEvidence, ...]:
    if not isinstance(value, str) or len(value) > 100_000:
        raise RegulationStatusRepositoryError("Legal-status resolver returned invalid evidence metadata.")
    try:
        payload = json.loads(value)
        if not isinstance(payload, list) or len(payload) > 5:
            raise ValueError
        return tuple(LegalClaimEvidence.model_validate(item) for item in payload)
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError):
        raise RegulationStatusRepositoryError("Legal-status resolver returned invalid evidence metadata.") from None


def _require_empty_evidence(value: Any) -> None:
    if not isinstance(value, str) or len(value) > 100_000:
        raise RegulationStatusRepositoryError("Legal-status abstention returned invalid evidence metadata.")
    try:
        if json.loads(value) != []:
            raise ValueError
    except (json.JSONDecodeError, TypeError, ValueError):
        raise RegulationStatusRepositoryError("Legal-status abstention returned invalid evidence metadata.") from None


async def resolve_regulation_status(
    pool: _Pool,
    *,
    instrument_id: str,
    as_of: date,
) -> RegulationStatusRecord:
    """Resolve one exact date through the database-owned, least-privilege boundary."""

    try:
        rows = await pool.fetch(_RESOLVE_STATUS_SQL, instrument_id, as_of)
    except (asyncpg.PostgresError, OSError, TimeoutError):
        raise RegulationStatusRepositoryError("Legal-status evidence is temporarily unavailable.") from None
    if not isinstance(rows, list | tuple) or len(rows) != 1:
        raise RegulationStatusRepositoryError("Legal-status resolver did not return exactly one record.")

    raw = _row_mapping(rows[0])
    evidence_json = raw.pop("evidence_json", None)
    resolved = raw.get("resolved") is True
    evidence = _parse_evidence(evidence_json) if resolved else ()
    legal_version = None
    if resolved:
        try:
            legal_version = ResolvedLegalVersion(
                legal_version_id=raw.pop("legal_version_id", None),
                version_key=raw.pop("version_key", None),
                legal_text_sha256=raw.pop("legal_text_sha256", None),
                version_review_record_sha256=raw.pop("version_review_record_sha256", None),
                amends_version_id=raw.pop("amends_version_id", None),
                consolidation_state=raw.pop("consolidation_state", None),
            )
        except ValidationError:
            raise RegulationStatusRepositoryError("Legal-status resolver returned an invalid record.") from None
    else:
        _require_empty_evidence(evidence_json)
        for field in (
            "legal_version_id",
            "version_key",
            "legal_text_sha256",
            "version_review_record_sha256",
            "amends_version_id",
            "consolidation_state",
        ):
            if raw.pop(field, None) is not None:
                raise RegulationStatusRepositoryError("Legal-status abstention returned claim metadata.")

    try:
        record = RegulationStatusRecord.model_validate(
            {
                **raw,
                "legal_version": legal_version,
                "evidence": evidence,
            }
        )
    except (ValidationError, TypeError, ValueError):
        raise RegulationStatusRepositoryError("Legal-status resolver returned an invalid record.") from None
    if record.instrument_id != instrument_id or record.as_of != as_of:
        raise RegulationStatusRepositoryError("Legal status does not match the requested instrument and date.")
    if any(
        (item.role == "status" and not item.valid_from <= as_of <= item.valid_through)
        or (item.claim_date is not None and item.claim_date > as_of)
        for item in record.evidence
    ):
        raise RegulationStatusRepositoryError("Legal evidence does not cover the requested date.")
    version = record.legal_version
    if version is not None and version.legal_version_id != legal_version_id_for(
        instrument_id=instrument_id,
        version_key=version.version_key,
        legal_text_sha256=version.legal_text_sha256,
    ):
        raise RegulationStatusRepositoryError("Legal-version identity is inconsistent.")
    for item in record.evidence:
        if (
            item.artifact_blob_id != blob_id_for(content_sha256=item.artifact_sha256)
            or item.artifact_id
            != artifact_id_for(
                blob_id=item.artifact_blob_id,
                canonical_uri=item.source_url,
                retrieved_at=item.artifact_retrieved_at,
            )
            or item.evidence_id
            != evidence_id_for(
                artifact_id=item.artifact_id,
                locator=item.evidence_locator,
                statement_sha256=item.evidence_statement_sha256,
                authority_level="authoritative",
            )
            or not item.claim_id.startswith("status_sha256_" if item.role == "status" else "event_sha256_")
        ):
            raise RegulationStatusRepositoryError("Legal-evidence identity is inconsistent.")
        # Required applicability claims have all identity components in this
        # projection. Optional relationship targets remain database-validated.
        if item.role == "status":
            expected_claim = status_assertion_id_for(
                legal_version_id=version.legal_version_id,
                status="effective",
                valid_from=item.valid_from,
                valid_through=item.valid_through,
                evidence_id=item.evidence_id,
            )
        elif item.role in {"publication", "effective"}:
            expected_claim = event_id_for(
                legal_version_id=version.legal_version_id,
                event_type=item.role,
                event_date=item.claim_date,
                evidence_id=item.evidence_id,
            )
        else:
            continue
        if item.claim_id != expected_claim:
            raise RegulationStatusRepositoryError("Legal-claim identity is inconsistent.")
    return record
