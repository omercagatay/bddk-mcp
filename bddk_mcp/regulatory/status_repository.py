"""Read-only access to the hardened canonical legal-status resolver."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from typing import Any, Protocol

import asyncpg
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from bddk_mcp.regulatory.legal_versions import ResolutionReason
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
            if not {"publication", "effective", "status"}.issubset(roles):
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
        return RegulationStatusRecord.model_validate(
            {
                **raw,
                "legal_version": legal_version,
                "evidence": evidence,
            }
        )
    except (ValidationError, TypeError, ValueError):
        raise RegulationStatusRepositoryError("Legal-status resolver returned an invalid record.") from None
