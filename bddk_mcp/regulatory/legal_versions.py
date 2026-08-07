"""Immutable models and canonical checksum for legal-version bundles.

Reconstructed 2026-08-07 from the field/attribute usage in
``bddk_mcp/regulatory/repository.py`` after the original untracked file was
lost. If the original is recovered, diff against this before replacing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from datetime import date, datetime
from enum import Enum, StrEnum


class ValidationState(StrEnum):
    UNVALIDATED = "unvalidated"
    MACHINE_VALIDATED = "machine_validated"
    HUMAN_VALIDATED = "human_validated"
    REJECTED = "rejected"


class AuthorityLevel(StrEnum):
    OFFICIAL_GAZETTE = "official_gazette"
    REGULATOR_PUBLICATION = "regulator_publication"
    AGGREGATOR = "aggregator"


class ConsolidationState(StrEnum):
    AS_ENACTED = "as_enacted"
    CONSOLIDATED = "consolidated"
    UNKNOWN = "unknown"


class LegalEventType(StrEnum):
    PUBLICATION = "publication"
    EFFECTIVE = "effective"
    EXPIRY = "expiry"
    REPEAL = "repeal"
    SUPERSESSION = "supersession"
    CONSOLIDATION = "consolidation"


class LegalStatus(StrEnum):
    IN_FORCE = "in_force"
    NOT_YET_EFFECTIVE = "not_yet_effective"
    REPEALED = "repealed"
    SUPERSEDED = "superseded"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ValidationRecord:
    state: ValidationState
    validated_by: str | None
    validated_at: datetime | None
    method: str | None
    review_record_sha256: str | None


@dataclass(frozen=True, slots=True)
class Evidence:
    evidence_id: str
    artifact_id: str
    locator: str
    statement_sha256: str
    authority_level: AuthorityLevel


@dataclass(frozen=True, slots=True)
class Instrument:
    instrument_id: str
    jurisdiction: str
    authority_code: str
    identity_key: str
    canonical_title: str
    instrument_type: str


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    artifact_id: str
    content_sha256: str
    canonical_uri: str
    source_authority: str
    media_type: str
    retrieved_at: datetime
    repository_document_id: str | None
    fixture_only: bool


@dataclass(frozen=True, slots=True)
class LegalEvent:
    event_id: str
    legal_version_id: str
    event_type: LegalEventType
    event_date: date | None
    evidence: Evidence
    validation: ValidationRecord
    target_legal_version_id: str | None = None


@dataclass(frozen=True, slots=True)
class LegalEventSet:
    publication: LegalEvent | None = None
    effective: LegalEvent | None = None
    expiry: LegalEvent | None = None
    repeal: LegalEvent | None = None
    supersession: LegalEvent | None = None
    consolidation: LegalEvent | None = None


@dataclass(frozen=True, slots=True)
class LegalStatusAssertion:
    assertion_id: str
    legal_version_id: str
    status: LegalStatus
    valid_from: date | None
    valid_through: date | None
    evidence: Evidence
    validation: ValidationRecord


@dataclass(frozen=True, slots=True)
class Provision:
    """Addressable unit of an instrument (madde, ilke, ek, ...).

    Bundle producers MUST build ``canonical_path`` via
    ``bddk_mcp.regulatory.bridge.canonical_provision_path`` so bundle import
    and every section→provision SQL join share a single normalization.
    """

    provision_id: str
    instrument_id: str
    kind: str
    canonical_path: str


@dataclass(frozen=True, slots=True)
class ProvisionOccurrence:
    legal_version_id: str
    provision_id: str
    normalized_text_sha256: str
    evidence: Evidence


@dataclass(frozen=True, slots=True)
class LegalVersion:
    legal_version_id: str
    instrument_id: str
    version_key: str
    legal_text_sha256: str
    predecessor_version_id: str | None
    consolidation_state: ConsolidationState
    validation: ValidationRecord
    events: LegalEventSet
    status_assertions: tuple[LegalStatusAssertion, ...]
    provisions: tuple[ProvisionOccurrence, ...]
    source_artifact_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LegalVersionBundle:
    bundle_id: str
    bundle_sha256: str
    schema_version: str
    fixture_only: bool
    instrument: Instrument
    artifacts: tuple[SourceArtifact, ...]
    versions: tuple[LegalVersion, ...]
    provisions: tuple[Provision, ...]


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def canonical_bundle_sha256(bundle: LegalVersionBundle) -> str:
    """Deterministic content hash of a bundle, excluding the hash field itself."""
    payload = _jsonable(bundle)
    assert isinstance(payload, dict)
    payload.pop("bundle_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
