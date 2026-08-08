"""Abstention-first canonical legal-version model.

This module deliberately separates source/extraction artifacts from legal
versions.  A parsed document, a changed extraction, or a newer retrieval
timestamp is not evidence that a legal provision became effective.  The
resolver therefore returns a version only when repository data contains
explicit, authoritative, reviewer-validated publication, effective-date, and
status evidence covering the requested date.

The model is jurisdiction-neutral.  It does not encode or infer any BDDK legal
facts and it does not scrape external sources.  The tracked pilot fixture is
synthetic and exercises the contract without asserting real-world currentness.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_INSTRUMENT_ID_PATTERN = r"^inst_sha256_[0-9a-f]{64}$"
_BLOB_ID_PATTERN = r"^blob_sha256_[0-9a-f]{64}$"
_ARTIFACT_ID_PATTERN = r"^art_sha256_[0-9a-f]{64}$"
_VERSION_ID_PATTERN = r"^ver_sha256_[0-9a-f]{64}$"
_PROVISION_ID_PATTERN = r"^prov_sha256_[0-9a-f]{64}$"
_EVIDENCE_ID_PATTERN = r"^evid_sha256_[0-9a-f]{64}$"
_EVENT_ID_PATTERN = r"^event_sha256_[0-9a-f]{64}$"
_ASSERTION_ID_PATTERN = r"^status_sha256_[0-9a-f]{64}$"
_BUNDLE_ID_PATTERN = r"^family_sha256_[0-9a-f]{64}$"
_MAX_BUNDLE_BYTES = 2 * 1024 * 1024
_MAX_SOURCE_QUERY_LENGTH = 1_000
_SENSITIVE_QUERY_KEY_RE = re.compile(
    r"(?:token|secret|password|passwd|signature|credential|api[-_]?key)",
    re.IGNORECASE,
)


class LegalVersionBundleError(ValueError):
    """Raised with a bounded, content-free legal-version import error."""


class ValidationState(StrEnum):
    """Human review state; only ``validated`` can support a resolution."""

    UNVALIDATED = "unvalidated"
    IN_REVIEW = "in_review"
    VALIDATED = "validated"
    REJECTED = "rejected"


class AuthorityLevel(StrEnum):
    """Provenance classification for a source assertion."""

    AUTHORITATIVE = "authoritative"
    SECONDARY = "secondary"
    REPOSITORY_FIXTURE = "repository_fixture"


class LegalEventType(StrEnum):
    """Legal lifecycle events represented by explicit evidence claims."""

    PUBLICATION = "publication"
    EFFECTIVE = "effective"
    EXPIRY = "expiry"
    REPEAL = "repeal"
    SUPERSESSION = "supersession"
    CONSOLIDATION = "consolidation"


class ConsolidationState(StrEnum):
    """Whether the version is an original, amendment, or consolidated text."""

    UNKNOWN = "unknown"
    ORIGINAL = "original"
    AMENDMENT = "amendment"
    CONSOLIDATED = "consolidated"


class LegalStatus(StrEnum):
    """Status asserted for a bounded date range."""

    EFFECTIVE = "effective"
    NOT_YET_EFFECTIVE = "not_yet_effective"
    EXPIRED = "expired"
    REPEALED = "repealed"
    SUPERSEDED = "superseded"
    UNKNOWN = "unknown"


class ResolutionReason(StrEnum):
    """Stable, machine-readable resolver outcomes."""

    RESOLVED = "resolved"
    FIXTURE_ONLY_DATA = "fixture_only_data"
    INSTRUMENT_NOT_FOUND = "instrument_not_found"
    NO_VALIDATED_VERSION = "no_validated_version"
    STATUS_NOT_VALIDATED_FOR_DATE = "status_not_validated_for_date"
    CONFLICTING_STATUS_EVIDENCE = "conflicting_status_evidence"
    AMBIGUOUS_VALIDATED_VERSIONS = "ambiguous_validated_versions"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _stable_id(prefix: str, *parts: str) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return f"{prefix}_sha256_{hashlib.sha256(payload).hexdigest()}"


def instrument_id_for(*, jurisdiction: str, authority_code: str, identity_key: str) -> str:
    """Build an identity independent of mutable titles and extracted text."""

    return _stable_id("inst", jurisdiction, authority_code, identity_key)


def blob_id_for(*, content_sha256: str) -> str:
    """Build the identity of one immutable set of acquired bytes."""

    return _stable_id("blob", content_sha256)


def _canonical_acquisition_time(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("retrieved_at must include a UTC offset")
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_canonical_source_uri(value: str) -> str:
    """Validate the one source-URI policy shared by ingestion and Citation v1."""

    if any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError("source URI must not contain whitespace or control characters")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or "#" in value
        or len(parsed.query) > _MAX_SOURCE_QUERY_LENGTH
    ):
        raise ValueError("source URI must be canonical HTTPS without credentials or fragments")
    if any(_SENSITIVE_QUERY_KEY_RE.search(key) for key, _ in parse_qsl(parsed.query, keep_blank_values=True)):
        raise ValueError("source URI contains a sensitive query parameter")
    return value


def artifact_id_for(*, blob_id: str, canonical_uri: str, retrieved_at: datetime) -> str:
    """Build an acquisition identity distinct from the acquired-byte identity.

    Reacquiring identical bytes at another canonical URI or at another time
    deliberately creates another acquisition record while reusing the blob.
    """

    return _stable_id("art", blob_id, canonical_uri, _canonical_acquisition_time(retrieved_at))


def legal_version_id_for(*, instrument_id: str, version_key: str, legal_text_sha256: str) -> str:
    """Build an immutable legal-version identity, distinct from extraction runs."""

    return _stable_id("ver", instrument_id, version_key, legal_text_sha256)


def provision_id_for(*, instrument_id: str, kind: str, canonical_path: str) -> str:
    """Build a logical provision identity that remains stable across versions."""

    return _stable_id("prov", instrument_id, kind, canonical_path)


def evidence_id_for(
    *,
    artifact_id: str,
    locator: str,
    statement_sha256: str,
    authority_level: AuthorityLevel | str,
) -> str:
    """Build an immutable identity for a bounded source assertion."""

    return _stable_id("evid", artifact_id, locator, statement_sha256, str(authority_level))


def event_id_for(
    *,
    legal_version_id: str,
    event_type: LegalEventType | str,
    event_date: date,
    evidence_id: str,
    target_legal_version_id: str | None = None,
) -> str:
    """Build an immutable temporal-event claim identity."""

    return _stable_id(
        "event",
        legal_version_id,
        str(event_type),
        event_date.isoformat(),
        evidence_id,
        target_legal_version_id or "",
    )


def status_assertion_id_for(
    *,
    legal_version_id: str,
    status: LegalStatus | str,
    valid_from: date,
    valid_through: date,
    evidence_id: str,
) -> str:
    """Build an immutable identity for a date-bounded applicability assertion."""

    return _stable_id(
        "status",
        legal_version_id,
        str(status),
        valid_from.isoformat(),
        valid_through.isoformat(),
        evidence_id,
    )


def bundle_id_for(*, instrument_id: str) -> str:
    """Build the stable identity for one instrument-family bundle."""

    return _stable_id("family", instrument_id)


class ValidationRecord(_FrozenModel):
    """Review provenance; validation never follows from ingestion alone."""

    state: ValidationState
    validated_by: str | None = Field(default=None, min_length=1, max_length=200)
    validated_at: datetime | None = None
    method: str | None = Field(default=None, min_length=1, max_length=500)
    review_record_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)

    @field_validator("validated_at")
    @classmethod
    def _require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("validated_at must include a UTC offset")
        return value.astimezone(UTC) if value is not None else None

    @model_validator(mode="after")
    def _validated_state_has_complete_review_provenance(self) -> ValidationRecord:
        fields = (self.validated_by, self.validated_at, self.method, self.review_record_sha256)
        completed = self.state in {ValidationState.VALIDATED, ValidationState.REJECTED}
        if completed and any(item is None for item in fields):
            raise ValueError("completed reviews require reviewer, time, method, and review-record hash")
        if not completed and any(item is not None for item in fields):
            raise ValueError("only completed reviews may carry review provenance")
        return self

    @property
    def is_validated(self) -> bool:
        return self.state is ValidationState.VALIDATED


class CanonicalInstrument(_FrozenModel):
    """Stable identity for a legal instrument independent of its versions."""

    instrument_id: str = Field(pattern=_INSTRUMENT_ID_PATTERN)
    jurisdiction: str = Field(min_length=2, max_length=50)
    authority_code: str = Field(min_length=1, max_length=100)
    identity_key: str = Field(min_length=1, max_length=300)
    canonical_title: str = Field(min_length=1, max_length=1000)
    instrument_type: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def _identity_matches_components(self) -> CanonicalInstrument:
        expected = instrument_id_for(
            jurisdiction=self.jurisdiction,
            authority_code=self.authority_code,
            identity_key=self.identity_key,
        )
        if self.instrument_id != expected:
            raise ValueError("instrument_id does not match its immutable identity components")
        return self


class SourceBlob(_FrozenModel):
    """Content-addressed identity for immutable acquired bytes."""

    blob_id: str = Field(pattern=_BLOB_ID_PATTERN)
    content_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _content_identity_matches(self) -> SourceBlob:
        if self.blob_id != blob_id_for(content_sha256=self.content_sha256):
            raise ValueError("blob_id does not match content_sha256")
        return self


class SourceArtifact(_FrozenModel):
    """Immutable source acquisition; extraction revisions are separate."""

    artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    blob_id: str = Field(pattern=_BLOB_ID_PATTERN)
    canonical_uri: str = Field(min_length=1, max_length=2000)
    source_authority: str = Field(min_length=1, max_length=200)
    media_type: str = Field(min_length=1, max_length=200)
    retrieved_at: datetime
    repository_document_id: str | None = Field(default=None, min_length=1, max_length=500)
    fixture_only: bool = False

    @field_validator("canonical_uri")
    @classmethod
    def _require_https_source(cls, value: str) -> str:
        return validate_canonical_source_uri(value)

    @field_validator("retrieved_at")
    @classmethod
    def _retrieval_time_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("retrieved_at must include a UTC offset")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _acquisition_identity_matches(self) -> SourceArtifact:
        expected = artifact_id_for(
            blob_id=self.blob_id,
            canonical_uri=self.canonical_uri,
            retrieved_at=self.retrieved_at,
        )
        if self.artifact_id != expected:
            raise ValueError("artifact_id does not match its immutable acquisition components")
        return self


class EvidenceReference(_FrozenModel):
    """A hash-and-locator reference; source text is intentionally not copied."""

    evidence_id: str = Field(pattern=_EVIDENCE_ID_PATTERN)
    artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    locator: str = Field(min_length=1, max_length=1000)
    statement_sha256: str = Field(pattern=_SHA256_PATTERN)
    authority_level: AuthorityLevel

    @model_validator(mode="after")
    def _identity_matches_components(self) -> EvidenceReference:
        expected = evidence_id_for(
            artifact_id=self.artifact_id,
            locator=self.locator,
            statement_sha256=self.statement_sha256,
            authority_level=self.authority_level,
        )
        if self.evidence_id != expected:
            raise ValueError("evidence_id does not match its immutable components")
        return self

    @property
    def is_authoritative(self) -> bool:
        return self.authority_level is AuthorityLevel.AUTHORITATIVE


class LegalEvent(_FrozenModel):
    """One dated lifecycle claim with separate evidence and review state."""

    event_id: str = Field(pattern=_EVENT_ID_PATTERN)
    legal_version_id: str = Field(pattern=_VERSION_ID_PATTERN)
    event_type: LegalEventType
    event_date: date
    evidence: EvidenceReference
    validation: ValidationRecord
    target_legal_version_id: str | None = Field(default=None, pattern=_VERSION_ID_PATTERN)

    @model_validator(mode="after")
    def _event_contract_is_consistent(self) -> LegalEvent:
        if self.event_type is LegalEventType.SUPERSESSION:
            if self.target_legal_version_id is None or self.target_legal_version_id == self.legal_version_id:
                raise ValueError("supersession requires a distinct target legal version")
        elif self.target_legal_version_id is not None:
            raise ValueError("only supersession events may target another legal version")
        expected = event_id_for(
            legal_version_id=self.legal_version_id,
            event_type=self.event_type,
            event_date=self.event_date,
            evidence_id=self.evidence.evidence_id,
            target_legal_version_id=self.target_legal_version_id,
        )
        if self.event_id != expected:
            raise ValueError("event_id does not match its immutable components")
        return self

    @property
    def supports_resolution(self) -> bool:
        return self.validation.is_validated and self.evidence.is_authoritative


class VersionEvents(_FrozenModel):
    """Named legal lifecycle claims; absence means unknown, never false."""

    publication: LegalEvent | None = None
    effective: LegalEvent | None = None
    expiry: LegalEvent | None = None
    repeal: LegalEvent | None = None
    supersession: LegalEvent | None = None
    consolidation: LegalEvent | None = None

    @model_validator(mode="after")
    def _field_names_match_event_types(self) -> VersionEvents:
        for field_name, expected_type in (
            ("publication", LegalEventType.PUBLICATION),
            ("effective", LegalEventType.EFFECTIVE),
            ("expiry", LegalEventType.EXPIRY),
            ("repeal", LegalEventType.REPEAL),
            ("supersession", LegalEventType.SUPERSESSION),
            ("consolidation", LegalEventType.CONSOLIDATION),
        ):
            event = getattr(self, field_name)
            if event is not None and event.event_type is not expected_type:
                raise ValueError(f"{field_name} contains the wrong event type")
        return self

    def terminal_events(self) -> tuple[LegalEvent, ...]:
        return tuple(event for event in (self.expiry, self.repeal, self.supersession) if event is not None)


class LegalStatusAssertion(_FrozenModel):
    """Explicit status evidence covering a closed date interval."""

    assertion_id: str = Field(pattern=_ASSERTION_ID_PATTERN)
    legal_version_id: str = Field(pattern=_VERSION_ID_PATTERN)
    status: LegalStatus
    valid_from: date
    valid_through: date
    evidence: EvidenceReference
    validation: ValidationRecord

    @model_validator(mode="after")
    def _assertion_contract_is_consistent(self) -> LegalStatusAssertion:
        if self.valid_through < self.valid_from:
            raise ValueError("status range ends before it starts")
        expected = status_assertion_id_for(
            legal_version_id=self.legal_version_id,
            status=self.status,
            valid_from=self.valid_from,
            valid_through=self.valid_through,
            evidence_id=self.evidence.evidence_id,
        )
        if self.assertion_id != expected:
            raise ValueError("assertion_id does not match its immutable components")
        return self

    @property
    def supports_resolution(self) -> bool:
        return self.validation.is_validated and self.evidence.is_authoritative

    def covers(self, as_of: date) -> bool:
        return self.valid_from <= as_of <= self.valid_through


class ProvisionIdentity(_FrozenModel):
    """Logical provision identity stable across legal versions."""

    provision_id: str = Field(pattern=_PROVISION_ID_PATTERN)
    instrument_id: str = Field(pattern=_INSTRUMENT_ID_PATTERN)
    kind: str = Field(min_length=1, max_length=100)
    canonical_path: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def _identity_matches_components(self) -> ProvisionIdentity:
        expected = provision_id_for(
            instrument_id=self.instrument_id,
            kind=self.kind,
            canonical_path=self.canonical_path,
        )
        if self.provision_id != expected:
            raise ValueError("provision_id does not match its immutable components")
        return self


class ProvisionOccurrence(_FrozenModel):
    """Version-specific provision content and its exact source location."""

    provision_id: str = Field(pattern=_PROVISION_ID_PATTERN)
    legal_version_id: str = Field(pattern=_VERSION_ID_PATTERN)
    provision_text_sha256: str = Field(pattern=_SHA256_PATTERN)
    document_section_id: int | None = Field(default=None, gt=0)
    evidence: EvidenceReference
    validation: ValidationRecord

    @model_validator(mode="after")
    def _evidence_identifies_the_exact_normalized_statement(self) -> ProvisionOccurrence:
        if self.evidence.statement_sha256 != self.provision_text_sha256:
            raise ValueError("provision evidence hash does not match normalized provision text")
        return self


class LegalVersion(_FrozenModel):
    """Canonical legal version, not an extraction revision."""

    legal_version_id: str = Field(pattern=_VERSION_ID_PATTERN)
    instrument_id: str = Field(pattern=_INSTRUMENT_ID_PATTERN)
    version_key: str = Field(min_length=1, max_length=300)
    legal_text_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_artifact_ids: tuple[str, ...] = Field(min_length=1)
    predecessor_version_id: str | None = Field(default=None, pattern=_VERSION_ID_PATTERN)
    consolidation_state: ConsolidationState = ConsolidationState.UNKNOWN
    validation: ValidationRecord
    events: VersionEvents
    status_assertions: tuple[LegalStatusAssertion, ...] = ()
    provisions: tuple[ProvisionOccurrence, ...] = ()

    @field_validator("source_artifact_ids")
    @classmethod
    def _source_artifacts_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not isinstance(item, str) for item in value) or len(set(value)) != len(value):
            raise ValueError("source artifact identities must be unique strings")
        if any(not re.fullmatch(_ARTIFACT_ID_PATTERN, item) for item in value):
            raise ValueError("source artifact identity is invalid")
        return value

    @model_validator(mode="after")
    def _version_contract_is_consistent(self) -> LegalVersion:
        expected = legal_version_id_for(
            instrument_id=self.instrument_id,
            version_key=self.version_key,
            legal_text_sha256=self.legal_text_sha256,
        )
        if self.legal_version_id != expected:
            raise ValueError("legal_version_id does not match its immutable components")
        if self.predecessor_version_id == self.legal_version_id:
            raise ValueError("a legal version cannot be its own predecessor")
        if self.source_artifact_ids != tuple(sorted(self.source_artifact_ids)):
            raise ValueError("source artifacts must use canonical identity order")
        if len({item.assertion_id for item in self.status_assertions}) != len(self.status_assertions):
            raise ValueError("status assertion identities are duplicated")
        if len({item.provision_id for item in self.provisions}) != len(self.provisions):
            raise ValueError("provision occurrence identities are duplicated")
        if self.status_assertions != tuple(sorted(self.status_assertions, key=lambda item: item.assertion_id)):
            raise ValueError("status assertions must use canonical identity order")
        if self.provisions != tuple(sorted(self.provisions, key=lambda item: item.provision_id)):
            raise ValueError("provision occurrences must use canonical identity order")
        for event in (
            self.events.publication,
            self.events.effective,
            self.events.expiry,
            self.events.repeal,
            self.events.supersession,
            self.events.consolidation,
        ):
            if event is not None and event.legal_version_id != self.legal_version_id:
                raise ValueError("event belongs to another legal version")
        if any(item.legal_version_id != self.legal_version_id for item in self.status_assertions):
            raise ValueError("status assertion belongs to another legal version")
        if any(item.legal_version_id != self.legal_version_id for item in self.provisions):
            raise ValueError("provision occurrence belongs to another legal version")
        if self.consolidation_state is not ConsolidationState.UNKNOWN and self.events.consolidation is None:
            raise ValueError("known consolidation state requires explicit consolidation evidence")
        return self


class LegalVersionBundle(_FrozenModel):
    """One deterministically imported instrument family and its amendment chain."""

    schema_version: Literal[1]
    bundle_id: str = Field(pattern=_BUNDLE_ID_PATTERN)
    bundle_sha256: str = Field(pattern=_SHA256_PATTERN)
    fixture_only: bool = False
    instrument: CanonicalInstrument
    blobs: tuple[SourceBlob, ...] = Field(min_length=1)
    artifacts: tuple[SourceArtifact, ...] = Field(min_length=1)
    provisions: tuple[ProvisionIdentity, ...] = ()
    versions: tuple[LegalVersion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _references_and_chain_are_closed(self) -> LegalVersionBundle:
        if self.bundle_id != bundle_id_for(instrument_id=self.instrument.instrument_id):
            raise ValueError("bundle_id does not match the instrument")
        if self.blobs != tuple(sorted(self.blobs, key=lambda item: item.blob_id)):
            raise ValueError("blobs must use canonical identity order")
        if self.artifacts != tuple(sorted(self.artifacts, key=lambda item: item.artifact_id)):
            raise ValueError("artifacts must use canonical identity order")
        if self.provisions != tuple(sorted(self.provisions, key=lambda item: item.provision_id)):
            raise ValueError("provisions must use canonical identity order")
        if self.versions != tuple(sorted(self.versions, key=lambda item: item.legal_version_id)):
            raise ValueError("legal versions must use canonical identity order")

        blobs = {item.blob_id: item for item in self.blobs}
        artifacts = {item.artifact_id: item for item in self.artifacts}
        provisions = {item.provision_id: item for item in self.provisions}
        versions = {item.legal_version_id: item for item in self.versions}
        if len(blobs) != len(self.blobs):
            raise ValueError("blob identities are duplicated")
        if len(artifacts) != len(self.artifacts):
            raise ValueError("artifact identities are duplicated")
        if len(provisions) != len(self.provisions):
            raise ValueError("provision identities are duplicated")
        if len(versions) != len(self.versions):
            raise ValueError("legal-version identities are duplicated")
        roots = [version for version in self.versions if version.predecessor_version_id is None]
        if len(roots) != 1:
            raise ValueError("a pilot family must contain exactly one amendment-chain root")
        if not self.fixture_only and any(item.fixture_only for item in self.artifacts):
            raise ValueError("a production bundle cannot contain fixture-only artifacts")
        if any(artifact.blob_id not in blobs for artifact in self.artifacts):
            raise ValueError("source artifact references an unknown blob")

        for provision in self.provisions:
            if provision.instrument_id != self.instrument.instrument_id:
                raise ValueError("provision belongs to another instrument")

        for version in self.versions:
            if version.instrument_id != self.instrument.instrument_id:
                raise ValueError("legal version belongs to another instrument")
            if any(artifact_id not in artifacts for artifact_id in version.source_artifact_ids):
                raise ValueError("legal version references an unknown artifact")
            acquisition_times = tuple(artifacts[item].retrieved_at for item in version.source_artifact_ids)
            if version.validation.validated_at is not None and version.validation.validated_at < max(acquisition_times):
                raise ValueError("legal-version review predates one of its source acquisitions")
            if version.predecessor_version_id is not None and version.predecessor_version_id not in versions:
                raise ValueError("legal version references an unknown predecessor")
            if version.predecessor_version_id is not None:
                predecessor = versions[version.predecessor_version_id]
                supersession = predecessor.events.supersession
                if supersession is None or supersession.target_legal_version_id != version.legal_version_id:
                    raise ValueError("amendment chain lacks matching supersession evidence")
            supersession = version.events.supersession
            if supersession is not None:
                target = versions.get(supersession.target_legal_version_id)
                if target is None or target.predecessor_version_id != version.legal_version_id:
                    raise ValueError("supersession evidence targets a version outside the amendment chain")
            for evidence in _version_evidence(version):
                if evidence.artifact_id not in artifacts:
                    raise ValueError("evidence references an unknown artifact")
                if evidence.artifact_id not in version.source_artifact_ids:
                    raise ValueError("version evidence must use one of its source artifacts")
            reviewed_claims = (
                *(
                    event
                    for event in (
                        version.events.publication,
                        version.events.effective,
                        version.events.expiry,
                        version.events.repeal,
                        version.events.supersession,
                        version.events.consolidation,
                    )
                    if event is not None
                ),
                *version.status_assertions,
                *version.provisions,
            )
            for claim in reviewed_claims:
                reviewed_at = claim.validation.validated_at
                acquired_at = artifacts[claim.evidence.artifact_id].retrieved_at
                if reviewed_at is not None and reviewed_at < acquired_at:
                    raise ValueError("claim review predates its source acquisition")
            for occurrence in version.provisions:
                if occurrence.provision_id not in provisions:
                    raise ValueError("version references an unknown provision")

        for version in self.versions:
            seen: set[str] = set()
            cursor: LegalVersion | None = version
            while cursor is not None:
                if cursor.legal_version_id in seen:
                    raise ValueError("amendment chain contains a predecessor cycle")
                seen.add(cursor.legal_version_id)
                cursor = versions.get(cursor.predecessor_version_id) if cursor.predecessor_version_id else None
        return self


class ResolutionResult(_FrozenModel):
    """Fail-closed result for current/as-of legal-version selection."""

    instrument_id: str
    as_of: date
    reason: ResolutionReason
    legal_version_id: str | None = None
    evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _resolved_and_abstained_shapes_are_distinct(self) -> ResolutionResult:
        if self.reason is ResolutionReason.RESOLVED:
            if self.legal_version_id is None or not self.evidence_ids:
                raise ValueError("resolved results require a version and evidence")
        elif self.legal_version_id is not None or self.evidence_ids:
            raise ValueError("abstention results cannot claim a version or evidence")
        return self

    @property
    def resolved(self) -> bool:
        return self.reason is ResolutionReason.RESOLVED


def _version_evidence(version: LegalVersion) -> tuple[EvidenceReference, ...]:
    events = (
        version.events.publication,
        version.events.effective,
        version.events.expiry,
        version.events.repeal,
        version.events.supersession,
        version.events.consolidation,
    )
    return (
        tuple(event.evidence for event in events if event is not None)
        + tuple(assertion.evidence for assertion in version.status_assertions)
        + tuple(occurrence.evidence for occurrence in version.provisions)
    )


def _canonical_mapping(value: LegalVersionBundle | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, LegalVersionBundle):
        data = value.model_dump(mode="json")
    else:
        data = dict(value)
    data.pop("bundle_sha256", None)
    return data


def canonical_bundle_sha256(value: LegalVersionBundle | dict[str, Any]) -> str:
    """Hash canonical JSON while excluding the manifest's own checksum field."""

    payload = json.dumps(
        _canonical_mapping(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LegalVersionBundleError("Legal-version bundle contains a duplicate JSON key; import refused.")
        result[key] = value
    return result


def load_legal_version_bundle(path: str | Path) -> LegalVersionBundle:
    """Load one bounded, checksum-verified family without exposing source text.

    The function performs no database or network I/O and is deterministic for
    identical bytes.  Validation errors are intentionally summarized so a
    production log cannot copy legal text or source excerpts.
    """

    source = Path(path)
    try:
        if source.is_symlink() or not source.is_file():
            raise LegalVersionBundleError("Legal-version bundle is missing or is not a regular file.")
        size = source.stat().st_size
        if size <= 0 or size > _MAX_BUNDLE_BYTES:
            raise LegalVersionBundleError("Legal-version bundle exceeds the allowed size or is empty.")
        raw = source.read_text(encoding="utf-8")
        mapping = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
        if not isinstance(mapping, dict):
            raise LegalVersionBundleError("Legal-version bundle root must be a JSON object.")
        bundle = LegalVersionBundle.model_validate(mapping)
        if bundle.bundle_sha256 != canonical_bundle_sha256(bundle):
            raise LegalVersionBundleError("Legal-version bundle checksum does not match; import refused.")
        return bundle
    except LegalVersionBundleError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValidationError, TypeError, ValueError):
        raise LegalVersionBundleError("Legal-version bundle failed schema validation; import refused.") from None


def _abstain(instrument_id: str, as_of: date, reason: ResolutionReason) -> ResolutionResult:
    return ResolutionResult(instrument_id=instrument_id, as_of=as_of, reason=reason)


def resolve_as_of(bundle: LegalVersionBundle, *, instrument_id: str, as_of: date) -> ResolutionResult:
    """Resolve a version only from explicit validated authoritative evidence.

    A status assertion is date-bounded.  The resolver does not extrapolate one
    day's or one interval's status beyond ``valid_through``, does not infer
    currentness from a missing repeal event, and does not prefer a newer
    extraction or version key.
    """

    # Repository fixtures exercise the schema but are never legal-status
    # evidence.  Check both the bundle marker and every artifact marker so a
    # malformed model assembled with ``model_construct``/``model_copy`` still
    # cannot promote fixture evidence.
    if bundle.fixture_only or any(artifact.fixture_only for artifact in bundle.artifacts):
        return _abstain(instrument_id, as_of, ResolutionReason.FIXTURE_ONLY_DATA)
    if bundle.instrument.instrument_id != instrument_id:
        return _abstain(instrument_id, as_of, ResolutionReason.INSTRUMENT_NOT_FOUND)

    validated_versions = [version for version in bundle.versions if version.validation.is_validated]
    if not validated_versions:
        return _abstain(instrument_id, as_of, ResolutionReason.NO_VALIDATED_VERSION)

    eligible: list[tuple[LegalVersion, tuple[str, ...]]] = []
    conflict = False
    for version in validated_versions:
        publication = version.events.publication
        effective = version.events.effective
        if (
            publication is None
            or effective is None
            or not publication.supports_resolution
            or not effective.supports_resolution
            or publication.event_date > as_of
            or effective.event_date > as_of
        ):
            continue

        terminal_events = tuple(event for event in version.events.terminal_events() if event.event_date <= as_of)
        if any(event.supports_resolution for event in terminal_events):
            continue
        if terminal_events:
            conflict = True
            continue

        all_covering = [assertion for assertion in version.status_assertions if assertion.covers(as_of)]
        if any(not assertion.supports_resolution for assertion in all_covering):
            conflict = True
            continue
        covering = [assertion for assertion in all_covering if assertion.supports_resolution]
        if any(assertion.status is not LegalStatus.EFFECTIVE for assertion in covering):
            conflict = True
            continue
        effective_assertions = [assertion for assertion in covering if assertion.status is LegalStatus.EFFECTIVE]
        if len(effective_assertions) > 1:
            conflict = True
            continue
        if len(effective_assertions) == 1:
            assertion = effective_assertions[0]
            eligible.append(
                (
                    version,
                    (
                        publication.evidence.evidence_id,
                        effective.evidence.evidence_id,
                        assertion.evidence.evidence_id,
                    ),
                )
            )

    if conflict:
        return _abstain(instrument_id, as_of, ResolutionReason.CONFLICTING_STATUS_EVIDENCE)
    if len(eligible) > 1:
        return _abstain(instrument_id, as_of, ResolutionReason.AMBIGUOUS_VALIDATED_VERSIONS)
    if not eligible:
        return _abstain(instrument_id, as_of, ResolutionReason.STATUS_NOT_VALIDATED_FOR_DATE)

    version, evidence_ids = eligible[0]
    return ResolutionResult(
        instrument_id=instrument_id,
        as_of=as_of,
        reason=ResolutionReason.RESOLVED,
        legal_version_id=version.legal_version_id,
        evidence_ids=evidence_ids,
    )


def resolve_current(
    bundle: LegalVersionBundle,
    *,
    instrument_id: str,
    current_date: date | None = None,
) -> ResolutionResult:
    """Resolve for today (or an injected clock date) under the same strict rules."""

    return resolve_as_of(bundle, instrument_id=instrument_id, as_of=current_date or date.today())
