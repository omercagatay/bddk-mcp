"""Strict, corpus-bound expert evaluation datasets.

The tracked pilot is intentionally a draft annotation workload.  Loading it
proves schema and corpus referential integrity; it does not make the proposed
evidence legally current or expert-approved.  Release use is a separate,
fail-closed operation.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bddk_mcp.citations import (
    CitationV1,
    TrustedCitationContext,
    render_normalized_range_excerpt,
    verify_normalized_range_citation,
)
from bddk_mcp.corpus_manifest import (
    CorpusArtifact,
    CorpusManifestError,
    CorpusManifestValidation,
    load_and_validate_corpus_manifest,
)

EXPERT_EVALUATION_DRAFT_PATH = Path(__file__).with_name("expert_evaluation_draft.yml")
_MAX_DATASET_BYTES = 4 * 1024 * 1024
_MAX_SIGNATURE_BYTES = 1_024
_MAX_PUBLIC_KEY_BYTES = 16_384
_MAX_LEGAL_PACK_BYTES = 32 * 1024 * 1024
_MAX_LEGAL_ATTESTATION_BYTES = 1 * 1024 * 1024


class ExpertEvaluationError(ValueError):
    """Raised when evaluation data cannot be trusted."""


class ExpertEvaluationReleaseError(ExpertEvaluationError):
    """Raised when draft or incomplete data is requested for release use."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


Domain = Literal[
    "tfrs9_ecl",
    "credit_classification_provisioning",
    "capital_adequacy",
    "irb_credit_risk",
    "icaap_isedes",
    "liquidity_risk",
    "interest_rate_risk",
    "operational_risk",
]
QueryClass = Literal[
    "specific_provision",
    "document_lookup",
    "semantic_retrieval",
    "table_formula",
    "currentness",
    "version_comparison",
    "amendment_tracking",
]
AnnotatorRole = Literal[
    "regulatory_domain_expert",
    "audit_practitioner",
    "retrieval_evaluator",
    "legal_reviewer",
    "dataset_owner",
]


class DatasetCorpusIdentity(_StrictModel):
    """Immutable identity of the corpus against which cases were proposed."""

    manifest_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_built_at: datetime
    documents_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunks_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("corpus_built_at")
    @classmethod
    def _timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("corpus_built_at must include a timezone")
        return value


class EvidenceReference(_StrictModel):
    """Proposed corpus evidence; this is not an audit-grade Citation v1."""

    evidence_id: str = Field(pattern=r"^ev-[a-z0-9][a-z0-9-]{2,95}$")
    document_id: str = Field(min_length=1, max_length=128)
    document_title: str = Field(min_length=1, max_length=500)
    document_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_url: str = Field(pattern=r"^https?://", max_length=2_000)
    granularity: Literal["document", "section"]
    section_type: str | None = Field(default=None, pattern=r"^[a-z_]{1,32}$")
    section_ref: str | None = Field(default=None, min_length=1, max_length=64)
    section_content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    citation_label: str = Field(min_length=1, max_length=500)
    citation_v1_status: Literal["pending_legal_mapping", "verified"] = "pending_legal_mapping"
    citation_v1_id: str | None = Field(default=None, pattern=r"^cite_sha256_[0-9a-f]{64}$")
    citation_v1: CitationV1 | None = None
    corpus_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    legal_currentness: Literal["not_verified"]

    @model_validator(mode="after")
    def _granularity_matches_locator(self) -> EvidenceReference:
        section_values = (self.section_type, self.section_ref, self.section_content_sha256)
        if self.granularity == "section" and any(value is None for value in section_values):
            raise ValueError("section evidence requires type, reference, and immutable hash")
        if self.granularity == "document" and any(value is not None for value in section_values):
            raise ValueError("document evidence cannot carry a partial section locator")
        if self.citation_v1_status == "verified":
            if self.citation_v1_id is None or self.citation_v1 is None:
                raise ValueError("verified Citation v1 evidence requires the complete signed-dataset citation bundle")
            if self.citation_v1_id != self.citation_v1.citation_id:
                raise ValueError("Citation v1 identity differs from its complete citation bundle")
            if self.granularity != "section":
                raise ValueError("verified Citation v1 evidence must identify an exact section")
        elif self.citation_v1_id is not None or self.citation_v1 is not None:
            raise ValueError("pending Citation v1 evidence cannot claim a verified identity or bundle")
        return self


class NoAnswerExpectation(_StrictModel):
    """Why the corpus cannot safely support the requested conclusion."""

    reason_code: Literal[
        "legal_currentness_not_modeled",
        "version_history_not_modeled",
        "corpus_freshness_not_guaranteed",
        "corpus_not_exhaustive",
    ]
    basis_corpus_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required_follow_up: str = Field(min_length=1, max_length=500)


class IndependentAnnotation(_StrictModel):
    """An independent annotation slot or completed annotation."""

    annotation_id: str = Field(pattern=r"^ann-[a-z0-9][a-z0-9-]{2,127}$")
    annotator_role: AnnotatorRole
    independent: Literal[True]
    status: Literal["pending", "completed"]
    annotator_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    verdict: Literal["supported", "abstain", "needs_changes"] | None = None
    selected_positive_evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    selected_hard_negative_evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def _status_is_truthful(self) -> IndependentAnnotation:
        completed_values = (self.annotator_id, self.verdict, self.completed_at)
        if self.status == "pending":
            if any(value is not None for value in completed_values):
                raise ValueError("pending annotation cannot imply completed work")
            if self.selected_positive_evidence_ids or self.selected_hard_negative_evidence_ids:
                raise ValueError("pending annotation cannot select evidence")
        elif any(value is None for value in completed_values):
            raise ValueError("completed annotation requires annotator, verdict, and timestamp")
        if self.completed_at is not None and (
            self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None
        ):
            raise ValueError("annotation completion time must include a timezone")
        return self


class Adjudication(_StrictModel):
    """Resolution after independent annotation, never implied for drafts."""

    status: Literal["pending", "completed"]
    adjudicator_role: Literal["dataset_owner", "legal_reviewer"]
    disagreement: Literal["unassessed", "present", "absent"]
    adjudicator_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    outcome: Literal["supported", "abstain", "reject"] | None = None
    resolved_positive_evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    resolved_hard_negative_evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    resolution_note: str | None = Field(default=None, min_length=1, max_length=500)
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def _status_is_truthful(self) -> Adjudication:
        completed_values = (self.adjudicator_id, self.outcome, self.completed_at)
        if self.status == "pending":
            if self.disagreement != "unassessed" or any(value is not None for value in completed_values):
                raise ValueError("pending adjudication cannot imply a resolved decision")
            if self.resolved_positive_evidence_ids or self.resolved_hard_negative_evidence_ids:
                raise ValueError("pending adjudication cannot resolve evidence")
            if self.resolution_note is not None:
                raise ValueError("pending adjudication cannot include a resolution note")
        else:
            if self.disagreement == "unassessed" or any(value is None for value in completed_values):
                raise ValueError("completed adjudication requires assessed disagreement and decision metadata")
            if self.disagreement == "present" and self.resolution_note is None:
                raise ValueError("a disagreement resolution requires a concise note")
        if self.completed_at is not None and (
            self.completed_at.tzinfo is None or self.completed_at.utcoffset() is None
        ):
            raise ValueError("adjudication completion time must include a timezone")
        return self


class Approval(_StrictModel):
    """Explicit owner approval; draft is the only state with no approval metadata."""

    state: Literal["draft", "owner_approved", "rejected"]
    owner_role: Literal["dataset_owner"]
    owner_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    decided_at: datetime | None = None

    @model_validator(mode="after")
    def _approval_is_explicit(self) -> Approval:
        if self.state == "draft" and (self.owner_id is not None or self.decided_at is not None):
            raise ValueError("draft approval cannot contain owner decision metadata")
        if self.state != "draft" and (self.owner_id is None or self.decided_at is None):
            raise ValueError("an owner decision requires owner identity and timestamp")
        if self.decided_at is not None and (self.decided_at.tzinfo is None or self.decided_at.utcoffset() is None):
            raise ValueError("approval decision time must include a timezone")
        return self


class ExpertEvaluationCase(_StrictModel):
    """One proposed case whose release grain becomes adjudicated query case."""

    case_id: str = Field(pattern=r"^tr-[a-z0-9][a-z0-9-]{2,127}$")
    query: str = Field(min_length=5, max_length=1_000)
    query_class: QueryClass
    domain: Domain
    answerability: Literal["supported", "abstain"]
    positive_evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    hard_negative_evidence_ids: list[str] = Field(default_factory=list, max_length=50)
    no_answer: NoAnswerExpectation | None = None
    annotations: list[IndependentAnnotation] = Field(min_length=2, max_length=10)
    adjudication: Adjudication
    approval: Approval

    @field_validator("positive_evidence_ids", "hard_negative_evidence_ids")
    @classmethod
    def _evidence_ids_are_unique(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("case evidence references must be unique")
        return values

    @model_validator(mode="after")
    def _answerability_has_testable_ground_truth(self) -> ExpertEvaluationCase:
        if set(self.positive_evidence_ids) & set(self.hard_negative_evidence_ids):
            raise ValueError("positive and hard-negative evidence must be disjoint")
        if self.answerability == "supported":
            if not self.positive_evidence_ids or not self.hard_negative_evidence_ids:
                raise ValueError("supported cases require positive evidence and at least one hard negative")
            if self.no_answer is not None:
                raise ValueError("supported cases cannot carry a no-answer expectation")
        elif self.positive_evidence_ids or self.hard_negative_evidence_ids or self.no_answer is None:
            raise ValueError("abstention cases require only an explicit no-answer expectation")
        if self.query_class in {"currentness", "version_comparison", "amendment_tracking"} and (
            self.answerability != "abstain"
        ):
            raise ValueError("unmodeled legal status and history query classes must remain abstention cases")

        annotation_ids = [item.annotation_id for item in self.annotations]
        annotation_roles = [item.annotator_role for item in self.annotations]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError("annotation IDs must be unique within a case")
        if len(annotation_roles) != len(set(annotation_roles)):
            raise ValueError("independent annotation slots require distinct roles")
        return self


class DatasetIntegrity(_StrictModel):
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    signature_status: Literal["not_configured", "verified"] = "not_configured"
    signature_algorithm: Literal["ed25519"] | None = None
    signature_reference: str | None = Field(default=None, min_length=1, max_length=255)
    signature_public_key_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("signature_reference")
    @classmethod
    def _safe_signature_reference(cls, value: str | None) -> str | None:
        if value is None:
            return None
        candidate = Path(value)
        if candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
            raise ValueError("dataset signature reference must be a normalized relative path")
        return candidate.as_posix()

    @model_validator(mode="after")
    def _signature_state_is_complete(self) -> DatasetIntegrity:
        values = (self.signature_algorithm, self.signature_reference, self.signature_public_key_sha256)
        if self.signature_status == "verified" and any(value is None for value in values):
            raise ValueError("verified dataset signature requires algorithm, reference, and trusted-key hash")
        if self.signature_status == "not_configured" and any(value is not None for value in values):
            raise ValueError("dataset signature metadata is forbidden when signing is not configured")
        return self


class ValidatedLegalCitationPack(_StrictModel):
    """Bounded export from the curator-controlled validated-citation relation."""

    schema_version: Literal[1]
    export_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    source_relation: Literal["public.regulatory_validated_section_citations"]
    exported_at: datetime
    citations: list[CitationV1] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def _canonical_unique_citations(self) -> ValidatedLegalCitationPack:
        if self.exported_at.tzinfo is None or self.exported_at.utcoffset() is None:
            raise ValueError("legal citation pack timestamp must include a timezone")
        identities = [citation.citation_id for citation in self.citations]
        if len(identities) != len(set(identities)) or identities != sorted(identities):
            raise ValueError("legal citation pack identities must be unique and canonically ordered")
        return self


class LegalAttestationIntegrity(_StrictModel):
    attestation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    signature_algorithm: Literal["ed25519"]
    signature_reference: str = Field(min_length=1, max_length=255)
    signature_public_key_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("signature_reference")
    @classmethod
    def _safe_signature_reference(cls, value: str) -> str:
        candidate = Path(value)
        if candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
            raise ValueError("legal attestation signature reference must be a normalized relative path")
        return candidate.as_posix()


class LegalCitationAttestation(_StrictModel):
    """Detached legal-curator approval for one exact validated citation pack."""

    schema_version: Literal[1]
    attestation_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    curator_role: Literal["legal_curator"]
    pack_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    citation_ids: list[str] = Field(min_length=1, max_length=10_000)
    attested_at: datetime
    integrity: LegalAttestationIntegrity

    @field_validator("citation_ids")
    @classmethod
    def _canonical_citation_ids(cls, values: list[str]) -> list[str]:
        if any(not re.fullmatch(r"cite_sha256_[0-9a-f]{64}", value) for value in values):
            raise ValueError("legal attestation contains an invalid Citation v1 identity")
        if len(values) != len(set(values)) or values != sorted(values):
            raise ValueError("legal attestation citation identities must be unique and canonically ordered")
        return values

    @field_validator("attested_at")
    @classmethod
    def _attestation_time_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("legal attestation timestamp must include a timezone")
        return value


class ExpertEvaluationDataset(_StrictModel):
    """Versioned expert evaluation dataset contract."""

    schema_version: Literal[1]
    dataset_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    dataset_version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+(?:-draft\.[0-9]+)?$")
    language: Literal["tr"]
    intended_grain: Literal["one adjudicated query case"]
    intended_use: str = Field(min_length=1, max_length=1_000)
    created_at: datetime
    corpus: DatasetCorpusIdentity
    evidence_catalog: list[EvidenceReference] = Field(min_length=1, max_length=10_000)
    cases: list[ExpertEvaluationCase] = Field(min_length=1, max_length=10_000)
    limitations: list[str] = Field(min_length=1, max_length=100)
    approval: Approval
    integrity: DatasetIntegrity

    @field_validator("limitations")
    @classmethod
    def _limitations_are_distinct(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value or len(value) > 500 for value in normalized):
            raise ValueError("limitations must be non-empty and at most 500 characters")
        if len(normalized) != len(set(normalized)):
            raise ValueError("limitations must be unique")
        return normalized

    @model_validator(mode="after")
    def _keys_and_references_are_consistent(self) -> ExpertEvaluationDataset:
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("dataset creation time must include a timezone")

        case_ids = [case.case_id for case in self.cases]
        evidence_ids = [evidence.evidence_id for evidence in self.evidence_catalog]
        annotation_ids = [annotation.annotation_id for case in self.cases for annotation in case.annotations]
        for label, values in (
            ("case", case_ids),
            ("evidence", evidence_ids),
            ("annotation", annotation_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} IDs must be globally unique")

        catalog = set(evidence_ids)
        for evidence in self.evidence_catalog:
            if evidence.corpus_manifest_sha256 != self.corpus.manifest_sha256:
                raise ValueError("evidence is bound to a different corpus manifest")
        for case in self.cases:
            proposed = set(case.positive_evidence_ids) | set(case.hard_negative_evidence_ids)
            if not proposed <= catalog:
                raise ValueError("case references evidence missing from the catalog")
            if case.no_answer is not None and (
                case.no_answer.basis_corpus_manifest_sha256 != self.corpus.manifest_sha256
            ):
                raise ValueError("no-answer basis is bound to a different corpus manifest")
            for annotation in case.annotations:
                if not set(annotation.selected_positive_evidence_ids) <= set(case.positive_evidence_ids):
                    raise ValueError("annotation relabeled evidence outside the case's positive set")
                if not set(annotation.selected_hard_negative_evidence_ids) <= set(case.hard_negative_evidence_ids):
                    raise ValueError("annotation relabeled evidence outside the case's hard-negative set")
            if not set(case.adjudication.resolved_positive_evidence_ids) <= set(case.positive_evidence_ids):
                raise ValueError("adjudication relabeled evidence outside the case's positive set")
            if not set(case.adjudication.resolved_hard_negative_evidence_ids) <= set(case.hard_negative_evidence_ids):
                raise ValueError("adjudication relabeled evidence outside the case's hard-negative set")
        return self


@dataclass(frozen=True, slots=True)
class ExpertEvaluationValidation:
    dataset: ExpertEvaluationDataset
    dataset_sha256: str
    corpus_validation: CorpusManifestValidation
    dataset_signature_verified: bool
    legal_attestation_verified: bool
    legal_attestation_key_sha256: str | None


@dataclass(frozen=True, slots=True)
class ExpertDatasetQualityProfile:
    """Safe aggregate profile with no query, corpus text, or annotator identity."""

    dataset_id: str
    dataset_version: str
    dataset_sha256: str
    corpus_manifest_id: str
    corpus_manifest_sha256: str
    case_count: int
    evidence_count: int
    domain_counts: dict[str, int]
    query_class_counts: dict[str, int]
    answerability_counts: dict[str, int]
    case_approval_counts: dict[str, int]
    annotation_status_counts: dict[str, int]
    adjudication_status_counts: dict[str, int]
    release_blocker_counts: dict[str, int]
    currentness_unverified_evidence_count: int
    citation_v1_status_counts: dict[str, int]
    dataset_signature_verified: bool
    legal_attestation_verified: bool
    corpus_signature_verified: bool
    corpus_freshness_quantified: bool
    corpus_freshness_measured: bool
    release_ready: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def canonical_dataset_payload(raw_dataset: dict[str, Any]) -> bytes:
    """Canonical JSON payload covered by the dataset self-checksum."""

    try:
        payload = json.loads(json.dumps(raw_dataset, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ExpertEvaluationError("expert dataset canonicalization failed") from exc
    integrity = payload.get("integrity")
    if not isinstance(integrity, dict):
        raise ExpertEvaluationError("expert dataset integrity section is missing")
    integrity.pop("dataset_sha256", None)
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_dataset_sha256(raw_dataset: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_dataset_payload(raw_dataset)).hexdigest()


def _bounded_regular_bytes(path: Path, *, label: str, maximum_bytes: int) -> bytes:
    try:
        metadata = path.stat()
    except FileNotFoundError as exc:
        raise ExpertEvaluationError(f"expert dataset {label} is missing") from exc
    if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= maximum_bytes:
        raise ExpertEvaluationError(f"expert dataset {label} is not a bounded regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ExpertEvaluationError(f"expert dataset {label} could not be read") from exc


def _verify_dataset_signature(
    raw_dataset: dict[str, Any],
    dataset: ExpertEvaluationDataset,
    *,
    dataset_path: Path,
    trusted_signing_key: Path | None,
) -> bool:
    integrity = dataset.integrity
    if integrity.signature_status != "verified":
        return False
    if trusted_signing_key is None:
        raise ExpertEvaluationError("verified expert dataset requires a separately supplied trusted public key")

    key_path = trusted_signing_key.resolve()
    key_bytes = _bounded_regular_bytes(key_path, label="trusted signing key", maximum_bytes=_MAX_PUBLIC_KEY_BYTES)
    if hashlib.sha256(key_bytes).hexdigest() != integrity.signature_public_key_sha256:
        raise ExpertEvaluationError("trusted expert-dataset signing-key hash differs from the dataset")

    signature_path = (dataset_path.parent / (integrity.signature_reference or "")).resolve()
    if not signature_path.is_relative_to(dataset_path.parent):
        raise ExpertEvaluationError("expert dataset signature escaped its approved directory")
    signature = _bounded_regular_bytes(signature_path, label="detached signature", maximum_bytes=_MAX_SIGNATURE_BYTES)
    try:
        public_key = serialization.load_pem_public_key(key_bytes)
        if not isinstance(public_key, Ed25519PublicKey):
            raise ValueError("unsupported public key type")
        public_key.verify(signature, canonical_dataset_payload(raw_dataset))
    except (InvalidSignature, TypeError, ValueError):
        raise ExpertEvaluationError("expert dataset detached signature verification failed") from None
    return True


def canonical_legal_attestation_payload(raw_attestation: dict[str, Any]) -> bytes:
    """Canonical bytes covered by the legal curator's detached signature."""

    try:
        payload = json.loads(json.dumps(raw_attestation, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ExpertEvaluationError("legal citation attestation canonicalization failed") from exc
    integrity = payload.get("integrity")
    if not isinstance(integrity, dict):
        raise ExpertEvaluationError("legal citation attestation integrity is missing")
    integrity.pop("attestation_sha256", None)
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_legal_attestation_sha256(raw_attestation: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_legal_attestation_payload(raw_attestation)).hexdigest()


def _load_bounded_mapping(path: Path, *, label: str, maximum_bytes: int) -> tuple[dict[str, Any], bytes]:
    raw_bytes = _bounded_regular_bytes(path, label=label, maximum_bytes=maximum_bytes)
    try:
        raw = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise ExpertEvaluationError(f"expert dataset {label} is invalid") from exc
    if not isinstance(raw, dict):
        raise ExpertEvaluationError(f"expert dataset {label} must be a mapping")
    return raw, raw_bytes


def _verify_legal_attestation(
    dataset: ExpertEvaluationDataset,
    *,
    legal_pack_path: Path,
    attestation_path: Path,
    trusted_signing_key: Path,
    current: datetime,
) -> str:
    raw_pack, pack_bytes = _load_bounded_mapping(
        legal_pack_path.resolve(), label="validated legal citation pack", maximum_bytes=_MAX_LEGAL_PACK_BYTES
    )
    raw_attestation, _ = _load_bounded_mapping(
        attestation_path.resolve(), label="legal citation attestation", maximum_bytes=_MAX_LEGAL_ATTESTATION_BYTES
    )
    try:
        pack = ValidatedLegalCitationPack.model_validate(raw_pack)
        attestation = LegalCitationAttestation.model_validate(raw_attestation)
    except (ValueError, RecursionError) as exc:
        raise ExpertEvaluationError("legal citation pack or attestation schema validation failed") from exc
    if attestation.integrity.attestation_sha256 != canonical_legal_attestation_sha256(raw_attestation):
        raise ExpertEvaluationError("legal citation attestation checksum mismatch")
    if hashlib.sha256(pack_bytes).hexdigest() != attestation.pack_sha256:
        raise ExpertEvaluationError("validated legal citation pack hash differs from its attestation")

    key_path = trusted_signing_key.resolve()
    key_bytes = _bounded_regular_bytes(
        key_path, label="trusted legal-curator signing key", maximum_bytes=_MAX_PUBLIC_KEY_BYTES
    )
    key_sha256 = hashlib.sha256(key_bytes).hexdigest()
    if key_sha256 != attestation.integrity.signature_public_key_sha256:
        raise ExpertEvaluationError("trusted legal-curator key hash differs from the attestation")
    signature_path = (attestation_path.resolve().parent / attestation.integrity.signature_reference).resolve()
    if not signature_path.is_relative_to(attestation_path.resolve().parent):
        raise ExpertEvaluationError("legal citation attestation signature escaped its approved directory")
    signature = _bounded_regular_bytes(
        signature_path, label="legal-curator detached signature", maximum_bytes=_MAX_SIGNATURE_BYTES
    )
    try:
        public_key = serialization.load_pem_public_key(key_bytes)
        if not isinstance(public_key, Ed25519PublicKey):
            raise ValueError("unsupported public key type")
        public_key.verify(signature, canonical_legal_attestation_payload(raw_attestation))
    except (InvalidSignature, TypeError, ValueError):
        raise ExpertEvaluationError("legal citation attestation signature verification failed") from None

    if pack.exported_at > attestation.attested_at or attestation.attested_at > current:
        raise ExpertEvaluationError("legal citation pack and attestation timestamps are inconsistent")
    pack_by_id = {citation.citation_id: citation for citation in pack.citations}
    if list(pack_by_id) != attestation.citation_ids:
        raise ExpertEvaluationError("legal citation attestation inventory differs from its pack")
    dataset_citations = {
        evidence.citation_v1.citation_id: evidence.citation_v1
        for evidence in dataset.evidence_catalog
        if evidence.citation_v1 is not None
    }
    if set(dataset_citations) != set(pack_by_id):
        raise ExpertEvaluationError("expert dataset Citation v1 inventory differs from the validated legal pack")
    if any(dataset_citations[citation_id] != citation for citation_id, citation in pack_by_id.items()):
        raise ExpertEvaluationError("expert dataset Citation v1 bundle differs from the validated legal pack")
    if any(citation.generated_at > pack.exported_at for citation in pack.citations):
        raise ExpertEvaluationError("validated legal pack export predates a Citation v1 bundle")
    return key_sha256


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        metadata = path.stat()
    except FileNotFoundError as exc:
        raise ExpertEvaluationError("expert evaluation dataset is missing") from exc
    if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= _MAX_DATASET_BYTES:
        raise ExpertEvaluationError("expert evaluation dataset is not a bounded regular file")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ExpertEvaluationError("expert evaluation dataset YAML is invalid") from exc
    if not isinstance(raw, dict):
        raise ExpertEvaluationError("expert evaluation dataset must be a mapping")
    return raw


def _artifact_for_role(validation: CorpusManifestValidation, role: str) -> CorpusArtifact:
    matching = [artifact for artifact in validation.manifest.artifacts if artifact.role == role]
    if len(matching) != 1:
        raise ExpertEvaluationError(f"corpus manifest requires exactly one {role} artifact")
    return matching[0]


def _load_bound_corpus_json(corpus_root: Path, artifact: CorpusArtifact) -> Any:
    """Read the exact corpus bytes whose identity the manifest validated."""

    root = corpus_root.resolve()
    path = (root / artifact.path).resolve()
    if not path.is_relative_to(root):
        raise ExpertEvaluationError("validated corpus artifact escaped its approved root")
    try:
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != artifact.bytes:
            raise ExpertEvaluationError("validated corpus artifact changed after manifest validation")
        with path.open("rb") as handle:
            payload = handle.read(artifact.bytes + 1)
    except OSError as exc:
        raise ExpertEvaluationError("validated corpus artifact could not be read") from exc
    if len(payload) != artifact.bytes or hashlib.sha256(payload).hexdigest() != artifact.sha256:
        raise ExpertEvaluationError("validated corpus artifact changed after manifest validation")
    try:
        return json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExpertEvaluationError("validated corpus artifact is not valid JSON") from exc


def _load_corpus_indexes(
    validation: CorpusManifestValidation,
    corpus_root: Path,
) -> tuple[dict[str, dict[str, Any]], set[tuple[str, str, str, str]]]:
    document_rows = _load_bound_corpus_json(corpus_root, _artifact_for_role(validation, "documents"))
    chunk_rows = _load_bound_corpus_json(corpus_root, _artifact_for_role(validation, "chunks"))
    if not isinstance(document_rows, list) or not isinstance(chunk_rows, list):
        raise ExpertEvaluationError("validated corpus indexes have an unsupported shape")

    documents: dict[str, dict[str, Any]] = {}
    for row in document_rows:
        if not isinstance(row, dict):
            raise ExpertEvaluationError("documents index contains a non-object row")
        document_id = row.get("document_id")
        if not isinstance(document_id, str) or document_id in documents:
            raise ExpertEvaluationError("documents index has missing or duplicate identifiers")
        documents[document_id] = row

    sections: set[tuple[str, str, str, str]] = set()
    for row in chunk_rows:
        if not isinstance(row, dict):
            raise ExpertEvaluationError("chunks index contains a non-object row")
        values = (
            row.get("doc_id"),
            row.get("section_type"),
            row.get("section_ref"),
            row.get("section_content_hash"),
        )
        if all(isinstance(value, str) and value for value in values):
            sections.add(values)  # type: ignore[arg-type]
    return documents, sections


def _verify_corpus_binding(
    dataset: ExpertEvaluationDataset,
    validation: CorpusManifestValidation,
    *,
    corpus_root: Path,
) -> None:
    manifest = validation.manifest
    if dataset.corpus.manifest_id != manifest.manifest_id:
        raise ExpertEvaluationError("expert dataset corpus manifest ID has drifted")
    if dataset.corpus.manifest_sha256 != validation.manifest_sha256:
        raise ExpertEvaluationError("expert dataset corpus manifest checksum has drifted")
    if dataset.corpus.corpus_built_at != manifest.freshness.corpus_built_at:
        raise ExpertEvaluationError("expert dataset corpus build identity has drifted")
    documents_artifact = _artifact_for_role(validation, "documents")
    chunks_artifact = _artifact_for_role(validation, "chunks")
    if (
        dataset.corpus.documents_sha256 != documents_artifact.sha256
        or dataset.corpus.chunks_sha256 != chunks_artifact.sha256
    ):
        raise ExpertEvaluationError("expert dataset corpus artifact identity has drifted")

    documents, sections = _load_corpus_indexes(validation, corpus_root)
    for evidence in dataset.evidence_catalog:
        row = documents.get(evidence.document_id)
        if row is None:
            raise ExpertEvaluationError("expert evidence references a missing corpus document")
        expected_metadata = (
            evidence.document_title,
            evidence.document_content_sha256,
            evidence.source_url,
        )
        observed_metadata = (row.get("title"), row.get("content_hash"), row.get("source_url"))
        if expected_metadata != observed_metadata:
            raise ExpertEvaluationError("expert evidence document metadata or hash has drifted")
        if evidence.granularity == "section":
            locator = (
                evidence.document_id,
                evidence.section_type or "",
                evidence.section_ref or "",
                evidence.section_content_sha256 or "",
            )
            if locator not in sections:
                raise ExpertEvaluationError("expert evidence section locator or hash has drifted")
        citation = evidence.citation_v1
        if citation is not None:
            citation_identity = (
                citation.citation_id,
                citation.source_document_id,
                citation.normalized_document_sha256,
                citation.source_url,
                citation.provision_text_sha256,
            )
            expected_identity = (
                evidence.citation_v1_id,
                evidence.document_id,
                evidence.document_content_sha256,
                evidence.source_url,
                evidence.section_content_sha256,
            )
            if citation_identity != expected_identity:
                raise ExpertEvaluationError("expert evidence Citation v1 bundle differs from its corpus locator")
            normalized_document = row.get("markdown_content")
            if not isinstance(normalized_document, str):
                raise ExpertEvaluationError("expert evidence Citation v1 document text is unavailable")
            context_data = citation.model_dump()
            for field in ("schema_version", "citation_id", "generated_at"):
                context_data.pop(field, None)
            try:
                expected_context = TrustedCitationContext.model_validate(context_data)
                rendered_excerpt = render_normalized_range_excerpt(citation, normalized_document)
                verification = verify_normalized_range_citation(
                    citation,
                    normalized_document=normalized_document,
                    rendered_excerpt=rendered_excerpt,
                    expected=expected_context,
                )
            except ValueError as exc:
                raise ExpertEvaluationError("expert evidence Citation v1 reconstruction failed") from exc
            if not verification.valid:
                raise ExpertEvaluationError("expert evidence Citation v1 reconstruction failed")


def _verify_temporal_order(dataset: ExpertEvaluationDataset, *, current: datetime) -> None:
    lower_bound = max(dataset.created_at, dataset.corpus.corpus_built_at)
    case_decisions: list[datetime] = []
    for evidence in dataset.evidence_catalog:
        if evidence.citation_v1 is not None:
            generated_at = evidence.citation_v1.generated_at
            if generated_at < dataset.corpus.corpus_built_at or generated_at > current:
                raise ExpertEvaluationError("expert evidence Citation v1 timestamp is outside the dataset window")

    for case in dataset.cases:
        annotation_times: list[datetime] = []
        for annotation in case.annotations:
            if annotation.completed_at is not None:
                if annotation.completed_at < lower_bound or annotation.completed_at > current:
                    raise ExpertEvaluationError("expert annotation timestamp is outside the review window")
                annotation_times.append(annotation.completed_at)

        adjudicated_at = case.adjudication.completed_at
        if adjudicated_at is not None:
            if adjudicated_at < lower_bound or adjudicated_at > current:
                raise ExpertEvaluationError("expert adjudication timestamp is outside the review window")
            if annotation_times and adjudicated_at < max(annotation_times):
                raise ExpertEvaluationError("expert adjudication predates an independent annotation")

        approved_at = case.approval.decided_at
        if approved_at is not None:
            if adjudicated_at is None or approved_at < adjudicated_at or approved_at > current:
                raise ExpertEvaluationError("expert case approval is outside the adjudicated review window")
            case_decisions.append(approved_at)

    dataset_approved_at = dataset.approval.decided_at
    if dataset_approved_at is not None:
        if dataset_approved_at < lower_bound or dataset_approved_at > current:
            raise ExpertEvaluationError("expert dataset approval is outside the review window")
        if len(case_decisions) != len(dataset.cases) or dataset_approved_at < max(case_decisions):
            raise ExpertEvaluationError("expert dataset approval predates complete case approval")


def load_expert_evaluation_dataset(
    path: str | Path = EXPERT_EVALUATION_DRAFT_PATH,
    *,
    corpus_manifest_path: str | Path | None = None,
    corpus_root: str | Path | None = None,
    trusted_dataset_signing_key: str | Path | None = None,
    trusted_corpus_signing_key: str | Path | None = None,
    validated_legal_pack_path: str | Path | None = None,
    legal_attestation_path: str | Path | None = None,
    trusted_legal_attestation_key: str | Path | None = None,
    now: datetime | None = None,
    require_release_ready: bool = False,
) -> ExpertEvaluationValidation:
    """Validate schema, self-checksum, corpus binding, and optional release gate."""

    dataset_path = Path(path).resolve()
    raw = _load_yaml_mapping(dataset_path)
    try:
        dataset = ExpertEvaluationDataset.model_validate(raw)
        checksum = canonical_dataset_sha256(raw)
    except ExpertEvaluationError:
        raise
    except (ValueError, RecursionError) as exc:
        raise ExpertEvaluationError("expert evaluation dataset schema validation failed") from exc
    if dataset.integrity.dataset_sha256 != checksum:
        raise ExpertEvaluationError("expert evaluation dataset checksum mismatch")
    dataset_signature_verified = _verify_dataset_signature(
        raw,
        dataset,
        dataset_path=dataset_path,
        trusted_signing_key=(
            Path(trusted_dataset_signing_key).resolve() if trusted_dataset_signing_key is not None else None
        ),
    )

    root = Path(corpus_root or Path(__file__).resolve().parents[1] / "seed_data").resolve()
    manifest_path = Path(corpus_manifest_path or root / "corpus_scope.yml").resolve()
    try:
        corpus_validation = load_and_validate_corpus_manifest(
            manifest_path,
            corpus_root=root,
            now=now,
            trusted_signing_key=(
                Path(trusted_corpus_signing_key).resolve() if trusted_corpus_signing_key is not None else None
            ),
        )
    except CorpusManifestError as exc:
        raise ExpertEvaluationError("expert dataset corpus validation failed") from exc
    _verify_corpus_binding(dataset, corpus_validation, corpus_root=root)

    current = now or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ExpertEvaluationError("expert dataset validation time must include a timezone")
    if dataset.created_at > current:
        raise ExpertEvaluationError("expert evaluation dataset has a future creation time")
    if dataset.created_at < dataset.corpus.corpus_built_at:
        raise ExpertEvaluationError("expert evaluation dataset predates its bound corpus")
    _verify_temporal_order(dataset, current=current)

    legal_inputs = (validated_legal_pack_path, legal_attestation_path, trusted_legal_attestation_key)
    if any(value is not None for value in legal_inputs) and not all(value is not None for value in legal_inputs):
        raise ExpertEvaluationError("validated legal pack, attestation, and separate trust anchor are all required")
    legal_attestation_key_sha256: str | None = None
    if all(value is not None for value in legal_inputs):
        legal_attestation_key_sha256 = _verify_legal_attestation(
            dataset,
            legal_pack_path=Path(str(validated_legal_pack_path)).resolve(),
            attestation_path=Path(str(legal_attestation_path)).resolve(),
            trusted_signing_key=Path(str(trusted_legal_attestation_key)).resolve(),
            current=current,
        )

    result = ExpertEvaluationValidation(
        dataset=dataset,
        dataset_sha256=checksum,
        corpus_validation=corpus_validation,
        dataset_signature_verified=dataset_signature_verified,
        legal_attestation_verified=legal_attestation_key_sha256 is not None,
        legal_attestation_key_sha256=legal_attestation_key_sha256,
    )
    if require_release_ready:
        require_expert_dataset_release_ready(result)
    return result


_MINIMUM_RELEASE_CASES = 20
_MINIMUM_RELEASE_DOMAINS = 5
_REQUIRED_RELEASE_QUERY_CLASSES = frozenset(
    {"specific_provision", "semantic_retrieval", "currentness", "table_formula"}
)


def _release_blockers(validation: ExpertEvaluationValidation) -> Counter[str]:
    dataset = validation.dataset
    blockers: Counter[str] = Counter()
    if not validation.dataset_signature_verified:
        blockers["dataset_signature_not_verified"] += 1
    if not validation.legal_attestation_verified:
        blockers["legal_citation_attestation_not_verified"] += 1
    elif (
        dataset.integrity.signature_public_key_sha256 is not None
        and validation.legal_attestation_key_sha256 == dataset.integrity.signature_public_key_sha256
    ):
        blockers["dataset_and_legal_signers_not_separated"] += 1
    corpus_integrity = validation.corpus_validation.manifest.integrity
    if corpus_integrity.signature_status != "verified":
        blockers["corpus_signature_not_verified"] += 1
    freshness = validation.corpus_validation.manifest.freshness
    freshness_values = (
        freshness.source_detection_slo_seconds,
        freshness.publication_slo_seconds,
        freshness.max_manifest_age_seconds,
    )
    blockers["unquantified_corpus_freshness_objectives"] += sum(value is None for value in freshness_values)
    if freshness.slo_evidence_status != "measured":
        blockers["corpus_freshness_slo_not_measured"] += 1
    if len(dataset.cases) < _MINIMUM_RELEASE_CASES:
        blockers["minimum_case_count_not_met"] += _MINIMUM_RELEASE_CASES - len(dataset.cases)
    domain_count = len({case.domain for case in dataset.cases})
    if domain_count < _MINIMUM_RELEASE_DOMAINS:
        blockers["minimum_domain_coverage_not_met"] += _MINIMUM_RELEASE_DOMAINS - domain_count
    query_classes = {case.query_class for case in dataset.cases}
    blockers["missing_required_query_classes"] += len(_REQUIRED_RELEASE_QUERY_CLASSES - query_classes)
    answerability = {case.answerability for case in dataset.cases}
    blockers["missing_answerability_classes"] += len({"supported", "abstain"} - answerability)
    if dataset.approval.state != "owner_approved":
        blockers["dataset_not_owner_approved"] += 1
    if "-draft." in dataset.dataset_version:
        blockers["draft_dataset_version"] += 1
    blockers["pending_citation_v1_evidence"] += sum(
        evidence.citation_v1_status != "verified" for evidence in dataset.evidence_catalog
    )

    for case in dataset.cases:
        if case.approval.state != "owner_approved":
            blockers["case_not_owner_approved"] += 1
        completed = [annotation for annotation in case.annotations if annotation.status == "completed"]
        blockers["pending_annotations"] += len(case.annotations) - len(completed)
        if len(completed) < 2:
            blockers["insufficient_independent_annotations"] += 1
        else:
            annotator_ids = [annotation.annotator_id for annotation in completed]
            roles = [annotation.annotator_role for annotation in completed]
            if None in annotator_ids or len(annotator_ids) != len(set(annotator_ids)):
                blockers["non_independent_annotators"] += 1
            if len(roles) != len(set(roles)):
                blockers["non_independent_annotation_roles"] += 1
            if any(annotation.verdict == "needs_changes" for annotation in completed):
                blockers["annotation_needs_changes"] += 1
            for annotation in completed:
                if annotation.verdict == "supported" and not annotation.selected_positive_evidence_ids:
                    blockers["annotation_missing_positive_evidence"] += 1
                if annotation.verdict == "abstain" and (
                    annotation.selected_positive_evidence_ids or annotation.selected_hard_negative_evidence_ids
                ):
                    blockers["abstention_annotation_selected_evidence"] += 1
        if case.adjudication.status != "completed":
            blockers["pending_adjudications"] += 1
        else:
            expected_outcome = "supported" if case.answerability == "supported" else "abstain"
            if case.adjudication.outcome != expected_outcome:
                blockers["adjudication_outcome_mismatch"] += 1
            if case.answerability == "supported" and not case.adjudication.resolved_positive_evidence_ids:
                blockers["adjudication_missing_positive_evidence"] += 1
            if case.answerability == "abstain" and (
                case.adjudication.resolved_positive_evidence_ids
                or case.adjudication.resolved_hard_negative_evidence_ids
            ):
                blockers["abstention_adjudication_selected_evidence"] += 1
            completed_verdicts = {annotation.verdict for annotation in completed}
            disagreement_exists = len(completed_verdicts) > 1 or any(
                verdict != case.adjudication.outcome for verdict in completed_verdicts
            )
            expected_disagreement = "present" if disagreement_exists else "absent"
            if completed and case.adjudication.disagreement != expected_disagreement:
                blockers["disagreement_state_mismatch"] += 1
        if (
            case.approval.decided_at is not None
            and case.adjudication.completed_at is not None
            and case.approval.decided_at < case.adjudication.completed_at
        ):
            blockers["approval_predates_adjudication"] += 1
    completed_at = [case.adjudication.completed_at for case in dataset.cases]
    if (
        dataset.approval.decided_at is not None
        and all(value is not None for value in completed_at)
        and dataset.approval.decided_at < max(value for value in completed_at if value is not None)
    ):
        blockers["dataset_approval_predates_adjudication"] += 1
    return +blockers


def require_expert_dataset_release_ready(validation: ExpertEvaluationValidation) -> None:
    """Refuse benchmark release use until independent review is complete."""

    blockers = _release_blockers(validation)
    if blockers:
        summary = ", ".join(f"{name}={count}" for name, count in sorted(blockers.items()))
        raise ExpertEvaluationReleaseError(f"expert evaluation dataset is not release-ready ({summary})")


def profile_expert_evaluation_dataset(
    validation: ExpertEvaluationValidation,
) -> ExpertDatasetQualityProfile:
    """Return an aggregate-only data-quality profile safe for logs and reports."""

    dataset = validation.dataset
    blockers = _release_blockers(validation)
    annotations = [annotation for case in dataset.cases for annotation in case.annotations]
    return ExpertDatasetQualityProfile(
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        dataset_sha256=validation.dataset_sha256,
        corpus_manifest_id=dataset.corpus.manifest_id,
        corpus_manifest_sha256=dataset.corpus.manifest_sha256,
        case_count=len(dataset.cases),
        evidence_count=len(dataset.evidence_catalog),
        domain_counts=dict(sorted(Counter(case.domain for case in dataset.cases).items())),
        query_class_counts=dict(sorted(Counter(case.query_class for case in dataset.cases).items())),
        answerability_counts=dict(sorted(Counter(case.answerability for case in dataset.cases).items())),
        case_approval_counts=dict(sorted(Counter(case.approval.state for case in dataset.cases).items())),
        annotation_status_counts=dict(sorted(Counter(item.status for item in annotations).items())),
        adjudication_status_counts=dict(sorted(Counter(case.adjudication.status for case in dataset.cases).items())),
        release_blocker_counts=dict(sorted(blockers.items())),
        currentness_unverified_evidence_count=sum(
            evidence.legal_currentness == "not_verified" for evidence in dataset.evidence_catalog
        ),
        citation_v1_status_counts=dict(
            sorted(Counter(evidence.citation_v1_status for evidence in dataset.evidence_catalog).items())
        ),
        dataset_signature_verified=validation.dataset_signature_verified,
        legal_attestation_verified=validation.legal_attestation_verified,
        corpus_signature_verified=(validation.corpus_validation.manifest.integrity.signature_status == "verified"),
        corpus_freshness_quantified=all(
            value is not None
            for value in (
                validation.corpus_validation.manifest.freshness.source_detection_slo_seconds,
                validation.corpus_validation.manifest.freshness.publication_slo_seconds,
                validation.corpus_validation.manifest.freshness.max_manifest_age_seconds,
            )
        ),
        corpus_freshness_measured=(validation.corpus_validation.manifest.freshness.slo_evidence_status == "measured"),
        release_ready=not blockers,
    )
