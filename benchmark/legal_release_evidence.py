"""Fail-closed evidence for retaining and reviewing authoritative legal bytes.

Citation v1 proves a database relationship and a normalized-text range.  It
does not, by itself, prove that the acquired source bytes, acquisition record,
or source-page mapping were retained for an auditor.  This module verifies a
separately signed, append-only checkpoint over those external artifacts.

The caller must separately provide the bank-approved hash of the latest
checkpoint.  A checkpoint cannot truthfully declare itself to be latest.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bddk_mcp.citations import CitationV1
from benchmark.signing import ed25519_public_key_fingerprint_sha256

_MAX_CHECKPOINT_BYTES = 8 * 1024 * 1024
_MAX_PUBLIC_KEY_BYTES = 16 * 1024
_MAX_SIGNATURE_BYTES = 1_024
_MAX_SOURCE_BYTES = 256 * 1024 * 1024
_MAX_ACQUISITION_RECORD_BYTES = 1 * 1024 * 1024
_MAX_PAGE_PROOF_BYTES = 32 * 1024 * 1024
_MAX_PAGE_TEXT_BYTES = 8 * 1024 * 1024
_MAX_CITATION_EXCERPT_BYTES = 128 * 1024
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_ARTIFACT_ID_PATTERN = r"^art_sha256_[0-9a-f]{64}$"
_BLOB_ID_PATTERN = r"^blob_sha256_[0-9a-f]{64}$"
_CITATION_ID_PATTERN = r"^cite_sha256_[0-9a-f]{64}$"


class LegalReleaseEvidenceError(ValueError):
    """Raised when a legal-release checkpoint or retained artifact is invalid."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _safe_relative_reference(value: str) -> str:
    candidate = Path(value)
    if candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError("artifact reference must be a normalized relative path")
    return candidate.as_posix()


class SealedFile(_StrictModel):
    reference: str = Field(min_length=1, max_length=500)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    bytes: int = Field(ge=1)

    _normalize_reference = field_validator("reference")(_safe_relative_reference)


class LegalArtifactEvidence(_StrictModel):
    artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    blob_id: str = Field(pattern=_BLOB_ID_PATTERN)
    citation_ids: tuple[str, ...] = Field(min_length=1, max_length=10_000)
    source_bytes: SealedFile
    acquisition_record: SealedFile
    page_mapping_proof: SealedFile

    @field_validator("citation_ids")
    @classmethod
    def _canonical_citation_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(re.fullmatch(_CITATION_ID_PATTERN, item) is None for item in value):
            raise ValueError("legal artifact evidence contains an invalid citation identity")
        if len(value) != len(set(value)) or value != tuple(sorted(value)):
            raise ValueError("legal artifact citation identities must be unique and canonically ordered")
        return value


class CheckpointIntegrity(_StrictModel):
    checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    signature_algorithm: Literal["ed25519"]
    signature_reference: str = Field(min_length=1, max_length=500)
    signature_public_key_sha256: str = Field(pattern=_SHA256_PATTERN)

    _normalize_signature_reference = field_validator("signature_reference")(_safe_relative_reference)


class LegalReleaseCheckpoint(_StrictModel):
    schema_version: Literal[1]
    checkpoint_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    legal_pack_sha256: str = Field(pattern=_SHA256_PATTERN)
    corpus_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    created_at: datetime
    predecessor_checkpoint_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    artifacts: tuple[LegalArtifactEvidence, ...] = Field(min_length=1, max_length=10_000)
    integrity: CheckpointIntegrity

    @field_validator("created_at")
    @classmethod
    def _aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("legal release checkpoint timestamp must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _canonical_artifacts(self) -> LegalReleaseCheckpoint:
        identities = tuple(item.artifact_id for item in self.artifacts)
        if len(identities) != len(set(identities)) or identities != tuple(sorted(identities)):
            raise ValueError("legal release artifacts must be unique and canonically ordered")
        return self


class AcquisitionRecord(_StrictModel):
    schema_version: Literal[1]
    artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    blob_id: str = Field(pattern=_BLOB_ID_PATTERN)
    canonical_uri: str = Field(pattern=r"^https://", max_length=2_000)
    retrieved_at: datetime
    captured_at: datetime
    source_authority: str = Field(min_length=1, max_length=200)
    media_type: str = Field(min_length=1, max_length=200)
    response_status: int = Field(ge=200, le=299)
    response_body_sha256: str = Field(pattern=_SHA256_PATTERN)
    response_body_bytes: int = Field(ge=1)

    @field_validator("retrieved_at", "captured_at")
    @classmethod
    def _aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("acquisition evidence timestamps must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _capture_follows_retrieval(self) -> AcquisitionRecord:
        if self.captured_at < self.retrieved_at:
            raise ValueError("acquisition evidence capture predates retrieval")
        return self


class SourcePage(_StrictModel):
    page_number: int = Field(ge=1)
    rendered_text: SealedFile


class CitationPageMapping(_StrictModel):
    citation_id: str = Field(pattern=_CITATION_ID_PATTERN)
    page_numbers: tuple[int, ...] = Field(min_length=1, max_length=1_000)
    rendered_excerpt: SealedFile

    @field_validator("page_numbers")
    @classmethod
    def _canonical_pages(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(item < 1 for item in value) or len(value) != len(set(value)) or value != tuple(sorted(value)):
            raise ValueError("citation page numbers must be positive, unique, and canonically ordered")
        return value


class PageMappingProof(_StrictModel):
    schema_version: Literal[1]
    proof_method: Literal["reviewed_source_page_mapping_v1"]
    mapping_profile: Literal["exact_utf8_excerpt_in_concatenated_page_text_v1"]
    artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    source_bytes_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_bytes: int = Field(ge=1)
    pages: tuple[SourcePage, ...] = Field(min_length=1, max_length=100_000)
    citation_mappings: tuple[CitationPageMapping, ...] = Field(min_length=1, max_length=10_000)
    reviewed_by_role: Literal["legal_source_reviewer"]
    reviewed_at: datetime

    @field_validator("reviewed_at")
    @classmethod
    def _aware_reviewed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("page mapping review timestamp must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _canonical_and_closed_mapping(self) -> PageMappingProof:
        page_numbers = tuple(item.page_number for item in self.pages)
        if len(page_numbers) != len(set(page_numbers)) or page_numbers != tuple(sorted(page_numbers)):
            raise ValueError("source pages must be unique and canonically ordered")
        page_references = tuple(item.rendered_text.reference for item in self.pages)
        if len(page_references) != len(set(page_references)):
            raise ValueError("source pages must use distinct retained-text artifacts")
        citation_ids = tuple(item.citation_id for item in self.citation_mappings)
        if len(citation_ids) != len(set(citation_ids)) or citation_ids != tuple(sorted(citation_ids)):
            raise ValueError("page-mapped citations must be unique and canonically ordered")
        known_pages = set(page_numbers)
        if any(not set(item.page_numbers) <= known_pages for item in self.citation_mappings):
            raise ValueError("citation page mapping references an unknown source page")
        return self


@dataclass(frozen=True, slots=True)
class LegalReleaseEvidenceValidation:
    checkpoint_sha256: str
    signing_key_sha256: str
    signing_key_fingerprint_sha256: str
    latest_checkpoint_verified: bool
    artifact_count: int
    citation_count: int


def canonical_checkpoint_payload(raw_checkpoint: dict[str, Any]) -> bytes:
    """Return deterministic bytes covered by checksum and detached signature."""

    try:
        checkpoint = LegalReleaseCheckpoint.model_validate(raw_checkpoint)
    except (ValueError, RecursionError) as exc:
        raise LegalReleaseEvidenceError("legal release checkpoint schema validation failed") from exc
    payload = checkpoint.model_dump(mode="json")
    payload["integrity"].pop("checkpoint_sha256", None)
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_checkpoint_sha256(raw_checkpoint: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_checkpoint_payload(raw_checkpoint)).hexdigest()


def _bounded_regular_bytes(path: Path, *, label: str, maximum_bytes: int) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LegalReleaseEvidenceError(f"{label} is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= maximum_bytes:
        raise LegalReleaseEvidenceError(f"{label} is not a bounded regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise LegalReleaseEvidenceError(f"{label} could not be read") from exc


def _load_mapping(
    path: Path, *, label: str, maximum_bytes: int, json_only: bool = False
) -> tuple[dict[str, Any], bytes]:
    raw_bytes = _bounded_regular_bytes(path, label=label, maximum_bytes=maximum_bytes)
    try:
        raw = json.loads(raw_bytes) if json_only else yaml.safe_load(raw_bytes)
    except (UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise LegalReleaseEvidenceError(f"{label} is invalid") from exc
    if not isinstance(raw, dict):
        raise LegalReleaseEvidenceError(f"{label} must be an object")
    return raw, raw_bytes


def _resolve_reference(root: Path, reference: str, *, label: str) -> Path:
    candidate = (root / reference).resolve()
    if not candidate.is_relative_to(root):
        raise LegalReleaseEvidenceError(f"{label} escaped its approved root")
    return candidate


def _verify_sealed_file(root: Path, sealed: SealedFile, *, label: str, maximum_bytes: int) -> tuple[Path, bytes]:
    if sealed.bytes > maximum_bytes:
        raise LegalReleaseEvidenceError(f"{label} exceeds its size bound")
    path = _resolve_reference(root, sealed.reference, label=label)
    raw_bytes = _bounded_regular_bytes(path, label=label, maximum_bytes=maximum_bytes)
    if len(raw_bytes) != sealed.bytes or hashlib.sha256(raw_bytes).hexdigest() != sealed.sha256:
        raise LegalReleaseEvidenceError(f"{label} differs from its sealed identity")
    return path, raw_bytes


def _verify_checkpoint_signature(
    checkpoint_path: Path,
    trusted_signing_key: Path,
    *,
    current: datetime,
) -> tuple[LegalReleaseCheckpoint, dict[str, Any], str, str]:
    raw, _ = _load_mapping(
        checkpoint_path,
        label="legal release checkpoint",
        maximum_bytes=_MAX_CHECKPOINT_BYTES,
    )
    try:
        checkpoint = LegalReleaseCheckpoint.model_validate(raw)
    except (ValueError, RecursionError) as exc:
        raise LegalReleaseEvidenceError("legal release checkpoint schema validation failed") from exc
    checksum = canonical_checkpoint_sha256(raw)
    if checkpoint.integrity.checkpoint_sha256 != checksum:
        raise LegalReleaseEvidenceError("legal release checkpoint checksum mismatch")
    if checkpoint.created_at > current:
        raise LegalReleaseEvidenceError("legal release checkpoint is from the future")

    key_bytes = _bounded_regular_bytes(
        trusted_signing_key.resolve(),
        label="trusted legal-release signing key",
        maximum_bytes=_MAX_PUBLIC_KEY_BYTES,
    )
    key_sha256 = hashlib.sha256(key_bytes).hexdigest()
    if key_sha256 != checkpoint.integrity.signature_public_key_sha256:
        raise LegalReleaseEvidenceError("trusted legal-release key hash differs from the checkpoint")
    signature_path = _resolve_reference(
        checkpoint_path.parent.resolve(),
        checkpoint.integrity.signature_reference,
        label="legal release signature",
    )
    signature = _bounded_regular_bytes(
        signature_path,
        label="legal release detached signature",
        maximum_bytes=_MAX_SIGNATURE_BYTES,
    )
    try:
        public_key = serialization.load_pem_public_key(key_bytes)
        if not isinstance(public_key, Ed25519PublicKey):
            raise ValueError("unsupported public key type")
        public_key.verify(signature, canonical_checkpoint_payload(raw))
    except (InvalidSignature, TypeError, ValueError):
        raise LegalReleaseEvidenceError("legal release checkpoint signature verification failed") from None
    return checkpoint, raw, key_sha256, ed25519_public_key_fingerprint_sha256(public_key)


def validate_legal_release_evidence(
    *,
    checkpoint_path: str | Path,
    trusted_signing_key: str | Path,
    source_root: str | Path,
    legal_pack_sha256: str,
    legal_pack_exported_at: datetime,
    corpus_manifest_sha256: str,
    citations: Sequence[CitationV1],
    now: datetime,
    predecessor_checkpoint_path: str | Path | None = None,
    trusted_latest_checkpoint_sha256: str | None = None,
) -> LegalReleaseEvidenceValidation:
    """Verify a signed checkpoint and every referenced retained evidence file."""

    current = now.astimezone(UTC) if now.tzinfo is not None and now.utcoffset() is not None else None
    if current is None:
        raise LegalReleaseEvidenceError("legal release validation time must include a timezone")
    checkpoint_file = Path(checkpoint_path).resolve()
    trusted_key = Path(trusted_signing_key).resolve()
    checkpoint, _, key_sha256, key_fingerprint = _verify_checkpoint_signature(
        checkpoint_file,
        trusted_key,
        current=current,
    )
    if checkpoint.legal_pack_sha256 != legal_pack_sha256:
        raise LegalReleaseEvidenceError("legal release checkpoint refers to a different legal pack")
    if checkpoint.corpus_manifest_sha256 != corpus_manifest_sha256:
        raise LegalReleaseEvidenceError("legal release checkpoint refers to a different corpus manifest")
    exported_at = legal_pack_exported_at.astimezone(UTC)
    if checkpoint.created_at < exported_at:
        raise LegalReleaseEvidenceError("legal release checkpoint predates the validated legal pack")

    predecessor_path = Path(predecessor_checkpoint_path).resolve() if predecessor_checkpoint_path else None
    if (checkpoint.predecessor_checkpoint_sha256 is None) != (predecessor_path is None):
        raise LegalReleaseEvidenceError("legal release predecessor checkpoint evidence is incomplete")
    if predecessor_path is not None:
        predecessor, _, predecessor_key_sha256, predecessor_key_fingerprint = _verify_checkpoint_signature(
            predecessor_path,
            trusted_key,
            current=current,
        )
        if predecessor.integrity.checkpoint_sha256 != checkpoint.predecessor_checkpoint_sha256:
            raise LegalReleaseEvidenceError("legal release predecessor checksum differs from the checkpoint")
        if predecessor.created_at >= checkpoint.created_at:
            raise LegalReleaseEvidenceError("legal release predecessor does not predate the checkpoint")
        if predecessor_key_sha256 != key_sha256:
            raise LegalReleaseEvidenceError("legal release predecessor uses a different trust anchor")
        if predecessor_key_fingerprint != key_fingerprint:
            raise LegalReleaseEvidenceError("legal release predecessor uses a different signer")

    latest_verified = False
    if trusted_latest_checkpoint_sha256 is not None:
        if re.fullmatch(_SHA256_PATTERN, trusted_latest_checkpoint_sha256) is None:
            raise LegalReleaseEvidenceError("trusted latest legal checkpoint hash is invalid")
        if trusted_latest_checkpoint_sha256 != checkpoint.integrity.checkpoint_sha256:
            raise LegalReleaseEvidenceError("legal release checkpoint is not the bank-approved latest checkpoint")
        latest_verified = True

    citation_by_id = {citation.citation_id: citation for citation in citations}
    if not citation_by_id or len(citation_by_id) != len(citations):
        raise LegalReleaseEvidenceError("validated legal pack citation identities are empty or duplicated")
    citations_by_artifact: dict[str, list[CitationV1]] = {}
    for citation in citations:
        citations_by_artifact.setdefault(citation.artifact_id, []).append(citation)
        if citation.generated_at > checkpoint.created_at:
            raise LegalReleaseEvidenceError("legal release checkpoint predates a cited Citation v1 bundle")

    evidence_by_artifact = {item.artifact_id: item for item in checkpoint.artifacts}
    if set(evidence_by_artifact) != set(citations_by_artifact):
        raise LegalReleaseEvidenceError("legal release artifact inventory differs from the validated legal pack")

    root = Path(source_root).resolve()
    for artifact_id, artifact_citations in citations_by_artifact.items():
        evidence = evidence_by_artifact[artifact_id]
        expected_citation_ids = tuple(sorted(item.citation_id for item in artifact_citations))
        if evidence.citation_ids != expected_citation_ids:
            raise LegalReleaseEvidenceError("legal release citation inventory differs from its artifact evidence")
        first = artifact_citations[0]
        if any(
            item.artifact_blob_id != first.artifact_blob_id
            or item.artifact_sha256 != first.artifact_sha256
            or item.source_url != first.source_url
            or item.artifact_retrieved_at != first.artifact_retrieved_at
            for item in artifact_citations
        ):
            raise LegalReleaseEvidenceError("citations disagree on immutable source acquisition identity")
        if evidence.blob_id != first.artifact_blob_id or evidence.source_bytes.sha256 != first.artifact_sha256:
            raise LegalReleaseEvidenceError("retained source identity differs from Citation v1")

        _, source_bytes = _verify_sealed_file(
            root,
            evidence.source_bytes,
            label="retained authoritative source bytes",
            maximum_bytes=_MAX_SOURCE_BYTES,
        )
        acquisition_path, _ = _verify_sealed_file(
            root,
            evidence.acquisition_record,
            label="retained source acquisition record",
            maximum_bytes=_MAX_ACQUISITION_RECORD_BYTES,
        )
        page_proof_path, _ = _verify_sealed_file(
            root,
            evidence.page_mapping_proof,
            label="retained source page-mapping proof",
            maximum_bytes=_MAX_PAGE_PROOF_BYTES,
        )

        raw_acquisition, _ = _load_mapping(
            acquisition_path,
            label="retained source acquisition record",
            maximum_bytes=_MAX_ACQUISITION_RECORD_BYTES,
            json_only=True,
        )
        raw_page_proof, _ = _load_mapping(
            page_proof_path,
            label="retained source page-mapping proof",
            maximum_bytes=_MAX_PAGE_PROOF_BYTES,
            json_only=True,
        )
        try:
            acquisition = AcquisitionRecord.model_validate(raw_acquisition)
            page_proof = PageMappingProof.model_validate(raw_page_proof)
        except (ValueError, RecursionError) as exc:
            raise LegalReleaseEvidenceError("retained legal source evidence schema validation failed") from exc

        if (
            acquisition.artifact_id != artifact_id
            or acquisition.blob_id != first.artifact_blob_id
            or acquisition.canonical_uri != first.source_url
            or acquisition.retrieved_at != first.artifact_retrieved_at
            or acquisition.response_body_sha256 != first.artifact_sha256
            or acquisition.response_body_bytes != len(source_bytes)
            or acquisition.captured_at > checkpoint.created_at
        ):
            raise LegalReleaseEvidenceError("retained acquisition record differs from Citation v1 or source bytes")
        if (
            page_proof.artifact_id != artifact_id
            or page_proof.source_bytes_sha256 != first.artifact_sha256
            or page_proof.source_bytes != len(source_bytes)
            or page_proof.reviewed_at > checkpoint.created_at
        ):
            raise LegalReleaseEvidenceError("retained page-mapping proof differs from Citation v1 or source bytes")
        page_text_by_number: dict[int, str] = {}
        for page in page_proof.pages:
            _, page_text_bytes = _verify_sealed_file(
                root,
                page.rendered_text,
                label="retained source-page text",
                maximum_bytes=_MAX_PAGE_TEXT_BYTES,
            )
            try:
                page_text_by_number[page.page_number] = page_text_bytes.decode("utf-8")
            except UnicodeError:
                raise LegalReleaseEvidenceError("retained source-page text is not UTF-8") from None
        mappings = {item.citation_id: item for item in page_proof.citation_mappings}
        if set(mappings) != set(expected_citation_ids):
            raise LegalReleaseEvidenceError("page-mapping citation inventory differs from the validated legal pack")
        for citation in artifact_citations:
            mapping = mappings[citation.citation_id]
            if mapping.rendered_excerpt.sha256 != citation.excerpt_sha256:
                raise LegalReleaseEvidenceError("page-mapping excerpt identity differs from Citation v1")
            _, excerpt_bytes = _verify_sealed_file(
                root,
                mapping.rendered_excerpt,
                label="retained Citation v1 excerpt",
                maximum_bytes=_MAX_CITATION_EXCERPT_BYTES,
            )
            try:
                excerpt = excerpt_bytes.decode("utf-8")
            except UnicodeError:
                raise LegalReleaseEvidenceError("retained Citation v1 excerpt is not UTF-8 text") from None
            if len(excerpt) != citation.excerpt_length:
                raise LegalReleaseEvidenceError("retained Citation v1 excerpt length differs from Citation v1")
            mapped_page_text = "\n".join(page_text_by_number[number] for number in mapping.page_numbers)
            if excerpt not in mapped_page_text:
                raise LegalReleaseEvidenceError("retained Citation v1 excerpt is absent from its mapped pages")

    return LegalReleaseEvidenceValidation(
        checkpoint_sha256=checkpoint.integrity.checkpoint_sha256,
        signing_key_sha256=key_sha256,
        signing_key_fingerprint_sha256=key_fingerprint,
        latest_checkpoint_verified=latest_verified,
        artifact_count=len(evidence_by_artifact),
        citation_count=len(citation_by_id),
    )


__all__ = (
    "LegalReleaseEvidenceError",
    "LegalReleaseEvidenceValidation",
    "canonical_checkpoint_payload",
    "canonical_checkpoint_sha256",
    "validate_legal_release_evidence",
)
