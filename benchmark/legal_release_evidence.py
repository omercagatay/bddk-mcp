"""Fail-closed evidence for retaining and reviewing authoritative legal bytes.

Citation v1 proves a database relationship and a normalized-text range.  It
does not, by itself, prove that the acquired source bytes, acquisition record,
or source-page mapping were retained for an auditor.  This module verifies a
separately signed, append-only checkpoint over those external artifacts.

The verifier must receive an independent latest-checkpoint anchor: an explicit
hash in policy-free development or the approved head from a separately verified
trust policy. A checkpoint cannot truthfully declare itself to be latest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bddk_mcp.citations import CitationV1
from bddk_mcp.release_yaml import ReleaseYamlError, load_bounded_release_yaml
from benchmark.signing import ed25519_public_key_fingerprint_sha256

_MAX_CHECKPOINT_BYTES = 8 * 1024 * 1024
_MAX_PUBLIC_KEY_BYTES = 16 * 1024
_MAX_SIGNATURE_BYTES = 1_024
_MAX_SOURCE_BYTES = 256 * 1024 * 1024
_MAX_ACQUISITION_RECORD_BYTES = 1 * 1024 * 1024
_MAX_PAGE_PROOF_BYTES = 32 * 1024 * 1024
_MAX_PAGE_TEXT_BYTES = 8 * 1024 * 1024
_MAX_CITATION_EXCERPT_BYTES = 128 * 1024
_MAX_CHAIN_CHECKPOINTS = 256
_MAX_RETAINED_FILES_PER_CHAIN = 100_000
_MAX_RETAINED_BYTES_PER_CHAIN = 8 * 1024 * 1024 * 1024
_MAX_PAGE_TEXT_BYTES_PER_ARTIFACT = 256 * 1024 * 1024
_HASH_BLOCK_BYTES = 1024 * 1024
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_ARTIFACT_ID_PATTERN = r"^art_sha256_[0-9a-f]{64}$"
_BLOB_ID_PATTERN = r"^blob_sha256_[0-9a-f]{64}$"
_CITATION_ID_PATTERN = r"^cite_sha256_[0-9a-f]{64}$"
_OWNER_ID_PATTERN = r"^[a-z0-9][a-z0-9._:@/-]{2,127}$"


class LegalReleaseEvidenceError(ValueError):
    """Raised when a legal-release checkpoint or retained artifact is invalid."""


class _DuplicateJsonKeyError(ValueError):
    """Internal marker for ambiguous release-significant JSON."""


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError("duplicate JSON object key")
        result[key] = value
    return result


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _safe_relative_reference(value: str) -> str:
    if "\\" in value or any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("artifact reference must use printable POSIX path characters")
    candidate = Path(value)
    if candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError("artifact reference must be a normalized relative path")
    return candidate.as_posix()


def _as_utc(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise LegalReleaseEvidenceError(f"{label} must include a timezone")
    return value.astimezone(UTC)


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
    schema_version: Literal[2]
    checkpoint_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    signer_role: Literal["legal_release_certifier"]
    legal_pack_sha256: str = Field(pattern=_SHA256_PATTERN)
    corpus_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    created_at: datetime
    predecessor_checkpoint_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    predecessor_checkpoint_reference: str | None = Field(default=None, max_length=500)
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
        predecessor_values = (self.predecessor_checkpoint_sha256, self.predecessor_checkpoint_reference)
        if (predecessor_values[0] is None) != (predecessor_values[1] is None):
            raise ValueError("legal release predecessor hash and reference must be supplied together")
        return self

    @field_validator("predecessor_checkpoint_reference")
    @classmethod
    def _safe_predecessor_reference(cls, value: str | None) -> str | None:
        return _safe_relative_reference(value) if value is not None else None


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
    schema_version: Literal[1, 2]
    proof_method: Literal["reviewed_source_page_mapping_v1", "reviewed_source_page_mapping_v2"]
    mapping_profile: Literal["exact_utf8_excerpt_in_concatenated_page_text_v1"]
    artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    source_bytes_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_bytes: int = Field(ge=1)
    pages: tuple[SourcePage, ...] = Field(min_length=1, max_length=100_000)
    citation_mappings: tuple[CitationPageMapping, ...] = Field(min_length=1, max_length=10_000)
    reviewed_by_role: Literal["legal_source_reviewer"]
    reviewed_by_owner_id: str | None = Field(default=None, pattern=_OWNER_ID_PATTERN)
    reviewed_at: datetime

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("page mapping schema version must be an integer")
        return value

    @field_validator("reviewed_at", mode="before")
    @classmethod
    def _reject_numeric_reviewed_at(cls, value: object) -> object:
        if isinstance(value, (bool, int, float)):
            raise ValueError("page mapping review timestamp must be ISO text or a datetime")
        if isinstance(value, str) and re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
            value.strip(),
        ):
            raise ValueError("page mapping review timestamp must not be a numeric string")
        return value

    @field_validator("reviewed_at")
    @classmethod
    def _aware_reviewed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("page mapping review timestamp must include a timezone")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _canonical_and_closed_mapping(self) -> PageMappingProof:
        if self.schema_version == 1 and (
            self.proof_method != "reviewed_source_page_mapping_v1" or self.reviewed_by_owner_id is not None
        ):
            raise ValueError("page mapping v1 cannot claim a policy-bound reviewer")
        if self.schema_version == 2 and (
            self.proof_method != "reviewed_source_page_mapping_v2" or self.reviewed_by_owner_id is None
        ):
            raise ValueError("page mapping v2 requires a policy-bound reviewer owner")
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
    checkpoint_created_at: datetime
    signing_key_sha256: str
    signing_key_fingerprint_sha256: str
    latest_checkpoint_verified: bool
    chain_checkpoint_count: int
    genesis_checkpoint_sha256: str
    artifact_count: int
    citation_count: int
    chain_signers: tuple[LegalReleaseCheckpointSigner, ...]
    configured_signing_key_fingerprints_sha256: tuple[str, ...]
    source_reviews: tuple[LegalSourceReview, ...]


@dataclass(frozen=True, slots=True)
class LegalReleaseCheckpointSigner:
    checkpoint_sha256: str
    checkpoint_created_at: datetime
    signing_key_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class LegalSourceReview:
    checkpoint_sha256: str
    artifact_id: str
    proof_schema_version: int
    reviewer_owner_id: str | None
    reviewed_at: datetime


@dataclass(frozen=True, slots=True)
class _TrustedSigningKey:
    key_sha256: str
    key_fingerprint_sha256: str
    public_key: Ed25519PublicKey


@dataclass(frozen=True, slots=True)
class _RetainedArtifact:
    source_bytes_size: int
    acquisition: AcquisitionRecord
    page_proof: PageMappingProof
    mapping_by_citation_id: dict[str, CitationPageMapping]
    excerpt_length_by_citation_id: dict[str, int]


@dataclass(slots=True)
class _RetentionBudget:
    file_count: int = 0
    declared_bytes: int = 0

    def consume(self, sealed: SealedFile) -> None:
        self.file_count += 1
        self.declared_bytes += sealed.bytes
        if self.file_count > _MAX_RETAINED_FILES_PER_CHAIN:
            raise LegalReleaseEvidenceError("legal release retained-file count exceeds its chain bound")
        if self.declared_bytes > _MAX_RETAINED_BYTES_PER_CHAIN:
            raise LegalReleaseEvidenceError("legal release retained bytes exceed their chain bound")


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
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise LegalReleaseEvidenceError(f"{label} is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= maximum_bytes:
            raise LegalReleaseEvidenceError(f"{label} is not a bounded regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read(maximum_bytes + 1)
        if len(payload) != metadata.st_size:
            raise LegalReleaseEvidenceError(f"{label} changed while it was read")
        return payload
    except OSError as exc:
        raise LegalReleaseEvidenceError(f"{label} could not be read") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _load_mapping(
    path: Path, *, label: str, maximum_bytes: int, json_only: bool = False
) -> tuple[dict[str, Any], bytes]:
    raw_bytes = _bounded_regular_bytes(path, label=label, maximum_bytes=maximum_bytes)
    return _parse_mapping_bytes(raw_bytes, label=label, json_only=json_only), raw_bytes


def _parse_mapping_bytes(raw_bytes: bytes, *, label: str, json_only: bool = False) -> dict[str, Any]:
    try:
        raw = (
            json.loads(raw_bytes, object_pairs_hook=_unique_json_object)
            if json_only
            else load_bounded_release_yaml(raw_bytes, maximum_bytes=len(raw_bytes))
        )
    except (UnicodeError, json.JSONDecodeError, _DuplicateJsonKeyError, ReleaseYamlError) as exc:
        raise LegalReleaseEvidenceError(f"{label} is invalid") from exc
    if not isinstance(raw, dict):
        raise LegalReleaseEvidenceError(f"{label} must be an object")
    return raw


def _resolve_reference(root: Path, reference: str, *, label: str) -> Path:
    unresolved = root / reference
    cursor = root
    for part in Path(reference).parts:
        cursor /= part
        try:
            if stat.S_ISLNK(cursor.lstat().st_mode):
                raise LegalReleaseEvidenceError(f"{label} uses a symbolic-link path")
        except FileNotFoundError:
            break
    candidate = unresolved.resolve()
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


def _verify_sealed_file_streaming(
    root: Path,
    sealed: SealedFile,
    *,
    label: str,
    maximum_bytes: int,
) -> tuple[Path, int]:
    """Verify a sealed file without retaining its potentially large body."""

    if sealed.bytes > maximum_bytes:
        raise LegalReleaseEvidenceError(f"{label} exceeds its size bound")
    path = _resolve_reference(root, sealed.reference, label=label)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise LegalReleaseEvidenceError(f"{label} is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != sealed.bytes:
            raise LegalReleaseEvidenceError(f"{label} differs from its sealed identity")
        digest = hashlib.sha256()
        read_bytes = 0
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            while block := handle.read(_HASH_BLOCK_BYTES):
                read_bytes += len(block)
                if read_bytes > maximum_bytes:
                    raise LegalReleaseEvidenceError(f"{label} exceeds its size bound")
                digest.update(block)
        if read_bytes != sealed.bytes or digest.hexdigest() != sealed.sha256:
            raise LegalReleaseEvidenceError(f"{label} differs from its sealed identity")
        return path, read_bytes
    except OSError as exc:
        raise LegalReleaseEvidenceError(f"{label} could not be read") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _verify_checkpoint_signature(
    checkpoint_path: Path,
    trusted_keys_by_sha256: dict[str, _TrustedSigningKey],
    *,
    current: datetime,
    label: str = "legal release checkpoint",
) -> tuple[LegalReleaseCheckpoint, dict[str, Any], str, str]:
    raw, _ = _load_mapping(
        checkpoint_path,
        label=label,
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

    trusted_key = trusted_keys_by_sha256.get(checkpoint.integrity.signature_public_key_sha256)
    if trusted_key is None:
        raise LegalReleaseEvidenceError("legal release checkpoint uses an untrusted signing key")
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
        trusted_key.public_key.verify(signature, canonical_checkpoint_payload(raw))
    except InvalidSignature:
        raise LegalReleaseEvidenceError("legal release checkpoint signature verification failed") from None
    return (
        checkpoint,
        raw,
        trusted_key.key_sha256,
        trusted_key.key_fingerprint_sha256,
    )


def _load_trusted_signing_keyring(
    signing_keys: Sequence[str | Path],
) -> tuple[dict[str, _TrustedSigningKey], str]:
    if not 1 <= len(signing_keys) <= 32:
        raise LegalReleaseEvidenceError("legal release signing keyring exceeds its bound")
    by_sha256: dict[str, _TrustedSigningKey] = {}
    fingerprints: set[str] = set()
    primary_key_sha256: str | None = None
    for position, signing_key in enumerate(signing_keys):
        key_bytes = _bounded_regular_bytes(
            Path(signing_key),
            label="trusted legal-release signing key",
            maximum_bytes=_MAX_PUBLIC_KEY_BYTES,
        )
        key_sha256 = hashlib.sha256(key_bytes).hexdigest()
        try:
            public_key = serialization.load_pem_public_key(key_bytes)
            if not isinstance(public_key, Ed25519PublicKey):
                raise ValueError("unsupported public key type")
        except (TypeError, ValueError):
            raise LegalReleaseEvidenceError("trusted legal-release signing key is invalid") from None
        fingerprint = ed25519_public_key_fingerprint_sha256(public_key)
        if key_sha256 in by_sha256 or fingerprint in fingerprints:
            raise LegalReleaseEvidenceError("legal release signing keyring contains a duplicate signer")
        by_sha256[key_sha256] = _TrustedSigningKey(
            key_sha256=key_sha256,
            key_fingerprint_sha256=fingerprint,
            public_key=public_key,
        )
        fingerprints.add(fingerprint)
        if position == 0:
            primary_key_sha256 = key_sha256
    if primary_key_sha256 is None:  # pragma: no cover - guarded by the length bound
        raise LegalReleaseEvidenceError("legal release signing keyring is empty")
    return by_sha256, primary_key_sha256


def _verify_checkpoint_retention(
    checkpoint: LegalReleaseCheckpoint,
    *,
    root: Path,
    budget: _RetentionBudget,
) -> dict[str, _RetainedArtifact]:
    """Re-hash every retained artifact named by one signed checkpoint."""

    retained: dict[str, _RetainedArtifact] = {}
    for evidence in checkpoint.artifacts:
        budget.consume(evidence.source_bytes)
        _, source_bytes_size = _verify_sealed_file_streaming(
            root,
            evidence.source_bytes,
            label="retained authoritative source bytes",
            maximum_bytes=_MAX_SOURCE_BYTES,
        )
        budget.consume(evidence.acquisition_record)
        _, acquisition_bytes = _verify_sealed_file(
            root,
            evidence.acquisition_record,
            label="retained source acquisition record",
            maximum_bytes=_MAX_ACQUISITION_RECORD_BYTES,
        )
        budget.consume(evidence.page_mapping_proof)
        _, page_proof_bytes = _verify_sealed_file(
            root,
            evidence.page_mapping_proof,
            label="retained source page-mapping proof",
            maximum_bytes=_MAX_PAGE_PROOF_BYTES,
        )
        raw_acquisition = _parse_mapping_bytes(
            acquisition_bytes,
            label="retained source acquisition record",
            json_only=True,
        )
        raw_page_proof = _parse_mapping_bytes(
            page_proof_bytes,
            label="retained source page-mapping proof",
            json_only=True,
        )
        try:
            acquisition = AcquisitionRecord.model_validate(raw_acquisition)
            page_proof = PageMappingProof.model_validate(raw_page_proof)
        except (ValueError, RecursionError):
            raise LegalReleaseEvidenceError("retained legal source evidence schema validation failed") from None

        if (
            acquisition.artifact_id != evidence.artifact_id
            or acquisition.blob_id != evidence.blob_id
            or acquisition.response_body_sha256 != evidence.source_bytes.sha256
            or acquisition.response_body_bytes != source_bytes_size
            or acquisition.captured_at > checkpoint.created_at
        ):
            raise LegalReleaseEvidenceError("retained acquisition record differs from its signed checkpoint")
        if (
            page_proof.artifact_id != evidence.artifact_id
            or page_proof.source_bytes_sha256 != evidence.source_bytes.sha256
            or page_proof.source_bytes != source_bytes_size
            or page_proof.reviewed_at < acquisition.captured_at
            or page_proof.reviewed_at > checkpoint.created_at
        ):
            raise LegalReleaseEvidenceError("retained page-mapping proof differs from its signed checkpoint")

        page_text_by_number: dict[int, str] = {}
        page_text_declared_bytes = 0
        for page in page_proof.pages:
            budget.consume(page.rendered_text)
            page_text_declared_bytes += page.rendered_text.bytes
            if page_text_declared_bytes > _MAX_PAGE_TEXT_BYTES_PER_ARTIFACT:
                raise LegalReleaseEvidenceError("retained source-page text exceeds its per-artifact bound")
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
        if set(mappings) != set(evidence.citation_ids):
            raise LegalReleaseEvidenceError("page-mapping citation inventory differs from its signed checkpoint")
        excerpt_lengths: dict[str, int] = {}
        for citation_id, mapping in mappings.items():
            budget.consume(mapping.rendered_excerpt)
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
            mapped_page_text = "\n".join(page_text_by_number[number] for number in mapping.page_numbers)
            if excerpt not in mapped_page_text:
                raise LegalReleaseEvidenceError("retained Citation v1 excerpt is absent from its mapped pages")
            excerpt_lengths[citation_id] = len(excerpt)
        retained[evidence.artifact_id] = _RetainedArtifact(
            source_bytes_size=source_bytes_size,
            acquisition=acquisition,
            page_proof=page_proof,
            mapping_by_citation_id=mappings,
            excerpt_length_by_citation_id=excerpt_lengths,
        )
    return retained


def _source_reviews(
    checkpoint: LegalReleaseCheckpoint,
    retained: dict[str, _RetainedArtifact],
) -> tuple[LegalSourceReview, ...]:
    return tuple(
        LegalSourceReview(
            checkpoint_sha256=checkpoint.integrity.checkpoint_sha256,
            artifact_id=artifact_id,
            proof_schema_version=artifact.page_proof.schema_version,
            reviewer_owner_id=artifact.page_proof.reviewed_by_owner_id,
            reviewed_at=artifact.page_proof.reviewed_at,
        )
        for artifact_id, artifact in sorted(retained.items())
    )


def validate_legal_release_evidence(
    *,
    checkpoint_path: str | Path,
    trusted_signing_key: str | Path,
    source_root: str | Path,
    legal_pack_sha256: str,
    legal_pack_exported_at: datetime,
    legal_pack_attested_at: datetime,
    corpus_manifest_sha256: str,
    corpus_approved_at: datetime,
    citations: Sequence[CitationV1],
    now: datetime,
    predecessor_checkpoint_path: str | Path | None = None,
    trusted_latest_checkpoint_sha256: str | None = None,
    trusted_predecessor_signing_keys: Sequence[str | Path] = (),
) -> LegalReleaseEvidenceValidation:
    """Verify a signed checkpoint and every referenced retained evidence file."""

    current = now.astimezone(UTC) if now.tzinfo is not None and now.utcoffset() is not None else None
    if current is None:
        raise LegalReleaseEvidenceError("legal release validation time must include a timezone")
    checkpoint_file = Path(checkpoint_path).resolve()
    trusted_keyring, primary_key_sha256 = _load_trusted_signing_keyring(
        (trusted_signing_key, *trusted_predecessor_signing_keys)
    )
    configured_signing_key_fingerprints = tuple(
        trusted_key.key_fingerprint_sha256 for trusted_key in trusted_keyring.values()
    )
    checkpoint, _, key_sha256, key_fingerprint = _verify_checkpoint_signature(
        checkpoint_file,
        trusted_keyring,
        current=current,
    )
    if key_sha256 != primary_key_sha256:
        raise LegalReleaseEvidenceError("latest legal release checkpoint does not use the primary signing key")
    if checkpoint.legal_pack_sha256 != legal_pack_sha256:
        raise LegalReleaseEvidenceError("legal release checkpoint refers to a different legal pack")
    if checkpoint.corpus_manifest_sha256 != corpus_manifest_sha256:
        raise LegalReleaseEvidenceError("legal release checkpoint refers to a different corpus manifest")
    exported_at = _as_utc(legal_pack_exported_at, label="legal pack export timestamp")
    if checkpoint.created_at < exported_at:
        raise LegalReleaseEvidenceError("legal release checkpoint predates the validated legal pack")
    attested_at = _as_utc(legal_pack_attested_at, label="legal-curator approval timestamp")
    if checkpoint.created_at < attested_at:
        raise LegalReleaseEvidenceError("legal release checkpoint predates legal-curator approval")
    approved_at = _as_utc(corpus_approved_at, label="corpus approval timestamp")
    if checkpoint.created_at < approved_at:
        raise LegalReleaseEvidenceError("legal release checkpoint predates corpus build or scope approval")

    root = Path(source_root).resolve()
    retention_budget = _RetentionBudget()
    retained_by_artifact = _verify_checkpoint_retention(checkpoint, root=root, budget=retention_budget)
    source_review_batches = [_source_reviews(checkpoint, retained_by_artifact)]
    supplied_predecessor = Path(predecessor_checkpoint_path).resolve() if predecessor_checkpoint_path else None
    chain_path = checkpoint_file
    chain_checkpoint = checkpoint
    seen_checkpoints = {checkpoint.integrity.checkpoint_sha256}
    chain_signers = [
        LegalReleaseCheckpointSigner(
            checkpoint_sha256=checkpoint.integrity.checkpoint_sha256,
            checkpoint_created_at=checkpoint.created_at,
            signing_key_fingerprint_sha256=key_fingerprint,
        )
    ]
    for depth in range(_MAX_CHAIN_CHECKPOINTS):
        predecessor_sha256 = chain_checkpoint.predecessor_checkpoint_sha256
        predecessor_reference = chain_checkpoint.predecessor_checkpoint_reference
        if predecessor_sha256 is None or predecessor_reference is None:
            if depth == 0 and supplied_predecessor is not None:
                raise LegalReleaseEvidenceError("a genesis legal release checkpoint cannot have predecessor input")
            break
        predecessor_path = _resolve_reference(
            chain_path.parent.resolve(),
            predecessor_reference,
            label="legal release predecessor checkpoint",
        )
        if depth == 0 and supplied_predecessor is not None and predecessor_path != supplied_predecessor:
            raise LegalReleaseEvidenceError("supplied legal release predecessor differs from the signed reference")
        predecessor, _, _, predecessor_key_fingerprint = _verify_checkpoint_signature(
            predecessor_path,
            trusted_keyring,
            current=current,
            label="legal release predecessor checkpoint",
        )
        if predecessor.integrity.checkpoint_sha256 != predecessor_sha256:
            raise LegalReleaseEvidenceError("legal release predecessor checksum differs from the checkpoint")
        if predecessor.integrity.checkpoint_sha256 in seen_checkpoints:
            raise LegalReleaseEvidenceError("legal release predecessor chain contains a cycle")
        if predecessor.created_at >= chain_checkpoint.created_at:
            raise LegalReleaseEvidenceError("legal release predecessor does not predate the checkpoint")
        predecessor_retained = _verify_checkpoint_retention(predecessor, root=root, budget=retention_budget)
        source_review_batches.append(_source_reviews(predecessor, predecessor_retained))
        seen_checkpoints.add(predecessor.integrity.checkpoint_sha256)
        chain_signers.append(
            LegalReleaseCheckpointSigner(
                checkpoint_sha256=predecessor.integrity.checkpoint_sha256,
                checkpoint_created_at=predecessor.created_at,
                signing_key_fingerprint_sha256=predecessor_key_fingerprint,
            )
        )
        chain_path = predecessor_path
        chain_checkpoint = predecessor
    else:
        raise LegalReleaseEvidenceError("legal release predecessor chain exceeds its verification bound")

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

    for artifact_id, artifact_citations in citations_by_artifact.items():
        evidence = evidence_by_artifact[artifact_id]
        retained = retained_by_artifact[artifact_id]
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

        if (
            retained.acquisition.canonical_uri != first.source_url
            or retained.acquisition.retrieved_at != first.artifact_retrieved_at
            or retained.acquisition.response_body_sha256 != first.artifact_sha256
        ):
            raise LegalReleaseEvidenceError("retained acquisition record differs from Citation v1 or source bytes")
        if (
            retained.page_proof.source_bytes_sha256 != first.artifact_sha256
            or retained.page_proof.source_bytes != retained.source_bytes_size
        ):
            raise LegalReleaseEvidenceError("retained page-mapping proof differs from Citation v1 or source bytes")
        for citation in artifact_citations:
            mapping = retained.mapping_by_citation_id[citation.citation_id]
            if mapping.rendered_excerpt.sha256 != citation.excerpt_sha256:
                raise LegalReleaseEvidenceError("page-mapping excerpt identity differs from Citation v1")
            if retained.excerpt_length_by_citation_id[citation.citation_id] != citation.excerpt_length:
                raise LegalReleaseEvidenceError("retained Citation v1 excerpt length differs from Citation v1")

    return LegalReleaseEvidenceValidation(
        checkpoint_sha256=checkpoint.integrity.checkpoint_sha256,
        checkpoint_created_at=checkpoint.created_at,
        signing_key_sha256=key_sha256,
        signing_key_fingerprint_sha256=key_fingerprint,
        latest_checkpoint_verified=latest_verified,
        chain_checkpoint_count=len(seen_checkpoints),
        genesis_checkpoint_sha256=chain_checkpoint.integrity.checkpoint_sha256,
        artifact_count=len(evidence_by_artifact),
        citation_count=len(citation_by_id),
        chain_signers=tuple(reversed(chain_signers)),
        configured_signing_key_fingerprints_sha256=configured_signing_key_fingerprints,
        source_reviews=tuple(review for batch in reversed(source_review_batches) for review in batch),
    )


__all__ = (
    "LegalReleaseEvidenceError",
    "LegalReleaseEvidenceValidation",
    "LegalReleaseCheckpointSigner",
    "LegalSourceReview",
    "canonical_checkpoint_payload",
    "canonical_checkpoint_sha256",
    "validate_legal_release_evidence",
)
