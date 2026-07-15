"""Fail-closed scope and artifact identity for a reviewed regulatory corpus.

The manifest deliberately describes *selection and provenance*, not legal
applicability.  A valid checksum proves that the reviewed declaration and its
artifact identities agree; it is not a digital signature and it does not turn
an extraction snapshot into an authoritative legal version.
"""

from __future__ import annotations

import hashlib
import json
import math
import stat
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bddk_mcp.release_yaml import ReleaseYamlError, load_bounded_release_yaml

CORPUS_MANIFEST_FILENAME = "corpus_scope.yml"
CORPUS_SCOPE_WARNING = (
    "This corpus is a job-specific selection, not exhaustive BDDK coverage. "
    "Freshness and legal applicability must be verified against authoritative sources."
)
_MAX_MANIFEST_BYTES = 1_000_000
_MAX_JSON_ARTIFACT_BYTES = 128 * 1024 * 1024
_MAX_SIGNATURE_BYTES = 1_024
_MAX_PUBLIC_KEY_BYTES = 16_384


class CorpusManifestError(ValueError):
    """Raised when corpus scope, integrity, or freshness cannot be trusted."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CorpusArtifact(_StrictModel):
    """One immutable artifact covered by the scope declaration."""

    role: Literal["documents", "chunks", "decision_cache", "other"]
    path: str = Field(min_length=1, max_length=255)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0, le=_MAX_JSON_ARTIFACT_BYTES)
    records: int | None = Field(default=None, ge=0)

    @field_validator("path")
    @classmethod
    def _safe_relative_path(cls, value: str) -> str:
        candidate = Path(value)
        if candidate.is_absolute() or not candidate.parts or any(part in {"", ".", ".."} for part in candidate.parts):
            raise ValueError("artifact path must be a normalized relative path")
        return candidate.as_posix()


class CorpusFreshness(_StrictModel):
    """Observed extraction times and owner-defined service objectives."""

    source_observed_start: datetime
    source_observed_end: datetime
    corpus_built_at: datetime
    scope_reviewed_at: datetime
    business_expectation: str = Field(min_length=1, max_length=100)
    source_detection_slo_seconds: int | None = Field(default=None, ge=1)
    publication_slo_seconds: int | None = Field(default=None, ge=1)
    max_manifest_age_seconds: int | None = Field(default=None, ge=1)
    slo_evidence_status: Literal["not_measured", "measured"] = "not_measured"

    @model_validator(mode="after")
    def _chronology_and_timezones(self) -> CorpusFreshness:
        values = (
            self.source_observed_start,
            self.source_observed_end,
            self.corpus_built_at,
            self.scope_reviewed_at,
        )
        if any(value.tzinfo is None or value.utcoffset() is None for value in values):
            raise ValueError("freshness timestamps must include a timezone")
        if self.source_observed_start > self.source_observed_end:
            raise ValueError("source observation range is reversed")
        if self.source_observed_end > self.corpus_built_at:
            raise ValueError("corpus build predates the latest source observation")
        if self.corpus_built_at > self.scope_reviewed_at:
            raise ValueError("scope review predates the corpus build")
        objectives = (
            self.source_detection_slo_seconds,
            self.publication_slo_seconds,
            self.max_manifest_age_seconds,
        )
        if self.slo_evidence_status == "measured" and any(value is None for value in objectives):
            raise ValueError("measured freshness requires all numeric objectives")
        return self


class CorpusIntegrity(_StrictModel):
    """Self-checksum plus an explicit, non-implied signature state."""

    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
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
            raise ValueError("signature reference must be a normalized relative path")
        return candidate.as_posix()

    @model_validator(mode="after")
    def _signature_reference_matches_status(self) -> CorpusIntegrity:
        signature_fields = (
            self.signature_algorithm,
            self.signature_reference,
            self.signature_public_key_sha256,
        )
        if self.signature_status == "verified" and any(value is None for value in signature_fields):
            raise ValueError("verified signature requires algorithm, reference, and trusted-key hash")
        if self.signature_status == "not_configured" and any(value is not None for value in signature_fields):
            raise ValueError("signature metadata is not allowed when signing is not configured")
        return self


class CorpusScopeManifest(_StrictModel):
    """Version 1 machine-readable corpus selection contract."""

    schema_version: Literal[1]
    manifest_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    selection_owner: str = Field(min_length=1, max_length=200)
    purpose: str = Field(min_length=1, max_length=1_000)
    exhaustive: Literal[False]
    included_source_classes: list[str] = Field(min_length=1, max_length=100)
    excluded_source_classes: list[str] = Field(min_length=1, max_length=100)
    known_gaps: list[str] = Field(min_length=1, max_length=100)
    freshness: CorpusFreshness
    artifacts: list[CorpusArtifact] = Field(min_length=1, max_length=100)
    integrity: CorpusIntegrity

    @field_validator("included_source_classes", "excluded_source_classes", "known_gaps")
    @classmethod
    def _normalized_nonempty_values(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value or len(value) > 500 for value in normalized):
            raise ValueError("scope entries must be non-empty and at most 500 characters")
        if len(set(normalized)) != len(normalized):
            raise ValueError("scope entries must be unique")
        return normalized

    @model_validator(mode="after")
    def _scope_and_artifacts_are_unambiguous(self) -> CorpusScopeManifest:
        overlap = set(self.included_source_classes) & set(self.excluded_source_classes)
        if overlap:
            raise ValueError("included and excluded source classes overlap")
        paths = [artifact.path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("artifact paths must be unique")
        non_other_roles = [artifact.role for artifact in self.artifacts if artifact.role != "other"]
        if len(non_other_roles) != len(set(non_other_roles)):
            raise ValueError("well-known artifact roles must be unique")
        if "documents" not in non_other_roles:
            raise ValueError("a documents artifact is required")
        return self


@dataclass(frozen=True, slots=True)
class CorpusManifestValidation:
    manifest: CorpusScopeManifest
    manifest_sha256: str
    signing_key_fingerprint_sha256: str | None
    warnings: tuple[str, ...]


def canonical_manifest_payload(raw_manifest: dict[str, Any]) -> bytes:
    """Serialize the declaration signed and hashed by the integrity contract."""

    payload = json.loads(json.dumps(raw_manifest, ensure_ascii=False, allow_nan=False))
    integrity = payload.get("integrity")
    if not isinstance(integrity, dict):
        raise CorpusManifestError("corpus manifest integrity section is missing")
    integrity.pop("manifest_sha256", None)
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_manifest_sha256(raw_manifest: dict[str, Any]) -> str:
    """Hash canonical JSON after removing only the checksum's own value."""

    return hashlib.sha256(canonical_manifest_payload(raw_manifest)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _bounded_regular_file(path: Path, *, label: str, maximum_bytes: int) -> bytes:
    try:
        metadata = path.stat()
    except FileNotFoundError as exc:
        raise CorpusManifestError(f"corpus {label} is missing") from exc
    if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= maximum_bytes:
        raise CorpusManifestError(f"corpus {label} is not a bounded regular file")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise CorpusManifestError(f"corpus {label} could not be read") from exc


def _verify_manifest_signature(
    raw_manifest: dict[str, Any],
    manifest: CorpusScopeManifest,
    *,
    corpus_root: Path,
    trusted_signing_key: Path | None,
) -> str | None:
    integrity = manifest.integrity
    if integrity.signature_status != "verified":
        return None
    if trusted_signing_key is None:
        raise CorpusManifestError("verified corpus signature requires a separately supplied trusted public key")

    key_path = trusted_signing_key.resolve()
    key_bytes = _bounded_regular_file(key_path, label="trusted signing key", maximum_bytes=_MAX_PUBLIC_KEY_BYTES)
    if hashlib.sha256(key_bytes).hexdigest() != integrity.signature_public_key_sha256:
        raise CorpusManifestError("trusted corpus signing-key hash differs from the manifest")
    signature_path = _artifact_path(corpus_root, integrity.signature_reference or "")
    signature = _bounded_regular_file(
        signature_path,
        label="detached signature",
        maximum_bytes=_MAX_SIGNATURE_BYTES,
    )
    try:
        public_key = serialization.load_pem_public_key(key_bytes)
        if not isinstance(public_key, Ed25519PublicKey):
            raise ValueError("unsupported public key type")
        public_key.verify(signature, canonical_manifest_payload(raw_manifest))
    except (InvalidSignature, TypeError, ValueError):
        raise CorpusManifestError("corpus manifest detached signature verification failed") from None
    canonical_key = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(canonical_key).hexdigest()


def _artifact_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise CorpusManifestError("corpus artifact escapes the approved root")
    try:
        metadata = candidate.stat()
    except FileNotFoundError as exc:
        raise CorpusManifestError(f"corpus artifact is missing: {relative_path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise CorpusManifestError(f"corpus artifact is not a regular file: {relative_path}")
    return candidate


def _load_json_records(path: Path, artifact: CorpusArtifact) -> list[Any] | dict[str, Any]:
    if path.stat().st_size > _MAX_JSON_ARTIFACT_BYTES:
        raise CorpusManifestError(f"corpus JSON artifact exceeds the verification limit: {artifact.path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CorpusManifestError(f"corpus JSON artifact is invalid: {artifact.path}") from exc
    if not isinstance(value, (list, dict)):
        raise CorpusManifestError(f"corpus JSON artifact must contain a list or object: {artifact.path}")
    if artifact.records is not None and len(value) != artifact.records:
        raise CorpusManifestError(f"corpus artifact record count differs from the manifest: {artifact.path}")
    return value


def _timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise CorpusManifestError(f"documents artifact contains an invalid {field} timestamp")
    try:
        return datetime.fromtimestamp(float(value), UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise CorpusManifestError(f"documents artifact contains an invalid {field} timestamp") from exc


def _verify_document_freshness(records: list[Any] | dict[str, Any], manifest: CorpusScopeManifest) -> None:
    rows = list(records.values()) if isinstance(records, dict) else records
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise CorpusManifestError("documents artifact must contain document objects")
    downloaded = [_timestamp(row.get("downloaded_at"), field="downloaded_at") for row in rows]
    extracted = [_timestamp(row.get("extracted_at"), field="extracted_at") for row in rows]
    if any(extracted_at < downloaded_at for downloaded_at, extracted_at in zip(downloaded, extracted, strict=True)):
        raise CorpusManifestError("documents artifact contains extraction timestamps before download")
    tolerance = 1.0
    expected = manifest.freshness
    if abs((min(downloaded) - expected.source_observed_start).total_seconds()) > tolerance:
        raise CorpusManifestError("source observation start differs from the documents artifact")
    if abs((max(downloaded) - expected.source_observed_end).total_seconds()) > tolerance:
        raise CorpusManifestError("source observation end differs from the documents artifact")
    if abs((max(extracted) - expected.corpus_built_at).total_seconds()) > tolerance:
        raise CorpusManifestError("corpus build time differs from the documents artifact")
    if expected.slo_evidence_status == "measured":
        authoritative = [
            _timestamp(row.get("authoritative_published_at"), field="authoritative_published_at") for row in rows
        ]
        detected = [_timestamp(row.get("source_detected_at"), field="source_detected_at") for row in rows]
        published = [_timestamp(row.get("retrieval_published_at"), field="retrieval_published_at") for row in rows]
        for authoritative_at, detected_at, downloaded_at, extracted_at, published_at in zip(
            authoritative, detected, downloaded, extracted, published, strict=True
        ):
            if not authoritative_at <= detected_at <= downloaded_at <= extracted_at <= published_at:
                raise CorpusManifestError("documents artifact contains an invalid measured freshness sequence")
            if (detected_at - authoritative_at).total_seconds() > (expected.source_detection_slo_seconds or 0):
                raise CorpusManifestError("documents artifact exceeds the source-detection SLO")
            if (published_at - detected_at).total_seconds() > (expected.publication_slo_seconds or 0):
                raise CorpusManifestError("documents artifact exceeds the retrieval-publication SLO")


def load_and_validate_corpus_manifest(
    manifest_path: Path,
    *,
    corpus_root: Path | None = None,
    now: datetime | None = None,
    require_quantified_freshness: bool = False,
    require_measured_freshness: bool = False,
    require_verified_signature: bool = False,
    trusted_signing_key: Path | None = None,
) -> CorpusManifestValidation:
    """Load one manifest and verify its checksum, artifacts, and freshness.

    The function never repairs or rewrites a corpus.  Missing policy values are
    reported as warnings for research/local operation and become hard failures
    when a production-oriented requirement flag is supplied.
    """

    path = manifest_path.resolve()
    root = (corpus_root or path.parent).resolve()
    try:
        metadata = path.stat()
    except FileNotFoundError as exc:
        raise CorpusManifestError(f"required corpus manifest is missing: {CORPUS_MANIFEST_FILENAME}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MAX_MANIFEST_BYTES:
        raise CorpusManifestError("corpus manifest is not a bounded regular file")
    try:
        raw = load_bounded_release_yaml(
            path.read_text(encoding="utf-8"),
            maximum_bytes=_MAX_MANIFEST_BYTES,
        )
    except (OSError, UnicodeError, ReleaseYamlError) as exc:
        raise CorpusManifestError("corpus manifest YAML is invalid") from exc
    if not isinstance(raw, dict):
        raise CorpusManifestError("corpus manifest must be a mapping")

    try:
        manifest = CorpusScopeManifest.model_validate(raw)
    except (ValueError, RecursionError) as exc:
        raise CorpusManifestError("corpus manifest schema validation failed") from exc
    try:
        expected_checksum = canonical_manifest_sha256(raw)
    except (TypeError, ValueError, RecursionError) as exc:
        raise CorpusManifestError("corpus manifest canonicalization failed") from exc
    if manifest.integrity.manifest_sha256 != expected_checksum:
        raise CorpusManifestError("corpus manifest checksum mismatch")
    signing_key_fingerprint = _verify_manifest_signature(
        raw,
        manifest,
        corpus_root=root,
        trusted_signing_key=trusted_signing_key,
    )

    warnings: list[str] = [CORPUS_SCOPE_WARNING]
    current = now or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise CorpusManifestError("validation time must include a timezone")
    if manifest.freshness.scope_reviewed_at > current or manifest.freshness.corpus_built_at > current:
        raise CorpusManifestError("corpus manifest contains a future timestamp")

    freshness_values = (
        manifest.freshness.source_detection_slo_seconds,
        manifest.freshness.publication_slo_seconds,
        manifest.freshness.max_manifest_age_seconds,
    )
    if any(value is None for value in freshness_values):
        if require_quantified_freshness:
            raise CorpusManifestError("corpus freshness objectives are not quantified")
        warnings.append("Corpus freshness objectives are not yet quantified; 'immediate' is not a testable SLO.")
    elif current - manifest.freshness.corpus_built_at > timedelta(
        seconds=manifest.freshness.max_manifest_age_seconds or 0
    ):
        raise CorpusManifestError("corpus manifest is stale under its declared maximum age")
    if manifest.freshness.slo_evidence_status != "measured":
        if require_measured_freshness:
            raise CorpusManifestError("corpus freshness SLO compliance is not measured")
        warnings.append("Corpus freshness objectives, if declared, are not measured against per-document events.")

    if manifest.integrity.signature_status != "verified":
        if require_verified_signature:
            raise CorpusManifestError("corpus manifest signature is not verified")
        warnings.append("Corpus manifest checksum is verified, but no digital signature is configured.")

    for artifact in manifest.artifacts:
        artifact_path = _artifact_path(root, artifact.path)
        if artifact_path.stat().st_size != artifact.bytes:
            raise CorpusManifestError(f"corpus artifact size differs from the manifest: {artifact.path}")
        if _file_sha256(artifact_path) != artifact.sha256:
            raise CorpusManifestError(f"corpus artifact checksum differs from the manifest: {artifact.path}")
        records = _load_json_records(artifact_path, artifact) if artifact.records is not None else None
        if artifact.role == "documents":
            if records is None:
                records = _load_json_records(artifact_path, artifact)
            _verify_document_freshness(records, manifest)

    return CorpusManifestValidation(
        manifest=manifest,
        manifest_sha256=expected_checksum,
        signing_key_fingerprint_sha256=signing_key_fingerprint,
        warnings=tuple(warnings),
    )
