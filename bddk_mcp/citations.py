"""Versioned, reconstructable citations for normalized regulatory text.

Citation v1 is deliberately narrower than a general legal citation engine.  It
supports an exact range in normalized Markdown and keeps that coordinate system
distinct from true source-document pages.  A citation is emitted only when the
caller supplies a validated legal-version mapping and authoritative,
non-fixture artifact provenance.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bddk_mcp.quality.markdown_quality import sanitize_markdown_for_context
from bddk_mcp.regulatory.legal_versions import (
    AuthorityLevel,
    artifact_id_for,
    blob_id_for,
    evidence_id_for,
    instrument_id_for,
    legal_version_id_for,
    provision_id_for,
    validate_canonical_source_uri,
)
from bddk_mcp.regulatory.text_profile import PROVISION_BOUNDARY_WHITESPACE_V1

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_CITATION_ID_PATTERN = r"^cite_sha256_[0-9a-f]{64}$"
_INSTRUMENT_ID_PATTERN = r"^inst_sha256_[0-9a-f]{64}$"
_BLOB_ID_PATTERN = r"^blob_sha256_[0-9a-f]{64}$"
_ARTIFACT_ID_PATTERN = r"^art_sha256_[0-9a-f]{64}$"
_VERSION_ID_PATTERN = r"^ver_sha256_[0-9a-f]{64}$"
_PROVISION_ID_PATTERN = r"^prov_sha256_[0-9a-f]{64}$"

_SECTION_RETRIEVAL_PROFILE = {
    "profile": "bddk-mcp-exact-section-citation",
    "profile_version": 1,
    "coordinate_system": "unicode_codepoint_offsets_in_normalized_markdown_v1",
    "offset_unit": "python_unicode_code_points",
    "unicode_normalization": "none_preserve_stored_form",
    "line_endings": "preserve_normalized_document_storage",
    "storage_transform": "sanitize_markdown_for_storage_v1",
    "provision_transform": "strip_explicit_unicode_whitespace_v1",
    "render_transform": "sanitize_markdown_for_context_v1",
    "render_max_line_length": 1000,
}
# Explicitly pin the code points accepted at provision boundaries.  This is
# Python's historical whitespace set for the supported runtimes, expressed as
# data so a future Unicode database update cannot silently change Citation v1.
_QUALITY_FLAG_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CitationQuality(_StrictFrozenModel):
    """Quality state copied from the retrieval result at citation time."""

    label: Literal["clean", "warning", "fail", "unknown"]
    flags: tuple[str, ...] = Field(default=(), max_length=32)
    warning: str | None = Field(default=None, max_length=500)

    @field_validator("flags")
    @classmethod
    def _stable_unique_flags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not _QUALITY_FLAG_RE.fullmatch(item) for item in value):
            raise ValueError("citation quality flags must be bounded stable identifiers")
        if len(value) != len(set(value)) or value != tuple(sorted(value)):
            raise ValueError("citation quality flags must be unique and canonically ordered")
        return value


class NormalizedTextRange(_StrictFrozenModel):
    """Half-open Unicode-codepoint offsets in normalized Markdown, not pages."""

    kind: Literal["normalized_text_range"] = "normalized_text_range"
    coordinate_system: Literal["unicode_codepoint_offsets_in_normalized_markdown_v1"] = (
        "unicode_codepoint_offsets_in_normalized_markdown_v1"
    )
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    normalized_range_sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _range_is_forward(self) -> NormalizedTextRange:
        if self.end_char <= self.start_char:
            raise ValueError("normalized citation range must be non-empty and forward")
        return self


class TrustedCitationContext(_StrictFrozenModel):
    """Complete context produced by the reviewed v0004 evidence join."""

    instrument_id: str = Field(pattern=_INSTRUMENT_ID_PATTERN)
    instrument_jurisdiction: str = Field(min_length=2, max_length=50)
    instrument_authority_code: str = Field(min_length=1, max_length=100)
    instrument_identity_key: str = Field(min_length=1, max_length=300)
    legal_version_id: str = Field(pattern=_VERSION_ID_PATTERN)
    legal_version_key: str = Field(min_length=1, max_length=300)
    legal_validation_state: Literal["validated"] = "validated"
    legal_validation_record_sha256: str = Field(pattern=_SHA256_PATTERN)
    provision_validation_state: Literal["validated"] = "validated"
    provision_validation_record_sha256: str = Field(pattern=_SHA256_PATTERN)
    artifact_id: str = Field(pattern=_ARTIFACT_ID_PATTERN)
    artifact_blob_id: str = Field(pattern=_BLOB_ID_PATTERN)
    artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    artifact_fixture_only: Literal[False] = False
    evidence_authority: Literal["authoritative"] = "authoritative"
    source_url: str = Field(min_length=1, max_length=2_000)
    artifact_retrieved_at: datetime
    source_document_id: str = Field(min_length=1, max_length=500)
    normalized_document_sha256: str = Field(pattern=_SHA256_PATTERN)
    evidence_id: str = Field(pattern=r"^evid_sha256_[0-9a-f]{64}$")
    evidence_locator: str = Field(min_length=1, max_length=1_000)
    evidence_statement_sha256: str = Field(pattern=_SHA256_PATTERN)
    provision_id: str = Field(pattern=_PROVISION_ID_PATTERN)
    provision_kind: str = Field(min_length=1, max_length=100)
    provision_path: str = Field(min_length=1, max_length=500)
    provision_text_sha256: str = Field(pattern=_SHA256_PATTERN)
    locator: NormalizedTextRange
    excerpt_sha256: str = Field(pattern=_SHA256_PATTERN)
    excerpt_length: int = Field(ge=1, le=30_000)
    provision_transform: Literal["strip_explicit_unicode_whitespace_v1"] = "strip_explicit_unicode_whitespace_v1"
    render_transform: Literal["sanitize_markdown_for_context_v1"] = "sanitize_markdown_for_context_v1"
    retrieval_profile_sha256: str = Field(pattern=_SHA256_PATTERN)
    quality: CitationQuality

    @field_validator("source_url")
    @classmethod
    def _authoritative_https_url(cls, value: str) -> str:
        return validate_canonical_source_uri(value)

    @field_validator("artifact_retrieved_at")
    @classmethod
    def _timezone_is_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("citation timestamps must include a UTC offset")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _trusted_relationships_are_closed(self) -> TrustedCitationContext:
        if self.retrieval_profile_sha256 != section_retrieval_profile_sha256():
            raise ValueError("citation retrieval profile is not supported by this verifier")
        if self.instrument_id != instrument_id_for(
            jurisdiction=self.instrument_jurisdiction,
            authority_code=self.instrument_authority_code,
            identity_key=self.instrument_identity_key,
        ):
            raise ValueError("citation instrument identity does not match its canonical components")
        if self.legal_version_id != legal_version_id_for(
            instrument_id=self.instrument_id,
            version_key=self.legal_version_key,
            legal_text_sha256=self.normalized_document_sha256,
        ):
            raise ValueError("citation legal-version identity does not match its canonical components")
        if self.artifact_blob_id != blob_id_for(content_sha256=self.artifact_sha256):
            raise ValueError("citation blob identity does not match its content hash")
        if self.artifact_id != artifact_id_for(
            blob_id=self.artifact_blob_id,
            canonical_uri=self.source_url,
            retrieved_at=self.artifact_retrieved_at,
        ):
            raise ValueError("citation artifact identity does not match its acquisition components")
        if self.provision_id != provision_id_for(
            instrument_id=self.instrument_id,
            kind=self.provision_kind,
            canonical_path=self.provision_path,
        ):
            raise ValueError("citation provision identity does not match its canonical components")
        if self.evidence_statement_sha256 != self.provision_text_sha256:
            raise ValueError("citation evidence statement and provision hashes differ")
        if self.evidence_id != evidence_id_for(
            artifact_id=self.artifact_id,
            locator=self.evidence_locator,
            statement_sha256=self.evidence_statement_sha256,
            authority_level=AuthorityLevel.AUTHORITATIVE,
        ):
            raise ValueError("citation evidence identity does not match its canonical components")
        return self


class CitationV1(TrustedCitationContext):
    """A content-addressed citation to one validated provision occurrence."""

    schema_version: Literal["1.0"] = "1.0"
    citation_id: str = Field(pattern=_CITATION_ID_PATTERN)
    generated_at: datetime

    @field_validator("generated_at")
    @classmethod
    def _generated_time_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("citation timestamps must include a UTC offset")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _citation_identity_matches(self) -> CitationV1:
        if self.generated_at < self.artifact_retrieved_at:
            raise ValueError("citation generation cannot predate artifact retrieval")
        if self.citation_id != citation_id_for(self):
            raise ValueError("citation_id does not match its immutable evidence fields")
        return self


ExpectedCitationIdentity = TrustedCitationContext


class CitationVerificationResult(_StrictFrozenModel):
    """Deterministic verification outcome without returning source text."""

    valid: bool
    failure_codes: tuple[str, ...] = ()
    citation_id: str

    @model_validator(mode="after")
    def _result_shape_is_consistent(self) -> CitationVerificationResult:
        if self.valid == bool(self.failure_codes):
            raise ValueError("valid citations have no failure codes and invalid citations have at least one")
        return self


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _strip_provision_boundaries_v1(value: str) -> str:
    return value.strip(PROVISION_BOUNDARY_WHITESPACE_V1)


def section_retrieval_profile_sha256() -> str:
    """Return the immutable profile for the exact-section Citation v1 path."""

    payload = json.dumps(_SECTION_RETRIEVAL_PROFILE, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _citation_identity_mapping(citation: CitationV1 | dict[str, object]) -> dict[str, object]:
    if isinstance(citation, CitationV1):
        data = citation.model_dump(mode="json")
    else:
        data = dict(citation)
    data.pop("citation_id", None)
    data.pop("generated_at", None)
    quality = data.get("quality")
    if isinstance(quality, dict):
        quality.pop("warning", None)
    return _json_compatible(data)


def _json_compatible(value):
    """Match Pydantic's JSON-mode datetime encoding before model validation."""

    if isinstance(value, datetime):
        encoded = value.isoformat()
        return encoded.removesuffix("+00:00") + "Z" if encoded.endswith("+00:00") else encoded
    if isinstance(value, dict):
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def citation_id_for(citation: CitationV1 | dict[str, object]) -> str:
    """Build a stable identity from evidence fields, excluding request time."""

    payload = json.dumps(
        _citation_identity_mapping(citation),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"cite_sha256_{hashlib.sha256(payload).hexdigest()}"


def build_normalized_range_citation(
    *,
    trusted: TrustedCitationContext,
    provision_text: str,
    normalized_source_range: str,
    rendered_excerpt: str,
    generated_at: datetime | None = None,
) -> CitationV1:
    """Construct Citation v1 only from a complete, independently checked context."""

    expected_provision = _strip_provision_boundaries_v1(normalized_source_range)
    if provision_text != expected_provision:
        raise ValueError("stored provision cannot be reconstructed from its normalized range")
    expected_excerpt = sanitize_markdown_for_context(expected_provision)
    if rendered_excerpt != expected_excerpt:
        raise ValueError("returned excerpt cannot be reconstructed from the normalized range")
    if trusted.locator.end_char - trusted.locator.start_char != len(normalized_source_range):
        raise ValueError("normalized source range length does not match its offsets")
    comparisons = (
        (trusted.provision_text_sha256, _sha256_text(provision_text)),
        (trusted.evidence_statement_sha256, _sha256_text(provision_text)),
        (trusted.locator.normalized_range_sha256, _sha256_text(normalized_source_range)),
        (trusted.excerpt_sha256, _sha256_text(rendered_excerpt)),
        (trusted.excerpt_length, len(rendered_excerpt)),
    )
    if any(observed != expected for observed, expected in comparisons):
        raise ValueError("trusted citation context differs from reconstructed evidence")

    data: dict[str, object] = trusted.model_dump(mode="json")
    data.update({"schema_version": "1.0", "generated_at": generated_at or datetime.now(UTC)})
    data["citation_id"] = citation_id_for(data)
    return CitationV1.model_validate(data)


def render_normalized_range_excerpt(citation: CitationV1, normalized_document: str) -> str:
    """Reconstruct the public excerpt for a Citation v1 normalized range."""

    start = citation.locator.start_char
    end = citation.locator.end_char
    if end > len(normalized_document):
        raise ValueError("normalized citation range is outside the document")
    source_range = normalized_document[start:end]
    return sanitize_markdown_for_context(_strip_provision_boundaries_v1(source_range))


def verify_normalized_range_citation(
    citation: CitationV1,
    *,
    normalized_document: str,
    rendered_excerpt: str,
    expected: TrustedCitationContext,
) -> CitationVerificationResult:
    """Reconstruct a normalized-range citation and fail on any mismatch."""

    failures: list[str] = []
    for field in TrustedCitationContext.model_fields:
        if getattr(citation, field) != getattr(expected, field):
            failures.append(f"{field}_mismatch")

    if citation.citation_id != citation_id_for(citation):
        failures.append("citation_id_mismatch")
    if _sha256_text(normalized_document) != citation.normalized_document_sha256:
        failures.append("normalized_document_sha256_mismatch")

    start = citation.locator.start_char
    end = citation.locator.end_char
    if end > len(normalized_document):
        failures.append("normalized_range_out_of_bounds")
    else:
        source_range = normalized_document[start:end]
        if _sha256_text(source_range) != citation.locator.normalized_range_sha256:
            failures.append("normalized_range_sha256_mismatch")
        provision_text = _strip_provision_boundaries_v1(source_range)
        if _sha256_text(provision_text) != citation.provision_text_sha256:
            failures.append("provision_text_sha256_mismatch")
        if sanitize_markdown_for_context(provision_text) != rendered_excerpt:
            failures.append("excerpt_reconstruction_mismatch")

    if len(rendered_excerpt) != citation.excerpt_length:
        failures.append("excerpt_length_mismatch")
    if _sha256_text(rendered_excerpt) != citation.excerpt_sha256:
        failures.append("excerpt_sha256_mismatch")

    unique_failures = tuple(dict.fromkeys(failures))
    return CitationVerificationResult(
        valid=not unique_failures,
        failure_codes=unique_failures,
        citation_id=citation.citation_id,
    )
