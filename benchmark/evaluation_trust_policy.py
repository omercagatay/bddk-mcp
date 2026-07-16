"""Bank-signed authorization policy for expert-evaluation release evidence.

The evaluation artifacts prove cryptographic consistency.  This module adds a
separate authority boundary: an exact policy file signed by a configured
Ed25519 root intended for bank control binds operational signer fingerprints to
roles and opaque owner identities, authorizes separate legal-source reviewer
owners, constrains validity/revocation, declares deployment scope, and approves
one legal-release checkpoint head. Runtime verification cannot by itself prove
who controls that configured root or who performed a declared review.

Owner IDs and labels are not copied into public evidence. Only bounded policy
identifiers, fingerprints, booleans, and aggregate counts are emitted.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, field_validator, model_validator

from bddk_mcp.release_yaml import ReleaseYamlError, load_bounded_release_yaml
from benchmark.signing import ed25519_public_key_fingerprint_sha256

_MAX_POLICY_BYTES = 256 * 1024
_MAX_PUBLIC_KEY_BYTES = 16 * 1024
_MAX_SIGNATURE_BYTES = 1_024
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._:@/-]{2,127}$"
_ARTIFACT_ID_PATTERN = r"^art_sha256_[0-9a-f]{64}$"

EvaluationSignerRole = Literal[
    "corpus_scope_approver",
    "expert_dataset_owner",
    "legal_curator",
    "legal_release_certifier",
]


class EvaluationTrustPolicyError(ValueError):
    """Raised when bank trust-policy verification or authorization fails."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("trust-policy timestamps must include a timezone")
    return value.astimezone(UTC)


def _reject_numeric_timestamp(value: object) -> object:
    if isinstance(value, (bool, int, float)):
        raise ValueError("trust-policy timestamps must be ISO text or datetime values")
    if isinstance(value, str) and re.fullmatch(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
        value.strip(),
    ):
        raise ValueError("trust-policy timestamps must not be numeric strings")
    return value


_PolicyTimestamp = Annotated[
    datetime,
    BeforeValidator(_reject_numeric_timestamp),
    AfterValidator(_aware_utc),
]


def _audit_label(value: str) -> str:
    if value != value.strip() or any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError("trust-policy labels must be trimmed printable text")
    return value


class AuthorizedSigner(_StrictModel):
    """One canonical operational key authorization owned by one bank role."""

    key_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    owner_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    owner_label: str = Field(min_length=1, max_length=200)
    key_fingerprint_sha256: str = Field(pattern=_SHA256_PATTERN)
    valid_from: _PolicyTimestamp
    valid_until: _PolicyTimestamp
    replaces_key_id: str | None = Field(default=None, pattern=_IDENTIFIER_PATTERN)

    _validate_owner_label = field_validator("owner_label")(_audit_label)

    @model_validator(mode="after")
    def _ordered_window(self) -> AuthorizedSigner:
        if self.valid_from >= self.valid_until:
            raise ValueError("trust-policy signer validity window is empty")
        return self


class EvaluationSignerRegistry(_StrictModel):
    corpus_scope_approver: tuple[AuthorizedSigner, ...] = Field(min_length=1, max_length=32)
    expert_dataset_owner: tuple[AuthorizedSigner, ...] = Field(min_length=1, max_length=32)
    legal_curator: tuple[AuthorizedSigner, ...] = Field(min_length=1, max_length=32)
    legal_release_certifier: tuple[AuthorizedSigner, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def _separated_and_canonical(self) -> EvaluationSignerRegistry:
        fingerprint_roles: dict[str, str] = {}
        owner_roles: dict[str, str] = {}
        owner_labels: dict[str, str] = {}
        all_key_ids: set[str] = set()
        for role in (
            "corpus_scope_approver",
            "expert_dataset_owner",
            "legal_curator",
            "legal_release_certifier",
        ):
            entries = getattr(self, role)
            order = tuple(entry.key_id for entry in entries)
            if len(order) != len(set(order)) or order != tuple(sorted(order)):
                raise ValueError("trust-policy signer entries must be unique and canonically ordered")
            if all_key_ids & set(order):
                raise ValueError("trust-policy key IDs must be globally unique")
            all_key_ids.update(order)
            by_key_id = {entry.key_id: entry for entry in entries}
            roots = tuple(entry for entry in entries if entry.replaces_key_id is None)
            if len(roots) != 1:
                raise ValueError("each trust-policy role must have one key-rotation root")
            for entry in entries:
                if entry.replaces_key_id is not None:
                    predecessor = by_key_id.get(entry.replaces_key_id)
                    if predecessor is None or predecessor.key_id == entry.key_id:
                        raise ValueError("trust-policy key rotation references an unknown predecessor")
                    if entry.valid_from < predecessor.valid_from:
                        raise ValueError("trust-policy replacement key predates its predecessor")
                if entry.key_fingerprint_sha256 in fingerprint_roles:
                    raise ValueError("trust-policy operational key fingerprints must be globally unique")
                fingerprint_roles[entry.key_fingerprint_sha256] = role
                previous_owner_role = owner_roles.setdefault(entry.owner_id, role)
                if previous_owner_role != role:
                    raise ValueError("one owner cannot control multiple separated roles")
                previous_owner_label = owner_labels.setdefault(entry.owner_id, entry.owner_label)
                if previous_owner_label != entry.owner_label:
                    raise ValueError("one trust-policy owner ID cannot have multiple labels")
            for entry in entries:
                seen: set[str] = set()
                cursor = entry
                while cursor.replaces_key_id is not None:
                    if cursor.key_id in seen:
                        raise ValueError("trust-policy key rotation contains a cycle")
                    seen.add(cursor.key_id)
                    cursor = by_key_id[cursor.replaces_key_id]
                if cursor.key_id != roots[0].key_id:
                    raise ValueError("trust-policy key rotation is disconnected")
        return self

    def entries_for(self, role: EvaluationSignerRole) -> tuple[AuthorizedSigner, ...]:
        return getattr(self, role)

    def all_entries(self) -> tuple[AuthorizedSigner, ...]:
        return tuple(
            entry
            for role in (
                "corpus_scope_approver",
                "expert_dataset_owner",
                "legal_curator",
                "legal_release_certifier",
            )
            for entry in self.entries_for(role)
        )


class AuthorizedLegalSourceReviewer(_StrictModel):
    owner_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    owner_label: str = Field(min_length=1, max_length=200)
    role: Literal["legal_source_reviewer"]
    valid_from: _PolicyTimestamp
    valid_until: _PolicyTimestamp

    _validate_owner_label = field_validator("owner_label")(_audit_label)

    @model_validator(mode="after")
    def _ordered_window(self) -> AuthorizedLegalSourceReviewer:
        if self.valid_from >= self.valid_until:
            raise ValueError("legal source reviewer validity window is empty")
        return self


class ReviewerRevocation(_StrictModel):
    owner_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    revoked_at: _PolicyTimestamp
    reason_code: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")


class KeyRevocation(_StrictModel):
    key_fingerprint_sha256: str = Field(pattern=_SHA256_PATTERN)
    revoked_at: _PolicyTimestamp
    reason_code: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")


class CheckpointRevocation(_StrictModel):
    checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    revoked_at: _PolicyTimestamp
    reason_code: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")


class ApprovedRelease(_StrictModel):
    dataset_sha256: str = Field(pattern=_SHA256_PATTERN)
    corpus_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    legal_pack_sha256: str = Field(pattern=_SHA256_PATTERN)
    legal_attestation_sha256: str = Field(pattern=_SHA256_PATTERN)
    legal_release_checkpoint_sha256: str = Field(pattern=_SHA256_PATTERN)
    approved_at: _PolicyTimestamp
    approval_record_id: str = Field(pattern=_IDENTIFIER_PATTERN)


class EvaluationTrustPolicy(_StrictModel):
    """Closed schema signed verbatim by the bank policy authority."""

    schema_version: Literal[2]
    purpose: Literal["bddk_mcp_expert_evaluation_release"]
    policy_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    policy_version: int = Field(strict=True, ge=1)
    supersedes_policy_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    organization_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    environment_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    issuer_id: str = Field(pattern=_IDENTIFIER_PATTERN)
    issuer_label: str = Field(min_length=1, max_length=200)
    issuer_role: Literal["bank_trust_policy_authority"]
    deployment_scope: Literal["bank_production"]
    issued_at: _PolicyTimestamp
    valid_from: _PolicyTimestamp
    valid_until: _PolicyTimestamp
    approved_release: ApprovedRelease
    authorized_signers: EvaluationSignerRegistry
    authorized_legal_source_reviewers: tuple[AuthorizedLegalSourceReviewer, ...] = Field(min_length=1, max_length=256)
    revoked_keys: tuple[KeyRevocation, ...] = Field(default=(), max_length=256)
    revoked_legal_source_reviewers: tuple[ReviewerRevocation, ...] = Field(default=(), max_length=256)
    revoked_legal_release_checkpoints: tuple[CheckpointRevocation, ...] = Field(default=(), max_length=256)

    _validate_issuer_label = field_validator("issuer_label")(_audit_label)

    @field_validator("schema_version", mode="before")
    @classmethod
    def _strict_schema_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("trust-policy schema version must be an integer")
        return value

    @model_validator(mode="after")
    def _closed_timeline_and_revocations(self) -> EvaluationTrustPolicy:
        if not self.issued_at <= self.valid_from < self.valid_until:
            raise ValueError("trust-policy validity timeline is invalid")
        if not self.valid_from <= self.approved_release.approved_at < self.valid_until:
            raise ValueError("checkpoint approval is outside the policy timeline")
        if (self.policy_version == 1) != (self.supersedes_policy_sha256 is None):
            raise ValueError("trust-policy predecessor is inconsistent with its version")

        key_order = tuple(
            (item.key_fingerprint_sha256, item.revoked_at, item.reason_code) for item in self.revoked_keys
        )
        if len({item[0] for item in key_order}) != len(key_order) or key_order != tuple(sorted(key_order)):
            raise ValueError("trust-policy key revocations must be unique and canonically ordered")
        known_fingerprints = {entry.key_fingerprint_sha256 for entry in self.authorized_signers.all_entries()}
        if any(item.key_fingerprint_sha256 not in known_fingerprints for item in self.revoked_keys):
            raise ValueError("trust-policy key revocation references an unknown signer")
        if self.issuer_id in {entry.owner_id for entry in self.authorized_signers.all_entries()}:
            raise ValueError("trust-policy authority must be separate from operational signer owners")
        reviewer_order = tuple(item.owner_id for item in self.authorized_legal_source_reviewers)
        if len(reviewer_order) != len(set(reviewer_order)) or reviewer_order != tuple(sorted(reviewer_order)):
            raise ValueError("legal source reviewers must be unique and canonically ordered")
        operational_owners = {entry.owner_id for entry in self.authorized_signers.all_entries()}
        if self.issuer_id in reviewer_order or operational_owners & set(reviewer_order):
            raise ValueError("legal source reviewers must be separate from policy and signer owners")
        reviewer_revocation_order = tuple(
            (item.owner_id, item.revoked_at, item.reason_code) for item in self.revoked_legal_source_reviewers
        )
        if (
            len({item[0] for item in reviewer_revocation_order}) != len(reviewer_revocation_order)
            or reviewer_revocation_order != tuple(sorted(reviewer_revocation_order))
            or not {item[0] for item in reviewer_revocation_order} <= set(reviewer_order)
        ):
            raise ValueError("legal source reviewer revocations are invalid or non-canonical")
        checkpoint_order = tuple(
            (item.checkpoint_sha256, item.revoked_at, item.reason_code)
            for item in self.revoked_legal_release_checkpoints
        )
        if len({item[0] for item in checkpoint_order}) != len(checkpoint_order) or checkpoint_order != tuple(
            sorted(checkpoint_order)
        ):
            raise ValueError("trust-policy checkpoint revocations must be unique and canonically ordered")
        return self


@dataclass(frozen=True, slots=True)
class SignedEvaluationTrustPolicy:
    """Verified policy plus the minimum safe evidence needed by the preflight."""

    policy: EvaluationTrustPolicy
    policy_sha256: str
    policy_signing_key_fingerprint_sha256: str


@dataclass(frozen=True, slots=True)
class EvaluationTrustAuthorization:
    policy_id: str
    policy_version: int
    policy_sha256: str
    policy_signing_key_fingerprint_sha256: str
    policy_valid_until: datetime
    approved_checkpoint_sha256: str
    authorized_owner_count: int
    authorized_reviewer_count: int
    policy_bound_legal_source_review_count: int


def _bounded_regular_bytes(path: Path, *, label: str, maximum_bytes: int) -> bytes:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise EvaluationTrustPolicyError(f"{label} is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= maximum_bytes:
            raise EvaluationTrustPolicyError(f"{label} is not a bounded regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read(maximum_bytes + 1)
        if len(payload) != metadata.st_size:
            raise EvaluationTrustPolicyError(f"{label} changed while it was read")
        return payload
    except OSError as exc:
        raise EvaluationTrustPolicyError(f"{label} could not be read") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _current_utc(current: datetime) -> datetime:
    if current.tzinfo is None or current.utcoffset() is None:
        raise EvaluationTrustPolicyError("trust-policy validation time must include a timezone")
    return current.astimezone(UTC)


def load_signed_evaluation_trust_policy(
    policy_path: str | Path,
    signature_path: str | Path,
    trusted_policy_signing_key: str | Path,
    *,
    current: datetime,
    expected_organization_id: str | None = None,
    expected_environment_id: str | None = None,
    expected_deployment_scope: str | None = None,
) -> SignedEvaluationTrustPolicy:
    """Verify an exact signed policy and its current global validity."""

    now = _current_utc(current)
    policy_bytes = _bounded_regular_bytes(
        Path(policy_path), label="evaluation trust policy", maximum_bytes=_MAX_POLICY_BYTES
    )
    signature = _bounded_regular_bytes(
        Path(signature_path),
        label="evaluation trust-policy detached signature",
        maximum_bytes=_MAX_SIGNATURE_BYTES,
    )
    key_bytes = _bounded_regular_bytes(
        Path(trusted_policy_signing_key),
        label="trusted evaluation policy signing key",
        maximum_bytes=_MAX_PUBLIC_KEY_BYTES,
    )
    try:
        public_key = serialization.load_pem_public_key(key_bytes)
        if not isinstance(public_key, Ed25519PublicKey):
            raise ValueError("unsupported public key type")
        public_key.verify(signature, policy_bytes)
    except (InvalidSignature, TypeError, ValueError):
        raise EvaluationTrustPolicyError("evaluation trust-policy signature verification failed") from None

    try:
        raw = load_bounded_release_yaml(policy_bytes, maximum_bytes=_MAX_POLICY_BYTES)
        if not isinstance(raw, dict):
            raise ValueError("policy is not an object")
        policy = EvaluationTrustPolicy.model_validate(raw)
    except (ReleaseYamlError, ValueError, RecursionError):
        raise EvaluationTrustPolicyError("evaluation trust-policy schema validation failed") from None

    if policy.issued_at > now:
        raise EvaluationTrustPolicyError("evaluation trust policy is from the future")
    if not policy.valid_from <= now < policy.valid_until:
        raise EvaluationTrustPolicyError("evaluation trust policy is not currently valid")
    if policy.approved_release.approved_at > now:
        raise EvaluationTrustPolicyError("evaluation trust-policy approval is from the future")
    expected_scope = (
        expected_organization_id,
        expected_environment_id,
        expected_deployment_scope,
    )
    if any(value is not None for value in expected_scope):
        if not all(value is not None for value in expected_scope):
            raise EvaluationTrustPolicyError("expected evaluation trust-policy scope is incomplete")
        if expected_scope != (
            policy.organization_id,
            policy.environment_id,
            policy.deployment_scope,
        ):
            raise EvaluationTrustPolicyError("evaluation trust policy is for a different deployment scope")

    root_fingerprint = ed25519_public_key_fingerprint_sha256(public_key)
    if any(entry.key_fingerprint_sha256 == root_fingerprint for entry in policy.authorized_signers.all_entries()):
        raise EvaluationTrustPolicyError("policy root cannot also be an operational evaluation signer")

    return SignedEvaluationTrustPolicy(
        policy=policy,
        policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        policy_signing_key_fingerprint_sha256=root_fingerprint,
    )


def _authorized_entry(
    signed_policy: SignedEvaluationTrustPolicy,
    *,
    role: EvaluationSignerRole,
    fingerprint: str,
    authorized_at: datetime,
    current: datetime,
) -> AuthorizedSigner:
    if re.fullmatch(_SHA256_PATTERN, fingerprint) is None:
        raise EvaluationTrustPolicyError("evaluation signer fingerprint is invalid")
    event_time = _current_utc(authorized_at)
    if event_time > current:
        raise EvaluationTrustPolicyError("evaluation signer authorization event is from the future")
    matches = tuple(
        entry
        for entry in signed_policy.policy.authorized_signers.entries_for(role)
        if entry.key_fingerprint_sha256 == fingerprint and entry.valid_from <= event_time < entry.valid_until
    )
    if len(matches) != 1:
        raise EvaluationTrustPolicyError("evaluation signer is not authorized for its role and time")
    for revocation in signed_policy.policy.revoked_keys:
        if revocation.key_fingerprint_sha256 == fingerprint and revocation.revoked_at <= current:
            raise EvaluationTrustPolicyError("evaluation signer has been revoked")
    return matches[0]


def _require_forward_key_rotation(
    entries: Sequence[AuthorizedSigner],
    registry: EvaluationSignerRegistry,
) -> None:
    by_key_id = {entry.key_id: entry for entry in registry.legal_release_certifier}
    previous = entries[0]
    for current in entries[1:]:
        if current.key_id == previous.key_id:
            continue
        cursor = current
        seen: set[str] = set()
        while cursor.replaces_key_id is not None and cursor.replaces_key_id != previous.key_id:
            if cursor.key_id in seen:
                raise EvaluationTrustPolicyError("legal release key rotation contains a cycle")
            seen.add(cursor.key_id)
            cursor = by_key_id[cursor.replaces_key_id]
        if cursor.replaces_key_id != previous.key_id:
            raise EvaluationTrustPolicyError("legal release checkpoint signers do not follow approved rotation")
        previous = current


def authorize_evaluation_trust_chain(
    signed_policy: SignedEvaluationTrustPolicy,
    *,
    corpus_signer_fingerprint_sha256: str,
    corpus_authorized_at: datetime,
    dataset_signer_fingerprint_sha256: str,
    dataset_authorized_at: datetime,
    legal_curator_fingerprint_sha256: str,
    legal_curator_authorized_at: datetime,
    legal_release_signer_fingerprint_sha256: str,
    legal_release_authorized_at: datetime,
    legal_release_chain_signers: Sequence[tuple[str, datetime, str]],
    legal_release_configured_key_fingerprints_sha256: Sequence[str],
    legal_source_reviews: Sequence[tuple[str, str, str | None, datetime, int]],
    dataset_sha256: str,
    corpus_manifest_sha256: str,
    legal_pack_sha256: str,
    legal_attestation_sha256: str,
    legal_release_checkpoint_sha256: str,
    current: datetime,
) -> EvaluationTrustAuthorization:
    """Authorize the four observed roles and the exact approved release head."""

    now = _current_utc(current)
    policy = signed_policy.policy
    if not policy.valid_from <= now < policy.valid_until:
        raise EvaluationTrustPolicyError("evaluation trust policy expired during validation")
    approved = policy.approved_release
    observed_release = (
        dataset_sha256,
        corpus_manifest_sha256,
        legal_pack_sha256,
        legal_attestation_sha256,
        legal_release_checkpoint_sha256,
    )
    approved_release = (
        approved.dataset_sha256,
        approved.corpus_manifest_sha256,
        approved.legal_pack_sha256,
        approved.legal_attestation_sha256,
        approved.legal_release_checkpoint_sha256,
    )
    if observed_release != approved_release:
        raise EvaluationTrustPolicyError("evaluation artifacts differ from the policy-approved release")
    if not legal_release_chain_signers:
        raise EvaluationTrustPolicyError("legal release signer history is empty")
    checkpoint_ids = tuple(item[2] for item in legal_release_chain_signers)
    checkpoint_times = tuple(_current_utc(item[1]) for item in legal_release_chain_signers)
    checkpoint_time_by_id = dict(zip(checkpoint_ids, checkpoint_times, strict=True))
    if (
        len(checkpoint_ids) != len(set(checkpoint_ids))
        or checkpoint_times != tuple(sorted(checkpoint_times))
        or legal_release_chain_signers[-1][0] != legal_release_signer_fingerprint_sha256
        or checkpoint_times[-1] != _current_utc(legal_release_authorized_at)
        or checkpoint_ids[-1] != legal_release_checkpoint_sha256
    ):
        raise EvaluationTrustPolicyError("legal release signer history is inconsistent")
    for checkpoint_sha256 in checkpoint_ids:
        if re.fullmatch(_SHA256_PATTERN, checkpoint_sha256) is None:
            raise EvaluationTrustPolicyError("legal release signer history contains an invalid checkpoint")
        for revocation in policy.revoked_legal_release_checkpoints:
            if revocation.checkpoint_sha256 == checkpoint_sha256 and revocation.revoked_at <= now:
                raise EvaluationTrustPolicyError("legal release checkpoint has been revoked")
    role_inputs: tuple[tuple[EvaluationSignerRole, str, datetime], ...] = (
        ("corpus_scope_approver", corpus_signer_fingerprint_sha256, corpus_authorized_at),
        ("expert_dataset_owner", dataset_signer_fingerprint_sha256, dataset_authorized_at),
        ("legal_curator", legal_curator_fingerprint_sha256, legal_curator_authorized_at),
        ("legal_release_certifier", legal_release_signer_fingerprint_sha256, legal_release_authorized_at),
    )
    selected = tuple(
        _authorized_entry(
            signed_policy,
            role=role,
            fingerprint=fingerprint,
            authorized_at=authorized_at,
            current=now,
        )
        for role, fingerprint, authorized_at in role_inputs
    )
    owners = {entry.owner_id for entry in selected}
    if len(owners) != len(selected):
        raise EvaluationTrustPolicyError("evaluation signer owners are not separated across roles")
    historical_release_entries = tuple(
        _authorized_entry(
            signed_policy,
            role="legal_release_certifier",
            fingerprint=fingerprint,
            authorized_at=authorized_at,
            current=now,
        )
        for fingerprint, authorized_at, _checkpoint_sha256 in legal_release_chain_signers
    )
    _require_forward_key_rotation(historical_release_entries, policy.authorized_signers)
    owners.update(entry.owner_id for entry in historical_release_entries)
    configured_key_fingerprints = tuple(legal_release_configured_key_fingerprints_sha256)
    observed_release_key_fingerprints = {item[0] for item in legal_release_chain_signers}
    effective_revoked_key_fingerprints = {
        item.key_fingerprint_sha256 for item in policy.revoked_keys if item.revoked_at <= now
    }
    authorized_release_key_fingerprints = {
        entry.key_fingerprint_sha256 for entry in policy.authorized_signers.legal_release_certifier
    } - effective_revoked_key_fingerprints
    if (
        not configured_key_fingerprints
        or len(configured_key_fingerprints) != len(set(configured_key_fingerprints))
        or configured_key_fingerprints[0] != legal_release_signer_fingerprint_sha256
        or not observed_release_key_fingerprints <= set(configured_key_fingerprints)
        or not set(configured_key_fingerprints) <= authorized_release_key_fingerprints
    ):
        raise EvaluationTrustPolicyError("configured legal release keyring is not authorized by the policy")

    if not legal_source_reviews:
        raise EvaluationTrustPolicyError("legal source reviewer history is empty")
    review_identities = tuple((review[0], review[1]) for review in legal_source_reviews)
    if len(review_identities) != len(set(review_identities)):
        raise EvaluationTrustPolicyError("legal source reviewer history contains a duplicate artifact review")
    if {identity[0] for identity in review_identities} != set(checkpoint_ids):
        raise EvaluationTrustPolicyError("legal source reviewer history differs from the checkpoint chain")
    reviewer_owners: set[str] = set()
    review_times: list[datetime] = []
    reviewer_by_owner = {reviewer.owner_id: reviewer for reviewer in policy.authorized_legal_source_reviewers}
    for checkpoint_sha256, artifact_id, owner_id, reviewed_at, proof_schema_version in legal_source_reviews:
        if (
            re.fullmatch(_SHA256_PATTERN, checkpoint_sha256) is None
            or re.fullmatch(_ARTIFACT_ID_PATTERN, artifact_id) is None
        ):
            raise EvaluationTrustPolicyError("legal source reviewer history contains an invalid identity")
        if type(proof_schema_version) is not int or proof_schema_version != 2 or owner_id is None:
            raise EvaluationTrustPolicyError("policy authorization requires page-mapping proof v2")
        review_time = _current_utc(reviewed_at)
        checkpoint_time = checkpoint_time_by_id[checkpoint_sha256]
        if review_time > checkpoint_time or checkpoint_time > now:
            raise EvaluationTrustPolicyError("legal source review chronology is invalid")
        reviewer = reviewer_by_owner.get(owner_id)
        if reviewer is None or not reviewer.valid_from <= review_time < reviewer.valid_until:
            raise EvaluationTrustPolicyError("legal source reviewer is not authorized for the review time")
        if any(
            revocation.owner_id == owner_id and revocation.revoked_at <= now
            for revocation in policy.revoked_legal_source_reviewers
        ):
            raise EvaluationTrustPolicyError("legal source reviewer has been revoked")
        reviewer_owners.add(owner_id)
        review_times.append(review_time)

    artifact_authorization_times = (
        _current_utc(corpus_authorized_at),
        _current_utc(dataset_authorized_at),
        _current_utc(legal_curator_authorized_at),
        *checkpoint_times,
        *review_times,
    )
    if policy.approved_release.approved_at < max(artifact_authorization_times):
        raise EvaluationTrustPolicyError("trust-policy approval predates its approved release evidence")

    return EvaluationTrustAuthorization(
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
        policy_sha256=signed_policy.policy_sha256,
        policy_signing_key_fingerprint_sha256=signed_policy.policy_signing_key_fingerprint_sha256,
        policy_valid_until=policy.valid_until,
        approved_checkpoint_sha256=policy.approved_release.legal_release_checkpoint_sha256,
        authorized_owner_count=len(owners),
        authorized_reviewer_count=len(reviewer_owners),
        policy_bound_legal_source_review_count=len(legal_source_reviews),
    )


__all__ = (
    "ApprovedRelease",
    "AuthorizedLegalSourceReviewer",
    "AuthorizedSigner",
    "CheckpointRevocation",
    "EvaluationSignerRegistry",
    "EvaluationTrustAuthorization",
    "EvaluationTrustPolicy",
    "EvaluationTrustPolicyError",
    "KeyRevocation",
    "ReviewerRevocation",
    "SignedEvaluationTrustPolicy",
    "authorize_evaluation_trust_chain",
    "load_signed_evaluation_trust_policy",
)
