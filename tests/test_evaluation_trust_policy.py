"""Tests for the signed expert-evaluation trust-policy boundary."""

from __future__ import annotations

import hashlib
import json
import traceback
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import benchmark.release_preflight as release_preflight
from benchmark.evaluation_trust_policy import (
    EvaluationTrustPolicyError,
    authorize_evaluation_trust_chain,
    load_signed_evaluation_trust_policy,
)
from benchmark.release_preflight import ReleasePreflightInputs, run_release_preflight
from benchmark.signing import ed25519_public_key_fingerprint_sha256

NOW = datetime(2026, 7, 16, 12, tzinfo=UTC)
EVENT = datetime(2026, 7, 10, 12, tzinfo=UTC)
HASHES = {
    "dataset": "1" * 64,
    "corpus": "2" * 64,
    "pack": "3" * 64,
    "attestation": "4" * 64,
    "checkpoint": "5" * 64,
}
ARTIFACT_ID = "art_sha256_" + "a" * 64


def _public_pem(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def _fingerprint(key: Ed25519PrivateKey) -> str:
    return ed25519_public_key_fingerprint_sha256(key.public_key())


def _keys() -> dict[str, Ed25519PrivateKey]:
    return {
        role: Ed25519PrivateKey.generate()
        for role in ("root", "corpus", "dataset", "curator", "release", "release_old")
    }


def _entry(
    key_id: str,
    owner_id: str,
    key: Ed25519PrivateKey,
    *,
    valid_from: str = "2025-01-01T00:00:00Z",
    valid_until: str = "2027-01-01T00:00:00Z",
    replaces_key_id: str | None = None,
) -> dict:
    return {
        "key_id": key_id,
        "owner_id": owner_id,
        "owner_label": f"Display label for {owner_id}",
        "key_fingerprint_sha256": _fingerprint(key),
        "valid_from": valid_from,
        "valid_until": valid_until,
        "replaces_key_id": replaces_key_id,
    }


def _reviewer(owner_id: str, *, owner_label: str | None = None) -> dict:
    return {
        "owner_id": owner_id,
        "owner_label": owner_label or f"Display label for {owner_id}",
        "role": "legal_source_reviewer",
        "valid_from": "2025-01-01T00:00:00Z",
        "valid_until": "2027-01-01T00:00:00Z",
    }


def _policy(keys: dict[str, Ed25519PrivateKey]) -> dict:
    return {
        "schema_version": 2,
        "purpose": "bddk_mcp_expert_evaluation_release",
        "policy_id": "bank-evaluation-policy",
        "policy_version": 1,
        "supersedes_policy_sha256": None,
        "organization_id": "bank-org",
        "environment_id": "openshift-production",
        "issuer_id": "bank-policy-authority",
        "issuer_label": "Bank policy authority display label",
        "issuer_role": "bank_trust_policy_authority",
        "deployment_scope": "bank_production",
        "issued_at": "2026-07-01T00:00:00Z",
        "valid_from": "2026-07-01T00:00:00Z",
        "valid_until": "2026-08-01T00:00:00Z",
        "approved_release": {
            "dataset_sha256": HASHES["dataset"],
            "corpus_manifest_sha256": HASHES["corpus"],
            "legal_pack_sha256": HASHES["pack"],
            "legal_attestation_sha256": HASHES["attestation"],
            "legal_release_checkpoint_sha256": HASHES["checkpoint"],
            "approved_at": "2026-07-13T00:00:00Z",
            "approval_record_id": "change-record-123",
        },
        "authorized_signers": {
            "corpus_scope_approver": [_entry("corpus-key-v1", "corpus-owner", keys["corpus"])],
            "expert_dataset_owner": [_entry("dataset-key-v1", "dataset-owner", keys["dataset"])],
            "legal_curator": [_entry("curator-key-v1", "curator-owner", keys["curator"])],
            "legal_release_certifier": [_entry("release-key-v1", "release-owner", keys["release"])],
        },
        "authorized_legal_source_reviewers": [
            _reviewer("page-reviewer", owner_label="Legal source reviewer display label")
        ],
        "revoked_keys": [],
        "revoked_legal_source_reviewers": [],
        "revoked_legal_release_checkpoints": [],
    }


def _write_signed_policy(
    tmp_path: Path,
    policy: dict,
    root: Ed25519PrivateKey,
    *,
    policy_bytes: bytes | None = None,
) -> tuple[Path, Path, Path]:
    payload = policy_bytes or yaml.safe_dump(policy, sort_keys=False).encode("utf-8")
    policy_path = tmp_path / "policy.yml"
    signature_path = tmp_path / "policy.sig"
    key_path = tmp_path / "root.pem"
    policy_path.write_bytes(payload)
    signature_path.write_bytes(root.sign(payload))
    key_path.write_bytes(_public_pem(root))
    return policy_path, signature_path, key_path


def _load(tmp_path: Path, policy: dict, keys: dict[str, Ed25519PrivateKey]):
    paths = _write_signed_policy(tmp_path, policy, keys["root"])
    return load_signed_evaluation_trust_policy(*paths, current=NOW)


def _authorize(signed, keys: dict[str, Ed25519PrivateKey], **overrides):
    arguments = {
        "corpus_signer_fingerprint_sha256": _fingerprint(keys["corpus"]),
        "corpus_authorized_at": EVENT,
        "dataset_signer_fingerprint_sha256": _fingerprint(keys["dataset"]),
        "dataset_authorized_at": EVENT,
        "legal_curator_fingerprint_sha256": _fingerprint(keys["curator"]),
        "legal_curator_authorized_at": EVENT,
        "legal_release_signer_fingerprint_sha256": _fingerprint(keys["release"]),
        "legal_release_authorized_at": EVENT,
        "dataset_sha256": HASHES["dataset"],
        "corpus_manifest_sha256": HASHES["corpus"],
        "legal_pack_sha256": HASHES["pack"],
        "legal_attestation_sha256": HASHES["attestation"],
        "legal_release_checkpoint_sha256": HASHES["checkpoint"],
        "current": NOW,
    }
    arguments.update(overrides)
    arguments.setdefault(
        "legal_release_chain_signers",
        [
            (
                arguments["legal_release_signer_fingerprint_sha256"],
                arguments["legal_release_authorized_at"],
                arguments["legal_release_checkpoint_sha256"],
            )
        ],
    )
    arguments.setdefault(
        "legal_source_reviews",
        [
            (
                arguments["legal_release_checkpoint_sha256"],
                ARTIFACT_ID,
                "page-reviewer",
                arguments["legal_release_authorized_at"],
                2,
            )
        ],
    )
    arguments.setdefault(
        "legal_release_configured_key_fingerprints_sha256",
        tuple(
            dict.fromkeys(
                (
                    arguments["legal_release_signer_fingerprint_sha256"],
                    *(item[0] for item in arguments["legal_release_chain_signers"]),
                )
            )
        ),
    )
    return authorize_evaluation_trust_chain(signed, **arguments)


def test_signed_policy_authorizes_exact_release_without_exposing_owner_ids(tmp_path: Path) -> None:
    keys = _keys()
    policy = _policy(keys)
    signed = _load(tmp_path, policy, keys)

    authorization = _authorize(signed, keys)

    assert signed.policy.schema_version == 2
    assert authorization.policy_id == "bank-evaluation-policy"
    assert authorization.policy_version == 1
    assert authorization.authorized_owner_count == 4
    assert authorization.authorized_reviewer_count == 1
    assert authorization.policy_bound_legal_source_review_count == 1
    assert authorization.approved_checkpoint_sha256 == HASHES["checkpoint"]
    assert authorization.policy_sha256 == hashlib.sha256((tmp_path / "policy.yml").read_bytes()).hexdigest()
    assert authorization.policy_signing_key_fingerprint_sha256 == _fingerprint(keys["root"])
    assert "corpus-owner" not in repr(authorization)
    assert "dataset-owner" not in repr(authorization)
    assert "Display label" not in repr(authorization)


def test_review_counts_distinguish_observed_owners_from_artifact_reviews(tmp_path: Path) -> None:
    keys = _keys()
    policy = _policy(keys)
    policy["authorized_legal_source_reviewers"] = [
        _reviewer("a-reviewer"),
        _reviewer("page-reviewer"),
    ]
    signed = _load(tmp_path, policy, keys)

    authorization = _authorize(
        signed,
        keys,
        legal_source_reviews=[
            (HASHES["checkpoint"], "art_sha256_" + "a" * 64, "a-reviewer", EVENT, 2),
            (HASHES["checkpoint"], "art_sha256_" + "b" * 64, "page-reviewer", EVENT, 2),
        ],
    )

    assert authorization.authorized_reviewer_count == 2
    assert authorization.policy_bound_legal_source_review_count == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("valid_from", "2026-07-17T00:00:00Z"),
        ("valid_until", "2026-07-16T12:00:00Z"),
        ("issued_at", "2026-07-17T00:00:00Z"),
    ],
)
def test_policy_must_be_issued_and_current(tmp_path: Path, field: str, value: str) -> None:
    keys = _keys()
    policy = _policy(keys)
    policy[field] = value

    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, policy, keys)


def test_policy_approval_must_follow_validity_and_every_approved_artifact(tmp_path: Path) -> None:
    keys = _keys()
    policy = _policy(keys)
    policy["valid_from"] = "2026-07-14T00:00:00Z"
    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, policy, keys)

    signed = _load(tmp_path, _policy(keys), keys)
    with pytest.raises(EvaluationTrustPolicyError):
        _authorize(
            signed,
            keys,
            legal_release_authorized_at=datetime(2026, 7, 14, tzinfo=UTC),
        )


def test_future_authorization_events_fail_closed(tmp_path: Path) -> None:
    keys = _keys()
    signed = _load(tmp_path, _policy(keys), keys)
    future = datetime(2026, 7, 17, tzinfo=UTC)

    with pytest.raises(EvaluationTrustPolicyError):
        _authorize(signed, keys, dataset_authorized_at=future)


def test_policy_tamper_and_wrong_root_fail_signature_verification(tmp_path: Path) -> None:
    keys = _keys()
    policy_path, signature_path, key_path = _write_signed_policy(tmp_path, _policy(keys), keys["root"])
    policy_path.write_bytes(policy_path.read_bytes() + b"\n")

    with pytest.raises(EvaluationTrustPolicyError):
        load_signed_evaluation_trust_policy(policy_path, signature_path, key_path, current=NOW)

    policy_path, signature_path, _ = _write_signed_policy(tmp_path, _policy(keys), keys["root"])
    wrong_key = tmp_path / "wrong.pem"
    wrong_key.write_bytes(_public_pem(Ed25519PrivateKey.generate()))
    with pytest.raises(EvaluationTrustPolicyError):
        load_signed_evaluation_trust_policy(policy_path, signature_path, wrong_key, current=NOW)


def test_policy_rejects_ambiguous_yaml_and_symbolic_links(tmp_path: Path) -> None:
    keys = _keys()
    duplicate = b"schema_version: 1\nschema_version: 1\n"
    paths = _write_signed_policy(tmp_path, {}, keys["root"], policy_bytes=duplicate)
    with pytest.raises(EvaluationTrustPolicyError):
        load_signed_evaluation_trust_policy(*paths, current=NOW)

    actual = tmp_path / "actual.yml"
    actual.write_bytes(yaml.safe_dump(_policy(keys), sort_keys=False).encode())
    link = tmp_path / "linked.yml"
    link.symlink_to(actual)
    signature = tmp_path / "linked.sig"
    signature.write_bytes(keys["root"].sign(actual.read_bytes()))
    root = tmp_path / "linked-root.pem"
    root.write_bytes(_public_pem(keys["root"]))
    with pytest.raises(EvaluationTrustPolicyError):
        load_signed_evaluation_trust_policy(link, signature, root, current=NOW)


def test_policy_rejects_coercible_scalar_types(tmp_path: Path) -> None:
    keys = _keys()
    invalid_policies = []
    for field, value in (
        ("schema_version", 1),
        ("schema_version", True),
        ("schema_version", 2.0),
        ("policy_version", True),
        ("policy_version", 1.0),
        ("policy_version", "1"),
        ("issued_at", 1_752_000_000),
        ("issued_at", "1752000000"),
    ):
        policy = _policy(keys)
        policy[field] = value
        invalid_policies.append(policy)
    signer_timestamp = _policy(keys)
    signer_timestamp["authorized_signers"]["corpus_scope_approver"][0]["valid_from"] = 1_752_000_000
    invalid_policies.append(signer_timestamp)
    reviewer_timestamp = _policy(keys)
    reviewer_timestamp["authorized_legal_source_reviewers"][0]["valid_from"] = 1_752_000_000
    invalid_policies.append(reviewer_timestamp)
    reviewer_numeric_string = _policy(keys)
    reviewer_numeric_string["authorized_legal_source_reviewers"][0]["valid_from"] = "1752000000"
    invalid_policies.append(reviewer_numeric_string)
    approval_timestamp = _policy(keys)
    approval_timestamp["approved_release"]["approved_at"] = 1_752_000_000
    invalid_policies.append(approval_timestamp)

    for policy in invalid_policies:
        with pytest.raises(EvaluationTrustPolicyError):
            _load(tmp_path, policy, keys)


def test_policy_schema_failure_traceback_does_not_leak_owner_registry(tmp_path: Path) -> None:
    keys = _keys()
    policy = _policy(keys)
    policy["authorized_legal_source_reviewers"][0]["owner_label"] = "SENSITIVE REVIEWER LABEL "

    with pytest.raises(EvaluationTrustPolicyError) as captured:
        _load(tmp_path, policy, keys)

    rendered = "".join(traceback.format_exception(captured.value))
    assert captured.value.__cause__ is None
    assert "SENSITIVE REVIEWER LABEL" not in rendered
    assert "page-reviewer" not in rendered


def test_policy_scope_must_match_independent_deployment_expectations(tmp_path: Path) -> None:
    keys = _keys()
    paths = _write_signed_policy(tmp_path, _policy(keys), keys["root"])
    assert (
        load_signed_evaluation_trust_policy(
            *paths,
            current=NOW,
            expected_organization_id="bank-org",
            expected_environment_id="openshift-production",
            expected_deployment_scope="bank_production",
        ).policy.environment_id
        == "openshift-production"
    )

    for expected in (
        {
            "expected_organization_id": "other-bank",
            "expected_environment_id": "openshift-production",
            "expected_deployment_scope": "bank_production",
        },
        {
            "expected_organization_id": "bank-org",
            "expected_environment_id": "other-environment",
            "expected_deployment_scope": "bank_production",
        },
        {
            "expected_organization_id": "bank-org",
            "expected_environment_id": "openshift-production",
            "expected_deployment_scope": "development",
        },
        {"expected_organization_id": "bank-org"},
    ):
        with pytest.raises(EvaluationTrustPolicyError):
            load_signed_evaluation_trust_policy(*paths, current=NOW, **expected)


def test_policy_root_cannot_be_an_operational_signer(tmp_path: Path) -> None:
    keys = _keys()
    policy = _policy(keys)
    policy["authorized_signers"]["corpus_scope_approver"][0]["key_fingerprint_sha256"] = _fingerprint(keys["root"])

    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, policy, keys)


def test_roles_require_distinct_keys_and_distinct_owner_authorities(tmp_path: Path) -> None:
    keys = _keys()
    same_key = _policy(keys)
    same_key["authorized_signers"]["expert_dataset_owner"][0]["key_fingerprint_sha256"] = _fingerprint(keys["corpus"])
    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, same_key, keys)

    same_owner = _policy(keys)
    same_owner["authorized_signers"]["expert_dataset_owner"][0]["owner_id"] = "corpus-owner"
    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, same_owner, keys)

    issuer_is_operator = _policy(keys)
    issuer_is_operator["issuer_id"] = "corpus-owner"
    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, issuer_is_operator, keys)

    reviewer_is_operator = _policy(keys)
    reviewer_is_operator["authorized_legal_source_reviewers"][0]["owner_id"] = "corpus-owner"
    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, reviewer_is_operator, keys)

    reviewer_is_issuer = _policy(keys)
    reviewer_is_issuer["authorized_legal_source_reviewers"][0]["owner_id"] = "bank-policy-authority"
    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, reviewer_is_issuer, keys)


def test_reviewer_registry_is_unique_canonical_and_label_safe(tmp_path: Path) -> None:
    keys = _keys()
    duplicate = _policy(keys)
    duplicate["authorized_legal_source_reviewers"].append(deepcopy(duplicate["authorized_legal_source_reviewers"][0]))
    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, duplicate, keys)

    reversed_order = _policy(keys)
    reversed_order["authorized_legal_source_reviewers"] = [
        _reviewer("z-reviewer"),
        _reviewer("a-reviewer"),
    ]
    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, reversed_order, keys)

    invalid_label = _policy(keys)
    invalid_label["authorized_legal_source_reviewers"][0]["owner_label"] = " reviewer"
    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, invalid_label, keys)

    canonical = _policy(keys)
    canonical["authorized_legal_source_reviewers"] = [
        _reviewer("a-reviewer"),
        _reviewer("page-reviewer"),
    ]
    assert len(_load(tmp_path, canonical, keys).policy.authorized_legal_source_reviewers) == 2


def test_reviewer_revocations_are_known_unique_canonical_and_effective(tmp_path: Path) -> None:
    keys = _keys()
    unknown = _policy(keys)
    unknown["revoked_legal_source_reviewers"] = [
        {
            "owner_id": "unknown-reviewer",
            "revoked_at": "2026-07-15T00:00:00Z",
            "reason_code": "review_authority_withdrawn",
        }
    ]
    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, unknown, keys)

    duplicate = _policy(keys)
    duplicate["revoked_legal_source_reviewers"] = [
        {
            "owner_id": "page-reviewer",
            "revoked_at": "2026-07-14T00:00:00Z",
            "reason_code": "review_authority_withdrawn",
        },
        {
            "owner_id": "page-reviewer",
            "revoked_at": "2026-07-15T00:00:00Z",
            "reason_code": "identity_compromise",
        },
    ]
    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, duplicate, keys)

    noncanonical = _policy(keys)
    noncanonical["authorized_legal_source_reviewers"] = [
        _reviewer("a-reviewer"),
        _reviewer("page-reviewer"),
    ]
    noncanonical["revoked_legal_source_reviewers"] = [
        {
            "owner_id": "page-reviewer",
            "revoked_at": "2026-07-15T00:00:00Z",
            "reason_code": "review_authority_withdrawn",
        },
        {
            "owner_id": "a-reviewer",
            "revoked_at": "2026-07-15T00:00:00Z",
            "reason_code": "review_authority_withdrawn",
        },
    ]
    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, noncanonical, keys)

    future = _policy(keys)
    future["revoked_legal_source_reviewers"] = [
        {
            "owner_id": "page-reviewer",
            "revoked_at": "2026-07-16T12:00:01Z",
            "reason_code": "review_authority_withdrawn",
        }
    ]
    assert _authorize(_load(tmp_path, future, keys), keys).authorized_reviewer_count == 1

    boundary = _policy(keys)
    boundary["revoked_legal_source_reviewers"] = [
        {
            "owner_id": "page-reviewer",
            "revoked_at": NOW.isoformat(),
            "reason_code": "review_authority_withdrawn",
        }
    ]
    with pytest.raises(EvaluationTrustPolicyError):
        _authorize(_load(tmp_path, boundary, keys), keys)


@pytest.mark.parametrize("label", [" ", " leading", "trailing ", "control\nlabel"])
def test_policy_owner_and_issuer_labels_are_trimmed_printable_text(tmp_path: Path, label: str) -> None:
    keys = _keys()
    owner_label = _policy(keys)
    owner_label["authorized_signers"]["corpus_scope_approver"][0]["owner_label"] = label
    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, owner_label, keys)

    issuer_label = _policy(keys)
    issuer_label["issuer_label"] = label
    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, issuer_label, keys)


def test_same_canonical_key_cannot_be_aliased_as_a_rotation(tmp_path: Path) -> None:
    keys = _keys()
    policy = _policy(keys)
    policy["authorized_signers"]["legal_release_certifier"] = [
        _entry("release-key-v1", "release-owner", keys["release"]),
        _entry(
            "release-key-v2",
            "release-owner",
            keys["release"],
            valid_from="2026-07-11T00:00:00Z",
            replaces_key_id="release-key-v1",
        ),
    ]

    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, policy, keys)


@pytest.mark.parametrize("shape", ["two_roots", "cycle"])
def test_rotation_graph_rejects_disconnected_roots_and_cycles(tmp_path: Path, shape: str) -> None:
    keys = _keys()
    policy = _policy(keys)
    first = _entry("release-key-v1", "release-owner", keys["release_old"])
    second = _entry("release-key-v2", "release-owner", keys["release"])
    if shape == "cycle":
        first["replaces_key_id"] = "release-key-v2"
        second["replaces_key_id"] = "release-key-v1"
    policy["authorized_signers"]["legal_release_certifier"] = [first, second]

    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, policy, keys)


def test_wrong_role_and_wrong_artifact_binding_fail_closed(tmp_path: Path) -> None:
    keys = _keys()
    signed = _load(tmp_path, _policy(keys), keys)

    with pytest.raises(EvaluationTrustPolicyError):
        _authorize(signed, keys, dataset_signer_fingerprint_sha256=_fingerprint(keys["corpus"]))
    with pytest.raises(EvaluationTrustPolicyError):
        _authorize(signed, keys, dataset_sha256="f" * 64)


def test_effective_key_and_checkpoint_revocation_fail_closed(tmp_path: Path) -> None:
    keys = _keys()
    revoked_key_policy = _policy(keys)
    revoked_key_policy["revoked_keys"] = [
        {
            "key_fingerprint_sha256": _fingerprint(keys["dataset"]),
            "revoked_at": "2026-07-15T00:00:00Z",
            "reason_code": "key_compromise",
        }
    ]
    with pytest.raises(EvaluationTrustPolicyError):
        _authorize(_load(tmp_path, revoked_key_policy, keys), keys)

    checkpoint_policy = _policy(keys)
    checkpoint_policy["revoked_legal_release_checkpoints"] = [
        {
            "checkpoint_sha256": HASHES["checkpoint"],
            "revoked_at": "2026-07-15T00:00:00Z",
            "reason_code": "evidence_withdrawn",
        }
    ]
    with pytest.raises(EvaluationTrustPolicyError):
        _authorize(_load(tmp_path, checkpoint_policy, keys), keys)

    predecessor_policy = _policy(keys)
    predecessor_policy["revoked_legal_release_checkpoints"] = [
        {
            "checkpoint_sha256": "6" * 64,
            "revoked_at": "2026-07-15T00:00:00Z",
            "reason_code": "evidence_withdrawn",
        }
    ]
    predecessor_time = datetime(2026, 7, 9, tzinfo=UTC)
    with pytest.raises(EvaluationTrustPolicyError):
        _authorize(
            _load(tmp_path, predecessor_policy, keys),
            keys,
            legal_release_chain_signers=[
                (_fingerprint(keys["release"]), predecessor_time, "6" * 64),
                (_fingerprint(keys["release"]), EVENT, HASHES["checkpoint"]),
            ],
            legal_source_reviews=[
                ("6" * 64, ARTIFACT_ID, "page-reviewer", predecessor_time, 2),
                (HASHES["checkpoint"], ARTIFACT_ID, "page-reviewer", EVENT, 2),
            ],
        )

    future_policy = _policy(keys)
    future_policy["revoked_keys"] = [
        {
            "key_fingerprint_sha256": _fingerprint(keys["dataset"]),
            "revoked_at": "2026-07-16T12:00:01Z",
            "reason_code": "key_compromise",
        }
    ]
    future_policy["revoked_legal_release_checkpoints"] = [
        {
            "checkpoint_sha256": HASHES["checkpoint"],
            "revoked_at": "2026-07-16T12:00:01Z",
            "reason_code": "evidence_withdrawn",
        }
    ]
    assert _authorize(_load(tmp_path, future_policy, keys), keys).authorized_owner_count == 4

    boundary_policy = _policy(keys)
    boundary_policy["revoked_legal_release_checkpoints"] = [
        {
            "checkpoint_sha256": HASHES["checkpoint"],
            "revoked_at": NOW.isoformat(),
            "reason_code": "evidence_withdrawn",
        }
    ]
    with pytest.raises(EvaluationTrustPolicyError):
        _authorize(_load(tmp_path, boundary_policy, keys), keys)


def test_unknown_key_revocation_fails_but_out_of_chain_checkpoint_is_denylisted(tmp_path: Path) -> None:
    keys = _keys()
    unknown_key = _policy(keys)
    unknown_key["revoked_keys"] = [
        {
            "key_fingerprint_sha256": "f" * 64,
            "revoked_at": "2026-07-15T00:00:00Z",
            "reason_code": "key_compromise",
        }
    ]
    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, unknown_key, keys)

    unrelated_checkpoint = _policy(keys)
    unrelated_checkpoint["revoked_legal_release_checkpoints"] = [
        {
            "checkpoint_sha256": "e" * 64,
            "revoked_at": "2026-07-15T00:00:00Z",
            "reason_code": "evidence_withdrawn",
        }
    ]
    authorization = _authorize(_load(tmp_path, unrelated_checkpoint, keys), keys)
    assert authorization.approved_checkpoint_sha256 == HASHES["checkpoint"]

    revoked_head = deepcopy(unrelated_checkpoint)
    revoked_head["approved_release"]["legal_release_checkpoint_sha256"] = "e" * 64
    with pytest.raises(EvaluationTrustPolicyError):
        _authorize(
            _load(tmp_path, revoked_head, keys),
            keys,
            legal_release_checkpoint_sha256="e" * 64,
            legal_release_chain_signers=[(_fingerprint(keys["release"]), EVENT, "e" * 64)],
            legal_source_reviews=[("e" * 64, ARTIFACT_ID, "page-reviewer", EVENT, 2)],
        )


def test_configured_legal_release_keyring_must_be_policy_authorized(tmp_path: Path) -> None:
    keys = _keys()
    signed = _load(tmp_path, _policy(keys), keys)

    with pytest.raises(EvaluationTrustPolicyError):
        _authorize(
            signed,
            keys,
            legal_release_configured_key_fingerprints_sha256=[
                _fingerprint(keys["release"]),
                _fingerprint(keys["release_old"]),
            ],
        )

    rotation_policy = _policy(keys)
    rotation_policy["authorized_signers"]["legal_release_certifier"] = [
        _entry("release-key-v1", "release-owner", keys["release_old"]),
        _entry(
            "release-key-v2",
            "release-owner",
            keys["release"],
            replaces_key_id="release-key-v1",
        ),
    ]
    with pytest.raises(EvaluationTrustPolicyError):
        _authorize(
            _load(tmp_path, rotation_policy, keys),
            keys,
            legal_release_configured_key_fingerprints_sha256=[_fingerprint(keys["release_old"])],
        )
    rotation_signed = _load(tmp_path, rotation_policy, keys)
    configured_rotation = [
        _fingerprint(keys["release"]),
        _fingerprint(keys["release_old"]),
    ]
    assert (
        _authorize(
            rotation_signed,
            keys,
            legal_release_configured_key_fingerprints_sha256=configured_rotation,
        ).authorized_owner_count
        == 4
    )
    for invalid_keyring in (list(reversed(configured_rotation)), [configured_rotation[0]] * 2):
        with pytest.raises(EvaluationTrustPolicyError):
            _authorize(
                rotation_signed,
                keys,
                legal_release_configured_key_fingerprints_sha256=invalid_keyring,
            )

    revoked_unused_key_policy = deepcopy(rotation_policy)
    revoked_unused_key_policy["revoked_keys"] = [
        {
            "key_fingerprint_sha256": _fingerprint(keys["release_old"]),
            "revoked_at": "2026-07-15T00:00:00Z",
            "reason_code": "key_compromise",
        }
    ]
    with pytest.raises(EvaluationTrustPolicyError):
        _authorize(
            _load(tmp_path, revoked_unused_key_policy, keys),
            keys,
            legal_release_configured_key_fingerprints_sha256=[
                _fingerprint(keys["release"]),
                _fingerprint(keys["release_old"]),
            ],
        )


def test_page_reviewer_requires_v2_identity_window_and_non_revoked_owner(tmp_path: Path) -> None:
    keys = _keys()
    signed = _load(tmp_path, _policy(keys), keys)

    for invalid_reviews in (
        [],
        [(HASHES["checkpoint"], ARTIFACT_ID, None, EVENT, 1)],
        [(HASHES["checkpoint"], ARTIFACT_ID, "page-reviewer", EVENT, 1)],
        [(HASHES["checkpoint"], ARTIFACT_ID, "page-reviewer", EVENT, 2.0)],
        [(HASHES["checkpoint"], ARTIFACT_ID, "unknown-reviewer", EVENT, 2)],
        [(HASHES["checkpoint"], "artifact-not-a-hash", "page-reviewer", EVENT, 2)],
        [
            (
                HASHES["checkpoint"],
                ARTIFACT_ID,
                "page-reviewer",
                datetime(2024, 1, 1, tzinfo=UTC),
                2,
            )
        ],
    ):
        with pytest.raises(EvaluationTrustPolicyError):
            _authorize(signed, keys, legal_source_reviews=invalid_reviews)

    revoked = _policy(keys)
    revoked["revoked_legal_source_reviewers"] = [
        {
            "owner_id": "page-reviewer",
            "revoked_at": "2026-07-15T00:00:00Z",
            "reason_code": "review_authority_withdrawn",
        }
    ]
    with pytest.raises(EvaluationTrustPolicyError):
        _authorize(_load(tmp_path, revoked, keys), keys)

    boundary = _policy(keys)
    boundary["authorized_legal_source_reviewers"][0]["valid_until"] = EVENT.isoformat()
    with pytest.raises(EvaluationTrustPolicyError):
        _authorize(_load(tmp_path, boundary, keys), keys)


def test_reviewer_history_rejects_duplicates_unrelated_checkpoints_and_late_reviews(tmp_path: Path) -> None:
    keys = _keys()
    signed = _load(tmp_path, _policy(keys), keys)
    valid = (HASHES["checkpoint"], ARTIFACT_ID, "page-reviewer", EVENT, 2)

    for invalid_reviews in (
        [valid, valid],
        [("6" * 64, ARTIFACT_ID, "page-reviewer", EVENT, 2)],
        [
            (
                HASHES["checkpoint"],
                ARTIFACT_ID,
                "page-reviewer",
                datetime(2026, 7, 10, 13, tzinfo=UTC),
                2,
            )
        ],
    ):
        with pytest.raises(EvaluationTrustPolicyError):
            _authorize(signed, keys, legal_source_reviews=invalid_reviews)


def test_v1_review_in_any_predecessor_blocks_policy_bound_chain(tmp_path: Path) -> None:
    keys = _keys()
    signed = _load(tmp_path, _policy(keys), keys)
    predecessor_time = datetime(2026, 7, 9, tzinfo=UTC)
    chain = [
        (_fingerprint(keys["release"]), predecessor_time, "6" * 64),
        (_fingerprint(keys["release"]), EVENT, HASHES["checkpoint"]),
    ]
    reviews = [
        ("6" * 64, ARTIFACT_ID, None, predecessor_time, 1),
        (HASHES["checkpoint"], ARTIFACT_ID, "page-reviewer", EVENT, 2),
    ]

    with pytest.raises(EvaluationTrustPolicyError):
        _authorize(
            signed,
            keys,
            legal_release_chain_signers=chain,
            legal_source_reviews=reviews,
        )


def test_rotation_preserves_retired_history_but_rejects_compromised_key(tmp_path: Path) -> None:
    keys = _keys()
    policy = _policy(keys)
    policy["authorized_signers"]["legal_release_certifier"] = [
        _entry(
            "release-key-v1",
            "release-owner",
            keys["release_old"],
            valid_until="2026-07-12T00:00:00Z",
        ),
        _entry(
            "release-key-v2",
            "release-owner",
            keys["release"],
            valid_from="2026-07-11T00:00:00Z",
            replaces_key_id="release-key-v1",
        ),
    ]
    signed = _load(tmp_path, policy, keys)

    retired = _authorize(
        signed,
        keys,
        legal_release_signer_fingerprint_sha256=_fingerprint(keys["release_old"]),
        legal_release_authorized_at=datetime(2026, 7, 10, tzinfo=UTC),
    )
    assert retired.authorized_owner_count == 4
    assert (
        _authorize(
            signed,
            keys,
            legal_release_authorized_at=datetime(2026, 7, 12, tzinfo=UTC),
            legal_release_chain_signers=[
                (
                    _fingerprint(keys["release_old"]),
                    datetime(2026, 7, 10, tzinfo=UTC),
                    "6" * 64,
                ),
                (
                    _fingerprint(keys["release"]),
                    datetime(2026, 7, 12, tzinfo=UTC),
                    HASHES["checkpoint"],
                ),
            ],
            legal_source_reviews=[
                (
                    "6" * 64,
                    ARTIFACT_ID,
                    "page-reviewer",
                    datetime(2026, 7, 10, tzinfo=UTC),
                    2,
                ),
                (
                    HASHES["checkpoint"],
                    ARTIFACT_ID,
                    "page-reviewer",
                    datetime(2026, 7, 12, tzinfo=UTC),
                    2,
                ),
            ],
        ).authorized_owner_count
        == 4
    )

    compromised = deepcopy(policy)
    compromised["revoked_keys"] = [
        {
            "key_fingerprint_sha256": _fingerprint(keys["release_old"]),
            "revoked_at": "2026-07-15T00:00:00Z",
            "reason_code": "key_compromise",
        }
    ]
    with pytest.raises(EvaluationTrustPolicyError):
        _authorize(
            _load(tmp_path, compromised, keys),
            keys,
            legal_release_signer_fingerprint_sha256=_fingerprint(keys["release_old"]),
            legal_release_authorized_at=datetime(2026, 7, 10, tzinfo=UTC),
        )


def test_rotation_sequence_cannot_revert_from_new_key_to_its_predecessor(tmp_path: Path) -> None:
    keys = _keys()
    policy = _policy(keys)
    policy["authorized_signers"]["legal_release_certifier"] = [
        _entry(
            "release-key-v1",
            "release-owner",
            keys["release_old"],
            valid_until="2026-07-20T00:00:00Z",
        ),
        _entry(
            "release-key-v2",
            "release-owner",
            keys["release"],
            valid_from="2026-07-11T00:00:00Z",
            replaces_key_id="release-key-v1",
        ),
    ]
    signed = _load(tmp_path, policy, keys)
    old_event = datetime(2026, 7, 12, 10, tzinfo=UTC)
    new_event = datetime(2026, 7, 12, 9, tzinfo=UTC)

    with pytest.raises(EvaluationTrustPolicyError):
        _authorize(
            signed,
            keys,
            legal_release_signer_fingerprint_sha256=_fingerprint(keys["release_old"]),
            legal_release_authorized_at=old_event,
            legal_release_chain_signers=[
                (_fingerprint(keys["release"]), new_event, "6" * 64),
                (_fingerprint(keys["release_old"]), old_event, HASHES["checkpoint"]),
            ],
            legal_source_reviews=[
                ("6" * 64, ARTIFACT_ID, "page-reviewer", new_event, 2),
                (HASHES["checkpoint"], ARTIFACT_ID, "page-reviewer", old_event, 2),
            ],
        )


def test_policy_versions_require_a_predecessor_hash_after_version_one(tmp_path: Path) -> None:
    keys = _keys()
    policy = _policy(keys)
    policy["policy_version"] = 2

    with pytest.raises(EvaluationTrustPolicyError):
        _load(tmp_path, policy, keys)

    policy["supersedes_policy_sha256"] = "a" * 64
    assert _load(tmp_path, policy, keys).policy.policy_version == 2


def test_real_signed_policy_report_does_not_leak_owner_or_issuer_labels(tmp_path: Path, monkeypatch) -> None:
    keys = _keys()
    policy = _policy(keys)
    policy_path, signature_path, root_path = _write_signed_policy(tmp_path, policy, keys["root"])
    policy_sha256 = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    validation = SimpleNamespace(
        legal_release_checkpoint_sha256=HASHES["checkpoint"],
        legal_release_checkpoint_created_at=EVENT,
        legal_release_chain_checkpoint_count=1,
        legal_release_genesis_checkpoint_sha256=HASHES["checkpoint"],
        legal_release_signing_key_fingerprint_sha256=_fingerprint(keys["release"]),
        legal_release_chain_signers=(
            SimpleNamespace(
                checkpoint_sha256=HASHES["checkpoint"],
                checkpoint_created_at=EVENT,
                signing_key_fingerprint_sha256=_fingerprint(keys["release"]),
            ),
        ),
        legal_release_configured_key_fingerprints_sha256=(_fingerprint(keys["release"]),),
        legal_source_reviews=(
            SimpleNamespace(
                checkpoint_sha256=HASHES["checkpoint"],
                artifact_id=ARTIFACT_ID,
                reviewer_owner_id="page-reviewer",
                reviewed_at=EVENT,
                proof_schema_version=2,
            ),
        ),
        legal_pack_sha256=HASHES["pack"],
        legal_attestation_sha256=HASHES["attestation"],
        legal_attestation_key_fingerprint_sha256=_fingerprint(keys["curator"]),
        legal_attestation_attested_at=EVENT,
        dataset_signing_key_fingerprint_sha256=_fingerprint(keys["dataset"]),
        dataset=SimpleNamespace(approval=SimpleNamespace(decided_at=EVENT)),
        corpus_validation=SimpleNamespace(
            signing_key_fingerprint_sha256=_fingerprint(keys["corpus"]),
            manifest=SimpleNamespace(freshness=SimpleNamespace(scope_reviewed_at=EVENT)),
        ),
    )
    profile = SimpleNamespace(
        dataset_id="dataset-v1",
        dataset_version="1.0.0",
        dataset_sha256=HASHES["dataset"],
        corpus_manifest_id="corpus-v1",
        corpus_manifest_sha256=HASHES["corpus"],
        case_count=20,
        evidence_count=20,
        release_blocker_counts={},
    )
    monkeypatch.setattr(release_preflight, "_trusted_now", lambda: NOW)
    monkeypatch.setattr(release_preflight, "load_expert_evaluation_dataset", lambda *a, **k: validation)
    monkeypatch.setattr(release_preflight, "profile_expert_evaluation_dataset", lambda value: profile)
    monkeypatch.setattr(release_preflight, "require_expert_dataset_release_ready", lambda value: None)

    inputs = ReleasePreflightInputs(
        dataset=Path("unused.yml"),
        corpus_manifest=None,
        corpus_root=None,
        trusted_dataset_key=None,
        trusted_corpus_key=None,
        legal_pack=None,
        legal_attestation=None,
        trusted_legal_attestation_key=None,
        legal_release_checkpoint=None,
        legal_release_source_root=None,
        trusted_legal_release_key=None,
        predecessor_legal_release_checkpoint=None,
        trusted_latest_legal_checkpoint_sha256=None,
        bank_trust_policy=policy_path,
        bank_trust_policy_signature=signature_path,
        trusted_bank_policy_key=root_path,
        trust_mode="bank_policy",
        trusted_current_bank_policy_sha256=policy_sha256,
        trusted_current_bank_policy_version=1,
        trusted_bank_organization_id="bank-org",
        trusted_bank_environment_id="openshift-production",
        trusted_bank_deployment_scope="bank_production",
    )
    report = run_release_preflight(inputs)

    serialized = json.dumps(report, sort_keys=True)
    assert report["configured_root_policy_signature_verified"] is True
    assert report["policy_current_head_pin_verified"] is True
    assert report["policy_deployment_scope_pin_verified"] is True
    assert report["trust_policy_authorized_reviewer_count"] == 1
    assert report["policy_bound_legal_source_reviews_verified"] is True
    assert report["policy_bound_legal_source_review_count"] == 1
    assert "Display label" not in serialized
    assert "Bank policy authority display label" not in serialized
    assert "Legal source reviewer display label" not in serialized
    assert "corpus-owner" not in serialized

    validation.legal_source_reviews = (
        SimpleNamespace(
            checkpoint_sha256=HASHES["checkpoint"],
            artifact_id=ARTIFACT_ID,
            reviewer_owner_id=None,
            reviewed_at=EVENT,
            proof_schema_version=1,
        ),
    )
    with pytest.raises(EvaluationTrustPolicyError):
        run_release_preflight(inputs)
