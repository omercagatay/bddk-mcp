"""Tests for the signed expert-evaluation trust-policy boundary."""

from __future__ import annotations

import hashlib
import json
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


def _policy(keys: dict[str, Ed25519PrivateKey]) -> dict:
    return {
        "schema_version": 1,
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
        "revoked_keys": [],
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
    return authorize_evaluation_trust_chain(signed, **arguments)


def test_signed_policy_authorizes_exact_release_without_exposing_owner_ids(tmp_path: Path) -> None:
    keys = _keys()
    policy = _policy(keys)
    signed = _load(tmp_path, policy, keys)

    authorization = _authorize(signed, keys)

    assert authorization.policy_id == "bank-evaluation-policy"
    assert authorization.policy_version == 1
    assert authorization.authorized_owner_count == 4
    assert authorization.approved_checkpoint_sha256 == HASHES["checkpoint"]
    assert authorization.policy_sha256 == hashlib.sha256((tmp_path / "policy.yml").read_bytes()).hexdigest()
    assert authorization.policy_signing_key_fingerprint_sha256 == _fingerprint(keys["root"])
    assert "corpus-owner" not in repr(authorization)
    assert "dataset-owner" not in repr(authorization)
    assert "Display label" not in repr(authorization)


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


def test_unknown_key_and_out_of_chain_checkpoint_revocations_fail_closed(tmp_path: Path) -> None:
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
    with pytest.raises(EvaluationTrustPolicyError):
        _authorize(_load(tmp_path, unrelated_checkpoint, keys), keys)


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

    report = run_release_preflight(
        ReleasePreflightInputs(
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
        )
    )

    serialized = json.dumps(report, sort_keys=True)
    assert report["configured_root_policy_signature_verified"] is True
    assert report["policy_current_head_pin_verified"] is True
    assert "Display label" not in serialized
    assert "Bank policy authority display label" not in serialized
    assert "corpus-owner" not in serialized
