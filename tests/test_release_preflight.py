"""Tests for the executable expert-evaluation release preflight."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import benchmark.release_preflight as release_preflight
from benchmark.evaluation_trust_policy import EvaluationTrustAuthorization
from benchmark.expert_evaluation import ExpertEvaluationError
from benchmark.release_preflight import ReleasePreflightInputs, main, run_release_preflight


def test_tracked_draft_fails_preflight_without_exposing_case_content(capsys) -> None:
    result = main([])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert result == 1
    assert captured.out == ""
    assert payload == {
        "schema_version": 1,
        "status": "release_preflight_failed",
        "error_code": "EXPERT_EVALUATION_NOT_RELEASE_READY",
        "model_scores_authorized": False,
    }
    assert "Bugün itibarıyla" not in captured.err
    assert "markdown_content" not in captured.err


def test_unexpected_preflight_failure_is_path_and_content_free(capsys, monkeypatch) -> None:
    def fail_without_leaking(*args, **kwargs):
        raise OSError("/private/bank/source.pdf: secret page text")

    monkeypatch.setattr(release_preflight, "run_release_preflight", fail_without_leaking)

    result = main([])

    captured = capsys.readouterr()
    assert result == 3
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "schema_version": 1,
        "status": "release_preflight_failed",
        "error_code": "RELEASE_PREFLIGHT_INTERNAL_ERROR",
        "model_scores_authorized": False,
    }
    assert "/private/bank/source.pdf" not in captured.err
    assert "secret page text" not in captured.err


def test_validation_failure_uses_a_stable_content_free_exit(capsys, monkeypatch) -> None:
    def fail_validation(*args, **kwargs):
        raise ExpertEvaluationError("/private/source: untrusted legal content")

    monkeypatch.setattr(release_preflight, "run_release_preflight", fail_validation)

    result = main([])

    captured = capsys.readouterr()
    assert result == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error_code"] == "EXPERT_EVALUATION_VALIDATION_FAILED"
    assert "/private/source" not in captured.err
    assert "untrusted legal content" not in captured.err


def test_invalid_cli_arguments_do_not_echo_untrusted_values(capsys) -> None:
    result = main(["--unknown", "/private/source-secret.pdf"])

    captured = capsys.readouterr()
    assert result == 64
    assert captured.out == ""
    assert json.loads(captured.err)["error_code"] == "RELEASE_PREFLIGHT_ARGUMENTS_INVALID"
    assert "/private/source-secret.pdf" not in captured.err


def test_cli_does_not_accept_a_caller_controlled_clock(capsys) -> None:
    result = main(["--now", "2099-01-01T00:00:00Z"])

    captured = capsys.readouterr()
    assert result == 64
    assert "2099-01-01" not in captured.err


def test_required_or_partial_trust_policy_fails_with_content_free_error(capsys) -> None:
    result = main(["--trust-mode", "bank-policy"])

    captured = capsys.readouterr()
    assert result == 4
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "schema_version": 1,
        "status": "release_preflight_failed",
        "error_code": "EVALUATION_TRUST_POLICY_VALIDATION_FAILED",
        "bank_authorization_verified": False,
        "model_scores_authorized": False,
    }

    result = main(["--bank-trust-policy", "/private/bank/policy.yml"])
    captured = capsys.readouterr()
    assert result == 4
    assert "/private/bank/policy.yml" not in captured.err


def test_policy_scope_pins_are_required_only_in_bank_policy_mode(capsys) -> None:
    result = main(["--trusted-bank-organization-id", "bank-org"])
    captured = capsys.readouterr()
    assert result == 4
    assert json.loads(captured.err)["error_code"] == "EVALUATION_TRUST_POLICY_VALIDATION_FAILED"

    result = main(
        [
            "--trust-mode",
            "bank-policy",
            "--bank-trust-policy",
            "missing-policy.yml",
            "--bank-trust-policy-signature",
            "missing-policy.sig",
            "--trusted-bank-policy-key",
            "missing-root.pem",
            "--trusted-current-bank-policy-sha256",
            "f" * 64,
            "--trusted-current-bank-policy-version",
            "1",
        ]
    )
    captured = capsys.readouterr()
    assert result == 4
    assert json.loads(captured.err)["error_code"] == "EVALUATION_TRUST_POLICY_VALIDATION_FAILED"


def test_programmatic_policy_pins_are_strictly_typed() -> None:
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
        bank_trust_policy=Path("missing-policy.yml"),
        bank_trust_policy_signature=Path("missing-policy.sig"),
        trusted_bank_policy_key=Path("missing-root.pem"),
        trust_mode="bank_policy",
        trusted_current_bank_policy_sha256="a" * 64,
        trusted_current_bank_policy_version=1,
        trusted_bank_organization_id="bank-org",
        trusted_bank_environment_id="openshift-production",
        trusted_bank_deployment_scope="bank_production",
    )
    invalid_overrides = (
        {"trusted_current_bank_policy_sha256": "not-a-hash"},
        {"trusted_current_bank_policy_version": True},
        {"trusted_current_bank_policy_version": 1.0},
        {"trusted_bank_organization_id": "../bank"},
        {"trusted_bank_environment_id": "invalid environment"},
        {"trusted_bank_deployment_scope": "bank-production"},
    )

    for override in invalid_overrides:
        with pytest.raises(release_preflight.EvaluationTrustPolicyError):
            run_release_preflight(replace(inputs, **override))


def test_signed_policy_forbids_a_manual_latest_head(capsys) -> None:
    result = main(
        [
            "--bank-trust-policy",
            "missing-policy.yml",
            "--bank-trust-policy-signature",
            "missing-policy.sig",
            "--trusted-bank-policy-key",
            "missing-root.pem",
            "--trusted-latest-legal-checkpoint-sha256",
            "f" * 64,
        ]
    )

    captured = capsys.readouterr()
    assert result == 4
    assert captured.out == ""
    assert "missing" not in captured.err


def test_success_report_records_trust_identities_and_unsigned_checksum(monkeypatch) -> None:
    validation = SimpleNamespace(
        legal_release_checkpoint_sha256="1" * 64,
        legal_release_chain_checkpoint_count=3,
        legal_release_genesis_checkpoint_sha256="2" * 64,
        legal_release_signing_key_fingerprint_sha256="3" * 64,
        legal_pack_sha256="4" * 64,
        legal_attestation_sha256="5" * 64,
        legal_attestation_key_fingerprint_sha256="6" * 64,
        dataset_signing_key_fingerprint_sha256="7" * 64,
        corpus_validation=SimpleNamespace(signing_key_fingerprint_sha256="8" * 64),
        legal_release_chain_signers=(SimpleNamespace(signing_key_fingerprint_sha256="3" * 64),),
    )
    profile = SimpleNamespace(
        dataset_id="dataset-v1",
        dataset_version="1.0.0",
        dataset_sha256="9" * 64,
        corpus_manifest_id="corpus-v1",
        corpus_manifest_sha256="a" * 64,
        case_count=20,
        evidence_count=20,
        release_blocker_counts={},
    )
    readiness_calls = []
    monkeypatch.setattr(release_preflight, "load_expert_evaluation_dataset", lambda *args, **kwargs: validation)
    monkeypatch.setattr(release_preflight, "profile_expert_evaluation_dataset", lambda value: profile)
    monkeypatch.setattr(release_preflight, "require_expert_dataset_release_ready", readiness_calls.append)
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
    )
    monkeypatch.setattr(release_preflight, "_trusted_now", lambda: datetime(2026, 7, 16, tzinfo=UTC))

    report = run_release_preflight(inputs)

    checksum = report.pop("preflight_self_checksum_sha256")
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert checksum == release_preflight.hashlib.sha256(canonical.encode()).hexdigest()
    assert report["self_checksum_algorithm"] == "sha256_canonical_json_unsigned"
    assert report["status"] == "cryptographic_preflight_passed"
    assert report["schema_version"] == 2
    assert report["bank_authorization_verified"] is False
    assert report["configured_root_policy_signature_verified"] is False
    assert report["policy_approved_release_binding_verified"] is False
    assert report["policy_deployment_scope_pin_verified"] is False
    assert report["policy_bound_legal_source_reviews_verified"] is False
    assert report["policy_bound_legal_source_review_count"] == 0
    assert report["latest_checkpoint_anchor_provenance"] == "caller_supplied_argument"
    assert report["legal_release_chain_checkpoint_count"] == 3
    assert readiness_calls == [validation]


def test_signed_policy_supplies_latest_head_and_emits_only_safe_policy_evidence(monkeypatch) -> None:
    current = datetime(2026, 7, 16, tzinfo=UTC)
    validation = SimpleNamespace(
        legal_release_checkpoint_sha256="1" * 64,
        legal_release_checkpoint_created_at=datetime(2026, 7, 15, tzinfo=UTC),
        legal_release_chain_checkpoint_count=1,
        legal_release_genesis_checkpoint_sha256="1" * 64,
        legal_release_signing_key_fingerprint_sha256="2" * 64,
        legal_release_chain_signers=(
            SimpleNamespace(
                checkpoint_sha256="1" * 64,
                checkpoint_created_at=datetime(2026, 7, 15, tzinfo=UTC),
                signing_key_fingerprint_sha256="2" * 64,
            ),
        ),
        legal_release_configured_key_fingerprints_sha256=("2" * 64,),
        legal_source_reviews=(
            SimpleNamespace(
                checkpoint_sha256="1" * 64,
                artifact_id="art_sha256_" + "a" * 64,
                reviewer_owner_id="page-reviewer",
                reviewed_at=datetime(2026, 7, 15, tzinfo=UTC),
                proof_schema_version=2,
            ),
        ),
        legal_pack_sha256="3" * 64,
        legal_attestation_sha256="4" * 64,
        legal_attestation_key_fingerprint_sha256="5" * 64,
        legal_attestation_attested_at=datetime(2026, 7, 14, tzinfo=UTC),
        dataset_signing_key_fingerprint_sha256="6" * 64,
        dataset=SimpleNamespace(approval=SimpleNamespace(decided_at=datetime(2026, 7, 13, tzinfo=UTC))),
        corpus_validation=SimpleNamespace(
            signing_key_fingerprint_sha256="7" * 64,
            manifest=SimpleNamespace(freshness=SimpleNamespace(scope_reviewed_at=datetime(2026, 7, 12, tzinfo=UTC))),
        ),
    )
    profile = SimpleNamespace(
        dataset_id="dataset-v1",
        dataset_version="1.0.0",
        dataset_sha256="8" * 64,
        corpus_manifest_id="corpus-v1",
        corpus_manifest_sha256="9" * 64,
        case_count=20,
        evidence_count=20,
        release_blocker_counts={},
    )
    signed_policy = SimpleNamespace(
        policy_sha256="a" * 64,
        policy=SimpleNamespace(
            policy_version=1,
            approved_release=SimpleNamespace(legal_release_checkpoint_sha256="1" * 64),
        ),
    )
    authorization = EvaluationTrustAuthorization(
        policy_id="bank-policy-v1",
        policy_version=1,
        policy_sha256="a" * 64,
        policy_signing_key_fingerprint_sha256="b" * 64,
        policy_valid_until=datetime(2026, 8, 1, tzinfo=UTC),
        approved_checkpoint_sha256="1" * 64,
        authorized_owner_count=4,
        authorized_reviewer_count=1,
        policy_bound_legal_source_review_count=1,
    )
    load_calls = []

    def load_dataset(*args, **kwargs):
        load_calls.append(kwargs)
        return validation

    monkeypatch.setattr(release_preflight, "_trusted_now", lambda: current)
    monkeypatch.setattr(release_preflight, "load_signed_evaluation_trust_policy", lambda *a, **k: signed_policy)
    monkeypatch.setattr(release_preflight, "load_expert_evaluation_dataset", load_dataset)
    monkeypatch.setattr(release_preflight, "profile_expert_evaluation_dataset", lambda value: profile)
    monkeypatch.setattr(release_preflight, "require_expert_dataset_release_ready", lambda value: None)
    monkeypatch.setattr(release_preflight, "authorize_evaluation_trust_chain", lambda *a, **k: authorization)
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
        bank_trust_policy=Path("policy.yml"),
        bank_trust_policy_signature=Path("policy.sig"),
        trusted_bank_policy_key=Path("root.pem"),
        trust_mode="bank_policy",
        trusted_current_bank_policy_sha256="a" * 64,
        trusted_current_bank_policy_version=1,
        trusted_bank_organization_id="bank-org",
        trusted_bank_environment_id="openshift-production",
        trusted_bank_deployment_scope="bank_production",
    )

    report = run_release_preflight(inputs)

    assert load_calls[0]["trusted_latest_legal_checkpoint_sha256"] == "1" * 64
    assert load_calls[0]["now"] == current
    assert report["status"] == "configured_policy_head_preflight_passed"
    assert report["configured_root_policy_signature_verified"] is True
    assert report["policy_approved_release_binding_verified"] is True
    assert report["policy_current_head_pin_verified"] is True
    assert report["policy_deployment_scope_pin_verified"] is True
    assert report["bank_authorization_verified"] is False
    assert report["model_scores_authorized"] is False
    assert report["latest_checkpoint_anchor_provenance"] == "signed_evaluation_trust_policy"
    assert report["trust_policy_id"] == "bank-policy-v1"
    assert report["trust_policy_authorized_owner_count"] == 4
    assert report["trust_policy_authorized_reviewer_count"] == 1
    assert report["policy_bound_legal_source_reviews_verified"] is True
    assert report["policy_bound_legal_source_review_count"] == 1
    assert "owner_id" not in json.dumps(report)


def test_bank_policy_mode_rejects_a_stale_but_signed_policy_head(monkeypatch) -> None:
    monkeypatch.setattr(
        release_preflight,
        "load_signed_evaluation_trust_policy",
        lambda *a, **k: SimpleNamespace(
            policy_sha256="a" * 64,
            policy=SimpleNamespace(
                policy_version=1,
                approved_release=SimpleNamespace(legal_release_checkpoint_sha256="1" * 64),
            ),
        ),
    )
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
        bank_trust_policy=Path("policy.yml"),
        bank_trust_policy_signature=Path("policy.sig"),
        trusted_bank_policy_key=Path("root.pem"),
        trust_mode="bank_policy",
        trusted_current_bank_policy_sha256="b" * 64,
        trusted_current_bank_policy_version=2,
        trusted_bank_organization_id="bank-org",
        trusted_bank_environment_id="openshift-production",
        trusted_bank_deployment_scope="bank_production",
    )

    with pytest.raises(release_preflight.EvaluationTrustPolicyError):
        run_release_preflight(inputs)
