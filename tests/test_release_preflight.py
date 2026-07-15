"""Tests for the executable expert-evaluation release preflight."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import benchmark.release_preflight as release_preflight
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
    assert report["bank_authorization_verified"] is False
    assert report["latest_checkpoint_anchor_provenance"] == "caller_supplied_argument"
    assert report["legal_release_chain_checkpoint_count"] == 3
    assert readiness_calls == [validation]
