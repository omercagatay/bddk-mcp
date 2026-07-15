"""Regression tests for privacy-safe benchmark artifacts."""

from __future__ import annotations

import json

from benchmark.audit import REDACTED, canonical_sha256, sanitize_for_audit
from benchmark.report import console_report, diagnosis_report, save_json_results


def test_recursive_sanitizer_redacts_sensitive_keys_and_values_without_mutating_input():
    source = {
        "authorization": "Bearer abcdefghijklmnop",
        "nested": {
            "database_url": "postgresql://user:password@db/bddk",
            "text": "api_key=sk-secretsecretsecret source https://www.bddk.org.tr/doc",
        },
    }

    sanitized = sanitize_for_audit(source)

    assert sanitized["authorization"] == REDACTED
    assert sanitized["nested"]["database_url"] == REDACTED
    assert "sk-secretsecretsecret" not in sanitized["nested"]["text"]
    assert "https://www.bddk.org.tr/doc" in sanitized["nested"]["text"]
    assert source["authorization"] == "Bearer abcdefghijklmnop"


def test_result_writer_performs_final_redaction_before_disk(tmp_path):
    results = {
        "phase2": {
            "model": {
                "details": [
                    {
                        "final_answer": "Bearer abcdefghijklmnop",
                        "tool_evidence": [{"api_key": "sk-secretsecretsecret", "document_id": "943"}],
                    }
                ]
            }
        }
    }

    path = save_json_results(results, tmp_path)
    persisted = path.read_text(encoding="utf-8")
    decoded = json.loads(persisted)

    assert "abcdefghijklmnop" not in persisted
    assert "sk-secretsecretsecret" not in persisted
    assert decoded["phase2"]["model"]["details"][0]["tool_evidence"][0]["document_id"] == "943"


def test_canonical_hash_is_stable_after_redaction_and_does_not_depend_on_key_order():
    left = {"document_id": "943", "token": "first"}
    right = {"token": "second", "document_id": "943"}

    assert canonical_sha256(left) == canonical_sha256(right)


def test_text_redaction_covers_authorization_headers_jwts_and_common_provider_keys():
    # Build the detector-shaped synthetic key at runtime. Keeping the complete
    # literal in Git would create a new Gitleaks history finding on every commit
    # that carries this redaction regression test.
    synthetic_aws_key = "AKIA" + "ABCDEFGHIJKLMNOP"
    value = f"Authorization: Basic dXNlcjpwYXNzd29yZA== eyJabcdefghijk.abcdefghijkl.abcdefghijkl {synthetic_aws_key}"

    rendered = sanitize_for_audit(value)

    assert "dXNlcjpwYXNzd29yZA" not in rendered
    assert "eyJabcdefghijk" not in rendered
    assert synthetic_aws_key not in rendered
    assert rendered.count(REDACTED) == 3


def test_diagnosis_ignores_evaluation_evidence_metadata():
    report = diagnosis_report(
        {
            "evaluation_evidence": {
                "classification": "exploratory_not_release_evidence",
                "release_preflight_status": "not_executed",
            },
            "phase1a": {},
        }
    )

    assert "exploratory_not_release_evidence" not in report
    assert "release_preflight_status" not in report
    assert "EXPLORATORY ONLY" in report


def test_human_reports_never_turn_unauthorized_scores_into_deployment_advice():
    results = {
        "evaluation_evidence": {
            "classification": "exploratory_not_release_evidence",
            "model_scores_authorized": False,
        },
        "phase1a": {
            "model-a": {
                "tool_selection_accuracy": 1.0,
                "tool_consistency": 1.0,
                "avg_parameter_f1": 1.0,
                "avg_latency_s": 0.1,
            }
        },
        "phase1b": {"model-a": {"accuracy": 1.0, "macro_f1": 1.0}},
        "phase1c": {"model-a": {"accuracy": 1.0, "correct": 1}},
    }

    console = console_report(results)
    diagnosis = diagnosis_report(results)

    assert "EXPLORATORY ONLY" in console
    assert "do not authorize deployment" in console
    assert "EXPLORATORY PASS" in diagnosis
    assert "Do not deploy based on these scores" in diagnosis
    assert "Deploy with RAG" not in diagnosis
