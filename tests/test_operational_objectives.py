"""Fail-closed operational-objective decision contract tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from bddk_mcp.operational_objectives import (
    OperationalObjectivesError,
    canonical_decision_payload_sha256,
    load_operational_objectives,
)
from scripts import validate_operational_objectives

ROOT = Path(__file__).parents[1]
CONTRACT = ROOT / "docs" / "decisions" / "operational-objectives.v1.yml"


def _raw_contract() -> dict:
    value = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_contract(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "objectives.yml"
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _approved_contract() -> dict:
    raw = deepcopy(_raw_contract())
    for metric in raw["metrics"]:
        metric["target"]["approval_state"] = "approved"
        metric["target"]["value"] = 99.0 if metric["metric_id"] == "service_availability" else 1.0
        if metric["window"]["kind"] == "rolling_service_window":
            metric["window"]["duration_seconds"] = 2_592_000
        metric["source"]["verification_state"] = "verified"
        metric["alert"]["implementation_state"] = "verified"
        metric["alert"]["route_reference"] = "bank-alert-route-objectives-v1"
    raw["evidence_retention_policy"]["verification_state"] = "verified"
    raw["approval"] = {
        "state": "approved",
        "change_record_id": "BANK-CHANGE-2026-0001",
        "approvals": [],
    }
    decision_hash = canonical_decision_payload_sha256(raw)
    raw["approval"]["approvals"] = [
        {
            "role": "project_owner",
            "subject_id": "project-owner-001",
            "approved_at": "2026-01-01T00:00:00Z",
            "approved_decision_sha256": decision_hash,
            "approval_record_sha256": "a" * 64,
        },
        {
            "role": "bank_operations",
            "subject_id": "bank-operations-001",
            "approved_at": "2026-01-01T00:01:00Z",
            "approved_decision_sha256": decision_hash,
            "approval_record_sha256": "b" * 64,
        },
    ]
    return raw


def test_tracked_contract_is_complete_but_deliberately_not_production_eligible() -> None:
    validation = load_operational_objectives(CONTRACT)

    assert tuple(metric.metric_id for metric in validation.contract.metrics) == (
        "service_availability",
        "tool_latency",
        "source_detection_lag",
        "retrieval_publication_lag",
        "maximum_corpus_age",
        "recovery_point_objective",
        "recovery_time_objective",
        "evidence_retention",
    )
    assert all(metric.target.value is None for metric in validation.contract.metrics)
    rolling = [metric for metric in validation.contract.metrics if metric.window.kind == "rolling_service_window"]
    assert [metric.metric_id for metric in rolling] == ["service_availability", "tool_latency"]
    assert all(metric.window.duration_seconds is None for metric in rolling)
    assert not validation.production_eligible
    assert validation.readiness_reasons == (
        "decision_unapproved",
        "targets_unapproved",
        "rolling_windows_unapproved",
        "evidence_sources_unverified",
        "alert_mappings_unverified",
        "evidence_retention_registry_unverified",
    )


def test_production_gate_rejects_the_unapproved_tracked_decision() -> None:
    with pytest.raises(OperationalObjectivesError) as caught:
        load_operational_objectives(CONTRACT, require_production_approval=True)
    assert caught.value.code == "production_objectives_unapproved"
    assert "immediate" not in str(caught.value).lower()


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.update({"unknown": "forbidden"}),
        lambda value: value["metrics"].reverse(),
        lambda value: value["metrics"][0]["target"].update({"value": 99.9}),
        lambda value: value["metrics"][0]["window"].update({"duration_seconds": 3600}),
    ),
)
def test_contract_shape_targets_and_windows_fail_closed(tmp_path: Path, mutation) -> None:
    raw = _raw_contract()
    mutation(raw)

    with pytest.raises(OperationalObjectivesError, match="schema version 1") as caught:
        load_operational_objectives(_write_contract(tmp_path, raw))
    assert caught.value.code == "contract_invalid_schema"


def test_duplicate_keys_and_yaml_aliases_are_rejected(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.yml"
    duplicate.write_text("schema_version: 1\nschema_version: 1\n", encoding="utf-8")
    alias = tmp_path / "alias.yml"
    alias.write_text("schema_version: &version 1\ncopy: *version\n", encoding="utf-8")

    for path in (duplicate, alias):
        with pytest.raises(OperationalObjectivesError) as caught:
            load_operational_objectives(path)
        assert caught.value.code == "contract_invalid_yaml"


def test_complete_two_party_decision_is_bound_to_the_exact_payload(tmp_path: Path) -> None:
    raw = _approved_contract()
    path = _write_contract(tmp_path, raw)

    validation = load_operational_objectives(path, require_production_approval=True)
    assert validation.production_eligible
    assert validation.readiness_reasons == ()

    raw["metrics"][0]["target"]["value"] = 98.0
    with pytest.raises(OperationalObjectivesError) as caught:
        load_operational_objectives(_write_contract(tmp_path, raw), require_production_approval=True)
    assert caught.value.code == "approval_binding_mismatch"


def test_validation_cli_reports_hashes_but_never_claims_unapproved_targets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert validate_operational_objectives.main([str(CONTRACT)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "valid_unapproved"
    assert payload["production_eligible"] is False
    assert len(payload["contract_sha256"]) == 64
    assert len(payload["decision_payload_sha256"]) == 64

    assert (
        validate_operational_objectives.main(
            [str(CONTRACT), "--require-production-approval"]
        )
        == 2
    )
    failure = json.loads(capsys.readouterr().out)
    assert failure == {
        "error_code": "production_objectives_unapproved",
        "production_eligible": False,
        "schema_version": 1,
        "status": "failed",
    }
