"""Fail-closed decision contract for bank production service objectives.

The tracked contract is intentionally an unapproved template.  This module
validates definitions and approval evidence; it does not manufacture target
values, infer approval from repository membership, or treat local metrics as
bank-grade telemetry.
"""

from __future__ import annotations

import hashlib
import json
import math
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bddk_mcp.release_yaml import ReleaseYamlError, load_bounded_release_yaml

_MAX_CONTRACT_BYTES = 512 * 1024

MetricId = Literal[
    "service_availability",
    "tool_latency",
    "source_detection_lag",
    "retrieval_publication_lag",
    "maximum_corpus_age",
    "recovery_point_objective",
    "recovery_time_objective",
    "evidence_retention",
]
ApprovalState = Literal["unapproved", "approved"]
ImplementationState = Literal["not_implemented", "implemented_not_verified", "verified"]

_REQUIRED_METRIC_IDS = (
    "service_availability",
    "tool_latency",
    "source_detection_lag",
    "retrieval_publication_lag",
    "maximum_corpus_age",
    "recovery_point_objective",
    "recovery_time_objective",
    "evidence_retention",
)
_REQUIRED_EVIDENCE_CLASSES = (
    "service_objective_observations",
    "corpus_release_manifests_and_signatures",
    "source_acquisition_and_freshness_events",
    "recovery_and_restore_reports",
    "openshift_acceptance_reports",
    "supply_chain_attestations",
    "model_evaluation_reports",
    "objective_approval_records",
)

# These structural properties are part of schema version 1.  Changing one is a
# semantic contract change, not an in-place tuning of a target.
_METRIC_SHAPES: dict[str, dict[str, str]] = {
    "service_availability": {
        "metric_role": "primary",
        "unit": "percent",
        "statistic": "ratio",
        "grain": "eligible_mcp_tool_request",
        "window_kind": "rolling_service_window",
        "comparator": "greater_than_or_equal",
        "source_id": "mcp_request_boundary_telemetry",
    },
    "tool_latency": {
        "metric_role": "guardrail",
        "unit": "milliseconds",
        "statistic": "p95",
        "grain": "successful_mcp_tool_request",
        "window_kind": "rolling_service_window",
        "comparator": "less_than_or_equal",
        "source_id": "mcp_request_boundary_telemetry",
    },
    "source_detection_lag": {
        "metric_role": "driver",
        "unit": "seconds",
        "statistic": "maximum",
        "grain": "eligible_authoritative_source_item",
        "window_kind": "corpus_release",
        "comparator": "less_than_or_equal",
        "source_id": "corpus_document_freshness_events",
    },
    "retrieval_publication_lag": {
        "metric_role": "primary",
        "unit": "seconds",
        "statistic": "maximum",
        "grain": "eligible_corpus_document",
        "window_kind": "corpus_release",
        "comparator": "less_than_or_equal",
        "source_id": "corpus_document_freshness_events",
    },
    "maximum_corpus_age": {
        "metric_role": "guardrail",
        "unit": "seconds",
        "statistic": "maximum",
        "grain": "active_corpus_release",
        "window_kind": "point_in_time",
        "comparator": "less_than_or_equal",
        "source_id": "active_corpus_release_identity",
    },
    "recovery_point_objective": {
        "metric_role": "guardrail",
        "unit": "seconds",
        "statistic": "maximum",
        "grain": "recovery_event",
        "window_kind": "recovery_event",
        "comparator": "less_than_or_equal",
        "source_id": "bank_recovery_point_catalog",
    },
    "recovery_time_objective": {
        "metric_role": "primary",
        "unit": "seconds",
        "statistic": "maximum",
        "grain": "recovery_event",
        "window_kind": "recovery_event",
        "comparator": "less_than_or_equal",
        "source_id": "recovery_drill_evidence",
    },
    "evidence_retention": {
        "metric_role": "guardrail",
        "unit": "days",
        "statistic": "minimum",
        "grain": "required_evidence_class",
        "window_kind": "point_in_time",
        "comparator": "greater_than_or_equal",
        "source_id": "bank_evidence_retention_registry",
    },
}


class OperationalObjectivesError(RuntimeError):
    """Privacy-safe operational-objective validation failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MeasurementWindow(_StrictModel):
    kind: Literal["rolling_service_window", "corpus_release", "point_in_time", "recovery_event"]
    duration_seconds: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _duration_only_applies_to_rolling_windows(self) -> MeasurementWindow:
        if self.kind != "rolling_service_window" and self.duration_seconds is not None:
            raise ValueError("only rolling service windows have a duration")
        return self


class ObjectiveTarget(_StrictModel):
    approval_state: ApprovalState
    comparator: Literal["greater_than_or_equal", "less_than_or_equal"]
    value: float | None = None

    @field_validator("value", mode="before")
    @classmethod
    def _numeric_target_is_not_boolean(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("target value must be numeric, not boolean")
        return value

    @model_validator(mode="after")
    def _value_matches_approval_state(self) -> ObjectiveTarget:
        if self.approval_state == "unapproved" and self.value is not None:
            raise ValueError("unapproved target values must remain unset")
        if self.approval_state == "approved":
            if self.value is None or not math.isfinite(self.value) or self.value <= 0:
                raise ValueError("approved target values must be finite and greater than zero")
        return self


class MetricOwner(_StrictModel):
    accountable_role: Literal["project_owner"]
    operational_role: Literal["bank_operations"]
    evidence_producer_role: Literal[
        "platform_observability",
        "regulatory_ingestion",
        "database_recovery",
        "records_management",
    ]


class EvidenceSource(_StrictModel):
    source_id: Literal[
        "mcp_request_boundary_telemetry",
        "corpus_document_freshness_events",
        "active_corpus_release_identity",
        "bank_recovery_point_catalog",
        "recovery_drill_evidence",
        "bank_evidence_retention_registry",
    ]
    definition: str = Field(min_length=20, max_length=2_000)
    required_fields: tuple[str, ...] = Field(min_length=1, max_length=30)
    verification_state: ImplementationState

    @field_validator("required_fields")
    @classmethod
    def _required_fields_are_unique(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value or len(value) > 100 for value in normalized):
            raise ValueError("source fields must be bounded non-empty identifiers")
        if len(set(normalized)) != len(normalized):
            raise ValueError("source fields must be unique")
        return normalized


class AlertMapping(_StrictModel):
    rule_id: str = Field(pattern=r"^[A-Z][A-Za-z0-9]{2,127}$")
    breach_condition: str = Field(min_length=20, max_length=2_000)
    missing_data_condition: str = Field(min_length=20, max_length=2_000)
    decision_action: str = Field(min_length=20, max_length=1_000)
    route_owner_role: Literal["bank_operations"]
    route_reference: str | None = Field(default=None, min_length=1, max_length=300)
    runbook: str = Field(pattern=r"^docs/[A-Za-z0-9_./-]+\.md$")
    implementation_state: ImplementationState

    @model_validator(mode="after")
    def _route_is_verified_with_the_rule(self) -> AlertMapping:
        if self.implementation_state == "verified" and self.route_reference is None:
            raise ValueError("verified alert mappings require an approved route reference")
        if self.implementation_state == "not_implemented" and self.route_reference is not None:
            raise ValueError("unimplemented alert mappings cannot claim a route")
        return self


class ObjectiveMetric(_StrictModel):
    metric_id: MetricId
    metric_role: Literal["primary", "driver", "guardrail"]
    decision_use: str = Field(min_length=20, max_length=2_000)
    definition: str = Field(min_length=20, max_length=3_000)
    unit: Literal["percent", "milliseconds", "seconds", "days"]
    statistic: Literal["ratio", "p95", "maximum", "minimum"]
    window: MeasurementWindow
    grain: Literal[
        "eligible_mcp_tool_request",
        "successful_mcp_tool_request",
        "eligible_authoritative_source_item",
        "eligible_corpus_document",
        "active_corpus_release",
        "recovery_event",
        "required_evidence_class",
    ]
    numerator: str = Field(min_length=20, max_length=3_000)
    denominator: str = Field(min_length=20, max_length=3_000)
    exclusions: tuple[str, ...] = Field(min_length=1, max_length=30)
    source: EvidenceSource
    owner: MetricOwner
    target: ObjectiveTarget
    alert: AlertMapping
    guardrails: tuple[str, ...] = Field(min_length=1, max_length=20)
    caveats: tuple[str, ...] = Field(min_length=1, max_length=20)

    @field_validator("exclusions", "guardrails", "caveats")
    @classmethod
    def _bounded_unique_statements(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(len(value) < 10 or len(value) > 1_000 for value in normalized):
            raise ValueError("metric statements must contain 10 through 1,000 characters")
        if len(set(normalized)) != len(normalized):
            raise ValueError("metric statements must be unique")
        return normalized


class ApprovalEvidence(_StrictModel):
    role: Literal["project_owner", "bank_operations"]
    subject_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._@/-]{2,199}$")
    approved_at: datetime
    approved_decision_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_record_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("approved_at")
    @classmethod
    def _approval_time_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("approval timestamps must include a timezone")
        if value > datetime.now(UTC):
            raise ValueError("approval timestamps cannot be in the future")
        return value


class DecisionApproval(_StrictModel):
    state: ApprovalState
    change_record_id: str | None = Field(default=None, min_length=3, max_length=200)
    approvals: tuple[ApprovalEvidence, ...] = Field(default=(), max_length=2)

    @model_validator(mode="after")
    def _two_party_approval_is_explicit(self) -> DecisionApproval:
        if self.state == "unapproved":
            if self.change_record_id is not None or self.approvals:
                raise ValueError("unapproved decisions cannot contain approval claims")
            return self

        if self.change_record_id is None:
            raise ValueError("approved decisions require a bank change-record identifier")
        roles = [approval.role for approval in self.approvals]
        if sorted(roles) != ["bank_operations", "project_owner"]:
            raise ValueError("approved decisions require project-owner and bank-operations evidence")
        subjects = [approval.subject_id for approval in self.approvals]
        if len(set(subjects)) != 2:
            raise ValueError("project-owner and bank-operations approvals require separate subjects")
        approval_records = [approval.approval_record_sha256 for approval in self.approvals]
        if len(set(approval_records)) != 2:
            raise ValueError("project-owner and bank-operations approvals require separate approval records")
        return self


class EvidenceRetentionPolicy(_StrictModel):
    required_classes: tuple[str, ...] = Field(min_length=1)
    registry_source_id: Literal["bank_evidence_retention_registry"]
    owner_role: Literal["records_management"]
    legal_hold_rule: str = Field(min_length=20, max_length=1_000)
    disposal_rule: str = Field(min_length=20, max_length=1_000)
    verification_state: ImplementationState

    @model_validator(mode="after")
    def _required_classes_are_exact(self) -> EvidenceRetentionPolicy:
        if self.required_classes != _REQUIRED_EVIDENCE_CLASSES:
            raise ValueError("evidence classes must match the version-1 ordered inventory")
        return self


class OperationalObjectivesContract(_StrictModel):
    """Versioned service-objective definitions and explicit approval state."""

    schema_version: Literal[1]
    decision_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]{2,127}$")
    environment_class: Literal["bank_onprem_openshift_production"]
    purpose: str = Field(min_length=20, max_length=2_000)
    metrics: tuple[ObjectiveMetric, ...] = Field(min_length=8, max_length=8)
    evidence_retention_policy: EvidenceRetentionPolicy
    approval: DecisionApproval

    @model_validator(mode="after")
    def _complete_and_fail_closed(self) -> OperationalObjectivesContract:
        metric_ids = tuple(metric.metric_id for metric in self.metrics)
        if metric_ids != _REQUIRED_METRIC_IDS:
            raise ValueError("metrics must match the version-1 ordered inventory")

        alert_ids = [metric.alert.rule_id for metric in self.metrics]
        if len(set(alert_ids)) != len(alert_ids):
            raise ValueError("alert rule identifiers must be unique")

        for metric in self.metrics:
            expected = _METRIC_SHAPES[metric.metric_id]
            actual = {
                "metric_role": metric.metric_role,
                "unit": metric.unit,
                "statistic": metric.statistic,
                "grain": metric.grain,
                "window_kind": metric.window.kind,
                "comparator": metric.target.comparator,
                "source_id": metric.source.source_id,
            }
            if actual != expected:
                raise ValueError(f"{metric.metric_id} does not match the version-1 metric shape")
            if metric.metric_id == "service_availability" and metric.target.value is not None:
                if metric.target.value > 100:
                    raise ValueError("availability cannot exceed 100 percent")

        if self.approval.state == "unapproved":
            if any(metric.target.approval_state != "unapproved" for metric in self.metrics):
                raise ValueError("unapproved decisions require every target to remain unapproved")
            if any(metric.window.duration_seconds is not None for metric in self.metrics):
                raise ValueError("unapproved decisions cannot contain a rolling-window duration")
            return self

        for metric in self.metrics:
            if metric.target.approval_state != "approved":
                raise ValueError("approved decisions require every target to be approved")
            if metric.window.kind == "rolling_service_window" and metric.window.duration_seconds is None:
                raise ValueError("approved rolling service metrics require an approved window duration")
            if metric.source.verification_state != "verified":
                raise ValueError("approved decisions require every evidence source to be verified")
            if metric.alert.implementation_state != "verified":
                raise ValueError("approved decisions require every alert mapping to be verified")
        service_window_durations = {
            metric.window.duration_seconds for metric in self.metrics if metric.window.kind == "rolling_service_window"
        }
        if len(service_window_durations) != 1:
            raise ValueError("availability and latency must use the same approved rolling-window duration")
        if self.evidence_retention_policy.verification_state != "verified":
            raise ValueError("approved decisions require a verified evidence-retention registry")
        return self


@dataclass(frozen=True, slots=True)
class OperationalObjectivesValidation:
    contract: OperationalObjectivesContract
    contract_sha256: str
    decision_payload_sha256: str
    production_eligible: bool
    readiness_reasons: tuple[str, ...]


def production_readiness_reasons(contract: OperationalObjectivesContract) -> tuple[str, ...]:
    """Return stable, content-free reasons why the contract cannot gate production."""

    reasons: list[str] = []
    if contract.approval.state != "approved":
        reasons.append("decision_unapproved")
    if any(metric.target.approval_state != "approved" or metric.target.value is None for metric in contract.metrics):
        reasons.append("targets_unapproved")
    if any(
        metric.window.kind == "rolling_service_window" and metric.window.duration_seconds is None
        for metric in contract.metrics
    ):
        reasons.append("rolling_windows_unapproved")
    if any(metric.source.verification_state != "verified" for metric in contract.metrics):
        reasons.append("evidence_sources_unverified")
    if any(metric.alert.implementation_state != "verified" for metric in contract.metrics):
        reasons.append("alert_mappings_unverified")
    if contract.evidence_retention_policy.verification_state != "verified":
        reasons.append("evidence_retention_registry_unverified")
    return tuple(reasons)


def canonical_decision_payload_sha256(raw_contract: dict[str, Any]) -> str:
    """Hash all decision semantics while excluding only external approval evidence."""

    try:
        payload = json.loads(json.dumps(raw_contract, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError, RecursionError) as exc:
        raise OperationalObjectivesError(
            "contract_not_canonical",
            "operational-objectives decision payload cannot be canonicalized",
        ) from exc
    payload.pop("approval", None)
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def load_operational_objectives(
    contract_path: Path,
    *,
    require_production_approval: bool = False,
) -> OperationalObjectivesValidation:
    """Read one bounded contract and optionally enforce every production gate."""

    path = contract_path.resolve()
    try:
        metadata = path.stat()
    except FileNotFoundError as exc:
        raise OperationalObjectivesError("contract_missing", "operational-objectives contract is missing") from exc
    if not stat.S_ISREG(metadata.st_mode) or not 1 <= metadata.st_size <= _MAX_CONTRACT_BYTES:
        raise OperationalObjectivesError(
            "contract_not_bounded",
            "operational-objectives contract must be a bounded regular file",
        )
    try:
        raw_bytes = path.read_bytes()
        raw = load_bounded_release_yaml(raw_bytes, maximum_bytes=_MAX_CONTRACT_BYTES)
    except (OSError, UnicodeError, ReleaseYamlError) as exc:
        raise OperationalObjectivesError("contract_invalid_yaml", "operational-objectives YAML is invalid") from exc
    if not isinstance(raw, dict):
        raise OperationalObjectivesError("contract_invalid_schema", "operational-objectives contract must be a mapping")
    try:
        contract = OperationalObjectivesContract.model_validate(raw)
    except (ValueError, RecursionError) as exc:
        raise OperationalObjectivesError(
            "contract_invalid_schema",
            "operational-objectives contract does not satisfy schema version 1",
        ) from exc

    decision_payload_sha256 = canonical_decision_payload_sha256(raw)
    if contract.approval.state == "approved" and any(
        approval.approved_decision_sha256 != decision_payload_sha256 for approval in contract.approval.approvals
    ):
        raise OperationalObjectivesError(
            "approval_binding_mismatch",
            "operational-objective approvals do not bind the exact decision payload",
        )

    reasons = production_readiness_reasons(contract)
    if require_production_approval and reasons:
        raise OperationalObjectivesError(
            "production_objectives_unapproved",
            "production objectives are incomplete or not jointly approved",
        )
    return OperationalObjectivesValidation(
        contract=contract,
        contract_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        decision_payload_sha256=decision_payload_sha256,
        production_eligible=not reasons,
        readiness_reasons=reasons,
    )
