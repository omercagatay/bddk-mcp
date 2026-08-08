"""Executable preflight for expert-evaluation release evidence.

This command validates the release trust chain and emits only aggregate,
path-free evidence.  It does not run a model and does not promote the existing
Phase 1/2 case sets to release-grade evaluation datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from benchmark.evaluation_trust_policy import (
    EvaluationTrustAuthorization,
    EvaluationTrustPolicyError,
    authorize_evaluation_trust_chain,
    load_signed_evaluation_trust_policy,
)
from benchmark.expert_evaluation import (
    EXPERT_EVALUATION_DRAFT_PATH,
    ExpertEvaluationError,
    ExpertEvaluationReleaseError,
    load_expert_evaluation_dataset,
    profile_expert_evaluation_dataset,
    require_expert_dataset_release_ready,
)


class ReleasePreflightArgumentError(ValueError):
    """Raised without echoing an untrusted CLI argument."""


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ReleasePreflightArgumentError("release preflight arguments are invalid")


def _sha256_argument(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise argparse.ArgumentTypeError("invalid SHA-256 value")
    return value


def _positive_policy_version(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("invalid policy version") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("invalid policy version")
    return parsed


def _policy_identifier(value: str) -> str:
    if re.fullmatch(r"[a-z0-9][a-z0-9._:@/-]{2,127}", value) is None:
        raise argparse.ArgumentTypeError("invalid policy scope identifier")
    return value


@dataclass(frozen=True, slots=True)
class ReleasePreflightInputs:
    dataset: Path
    corpus_manifest: Path | None
    corpus_root: Path | None
    trusted_dataset_key: Path | None
    trusted_corpus_key: Path | None
    legal_pack: Path | None
    legal_attestation: Path | None
    trusted_legal_attestation_key: Path | None
    legal_release_checkpoint: Path | None
    legal_release_source_root: Path | None
    trusted_legal_release_key: Path | None
    predecessor_legal_release_checkpoint: Path | None
    trusted_latest_legal_checkpoint_sha256: str | None
    bank_trust_policy: Path | None = None
    bank_trust_policy_signature: Path | None = None
    trusted_bank_policy_key: Path | None = None
    trust_mode: Literal["development", "bank_policy"] = "development"
    trusted_current_bank_policy_sha256: str | None = None
    trusted_current_bank_policy_version: int | None = None
    trusted_legal_release_predecessor_keys: tuple[Path, ...] = ()
    trusted_bank_organization_id: str | None = None
    trusted_bank_environment_id: str | None = None
    trusted_bank_deployment_scope: str | None = None


def _validate_programmatic_policy_pins(inputs: ReleasePreflightInputs) -> None:
    if inputs.trusted_current_bank_policy_sha256 is not None and (
        not isinstance(inputs.trusted_current_bank_policy_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", inputs.trusted_current_bank_policy_sha256) is None
    ):
        raise EvaluationTrustPolicyError("current bank-policy SHA-256 pin is invalid")
    if inputs.trusted_current_bank_policy_version is not None and (
        type(inputs.trusted_current_bank_policy_version) is not int or inputs.trusted_current_bank_policy_version < 1
    ):
        raise EvaluationTrustPolicyError("current bank-policy version pin is invalid")
    for value in (inputs.trusted_bank_organization_id, inputs.trusted_bank_environment_id):
        if value is not None and (
            not isinstance(value, str) or re.fullmatch(r"[a-z0-9][a-z0-9._:@/-]{2,127}", value) is None
        ):
            raise EvaluationTrustPolicyError("bank-policy deployment identity pin is invalid")
    if inputs.trusted_bank_deployment_scope is not None and inputs.trusted_bank_deployment_scope != "bank_production":
        raise EvaluationTrustPolicyError("bank-policy deployment scope pin is invalid")


def _trusted_now() -> datetime:
    """Read the process clock; release callers cannot inject validation time."""

    return datetime.now(UTC)


def _canonical_sha256(value: dict) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_release_preflight(inputs: ReleasePreflightInputs) -> dict:
    """Validate the complete gate and return a content-free evidence report."""

    current = _trusted_now().astimezone(UTC)
    if inputs.trust_mode not in {"development", "bank_policy"}:
        raise EvaluationTrustPolicyError("evaluation trust mode is invalid")
    _validate_programmatic_policy_pins(inputs)
    policy_inputs = (
        inputs.bank_trust_policy,
        inputs.bank_trust_policy_signature,
        inputs.trusted_bank_policy_key,
    )
    if any(value is not None for value in policy_inputs) and not all(value is not None for value in policy_inputs):
        raise EvaluationTrustPolicyError("signed evaluation trust-policy inputs are incomplete")
    policy_head_inputs = (
        inputs.trusted_current_bank_policy_sha256,
        inputs.trusted_current_bank_policy_version,
    )
    policy_scope_inputs = (
        inputs.trusted_bank_organization_id,
        inputs.trusted_bank_environment_id,
        inputs.trusted_bank_deployment_scope,
    )
    if inputs.trust_mode == "bank_policy":
        if not all(value is not None for value in policy_inputs):
            raise EvaluationTrustPolicyError("bank-policy mode requires a signed evaluation trust policy")
        if not all(value is not None for value in policy_head_inputs):
            raise EvaluationTrustPolicyError("bank-policy mode requires a pinned current policy head")
        if not all(value is not None for value in policy_scope_inputs):
            raise EvaluationTrustPolicyError("bank-policy mode requires a pinned deployment scope")
    elif any(value is not None for value in (*policy_head_inputs, *policy_scope_inputs)):
        raise EvaluationTrustPolicyError("bank-policy pins are forbidden in development mode")

    signed_policy = None
    if all(value is not None for value in policy_inputs):
        if inputs.trusted_latest_legal_checkpoint_sha256 is not None:
            raise EvaluationTrustPolicyError("manual latest-head input is forbidden with a signed trust policy")
        signed_policy = load_signed_evaluation_trust_policy(
            inputs.bank_trust_policy,
            inputs.bank_trust_policy_signature,
            inputs.trusted_bank_policy_key,
            current=current,
            expected_organization_id=(
                inputs.trusted_bank_organization_id if inputs.trust_mode == "bank_policy" else None
            ),
            expected_environment_id=(
                inputs.trusted_bank_environment_id if inputs.trust_mode == "bank_policy" else None
            ),
            expected_deployment_scope=(
                inputs.trusted_bank_deployment_scope if inputs.trust_mode == "bank_policy" else None
            ),
        )
        if inputs.trust_mode == "bank_policy" and (
            signed_policy.policy_sha256 != inputs.trusted_current_bank_policy_sha256
            or signed_policy.policy.policy_version != inputs.trusted_current_bank_policy_version
        ):
            raise EvaluationTrustPolicyError("signed evaluation trust policy is not the pinned current policy")
    trusted_latest_checkpoint = (
        signed_policy.policy.approved_release.legal_release_checkpoint_sha256
        if signed_policy is not None
        else inputs.trusted_latest_legal_checkpoint_sha256
    )
    validation = load_expert_evaluation_dataset(
        inputs.dataset,
        corpus_manifest_path=inputs.corpus_manifest,
        corpus_root=inputs.corpus_root,
        trusted_dataset_signing_key=inputs.trusted_dataset_key,
        trusted_corpus_signing_key=inputs.trusted_corpus_key,
        validated_legal_pack_path=inputs.legal_pack,
        legal_attestation_path=inputs.legal_attestation,
        trusted_legal_attestation_key=inputs.trusted_legal_attestation_key,
        legal_release_checkpoint_path=inputs.legal_release_checkpoint,
        legal_release_source_root=inputs.legal_release_source_root,
        trusted_legal_release_signing_key=inputs.trusted_legal_release_key,
        trusted_legal_release_predecessor_signing_keys=(inputs.trusted_legal_release_predecessor_keys),
        predecessor_legal_release_checkpoint_path=inputs.predecessor_legal_release_checkpoint,
        trusted_latest_legal_checkpoint_sha256=trusted_latest_checkpoint,
        now=current,
    )
    profile = profile_expert_evaluation_dataset(validation)
    require_expert_dataset_release_ready(validation)
    policy_authorization: EvaluationTrustAuthorization | None = None
    if signed_policy is not None:
        dataset_authorized_at = validation.dataset.approval.decided_at
        corpus_authorized_at = validation.corpus_validation.manifest.freshness.scope_reviewed_at
        required_policy_values = (
            validation.corpus_validation.signing_key_fingerprint_sha256,
            validation.dataset_signing_key_fingerprint_sha256,
            validation.legal_attestation_key_fingerprint_sha256,
            validation.legal_attestation_attested_at,
            validation.legal_release_signing_key_fingerprint_sha256,
            validation.legal_release_checkpoint_created_at,
            validation.legal_release_checkpoint_sha256,
            validation.legal_pack_sha256,
            validation.legal_attestation_sha256,
            dataset_authorized_at,
            validation.legal_release_chain_signers,
            validation.legal_release_configured_key_fingerprints_sha256,
            validation.legal_source_reviews,
        )
        if any(value is None for value in required_policy_values):
            raise EvaluationTrustPolicyError("release evidence lacks a policy authorization identity")
        policy_authorization = authorize_evaluation_trust_chain(
            signed_policy,
            corpus_signer_fingerprint_sha256=validation.corpus_validation.signing_key_fingerprint_sha256,
            corpus_authorized_at=corpus_authorized_at,
            dataset_signer_fingerprint_sha256=validation.dataset_signing_key_fingerprint_sha256,
            dataset_authorized_at=dataset_authorized_at,
            legal_curator_fingerprint_sha256=validation.legal_attestation_key_fingerprint_sha256,
            legal_curator_authorized_at=validation.legal_attestation_attested_at,
            legal_release_signer_fingerprint_sha256=(validation.legal_release_signing_key_fingerprint_sha256),
            legal_release_authorized_at=validation.legal_release_checkpoint_created_at,
            legal_release_chain_signers=tuple(
                (
                    signer.signing_key_fingerprint_sha256,
                    signer.checkpoint_created_at,
                    signer.checkpoint_sha256,
                )
                for signer in validation.legal_release_chain_signers
            ),
            legal_release_configured_key_fingerprints_sha256=(
                validation.legal_release_configured_key_fingerprints_sha256
            ),
            legal_source_reviews=tuple(
                (
                    review.checkpoint_sha256,
                    review.artifact_id,
                    review.reviewer_owner_id,
                    review.reviewed_at,
                    review.proof_schema_version,
                )
                for review in validation.legal_source_reviews
            ),
            dataset_sha256=profile.dataset_sha256,
            corpus_manifest_sha256=profile.corpus_manifest_sha256,
            legal_pack_sha256=validation.legal_pack_sha256,
            legal_attestation_sha256=validation.legal_attestation_sha256,
            legal_release_checkpoint_sha256=validation.legal_release_checkpoint_sha256,
            current=current,
        )
    validated_at = current.isoformat()
    policy_verified = policy_authorization is not None
    policy_head_pin_verified = policy_verified and inputs.trust_mode == "bank_policy"
    report = {
        "schema_version": 2,
        "status": (
            "configured_policy_head_preflight_passed"
            if policy_head_pin_verified
            else "signed_policy_preflight_passed"
            if policy_verified
            else "cryptographic_preflight_passed"
        ),
        "scope": (
            "configured_root_signed_expert_evaluation_trust_chain"
            if policy_verified
            else "operator_supplied_expert_evaluation_trust_chain"
        ),
        "bank_authorization_verified": False,
        "reason_bank_authorization_not_verified": (
            "bank_controlled_mount_and_promotion_not_attested_by_source_checkout"
            if policy_head_pin_verified
            else "configured_policy_root_ownership_not_attested"
            if policy_verified
            else "bank_signed_trust_policy_not_verified"
        ),
        "trust_mode": inputs.trust_mode,
        "configured_root_policy_signature_verified": policy_verified,
        "policy_approved_release_binding_verified": policy_verified,
        "policy_current_head_pin_verified": policy_head_pin_verified,
        "policy_deployment_scope_pin_verified": policy_head_pin_verified,
        "configured_policy_input_provenance": ("caller_or_deployment_supplied" if policy_verified else "not_supplied"),
        "model_scores_authorized": False,
        "reason_model_scores_not_authorized": "expert_dataset_execution_not_implemented",
        "authorized_capabilities": (["policy_bound_release_evidence_validation"] if policy_verified else []),
        "unsupported_capabilities": [
            "model_score_release_authorization",
            "currentness_scoring",
            "version_comparison_scoring",
            "amendment_tracking_scoring",
        ],
        "self_checksum_algorithm": "sha256_canonical_json_unsigned",
        "validated_at": validated_at,
        "dataset_id": profile.dataset_id,
        "dataset_version": profile.dataset_version,
        "dataset_sha256": profile.dataset_sha256,
        "corpus_manifest_id": profile.corpus_manifest_id,
        "corpus_manifest_sha256": profile.corpus_manifest_sha256,
        "corpus_signing_key_fingerprint_sha256": validation.corpus_validation.signing_key_fingerprint_sha256,
        "dataset_signing_key_fingerprint_sha256": validation.dataset_signing_key_fingerprint_sha256,
        "legal_pack_sha256": validation.legal_pack_sha256,
        "legal_attestation_sha256": validation.legal_attestation_sha256,
        "legal_attestation_key_fingerprint_sha256": validation.legal_attestation_key_fingerprint_sha256,
        "legal_release_checkpoint_sha256": validation.legal_release_checkpoint_sha256,
        "legal_release_signing_key_fingerprint_sha256": validation.legal_release_signing_key_fingerprint_sha256,
        "legal_release_chain_checkpoint_count": validation.legal_release_chain_checkpoint_count,
        "legal_release_chain_signer_fingerprints_sha256": sorted(
            {signer.signing_key_fingerprint_sha256 for signer in validation.legal_release_chain_signers}
        ),
        "legal_release_genesis_checkpoint_sha256": validation.legal_release_genesis_checkpoint_sha256,
        "latest_checkpoint_anchor_provenance": (
            "signed_evaluation_trust_policy" if policy_verified else "caller_supplied_argument"
        ),
        "trust_policy_id": policy_authorization.policy_id if policy_authorization else None,
        "trust_policy_version": policy_authorization.policy_version if policy_authorization else None,
        "trust_policy_sha256": policy_authorization.policy_sha256 if policy_authorization else None,
        "trust_policy_signing_key_fingerprint_sha256": (
            policy_authorization.policy_signing_key_fingerprint_sha256 if policy_authorization else None
        ),
        "trust_policy_valid_until": (
            policy_authorization.policy_valid_until.isoformat() if policy_authorization else None
        ),
        "trust_policy_authorized_owner_count": (
            policy_authorization.authorized_owner_count if policy_authorization else 0
        ),
        "trust_policy_authorized_reviewer_count": (
            policy_authorization.authorized_reviewer_count if policy_authorization else 0
        ),
        "policy_bound_legal_source_reviews_verified": policy_authorization is not None,
        "policy_bound_legal_source_review_count": (
            policy_authorization.policy_bound_legal_source_review_count if policy_authorization else 0
        ),
        "case_count": profile.case_count,
        "evidence_count": profile.evidence_count,
        "release_blocker_counts": profile.release_blocker_counts,
    }
    return {**report, "preflight_self_checksum_sha256": _canonical_sha256(report)}


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(description="Validate expert-evaluation release evidence without running a model.")
    parser.add_argument("--dataset", type=Path, default=EXPERT_EVALUATION_DRAFT_PATH)
    parser.add_argument("--corpus-manifest", type=Path)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--trusted-dataset-key", type=Path)
    parser.add_argument("--trusted-corpus-key", type=Path)
    parser.add_argument("--legal-pack", type=Path)
    parser.add_argument("--legal-attestation", type=Path)
    parser.add_argument("--trusted-legal-attestation-key", type=Path)
    parser.add_argument("--legal-release-checkpoint", type=Path)
    parser.add_argument("--legal-release-source-root", type=Path)
    parser.add_argument("--trusted-legal-release-key", type=Path)
    parser.add_argument("--trusted-legal-release-predecessor-key", type=Path, action="append", default=[])
    parser.add_argument("--predecessor-legal-release-checkpoint", type=Path)
    parser.add_argument("--trusted-latest-legal-checkpoint-sha256")
    parser.add_argument("--bank-trust-policy", type=Path)
    parser.add_argument("--bank-trust-policy-signature", type=Path)
    parser.add_argument("--trusted-bank-policy-key", type=Path)
    parser.add_argument("--trust-mode", choices=("development", "bank-policy"), default="development")
    parser.add_argument("--trusted-current-bank-policy-sha256", type=_sha256_argument)
    parser.add_argument("--trusted-current-bank-policy-version", type=_positive_policy_version)
    parser.add_argument("--trusted-bank-organization-id", type=_policy_identifier)
    parser.add_argument("--trusted-bank-environment-id", type=_policy_identifier)
    parser.add_argument("--trusted-bank-deployment-scope", choices=("bank-production",))
    return parser


def _inputs(args: argparse.Namespace) -> ReleasePreflightInputs:
    return ReleasePreflightInputs(
        dataset=args.dataset,
        corpus_manifest=args.corpus_manifest,
        corpus_root=args.corpus_root,
        trusted_dataset_key=args.trusted_dataset_key,
        trusted_corpus_key=args.trusted_corpus_key,
        legal_pack=args.legal_pack,
        legal_attestation=args.legal_attestation,
        trusted_legal_attestation_key=args.trusted_legal_attestation_key,
        legal_release_checkpoint=args.legal_release_checkpoint,
        legal_release_source_root=args.legal_release_source_root,
        trusted_legal_release_key=args.trusted_legal_release_key,
        predecessor_legal_release_checkpoint=args.predecessor_legal_release_checkpoint,
        trusted_latest_legal_checkpoint_sha256=args.trusted_latest_legal_checkpoint_sha256,
        bank_trust_policy=args.bank_trust_policy,
        bank_trust_policy_signature=args.bank_trust_policy_signature,
        trusted_bank_policy_key=args.trusted_bank_policy_key,
        trust_mode=args.trust_mode.replace("-", "_"),
        trusted_current_bank_policy_sha256=args.trusted_current_bank_policy_sha256,
        trusted_current_bank_policy_version=args.trusted_current_bank_policy_version,
        trusted_bank_organization_id=args.trusted_bank_organization_id,
        trusted_bank_environment_id=args.trusted_bank_environment_id,
        trusted_bank_deployment_scope=(
            args.trusted_bank_deployment_scope.replace("-", "_")
            if args.trusted_bank_deployment_scope is not None
            else None
        ),
        trusted_legal_release_predecessor_keys=tuple(args.trusted_legal_release_predecessor_key),
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        inputs = _inputs(args)
        report = run_release_preflight(inputs)
    except ReleasePreflightArgumentError:
        failure = {
            "schema_version": 1,
            "status": "release_preflight_failed",
            "error_code": "RELEASE_PREFLIGHT_ARGUMENTS_INVALID",
            "model_scores_authorized": False,
        }
        print(json.dumps(failure, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 64
    except ExpertEvaluationReleaseError:
        # Reloading is unnecessary: the release exception is based entirely on
        # aggregate blocker counts already covered by focused library tests.
        failure = {
            "schema_version": 1,
            "status": "release_preflight_failed",
            "error_code": "EXPERT_EVALUATION_NOT_RELEASE_READY",
            "model_scores_authorized": False,
        }
        print(json.dumps(failure, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 1
    except ExpertEvaluationError:
        failure = {
            "schema_version": 1,
            "status": "release_preflight_failed",
            "error_code": "EXPERT_EVALUATION_VALIDATION_FAILED",
            "model_scores_authorized": False,
        }
        print(json.dumps(failure, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    except EvaluationTrustPolicyError:
        failure = {
            "schema_version": 1,
            "status": "release_preflight_failed",
            "error_code": "EVALUATION_TRUST_POLICY_VALIDATION_FAILED",
            "bank_authorization_verified": False,
            "model_scores_authorized": False,
        }
        print(json.dumps(failure, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 4
    except Exception:
        # The preflight is often executed in CI or an operator job.  Never let
        # an unexpected filesystem/provider exception print paths, inputs, or
        # document content through Python's default traceback.
        failure = {
            "schema_version": 1,
            "status": "release_preflight_failed",
            "error_code": "RELEASE_PREFLIGHT_INTERNAL_ERROR",
            "model_scores_authorized": False,
        }
        print(json.dumps(failure, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 3
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())


__all__ = ("ReleasePreflightInputs", "main", "run_release_preflight")
