"""Executable preflight for expert-evaluation release evidence.

This command validates the release trust chain and emits only aggregate,
path-free evidence.  It does not run a model and does not promote the existing
Phase 1/2 case sets to release-grade evaluation datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from benchmark.expert_evaluation import (
    EXPERT_EVALUATION_DRAFT_PATH,
    ExpertEvaluationError,
    ExpertEvaluationReleaseError,
    load_expert_evaluation_dataset,
    profile_expert_evaluation_dataset,
    require_expert_dataset_release_ready,
)


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
    now: datetime | None


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
        predecessor_legal_release_checkpoint_path=inputs.predecessor_legal_release_checkpoint,
        trusted_latest_legal_checkpoint_sha256=inputs.trusted_latest_legal_checkpoint_sha256,
        now=inputs.now,
    )
    profile = profile_expert_evaluation_dataset(validation)
    require_expert_dataset_release_ready(validation)
    validated_at = (inputs.now or datetime.now(UTC)).astimezone(UTC).isoformat()
    report = {
        "schema_version": 1,
        "status": "release_preflight_passed",
        "scope": "expert_evaluation_trust_chain_only",
        "model_scores_authorized": False,
        "reason_model_scores_not_authorized": "expert_dataset_execution_not_implemented",
        "validated_at": validated_at,
        "dataset_id": profile.dataset_id,
        "dataset_version": profile.dataset_version,
        "dataset_sha256": profile.dataset_sha256,
        "corpus_manifest_id": profile.corpus_manifest_id,
        "corpus_manifest_sha256": profile.corpus_manifest_sha256,
        "legal_release_checkpoint_sha256": validation.legal_release_checkpoint_sha256,
        "case_count": profile.case_count,
        "evidence_count": profile.evidence_count,
        "release_blocker_counts": profile.release_blocker_counts,
    }
    return {**report, "preflight_sha256": _canonical_sha256(report)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate expert-evaluation release evidence without running a model.")
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
    parser.add_argument("--predecessor-legal-release-checkpoint", type=Path)
    parser.add_argument("--trusted-latest-legal-checkpoint-sha256")
    parser.add_argument("--now", type=datetime.fromisoformat, help=argparse.SUPPRESS)
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
        now=args.now,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = _inputs(args)
    try:
        report = run_release_preflight(inputs)
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
