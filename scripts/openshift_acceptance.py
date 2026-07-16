"""Run the offline, secret-free OpenShift acceptance preflight."""

from __future__ import annotations

import argparse
from pathlib import Path

from bddk_mcp.openshift_acceptance import (
    OpenShiftAcceptanceError,
    run_openshift_preflight,
    sanitized_failure_evidence,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate repository-controlled OpenShift deployment contracts. "
            "This command does not contact a cluster and never certifies production readiness."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Secret-free acceptance YAML")
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing deploy/openshift (default: script checkout)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        evidence = run_openshift_preflight(args.config, args.repository_root)
    except OpenShiftAcceptanceError as exc:
        print(sanitized_failure_evidence(exc), end="")
        return 2
    except Exception:
        error = OpenShiftAcceptanceError(
            "unexpected-preflight-failure",
            "repository preflight failed unexpectedly; no bank acceptance evidence was produced",
        )
        print(sanitized_failure_evidence(error), end="")
        return 3
    print(evidence.to_json(), end="")
    return 0 if evidence.status == "preflight_passed_external_gates_pending" else 1


if __name__ == "__main__":
    raise SystemExit(main())
