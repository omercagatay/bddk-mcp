#!/usr/bin/env python3
"""Validate the bank production objective decision without inventing targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bddk_mcp.operational_objectives import OperationalObjectivesError, load_operational_objectives

DEFAULT_CONTRACT = Path("docs/decisions/operational-objectives.v1.yml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", nargs="?", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--require-production-approval",
        action="store_true",
        help="fail unless targets, windows, evidence, alerts, retention, and two-party approval are complete",
    )
    return parser


def _json(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validation = load_operational_objectives(
            args.contract,
            require_production_approval=args.require_production_approval,
        )
    except OperationalObjectivesError as error:
        print(
            _json(
                {
                    "schema_version": 1,
                    "status": "failed",
                    "error_code": error.code,
                    "production_eligible": False,
                }
            )
        )
        return 2

    print(
        _json(
            {
                "schema_version": 1,
                "status": "production_eligible" if validation.production_eligible else "valid_unapproved",
                "contract_sha256": validation.contract_sha256,
                "decision_payload_sha256": validation.decision_payload_sha256,
                "production_eligible": validation.production_eligible,
                "readiness_reasons": validation.readiness_reasons,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
