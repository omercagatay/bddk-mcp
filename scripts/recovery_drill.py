"""Run guarded migration-rehearsal and logical-restore evidence workflows."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from functools import partial
from pathlib import Path
from time import time

import asyncpg

from bddk_mcp.db_identity import assert_database_connection_identity
from bddk_mcp.db_transport import assert_database_transport
from bddk_mcp.operations.recovery import (
    RecoveryDrillError,
    require_disposable_acknowledgement,
    run_backup_restore_drill,
    run_populated_v2_rehearsal,
    validate_disposable_target_name,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Produce privacy-safe evidence from an explicitly marked disposable database. "
            "Database URLs, the guard token, and the exact acknowledgement are read only from environment variables."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rehearsal = subparsers.add_parser(
        "migration-rehearsal",
        help="rehearse the populated-v2 migration and complete vector publication",
    )
    rehearsal.add_argument("--expected-target", required=True)
    rehearsal.add_argument("--report", type=Path)

    restore = subparsers.add_parser(
        "restore-drill",
        help="dump a read-only source snapshot and restore it on an isolated disposable cluster",
    )
    restore.add_argument("--expected-source", required=True)
    restore.add_argument("--expected-admin", required=True)
    restore.add_argument("--target", required=True)
    restore.add_argument("--report", type=Path)
    return parser


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RecoveryDrillError("required_recovery_environment_missing")
    return value


def _write_report(report: str, output: Path | None) -> None:
    if output is None:
        print(report)
        return
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as report_file:
        report_file.write(report)
        report_file.write("\n")


def _failure_report(command: str, target: str, started: int, error_code: str) -> str:
    payload = {
        "schema_version": 2,
        "workflow": command.replace("-", "_"),
        "status": "failed",
        "target_fingerprint_sha256": hashlib.sha256(target.encode("utf-8")).hexdigest(),
        "started_at_epoch": started,
        "elapsed_ms": max(0, int((time() - started) * 1000)),
        "error_code": error_code,
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


async def _run_migration_rehearsal(args: argparse.Namespace, guard: str, acknowledgement: str) -> str:
    schema_owner_dsn = assert_database_transport(_required_environment("BDDK_SCHEMA_OWNER_DATABASE_URL"))
    ingestion_dsn = assert_database_transport(_required_environment("BDDK_INGESTION_DATABASE_URL"))
    schema_pool: asyncpg.Pool | None = None
    ingestion_pool: asyncpg.Pool | None = None
    try:
        schema_pool = await asyncpg.create_pool(schema_owner_dsn, min_size=2, max_size=3, timeout=10)
        ingestion_pool = await asyncpg.create_pool(
            ingestion_dsn,
            min_size=2,
            max_size=3,
            timeout=10,
            init=partial(assert_database_connection_identity, profile="ingestion"),
        )
        evidence = await run_populated_v2_rehearsal(
            schema_pool,
            ingestion_pool,
            expected_target=args.expected_target,
            guard_token=guard,
            acknowledgement=acknowledgement,
        )
        return evidence.to_json()
    finally:
        if ingestion_pool is not None:
            await ingestion_pool.close()
        if schema_pool is not None:
            await schema_pool.close()


async def _run(args: argparse.Namespace) -> str:
    guard = _required_environment("BDDK_RECOVERY_GUARD_TOKEN")
    acknowledgement = _required_environment("BDDK_RECOVERY_ACKNOWLEDGEMENT")
    require_disposable_acknowledgement(acknowledgement)
    if args.command == "migration-rehearsal":
        validate_disposable_target_name(args.expected_target)
        return await _run_migration_rehearsal(args, guard, acknowledgement)
    if args.command == "restore-drill":
        validate_disposable_target_name(args.target)
        evidence = await run_backup_restore_drill(
            source_dsn=_required_environment("BDDK_RECOVERY_SOURCE_DATABASE_URL"),
            admin_dsn=_required_environment("BDDK_RECOVERY_ADMIN_DATABASE_URL"),
            expected_source_database=args.expected_source,
            expected_admin_database=args.expected_admin,
            target_database=args.target,
            guard_token=guard,
            acknowledgement=acknowledgement,
        )
        return evidence.to_json()
    raise RecoveryDrillError("unsupported_recovery_workflow")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    target = args.expected_target if args.command == "migration-rehearsal" else args.target
    started = int(time())
    try:
        report = asyncio.run(_run(args))
        _write_report(report, args.report)
        return 0
    except RecoveryDrillError as error:
        failure = _failure_report(args.command, target, started, error.code)
        try:
            _write_report(failure, args.report)
        except OSError:
            print(failure)
        return 2
    except Exception:
        failure = _failure_report(args.command, target, started, "unexpected_recovery_failure")
        try:
            _write_report(failure, args.report)
        except OSError:
            print(failure)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
