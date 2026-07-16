"""Safety contracts and opt-in PostgreSQL proof for recovery workflows."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from dataclasses import replace

import pytest

from bddk_mcp.migrations import LATEST_SCHEMA_VERSION, MIGRATIONS
from bddk_mcp.migrations.v0005_corpus_release_publication import CORPUS_EPOCH_TRACKED_TABLES
from bddk_mcp.migrations.v0007_retained_corpus_generations import RETAINED_CORPUS_RELATIONS
from bddk_mcp.operations import recovery
from bddk_mcp.operations.recovery import (
    DISPOSABLE_ACKNOWLEDGEMENT,
    IdentitySequenceEvidence,
    RecoveryDrillError,
    RecoveryEvidence,
    RelationEvidence,
    SnapshotEvidence,
    _database_dsn,
    _parse_pg_environment,
    _run_pg_tool,
    assert_disposable_database_target,
    collect_snapshot_evidence,
    require_disposable_acknowledgement,
    run_populated_v2_rehearsal,
    validate_admin_database_name,
    validate_disposable_target_name,
)
from bddk_mcp.store.vector_store import retrieval_profile_hash
from scripts.recovery_drill import _failure_report, build_parser


class _GuardPool:
    def __init__(self, row):
        self.row = row

    async def fetchrow(self, _query, *_args):
        return self.row


class _PinnedPool(recovery._PinnedPool):
    """Test alias for the production pinned-connection contract."""


def _sequence() -> IdentitySequenceEvidence:
    return IdentitySequenceEvidence(
        sequence_name="bddk_meta.corpus_release_activations_activation_sequence_seq",
        owned_by="bddk_meta.corpus_release_activations.activation_sequence",
        identity_generation="always",
        last_value=7,
        is_called=True,
        start_value=1,
        increment_by=1,
        minimum_value=1,
        maximum_value=9_223_372_036_854_775_807,
        cache_size=1,
        cycle=False,
        next_candidate=8,
        maximum_retained_activation=7,
    )


def _snapshot(fingerprint: str = "a" * 64) -> SnapshotEvidence:
    return SnapshotEvidence(
        migration_version=3,
        migration_checksum="b" * 64,
        database_encoding="UTF8",
        database_collation="C",
        database_character_classification="C",
        database_locale_provider="c",
        database_locale=None,
        database_icu_rules=None,
        database_collation_version="2.36",
        database_collation_actual_version="2.36",
        logical_fingerprint_sha256=fingerprint,
        database_bytes=100,
        wal_lsn="0/10",
        relations={"public.documents": RelationEvidence(rows=1, heap_bytes=10, total_bytes=20)},
        catalog_valid=True,
        catalog_failures=(),
        readiness_ready=True,
        readiness_issues=(),
        active_corpus_release_id="corpus_release_sha256_" + "d" * 64,
        activation_sequence=_sequence(),
    )


def test_disposable_acknowledgement_is_exact() -> None:
    require_disposable_acknowledgement(DISPOSABLE_ACKNOWLEDGEMENT)

    for invalid in ("", "yes", DISPOSABLE_ACKNOWLEDGEMENT.lower(), DISPOSABLE_ACKNOWLEDGEMENT + " "):
        with pytest.raises(RecoveryDrillError, match="disposable_acknowledgement_required"):
            require_disposable_acknowledgement(invalid)


@pytest.mark.parametrize(
    "name",
    (
        "bddk_v2_rehearsal_scale01",
        "bddk_restore_drill_20260715",
    ),
)
def test_disposable_target_names_are_unmistakable(name: str) -> None:
    assert validate_disposable_target_name(name) == name


@pytest.mark.parametrize(
    "name",
    (
        "bddk",
        "bddk_prod",
        "bddk_test",
        "bddk_restore_drill_",
        "BDDk_restore_drill_x",
        "bddk_restore_drill_x;drop_database",
    ),
)
def test_non_disposable_target_names_are_rejected(name: str) -> None:
    with pytest.raises(RecoveryDrillError, match="unsafe_disposable_target_name"):
        validate_disposable_target_name(name)


def test_admin_database_name_is_dedicated() -> None:
    assert validate_admin_database_name("bddk_recovery_admin") == "bddk_recovery_admin"
    assert validate_admin_database_name("bddk_recovery_admin_ci1") == "bddk_recovery_admin_ci1"
    with pytest.raises(RecoveryDrillError, match="unsafe_recovery_admin_database"):
        validate_admin_database_name("postgres")


@pytest.mark.asyncio
async def test_database_guard_requires_name_and_independent_hash() -> None:
    token = "recovery-guard-token-that-is-longer-than-32-characters"
    target = "bddk_v2_rehearsal_scale01"
    row = {
        "database_name": target,
        "guard_hash": hashlib.sha256(token.encode()).hexdigest(),
    }

    await assert_disposable_database_target(_GuardPool(row), target, token)

    for invalid_row in (
        {**row, "database_name": "bddk_v2_rehearsal_other"},
        {**row, "guard_hash": "0" * 64},
        None,
    ):
        with pytest.raises(RecoveryDrillError, match="disposable_database_guard_failed"):
            await assert_disposable_database_target(_GuardPool(invalid_row), target, token)


def test_pg_url_is_translated_to_environment_without_argv_credentials() -> None:
    dsn = (
        "postgresql://backup-user:super-secret@db.example.test:5433/source_db"
        "?sslmode=verify-full&sslrootcert=%2Fbank%2Fca.pem"
    )
    parsed = _parse_pg_environment(dsn)

    assert parsed.database_name == "source_db"
    assert parsed.environment["PGPASSWORD"] == "super-secret"
    assert parsed.environment["PGSSLROOTCERT"] == "/bank/ca.pem"
    assert dsn not in repr(parsed)
    assert "super-secret" not in repr(parsed)

    with pytest.raises(RecoveryDrillError, match="unsupported_database_url"):
        _parse_pg_environment(dsn + "&options=-c%20role%3Dsuperuser")


def test_target_dsn_replaces_database_and_encodes_ephemeral_login() -> None:
    dsn = _database_dsn(
        "postgresql://admin:old@restore.example.test:5432/bddk_recovery_admin?sslmode=verify-full",
        "bddk_restore_drill_20260715",
        username="temporary user",
        password="new:/secret",
        role="bddk_schema_owner",
    )

    assert "temporary%20user:new%3A%2Fsecret@" in dsn
    assert "/bddk_restore_drill_20260715?" in dsn
    assert "role%3Dbddk_schema_owner" in dsn
    assert "old" not in dsn


@pytest.mark.parametrize("variable", recovery._RUNTIME_DATABASE_URL_VARIABLES)
def test_recovery_admin_rejects_every_runtime_endpoint_identity(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    for name in recovery._RUNTIME_DATABASE_URL_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        variable,
        "postgres://runtime%20user:runtime-password@DB.BANK.EXAMPLE/bddk%5Fruntime"
        "?sslmode=require&application_name=serving",
    )

    with pytest.raises(RecoveryDrillError, match="recovery_admin_reuses_runtime_identity"):
        recovery._assert_dsn_not_runtime(
            "postgresql://runtime%20user:different-password@db.bank.example:5432/bddk_runtime?sslmode=verify-full"
        )


def test_recovery_admin_endpoint_identity_preserves_database_user_and_nondefault_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in recovery._RUNTIME_DATABASE_URL_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        "BDDK_DATABASE_URL",
        "postgresql://runtime@db.bank.example:5433/bddk_runtime",
    )

    recovery._assert_dsn_not_runtime("postgresql://admin@db.bank.example:5433/bddk_runtime")
    recovery._assert_dsn_not_runtime("postgresql://runtime@db.bank.example:5432/bddk_runtime")
    recovery._assert_dsn_not_runtime("postgresql://runtime@db.bank.example:5433/bddk_recovery_admin")


@pytest.mark.asyncio
async def test_pg_tool_keeps_database_credentials_out_of_process_arguments(monkeypatch) -> None:
    captured: dict = {}

    class _Process:
        async def wait(self):
            return 0

    async def fake_subprocess(*args, **kwargs):
        captured["args"] = args
        captured["env"] = kwargs["env"]
        return _Process()

    monkeypatch.setattr(recovery.shutil, "which", lambda _name: "/usr/bin/pg_dump")
    monkeypatch.setattr(recovery.asyncio, "create_subprocess_exec", fake_subprocess)
    environment = {"PATH": "/usr/bin", "PGDATABASE": "source", "PGPASSWORD": "secret-value"}

    evidence = await _run_pg_tool(
        "pg_dump",
        ["--format=custom", "--file=/tmp/snapshot.dump"],
        environment,
        failure_code="logical_backup_failed",
    )

    assert evidence.elapsed_ms >= 0
    assert "secret-value" not in repr(captured["args"])
    assert captured["env"]["PGPASSWORD"] == "secret-value"


def test_recovery_fingerprints_hash_actual_text_and_embedding_serialization() -> None:
    queries = dict(recovery._SAFE_FINGERPRINT_QUERIES)

    for label, column in (
        ("documents", "markdown_content"),
        ("sections", "content"),
        ("chunks", "chunk_text"),
    ):
        query = " ".join(queries[label].split())
        assert (
            f"pg_catalog.encode( pg_catalog.sha256( pg_catalog.convert_to(COALESCE({column}, ''), 'UTF8') ), "
            "'hex' ) AS content_digest"
        ) in query
    chunk_query = " ".join(queries["chunks"].split())
    assert "pg_catalog.encode( pg_catalog.sha256(public.vector_send(embedding)), 'hex' )" in chunk_query
    assert "has_embedding" not in chunk_query
    assert all("pg_catalog.md5(" not in query for query in queries.values())
    assert (
        "pg_catalog.encode( pg_catalog.sha256(COALESCE(pdf_blob, ''::pg_catalog.bytea)), 'hex' ) AS pdf_digest"
        in " ".join(queries["documents"].split())
    )


@pytest.mark.asyncio
async def test_relation_fingerprint_streams_without_materializing_rows() -> None:
    class _Row(dict):
        pass

    class _Connection:
        def cursor(self, _query, *, prefetch):
            assert prefetch == 17

            async def rows():
                for value in range(10_000):
                    yield _Row(value=value)

            return rows()

    hasher = hashlib.sha256()
    count = await recovery._stream_relation_hash(
        _Connection(),
        hasher,
        "scale_fixture",
        "SELECT value ORDER BY value",
        prefetch=17,
    )

    assert count == 10_000
    assert len(hasher.hexdigest()) == 64


@pytest.mark.asyncio
async def test_pg_tool_timeout_terminates_then_kills_and_sanitizes(monkeypatch) -> None:
    never_finishes = asyncio.Event()

    class _Process:
        def __init__(self) -> None:
            self.terminate_calls = 0
            self.kill_calls = 0
            self.killed = False

        async def wait(self):
            if self.killed:
                return -9
            await never_finishes.wait()
            return 0

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1
            self.killed = True

    process = _Process()

    async def fake_subprocess(*_args, **_kwargs):
        return process

    monkeypatch.setattr(recovery.shutil, "which", lambda _name: "/usr/bin/pg_dump")
    monkeypatch.setattr(recovery.asyncio, "create_subprocess_exec", fake_subprocess)
    monkeypatch.setattr(recovery, "_PG_TOOL_TERMINATION_GRACE_SECONDS", 0.001)

    with pytest.raises(RecoveryDrillError) as exc_info:
        await _run_pg_tool(
            "pg_dump",
            ["--file=/private/snapshot.dump"],
            {"PGPASSWORD": "private-secret"},
            failure_code="logical_backup_failed",
            timeout_seconds=0.001,
        )

    assert exc_info.value.code == "pg_tool_timed_out"
    assert str(exc_info.value) == "pg_tool_timed_out"
    assert "private" not in str(exc_info.value)
    assert process.terminate_calls == 1
    assert process.kill_calls == 1


@pytest.mark.parametrize("value", ["29", "21601", "1.5", "secret-value"])
def test_pg_tool_timeout_environment_is_bounded_and_sanitized(monkeypatch, value: str) -> None:
    monkeypatch.setenv("BDDK_RECOVERY_PG_TOOL_TIMEOUT_SECONDS", value)

    with pytest.raises(RecoveryDrillError) as exc_info:
        recovery._configured_pg_tool_timeout_seconds()

    assert exc_info.value.code == "pg_tool_timeout_configuration_invalid"
    assert value not in str(exc_info.value)


def test_report_schema_contains_no_target_name_secret_or_corpus_text() -> None:
    evidence = RecoveryEvidence(
        schema_version=2,
        workflow="logical_backup_restore_drill",
        status="passed",
        target_fingerprint_sha256=hashlib.sha256(b"bddk_restore_drill_private").hexdigest(),
        started_at_epoch=1,
        elapsed_ms=2,
        source=_snapshot(),
        restored=_snapshot(),
        dump_bytes=10,
        dump_sha256="c" * 64,
        identities_verified=True,
    )
    report = evidence.to_json()

    json.loads(report)
    assert "bddk_restore_drill_private" not in report
    assert "password" not in report
    assert "regulatory corpus body" not in report
    assert "postgresql://" not in report
    payload = json.loads(report)
    assert payload["schema_version"] == 2
    assert set(payload) == {
        "backup_elapsed_ms",
        "default_refusal_proved",
        "dump_bytes",
        "dump_sha256",
        "elapsed_ms",
        "identities_verified",
        "lock_samples",
        "maximum_lock_waiters",
        "migration_elapsed_ms",
        "reindex_current",
        "reindex_elapsed_ms",
        "reindex_published",
        "reindex_scanned",
        "restore_elapsed_ms",
        "restored",
        "schema_version",
        "source",
        "started_at_epoch",
        "status",
        "target_fingerprint_sha256",
        "wal_generated_bytes",
        "workflow",
    }
    assert set(payload["source"]) == {
        "activation_sequence",
        "active_corpus_release_id",
        "catalog_failures",
        "catalog_valid",
        "database_bytes",
        "database_character_classification",
        "database_collation_actual_version",
        "database_collation_version",
        "database_collation",
        "database_encoding",
        "database_icu_rules",
        "database_locale",
        "database_locale_provider",
        "logical_fingerprint_sha256",
        "migration_checksum",
        "migration_version",
        "readiness_issues",
        "readiness_ready",
        "relations",
        "wal_lsn",
    }
    assert set(payload["source"]["activation_sequence"]) == {
        "cache_size",
        "cycle",
        "identity_generation",
        "increment_by",
        "is_called",
        "last_value",
        "maximum_retained_activation",
        "maximum_value",
        "minimum_value",
        "next_candidate",
        "owned_by",
        "sequence_name",
        "start_value",
    }


def test_failure_report_is_bounded_and_hashes_target() -> None:
    report = _failure_report(
        "restore-drill",
        "bddk_restore_drill_private",
        1,
        "logical_restore_failed",
    )
    payload = json.loads(report)

    assert payload["status"] == "failed"
    assert payload["error_code"] == "logical_restore_failed"
    assert "bddk_restore_drill_private" not in report


def test_cli_never_accepts_database_urls_or_guard_secrets_as_arguments() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    source = inspect.getsource(recovery.run_backup_restore_drill)

    assert "--dsn" not in help_text
    assert "--guard-token" not in help_text
    assert "DROP DATABASE" not in source
    assert "--no-owner" in source
    assert "--no-privileges" in source
    assert "--single-transaction" in source


def test_recovery_evidence_covers_legal_version_relations_in_fk_safe_order() -> None:
    expected = (
        "public.regulatory_instruments",
        "public.regulatory_family_imports",
        "public.regulatory_source_blobs",
        "public.regulatory_source_artifacts",
        "public.regulatory_evidence",
        "public.regulatory_legal_versions",
        "public.regulatory_legal_version_artifacts",
        "public.regulatory_legal_events",
        "public.regulatory_legal_status_assertions",
        "public.regulatory_provisions",
        "public.regulatory_legal_version_provisions",
        "public.regulatory_validated_section_citations",
    )
    inventory_positions = tuple(recovery._MANAGED_RELATIONS.index(relation) for relation in expected)
    fingerprint_labels = {label for label, _query in recovery._SAFE_FINGERPRINT_QUERIES}

    assert inventory_positions == tuple(sorted(inventory_positions))
    assert len(recovery._MANAGED_RELATIONS) == len(set(recovery._MANAGED_RELATIONS))
    assert {relation.removeprefix("public.") for relation in expected} <= fingerprint_labels
    for relation in expected:
        label = relation.removeprefix("public.")
        query = dict(recovery._SAFE_FINGERPRINT_QUERIES)[label]
        assert f"FROM {relation}" in query
        assert "ORDER BY" in query


def test_recovery_inventory_covers_release_epoch_views_and_identity_sequence() -> None:
    assert len(recovery._MANAGED_RELATIONS) == 51
    assert {
        "bddk_meta.corpus_state_epoch",
        "bddk_meta.corpus_releases",
        "bddk_meta.corpus_release_activations",
        "bddk_meta.active_corpus_release",
        "bddk_meta.corpus_release_activations_activation_sequence_seq",
    } <= set(recovery._MANAGED_RELATIONS)
    fingerprint_labels = {label for label, _query in recovery._SAFE_FINGERPRINT_QUERIES}
    assert {
        "corpus_state_epoch",
        "corpus_releases",
        "corpus_release_activations",
        "active_corpus_release",
        "corpus_release_activation_sequence",
    } <= fingerprint_labels


def test_recovery_evidence_covers_retained_generations_in_dependency_order() -> None:
    retained_members = tuple(f"bddk_retained.{relation}" for relation in RETAINED_CORPUS_RELATIONS)
    retained_evidence = (
        "bddk_meta.corpus_generations",
        *retained_members,
        "bddk_meta.corpus_generation_relation_inventory",
        "bddk_meta.corpus_generation_seals",
        "bddk_meta.corpus_retained_releases",
        "bddk_meta.corpus_release_retention_status",
    )
    inventory_positions = tuple(recovery._MANAGED_RELATIONS.index(relation) for relation in retained_evidence)
    queries = dict(recovery._SAFE_FINGERPRINT_QUERIES)
    query_labels = tuple(label for label, _query in recovery._SAFE_FINGERPRINT_QUERIES)
    fingerprint_order = (
        "corpus_generations",
        *(f"retained_{relation}" for relation in RETAINED_CORPUS_RELATIONS),
        "corpus_generation_relation_inventory",
        "corpus_generation_seals",
        "corpus_retained_releases",
        "corpus_release_retention_status",
    )
    fingerprint_positions = tuple(query_labels.index(label) for label in fingerprint_order)

    assert inventory_positions == tuple(sorted(inventory_positions))
    assert fingerprint_positions == tuple(sorted(fingerprint_positions))
    assert len(retained_members) == 17
    assert len(retained_members) == len(set(retained_members))
    for relation in RETAINED_CORPUS_RELATIONS:
        query = " ".join(queries[f"retained_{relation}"].split())
        assert f"FROM bddk_retained.{relation} AS member" in query
        assert "bddk_meta.retained_row_sha256(member, false)" in query
        assert "AS row_sha256" in query
        assert query.endswith("ORDER BY row_sha256")
        assert "SELECT *" not in query.upper()

    assert "FROM bddk_meta.corpus_generation_seals" in queries["corpus_generation_seals"]
    assert "inventory_sha256" in queries["corpus_generation_seals"]
    assert "FROM bddk_meta.corpus_retained_releases" in queries["corpus_retained_releases"]
    assert "seal_id" in queries["corpus_retained_releases"]
    assert "FROM bddk_meta.corpus_release_retention_status" in queries["corpus_release_retention_status"]


def test_retained_generation_seal_gate_recomputes_the_exact_v7_inventory_contract() -> None:
    query = " ".join(recovery._RETAINED_GENERATION_SEAL_VALIDATION_SQL.split())

    assert query.count("FROM bddk_retained.") == 2 * len(RETAINED_CORPUS_RELATIONS)
    for position, relation in enumerate(RETAINED_CORPUS_RELATIONS, start=1):
        assert f"{position}::pg_catalog.int4 AS relation_position" in query
        assert f"'{relation}'::pg_catalog.text AS relation_name" in query
        assert f"FROM bddk_retained.{relation} AS member" in query
        assert f"FROM bddk_retained.{relation} AS member LEFT JOIN bddk_meta.corpus_generations AS generation" in query
    assert "bddk_meta.retained_row_sha256(member, true)" in query
    assert "pg_catalog.string_agg(row_sha256, '' ORDER BY row_sha256)" in query
    assert "ORDER BY fresh.relation_position" in query
    assert "bddk_meta.corpus_fingerprint_frame('1')" in query
    assert "bddk_meta.retained_corpus_state_sha256(" in query
    assert "inventory.row_count IS DISTINCT FROM fresh.row_count" in query
    assert "inventory.relation_sha256 IS DISTINCT FROM fresh.relation_sha256" in query
    assert "sealed_inventory_sha256 IS DISTINCT FROM fresh_inventory_sha256" in query
    assert "binding.release_id = validation.source_release_id" in query
    assert "seal.generation_id = binding.generation_id" in query
    assert "seal.corpus_state_sha256 = binding.corpus_state_sha256" in query
    assert "release.release_id = binding.release_id" in query
    assert "release.retrieval_profile_sha256 = binding.retrieval_profile_sha256" in query
    assert "SELECT member.*" not in query


@pytest.mark.asyncio
async def test_retained_generation_seal_gate_is_boolean_and_sanitizes_failures() -> None:
    class _Pool:
        def __init__(self, result):
            self.result = result

        async def fetchval(self, query):
            assert query == recovery._RETAINED_GENERATION_SEAL_VALIDATION_SQL
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

    await recovery._assert_retained_generation_seals(_Pool(True))

    for invalid in (False, None, 1, RuntimeError("private generation and document identifiers")):
        with pytest.raises(RecoveryDrillError) as captured:
            await recovery._assert_retained_generation_seals(_Pool(invalid))
        assert captured.value.code == "retained_generation_seal_invalid"
        assert str(captured.value) == "retained_generation_seal_invalid"
        assert captured.value.__cause__ is None
        assert "private" not in str(captured.value)


@pytest.mark.asyncio
async def test_identity_sequence_evidence_requires_collision_safe_generated_always_contract() -> None:
    row = {
        "last_value": 7,
        "is_called": True,
        "seqstart": 1,
        "seqincrement": 1,
        "seqmax": 9_223_372_036_854_775_807,
        "seqmin": 1,
        "seqcache": 1,
        "seqcycle": False,
        "attidentity": "a",
        "deptype": "i",
        "maximum_activation_sequence": 7,
    }

    class _Pool:
        async def fetch(self, _query):
            return [row]

    assert await recovery._collect_activation_sequence_evidence(_Pool()) == _sequence()

    class _AsyncpgCatalogPool:
        async def fetch(self, _query):
            return [{**row, "attidentity": b"a", "deptype": b"i"}]

    assert await recovery._collect_activation_sequence_evidence(_AsyncpgCatalogPool()) == _sequence()

    invalid_cases = (
        {**row, "attidentity": "d"},
        {**row, "seqcache": 2},
        {**row, "seqcycle": True},
        {**row, "seqincrement": -1},
    )
    for invalid in invalid_cases:

        class _InvalidPool:
            async def fetch(self, _query, value=invalid):
                return [value]

        with pytest.raises(RecoveryDrillError, match="identity_sequence_contract_invalid"):
            await recovery._collect_activation_sequence_evidence(_InvalidPool())

    class _CollisionPool:
        async def fetch(self, _query):
            return [{**row, "last_value": 7, "is_called": False}]

    with pytest.raises(RecoveryDrillError, match="identity_sequence_collision_risk"):
        await recovery._collect_activation_sequence_evidence(_CollisionPool())


def test_logical_snapshot_requires_same_non_null_release_and_sequence_state() -> None:
    baseline = _snapshot()
    assert recovery._same_logical_snapshot(baseline, baseline)
    assert not recovery._same_logical_snapshot(
        baseline,
        replace(baseline, active_corpus_release_id=None),
    )
    assert not recovery._same_logical_snapshot(
        baseline,
        replace(baseline, activation_sequence=replace(_sequence(), last_value=8, next_candidate=9)),
    )
    assert not recovery._same_logical_snapshot(
        baseline,
        replace(baseline, database_collation="tr_TR.UTF-8"),
    )
    assert not recovery._same_logical_snapshot(
        baseline,
        replace(baseline, database_locale_provider="i", database_locale="tr-TR"),
    )


def test_database_locale_evidence_covers_pg17_provider_rules_and_versions() -> None:
    query = " ".join(recovery._DATABASE_LOCALE_SQL.split())

    for field in (
        "datlocprovider",
        "datlocale",
        "daticurules",
        "datcollversion",
        "pg_database_collation_actual_version",
    ):
        assert field in query


async def _downgrade_to_v2(connection) -> None:
    await connection.execute("DROP SCHEMA IF EXISTS bddk_retained CASCADE")
    await connection.execute("DROP VIEW IF EXISTS bddk_meta.corpus_release_retention_status")
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.inspect_retained_generation_storage(pg_catalog.text)")
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.retain_active_corpus_generation(pg_catalog.text)")
    await connection.execute(
        "DROP FUNCTION IF EXISTS bddk_meta.retained_corpus_state_sha256(pg_catalog.text, pg_catalog.text)"
    )
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.retained_row_sha256(anyelement, pg_catalog.bool)")
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.guard_retained_generation_member() CASCADE")
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.reject_retained_generation_mutation() CASCADE")
    await connection.execute(
        "DROP TABLE IF EXISTS bddk_meta.corpus_retained_releases, "
        "bddk_meta.corpus_generation_seals, "
        "bddk_meta.corpus_generation_relation_inventory, "
        "bddk_meta.corpus_generations CASCADE"
    )
    await connection.execute(
        "ALTER TABLE bddk_meta.corpus_releases DROP CONSTRAINT IF EXISTS corpus_releases_retention_identity_uq"
    )
    await connection.execute(
        "ALTER TABLE bddk_meta.corpus_release_activations "
        "DROP CONSTRAINT IF EXISTS corpus_release_activations_retention_identity_uq"
    )
    await connection.execute("DELETE FROM bddk_meta.schema_migrations WHERE version = 7")
    await connection.execute(
        "DROP FUNCTION IF EXISTS bddk_meta.resolve_regulation_status(pg_catalog.text, pg_catalog.date)"
    )
    await connection.execute("DELETE FROM bddk_meta.schema_migrations WHERE version = 6")
    await connection.execute("DROP VIEW IF EXISTS bddk_meta.active_corpus_release")
    for table_name in CORPUS_EPOCH_TRACKED_TABLES:
        await connection.execute(f"DROP TRIGGER IF EXISTS bump_corpus_state_epoch_on_change ON public.{table_name}")
    await connection.execute(
        "DROP TABLE IF EXISTS bddk_meta.corpus_release_activations, "
        "bddk_meta.corpus_releases, bddk_meta.corpus_state_epoch CASCADE"
    )
    await connection.execute(
        "DROP FUNCTION IF EXISTS bddk_meta.publish_verified_corpus_release("
        "pg_catalog.text, pg_catalog.text, pg_catalog.text, pg_catalog.int4, "
        "pg_catalog.int4, pg_catalog.int4, pg_catalog.text)"
    )
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.corpus_retrieval_ready(pg_catalog.text)")
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.current_corpus_state_sha256(pg_catalog.text)")
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.corpus_fingerprint_frame(pg_catalog.text)")
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.bump_corpus_state_epoch()")
    await connection.execute("DROP FUNCTION IF EXISTS bddk_meta.reject_corpus_release_mutation()")
    await connection.execute("DROP VIEW IF EXISTS public.regulatory_validated_section_citations")
    for table in (
        "regulatory_legal_version_provisions",
        "regulatory_legal_status_assertions",
        "regulatory_legal_events",
        "regulatory_legal_version_artifacts",
        "regulatory_provisions",
        "regulatory_legal_versions",
        "regulatory_evidence",
        "regulatory_source_artifacts",
        "regulatory_source_blobs",
        "regulatory_family_imports",
        "regulatory_instruments",
    ):
        await connection.execute(f"DROP TABLE IF EXISTS public.{table}")
    for trigger_name in (
        "invalidate_retrieval_publication_on_chunk_insert",
        "invalidate_retrieval_publication_on_chunk_delete",
        "invalidate_retrieval_publication_on_chunk_update",
    ):
        await connection.execute(f"DROP TRIGGER IF EXISTS {trigger_name} ON public.document_chunks")
    await connection.execute("DROP FUNCTION IF EXISTS public.invalidate_retrieval_publication()")
    await connection.execute("DROP TABLE IF EXISTS public.document_retrieval_publications")
    await connection.execute("ALTER TABLE public.document_chunks DROP CONSTRAINT IF EXISTS document_chunks_document_fk")
    await connection.execute(
        "ALTER TABLE public.document_sections DROP CONSTRAINT IF EXISTS document_sections_document_fk"
    )
    await connection.execute("ALTER TABLE public.document_sections DROP COLUMN IF EXISTS source_content_hash")
    await connection.execute("DROP TABLE IF EXISTS bddk_meta.legacy_schema_adoptions")
    await connection.execute("DELETE FROM bddk_meta.schema_migrations WHERE version >= 3")


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_live_populated_v2_rehearsal_proves_refusal_migration_publication_and_readiness(
    pg_pool,
    monkeypatch,
) -> None:
    connection = await pg_pool.acquire()
    transaction = connection.transaction()
    await transaction.start()
    pool = _PinnedPool(connection)
    target = str(await connection.fetchval("SELECT current_database()"))
    guard_token = "live-recovery-guard-token-with-more-than-32-characters"
    body = "Proof body"
    content_hash = hashlib.sha256(body.encode()).hexdigest()
    vector = "[" + ",".join("0" for _ in range(768)) + "]"

    async def no_identity_check(*_args, **_kwargs):
        return None

    async def no_lock_monitor(_pool, task):
        await asyncio.shield(task)
        return 0, 1

    async def reindex(_pool):
        await connection.execute(
            "UPDATE public.document_chunks SET embedding = $1::public.vector, total_chunks = 1 WHERE doc_id = $2",
            vector,
            "recovery-proof",
        )
        await connection.execute(
            """
            INSERT INTO public.document_retrieval_publications (
                doc_id, content_hash, retrieval_profile_hash, expected_chunks
            ) VALUES ($1, $2, $3, 1)
            """,
            "recovery-proof",
            content_hash,
            retrieval_profile_hash(),
        )
        return {"reindex_scanned": 1, "reindex_published": 1, "reindex_current": 0}

    try:
        await _downgrade_to_v2(connection)
        await connection.execute(
            "SELECT pg_catalog.set_config('bddk.recovery_drill_guard', $1, true)",
            hashlib.sha256(guard_token.encode()).hexdigest(),
        )
        await connection.execute(
            """
            INSERT INTO public.decision_cache (document_id, title, content, cached_at)
            VALUES ('recovery-proof', 'Proof', $1, 1)
            """,
            body,
        )
        await connection.execute(
            """
            INSERT INTO public.documents (document_id, title, markdown_content, content_hash)
            VALUES ('recovery-proof', 'Proof', $1, $2)
            """,
            body,
            content_hash,
        )
        await connection.execute(
            """
            INSERT INTO public.document_sections (
                doc_id, section_type, section_ref, start_char, end_char, content, content_hash
            ) VALUES ('recovery-proof', 'article', '1', 0, 10, $1, $2)
            """,
            body,
            content_hash,
        )
        await connection.execute(
            """
            INSERT INTO public.document_chunks (
                doc_id, chunk_index, content_hash, chunk_text, total_chunks
            ) VALUES ('recovery-proof', 0, $1, $2, 1)
            """,
            content_hash,
            body,
        )

        monkeypatch.setattr(recovery, "validate_disposable_target_name", lambda name: name)
        monkeypatch.setattr(recovery, "assert_schema_owner_identity", no_identity_check)
        monkeypatch.setattr(recovery, "assert_database_identity", no_identity_check)
        monkeypatch.setattr(recovery, "_monitor_lock_waits", no_lock_monitor)

        report = await run_populated_v2_rehearsal(
            pool,  # type: ignore[arg-type]
            pool,  # type: ignore[arg-type]
            expected_target=target,
            guard_token=guard_token,
            acknowledgement=DISPOSABLE_ACKNOWLEDGEMENT,
            reindexer=reindex,
        )

        assert report.status == "passed"
        assert report.default_refusal_proved
        assert report.source.migration_version == 2
        assert report.restored.migration_version == LATEST_SCHEMA_VERSION
        assert report.restored.migration_checksum == MIGRATIONS[-1].checksum
        assert report.restored.catalog_valid
        assert report.restored.readiness_ready
        assert report.reindex_published == 1
        assert report.lock_samples == 1
        assert json.loads(report.to_json())["source"]["relations"]["public.documents"]["rows"] == 1
        regulatory_relations = {
            relation for relation in recovery._MANAGED_RELATIONS if relation.startswith("public.regulatory_")
        }
        assert not regulatory_relations.intersection(report.source.relations)
        assert regulatory_relations <= set(report.restored.relations)
        assert report.restored.relations["public.regulatory_validated_section_citations"] == RelationEvidence(
            rows=0,
            heap_bytes=0,
            total_bytes=0,
        )
    finally:
        await transaction.rollback()
        await pg_pool.release(connection)


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_live_snapshot_executes_every_retained_generation_fingerprint(pg_pool) -> None:
    evidence = await collect_snapshot_evidence(pg_pool, require_corpus=False)
    retained_members = {f"bddk_retained.{relation}" for relation in RETAINED_CORPUS_RELATIONS}

    assert retained_members <= set(evidence.relations)
    assert {
        "bddk_meta.corpus_generations",
        "bddk_meta.corpus_generation_relation_inventory",
        "bddk_meta.corpus_generation_seals",
        "bddk_meta.corpus_retained_releases",
        "bddk_meta.corpus_release_retention_status",
    } <= set(evidence.relations)
    assert len(evidence.logical_fingerprint_sha256) == 64
    assert evidence.database_encoding
    assert evidence.database_locale_provider in {"b", "c", "i"}
    assert evidence.database_collation_version == evidence.database_collation_actual_version


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_live_snapshot_refuses_retained_member_inventory_seal_and_binding_tamper(pg_pool) -> None:
    from bddk_mcp.corpus_generations import retain_active_corpus_generation
    from tests.test_corpus_publication import (
        _ensure_release_publisher_role,
        _insert_canonical_legal_state,
        _insert_ready_corpus,
        _publish,
    )

    connection = await pg_pool.acquire()
    transaction = connection.transaction()
    await transaction.start()
    document_id = "recovery-retained-seal-tamper"
    try:
        await _ensure_release_publisher_role(connection)
        content_hash = await _insert_ready_corpus(connection, document_id)
        await connection.execute(
            """
            INSERT INTO public.document_versions (
                document_id, version, content_hash, markdown_content, synced_at
            ) SELECT document_id, 1, content_hash, markdown_content, 1.0
              FROM public.documents WHERE document_id = $1
            """,
            document_id,
        )
        await _insert_canonical_legal_state(
            connection,
            document_id=document_id,
            content_hash=content_hash,
        )
        published = await _publish(connection, manifest_id="recovery-retained-seal-001")
        receipt = await retain_active_corpus_generation(
            connection,
            expected_release_id=str(published["release_id"]),
        )
        await connection.execute(
            """
            SELECT pg_catalog.set_config('TimeZone', 'Europe/Istanbul', true),
                   pg_catalog.set_config('DateStyle', 'German', true),
                   pg_catalog.set_config('IntervalStyle', 'sql_standard', true),
                   pg_catalog.set_config('bytea_output', 'escape', true),
                   pg_catalog.set_config('extra_float_digits', '0', true)
            """
        )
        baseline = await collect_snapshot_evidence(_PinnedPool(connection), require_corpus=False)
        assert len(baseline.logical_fingerprint_sha256) == 64
        await connection.execute(
            """
            SELECT pg_catalog.set_config('TimeZone', 'Pacific/Auckland', true),
                   pg_catalog.set_config('DateStyle', 'SQL, DMY', true),
                   pg_catalog.set_config('IntervalStyle', 'postgres_verbose', true),
                   pg_catalog.set_config('bytea_output', 'hex', true),
                   pg_catalog.set_config('extra_float_digits', '2', true)
            """
        )
        differently_configured = await collect_snapshot_evidence(
            _PinnedPool(connection),
            require_corpus=False,
        )
        assert differently_configured.logical_fingerprint_sha256 == baseline.logical_fingerprint_sha256

        tamper_cases = (
            (
                "bddk_retained.documents",
                "guard_retained_generation_member",
                "UPDATE bddk_retained.documents SET title = 'private retained tamper' WHERE generation_id = $1",
            ),
            (
                "bddk_meta.corpus_generation_relation_inventory",
                "guard_retained_generation_inventory",
                "UPDATE bddk_meta.corpus_generation_relation_inventory "
                "SET relation_sha256 = 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff' "
                "WHERE generation_id = $1 AND relation_name = 'documents'",
            ),
            (
                "bddk_meta.corpus_generation_seals",
                "reject_corpus_generation_seals_update_delete",
                "UPDATE bddk_meta.corpus_generation_seals "
                "SET inventory_sha256 = 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff' "
                "WHERE generation_id = $1",
            ),
            (
                "bddk_meta.corpus_retained_releases",
                "reject_corpus_retained_releases_update_delete",
                "DELETE FROM bddk_meta.corpus_retained_releases WHERE generation_id = $1",
            ),
        )
        for table, trigger, statement in tamper_cases:
            savepoint = connection.transaction()
            await savepoint.start()
            try:
                await connection.execute(f"ALTER TABLE {table} DISABLE TRIGGER {trigger}")
                await connection.execute(statement, receipt.generation_id)
                await connection.execute(f"ALTER TABLE {table} ENABLE TRIGGER {trigger}")

                with pytest.raises(RecoveryDrillError) as captured:
                    await collect_snapshot_evidence(_PinnedPool(connection), require_corpus=False)
                assert captured.value.code == "retained_generation_seal_invalid"
                assert captured.value.__cause__ is None
                assert receipt.generation_id not in str(captured.value)
                assert "private retained tamper" not in str(captured.value)
            finally:
                await savepoint.rollback()

        # A privileged restore can temporarily bypass both the immutable
        # member trigger and the generation FK. Orphan bytes must not sit
        # outside every per-generation inventory and escape recovery checks.
        savepoint = connection.transaction()
        await savepoint.start()
        try:
            orphan_generation_id = "corpus_generation_sha256_" + "9" * 64
            await connection.execute("ALTER TABLE bddk_retained.documents DISABLE TRIGGER ALL")
            await connection.execute(
                "INSERT INTO bddk_retained.documents "
                "SELECT $1, source.* FROM public.documents AS source WHERE document_id = $2",
                orphan_generation_id,
                document_id,
            )
            await connection.execute("ALTER TABLE bddk_retained.documents ENABLE TRIGGER ALL")

            with pytest.raises(RecoveryDrillError) as captured:
                await collect_snapshot_evidence(_PinnedPool(connection), require_corpus=False)
            assert captured.value.code == "retained_generation_seal_invalid"
            assert captured.value.__cause__ is None
            assert orphan_generation_id not in str(captured.value)
        finally:
            await savepoint.rollback()

        # Prove the all-bindings gate checks the complete seal tuple, not just
        # that each independently named generation and seal happens to exist.
        savepoint = connection.transaction()
        await savepoint.start()
        try:
            cross_release_id = "corpus_release_sha256_" + "e" * 64
            cross_state = "f" * 64
            cross_profile = "0" * 64
            await connection.execute(
                """
                INSERT INTO bddk_meta.corpus_releases (
                    release_id, manifest_id, manifest_sha256,
                    signer_key_sha256, freshness_policy_result,
                    source_detection_slo_seconds, publication_slo_seconds,
                    max_manifest_age_seconds, retrieval_profile_sha256,
                    corpus_state_sha256
                ) VALUES (
                    $1, 'cross-wired-release', $2, $3,
                    'quantified_measured_signature_verified_pass',
                    60, 120, 3600, $4, $5
                )
                """,
                cross_release_id,
                "1" * 64,
                "2" * 64,
                cross_profile,
                cross_state,
            )
            await connection.execute("ALTER TABLE bddk_meta.corpus_retained_releases DISABLE TRIGGER ALL")
            await connection.execute(
                """
                INSERT INTO bddk_meta.corpus_retained_releases (
                    release_id, seal_id, generation_id,
                    corpus_state_sha256, retrieval_profile_sha256,
                    retained_by_fingerprint_sha256
                ) VALUES ($1, $2, $3, $4, $5, $6)
                """,
                cross_release_id,
                receipt.seal_id,
                receipt.generation_id,
                cross_state,
                cross_profile,
                "3" * 64,
            )
            await connection.execute("ALTER TABLE bddk_meta.corpus_retained_releases ENABLE TRIGGER ALL")

            with pytest.raises(RecoveryDrillError) as captured:
                await collect_snapshot_evidence(_PinnedPool(connection), require_corpus=False)
            assert captured.value.code == "retained_generation_seal_invalid"
            assert captured.value.__cause__ is None
            assert cross_release_id not in str(captured.value)
        finally:
            await savepoint.rollback()
    finally:
        await transaction.rollback()
        await pg_pool.release(connection)


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_live_logical_fingerprint_detects_same_length_text_and_embedding_corruption(pg_pool) -> None:
    connection = await pg_pool.acquire()
    transaction = connection.transaction()
    await transaction.start()
    pool = _PinnedPool(connection)
    document_id = "recovery-fingerprint-corruption"
    document_text = "Document body A"
    changed_document_text = "Document body B"
    section_text = "Section body A"
    changed_section_text = "Section body B"
    chunk_text = "Chunk body A"
    changed_chunk_text = "Chunk body B"
    document_hash = hashlib.sha256(document_text.encode()).hexdigest()
    section_hash = hashlib.sha256(section_text.encode()).hexdigest()
    zero_vector = "[" + ",".join("0" for _ in range(768)) + "]"
    changed_vector = "[1," + ",".join("0" for _ in range(767)) + "]"

    assert len(document_text) == len(changed_document_text)
    assert len(section_text) == len(changed_section_text)
    assert len(chunk_text) == len(changed_chunk_text)
    assert len(zero_vector) == len(changed_vector)

    try:
        await connection.execute(
            """
            INSERT INTO public.documents (
                document_id, title, markdown_content, content_hash, pdf_blob
            ) VALUES ($1, 'Fingerprint proof', $2, $3, $4)
            """,
            document_id,
            document_text,
            document_hash,
            b"source-bytes-a",
        )
        await connection.execute(
            """
            INSERT INTO public.document_sections (
                doc_id, section_type, section_ref, start_char, end_char,
                content, content_hash, source_content_hash
            ) VALUES ($1, 'article', '1', 0, $2, $3, $4, $5)
            """,
            document_id,
            len(section_text),
            section_text,
            section_hash,
            document_hash,
        )
        await connection.execute(
            """
            INSERT INTO public.document_chunks (
                doc_id, chunk_index, content_hash, chunk_text, total_chunks, embedding
            ) VALUES ($1, 0, $2, $3, 1, $4::public.vector)
            """,
            document_id,
            document_hash,
            chunk_text,
            zero_vector,
        )

        baseline = await collect_snapshot_evidence(pool, require_corpus=False)

        await connection.execute(
            "UPDATE public.documents SET markdown_content = $1 WHERE document_id = $2",
            changed_document_text,
            document_id,
        )
        document_corruption = await collect_snapshot_evidence(pool, require_corpus=False)
        await connection.execute(
            "UPDATE public.documents SET markdown_content = $1 WHERE document_id = $2",
            document_text,
            document_id,
        )

        await connection.execute(
            "UPDATE public.document_sections SET content = $1 WHERE doc_id = $2",
            changed_section_text,
            document_id,
        )
        section_corruption = await collect_snapshot_evidence(pool, require_corpus=False)
        await connection.execute(
            "UPDATE public.document_sections SET content = $1 WHERE doc_id = $2",
            section_text,
            document_id,
        )

        await connection.execute(
            "UPDATE public.document_chunks SET chunk_text = $1 WHERE doc_id = $2",
            changed_chunk_text,
            document_id,
        )
        chunk_corruption = await collect_snapshot_evidence(pool, require_corpus=False)
        await connection.execute(
            "UPDATE public.document_chunks SET chunk_text = $1 WHERE doc_id = $2",
            chunk_text,
            document_id,
        )

        await connection.execute(
            "UPDATE public.document_chunks SET embedding = $1::public.vector WHERE doc_id = $2",
            changed_vector,
            document_id,
        )
        embedding_corruption = await collect_snapshot_evidence(pool, require_corpus=False)

        assert document_corruption.logical_fingerprint_sha256 != baseline.logical_fingerprint_sha256
        assert section_corruption.logical_fingerprint_sha256 != baseline.logical_fingerprint_sha256
        assert chunk_corruption.logical_fingerprint_sha256 != baseline.logical_fingerprint_sha256
        assert embedding_corruption.logical_fingerprint_sha256 != baseline.logical_fingerprint_sha256
    finally:
        await transaction.rollback()
        await pg_pool.release(connection)


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_live_snapshot_rejects_activation_identity_collision_and_restores_sequence_state(pg_pool) -> None:
    """setval is nontransactional, so the regression restores its exact prior state."""

    connection = await pg_pool.acquire()
    original = await connection.fetchrow(
        """
        SELECT last_value, is_called
        FROM bddk_meta.corpus_release_activations_activation_sequence_seq
        """
    )
    transaction = connection.transaction()
    await transaction.start()
    release_id = "corpus_release_sha256_" + "e" * 64
    try:
        await connection.execute(
            """
            INSERT INTO bddk_meta.corpus_releases (
                release_id, manifest_id, manifest_sha256, signer_key_sha256,
                freshness_policy_result, source_detection_slo_seconds,
                publication_slo_seconds, max_manifest_age_seconds,
                retrieval_profile_sha256, corpus_state_sha256
            ) VALUES (
                $1, 'recovery-sequence-collision-proof', $2, $3,
                'quantified_measured_signature_verified_pass', 60, 120, 3600,
                $4, $5
            )
            """,
            release_id,
            "1" * 64,
            "2" * 64,
            "3" * 64,
            "4" * 64,
        )
        activation_sequence = int(
            await connection.fetchval(
                """
                INSERT INTO bddk_meta.corpus_release_activations (
                    release_id, actor_fingerprint_sha256, corpus_epoch
                )
                SELECT $1, $2, epoch
                FROM bddk_meta.corpus_state_epoch
                WHERE singleton_id = TRUE
                RETURNING activation_sequence
                """,
                release_id,
                "5" * 64,
            )
        )
        await connection.fetchval(
            """
            SELECT pg_catalog.setval(
                'bddk_meta.corpus_release_activations_activation_sequence_seq'::pg_catalog.regclass,
                $1,
                FALSE
            )
            """,
            activation_sequence,
        )

        with pytest.raises(RecoveryDrillError, match="identity_sequence_collision_risk"):
            await collect_snapshot_evidence(_PinnedPool(connection), require_corpus=False)
    finally:
        await transaction.rollback()
        await connection.fetchval(
            """
            SELECT pg_catalog.setval(
                'bddk_meta.corpus_release_activations_activation_sequence_seq'::pg_catalog.regclass,
                $1,
                $2
            )
            """,
            int(original["last_value"]),
            bool(original["is_called"]),
        )
        await pg_pool.release(connection)
