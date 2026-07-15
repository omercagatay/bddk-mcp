"""Safety contracts and opt-in PostgreSQL proof for recovery workflows."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json

import pytest

from bddk_mcp.migrations import LATEST_SCHEMA_VERSION, MIGRATIONS
from bddk_mcp.operations import recovery
from bddk_mcp.operations.recovery import (
    DISPOSABLE_ACKNOWLEDGEMENT,
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


def _snapshot(fingerprint: str = "a" * 64) -> SnapshotEvidence:
    return SnapshotEvidence(
        migration_version=3,
        migration_checksum="b" * 64,
        logical_fingerprint_sha256=fingerprint,
        database_bytes=100,
        wal_lsn="0/10",
        relations={"public.documents": RelationEvidence(rows=1, heap_bytes=10, total_bytes=20)},
        catalog_valid=True,
        catalog_failures=(),
        readiness_ready=True,
        readiness_issues=(),
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
        schema_version=1,
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


async def _downgrade_to_v2(connection) -> None:
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
    await connection.execute(
        "DROP TRIGGER IF EXISTS invalidate_retrieval_publication_on_chunk_change ON public.document_chunks"
    )
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
