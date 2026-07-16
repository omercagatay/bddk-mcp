"""Tests for the packaged bddk-mcp command surface."""

from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bddk_mcp import __version__, cli
from bddk_mcp.corpus_coordination import SCHEMA_MIGRATION_ADVISORY_KEY


def test_parser_exposes_explicit_runtime_commands():
    parser = cli.build_parser()

    assert parser.parse_args(["serve", "--transport", "stdio"]).command == "serve"
    assert parser.parse_args(["serve", "--profile", "operator"]).profile == "operator"
    assert parser.parse_args(["migrate"]).command == "migrate"
    assert parser.parse_args(["migrate"]).allow_retrieval_publication_backfill is False
    assert (
        parser.parse_args(["migrate", "--allow-retrieval-publication-backfill"]).allow_retrieval_publication_backfill
        is True
    )
    assert parser.parse_args(["bootstrap", "--seed-dir", "/corpus"]).seed_dir == Path("/corpus")
    bootstrap = parser.parse_args(
        [
            "bootstrap",
            "--reindex-existing",
            "--require-quantified-freshness",
            "--require-measured-freshness",
            "--require-verified-signature",
            "--trusted-signing-key",
            "/trust/corpus.pem",
        ]
    )
    assert bootstrap.reindex_existing is True
    assert bootstrap.require_quantified_freshness is True
    assert bootstrap.require_measured_freshness is True
    assert bootstrap.require_verified_signature is True
    assert bootstrap.trusted_signing_key == Path("/trust/corpus.pem")
    publish = parser.parse_args(
        [
            "publish-corpus-release",
            "--seed-dir",
            "/corpus",
            "--trusted-signing-key",
            "/trust/corpus.pem",
        ]
    )
    assert publish.command == "publish-corpus-release"
    assert publish.seed_dir == Path("/corpus")
    assert publish.trusted_signing_key == Path("/trust/corpus.pem")
    stage = parser.parse_args(
        [
            "verify-and-stage-corpus-release",
            "--seed-dir",
            "/corpus",
            "--trusted-signing-key",
            "/trust/corpus.pem",
            "--verifier-revision-sha256",
            "a" * 64,
            "--verifier-image-digest",
            "sha256:" + "b" * 64,
            "--verification-valid-for-seconds",
            "900",
        ]
    )
    assert stage.command == "verify-and-stage-corpus-release"
    assert stage.verifier_revision_sha256 == "a" * 64
    assert stage.verifier_image_digest == "sha256:" + "b" * 64
    assert stage.verification_valid_for_seconds == 900
    activate = parser.parse_args(
        [
            "activate-corpus-release",
            "--request-id",
            "corpus_release_request_sha256_" + "c" * 64,
        ]
    )
    assert activate.command == "activate-corpus-release"
    assert activate.request_id == "corpus_release_request_sha256_" + "c" * 64
    retain = parser.parse_args(
        [
            "retain-corpus-generation",
            "--expected-release-id",
            "corpus_release_sha256_" + "a" * 64,
        ]
    )
    assert retain.command == "retain-corpus-generation"
    assert retain.expected_release_id == "corpus_release_sha256_" + "a" * 64
    verify = parser.parse_args(
        [
            "verify-corpus",
            "--require-quantified-freshness",
            "--require-measured-freshness",
            "--require-verified-signature",
        ]
    )
    assert verify.command == "verify-corpus"
    assert verify.require_quantified_freshness is True
    assert verify.require_measured_freshness is True
    assert verify.require_verified_signature is True
    assert verify.trusted_signing_key is None


def test_version_does_not_import_runtime(capsys):
    with pytest.raises(SystemExit, match="0"):
        cli.main(["--version"])

    assert capsys.readouterr().out.strip() == f"bddk-mcp {__version__}"


def test_migrate_command_forwards_only_explicit_maintenance_approvals(capsys):
    with (
        patch("bddk_mcp.cli._migrate") as migrate,
        patch("bddk_mcp.cli.asyncio.run") as run,
    ):
        cli.main(["migrate", "--adopt-legacy", "--allow-retrieval-publication-backfill"])

    migrate.assert_called_once_with(
        None,
        adopt_legacy=True,
        allow_retrieval_publication_backfill=True,
    )
    run.assert_called_once()
    run.call_args.args[0].close()
    assert capsys.readouterr().out.strip() == "Database schema is ready."


def test_bootstrap_reports_the_path_free_manifest_identity(capsys):
    result = {
        "decision_cache": 1,
        "documents": 2,
        "sections": 3,
        "chunks": 4,
        "embedded": 4,
        "reindex_published": 0,
        "corpus_manifest_id": "reviewed-corpus-v1",
        "corpus_manifest_sha256": "a" * 64,
        "corpus_scope_warnings": ["chunk_artifact_does_not_match_current_retrieval_profile"],
        "release_publication_required": True,
    }
    with (
        patch("bddk_mcp.cli._bootstrap") as bootstrap,
        patch("bddk_mcp.cli.asyncio.run", return_value=result) as run,
    ):
        cli.main(["bootstrap"])

    bootstrap.assert_called_once_with(
        None,
        None,
        False,
        reindex_existing=False,
        require_quantified_freshness=False,
        require_measured_freshness=False,
        require_verified_signature=False,
        trusted_signing_key=None,
    )
    run.assert_called_once()
    run.call_args.args[0].close()
    output = capsys.readouterr().out
    assert "Corpus manifest used: id=reviewed-corpus-v1" in output
    assert "sha256=" + "a" * 64 in output
    assert "WARNING: chunk_artifact_does_not_match_current_retrieval_profile" in output
    assert "Release publication required" in output
    assert "seed_data" not in output


def test_legacy_publish_release_command_fails_closed_with_migration_guidance(capsys):
    with pytest.raises(SystemExit, match="2"):
        cli.main(
            [
                "publish-corpus-release",
                "--seed-dir",
                "/corpus",
                "--trusted-signing-key",
                "/trust/corpus.pem",
            ]
        )

    error = capsys.readouterr().err
    assert "publish-corpus-release is disabled" in error
    assert "verify-and-stage-corpus-release" in error
    assert "activate-corpus-release" in error
    assert "/corpus" not in error
    assert "/trust" not in error


def test_stage_and_activation_commands_forward_separate_inputs(capsys):
    request_id = "corpus_release_request_sha256_" + "c" * 64
    staged = {"corpus_release_request": {"request_id": request_id}}
    active_release = {
        "release_id": "corpus_release_sha256_" + "a" * 64,
        "manifest_sha256": "b" * 64,
        "retrieval_profile_sha256": "d" * 64,
    }
    with (
        patch("bddk_mcp.cli._verify_and_stage_corpus_release") as stage,
        patch("bddk_mcp.cli.asyncio.run", return_value=staged) as run,
    ):
        cli.main(
            [
                "verify-and-stage-corpus-release",
                "--seed-dir",
                "/corpus",
                "--trusted-signing-key",
                "/trust/corpus.pem",
                "--verifier-revision-sha256",
                "a" * 64,
                "--verifier-image-digest",
                "sha256:" + "b" * 64,
            ]
        )
    stage.assert_called_once_with(
        None,
        Path("/corpus"),
        trusted_signing_key=Path("/trust/corpus.pem"),
        verifier_revision_sha256="a" * 64,
        verifier_image_digest="sha256:" + "b" * 64,
        valid_for_seconds=None,
    )
    run.call_args.args[0].close()
    assert request_id in capsys.readouterr().out

    with (
        patch("bddk_mcp.cli._activate_corpus_release") as activate,
        patch(
            "bddk_mcp.cli.asyncio.run",
            return_value={"active_corpus_release": active_release},
        ) as run,
    ):
        cli.main(["activate-corpus-release", "--request-id", request_id])
    activate.assert_called_once_with(None, request_id=request_id)
    run.call_args.args[0].close()
    output = capsys.readouterr().out
    assert active_release["release_id"] in output
    assert active_release["manifest_sha256"] in output


@pytest.mark.asyncio
async def test_activation_helper_uses_only_publisher_identity_and_request_id():
    from bddk_mcp.corpus_publication import (
        CorpusReleaseActivationReceipt,
        CorpusReleaseIdentity,
    )

    request_id = "corpus_release_request_sha256_" + "a" * 64
    release = CorpusReleaseIdentity(
        release_id="corpus_release_sha256_" + "b" * 64,
        manifest_id="release-test-001",
        manifest_sha256="c" * 64,
        signer_key_sha256="d" * 64,
        freshness_policy_result="quantified_measured_signature_verified_pass",
        source_detection_slo_seconds=60,
        publication_slo_seconds=120,
        max_manifest_age_seconds=3600,
        retrieval_profile_sha256="e" * 64,
        corpus_state_sha256="f" * 64,
        completed_at=datetime(2026, 7, 16, tzinfo=UTC),
    )
    receipt = CorpusReleaseActivationReceipt(
        request_id=request_id,
        activation_sequence=3,
        release=release,
    )
    connection = MagicMock()
    connection.execute = AsyncMock()
    transaction = connection.transaction.return_value
    acquire = MagicMock()
    acquire.__aenter__ = AsyncMock(return_value=connection)
    acquire.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value = acquire
    pool.close = AsyncMock()

    with (
        patch("bddk_mcp.db_transport.assert_database_transport", return_value="postgresql://verified") as transport,
        patch("bddk_mcp.db_identity.assert_database_connection_identity") as connection_identity,
        patch("bddk_mcp.db_identity.assert_database_identity", new=AsyncMock()) as identity,
        patch(
            "bddk_mcp.corpus_publication.activate_staged_corpus_release",
            new=AsyncMock(return_value=receipt),
        ) as activate,
        patch(
            "bddk_mcp.corpus_publication.inspect_active_corpus_release",
            new=AsyncMock(return_value=release),
        ) as inspect_active,
        patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)) as create_pool,
    ):
        result = await cli._activate_corpus_release(
            "postgresql://requested",
            request_id=request_id,
        )

    transport.assert_called_once_with("postgresql://requested")
    init = create_pool.await_args.kwargs["init"]
    assert init.func is connection_identity
    assert init.keywords == {"profile": "release-publisher"}
    identity.assert_awaited_once_with(pool, "release-publisher")
    activate.assert_awaited_once_with(connection, request_id=request_id)
    inspect_active.assert_awaited_once_with(connection)
    assert transaction.__aenter__.await_count == 1
    assert pool.close.await_count == 1
    assert result == {"schema_version": 1, **receipt.safe_dict()}


@pytest.mark.asyncio
async def test_activation_helper_sanitizes_database_connection_failures():
    request_id = "corpus_release_request_sha256_" + "a" * 64
    with (
        patch("bddk_mcp.db_transport.assert_database_transport", return_value="postgresql://verified"),
        patch("asyncpg.create_pool", new=AsyncMock(side_effect=RuntimeError("private DSN and principal"))),
    ):
        with pytest.raises(RuntimeError) as captured:
            await cli._activate_corpus_release(
                "postgresql://requested",
                request_id=request_id,
            )

    assert str(captured.value) == "Release-publisher database connection could not be established safely."
    assert "private" not in str(captured.value)


@pytest.mark.asyncio
async def test_staging_helper_uses_verifier_identity_and_checks_membership_inside_transaction(tmp_path):
    from types import SimpleNamespace

    from bddk_mcp.corpus_publication import CorpusReleaseRequestIdentity

    validation = SimpleNamespace(
        manifest_sha256="b" * 64,
        signature_sha256="4" * 64,
        warnings=(),
        manifest=SimpleNamespace(
            manifest_id="release-test-001",
            integrity=SimpleNamespace(signature_reference="manifest.sig"),
        ),
    )
    artifacts = {name: SimpleNamespace(role=name) for name in ("documents", "chunks", "decision_cache")}
    request = CorpusReleaseRequestIdentity(
        request_id="corpus_release_request_sha256_" + "a" * 64,
        release_id="corpus_release_sha256_" + "c" * 64,
        corpus_state_sha256="d" * 64,
        corpus_epoch=4,
        staged_at=datetime(2026, 7, 16, tzinfo=UTC),
        verification_expires_at=datetime(2026, 7, 16, 0, 15, tzinfo=UTC),
    )
    events: list[str] = []
    connection = MagicMock()
    connection.execute = AsyncMock()
    transaction = connection.transaction.return_value
    transaction.__aenter__ = AsyncMock(side_effect=lambda: events.append("transaction-enter"))
    transaction.__aexit__ = AsyncMock(side_effect=lambda *_args: events.append("transaction-exit"))
    acquire = MagicMock()
    acquire.__aenter__ = AsyncMock(return_value=connection)
    acquire.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value = acquire
    pool.close = AsyncMock()
    vector_store = SimpleNamespace(retrieval_profile_hash="e" * 64)

    async def stage_release(*_args, **_kwargs):
        events.append("stage")
        return request

    async def membership(*_args, **_kwargs):
        events.append("membership")

    with (
        patch("bddk_mcp.core.config.RELEASE_VERIFIER_REVISION_SHA256", "f" * 64),
        patch("bddk_mcp.core.config.RELEASE_VERIFIER_IMAGE_DIGEST", "sha256:" + "1" * 64),
        patch("bddk_mcp.db_transport.assert_database_transport", return_value="postgresql://verified"),
        patch("bddk_mcp.db_identity.assert_database_connection_identity") as connection_identity,
        patch("bddk_mcp.db_identity.assert_database_identity", new=AsyncMock()) as identity,
        patch("bddk_mcp.ingest.seed._manifest_seed_artifacts", return_value=(validation, artifacts)),
        patch("bddk_mcp.ingest.seed._load_manifest_bound_records", side_effect=[[], [], []]),
        patch("bddk_mcp.ingest.seed._validate_seed_documents"),
        patch("bddk_mcp.ingest.seed._validate_strict_seed_artifact_shapes"),
        patch("bddk_mcp.ingest.seed._expected_seed_sections", return_value={}),
        patch("bddk_mcp.ingest.seed._generate_seed_chunks", return_value=([], {})),
        patch("bddk_mcp.ingest.seed._record_chunk_artifact_match") as chunk_match,
        patch("bddk_mcp.ingest.seed._regenerate_seed_embedding_vectors", new=AsyncMock(return_value=[])),
        patch("bddk_mcp.ingest.seed._assert_strict_seed_membership", new=AsyncMock(side_effect=membership)) as member,
        patch("bddk_mcp.store.vector_store.VectorStore", return_value=vector_store),
        patch(
            "bddk_mcp.corpus_publication.strict_verification_evidence_sha256",
            return_value="2" * 64,
        ),
        patch(
            "bddk_mcp.corpus_publication.stage_strict_corpus_release",
            new=AsyncMock(side_effect=stage_release),
        ) as stage,
        patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)) as create_pool,
    ):
        chunk_match.side_effect = lambda result, **_kwargs: result.update(chunk_artifact_match=True)
        result = await cli._verify_and_stage_corpus_release(
            "postgresql://requested",
            tmp_path,
            trusted_signing_key=tmp_path / "trusted.pem",
            verifier_revision_sha256=None,
            verifier_image_digest=None,
            valid_for_seconds=None,
        )

    init = create_pool.await_args.kwargs["init"]
    assert init.func is connection_identity
    assert init.keywords == {"profile": "release-verifier"}
    identity.assert_awaited_once_with(pool, "release-verifier")
    stage.assert_awaited_once()
    assert stage.await_args.kwargs["signature_sha256"] == "4" * 64
    member.assert_awaited_once()
    assert events == ["transaction-enter", "stage", "membership", "transaction-exit"]
    assert result["corpus_release_request"] == request.safe_dict()


@pytest.mark.asyncio
async def test_retain_generation_uses_bounded_transaction_and_schema_only_preflight():
    from bddk_mcp.corpus_generations import (
        CorpusGenerationReceipt,
        CorpusGenerationStorageEvidence,
    )

    release_id = "corpus_release_sha256_" + "a" * 64
    receipt = CorpusGenerationReceipt(
        generation_id="corpus_generation_sha256_" + "b" * 64,
        seal_id="corpus_generation_seal_sha256_" + "c" * 64,
        release_id=release_id,
        source_activation_sequence=7,
        corpus_state_sha256="d" * 64,
        retrieval_profile_sha256="e" * 64,
        inventory_sha256="f" * 64,
        relation_count=17,
        row_count=42,
        retained_at=datetime(2026, 7, 16, 12, 30, tzinfo=UTC),
    )
    storage = CorpusGenerationStorageEvidence(
        generation_id=receipt.generation_id,
        relation_count=17,
        row_count=42,
        generation_logical_bytes=100,
        retained_store_heap_main_bytes=40,
        retained_store_heap_auxiliary_bytes=10,
        retained_store_toast_bytes=20,
        retained_store_index_bytes=30,
        retained_store_total_bytes=100,
    )
    events: list[tuple[object, ...]] = []

    async def execute(sql: str) -> None:
        events.append(("execute", sql))

    fetch_results = iter([None, "0/100", "0/1100", Decimal(4096)])

    async def fetchval(sql: str, *args: object) -> object:
        events.append(("fetchval", sql, *args))
        return next(fetch_results)

    connection = MagicMock()
    connection.execute = AsyncMock(side_effect=execute)
    connection.fetchval = AsyncMock(side_effect=fetchval)
    transaction = connection.transaction.return_value
    acquire = MagicMock()
    acquire.__aenter__ = AsyncMock(return_value=connection)
    acquire.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value = acquire
    pool.close = AsyncMock()

    with (
        patch("bddk_mcp.db_transport.assert_database_transport", return_value="postgresql://verified") as transport,
        patch("bddk_mcp.db_identity.assert_database_connection_identity") as connection_identity,
        patch("bddk_mcp.db_identity.assert_database_identity", new=AsyncMock()) as identity,
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=AsyncMock()) as readiness,
        patch(
            "bddk_mcp.corpus_generations.retain_active_corpus_generation", new=AsyncMock(return_value=receipt)
        ) as retain,
        patch(
            "bddk_mcp.corpus_generations.collect_generation_storage_evidence",
            new=AsyncMock(return_value=storage),
        ) as collect_storage,
        patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)) as create_pool,
    ):
        readiness.side_effect = lambda **_kwargs: events.append(("readiness",))
        retain.side_effect = lambda *_args, **_kwargs: events.append(("retain",)) or receipt
        collect_storage.side_effect = lambda *_args, **_kwargs: events.append(("storage",)) or storage
        result = await cli._retain_corpus_generation(
            "postgresql://requested",
            expected_release_id=release_id,
        )

    transport.assert_called_once_with("postgresql://requested")
    create_pool.assert_awaited_once()
    assert create_pool.await_args.args == ("postgresql://verified",)
    assert create_pool.await_args.kwargs["min_size"] == 1
    assert create_pool.await_args.kwargs["max_size"] == 1
    init = create_pool.await_args.kwargs["init"]
    assert init.func is connection_identity
    assert init.keywords == {"profile": "release-publisher"}
    identity.assert_awaited_once_with(pool, "release-publisher")
    readiness.assert_awaited_once_with(pool=connection, require_corpus=False)
    retain.assert_awaited_once_with(connection, expected_release_id=release_id)
    assert connection.execute.await_args_list[0].args == (
        f"SET LOCAL lock_timeout = '{cli._CORPUS_RETENTION_LOCK_TIMEOUT}'",
    )
    assert connection.execute.await_args_list[1].args == (
        f"SET LOCAL statement_timeout = '{cli._CORPUS_RETENTION_STATEMENT_TIMEOUT}'",
    )
    assert connection.fetchval.await_count == 4
    assert connection.fetchval.await_args_list[0].args == (
        cli._SCHEMA_MIGRATION_LOCK_SQL,
        SCHEMA_MIGRATION_ADVISORY_KEY,
    )
    assert connection.fetchval.await_args_list[1].args == (cli._CURRENT_WAL_INSERT_LSN_SQL,)
    assert connection.fetchval.await_args_list[2].args == (cli._CURRENT_WAL_INSERT_LSN_SQL,)
    assert connection.fetchval.await_args_list[3].args == (
        cli._OBSERVED_WAL_BYTES_SQL,
        "0/1100",
        "0/100",
    )
    assert events == [
        ("execute", f"SET LOCAL lock_timeout = '{cli._CORPUS_RETENTION_LOCK_TIMEOUT}'"),
        ("execute", f"SET LOCAL statement_timeout = '{cli._CORPUS_RETENTION_STATEMENT_TIMEOUT}'"),
        ("fetchval", cli._SCHEMA_MIGRATION_LOCK_SQL, SCHEMA_MIGRATION_ADVISORY_KEY),
        ("readiness",),
        ("fetchval", cli._CURRENT_WAL_INSERT_LSN_SQL),
        ("retain",),
        ("storage",),
        ("fetchval", cli._CURRENT_WAL_INSERT_LSN_SQL),
        ("fetchval", cli._OBSERVED_WAL_BYTES_SQL, "0/1100", "0/100"),
    ]
    collect_storage.assert_awaited_once_with(
        connection,
        generation_id=receipt.generation_id,
    )
    assert transaction.__aenter__.await_count == 2
    assert transaction.__aexit__.await_count == 2
    assert all(call.args == (None, None, None) for call in transaction.__aexit__.await_args_list)
    pool.close.assert_awaited_once_with()
    assert result["schema_version"] == 1
    assert result["retained_generation"] == receipt.safe_dict()
    assert result["storage_evidence"] == {
        **storage.safe_dict(),
        "observed_cluster_wal_bytes": 4096,
        "wal_attribution": "observed_cluster_interval_not_exclusive",
    }
    assert result["storage_evidence"]["observed_cluster_wal_bytes"] == 4096
    assert result["storage_evidence"]["wal_attribution"] == "observed_cluster_interval_not_exclusive"
    assert result["storage_evidence"]["backup_growth_bytes"] is None
    assert result["storage_evidence"]["backup_growth_status"] == "not_measured"


@pytest.mark.asyncio
@pytest.mark.parametrize("release_fails", (False, True))
async def test_retain_generation_preserves_committed_success_across_wal_and_pool_cleanup_failures(
    release_fails: bool,
):
    from bddk_mcp.corpus_generations import CorpusGenerationStorageEvidence

    release_id = "corpus_release_sha256_" + "a" * 64
    generation_id = "corpus_generation_sha256_" + "b" * 64
    receipt = MagicMock(generation_id=generation_id, relation_count=17, row_count=1)
    receipt.safe_dict.return_value = {"generation_id": generation_id, "release_id": release_id}
    storage = CorpusGenerationStorageEvidence(
        generation_id=generation_id,
        relation_count=17,
        row_count=1,
        generation_logical_bytes=10,
        retained_store_heap_main_bytes=4,
        retained_store_heap_auxiliary_bytes=1,
        retained_store_toast_bytes=2,
        retained_store_index_bytes=3,
        retained_store_total_bytes=10,
    )
    connection = MagicMock()
    connection.execute = AsyncMock()
    connection.fetchval = AsyncMock(side_effect=[None, "0/100", "0/1100", "invalid private value"])
    transaction = connection.transaction.return_value
    acquire = MagicMock()
    acquire.__aenter__ = AsyncMock(return_value=connection)
    acquire.__aexit__ = AsyncMock(
        side_effect=RuntimeError("private pool release failure") if release_fails else None,
        return_value=False,
    )
    pool = MagicMock()
    pool.acquire.return_value = acquire
    pool.close = AsyncMock(side_effect=None if release_fails else RuntimeError("private pool close failure"))
    pool.terminate = MagicMock()

    with (
        patch("bddk_mcp.db_transport.assert_database_transport", return_value="postgresql://verified"),
        patch("bddk_mcp.db_identity.assert_database_connection_identity"),
        patch("bddk_mcp.db_identity.assert_database_identity", new=AsyncMock()),
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=AsyncMock()),
        patch(
            "bddk_mcp.corpus_generations.retain_active_corpus_generation",
            new=AsyncMock(return_value=receipt),
        ),
        patch(
            "bddk_mcp.corpus_generations.collect_generation_storage_evidence",
            new=AsyncMock(return_value=storage),
        ),
        patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)),
    ):
        result = await cli._retain_corpus_generation(
            "postgresql://requested",
            expected_release_id=release_id,
        )

    assert transaction.__aexit__.await_count == 2
    assert all(call.args == (None, None, None) for call in transaction.__aexit__.await_args_list)
    acquire.__aexit__.assert_awaited_once_with(None, None, None)
    if release_fails:
        pool.close.assert_not_awaited()
        pool.terminate.assert_called_once_with()
    else:
        pool.close.assert_awaited_once_with()
        pool.terminate.assert_not_called()
    assert result["retained_generation"]["release_id"] == release_id
    assert result["storage_evidence"]["observed_cluster_wal_bytes"] is None
    assert result["storage_evidence"]["wal_attribution"] == "not_measured"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "mismatched_value"),
    (
        ("generation_id", "corpus_generation_sha256_" + "f" * 64),
        ("relation_count", 18),
        ("row_count", 2),
    ),
)
async def test_retain_generation_rejects_storage_evidence_that_does_not_match_receipt(
    field: str,
    mismatched_value: object,
):
    from bddk_mcp.corpus_generations import (
        CorpusGenerationReceipt,
        CorpusGenerationStorageEvidence,
    )

    release_id = "corpus_release_sha256_" + "a" * 64
    generation_id = "corpus_generation_sha256_" + "b" * 64
    receipt = CorpusGenerationReceipt(
        generation_id=generation_id,
        seal_id="corpus_generation_seal_sha256_" + "c" * 64,
        release_id=release_id,
        source_activation_sequence=7,
        corpus_state_sha256="d" * 64,
        retrieval_profile_sha256="e" * 64,
        inventory_sha256="f" * 64,
        relation_count=17,
        row_count=1,
        retained_at=datetime(2026, 7, 16, 12, 30, tzinfo=UTC),
    )
    storage = replace(
        CorpusGenerationStorageEvidence(
            generation_id=generation_id,
            relation_count=17,
            row_count=1,
            generation_logical_bytes=10,
            retained_store_heap_main_bytes=4,
            retained_store_heap_auxiliary_bytes=1,
            retained_store_toast_bytes=2,
            retained_store_index_bytes=3,
            retained_store_total_bytes=10,
        ),
        **{field: mismatched_value},
    )
    connection = MagicMock()
    connection.execute = AsyncMock()
    connection.fetchval = AsyncMock(side_effect=[None, "0/100"])
    transaction = connection.transaction.return_value
    acquire = MagicMock()
    acquire.__aenter__ = AsyncMock(return_value=connection)
    acquire.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value = acquire
    pool.close = AsyncMock()

    with (
        patch("bddk_mcp.db_transport.assert_database_transport", return_value="postgresql://verified"),
        patch("bddk_mcp.db_identity.assert_database_connection_identity"),
        patch("bddk_mcp.db_identity.assert_database_identity", new=AsyncMock()),
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=AsyncMock()),
        patch(
            "bddk_mcp.corpus_generations.retain_active_corpus_generation",
            new=AsyncMock(return_value=receipt),
        ),
        patch(
            "bddk_mcp.corpus_generations.collect_generation_storage_evidence",
            new=AsyncMock(return_value=storage),
        ),
        patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)),
    ):
        with pytest.raises(
            RuntimeError,
            match=r"^Retained corpus generation storage evidence is inconsistent\.$",
        ) as error:
            await cli._retain_corpus_generation(
                "postgresql://requested",
                expected_release_id=release_id,
            )

    assert error.value.__cause__ is None
    assert generation_id not in str(error.value)
    assert transaction.__aexit__.await_count == 2
    assert transaction.__aexit__.await_args.args[0] is not None


@pytest.mark.asyncio
async def test_retain_generation_wal_baseline_failure_rolls_back_only_its_savepoint():
    from bddk_mcp.corpus_generations import CorpusGenerationStorageEvidence

    release_id = "corpus_release_sha256_" + "a" * 64
    generation_id = "corpus_generation_sha256_" + "b" * 64
    receipt = MagicMock(generation_id=generation_id, relation_count=17, row_count=1)
    receipt.safe_dict.return_value = {"generation_id": generation_id, "release_id": release_id}
    storage = CorpusGenerationStorageEvidence(
        generation_id=generation_id,
        relation_count=17,
        row_count=1,
        generation_logical_bytes=10,
        retained_store_heap_main_bytes=4,
        retained_store_heap_auxiliary_bytes=1,
        retained_store_toast_bytes=2,
        retained_store_index_bytes=3,
        retained_store_total_bytes=10,
    )
    connection = MagicMock()
    connection.execute = AsyncMock()
    connection.fetchval = AsyncMock(side_effect=[None, RuntimeError("private wal permission failure")])
    transaction = connection.transaction.return_value
    transaction.__aexit__ = AsyncMock(return_value=False)
    acquire = MagicMock()
    acquire.__aenter__ = AsyncMock(return_value=connection)
    acquire.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value = acquire
    pool.close = AsyncMock()

    with (
        patch("bddk_mcp.db_transport.assert_database_transport", return_value="postgresql://verified"),
        patch("bddk_mcp.db_identity.assert_database_connection_identity"),
        patch("bddk_mcp.db_identity.assert_database_identity", new=AsyncMock()),
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=AsyncMock()),
        patch(
            "bddk_mcp.corpus_generations.retain_active_corpus_generation",
            new=AsyncMock(return_value=receipt),
        ) as retain,
        patch(
            "bddk_mcp.corpus_generations.collect_generation_storage_evidence",
            new=AsyncMock(return_value=storage),
        ),
        patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)),
    ):
        result = await cli._retain_corpus_generation(
            "postgresql://requested",
            expected_release_id=release_id,
        )

    retain.assert_awaited_once_with(connection, expected_release_id=release_id)
    assert transaction.__aenter__.await_count == 2
    assert transaction.__aexit__.await_count == 2
    assert transaction.__aexit__.await_args_list[0].args[0] is RuntimeError
    assert transaction.__aexit__.await_args_list[1].args == (None, None, None)
    assert result["storage_evidence"]["observed_cluster_wal_bytes"] is None
    assert result["storage_evidence"]["wal_attribution"] == "not_measured"


def test_retain_generation_prints_content_free_canonical_json(capsys):
    release_id = "corpus_release_sha256_" + "a" * 64
    result = {
        "storage_evidence": {
            "retained_store_total_bytes": 100,
            "generation_id": "corpus_generation_sha256_" + "b" * 64,
        },
        "schema_version": 1,
        "retained_generation": {
            "release_id": release_id,
            "inventory_sha256": "c" * 64,
        },
    }
    with (
        patch("bddk_mcp.cli._retain_corpus_generation") as retain,
        patch("bddk_mcp.cli.asyncio.run", return_value=result) as run,
    ):
        cli.main(["retain-corpus-generation", "--expected-release-id", release_id])

    retain.assert_called_once_with(None, expected_release_id=release_id)
    run.assert_called_once()
    run.call_args.args[0].close()
    output = capsys.readouterr().out.strip()
    assert output == json.dumps(result, sort_keys=True, separators=(",", ":"))
    assert json.loads(output) == result
    assert "markdown_content" not in output
    assert "postgresql://" not in output


@pytest.mark.asyncio
async def test_retain_generation_suppresses_untrusted_database_error_causes():
    release_id = "corpus_release_sha256_" + "a" * 64
    pool = MagicMock()
    pool.close = AsyncMock()
    with (
        patch("bddk_mcp.db_transport.assert_database_transport", return_value="postgresql://verified"),
        patch("asyncpg.create_pool", new=AsyncMock(side_effect=ValueError("postgresql://user:secret@db"))),
    ):
        with pytest.raises(RuntimeError, match=r"^Retained corpus generation operation failed\.$") as error:
            await cli._retain_corpus_generation(
                "postgresql://requested",
                expected_release_id=release_id,
            )

    assert error.value.__cause__ is None
    assert "secret" not in str(error.value)


@pytest.mark.asyncio
async def test_retain_generation_sanitizes_timeout_setup_failures_before_retention():
    release_id = "corpus_release_sha256_" + "a" * 64
    connection = MagicMock()
    connection.execute = AsyncMock(side_effect=RuntimeError("postgresql://user:secret@db"))
    acquire = MagicMock()
    acquire.__aenter__ = AsyncMock(return_value=connection)
    acquire.__aexit__ = AsyncMock(return_value=False)
    pool = MagicMock()
    pool.acquire.return_value = acquire
    pool.close = AsyncMock()

    with (
        patch("bddk_mcp.db_transport.assert_database_transport", return_value="postgresql://verified"),
        patch("bddk_mcp.db_identity.assert_database_connection_identity"),
        patch("bddk_mcp.db_identity.assert_database_identity", new=AsyncMock()),
        patch("bddk_mcp.corpus_generations.retain_active_corpus_generation", new=AsyncMock()) as retain,
        patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)),
    ):
        with pytest.raises(RuntimeError, match=r"^Retained corpus generation operation failed\.$") as error:
            await cli._retain_corpus_generation(
                "postgresql://requested",
                expected_release_id=release_id,
            )

    retain.assert_not_awaited()
    assert error.value.__cause__ is None
    assert "secret" not in str(error.value)


def test_retain_generation_is_not_an_mcp_tool():
    from bddk_mcp.tools.registry import OPERATOR_TOOL_NAMES, PUBLIC_TOOL_NAMES

    assert "retain-corpus-generation" not in PUBLIC_TOOL_NAMES
    assert "retain-corpus-generation" not in OPERATOR_TOOL_NAMES
    assert "retain_corpus_generation" not in PUBLIC_TOOL_NAMES
    assert "retain_corpus_generation" not in OPERATOR_TOOL_NAMES


def test_serve_applies_overrides_before_import():
    args = Namespace(profile="operator", transport="streamable-http", host="127.0.0.2", port=8123)

    with (
        patch.dict(cli.os.environ, {}, clear=False),
        patch("bddk_mcp.server.main") as server_main,
    ):
        cli._run_serve(args)

        server_main.assert_called_once_with()
        assert cli.os.environ["MCP_TRANSPORT"] == "streamable-http"
        assert cli.os.environ["BDDK_TOOL_PROFILE"] == "operator"
        assert cli.os.environ["MCP_HOST"] == "127.0.0.2"
        assert cli.os.environ["PORT"] == "8123"


def test_port_must_be_in_tcp_range():
    parser = cli.build_parser()

    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(["serve", "--port", "0"])
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(["serve", "--port", "65536"])


def test_verify_corpus_checks_tracked_artifacts_and_reports_safe_identity(capsys):
    cli.main(["verify-corpus"])

    output = capsys.readouterr().out
    assert "id=bddk-job-corpus-2026-07-15" in output
    assert "artifacts=3 exhaustive=false" in output
    assert "WARNING: This corpus is a job-specific selection" in output
    assert "markdown_content" not in output


def test_verify_corpus_production_requirements_fail_until_owner_policies_exist(capsys):
    with pytest.raises(SystemExit, match="2"):
        cli.main(["verify-corpus", "--require-quantified-freshness"])

    assert "freshness objectives are not quantified" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        cli.main(["verify-corpus", "--require-measured-freshness"])

    assert "SLO compliance is not measured" in capsys.readouterr().err
