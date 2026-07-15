"""Verified-corpus publication, tamper detection, and append-only evidence tests."""

from __future__ import annotations

import asyncio
import hashlib
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import asyncpg
import pytest

from bddk_mcp.corpus_publication import (
    CorpusPublicationError,
    inspect_active_corpus_release,
    publish_strict_corpus_release,
)

_PROFILE_SHA256 = "7" * 64
_ZERO_VECTOR = "[" + ",".join("0" for _ in range(768)) + "]"


@pytest.mark.asyncio
async def test_application_publication_contract_supplies_exact_sql_arguments() -> None:
    expected_completed_at = datetime(2026, 1, 2, tzinfo=UTC)
    returned_row = {
        "release_id": "corpus_release_sha256_" + "1" * 64,
        "manifest_id": "release-test-001",
        "manifest_sha256": "8" * 64,
        "signer_key_sha256": "9" * 64,
        "freshness_policy_result": "quantified_measured_signature_verified_pass",
        "source_detection_slo_seconds": 60,
        "publication_slo_seconds": 120,
        "max_manifest_age_seconds": 3600,
        "retrieval_profile_sha256": _PROFILE_SHA256,
        "corpus_state_sha256": "2" * 64,
        "completed_at": expected_completed_at,
    }

    class RecordingConnection:
        def __init__(self) -> None:
            self.arguments: tuple[object, ...] | None = None

        async def fetchrow(self, _query: str, *arguments: object):
            self.arguments = arguments
            return returned_row

    connection = RecordingConnection()
    validation = SimpleNamespace(
        manifest_sha256="8" * 64,
        manifest=SimpleNamespace(
            manifest_id="release-test-001",
            freshness=SimpleNamespace(
                source_detection_slo_seconds=60,
                publication_slo_seconds=120,
                max_manifest_age_seconds=3600,
                slo_evidence_status="measured",
            ),
            integrity=SimpleNamespace(
                signature_status="verified",
                signature_public_key_sha256="9" * 64,
            ),
        ),
    )

    identity = await publish_strict_corpus_release(
        connection,
        validation,
        retrieval_profile_sha256=_PROFILE_SHA256,
        require_quantified_freshness=True,
        require_measured_freshness=True,
        require_verified_signature=True,
    )

    assert connection.arguments == (
        "release-test-001",
        "8" * 64,
        "9" * 64,
        60,
        120,
        3600,
        _PROFILE_SHA256,
    )
    assert identity.completed_at == expected_completed_at


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


async def _ensure_release_publisher_role(connection: asyncpg.Connection) -> None:
    await connection.execute(
        """
        DO $role$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'bddk_release_publisher'
            ) THEN
                CREATE ROLE bddk_release_publisher NOLOGIN;
            END IF;
        END
        $role$
        """
    )


async def _insert_ready_corpus(connection: asyncpg.Connection, document_id: str) -> str:
    content = "# Madde 1\nDoğrulanmış düzenleme metni."
    content_hash = _sha256(content)
    section_hash = _sha256(content)
    await connection.execute(
        """
        INSERT INTO public.decision_cache (
            document_id, title, content, decision_date, decision_number,
            category, source_url, cached_at
        ) VALUES ($1, 'Kanıt', $2, '2026-01-01', '1', 'test',
                  'https://example.invalid/regulation', 1.0)
        """,
        document_id,
        content,
    )
    await connection.execute(
        """
        INSERT INTO public.documents (
            document_id, title, category, decision_date, decision_number,
            source_url, markdown_content, content_hash, total_pages, file_size
        ) VALUES ($1, 'Kanıt', 'test', '2026-01-01', '1',
                  'https://example.invalid/regulation', $2, $3, 1, $4)
        """,
        document_id,
        content,
        content_hash,
        len(content.encode("utf-8")),
    )
    await connection.execute(
        """
        INSERT INTO public.document_sections (
            doc_id, section_type, section_ref, heading, start_char, end_char,
            content, content_hash, source_content_hash
        ) VALUES ($1, 'article', '1', 'Madde 1', 0, $2, $3, $4, $5)
        """,
        document_id,
        len(content),
        content,
        section_hash,
        content_hash,
    )
    await connection.execute(
        """
        INSERT INTO public.document_chunks (
            doc_id, chunk_index, title, category, decision_date,
            decision_number, source_url, total_chunks, total_pages,
            content_hash, chunk_start_char, chunk_end_char, section_type,
            section_ref, section_start_char, section_end_char,
            section_content_hash, chunk_text, embedding
        ) VALUES (
            $1, 0, 'Kanıt', 'test', '2026-01-01', '1',
            'https://example.invalid/regulation', 1, 1, $2, 0, $3,
            'article', '1', 0, $3, $4, $5, $6::public.vector
        )
        """,
        document_id,
        content_hash,
        len(content),
        section_hash,
        content,
        _ZERO_VECTOR,
    )
    await connection.execute(
        """
        INSERT INTO public.document_retrieval_publications (
            doc_id, content_hash, retrieval_profile_hash, expected_chunks
        ) VALUES ($1, $2, $3, 1)
        """,
        document_id,
        content_hash,
        _PROFILE_SHA256,
    )
    return content_hash


async def _insert_canonical_legal_state(
    connection: asyncpg.Connection,
    *,
    document_id: str,
    content_hash: str,
) -> None:
    instrument_id = "inst_sha256_" + "1" * 64
    bundle_id = "family_sha256_" + "2" * 64
    blob_id = "blob_sha256_" + "3" * 64
    artifact_id = "art_sha256_" + "4" * 64
    evidence_id = "evid_sha256_" + "5" * 64
    version_id = "ver_sha256_" + "6" * 64
    event_id = "event_sha256_" + "7" * 64
    assertion_id = "status_sha256_" + "8" * 64
    provision_id = "prov_sha256_" + "9" * 64
    review_hash = "a" * 64
    section_id = await connection.fetchval(
        "SELECT id FROM public.document_sections WHERE doc_id = $1",
        document_id,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_instruments (
            instrument_id, jurisdiction, authority_code, identity_key,
            canonical_title, instrument_type
        ) VALUES ($1, 'TR', 'BDDK', 'test-regulation', 'Test Düzenlemesi', 'regulation')
        """,
        instrument_id,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_family_imports (
            bundle_id, bundle_sha256, instrument_id, schema_version,
            fixture_only, imported_by, member_manifest
        ) VALUES ($1, $2, $3, 1, false, 'reviewer',
                  '{"z":1,"documents":["proof"]}'::pg_catalog.jsonb)
        """,
        bundle_id,
        "b" * 64,
        instrument_id,
    )
    await connection.execute(
        "INSERT INTO public.regulatory_source_blobs (blob_id, content_sha256) VALUES ($1, $2)",
        blob_id,
        content_hash,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_source_artifacts (
            artifact_id, blob_id, canonical_uri, source_authority, media_type,
            retrieved_at, repository_document_id, fixture_only
        ) VALUES ($1, $2, 'https://example.invalid/regulation', 'BDDK',
                  'text/markdown', '2026-01-01T00:00:00Z', $3, false)
        """,
        artifact_id,
        blob_id,
        document_id,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_evidence (
            evidence_id, artifact_id, locator, statement_sha256, authority_level
        ) VALUES ($1, $2, 'article:1', $3, 'authoritative')
        """,
        evidence_id,
        artifact_id,
        content_hash,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_legal_versions (
            legal_version_id, instrument_id, version_key, legal_text_sha256,
            consolidation_state, validation_state, validated_by, validated_at,
            validation_method, review_record_sha256
        ) VALUES ($1, $2, '2026-original', $3, 'original', 'validated',
                  'reviewer', '2026-01-02T00:00:00Z', 'four-eyes', $4)
        """,
        version_id,
        instrument_id,
        content_hash,
        review_hash,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_legal_version_artifacts (
            legal_version_id, artifact_id, source_role
        ) VALUES ($1, $2, 'legal_text')
        """,
        version_id,
        artifact_id,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_legal_events (
            event_id, legal_version_id, event_type, event_date, evidence_id,
            validation_state, validated_by, validated_at, validation_method,
            review_record_sha256
        ) VALUES ($1, $2, 'publication', '2026-01-01', $3, 'validated',
                  'reviewer', '2026-01-02T00:00:00Z', 'four-eyes', $4)
        """,
        event_id,
        version_id,
        evidence_id,
        review_hash,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_legal_status_assertions (
            assertion_id, legal_version_id, legal_status, valid_from,
            valid_through, evidence_id, validation_state, validated_by,
            validated_at, validation_method, review_record_sha256
        ) VALUES ($1, $2, 'effective', '2026-01-01', '9999-12-31', $3,
                  'validated', 'reviewer', '2026-01-02T00:00:00Z',
                  'four-eyes', $4)
        """,
        assertion_id,
        version_id,
        evidence_id,
        review_hash,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_provisions (
            provision_id, instrument_id, provision_kind, canonical_path
        ) VALUES ($1, $2, 'article', 'article/1')
        """,
        provision_id,
        instrument_id,
    )
    await connection.execute(
        """
        INSERT INTO public.regulatory_legal_version_provisions (
            legal_version_id, provision_id, provision_text_sha256,
            document_section_id, evidence_id, validation_state, validated_by,
            validated_at, validation_method, review_record_sha256
        ) VALUES ($1, $2, $3, $4, $5, 'validated', 'reviewer',
                  '2026-01-02T00:00:00Z', 'four-eyes', $6)
        """,
        version_id,
        provision_id,
        content_hash,
        section_id,
        evidence_id,
        review_hash,
    )


async def _publish(
    connection: asyncpg.Connection,
    *,
    manifest_id: str = "release-test-001",
    manifest_sha256: str = "8" * 64,
):
    return await connection.fetchrow(
        """
        SELECT *
        FROM bddk_meta.publish_verified_corpus_release(
            $1, $2, $3, 60, 120, 3600, $4
        )
        """,
        manifest_id,
        manifest_sha256,
        "9" * 64,
        _PROFILE_SHA256,
    )


@asynccontextmanager
async def _rollback_savepoint(connection: asyncpg.Connection):
    savepoint = connection.transaction()
    await savepoint.start()
    try:
        yield
    finally:
        await savepoint.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_release_is_append_only_and_false_ready_mutations_invalidate_active_view(pg_pool) -> None:
    connection = await pg_pool.acquire()
    transaction = connection.transaction()
    await transaction.start()
    document_id = "corpus-release-integrity"
    try:
        await _ensure_release_publisher_role(connection)
        content_hash = await _insert_ready_corpus(connection, document_id)
        await _insert_canonical_legal_state(
            connection,
            document_id=document_id,
            content_hash=content_hash,
        )
        assert await connection.fetchval("SELECT bddk_meta.corpus_retrieval_ready($1)", _PROFILE_SHA256)

        # Privileged pre-release tampering cannot smuggle a replacement FTS
        # vector into an otherwise self-consistent corpus.
        for table, trigger, predicate in (
            ("document_sections", "trg_document_sections_tsv", "doc_id = $1"),
            ("document_chunks", "chunks_tsv_update", "doc_id = $1"),
        ):
            async with _rollback_savepoint(connection):
                await connection.execute(f"ALTER TABLE public.{table} DISABLE TRIGGER {trigger}")
                await connection.execute(
                    f"UPDATE public.{table} SET tsv = pg_catalog.to_tsvector('simple', 'tampered') WHERE {predicate}",
                    document_id,
                )
                await connection.execute(f"ALTER TABLE public.{table} ENABLE TRIGGER {trigger}")
                assert not await connection.fetchval("SELECT bddk_meta.corpus_retrieval_ready($1)", _PROFILE_SHA256)

        published = await _publish(connection)
        repeated = await _publish(connection)
        assert published is not None
        assert repeated["release_id"] == published["release_id"]
        assert await connection.fetchval("SELECT count(*) FROM bddk_meta.corpus_releases") == 1
        assert await connection.fetchval("SELECT count(*) FROM bddk_meta.corpus_release_activations") == 1
        assert (
            await connection.fetchval("SELECT release_id FROM bddk_meta.active_corpus_release")
            == published["release_id"]
        )

        # CURRENT_TIMESTAMP is transaction-stable, so these two activations
        # deliberately tie on completed_at.  The monotonic sequence—not wall
        # clock ordering—must select the second release.
        replacement = await _publish(
            connection,
            manifest_id="release-test-002",
            manifest_sha256="a" * 64,
        )
        assert replacement["release_id"] != published["release_id"]
        activation_times = await connection.fetch(
            "SELECT completed_at FROM bddk_meta.corpus_release_activations ORDER BY activation_sequence"
        )
        assert len(activation_times) == 2
        assert activation_times[0]["completed_at"] == activation_times[1]["completed_at"]
        assert (
            await connection.fetchval("SELECT release_id FROM bddk_meta.active_corpus_release")
            == replacement["release_id"]
        )

        for statement in (
            "UPDATE bddk_meta.corpus_releases SET manifest_id = 'changed'",
            "DELETE FROM bddk_meta.corpus_release_activations",
        ):
            async with _rollback_savepoint(connection):
                with pytest.raises(asyncpg.ObjectNotInPrerequisiteStateError):
                    await connection.execute(statement)

        # Every base table behind validated Citation v1 output is part of the
        # active release identity.  Curator changes cannot retain the previous
        # release ID merely because vector retrieval rows were untouched.
        for statement in (
            "UPDATE public.regulatory_evidence SET locator = 'article:1-mutated'",
            "UPDATE public.regulatory_legal_status_assertions SET legal_status = 'unknown'",
            "UPDATE public.regulatory_provisions SET canonical_path = 'article/1-mutated'",
        ):
            async with _rollback_savepoint(connection):
                await connection.execute(statement)
                assert await connection.fetchval("SELECT bddk_meta.corpus_retrieval_ready($1)", _PROFILE_SHA256)
                assert await connection.fetchval("SELECT count(*) FROM bddk_meta.active_corpus_release") == 0

        # An empty or unpublished document must not be hidden by one healthy
        # document elsewhere in the corpus.
        async with _rollback_savepoint(connection):
            await connection.execute(
                "INSERT INTO public.documents (document_id, title) VALUES ('unpublished-empty', 'Empty')"
            )
            assert not await connection.fetchval("SELECT bddk_meta.corpus_retrieval_ready($1)", _PROFILE_SHA256)
            assert await connection.fetchval("SELECT count(*) FROM bddk_meta.active_corpus_release") == 0

        # Section coordinates and content must resolve to the canonical source
        # range, not merely carry a self-consistent hash.
        async with _rollback_savepoint(connection):
            await connection.execute(
                "UPDATE public.document_sections SET end_char = end_char - 1 WHERE doc_id = $1",
                document_id,
            )
            assert not await connection.fetchval("SELECT bddk_meta.corpus_retrieval_ready($1)", _PROFILE_SHA256)
            assert await connection.fetchval("SELECT count(*) FROM bddk_meta.active_corpus_release") == 0

        # Chunk text, ranges, document metadata, and section lineage are all
        # checked against canonical rows.  The chunk trigger also removes the
        # now-stale current-profile publication.
        async with _rollback_savepoint(connection):
            await connection.execute(
                "UPDATE public.document_chunks SET chunk_end_char = chunk_end_char - 1 WHERE doc_id = $1",
                document_id,
            )
            assert not await connection.fetchval("SELECT bddk_meta.corpus_retrieval_ready($1)", _PROFILE_SHA256)
            assert await connection.fetchval("SELECT count(*) FROM bddk_meta.active_corpus_release") == 0

        async with _rollback_savepoint(connection):
            await connection.execute(
                "UPDATE public.document_chunks SET section_content_hash = repeat('a', 64) WHERE doc_id = $1",
                document_id,
            )
            assert not await connection.fetchval("SELECT bddk_meta.corpus_retrieval_ready($1)", _PROFILE_SHA256)

        # The state digest also binds non-retrieval release inputs.  A decision
        # cache timestamp mutation remains retrieval-ready but invalidates the
        # exact release identity immediately.
        async with _rollback_savepoint(connection):
            await connection.execute(
                "UPDATE public.decision_cache SET cached_at = cached_at + 1 WHERE document_id = $1",
                document_id,
            )
            assert await connection.fetchval("SELECT bddk_meta.corpus_retrieval_ready($1)", _PROFILE_SHA256)
            assert await connection.fetchval("SELECT count(*) FROM bddk_meta.active_corpus_release") == 0
    finally:
        await transaction.rollback()
        await pg_pool.release(connection)


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_concurrent_publication_serializes_to_one_activation(pg_pool) -> None:
    """Two ingestion workers cannot race duplicate latest activations."""

    document_id = "corpus-release-concurrent"
    release_id: str | None = None
    role_existed = bool(
        await pg_pool.fetchval("SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'bddk_release_publisher')")
    )
    try:
        async with pg_pool.acquire() as connection, connection.transaction():
            await _ensure_release_publisher_role(connection)
            await _insert_ready_corpus(connection, document_id)

        first, second = await asyncio.gather(
            _publish(pg_pool, manifest_id="release-concurrent-001"),
            _publish(pg_pool, manifest_id="release-concurrent-001"),
        )
        assert first is not None and second is not None
        release_id = str(first["release_id"])
        assert second["release_id"] == release_id
        assert (
            await pg_pool.fetchval(
                "SELECT count(*) FROM bddk_meta.corpus_release_activations WHERE release_id = $1",
                release_id,
            )
            == 1
        )
    finally:
        # Append-only triggers protect production evidence.  This explicit
        # superuser-only teardown is confined to the disposable PG test DB.
        async with pg_pool.acquire() as connection, connection.transaction():
            cleanup_release_ids = await connection.fetch(
                "SELECT release_id FROM bddk_meta.corpus_releases WHERE manifest_id = 'release-concurrent-001'"
            )
            await connection.execute(
                "ALTER TABLE bddk_meta.corpus_release_activations "
                "DISABLE TRIGGER reject_corpus_release_activation_update_delete"
            )
            for cleanup_release in cleanup_release_ids:
                await connection.execute(
                    "DELETE FROM bddk_meta.corpus_release_activations WHERE release_id = $1",
                    cleanup_release["release_id"],
                )
            await connection.execute(
                "ALTER TABLE bddk_meta.corpus_release_activations "
                "ENABLE TRIGGER reject_corpus_release_activation_update_delete"
            )
            await connection.execute(
                "ALTER TABLE bddk_meta.corpus_releases DISABLE TRIGGER reject_corpus_release_update_delete"
            )
            for cleanup_release in cleanup_release_ids:
                await connection.execute(
                    "DELETE FROM bddk_meta.corpus_releases WHERE release_id = $1",
                    cleanup_release["release_id"],
                )
            await connection.execute(
                "ALTER TABLE bddk_meta.corpus_releases ENABLE TRIGGER reject_corpus_release_update_delete"
            )
            await connection.execute("DELETE FROM public.documents WHERE document_id = $1", document_id)
            await connection.execute("DELETE FROM public.decision_cache WHERE document_id = $1", document_id)
            if not role_existed:
                await connection.execute("DROP ROLE IF EXISTS bddk_release_publisher")


@pytest.mark.asyncio
async def test_active_release_reader_sanitizes_database_failures() -> None:
    class FailingPool:
        async def fetchrow(self, *_args):
            raise RuntimeError("private path and principal")

    with pytest.raises(CorpusPublicationError) as exc_info:
        await inspect_active_corpus_release(FailingPool())

    assert "private path" not in str(exc_info.value)
    assert "principal" not in str(exc_info.value)
