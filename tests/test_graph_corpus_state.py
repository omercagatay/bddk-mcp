"""Graph edits are governed release state, including retained evidence."""

import asyncpg
import pytest

from bddk_mcp.corpus_generations import retain_active_corpus_generation
from bddk_mcp.migrations import migrate
from bddk_mcp.operations.recovery import _RETAINED_GENERATION_SEAL_VALIDATION_SQL, _SAFE_FINGERPRINT_QUERIES
from tests.test_corpus_publication import (
    _PROFILE_SHA256,
    _ensure_release_publisher_role,
    _insert_canonical_legal_state,
    _insert_ready_corpus,
    _insert_relation,
    _publish,
    _rollback_savepoint,
)
from tests.test_migrations import (
    _downgrade_current_schema_to_v10,
    _ensure_v8_release_roles,
    _PinnedPool,
    _session_authorization,
    _stage_v8_release,
)

pytestmark = pytest.mark.asyncio


async def test_graph_edits_invalidate_release_requests_and_preserve_retained_rows(pg_pool):
    async with pg_pool.acquire() as connection, _rollback_savepoint(connection):
        await _ensure_v8_release_roles(connection)
        content_hash = await _insert_ready_corpus(connection, "graph-release")
        await _insert_canonical_legal_state(connection, document_id="graph-release", content_hash=content_hash)
        await _insert_relation(connection)
        async with _session_authorization(connection, "bddk_release_verifier"):
            staged = await _stage_v8_release(connection)
        async with _session_authorization(connection, "bddk_release_publisher"):
            await connection.fetchrow(
                "SELECT * FROM bddk_meta.activate_staged_corpus_release($1)", staged["request_id"]
            )
        receipt = await retain_active_corpus_generation(connection, expected_release_id=staged["release_id"])
        assert receipt.relation_count == 18
        assert await connection.fetchval(_RETAINED_GENERATION_SEAL_VALIDATION_SQL)
        assert await connection.fetchval("SELECT count(*) FROM public.regulatory_validated_relations") == 1

        async with _session_authorization(connection, "bddk_release_verifier"):
            pending = await _stage_v8_release(connection, manifest_id="graph-release-next")
        graph_query = dict(_SAFE_FINGERPRINT_QUERIES)["regulatory_relations"]
        graph_fingerprint = await connection.fetch(graph_query)
        for mutation in (
            "UPDATE public.regulatory_relations SET validation_state = 'rejected'",
            "DELETE FROM public.regulatory_relations",
            "TRUNCATE public.regulatory_relations",
        ):
            async with _rollback_savepoint(connection):
                await connection.execute(mutation)
                assert await connection.fetch(graph_query) != graph_fingerprint
                assert not await connection.fetchval("SELECT count(*) FROM bddk_meta.active_corpus_release")
                assert not await connection.fetchval("SELECT count(*) FROM public.regulatory_validated_relations")
                assert (
                    await connection.fetchval("SELECT bddk_meta.current_corpus_state_sha256($1)", _PROFILE_SHA256)
                    != receipt.corpus_state_sha256
                )
                assert (
                    await connection.fetchval(
                        "SELECT bddk_meta.retained_corpus_state_sha256($1, $2)", receipt.generation_id, _PROFILE_SHA256
                    )
                    == receipt.corpus_state_sha256
                )
                async with _session_authorization(connection, "bddk_release_publisher"):
                    async with _rollback_savepoint(connection):
                        with pytest.raises(asyncpg.ObjectNotInPrerequisiteStateError):
                            await connection.fetchrow(
                                "SELECT * FROM bddk_meta.activate_staged_corpus_release($1)", pending["request_id"]
                            )
        async with _rollback_savepoint(connection):
            await connection.execute(
                "ALTER TABLE bddk_retained.regulatory_relations DISABLE TRIGGER guard_retained_generation_member"
            )
            await connection.execute("UPDATE bddk_retained.regulatory_relations SET confidence = 0.5")
            await connection.execute(
                "ALTER TABLE bddk_retained.regulatory_relations ENABLE TRIGGER guard_retained_generation_member"
            )
            assert not await connection.fetchval(_RETAINED_GENERATION_SEAL_VALIDATION_SQL)


async def test_upgrade_preserves_legacy_seals_and_requires_a_new_release(pg_pool):
    async with pg_pool.acquire() as connection, _rollback_savepoint(connection):
        await _downgrade_current_schema_to_v10(connection)
        await _ensure_release_publisher_role(connection)
        content_hash = await _insert_ready_corpus(connection, "legacy-graph-upgrade")
        await _insert_canonical_legal_state(connection, document_id="legacy-graph-upgrade", content_hash=content_hash)
        old_release = await _publish(connection)
        old_receipt = await retain_active_corpus_generation(connection, expected_release_id=old_release["release_id"])
        assert old_receipt.relation_count == 17
        # V10 never retained these later relation claims. The upgrade must not
        # rewrite the old seal to pretend otherwise.
        await _insert_relation(connection)
        await migrate(_PinnedPool(connection))
        assert not await connection.fetchval("SELECT count(*) FROM bddk_meta.active_corpus_release")
        assert await connection.fetchval(_RETAINED_GENERATION_SEAL_VALIDATION_SQL)
        assert (
            await connection.fetchval(
                "SELECT bddk_meta.retained_corpus_state_sha256($1, $2)", old_receipt.generation_id, _PROFILE_SHA256
            )
            == old_receipt.corpus_state_sha256
        )
        new_release = await _publish(connection)
        assert new_release["release_id"] != old_release["release_id"]
        new_receipt = await retain_active_corpus_generation(connection, expected_release_id=new_release["release_id"])
        assert new_receipt.relation_count == 18
        assert await connection.fetchval(_RETAINED_GENERATION_SEAL_VALIDATION_SQL)
