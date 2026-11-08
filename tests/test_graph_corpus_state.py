"""Graph edits are governed release state, including retained evidence."""

from typing import Final

import asyncpg
import pytest

from bddk_mcp.corpus_generations import retain_active_corpus_generation
from bddk_mcp.migrations import migrate
from bddk_mcp.migrations.v0011_graph_corpus_state import RETAINED_CORPUS_RELATIONS
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

# Retained-generation seal check, kept here after the recovery-drill module was removed.
_RETAINED_MEMBER_ORPHANS_SQL: Final[str] = "\nUNION ALL\n".join(
    f"""
    SELECT 1 AS orphaned
    FROM bddk_retained.{relation} AS member
    LEFT JOIN bddk_meta.corpus_generations AS generation
      ON generation.generation_id = member.generation_id
    WHERE generation.generation_id IS NULL
    """
    for relation in RETAINED_CORPUS_RELATIONS
)


def _retained_inventory_recomputation_select(position: int, relation: str) -> str:
    """Recompute one v7 inventory member without returning retained content."""

    if relation not in RETAINED_CORPUS_RELATIONS:
        raise ValueError("retained relation is not in the immutable migration inventory")
    return f"""
        SELECT generation.generation_id,
               {position}::pg_catalog.int4 AS relation_position,
               '{relation}'::pg_catalog.text AS relation_name,
               recomputed.row_count,
               recomputed.relation_sha256
        FROM bddk_meta.corpus_generations AS generation
        CROSS JOIN LATERAL (
            SELECT pg_catalog.count(*)::pg_catalog.int8 AS row_count,
                   pg_catalog.encode(
                       pg_catalog.sha256(
                           pg_catalog.convert_to(
                               COALESCE(
                                   pg_catalog.string_agg(row_sha256, '' ORDER BY row_sha256),
                                   ''
                               ),
                               'UTF8'
                           )
                       ),
                       'hex'
                   ) AS relation_sha256
            FROM (
                SELECT bddk_meta.retained_row_sha256(member, true) AS row_sha256
                FROM bddk_retained.{relation} AS member
                WHERE member.generation_id = generation.generation_id
            ) AS retained_rows
        ) AS recomputed
        WHERE {position} <= 17 OR generation.generation_schema_version = 2
        """


_FRESH_RETAINED_INVENTORY_SQL: Final[str] = "\nUNION ALL\n".join(
    _retained_inventory_recomputation_select(position, relation)
    for position, relation in enumerate(RETAINED_CORPUS_RELATIONS, start=1)
)
_RETAINED_GENERATION_SEAL_VALIDATION_SQL: Final[str] = f"""
WITH fresh_inventory AS MATERIALIZED (
    {_FRESH_RETAINED_INVENTORY_SQL}
),
recomputed_generations AS MATERIALIZED (
    SELECT generation.generation_id,
           CASE generation.generation_schema_version WHEN 1 THEN 17 ELSE 18 END AS expected_relation_count,
           generation.source_activation_sequence,
           generation.source_release_id,
           generation.corpus_state_sha256,
           generation.retrieval_profile_sha256,
           pg_catalog.count(fresh.relation_name)::pg_catalog.int4
               AS fresh_relation_count,
           COALESCE(pg_catalog.sum(fresh.row_count), 0)::pg_catalog.int8
               AS fresh_row_count,
           pg_catalog.encode(
               pg_catalog.sha256(
                   bddk_meta.corpus_fingerprint_frame('1')
                   || COALESCE(
                          pg_catalog.string_agg(
                              bddk_meta.corpus_fingerprint_frame(fresh.relation_name)
                              || bddk_meta.corpus_fingerprint_frame(
                                     fresh.row_count::pg_catalog.text
                                 )
                              || bddk_meta.corpus_fingerprint_frame(
                                     fresh.relation_sha256
                                 ),
                              pg_catalog.decode('', 'hex')
                              ORDER BY fresh.relation_position
                          ),
                          pg_catalog.decode('', 'hex')
                      )
               ),
               'hex'
           ) AS fresh_inventory_sha256
    FROM bddk_meta.corpus_generations AS generation
    JOIN fresh_inventory AS fresh
      ON fresh.generation_id = generation.generation_id
    GROUP BY generation.generation_id,
             generation.source_activation_sequence,
             generation.source_release_id,
             generation.corpus_state_sha256,
             generation.retrieval_profile_sha256
),
validation AS MATERIALIZED (
    SELECT recomputed.*,
           seal.seal_id,
           seal.corpus_state_sha256 AS sealed_state_sha256,
           seal.retrieval_profile_sha256 AS sealed_profile_sha256,
           seal.inventory_sha256 AS sealed_inventory_sha256,
           seal.relation_count AS sealed_relation_count,
           seal.row_count AS sealed_row_count
    FROM recomputed_generations AS recomputed
    LEFT JOIN bddk_meta.corpus_generation_seals AS seal
      ON seal.generation_id = recomputed.generation_id
)
SELECT NOT EXISTS (
    SELECT 1
    FROM validation
    WHERE fresh_relation_count <> expected_relation_count
       OR (
            SELECT pg_catalog.count(*)
            FROM bddk_meta.corpus_generation_relation_inventory AS inventory
            WHERE inventory.generation_id = validation.generation_id
          ) <> expected_relation_count
       OR EXISTS (
            SELECT 1
            FROM fresh_inventory AS fresh
            LEFT JOIN bddk_meta.corpus_generation_relation_inventory AS inventory
              ON inventory.generation_id = fresh.generation_id
             AND inventory.relation_name = fresh.relation_name
            WHERE fresh.generation_id = validation.generation_id
              AND (
                    inventory.generation_id IS NULL
                    OR inventory.row_count IS DISTINCT FROM fresh.row_count
                    OR inventory.relation_sha256 IS DISTINCT FROM fresh.relation_sha256
                  )
          )
       OR seal_id IS NULL
       OR sealed_relation_count IS DISTINCT FROM expected_relation_count
       OR sealed_row_count IS DISTINCT FROM fresh_row_count
       OR sealed_state_sha256 IS DISTINCT FROM corpus_state_sha256
       OR sealed_profile_sha256 IS DISTINCT FROM retrieval_profile_sha256
       OR sealed_inventory_sha256 IS DISTINCT FROM fresh_inventory_sha256
       OR bddk_meta.retained_corpus_state_sha256(
              validation.generation_id,
              validation.retrieval_profile_sha256
          ) IS DISTINCT FROM sealed_state_sha256
       OR validation.generation_id IS DISTINCT FROM (
              'corpus_generation_sha256_'
              || pg_catalog.encode(
                     pg_catalog.sha256(
                         bddk_meta.corpus_fingerprint_frame('1')
                         || bddk_meta.corpus_fingerprint_frame(corpus_state_sha256)
                         || bddk_meta.corpus_fingerprint_frame(retrieval_profile_sha256)
                     ),
                     'hex'
                 )
          )
       OR seal_id IS DISTINCT FROM (
              'corpus_generation_seal_sha256_'
              || pg_catalog.encode(
                     pg_catalog.sha256(
                         bddk_meta.corpus_fingerprint_frame('1')
                         || bddk_meta.corpus_fingerprint_frame(validation.generation_id)
                         || bddk_meta.corpus_fingerprint_frame(corpus_state_sha256)
                         || bddk_meta.corpus_fingerprint_frame(retrieval_profile_sha256)
                         || bddk_meta.corpus_fingerprint_frame(fresh_inventory_sha256)
                     ),
                     'hex'
                 )
          )
       OR NOT EXISTS (
            SELECT 1
            FROM bddk_meta.corpus_retained_releases AS binding
            WHERE binding.release_id = validation.source_release_id
              AND binding.seal_id = validation.seal_id
              AND binding.generation_id = validation.generation_id
              AND binding.corpus_state_sha256 = validation.corpus_state_sha256
              AND binding.retrieval_profile_sha256 =
                  validation.retrieval_profile_sha256
          )
)
AND NOT EXISTS (
    SELECT 1
    FROM bddk_meta.corpus_generation_seals AS seal
    LEFT JOIN bddk_meta.corpus_generations AS generation
      ON generation.generation_id = seal.generation_id
    WHERE generation.generation_id IS NULL
)
AND NOT EXISTS (
    SELECT 1
    FROM bddk_meta.corpus_generation_relation_inventory AS inventory
    LEFT JOIN bddk_meta.corpus_generations AS generation
      ON generation.generation_id = inventory.generation_id
    WHERE generation.generation_id IS NULL
)
AND NOT EXISTS (
    SELECT 1
    FROM bddk_meta.corpus_retained_releases AS binding
    LEFT JOIN bddk_meta.corpus_generations AS generation
      ON generation.generation_id = binding.generation_id
    LEFT JOIN bddk_meta.corpus_generation_seals AS seal
      ON seal.seal_id = binding.seal_id
     AND seal.generation_id = binding.generation_id
     AND seal.corpus_state_sha256 = binding.corpus_state_sha256
     AND seal.retrieval_profile_sha256 = binding.retrieval_profile_sha256
    LEFT JOIN bddk_meta.corpus_releases AS release
      ON release.release_id = binding.release_id
     AND release.corpus_state_sha256 = binding.corpus_state_sha256
     AND release.retrieval_profile_sha256 = binding.retrieval_profile_sha256
    WHERE generation.generation_id IS NULL
       OR seal.seal_id IS NULL
       OR release.release_id IS NULL
)
AND NOT EXISTS (
    SELECT 1
    FROM (
        {_RETAINED_MEMBER_ORPHANS_SQL}
    ) AS orphaned_members
)
AS retained_generation_seals_valid
"""

_GRAPH_RELATIONS_FINGERPRINT_SQL: Final[str] = (
    "SELECT bddk_meta.retained_row_sha256(member, false) AS row_sha256 "
    "FROM public.regulatory_relations AS member ORDER BY row_sha256"
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
        graph_query = _GRAPH_RELATIONS_FINGERPRINT_SQL
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
