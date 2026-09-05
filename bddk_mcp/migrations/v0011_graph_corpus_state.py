"""Migration 0011: include relation claims in release and retention identity.

New states have an explicit fingerprint format marker and new generations use
schema 2 (18 relations). Sealed schema-1 generations keep their original hash
and inventory. The epoch bump invalidates pre-upgrade activations and requests;
operators must verify, stage, and activate the corpus again after migration.
"""

from typing import Final

from bddk_mcp.corpus_coordination import CORPUS_MUTATION_ADVISORY_KEY
from bddk_mcp.migrations import v0007_retained_corpus_generations as v7
from bddk_mcp.migrations.model import Migration
from bddk_mcp.migrations.v0008_staged_corpus_releases import V0008_STAGED_CORPUS_RELEASES
from bddk_mcp.migrations.v0010_corpus_release_freshness_policy import V0010_CORPUS_RELEASE_FRESHNESS_POLICY

RETAINED_CORPUS_RELATIONS: Final[tuple[str, ...]] = (*v7.RETAINED_CORPUS_RELATIONS, "regulatory_relations")
CORPUS_EPOCH_TRACKED_TABLES: Final[tuple[str, ...]] = RETAINED_CORPUS_RELATIONS
_RELATION_ARRAY = "ARRAY[" + ",".join(f"'{relation}'" for relation in RETAINED_CORPUS_RELATIONS) + "]"
_SOURCE_TABLE_LOCKS = ",\n                       ".join(f"public.{relation}" for relation in RETAINED_CORPUS_RELATIONS)


def _replace(source: str, old: str, new: str, *, count: int = 1) -> str:
    if source.count(old) != count:
        raise RuntimeError(f"graph corpus migration source changed: {old}")
    return source.replace(old, new)


def _function(migration: Migration, name: str) -> str:
    source = next(s for s in migration.statements if f"CREATE FUNCTION bddk_meta.{name}(" in s)
    return _replace(source, "CREATE FUNCTION", "CREATE OR REPLACE FUNCTION")


def _state_fingerprint(*, retained: bool) -> str:
    source = v7._v5_state_fingerprint_statement() if retained else v7._canonical_current_state_fingerprint_statement()
    if retained:
        source = _replace(source, "CREATE FUNCTION", "CREATE OR REPLACE FUNCTION")
    relation = "bddk_retained.regulatory_relations" if retained else "public.regulatory_relations"
    predicate = "WHERE relation.generation_id = requested_generation_id" if retained else ""
    # The existing row hash canonicalizes every typed column, including review
    # provenance. The retained copy strips only its generation_id envelope.
    source = _replace(
        source,
        "), combined AS (",
        f"""    UNION ALL
            SELECT 'regulatory_relation', relation.relation_id,
                   pg_catalog.decode(bddk_meta.retained_row_sha256(relation, {str(retained).lower()}), 'hex')
            FROM {relation} AS relation
            {predicate}
        ), combined AS (""",
    )
    prefix = "bddk_meta.corpus_fingerprint_frame('2')"
    if retained:
        prefix = f"""CASE WHEN (
                           SELECT generation.generation_schema_version
                           FROM bddk_meta.corpus_generations AS generation
                           WHERE generation.generation_id = requested_generation_id
                       ) = 2 THEN {prefix}
                       ELSE pg_catalog.decode('', 'hex') END"""
    return _replace(
        source,
        "bddk_meta.corpus_fingerprint_frame(requested_retrieval_profile_sha256)",
        prefix + "\n                       || bddk_meta.corpus_fingerprint_frame(requested_retrieval_profile_sha256)",
    )


def _retain_function() -> str:
    source = _function(v7.V0007_RETAINED_CORPUS_GENERATIONS, "retain_active_corpus_generation")
    source = _replace(source, v7._SOURCE_TABLE_LOCKS, _SOURCE_TABLE_LOCKS)
    source = _replace(source, v7._RELATION_ARRAY, _RELATION_ARRAY, count=2)
    source = _replace(
        source, "selected_generation_id,\n                1,", "selected_generation_id,\n                2,"
    )
    source = _replace(source, "sealed_relation_count IS DISTINCT FROM 17", "sealed_relation_count IS DISTINCT FROM 18")
    source = _replace(source, "                17,", "                18,")
    return _replace(source, "17::pg_catalog.int4", "18::pg_catalog.int4")


def _storage_function() -> str:
    source = _function(v7.V0007_RETAINED_CORPUS_GENERATIONS, "inspect_retained_generation_storage")
    # Storage totals cover the whole retained store, while a generation's
    # relation count comes from its original seal (17 or 18).
    source = _replace(source, v7._RELATION_ARRAY, _RELATION_ARRAY)
    return _replace(
        source,
        "17::pg_catalog.int4",
        "(SELECT seal.relation_count FROM bddk_meta.corpus_generation_seals AS seal "
        "WHERE seal.generation_id = requested_generation_id)",
    )


V0011_GRAPH_CORPUS_STATE = Migration(
    version=11,
    name="graph_corpus_state",
    statements=(
        f"SELECT pg_catalog.pg_advisory_xact_lock({CORPUS_MUTATION_ADVISORY_KEY}::pg_catalog.int8)",
        """
        CREATE TRIGGER bump_corpus_state_epoch_on_change
        AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON public.regulatory_relations
        FOR EACH STATEMENT EXECUTE FUNCTION bddk_meta.bump_corpus_state_epoch()
        """,
        """
        ALTER TABLE bddk_meta.corpus_generations
        DROP CONSTRAINT corpus_generations_schema_check,
        ADD CONSTRAINT corpus_generations_schema_check CHECK (generation_schema_version IN (1, 2))
        """,
        """
        ALTER TABLE bddk_meta.corpus_generation_seals
        DROP CONSTRAINT corpus_generation_seals_relation_count_check,
        ADD CONSTRAINT corpus_generation_seals_relation_count_check CHECK (relation_count IN (17, 18))
        """,
        f"""
        ALTER TABLE bddk_meta.corpus_generation_relation_inventory
        DROP CONSTRAINT corpus_generation_relation_inventory_name_check,
        ADD CONSTRAINT corpus_generation_relation_inventory_name_check
            CHECK (relation_name = ANY({_RELATION_ARRAY}))
        """,
        """
        CREATE TABLE bddk_retained.regulatory_relations (
            generation_id pg_catalog.text NOT NULL,
            LIKE public.regulatory_relations INCLUDING STORAGE INCLUDING COMPRESSION,
            CONSTRAINT rt_18_generation_fk FOREIGN KEY (generation_id)
                REFERENCES bddk_meta.corpus_generations(generation_id),
            CONSTRAINT rt_18_pkey PRIMARY KEY (generation_id, relation_id)
        )
        """,
        *(
            v7._retained_foreign_key_statement(position, "regulatory_relations", (column,), target, (target_column,))
            for position, (column, target, target_column) in enumerate(
                (
                    ("source_instrument_id", "regulatory_instruments", "instrument_id"),
                    ("source_provision_id", "regulatory_provisions", "provision_id"),
                    ("target_instrument_id", "regulatory_instruments", "instrument_id"),
                    ("target_provision_id", "regulatory_provisions", "provision_id"),
                    ("evidence_id", "regulatory_evidence", "evidence_id"),
                ),
                start=len(v7._FOREIGN_KEYS) + 1,
            )
        ),
        """
        CREATE TRIGGER guard_retained_generation_member
        BEFORE INSERT OR UPDATE OR DELETE ON bddk_retained.regulatory_relations
        FOR EACH ROW EXECUTE FUNCTION bddk_meta.guard_retained_generation_member()
        """,
        """
        CREATE TRIGGER reject_retained_generation_truncate
        BEFORE TRUNCATE ON bddk_retained.regulatory_relations
        FOR EACH STATEMENT EXECUTE FUNCTION bddk_meta.reject_retained_generation_mutation()
        """,
        "REVOKE ALL PRIVILEGES ON TABLE bddk_retained.regulatory_relations FROM PUBLIC",
        _state_fingerprint(retained=False),
        _state_fingerprint(retained=True),
        _retain_function(),
        _storage_function(),
        *(
            _replace(_function(migration, name), v7._SOURCE_TABLE_LOCKS, _SOURCE_TABLE_LOCKS)
            for migration, name in (
                (V0010_CORPUS_RELEASE_FRESHNESS_POLICY, "stage_verified_corpus_release"),
                (V0008_STAGED_CORPUS_RELEASES, "activate_staged_corpus_release"),
            )
        ),
        "UPDATE bddk_meta.corpus_state_epoch SET epoch = epoch + 1 WHERE singleton_id",
    ),
)
