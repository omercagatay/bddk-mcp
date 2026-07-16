"""Migration 0007: immutable typed retention of one verified v5 corpus state.

This migration is deliberately additive.  It does not change the v5 active
release view, serving queries, or activation semantics.  A narrowly authorized
release publisher can atomically copy the exact 17-table state behind the
currently active release, reproduce its v5 fingerprint over typed retained
rows, seal the inventory, and bind that seal to the governed release.
"""

from __future__ import annotations

from typing import Final

from bddk_mcp.corpus_coordination import CORPUS_MUTATION_ADVISORY_KEY, SCHEMA_MIGRATION_ADVISORY_KEY
from bddk_mcp.migrations.model import Migration
from bddk_mcp.migrations.v0005_corpus_release_publication import (
    CORPUS_EPOCH_TRACKED_TABLES,
    V0005_CORPUS_RELEASE_PUBLICATION,
)

RETAINED_CORPUS_RELATIONS: Final[tuple[str, ...]] = CORPUS_EPOCH_TRACKED_TABLES
NONCANONICAL_FINGERPRINT_UPGRADE_SQLSTATE: Final[str] = "P7001"

_PRIMARY_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "decision_cache": ("document_id",),
    "documents": ("document_id",),
    "document_sections": ("id",),
    "document_versions": ("id",),
    "document_chunks": ("id",),
    "document_retrieval_publications": ("doc_id",),
    "regulatory_instruments": ("instrument_id",),
    "regulatory_family_imports": ("bundle_id", "bundle_sha256"),
    "regulatory_source_blobs": ("blob_id",),
    "regulatory_source_artifacts": ("artifact_id",),
    "regulatory_evidence": ("evidence_id",),
    "regulatory_legal_versions": ("legal_version_id",),
    "regulatory_legal_version_artifacts": ("legal_version_id", "artifact_id", "source_role"),
    "regulatory_legal_events": ("event_id",),
    "regulatory_legal_status_assertions": ("assertion_id",),
    "regulatory_provisions": ("provision_id",),
    "regulatory_legal_version_provisions": ("legal_version_id", "provision_id"),
}

_UNIQUE_KEYS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("document_sections", ("doc_id", "section_type", "section_ref", "content_hash")),
    ("document_versions", ("document_id", "version")),
    ("document_chunks", ("doc_id", "chunk_index")),
    ("regulatory_instruments", ("jurisdiction", "authority_code", "identity_key")),
    ("regulatory_source_blobs", ("content_sha256",)),
    ("regulatory_source_artifacts", ("blob_id", "canonical_uri", "retrieved_at")),
    ("regulatory_legal_versions", ("instrument_id", "version_key")),
    ("regulatory_legal_events", ("legal_version_id", "event_type")),
    ("regulatory_provisions", ("instrument_id", "provision_kind", "canonical_path")),
    ("regulatory_legal_version_provisions", ("document_section_id",)),
)

_FOREIGN_KEYS: Final[tuple[tuple[str, tuple[str, ...], str, tuple[str, ...]], ...]] = (
    ("document_sections", ("doc_id",), "documents", ("document_id",)),
    ("document_versions", ("document_id",), "documents", ("document_id",)),
    ("document_chunks", ("doc_id",), "documents", ("document_id",)),
    ("document_retrieval_publications", ("doc_id",), "documents", ("document_id",)),
    ("regulatory_family_imports", ("instrument_id",), "regulatory_instruments", ("instrument_id",)),
    (
        "regulatory_family_imports",
        ("bundle_id", "predecessor_bundle_sha256"),
        "regulatory_family_imports",
        ("bundle_id", "bundle_sha256"),
    ),
    ("regulatory_source_artifacts", ("blob_id",), "regulatory_source_blobs", ("blob_id",)),
    ("regulatory_source_artifacts", ("repository_document_id",), "documents", ("document_id",)),
    ("regulatory_evidence", ("artifact_id",), "regulatory_source_artifacts", ("artifact_id",)),
    ("regulatory_legal_versions", ("instrument_id",), "regulatory_instruments", ("instrument_id",)),
    (
        "regulatory_legal_versions",
        ("predecessor_version_id",),
        "regulatory_legal_versions",
        ("legal_version_id",),
    ),
    (
        "regulatory_legal_version_artifacts",
        ("legal_version_id",),
        "regulatory_legal_versions",
        ("legal_version_id",),
    ),
    (
        "regulatory_legal_version_artifacts",
        ("artifact_id",),
        "regulatory_source_artifacts",
        ("artifact_id",),
    ),
    ("regulatory_legal_events", ("legal_version_id",), "regulatory_legal_versions", ("legal_version_id",)),
    (
        "regulatory_legal_events",
        ("target_legal_version_id",),
        "regulatory_legal_versions",
        ("legal_version_id",),
    ),
    ("regulatory_legal_events", ("evidence_id",), "regulatory_evidence", ("evidence_id",)),
    (
        "regulatory_legal_status_assertions",
        ("legal_version_id",),
        "regulatory_legal_versions",
        ("legal_version_id",),
    ),
    (
        "regulatory_legal_status_assertions",
        ("evidence_id",),
        "regulatory_evidence",
        ("evidence_id",),
    ),
    ("regulatory_provisions", ("instrument_id",), "regulatory_instruments", ("instrument_id",)),
    (
        "regulatory_legal_version_provisions",
        ("legal_version_id",),
        "regulatory_legal_versions",
        ("legal_version_id",),
    ),
    (
        "regulatory_legal_version_provisions",
        ("provision_id",),
        "regulatory_provisions",
        ("provision_id",),
    ),
    (
        "regulatory_legal_version_provisions",
        ("document_section_id",),
        "document_sections",
        ("id",),
    ),
    (
        "regulatory_legal_version_provisions",
        ("evidence_id",),
        "regulatory_evidence",
        ("evidence_id",),
    ),
)

_CANONICAL_FINGERPRINT_FUNCTION_SETTINGS: Final[str] = """SET search_path = pg_catalog
        SET TimeZone = 'UTC'
        SET DateStyle = 'ISO, YMD'
        SET IntervalStyle = 'postgres'
        SET bytea_output = 'hex'
        SET extra_float_digits = 3"""


def _canonical_current_state_fingerprint_statement() -> str:
    """Recreate the v5 state hash with session-independent text rendering."""

    source = next(
        statement
        for statement in V0005_CORPUS_RELEASE_PUBLICATION.statements
        if "CREATE FUNCTION bddk_meta.current_corpus_state_sha256" in statement
    )
    source = source.replace("CREATE FUNCTION", "CREATE OR REPLACE FUNCTION", 1)
    if source.count("SET search_path = pg_catalog") != 1:
        raise RuntimeError("v5 state fingerprint settings changed")
    return source.replace(
        "SET search_path = pg_catalog",
        _CANONICAL_FINGERPRINT_FUNCTION_SETTINGS,
        1,
    )


def _retained_table_statement(position: int, relation: str) -> str:
    primary_columns = ", ".join(("generation_id", *_PRIMARY_KEYS[relation]))
    return f"""
        CREATE TABLE bddk_retained.{relation} (
            generation_id pg_catalog.text NOT NULL,
            LIKE public.{relation} INCLUDING STORAGE INCLUDING COMPRESSION,
            CONSTRAINT rt_{position:02d}_generation_fk
                FOREIGN KEY (generation_id)
                REFERENCES bddk_meta.corpus_generations(generation_id),
            CONSTRAINT rt_{position:02d}_pkey PRIMARY KEY ({primary_columns})
        )
        """


def _retained_unique_statement(position: int, relation: str, columns: tuple[str, ...]) -> str:
    qualified_columns = ", ".join(("generation_id", *columns))
    return f"""
        ALTER TABLE bddk_retained.{relation}
        ADD CONSTRAINT rt_uq_{position:02d} UNIQUE ({qualified_columns})
        """


def _retained_foreign_key_statement(
    position: int,
    relation: str,
    columns: tuple[str, ...],
    target_relation: str,
    target_columns: tuple[str, ...],
) -> str:
    source = ", ".join(("generation_id", *columns))
    target = ", ".join(("generation_id", *target_columns))
    return f"""
        ALTER TABLE bddk_retained.{relation}
        ADD CONSTRAINT rt_fk_{position:02d}
        FOREIGN KEY ({source})
        REFERENCES bddk_retained.{target_relation}({target})
        """


def _v5_state_fingerprint_statement() -> str:
    source = next(
        statement
        for statement in V0005_CORPUS_RELEASE_PUBLICATION.statements
        if "CREATE FUNCTION bddk_meta.current_corpus_state_sha256" in statement
    )
    source = source.replace(
        "CREATE FUNCTION bddk_meta.current_corpus_state_sha256(\n"
        "            requested_retrieval_profile_sha256 pg_catalog.text\n"
        "        )",
        "CREATE FUNCTION bddk_meta.retained_corpus_state_sha256(\n"
        "            requested_generation_id pg_catalog.text,\n"
        "            requested_retrieval_profile_sha256 pg_catalog.text\n"
        "        )",
        1,
    )
    aliases = {
        "decision_cache": "cache",
        "documents": "document",
        "document_sections": "section",
        "document_versions": "version",
        "document_chunks": "chunk",
        "regulatory_instruments": "instrument",
        "regulatory_family_imports": "import_record",
        "regulatory_source_blobs": "blob",
        "regulatory_source_artifacts": "artifact",
        "regulatory_evidence": "evidence",
        "regulatory_legal_versions": "version",
        "regulatory_legal_version_artifacts": "version_artifact",
        "regulatory_legal_events": "event",
        "regulatory_legal_status_assertions": "assertion",
        "regulatory_provisions": "provision",
        "regulatory_legal_version_provisions": "occurrence",
    }
    for relation, alias in aliases.items():
        needle = f"FROM public.{relation} AS {alias}"
        if source.count(needle) != 1:
            raise RuntimeError(f"v5 state fingerprint source changed for {relation}")
        source = source.replace(
            needle,
            f"FROM bddk_retained.{relation} AS {alias}\n"
            f"            WHERE {alias}.generation_id = requested_generation_id",
            1,
        )
    publication_needle = (
        "FROM public.document_retrieval_publications AS publication\n"
        "            WHERE publication.retrieval_profile_hash = requested_retrieval_profile_sha256"
    )
    if source.count(publication_needle) != 1:
        raise RuntimeError("v5 state fingerprint source changed for retrieval publications")
    source = source.replace(
        publication_needle,
        "FROM bddk_retained.document_retrieval_publications AS publication\n"
        "            WHERE publication.generation_id = requested_generation_id\n"
        "              AND publication.retrieval_profile_hash = requested_retrieval_profile_sha256",
        1,
    )
    if source.count("SET search_path = pg_catalog") != 1:
        raise RuntimeError("retained state fingerprint settings changed")
    return source.replace(
        "SET search_path = pg_catalog",
        _CANONICAL_FINGERPRINT_FUNCTION_SETTINGS,
        1,
    )


_RELATION_ARRAY = "ARRAY[" + ",".join(f"'{relation}'" for relation in RETAINED_CORPUS_RELATIONS) + "]"
_RELATION_CHECK = ", ".join(f"'{relation}'" for relation in RETAINED_CORPUS_RELATIONS)
_SOURCE_TABLE_LOCKS = ",\n                       ".join(f"public.{relation}" for relation in RETAINED_CORPUS_RELATIONS)

V0007_RETAINED_CORPUS_GENERATIONS = Migration(
    version=7,
    name="retained_corpus_generations",
    statements=(
        f"""
        DO $canonical_fingerprint_upgrade_guard$
        DECLARE
            active_state_sha256 pg_catalog.text;
            active_profile_sha256 pg_catalog.text;
            canonical_state_sha256 pg_catalog.text;
        BEGIN
            -- The runner already owns the schema-migration advisory lock.
            -- Join the global schema -> corpus lock order before observing the
            -- v5 active head, so a publisher that began first must commit or
            -- roll back before this guard decides whether v7 may proceed.
            PERFORM pg_catalog.pg_advisory_xact_lock(
                {CORPUS_MUTATION_ADVISORY_KEY}::pg_catalog.int8
            );
            PERFORM pg_catalog.set_config('TimeZone', 'UTC', true);
            PERFORM pg_catalog.set_config('DateStyle', 'ISO, YMD', true);
            PERFORM pg_catalog.set_config('IntervalStyle', 'postgres', true);
            PERFORM pg_catalog.set_config('bytea_output', 'hex', true);
            PERFORM pg_catalog.set_config('extra_float_digits', '3', true);

            SELECT active.corpus_state_sha256,
                   active.retrieval_profile_sha256
            INTO active_state_sha256,
                 active_profile_sha256
            FROM bddk_meta.active_corpus_release AS active;
            IF active_state_sha256 IS NOT NULL THEN
                canonical_state_sha256 := bddk_meta.current_corpus_state_sha256(
                    active_profile_sha256
                );
                IF canonical_state_sha256 IS DISTINCT FROM active_state_sha256 THEN
                    RAISE EXCEPTION
                        'active pre-v7 corpus release uses a noncanonical fingerprint; '
                        'republish it on schema v5 or v6 under the documented canonical settings before v7'
                        USING ERRCODE = '{NONCANONICAL_FINGERPRINT_UPGRADE_SQLSTATE}';
                END IF;
            END IF;
        END
        $canonical_fingerprint_upgrade_guard$
        """,
        """
        ALTER TABLE bddk_meta.corpus_releases
        ADD CONSTRAINT corpus_releases_retention_identity_uq
        UNIQUE (release_id, corpus_state_sha256, retrieval_profile_sha256)
        """,
        """
        ALTER TABLE bddk_meta.corpus_release_activations
        ADD CONSTRAINT corpus_release_activations_retention_identity_uq
        UNIQUE (activation_sequence, release_id)
        """,
        _canonical_current_state_fingerprint_statement(),
        "CREATE SCHEMA bddk_retained",
        """
        CREATE TABLE bddk_meta.corpus_generations (
            generation_id pg_catalog.text PRIMARY KEY,
            generation_schema_version pg_catalog.int4 NOT NULL,
            source_activation_sequence pg_catalog.int8 NOT NULL UNIQUE,
            source_release_id pg_catalog.text NOT NULL,
            corpus_state_sha256 pg_catalog.text NOT NULL,
            retrieval_profile_sha256 pg_catalog.text NOT NULL,
            staged_at pg_catalog.timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
            staged_by_fingerprint_sha256 pg_catalog.text NOT NULL,
            CONSTRAINT corpus_generations_id_check
                CHECK (generation_id ~ '^corpus_generation_sha256_[0-9a-f]{64}$'),
            CONSTRAINT corpus_generations_schema_check CHECK (generation_schema_version = 1),
            CONSTRAINT corpus_generations_state_hash_check
                CHECK (corpus_state_sha256 ~ '^[0-9a-f]{64}$'),
            CONSTRAINT corpus_generations_profile_hash_check
                CHECK (retrieval_profile_sha256 ~ '^[0-9a-f]{64}$'),
            CONSTRAINT corpus_generations_actor_hash_check
                CHECK (staged_by_fingerprint_sha256 ~ '^[0-9a-f]{64}$'),
            CONSTRAINT corpus_generations_activation_fk
                FOREIGN KEY (source_activation_sequence, source_release_id)
                REFERENCES bddk_meta.corpus_release_activations(
                    activation_sequence, release_id
                ),
            CONSTRAINT corpus_generations_release_fk
                FOREIGN KEY (source_release_id)
                REFERENCES bddk_meta.corpus_releases(release_id),
            CONSTRAINT corpus_generations_state_identity_uq
                UNIQUE (generation_id, corpus_state_sha256, retrieval_profile_sha256)
        )
        """,
        f"""
        CREATE TABLE bddk_meta.corpus_generation_relation_inventory (
            generation_id pg_catalog.text NOT NULL,
            relation_name pg_catalog.text NOT NULL,
            row_count pg_catalog.int8 NOT NULL,
            relation_sha256 pg_catalog.text NOT NULL,
            CONSTRAINT corpus_generation_relation_inventory_pkey
                PRIMARY KEY (generation_id, relation_name),
            CONSTRAINT corpus_generation_relation_inventory_generation_fk
                FOREIGN KEY (generation_id)
                REFERENCES bddk_meta.corpus_generations(generation_id),
            CONSTRAINT corpus_generation_relation_inventory_name_check
                CHECK (relation_name IN ({_RELATION_CHECK})),
            CONSTRAINT corpus_generation_relation_inventory_count_check CHECK (row_count >= 0),
            CONSTRAINT corpus_generation_relation_inventory_hash_check
                CHECK (relation_sha256 ~ '^[0-9a-f]{{64}}$')
        )
        """,
        """
        CREATE TABLE bddk_meta.corpus_generation_seals (
            seal_id pg_catalog.text PRIMARY KEY,
            generation_id pg_catalog.text NOT NULL UNIQUE,
            corpus_state_sha256 pg_catalog.text NOT NULL,
            retrieval_profile_sha256 pg_catalog.text NOT NULL,
            inventory_sha256 pg_catalog.text NOT NULL,
            relation_count pg_catalog.int4 NOT NULL,
            row_count pg_catalog.int8 NOT NULL,
            sealed_at pg_catalog.timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
            sealed_by_fingerprint_sha256 pg_catalog.text NOT NULL,
            CONSTRAINT corpus_generation_seals_id_check
                CHECK (seal_id ~ '^corpus_generation_seal_sha256_[0-9a-f]{64}$'),
            CONSTRAINT corpus_generation_seals_state_hash_check
                CHECK (corpus_state_sha256 ~ '^[0-9a-f]{64}$'),
            CONSTRAINT corpus_generation_seals_profile_hash_check
                CHECK (retrieval_profile_sha256 ~ '^[0-9a-f]{64}$'),
            CONSTRAINT corpus_generation_seals_inventory_hash_check
                CHECK (inventory_sha256 ~ '^[0-9a-f]{64}$'),
            CONSTRAINT corpus_generation_seals_relation_count_check CHECK (relation_count = 17),
            CONSTRAINT corpus_generation_seals_row_count_check CHECK (row_count >= 0),
            CONSTRAINT corpus_generation_seals_actor_hash_check
                CHECK (sealed_by_fingerprint_sha256 ~ '^[0-9a-f]{64}$'),
            CONSTRAINT corpus_generation_seals_generation_fk
                FOREIGN KEY (generation_id, corpus_state_sha256, retrieval_profile_sha256)
                REFERENCES bddk_meta.corpus_generations(
                    generation_id, corpus_state_sha256, retrieval_profile_sha256
                ),
            CONSTRAINT corpus_generation_seals_binding_uq
                UNIQUE (seal_id, generation_id, corpus_state_sha256, retrieval_profile_sha256)
        )
        """,
        """
        CREATE TABLE bddk_meta.corpus_retained_releases (
            release_id pg_catalog.text PRIMARY KEY,
            seal_id pg_catalog.text NOT NULL,
            generation_id pg_catalog.text NOT NULL,
            corpus_state_sha256 pg_catalog.text NOT NULL,
            retrieval_profile_sha256 pg_catalog.text NOT NULL,
            retained_at pg_catalog.timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
            retained_by_fingerprint_sha256 pg_catalog.text NOT NULL,
            CONSTRAINT corpus_retained_releases_actor_hash_check
                CHECK (retained_by_fingerprint_sha256 ~ '^[0-9a-f]{64}$'),
            CONSTRAINT corpus_retained_releases_release_fk
                FOREIGN KEY (release_id, corpus_state_sha256, retrieval_profile_sha256)
                REFERENCES bddk_meta.corpus_releases(
                    release_id, corpus_state_sha256, retrieval_profile_sha256
                ),
            CONSTRAINT corpus_retained_releases_seal_fk
                FOREIGN KEY (seal_id, generation_id, corpus_state_sha256, retrieval_profile_sha256)
                REFERENCES bddk_meta.corpus_generation_seals(
                    seal_id, generation_id, corpus_state_sha256, retrieval_profile_sha256
                )
        )
        """,
        """
        CREATE INDEX corpus_retained_releases_generation_idx
        ON bddk_meta.corpus_retained_releases (generation_id)
        """,
        """
        CREATE INDEX corpus_retained_releases_seal_idx
        ON bddk_meta.corpus_retained_releases (seal_id)
        """,
        *(
            _retained_table_statement(position, relation)
            for position, relation in enumerate(RETAINED_CORPUS_RELATIONS, start=1)
        ),
        *(
            _retained_unique_statement(position, relation, columns)
            for position, (relation, columns) in enumerate(_UNIQUE_KEYS, start=1)
        ),
        *(
            _retained_foreign_key_statement(position, relation, columns, target_relation, target_columns)
            for position, (relation, columns, target_relation, target_columns) in enumerate(
                _FOREIGN_KEYS,
                start=1,
            )
        ),
        _v5_state_fingerprint_statement(),
        """
        CREATE FUNCTION bddk_meta.retained_row_sha256(
            member anyelement,
            exclude_generation_id pg_catalog.bool
        )
        RETURNS pg_catalog.text
        LANGUAGE sql
        IMMUTABLE
        PARALLEL SAFE
        SET search_path = pg_catalog
        SET TimeZone = 'UTC'
        SET DateStyle = 'ISO, YMD'
        SET IntervalStyle = 'postgres'
        SET bytea_output = 'hex'
        SET extra_float_digits = 3
        AS $function$
        SELECT pg_catalog.encode(
                   pg_catalog.sha256(
                       pg_catalog.convert_to(
                           CASE WHEN exclude_generation_id
                                THEN (
                                    pg_catalog.to_jsonb(member) - 'generation_id'
                                )::pg_catalog.text
                                ELSE pg_catalog.to_jsonb(member)::pg_catalog.text
                           END,
                           'UTF8'
                       )
                   ),
                   'hex'
               )
        $function$
        """,
        """
        CREATE FUNCTION bddk_meta.guard_retained_generation_member()
        RETURNS trigger
        LANGUAGE plpgsql
        VOLATILE
        PARALLEL UNSAFE
        SECURITY DEFINER
        SET search_path = pg_catalog
        AS $function$
        DECLARE
            selected_generation_id pg_catalog.text;
            previous_generation_id pg_catalog.text;
            expected_generation_count pg_catalog.int4 := 1;
            locked_generation_count pg_catalog.int4;
        BEGIN
            selected_generation_id := CASE WHEN TG_OP = 'DELETE'
                                           THEN OLD.generation_id
                                           ELSE NEW.generation_id END;
            previous_generation_id := CASE WHEN TG_OP = 'UPDATE'
                                           THEN OLD.generation_id
                                           ELSE NULL END;
            IF previous_generation_id IS DISTINCT FROM selected_generation_id
               AND previous_generation_id IS NOT NULL THEN
                expected_generation_count := 2;
            END IF;

            -- Serialize the member check with generation creation/sealing.
            -- Without this row lock, an INSERT that starts while the parent
            -- generation is uncommitted can pass the trigger, wait in its FK,
            -- and append after the seal commits.
            PERFORM 1
            FROM bddk_meta.corpus_generations AS generation
            WHERE generation.generation_id = selected_generation_id
               OR generation.generation_id = previous_generation_id
            ORDER BY generation.generation_id
            FOR UPDATE;
            GET DIAGNOSTICS locked_generation_count = ROW_COUNT;
            IF locked_generation_count <> expected_generation_count THEN
                RAISE EXCEPTION 'retained corpus generation parent is unavailable'
                    USING ERRCODE = '55000';
            END IF;

            IF EXISTS (
                SELECT 1
                FROM bddk_meta.corpus_generation_seals AS seal
                WHERE seal.generation_id = selected_generation_id
            ) OR (
                previous_generation_id IS DISTINCT FROM selected_generation_id
                AND previous_generation_id IS NOT NULL
                AND EXISTS (
                    SELECT 1
                    FROM bddk_meta.corpus_generation_seals AS seal
                    WHERE seal.generation_id = previous_generation_id
                )
            ) THEN
                RAISE EXCEPTION 'sealed corpus generation evidence is append-only'
                    USING ERRCODE = '55000';
            END IF;
            RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
        END
        $function$
        """,
        """
        CREATE FUNCTION bddk_meta.reject_retained_generation_mutation()
        RETURNS trigger
        LANGUAGE plpgsql
        VOLATILE
        PARALLEL UNSAFE
        SECURITY DEFINER
        SET search_path = pg_catalog
        AS $function$
        BEGIN
            RAISE EXCEPTION 'retained corpus generation evidence is append-only'
                USING ERRCODE = '55000';
        END
        $function$
        """,
        *(
            f"""
            CREATE TRIGGER guard_retained_generation_member
            BEFORE INSERT OR UPDATE OR DELETE ON bddk_retained.{relation}
            FOR EACH ROW EXECUTE FUNCTION bddk_meta.guard_retained_generation_member()
            """
            for relation in RETAINED_CORPUS_RELATIONS
        ),
        *(
            f"""
            CREATE TRIGGER reject_retained_generation_truncate
            BEFORE TRUNCATE ON bddk_retained.{relation}
            FOR EACH STATEMENT EXECUTE FUNCTION bddk_meta.reject_retained_generation_mutation()
            """
            for relation in RETAINED_CORPUS_RELATIONS
        ),
        """
        CREATE TRIGGER guard_retained_generation_inventory
        BEFORE INSERT OR UPDATE OR DELETE ON bddk_meta.corpus_generation_relation_inventory
        FOR EACH ROW EXECUTE FUNCTION bddk_meta.guard_retained_generation_member()
        """,
        """
        CREATE TRIGGER reject_retained_generation_inventory_truncate
        BEFORE TRUNCATE ON bddk_meta.corpus_generation_relation_inventory
        FOR EACH STATEMENT EXECUTE FUNCTION bddk_meta.reject_retained_generation_mutation()
        """,
        *(
            f"""
            CREATE TRIGGER reject_{relation}_update_delete
            BEFORE UPDATE OR DELETE ON bddk_meta.{relation}
            FOR EACH ROW EXECUTE FUNCTION bddk_meta.reject_retained_generation_mutation()
            """
            for relation in ("corpus_generations", "corpus_generation_seals", "corpus_retained_releases")
        ),
        *(
            f"""
            CREATE TRIGGER reject_{relation}_truncate
            BEFORE TRUNCATE ON bddk_meta.{relation}
            FOR EACH STATEMENT EXECUTE FUNCTION bddk_meta.reject_retained_generation_mutation()
            """
            for relation in ("corpus_generations", "corpus_generation_seals", "corpus_retained_releases")
        ),
        f"""
        CREATE FUNCTION bddk_meta.retain_active_corpus_generation(
            expected_release_id pg_catalog.text
        )
        RETURNS TABLE (
            generation_id pg_catalog.text,
            seal_id pg_catalog.text,
            release_id pg_catalog.text,
            source_activation_sequence pg_catalog.int8,
            corpus_state_sha256 pg_catalog.text,
            retrieval_profile_sha256 pg_catalog.text,
            inventory_sha256 pg_catalog.text,
            relation_count pg_catalog.int4,
            row_count pg_catalog.int8,
            retained_at pg_catalog.timestamptz
        )
        LANGUAGE plpgsql
        VOLATILE
        PARALLEL UNSAFE
        SECURITY DEFINER
        SET search_path = pg_catalog
        AS $function$
        DECLARE
            selected_release_id pg_catalog.text;
            selected_activation_sequence pg_catalog.int8;
            selected_state_sha256 pg_catalog.text;
            selected_profile_sha256 pg_catalog.text;
            selected_generation_id pg_catalog.text;
            selected_seal_id pg_catalog.text;
            selected_inventory_sha256 pg_catalog.text;
            selected_retained_state_sha256 pg_catalog.text;
            selected_actor_fingerprint pg_catalog.text;
            selected_relation pg_catalog.text;
            selected_relation_count pg_catalog.int8;
            selected_relation_sha256 pg_catalog.text;
            recorded_relation_count pg_catalog.int8;
            recorded_relation_sha256 pg_catalog.text;
            sealed_inventory_sha256 pg_catalog.text;
            sealed_relation_count pg_catalog.int4;
            sealed_row_count pg_catalog.int8;
            recomputed_seal_id pg_catalog.text;
            release_already_bound pg_catalog.bool := false;
            selected_total_rows pg_catalog.int8 := 0;
            inventory_frames pg_catalog.bytea := pg_catalog.decode('', 'hex');
        BEGIN
            IF pg_catalog.to_regrole('bddk_release_publisher') IS NULL
               OR NOT pg_catalog.pg_has_role(
                   SESSION_USER,
                   pg_catalog.to_regrole('bddk_release_publisher'),
                   'MEMBER'
               ) THEN
                RAISE EXCEPTION 'retained corpus generation caller is not authorized'
                    USING ERRCODE = '42501';
            END IF;
            IF expected_release_id IS NULL
               OR expected_release_id !~ '^corpus_release_sha256_[0-9a-f]{{64}}$' THEN
                RAISE EXCEPTION 'expected corpus release identity is invalid'
                    USING ERRCODE = '22023';
            END IF;

            PERFORM pg_catalog.pg_advisory_xact_lock(
                {SCHEMA_MIGRATION_ADVISORY_KEY}::pg_catalog.int8
            );
            PERFORM pg_catalog.pg_advisory_xact_lock(
                {CORPUS_MUTATION_ADVISORY_KEY}::pg_catalog.int8
            );
            LOCK TABLE bddk_meta.corpus_release_activations,
                       bddk_meta.corpus_generations,
                       bddk_meta.corpus_generation_seals,
                       bddk_meta.corpus_retained_releases
                IN SHARE ROW EXCLUSIVE MODE;
            LOCK TABLE {_SOURCE_TABLE_LOCKS}
                IN SHARE MODE;

            SELECT active.release_id,
                   active.activation_sequence,
                   active.corpus_state_sha256,
                   active.retrieval_profile_sha256
            INTO selected_release_id,
                 selected_activation_sequence,
                 selected_state_sha256,
                 selected_profile_sha256
            FROM bddk_meta.active_corpus_release AS active;

            IF selected_release_id IS NULL OR selected_release_id <> expected_release_id THEN
                RAISE EXCEPTION 'expected corpus release is not the active release'
                    USING ERRCODE = '55000';
            END IF;

            IF NOT bddk_meta.corpus_retrieval_ready(selected_profile_sha256)
               OR bddk_meta.current_corpus_state_sha256(selected_profile_sha256)
                    <> selected_state_sha256 THEN
                RAISE EXCEPTION 'active corpus release changed during retention validation'
                    USING ERRCODE = '55000';
            END IF;

            selected_generation_id := 'corpus_generation_sha256_' || pg_catalog.encode(
                pg_catalog.sha256(
                    bddk_meta.corpus_fingerprint_frame('1')
                    || bddk_meta.corpus_fingerprint_frame(selected_state_sha256)
                    || bddk_meta.corpus_fingerprint_frame(selected_profile_sha256)
                ),
                'hex'
            );
            SELECT EXISTS (
                SELECT 1
                FROM bddk_meta.corpus_retained_releases AS binding
                WHERE binding.release_id = selected_release_id
            ) INTO release_already_bound;
            selected_actor_fingerprint := pg_catalog.encode(
                pg_catalog.sha256(pg_catalog.convert_to(SESSION_USER::pg_catalog.text, 'UTF8')),
                'hex'
            );

            -- Physical identity is content/profile-derived. A differently
            -- signed or governed release over the same exact state binds to
            -- the existing seal without copying a second generation.
            IF EXISTS (
                SELECT 1
                FROM bddk_meta.corpus_generations AS generation
                WHERE generation.generation_id = selected_generation_id
            ) THEN
                SELECT seal.seal_id,
                       seal.inventory_sha256,
                       seal.relation_count,
                       seal.row_count
                INTO selected_seal_id,
                     sealed_inventory_sha256,
                     sealed_relation_count,
                     sealed_row_count
                FROM bddk_meta.corpus_generations AS generation
                JOIN bddk_meta.corpus_generation_seals AS seal
                  ON seal.generation_id = generation.generation_id
                WHERE generation.generation_id = selected_generation_id
                  AND generation.corpus_state_sha256 = selected_state_sha256
                  AND generation.retrieval_profile_sha256 = selected_profile_sha256
                  AND seal.corpus_state_sha256 = selected_state_sha256
                  AND seal.retrieval_profile_sha256 = selected_profile_sha256
                FOR UPDATE OF generation, seal;
                IF NOT FOUND THEN
                    RAISE EXCEPTION 'retained corpus generation retry state is incomplete'
                        USING ERRCODE = '55000';
                END IF;

                -- A content-derived identity is reusable only after freshly
                -- reproducing every retained member hash and the v5 state
                -- hash.  This prevents a new governed release from binding
                -- to owner- or restore-tampered retained bytes merely because
                -- a syntactically valid seal row still exists.
                selected_total_rows := 0;
                inventory_frames := pg_catalog.decode('', 'hex');
                FOREACH selected_relation IN ARRAY {_RELATION_ARRAY}
                LOOP
                    EXECUTE pg_catalog.format(
                        'SELECT pg_catalog.count(*)::pg_catalog.int8, '
                        'pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(COALESCE('
                        'pg_catalog.string_agg(row_hash, '''' ORDER BY row_hash), ''''), ''UTF8'')), ''hex'') '
                        'FROM (SELECT bddk_meta.retained_row_sha256(member, true) AS row_hash '
                        'FROM bddk_retained.%I AS member '
                        'WHERE member.generation_id = $1) AS retained_rows',
                        selected_relation
                    ) INTO selected_relation_count, selected_relation_sha256
                      USING selected_generation_id;

                    SELECT inventory.row_count,
                           inventory.relation_sha256
                    INTO recorded_relation_count,
                         recorded_relation_sha256
                    FROM bddk_meta.corpus_generation_relation_inventory AS inventory
                    WHERE inventory.generation_id = selected_generation_id
                      AND inventory.relation_name = selected_relation;
                    IF NOT FOUND
                       OR recorded_relation_count IS DISTINCT FROM selected_relation_count
                       OR recorded_relation_sha256 IS DISTINCT FROM selected_relation_sha256 THEN
                        RAISE EXCEPTION 'retained corpus generation retry state is invalid'
                            USING ERRCODE = '55000';
                    END IF;

                    selected_total_rows := selected_total_rows + selected_relation_count;
                    inventory_frames := inventory_frames
                        || bddk_meta.corpus_fingerprint_frame(selected_relation)
                        || bddk_meta.corpus_fingerprint_frame(
                               selected_relation_count::pg_catalog.text
                           )
                        || bddk_meta.corpus_fingerprint_frame(selected_relation_sha256);
                END LOOP;

                selected_retained_state_sha256 := bddk_meta.retained_corpus_state_sha256(
                    selected_generation_id,
                    selected_profile_sha256
                );
                selected_inventory_sha256 := pg_catalog.encode(
                    pg_catalog.sha256(
                        bddk_meta.corpus_fingerprint_frame('1') || inventory_frames
                    ),
                    'hex'
                );
                recomputed_seal_id := 'corpus_generation_seal_sha256_' || pg_catalog.encode(
                    pg_catalog.sha256(
                        bddk_meta.corpus_fingerprint_frame('1')
                        || bddk_meta.corpus_fingerprint_frame(selected_generation_id)
                        || bddk_meta.corpus_fingerprint_frame(selected_state_sha256)
                        || bddk_meta.corpus_fingerprint_frame(selected_profile_sha256)
                        || bddk_meta.corpus_fingerprint_frame(selected_inventory_sha256)
                    ),
                    'hex'
                );
                IF selected_retained_state_sha256 IS DISTINCT FROM selected_state_sha256
                   OR sealed_inventory_sha256 IS DISTINCT FROM selected_inventory_sha256
                   OR sealed_relation_count IS DISTINCT FROM 17
                   OR sealed_row_count IS DISTINCT FROM selected_total_rows
                   OR selected_seal_id IS DISTINCT FROM recomputed_seal_id THEN
                    RAISE EXCEPTION 'retained corpus generation retry state is invalid'
                        USING ERRCODE = '55000';
                END IF;

                IF release_already_bound THEN
                    IF NOT EXISTS (
                        SELECT 1
                        FROM bddk_meta.corpus_retained_releases AS binding
                        WHERE binding.release_id = selected_release_id
                          AND binding.seal_id = selected_seal_id
                          AND binding.generation_id = selected_generation_id
                          AND binding.corpus_state_sha256 = selected_state_sha256
                          AND binding.retrieval_profile_sha256 = selected_profile_sha256
                    ) THEN
                        RAISE EXCEPTION 'retained corpus generation retry binding is invalid'
                            USING ERRCODE = '55000';
                    END IF;
                ELSE
                    INSERT INTO bddk_meta.corpus_retained_releases (
                        release_id,
                        seal_id,
                        generation_id,
                        corpus_state_sha256,
                        retrieval_profile_sha256,
                        retained_by_fingerprint_sha256
                    ) VALUES (
                        selected_release_id,
                        selected_seal_id,
                        selected_generation_id,
                        selected_state_sha256,
                        selected_profile_sha256,
                        selected_actor_fingerprint
                    );
                END IF;

                RETURN QUERY
                SELECT generation.generation_id,
                       seal.seal_id,
                       binding.release_id,
                       generation.source_activation_sequence,
                       seal.corpus_state_sha256,
                       seal.retrieval_profile_sha256,
                       seal.inventory_sha256,
                       seal.relation_count,
                       seal.row_count,
                       binding.retained_at
                FROM bddk_meta.corpus_retained_releases AS binding
                JOIN bddk_meta.corpus_generation_seals AS seal
                  ON seal.seal_id = binding.seal_id
                 AND seal.generation_id = binding.generation_id
                 AND seal.corpus_state_sha256 = binding.corpus_state_sha256
                 AND seal.retrieval_profile_sha256 = binding.retrieval_profile_sha256
                JOIN bddk_meta.corpus_generations AS generation
                  ON generation.generation_id = binding.generation_id
                WHERE binding.release_id = selected_release_id;
                RETURN;
            END IF;

            INSERT INTO bddk_meta.corpus_generations (
                generation_id,
                generation_schema_version,
                source_activation_sequence,
                source_release_id,
                corpus_state_sha256,
                retrieval_profile_sha256,
                staged_by_fingerprint_sha256
            ) VALUES (
                selected_generation_id,
                1,
                selected_activation_sequence,
                selected_release_id,
                selected_state_sha256,
                selected_profile_sha256,
                selected_actor_fingerprint
            );

            FOREACH selected_relation IN ARRAY {_RELATION_ARRAY}
            LOOP
                EXECUTE pg_catalog.format(
                    'INSERT INTO bddk_retained.%I SELECT $1, source.* FROM public.%I AS source',
                    selected_relation,
                    selected_relation
                ) USING selected_generation_id;
                GET DIAGNOSTICS selected_relation_count = ROW_COUNT;

                EXECUTE pg_catalog.format(
                    'SELECT pg_catalog.encode('
                    'pg_catalog.sha256(pg_catalog.convert_to(COALESCE('
                    'pg_catalog.string_agg(row_hash, '''' ORDER BY row_hash), ''''), ''UTF8'')), ''hex'') '
                    'FROM (SELECT bddk_meta.retained_row_sha256(member, true) AS row_hash '
                    'FROM bddk_retained.%I AS member '
                    'WHERE member.generation_id = $1) AS retained_rows',
                    selected_relation
                ) INTO selected_relation_sha256 USING selected_generation_id;

                INSERT INTO bddk_meta.corpus_generation_relation_inventory (
                    generation_id,
                    relation_name,
                    row_count,
                    relation_sha256
                ) VALUES (
                    selected_generation_id,
                    selected_relation,
                    selected_relation_count,
                    selected_relation_sha256
                );
                selected_total_rows := selected_total_rows + selected_relation_count;
                inventory_frames := inventory_frames
                    || bddk_meta.corpus_fingerprint_frame(selected_relation)
                    || bddk_meta.corpus_fingerprint_frame(
                           selected_relation_count::pg_catalog.text
                       )
                    || bddk_meta.corpus_fingerprint_frame(selected_relation_sha256);
            END LOOP;

            selected_retained_state_sha256 := bddk_meta.retained_corpus_state_sha256(
                selected_generation_id,
                selected_profile_sha256
            );
            IF selected_retained_state_sha256 <> selected_state_sha256 THEN
                RAISE EXCEPTION 'retained corpus generation differs from the active release'
                    USING ERRCODE = '55000';
            END IF;

            selected_inventory_sha256 := pg_catalog.encode(
                pg_catalog.sha256(
                    bddk_meta.corpus_fingerprint_frame('1') || inventory_frames
                ),
                'hex'
            );
            selected_seal_id := 'corpus_generation_seal_sha256_' || pg_catalog.encode(
                pg_catalog.sha256(
                    bddk_meta.corpus_fingerprint_frame('1')
                    || bddk_meta.corpus_fingerprint_frame(selected_generation_id)
                    || bddk_meta.corpus_fingerprint_frame(selected_state_sha256)
                    || bddk_meta.corpus_fingerprint_frame(selected_profile_sha256)
                    || bddk_meta.corpus_fingerprint_frame(selected_inventory_sha256)
                ),
                'hex'
            );

            INSERT INTO bddk_meta.corpus_generation_seals (
                seal_id,
                generation_id,
                corpus_state_sha256,
                retrieval_profile_sha256,
                inventory_sha256,
                relation_count,
                row_count,
                sealed_by_fingerprint_sha256
            ) VALUES (
                selected_seal_id,
                selected_generation_id,
                selected_state_sha256,
                selected_profile_sha256,
                selected_inventory_sha256,
                17,
                selected_total_rows,
                selected_actor_fingerprint
            );
            IF release_already_bound THEN
                RAISE EXCEPTION 'retained corpus generation retry binding is invalid'
                    USING ERRCODE = '55000';
            END IF;
            INSERT INTO bddk_meta.corpus_retained_releases (
                release_id,
                seal_id,
                generation_id,
                corpus_state_sha256,
                retrieval_profile_sha256,
                retained_by_fingerprint_sha256
            ) VALUES (
                selected_release_id,
                selected_seal_id,
                selected_generation_id,
                selected_state_sha256,
                selected_profile_sha256,
                selected_actor_fingerprint
            );

            RETURN QUERY
            SELECT selected_generation_id,
                   selected_seal_id,
                   selected_release_id,
                   selected_activation_sequence,
                   selected_state_sha256,
                   selected_profile_sha256,
                   selected_inventory_sha256,
                   17::pg_catalog.int4,
                   selected_total_rows,
                   binding.retained_at
            FROM bddk_meta.corpus_retained_releases AS binding
            WHERE binding.release_id = selected_release_id;
        END
        $function$
        """,
        f"""
        CREATE FUNCTION bddk_meta.inspect_retained_generation_storage(
            requested_generation_id pg_catalog.text
        )
        RETURNS TABLE (
            generation_id pg_catalog.text,
            relation_count pg_catalog.int4,
            row_count pg_catalog.int8,
            generation_logical_bytes pg_catalog.int8,
            retained_store_heap_main_bytes pg_catalog.int8,
            retained_store_heap_auxiliary_bytes pg_catalog.int8,
            retained_store_toast_bytes pg_catalog.int8,
            retained_store_index_bytes pg_catalog.int8,
            retained_store_total_bytes pg_catalog.int8
        )
        LANGUAGE plpgsql
        STABLE
        PARALLEL UNSAFE
        SECURITY DEFINER
        SET search_path = pg_catalog
        AS $function$
        DECLARE
            selected_relation pg_catalog.text;
            selected_oid pg_catalog.oid;
            selected_toast_oid pg_catalog.oid;
            selected_main pg_catalog.int8;
            selected_table pg_catalog.int8;
            selected_indexes pg_catalog.int8;
            selected_total pg_catalog.int8;
            selected_toast pg_catalog.int8;
            selected_logical pg_catalog.int8;
            total_main pg_catalog.int8 := 0;
            total_auxiliary pg_catalog.int8 := 0;
            total_toast pg_catalog.int8 := 0;
            total_indexes pg_catalog.int8 := 0;
            total_store pg_catalog.int8 := 0;
            total_logical pg_catalog.int8 := 0;
            total_rows pg_catalog.int8;
        BEGIN
            IF pg_catalog.to_regrole('bddk_release_publisher') IS NULL
               OR NOT pg_catalog.pg_has_role(
                   SESSION_USER,
                   pg_catalog.to_regrole('bddk_release_publisher'),
                   'MEMBER'
               ) THEN
                RAISE EXCEPTION 'retained corpus generation caller is not authorized'
                    USING ERRCODE = '42501';
            END IF;
            IF NOT EXISTS (
                SELECT 1 FROM bddk_meta.corpus_generation_seals AS seal
                WHERE seal.generation_id = requested_generation_id
            ) THEN
                RAISE EXCEPTION 'sealed corpus generation is unavailable'
                    USING ERRCODE = '55000';
            END IF;
            SELECT seal.row_count INTO STRICT total_rows
            FROM bddk_meta.corpus_generation_seals AS seal
            WHERE seal.generation_id = requested_generation_id;

            FOREACH selected_relation IN ARRAY {_RELATION_ARRAY}
            LOOP
                selected_oid := pg_catalog.to_regclass(
                    'bddk_retained.' || selected_relation
                );
                SELECT relation.reltoastrelid
                INTO STRICT selected_toast_oid
                FROM pg_catalog.pg_class AS relation
                WHERE relation.oid = selected_oid;
                selected_main := pg_catalog.pg_relation_size(selected_oid, 'main');
                selected_table := pg_catalog.pg_table_size(selected_oid);
                selected_indexes := pg_catalog.pg_indexes_size(selected_oid);
                selected_total := pg_catalog.pg_total_relation_size(selected_oid);
                selected_toast := CASE WHEN selected_toast_oid = 0 THEN 0
                                       ELSE pg_catalog.pg_total_relation_size(selected_toast_oid)
                                  END;
                EXECUTE pg_catalog.format(
                    'SELECT COALESCE(pg_catalog.sum(pg_catalog.pg_column_size(member)), 0)::pg_catalog.int8 '
                    'FROM bddk_retained.%I AS member WHERE member.generation_id = $1',
                    selected_relation
                ) INTO selected_logical USING requested_generation_id;
                total_main := total_main + selected_main;
                total_toast := total_toast + selected_toast;
                total_auxiliary := total_auxiliary + selected_table - selected_main - selected_toast;
                total_indexes := total_indexes + selected_indexes;
                total_store := total_store + selected_total;
                total_logical := total_logical + selected_logical;
            END LOOP;
            IF total_store <> total_main + total_auxiliary + total_toast + total_indexes THEN
                RAISE EXCEPTION 'retained corpus storage evidence does not reconcile'
                    USING ERRCODE = '55000';
            END IF;
            RETURN QUERY SELECT requested_generation_id,
                                17::pg_catalog.int4,
                                total_rows,
                                total_logical,
                                total_main,
                                total_auxiliary,
                                total_toast,
                                total_indexes,
                                total_store;
        END
        $function$
        """,
        """
        CREATE VIEW bddk_meta.corpus_release_retention_status
        WITH (security_barrier = true, security_invoker = false)
        AS
        SELECT release.release_id,
               CASE WHEN binding.release_id IS NULL
                    THEN 'legacy_v5_unretained'::pg_catalog.text
                    ELSE 'retained'::pg_catalog.text
               END AS retention_status,
               binding.generation_id,
               binding.seal_id,
               release.corpus_state_sha256,
               release.retrieval_profile_sha256,
               binding.retained_at
        FROM bddk_meta.corpus_releases AS release
        LEFT JOIN bddk_meta.corpus_retained_releases AS binding
          ON binding.release_id = release.release_id
        """,
        "REVOKE ALL PRIVILEGES ON SCHEMA bddk_retained FROM PUBLIC",
        "REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA bddk_retained FROM PUBLIC",
        "REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA bddk_meta FROM PUBLIC",
        """
        REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.retained_corpus_state_sha256(
            pg_catalog.text, pg_catalog.text
        ) FROM PUBLIC
        """,
        """
        REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.retained_row_sha256(
            anyelement, pg_catalog.bool
        ) FROM PUBLIC
        """,
        "REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.guard_retained_generation_member() FROM PUBLIC",
        "REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.reject_retained_generation_mutation() FROM PUBLIC",
        """
        REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.retain_active_corpus_generation(
            pg_catalog.text
        ) FROM PUBLIC
        """,
        """
        REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.inspect_retained_generation_storage(
            pg_catalog.text
        ) FROM PUBLIC
        """,
    ),
)

__all__ = (
    "NONCANONICAL_FINGERPRINT_UPGRADE_SQLSTATE",
    "RETAINED_CORPUS_RELATIONS",
    "V0007_RETAINED_CORPUS_GENERATIONS",
)
