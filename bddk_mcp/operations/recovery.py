"""Fail-closed migration rehearsal and logical restore evidence.

These workflows intentionally sit outside serving and ingestion startup.  They
may mutate only an independently marked, name-constrained disposable target.
Source inspection and logical backup run in a read-only repeatable-read
snapshot.  Reports contain counts, byte sizes, hashes and bounded status labels
only; DSNs, credentials, document identifiers and corpus text are never part of
the report model.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import time
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Final
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

import asyncpg

from bddk_mcp.catalog_integrity import inspect_catalog_integrity
from bddk_mcp.db_identity import assert_database_connection_identity, assert_database_identity
from bddk_mcp.db_lifecycle import (
    assert_schema_owner_identity,
    inspect_database_readiness,
)
from bddk_mcp.db_transport import assert_database_transport
from bddk_mcp.migrations import (
    LATEST_SCHEMA_VERSION,
    MIGRATIONS,
    MigrationScaleError,
    inspect_migration_state,
    migrate,
)
from bddk_mcp.migrations.v0007_retained_corpus_generations import RETAINED_CORPUS_RELATIONS
from bddk_mcp.observability.telemetry import assert_telemetry_writer_ready

DISPOSABLE_ACKNOWLEDGEMENT: Final[str] = "I_UNDERSTAND_THIS_MUTATES_ONLY_A_DISPOSABLE_RECOVERY_TARGET"
_GUARD_SETTING: Final[str] = "bddk.recovery_drill_guard"
_DISPOSABLE_TARGET_RE = re.compile(r"^bddk_(?:v2_rehearsal|restore_drill)_[a-z0-9][a-z0-9_]{0,31}$")
_ADMIN_DATABASE_RE = re.compile(r"^bddk_recovery_admin(?:_[a-z0-9][a-z0-9_]{0,31})?$")
_ROLE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_PG_TOOL_TIMEOUT_ENV: Final[str] = "BDDK_RECOVERY_PG_TOOL_TIMEOUT_SECONDS"
_PG_TOOL_TIMEOUT_DEFAULT_SECONDS: Final[int] = 1800
_PG_TOOL_TIMEOUT_MIN_SECONDS: Final[int] = 30
_PG_TOOL_TIMEOUT_MAX_SECONDS: Final[int] = 21600
_PG_TOOL_TERMINATION_GRACE_SECONDS: Final[int] = 10
_ACTIVATION_SEQUENCE: Final[str] = "bddk_meta.corpus_release_activations_activation_sequence_seq"
_ACTIVATION_TABLE: Final[str] = "bddk_meta.corpus_release_activations"
_ACTIVATION_COLUMN: Final[str] = "activation_sequence"
_RETAINED_GENERATION_SCHEMA_MIGRATION_VERSION: Final[int] = 7
_RUNTIME_DATABASE_URL_VARIABLES: Final[tuple[str, ...]] = (
    "BDDK_DATABASE_URL",
    "BDDK_OPERATOR_DATABASE_URL",
    "BDDK_SCHEMA_OWNER_DATABASE_URL",
    "BDDK_INGESTION_DATABASE_URL",
    "BDDK_TELEMETRY_DATABASE_URL",
    "BDDK_RELEASE_PUBLISHER_DATABASE_URL",
)
_DATABASE_LOCALE_SQL: Final[str] = """
SELECT pg_catalog.pg_encoding_to_char(database_record.encoding) AS database_encoding,
       database_record.datcollate AS database_collation,
       database_record.datctype AS database_character_classification,
       database_record.datlocprovider::pg_catalog.text AS database_locale_provider,
       database_record.datlocale AS database_locale,
       database_record.daticurules AS database_icu_rules,
       database_record.datcollversion AS database_collation_version,
       pg_catalog.pg_database_collation_actual_version(database_record.oid)
           AS database_collation_actual_version
FROM pg_catalog.pg_database AS database_record
WHERE database_record.datname = current_database()
"""
_RETAINED_MEMBER_RELATIONS: Final[tuple[str, ...]] = tuple(
    f"bddk_retained.{relation}" for relation in RETAINED_CORPUS_RELATIONS
)


def _retained_member_fingerprint_query(relation: str) -> str:
    """Return a fixed-identifier query that emits only one digest per retained row."""

    if relation not in RETAINED_CORPUS_RELATIONS:
        raise ValueError("retained relation is not in the immutable migration inventory")
    return f"""
        SELECT bddk_meta.retained_row_sha256(member, false) AS row_sha256
        FROM bddk_retained.{relation} AS member
        ORDER BY row_sha256
        """


_RETAINED_MEMBER_FINGERPRINT_QUERIES: Final[tuple[tuple[str, str], ...]] = tuple(
    (f"retained_{relation}", _retained_member_fingerprint_query(relation)) for relation in RETAINED_CORPUS_RELATIONS
)
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
    WHERE fresh_relation_count <> {len(RETAINED_CORPUS_RELATIONS)}
       OR (
            SELECT pg_catalog.count(*)
            FROM bddk_meta.corpus_generation_relation_inventory AS inventory
            WHERE inventory.generation_id = validation.generation_id
          ) <> {len(RETAINED_CORPUS_RELATIONS)}
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
       OR sealed_relation_count IS DISTINCT FROM {len(RETAINED_CORPUS_RELATIONS)}
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

_MANAGED_RELATIONS: Final[tuple[str, ...]] = (
    "bddk_meta.legacy_schema_adoptions",
    "bddk_meta.schema_migrations",
    "bddk_meta.corpus_state_epoch",
    "bddk_meta.corpus_releases",
    "bddk_meta.corpus_release_activations",
    "bddk_meta.corpus_generations",
    "bddk_meta.active_corpus_release",
    _ACTIVATION_SEQUENCE,
    "bddk_operator.operator_jobs",
    "public.decision_cache",
    "public.documents",
    "public.document_sections",
    "public.document_versions",
    "public.document_chunks",
    "public.document_retrieval_publications",
    "public.sync_metadata",
    "public.sync_failures",
    "public.tool_call_traces",
    # Parent relation groups precede dependants. Self-referential predecessor
    # rows still require the transaction-aware pg_restore path below.
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
    # Retained member relations follow their generation parent and preserve
    # the v5 source-table dependency order.  Seals and release bindings follow
    # the complete member inventory they attest.
    *_RETAINED_MEMBER_RELATIONS,
    "bddk_meta.corpus_generation_relation_inventory",
    "bddk_meta.corpus_generation_seals",
    "bddk_meta.corpus_retained_releases",
    "bddk_meta.corpus_release_retention_status",
)

_SAFE_FINGERPRINT_QUERIES: Final[tuple[tuple[str, str], ...]] = (
    (
        "migrations",
        """
        SELECT version, name, checksum
        FROM bddk_meta.schema_migrations
        ORDER BY version
        """,
    ),
    (
        "corpus_state_epoch",
        """
        SELECT singleton_id, epoch
        FROM bddk_meta.corpus_state_epoch
        ORDER BY singleton_id
        """,
    ),
    (
        "corpus_releases",
        """
        SELECT release_id, manifest_id, manifest_sha256, signer_key_sha256,
               freshness_policy_result, source_detection_slo_seconds,
               publication_slo_seconds, max_manifest_age_seconds,
               retrieval_profile_sha256, corpus_state_sha256, created_at
        FROM bddk_meta.corpus_releases
        ORDER BY release_id
        """,
    ),
    (
        "corpus_release_activations",
        """
        SELECT activation_sequence, release_id, completed_at,
               actor_fingerprint_sha256, corpus_epoch
        FROM bddk_meta.corpus_release_activations
        ORDER BY activation_sequence
        """,
    ),
    (
        "active_corpus_release",
        """
        SELECT release_id, manifest_id, manifest_sha256, signer_key_sha256,
               freshness_policy_result, source_detection_slo_seconds,
               publication_slo_seconds, max_manifest_age_seconds,
               retrieval_profile_sha256, corpus_state_sha256, created_at,
               activation_sequence, completed_at, actor_fingerprint_sha256,
               corpus_epoch
        FROM bddk_meta.active_corpus_release
        ORDER BY activation_sequence
        """,
    ),
    (
        "corpus_release_activation_sequence",
        """
        SELECT state.last_value, state.is_called,
               sequence.seqstart, sequence.seqincrement, sequence.seqmax,
               sequence.seqmin, sequence.seqcache, sequence.seqcycle,
               attribute.attidentity, dependency.deptype,
               COALESCE(activation.maximum_activation_sequence, 0)
                   AS maximum_activation_sequence
        FROM pg_catalog.pg_class AS sequence_relation
        JOIN pg_catalog.pg_namespace AS sequence_namespace
          ON sequence_namespace.oid = sequence_relation.relnamespace
        JOIN pg_catalog.pg_sequence AS sequence
          ON sequence.seqrelid = sequence_relation.oid
        JOIN pg_catalog.pg_depend AS dependency
          ON dependency.classid = 'pg_catalog.pg_class'::pg_catalog.regclass
         AND dependency.objid = sequence_relation.oid
         AND dependency.objsubid = 0
         AND dependency.refclassid = 'pg_catalog.pg_class'::pg_catalog.regclass
         AND dependency.deptype = 'i'
        JOIN pg_catalog.pg_class AS owner_relation
          ON owner_relation.oid = dependency.refobjid
        JOIN pg_catalog.pg_namespace AS owner_namespace
          ON owner_namespace.oid = owner_relation.relnamespace
        JOIN pg_catalog.pg_attribute AS attribute
          ON attribute.attrelid = owner_relation.oid
         AND attribute.attnum = dependency.refobjsubid
         AND NOT attribute.attisdropped
        CROSS JOIN bddk_meta.corpus_release_activations_activation_sequence_seq AS state
        CROSS JOIN LATERAL (
            SELECT MAX(activation_sequence) AS maximum_activation_sequence
            FROM bddk_meta.corpus_release_activations
        ) AS activation
        WHERE sequence_namespace.nspname = 'bddk_meta'
          AND sequence_relation.relname =
              'corpus_release_activations_activation_sequence_seq'
          AND sequence_relation.relkind = 'S'
          AND owner_namespace.nspname = 'bddk_meta'
          AND owner_relation.relname = 'corpus_release_activations'
          AND attribute.attname = 'activation_sequence'
        """,
    ),
    (
        "documents",
        """
        SELECT document_id, content_hash,
               pg_catalog.encode(
                   pg_catalog.sha256(
                       pg_catalog.convert_to(COALESCE(markdown_content, ''), 'UTF8')
                   ),
                   'hex'
               ) AS content_digest,
               pg_catalog.octet_length(COALESCE(markdown_content, '')) AS content_bytes,
               COALESCE(file_size, 0) AS source_bytes,
               pg_catalog.encode(
                   pg_catalog.sha256(COALESCE(pdf_blob, ''::pg_catalog.bytea)),
                   'hex'
               ) AS pdf_digest,
               title, category, decision_date, decision_number, source_url,
               downloaded_at, extracted_at, extraction_method, total_pages
        FROM public.documents
        ORDER BY document_id
        """,
    ),
    (
        "sections",
        """
        SELECT doc_id, section_type, section_ref, content_hash,
               pg_catalog.encode(
                   pg_catalog.sha256(
                       pg_catalog.convert_to(COALESCE(content, ''), 'UTF8')
                   ),
                   'hex'
               ) AS content_digest,
               pg_catalog.octet_length(COALESCE(content, '')) AS content_bytes,
               start_char, end_char, page_start, page_end
        FROM public.document_sections
        ORDER BY doc_id, id
        """,
    ),
    (
        "chunks",
        """
        SELECT doc_id, chunk_index, content_hash, section_content_hash,
               pg_catalog.encode(
                   pg_catalog.sha256(
                       pg_catalog.convert_to(COALESCE(chunk_text, ''), 'UTF8')
                   ),
                   'hex'
               ) AS content_digest,
               pg_catalog.octet_length(chunk_text) AS content_bytes,
               CASE
                   WHEN embedding IS NULL THEN NULL
                   ELSE pg_catalog.encode(
                       pg_catalog.sha256(public.vector_send(embedding)),
                       'hex'
                   )
               END AS embedding_digest,
               total_chunks, chunk_start_char, chunk_end_char
        FROM public.document_chunks
        ORDER BY doc_id, chunk_index
        """,
    ),
    (
        "publications",
        """
        SELECT doc_id, content_hash, retrieval_profile_hash, expected_chunks
        FROM public.document_retrieval_publications
        ORDER BY doc_id
        """,
    ),
    (
        "decision_cache",
        """
        SELECT document_id, title,
               pg_catalog.encode(
                   pg_catalog.sha256(pg_catalog.convert_to(COALESCE(content, ''), 'UTF8')),
                   'hex'
               ) AS content_digest,
               pg_catalog.octet_length(COALESCE(content, '')) AS content_bytes,
               decision_date, decision_number, category, source_url, cached_at
        FROM public.decision_cache
        ORDER BY document_id
        """,
    ),
    (
        "document_versions",
        """
        SELECT id, document_id, version, content_hash,
               pg_catalog.encode(
                   pg_catalog.sha256(
                       pg_catalog.convert_to(COALESCE(markdown_content, ''), 'UTF8')
                   ),
                   'hex'
               ) AS content_digest,
               pg_catalog.octet_length(COALESCE(markdown_content, '')) AS content_bytes,
               synced_at
        FROM public.document_versions
        ORDER BY id
        """,
    ),
    (
        "sync_metadata",
        """
        SELECT document_id, etag, last_modified, last_sync_at, sync_count
        FROM public.sync_metadata
        ORDER BY document_id
        """,
    ),
    (
        "sync_failures",
        """
        SELECT document_id,
               pg_catalog.encode(
                   pg_catalog.sha256(pg_catalog.convert_to(error, 'UTF8')),
                   'hex'
               ) AS error_digest,
               error_category, source_url, retryable, attempts,
               first_failed_at, last_failed_at
        FROM public.sync_failures
        ORDER BY document_id
        """,
    ),
    (
        "tool_call_traces",
        """
        SELECT id, created_at, tool_name, args_hash,
               pg_catalog.encode(
                   pg_catalog.sha256(
                       pg_catalog.convert_to(COALESCE(args_summary::pg_catalog.text, ''), 'UTF8')
                   ),
                   'hex'
               ) AS args_summary_digest,
               latency_ms, result_count, doc_ids,
               pg_catalog.encode(
                   pg_catalog.sha256(
                       pg_catalog.convert_to(COALESCE(quality_labels::pg_catalog.text, ''), 'UTF8')
                   ),
                   'hex'
               ) AS quality_digest,
               pg_catalog.encode(
                   pg_catalog.sha256(
                       pg_catalog.convert_to(COALESCE(relevance_stats::pg_catalog.text, ''), 'UTF8')
                   ),
                   'hex'
               ) AS relevance_digest,
               model_id, session_id
        FROM public.tool_call_traces
        ORDER BY id
        """,
    ),
    (
        "operator_jobs",
        """
        SELECT job_id, kind, state, args_fingerprint, idempotency_digest,
               created_at, updated_at, revision, started_at, finished_at,
               progress_total, progress_completed, progress_succeeded, progress_failed,
               pg_catalog.encode(
                   pg_catalog.sha256(
                       pg_catalog.convert_to(result_metrics::pg_catalog.text, 'UTF8')
                   ),
                   'hex'
               ) AS result_metrics_digest,
               error_code
        FROM bddk_operator.operator_jobs
        ORDER BY job_id
        """,
    ),
    (
        "legacy_adoptions",
        """
        SELECT migration_version, source_kind, verifier_version, target_checksum,
               pre_normalization_fingerprint, post_normalization_fingerprint,
               normalizations, adopted_by, adopted_session_user, adopted_at
        FROM bddk_meta.legacy_schema_adoptions
        ORDER BY migration_version
        """,
    ),
    (
        "regulatory_instruments",
        """
        SELECT instrument_id, jurisdiction, authority_code, identity_key,
               canonical_title, instrument_type, created_at
        FROM public.regulatory_instruments
        ORDER BY instrument_id
        """,
    ),
    (
        "regulatory_family_imports",
        """
        SELECT bundle_id, bundle_sha256, instrument_id, schema_version,
               fixture_only, imported_by, imported_current_user,
               imported_session_user, predecessor_bundle_sha256,
               member_manifest, imported_at
        FROM public.regulatory_family_imports
        ORDER BY bundle_id, bundle_sha256
        """,
    ),
    (
        "regulatory_source_blobs",
        """
        SELECT blob_id, content_sha256
        FROM public.regulatory_source_blobs
        ORDER BY blob_id
        """,
    ),
    (
        "regulatory_source_artifacts",
        """
        SELECT artifact_id, blob_id, canonical_uri, source_authority,
               media_type, retrieved_at, repository_document_id, fixture_only
        FROM public.regulatory_source_artifacts
        ORDER BY artifact_id
        """,
    ),
    (
        "regulatory_evidence",
        """
        SELECT evidence_id, artifact_id, locator, statement_sha256, authority_level
        FROM public.regulatory_evidence
        ORDER BY evidence_id
        """,
    ),
    (
        "regulatory_legal_versions",
        """
        SELECT legal_version_id, instrument_id, version_key, legal_text_sha256,
               predecessor_version_id, consolidation_state, validation_state,
               validated_by, validated_at, validation_method,
               review_record_sha256, created_at
        FROM public.regulatory_legal_versions
        ORDER BY legal_version_id
        """,
    ),
    (
        "regulatory_legal_version_artifacts",
        """
        SELECT legal_version_id, artifact_id, source_role
        FROM public.regulatory_legal_version_artifacts
        ORDER BY legal_version_id, artifact_id, source_role
        """,
    ),
    (
        "regulatory_legal_events",
        """
        SELECT event_id, legal_version_id, event_type, event_date, evidence_id,
               target_legal_version_id, validation_state, validated_by,
               validated_at, validation_method, review_record_sha256
        FROM public.regulatory_legal_events
        ORDER BY event_id
        """,
    ),
    (
        "regulatory_legal_status_assertions",
        """
        SELECT assertion_id, legal_version_id, legal_status, valid_from,
               valid_through, evidence_id, validation_state, validated_by,
               validated_at, validation_method, review_record_sha256
        FROM public.regulatory_legal_status_assertions
        ORDER BY assertion_id
        """,
    ),
    (
        "regulatory_provisions",
        """
        SELECT provision_id, instrument_id, provision_kind, canonical_path
        FROM public.regulatory_provisions
        ORDER BY provision_id
        """,
    ),
    (
        "regulatory_legal_version_provisions",
        """
        SELECT legal_version_id, provision_id, provision_text_sha256,
               document_section_id, evidence_id, validation_state, validated_by,
               validated_at, validation_method, review_record_sha256
        FROM public.regulatory_legal_version_provisions
        ORDER BY legal_version_id, provision_id
        """,
    ),
    (
        "regulatory_validated_section_citations",
        """
        SELECT document_section_id, source_document_id,
               normalized_document_sha256, normalized_section_sha256,
               instrument_id, instrument_jurisdiction, instrument_authority_code,
               instrument_identity_key, legal_version_id, legal_version_key,
               legal_text_sha256, review_record_sha256,
               provision_review_record_sha256, artifact_id, artifact_sha256,
               source_url, artifact_retrieved_at, evidence_id, evidence_locator,
               evidence_statement_sha256, provision_id, provision_kind,
               provision_path, provision_text_sha256
        FROM public.regulatory_validated_section_citations
        ORDER BY document_section_id
        """,
    ),
    (
        "corpus_generations",
        """
        SELECT generation_id, generation_schema_version,
               source_activation_sequence, source_release_id,
               corpus_state_sha256, retrieval_profile_sha256,
               staged_at, staged_by_fingerprint_sha256
        FROM bddk_meta.corpus_generations
        ORDER BY generation_id
        """,
    ),
    *_RETAINED_MEMBER_FINGERPRINT_QUERIES,
    (
        "corpus_generation_relation_inventory",
        """
        SELECT generation_id, relation_name, row_count, relation_sha256
        FROM bddk_meta.corpus_generation_relation_inventory
        ORDER BY generation_id, relation_name
        """,
    ),
    (
        "corpus_generation_seals",
        """
        SELECT seal_id, generation_id, corpus_state_sha256,
               retrieval_profile_sha256, inventory_sha256, relation_count,
               row_count, sealed_at, sealed_by_fingerprint_sha256
        FROM bddk_meta.corpus_generation_seals
        ORDER BY seal_id
        """,
    ),
    (
        "corpus_retained_releases",
        """
        SELECT release_id, seal_id, generation_id, corpus_state_sha256,
               retrieval_profile_sha256, retained_at,
               retained_by_fingerprint_sha256
        FROM bddk_meta.corpus_retained_releases
        ORDER BY release_id
        """,
    ),
    (
        "corpus_release_retention_status",
        """
        SELECT release_id, retention_status, generation_id, seal_id,
               corpus_state_sha256, retrieval_profile_sha256, retained_at
        FROM bddk_meta.corpus_release_retention_status
        ORDER BY release_id
        """,
    ),
)

_DATABASE_GUARD_SQL: Final[str] = """
SELECT current_database()::pg_catalog.text AS database_name,
       current_setting('bddk.recovery_drill_guard', true) AS guard_hash
"""
_CLUSTER_GUARD_SQL: Final[str] = """
SELECT current_database()::pg_catalog.text AS database_name,
       current_setting('bddk.recovery_drill_guard', true) AS guard_hash,
       role.rolsuper,
       role.rolcreatedb,
       control.system_identifier::pg_catalog.text AS system_identifier
FROM pg_catalog.pg_roles AS role
CROSS JOIN pg_catalog.pg_control_system() AS control
WHERE role.rolname = session_user
"""
_SOURCE_IDENTITY_SQL: Final[str] = """
SELECT current_database()::pg_catalog.text AS database_name,
       control.system_identifier::pg_catalog.text AS system_identifier
FROM pg_catalog.pg_control_system() AS control
"""


class RecoveryDrillError(RuntimeError):
    """A bounded, credential-free recovery workflow failure."""

    def __init__(self, code: str):
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code):
            code = "recovery_drill_failed"
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class RelationEvidence:
    """Privacy-safe scale evidence for one fixed application relation."""

    rows: int
    heap_bytes: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class IdentitySequenceEvidence:
    """Collision-safety evidence for the release activation identity."""

    sequence_name: str
    owned_by: str
    identity_generation: str
    last_value: int
    is_called: bool
    start_value: int
    increment_by: int
    minimum_value: int
    maximum_value: int
    cache_size: int
    cycle: bool
    next_candidate: int
    maximum_retained_activation: int


@dataclass(frozen=True, slots=True)
class SnapshotEvidence:
    """Read-only integrity and scale evidence for one database snapshot."""

    migration_version: int
    migration_checksum: str
    database_encoding: str
    database_collation: str
    database_character_classification: str
    database_locale_provider: str
    database_locale: str | None
    database_icu_rules: str | None
    database_collation_version: str | None
    database_collation_actual_version: str | None
    logical_fingerprint_sha256: str
    database_bytes: int
    wal_lsn: str
    relations: Mapping[str, RelationEvidence]
    catalog_valid: bool
    catalog_failures: tuple[str, ...]
    readiness_ready: bool
    readiness_issues: tuple[str, ...]
    active_corpus_release_id: str | None
    activation_sequence: IdentitySequenceEvidence | None


@dataclass(frozen=True, slots=True)
class RecoveryEvidence:
    """A stable report that is safe to retain outside the corpus store."""

    schema_version: int
    workflow: str
    status: str
    target_fingerprint_sha256: str
    started_at_epoch: int
    elapsed_ms: int
    source: SnapshotEvidence
    restored: SnapshotEvidence
    dump_bytes: int = 0
    dump_sha256: str = ""
    backup_elapsed_ms: int = 0
    restore_elapsed_ms: int = 0
    migration_elapsed_ms: int = 0
    reindex_elapsed_ms: int = 0
    wal_generated_bytes: int = 0
    maximum_lock_waiters: int = 0
    lock_samples: int = 0
    default_refusal_proved: bool = False
    reindex_scanned: int = 0
    reindex_published: int = 0
    reindex_current: int = 0
    identities_verified: bool = False

    def to_json(self) -> str:
        """Return deterministic JSON containing only the fixed report schema."""

        return json.dumps(asdict(self), ensure_ascii=True, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class _CommandEvidence:
    elapsed_ms: int


@dataclass(frozen=True, slots=True)
class _PgEnvironment:
    environment: Mapping[str, str] = field(repr=False)
    database_name: str
    hostname: str
    port: int


class _PinnedPool:
    """Expose one connection through the small pool API used by inspectors."""

    def __init__(self, connection: asyncpg.Connection):
        self._connection = connection

    @asynccontextmanager
    async def acquire(self):
        yield self._connection

    async def fetch(self, query: str, *args):
        return await self._connection.fetch(query, *args)

    async def fetchrow(self, query: str, *args):
        return await self._connection.fetchrow(query, *args)

    async def fetchval(self, query: str, *args):
        return await self._connection.fetchval(query, *args)

    async def execute(self, query: str, *args):
        return await self._connection.execute(query, *args)


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def _catalog_char(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii", errors="strict")
    return str(value)


def require_disposable_acknowledgement(value: str) -> None:
    """Require one exact opt-in before opening a mutating target connection."""

    if not hmac.compare_digest(value, DISPOSABLE_ACKNOWLEDGEMENT):
        raise RecoveryDrillError("disposable_acknowledgement_required")


def validate_disposable_target_name(name: str) -> str:
    """Accept only unmistakably disposable target database names."""

    if not _DISPOSABLE_TARGET_RE.fullmatch(name):
        raise RecoveryDrillError("unsafe_disposable_target_name")
    return name


def validate_admin_database_name(name: str) -> str:
    """Bind cluster mutation to a dedicated recovery-admin database."""

    if not _ADMIN_DATABASE_RE.fullmatch(name):
        raise RecoveryDrillError("unsafe_recovery_admin_database")
    return name


def _guard_digest(token: str) -> str:
    if len(token) < 32:
        raise RecoveryDrillError("recovery_guard_token_too_short")
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _target_fingerprint(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _normalized_database_identity(dsn: str) -> tuple[str, int, str, str]:
    """Return a credential- and option-independent PostgreSQL endpoint identity."""

    try:
        parsed = urlsplit(dsn.strip())
        if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
            raise ValueError
        database_name = unquote(parsed.path.removeprefix("/"))
        username = unquote(parsed.username or "")
        if not database_name:
            raise ValueError
        return (
            parsed.hostname.casefold(),
            parsed.port or 5432,
            database_name,
            username,
        )
    except (TypeError, ValueError):
        raise RecoveryDrillError("unsupported_database_url") from None


def _assert_dsn_not_runtime(dsn: str) -> None:
    candidate = dsn.strip()
    if not candidate:
        raise RecoveryDrillError("database_url_required")
    candidate_identity = _normalized_database_identity(candidate)
    for variable in _RUNTIME_DATABASE_URL_VARIABLES:
        configured = os.environ.get(variable, "").strip()
        if configured and _normalized_database_identity(configured) == candidate_identity:
            raise RecoveryDrillError("recovery_admin_reuses_runtime_identity")


def _parse_pg_environment(dsn: str, *, database_name: str | None = None) -> _PgEnvironment:
    """Translate one PostgreSQL URL into libpq environment without argv secrets."""

    try:
        parsed = urlsplit(dsn)
        query_items = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=False)
        query: dict[str, str] = {}
        for key, value in query_items:
            if key in query:
                raise ValueError
            query[key] = value
        if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
            raise ValueError
        selected_database = database_name or unquote(parsed.path.removeprefix("/"))
        if not selected_database or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,62}", selected_database):
            raise ValueError
        allowed_query = {
            "sslmode": "PGSSLMODE",
            "sslrootcert": "PGSSLROOTCERT",
            "sslcert": "PGSSLCERT",
            "sslkey": "PGSSLKEY",
            "sslcrl": "PGSSLCRL",
            "sslcrldir": "PGSSLCRLDIR",
            "connect_timeout": "PGCONNECT_TIMEOUT",
            "application_name": "PGAPPNAME",
            "target_session_attrs": "PGTARGETSESSIONATTRS",
        }
        if set(query) - set(allowed_query):
            raise ValueError
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PGHOST": parsed.hostname,
            "PGPORT": str(parsed.port or 5432),
            "PGDATABASE": selected_database,
        }
        if parsed.username is not None:
            environment["PGUSER"] = unquote(parsed.username)
        if parsed.password is not None:
            environment["PGPASSWORD"] = unquote(parsed.password)
        for key, env_name in allowed_query.items():
            if key in query:
                environment[env_name] = query[key]
        return _PgEnvironment(
            environment=environment,
            database_name=selected_database,
            hostname=parsed.hostname,
            port=parsed.port or 5432,
        )
    except (TypeError, ValueError):
        raise RecoveryDrillError("unsupported_database_url") from None


def _database_dsn(
    dsn: str, database_name: str, *, username: str | None = None, password: str | None = None, role: str | None = None
) -> str:
    """Return a URL for the same server without exposing it in reports or argv."""

    try:
        parsed = urlsplit(dsn)
        if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
            raise ValueError
        user = username if username is not None else unquote(parsed.username or "")
        secret = password if password is not None else unquote(parsed.password or "")
        userinfo = quote(user, safe="")
        if secret:
            userinfo += ":" + quote(secret, safe="")
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = f"{userinfo}@{host}" if userinfo else host
        if parsed.port is not None:
            netloc += f":{parsed.port}"
        query = parse_qsl(parsed.query, keep_blank_values=True)
        query = [(key, value) for key, value in query if key != "options"]
        if role is not None:
            query.append(("options", f"-c role={role}"))
        return urlunsplit((parsed.scheme, netloc, "/" + quote(database_name, safe=""), urlencode(query), ""))
    except (TypeError, ValueError):
        raise RecoveryDrillError("unsupported_database_url") from None


async def assert_disposable_database_target(pool: Any, expected_database: str, guard_token: str) -> None:
    """Verify name and independently provisioned database guard using SELECT only."""

    validate_disposable_target_name(expected_database)
    expected_guard = _guard_digest(guard_token)
    try:
        row = await pool.fetchrow(_DATABASE_GUARD_SQL)
        valid = (
            row is not None
            and str(_row_value(row, "database_name", "")) == expected_database
            and hmac.compare_digest(str(_row_value(row, "guard_hash", "")), expected_guard)
        )
    except Exception:
        valid = False
    if not valid:
        raise RecoveryDrillError("disposable_database_guard_failed")


async def _assert_disposable_cluster(
    pool: Any,
    *,
    expected_admin_database: str,
    guard_token: str,
) -> str:
    validate_admin_database_name(expected_admin_database)
    expected_guard = _guard_digest(guard_token)
    try:
        row = await pool.fetchrow(_CLUSTER_GUARD_SQL)
        valid = (
            row is not None
            and str(_row_value(row, "database_name", "")) == expected_admin_database
            and hmac.compare_digest(str(_row_value(row, "guard_hash", "")), expected_guard)
            and bool(_row_value(row, "rolsuper", False))
            and bool(_row_value(row, "rolcreatedb", False))
            and bool(str(_row_value(row, "system_identifier", "")))
        )
    except Exception:
        valid = False
        row = None
    if not valid:
        raise RecoveryDrillError("disposable_cluster_guard_failed")
    return str(_row_value(row, "system_identifier"))


def _begin_relation_hash(hasher: Any, label: str) -> None:
    hasher.update(label.encode("ascii"))
    hasher.update(b"\x00")


def _hash_row(hasher: Any, row: Any) -> None:
    values = [row[key] for key in row.keys()]
    encoded = json.dumps(values, ensure_ascii=True, default=str, separators=(",", ":")).encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)


async def _stream_relation_hash(
    connection: asyncpg.Connection,
    hasher: Any,
    label: str,
    query: str,
    *,
    prefetch: int = 256,
) -> int:
    """Hash an ordered relation through a bounded asyncpg server cursor."""

    if not 1 <= prefetch <= 4_096:
        raise RecoveryDrillError("fingerprint_prefetch_invalid")
    _begin_relation_hash(hasher, label)
    rows = 0
    try:
        async for row in connection.cursor(query, prefetch=prefetch):
            _hash_row(hasher, row)
            rows += 1
    except Exception:
        raise RecoveryDrillError("snapshot_fingerprint_failed") from None
    return rows


async def _relation_evidence(pool: Any) -> dict[str, RelationEvidence]:
    result: dict[str, RelationEvidence] = {}
    for relation in _MANAGED_RELATIONS:
        exists = await pool.fetchval("SELECT pg_catalog.to_regclass($1)::pg_catalog.text", relation)
        if exists is None:
            continue
        # Relation names are drawn only from the immutable module constant.
        row_count = int(await pool.fetchval(f"SELECT COUNT(*) FROM {relation}"))
        if relation in {
            "bddk_meta.active_corpus_release",
            "bddk_meta.corpus_release_retention_status",
            "public.regulatory_validated_section_citations",
        }:
            heap_bytes = total_bytes = 0
        else:
            sizes = await pool.fetchrow(
                """
                SELECT pg_catalog.pg_relation_size($1::pg_catalog.regclass) AS heap_bytes,
                       pg_catalog.pg_total_relation_size($1::pg_catalog.regclass) AS total_bytes
                """,
                relation,
            )
            heap_bytes = int(_row_value(sizes, "heap_bytes", 0) or 0)
            total_bytes = int(_row_value(sizes, "total_bytes", 0) or 0)
        result[relation] = RelationEvidence(
            rows=row_count,
            heap_bytes=heap_bytes,
            total_bytes=total_bytes,
        )
    return result


async def _collect_activation_sequence_evidence(pool: Any) -> IdentitySequenceEvidence:
    """Prove the identity sequence cannot collide with a retained activation."""

    try:
        rows = await pool.fetch(dict(_SAFE_FINGERPRINT_QUERIES)["corpus_release_activation_sequence"])
    except Exception:
        raise RecoveryDrillError("identity_sequence_evidence_failed") from None
    if len(rows) != 1:
        raise RecoveryDrillError("identity_sequence_contract_invalid")
    row = rows[0]
    try:
        last_value = int(_row_value(row, "last_value"))
        is_called = bool(_row_value(row, "is_called"))
        start_value = int(_row_value(row, "seqstart"))
        increment_by = int(_row_value(row, "seqincrement"))
        minimum_value = int(_row_value(row, "seqmin"))
        maximum_value = int(_row_value(row, "seqmax"))
        cache_size = int(_row_value(row, "seqcache"))
        cycle = bool(_row_value(row, "seqcycle"))
        maximum_retained = int(_row_value(row, "maximum_activation_sequence", 0) or 0)
        identity_generation = _catalog_char(_row_value(row, "attidentity", ""))
        dependency_type = _catalog_char(_row_value(row, "deptype", ""))
    except (TypeError, UnicodeError, ValueError):
        raise RecoveryDrillError("identity_sequence_contract_invalid") from None
    next_candidate = last_value + increment_by if is_called else last_value
    if (
        identity_generation != "a"
        or dependency_type != "i"
        or increment_by <= 0
        or cache_size != 1
        or cycle
        or not minimum_value <= start_value <= maximum_value
        or not minimum_value <= last_value <= maximum_value
        or not minimum_value <= next_candidate <= maximum_value
    ):
        raise RecoveryDrillError("identity_sequence_contract_invalid")
    if next_candidate <= maximum_retained:
        raise RecoveryDrillError("identity_sequence_collision_risk")
    return IdentitySequenceEvidence(
        sequence_name=_ACTIVATION_SEQUENCE,
        owned_by=f"{_ACTIVATION_TABLE}.{_ACTIVATION_COLUMN}",
        identity_generation="always",
        last_value=last_value,
        is_called=is_called,
        start_value=start_value,
        increment_by=increment_by,
        minimum_value=minimum_value,
        maximum_value=maximum_value,
        cache_size=cache_size,
        cycle=cycle,
        next_candidate=next_candidate,
        maximum_retained_activation=maximum_retained,
    )


async def _assert_retained_generation_seals(pool: Any) -> None:
    """Fail closed unless every v7 retained generation reproduces its seal."""

    try:
        valid = await pool.fetchval(_RETAINED_GENERATION_SEAL_VALIDATION_SQL)
    except Exception:
        raise RecoveryDrillError("retained_generation_seal_invalid") from None
    if valid is not True:
        raise RecoveryDrillError("retained_generation_seal_invalid")


async def _collect_snapshot_evidence_pinned(
    pool: Any,
    connection: asyncpg.Connection,
    *,
    require_corpus: bool,
    require_active_release: bool,
) -> SnapshotEvidence:
    migration = await inspect_migration_state(pool)
    relations = await _relation_evidence(pool)
    if migration.current_version >= _RETAINED_GENERATION_SCHEMA_MIGRATION_VERSION:
        await _assert_retained_generation_seals(pool)
    hasher = hashlib.sha256()
    for label, query in _SAFE_FINGERPRINT_QUERIES:
        relation_by_label = {
            "corpus_state_epoch": "bddk_meta.corpus_state_epoch",
            "corpus_releases": "bddk_meta.corpus_releases",
            "corpus_release_activations": "bddk_meta.corpus_release_activations",
            "corpus_generations": "bddk_meta.corpus_generations",
            "active_corpus_release": "bddk_meta.active_corpus_release",
            "corpus_release_activation_sequence": _ACTIVATION_SEQUENCE,
            "publications": "public.document_retrieval_publications",
            "decision_cache": "public.decision_cache",
            "document_versions": "public.document_versions",
            "sync_metadata": "public.sync_metadata",
            "sync_failures": "public.sync_failures",
            "tool_call_traces": "public.tool_call_traces",
            "operator_jobs": "bddk_operator.operator_jobs",
            "legacy_adoptions": "bddk_meta.legacy_schema_adoptions",
            "regulatory_instruments": "public.regulatory_instruments",
            "regulatory_family_imports": "public.regulatory_family_imports",
            "regulatory_source_blobs": "public.regulatory_source_blobs",
            "regulatory_source_artifacts": "public.regulatory_source_artifacts",
            "regulatory_evidence": "public.regulatory_evidence",
            "regulatory_legal_versions": "public.regulatory_legal_versions",
            "regulatory_legal_version_artifacts": "public.regulatory_legal_version_artifacts",
            "regulatory_legal_events": "public.regulatory_legal_events",
            "regulatory_legal_status_assertions": "public.regulatory_legal_status_assertions",
            "regulatory_provisions": "public.regulatory_provisions",
            "regulatory_legal_version_provisions": "public.regulatory_legal_version_provisions",
            "regulatory_validated_section_citations": "public.regulatory_validated_section_citations",
            **{f"retained_{relation}": f"bddk_retained.{relation}" for relation in RETAINED_CORPUS_RELATIONS},
            "corpus_generation_relation_inventory": ("bddk_meta.corpus_generation_relation_inventory"),
            "corpus_generation_seals": "bddk_meta.corpus_generation_seals",
            "corpus_retained_releases": "bddk_meta.corpus_retained_releases",
            "corpus_release_retention_status": "bddk_meta.corpus_release_retention_status",
        }
        relation = relation_by_label.get(label)
        if relation is not None and relation not in relations:
            continue
        await _stream_relation_hash(connection, hasher, label, query)

    catalog_failures: tuple[str, ...] = ()
    if migration.current:
        catalog_failures = (await inspect_catalog_integrity(pool)).failures
    readiness = await inspect_database_readiness(
        pool,
        require_corpus=require_corpus,
        require_active_release=require_active_release,
    )
    readiness_issues = (
        readiness.missing_extensions
        + readiness.missing_relations
        + readiness.missing_columns
        + readiness.catalog_issues
        + readiness.corpus_issues
    )
    checksum = ""
    if migration.current_version:
        checksum = MIGRATIONS[migration.current_version - 1].checksum
    activation_sequence: IdentitySequenceEvidence | None = None
    if _ACTIVATION_SEQUENCE in relations:
        activation_sequence = await _collect_activation_sequence_evidence(pool)
    elif migration.current:
        raise RecoveryDrillError("identity_sequence_contract_invalid")
    active_release = readiness.active_corpus_release
    active_release_id = active_release.release_id if active_release is not None else None
    if require_active_release and active_release_id is None:
        raise RecoveryDrillError("active_corpus_release_required")
    database_locale = await pool.fetchrow(_DATABASE_LOCALE_SQL)
    database_encoding = str(_row_value(database_locale, "database_encoding", ""))
    database_collation = str(_row_value(database_locale, "database_collation", ""))
    database_character_classification = str(_row_value(database_locale, "database_character_classification", ""))
    database_locale_provider = str(_row_value(database_locale, "database_locale_provider", ""))
    optional_locale_values = {
        name: (None if _row_value(database_locale, name) is None else str(_row_value(database_locale, name)))
        for name in (
            "database_locale",
            "database_icu_rules",
            "database_collation_version",
            "database_collation_actual_version",
        )
    }
    if (
        not database_encoding
        or not database_collation
        or not database_character_classification
        or not database_locale_provider
        or optional_locale_values["database_collation_version"]
        != optional_locale_values["database_collation_actual_version"]
    ):
        raise RecoveryDrillError("database_locale_contract_invalid")
    return SnapshotEvidence(
        migration_version=migration.current_version,
        migration_checksum=checksum,
        database_encoding=database_encoding,
        database_collation=database_collation,
        database_character_classification=database_character_classification,
        database_locale_provider=database_locale_provider,
        database_locale=optional_locale_values["database_locale"],
        database_icu_rules=optional_locale_values["database_icu_rules"],
        database_collation_version=optional_locale_values["database_collation_version"],
        database_collation_actual_version=optional_locale_values["database_collation_actual_version"],
        logical_fingerprint_sha256=hasher.hexdigest(),
        database_bytes=int(await pool.fetchval("SELECT pg_catalog.pg_database_size(current_database())")),
        wal_lsn=str(await pool.fetchval("SELECT pg_catalog.pg_current_wal_lsn()::pg_catalog.text")),
        relations=relations,
        catalog_valid=not catalog_failures if migration.current else False,
        catalog_failures=catalog_failures,
        readiness_ready=readiness.ready,
        readiness_issues=tuple(readiness_issues),
        active_corpus_release_id=active_release_id,
        activation_sequence=activation_sequence,
    )


async def collect_snapshot_evidence(
    pool: Any,
    *,
    require_corpus: bool,
    require_active_release: bool = False,
) -> SnapshotEvidence:
    """Collect fixed-field integrity evidence from one bounded, repeatable snapshot."""

    if isinstance(pool, _PinnedPool):
        return await _collect_snapshot_evidence_pinned(
            pool,
            pool._connection,
            require_corpus=require_corpus,
            require_active_release=require_active_release,
        )
    async with pool.acquire() as connection:
        async with connection.transaction(isolation="repeatable_read", readonly=True):
            return await _collect_snapshot_evidence_pinned(
                _PinnedPool(connection),
                connection,
                require_corpus=require_corpus,
                require_active_release=require_active_release,
            )


async def _wal_difference(pool: Any, start: str, end: str) -> int:
    return int(
        await pool.fetchval(
            "SELECT pg_catalog.pg_wal_lsn_diff("
            "$1::pg_catalog.text::pg_catalog.pg_lsn, "
            "$2::pg_catalog.text::pg_catalog.pg_lsn"
            ")::pg_catalog.int8",
            end,
            start,
        )
        or 0
    )


async def _monitor_lock_waits(pool: Any, task: asyncio.Task[Any]) -> tuple[int, int]:
    maximum = samples = 0
    while True:
        try:
            waiting = int(
                await pool.fetchval(
                    """
                    SELECT COUNT(*)
                    FROM pg_catalog.pg_stat_activity
                    WHERE datname = current_database()
                      AND pid <> pg_catalog.pg_backend_pid()
                      AND wait_event_type = 'Lock'
                    """
                )
                or 0
            )
            maximum = max(maximum, waiting)
            samples += 1
        except Exception:
            raise RecoveryDrillError("lock_monitor_failed") from None
        if task.done():
            break
        await asyncio.sleep(0.05)
    return maximum, samples


async def run_populated_v2_rehearsal(
    schema_owner_pool: asyncpg.Pool,
    ingestion_pool: asyncpg.Pool,
    *,
    expected_target: str,
    guard_token: str,
    acknowledgement: str,
    reindexer: Callable[[asyncpg.Pool], Awaitable[Mapping[str, int]]] | None = None,
) -> RecoveryEvidence:
    """Rehearse the existing v2-to-v3 path on one marked disposable restore."""

    require_disposable_acknowledgement(acknowledgement)
    await assert_disposable_database_target(schema_owner_pool, expected_target, guard_token)
    await assert_disposable_database_target(ingestion_pool, expected_target, guard_token)
    await assert_schema_owner_identity(schema_owner_pool, expected_target)
    await assert_database_identity(ingestion_pool, "ingestion")

    started_wall = int(time.time())
    started = time.perf_counter()
    state = await inspect_migration_state(schema_owner_pool)
    if state.current_version != 2:
        raise RecoveryDrillError("rehearsal_requires_populated_v2")
    source = await collect_snapshot_evidence(schema_owner_pool, require_corpus=False)
    documents = source.relations.get("public.documents")
    sections = source.relations.get("public.document_sections")
    chunks = source.relations.get("public.document_chunks")
    if not documents or not sections or not chunks or min(documents.rows, sections.rows, chunks.rows) < 1:
        raise RecoveryDrillError("rehearsal_requires_populated_v2")

    try:
        await migrate(schema_owner_pool)
    except MigrationScaleError:
        default_refusal = True
    else:
        raise RecoveryDrillError("populated_v2_default_refusal_missing")
    if (await inspect_migration_state(schema_owner_pool)).current_version != 2:
        raise RecoveryDrillError("default_refusal_changed_schema")

    migration_task = asyncio.create_task(migrate(schema_owner_pool, allow_retrieval_publication_backfill=True))
    monitor_task = asyncio.create_task(_monitor_lock_waits(schema_owner_pool, migration_task))
    migration_started = time.perf_counter()
    try:
        migrated = await migration_task
        maximum_lock_waiters, lock_samples = await monitor_task
    except Exception:
        monitor_task.cancel()
        await asyncio.gather(monitor_task, return_exceptions=True)
        raise
    if not migrated.current:
        raise RecoveryDrillError("migration_did_not_reach_current")
    migration_elapsed_ms = round((time.perf_counter() - migration_started) * 1000)

    if reindexer is None:
        from bddk_mcp.ingest.seed import reindex_existing_documents

        reindexer = reindex_existing_documents
    try:
        reindex_started = time.perf_counter()
        reindex = dict(await reindexer(ingestion_pool))
        reindex_elapsed_ms = round((time.perf_counter() - reindex_started) * 1000)
    except Exception:
        raise RecoveryDrillError("reindex_failed") from None

    restored = await collect_snapshot_evidence(schema_owner_pool, require_corpus=True)
    if (
        restored.migration_version != LATEST_SCHEMA_VERSION
        or not restored.catalog_valid
        or not restored.readiness_ready
    ):
        raise RecoveryDrillError("post_migration_integrity_failed")
    wal_generated = await _wal_difference(
        schema_owner_pool,
        source.wal_lsn,
        restored.wal_lsn,
    )
    return RecoveryEvidence(
        schema_version=2,
        workflow="populated_v2_migration_rehearsal",
        status="passed",
        target_fingerprint_sha256=_target_fingerprint(expected_target),
        started_at_epoch=started_wall,
        elapsed_ms=round((time.perf_counter() - started) * 1000),
        source=source,
        restored=restored,
        wal_generated_bytes=max(0, wal_generated),
        maximum_lock_waiters=maximum_lock_waiters,
        lock_samples=lock_samples,
        default_refusal_proved=default_refusal,
        migration_elapsed_ms=migration_elapsed_ms,
        reindex_elapsed_ms=reindex_elapsed_ms,
        reindex_scanned=int(reindex.get("reindex_scanned", 0)),
        reindex_published=int(reindex.get("reindex_published", 0)),
        reindex_current=int(reindex.get("reindex_current", 0)),
        identities_verified=True,
    )


async def _run_pg_tool(
    executable: str,
    arguments: list[str],
    environment: Mapping[str, str],
    *,
    failure_code: str,
    timeout_seconds: float | None = None,
) -> _CommandEvidence:
    selected_timeout = _configured_pg_tool_timeout_seconds() if timeout_seconds is None else timeout_seconds
    if not (0 < selected_timeout <= _PG_TOOL_TIMEOUT_MAX_SECONDS):
        raise RecoveryDrillError("pg_tool_timeout_configuration_invalid")
    binary = shutil.which(executable)
    if binary is None:
        raise RecoveryDrillError(f"{executable}_unavailable")
    started = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        binary,
        *arguments,
        env=dict(environment),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        return_code = await asyncio.wait_for(process.wait(), timeout=selected_timeout)
    except TimeoutError:
        await _terminate_pg_tool(process)
        raise RecoveryDrillError("pg_tool_timed_out") from None
    except asyncio.CancelledError:
        await _terminate_pg_tool(process)
        raise
    if return_code != 0:
        raise RecoveryDrillError(failure_code)
    return _CommandEvidence(elapsed_ms=round((time.perf_counter() - started) * 1000))


def _configured_pg_tool_timeout_seconds() -> int:
    value = os.environ.get(_PG_TOOL_TIMEOUT_ENV, "").strip()
    if not value:
        return _PG_TOOL_TIMEOUT_DEFAULT_SECONDS
    if not value.isascii() or not value.isdecimal():
        raise RecoveryDrillError("pg_tool_timeout_configuration_invalid")
    timeout = int(value)
    if not (_PG_TOOL_TIMEOUT_MIN_SECONDS <= timeout <= _PG_TOOL_TIMEOUT_MAX_SECONDS):
        raise RecoveryDrillError("pg_tool_timeout_configuration_invalid")
    return timeout


async def _terminate_pg_tool(process: Any) -> None:
    """Reap a timed-out child without exposing its arguments or environment."""

    try:
        process.terminate()
    except Exception:
        pass
    try:
        await asyncio.wait_for(
            process.wait(),
            timeout=_PG_TOOL_TERMINATION_GRACE_SECONDS,
        )
        return
    except Exception:
        pass

    try:
        process.kill()
    except Exception:
        pass
    try:
        await asyncio.wait_for(
            process.wait(),
            timeout=_PG_TOOL_TERMINATION_GRACE_SECONDS,
        )
    except Exception:
        # A platform-level process-reaping failure must not turn the workflow
        # into an unbounded wait or disclose subprocess details.
        return


async def _execute_asset(connection: asyncpg.Connection, sql_path: Path, expected_database: str) -> None:
    try:
        sql = sql_path.read_text(encoding="utf-8")
        async with connection.transaction():
            await connection.fetchval(
                "SELECT pg_catalog.set_config('bddk.expected_database', $1, true)",
                expected_database,
            )
            await connection.execute(sql)
    except (OSError, UnicodeError, asyncpg.PostgresError):
        raise RecoveryDrillError("database_role_asset_failed") from None


async def _provision_verification_logins(
    connection: asyncpg.Connection,
) -> tuple[dict[str, tuple[str, str, str | None]], tuple[str, ...]]:
    suffix = secrets.token_hex(5)
    specs = {
        "schema_owner": (("bddk_schema_owner",), "bddk_schema_owner"),
        "public": (("bddk_public_reader",), None),
        "ingestion": (("bddk_ingestion",), None),
        "release_publisher": (("bddk_release_publisher",), None),
        "operator": (("bddk_public_reader", "bddk_ingestion", "bddk_operator_runtime"), None),
        "telemetry": (("bddk_telemetry_writer",), None),
    }
    result: dict[str, tuple[str, str, str | None]] = {}
    created: list[str] = []
    try:
        for profile, (memberships, role) in specs.items():
            name = f"bddk_recovery_{profile}_{suffix}"
            if not _ROLE_NAME_RE.fullmatch(name):
                raise RecoveryDrillError("verification_login_name_failed")
            password = secrets.token_urlsafe(32)
            quoted_password = await connection.fetchval("SELECT pg_catalog.quote_literal($1)", password)
            expires = datetime.now(UTC) + timedelta(minutes=15)
            quoted_expiry = await connection.fetchval(
                "SELECT pg_catalog.quote_literal($1)",
                expires.isoformat(),
            )
            await connection.execute(
                f"CREATE ROLE {name} LOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE "
                f"NOREPLICATION NOBYPASSRLS PASSWORD {quoted_password} VALID UNTIL "
                f"{quoted_expiry}"
            )
            created.append(name)
            await connection.execute(f"GRANT {', '.join(memberships)} TO {name}")
            result[profile] = (name, password, role)
    except Exception:
        for name in reversed(created):
            try:
                await connection.execute(f"DROP ROLE IF EXISTS {name}")
            except asyncpg.PostgresError:
                pass
        raise RecoveryDrillError("verification_login_provisioning_failed") from None
    return result, tuple(created)


async def _drop_verification_logins(connection: asyncpg.Connection, names: tuple[str, ...]) -> None:
    for name in reversed(names):
        if _ROLE_NAME_RE.fullmatch(name):
            await connection.execute(f"DROP ROLE IF EXISTS {name}")


async def _verify_restored_identities(
    admin_dsn: str,
    target_name: str,
    login_specs: Mapping[str, tuple[str, str, str | None]],
) -> SnapshotEvidence:
    pools: list[asyncpg.Pool] = []
    by_profile: dict[str, asyncpg.Pool] = {}
    try:
        for profile, (username, password, role) in login_specs.items():
            dsn = _database_dsn(admin_dsn, target_name, username=username, password=password, role=role)
            init = None
            identity_profile = "release-publisher" if profile == "release_publisher" else profile
            if identity_profile in {"public", "ingestion", "release-publisher", "operator"}:
                init = partial(assert_database_connection_identity, profile=identity_profile)
            pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2, timeout=10, init=init)
            pools.append(pool)
            by_profile[profile] = pool

        await assert_schema_owner_identity(by_profile["schema_owner"], target_name)
        for profile in ("public", "ingestion", "operator"):
            await assert_database_identity(by_profile[profile], profile)  # type: ignore[arg-type]
        await assert_database_identity(by_profile["release_publisher"], "release-publisher")
        await assert_telemetry_writer_ready(by_profile["telemetry"])
        evidence = await collect_snapshot_evidence(
            by_profile["schema_owner"],
            require_corpus=True,
            require_active_release=True,
        )
        if not evidence.readiness_ready or not evidence.catalog_valid:
            raise RecoveryDrillError("restored_database_not_ready")
        return evidence
    except RecoveryDrillError:
        raise
    except Exception:
        raise RecoveryDrillError("restored_identity_verification_failed") from None
    finally:
        for pool in reversed(pools):
            await pool.close()


def _same_logical_snapshot(source: SnapshotEvidence, restored: SnapshotEvidence) -> bool:
    if source.logical_fingerprint_sha256 != restored.logical_fingerprint_sha256:
        return False
    if (
        source.migration_version != restored.migration_version
        or source.migration_checksum != restored.migration_checksum
        or source.database_encoding != restored.database_encoding
        or source.database_collation != restored.database_collation
        or source.database_character_classification != restored.database_character_classification
        or source.database_locale_provider != restored.database_locale_provider
        or source.database_locale != restored.database_locale
        or source.database_icu_rules != restored.database_icu_rules
        or source.database_collation_version != restored.database_collation_version
        or source.database_collation_actual_version != restored.database_collation_actual_version
    ):
        return False
    if (
        source.active_corpus_release_id is None
        or restored.active_corpus_release_id is None
        or source.active_corpus_release_id != restored.active_corpus_release_id
        or source.activation_sequence is None
        or restored.activation_sequence is None
        or source.activation_sequence != restored.activation_sequence
    ):
        return False
    return {name: relation.rows for name, relation in source.relations.items()} == {
        name: relation.rows for name, relation in restored.relations.items()
    }


async def run_backup_restore_drill(
    *,
    source_dsn: str,
    admin_dsn: str,
    expected_source_database: str,
    expected_admin_database: str,
    target_database: str,
    guard_token: str,
    acknowledgement: str,
    repository_root: Path | None = None,
) -> RecoveryEvidence:
    """Dump one read-only source snapshot and restore it on an isolated cluster."""

    require_disposable_acknowledgement(acknowledgement)
    validate_admin_database_name(expected_admin_database)
    validate_disposable_target_name(target_database)
    if expected_source_database == target_database:
        raise RecoveryDrillError("source_and_target_must_differ")
    _assert_dsn_not_runtime(admin_dsn)
    source_dsn = assert_database_transport(source_dsn)
    admin_dsn = assert_database_transport(admin_dsn)
    source_env = _parse_pg_environment(source_dsn)
    admin_env = _parse_pg_environment(admin_dsn)
    if source_env.database_name != expected_source_database:
        raise RecoveryDrillError("source_database_url_mismatch")
    if admin_env.database_name != expected_admin_database:
        raise RecoveryDrillError("admin_database_url_mismatch")

    root = repository_root or Path(__file__).resolve().parents[2]
    roles_sql = root / "deploy/postgres/01_roles.sql"
    grants_sql = root / "deploy/postgres/02_grants.sql"
    if not roles_sql.is_file() or not grants_sql.is_file():
        raise RecoveryDrillError("database_role_assets_unavailable")

    started_wall = int(time.time())
    started = time.perf_counter()
    source_pool: asyncpg.Pool | None = None
    admin_pool: asyncpg.Pool | None = None
    target_admin_pool: asyncpg.Pool | None = None
    verification_logins: tuple[str, ...] = ()
    try:

        async def source_read_only(connection: asyncpg.Connection) -> None:
            await connection.execute("SET default_transaction_read_only = on")

        source_pool = await asyncpg.create_pool(
            source_dsn,
            min_size=1,
            max_size=2,
            timeout=10,
            init=source_read_only,
        )
        admin_pool = await asyncpg.create_pool(admin_dsn, min_size=1, max_size=2, timeout=10)
        target_system_identifier = await _assert_disposable_cluster(
            admin_pool,
            expected_admin_database=expected_admin_database,
            guard_token=guard_token,
        )
        source_identity = await source_pool.fetchrow(_SOURCE_IDENTITY_SQL)
        if (
            source_identity is None
            or str(_row_value(source_identity, "database_name", "")) != expected_source_database
            or not str(_row_value(source_identity, "system_identifier", ""))
        ):
            raise RecoveryDrillError("source_identity_failed")
        if hmac.compare_digest(
            str(_row_value(source_identity, "system_identifier")),
            target_system_identifier,
        ):
            raise RecoveryDrillError("restore_cluster_must_be_isolated")
        if await admin_pool.fetchval(
            "SELECT EXISTS(SELECT 1 FROM pg_catalog.pg_database WHERE datname = $1)",
            target_database,
        ):
            raise RecoveryDrillError("restore_target_already_exists")

        with TemporaryDirectory(prefix="bddk-recovery-") as temporary:
            dump_path = Path(temporary) / "snapshot.dump"
            dump_path.touch(mode=0o600)
            async with source_pool.acquire() as source_connection:
                transaction = source_connection.transaction(isolation="repeatable_read", readonly=True)
                await transaction.start()
                try:
                    snapshot_id = str(await source_connection.fetchval("SELECT pg_catalog.pg_export_snapshot()"))
                    source = await collect_snapshot_evidence(
                        _PinnedPool(source_connection),
                        require_corpus=True,
                        require_active_release=True,
                    )
                    if (
                        source.migration_version != LATEST_SCHEMA_VERSION
                        or not source.catalog_valid
                        or not source.readiness_ready
                    ):
                        raise RecoveryDrillError("source_database_not_ready")
                    backup_command = await _run_pg_tool(
                        "pg_dump",
                        [
                            "--format=custom",
                            "--no-owner",
                            "--no-privileges",
                            "--snapshot=" + snapshot_id,
                            "--file=" + str(dump_path),
                        ],
                        source_env.environment,
                        failure_code="logical_backup_failed",
                    )
                finally:
                    await transaction.rollback()

            dump_bytes = dump_path.stat().st_size
            if dump_bytes < 1:
                raise RecoveryDrillError("logical_backup_empty")
            dump_sha256_hasher = hashlib.sha256()
            with dump_path.open("rb") as dump_file:
                for block in iter(lambda: dump_file.read(1024 * 1024), b""):
                    dump_sha256_hasher.update(block)
            dump_sha256 = dump_sha256_hasher.hexdigest()

            async with admin_pool.acquire() as admin_connection:
                await admin_connection.execute(f'CREATE DATABASE "{target_database}" TEMPLATE template0')
                await admin_connection.execute(
                    f"ALTER DATABASE \"{target_database}\" SET {_GUARD_SETTING} = '{_guard_digest(guard_token)}'"
                )

            target_admin_dsn = _database_dsn(admin_dsn, target_database)
            target_admin_pool = await asyncpg.create_pool(target_admin_dsn, min_size=1, max_size=2, timeout=10)
            await assert_disposable_database_target(target_admin_pool, target_database, guard_token)
            async with target_admin_pool.acquire() as target_connection:
                await _execute_asset(target_connection, roles_sql, target_database)

            target_environment = dict(admin_env.environment)
            target_environment["PGDATABASE"] = target_database
            restore_command = await _run_pg_tool(
                "pg_restore",
                [
                    "--dbname=" + target_database,
                    "--exit-on-error",
                    "--single-transaction",
                    "--no-owner",
                    "--no-privileges",
                    str(dump_path),
                ],
                target_environment,
                failure_code="logical_restore_failed",
            )
            async with target_admin_pool.acquire() as target_connection:
                await _execute_asset(target_connection, grants_sql, target_database)
                login_specs, verification_logins = await _provision_verification_logins(target_connection)

            restored = await _verify_restored_identities(admin_dsn, target_database, login_specs)
            if not _same_logical_snapshot(source, restored):
                raise RecoveryDrillError("restored_logical_fingerprint_mismatch")

            return RecoveryEvidence(
                schema_version=2,
                workflow="logical_backup_restore_drill",
                status="passed",
                target_fingerprint_sha256=_target_fingerprint(target_database),
                started_at_epoch=started_wall,
                elapsed_ms=round((time.perf_counter() - started) * 1000),
                source=source,
                restored=restored,
                dump_bytes=dump_bytes,
                dump_sha256=dump_sha256,
                backup_elapsed_ms=backup_command.elapsed_ms,
                restore_elapsed_ms=restore_command.elapsed_ms,
                identities_verified=True,
            )
    except RecoveryDrillError:
        raise
    except Exception:
        raise RecoveryDrillError("backup_restore_drill_failed") from None
    finally:
        cleanup_failed = False
        if target_admin_pool is not None and verification_logins:
            try:
                async with target_admin_pool.acquire() as connection:
                    await _drop_verification_logins(connection, verification_logins)
            except Exception:
                cleanup_failed = True
        if target_admin_pool is not None:
            await target_admin_pool.close()
        if admin_pool is not None:
            await admin_pool.close()
        if source_pool is not None:
            await source_pool.close()
        # The restored database is deliberately retained as evidence.  This
        # workflow never drops an existing database, even after failure.
        if cleanup_failed:
            raise RecoveryDrillError("verification_login_cleanup_failed")
