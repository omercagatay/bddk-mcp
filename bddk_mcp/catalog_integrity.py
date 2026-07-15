"""SELECT-only attestation of retrieval-critical PostgreSQL objects.

The migration ledger proves which statements were recorded as applied.  It
does not, by itself, prove that a later administrator has not disabled a
trigger, replaced a function, dropped a foreign key, or rebuilt a search index
under the same name.  This module checks the small set of objects whose drift
could make retrieval silently stale or unsupported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

import asyncpg

from bddk_mcp.migrations.v0005_corpus_release_publication import V0005_CORPUS_RELEASE_PUBLICATION
from bddk_mcp.migrations.v0006_legal_status_resolver import V0006_LEGAL_STATUS_RESOLVER
from bddk_mcp.regulatory.text_profile import PROVISION_BOUNDARY_CODEPOINTS_V1

_CORPUS_RELEASE_RELATIONS_SQL = """
SELECT relation.relname,
       relation.relkind,
       owner.rolname AS owner_name,
       ledger_owner.rolname AS ledger_owner_name,
       COALESCE(relation.reloptions, ARRAY[]::pg_catalog.text[]) AS options,
       CASE WHEN relation.relkind = 'S' THEN ARRAY[]::pg_catalog.name[]
            ELSE COALESCE(
                ARRAY(
                    SELECT attribute.attname
                    FROM pg_catalog.pg_attribute AS attribute
                    WHERE attribute.attrelid = relation.oid
                      AND attribute.attnum > 0
                      AND NOT attribute.attisdropped
                    ORDER BY attribute.attnum
                ),
                ARRAY[]::pg_catalog.name[]
            )
       END AS columns
FROM pg_catalog.pg_class AS relation
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = relation.relnamespace
JOIN pg_catalog.pg_roles AS owner
  ON owner.oid = relation.relowner
JOIN pg_catalog.pg_class AS ledger
  ON ledger.oid = 'bddk_meta.schema_migrations'::pg_catalog.regclass
JOIN pg_catalog.pg_roles AS ledger_owner
  ON ledger_owner.oid = ledger.relowner
WHERE namespace.nspname = 'bddk_meta'
  AND relation.relname = ANY($1::pg_catalog.text[])
ORDER BY relation.relname
"""

_CORPUS_RELEASE_CONSTRAINTS_SQL = """
SELECT relation.relname,
       constraint_record.conname,
       constraint_record.contype,
       constraint_record.convalidated,
       pg_catalog.pg_get_constraintdef(constraint_record.oid, false) AS definition
FROM pg_catalog.pg_constraint AS constraint_record
JOIN pg_catalog.pg_class AS relation
  ON relation.oid = constraint_record.conrelid
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = relation.relnamespace
WHERE namespace.nspname = 'bddk_meta'
  AND relation.relname = ANY($1::pg_catalog.text[])
ORDER BY relation.relname, constraint_record.conname
"""

_CORPUS_RELEASE_TRIGGERS_SQL = """
SELECT relation.relname,
       trigger_record.tgname,
       trigger_record.tgenabled,
       trigger_record.tgtype,
       routine_namespace.nspname || '.' || routine.proname || '('
           || pg_catalog.pg_get_function_identity_arguments(routine.oid) || ')'
           AS function_identity
FROM pg_catalog.pg_trigger AS trigger_record
JOIN pg_catalog.pg_class AS relation
  ON relation.oid = trigger_record.tgrelid
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = relation.relnamespace
JOIN pg_catalog.pg_proc AS routine
  ON routine.oid = trigger_record.tgfoid
JOIN pg_catalog.pg_namespace AS routine_namespace
  ON routine_namespace.oid = routine.pronamespace
WHERE namespace.nspname = 'bddk_meta'
  AND relation.relname = ANY($1::pg_catalog.text[])
  AND NOT trigger_record.tgisinternal
ORDER BY relation.relname, trigger_record.tgname
"""

_CORPUS_RELEASE_ROUTINES_SQL = """
SELECT routine.proname || '(' || pg_catalog.oidvectortypes(routine.proargtypes) || ')'
           AS function_identity,
       language.lanname AS language,
       routine.provolatile,
       routine.proparallel,
       routine.prosecdef,
       routine.proleakproof,
       COALESCE(routine.proconfig, ARRAY[]::pg_catalog.text[]) AS configuration,
       routine.prosrc AS source,
       owner.rolname AS owner_name,
       ledger_owner.rolname AS ledger_owner_name,
       EXISTS (
           SELECT 1
           FROM pg_catalog.aclexplode(
               COALESCE(routine.proacl, pg_catalog.acldefault('f'::"char", routine.proowner))
           ) AS acl
           WHERE acl.grantee = 0
             AND acl.privilege_type = 'EXECUTE'
       ) AS public_can_execute
FROM pg_catalog.pg_proc AS routine
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = routine.pronamespace
JOIN pg_catalog.pg_language AS language
  ON language.oid = routine.prolang
JOIN pg_catalog.pg_roles AS owner
  ON owner.oid = routine.proowner
JOIN pg_catalog.pg_class AS ledger
  ON ledger.oid = 'bddk_meta.schema_migrations'::pg_catalog.regclass
JOIN pg_catalog.pg_roles AS ledger_owner
  ON ledger_owner.oid = ledger.relowner
WHERE namespace.nspname = 'bddk_meta'
  AND routine.proname = ANY($1::pg_catalog.text[])
ORDER BY routine.proname, pg_catalog.oidvectortypes(routine.proargtypes)
"""

_ACTIVE_CORPUS_RELEASE_VIEW_SQL = """
SELECT pg_catalog.pg_get_viewdef(relation.oid, false) AS definition,
       COALESCE(
           ARRAY(
               SELECT DISTINCT dependency_namespace.nspname || '.' || dependency_relation.relname
               FROM pg_catalog.pg_rewrite AS rewrite
               JOIN pg_catalog.pg_depend AS dependency
                 ON dependency.classid = 'pg_catalog.pg_rewrite'::pg_catalog.regclass
                AND dependency.objid = rewrite.oid
                AND dependency.refclassid = 'pg_catalog.pg_class'::pg_catalog.regclass
               JOIN pg_catalog.pg_class AS dependency_relation
                 ON dependency_relation.oid = dependency.refobjid
               JOIN pg_catalog.pg_namespace AS dependency_namespace
                 ON dependency_namespace.oid = dependency_relation.relnamespace
               WHERE rewrite.ev_class = relation.oid
                 AND dependency_relation.oid <> relation.oid
               ORDER BY dependency_namespace.nspname || '.' || dependency_relation.relname
           ),
           ARRAY[]::pg_catalog.text[]
       ) AS dependencies
FROM pg_catalog.pg_class AS relation
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = relation.relnamespace
WHERE namespace.nspname = 'bddk_meta'
  AND relation.relname = 'active_corpus_release'
  AND relation.relkind = 'v'
"""

_LEGAL_STATUS_ROUTINE_SQL = """
SELECT routine.proname || '(' || pg_catalog.oidvectortypes(routine.proargtypes) || ')'
           AS function_identity,
       language.lanname AS language,
       routine.provolatile,
       routine.proparallel,
       routine.prosecdef,
       routine.proleakproof,
       routine.proisstrict,
       routine.proretset,
       pg_catalog.pg_get_function_result(routine.oid) AS result_type,
       COALESCE(routine.proconfig, ARRAY[]::pg_catalog.text[]) AS configuration,
       routine.prosrc AS source,
       owner.rolname AS owner_name,
       ledger_owner.rolname AS ledger_owner_name,
       EXISTS (
           SELECT 1
           FROM pg_catalog.aclexplode(
               COALESCE(routine.proacl, pg_catalog.acldefault('f'::"char", routine.proowner))
           ) AS acl
           WHERE acl.grantee = 0
             AND acl.privilege_type = 'EXECUTE'
       ) AS public_can_execute
FROM pg_catalog.pg_proc AS routine
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = routine.pronamespace
JOIN pg_catalog.pg_language AS language
  ON language.oid = routine.prolang
JOIN pg_catalog.pg_roles AS owner
  ON owner.oid = routine.proowner
JOIN pg_catalog.pg_class AS ledger
  ON ledger.oid = 'bddk_meta.schema_migrations'::pg_catalog.regclass
JOIN pg_catalog.pg_roles AS ledger_owner
  ON ledger_owner.oid = ledger.relowner
WHERE namespace.nspname = 'bddk_meta'
  AND routine.proname = 'resolve_regulation_status'
"""

_CONSTRAINTS_SQL = """
SELECT namespace.nspname || '.' || relation.relname AS table_name,
       constraint_record.conname,
       constraint_record.contype,
       constraint_record.convalidated,
       pg_catalog.pg_get_constraintdef(constraint_record.oid, false) AS definition
FROM pg_catalog.pg_constraint AS constraint_record
JOIN pg_catalog.pg_class AS relation
  ON relation.oid = constraint_record.conrelid
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = relation.relnamespace
WHERE namespace.nspname = 'public'
  AND relation.relname = ANY($1::pg_catalog.text[])
ORDER BY relation.relname, constraint_record.conname
"""

_TRIGGERS_SQL = """
SELECT namespace.nspname || '.' || relation.relname AS table_name,
       trigger_record.tgname,
       trigger_record.tgenabled,
       trigger_record.tgtype,
       routine.proname || '(' || pg_catalog.pg_get_function_identity_arguments(routine.oid) || ')'
           AS function_identity
FROM pg_catalog.pg_trigger AS trigger_record
JOIN pg_catalog.pg_class AS relation
  ON relation.oid = trigger_record.tgrelid
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = relation.relnamespace
JOIN pg_catalog.pg_proc AS routine
  ON routine.oid = trigger_record.tgfoid
WHERE namespace.nspname = 'public'
  AND relation.relname = ANY($1::pg_catalog.text[])
  AND NOT trigger_record.tgisinternal
ORDER BY relation.relname, trigger_record.tgname
"""

_INDEXES_SQL = """
SELECT index_relation.relname AS index_name,
       access_method.amname AS method,
       index_record.indisunique,
       index_record.indisprimary,
       index_record.indisvalid,
       index_record.indisready,
       COALESCE(
           ARRAY(
               SELECT pg_catalog.pg_get_indexdef(
                   index_record.indexrelid,
                   key_position,
                   true
               )
               FROM pg_catalog.generate_series(
                   1,
                   index_record.indnkeyatts
               ) AS key_position
           ),
           ARRAY[]::pg_catalog.text[]
       ) AS keys,
       COALESCE(
           ARRAY(
               SELECT operator_class.opcname
               FROM pg_catalog.unnest(index_record.indclass)
                    WITH ORDINALITY AS class_oid(oid, position)
               JOIN pg_catalog.pg_opclass AS operator_class
                 ON operator_class.oid = class_oid.oid
               WHERE class_oid.position <= index_record.indnkeyatts
               ORDER BY class_oid.position
           ),
           ARRAY[]::pg_catalog.text[]
       ) AS opclasses,
       COALESCE(index_relation.reloptions, ARRAY[]::pg_catalog.text[]) AS options
FROM pg_catalog.pg_index AS index_record
JOIN pg_catalog.pg_class AS index_relation
  ON index_relation.oid = index_record.indexrelid
JOIN pg_catalog.pg_class AS table_relation
  ON table_relation.oid = index_record.indrelid
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = table_relation.relnamespace
JOIN pg_catalog.pg_am AS access_method
  ON access_method.oid = index_relation.relam
WHERE namespace.nspname = 'public'
  AND index_relation.relname = ANY($1::pg_catalog.text[])
ORDER BY index_relation.relname
"""

_ROUTINES_SQL = """
SELECT routine.proname || '(' || pg_catalog.pg_get_function_identity_arguments(routine.oid) || ')'
           AS function_identity,
       language.lanname AS language,
       routine.provolatile,
       routine.proparallel,
       routine.prosecdef,
       routine.proleakproof,
       COALESCE(routine.proconfig, ARRAY[]::pg_catalog.text[]) AS configuration,
       routine.prosrc AS source
FROM pg_catalog.pg_proc AS routine
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = routine.pronamespace
JOIN pg_catalog.pg_language AS language
  ON language.oid = routine.prolang
WHERE namespace.nspname = 'public'
  AND routine.proname = ANY($1::pg_catalog.text[])
ORDER BY routine.proname, pg_catalog.pg_get_function_identity_arguments(routine.oid)
"""

_CITATION_VIEW_SQL = """
SELECT relation.relkind,
       owner.rolname AS owner_name,
       (
           SELECT ledger_owner.rolname
           FROM pg_catalog.pg_class AS ledger
           JOIN pg_catalog.pg_namespace AS ledger_namespace
             ON ledger_namespace.oid = ledger.relnamespace
           JOIN pg_catalog.pg_roles AS ledger_owner
             ON ledger_owner.oid = ledger.relowner
           WHERE ledger_namespace.nspname = 'bddk_meta'
             AND ledger.relname = 'schema_migrations'
             AND ledger.relkind IN ('r', 'p')
       ) AS ledger_owner_name,
       COALESCE(relation.reloptions, ARRAY[]::pg_catalog.text[]) AS options,
       pg_catalog.pg_get_viewdef(relation.oid, false) AS definition,
       COALESCE(
           ARRAY(
               SELECT attribute.attname
               FROM pg_catalog.pg_attribute AS attribute
               WHERE attribute.attrelid = relation.oid
                 AND attribute.attnum > 0
                 AND NOT attribute.attisdropped
               ORDER BY attribute.attnum
           ),
           ARRAY[]::pg_catalog.name[]
       ) AS columns,
       COALESCE(
           ARRAY(
               SELECT DISTINCT dependency_namespace.nspname || '.' || dependency_relation.relname
               FROM pg_catalog.pg_rewrite AS rewrite
               JOIN pg_catalog.pg_depend AS dependency
                 ON dependency.classid = 'pg_catalog.pg_rewrite'::pg_catalog.regclass
                AND dependency.objid = rewrite.oid
                AND dependency.refclassid = 'pg_catalog.pg_class'::pg_catalog.regclass
               JOIN pg_catalog.pg_class AS dependency_relation
                 ON dependency_relation.oid = dependency.refobjid
               JOIN pg_catalog.pg_namespace AS dependency_namespace
                 ON dependency_namespace.oid = dependency_relation.relnamespace
               WHERE rewrite.ev_class = relation.oid
                 AND dependency_relation.oid <> relation.oid
                 AND dependency_namespace.nspname = 'public'
               ORDER BY dependency_namespace.nspname || '.' || dependency_relation.relname
           ),
           ARRAY[]::pg_catalog.text[]
       ) AS dependencies
FROM pg_catalog.pg_class AS relation
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = relation.relnamespace
JOIN pg_catalog.pg_roles AS owner
  ON owner.oid = relation.relowner
WHERE namespace.nspname = 'public'
  AND relation.relname = 'regulatory_validated_section_citations'
"""

_V4_TABLES: Final[tuple[str, ...]] = (
    "regulatory_evidence",
    "regulatory_family_imports",
    "regulatory_instruments",
    "regulatory_legal_events",
    "regulatory_legal_status_assertions",
    "regulatory_legal_version_artifacts",
    "regulatory_legal_version_provisions",
    "regulatory_legal_versions",
    "regulatory_provisions",
    "regulatory_source_artifacts",
    "regulatory_source_blobs",
)

_V4_CONSTRAINT_CATALOG_SQL = """
WITH catalog_items AS (
    SELECT pg_catalog.jsonb_build_array(
               relation.relname,
               constraint_record.conname,
               constraint_record.contype,
               constraint_record.convalidated,
               pg_catalog.pg_get_constraintdef(constraint_record.oid, false)
           )::pg_catalog.text AS item
    FROM pg_catalog.pg_constraint AS constraint_record
    JOIN pg_catalog.pg_class AS relation
      ON relation.oid = constraint_record.conrelid
    JOIN pg_catalog.pg_namespace AS namespace
      ON namespace.oid = relation.relnamespace
    WHERE namespace.nspname = 'public'
      AND relation.relname = ANY($1::pg_catalog.text[])
)
SELECT pg_catalog.count(*)::pg_catalog.int4 AS object_count,
       pg_catalog.encode(
           pg_catalog.sha256(
               pg_catalog.convert_to(
                   pg_catalog.string_agg(item, E'\\n' ORDER BY item),
                   'UTF8'
               )
           ),
           'hex'
       ) AS v4_constraint_catalog_sha256
FROM catalog_items
"""

_V4_INDEX_CATALOG_SQL = """
WITH catalog_items AS (
    SELECT pg_catalog.jsonb_build_array(
               table_relation.relname,
               index_relation.relname,
               access_method.amname,
               index_record.indisunique,
               index_record.indisprimary,
               index_record.indisvalid,
               index_record.indisready,
               index_record.indisclustered,
               index_record.indisreplident,
               index_record.indnullsnotdistinct,
               pg_catalog.pg_get_indexdef(index_record.indexrelid, 0, false),
               COALESCE(index_relation.reloptions, ARRAY[]::pg_catalog.text[])
           )::pg_catalog.text AS item
    FROM pg_catalog.pg_index AS index_record
    JOIN pg_catalog.pg_class AS index_relation
      ON index_relation.oid = index_record.indexrelid
    JOIN pg_catalog.pg_class AS table_relation
      ON table_relation.oid = index_record.indrelid
    JOIN pg_catalog.pg_namespace AS namespace
      ON namespace.oid = table_relation.relnamespace
    JOIN pg_catalog.pg_am AS access_method
      ON access_method.oid = index_relation.relam
    WHERE namespace.nspname = 'public'
      AND table_relation.relname = ANY($1::pg_catalog.text[])
)
SELECT pg_catalog.count(*)::pg_catalog.int4 AS object_count,
       pg_catalog.encode(
           pg_catalog.sha256(
               pg_catalog.convert_to(
                   pg_catalog.string_agg(item, E'\\n' ORDER BY item),
                   'UTF8'
               )
           ),
           'hex'
       ) AS v4_index_catalog_sha256
FROM catalog_items
"""

_EXPECTED_V4_CONSTRAINT_COUNT: Final[int] = 69
_EXPECTED_V4_CONSTRAINT_CATALOG_SHA256: Final[str] = "145336033f6236c27c925a4c2339423d398b7358da5c0d589bee3e96cbec0e3a"
_EXPECTED_V4_INDEX_COUNT: Final[int] = 21
_EXPECTED_V4_INDEX_CATALOG_SHA256: Final[str] = "7917cee6cca9a86c358cee10916c0b7fcccebfdaef9ad18bf2ffab5415896c04"


def _normalize_sql(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return text.replace("public.", "").replace("::text", "")


def _normalize_view_sql(value: Any) -> str:
    """Normalize PostgreSQL 17's view deparse without weakening source DDL."""

    return _normalize_sql(value).replace('"', "").replace("pg_catalog.", "").replace("::name", "")


def _v5_function_source(name: str) -> str:
    """Extract the immutable function body from migration v0005."""

    prefix = f"CREATE FUNCTION bddk_meta.{name}("
    statement = next(
        (item for item in V0005_CORPUS_RELEASE_PUBLICATION.statements if item.strip().startswith(prefix)),
        None,
    )
    if statement is None:
        raise RuntimeError(f"migration v0005 is missing {name}")
    match = re.search(r"\bAS \$function\$\s*(.*?)\s*\$function\$\s*$", statement, re.DOTALL)
    if match is None:
        raise RuntimeError(f"migration v0005 has an invalid {name} body")
    return match.group(1)


def _v6_legal_status_function_source() -> str:
    """Extract the immutable resolver body from migration v0006."""

    prefix = "CREATE FUNCTION bddk_meta.resolve_regulation_status("
    statement = next(
        (item for item in V0006_LEGAL_STATUS_RESOLVER.statements if item.strip().startswith(prefix)),
        None,
    )
    if statement is None:
        raise RuntimeError("migration v0006 is missing the legal-status resolver")
    match = re.search(r"\bAS \$function\$\s*(.*?)\s*\$function\$\s*$", statement, re.DOTALL)
    if match is None:
        raise RuntimeError("migration v0006 has an invalid legal-status resolver body")
    return match.group(1)


_CORPUS_RELEASE_RELATIONS: Final[dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]]] = {
    "active_corpus_release": (
        "v",
        (
            "release_id",
            "manifest_id",
            "manifest_sha256",
            "signer_key_sha256",
            "freshness_policy_result",
            "source_detection_slo_seconds",
            "publication_slo_seconds",
            "max_manifest_age_seconds",
            "retrieval_profile_sha256",
            "corpus_state_sha256",
            "completed_at",
        ),
        ("security_barrier=true", "security_invoker=false"),
    ),
    "corpus_release_activations": (
        "r",
        (
            "activation_sequence",
            "release_id",
            "completed_at",
            "actor_fingerprint_sha256",
        ),
        (),
    ),
    "corpus_release_activations_activation_sequence_seq": ("S", (), ()),
    "corpus_releases": (
        "r",
        (
            "release_id",
            "manifest_id",
            "manifest_sha256",
            "signer_key_sha256",
            "freshness_policy_result",
            "source_detection_slo_seconds",
            "publication_slo_seconds",
            "max_manifest_age_seconds",
            "retrieval_profile_sha256",
            "corpus_state_sha256",
            "created_at",
        ),
        (),
    ),
}

_CORPUS_RELEASE_CONSTRAINTS: Final[dict[tuple[str, str], tuple[str, str]]] = {
    ("corpus_releases", "corpus_releases_pkey"): ("p", "PRIMARY KEY (release_id)"),
    ("corpus_releases", "corpus_releases_id_check"): (
        "c",
        "CHECK ((release_id ~ '^corpus_release_sha256_[0-9a-f]{64}$'))",
    ),
    ("corpus_releases", "corpus_releases_manifest_id_check"): (
        "c",
        "CHECK ((manifest_id ~ '^[a-z0-9][a-z0-9._-]{2,127}$'))",
    ),
    ("corpus_releases", "corpus_releases_manifest_hash_check"): (
        "c",
        "CHECK ((manifest_sha256 ~ '^[0-9a-f]{64}$'))",
    ),
    ("corpus_releases", "corpus_releases_signer_hash_check"): (
        "c",
        "CHECK ((signer_key_sha256 ~ '^[0-9a-f]{64}$'))",
    ),
    ("corpus_releases", "corpus_releases_policy_result_check"): (
        "c",
        "CHECK ((freshness_policy_result = 'quantified_measured_signature_verified_pass'))",
    ),
    ("corpus_releases", "corpus_releases_source_detection_slo_check"): (
        "c",
        "CHECK ((source_detection_slo_seconds > 0))",
    ),
    ("corpus_releases", "corpus_releases_publication_slo_check"): (
        "c",
        "CHECK ((publication_slo_seconds > 0))",
    ),
    ("corpus_releases", "corpus_releases_max_age_check"): (
        "c",
        "CHECK ((max_manifest_age_seconds > 0))",
    ),
    ("corpus_releases", "corpus_releases_profile_hash_check"): (
        "c",
        "CHECK ((retrieval_profile_sha256 ~ '^[0-9a-f]{64}$'))",
    ),
    ("corpus_releases", "corpus_releases_state_hash_check"): (
        "c",
        "CHECK ((corpus_state_sha256 ~ '^[0-9a-f]{64}$'))",
    ),
    ("corpus_release_activations", "corpus_release_activations_pkey"): (
        "p",
        "PRIMARY KEY (activation_sequence)",
    ),
    ("corpus_release_activations", "corpus_release_activations_release_fk"): (
        "f",
        "FOREIGN KEY (release_id) REFERENCES bddk_meta.corpus_releases(release_id)",
    ),
    ("corpus_release_activations", "corpus_release_activations_actor_hash_check"): (
        "c",
        "CHECK ((actor_fingerprint_sha256 ~ '^[0-9a-f]{64}$'))",
    ),
}

_CORPUS_RELEASE_TRIGGERS: Final[dict[tuple[str, str], tuple[str, int]]] = {
    ("corpus_releases", "reject_corpus_release_update_delete"): (
        "bddk_meta.reject_corpus_release_mutation()",
        27,
    ),
    ("corpus_release_activations", "reject_corpus_release_activation_update_delete"): (
        "bddk_meta.reject_corpus_release_mutation()",
        27,
    ),
}

_CORPUS_RELEASE_ROUTINES: Final[dict[str, tuple[str, str, str, bool, str]]] = {
    "corpus_fingerprint_frame(text)": (
        "sql",
        "i",
        "s",
        False,
        _v5_function_source("corpus_fingerprint_frame"),
    ),
    "corpus_retrieval_ready(text)": (
        "sql",
        "s",
        "u",
        True,
        _v5_function_source("corpus_retrieval_ready"),
    ),
    "current_corpus_state_sha256(text)": (
        "sql",
        "s",
        "u",
        True,
        _v5_function_source("current_corpus_state_sha256"),
    ),
    "publish_verified_corpus_release(text, text, text, integer, integer, integer, text)": (
        "plpgsql",
        "v",
        "u",
        True,
        _v5_function_source("publish_verified_corpus_release"),
    ),
    "reject_corpus_release_mutation()": (
        "plpgsql",
        "v",
        "u",
        False,
        _v5_function_source("reject_corpus_release_mutation"),
    ),
}

_ACTIVE_CORPUS_RELEASE_DEPENDENCIES: Final[tuple[str, ...]] = (
    "bddk_meta.corpus_release_activations",
    "bddk_meta.corpus_releases",
)
_ACTIVE_CORPUS_RELEASE_REQUIRED_DEFINITION: Final[tuple[str, ...]] = (
    "activation.activation_sequence = ( select max(latest.activation_sequence) as max",
    "release.release_id = activation.release_id",
    "release.corpus_state_sha256 = bddk_meta.current_corpus_state_sha256(release.retrieval_profile_sha256)",
    "bddk_meta.corpus_retrieval_ready(release.retrieval_profile_sha256)",
)
_LEGAL_STATUS_RESULT_TYPE: Final[str] = (
    "TABLE(resolved boolean, reason text, instrument_id text, as_of date, legal_version_id text, "
    "version_key text, legal_text_sha256 text, version_review_record_sha256 text, amends_version_id text, "
    "consolidation_state text, evidence_json text)"
)


_EXPECTED_CONSTRAINTS: Final[dict[tuple[str, str], tuple[str, str]]] = {
    ("public.document_chunks", "document_chunks_document_fk"): (
        "f",
        "FOREIGN KEY (doc_id) REFERENCES documents(document_id) ON DELETE CASCADE",
    ),
    ("public.document_chunks", "document_chunks_document_index_uq"): (
        "u",
        "UNIQUE (doc_id, chunk_index)",
    ),
    ("public.document_chunks", "document_chunks_pkey"): ("p", "PRIMARY KEY (id)"),
    ("public.document_sections", "document_sections_document_fk"): (
        "f",
        "FOREIGN KEY (doc_id) REFERENCES documents(document_id) ON DELETE CASCADE",
    ),
    ("public.document_sections", "document_sections_identity_uq"): (
        "u",
        "UNIQUE (doc_id, section_type, section_ref, content_hash)",
    ),
    ("public.document_sections", "document_sections_pkey"): ("p", "PRIMARY KEY (id)"),
    ("public.documents", "documents_pkey"): ("p", "PRIMARY KEY (document_id)"),
    ("public.document_retrieval_publications", "document_retrieval_publications_pkey"): (
        "p",
        "PRIMARY KEY (doc_id)",
    ),
    ("public.document_retrieval_publications", "document_retrieval_publications_document_fk"): (
        "f",
        "FOREIGN KEY (doc_id) REFERENCES documents(document_id) ON DELETE CASCADE",
    ),
    ("public.document_retrieval_publications", "document_retrieval_publications_content_hash_check"): (
        "c",
        "CHECK ((content_hash ~ '^[0-9a-f]{64}$'))",
    ),
    ("public.document_retrieval_publications", "document_retrieval_publications_profile_hash_check"): (
        "c",
        "CHECK ((retrieval_profile_hash ~ '^[0-9a-f]{64}$'))",
    ),
    ("public.document_retrieval_publications", "document_retrieval_publications_expected_chunks_check"): (
        "c",
        "CHECK ((expected_chunks > 0))",
    ),
}

_EXPECTED_TRIGGERS: Final[dict[tuple[str, str], tuple[str, int]]] = {
    ("public.documents", "trg_documents_tsv"): ("documents_tsv_trigger()", 23),
    ("public.document_sections", "trg_document_sections_tsv"): (
        "document_sections_tsv_trigger()",
        23,
    ),
    ("public.document_chunks", "chunks_tsv_update"): ("chunks_tsv_trigger()", 23),
    ("public.document_chunks", "invalidate_retrieval_publication_on_chunk_change"): (
        "invalidate_retrieval_publication()",
        29,
    ),
}

_EXPECTED_INDEXES: Final[dict[str, tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...]]]] = {
    "idx_documents_tsv": ("gin", ("tsv",), ("tsvector_ops",), ()),
    "idx_document_sections_doc_id": ("btree", ("doc_id",), ("text_ops",), ()),
    "idx_document_sections_tsv": ("gin", ("tsv",), ("tsvector_ops",), ()),
    "idx_chunks_doc_id": ("btree", ("doc_id",), ("text_ops",), ()),
    "idx_chunks_tsv": ("gin", ("tsv",), ("tsvector_ops",), ()),
    "idx_chunks_embedding_hnsw": (
        "hnsw",
        ("embedding",),
        ("vector_cosine_ops",),
        ("ef_construction=64", "m=16"),
    ),
}

_EXPECTED_ROUTINES: Final[dict[str, tuple[str, str, str, str]]] = {
    "immutable_unaccent(text)": (
        "sql",
        "i",
        "s",
        "SELECT public.unaccent($1)",
    ),
    "documents_tsv_trigger()": (
        "plpgsql",
        "v",
        "u",
        """
        BEGIN
            NEW.tsv :=
                pg_catalog.to_tsvector(
                    'simple'::pg_catalog.regconfig,
                    public.immutable_unaccent(COALESCE(NEW.title, ''))
                )
                || pg_catalog.to_tsvector(
                    'simple'::pg_catalog.regconfig,
                    public.immutable_unaccent(COALESCE(NEW.markdown_content, ''))
                )
                || pg_catalog.to_tsvector(
                    'simple'::pg_catalog.regconfig,
                    public.immutable_unaccent(COALESCE(NEW.category, ''))
                );
            RETURN NEW;
        END
        """,
    ),
    "document_sections_tsv_trigger()": (
        "plpgsql",
        "v",
        "u",
        """
        BEGIN
            NEW.tsv :=
                pg_catalog.to_tsvector(
                    'simple'::pg_catalog.regconfig,
                    public.immutable_unaccent(COALESCE(NEW.heading, ''))
                )
                || pg_catalog.to_tsvector(
                    'simple'::pg_catalog.regconfig,
                    public.immutable_unaccent(COALESCE(NEW.content, ''))
                );
            RETURN NEW;
        END
        """,
    ),
    "chunks_tsv_trigger()": (
        "plpgsql",
        "v",
        "u",
        """
        BEGIN
            NEW.tsv :=
                pg_catalog.to_tsvector(
                    'simple'::pg_catalog.regconfig,
                    public.immutable_unaccent(COALESCE(NEW.title, ''))
                )
                || pg_catalog.to_tsvector(
                    'simple'::pg_catalog.regconfig,
                    public.immutable_unaccent(COALESCE(NEW.chunk_text, ''))
                );
            RETURN NEW;
        END
        """,
    ),
    "invalidate_retrieval_publication()": (
        "plpgsql",
        "v",
        "u",
        """
        BEGIN
            IF TG_OP = 'INSERT' THEN
                DELETE FROM public.document_retrieval_publications
                WHERE doc_id = NEW.doc_id;
                RETURN NEW;
            END IF;
            DELETE FROM public.document_retrieval_publications
            WHERE doc_id = OLD.doc_id;
            IF TG_OP = 'UPDATE' THEN
                IF OLD.doc_id IS DISTINCT FROM NEW.doc_id THEN
                    DELETE FROM public.document_retrieval_publications
                    WHERE doc_id = NEW.doc_id;
                END IF;
                RETURN NEW;
            END IF;
            RETURN OLD;
        END
        """,
    ),
}

_CITATION_VIEW_DEPENDENCIES: Final[tuple[str, ...]] = (
    "public.document_sections",
    "public.documents",
    "public.regulatory_evidence",
    "public.regulatory_instruments",
    "public.regulatory_legal_version_artifacts",
    "public.regulatory_legal_version_provisions",
    "public.regulatory_legal_versions",
    "public.regulatory_provisions",
    "public.regulatory_source_artifacts",
    "public.regulatory_source_blobs",
)
_CITATION_VIEW_COLUMNS: Final[tuple[str, ...]] = (
    "document_section_id",
    "source_document_id",
    "normalized_document_sha256",
    "normalized_section_sha256",
    "instrument_id",
    "instrument_jurisdiction",
    "instrument_authority_code",
    "instrument_identity_key",
    "legal_version_id",
    "legal_version_key",
    "legal_text_sha256",
    "review_record_sha256",
    "provision_review_record_sha256",
    "artifact_id",
    "artifact_blob_id",
    "artifact_sha256",
    "source_url",
    "artifact_retrieved_at",
    "evidence_id",
    "evidence_locator",
    "evidence_statement_sha256",
    "provision_id",
    "provision_kind",
    "provision_path",
    "provision_text_sha256",
)
_CITATION_VIEW_REQUIRED_DEFINITION: Final[tuple[str, ...]] = (
    "occurrence.document_section_id",
    "section.doc_id as source_document_id",
    "document.content_hash as normalized_document_sha256",
    "section.content_hash as normalized_section_sha256",
    "version.instrument_id",
    "instrument.jurisdiction as instrument_jurisdiction",
    "instrument.authority_code as instrument_authority_code",
    "instrument.identity_key as instrument_identity_key",
    "version.legal_version_id",
    "version.version_key as legal_version_key",
    "version.legal_text_sha256",
    "version.review_record_sha256",
    "occurrence.review_record_sha256 as provision_review_record_sha256",
    "artifact.artifact_id",
    "artifact.blob_id as artifact_blob_id",
    "blob.content_sha256 as artifact_sha256",
    "artifact.canonical_uri as source_url",
    "artifact.retrieved_at as artifact_retrieved_at",
    "evidence.evidence_id",
    "evidence.locator as evidence_locator",
    "evidence.statement_sha256 as evidence_statement_sha256",
    "provision.provision_id",
    "provision.provision_kind",
    "provision.canonical_path as provision_path",
    "occurrence.provision_text_sha256",
    "section.id = occurrence.document_section_id",
    "section.content_hash = occurrence.provision_text_sha256",
    "section.content_hash = pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(section.content, 'utf8')), 'hex')",
    "document.document_id = section.doc_id",
    "document.content_hash = section.source_content_hash",
    "document.content_hash = pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(document.markdown_content, 'utf8')), 'hex')",
    "section.content = pg_catalog.btrim(pg_catalog.substr(document.markdown_content, (section.start_char + 1), (section.end_char - section.start_char))",
    "version.legal_version_id = occurrence.legal_version_id",
    "version.legal_text_sha256 = document.content_hash",
    "instrument.instrument_id = version.instrument_id",
    "provision.provision_id = occurrence.provision_id",
    "provision.instrument_id = version.instrument_id",
    "evidence.evidence_id = occurrence.evidence_id",
    "evidence.statement_sha256 = occurrence.provision_text_sha256",
    "artifact.artifact_id = evidence.artifact_id",
    "artifact.repository_document_id = section.doc_id",
    "blob.blob_id = artifact.blob_id",
    "version_artifact.legal_version_id = version.legal_version_id",
    "version_artifact.artifact_id = artifact.artifact_id",
    "version_artifact.source_role = 'legal_text'",
    "occurrence.validation_state = 'validated'",
    "version.validation_state = 'validated'",
    "evidence.authority_level = 'authoritative'",
    "artifact.fixture_only = false",
    *(f"pg_catalog.chr({codepoint})" for codepoint in PROVISION_BOUNDARY_CODEPOINTS_V1),
)


@dataclass(frozen=True, slots=True)
class CatalogIntegrity:
    """Bounded labels for retrieval-critical catalog drift."""

    failures: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.failures


def _value(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def _catalog_char(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii", errors="strict")
    return str(value)


async def inspect_catalog_integrity(pool: asyncpg.Pool) -> CatalogIntegrity:
    """Verify critical constraints, triggers, indexes, and function bodies."""

    failures: list[str] = []

    constraint_rows = await pool.fetch(
        _CONSTRAINTS_SQL,
        sorted({table.partition(".")[2] for table, _name in _EXPECTED_CONSTRAINTS}),
    )
    actual_constraints = {
        (str(_value(row, "table_name")), str(_value(row, "conname"))): (
            _catalog_char(_value(row, "contype")),
            bool(_value(row, "convalidated", False)),
            _normalize_sql(_value(row, "definition")),
        )
        for row in constraint_rows
    }
    for key, (constraint_type, definition) in _EXPECTED_CONSTRAINTS.items():
        if actual_constraints.get(key) != (constraint_type, True, _normalize_sql(definition)):
            failures.append(f"constraint:{key[0]}.{key[1]}")

    trigger_rows = await pool.fetch(
        _TRIGGERS_SQL,
        sorted({table.partition(".")[2] for table, _name in _EXPECTED_TRIGGERS}),
    )
    actual_triggers = {
        (str(_value(row, "table_name")), str(_value(row, "tgname"))): (
            str(_value(row, "function_identity")),
            int(_value(row, "tgtype", -1)),
            _catalog_char(_value(row, "tgenabled")),
        )
        for row in trigger_rows
    }
    for key, (function_identity, trigger_type) in _EXPECTED_TRIGGERS.items():
        if actual_triggers.get(key) != (function_identity, trigger_type, "O"):
            failures.append(f"trigger:{key[0]}.{key[1]}")

    index_rows = await pool.fetch(_INDEXES_SQL, sorted(_EXPECTED_INDEXES))
    actual_indexes = {
        str(_value(row, "index_name")): (
            str(_value(row, "method")),
            bool(_value(row, "indisunique", False)),
            bool(_value(row, "indisprimary", False)),
            bool(_value(row, "indisvalid", False)),
            bool(_value(row, "indisready", False)),
            tuple(str(item) for item in (_value(row, "keys", ()) or ())),
            tuple(str(item) for item in (_value(row, "opclasses", ()) or ())),
            tuple(sorted(str(item) for item in (_value(row, "options", ()) or ()))),
        )
        for row in index_rows
    }
    for name, (method, keys, opclasses, options) in _EXPECTED_INDEXES.items():
        if actual_indexes.get(name) != (
            method,
            False,
            False,
            True,
            True,
            keys,
            opclasses,
            tuple(sorted(options)),
        ):
            failures.append(f"index:public.{name}")

    routine_rows = await pool.fetch(
        _ROUTINES_SQL,
        sorted(identity.partition("(")[0] for identity in _EXPECTED_ROUTINES),
    )
    actual_routines = {
        str(_value(row, "function_identity")): (
            str(_value(row, "language")),
            _catalog_char(_value(row, "provolatile")),
            _catalog_char(_value(row, "proparallel")),
            bool(_value(row, "prosecdef", True)),
            bool(_value(row, "proleakproof", True)),
            tuple(str(item) for item in (_value(row, "configuration", ()) or ())),
            _normalize_sql(_value(row, "source")),
        )
        for row in routine_rows
    }
    for identity, (language, volatility, parallel, source) in _EXPECTED_ROUTINES.items():
        if actual_routines.get(identity) != (
            language,
            volatility,
            parallel,
            False,
            False,
            ("search_path=pg_catalog, public",),
            _normalize_sql(source),
        ):
            failures.append(f"routine:public.{identity}")

    v4_constraints = await pool.fetchrow(_V4_CONSTRAINT_CATALOG_SQL, list(_V4_TABLES))
    if (
        int(_value(v4_constraints, "object_count", -1)) != _EXPECTED_V4_CONSTRAINT_COUNT
        or str(_value(v4_constraints, "v4_constraint_catalog_sha256", "")) != _EXPECTED_V4_CONSTRAINT_CATALOG_SHA256
    ):
        failures.append("constraints:public.regulatory_v4_exact")

    v4_indexes = await pool.fetchrow(_V4_INDEX_CATALOG_SQL, list(_V4_TABLES))
    if (
        int(_value(v4_indexes, "object_count", -1)) != _EXPECTED_V4_INDEX_COUNT
        or str(_value(v4_indexes, "v4_index_catalog_sha256", "")) != _EXPECTED_V4_INDEX_CATALOG_SHA256
    ):
        failures.append("indexes:public.regulatory_v4_exact")

    citation_view = await pool.fetchrow(_CITATION_VIEW_SQL)
    view_definition = _normalize_view_sql(_value(citation_view, "definition"))
    view_valid = bool(
        citation_view is not None
        and _catalog_char(_value(citation_view, "relkind")) == "v"
        and bool(str(_value(citation_view, "ledger_owner_name", "")))
        and str(_value(citation_view, "owner_name")) == str(_value(citation_view, "ledger_owner_name", ""))
        and tuple(sorted(str(item) for item in (_value(citation_view, "options", ()) or ())))
        == ("security_barrier=true", "security_invoker=false")
        and tuple(str(item) for item in (_value(citation_view, "dependencies", ()) or ()))
        == _CITATION_VIEW_DEPENDENCIES
        and tuple(str(item) for item in (_value(citation_view, "columns", ()) or ())) == _CITATION_VIEW_COLUMNS
        and all(_normalize_view_sql(fragment) in view_definition for fragment in _CITATION_VIEW_REQUIRED_DEFINITION)
        and " or " not in view_definition
        and " union " not in view_definition
    )
    if not view_valid:
        failures.append("view:public.regulatory_validated_section_citations")

    release_relation_rows = await pool.fetch(
        _CORPUS_RELEASE_RELATIONS_SQL,
        sorted(_CORPUS_RELEASE_RELATIONS),
    )
    actual_release_relations = {
        str(_value(row, "relname")): (
            _catalog_char(_value(row, "relkind")),
            tuple(str(item) for item in (_value(row, "columns", ()) or ())),
            tuple(sorted(str(item) for item in (_value(row, "options", ()) or ()))),
            str(_value(row, "owner_name", "")),
            str(_value(row, "ledger_owner_name", "")),
        )
        for row in release_relation_rows
    }
    expected_release_relation_names = set(_CORPUS_RELEASE_RELATIONS)
    if set(actual_release_relations) != expected_release_relation_names:
        failures.append("relations:bddk_meta.corpus_release_exact")
    else:
        for name, (relation_kind, columns, options) in _CORPUS_RELEASE_RELATIONS.items():
            actual = actual_release_relations[name]
            if actual[:3] != (relation_kind, columns, options) or not actual[3] or actual[3] != actual[4]:
                failures.append(f"relation:bddk_meta.{name}")

    release_constraint_rows = await pool.fetch(
        _CORPUS_RELEASE_CONSTRAINTS_SQL,
        ["corpus_release_activations", "corpus_releases"],
    )
    actual_release_constraints = {
        (str(_value(row, "relname")), str(_value(row, "conname"))): (
            _catalog_char(_value(row, "contype")),
            bool(_value(row, "convalidated", False)),
            _normalize_sql(_value(row, "definition")),
        )
        for row in release_constraint_rows
    }
    expected_release_constraints = {
        key: (constraint_type, True, _normalize_sql(definition))
        for key, (constraint_type, definition) in _CORPUS_RELEASE_CONSTRAINTS.items()
    }
    if actual_release_constraints != expected_release_constraints:
        failures.append("constraints:bddk_meta.corpus_release_exact")

    release_trigger_rows = await pool.fetch(
        _CORPUS_RELEASE_TRIGGERS_SQL,
        ["corpus_release_activations", "corpus_releases"],
    )
    actual_release_triggers = {
        (str(_value(row, "relname")), str(_value(row, "tgname"))): (
            str(_value(row, "function_identity")),
            int(_value(row, "tgtype", -1)),
            _catalog_char(_value(row, "tgenabled")),
        )
        for row in release_trigger_rows
    }
    expected_release_triggers = {
        key: (function_identity, trigger_type, "O")
        for key, (function_identity, trigger_type) in _CORPUS_RELEASE_TRIGGERS.items()
    }
    if actual_release_triggers != expected_release_triggers:
        failures.append("triggers:bddk_meta.corpus_release_exact")

    release_routine_rows = await pool.fetch(
        _CORPUS_RELEASE_ROUTINES_SQL,
        sorted(identity.partition("(")[0] for identity in _CORPUS_RELEASE_ROUTINES),
    )
    actual_release_routines = {
        str(_value(row, "function_identity")): (
            str(_value(row, "language")),
            _catalog_char(_value(row, "provolatile")),
            _catalog_char(_value(row, "proparallel")),
            bool(_value(row, "prosecdef", False)),
            bool(_value(row, "proleakproof", True)),
            tuple(str(item) for item in (_value(row, "configuration", ()) or ())),
            _normalize_sql(_value(row, "source")),
            str(_value(row, "owner_name", "")),
            str(_value(row, "ledger_owner_name", "")),
            bool(_value(row, "public_can_execute", True)),
        )
        for row in release_routine_rows
    }
    if set(actual_release_routines) != set(_CORPUS_RELEASE_ROUTINES):
        failures.append("routines:bddk_meta.corpus_release_exact")
    else:
        for identity, (language, volatility, parallel, security_definer, source) in _CORPUS_RELEASE_ROUTINES.items():
            actual = actual_release_routines[identity]
            expected_prefix = (
                language,
                volatility,
                parallel,
                security_definer,
                False,
                ("search_path=pg_catalog",),
                _normalize_sql(source),
            )
            if actual[:7] != expected_prefix or not actual[7] or actual[7] != actual[8] or actual[9]:
                failures.append(f"routine:bddk_meta.{identity}")

    active_release_view = await pool.fetchrow(_ACTIVE_CORPUS_RELEASE_VIEW_SQL)
    active_release_definition = _normalize_view_sql(_value(active_release_view, "definition"))
    if not (
        active_release_view is not None
        and tuple(str(item) for item in (_value(active_release_view, "dependencies", ()) or ()))
        == _ACTIVE_CORPUS_RELEASE_DEPENDENCIES
        and all(
            _normalize_view_sql(fragment) in active_release_definition
            for fragment in _ACTIVE_CORPUS_RELEASE_REQUIRED_DEFINITION
        )
        and " union " not in active_release_definition
        and " or " not in active_release_definition
    ):
        failures.append("view:bddk_meta.active_corpus_release")

    legal_status_routine = await pool.fetchrow(_LEGAL_STATUS_ROUTINE_SQL)
    if not (
        legal_status_routine is not None
        and str(_value(legal_status_routine, "function_identity")) == "resolve_regulation_status(text, date)"
        and str(_value(legal_status_routine, "language")) == "sql"
        and _catalog_char(_value(legal_status_routine, "provolatile")) == "s"
        and _catalog_char(_value(legal_status_routine, "proparallel")) == "s"
        and bool(_value(legal_status_routine, "prosecdef", False))
        and not bool(_value(legal_status_routine, "proleakproof", True))
        and bool(_value(legal_status_routine, "proisstrict", False))
        and bool(_value(legal_status_routine, "proretset", False))
        and _normalize_sql(_value(legal_status_routine, "result_type")) == _normalize_sql(_LEGAL_STATUS_RESULT_TYPE)
        and tuple(str(item) for item in (_value(legal_status_routine, "configuration", ()) or ()))
        == ("search_path=pg_catalog",)
        and _normalize_sql(_value(legal_status_routine, "source")) == _normalize_sql(_v6_legal_status_function_source())
        and bool(str(_value(legal_status_routine, "owner_name", "")))
        and str(_value(legal_status_routine, "owner_name"))
        == str(_value(legal_status_routine, "ledger_owner_name", ""))
        and not bool(_value(legal_status_routine, "public_can_execute", True))
    ):
        failures.append("routine:bddk_meta.resolve_regulation_status(text, date)")

    return CatalogIntegrity(tuple(sorted(failures)))
