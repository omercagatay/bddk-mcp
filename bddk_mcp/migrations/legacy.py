"""Fail-closed adoption of the final pre-ledger PostgreSQL schema.

This module intentionally supports one legacy shape only: the schema emitted
by the last ad-hoc initializers immediately before global migration v0001 was
introduced.  Inspection reads PostgreSQL catalogs only.  The adoption runner
may subsequently normalize a small, enumerated set of metadata differences;
it never updates, deletes, or rewrites corpus rows.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Final

from bddk_mcp.migrations.v0001_core import V0001_CORE

LEGACY_VERIFIER_VERSION: Final[str] = "pre-ledger-v1-catalog-verifier/1"
LEGACY_SOURCE_KIND: Final[str] = "bddk-mcp-pre-ledger-initializers"


class LegacyAdoptionError(RuntimeError):
    """A sanitized refusal to adopt an unmanaged legacy schema."""


@dataclass(frozen=True, slots=True)
class ColumnSpec:
    name: str
    data_type: str
    not_null: bool = False
    default: str | None = None
    identity: str = ""


@dataclass(frozen=True, slots=True)
class IndexSpec:
    table: str
    name: str
    method: str
    keys: tuple[str, ...]
    opclasses: tuple[str, ...]
    unique: bool = False
    primary: bool = False
    options: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LegacyInspection:
    """Validated structural snapshot containing no corpus values."""

    fingerprint: str
    serial_columns: tuple[str, ...]
    legacy_constraints: tuple[str, ...]
    legacy_functions: tuple[str, ...]
    manifest: dict[str, Any]

    @property
    def normalizations(self) -> tuple[str, ...]:
        return self.serial_columns + self.legacy_constraints + self.legacy_functions


def _c(
    name: str,
    data_type: str = "text",
    *,
    not_null: bool = False,
    default: str | None = None,
    identity: str = "",
) -> ColumnSpec:
    return ColumnSpec(name, data_type, not_null, default, identity)


_TABLE_COLUMNS: Final[dict[str, tuple[ColumnSpec, ...]]] = {
    "decision_cache": (
        _c("document_id", not_null=True),
        _c("title", not_null=True, default="''::text"),
        _c("content", default="''::text"),
        _c("decision_date", default="''::text"),
        _c("decision_number", default="''::text"),
        _c("category", default="''::text"),
        _c("source_url", default="''::text"),
        _c("cached_at", "double precision", not_null=True),
    ),
    "document_chunks": (
        _c("id", "integer", not_null=True, identity="d"),
        _c("doc_id", not_null=True),
        _c("chunk_index", "integer", not_null=True),
        _c("title", default="''::text"),
        _c("category", default="''::text"),
        _c("decision_date", default="''::text"),
        _c("decision_number", default="''::text"),
        _c("source_url", default="''::text"),
        _c("total_chunks", "integer", default="1"),
        _c("total_pages", "integer", default="1"),
        _c("content_hash", default="''::text"),
        _c("chunk_start_char", "integer"),
        _c("chunk_end_char", "integer"),
        _c("section_type", default="''::text"),
        _c("section_ref", default="''::text"),
        _c("section_start_char", "integer"),
        _c("section_end_char", "integer"),
        _c("section_content_hash", default="''::text"),
        _c("chunk_text", not_null=True),
        _c("embedding", "vector(768)"),
        _c("tsv", "tsvector"),
    ),
    "document_sections": (
        _c("id", "integer", not_null=True, identity="d"),
        _c("doc_id", not_null=True),
        _c("section_type", not_null=True),
        _c("section_ref", not_null=True),
        _c("heading", default="''::text"),
        _c("start_char", "integer", not_null=True),
        _c("end_char", "integer", not_null=True),
        _c("content", not_null=True),
        _c("content_hash", not_null=True),
        _c("page_start", "integer"),
        _c("page_end", "integer"),
        _c("tsv", "tsvector"),
    ),
    "document_versions": (
        _c("id", "integer", not_null=True, identity="d"),
        _c("document_id", not_null=True),
        _c("version", "integer", not_null=True, default="1"),
        _c("content_hash", not_null=True),
        _c("markdown_content", default="''::text"),
        _c("synced_at", "double precision", not_null=True),
    ),
    "documents": (
        _c("document_id", not_null=True),
        _c("title", not_null=True),
        _c("category", default="''::text"),
        _c("decision_date", default="''::text"),
        _c("decision_number", default="''::text"),
        _c("source_url", default="''::text"),
        _c("pdf_blob", "bytea"),
        _c("markdown_content", default="''::text"),
        _c("content_hash", default="''::text"),
        _c("downloaded_at", "double precision"),
        _c("extracted_at", "double precision"),
        _c("extraction_method", default="'markitdown'::text"),
        _c("total_pages", "integer", default="1"),
        _c("file_size", "integer", default="0"),
        _c("tsv", "tsvector"),
    ),
    "sync_failures": (
        _c("document_id", not_null=True),
        _c("error", not_null=True),
        _c("error_category", not_null=True, default="'unknown'::text"),
        _c("source_url", default="''::text"),
        _c("retryable", "boolean", default="true"),
        _c("attempts", "integer", default="1"),
        _c("first_failed_at", "double precision"),
        _c("last_failed_at", "double precision"),
    ),
    "sync_metadata": (
        _c("document_id", not_null=True),
        _c("etag", default="''::text"),
        _c("last_modified", default="''::text"),
        _c("last_sync_at", "double precision"),
        _c("sync_count", "integer", default="0"),
    ),
    "tool_call_traces": (
        _c("id", "bigint", not_null=True, identity="d"),
        _c("created_at", "timestamp with time zone", default="now()"),
        _c("tool_name", not_null=True),
        _c("args_hash", not_null=True),
        _c("args_summary", "jsonb"),
        _c("latency_ms", "integer"),
        _c("result_count", "integer"),
        _c("doc_ids", "text[]"),
        _c("quality_labels", "jsonb"),
        _c("relevance_stats", "jsonb"),
        _c("model_id"),
        _c("session_id"),
    ),
}

_SERIAL_COLUMNS: Final[dict[str, tuple[str, str]]] = {
    "document_chunks.id": ("public.document_chunks_id_seq", "integer"),
    "document_sections.id": ("public.document_sections_id_seq", "integer"),
    "document_versions.id": ("public.document_versions_id_seq", "integer"),
    "tool_call_traces.id": ("public.tool_call_traces_id_seq", "bigint"),
}

_CONSTRAINTS: Final[dict[tuple[str, str], tuple[str, str]]] = {
    ("decision_cache", "decision_cache_pkey"): ("p", "PRIMARY KEY (document_id)"),
    ("document_chunks", "document_chunks_document_index_uq"): ("u", "UNIQUE (doc_id, chunk_index)"),
    ("document_chunks", "document_chunks_pkey"): ("p", "PRIMARY KEY (id)"),
    ("document_sections", "document_sections_identity_uq"): (
        "u",
        "UNIQUE (doc_id, section_type, section_ref, content_hash)",
    ),
    ("document_sections", "document_sections_pkey"): ("p", "PRIMARY KEY (id)"),
    ("document_versions", "document_versions_document_version_uq"): (
        "u",
        "UNIQUE (document_id, version)",
    ),
    ("document_versions", "document_versions_pkey"): ("p", "PRIMARY KEY (id)"),
    ("documents", "documents_pkey"): ("p", "PRIMARY KEY (document_id)"),
    ("sync_failures", "sync_failures_pkey"): ("p", "PRIMARY KEY (document_id)"),
    ("sync_metadata", "sync_metadata_pkey"): ("p", "PRIMARY KEY (document_id)"),
    ("tool_call_traces", "tool_call_traces_pkey"): ("p", "PRIMARY KEY (id)"),
}

_LEGACY_CONSTRAINT_NAMES: Final[dict[tuple[str, str], str]] = {
    ("document_chunks", "document_chunks_doc_id_chunk_index_key"): "document_chunks_document_index_uq",
    (
        "document_sections",
        "document_sections_doc_id_section_type_section_ref_content_hash_",
    ): "document_sections_identity_uq",
    ("document_versions", "document_versions_document_id_version_key"): "document_versions_document_version_uq",
}


def _idx(
    table: str,
    name: str,
    method: str,
    keys: tuple[str, ...],
    opclasses: tuple[str, ...],
    *,
    unique: bool = False,
    primary: bool = False,
    options: tuple[str, ...] = (),
) -> IndexSpec:
    return IndexSpec(table, name, method, keys, opclasses, unique, primary, options)


_INDEXES: Final[tuple[IndexSpec, ...]] = (
    _idx("decision_cache", "decision_cache_pkey", "btree", ("document_id",), ("text_ops",), unique=True, primary=True),
    _idx("decision_cache", "idx_decision_cache_category", "btree", ("category",), ("text_ops",)),
    _idx(
        "document_chunks",
        "document_chunks_document_index_uq",
        "btree",
        ("doc_id", "chunk_index"),
        ("text_ops", "int4_ops"),
        unique=True,
    ),
    _idx("document_chunks", "document_chunks_pkey", "btree", ("id",), ("int4_ops",), unique=True, primary=True),
    _idx("document_chunks", "idx_chunks_doc_id", "btree", ("doc_id",), ("text_ops",)),
    _idx(
        "document_chunks",
        "idx_chunks_embedding_hnsw",
        "hnsw",
        ("embedding",),
        ("vector_cosine_ops",),
        options=("ef_construction=64", "m=16"),
    ),
    _idx(
        "document_chunks",
        "idx_chunks_section_ref",
        "btree",
        ("section_type", "section_ref"),
        ("text_ops", "text_ops"),
    ),
    _idx("document_chunks", "idx_chunks_tsv", "gin", ("tsv",), ("tsvector_ops",)),
    _idx(
        "document_sections",
        "document_sections_identity_uq",
        "btree",
        ("doc_id", "section_type", "section_ref", "content_hash"),
        ("text_ops", "text_ops", "text_ops", "text_ops"),
        unique=True,
    ),
    _idx("document_sections", "document_sections_pkey", "btree", ("id",), ("int4_ops",), unique=True, primary=True),
    _idx("document_sections", "idx_document_sections_doc_id", "btree", ("doc_id",), ("text_ops",)),
    _idx(
        "document_sections",
        "idx_document_sections_ref",
        "btree",
        ("section_type", "section_ref"),
        ("text_ops", "text_ops"),
    ),
    _idx("document_sections", "idx_document_sections_tsv", "gin", ("tsv",), ("tsvector_ops",)),
    _idx(
        "document_versions",
        "document_versions_document_version_uq",
        "btree",
        ("document_id", "version"),
        ("text_ops", "int4_ops"),
        unique=True,
    ),
    _idx("document_versions", "document_versions_pkey", "btree", ("id",), ("int4_ops",), unique=True, primary=True),
    _idx("document_versions", "idx_versions_doc_id", "btree", ("document_id",), ("text_ops",)),
    _idx("documents", "documents_pkey", "btree", ("document_id",), ("text_ops",), unique=True, primary=True),
    _idx("documents", "idx_documents_category", "btree", ("category",), ("text_ops",)),
    _idx("documents", "idx_documents_date", "btree", ("decision_date",), ("text_ops",)),
    _idx("documents", "idx_documents_tsv", "gin", ("tsv",), ("tsvector_ops",)),
    _idx("sync_failures", "sync_failures_pkey", "btree", ("document_id",), ("text_ops",), unique=True, primary=True),
    _idx("sync_metadata", "sync_metadata_pkey", "btree", ("document_id",), ("text_ops",), unique=True, primary=True),
    _idx("tool_call_traces", "idx_tool_call_traces_created_at", "btree", ("created_at",), ("timestamptz_ops",)),
    _idx("tool_call_traces", "idx_tool_call_traces_doc_ids", "gin", ("doc_ids",), ("array_ops",)),
    _idx("tool_call_traces", "idx_tool_call_traces_tool_name", "btree", ("tool_name",), ("text_ops",)),
    _idx("tool_call_traces", "tool_call_traces_pkey", "btree", ("id",), ("int8_ops",), unique=True, primary=True),
)

_LEGACY_INDEX_NAMES: Final[dict[tuple[str, str], str]] = {
    (table, legacy): canonical for (table, legacy), canonical in _LEGACY_CONSTRAINT_NAMES.items()
}

_CANONICAL_FUNCTION_SOURCES: Final[dict[str, str]] = {
    "chunks_tsv_trigger()": """
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
    "document_sections_tsv_trigger()": """
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
    "documents_tsv_trigger()": """
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
    "immutable_unaccent(text)": "SELECT public.unaccent($1)",
}

_LEGACY_FUNCTION_SOURCES: Final[dict[str, str]] = {
    "chunks_tsv_trigger()": """
        BEGIN
            NEW.tsv := to_tsvector('simple', immutable_unaccent(coalesce(NEW.title, '')))
                    || to_tsvector('simple', immutable_unaccent(coalesce(NEW.chunk_text, '')));
            RETURN NEW;
        END;
    """,
    "document_sections_tsv_trigger()": """
        BEGIN
            NEW.tsv :=
                to_tsvector('simple', immutable_unaccent(coalesce(NEW.heading, '')))
                || to_tsvector('simple', immutable_unaccent(coalesce(NEW.content, '')));
            RETURN NEW;
        END;
    """,
    "documents_tsv_trigger()": """
        BEGIN
            NEW.tsv :=
                to_tsvector('simple', immutable_unaccent(coalesce(NEW.title, '')))
                || to_tsvector('simple', immutable_unaccent(coalesce(NEW.markdown_content, '')))
                || to_tsvector('simple', immutable_unaccent(coalesce(NEW.category, '')));
            RETURN NEW;
        END;
    """,
    "immutable_unaccent(text)": "SELECT unaccent($1);",
}

_FUNCTION_METADATA: Final[dict[str, tuple[str, str, str, str]]] = {
    "chunks_tsv_trigger()": ("trigger", "plpgsql", "v", "u"),
    "document_sections_tsv_trigger()": ("trigger", "plpgsql", "v", "u"),
    "documents_tsv_trigger()": ("trigger", "plpgsql", "v", "u"),
    "immutable_unaccent(text)": ("text", "sql", "i", "s"),
}

_TRIGGERS: Final[dict[tuple[str, str], str]] = {
    ("document_chunks", "chunks_tsv_update"): "chunks_tsv_trigger()",
    ("document_sections", "trg_document_sections_tsv"): "document_sections_tsv_trigger()",
    ("documents", "trg_documents_tsv"): "documents_tsv_trigger()",
}

_RELATIONS_SQL = """
SELECT relation.relname,
       relation.relkind,
       relation.relpersistence,
       relation.relrowsecurity,
       relation.relforcerowsecurity,
       relation.relreplident,
       relation.relispartition,
       relation.reloptions IS NULL AS default_relation_options,
       relation.relacl IS NULL AS no_relation_acl,
       relation.reltablespace = 0 AS default_tablespace,
       NOT EXISTS (
           SELECT 1
           FROM pg_catalog.pg_attribute AS dropped_attribute
           WHERE dropped_attribute.attrelid = relation.oid
             AND dropped_attribute.attnum > 0
             AND dropped_attribute.attisdropped
       ) AS no_dropped_columns,
       relation.relam = (
           SELECT access_method.oid
           FROM pg_catalog.pg_am AS access_method
           WHERE access_method.amname = 'heap' AND access_method.amtype = 't'
       ) AS default_access_method,
       relation.relowner = (SELECT oid FROM pg_catalog.pg_roles WHERE rolname = CURRENT_USER) AS owned_by_current_user
FROM pg_catalog.pg_class AS relation
JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
WHERE namespace.nspname = 'public'
  AND relation.relname = ANY($1::pg_catalog.text[])
ORDER BY relation.relname
"""

_COLUMNS_SQL = """
SELECT relation.relname AS table_name,
       attribute.attnum,
       attribute.attname,
       pg_catalog.format_type(attribute.atttypid, attribute.atttypmod) AS data_type,
       attribute.attnotnull,
       attribute.attidentity,
       attribute.attgenerated,
       pg_catalog.pg_get_expr(default_value.adbin, default_value.adrelid) AS default_expression,
       attribute.attinhcount,
       attribute.attislocal,
       COALESCE(attribute.attstattarget = -1, true) AS default_statistics_target,
       attribute.atthasmissing,
       attribute.attcollation = type_record.typcollation AS default_collation,
       attribute.attstorage = type_record.typstorage AS default_storage,
       attribute.attalign = type_record.typalign AS default_alignment,
       attribute.attbyval = type_record.typbyval AS default_pass_by_value,
       attribute.attacl IS NULL AS no_column_acl,
       attribute.attoptions IS NULL AS no_attribute_options,
       attribute.attfdwoptions IS NULL AS no_fdw_options
FROM pg_catalog.pg_class AS relation
JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
JOIN pg_catalog.pg_attribute AS attribute ON attribute.attrelid = relation.oid
JOIN pg_catalog.pg_type AS type_record ON type_record.oid = attribute.atttypid
LEFT JOIN pg_catalog.pg_attrdef AS default_value
  ON default_value.adrelid = relation.oid
 AND default_value.adnum = attribute.attnum
WHERE namespace.nspname = 'public'
  AND relation.relname = ANY($1::pg_catalog.text[])
  AND attribute.attnum > 0
  AND NOT attribute.attisdropped
ORDER BY relation.relname, attribute.attnum
"""

_CONSTRAINTS_SQL = """
SELECT relation.relname AS table_name,
       constraint_record.conname,
       constraint_record.contype,
       pg_catalog.pg_get_constraintdef(constraint_record.oid, true) AS definition,
       constraint_record.condeferrable,
       constraint_record.condeferred,
       constraint_record.convalidated,
       constraint_record.connoinherit,
       constraint_record.conislocal,
       constraint_record.coninhcount
FROM pg_catalog.pg_constraint AS constraint_record
JOIN pg_catalog.pg_class AS relation ON relation.oid = constraint_record.conrelid
JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
WHERE namespace.nspname = 'public'
  AND relation.relname = ANY($1::pg_catalog.text[])
ORDER BY relation.relname, constraint_record.conname
"""

_INDEXES_SQL = """
SELECT table_relation.relname AS table_name,
       index_relation.relname AS index_name,
       access_method.amname AS method,
       index_record.indisunique,
       index_record.indisprimary,
       index_record.indisvalid,
       index_record.indisready,
       index_record.indislive,
       index_record.indisclustered,
       index_record.indisreplident,
       index_record.indcheckxmin,
       index_record.indisexclusion,
       index_record.indimmediate,
       index_record.indnkeyatts,
       index_record.indnatts,
       COALESCE(
           (pg_catalog.to_jsonb(index_record) ->> 'indnullsnotdistinct')::pg_catalog.bool,
           false
       ) AS indnullsnotdistinct,
       index_relation.reltablespace = 0 AS default_tablespace,
       index_relation.relpersistence = 'p' AS persistent,
       index_relation.relowner = (SELECT oid FROM pg_catalog.pg_roles WHERE rolname = CURRENT_USER)
           AS owned_by_current_user,
       ARRAY(
           SELECT pg_catalog.pg_get_indexdef(index_record.indexrelid, position, true)
           FROM pg_catalog.generate_series(1, index_record.indnkeyatts) AS position
           ORDER BY position
       ) AS keys,
       ARRAY(
           SELECT operator_class.opcname
           FROM unnest(index_record.indclass::pg_catalog.oid[]) WITH ORDINALITY AS selected(oid, position)
           JOIN pg_catalog.pg_opclass AS operator_class ON operator_class.oid = selected.oid
           WHERE selected.position <= index_record.indnkeyatts
           ORDER BY selected.position
       ) AS opclasses,
       index_record.indoption::pg_catalog.int2[] AS index_options,
       pg_catalog.pg_get_expr(index_record.indpred, index_record.indrelid) AS predicate,
       ARRAY(SELECT unnest(COALESCE(index_relation.reloptions, ARRAY[]::pg_catalog.text[])) ORDER BY 1) AS options
FROM pg_catalog.pg_index AS index_record
JOIN pg_catalog.pg_class AS index_relation ON index_relation.oid = index_record.indexrelid
JOIN pg_catalog.pg_class AS table_relation ON table_relation.oid = index_record.indrelid
JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = table_relation.relnamespace
JOIN pg_catalog.pg_am AS access_method ON access_method.oid = index_relation.relam
WHERE namespace.nspname = 'public'
  AND table_relation.relname = ANY($1::pg_catalog.text[])
ORDER BY table_relation.relname, index_relation.relname
"""

_FUNCTIONS_SQL = """
SELECT function_record.proname || '(' || pg_catalog.pg_get_function_identity_arguments(function_record.oid) || ')' AS identity,
       pg_catalog.pg_get_function_result(function_record.oid) AS result_type,
       language.lanname AS language,
       function_record.provolatile,
       function_record.proparallel,
       function_record.prosecdef,
       function_record.proleakproof,
       function_record.proisstrict,
       function_record.proretset,
       function_record.prokind,
       function_record.procost,
       function_record.prorows,
       function_record.prosupport = 0 AS no_support_function,
       function_record.proacl IS NULL AS no_function_acl,
       function_record.proconfig,
       function_record.prosrc,
       function_record.proowner = (SELECT oid FROM pg_catalog.pg_roles WHERE rolname = CURRENT_USER) AS owned_by_current_user
FROM pg_catalog.pg_proc AS function_record
JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = function_record.pronamespace
JOIN pg_catalog.pg_language AS language ON language.oid = function_record.prolang
WHERE namespace.nspname = 'public'
  AND function_record.proname = ANY($1::pg_catalog.text[])
ORDER BY identity
"""

_TRIGGERS_SQL = """
SELECT relation.relname AS table_name,
       trigger_record.tgname,
       trigger_record.tgtype,
       trigger_record.tgenabled,
       function_record.proname || '(' || pg_catalog.pg_get_function_identity_arguments(function_record.oid) || ')' AS function_identity,
       trigger_record.tgargs = ''::pg_catalog.bytea AS empty_arguments,
       trigger_record.tgconstraint = 0 AS not_constraint_trigger,
       NOT trigger_record.tgdeferrable AS not_deferrable,
       NOT trigger_record.tginitdeferred AS not_initially_deferred,
       trigger_record.tgqual IS NULL AS no_when_clause,
       trigger_record.tgoldtable IS NULL AS no_old_transition_table,
       trigger_record.tgnewtable IS NULL AS no_new_transition_table
FROM pg_catalog.pg_trigger AS trigger_record
JOIN pg_catalog.pg_class AS relation ON relation.oid = trigger_record.tgrelid
JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
JOIN pg_catalog.pg_proc AS function_record ON function_record.oid = trigger_record.tgfoid
WHERE namespace.nspname = 'public'
  AND relation.relname = ANY($1::pg_catalog.text[])
  AND NOT trigger_record.tgisinternal
ORDER BY relation.relname, trigger_record.tgname
"""

_SEQUENCES_SQL = """
SELECT sequence_relation.relname AS sequence_name,
       pg_catalog.format_type(sequence_record.seqtypid, NULL) AS data_type,
       sequence_record.seqstart,
       sequence_record.seqincrement,
       sequence_record.seqmax,
       sequence_record.seqmin,
       sequence_record.seqcache,
       sequence_record.seqcycle,
       table_relation.relname AS owned_table,
       attribute.attname AS owned_column,
       dependency.deptype,
       sequence_relation.relacl IS NULL AS no_sequence_acl,
       sequence_relation.relowner = (SELECT oid FROM pg_catalog.pg_roles WHERE rolname = CURRENT_USER) AS owned_by_current_user
FROM pg_catalog.pg_class AS sequence_relation
JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = sequence_relation.relnamespace
JOIN pg_catalog.pg_sequence AS sequence_record ON sequence_record.seqrelid = sequence_relation.oid
LEFT JOIN pg_catalog.pg_depend AS dependency
  ON dependency.classid = 'pg_catalog.pg_class'::pg_catalog.regclass
 AND dependency.objid = sequence_relation.oid
 AND dependency.objsubid = 0
 AND dependency.refclassid = 'pg_catalog.pg_class'::pg_catalog.regclass
 AND dependency.deptype IN ('a', 'i')
LEFT JOIN pg_catalog.pg_class AS table_relation ON table_relation.oid = dependency.refobjid
LEFT JOIN pg_catalog.pg_attribute AS attribute
  ON attribute.attrelid = dependency.refobjid
 AND attribute.attnum = dependency.refobjsubid
WHERE namespace.nspname = 'public'
  AND sequence_relation.relname = ANY($1::pg_catalog.text[])
ORDER BY sequence_relation.relname
"""

_RULES_SQL = """
SELECT relation.relname AS table_name, rewrite_rule.rulename
FROM pg_catalog.pg_rewrite AS rewrite_rule
JOIN pg_catalog.pg_class AS relation ON relation.oid = rewrite_rule.ev_class
JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
WHERE namespace.nspname = 'public'
  AND relation.relname = ANY($1::pg_catalog.text[])
  AND rewrite_rule.rulename <> '_RETURN'
ORDER BY relation.relname, rewrite_rule.rulename
"""

_PUBLICATION_MEMBERSHIP_SQL = """
SELECT publication.pubname, relation.relname AS table_name
FROM pg_catalog.pg_publication_rel AS membership
JOIN pg_catalog.pg_publication AS publication ON publication.oid = membership.prpubid
JOIN pg_catalog.pg_class AS relation ON relation.oid = membership.prrelid
JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
WHERE namespace.nspname = 'public'
  AND relation.relname = ANY($1::pg_catalog.text[])
ORDER BY publication.pubname, relation.relname
"""

_EXTENSIONS_SQL = """
SELECT extension.extname, namespace.nspname AS extension_schema
FROM pg_catalog.pg_extension AS extension
JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = extension.extnamespace
WHERE extension.extname = ANY($1::pg_catalog.text[])
ORDER BY extension.extname
"""

_IDENTITY_SQL = """
SELECT CURRENT_USER AS current_user,
       pg_catalog.has_schema_privilege(CURRENT_USER, 'public', 'USAGE') AS has_usage,
       pg_catalog.has_schema_privilege(CURRENT_USER, 'public', 'CREATE') AS has_create,
       pg_catalog.has_database_privilege(CURRENT_USER, CURRENT_DATABASE(), 'CREATE') AS has_database_create,
       NOT role_record.rolsuper
           AND NOT role_record.rolcreatedb
           AND NOT role_record.rolcreaterole
           AND NOT role_record.rolreplication
           AND NOT role_record.rolbypassrls AS restricted_schema_owner,
       pg_catalog.to_regnamespace('bddk_meta') IS NULL AS meta_schema_absent,
       pg_catalog.to_regnamespace('bddk_operator') IS NULL AS operator_schema_absent
FROM pg_catalog.pg_roles AS role_record
WHERE role_record.rolname = CURRENT_USER
"""

_APPLICATION_SCHEMAS_SQL = """
SELECT namespace.nspname
FROM pg_catalog.pg_namespace AS namespace
WHERE namespace.nspname <> 'information_schema'
  AND namespace.nspname !~ '^pg_'
ORDER BY namespace.nspname
"""

_APPLICATION_RELATIONS_SQL = """
SELECT relation.relname, relation.relkind
FROM pg_catalog.pg_class AS relation
JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = relation.relnamespace
WHERE namespace.nspname = 'public'
  AND relation.relkind IN ('r', 'p', 'v', 'm', 'S', 'f')
  AND NOT EXISTS (
      SELECT 1
      FROM pg_catalog.pg_depend AS dependency
      WHERE dependency.classid = 'pg_catalog.pg_class'::pg_catalog.regclass
        AND dependency.objid = relation.oid
        AND dependency.deptype = 'e'
  )
ORDER BY relation.relkind, relation.relname
"""

_APPLICATION_FUNCTIONS_SQL = """
SELECT function_record.proname || '('
       || pg_catalog.pg_get_function_identity_arguments(function_record.oid)
       || ')' AS identity
FROM pg_catalog.pg_proc AS function_record
JOIN pg_catalog.pg_namespace AS namespace ON namespace.oid = function_record.pronamespace
WHERE namespace.nspname = 'public'
  AND NOT EXISTS (
      SELECT 1
      FROM pg_catalog.pg_depend AS dependency
      WHERE dependency.classid = 'pg_catalog.pg_proc'::pg_catalog.regclass
        AND dependency.objid = function_record.oid
        AND dependency.deptype = 'e'
  )
ORDER BY identity
"""

_LOCK_MANAGED_TABLES_SQL = """
LOCK TABLE
    public.decision_cache,
    public.document_chunks,
    public.document_sections,
    public.document_versions,
    public.documents,
    public.sync_failures,
    public.sync_metadata,
    public.tool_call_traces
IN ACCESS EXCLUSIVE MODE
"""


def _normalize_sql(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value.strip()).lower().rstrip(";")


def _value(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def _catalog_char(value: Any) -> str:
    """Decode PostgreSQL's internal ``\"char\"`` type as returned by asyncpg."""

    if isinstance(value, bytes):
        if value in {b"", b"\x00"}:
            return ""
        return value.decode("ascii")
    return str(value or "")


def _fingerprint(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _adoption_refusal(categories: set[str]) -> LegacyAdoptionError:
    shown = ", ".join(sorted(categories)[:8])
    suffix = "" if len(categories) <= 8 else f" (+{len(categories) - 8} more)"
    return LegacyAdoptionError(
        "Legacy adoption refused because the unmanaged schema is not the exact supported pre-ledger shape "
        f"({shown}{suffix}). No corpus rows were changed. Take a verified database backup and follow the "
        "blue-green data-only procedure in docs/LEGACY_DATABASE_UPGRADE.md; do not restore legacy DDL over a "
        "managed database."
    )


async def inspect_legacy_v1(connection: Any, *, allow_known_legacy: bool) -> LegacyInspection:
    """Verify the complete v0001 catalog shape using SELECT statements only."""

    mismatches: set[str] = set()
    serial_columns: list[str] = []
    legacy_constraints: list[str] = []
    legacy_functions: list[str] = []
    table_names = sorted(_TABLE_COLUMNS)

    identity = await connection.fetchrow(_IDENTITY_SQL)
    if (
        not bool(_value(identity, "has_usage"))
        or not bool(_value(identity, "has_create"))
        or not bool(_value(identity, "has_database_create"))
        or not bool(_value(identity, "restricted_schema_owner"))
    ):
        mismatches.add("schema-owner-identity")
    if not bool(_value(identity, "meta_schema_absent")):
        mismatches.add("preexisting-bddk-meta")
    if not bool(_value(identity, "operator_schema_absent")):
        mismatches.add("preexisting-bddk-operator")

    schema_rows = await connection.fetch(_APPLICATION_SCHEMAS_SQL)
    application_schemas = tuple(str(_value(row, "nspname")) for row in schema_rows)
    if application_schemas != ("public",):
        mismatches.add("unexpected-application-schemas")

    application_relation_rows = await connection.fetch(_APPLICATION_RELATIONS_SQL)
    application_relations = tuple(
        (str(_value(row, "relname")), _catalog_char(_value(row, "relkind"))) for row in application_relation_rows
    )
    expected_application_relations = tuple(
        sorted(
            [(name, "r") for name in _TABLE_COLUMNS]
            + [(value[0].partition(".")[2], "S") for value in _SERIAL_COLUMNS.values()],
            key=lambda value: (value[1], value[0]),
        )
    )
    if application_relations != expected_application_relations:
        mismatches.add("unexpected-application-relations")

    application_function_rows = await connection.fetch(_APPLICATION_FUNCTIONS_SQL)
    application_functions = tuple(str(_value(row, "identity")) for row in application_function_rows)
    if application_functions != tuple(sorted(_FUNCTION_METADATA)):
        mismatches.add("unexpected-application-functions")

    extension_rows = await connection.fetch(_EXTENSIONS_SQL, ["unaccent", "vector"])
    extensions = {str(_value(row, "extname")): str(_value(row, "extension_schema")) for row in extension_rows}
    if extensions != {"unaccent": "public", "vector": "public"}:
        mismatches.add("extension-placement")

    relation_rows = await connection.fetch(_RELATIONS_SQL, table_names)
    relations: dict[str, dict[str, Any]] = {}
    for row in relation_rows:
        name = str(_value(row, "relname"))
        relations[name] = {
            "kind": _catalog_char(_value(row, "relkind")),
            "persistence": _catalog_char(_value(row, "relpersistence")),
            "row_security": bool(_value(row, "relrowsecurity")),
            "force_row_security": bool(_value(row, "relforcerowsecurity")),
            "replica_identity": _catalog_char(_value(row, "relreplident")),
            "partition": bool(_value(row, "relispartition")),
            "default_relation_options": bool(_value(row, "default_relation_options")),
            "no_relation_acl": bool(_value(row, "no_relation_acl")),
            "default_tablespace": bool(_value(row, "default_tablespace")),
            "no_dropped_columns": bool(_value(row, "no_dropped_columns")),
            "default_access_method": bool(_value(row, "default_access_method")),
            "owned_by_current_user": bool(_value(row, "owned_by_current_user")),
        }
    expected_relation = {
        "kind": "r",
        "persistence": "p",
        "row_security": False,
        "force_row_security": False,
        "replica_identity": "d",
        "partition": False,
        "default_relation_options": True,
        "no_relation_acl": True,
        "default_tablespace": True,
        "no_dropped_columns": True,
        "default_access_method": True,
        "owned_by_current_user": True,
    }
    if set(relations) != set(table_names):
        mismatches.add("relations")
    for name, properties in relations.items():
        if properties != expected_relation:
            mismatches.add(f"relation:{name}")

    column_rows = await connection.fetch(_COLUMNS_SQL, table_names)
    actual_columns: dict[str, list[dict[str, Any]]] = {name: [] for name in table_names}
    for row in column_rows:
        table = str(_value(row, "table_name"))
        if table not in actual_columns:
            mismatches.add("unexpected-column-relation")
            continue
        actual_columns[table].append(
            {
                "position": int(_value(row, "attnum", -1)),
                "name": str(_value(row, "attname")),
                "type": str(_value(row, "data_type")),
                "not_null": bool(_value(row, "attnotnull")),
                "identity": _catalog_char(_value(row, "attidentity", "")),
                "generated": _catalog_char(_value(row, "attgenerated", "")),
                "default": str(_value(row, "default_expression"))
                if _value(row, "default_expression") is not None
                else None,
                "inheritance_count": int(_value(row, "attinhcount", -1)),
                "local": bool(_value(row, "attislocal")),
                "default_statistics_target": bool(_value(row, "default_statistics_target")),
                "has_missing_value": bool(_value(row, "atthasmissing")),
                "default_collation": bool(_value(row, "default_collation")),
                "default_storage": bool(_value(row, "default_storage")),
                "default_alignment": bool(_value(row, "default_alignment")),
                "default_pass_by_value": bool(_value(row, "default_pass_by_value")),
                "no_column_acl": bool(_value(row, "no_column_acl")),
                "no_attribute_options": bool(_value(row, "no_attribute_options")),
                "no_fdw_options": bool(_value(row, "no_fdw_options")),
            }
        )

    for table, expected_columns in _TABLE_COLUMNS.items():
        rows = actual_columns[table]
        if len(rows) != len(expected_columns):
            mismatches.add(f"columns:{table}")
            continue
        for position, (actual, expected) in enumerate(zip(rows, expected_columns, strict=True), start=1):
            key = f"{table}.{expected.name}"
            expected_base = {
                "position": position,
                "name": expected.name,
                "type": expected.data_type,
                "not_null": expected.not_null,
                "identity": expected.identity,
                "generated": "",
                "default": expected.default,
                "inheritance_count": 0,
                "local": True,
                "default_statistics_target": True,
                "has_missing_value": False,
                "default_collation": True,
                "default_storage": True,
                "default_alignment": True,
                "default_pass_by_value": True,
                "no_column_acl": True,
                "no_attribute_options": True,
                "no_fdw_options": True,
            }
            if actual == expected_base:
                continue
            sequence = _SERIAL_COLUMNS.get(key)
            sequence_name = sequence[0].partition(".")[2] if sequence is not None else ""
            serial_default = _normalize_sql(actual["default"])
            has_known_serial_default = serial_default in {
                f"nextval('{sequence_name}'::regclass)",
                f"nextval('public.{sequence_name}'::regclass)",
            }
            legacy_base = {**expected_base, "identity": "", "default": actual["default"]}
            if allow_known_legacy and sequence is not None and has_known_serial_default and actual == legacy_base:
                serial_columns.append(f"serial-to-identity:{key}")
                continue
            mismatches.add(f"column:{key}")

    constraint_rows = await connection.fetch(_CONSTRAINTS_SQL, table_names)
    constraints: dict[tuple[str, str], dict[str, Any]] = {}
    for row in constraint_rows:
        key = (str(_value(row, "table_name")), str(_value(row, "conname")))
        constraints[key] = {
            "type": _catalog_char(_value(row, "contype")),
            "definition": str(_value(row, "definition")),
            "deferrable": bool(_value(row, "condeferrable")),
            "deferred": bool(_value(row, "condeferred")),
            "validated": bool(_value(row, "convalidated")),
            "no_inherit": bool(_value(row, "connoinherit")),
            "local": bool(_value(row, "conislocal")),
            "inheritance_count": int(_value(row, "coninhcount", -1)),
        }
    canonical_constraints: dict[tuple[str, str], dict[str, Any]] = {
        key: {
            "type": value[0],
            "definition": value[1],
            "deferrable": False,
            "deferred": False,
            "validated": True,
            "no_inherit": True,
            "local": True,
            "inheritance_count": 0,
        }
        for key, value in _CONSTRAINTS.items()
    }
    normalized_constraints = dict(constraints)
    if allow_known_legacy:
        for old_key, new_name in _LEGACY_CONSTRAINT_NAMES.items():
            if old_key in normalized_constraints:
                new_key = (old_key[0], new_name)
                if new_key in normalized_constraints:
                    mismatches.add(f"constraint-duplicate:{old_key[0]}")
                else:
                    normalized_constraints[new_key] = normalized_constraints.pop(old_key)
                    legacy_constraints.append(f"rename-constraint:{old_key[0]}.{old_key[1]}->{new_name}")
    if normalized_constraints != canonical_constraints:
        mismatches.add("constraints")

    index_rows = await connection.fetch(_INDEXES_SQL, table_names)
    indexes: dict[tuple[str, str], dict[str, Any]] = {}
    for row in index_rows:
        key = (str(_value(row, "table_name")), str(_value(row, "index_name")))
        indexes[key] = {
            "method": str(_value(row, "method")),
            "unique": bool(_value(row, "indisunique")),
            "primary": bool(_value(row, "indisprimary")),
            "valid": bool(_value(row, "indisvalid")),
            "ready": bool(_value(row, "indisready")),
            "live": bool(_value(row, "indislive")),
            "clustered": bool(_value(row, "indisclustered")),
            "replica_identity": bool(_value(row, "indisreplident")),
            "check_xmin": bool(_value(row, "indcheckxmin")),
            "exclusion": bool(_value(row, "indisexclusion")),
            "immediate": bool(_value(row, "indimmediate")),
            "key_count": int(_value(row, "indnkeyatts", -1)),
            "attribute_count": int(_value(row, "indnatts", -1)),
            "nulls_not_distinct": bool(_value(row, "indnullsnotdistinct")),
            "default_tablespace": bool(_value(row, "default_tablespace")),
            "persistent": bool(_value(row, "persistent")),
            "owned_by_current_user": bool(_value(row, "owned_by_current_user")),
            "keys": tuple(str(item) for item in (_value(row, "keys", []) or [])),
            "opclasses": tuple(str(item) for item in (_value(row, "opclasses", []) or [])),
            "index_options": tuple(int(item) for item in (_value(row, "index_options", []) or [])),
            "predicate": _normalize_sql(_value(row, "predicate")),
            "options": tuple(str(item) for item in (_value(row, "options", []) or [])),
        }
    canonical_indexes = {
        (spec.table, spec.name): {
            "method": spec.method,
            "unique": spec.unique,
            "primary": spec.primary,
            "valid": True,
            "ready": True,
            "live": True,
            "clustered": False,
            "replica_identity": False,
            "check_xmin": False,
            "exclusion": False,
            "immediate": True,
            "key_count": len(spec.keys),
            "attribute_count": len(spec.keys),
            "nulls_not_distinct": False,
            "default_tablespace": True,
            "persistent": True,
            "owned_by_current_user": True,
            "keys": spec.keys,
            "opclasses": spec.opclasses,
            "index_options": tuple(
                3 if spec.table == "tool_call_traces" and spec.name == "idx_tool_call_traces_created_at" else 0
                for _key in spec.keys
            ),
            "predicate": "",
            "options": spec.options,
        }
        for spec in _INDEXES
    }
    normalized_indexes = dict(indexes)
    if allow_known_legacy:
        for old_key, new_name in _LEGACY_INDEX_NAMES.items():
            if old_key in normalized_indexes:
                new_key = (old_key[0], new_name)
                if new_key in normalized_indexes:
                    mismatches.add(f"index-duplicate:{old_key[0]}")
                else:
                    normalized_indexes[new_key] = normalized_indexes.pop(old_key)
    if normalized_indexes != canonical_indexes:
        mismatches.add("indexes")

    function_names = sorted({identity.partition("(")[0] for identity in _FUNCTION_METADATA})
    function_rows = await connection.fetch(_FUNCTIONS_SQL, function_names)
    functions: dict[str, dict[str, Any]] = {}
    for row in function_rows:
        identity_name = str(_value(row, "identity"))
        functions[identity_name] = {
            "result": str(_value(row, "result_type")),
            "language": str(_value(row, "language")),
            "volatility": _catalog_char(_value(row, "provolatile")),
            "parallel": _catalog_char(_value(row, "proparallel")),
            "security_definer": bool(_value(row, "prosecdef")),
            "leakproof": bool(_value(row, "proleakproof")),
            "strict": bool(_value(row, "proisstrict")),
            "set_returning": bool(_value(row, "proretset")),
            "kind": _catalog_char(_value(row, "prokind")),
            "cost": float(_value(row, "procost", -1)),
            "rows": float(_value(row, "prorows", -1)),
            "no_support_function": bool(_value(row, "no_support_function")),
            "no_function_acl": bool(_value(row, "no_function_acl")),
            "configuration": tuple(str(item) for item in (_value(row, "proconfig", []) or [])),
            "source": _normalize_sql(str(_value(row, "prosrc", ""))),
            "owned_by_current_user": bool(_value(row, "owned_by_current_user")),
        }
    if set(functions) != set(_FUNCTION_METADATA):
        mismatches.add("functions")
    for identity_name, metadata in _FUNCTION_METADATA.items():
        actual = functions.get(identity_name)
        if actual is None:
            continue
        expected = {
            "result": metadata[0],
            "language": metadata[1],
            "volatility": metadata[2],
            "parallel": metadata[3],
            "security_definer": False,
            "leakproof": False,
            "strict": False,
            "set_returning": False,
            "kind": "f",
            "cost": 100.0,
            "rows": 0.0,
            "no_support_function": True,
            "no_function_acl": True,
            "configuration": ("search_path=pg_catalog, public",),
            "source": _normalize_sql(_CANONICAL_FUNCTION_SOURCES[identity_name]),
            "owned_by_current_user": True,
        }
        if actual == expected:
            continue
        legacy = {
            **expected,
            "configuration": (),
            "source": _normalize_sql(_LEGACY_FUNCTION_SOURCES[identity_name]),
        }
        legacy_parallel_modes = {metadata[3]}
        if identity_name == "immutable_unaccent(text)":
            # VectorStore.initialize historically replaced the DocumentStore
            # definition without PARALLEL SAFE. Both execution orders shipped;
            # the canonical replacement below hardens either known variant.
            legacy_parallel_modes.add("u")
        if (
            allow_known_legacy
            and actual["parallel"] in legacy_parallel_modes
            and {**actual, "parallel": metadata[3]} == legacy
        ):
            legacy_functions.append(f"harden-function:{identity_name}")
            continue
        mismatches.add(f"function:{identity_name}")

    trigger_rows = await connection.fetch(_TRIGGERS_SQL, table_names)
    triggers: dict[tuple[str, str], dict[str, Any]] = {}
    for row in trigger_rows:
        key = (str(_value(row, "table_name")), str(_value(row, "tgname")))
        triggers[key] = {
            "type": int(_value(row, "tgtype", -1)),
            "enabled": _catalog_char(_value(row, "tgenabled")),
            "function": str(_value(row, "function_identity")),
            "empty_arguments": bool(_value(row, "empty_arguments")),
            "not_constraint_trigger": bool(_value(row, "not_constraint_trigger")),
            "not_deferrable": bool(_value(row, "not_deferrable")),
            "not_initially_deferred": bool(_value(row, "not_initially_deferred")),
            "no_when_clause": bool(_value(row, "no_when_clause")),
            "no_old_transition_table": bool(_value(row, "no_old_transition_table")),
            "no_new_transition_table": bool(_value(row, "no_new_transition_table")),
        }
    canonical_triggers = {
        key: {
            "type": 23,  # ROW | BEFORE | INSERT | UPDATE
            "enabled": "O",
            "function": function,
            "empty_arguments": True,
            "not_constraint_trigger": True,
            "not_deferrable": True,
            "not_initially_deferred": True,
            "no_when_clause": True,
            "no_old_transition_table": True,
            "no_new_transition_table": True,
        }
        for key, function in _TRIGGERS.items()
    }
    if triggers != canonical_triggers:
        mismatches.add("triggers")

    rule_rows = await connection.fetch(_RULES_SQL, table_names)
    rules = tuple((str(_value(row, "table_name")), str(_value(row, "rulename"))) for row in rule_rows)
    if rules:
        mismatches.add("rewrite-rules")

    publication_rows = await connection.fetch(_PUBLICATION_MEMBERSHIP_SQL, table_names)
    publication_memberships = tuple(
        (str(_value(row, "pubname")), str(_value(row, "table_name"))) for row in publication_rows
    )
    if publication_memberships:
        mismatches.add("logical-publications")

    sequence_names = sorted(value[0].partition(".")[2] for value in _SERIAL_COLUMNS.values())
    sequence_rows = await connection.fetch(_SEQUENCES_SQL, sequence_names)
    sequences: dict[str, dict[str, Any]] = {}
    for row in sequence_rows:
        name = str(_value(row, "sequence_name"))
        sequences[name] = {
            "type": str(_value(row, "data_type")),
            "start": int(_value(row, "seqstart", -1)),
            "increment": int(_value(row, "seqincrement", -1)),
            "maximum": int(_value(row, "seqmax", -1)),
            "minimum": int(_value(row, "seqmin", -1)),
            "cache": int(_value(row, "seqcache", -1)),
            "cycle": bool(_value(row, "seqcycle")),
            "owned_table": str(_value(row, "owned_table")),
            "owned_column": str(_value(row, "owned_column")),
            "dependency": _catalog_char(_value(row, "deptype")),
            "no_sequence_acl": bool(_value(row, "no_sequence_acl")),
            "owned_by_current_user": bool(_value(row, "owned_by_current_user")),
        }
    expected_sequence_names = {value[0].partition(".")[2] for value in _SERIAL_COLUMNS.values()}
    if set(sequences) != expected_sequence_names:
        mismatches.add("sequences")
    for column_key, (qualified_sequence, data_type) in _SERIAL_COLUMNS.items():
        table, column = column_key.split(".", 1)
        name = qualified_sequence.partition(".")[2]
        expected_sequence = {
            "type": data_type,
            "start": 1,
            "increment": 1,
            "maximum": 9223372036854775807 if data_type == "bigint" else 2147483647,
            "minimum": 1,
            "cache": 1,
            "cycle": False,
            "owned_table": table,
            "owned_column": column,
            "dependency": "i",
            "no_sequence_acl": True,
            "owned_by_current_user": True,
        }
        actual = sequences.get(name)
        if actual == expected_sequence:
            continue
        legacy_sequence = {**expected_sequence, "dependency": "a"}
        if allow_known_legacy and actual == legacy_sequence and f"serial-to-identity:{column_key}" in serial_columns:
            continue
        mismatches.add(f"sequence:{name}")

    manifest = {
        "target_migration": {"version": V0001_CORE.version, "checksum": V0001_CORE.checksum},
        "application_schemas": application_schemas,
        "application_relations": application_relations,
        "application_functions": application_functions,
        "extensions": extensions,
        "relations": relations,
        "columns": actual_columns,
        "constraints": {f"{key[0]}.{key[1]}": value for key, value in constraints.items()},
        "indexes": {f"{key[0]}.{key[1]}": value for key, value in indexes.items()},
        "functions": functions,
        "triggers": {f"{key[0]}.{key[1]}": value for key, value in triggers.items()},
        "rules": rules,
        "logical_publications": publication_memberships,
        "sequences": sequences,
    }
    if mismatches:
        raise _adoption_refusal(mismatches)
    return LegacyInspection(
        fingerprint=_fingerprint(manifest),
        serial_columns=tuple(sorted(serial_columns)),
        legacy_constraints=tuple(sorted(legacy_constraints)),
        legacy_functions=tuple(sorted(legacy_functions)),
        manifest=manifest,
    )


async def lock_legacy_v1_tables(connection: Any) -> None:
    """Hold all v0001 tables stable before sequence-state normalization."""

    await connection.execute(_LOCK_MANAGED_TABLES_SQL)


async def _normalize_serial_column(connection: Any, column_key: str, qualified_sequence: str) -> None:
    table, column = column_key.split(".", 1)
    sequence_state = await connection.fetchrow(
        f"SELECT last_value, is_called FROM {qualified_sequence}"  # noqa: S608 - fixed internal identifiers
    )
    maximum = await connection.fetchval(
        f"SELECT pg_catalog.max({column}) FROM public.{table}"  # noqa: S608 - fixed internal identifiers
    )
    last_value = int(_value(sequence_state, "last_value", 1))
    is_called = bool(_value(sequence_state, "is_called", False))
    next_from_sequence = last_value + 1 if is_called else last_value
    next_value = max(1, next_from_sequence, int(maximum) + 1 if maximum is not None else 1)
    sequence_type = _SERIAL_COLUMNS[column_key][1]
    sequence_maximum = 9223372036854775807 if sequence_type == "bigint" else 2147483647
    if next_value > sequence_maximum:
        raise LegacyAdoptionError(
            "Legacy adoption refused because an identity sequence has no safe next value. No corpus rows were "
            "changed. Follow the blue-green data-only procedure in docs/LEGACY_DATABASE_UPGRADE.md."
        )

    await connection.execute(
        f"ALTER TABLE public.{table} ALTER COLUMN {column} DROP DEFAULT"  # noqa: S608
    )
    await connection.execute(f"DROP SEQUENCE {qualified_sequence}")  # noqa: S608
    await connection.execute(
        f"ALTER TABLE public.{table} ALTER COLUMN {column} "  # noqa: S608
        f"ADD GENERATED BY DEFAULT AS IDENTITY (SEQUENCE NAME {qualified_sequence})"
    )
    if next_value == 1:
        await connection.fetchval("SELECT pg_catalog.setval($1::pg_catalog.regclass, 1, false)", qualified_sequence)
    else:
        await connection.fetchval(
            "SELECT pg_catalog.setval($1::pg_catalog.regclass, $2::pg_catalog.int8, true)",
            qualified_sequence,
            next_value - 1,
        )


async def normalize_legacy_v1(connection: Any, inspection: LegacyInspection) -> None:
    """Apply only the reviewed legacy metadata normalizations transactionally."""

    for item in inspection.serial_columns:
        column_key = item.removeprefix("serial-to-identity:")
        await _normalize_serial_column(connection, column_key, _SERIAL_COLUMNS[column_key][0])

    for item in inspection.legacy_constraints:
        payload = item.removeprefix("rename-constraint:")
        qualified_old, new_name = payload.split("->", 1)
        table, old_name = qualified_old.split(".", 1)
        await connection.execute(
            f"ALTER TABLE public.{table} RENAME CONSTRAINT {old_name} TO {new_name}"  # noqa: S608
        )

    functions_to_replace = {
        item.removeprefix("harden-function:").partition("(")[0] for item in inspection.legacy_functions
    }
    for statement in V0001_CORE.statements:
        normalized = statement.lstrip()
        if not normalized.startswith("CREATE FUNCTION public."):
            continue
        function_name = normalized.removeprefix("CREATE FUNCTION public.").partition("(")[0]
        if function_name in functions_to_replace:
            await connection.execute(statement.replace("CREATE FUNCTION", "CREATE OR REPLACE FUNCTION", 1))
