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


def _normalize_sql(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return text.replace("public.", "").replace("::text", "")


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

    return CatalogIntegrity(tuple(sorted(failures)))
