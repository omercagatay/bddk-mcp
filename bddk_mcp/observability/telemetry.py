"""Optional privacy-safe tool-call telemetry."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

import asyncpg

from bddk_mcp.core.config import TELEMETRY_ENABLED, TELEMETRY_MODEL_ID, TELEMETRY_SESSION_ID, TELEMETRY_STORE_TEXT
from bddk_mcp.db_compatibility import PostgreSQLCompatibilityError, assert_supported_postgresql

logger = logging.getLogger(__name__)

_TEXT_ARG_KEYS = {"query", "keywords", "heading", "prompt", "question", "text"}
_SAFE_ARG_KEYS = {
    "active_only",
    "category",
    "column",
    "currency",
    "date_from",
    "date_to",
    "days",
    "document_id",
    "include_neighbors",
    "institution_type",
    "limit",
    "lookback_weeks",
    "metric_id",
    "month",
    "page",
    "page_number",
    "page_size",
    "party_code",
    "period",
    "section_ref",
    "section_type",
    "table_no",
    "year",
}

_TELEMETRY_PRIVILEGES_SQL = """
WITH RECURSIVE target AS (
    SELECT to_regclass('public.tool_call_traces') AS relation_oid,
           to_regclass('public.tool_call_traces_id_seq') AS sequence_oid
), session_role AS (
    SELECT role.oid,
           role.rolsuper,
           role.rolcreaterole,
           role.rolcreatedb,
           role.rolreplication,
           role.rolbypassrls
    FROM pg_catalog.pg_roles AS role
    WHERE role.rolname = session_user
), role_closure(role_oid) AS (
    SELECT oid FROM session_role
    UNION
    SELECT membership.roleid
    FROM pg_catalog.pg_auth_members AS membership
    JOIN role_closure AS inherited ON inherited.role_oid = membership.member
), requested_relations(schema_name, relation_name) AS (
    VALUES
        ('public', 'decision_cache'),
        ('public', 'documents'),
        ('public', 'document_sections'),
        ('public', 'document_versions'),
        ('public', 'document_chunks'),
        ('public', 'document_retrieval_publications'),
        ('public', 'sync_metadata'),
        ('public', 'sync_failures'),
        ('public', 'regulatory_instruments'),
        ('public', 'regulatory_family_imports'),
        ('public', 'regulatory_source_blobs'),
        ('public', 'regulatory_source_artifacts'),
        ('public', 'regulatory_evidence'),
        ('public', 'regulatory_legal_versions'),
        ('public', 'regulatory_legal_version_artifacts'),
        ('public', 'regulatory_legal_events'),
        ('public', 'regulatory_legal_status_assertions'),
        ('public', 'regulatory_provisions'),
        ('public', 'regulatory_legal_version_provisions'),
        ('public', 'regulatory_validated_section_citations'),
        ('bddk_meta', 'schema_migrations'),
        ('bddk_meta', 'legacy_schema_adoptions'),
        ('bddk_meta', 'corpus_releases'),
        ('bddk_meta', 'corpus_release_activations'),
        ('bddk_meta', 'corpus_state_epoch'),
        ('bddk_meta', 'active_corpus_release'),
        ('bddk_operator', 'operator_jobs')
), other_relations(relation_oid) AS (
    SELECT relation.oid
    FROM requested_relations AS requested
    LEFT JOIN pg_catalog.pg_namespace AS namespace
      ON namespace.nspname = requested.schema_name
    LEFT JOIN pg_catalog.pg_class AS relation
      ON relation.relnamespace = namespace.oid
     AND relation.relname = requested.relation_name
), requested_sequences(schema_name, sequence_name) AS (
    VALUES
        ('public', 'document_sections_id_seq'),
        ('public', 'document_versions_id_seq'),
        ('public', 'document_chunks_id_seq'),
        ('bddk_meta', 'corpus_release_activations_activation_sequence_seq')
), other_sequences(sequence_oid) AS (
    SELECT sequence.oid
    FROM requested_sequences AS requested
    LEFT JOIN pg_catalog.pg_namespace AS namespace
      ON namespace.nspname = requested.schema_name
    LEFT JOIN pg_catalog.pg_class AS sequence
      ON sequence.relnamespace = namespace.oid
     AND sequence.relname = requested.sequence_name
     AND sequence.relkind = 'S'
), requested_functions(schema_name, function_name, argument_types) AS (
    VALUES
        ('public', 'immutable_unaccent', 'text'),
        ('public', 'documents_tsv_trigger', ''),
        ('public', 'document_sections_tsv_trigger', ''),
        ('public', 'chunks_tsv_trigger', ''),
        ('public', 'invalidate_retrieval_publication', ''),
        ('bddk_meta', 'corpus_fingerprint_frame', 'text'),
        ('bddk_meta', 'bump_corpus_state_epoch', ''),
        ('bddk_meta', 'current_corpus_state_sha256', 'text'),
        ('bddk_meta', 'corpus_retrieval_ready', 'text'),
        ('bddk_meta', 'reject_corpus_release_mutation', ''),
        (
            'bddk_meta',
            'publish_verified_corpus_release',
            'text, text, text, integer, integer, integer, text'
        ),
        ('bddk_meta', 'resolve_regulation_status', 'text, date')
), application_functions(function_oid) AS (
    SELECT routine.oid
    FROM requested_functions AS requested
    LEFT JOIN pg_catalog.pg_namespace AS namespace
      ON namespace.nspname = requested.schema_name
    LEFT JOIN pg_catalog.pg_proc AS routine
      ON routine.pronamespace = namespace.oid
     AND routine.proname = requested.function_name
     AND pg_catalog.oidvectortypes(routine.proargtypes) = requested.argument_types
)
SELECT relation_oid IS NOT NULL AS relation_exists,
       sequence_oid IS NOT NULL AS sequence_exists,
       current_user = session_user AS session_is_current,
       COALESCE((
           SELECT NOT (rolsuper OR rolcreaterole OR rolcreatedb OR rolreplication OR rolbypassrls)
           FROM session_role
       ), false) AS identity_hardened,
       pg_has_role(session_user, 'bddk_telemetry_writer', 'MEMBER')
           AND NOT EXISTS (
               SELECT 1
               FROM role_closure
               JOIN pg_catalog.pg_roles AS inherited_role ON inherited_role.oid = role_closure.role_oid
               WHERE inherited_role.rolname NOT IN (session_user, 'bddk_telemetry_writer')
           ) AS membership_isolated,
       NOT has_database_privilege(current_user, current_database(), 'CREATE')
           AND NOT has_database_privilege(current_user, current_database(), 'TEMPORARY')
           AS database_capabilities_isolated,
       has_schema_privilege(current_user, 'public', 'USAGE') AS schema_usage,
       NOT has_schema_privilege(current_user, 'public', 'CREATE')
           AND NOT has_schema_privilege(current_user, 'bddk_meta', 'USAGE')
           AND NOT has_schema_privilege(current_user, 'bddk_meta', 'CREATE')
           AND NOT has_schema_privilege(current_user, 'bddk_operator', 'USAGE')
           AND NOT has_schema_privilege(current_user, 'bddk_operator', 'CREATE')
           AS application_schemas_isolated,
       NOT EXISTS (
           SELECT 1
           FROM other_relations
           WHERE relation_oid IS NULL
              OR has_any_column_privilege(current_user, relation_oid, 'SELECT')
              OR has_any_column_privilege(current_user, relation_oid, 'INSERT')
              OR has_any_column_privilege(current_user, relation_oid, 'UPDATE')
              OR has_any_column_privilege(current_user, relation_oid, 'REFERENCES')
              OR has_table_privilege(current_user, relation_oid, 'DELETE')
              OR has_table_privilege(current_user, relation_oid, 'TRUNCATE')
              OR has_table_privilege(current_user, relation_oid, 'TRIGGER')
       ) AS other_relations_isolated,
       NOT EXISTS (
           SELECT 1
           FROM other_sequences
           WHERE sequence_oid IS NULL
              OR has_sequence_privilege(current_user, sequence_oid, 'USAGE')
              OR has_sequence_privilege(current_user, sequence_oid, 'SELECT')
              OR has_sequence_privilege(current_user, sequence_oid, 'UPDATE')
       ) AS other_sequences_isolated,
       NOT EXISTS (
           SELECT 1
           FROM application_functions
           WHERE function_oid IS NULL
              OR pg_catalog.has_function_privilege(current_user, function_oid, 'EXECUTE')
       ) AS application_functions_isolated,
       CASE WHEN relation_oid IS NULL THEN false
            ELSE (
                has_column_privilege(current_user, relation_oid, 'tool_name', 'INSERT')
                AND has_column_privilege(current_user, relation_oid, 'args_hash', 'INSERT')
                AND has_column_privilege(current_user, relation_oid, 'args_summary', 'INSERT')
                AND has_column_privilege(current_user, relation_oid, 'latency_ms', 'INSERT')
                AND has_column_privilege(current_user, relation_oid, 'result_count', 'INSERT')
                AND has_column_privilege(current_user, relation_oid, 'doc_ids', 'INSERT')
                AND has_column_privilege(current_user, relation_oid, 'quality_labels', 'INSERT')
                AND has_column_privilege(current_user, relation_oid, 'relevance_stats', 'INSERT')
                AND has_column_privilege(current_user, relation_oid, 'model_id', 'INSERT')
                AND has_column_privilege(current_user, relation_oid, 'session_id', 'INSERT')
            ) END AS can_insert_required_columns,
       CASE WHEN relation_oid IS NULL THEN false
            ELSE (
                has_column_privilege(current_user, relation_oid, 'id', 'INSERT')
                OR has_column_privilege(current_user, relation_oid, 'created_at', 'INSERT')
            ) END AS can_insert_managed_columns,
       CASE WHEN relation_oid IS NULL THEN false
            ELSE has_any_column_privilege(current_user, relation_oid, 'SELECT') END AS can_select,
       CASE WHEN relation_oid IS NULL THEN false
            ELSE has_any_column_privilege(current_user, relation_oid, 'UPDATE') END AS can_update,
       CASE WHEN relation_oid IS NULL THEN false
            ELSE has_table_privilege(current_user, relation_oid, 'DELETE') END AS can_delete,
       CASE WHEN relation_oid IS NULL THEN false
            ELSE has_table_privilege(current_user, relation_oid, 'TRUNCATE') END AS can_truncate,
       CASE WHEN sequence_oid IS NULL THEN false
            ELSE has_sequence_privilege(current_user, sequence_oid, 'USAGE') END AS sequence_usage,
       CASE WHEN sequence_oid IS NULL THEN false
            ELSE has_sequence_privilege(current_user, sequence_oid, 'SELECT') END AS sequence_select,
       CASE WHEN sequence_oid IS NULL THEN false
            ELSE has_sequence_privilege(current_user, sequence_oid, 'UPDATE') END AS sequence_update
FROM target
"""


class TelemetryIdentityError(RuntimeError):
    """The telemetry pool is missing its exact least-privilege contract."""


async def assert_telemetry_writer_ready(
    pool: asyncpg.Pool,
    *,
    require_session_identity: bool = True,
) -> None:
    """Prove that the configured identity is INSERT-only for trace rows."""

    try:
        await assert_supported_postgresql(pool)
        privileges = await pool.fetchrow(_TELEMETRY_PRIVILEGES_SQL)
        identity_allowed = bool(
            not require_session_identity
            or (
                privileges
                and privileges["session_is_current"]
                and privileges["identity_hardened"]
                and privileges["membership_isolated"]
            )
        )
        allowed = bool(
            privileges
            and privileges["relation_exists"]
            and privileges["sequence_exists"]
            and identity_allowed
            and privileges["database_capabilities_isolated"]
            and privileges["schema_usage"]
            and privileges["application_schemas_isolated"]
            and privileges["other_relations_isolated"]
            and privileges["other_sequences_isolated"]
            and privileges["application_functions_isolated"]
            and privileges["can_insert_required_columns"]
            and privileges["sequence_usage"]
        )
        forbidden = bool(
            privileges
            and any(
                privileges[key]
                for key in (
                    "can_select",
                    "can_update",
                    "can_delete",
                    "can_truncate",
                    "can_insert_managed_columns",
                    "sequence_select",
                    "sequence_update",
                )
            )
        )
        if not allowed or forbidden:
            raise TelemetryIdentityError("Telemetry database identity is not INSERT-only for public.tool_call_traces.")
    except TelemetryIdentityError:
        raise
    except PostgreSQLCompatibilityError as exc:
        raise TelemetryIdentityError(str(exc)) from None
    except (asyncpg.PostgresError, OSError, KeyError, TypeError):
        raise TelemetryIdentityError(
            "Telemetry database identity could not be verified against the required INSERT-only contract."
        ) from None


def elapsed_ms(start: float) -> int:
    """Return elapsed milliseconds since a perf_counter start timestamp."""
    return max(0, int((time.perf_counter() - start) * 1000))


def args_hash(args: dict[str, Any]) -> str:
    """Return a stable SHA-256 hash of the full argument payload."""
    payload = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def summarize_args(args: dict[str, Any], *, store_text: bool = False) -> dict[str, Any]:
    """Build a privacy-safe args summary.

    Text-like user inputs are hashed and length-counted by default. Raw text is
    only included when store_text=True, which is controlled by an explicit env var.
    """
    summary: dict[str, Any] = {}
    for key, value in sorted(args.items()):
        if value is None:
            continue
        if key in _TEXT_ARG_KEYS:
            summary[key] = _text_summary(str(value), include_value=store_text)
        elif key in _SAFE_ARG_KEYS or isinstance(value, (bool, int, float)):
            summary[key] = value
        elif isinstance(value, str):
            # Unknown string fields are user-controlled by default. Never let
            # a newly added tool argument become raw telemetry merely because
            # it is short; only reviewed identifier keys may pass unchanged.
            summary[key] = _text_summary(value, include_value=store_text)
        else:
            summary[key] = {"type": type(value).__name__}
    return summary


def relevance_stats_from_hits(hits: list[dict]) -> dict[str, Any]:
    """Summarize relevance fields without storing snippets or queries."""
    if not hits:
        return {"result_count": 0}
    relevances = [float(hit.get("relevance", 0.0) or 0.0) for hit in hits]
    match_types = sorted({str(hit.get("match_type", "")) for hit in hits if hit.get("match_type")})
    return {
        "result_count": len(hits),
        "max_relevance": round(max(relevances), 4),
        "min_relevance": round(min(relevances), 4),
        "avg_relevance": round(sum(relevances) / len(relevances), 4),
        "match_types": match_types,
    }


def quality_labels_from_hits(hits: list[dict]) -> dict[str, dict[str, Any]]:
    """Collect quality labels/flags keyed by document ID from search hits."""
    labels: dict[str, dict[str, Any]] = {}
    for hit in hits:
        doc_id = hit.get("doc_id") or hit.get("document_id")
        if not doc_id:
            continue
        labels[str(doc_id)] = {
            "label": hit.get("quality_label", "unknown"),
            "flags": hit.get("quality_flags", []),
        }
    return labels


def unique_doc_ids(values: list[str | None]) -> list[str]:
    """Return doc IDs in first-seen order, dropping blanks."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


async def record_tool_call_trace(
    pool,
    *,
    tool_name: str,
    args: dict[str, Any],
    latency_ms: int,
    result_count: int | None = None,
    doc_ids: list[str] | None = None,
    quality_labels: dict[str, Any] | None = None,
    relevance_stats: dict[str, Any] | None = None,
    model_id: str | None = None,
    session_id: str | None = None,
) -> bool:
    """Persist a tool-call trace when telemetry is enabled.

    Telemetry is best-effort and never raises into the public tool path.
    """
    if not TELEMETRY_ENABLED or pool is None:
        return False

    try:
        await pool.execute(
            """
            INSERT INTO public.tool_call_traces (
                tool_name, args_hash, args_summary, latency_ms, result_count,
                doc_ids, quality_labels, relevance_stats, model_id, session_id
            )
            VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7::jsonb, $8::jsonb, $9, $10)
            """,
            tool_name,
            args_hash(args),
            json.dumps(summarize_args(args, store_text=TELEMETRY_STORE_TEXT), ensure_ascii=False),
            latency_ms,
            result_count,
            doc_ids or [],
            json.dumps(quality_labels or {}, ensure_ascii=False),
            json.dumps(relevance_stats or {}, ensure_ascii=False),
            model_id if model_id is not None else TELEMETRY_MODEL_ID or None,
            session_id if session_id is not None else TELEMETRY_SESSION_ID or None,
        )
        return True
    except Exception as error:
        logger.debug(
            "tool telemetry write failed for %s (error_type=%s)",
            tool_name,
            type(error).__name__,
        )
        return False


def _text_summary(value: str, *, include_value: bool = False) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        "chars": len(value),
        "words": len(value.split()),
    }
    if include_value:
        summary["value"] = value
    return summary
