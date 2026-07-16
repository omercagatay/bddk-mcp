"""Privacy-safe persistence of one strictly verified corpus release identity."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from bddk_mcp.catalog_integrity import inspect_catalog_integrity
from bddk_mcp.corpus_manifest import CorpusManifestValidation
from bddk_mcp.migrations import inspect_migration_state

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_ID_RE = re.compile(r"^corpus_release_sha256_[0-9a-f]{64}$")
STRICT_FRESHNESS_POLICY_RESULT = "quantified_measured_signature_verified_pass"

_ACTIVE_RELEASE_SQL = """
SELECT release_id,
       manifest_id,
       manifest_sha256,
       signer_key_sha256,
       freshness_policy_result,
       source_detection_slo_seconds,
       publication_slo_seconds,
       max_manifest_age_seconds,
       retrieval_profile_sha256,
       corpus_state_sha256,
       completed_at
FROM bddk_meta.active_corpus_release
"""

_PUBLISH_RELEASE_SQL = """
WITH canonical_publication_inputs AS MATERIALIZED (
    SELECT $1::pg_catalog.text AS manifest_id,
           $2::pg_catalog.text AS manifest_sha256,
           $3::pg_catalog.text AS signer_key_sha256,
           $4::pg_catalog.int4 AS source_detection_slo_seconds,
           $5::pg_catalog.int4 AS publication_slo_seconds,
           $6::pg_catalog.int4 AS max_manifest_age_seconds,
           $7::pg_catalog.text AS retrieval_profile_sha256,
           pg_catalog.set_config('TimeZone', 'UTC', true) AS timezone_setting,
           pg_catalog.set_config('DateStyle', 'ISO, YMD', true) AS datestyle_setting,
           pg_catalog.set_config('IntervalStyle', 'postgres', true) AS intervalstyle_setting,
           pg_catalog.set_config('bytea_output', 'hex', true) AS bytea_output_setting,
           pg_catalog.set_config('extra_float_digits', '3', true) AS float_digits_setting
)
SELECT published.release_id,
       published.manifest_id,
       published.manifest_sha256,
       published.signer_key_sha256,
       published.freshness_policy_result,
       published.source_detection_slo_seconds,
       published.publication_slo_seconds,
       published.max_manifest_age_seconds,
       published.retrieval_profile_sha256,
       published.corpus_state_sha256,
       published.completed_at
FROM canonical_publication_inputs AS inputs
CROSS JOIN LATERAL bddk_meta.publish_verified_corpus_release(
    inputs.manifest_id,
    inputs.manifest_sha256,
    inputs.signer_key_sha256,
    inputs.source_detection_slo_seconds,
    inputs.publication_slo_seconds,
    inputs.max_manifest_age_seconds,
    inputs.retrieval_profile_sha256
) AS published
WHERE inputs.timezone_setting = 'UTC'
  AND inputs.datestyle_setting = 'ISO, YMD'
  AND inputs.intervalstyle_setting = 'postgres'
  AND inputs.bytea_output_setting = 'hex'
  AND inputs.float_digits_setting = '3'
"""
_CORPUS_PUBLICATION_READY_SQL = "SELECT bddk_meta.corpus_retrieval_ready($1)"


class CorpusPublicationError(RuntimeError):
    """Raised when verified release evidence cannot be safely persisted or read."""


@dataclass(frozen=True, slots=True)
class CorpusReleaseIdentity:
    """Path-free, content-free identity safe for readiness and operator evidence."""

    release_id: str
    manifest_id: str
    manifest_sha256: str
    signer_key_sha256: str
    freshness_policy_result: str
    source_detection_slo_seconds: int
    publication_slo_seconds: int
    max_manifest_age_seconds: int
    retrieval_profile_sha256: str
    corpus_state_sha256: str
    completed_at: datetime

    def safe_dict(self) -> dict[str, Any]:
        """Return a JSON-ready object containing no paths, principal names, or corpus text."""

        value = asdict(self)
        value["completed_at"] = self.completed_at.isoformat()
        return value


def is_strict_release_request(
    *,
    require_quantified_freshness: bool,
    require_measured_freshness: bool,
    require_verified_signature: bool,
) -> bool:
    """Return true only for the complete production-oriented policy gate."""

    return all(
        (
            require_quantified_freshness,
            require_measured_freshness,
            require_verified_signature,
        )
    )


def _value(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def _identity_from_row(row: Any) -> CorpusReleaseIdentity:
    """Parse DB evidence fail closed before it reaches a health or tool response."""

    try:
        identity = CorpusReleaseIdentity(
            release_id=str(_value(row, "release_id", "")),
            manifest_id=str(_value(row, "manifest_id", "")),
            manifest_sha256=str(_value(row, "manifest_sha256", "")),
            signer_key_sha256=str(_value(row, "signer_key_sha256", "")),
            freshness_policy_result=str(_value(row, "freshness_policy_result", "")),
            source_detection_slo_seconds=int(_value(row, "source_detection_slo_seconds", 0)),
            publication_slo_seconds=int(_value(row, "publication_slo_seconds", 0)),
            max_manifest_age_seconds=int(_value(row, "max_manifest_age_seconds", 0)),
            retrieval_profile_sha256=str(_value(row, "retrieval_profile_sha256", "")),
            corpus_state_sha256=str(_value(row, "corpus_state_sha256", "")),
            completed_at=_value(row, "completed_at"),
        )
    except (TypeError, ValueError):
        raise CorpusPublicationError("Active corpus release evidence is invalid.") from None
    if (
        not _RELEASE_ID_RE.fullmatch(identity.release_id)
        or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", identity.manifest_id)
        or any(
            not _SHA256_RE.fullmatch(value)
            for value in (
                identity.manifest_sha256,
                identity.signer_key_sha256,
                identity.retrieval_profile_sha256,
                identity.corpus_state_sha256,
            )
        )
        or identity.freshness_policy_result != STRICT_FRESHNESS_POLICY_RESULT
        or min(
            identity.source_detection_slo_seconds,
            identity.publication_slo_seconds,
            identity.max_manifest_age_seconds,
        )
        <= 0
        or not isinstance(identity.completed_at, datetime)
        or identity.completed_at.tzinfo is None
        or identity.completed_at.utcoffset() is None
    ):
        raise CorpusPublicationError("Active corpus release evidence is invalid.")
    return identity


def _strict_publication_values(
    validation: CorpusManifestValidation,
    *,
    retrieval_profile_sha256: str,
    require_quantified_freshness: bool,
    require_measured_freshness: bool,
    require_verified_signature: bool,
) -> tuple[str, str, str, int, int, int, str]:
    if not is_strict_release_request(
        require_quantified_freshness=require_quantified_freshness,
        require_measured_freshness=require_measured_freshness,
        require_verified_signature=require_verified_signature,
    ):
        raise CorpusPublicationError("Corpus release activation requires the complete strict policy gate.")

    manifest = validation.manifest
    freshness = manifest.freshness
    signer_key_sha256 = manifest.integrity.signature_public_key_sha256
    values = (
        freshness.source_detection_slo_seconds,
        freshness.publication_slo_seconds,
        freshness.max_manifest_age_seconds,
    )
    if (
        manifest.integrity.signature_status != "verified"
        or signer_key_sha256 is None
        or freshness.slo_evidence_status != "measured"
        or any(value is None or value <= 0 for value in values)
        or not _SHA256_RE.fullmatch(retrieval_profile_sha256)
    ):
        raise CorpusPublicationError("Verified corpus release policy evidence is incomplete.")
    return (
        manifest.manifest_id,
        validation.manifest_sha256,
        signer_key_sha256,
        int(values[0]),
        int(values[1]),
        int(values[2]),
        retrieval_profile_sha256,
    )


async def publish_strict_corpus_release(
    connection: Any,
    validation: CorpusManifestValidation,
    *,
    retrieval_profile_sha256: str,
    require_quantified_freshness: bool,
    require_measured_freshness: bool,
    require_verified_signature: bool,
) -> CorpusReleaseIdentity:
    """Append and activate evidence inside the caller's final transaction."""

    values = _strict_publication_values(
        validation,
        retrieval_profile_sha256=retrieval_profile_sha256,
        require_quantified_freshness=require_quantified_freshness,
        require_measured_freshness=require_measured_freshness,
        require_verified_signature=require_verified_signature,
    )
    try:
        row = await connection.fetchrow(_PUBLISH_RELEASE_SQL, *values)
    except Exception:
        raise CorpusPublicationError("Verified corpus release evidence could not be activated.") from None
    if row is None:
        raise CorpusPublicationError("Verified corpus release evidence could not be activated.")
    return _identity_from_row(row)


async def inspect_active_corpus_release(pool: Any) -> CorpusReleaseIdentity | None:
    """Read and validate the active release view without exposing base evidence."""

    try:
        row = await pool.fetchrow(_ACTIVE_RELEASE_SQL)
    except Exception:
        raise CorpusPublicationError("Active corpus release evidence could not be verified.") from None
    return None if row is None else _identity_from_row(row)


async def assert_release_publication_ready(
    pool: Any,
    *,
    retrieval_profile_sha256: str,
    require_active_release: bool,
) -> CorpusReleaseIdentity | None:
    """Attest the exact v5/v6-remediation or v7 publication boundary.

    This is not serving readiness.  Schema v5/v6 is accepted only so the signed
    publication command can append a canonical replacement release before the
    v7 migration guard is retried.  The migration ledger, complete managed
    catalog for that version, and selected-profile corpus readiness all remain
    fail-closed.
    """

    if _SHA256_RE.fullmatch(retrieval_profile_sha256) is None:
        raise CorpusPublicationError("Corpus release publication profile is invalid.")
    try:
        migration_state = await inspect_migration_state(pool)
        schema_version = migration_state.current_version
        if schema_version not in {5, 6, 7}:
            raise CorpusPublicationError("Corpus release publication requires schema version 5, 6, or 7.")
        catalog = await inspect_catalog_integrity(
            pool,
            expected_schema_version=schema_version,
        )
        if not catalog.valid:
            raise CorpusPublicationError("Corpus release publication catalog integrity verification failed.")
        corpus_ready = await pool.fetchval(
            _CORPUS_PUBLICATION_READY_SQL,
            retrieval_profile_sha256,
        )
        if corpus_ready is not True:
            raise CorpusPublicationError("Corpus release publication corpus readiness verification failed.")
        active_release = await inspect_active_corpus_release(pool)
        if require_active_release and active_release is None:
            raise CorpusPublicationError("Corpus release publication did not produce an active identity.")
        return active_release
    except CorpusPublicationError:
        raise
    except Exception:
        raise CorpusPublicationError("Corpus release publication readiness could not be verified.") from None


__all__ = (
    "CorpusPublicationError",
    "CorpusReleaseIdentity",
    "STRICT_FRESHNESS_POLICY_RESULT",
    "assert_release_publication_ready",
    "inspect_active_corpus_release",
    "is_strict_release_request",
    "publish_strict_corpus_release",
)
