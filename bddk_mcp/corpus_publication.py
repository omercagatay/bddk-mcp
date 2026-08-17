"""Privacy-safe persistence of one strictly verified corpus release identity."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from bddk_mcp.catalog_integrity import inspect_catalog_integrity
from bddk_mcp.corpus_manifest import CorpusManifestValidation
from bddk_mcp.migrations import inspect_migration_state
from bddk_mcp.migrations.v0010_corpus_release_freshness_policy import (
    ADMISSIBLE_FRESHNESS_POLICY_RESULTS as _ADMISSIBLE_POLICY_RESULTS,
)
from bddk_mcp.migrations.v0010_corpus_release_freshness_policy import (
    MEASURED_FRESHNESS_POLICY_RESULT,
    UNMEASURED_FRESHNESS_POLICY_RESULT,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RELEASE_ID_RE = re.compile(r"^corpus_release_sha256_[0-9a-f]{64}$")
_REQUEST_ID_RE = re.compile(r"^corpus_release_request_sha256_[0-9a-f]{64}$")
_IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
STRICT_FRESHNESS_POLICY_RESULT = MEASURED_FRESHNESS_POLICY_RESULT
ADMISSIBLE_FRESHNESS_POLICY_RESULTS = frozenset(_ADMISSIBLE_POLICY_RESULTS)

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
_STAGE_RELEASE_SQL = """
WITH canonical_staging_inputs AS MATERIALIZED (
    SELECT $1::pg_catalog.text AS manifest_id,
           $2::pg_catalog.text AS manifest_sha256,
           $3::pg_catalog.text AS signature_sha256,
           $4::pg_catalog.text AS signer_key_sha256,
           $5::pg_catalog.text AS verification_evidence_sha256,
           $6::pg_catalog.text AS freshness_policy_result,
           $7::pg_catalog.int4 AS source_detection_slo_seconds,
           $8::pg_catalog.int4 AS publication_slo_seconds,
           $9::pg_catalog.int4 AS max_manifest_age_seconds,
           $10::pg_catalog.text AS retrieval_profile_sha256,
           $11::pg_catalog.text AS verifier_revision_sha256,
           $12::pg_catalog.text AS verifier_image_digest,
           $13::pg_catalog.int4 AS valid_for_seconds,
           pg_catalog.set_config('TimeZone', 'UTC', true) AS timezone_setting,
           pg_catalog.set_config('DateStyle', 'ISO, YMD', true) AS datestyle_setting,
           pg_catalog.set_config('IntervalStyle', 'postgres', true) AS intervalstyle_setting,
           pg_catalog.set_config('bytea_output', 'hex', true) AS bytea_output_setting,
           pg_catalog.set_config('extra_float_digits', '3', true) AS float_digits_setting
)
SELECT staged.request_id,
       staged.release_id,
       staged.corpus_state_sha256,
       staged.corpus_epoch,
       staged.staged_at,
       staged.verification_expires_at
FROM canonical_staging_inputs AS inputs
CROSS JOIN LATERAL bddk_meta.stage_verified_corpus_release(
    inputs.manifest_id,
    inputs.manifest_sha256,
    inputs.signature_sha256,
    inputs.signer_key_sha256,
    inputs.verification_evidence_sha256,
    inputs.freshness_policy_result,
    inputs.source_detection_slo_seconds,
    inputs.publication_slo_seconds,
    inputs.max_manifest_age_seconds,
    inputs.retrieval_profile_sha256,
    inputs.verifier_revision_sha256,
    inputs.verifier_image_digest,
    inputs.valid_for_seconds
) AS staged
WHERE inputs.timezone_setting = 'UTC'
  AND inputs.datestyle_setting = 'ISO, YMD'
  AND inputs.intervalstyle_setting = 'postgres'
  AND inputs.bytea_output_setting = 'hex'
  AND inputs.float_digits_setting = '3'
"""
_ACTIVATE_RELEASE_SQL = """
SELECT activated.request_id,
       activated.release_id,
       activated.manifest_id,
       activated.manifest_sha256,
       activated.signer_key_sha256,
       activated.freshness_policy_result,
       activated.source_detection_slo_seconds,
       activated.publication_slo_seconds,
       activated.max_manifest_age_seconds,
       activated.retrieval_profile_sha256,
       activated.corpus_state_sha256,
       activated.activation_sequence,
       activated.completed_at
FROM bddk_meta.activate_staged_corpus_release($1::pg_catalog.text) AS activated
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


@dataclass(frozen=True, slots=True)
class CorpusReleaseRequestIdentity:
    """Content-free identity of one short-lived independent verification."""

    request_id: str
    release_id: str
    corpus_state_sha256: str
    corpus_epoch: int
    staged_at: datetime
    verification_expires_at: datetime

    def safe_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["staged_at"] = self.staged_at.isoformat()
        value["verification_expires_at"] = self.verification_expires_at.isoformat()
        return value


@dataclass(frozen=True, slots=True)
class CorpusReleaseActivationReceipt:
    """Activation evidence returned to the publisher without request-table access."""

    request_id: str
    activation_sequence: int
    release: CorpusReleaseIdentity

    def safe_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "activation_sequence": self.activation_sequence,
            "active_corpus_release": self.release.safe_dict(),
        }


def is_strict_release_request(
    *,
    require_quantified_freshness: bool,
    require_measured_freshness: bool,
    require_verified_signature: bool,
) -> bool:
    """Return true only for the complete production-oriented policy gate.

    Measured freshness is the strongest admissible level but no longer the only
    one: schema v10 also admits an explicitly recorded quantified-unmeasured
    release.  Quantified objectives and a verified signature stay mandatory, so
    a caller that waives either is never staging a governed release.
    """

    del require_measured_freshness
    return require_quantified_freshness and require_verified_signature


def freshness_policy_result_for(validation: CorpusManifestValidation) -> str:
    """Derive the policy level the manifest actually proves.

    The level is read from verified manifest evidence rather than asserted by a
    caller, so an operator flag can permit the weaker state but never relabel an
    unmeasured corpus as measured.
    """

    manifest = validation.manifest
    if manifest.integrity.signature_status != "verified":
        raise CorpusPublicationError("Verified corpus release policy evidence is incomplete.")
    if any(
        value is None or value <= 0
        for value in (
            manifest.freshness.source_detection_slo_seconds,
            manifest.freshness.publication_slo_seconds,
            manifest.freshness.max_manifest_age_seconds,
        )
    ):
        raise CorpusPublicationError("Verified corpus release policy evidence is incomplete.")
    if manifest.freshness.slo_evidence_status == "measured":
        return MEASURED_FRESHNESS_POLICY_RESULT
    return UNMEASURED_FRESHNESS_POLICY_RESULT


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
        or identity.freshness_policy_result not in ADMISSIBLE_FRESHNESS_POLICY_RESULTS
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


def _request_from_row(row: Any) -> CorpusReleaseRequestIdentity:
    try:
        request = CorpusReleaseRequestIdentity(
            request_id=str(_value(row, "request_id", "")),
            release_id=str(_value(row, "release_id", "")),
            corpus_state_sha256=str(_value(row, "corpus_state_sha256", "")),
            corpus_epoch=int(_value(row, "corpus_epoch", -1)),
            staged_at=_value(row, "staged_at"),
            verification_expires_at=_value(row, "verification_expires_at"),
        )
    except (TypeError, ValueError):
        raise CorpusPublicationError("Corpus release request evidence is invalid.") from None
    if (
        _REQUEST_ID_RE.fullmatch(request.request_id) is None
        or _RELEASE_ID_RE.fullmatch(request.release_id) is None
        or _SHA256_RE.fullmatch(request.corpus_state_sha256) is None
        or request.corpus_epoch < 0
        or any(
            not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None
            for value in (request.staged_at, request.verification_expires_at)
        )
        or request.verification_expires_at <= request.staged_at
    ):
        raise CorpusPublicationError("Corpus release request evidence is invalid.")
    return request


def _activation_from_row(row: Any) -> CorpusReleaseActivationReceipt:
    release = _identity_from_row(row)
    try:
        request_id = str(_value(row, "request_id", ""))
        activation_sequence = int(_value(row, "activation_sequence", 0))
    except (TypeError, ValueError):
        raise CorpusPublicationError("Corpus release activation evidence is invalid.") from None
    if _REQUEST_ID_RE.fullmatch(request_id) is None or activation_sequence <= 0:
        raise CorpusPublicationError("Corpus release activation evidence is invalid.")
    return CorpusReleaseActivationReceipt(
        request_id=request_id,
        activation_sequence=activation_sequence,
        release=release,
    )


def _strict_publication_values(
    validation: CorpusManifestValidation,
    *,
    retrieval_profile_sha256: str,
    require_quantified_freshness: bool,
    require_measured_freshness: bool,
    require_verified_signature: bool,
) -> tuple[str, str, str, str, int, int, int, str]:
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
    policy_result = freshness_policy_result_for(validation)
    if (
        manifest.integrity.signature_status != "verified"
        or signer_key_sha256 is None
        or (require_measured_freshness and policy_result != MEASURED_FRESHNESS_POLICY_RESULT)
        or any(value is None or value <= 0 for value in values)
        or not _SHA256_RE.fullmatch(retrieval_profile_sha256)
    ):
        raise CorpusPublicationError("Verified corpus release policy evidence is incomplete.")
    return (
        manifest.manifest_id,
        validation.manifest_sha256,
        signer_key_sha256,
        policy_result,
        int(values[0]),
        int(values[1]),
        int(values[2]),
        retrieval_profile_sha256,
    )


def strict_verification_evidence_sha256(
    validation: CorpusManifestValidation,
    *,
    signature_sha256: str,
    retrieval_profile_sha256: str,
    verifier_revision_sha256: str,
    verifier_image_digest: str,
    verification_run_sha256: str,
) -> str:
    """Hash a bounded canonical receipt for the checks the verifier will commit.

    The manifest checksum already binds artifact paths and bytes.  This receipt
    additionally binds the detached signature, retrieval profile, and immutable
    verifier build that performed the exact database-membership comparison.
    """

    manifest = validation.manifest
    signer_key_sha256 = manifest.integrity.signature_public_key_sha256
    if (
        manifest.integrity.signature_status != "verified"
        or signer_key_sha256 is None
        or any(
            _SHA256_RE.fullmatch(value) is None
            for value in (
                validation.manifest_sha256,
                signature_sha256,
                signer_key_sha256,
                retrieval_profile_sha256,
                verifier_revision_sha256,
                verification_run_sha256,
            )
        )
        or _IMAGE_DIGEST_RE.fullmatch(verifier_image_digest) is None
    ):
        raise CorpusPublicationError("Verified corpus release evidence is incomplete.")
    payload = {
        "schema_version": 2,
        "verification_profile": "strict_corpus_release_membership_v2",
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": validation.manifest_sha256,
        "signature_sha256": signature_sha256,
        "signer_key_sha256": signer_key_sha256,
        "freshness_policy_result": freshness_policy_result_for(validation),
        "retrieval_profile_sha256": retrieval_profile_sha256,
        "verifier_revision_sha256": verifier_revision_sha256,
        "verifier_image_digest": verifier_image_digest,
        "verification_run_sha256": verification_run_sha256,
        "artifacts": [
            {
                "role": artifact.role,
                "path": artifact.path,
                "sha256": artifact.sha256,
                "bytes": artifact.bytes,
                "records": artifact.records,
            }
            for artifact in sorted(manifest.artifacts, key=lambda artifact: artifact.path)
        ],
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


async def stage_strict_corpus_release(
    connection: Any,
    validation: CorpusManifestValidation,
    *,
    signature_sha256: str,
    verification_evidence_sha256: str,
    retrieval_profile_sha256: str,
    verifier_revision_sha256: str,
    verifier_image_digest: str,
    valid_for_seconds: int,
    require_measured_freshness: bool = True,
) -> CorpusReleaseRequestIdentity:
    """Stage independent verification evidence without activating a release."""

    base_values = _strict_publication_values(
        validation,
        retrieval_profile_sha256=retrieval_profile_sha256,
        require_quantified_freshness=True,
        require_measured_freshness=require_measured_freshness,
        require_verified_signature=True,
    )
    if (
        any(
            _SHA256_RE.fullmatch(value) is None
            for value in (signature_sha256, verification_evidence_sha256, verifier_revision_sha256)
        )
        or _IMAGE_DIGEST_RE.fullmatch(verifier_image_digest) is None
        or isinstance(valid_for_seconds, bool)
        or not isinstance(valid_for_seconds, int)
        or not 60 <= valid_for_seconds <= 3600
    ):
        raise CorpusPublicationError("Verified corpus release staging evidence is invalid.")
    values = (
        base_values[0],
        base_values[1],
        signature_sha256,
        base_values[2],
        verification_evidence_sha256,
        base_values[3],
        base_values[4],
        base_values[5],
        base_values[6],
        base_values[7],
        verifier_revision_sha256,
        verifier_image_digest,
        valid_for_seconds,
    )
    try:
        row = await connection.fetchrow(_STAGE_RELEASE_SQL, *values)
    except Exception:
        raise CorpusPublicationError("Verified corpus release evidence could not be staged.") from None
    if row is None:
        raise CorpusPublicationError("Verified corpus release evidence could not be staged.")
    return _request_from_row(row)


async def activate_staged_corpus_release(
    connection: Any,
    *,
    request_id: str,
) -> CorpusReleaseActivationReceipt:
    """Activate only one unexpired, unused request selected by opaque identity."""

    if _REQUEST_ID_RE.fullmatch(request_id) is None:
        raise CorpusPublicationError("Corpus release request identity is invalid.")
    try:
        row = await connection.fetchrow(_ACTIVATE_RELEASE_SQL, request_id)
    except Exception:
        raise CorpusPublicationError("Staged corpus release could not be activated.") from None
    if row is None:
        raise CorpusPublicationError("Staged corpus release could not be activated.")
    receipt = _activation_from_row(row)
    if receipt.request_id != request_id:
        raise CorpusPublicationError("Corpus release activation evidence is invalid.")
    return receipt


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
    # The retired v5 routine hardcodes the measured policy literal, so it cannot
    # express — and must never silently mislabel — a v10 unmeasured release.
    if values[3] != MEASURED_FRESHNESS_POLICY_RESULT:
        raise CorpusPublicationError("Verified corpus release policy evidence is incomplete.")
    try:
        row = await connection.fetchrow(_PUBLISH_RELEASE_SQL, *values[:3], *values[4:])
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
    "ADMISSIBLE_FRESHNESS_POLICY_RESULTS",
    "CorpusPublicationError",
    "CorpusReleaseActivationReceipt",
    "CorpusReleaseIdentity",
    "CorpusReleaseRequestIdentity",
    "MEASURED_FRESHNESS_POLICY_RESULT",
    "STRICT_FRESHNESS_POLICY_RESULT",
    "UNMEASURED_FRESHNESS_POLICY_RESULT",
    "activate_staged_corpus_release",
    "assert_release_publication_ready",
    "freshness_policy_result_for",
    "inspect_active_corpus_release",
    "is_strict_release_request",
    "publish_strict_corpus_release",
    "stage_strict_corpus_release",
    "strict_verification_evidence_sha256",
)
