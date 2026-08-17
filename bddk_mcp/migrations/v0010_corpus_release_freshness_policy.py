"""Migration 0010: admit an explicit quantified-but-unmeasured release policy.

Schema v5 and v8 accept exactly one ``freshness_policy_result``, which requires
per-document authoritative-publication, detection, download, extraction, and
retrieval-publication events.  A batch-snapshot corpus has no such event
sequence, so a signed and quantified corpus could never be activated at all.

This migration widens that gate to a closed two-value set and keeps the weaker
state explicit rather than implicit.  Quantified objectives and a verified
signature remain unconditional: no unsigned or unquantified value is ever
admissible.  The policy result is a staging *input* rather than a constant, so
the verifier records the level it actually proved, and because that value is
fingerprinted into both the release and request identities, a measured and an
unmeasured release over identical corpus state can never share an identity or
be substituted for one another.
"""

from __future__ import annotations

from typing import Final

from bddk_mcp.corpus_coordination import CORPUS_MUTATION_ADVISORY_KEY
from bddk_mcp.migrations.model import Migration
from bddk_mcp.migrations.v0005_corpus_release_publication import CORPUS_EPOCH_TRACKED_TABLES
from bddk_mcp.migrations.v0008_staged_corpus_releases import (
    MAX_RELEASE_REQUEST_TTL_SECONDS,
    MIN_RELEASE_REQUEST_TTL_SECONDS,
    RELEASE_REQUEST_ID_PREFIX,
)

MEASURED_FRESHNESS_POLICY_RESULT: Final[str] = "quantified_measured_signature_verified_pass"
UNMEASURED_FRESHNESS_POLICY_RESULT: Final[str] = "quantified_unmeasured_signature_verified_pass"
ADMISSIBLE_FRESHNESS_POLICY_RESULTS: Final[tuple[str, ...]] = (
    MEASURED_FRESHNESS_POLICY_RESULT,
    UNMEASURED_FRESHNESS_POLICY_RESULT,
)

_ADMISSIBLE_POLICY_SQL_LIST: Final[str] = ", ".join(f"'{value}'" for value in ADMISSIBLE_FRESHNESS_POLICY_RESULTS)

_SOURCE_TABLE_LOCKS = ",\n                       ".join(
    f"public.{relation}" for relation in CORPUS_EPOCH_TRACKED_TABLES
)

V0010_CORPUS_RELEASE_FRESHNESS_POLICY = Migration(
    version=10,
    name="corpus_release_freshness_policy",
    statements=(
        """
        ALTER TABLE bddk_meta.corpus_releases
            DROP CONSTRAINT corpus_releases_policy_result_check
        """,
        f"""
        ALTER TABLE bddk_meta.corpus_releases
            ADD CONSTRAINT corpus_releases_policy_result_check
            CHECK (freshness_policy_result IN ({_ADMISSIBLE_POLICY_SQL_LIST}))
        """,
        """
        ALTER TABLE bddk_meta.corpus_release_requests
            DROP CONSTRAINT corpus_release_requests_policy_result_check
        """,
        f"""
        ALTER TABLE bddk_meta.corpus_release_requests
            ADD CONSTRAINT corpus_release_requests_policy_result_check
            CHECK (freshness_policy_result IN ({_ADMISSIBLE_POLICY_SQL_LIST}))
        """,
        # The staged policy level becomes a verifier input, so the v8 routine is
        # replaced outright rather than kept as a silently measured-only path.
        """
        DROP FUNCTION bddk_meta.stage_verified_corpus_release(
            pg_catalog.text,
            pg_catalog.text,
            pg_catalog.text,
            pg_catalog.text,
            pg_catalog.text,
            pg_catalog.int4,
            pg_catalog.int4,
            pg_catalog.int4,
            pg_catalog.text,
            pg_catalog.text,
            pg_catalog.text,
            pg_catalog.int4
        )
        """,
        f"""
        CREATE FUNCTION bddk_meta.stage_verified_corpus_release(
            requested_manifest_id pg_catalog.text,
            requested_manifest_sha256 pg_catalog.text,
            requested_signature_sha256 pg_catalog.text,
            requested_signer_key_sha256 pg_catalog.text,
            requested_verification_evidence_sha256 pg_catalog.text,
            requested_freshness_policy_result pg_catalog.text,
            requested_source_detection_slo_seconds pg_catalog.int4,
            requested_publication_slo_seconds pg_catalog.int4,
            requested_max_manifest_age_seconds pg_catalog.int4,
            requested_retrieval_profile_sha256 pg_catalog.text,
            requested_verifier_revision_sha256 pg_catalog.text,
            requested_verifier_image_digest pg_catalog.text,
            requested_valid_for_seconds pg_catalog.int4
        )
        RETURNS TABLE (
            request_id pg_catalog.text,
            release_id pg_catalog.text,
            corpus_state_sha256 pg_catalog.text,
            corpus_epoch pg_catalog.int8,
            staged_at pg_catalog.timestamptz,
            verification_expires_at pg_catalog.timestamptz
        )
        LANGUAGE plpgsql
        VOLATILE
        PARALLEL UNSAFE
        SECURITY DEFINER
        SET search_path = pg_catalog
        AS $function$
        DECLARE
            selected_request_id pg_catalog.text;
            selected_release_id pg_catalog.text;
            selected_state_sha256 pg_catalog.text;
            selected_corpus_epoch pg_catalog.int8;
            selected_staged_at pg_catalog.timestamptz;
            selected_expires_at pg_catalog.timestamptz;
            selected_actor_fingerprint pg_catalog.text;
            persisted_request bddk_meta.corpus_release_requests%ROWTYPE;
        BEGIN
            IF pg_catalog.to_regrole('bddk_release_verifier') IS NULL
               OR NOT pg_catalog.pg_has_role(
                   SESSION_USER,
                   pg_catalog.to_regrole('bddk_release_verifier'),
                   'MEMBER'
               )
               OR (
                   pg_catalog.to_regrole('bddk_release_publisher') IS NOT NULL
                   AND pg_catalog.pg_has_role(
                       SESSION_USER,
                       pg_catalog.to_regrole('bddk_release_publisher'),
                       'MEMBER'
                   )
               ) THEN
                RAISE EXCEPTION 'verified corpus release staging caller is not authorized'
                    USING ERRCODE = '42501';
            END IF;
            IF requested_manifest_id IS NULL
               OR requested_manifest_sha256 IS NULL
               OR requested_signature_sha256 IS NULL
               OR requested_signer_key_sha256 IS NULL
               OR requested_verification_evidence_sha256 IS NULL
               OR requested_freshness_policy_result IS NULL
               OR requested_source_detection_slo_seconds IS NULL
               OR requested_publication_slo_seconds IS NULL
               OR requested_max_manifest_age_seconds IS NULL
               OR requested_retrieval_profile_sha256 IS NULL
               OR requested_verifier_revision_sha256 IS NULL
               OR requested_verifier_image_digest IS NULL
               OR requested_valid_for_seconds IS NULL
               OR requested_manifest_id !~ '^[a-z0-9][a-z0-9._-]{{2,127}}$'
               OR requested_manifest_sha256 !~ '^[0-9a-f]{{64}}$'
               OR requested_signature_sha256 !~ '^[0-9a-f]{{64}}$'
               OR requested_signer_key_sha256 !~ '^[0-9a-f]{{64}}$'
               OR requested_verification_evidence_sha256 !~ '^[0-9a-f]{{64}}$'
               OR requested_freshness_policy_result NOT IN ({_ADMISSIBLE_POLICY_SQL_LIST})
               OR requested_retrieval_profile_sha256 !~ '^[0-9a-f]{{64}}$'
               OR requested_verifier_revision_sha256 !~ '^[0-9a-f]{{64}}$'
               OR requested_verifier_image_digest !~ '^sha256:[0-9a-f]{{64}}$'
               OR requested_source_detection_slo_seconds <= 0
               OR requested_publication_slo_seconds <= 0
               OR requested_max_manifest_age_seconds <= 0
               OR requested_valid_for_seconds NOT BETWEEN {MIN_RELEASE_REQUEST_TTL_SECONDS}
                                                          AND {MAX_RELEASE_REQUEST_TTL_SECONDS} THEN
                RAISE EXCEPTION 'verified corpus release staging identity is invalid'
                    USING ERRCODE = '22023';
            END IF;

            PERFORM pg_catalog.pg_advisory_xact_lock(
                {CORPUS_MUTATION_ADVISORY_KEY}::pg_catalog.int8
            );
            LOCK TABLE {_SOURCE_TABLE_LOCKS}
                IN SHARE MODE;

            SELECT epoch.epoch
            INTO STRICT selected_corpus_epoch
            FROM bddk_meta.corpus_state_epoch AS epoch
            WHERE epoch.singleton_id;
            IF NOT bddk_meta.corpus_retrieval_ready(requested_retrieval_profile_sha256) THEN
                RAISE EXCEPTION 'verified corpus release staging state is not retrieval-ready'
                    USING ERRCODE = '55000';
            END IF;

            selected_state_sha256 := bddk_meta.current_corpus_state_sha256(
                requested_retrieval_profile_sha256
            );
            selected_release_id := 'corpus_release_sha256_' || pg_catalog.encode(
                pg_catalog.sha256(
                    bddk_meta.corpus_fingerprint_frame('1')
                    || bddk_meta.corpus_fingerprint_frame(requested_manifest_id)
                    || bddk_meta.corpus_fingerprint_frame(requested_manifest_sha256)
                    || bddk_meta.corpus_fingerprint_frame(requested_signer_key_sha256)
                    || bddk_meta.corpus_fingerprint_frame(requested_freshness_policy_result)
                    || bddk_meta.corpus_fingerprint_frame(
                           requested_source_detection_slo_seconds::pg_catalog.text
                       )
                    || bddk_meta.corpus_fingerprint_frame(
                           requested_publication_slo_seconds::pg_catalog.text
                       )
                    || bddk_meta.corpus_fingerprint_frame(
                           requested_max_manifest_age_seconds::pg_catalog.text
                       )
                    || bddk_meta.corpus_fingerprint_frame(requested_retrieval_profile_sha256)
                    || bddk_meta.corpus_fingerprint_frame(selected_state_sha256)
                ),
                'hex'
            );
            selected_actor_fingerprint := pg_catalog.encode(
                pg_catalog.sha256(pg_catalog.convert_to(SESSION_USER::pg_catalog.text, 'UTF8')),
                'hex'
            );
            selected_request_id := '{RELEASE_REQUEST_ID_PREFIX}' || pg_catalog.encode(
                pg_catalog.sha256(
                    bddk_meta.corpus_fingerprint_frame('1')
                    || bddk_meta.corpus_fingerprint_frame(selected_release_id)
                    || bddk_meta.corpus_fingerprint_frame(requested_manifest_id)
                    || bddk_meta.corpus_fingerprint_frame(requested_manifest_sha256)
                    || bddk_meta.corpus_fingerprint_frame(requested_signature_sha256)
                    || bddk_meta.corpus_fingerprint_frame(requested_signer_key_sha256)
                    || bddk_meta.corpus_fingerprint_frame(
                           requested_verification_evidence_sha256
                       )
                    || bddk_meta.corpus_fingerprint_frame(requested_freshness_policy_result)
                    || bddk_meta.corpus_fingerprint_frame(
                           requested_source_detection_slo_seconds::pg_catalog.text
                       )
                    || bddk_meta.corpus_fingerprint_frame(
                           requested_publication_slo_seconds::pg_catalog.text
                       )
                    || bddk_meta.corpus_fingerprint_frame(
                           requested_max_manifest_age_seconds::pg_catalog.text
                       )
                    || bddk_meta.corpus_fingerprint_frame(requested_retrieval_profile_sha256)
                    || bddk_meta.corpus_fingerprint_frame(selected_state_sha256)
                    || bddk_meta.corpus_fingerprint_frame(
                           selected_corpus_epoch::pg_catalog.text
                       )
                    || bddk_meta.corpus_fingerprint_frame(requested_verifier_revision_sha256)
                    || bddk_meta.corpus_fingerprint_frame(requested_verifier_image_digest)
                    || bddk_meta.corpus_fingerprint_frame(
                           requested_valid_for_seconds::pg_catalog.text
                       )
                    || bddk_meta.corpus_fingerprint_frame(selected_actor_fingerprint)
                ),
                'hex'
            );
            selected_staged_at := pg_catalog.clock_timestamp();
            selected_expires_at := selected_staged_at
                + requested_valid_for_seconds * pg_catalog.interval '1 second';

            INSERT INTO bddk_meta.corpus_release_requests (
                request_id,
                release_id,
                manifest_id,
                manifest_sha256,
                signature_sha256,
                signer_key_sha256,
                verification_evidence_sha256,
                freshness_policy_result,
                source_detection_slo_seconds,
                publication_slo_seconds,
                max_manifest_age_seconds,
                retrieval_profile_sha256,
                corpus_state_sha256,
                corpus_epoch,
                verifier_revision_sha256,
                verifier_image_digest,
                valid_for_seconds,
                staged_at,
                verification_expires_at,
                verifier_fingerprint_sha256
            ) VALUES (
                selected_request_id,
                selected_release_id,
                requested_manifest_id,
                requested_manifest_sha256,
                requested_signature_sha256,
                requested_signer_key_sha256,
                requested_verification_evidence_sha256,
                requested_freshness_policy_result,
                requested_source_detection_slo_seconds,
                requested_publication_slo_seconds,
                requested_max_manifest_age_seconds,
                requested_retrieval_profile_sha256,
                selected_state_sha256,
                selected_corpus_epoch,
                requested_verifier_revision_sha256,
                requested_verifier_image_digest,
                requested_valid_for_seconds,
                selected_staged_at,
                selected_expires_at,
                selected_actor_fingerprint
            ) ON CONFLICT ON CONSTRAINT corpus_release_requests_pkey DO NOTHING;

            SELECT request.*
            INTO STRICT persisted_request
            FROM bddk_meta.corpus_release_requests AS request
            WHERE request.request_id = selected_request_id;
            IF persisted_request.release_id IS DISTINCT FROM selected_release_id
               OR persisted_request.manifest_id IS DISTINCT FROM requested_manifest_id
               OR persisted_request.manifest_sha256 IS DISTINCT FROM requested_manifest_sha256
               OR persisted_request.signature_sha256 IS DISTINCT FROM requested_signature_sha256
               OR persisted_request.signer_key_sha256 IS DISTINCT FROM requested_signer_key_sha256
               OR persisted_request.verification_evidence_sha256
                    IS DISTINCT FROM requested_verification_evidence_sha256
               OR persisted_request.freshness_policy_result
                    IS DISTINCT FROM requested_freshness_policy_result
               OR persisted_request.source_detection_slo_seconds
                    IS DISTINCT FROM requested_source_detection_slo_seconds
               OR persisted_request.publication_slo_seconds
                    IS DISTINCT FROM requested_publication_slo_seconds
               OR persisted_request.max_manifest_age_seconds
                    IS DISTINCT FROM requested_max_manifest_age_seconds
               OR persisted_request.retrieval_profile_sha256
                    IS DISTINCT FROM requested_retrieval_profile_sha256
               OR persisted_request.corpus_state_sha256 IS DISTINCT FROM selected_state_sha256
               OR persisted_request.corpus_epoch IS DISTINCT FROM selected_corpus_epoch
               OR persisted_request.verifier_revision_sha256
                    IS DISTINCT FROM requested_verifier_revision_sha256
               OR persisted_request.verifier_image_digest
                    IS DISTINCT FROM requested_verifier_image_digest
               OR persisted_request.valid_for_seconds
                    IS DISTINCT FROM requested_valid_for_seconds
               OR persisted_request.verifier_fingerprint_sha256
                    IS DISTINCT FROM selected_actor_fingerprint THEN
                RAISE EXCEPTION 'verified corpus release staging identity collision'
                    USING ERRCODE = '55000';
            END IF;

            RETURN QUERY
            SELECT persisted_request.request_id,
                   persisted_request.release_id,
                   persisted_request.corpus_state_sha256,
                   persisted_request.corpus_epoch,
                   persisted_request.staged_at,
                   persisted_request.verification_expires_at;
        END
        $function$
        """,
        """
        REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.stage_verified_corpus_release(
            pg_catalog.text,
            pg_catalog.text,
            pg_catalog.text,
            pg_catalog.text,
            pg_catalog.text,
            pg_catalog.text,
            pg_catalog.int4,
            pg_catalog.int4,
            pg_catalog.int4,
            pg_catalog.text,
            pg_catalog.text,
            pg_catalog.text,
            pg_catalog.int4
        ) FROM PUBLIC
        """,
    ),
)

__all__ = (
    "ADMISSIBLE_FRESHNESS_POLICY_RESULTS",
    "MEASURED_FRESHNESS_POLICY_RESULT",
    "UNMEASURED_FRESHNESS_POLICY_RESULT",
    "V0010_CORPUS_RELEASE_FRESHNESS_POLICY",
)
