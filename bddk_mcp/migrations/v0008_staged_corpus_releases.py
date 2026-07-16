"""Migration 0008: separate corpus verification from release activation.

The v5 publication routine accepts verifier-controlled claims from a release
publisher.  This additive migration replaces that runtime path with an
append-only, short-lived request staged by a distinct verifier identity.  A
publisher can activate only the exact database state and policy evidence
already sealed into one request ID; it cannot supply or alter verifier claims.
"""

from __future__ import annotations

from typing import Final

from bddk_mcp.corpus_coordination import CORPUS_MUTATION_ADVISORY_KEY
from bddk_mcp.migrations.model import Migration
from bddk_mcp.migrations.v0005_corpus_release_publication import CORPUS_EPOCH_TRACKED_TABLES

RELEASE_REQUEST_ID_PREFIX: Final[str] = "corpus_release_request_sha256_"
MIN_RELEASE_REQUEST_TTL_SECONDS: Final[int] = 60
MAX_RELEASE_REQUEST_TTL_SECONDS: Final[int] = 3600

_SOURCE_TABLE_LOCKS = ",\n                       ".join(
    f"public.{relation}" for relation in CORPUS_EPOCH_TRACKED_TABLES
)

V0008_STAGED_CORPUS_RELEASES = Migration(
    version=8,
    name="staged_corpus_releases",
    statements=(
        f"""
        CREATE TABLE bddk_meta.corpus_release_requests (
            request_id pg_catalog.text PRIMARY KEY,
            release_id pg_catalog.text NOT NULL,
            manifest_id pg_catalog.text NOT NULL,
            manifest_sha256 pg_catalog.text NOT NULL,
            signature_sha256 pg_catalog.text NOT NULL,
            signer_key_sha256 pg_catalog.text NOT NULL,
            verification_evidence_sha256 pg_catalog.text NOT NULL,
            freshness_policy_result pg_catalog.text NOT NULL,
            source_detection_slo_seconds pg_catalog.int4 NOT NULL,
            publication_slo_seconds pg_catalog.int4 NOT NULL,
            max_manifest_age_seconds pg_catalog.int4 NOT NULL,
            retrieval_profile_sha256 pg_catalog.text NOT NULL,
            corpus_state_sha256 pg_catalog.text NOT NULL,
            corpus_epoch pg_catalog.int8 NOT NULL,
            verifier_revision_sha256 pg_catalog.text NOT NULL,
            verifier_image_digest pg_catalog.text NOT NULL,
            valid_for_seconds pg_catalog.int4 NOT NULL,
            staged_at pg_catalog.timestamptz NOT NULL,
            verification_expires_at pg_catalog.timestamptz NOT NULL,
            verifier_fingerprint_sha256 pg_catalog.text NOT NULL,
            CONSTRAINT corpus_release_requests_id_check
                CHECK (request_id ~ '^{RELEASE_REQUEST_ID_PREFIX}[0-9a-f]{{64}}$'),
            CONSTRAINT corpus_release_requests_release_id_check
                CHECK (release_id ~ '^corpus_release_sha256_[0-9a-f]{{64}}$'),
            CONSTRAINT corpus_release_requests_manifest_id_check
                CHECK (manifest_id ~ '^[a-z0-9][a-z0-9._-]{{2,127}}$'),
            CONSTRAINT corpus_release_requests_manifest_hash_check
                CHECK (manifest_sha256 ~ '^[0-9a-f]{{64}}$'),
            CONSTRAINT corpus_release_requests_signature_hash_check
                CHECK (signature_sha256 ~ '^[0-9a-f]{{64}}$'),
            CONSTRAINT corpus_release_requests_signer_hash_check
                CHECK (signer_key_sha256 ~ '^[0-9a-f]{{64}}$'),
            CONSTRAINT corpus_release_requests_evidence_hash_check
                CHECK (verification_evidence_sha256 ~ '^[0-9a-f]{{64}}$'),
            CONSTRAINT corpus_release_requests_policy_result_check
                CHECK (freshness_policy_result = 'quantified_measured_signature_verified_pass'),
            CONSTRAINT corpus_release_requests_source_slo_check
                CHECK (source_detection_slo_seconds > 0),
            CONSTRAINT corpus_release_requests_publication_slo_check
                CHECK (publication_slo_seconds > 0),
            CONSTRAINT corpus_release_requests_max_age_check
                CHECK (max_manifest_age_seconds > 0),
            CONSTRAINT corpus_release_requests_profile_hash_check
                CHECK (retrieval_profile_sha256 ~ '^[0-9a-f]{{64}}$'),
            CONSTRAINT corpus_release_requests_state_hash_check
                CHECK (corpus_state_sha256 ~ '^[0-9a-f]{{64}}$'),
            CONSTRAINT corpus_release_requests_epoch_check CHECK (corpus_epoch >= 0),
            CONSTRAINT corpus_release_requests_revision_hash_check
                CHECK (verifier_revision_sha256 ~ '^[0-9a-f]{{64}}$'),
            CONSTRAINT corpus_release_requests_image_digest_check
                CHECK (verifier_image_digest ~ '^sha256:[0-9a-f]{{64}}$'),
            CONSTRAINT corpus_release_requests_ttl_check
                CHECK (valid_for_seconds BETWEEN {MIN_RELEASE_REQUEST_TTL_SECONDS}
                                             AND {MAX_RELEASE_REQUEST_TTL_SECONDS}),
            CONSTRAINT corpus_release_requests_expiry_check
                CHECK (
                    verification_expires_at = staged_at
                        + valid_for_seconds * pg_catalog.interval '1 second'
                ),
            CONSTRAINT corpus_release_requests_actor_hash_check
                CHECK (verifier_fingerprint_sha256 ~ '^[0-9a-f]{{64}}$'),
            CONSTRAINT corpus_release_requests_release_identity_uq
                UNIQUE (request_id, release_id)
        )
        """,
        """
        CREATE TABLE bddk_meta.corpus_release_request_activations (
            request_id pg_catalog.text PRIMARY KEY,
            activation_sequence pg_catalog.int8 NOT NULL UNIQUE,
            release_id pg_catalog.text NOT NULL,
            bound_at pg_catalog.timestamptz NOT NULL,
            publisher_fingerprint_sha256 pg_catalog.text NOT NULL,
            CONSTRAINT corpus_release_request_activations_request_fk
                FOREIGN KEY (request_id, release_id)
                REFERENCES bddk_meta.corpus_release_requests(request_id, release_id),
            CONSTRAINT corpus_release_request_activations_activation_fk
                FOREIGN KEY (activation_sequence, release_id)
                REFERENCES bddk_meta.corpus_release_activations(activation_sequence, release_id),
            CONSTRAINT corpus_release_request_activations_actor_hash_check
                CHECK (publisher_fingerprint_sha256 ~ '^[0-9a-f]{64}$')
        )
        """,
        """
        CREATE TRIGGER reject_corpus_release_request_update_delete
        BEFORE UPDATE OR DELETE ON bddk_meta.corpus_release_requests
        FOR EACH ROW EXECUTE FUNCTION bddk_meta.reject_corpus_release_mutation()
        """,
        """
        CREATE TRIGGER reject_corpus_release_request_activation_update_delete
        BEFORE UPDATE OR DELETE ON bddk_meta.corpus_release_request_activations
        FOR EACH ROW EXECUTE FUNCTION bddk_meta.reject_corpus_release_mutation()
        """,
        f"""
        CREATE FUNCTION bddk_meta.stage_verified_corpus_release(
            requested_manifest_id pg_catalog.text,
            requested_manifest_sha256 pg_catalog.text,
            requested_signature_sha256 pg_catalog.text,
            requested_signer_key_sha256 pg_catalog.text,
            requested_verification_evidence_sha256 pg_catalog.text,
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
                    || bddk_meta.corpus_fingerprint_frame(
                           'quantified_measured_signature_verified_pass'
                       )
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
                    || bddk_meta.corpus_fingerprint_frame(
                           'quantified_measured_signature_verified_pass'
                       )
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
                'quantified_measured_signature_verified_pass',
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
        f"""
        CREATE FUNCTION bddk_meta.activate_staged_corpus_release(
            requested_request_id pg_catalog.text
        )
        RETURNS TABLE (
            request_id pg_catalog.text,
            release_id pg_catalog.text,
            manifest_id pg_catalog.text,
            manifest_sha256 pg_catalog.text,
            signer_key_sha256 pg_catalog.text,
            freshness_policy_result pg_catalog.text,
            source_detection_slo_seconds pg_catalog.int4,
            publication_slo_seconds pg_catalog.int4,
            max_manifest_age_seconds pg_catalog.int4,
            retrieval_profile_sha256 pg_catalog.text,
            corpus_state_sha256 pg_catalog.text,
            activation_sequence pg_catalog.int8,
            completed_at pg_catalog.timestamptz
        )
        LANGUAGE plpgsql
        VOLATILE
        PARALLEL UNSAFE
        SECURITY DEFINER
        SET search_path = pg_catalog
        AS $function$
        DECLARE
            selected_request bddk_meta.corpus_release_requests%ROWTYPE;
            selected_live_epoch pg_catalog.int8;
            selected_live_state_sha256 pg_catalog.text;
            recomputed_release_id pg_catalog.text;
            selected_activation_sequence pg_catalog.int8;
            selected_completed_at pg_catalog.timestamptz;
            selected_actor_fingerprint pg_catalog.text;
        BEGIN
            IF pg_catalog.to_regrole('bddk_release_publisher') IS NULL
               OR NOT pg_catalog.pg_has_role(
                   SESSION_USER,
                   pg_catalog.to_regrole('bddk_release_publisher'),
                   'MEMBER'
               )
               OR (
                   pg_catalog.to_regrole('bddk_release_verifier') IS NOT NULL
                   AND pg_catalog.pg_has_role(
                       SESSION_USER,
                       pg_catalog.to_regrole('bddk_release_verifier'),
                       'MEMBER'
                   )
               ) THEN
                RAISE EXCEPTION 'staged corpus release activation caller is not authorized'
                    USING ERRCODE = '42501';
            END IF;
            IF requested_request_id IS NULL
               OR requested_request_id !~ '^{RELEASE_REQUEST_ID_PREFIX}[0-9a-f]{{64}}$' THEN
                RAISE EXCEPTION 'staged corpus release request identity is invalid'
                    USING ERRCODE = '22023';
            END IF;

            PERFORM pg_catalog.pg_advisory_xact_lock(
                {CORPUS_MUTATION_ADVISORY_KEY}::pg_catalog.int8
            );
            LOCK TABLE bddk_meta.corpus_release_requests,
                       bddk_meta.corpus_release_request_activations,
                       bddk_meta.corpus_release_activations,
                       bddk_meta.corpus_releases
                IN SHARE ROW EXCLUSIVE MODE;
            LOCK TABLE {_SOURCE_TABLE_LOCKS}
                IN SHARE MODE;

            SELECT request.*
            INTO STRICT selected_request
            FROM bddk_meta.corpus_release_requests AS request
            WHERE request.request_id = requested_request_id
            FOR UPDATE;
            IF EXISTS (
                SELECT 1
                FROM bddk_meta.corpus_release_request_activations AS binding
                WHERE binding.request_id = requested_request_id
            ) THEN
                RAISE EXCEPTION 'staged corpus release request was already activated'
                    USING ERRCODE = '55000';
            END IF;
            IF pg_catalog.clock_timestamp() >= selected_request.verification_expires_at THEN
                RAISE EXCEPTION 'staged corpus release request has expired'
                    USING ERRCODE = '55000';
            END IF;

            SELECT epoch.epoch
            INTO STRICT selected_live_epoch
            FROM bddk_meta.corpus_state_epoch AS epoch
            WHERE epoch.singleton_id;
            IF selected_live_epoch IS DISTINCT FROM selected_request.corpus_epoch THEN
                RAISE EXCEPTION 'staged corpus release corpus epoch has changed'
                    USING ERRCODE = '55000';
            END IF;
            IF NOT bddk_meta.corpus_retrieval_ready(
                selected_request.retrieval_profile_sha256
            ) THEN
                RAISE EXCEPTION 'staged corpus release state is not retrieval-ready'
                    USING ERRCODE = '55000';
            END IF;
            selected_live_state_sha256 := bddk_meta.current_corpus_state_sha256(
                selected_request.retrieval_profile_sha256
            );
            IF selected_live_state_sha256 IS DISTINCT FROM selected_request.corpus_state_sha256 THEN
                RAISE EXCEPTION 'staged corpus release state has changed'
                    USING ERRCODE = '55000';
            END IF;

            recomputed_release_id := 'corpus_release_sha256_' || pg_catalog.encode(
                pg_catalog.sha256(
                    bddk_meta.corpus_fingerprint_frame('1')
                    || bddk_meta.corpus_fingerprint_frame(selected_request.manifest_id)
                    || bddk_meta.corpus_fingerprint_frame(selected_request.manifest_sha256)
                    || bddk_meta.corpus_fingerprint_frame(selected_request.signer_key_sha256)
                    || bddk_meta.corpus_fingerprint_frame(
                           selected_request.freshness_policy_result
                       )
                    || bddk_meta.corpus_fingerprint_frame(
                           selected_request.source_detection_slo_seconds::pg_catalog.text
                       )
                    || bddk_meta.corpus_fingerprint_frame(
                           selected_request.publication_slo_seconds::pg_catalog.text
                       )
                    || bddk_meta.corpus_fingerprint_frame(
                           selected_request.max_manifest_age_seconds::pg_catalog.text
                       )
                    || bddk_meta.corpus_fingerprint_frame(
                           selected_request.retrieval_profile_sha256
                       )
                    || bddk_meta.corpus_fingerprint_frame(selected_live_state_sha256)
                ),
                'hex'
            );
            IF recomputed_release_id IS DISTINCT FROM selected_request.release_id THEN
                RAISE EXCEPTION 'staged corpus release identity is invalid'
                    USING ERRCODE = '55000';
            END IF;
            IF pg_catalog.clock_timestamp() >= selected_request.verification_expires_at THEN
                RAISE EXCEPTION 'staged corpus release request has expired'
                    USING ERRCODE = '55000';
            END IF;

            INSERT INTO bddk_meta.corpus_releases (
                release_id,
                manifest_id,
                manifest_sha256,
                signer_key_sha256,
                freshness_policy_result,
                source_detection_slo_seconds,
                publication_slo_seconds,
                max_manifest_age_seconds,
                retrieval_profile_sha256,
                corpus_state_sha256
            ) VALUES (
                selected_request.release_id,
                selected_request.manifest_id,
                selected_request.manifest_sha256,
                selected_request.signer_key_sha256,
                selected_request.freshness_policy_result,
                selected_request.source_detection_slo_seconds,
                selected_request.publication_slo_seconds,
                selected_request.max_manifest_age_seconds,
                selected_request.retrieval_profile_sha256,
                selected_request.corpus_state_sha256
            ) ON CONFLICT ON CONSTRAINT corpus_releases_pkey DO NOTHING;
            IF NOT EXISTS (
                SELECT 1
                FROM bddk_meta.corpus_releases AS release
                WHERE release.release_id = selected_request.release_id
                  AND release.manifest_id = selected_request.manifest_id
                  AND release.manifest_sha256 = selected_request.manifest_sha256
                  AND release.signer_key_sha256 = selected_request.signer_key_sha256
                  AND release.freshness_policy_result = selected_request.freshness_policy_result
                  AND release.source_detection_slo_seconds
                        = selected_request.source_detection_slo_seconds
                  AND release.publication_slo_seconds
                        = selected_request.publication_slo_seconds
                  AND release.max_manifest_age_seconds
                        = selected_request.max_manifest_age_seconds
                  AND release.retrieval_profile_sha256
                        = selected_request.retrieval_profile_sha256
                  AND release.corpus_state_sha256 = selected_request.corpus_state_sha256
            ) THEN
                RAISE EXCEPTION 'staged corpus release conflicts with persisted evidence'
                    USING ERRCODE = '55000';
            END IF;

            selected_actor_fingerprint := pg_catalog.encode(
                pg_catalog.sha256(pg_catalog.convert_to(SESSION_USER::pg_catalog.text, 'UTF8')),
                'hex'
            );
            INSERT INTO bddk_meta.corpus_release_activations (
                release_id,
                corpus_epoch,
                actor_fingerprint_sha256
            ) VALUES (
                selected_request.release_id,
                selected_request.corpus_epoch,
                selected_actor_fingerprint
            )
            RETURNING corpus_release_activations.activation_sequence,
                      corpus_release_activations.completed_at
            INTO selected_activation_sequence,
                 selected_completed_at;

            INSERT INTO bddk_meta.corpus_release_request_activations (
                request_id,
                activation_sequence,
                release_id,
                bound_at,
                publisher_fingerprint_sha256
            ) VALUES (
                selected_request.request_id,
                selected_activation_sequence,
                selected_request.release_id,
                selected_completed_at,
                selected_actor_fingerprint
            );

            RETURN QUERY
            SELECT selected_request.request_id,
                   selected_request.release_id,
                   selected_request.manifest_id,
                   selected_request.manifest_sha256,
                   selected_request.signer_key_sha256,
                   selected_request.freshness_policy_result,
                   selected_request.source_detection_slo_seconds,
                   selected_request.publication_slo_seconds,
                   selected_request.max_manifest_age_seconds,
                   selected_request.retrieval_profile_sha256,
                   selected_request.corpus_state_sha256,
                   selected_activation_sequence,
                   selected_completed_at;
        EXCEPTION
            WHEN NO_DATA_FOUND THEN
                RAISE EXCEPTION 'staged corpus release request is unavailable'
                    USING ERRCODE = '55000';
        END
        $function$
        """,
        "REVOKE ALL PRIVILEGES ON TABLE bddk_meta.corpus_release_requests FROM PUBLIC",
        "REVOKE ALL PRIVILEGES ON TABLE bddk_meta.corpus_release_request_activations FROM PUBLIC",
        """
        REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.stage_verified_corpus_release(
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
        """
        REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.activate_staged_corpus_release(
            pg_catalog.text
        ) FROM PUBLIC
        """,
        """
        DO $retire_legacy_publisher$
        DECLARE
            selected_grantee pg_catalog.name;
            legacy_routine pg_catalog.regprocedure :=
                'bddk_meta.publish_verified_corpus_release('
                'pg_catalog.text, pg_catalog.text, pg_catalog.text, '
                'pg_catalog.int4, pg_catalog.int4, pg_catalog.int4, '
                'pg_catalog.text)'::pg_catalog.regprocedure;
        BEGIN
            REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.publish_verified_corpus_release(
                pg_catalog.text,
                pg_catalog.text,
                pg_catalog.text,
                pg_catalog.int4,
                pg_catalog.int4,
                pg_catalog.int4,
                pg_catalog.text
            ) FROM PUBLIC;
            FOR selected_grantee IN
                SELECT grantee.rolname
                FROM pg_catalog.pg_proc AS routine
                CROSS JOIN LATERAL pg_catalog.aclexplode(
                    COALESCE(
                        routine.proacl,
                        pg_catalog.acldefault('f'::"char", routine.proowner)
                    )
                ) AS acl
                JOIN pg_catalog.pg_roles AS grantee ON grantee.oid = acl.grantee
                WHERE routine.oid = legacy_routine
                  AND acl.grantee <> routine.proowner
                  AND acl.privilege_type = 'EXECUTE'
            LOOP
                EXECUTE pg_catalog.format(
                    'REVOKE EXECUTE ON FUNCTION %s FROM %I CASCADE',
                    legacy_routine,
                    selected_grantee
                );
            END LOOP;
        END
        $retire_legacy_publisher$
        """,
    ),
)

__all__ = (
    "MAX_RELEASE_REQUEST_TTL_SECONDS",
    "MIN_RELEASE_REQUEST_TTL_SECONDS",
    "RELEASE_REQUEST_ID_PREFIX",
    "V0008_STAGED_CORPUS_RELEASES",
)
