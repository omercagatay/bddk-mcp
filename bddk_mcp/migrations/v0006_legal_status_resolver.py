"""Migration 0006: least-privilege, abstention-first legal-status resolver."""

from bddk_mcp.migrations.model import Migration

V0006_LEGAL_STATUS_RESOLVER = Migration(
    version=6,
    name="validated_legal_status_resolver",
    statements=(
        r"""
        CREATE FUNCTION bddk_meta.resolve_regulation_status(
            pg_catalog.text,
            pg_catalog.date
        )
        RETURNS TABLE (
            resolved pg_catalog.bool,
            reason pg_catalog.text,
            instrument_id pg_catalog.text,
            as_of pg_catalog.date,
            legal_version_id pg_catalog.text,
            version_key pg_catalog.text,
            legal_text_sha256 pg_catalog.text,
            version_review_record_sha256 pg_catalog.text,
            amends_version_id pg_catalog.text,
            consolidation_state pg_catalog.text,
            evidence_json pg_catalog.text
        )
        LANGUAGE sql
        STABLE
        STRICT
        PARALLEL SAFE
        SECURITY DEFINER
        SET search_path = pg_catalog
        AS $function$
        WITH instrument_row AS (
            SELECT instrument.instrument_id
            FROM public.regulatory_instruments AS instrument
            WHERE instrument.instrument_id = $1
        ), fixture_signal AS (
            SELECT COALESCE(
                pg_catalog.bool_or(signal.fixture_only),
                false
            ) AS present
            FROM (
                SELECT family.fixture_only
                FROM public.regulatory_family_imports AS family
                WHERE family.instrument_id = $1
                UNION ALL
                SELECT artifact.fixture_only
                FROM public.regulatory_legal_versions AS version
                JOIN public.regulatory_legal_version_artifacts AS version_artifact
                  ON version_artifact.legal_version_id = version.legal_version_id
                 AND version_artifact.source_role = 'legal_text'
                JOIN public.regulatory_source_artifacts AS artifact
                  ON artifact.artifact_id = version_artifact.artifact_id
                WHERE version.instrument_id = $1
            ) AS signal
        ), evidence_sources AS (
            SELECT evidence.evidence_id,
                   evidence.locator,
                   evidence.statement_sha256,
                   evidence.authority_level,
                   artifact.artifact_id,
                   artifact.blob_id AS artifact_blob_id,
                   blob.content_sha256 AS artifact_sha256,
                   artifact.canonical_uri AS source_url,
                   artifact.source_authority,
                   artifact.retrieved_at AS artifact_retrieved_at,
                   artifact.fixture_only
            FROM public.regulatory_evidence AS evidence
            JOIN public.regulatory_source_artifacts AS artifact
              ON artifact.artifact_id = evidence.artifact_id
            JOIN public.regulatory_source_blobs AS blob
              ON blob.blob_id = artifact.blob_id
        ), validated_versions AS (
            SELECT version.*
            FROM public.regulatory_legal_versions AS version
            WHERE version.instrument_id = $1
              AND version.validation_state = 'validated'
        ), publication_ready AS (
            SELECT version.legal_version_id,
                   event.event_id AS claim_id,
                   event.event_date AS claim_date,
                   event.evidence_id,
                   event.review_record_sha256 AS claim_review_record_sha256,
                   source.locator AS evidence_locator,
                   source.statement_sha256 AS evidence_statement_sha256,
                   source.artifact_id,
                   source.artifact_blob_id,
                   source.artifact_sha256,
                   source.source_url,
                   source.source_authority,
                   source.artifact_retrieved_at
            FROM validated_versions AS version
            JOIN public.regulatory_legal_events AS event
              ON event.legal_version_id = version.legal_version_id
             AND event.event_type = 'publication'
             AND event.event_date <= $2
             AND event.validation_state = 'validated'
            JOIN evidence_sources AS source
              ON source.evidence_id = event.evidence_id
             AND source.authority_level = 'authoritative'
             AND source.fixture_only = false
            WHERE EXISTS (
                SELECT 1
                FROM public.regulatory_legal_version_artifacts AS version_artifact
                WHERE version_artifact.legal_version_id = version.legal_version_id
                  AND version_artifact.artifact_id = source.artifact_id
                  AND version_artifact.source_role = 'legal_text'
            )
        ), effective_ready AS (
            SELECT version.legal_version_id,
                   event.event_id AS claim_id,
                   event.event_date AS claim_date,
                   event.evidence_id,
                   event.review_record_sha256 AS claim_review_record_sha256,
                   source.locator AS evidence_locator,
                   source.statement_sha256 AS evidence_statement_sha256,
                   source.artifact_id,
                   source.artifact_blob_id,
                   source.artifact_sha256,
                   source.source_url,
                   source.source_authority,
                   source.artifact_retrieved_at
            FROM validated_versions AS version
            JOIN public.regulatory_legal_events AS event
              ON event.legal_version_id = version.legal_version_id
             AND event.event_type = 'effective'
             AND event.event_date <= $2
             AND event.validation_state = 'validated'
            JOIN evidence_sources AS source
              ON source.evidence_id = event.evidence_id
             AND source.authority_level = 'authoritative'
             AND source.fixture_only = false
            WHERE EXISTS (
                SELECT 1
                FROM public.regulatory_legal_version_artifacts AS version_artifact
                WHERE version_artifact.legal_version_id = version.legal_version_id
                  AND version_artifact.artifact_id = source.artifact_id
                  AND version_artifact.source_role = 'legal_text'
            )
        ), ready_versions AS (
            SELECT version.*
            FROM validated_versions AS version
            JOIN publication_ready AS publication
              ON publication.legal_version_id = version.legal_version_id
            JOIN effective_ready AS effective
              ON effective.legal_version_id = version.legal_version_id
        ), evaluated_versions AS (
            SELECT version.*,
                   terminal.total_count AS terminal_signal_count,
                   terminal.supported_count AS supported_terminal_count,
                   status.invalid_count AS invalid_status_count,
                   status.effective_count AS effective_status_count,
                   status.effective_assertion_id
            FROM ready_versions AS version
            CROSS JOIN LATERAL (
                SELECT pg_catalog.count(*)::pg_catalog.int4 AS total_count,
                       pg_catalog.count(*) FILTER (
                           WHERE event.validation_state = 'validated'
                             AND source.authority_level = 'authoritative'
                             AND source.fixture_only = false
                             AND EXISTS (
                                 SELECT 1
                                 FROM public.regulatory_legal_version_artifacts AS version_artifact
                                 WHERE version_artifact.legal_version_id = version.legal_version_id
                                   AND version_artifact.artifact_id = source.artifact_id
                                   AND version_artifact.source_role = 'legal_text'
                             )
                       )::pg_catalog.int4 AS supported_count
                FROM public.regulatory_legal_events AS event
                LEFT JOIN evidence_sources AS source
                  ON source.evidence_id = event.evidence_id
                WHERE event.legal_version_id = version.legal_version_id
                  AND event.event_type IN ('expiry', 'repeal', 'supersession')
                  AND event.event_date <= $2
            ) AS terminal
            CROSS JOIN LATERAL (
                SELECT pg_catalog.count(*) FILTER (
                           WHERE assertion.validation_state <> 'validated'
                              OR source.authority_level IS DISTINCT FROM 'authoritative'
                              OR source.fixture_only IS DISTINCT FROM false
                              OR assertion.legal_status <> 'effective'
                              OR NOT EXISTS (
                                  SELECT 1
                                  FROM public.regulatory_legal_version_artifacts AS version_artifact
                                  WHERE version_artifact.legal_version_id = version.legal_version_id
                                    AND version_artifact.artifact_id = source.artifact_id
                                    AND version_artifact.source_role = 'legal_text'
                              )
                       )::pg_catalog.int4 AS invalid_count,
                       pg_catalog.count(*) FILTER (
                           WHERE assertion.validation_state = 'validated'
                             AND assertion.legal_status = 'effective'
                             AND source.authority_level = 'authoritative'
                             AND source.fixture_only = false
                             AND EXISTS (
                                 SELECT 1
                                 FROM public.regulatory_legal_version_artifacts AS version_artifact
                                 WHERE version_artifact.legal_version_id = version.legal_version_id
                                   AND version_artifact.artifact_id = source.artifact_id
                                   AND version_artifact.source_role = 'legal_text'
                             )
                       )::pg_catalog.int4 AS effective_count,
                       pg_catalog.max(assertion.assertion_id) FILTER (
                           WHERE assertion.validation_state = 'validated'
                             AND assertion.legal_status = 'effective'
                             AND source.authority_level = 'authoritative'
                             AND source.fixture_only = false
                             AND EXISTS (
                                 SELECT 1
                                 FROM public.regulatory_legal_version_artifacts AS version_artifact
                                 WHERE version_artifact.legal_version_id = version.legal_version_id
                                   AND version_artifact.artifact_id = source.artifact_id
                                   AND version_artifact.source_role = 'legal_text'
                             )
                       ) AS effective_assertion_id
                FROM public.regulatory_legal_status_assertions AS assertion
                LEFT JOIN evidence_sources AS source
                  ON source.evidence_id = assertion.evidence_id
                WHERE assertion.legal_version_id = version.legal_version_id
                  AND assertion.valid_from <= $2
                  AND assertion.valid_through >= $2
            ) AS status
        ), conflict_state AS (
            SELECT EXISTS (
                SELECT 1
                FROM evaluated_versions AS version
                WHERE (
                    version.terminal_signal_count > 0
                    AND version.supported_terminal_count = 0
                ) OR (
                    version.terminal_signal_count = 0
                    AND (
                        version.invalid_status_count > 0
                        OR version.effective_status_count > 1
                    )
                )
            ) AS present
        ), candidates AS (
            SELECT version.*
            FROM evaluated_versions AS version
            WHERE version.terminal_signal_count = 0
              AND version.invalid_status_count = 0
              AND version.effective_status_count = 1
        ), candidate_count AS (
            SELECT pg_catalog.count(*)::pg_catalog.int4 AS value
            FROM candidates
        ), selected AS (
            SELECT candidate.*
            FROM candidates AS candidate
            ORDER BY candidate.legal_version_id
            LIMIT 1
        ), amendment_evidence_candidates AS (
            SELECT selected.legal_version_id,
                   selected.predecessor_version_id AS amends_version_id,
                   event.event_id AS claim_id,
                   event.event_date AS claim_date,
                   event.evidence_id,
                   event.review_record_sha256 AS claim_review_record_sha256,
                   source.locator AS evidence_locator,
                   source.statement_sha256 AS evidence_statement_sha256,
                   source.artifact_id,
                   source.artifact_blob_id,
                   source.artifact_sha256,
                   source.source_url,
                   source.source_authority,
                   source.artifact_retrieved_at
            FROM selected
            JOIN public.regulatory_legal_events AS event
              ON event.legal_version_id = selected.predecessor_version_id
             AND event.event_type = 'supersession'
             AND event.target_legal_version_id = selected.legal_version_id
             AND event.event_date <= $2
             AND event.validation_state = 'validated'
            JOIN evidence_sources AS source
              ON source.evidence_id = event.evidence_id
             AND source.authority_level = 'authoritative'
             AND source.fixture_only = false
            WHERE EXISTS (
                SELECT 1
                FROM public.regulatory_legal_version_artifacts AS version_artifact
                WHERE version_artifact.legal_version_id = selected.predecessor_version_id
                  AND version_artifact.artifact_id = source.artifact_id
                  AND version_artifact.source_role = 'legal_text'
            )
        ), amendment_evidence AS (
            SELECT candidate.*
            FROM amendment_evidence_candidates AS candidate
            WHERE (
                SELECT pg_catalog.count(*)
                FROM amendment_evidence_candidates
            ) = 1
        ), consolidation_evidence_candidates AS (
            SELECT selected.legal_version_id,
                   event.event_id AS claim_id,
                   event.event_date AS claim_date,
                   event.evidence_id,
                   event.review_record_sha256 AS claim_review_record_sha256,
                   source.locator AS evidence_locator,
                   source.statement_sha256 AS evidence_statement_sha256,
                   source.artifact_id,
                   source.artifact_blob_id,
                   source.artifact_sha256,
                   source.source_url,
                   source.source_authority,
                   source.artifact_retrieved_at
            FROM selected
            JOIN public.regulatory_legal_events AS event
              ON event.legal_version_id = selected.legal_version_id
             AND event.event_type = 'consolidation'
             AND event.event_date <= $2
             AND event.validation_state = 'validated'
            JOIN evidence_sources AS source
              ON source.evidence_id = event.evidence_id
             AND source.authority_level = 'authoritative'
             AND source.fixture_only = false
            WHERE EXISTS (
                SELECT 1
                FROM public.regulatory_legal_version_artifacts AS version_artifact
                WHERE version_artifact.legal_version_id = selected.legal_version_id
                  AND version_artifact.artifact_id = source.artifact_id
                  AND version_artifact.source_role = 'legal_text'
            )
        ), consolidation_evidence AS (
            SELECT candidate.*
            FROM consolidation_evidence_candidates AS candidate
            WHERE (
                SELECT pg_catalog.count(*)
                FROM consolidation_evidence_candidates
            ) = 1
        ), resolved_payload AS (
            SELECT selected.legal_version_id,
                   selected.version_key,
                   selected.legal_text_sha256,
                   selected.review_record_sha256 AS version_review_record_sha256,
                   amendment.amends_version_id,
                   CASE
                       WHEN selected.consolidation_state = 'unknown' THEN 'unknown'
                       WHEN consolidation.claim_id IS NOT NULL THEN selected.consolidation_state
                       ELSE 'unknown'
                   END AS consolidation_state,
                   (
                       pg_catalog.jsonb_build_array(
                           pg_catalog.jsonb_build_object(
                               'role', 'publication',
                               'claim_id', publication.claim_id,
                               'claim_date', publication.claim_date,
                               'evidence_id', publication.evidence_id,
                               'evidence_locator', publication.evidence_locator,
                               'evidence_statement_sha256', publication.evidence_statement_sha256,
                               'claim_review_record_sha256', publication.claim_review_record_sha256,
                               'artifact_id', publication.artifact_id,
                               'artifact_blob_id', publication.artifact_blob_id,
                               'artifact_sha256', publication.artifact_sha256,
                               'source_url', publication.source_url,
                               'source_authority', publication.source_authority,
                               'artifact_retrieved_at', publication.artifact_retrieved_at
                           ),
                           pg_catalog.jsonb_build_object(
                               'role', 'effective',
                               'claim_id', effective.claim_id,
                               'claim_date', effective.claim_date,
                               'evidence_id', effective.evidence_id,
                               'evidence_locator', effective.evidence_locator,
                               'evidence_statement_sha256', effective.evidence_statement_sha256,
                               'claim_review_record_sha256', effective.claim_review_record_sha256,
                               'artifact_id', effective.artifact_id,
                               'artifact_blob_id', effective.artifact_blob_id,
                               'artifact_sha256', effective.artifact_sha256,
                               'source_url', effective.source_url,
                               'source_authority', effective.source_authority,
                               'artifact_retrieved_at', effective.artifact_retrieved_at
                           ),
                           pg_catalog.jsonb_build_object(
                               'role', 'status',
                               'claim_id', status.assertion_id,
                               'valid_from', status.valid_from,
                               'valid_through', status.valid_through,
                               'evidence_id', status.evidence_id,
                               'evidence_locator', status_source.locator,
                               'evidence_statement_sha256', status_source.statement_sha256,
                               'claim_review_record_sha256', status.review_record_sha256,
                               'artifact_id', status_source.artifact_id,
                               'artifact_blob_id', status_source.artifact_blob_id,
                               'artifact_sha256', status_source.artifact_sha256,
                               'source_url', status_source.source_url,
                               'source_authority', status_source.source_authority,
                               'artifact_retrieved_at', status_source.artifact_retrieved_at
                           )
                       )
                       || CASE WHEN amendment.claim_id IS NULL THEN '[]'::pg_catalog.jsonb ELSE
                           pg_catalog.jsonb_build_array(
                               pg_catalog.jsonb_build_object(
                                   'role', 'predecessor_supersession',
                                   'claim_id', amendment.claim_id,
                                   'claim_date', amendment.claim_date,
                                   'evidence_id', amendment.evidence_id,
                                   'evidence_locator', amendment.evidence_locator,
                                   'evidence_statement_sha256', amendment.evidence_statement_sha256,
                                   'claim_review_record_sha256', amendment.claim_review_record_sha256,
                                   'artifact_id', amendment.artifact_id,
                                   'artifact_blob_id', amendment.artifact_blob_id,
                                   'artifact_sha256', amendment.artifact_sha256,
                                   'source_url', amendment.source_url,
                                   'source_authority', amendment.source_authority,
                                   'artifact_retrieved_at', amendment.artifact_retrieved_at
                               )
                           )
                       END
                       || CASE WHEN consolidation.claim_id IS NULL THEN '[]'::pg_catalog.jsonb ELSE
                           pg_catalog.jsonb_build_array(
                               pg_catalog.jsonb_build_object(
                                   'role', 'consolidation',
                                   'claim_id', consolidation.claim_id,
                                   'claim_date', consolidation.claim_date,
                                   'evidence_id', consolidation.evidence_id,
                                   'evidence_locator', consolidation.evidence_locator,
                                   'evidence_statement_sha256', consolidation.evidence_statement_sha256,
                                   'claim_review_record_sha256', consolidation.claim_review_record_sha256,
                                   'artifact_id', consolidation.artifact_id,
                                   'artifact_blob_id', consolidation.artifact_blob_id,
                                   'artifact_sha256', consolidation.artifact_sha256,
                                   'source_url', consolidation.source_url,
                                   'source_authority', consolidation.source_authority,
                                   'artifact_retrieved_at', consolidation.artifact_retrieved_at
                               )
                           )
                       END
                   )::pg_catalog.text AS evidence_json
            FROM selected
            JOIN publication_ready AS publication
              ON publication.legal_version_id = selected.legal_version_id
            JOIN effective_ready AS effective
              ON effective.legal_version_id = selected.legal_version_id
            JOIN public.regulatory_legal_status_assertions AS status
              ON status.assertion_id = selected.effective_assertion_id
            JOIN evidence_sources AS status_source
              ON status_source.evidence_id = status.evidence_id
            LEFT JOIN amendment_evidence AS amendment
              ON amendment.legal_version_id = selected.legal_version_id
            LEFT JOIN consolidation_evidence AS consolidation
              ON consolidation.legal_version_id = selected.legal_version_id
        ), decision AS (
            SELECT CASE
                       WHEN NOT EXISTS (SELECT 1 FROM instrument_row) THEN 'instrument_not_found'
                       WHEN (SELECT present FROM fixture_signal) THEN 'fixture_only_data'
                       WHEN NOT EXISTS (SELECT 1 FROM validated_versions) THEN 'no_validated_version'
                       WHEN (SELECT present FROM conflict_state) THEN 'conflicting_status_evidence'
                       WHEN (SELECT value FROM candidate_count) > 1 THEN 'ambiguous_validated_versions'
                       WHEN (SELECT value FROM candidate_count) = 0 THEN 'status_not_validated_for_date'
                       ELSE 'resolved'
                   END AS reason
        )
        SELECT decision.reason = 'resolved' AS resolved,
               decision.reason,
               $1 AS instrument_id,
               $2 AS as_of,
               CASE WHEN decision.reason = 'resolved' THEN payload.legal_version_id END,
               CASE WHEN decision.reason = 'resolved' THEN payload.version_key END,
               CASE WHEN decision.reason = 'resolved' THEN payload.legal_text_sha256 END,
               CASE WHEN decision.reason = 'resolved' THEN payload.version_review_record_sha256 END,
               CASE WHEN decision.reason = 'resolved' THEN payload.amends_version_id END,
               CASE WHEN decision.reason = 'resolved' THEN payload.consolidation_state END,
               CASE WHEN decision.reason = 'resolved' THEN payload.evidence_json ELSE '[]' END
        FROM decision
        LEFT JOIN resolved_payload AS payload ON decision.reason = 'resolved'
        $function$
        """,
        """
        REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.resolve_regulation_status(
            pg_catalog.text,
            pg_catalog.date
        ) FROM PUBLIC
        """,
    ),
)
