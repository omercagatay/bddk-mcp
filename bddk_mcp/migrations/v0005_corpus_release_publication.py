"""Migration 0005: immutable verified-corpus release publication evidence.

The two-table model deliberately separates immutable release evidence from a
successful activation event.  An active identity is exposed only while the
current corpus still has the exact privacy-safe state fingerprint captured by
that activation.  It is therefore not possible for a failed replacement or a
later corpus mutation to leave an older release looking active.
"""

from typing import Final

from bddk_mcp.corpus_coordination import CORPUS_MUTATION_ADVISORY_KEY
from bddk_mcp.migrations.model import Migration

CORPUS_EPOCH_TRACKED_TABLES: Final[tuple[str, ...]] = (
    "decision_cache",
    "documents",
    "document_sections",
    "document_versions",
    "document_chunks",
    "document_retrieval_publications",
    "regulatory_instruments",
    "regulatory_family_imports",
    "regulatory_source_blobs",
    "regulatory_source_artifacts",
    "regulatory_evidence",
    "regulatory_legal_versions",
    "regulatory_legal_version_artifacts",
    "regulatory_legal_events",
    "regulatory_legal_status_assertions",
    "regulatory_provisions",
    "regulatory_legal_version_provisions",
)

V0005_CORPUS_RELEASE_PUBLICATION = Migration(
    version=5,
    name="verified_corpus_release_publication",
    statements=(
        """
        CREATE TABLE bddk_meta.corpus_state_epoch (
            singleton_id pg_catalog.bool PRIMARY KEY DEFAULT true,
            epoch pg_catalog.int8 NOT NULL DEFAULT 0,
            CONSTRAINT corpus_state_epoch_singleton_check CHECK (singleton_id),
            CONSTRAINT corpus_state_epoch_nonnegative_check CHECK (epoch >= 0)
        )
        """,
        """
        INSERT INTO bddk_meta.corpus_state_epoch (singleton_id, epoch)
        VALUES (true, 0)
        """,
        """
        CREATE TABLE bddk_meta.corpus_releases (
            release_id pg_catalog.text PRIMARY KEY,
            manifest_id pg_catalog.text NOT NULL,
            manifest_sha256 pg_catalog.text NOT NULL,
            signer_key_sha256 pg_catalog.text NOT NULL,
            freshness_policy_result pg_catalog.text NOT NULL,
            source_detection_slo_seconds pg_catalog.int4 NOT NULL,
            publication_slo_seconds pg_catalog.int4 NOT NULL,
            max_manifest_age_seconds pg_catalog.int4 NOT NULL,
            retrieval_profile_sha256 pg_catalog.text NOT NULL,
            corpus_state_sha256 pg_catalog.text NOT NULL,
            created_at pg_catalog.timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT corpus_releases_id_check
                CHECK (release_id ~ '^corpus_release_sha256_[0-9a-f]{64}$'),
            CONSTRAINT corpus_releases_manifest_id_check
                CHECK (manifest_id ~ '^[a-z0-9][a-z0-9._-]{2,127}$'),
            CONSTRAINT corpus_releases_manifest_hash_check
                CHECK (manifest_sha256 ~ '^[0-9a-f]{64}$'),
            CONSTRAINT corpus_releases_signer_hash_check
                CHECK (signer_key_sha256 ~ '^[0-9a-f]{64}$'),
            CONSTRAINT corpus_releases_policy_result_check
                CHECK (freshness_policy_result = 'quantified_measured_signature_verified_pass'),
            CONSTRAINT corpus_releases_source_detection_slo_check
                CHECK (source_detection_slo_seconds > 0),
            CONSTRAINT corpus_releases_publication_slo_check
                CHECK (publication_slo_seconds > 0),
            CONSTRAINT corpus_releases_max_age_check
                CHECK (max_manifest_age_seconds > 0),
            CONSTRAINT corpus_releases_profile_hash_check
                CHECK (retrieval_profile_sha256 ~ '^[0-9a-f]{64}$'),
            CONSTRAINT corpus_releases_state_hash_check
                CHECK (corpus_state_sha256 ~ '^[0-9a-f]{64}$')
        )
        """,
        """
        CREATE TABLE bddk_meta.corpus_release_activations (
            activation_sequence pg_catalog.int8 GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            release_id pg_catalog.text NOT NULL,
            corpus_epoch pg_catalog.int8 NOT NULL,
            completed_at pg_catalog.timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
            actor_fingerprint_sha256 pg_catalog.text NOT NULL,
            CONSTRAINT corpus_release_activations_release_fk
                FOREIGN KEY (release_id)
                REFERENCES bddk_meta.corpus_releases(release_id),
            CONSTRAINT corpus_release_activations_actor_hash_check
                CHECK (actor_fingerprint_sha256 ~ '^[0-9a-f]{64}$')
        )
        """,
        """
        CREATE FUNCTION bddk_meta.bump_corpus_state_epoch()
        RETURNS trigger
        LANGUAGE plpgsql
        VOLATILE
        PARALLEL UNSAFE
        SECURITY DEFINER
        SET search_path = pg_catalog
        AS $function$
        BEGIN
            UPDATE bddk_meta.corpus_state_epoch
            SET epoch = epoch + 1
            WHERE singleton_id;
            RETURN NULL;
        END
        $function$
        """,
        *(
            f"""
            CREATE TRIGGER bump_corpus_state_epoch_on_change
            AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON public.{table_name}
            FOR EACH STATEMENT EXECUTE FUNCTION bddk_meta.bump_corpus_state_epoch()
            """
            for table_name in CORPUS_EPOCH_TRACKED_TABLES
        ),
        """
        DROP TRIGGER IF EXISTS invalidate_retrieval_publication_on_chunk_change
        ON public.document_chunks
        """,
        """
        CREATE OR REPLACE FUNCTION public.invalidate_retrieval_publication()
        RETURNS trigger
        LANGUAGE plpgsql
        SET search_path = pg_catalog, public
        AS $function$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                DELETE FROM public.document_retrieval_publications AS publication
                WHERE publication.doc_id IN (
                    SELECT DISTINCT changed.doc_id
                    FROM changed_chunks AS changed
                );
            ELSIF TG_OP = 'DELETE' THEN
                DELETE FROM public.document_retrieval_publications AS publication
                WHERE publication.doc_id IN (
                    SELECT DISTINCT changed.doc_id
                    FROM changed_chunks AS changed
                );
            ELSE
                DELETE FROM public.document_retrieval_publications AS publication
                WHERE publication.doc_id IN (
                    SELECT old_chunk.doc_id FROM old_chunks AS old_chunk
                    UNION
                    SELECT new_chunk.doc_id FROM new_chunks AS new_chunk
                );
            END IF;
            RETURN NULL;
        END
        $function$
        """,
        """
        CREATE TRIGGER invalidate_retrieval_publication_on_chunk_insert
        AFTER INSERT ON public.document_chunks
        REFERENCING NEW TABLE AS changed_chunks
        FOR EACH STATEMENT EXECUTE FUNCTION public.invalidate_retrieval_publication()
        """,
        """
        CREATE TRIGGER invalidate_retrieval_publication_on_chunk_delete
        AFTER DELETE ON public.document_chunks
        REFERENCING OLD TABLE AS changed_chunks
        FOR EACH STATEMENT EXECUTE FUNCTION public.invalidate_retrieval_publication()
        """,
        """
        CREATE TRIGGER invalidate_retrieval_publication_on_chunk_update
        AFTER UPDATE ON public.document_chunks
        REFERENCING OLD TABLE AS old_chunks NEW TABLE AS new_chunks
        FOR EACH STATEMENT EXECUTE FUNCTION public.invalidate_retrieval_publication()
        """,
        """
        CREATE FUNCTION bddk_meta.corpus_fingerprint_frame(value pg_catalog.text)
        RETURNS pg_catalog.bytea
        LANGUAGE sql
        IMMUTABLE
        PARALLEL SAFE
        SET search_path = pg_catalog
        AS $function$
        SELECT CASE
            WHEN value IS NULL THEN pg_catalog.decode('00', 'hex')
            ELSE pg_catalog.decode('01', 'hex')
                 || pg_catalog.int8send(
                        pg_catalog.octet_length(pg_catalog.convert_to(value, 'UTF8'))::pg_catalog.int8
                    )
                 || pg_catalog.convert_to(value, 'UTF8')
        END
        $function$
        """,
        """
        CREATE FUNCTION bddk_meta.current_corpus_state_sha256(
            requested_retrieval_profile_sha256 pg_catalog.text
        )
        RETURNS pg_catalog.text
        LANGUAGE sql
        STABLE
        PARALLEL UNSAFE
        SECURITY DEFINER
        SET search_path = pg_catalog
        AS $function$
        WITH state_items(item_kind, item_id, payload_sha256) AS (
            SELECT 'decision_cache', cache.document_id,
                   pg_catalog.sha256(
                       bddk_meta.corpus_fingerprint_frame(cache.title)
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(
                                  pg_catalog.sha256(pg_catalog.convert_to(cache.content, 'UTF8')),
                                  'hex'
                              )
                          )
                       || bddk_meta.corpus_fingerprint_frame(cache.decision_date)
                       || bddk_meta.corpus_fingerprint_frame(cache.decision_number)
                       || bddk_meta.corpus_fingerprint_frame(cache.category)
                       || bddk_meta.corpus_fingerprint_frame(cache.source_url)
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(pg_catalog.float8send(cache.cached_at), 'hex')
                          )
                   )
            FROM public.decision_cache AS cache
            UNION ALL
            SELECT 'document', document.document_id,
                   pg_catalog.sha256(
                       bddk_meta.corpus_fingerprint_frame(document.title)
                       || bddk_meta.corpus_fingerprint_frame(document.category)
                       || bddk_meta.corpus_fingerprint_frame(document.decision_date)
                       || bddk_meta.corpus_fingerprint_frame(document.decision_number)
                       || bddk_meta.corpus_fingerprint_frame(document.source_url)
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(pg_catalog.sha256(document.pdf_blob), 'hex')
                          )
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(
                                  pg_catalog.sha256(
                                      pg_catalog.convert_to(document.markdown_content, 'UTF8')
                                  ),
                                  'hex'
                              )
                          )
                       || bddk_meta.corpus_fingerprint_frame(document.content_hash)
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(pg_catalog.float8send(document.downloaded_at), 'hex')
                          )
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(pg_catalog.float8send(document.extracted_at), 'hex')
                          )
                       || bddk_meta.corpus_fingerprint_frame(document.extraction_method)
                       || bddk_meta.corpus_fingerprint_frame(document.total_pages::pg_catalog.text)
                       || bddk_meta.corpus_fingerprint_frame(document.file_size::pg_catalog.text)
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(
                                  pg_catalog.sha256(
                                      pg_catalog.convert_to(document.tsv::pg_catalog.text, 'UTF8')
                                  ),
                                  'hex'
                              )
                          )
                   )
            FROM public.documents AS document
            UNION ALL
            SELECT 'document_section', section.id::pg_catalog.text,
                   pg_catalog.sha256(
                       bddk_meta.corpus_fingerprint_frame(section.doc_id)
                       || bddk_meta.corpus_fingerprint_frame(section.section_type)
                       || bddk_meta.corpus_fingerprint_frame(section.section_ref)
                       || bddk_meta.corpus_fingerprint_frame(section.heading)
                       || bddk_meta.corpus_fingerprint_frame(section.start_char::pg_catalog.text)
                       || bddk_meta.corpus_fingerprint_frame(section.end_char::pg_catalog.text)
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(
                                  pg_catalog.sha256(pg_catalog.convert_to(section.content, 'UTF8')),
                                  'hex'
                              )
                          )
                       || bddk_meta.corpus_fingerprint_frame(section.content_hash)
                       || bddk_meta.corpus_fingerprint_frame(section.page_start::pg_catalog.text)
                       || bddk_meta.corpus_fingerprint_frame(section.page_end::pg_catalog.text)
                       || bddk_meta.corpus_fingerprint_frame(section.source_content_hash)
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(
                                  pg_catalog.sha256(
                                      pg_catalog.convert_to(section.tsv::pg_catalog.text, 'UTF8')
                                  ),
                                  'hex'
                              )
                          )
                   )
            FROM public.document_sections AS section
            UNION ALL
            SELECT 'document_version', version.id::pg_catalog.text,
                   pg_catalog.sha256(
                       bddk_meta.corpus_fingerprint_frame(version.document_id)
                       || bddk_meta.corpus_fingerprint_frame(version.version::pg_catalog.text)
                       || bddk_meta.corpus_fingerprint_frame(version.content_hash)
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(
                                  pg_catalog.sha256(
                                      pg_catalog.convert_to(version.markdown_content, 'UTF8')
                                  ),
                                  'hex'
                              )
                          )
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(pg_catalog.float8send(version.synced_at), 'hex')
                          )
                   )
            FROM public.document_versions AS version
            UNION ALL
            SELECT 'document_chunk', chunk.id::pg_catalog.text,
                   pg_catalog.sha256(
                       bddk_meta.corpus_fingerprint_frame(chunk.doc_id)
                       || bddk_meta.corpus_fingerprint_frame(chunk.chunk_index::pg_catalog.text)
                       || bddk_meta.corpus_fingerprint_frame(chunk.title)
                       || bddk_meta.corpus_fingerprint_frame(chunk.category)
                       || bddk_meta.corpus_fingerprint_frame(chunk.decision_date)
                       || bddk_meta.corpus_fingerprint_frame(chunk.decision_number)
                       || bddk_meta.corpus_fingerprint_frame(chunk.source_url)
                       || bddk_meta.corpus_fingerprint_frame(chunk.total_chunks::pg_catalog.text)
                       || bddk_meta.corpus_fingerprint_frame(chunk.total_pages::pg_catalog.text)
                       || bddk_meta.corpus_fingerprint_frame(chunk.content_hash)
                       || bddk_meta.corpus_fingerprint_frame(chunk.chunk_start_char::pg_catalog.text)
                       || bddk_meta.corpus_fingerprint_frame(chunk.chunk_end_char::pg_catalog.text)
                       || bddk_meta.corpus_fingerprint_frame(chunk.section_type)
                       || bddk_meta.corpus_fingerprint_frame(chunk.section_ref)
                       || bddk_meta.corpus_fingerprint_frame(chunk.section_start_char::pg_catalog.text)
                       || bddk_meta.corpus_fingerprint_frame(chunk.section_end_char::pg_catalog.text)
                       || bddk_meta.corpus_fingerprint_frame(chunk.section_content_hash)
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(
                                  pg_catalog.sha256(pg_catalog.convert_to(chunk.chunk_text, 'UTF8')),
                                  'hex'
                              )
                          )
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(
                                  pg_catalog.sha256(
                                      pg_catalog.convert_to(chunk.embedding::pg_catalog.text, 'UTF8')
                                  ),
                                  'hex'
                              )
                          )
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(
                                  pg_catalog.sha256(
                                      pg_catalog.convert_to(chunk.tsv::pg_catalog.text, 'UTF8')
                                  ),
                                  'hex'
                              )
                          )
                   )
            FROM public.document_chunks AS chunk
            UNION ALL
            SELECT 'regulatory_instrument', instrument.instrument_id,
                   pg_catalog.sha256(
                       bddk_meta.corpus_fingerprint_frame(instrument.jurisdiction)
                       || bddk_meta.corpus_fingerprint_frame(instrument.authority_code)
                       || bddk_meta.corpus_fingerprint_frame(instrument.identity_key)
                       || bddk_meta.corpus_fingerprint_frame(instrument.canonical_title)
                       || bddk_meta.corpus_fingerprint_frame(instrument.instrument_type)
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(
                                  pg_catalog.timestamptz_send(instrument.created_at),
                                  'hex'
                              )
                          )
                   )
            FROM public.regulatory_instruments AS instrument
            UNION ALL
            SELECT 'regulatory_family_import',
                   pg_catalog.encode(
                       pg_catalog.sha256(
                           bddk_meta.corpus_fingerprint_frame(import_record.bundle_id)
                           || bddk_meta.corpus_fingerprint_frame(import_record.bundle_sha256)
                       ),
                       'hex'
                   ),
                   pg_catalog.sha256(
                       bddk_meta.corpus_fingerprint_frame(import_record.bundle_id)
                       || bddk_meta.corpus_fingerprint_frame(import_record.bundle_sha256)
                       || bddk_meta.corpus_fingerprint_frame(import_record.instrument_id)
                       || bddk_meta.corpus_fingerprint_frame(
                              import_record.schema_version::pg_catalog.text
                          )
                       || bddk_meta.corpus_fingerprint_frame(
                              import_record.fixture_only::pg_catalog.text
                          )
                       || bddk_meta.corpus_fingerprint_frame(import_record.imported_by)
                       || bddk_meta.corpus_fingerprint_frame(import_record.imported_current_user)
                       || bddk_meta.corpus_fingerprint_frame(import_record.imported_session_user)
                       || bddk_meta.corpus_fingerprint_frame(
                              import_record.predecessor_bundle_sha256
                          )
                       || bddk_meta.corpus_fingerprint_frame(
                              import_record.member_manifest::pg_catalog.text
                          )
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(
                                  pg_catalog.timestamptz_send(import_record.imported_at),
                                  'hex'
                              )
                          )
                   )
            FROM public.regulatory_family_imports AS import_record
            UNION ALL
            SELECT 'regulatory_source_blob', blob.blob_id,
                   pg_catalog.sha256(
                       bddk_meta.corpus_fingerprint_frame(blob.content_sha256)
                   )
            FROM public.regulatory_source_blobs AS blob
            UNION ALL
            SELECT 'regulatory_source_artifact', artifact.artifact_id,
                   pg_catalog.sha256(
                       bddk_meta.corpus_fingerprint_frame(artifact.blob_id)
                       || bddk_meta.corpus_fingerprint_frame(artifact.canonical_uri)
                       || bddk_meta.corpus_fingerprint_frame(artifact.source_authority)
                       || bddk_meta.corpus_fingerprint_frame(artifact.media_type)
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(
                                  pg_catalog.timestamptz_send(artifact.retrieved_at),
                                  'hex'
                              )
                          )
                       || bddk_meta.corpus_fingerprint_frame(artifact.repository_document_id)
                       || bddk_meta.corpus_fingerprint_frame(
                              artifact.fixture_only::pg_catalog.text
                          )
                   )
            FROM public.regulatory_source_artifacts AS artifact
            UNION ALL
            SELECT 'regulatory_evidence', evidence.evidence_id,
                   pg_catalog.sha256(
                       bddk_meta.corpus_fingerprint_frame(evidence.artifact_id)
                       || bddk_meta.corpus_fingerprint_frame(evidence.locator)
                       || bddk_meta.corpus_fingerprint_frame(evidence.statement_sha256)
                       || bddk_meta.corpus_fingerprint_frame(evidence.authority_level)
                   )
            FROM public.regulatory_evidence AS evidence
            UNION ALL
            SELECT 'regulatory_legal_version', version.legal_version_id,
                   pg_catalog.sha256(
                       bddk_meta.corpus_fingerprint_frame(version.instrument_id)
                       || bddk_meta.corpus_fingerprint_frame(version.version_key)
                       || bddk_meta.corpus_fingerprint_frame(version.legal_text_sha256)
                       || bddk_meta.corpus_fingerprint_frame(version.predecessor_version_id)
                       || bddk_meta.corpus_fingerprint_frame(version.consolidation_state)
                       || bddk_meta.corpus_fingerprint_frame(version.validation_state)
                       || bddk_meta.corpus_fingerprint_frame(version.validated_by)
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(
                                  pg_catalog.timestamptz_send(version.validated_at),
                                  'hex'
                              )
                          )
                       || bddk_meta.corpus_fingerprint_frame(version.validation_method)
                       || bddk_meta.corpus_fingerprint_frame(version.review_record_sha256)
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(
                                  pg_catalog.timestamptz_send(version.created_at),
                                  'hex'
                              )
                          )
                   )
            FROM public.regulatory_legal_versions AS version
            UNION ALL
            SELECT 'regulatory_legal_version_artifact',
                   pg_catalog.encode(
                       pg_catalog.sha256(
                           bddk_meta.corpus_fingerprint_frame(version_artifact.legal_version_id)
                           || bddk_meta.corpus_fingerprint_frame(version_artifact.artifact_id)
                           || bddk_meta.corpus_fingerprint_frame(version_artifact.source_role)
                       ),
                       'hex'
                   ),
                   pg_catalog.sha256(
                       bddk_meta.corpus_fingerprint_frame(version_artifact.legal_version_id)
                       || bddk_meta.corpus_fingerprint_frame(version_artifact.artifact_id)
                       || bddk_meta.corpus_fingerprint_frame(version_artifact.source_role)
                   )
            FROM public.regulatory_legal_version_artifacts AS version_artifact
            UNION ALL
            SELECT 'regulatory_legal_event', event.event_id,
                   pg_catalog.sha256(
                       bddk_meta.corpus_fingerprint_frame(event.legal_version_id)
                       || bddk_meta.corpus_fingerprint_frame(event.event_type)
                       || bddk_meta.corpus_fingerprint_frame(event.event_date::pg_catalog.text)
                       || bddk_meta.corpus_fingerprint_frame(event.evidence_id)
                       || bddk_meta.corpus_fingerprint_frame(event.target_legal_version_id)
                       || bddk_meta.corpus_fingerprint_frame(event.validation_state)
                       || bddk_meta.corpus_fingerprint_frame(event.validated_by)
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(
                                  pg_catalog.timestamptz_send(event.validated_at),
                                  'hex'
                              )
                          )
                       || bddk_meta.corpus_fingerprint_frame(event.validation_method)
                       || bddk_meta.corpus_fingerprint_frame(event.review_record_sha256)
                   )
            FROM public.regulatory_legal_events AS event
            UNION ALL
            SELECT 'regulatory_legal_status_assertion', assertion.assertion_id,
                   pg_catalog.sha256(
                       bddk_meta.corpus_fingerprint_frame(assertion.legal_version_id)
                       || bddk_meta.corpus_fingerprint_frame(assertion.legal_status)
                       || bddk_meta.corpus_fingerprint_frame(assertion.valid_from::pg_catalog.text)
                       || bddk_meta.corpus_fingerprint_frame(assertion.valid_through::pg_catalog.text)
                       || bddk_meta.corpus_fingerprint_frame(assertion.evidence_id)
                       || bddk_meta.corpus_fingerprint_frame(assertion.validation_state)
                       || bddk_meta.corpus_fingerprint_frame(assertion.validated_by)
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(
                                  pg_catalog.timestamptz_send(assertion.validated_at),
                                  'hex'
                              )
                          )
                       || bddk_meta.corpus_fingerprint_frame(assertion.validation_method)
                       || bddk_meta.corpus_fingerprint_frame(assertion.review_record_sha256)
                   )
            FROM public.regulatory_legal_status_assertions AS assertion
            UNION ALL
            SELECT 'regulatory_provision', provision.provision_id,
                   pg_catalog.sha256(
                       bddk_meta.corpus_fingerprint_frame(provision.instrument_id)
                       || bddk_meta.corpus_fingerprint_frame(provision.provision_kind)
                       || bddk_meta.corpus_fingerprint_frame(provision.canonical_path)
                   )
            FROM public.regulatory_provisions AS provision
            UNION ALL
            SELECT 'regulatory_legal_version_provision',
                   pg_catalog.encode(
                       pg_catalog.sha256(
                           bddk_meta.corpus_fingerprint_frame(occurrence.legal_version_id)
                           || bddk_meta.corpus_fingerprint_frame(occurrence.provision_id)
                       ),
                       'hex'
                   ),
                   pg_catalog.sha256(
                       bddk_meta.corpus_fingerprint_frame(occurrence.legal_version_id)
                       || bddk_meta.corpus_fingerprint_frame(occurrence.provision_id)
                       || bddk_meta.corpus_fingerprint_frame(occurrence.provision_text_sha256)
                       || bddk_meta.corpus_fingerprint_frame(
                              occurrence.document_section_id::pg_catalog.text
                          )
                       || bddk_meta.corpus_fingerprint_frame(occurrence.evidence_id)
                       || bddk_meta.corpus_fingerprint_frame(occurrence.validation_state)
                       || bddk_meta.corpus_fingerprint_frame(occurrence.validated_by)
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(
                                  pg_catalog.timestamptz_send(occurrence.validated_at),
                                  'hex'
                              )
                          )
                       || bddk_meta.corpus_fingerprint_frame(occurrence.validation_method)
                       || bddk_meta.corpus_fingerprint_frame(occurrence.review_record_sha256)
                   )
            FROM public.regulatory_legal_version_provisions AS occurrence
            UNION ALL
            SELECT 'retrieval_publication', publication.doc_id,
                   pg_catalog.sha256(
                       bddk_meta.corpus_fingerprint_frame(publication.content_hash)
                       || bddk_meta.corpus_fingerprint_frame(publication.retrieval_profile_hash)
                       || bddk_meta.corpus_fingerprint_frame(publication.expected_chunks::pg_catalog.text)
                       || bddk_meta.corpus_fingerprint_frame(
                              pg_catalog.encode(
                                  pg_catalog.timestamptz_send(publication.published_at),
                                  'hex'
                              )
                          )
                   )
            FROM public.document_retrieval_publications AS publication
            WHERE publication.retrieval_profile_hash = requested_retrieval_profile_sha256
        ), combined AS (
            SELECT COALESCE(
                       pg_catalog.string_agg(
                           pg_catalog.sha256(
                               bddk_meta.corpus_fingerprint_frame(item_kind)
                               || bddk_meta.corpus_fingerprint_frame(item_id)
                               || payload_sha256
                           ),
                           pg_catalog.decode('', 'hex')
                           ORDER BY item_kind, item_id
                       ),
                       pg_catalog.decode('', 'hex')
                   ) AS item_hashes
            FROM state_items
        )
        SELECT pg_catalog.encode(
                   pg_catalog.sha256(
                       bddk_meta.corpus_fingerprint_frame(requested_retrieval_profile_sha256)
                       || combined.item_hashes
                   ),
                   'hex'
               )
        FROM combined
        $function$
        """,
        """
        CREATE FUNCTION bddk_meta.corpus_retrieval_ready(
            requested_retrieval_profile_sha256 pg_catalog.text
        )
        RETURNS pg_catalog.bool
        LANGUAGE sql
        STABLE
        PARALLEL UNSAFE
        SECURITY DEFINER
        SET search_path = pg_catalog
        AS $function$
        WITH section_integrity AS (
            SELECT section.*,
                   document.content_hash AS document_content_hash,
                   pg_catalog.char_length(document.markdown_content) AS document_length,
                   pg_catalog.btrim(
                       pg_catalog.substr(
                           document.markdown_content,
                           section.start_char + 1,
                           section.end_char - section.start_char
                       ),
                       E' \t\n\r\f\v'
                   ) AS source_content,
                   pg_catalog.regexp_replace(
                       section.content,
                       E'\\n\\n\\[BÖLÜM KESİLDİ: içerik [0-9]+ karakterden 20000 karaktere '
                       'kısaltıldı — tam metin için get_bddk_document kullanın\\]$',
                       ''
                   ) AS content_without_truncation_marker,
                   section.content ~
                       E'\\n\\n\\[BÖLÜM KESİLDİ: içerik [0-9]+ karakterden 20000 karaktere '
                       'kısaltıldı — tam metin için get_bddk_document kullanın\\]$'
                       AS has_valid_truncation_marker
            FROM public.document_sections AS section
            LEFT JOIN public.documents AS document
              ON document.document_id = section.doc_id
        ), chunk_integrity AS (
            SELECT publication.doc_id,
                   publication.content_hash,
                   publication.expected_chunks,
                   pg_catalog.count(chunk.id)::pg_catalog.int4 AS actual_chunks,
                   pg_catalog.min(chunk.chunk_index)::pg_catalog.int4 AS first_index,
                   pg_catalog.max(chunk.chunk_index)::pg_catalog.int4 AS last_index,
                   pg_catalog.bool_and(chunk.content_hash = publication.content_hash) AS hashes_match,
                   pg_catalog.bool_and(chunk.total_chunks = publication.expected_chunks) AS totals_match,
                   pg_catalog.bool_and(chunk.embedding IS NOT NULL) AS embeddings_complete
            FROM public.document_retrieval_publications AS publication
            LEFT JOIN public.document_chunks AS chunk
              ON chunk.doc_id = publication.doc_id
            WHERE publication.retrieval_profile_hash = requested_retrieval_profile_sha256
            GROUP BY publication.doc_id, publication.content_hash, publication.expected_chunks
        )
        SELECT requested_retrieval_profile_sha256 ~ '^[0-9a-f]{64}$'
           AND EXISTS (SELECT 1 FROM public.decision_cache LIMIT 1)
           AND EXISTS (SELECT 1 FROM public.documents LIMIT 1)
           AND EXISTS (SELECT 1 FROM public.document_sections LIMIT 1)
           AND EXISTS (SELECT 1 FROM public.document_chunks LIMIT 1)
           AND NOT EXISTS (
                SELECT 1
                FROM public.documents AS document
                WHERE COALESCE(document.markdown_content, '') = ''
                   OR document.tsv IS DISTINCT FROM (
                       pg_catalog.to_tsvector(
                           'simple'::pg_catalog.regconfig,
                           public.immutable_unaccent(COALESCE(document.title, ''))
                       )
                       || pg_catalog.to_tsvector(
                           'simple'::pg_catalog.regconfig,
                           public.immutable_unaccent(COALESCE(document.markdown_content, ''))
                       )
                       || pg_catalog.to_tsvector(
                           'simple'::pg_catalog.regconfig,
                           public.immutable_unaccent(COALESCE(document.category, ''))
                       )
                   )
                   OR document.content_hash !~ '^[0-9a-f]{64}$'
                   OR document.content_hash IS DISTINCT FROM pg_catalog.encode(
                       pg_catalog.sha256(
                           pg_catalog.convert_to(document.markdown_content, 'UTF8')
                       ),
                       'hex'
                   )
                   OR NOT EXISTS (
                       SELECT 1
                       FROM public.document_chunks AS chunk
                       WHERE chunk.doc_id = document.document_id
                   )
                   OR NOT EXISTS (
                       SELECT 1
                       FROM public.document_retrieval_publications AS publication
                       WHERE publication.doc_id = document.document_id
                         AND publication.content_hash = document.content_hash
                         AND publication.retrieval_profile_hash = requested_retrieval_profile_sha256
                   )
                LIMIT 1
           )
           AND NOT EXISTS (
                SELECT 1
                FROM section_integrity AS section
                WHERE section.document_content_hash IS NULL
                   OR section.tsv IS DISTINCT FROM (
                       pg_catalog.to_tsvector(
                           'simple'::pg_catalog.regconfig,
                           public.immutable_unaccent(COALESCE(section.heading, ''))
                       )
                       || pg_catalog.to_tsvector(
                           'simple'::pg_catalog.regconfig,
                           public.immutable_unaccent(COALESCE(section.content, ''))
                       )
                   )
                   OR section.start_char < 0
                   OR section.end_char <= section.start_char
                   OR section.end_char > section.document_length
                   OR section.source_content_hash IS DISTINCT FROM section.document_content_hash
                   OR section.content_hash !~ '^[0-9a-f]{64}$'
                   OR section.content_hash IS DISTINCT FROM pg_catalog.encode(
                       pg_catalog.sha256(pg_catalog.convert_to(section.content, 'UTF8')),
                       'hex'
                   )
                   OR (
                       section.content IS DISTINCT FROM section.source_content
                       AND NOT (
                           section.end_char - section.start_char = 20000
                           AND section.has_valid_truncation_marker
                           AND section.content_without_truncation_marker = section.source_content
                       )
                   )
                LIMIT 1
           )
           AND NOT EXISTS (
                SELECT 1
                FROM public.document_chunks AS chunk
                LEFT JOIN public.documents AS document
                  ON document.document_id = chunk.doc_id
                WHERE document.document_id IS NULL
                   OR chunk.tsv IS DISTINCT FROM (
                       pg_catalog.to_tsvector(
                           'simple'::pg_catalog.regconfig,
                           public.immutable_unaccent(COALESCE(chunk.title, ''))
                       )
                       || pg_catalog.to_tsvector(
                           'simple'::pg_catalog.regconfig,
                           public.immutable_unaccent(COALESCE(chunk.chunk_text, ''))
                       )
                   )
                   OR chunk.embedding IS NULL
                   OR chunk.content_hash IS DISTINCT FROM document.content_hash
                   OR chunk.title IS DISTINCT FROM document.title
                   OR chunk.category IS DISTINCT FROM document.category
                   OR chunk.decision_date IS DISTINCT FROM document.decision_date
                   OR chunk.decision_number IS DISTINCT FROM document.decision_number
                   OR chunk.source_url IS DISTINCT FROM document.source_url
                   OR chunk.total_pages IS DISTINCT FROM document.total_pages
                   OR chunk.chunk_start_char IS NULL
                   OR chunk.chunk_end_char IS NULL
                   OR chunk.chunk_start_char < 0
                   OR chunk.chunk_end_char <= chunk.chunk_start_char
                   OR chunk.chunk_end_char > pg_catalog.char_length(document.markdown_content)
                   OR chunk.chunk_text IS DISTINCT FROM pg_catalog.substr(
                       document.markdown_content,
                       chunk.chunk_start_char + 1,
                       chunk.chunk_end_char - chunk.chunk_start_char
                   )
                   OR NOT (
                       (
                           chunk.section_type = ''
                           AND chunk.section_ref = ''
                           AND chunk.section_start_char IS NULL
                           AND chunk.section_end_char IS NULL
                           AND chunk.section_content_hash = ''
                           AND NOT EXISTS (
                               SELECT 1
                               FROM public.document_sections AS overlapping_section
                               WHERE overlapping_section.doc_id = chunk.doc_id
                                 AND chunk.chunk_start_char < overlapping_section.end_char
                                 AND chunk.chunk_end_char > overlapping_section.start_char
                           )
                       )
                       OR EXISTS (
                           SELECT 1
                           FROM public.document_sections AS linked_section
                           WHERE linked_section.doc_id = chunk.doc_id
                             AND linked_section.section_type = chunk.section_type
                             AND linked_section.section_ref = chunk.section_ref
                             AND linked_section.start_char = chunk.section_start_char
                             AND linked_section.end_char = chunk.section_end_char
                             AND linked_section.content_hash = chunk.section_content_hash
                             AND chunk.chunk_start_char < linked_section.end_char
                             AND chunk.chunk_end_char > linked_section.start_char
                       )
                   )
                LIMIT 1
           )
           AND NOT EXISTS (
                SELECT 1
                FROM chunk_integrity
                WHERE actual_chunks <> expected_chunks
                   OR first_index <> 0
                   OR last_index <> expected_chunks - 1
                   OR NOT COALESCE(hashes_match, false)
                   OR NOT COALESCE(totals_match, false)
                   OR NOT COALESCE(embeddings_complete, false)
                LIMIT 1
           )
        $function$
        """,
        """
        CREATE FUNCTION bddk_meta.reject_corpus_release_mutation()
        RETURNS trigger
        LANGUAGE plpgsql
        SET search_path = pg_catalog
        AS $function$
        BEGIN
            RAISE EXCEPTION 'corpus release evidence is append-only'
                USING ERRCODE = '55000';
        END
        $function$
        """,
        """
        CREATE TRIGGER reject_corpus_release_update_delete
        BEFORE UPDATE OR DELETE ON bddk_meta.corpus_releases
        FOR EACH ROW EXECUTE FUNCTION bddk_meta.reject_corpus_release_mutation()
        """,
        """
        CREATE TRIGGER reject_corpus_release_activation_update_delete
        BEFORE UPDATE OR DELETE ON bddk_meta.corpus_release_activations
        FOR EACH ROW EXECUTE FUNCTION bddk_meta.reject_corpus_release_mutation()
        """,
        """
        CREATE FUNCTION bddk_meta.publish_verified_corpus_release(
            requested_manifest_id pg_catalog.text,
            requested_manifest_sha256 pg_catalog.text,
            requested_signer_key_sha256 pg_catalog.text,
            requested_source_detection_slo_seconds pg_catalog.int4,
            requested_publication_slo_seconds pg_catalog.int4,
            requested_max_manifest_age_seconds pg_catalog.int4,
            requested_retrieval_profile_sha256 pg_catalog.text
        )
        RETURNS TABLE (
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
            completed_at pg_catalog.timestamptz
        )
        LANGUAGE plpgsql
        VOLATILE
        PARALLEL UNSAFE
        SECURITY DEFINER
        SET search_path = pg_catalog
        AS $function$
        DECLARE
            selected_state_sha256 pg_catalog.text;
            selected_release_id pg_catalog.text;
            selected_actor_fingerprint pg_catalog.text;
            selected_corpus_epoch pg_catalog.int8;
        BEGIN
            IF pg_catalog.to_regrole('bddk_release_publisher') IS NULL THEN
                RAISE EXCEPTION 'verified corpus release caller is not authorized'
                    USING ERRCODE = '42501';
            END IF;
            IF NOT pg_catalog.pg_has_role(
                SESSION_USER,
                pg_catalog.to_regrole('bddk_release_publisher'),
                'MEMBER'
            ) THEN
                RAISE EXCEPTION 'verified corpus release caller is not authorized'
                    USING ERRCODE = '42501';
            END IF;
            IF requested_manifest_id IS NULL
               OR requested_manifest_sha256 IS NULL
               OR requested_signer_key_sha256 IS NULL
               OR requested_source_detection_slo_seconds IS NULL
               OR requested_publication_slo_seconds IS NULL
               OR requested_max_manifest_age_seconds IS NULL
               OR requested_retrieval_profile_sha256 IS NULL
               OR requested_manifest_id !~ '^[a-z0-9][a-z0-9._-]{2,127}$'
               OR requested_manifest_sha256 !~ '^[0-9a-f]{64}$'
               OR requested_signer_key_sha256 !~ '^[0-9a-f]{64}$'
               OR requested_retrieval_profile_sha256 !~ '^[0-9a-f]{64}$'
               OR requested_source_detection_slo_seconds <= 0
               OR requested_publication_slo_seconds <= 0
               OR requested_max_manifest_age_seconds <= 0 THEN
                RAISE EXCEPTION 'verified corpus release identity is invalid'
                    USING ERRCODE = '22023';
            END IF;

            PERFORM pg_catalog.pg_advisory_xact_lock(
                __CORPUS_MUTATION_ADVISORY_KEY__::pg_catalog.int8
            );
            LOCK TABLE bddk_meta.corpus_release_activations IN SHARE ROW EXCLUSIVE MODE;
            LOCK TABLE public.decision_cache,
                       public.documents,
                       public.document_sections,
                       public.document_versions,
                       public.document_chunks,
                       public.document_retrieval_publications,
                       public.regulatory_instruments,
                       public.regulatory_family_imports,
                       public.regulatory_source_blobs,
                       public.regulatory_source_artifacts,
                       public.regulatory_evidence,
                       public.regulatory_legal_versions,
                       public.regulatory_legal_version_artifacts,
                       public.regulatory_legal_events,
                       public.regulatory_legal_status_assertions,
                       public.regulatory_provisions,
                       public.regulatory_legal_version_provisions
                IN SHARE MODE;

            SELECT epoch.epoch
            INTO STRICT selected_corpus_epoch
            FROM bddk_meta.corpus_state_epoch AS epoch
            WHERE epoch.singleton_id;

            IF NOT bddk_meta.corpus_retrieval_ready(requested_retrieval_profile_sha256) THEN
                RAISE EXCEPTION 'verified corpus release is not retrieval-ready'
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
                selected_release_id,
                requested_manifest_id,
                requested_manifest_sha256,
                requested_signer_key_sha256,
                'quantified_measured_signature_verified_pass',
                requested_source_detection_slo_seconds,
                requested_publication_slo_seconds,
                requested_max_manifest_age_seconds,
                requested_retrieval_profile_sha256,
                selected_state_sha256
            ) ON CONFLICT ON CONSTRAINT corpus_releases_pkey DO NOTHING;

            IF NOT EXISTS (
                SELECT 1
                FROM bddk_meta.corpus_release_activations AS activation
                WHERE activation.activation_sequence = (
                    SELECT MAX(latest.activation_sequence)
                    FROM bddk_meta.corpus_release_activations AS latest
                )
                  AND activation.release_id = selected_release_id
                  AND activation.corpus_epoch = selected_corpus_epoch
            ) THEN
                INSERT INTO bddk_meta.corpus_release_activations (
                    release_id,
                    corpus_epoch,
                    actor_fingerprint_sha256
                ) VALUES (
                    selected_release_id,
                    selected_corpus_epoch,
                    selected_actor_fingerprint
                );
            END IF;

            RETURN QUERY
            SELECT release.release_id,
                   release.manifest_id,
                   release.manifest_sha256,
                   release.signer_key_sha256,
                   release.freshness_policy_result,
                   release.source_detection_slo_seconds,
                   release.publication_slo_seconds,
                   release.max_manifest_age_seconds,
                   release.retrieval_profile_sha256,
                   release.corpus_state_sha256,
                   activation.completed_at
            FROM bddk_meta.corpus_release_activations AS activation
            JOIN bddk_meta.corpus_releases AS release
              ON release.release_id = activation.release_id
            WHERE activation.activation_sequence = (
                SELECT MAX(latest.activation_sequence)
                FROM bddk_meta.corpus_release_activations AS latest
            )
              AND release.release_id = selected_release_id
              AND activation.corpus_epoch = selected_corpus_epoch;
        END
        $function$
        """.replace("__CORPUS_MUTATION_ADVISORY_KEY__", str(CORPUS_MUTATION_ADVISORY_KEY)),
        """
        CREATE VIEW bddk_meta.active_corpus_release
        WITH (security_barrier = true, security_invoker = false)
        AS
        SELECT release.release_id,
               release.manifest_id,
               release.manifest_sha256,
               release.signer_key_sha256,
               release.freshness_policy_result,
               release.source_detection_slo_seconds,
               release.publication_slo_seconds,
               release.max_manifest_age_seconds,
               release.retrieval_profile_sha256,
               release.corpus_state_sha256,
               release.created_at,
               activation.activation_sequence,
               activation.completed_at,
               activation.actor_fingerprint_sha256,
               activation.corpus_epoch
        FROM bddk_meta.corpus_release_activations AS activation
        JOIN bddk_meta.corpus_releases AS release
          ON release.release_id = activation.release_id
        CROSS JOIN bddk_meta.corpus_state_epoch AS epoch
        WHERE activation.activation_sequence = (
            SELECT MAX(latest.activation_sequence)
            FROM bddk_meta.corpus_release_activations AS latest
        )
          AND epoch.singleton_id
          AND activation.corpus_epoch = epoch.epoch
        """,
        """
        REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.corpus_fingerprint_frame(pg_catalog.text) FROM PUBLIC
        """,
        """
        REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.bump_corpus_state_epoch() FROM PUBLIC
        """,
        """
        REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.current_corpus_state_sha256(pg_catalog.text) FROM PUBLIC
        """,
        """
        REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.corpus_retrieval_ready(pg_catalog.text) FROM PUBLIC
        """,
        """
        REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.reject_corpus_release_mutation() FROM PUBLIC
        """,
        """
        REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.publish_verified_corpus_release(
            pg_catalog.text,
            pg_catalog.text,
            pg_catalog.text,
            pg_catalog.int4,
            pg_catalog.int4,
            pg_catalog.int4,
            pg_catalog.text
        ) FROM PUBLIC
        """,
    ),
)
