"""Migration 0004: canonical instrument, legal-version, and provision identities.

The schema stores claims and their validation provenance.  It intentionally
does not derive legal effective dates from ``public.document_versions``:
those rows represent extraction history, not authoritative legal versions.
"""

from bddk_mcp.migrations.model import Migration
from bddk_mcp.regulatory.text_profile import POSTGRES_PROVISION_BOUNDARY_WHITESPACE_V1

V0004_CANONICAL_LEGAL_VERSIONS = Migration(
    version=4,
    name="canonical_legal_version_pilot",
    statements=(
        """
        CREATE TABLE public.regulatory_instruments (
            instrument_id pg_catalog.text PRIMARY KEY,
            jurisdiction pg_catalog.text NOT NULL,
            authority_code pg_catalog.text NOT NULL,
            identity_key pg_catalog.text NOT NULL,
            canonical_title pg_catalog.text NOT NULL,
            instrument_type pg_catalog.text NOT NULL,
            created_at pg_catalog.timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT regulatory_instruments_id_check
                CHECK (instrument_id ~ '^inst_sha256_[0-9a-f]{64}$'),
            CONSTRAINT regulatory_instruments_identity_uq
                UNIQUE (jurisdiction, authority_code, identity_key)
        )
        """,
        """
        CREATE TABLE public.regulatory_family_imports (
            bundle_id pg_catalog.text NOT NULL,
            bundle_sha256 pg_catalog.text NOT NULL,
            instrument_id pg_catalog.text NOT NULL,
            schema_version pg_catalog.int4 NOT NULL,
            fixture_only pg_catalog.bool NOT NULL,
            imported_by pg_catalog.text NOT NULL,
            imported_current_user pg_catalog.text NOT NULL DEFAULT CURRENT_USER,
            imported_session_user pg_catalog.text NOT NULL DEFAULT SESSION_USER,
            predecessor_bundle_sha256 pg_catalog.text,
            member_manifest pg_catalog.jsonb NOT NULL,
            imported_at pg_catalog.timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT regulatory_family_imports_pkey
                PRIMARY KEY (bundle_id, bundle_sha256),
            CONSTRAINT regulatory_family_imports_id_check
                CHECK (bundle_id ~ '^family_sha256_[0-9a-f]{64}$'),
            CONSTRAINT regulatory_family_imports_hash_check
                CHECK (bundle_sha256 ~ '^[0-9a-f]{64}$'),
            CONSTRAINT regulatory_family_imports_schema_check
                CHECK (schema_version = 1),
            CONSTRAINT regulatory_family_imports_predecessor_hash_check
                CHECK (predecessor_bundle_sha256 IS NULL OR predecessor_bundle_sha256 ~ '^[0-9a-f]{64}$'),
            CONSTRAINT regulatory_family_imports_instrument_fk
                FOREIGN KEY (instrument_id)
                REFERENCES public.regulatory_instruments(instrument_id),
            CONSTRAINT regulatory_family_imports_predecessor_fk
                FOREIGN KEY (bundle_id, predecessor_bundle_sha256)
                REFERENCES public.regulatory_family_imports(bundle_id, bundle_sha256),
            CONSTRAINT regulatory_family_imports_not_own_predecessor_check
                CHECK (predecessor_bundle_sha256 IS DISTINCT FROM bundle_sha256)
        )
        """,
        """
        CREATE TABLE public.regulatory_source_blobs (
            blob_id pg_catalog.text PRIMARY KEY,
            content_sha256 pg_catalog.text NOT NULL,
            CONSTRAINT regulatory_source_blobs_id_check
                CHECK (blob_id ~ '^blob_sha256_[0-9a-f]{64}$'),
            CONSTRAINT regulatory_source_blobs_hash_check
                CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
            CONSTRAINT regulatory_source_blobs_content_uq
                UNIQUE (content_sha256)
        )
        """,
        """
        CREATE TABLE public.regulatory_source_artifacts (
            artifact_id pg_catalog.text PRIMARY KEY,
            blob_id pg_catalog.text NOT NULL,
            canonical_uri pg_catalog.text NOT NULL,
            source_authority pg_catalog.text NOT NULL,
            media_type pg_catalog.text NOT NULL,
            retrieved_at pg_catalog.timestamptz NOT NULL,
            repository_document_id pg_catalog.text,
            fixture_only pg_catalog.bool NOT NULL DEFAULT false,
            CONSTRAINT regulatory_source_artifacts_id_check
                CHECK (artifact_id ~ '^art_sha256_[0-9a-f]{64}$'),
            CONSTRAINT regulatory_source_artifacts_blob_fk
                FOREIGN KEY (blob_id)
                REFERENCES public.regulatory_source_blobs(blob_id),
            CONSTRAINT regulatory_source_artifacts_document_fk
                FOREIGN KEY (repository_document_id)
                REFERENCES public.documents(document_id),
            CONSTRAINT regulatory_source_artifacts_acquisition_uq
                UNIQUE (blob_id, canonical_uri, retrieved_at)
        )
        """,
        """
        CREATE TABLE public.regulatory_evidence (
            evidence_id pg_catalog.text PRIMARY KEY,
            artifact_id pg_catalog.text NOT NULL,
            locator pg_catalog.text NOT NULL,
            statement_sha256 pg_catalog.text NOT NULL,
            authority_level pg_catalog.text NOT NULL,
            CONSTRAINT regulatory_evidence_id_check
                CHECK (evidence_id ~ '^evid_sha256_[0-9a-f]{64}$'),
            CONSTRAINT regulatory_evidence_statement_hash_check
                CHECK (statement_sha256 ~ '^[0-9a-f]{64}$'),
            CONSTRAINT regulatory_evidence_authority_check
                CHECK (authority_level IN ('authoritative', 'secondary', 'repository_fixture')),
            CONSTRAINT regulatory_evidence_artifact_fk
                FOREIGN KEY (artifact_id)
                REFERENCES public.regulatory_source_artifacts(artifact_id)
        )
        """,
        """
        CREATE TABLE public.regulatory_legal_versions (
            legal_version_id pg_catalog.text PRIMARY KEY,
            instrument_id pg_catalog.text NOT NULL,
            version_key pg_catalog.text NOT NULL,
            legal_text_sha256 pg_catalog.text NOT NULL,
            predecessor_version_id pg_catalog.text,
            consolidation_state pg_catalog.text NOT NULL DEFAULT 'unknown',
            validation_state pg_catalog.text NOT NULL DEFAULT 'unvalidated',
            validated_by pg_catalog.text,
            validated_at pg_catalog.timestamptz,
            validation_method pg_catalog.text,
            review_record_sha256 pg_catalog.text,
            created_at pg_catalog.timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT regulatory_legal_versions_id_check
                CHECK (legal_version_id ~ '^ver_sha256_[0-9a-f]{64}$'),
            CONSTRAINT regulatory_legal_versions_text_hash_check
                CHECK (legal_text_sha256 ~ '^[0-9a-f]{64}$'),
            CONSTRAINT regulatory_legal_versions_consolidation_check
                CHECK (consolidation_state IN ('unknown', 'original', 'amendment', 'consolidated')),
            CONSTRAINT regulatory_legal_versions_validation_state_check
                CHECK (validation_state IN ('unvalidated', 'in_review', 'validated', 'rejected')),
            CONSTRAINT regulatory_legal_versions_validation_fields_check CHECK (
                (
                    validation_state IN ('validated', 'rejected')
                    AND validated_by IS NOT NULL
                    AND validated_at IS NOT NULL
                    AND validation_method IS NOT NULL
                    AND review_record_sha256 ~ '^[0-9a-f]{64}$'
                ) OR (
                    validation_state NOT IN ('validated', 'rejected')
                    AND validated_by IS NULL
                    AND validated_at IS NULL
                    AND validation_method IS NULL
                    AND review_record_sha256 IS NULL
                )
            ),
            CONSTRAINT regulatory_legal_versions_instrument_fk
                FOREIGN KEY (instrument_id)
                REFERENCES public.regulatory_instruments(instrument_id),
            CONSTRAINT regulatory_legal_versions_predecessor_fk
                FOREIGN KEY (predecessor_version_id)
                REFERENCES public.regulatory_legal_versions(legal_version_id),
            CONSTRAINT regulatory_legal_versions_not_own_predecessor_check
                CHECK (predecessor_version_id IS DISTINCT FROM legal_version_id),
            CONSTRAINT regulatory_legal_versions_instrument_key_uq
                UNIQUE (instrument_id, version_key)
        )
        """,
        """
        CREATE TABLE public.regulatory_legal_version_artifacts (
            legal_version_id pg_catalog.text NOT NULL,
            artifact_id pg_catalog.text NOT NULL,
            source_role pg_catalog.text NOT NULL DEFAULT 'legal_text',
            CONSTRAINT regulatory_legal_version_artifacts_pkey
                PRIMARY KEY (legal_version_id, artifact_id, source_role),
            CONSTRAINT regulatory_legal_version_artifacts_version_fk
                FOREIGN KEY (legal_version_id)
                REFERENCES public.regulatory_legal_versions(legal_version_id),
            CONSTRAINT regulatory_legal_version_artifacts_artifact_fk
                FOREIGN KEY (artifact_id)
                REFERENCES public.regulatory_source_artifacts(artifact_id)
        )
        """,
        """
        CREATE TABLE public.regulatory_legal_events (
            event_id pg_catalog.text PRIMARY KEY,
            legal_version_id pg_catalog.text NOT NULL,
            event_type pg_catalog.text NOT NULL,
            event_date pg_catalog.date NOT NULL,
            evidence_id pg_catalog.text NOT NULL,
            target_legal_version_id pg_catalog.text,
            validation_state pg_catalog.text NOT NULL DEFAULT 'unvalidated',
            validated_by pg_catalog.text,
            validated_at pg_catalog.timestamptz,
            validation_method pg_catalog.text,
            review_record_sha256 pg_catalog.text,
            CONSTRAINT regulatory_legal_events_id_check
                CHECK (event_id ~ '^event_sha256_[0-9a-f]{64}$'),
            CONSTRAINT regulatory_legal_events_type_check
                CHECK (event_type IN (
                    'publication', 'effective', 'expiry', 'repeal', 'supersession', 'consolidation'
                )),
            CONSTRAINT regulatory_legal_events_target_check CHECK (
                (event_type = 'supersession' AND target_legal_version_id IS NOT NULL
                    AND target_legal_version_id IS DISTINCT FROM legal_version_id)
                OR (event_type <> 'supersession' AND target_legal_version_id IS NULL)
            ),
            CONSTRAINT regulatory_legal_events_validation_state_check
                CHECK (validation_state IN ('unvalidated', 'in_review', 'validated', 'rejected')),
            CONSTRAINT regulatory_legal_events_validation_fields_check CHECK (
                (
                    validation_state IN ('validated', 'rejected')
                    AND validated_by IS NOT NULL
                    AND validated_at IS NOT NULL
                    AND validation_method IS NOT NULL
                    AND review_record_sha256 ~ '^[0-9a-f]{64}$'
                ) OR (
                    validation_state NOT IN ('validated', 'rejected')
                    AND validated_by IS NULL
                    AND validated_at IS NULL
                    AND validation_method IS NULL
                    AND review_record_sha256 IS NULL
                )
            ),
            CONSTRAINT regulatory_legal_events_version_fk
                FOREIGN KEY (legal_version_id)
                REFERENCES public.regulatory_legal_versions(legal_version_id),
            CONSTRAINT regulatory_legal_events_target_version_fk
                FOREIGN KEY (target_legal_version_id)
                REFERENCES public.regulatory_legal_versions(legal_version_id),
            CONSTRAINT regulatory_legal_events_evidence_fk
                FOREIGN KEY (evidence_id)
                REFERENCES public.regulatory_evidence(evidence_id),
            CONSTRAINT regulatory_legal_events_version_type_uq
                UNIQUE (legal_version_id, event_type)
        )
        """,
        """
        CREATE TABLE public.regulatory_legal_status_assertions (
            assertion_id pg_catalog.text PRIMARY KEY,
            legal_version_id pg_catalog.text NOT NULL,
            legal_status pg_catalog.text NOT NULL,
            valid_from pg_catalog.date NOT NULL,
            valid_through pg_catalog.date NOT NULL,
            evidence_id pg_catalog.text NOT NULL,
            validation_state pg_catalog.text NOT NULL DEFAULT 'unvalidated',
            validated_by pg_catalog.text,
            validated_at pg_catalog.timestamptz,
            validation_method pg_catalog.text,
            review_record_sha256 pg_catalog.text,
            CONSTRAINT regulatory_legal_status_assertions_id_check
                CHECK (assertion_id ~ '^status_sha256_[0-9a-f]{64}$'),
            CONSTRAINT regulatory_legal_status_assertions_status_check
                CHECK (legal_status IN (
                    'effective', 'not_yet_effective', 'expired', 'repealed', 'superseded', 'unknown'
                )),
            CONSTRAINT regulatory_legal_status_assertions_range_check
                CHECK (valid_through >= valid_from),
            CONSTRAINT regulatory_legal_status_assertions_validation_state_check
                CHECK (validation_state IN ('unvalidated', 'in_review', 'validated', 'rejected')),
            CONSTRAINT regulatory_legal_status_assertions_validation_fields_check CHECK (
                (
                    validation_state IN ('validated', 'rejected')
                    AND validated_by IS NOT NULL
                    AND validated_at IS NOT NULL
                    AND validation_method IS NOT NULL
                    AND review_record_sha256 ~ '^[0-9a-f]{64}$'
                ) OR (
                    validation_state NOT IN ('validated', 'rejected')
                    AND validated_by IS NULL
                    AND validated_at IS NULL
                    AND validation_method IS NULL
                    AND review_record_sha256 IS NULL
                )
            ),
            CONSTRAINT regulatory_legal_status_assertions_version_fk
                FOREIGN KEY (legal_version_id)
                REFERENCES public.regulatory_legal_versions(legal_version_id),
            CONSTRAINT regulatory_legal_status_assertions_evidence_fk
                FOREIGN KEY (evidence_id)
                REFERENCES public.regulatory_evidence(evidence_id)
        )
        """,
        """
        CREATE TABLE public.regulatory_provisions (
            provision_id pg_catalog.text PRIMARY KEY,
            instrument_id pg_catalog.text NOT NULL,
            provision_kind pg_catalog.text NOT NULL,
            canonical_path pg_catalog.text NOT NULL,
            CONSTRAINT regulatory_provisions_id_check
                CHECK (provision_id ~ '^prov_sha256_[0-9a-f]{64}$'),
            CONSTRAINT regulatory_provisions_instrument_fk
                FOREIGN KEY (instrument_id)
                REFERENCES public.regulatory_instruments(instrument_id),
            CONSTRAINT regulatory_provisions_path_uq
                UNIQUE (instrument_id, provision_kind, canonical_path)
        )
        """,
        """
        CREATE TABLE public.regulatory_legal_version_provisions (
            legal_version_id pg_catalog.text NOT NULL,
            provision_id pg_catalog.text NOT NULL,
            provision_text_sha256 pg_catalog.text NOT NULL,
            document_section_id pg_catalog.int4,
            evidence_id pg_catalog.text NOT NULL,
            validation_state pg_catalog.text NOT NULL DEFAULT 'unvalidated',
            validated_by pg_catalog.text,
            validated_at pg_catalog.timestamptz,
            validation_method pg_catalog.text,
            review_record_sha256 pg_catalog.text,
            CONSTRAINT regulatory_legal_version_provisions_pkey
                PRIMARY KEY (legal_version_id, provision_id),
            CONSTRAINT regulatory_legal_version_provisions_text_hash_check
                CHECK (provision_text_sha256 ~ '^[0-9a-f]{64}$'),
            CONSTRAINT regulatory_legal_version_provisions_validation_state_check
                CHECK (validation_state IN ('unvalidated', 'in_review', 'validated', 'rejected')),
            CONSTRAINT regulatory_legal_version_provisions_validation_fields_check CHECK (
                (
                    validation_state IN ('validated', 'rejected')
                    AND validated_by IS NOT NULL
                    AND validated_at IS NOT NULL
                    AND validation_method IS NOT NULL
                    AND review_record_sha256 ~ '^[0-9a-f]{64}$'
                ) OR (
                    validation_state NOT IN ('validated', 'rejected')
                    AND validated_by IS NULL
                    AND validated_at IS NULL
                    AND validation_method IS NULL
                    AND review_record_sha256 IS NULL
                )
            ),
            CONSTRAINT regulatory_legal_version_provisions_version_fk
                FOREIGN KEY (legal_version_id)
                REFERENCES public.regulatory_legal_versions(legal_version_id),
            CONSTRAINT regulatory_legal_version_provisions_provision_fk
                FOREIGN KEY (provision_id)
                REFERENCES public.regulatory_provisions(provision_id),
            CONSTRAINT regulatory_legal_version_provisions_section_fk
                FOREIGN KEY (document_section_id)
                REFERENCES public.document_sections(id),
            CONSTRAINT regulatory_legal_version_provisions_evidence_fk
                FOREIGN KEY (evidence_id)
                REFERENCES public.regulatory_evidence(evidence_id),
            CONSTRAINT regulatory_legal_version_provisions_section_uq
                UNIQUE (document_section_id)
        )
        """,
        f"""
        CREATE VIEW public.regulatory_validated_section_citations
        WITH (security_barrier = true, security_invoker = false)
        AS
        SELECT occurrence.document_section_id,
               section.doc_id AS source_document_id,
               document.content_hash AS normalized_document_sha256,
               section.content_hash AS normalized_section_sha256,
               version.instrument_id,
               instrument.jurisdiction AS instrument_jurisdiction,
               instrument.authority_code AS instrument_authority_code,
               instrument.identity_key AS instrument_identity_key,
               version.legal_version_id,
               version.version_key AS legal_version_key,
               version.legal_text_sha256,
               version.review_record_sha256,
               occurrence.review_record_sha256 AS provision_review_record_sha256,
               artifact.artifact_id,
               artifact.blob_id AS artifact_blob_id,
               blob.content_sha256 AS artifact_sha256,
               artifact.canonical_uri AS source_url,
               artifact.retrieved_at AS artifact_retrieved_at,
               evidence.evidence_id,
               evidence.locator AS evidence_locator,
               evidence.statement_sha256 AS evidence_statement_sha256,
               provision.provision_id,
               provision.provision_kind,
               provision.canonical_path AS provision_path,
               occurrence.provision_text_sha256
        FROM public.regulatory_legal_version_provisions AS occurrence
        JOIN public.document_sections AS section
          ON section.id = occurrence.document_section_id
         AND section.content_hash = occurrence.provision_text_sha256
         AND section.content_hash = pg_catalog.encode(
             pg_catalog.sha256(pg_catalog.convert_to(section.content, 'UTF8')),
             'hex'
         )
        JOIN public.documents AS document
          ON document.document_id = section.doc_id
         AND document.content_hash = section.source_content_hash
         AND document.content_hash = pg_catalog.encode(
             pg_catalog.sha256(pg_catalog.convert_to(document.markdown_content, 'UTF8')),
             'hex'
         )
         AND section.content = pg_catalog.btrim(
             pg_catalog.substr(
                 document.markdown_content,
                 section.start_char + 1,
                 section.end_char - section.start_char
             ),
             {POSTGRES_PROVISION_BOUNDARY_WHITESPACE_V1}
         )
        JOIN public.regulatory_legal_versions AS version
          ON version.legal_version_id = occurrence.legal_version_id
         AND version.legal_text_sha256 = document.content_hash
        JOIN public.regulatory_instruments AS instrument
          ON instrument.instrument_id = version.instrument_id
        JOIN public.regulatory_provisions AS provision
          ON provision.provision_id = occurrence.provision_id
         AND provision.instrument_id = version.instrument_id
        JOIN public.regulatory_evidence AS evidence
          ON evidence.evidence_id = occurrence.evidence_id
         AND evidence.statement_sha256 = occurrence.provision_text_sha256
        JOIN public.regulatory_source_artifacts AS artifact
          ON artifact.artifact_id = evidence.artifact_id
         AND artifact.repository_document_id = section.doc_id
        JOIN public.regulatory_source_blobs AS blob
          ON blob.blob_id = artifact.blob_id
        JOIN public.regulatory_legal_version_artifacts AS version_artifact
          ON version_artifact.legal_version_id = version.legal_version_id
         AND version_artifact.artifact_id = artifact.artifact_id
         AND version_artifact.source_role = 'legal_text'
        WHERE occurrence.validation_state = 'validated'
          AND version.validation_state = 'validated'
          AND evidence.authority_level = 'authoritative'
          AND artifact.fixture_only = false
        """,
        """
        CREATE INDEX idx_regulatory_versions_instrument
        ON public.regulatory_legal_versions (instrument_id, legal_version_id)
        """,
        """
        CREATE INDEX idx_regulatory_status_as_of
        ON public.regulatory_legal_status_assertions (
            legal_version_id, valid_from, valid_through, legal_status
        )
        """,
        """
        CREATE INDEX idx_regulatory_events_version_date
        ON public.regulatory_legal_events (legal_version_id, event_date, event_type)
        """,
    ),
)
