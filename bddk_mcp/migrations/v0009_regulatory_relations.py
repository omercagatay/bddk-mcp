"""Migration 0009: typed, evidence-backed cross-reference edges between instruments.

Edges are claims with review provenance, exactly like legal versions: the
base table stores machine-extracted candidates and human verdicts, while the
runtime reads only the ``regulatory_validated_*`` views, which expose
validated rows backed by non-fixture artifacts.  Unvalidated or rejected
edges are therefore never servable by any workload LOGIN that lacks base
table privileges.
"""

from bddk_mcp.migrations.model import Migration

V0009_REGULATORY_RELATIONS = Migration(
    version=9,
    name="regulatory_relation_edges",
    statements=(
        """
        CREATE TABLE public.regulatory_relations (
            relation_id pg_catalog.text PRIMARY KEY,
            relation_type pg_catalog.text NOT NULL,
            source_instrument_id pg_catalog.text NOT NULL,
            source_provision_id pg_catalog.text,
            target_instrument_id pg_catalog.text,
            target_provision_id pg_catalog.text,
            target_external_ref pg_catalog.text,
            evidence_id pg_catalog.text NOT NULL,
            extraction_method pg_catalog.text NOT NULL,
            confidence pg_catalog.float4 NOT NULL,
            validation_state pg_catalog.text NOT NULL DEFAULT 'unvalidated',
            validated_by pg_catalog.text,
            validated_at pg_catalog.timestamptz,
            validation_method pg_catalog.text,
            review_record_sha256 pg_catalog.text,
            CONSTRAINT regulatory_relations_id_check
                CHECK (relation_id ~ '^rel_sha256_[0-9a-f]{64}$'),
            CONSTRAINT regulatory_relations_type_check
                CHECK (relation_type IN (
                    'amends', 'repeals', 'replaces', 'consolidates',
                    'implements', 'cites', 'defines', 'exception_to'
                )),
            CONSTRAINT regulatory_relations_target_check
                CHECK (target_instrument_id IS NOT NULL OR target_external_ref IS NOT NULL),
            CONSTRAINT regulatory_relations_provision_targets_check
                CHECK (target_provision_id IS NULL OR target_instrument_id IS NOT NULL),
            CONSTRAINT regulatory_relations_confidence_check
                CHECK (confidence >= 0.0 AND confidence <= 1.0),
            CONSTRAINT regulatory_relations_extraction_check
                CHECK (extraction_method <> ''),
            CONSTRAINT regulatory_relations_external_ref_check
                CHECK (target_external_ref IS NULL OR target_external_ref <> ''),
            CONSTRAINT regulatory_relations_validation_state_check
                CHECK (validation_state IN ('unvalidated', 'in_review', 'validated', 'rejected')),
            CONSTRAINT regulatory_relations_validation_fields_check CHECK (
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
            CONSTRAINT regulatory_relations_source_instrument_fk
                FOREIGN KEY (source_instrument_id)
                REFERENCES public.regulatory_instruments(instrument_id),
            CONSTRAINT regulatory_relations_source_provision_fk
                FOREIGN KEY (source_provision_id)
                REFERENCES public.regulatory_provisions(provision_id),
            CONSTRAINT regulatory_relations_target_instrument_fk
                FOREIGN KEY (target_instrument_id)
                REFERENCES public.regulatory_instruments(instrument_id),
            CONSTRAINT regulatory_relations_target_provision_fk
                FOREIGN KEY (target_provision_id)
                REFERENCES public.regulatory_provisions(provision_id),
            CONSTRAINT regulatory_relations_evidence_fk
                FOREIGN KEY (evidence_id)
                REFERENCES public.regulatory_evidence(evidence_id)
        )
        """,
        """
        CREATE INDEX idx_regulatory_relations_source
        ON public.regulatory_relations (source_instrument_id, relation_type)
        """,
        """
        CREATE INDEX idx_regulatory_relations_target
        ON public.regulatory_relations (target_instrument_id, relation_type)
        """,
        """
        CREATE VIEW public.regulatory_validated_relations
        WITH (security_barrier = true, security_invoker = false)
        AS
        SELECT relation.relation_id,
               relation.relation_type,
               relation.source_instrument_id,
               relation.source_provision_id,
               relation.target_instrument_id,
               relation.target_provision_id,
               relation.target_external_ref,
               relation.confidence,
               relation.review_record_sha256,
               evidence.evidence_id,
               evidence.locator AS evidence_locator,
               evidence.statement_sha256 AS evidence_statement_sha256,
               evidence.authority_level AS evidence_authority_level
        FROM public.regulatory_relations AS relation
        JOIN public.regulatory_evidence AS evidence
          ON evidence.evidence_id = relation.evidence_id
        JOIN public.regulatory_source_artifacts AS artifact
          ON artifact.artifact_id = evidence.artifact_id
        WHERE relation.validation_state = 'validated'
          AND artifact.fixture_only = false
        """,
        """
        CREATE VIEW public.regulatory_validated_legal_versions
        WITH (security_barrier = true, security_invoker = false)
        AS
        SELECT DISTINCT
               version.legal_version_id,
               version.instrument_id,
               version.version_key,
               version.legal_text_sha256,
               version.predecessor_version_id,
               version.consolidation_state,
               version.review_record_sha256,
               instrument.identity_key AS instrument_identity_key,
               instrument.canonical_title AS instrument_canonical_title,
               artifact.repository_document_id
        FROM public.regulatory_legal_versions AS version
        JOIN public.regulatory_instruments AS instrument
          ON instrument.instrument_id = version.instrument_id
        JOIN public.regulatory_legal_version_artifacts AS version_artifact
          ON version_artifact.legal_version_id = version.legal_version_id
         AND version_artifact.source_role = 'legal_text'
        JOIN public.regulatory_source_artifacts AS artifact
          ON artifact.artifact_id = version_artifact.artifact_id
        WHERE version.validation_state = 'validated'
          AND artifact.fixture_only = false
        """,
        """
        CREATE VIEW public.regulatory_validated_legal_events
        WITH (security_barrier = true, security_invoker = false)
        AS
        SELECT event.event_id,
               event.legal_version_id,
               event.event_type,
               event.event_date,
               event.evidence_id,
               event.target_legal_version_id
        FROM public.regulatory_legal_events AS event
        JOIN public.regulatory_legal_versions AS version
          ON version.legal_version_id = event.legal_version_id
        WHERE event.validation_state = 'validated'
          AND version.validation_state = 'validated'
        """,
    ),
)
