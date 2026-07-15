-- Exact bddk-mcp runtime privileges.
--
-- Run as the database owner or an approved database administrator after every
-- successful schema migration.  Missing expected relations are a hard error:
-- do not hide an incomplete or out-of-order deployment with conditional GRANTs.

DO $target_database$
DECLARE
    expected_database text := current_setting('bddk.expected_database', true);
BEGIN
    IF expected_database IS NULL OR btrim(expected_database) = '' THEN
        RAISE EXCEPTION 'bddk.expected_database must be set before grant reconciliation';
    END IF;
    IF current_database() <> expected_database THEN
        RAISE EXCEPTION 'grant reconciliation target database does not match the approved database';
    END IF;
END
$target_database$;

-- Normalize ownership for installations whose historical migrations ran as a
-- LOGIN role.  Subsequent migrations must SET ROLE bddk_schema_owner.
ALTER SCHEMA bddk_meta OWNER TO bddk_schema_owner;
ALTER SCHEMA bddk_operator OWNER TO bddk_schema_owner;

ALTER TABLE bddk_meta.schema_migrations OWNER TO bddk_schema_owner;
ALTER TABLE bddk_meta.legacy_schema_adoptions OWNER TO bddk_schema_owner;
ALTER TABLE bddk_meta.corpus_releases OWNER TO bddk_schema_owner;
ALTER TABLE bddk_meta.corpus_release_activations OWNER TO bddk_schema_owner;
ALTER TABLE bddk_meta.corpus_state_epoch OWNER TO bddk_schema_owner;
ALTER VIEW bddk_meta.active_corpus_release OWNER TO bddk_schema_owner;
ALTER TABLE public.decision_cache OWNER TO bddk_schema_owner;
ALTER TABLE public.documents OWNER TO bddk_schema_owner;
ALTER TABLE public.document_sections OWNER TO bddk_schema_owner;
ALTER TABLE public.document_versions OWNER TO bddk_schema_owner;
ALTER TABLE public.document_chunks OWNER TO bddk_schema_owner;
ALTER TABLE public.document_retrieval_publications OWNER TO bddk_schema_owner;
ALTER TABLE public.tool_call_traces OWNER TO bddk_schema_owner;
ALTER TABLE public.sync_metadata OWNER TO bddk_schema_owner;
ALTER TABLE public.sync_failures OWNER TO bddk_schema_owner;
ALTER TABLE public.regulatory_instruments OWNER TO bddk_schema_owner;
ALTER TABLE public.regulatory_family_imports OWNER TO bddk_schema_owner;
ALTER TABLE public.regulatory_source_blobs OWNER TO bddk_schema_owner;
ALTER TABLE public.regulatory_source_artifacts OWNER TO bddk_schema_owner;
ALTER TABLE public.regulatory_evidence OWNER TO bddk_schema_owner;
ALTER TABLE public.regulatory_legal_versions OWNER TO bddk_schema_owner;
ALTER TABLE public.regulatory_legal_version_artifacts OWNER TO bddk_schema_owner;
ALTER TABLE public.regulatory_legal_events OWNER TO bddk_schema_owner;
ALTER TABLE public.regulatory_legal_status_assertions OWNER TO bddk_schema_owner;
ALTER TABLE public.regulatory_provisions OWNER TO bddk_schema_owner;
ALTER TABLE public.regulatory_legal_version_provisions OWNER TO bddk_schema_owner;
ALTER VIEW public.regulatory_validated_section_citations OWNER TO bddk_schema_owner;
ALTER TABLE bddk_operator.operator_jobs OWNER TO bddk_schema_owner;

ALTER SEQUENCE public.document_sections_id_seq OWNER TO bddk_schema_owner;
ALTER SEQUENCE public.document_versions_id_seq OWNER TO bddk_schema_owner;
ALTER SEQUENCE public.document_chunks_id_seq OWNER TO bddk_schema_owner;
ALTER SEQUENCE public.tool_call_traces_id_seq OWNER TO bddk_schema_owner;
ALTER SEQUENCE bddk_meta.corpus_release_activations_activation_sequence_seq
    OWNER TO bddk_schema_owner;

ALTER FUNCTION public.immutable_unaccent(pg_catalog.text) OWNER TO bddk_schema_owner;
ALTER FUNCTION public.documents_tsv_trigger() OWNER TO bddk_schema_owner;
ALTER FUNCTION public.document_sections_tsv_trigger() OWNER TO bddk_schema_owner;
ALTER FUNCTION public.chunks_tsv_trigger() OWNER TO bddk_schema_owner;
ALTER FUNCTION public.invalidate_retrieval_publication() OWNER TO bddk_schema_owner;
ALTER FUNCTION bddk_meta.corpus_fingerprint_frame(pg_catalog.text) OWNER TO bddk_schema_owner;
ALTER FUNCTION bddk_meta.bump_corpus_state_epoch() OWNER TO bddk_schema_owner;
ALTER FUNCTION bddk_meta.current_corpus_state_sha256(pg_catalog.text) OWNER TO bddk_schema_owner;
ALTER FUNCTION bddk_meta.corpus_retrieval_ready(pg_catalog.text) OWNER TO bddk_schema_owner;
ALTER FUNCTION bddk_meta.reject_corpus_release_mutation() OWNER TO bddk_schema_owner;
ALTER FUNCTION bddk_meta.publish_verified_corpus_release(
    pg_catalog.text,
    pg_catalog.text,
    pg_catalog.text,
    pg_catalog.int4,
    pg_catalog.int4,
    pg_catalog.int4,
    pg_catalog.text
) OWNER TO bddk_schema_owner;
ALTER FUNCTION bddk_meta.resolve_regulation_status(
    pg_catalog.text,
    pg_catalog.date
) OWNER TO bddk_schema_owner;

-- Revoke first so rerunning this file also removes privileges deleted from the
-- reviewed matrix.  No runtime role receives CREATE on an application schema.
REVOKE ALL PRIVILEGES ON SCHEMA public
    FROM bddk_public_reader, bddk_ingestion, bddk_release_publisher,
         bddk_operator_runtime, bddk_telemetry_writer;
REVOKE ALL PRIVILEGES ON SCHEMA bddk_operator
    FROM bddk_public_reader, bddk_ingestion, bddk_release_publisher,
         bddk_operator_runtime, bddk_telemetry_writer;
REVOKE ALL PRIVILEGES ON SCHEMA bddk_meta
    FROM bddk_public_reader, bddk_ingestion, bddk_release_publisher,
         bddk_operator_runtime, bddk_telemetry_writer;

REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public
    FROM bddk_public_reader, bddk_ingestion, bddk_release_publisher,
         bddk_operator_runtime, bddk_telemetry_writer;
REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public
    FROM bddk_public_reader, bddk_ingestion, bddk_release_publisher,
         bddk_operator_runtime, bddk_telemetry_writer;
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA bddk_operator
    FROM bddk_public_reader, bddk_ingestion, bddk_release_publisher,
         bddk_operator_runtime, bddk_telemetry_writer;
REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA bddk_operator
    FROM bddk_public_reader, bddk_ingestion, bddk_release_publisher,
         bddk_operator_runtime, bddk_telemetry_writer;
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA bddk_meta
    FROM bddk_public_reader, bddk_ingestion, bddk_release_publisher,
         bddk_operator_runtime, bddk_telemetry_writer;
REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA bddk_meta
    FROM bddk_public_reader, bddk_ingestion, bddk_release_publisher,
         bddk_operator_runtime, bddk_telemetry_writer;

-- Canonical legal-version base tables are denied to serving and ingestion
-- workloads. The release-publisher receives read-only access later solely to
-- prove that no unsigned legal state is present during exact corpus admission.
REVOKE ALL PRIVILEGES ON TABLE
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
FROM PUBLIC, bddk_public_reader, bddk_ingestion, bddk_release_publisher,
     bddk_operator_runtime, bddk_telemetry_writer;

REVOKE ALL PRIVILEGES ON TABLE public.regulatory_validated_section_citations
FROM PUBLIC, bddk_public_reader, bddk_ingestion, bddk_release_publisher,
     bddk_operator_runtime, bddk_telemetry_writer;

-- Verified corpus evidence is immutable and owner-only. Runtimes see only the
-- content-free active view. Only the bootstrap-only release-publisher
-- capability can append through the reviewed SECURITY DEFINER function; no
-- runtime role receives base-table or sequence access.
REVOKE ALL PRIVILEGES ON TABLE
    bddk_meta.corpus_releases,
    bddk_meta.corpus_release_activations,
    bddk_meta.corpus_state_epoch,
    bddk_meta.active_corpus_release
FROM PUBLIC, bddk_public_reader, bddk_ingestion, bddk_release_publisher,
     bddk_operator_runtime, bddk_telemetry_writer;

REVOKE ALL PRIVILEGES ON SCHEMA bddk_operator FROM PUBLIC;
REVOKE ALL PRIVILEGES ON SCHEMA bddk_meta FROM PUBLIC;
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA bddk_operator FROM PUBLIC;
REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA bddk_operator FROM PUBLIC;
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA bddk_meta FROM PUBLIC;
REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA bddk_meta FROM PUBLIC;

ALTER DEFAULT PRIVILEGES FOR ROLE bddk_schema_owner IN SCHEMA bddk_operator
    REVOKE ALL PRIVILEGES ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE bddk_schema_owner IN SCHEMA bddk_operator
    REVOKE ALL PRIVILEGES ON SEQUENCES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE bddk_schema_owner IN SCHEMA bddk_operator
    REVOKE ALL PRIVILEGES ON FUNCTIONS FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE bddk_schema_owner IN SCHEMA bddk_meta
    REVOKE ALL PRIVILEGES ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE bddk_schema_owner IN SCHEMA bddk_meta
    REVOKE ALL PRIVILEGES ON SEQUENCES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE bddk_schema_owner IN SCHEMA bddk_meta
    REVOKE ALL PRIVILEGES ON FUNCTIONS FROM PUBLIC;

-- Revoke only application-owned functions, not vector/unaccent extension
-- functions in the same schema.  Search roles need the immutable wrapper;
-- trigger functions are invoked by their existing triggers, not directly.
REVOKE ALL PRIVILEGES ON FUNCTION public.immutable_unaccent(pg_catalog.text) FROM PUBLIC;
REVOKE ALL PRIVILEGES ON FUNCTION public.documents_tsv_trigger() FROM PUBLIC;
REVOKE ALL PRIVILEGES ON FUNCTION public.document_sections_tsv_trigger() FROM PUBLIC;
REVOKE ALL PRIVILEGES ON FUNCTION public.chunks_tsv_trigger() FROM PUBLIC;
REVOKE ALL PRIVILEGES ON FUNCTION public.invalidate_retrieval_publication() FROM PUBLIC;
REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.corpus_fingerprint_frame(pg_catalog.text)
FROM PUBLIC, bddk_public_reader, bddk_ingestion, bddk_release_publisher,
     bddk_operator_runtime, bddk_telemetry_writer;
REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.bump_corpus_state_epoch()
FROM PUBLIC, bddk_public_reader, bddk_ingestion, bddk_release_publisher,
     bddk_operator_runtime, bddk_telemetry_writer;
REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.current_corpus_state_sha256(pg_catalog.text)
FROM PUBLIC, bddk_public_reader, bddk_ingestion, bddk_release_publisher,
     bddk_operator_runtime, bddk_telemetry_writer;
REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.corpus_retrieval_ready(pg_catalog.text)
FROM PUBLIC, bddk_public_reader, bddk_ingestion, bddk_release_publisher,
     bddk_operator_runtime, bddk_telemetry_writer;
REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.reject_corpus_release_mutation()
FROM PUBLIC, bddk_public_reader, bddk_ingestion, bddk_release_publisher,
     bddk_operator_runtime, bddk_telemetry_writer;
REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.publish_verified_corpus_release(
    pg_catalog.text,
    pg_catalog.text,
    pg_catalog.text,
    pg_catalog.int4,
    pg_catalog.int4,
    pg_catalog.int4,
    pg_catalog.text
) FROM PUBLIC, bddk_public_reader, bddk_ingestion, bddk_release_publisher,
       bddk_operator_runtime, bddk_telemetry_writer;
REVOKE ALL PRIVILEGES ON FUNCTION bddk_meta.resolve_regulation_status(
    pg_catalog.text,
    pg_catalog.date
) FROM PUBLIC, bddk_public_reader, bddk_ingestion, bddk_release_publisher,
       bddk_operator_runtime, bddk_telemetry_writer;

-- Public MCP corpus: read-only, with no access to operational failure or trace
-- tables.  document_versions is included because get_document_history is public.
GRANT USAGE ON SCHEMA public TO bddk_public_reader;
GRANT SELECT ON TABLE
    public.decision_cache,
    public.documents,
    public.document_sections,
    public.document_versions,
    public.document_chunks,
    public.document_retrieval_publications,
    public.regulatory_validated_section_citations,
    bddk_meta.active_corpus_release
TO bddk_public_reader;
GRANT USAGE ON SCHEMA bddk_meta TO bddk_public_reader;
GRANT SELECT ON TABLE bddk_meta.schema_migrations TO bddk_public_reader;
GRANT EXECUTE ON FUNCTION public.immutable_unaccent(pg_catalog.text) TO bddk_public_reader;
GRANT EXECUTE ON FUNCTION bddk_meta.resolve_regulation_status(
    pg_catalog.text,
    pg_catalog.date
) TO bddk_public_reader;

-- Bootstrap/synchronization workers may mutate the corpus and its operational
-- sync state, but cannot read or write telemetry or operator-job relations.
GRANT USAGE ON SCHEMA public TO bddk_ingestion;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE
    public.decision_cache,
    public.documents,
    public.document_sections,
    public.document_versions,
    public.document_chunks,
    public.document_retrieval_publications,
    public.sync_metadata,
    public.sync_failures
TO bddk_ingestion;
GRANT USAGE ON SEQUENCE
    public.document_sections_id_seq,
    public.document_versions_id_seq,
    public.document_chunks_id_seq
TO bddk_ingestion;
GRANT USAGE ON SCHEMA bddk_meta TO bddk_ingestion;
GRANT SELECT ON TABLE bddk_meta.schema_migrations TO bddk_ingestion;
GRANT EXECUTE ON FUNCTION public.immutable_unaccent(pg_catalog.text) TO bddk_ingestion;
GRANT SELECT ON TABLE bddk_meta.active_corpus_release TO bddk_ingestion;
-- Release publication is a separate two-person/two-credential stage. The
-- publisher can inspect the exact corpus and canonical legal state, but cannot
-- mutate either; the only write path is the reviewed SECURITY DEFINER routine.
GRANT USAGE ON SCHEMA public TO bddk_release_publisher;
GRANT USAGE ON SCHEMA bddk_meta TO bddk_release_publisher;
GRANT SELECT ON TABLE
    public.decision_cache,
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
    public.regulatory_legal_version_provisions,
    bddk_meta.schema_migrations,
    bddk_meta.active_corpus_release
TO bddk_release_publisher;
GRANT EXECUTE ON FUNCTION bddk_meta.current_corpus_state_sha256(pg_catalog.text)
TO bddk_release_publisher;
GRANT EXECUTE ON FUNCTION bddk_meta.corpus_retrieval_ready(pg_catalog.text)
TO bddk_release_publisher;
GRANT EXECUTE ON FUNCTION bddk_meta.publish_verified_corpus_release(
    pg_catalog.text,
    pg_catalog.text,
    pg_catalog.text,
    pg_catalog.int4,
    pg_catalog.int4,
    pg_catalog.int4,
    pg_catalog.text
) TO bddk_release_publisher;

-- Operator lifecycle state is isolated in its own schema.  The runtime may
-- update/prune jobs and can only read the global migration ledger.
GRANT USAGE ON SCHEMA bddk_operator TO bddk_operator_runtime;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE bddk_operator.operator_jobs
    TO bddk_operator_runtime;
GRANT USAGE ON SCHEMA bddk_meta TO bddk_operator_runtime;
GRANT SELECT ON TABLE bddk_meta.schema_migrations TO bddk_operator_runtime;

-- Telemetry is append-only.  Column-level INSERT prevents callers from
-- overriding the generated id or created_at fields; sequence USAGE permits the
-- BIGSERIAL default without granting SELECT, UPDATE, or DELETE on trace rows.
GRANT USAGE ON SCHEMA public TO bddk_telemetry_writer;
GRANT INSERT (
    tool_name,
    args_hash,
    args_summary,
    latency_ms,
    result_count,
    doc_ids,
    quality_labels,
    relevance_stats,
    model_id,
    session_id
) ON TABLE public.tool_call_traces TO bddk_telemetry_writer;
GRANT USAGE ON SEQUENCE public.tool_call_traces_id_seq TO bddk_telemetry_writer;
