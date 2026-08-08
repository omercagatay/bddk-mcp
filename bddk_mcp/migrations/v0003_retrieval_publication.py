"""Migration 0003: fail-closed retrieval publication integrity."""

from bddk_mcp.migrations.model import Migration

V0003_RETRIEVAL_PUBLICATION = Migration(
    version=3,
    name="retrieval_publication_integrity",
    statements=(
        """
        CREATE TABLE bddk_meta.legacy_schema_adoptions (
            migration_version pg_catalog.int4 PRIMARY KEY,
            source_kind pg_catalog.text NOT NULL,
            verifier_version pg_catalog.text NOT NULL,
            target_checksum pg_catalog.text NOT NULL,
            pre_normalization_fingerprint pg_catalog.text NOT NULL,
            post_normalization_fingerprint pg_catalog.text NOT NULL,
            normalizations pg_catalog.text[] NOT NULL,
            adopted_by pg_catalog.text NOT NULL,
            adopted_session_user pg_catalog.text NOT NULL,
            adopted_at pg_catalog.timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT legacy_schema_adoptions_migration_fk
                FOREIGN KEY (migration_version)
                REFERENCES bddk_meta.schema_migrations(version),
            CONSTRAINT legacy_schema_adoptions_version_check CHECK (migration_version = 1),
            CONSTRAINT legacy_schema_adoptions_source_check
                CHECK (source_kind = 'bddk-mcp-pre-ledger-initializers'),
            CONSTRAINT legacy_schema_adoptions_target_checksum_check
                CHECK (target_checksum ~ '^[0-9a-f]{64}$'),
            CONSTRAINT legacy_schema_adoptions_pre_fingerprint_check
                CHECK (pre_normalization_fingerprint ~ '^[0-9a-f]{64}$'),
            CONSTRAINT legacy_schema_adoptions_post_fingerprint_check
                CHECK (post_normalization_fingerprint ~ '^[0-9a-f]{64}$')
        )
        """,
        """
        ALTER TABLE public.document_sections
        ADD COLUMN source_content_hash pg_catalog.text NOT NULL DEFAULT ''
        """,
        """
        ALTER TABLE public.document_sections
        DISABLE TRIGGER trg_document_sections_tsv
        """,
        """
        UPDATE public.document_sections AS section
        SET source_content_hash = document.content_hash
        FROM public.documents AS document
        WHERE document.document_id = section.doc_id
          AND section.source_content_hash IS DISTINCT FROM document.content_hash
        """,
        """
        ALTER TABLE public.document_sections
        ENABLE TRIGGER trg_document_sections_tsv
        """,
        """
        ALTER TABLE public.document_sections
        ALTER COLUMN source_content_hash DROP DEFAULT
        """,
        """
        ALTER TABLE public.document_sections
        ADD CONSTRAINT document_sections_document_fk
        FOREIGN KEY (doc_id) REFERENCES public.documents(document_id)
        ON DELETE CASCADE
        """,
        """
        ALTER TABLE public.document_chunks
        ADD CONSTRAINT document_chunks_document_fk
        FOREIGN KEY (doc_id) REFERENCES public.documents(document_id)
        ON DELETE CASCADE
        """,
        """
        CREATE TABLE public.document_retrieval_publications (
            doc_id pg_catalog.text PRIMARY KEY,
            content_hash pg_catalog.text NOT NULL,
            retrieval_profile_hash pg_catalog.text NOT NULL,
            expected_chunks pg_catalog.int4 NOT NULL,
            published_at pg_catalog.timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT document_retrieval_publications_document_fk
                FOREIGN KEY (doc_id) REFERENCES public.documents(document_id)
                ON DELETE CASCADE,
            CONSTRAINT document_retrieval_publications_content_hash_check
                CHECK (content_hash ~ '^[0-9a-f]{64}$'),
            CONSTRAINT document_retrieval_publications_profile_hash_check
                CHECK (retrieval_profile_hash ~ '^[0-9a-f]{64}$'),
            CONSTRAINT document_retrieval_publications_expected_chunks_check
                CHECK (expected_chunks > 0)
        )
        """,
        """
        CREATE FUNCTION public.invalidate_retrieval_publication()
        RETURNS trigger
        LANGUAGE plpgsql
        SET search_path = pg_catalog, public
        AS $function$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                DELETE FROM public.document_retrieval_publications
                WHERE doc_id = NEW.doc_id;
                RETURN NEW;
            END IF;
            DELETE FROM public.document_retrieval_publications
            WHERE doc_id = OLD.doc_id;
            IF TG_OP = 'UPDATE' THEN
                IF OLD.doc_id IS DISTINCT FROM NEW.doc_id THEN
                    DELETE FROM public.document_retrieval_publications
                    WHERE doc_id = NEW.doc_id;
                END IF;
                RETURN NEW;
            END IF;
            RETURN OLD;
        END
        $function$
        """,
        """
        CREATE TRIGGER invalidate_retrieval_publication_on_chunk_change
        AFTER INSERT OR UPDATE OR DELETE ON public.document_chunks
        FOR EACH ROW EXECUTE FUNCTION public.invalidate_retrieval_publication()
        """,
    ),
)
