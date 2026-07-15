"""Live-catalog attestation tests for retrieval-critical PostgreSQL objects."""

from __future__ import annotations

import pytest

from bddk_mcp.catalog_integrity import inspect_catalog_integrity
from bddk_mcp.db_lifecycle import inspect_database_readiness
from bddk_mcp.migrations.v0004_canonical_legal_versions import V0004_CANONICAL_LEGAL_VERSIONS
from bddk_mcp.regulatory.text_profile import PROVISION_BOUNDARY_CODEPOINTS_V1


def test_citation_view_source_ddl_schema_qualifies_the_retained_text_hash_gates() -> None:
    ddl = " ".join(" ".join(statement.split()) for statement in V0004_CANONICAL_LEGAL_VERSIONS.statements)

    assert (
        "document.content_hash = pg_catalog.encode( "
        "pg_catalog.sha256(pg_catalog.convert_to(document.markdown_content, 'UTF8')), 'hex' )"
    ) in ddl
    assert (
        "section.content_hash = pg_catalog.encode( "
        "pg_catalog.sha256(pg_catalog.convert_to(section.content, 'UTF8')), 'hex' )"
    ) in ddl
    assert "section.content = pg_catalog.btrim(" in ddl
    for codepoint in PROVISION_BOUNDARY_CODEPOINTS_V1:
        assert f"pg_catalog.chr({codepoint})" in ddl


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_current_migrations_have_valid_retrieval_catalog(pg_pool) -> None:
    report = await inspect_catalog_integrity(pg_pool)

    assert report.valid
    assert report.failures == ()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_disabled_publication_invalidation_trigger_fails_readiness(pg_pool) -> None:
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await connection.execute(
                "ALTER TABLE public.document_chunks DISABLE TRIGGER invalidate_retrieval_publication_on_chunk_change"
            )

            report = await inspect_catalog_integrity(connection)
            readiness = await inspect_database_readiness(connection, require_corpus=False)

            expected = "trigger:public.document_chunks.invalidate_retrieval_publication_on_chunk_change"
            assert expected in report.failures
            assert expected in readiness.catalog_issues
            assert not readiness.ready
        finally:
            await transaction.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_replaced_invalidation_function_fails_readiness(pg_pool) -> None:
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await connection.execute(
                """
                CREATE OR REPLACE FUNCTION public.invalidate_retrieval_publication()
                RETURNS trigger
                LANGUAGE plpgsql
                SET search_path = pg_catalog, public
                AS $function$
                BEGIN
                    RETURN COALESCE(NEW, OLD);
                END
                $function$
                """
            )

            report = await inspect_catalog_integrity(connection)

            assert "routine:public.invalidate_retrieval_publication()" in report.failures
        finally:
            await transaction.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_missing_retrieval_index_fails_readiness(pg_pool) -> None:
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await connection.execute("DROP INDEX public.idx_chunks_tsv")

            report = await inspect_catalog_integrity(connection)

            assert "index:public.idx_chunks_tsv" in report.failures
        finally:
            await transaction.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_weakened_validated_citation_view_fails_readiness(pg_pool) -> None:
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await connection.execute(
                "ALTER VIEW public.regulatory_validated_section_citations SET (security_barrier = false)"
            )

            report = await inspect_catalog_integrity(connection)
            readiness = await inspect_database_readiness(connection, require_corpus=False)

            expected = "view:public.regulatory_validated_section_citations"
            assert expected in report.failures
            assert expected in readiness.catalog_issues
            assert not readiness.ready
        finally:
            await transaction.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_weakened_v4_constraint_fails_exact_catalog_attestation(pg_pool) -> None:
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await connection.execute(
                "ALTER TABLE public.regulatory_source_blobs DROP CONSTRAINT regulatory_source_blobs_hash_check"
            )
            await connection.execute(
                "ALTER TABLE public.regulatory_source_blobs "
                "ADD CONSTRAINT regulatory_source_blobs_hash_check CHECK (content_sha256 IS NOT NULL)"
            )

            report = await inspect_catalog_integrity(connection)
            readiness = await inspect_database_readiness(connection, require_corpus=False)

            expected = "constraints:public.regulatory_v4_exact"
            assert expected in report.failures
            assert expected in readiness.catalog_issues
            assert not readiness.ready
        finally:
            await transaction.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_replaced_v4_index_fails_exact_catalog_attestation(pg_pool) -> None:
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await connection.execute("DROP INDEX public.idx_regulatory_events_version_date")
            await connection.execute(
                "CREATE INDEX idx_regulatory_events_version_date ON public.regulatory_legal_events (event_id)"
            )

            report = await inspect_catalog_integrity(connection)
            readiness = await inspect_database_readiness(connection, require_corpus=False)

            expected = "indexes:public.regulatory_v4_exact"
            assert expected in report.failures
            assert expected in readiness.catalog_issues
            assert not readiness.ready
        finally:
            await transaction.rollback()
