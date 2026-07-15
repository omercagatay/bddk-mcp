"""Live-catalog attestation tests for retrieval-critical PostgreSQL objects."""

from __future__ import annotations

import pytest

from bddk_mcp.catalog_integrity import inspect_catalog_integrity
from bddk_mcp.db_lifecycle import inspect_database_readiness


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
