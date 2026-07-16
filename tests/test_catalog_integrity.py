"""Live-catalog attestation tests for retrieval-critical PostgreSQL objects."""

from __future__ import annotations

import pytest

from bddk_mcp.catalog_integrity import inspect_catalog_integrity
from bddk_mcp.db_lifecycle import inspect_database_readiness
from bddk_mcp.migrations.v0004_canonical_legal_versions import V0004_CANONICAL_LEGAL_VERSIONS
from bddk_mcp.migrations.v0005_corpus_release_publication import CORPUS_EPOCH_TRACKED_TABLES
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
async def test_each_tracked_corpus_statement_advances_the_singleton_epoch(pg_pool) -> None:
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            previous = await connection.fetchval("SELECT epoch FROM bddk_meta.corpus_state_epoch WHERE singleton_id")
            assert previous is not None

            for table_name in CORPUS_EPOCH_TRACKED_TABLES:
                await connection.execute(f"DELETE FROM public.{table_name} WHERE false")
                current = await connection.fetchval("SELECT epoch FROM bddk_meta.corpus_state_epoch WHERE singleton_id")
                # A chunk statement also invokes the set-based publication
                # invalidator, whose publication DELETE is independently
                # tracked even when its transition table is empty.
                expected_delta = 2 if table_name == "document_chunks" else 1
                assert current == previous + expected_delta
                previous = current
        finally:
            await transaction.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_disabled_publication_invalidation_trigger_fails_readiness(pg_pool) -> None:
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await connection.execute(
                "ALTER TABLE public.document_chunks DISABLE TRIGGER invalidate_retrieval_publication_on_chunk_insert"
            )

            report = await inspect_catalog_integrity(connection)
            readiness = await inspect_database_readiness(connection, require_corpus=False)

            expected = "trigger:public.document_chunks.invalidate_retrieval_publication_on_chunk_insert"
            assert expected in report.failures
            assert expected in readiness.catalog_issues
            assert not readiness.ready
        finally:
            await transaction.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_unexpected_public_trigger_fails_exact_catalog_attestation(pg_pool) -> None:
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await connection.execute(
                """
                CREATE TRIGGER unexpected_documents_tsv_trigger
                BEFORE INSERT ON public.documents
                FOR EACH ROW EXECUTE FUNCTION public.documents_tsv_trigger()
                """
            )

            report = await inspect_catalog_integrity(connection)

            assert "triggers:public.exact" in report.failures
        finally:
            await transaction.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_changed_transition_table_alias_fails_trigger_attestation(pg_pool) -> None:
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await connection.execute(
                "DROP TRIGGER invalidate_retrieval_publication_on_chunk_insert ON public.document_chunks"
            )
            await connection.execute(
                """
                CREATE TRIGGER invalidate_retrieval_publication_on_chunk_insert
                AFTER INSERT ON public.document_chunks
                REFERENCING NEW TABLE AS unexpected_chunks
                FOR EACH STATEMENT EXECUTE FUNCTION public.invalidate_retrieval_publication()
                """
            )

            report = await inspect_catalog_integrity(connection)

            assert "trigger:public.document_chunks.invalidate_retrieval_publication_on_chunk_insert" in report.failures
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
async def test_weakened_legal_status_resolver_fails_readiness(pg_pool) -> None:
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await connection.execute(
                "ALTER FUNCTION bddk_meta.resolve_regulation_status(pg_catalog.text, pg_catalog.date) SECURITY INVOKER"
            )

            report = await inspect_catalog_integrity(connection)
            readiness = await inspect_database_readiness(connection, require_corpus=False)

            expected = "routine:bddk_meta.resolve_regulation_status(text, date)"
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


@pytest.mark.postgres
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tamper_sql",
    [
        "ALTER TABLE bddk_retained.documents DISABLE TRIGGER guard_retained_generation_member",
        "ALTER TABLE bddk_retained.decision_cache SET UNLOGGED",
        "ALTER TABLE bddk_retained.documents ENABLE ROW LEVEL SECURITY",
        "ALTER FUNCTION bddk_meta.guard_retained_generation_member() SECURITY INVOKER",
        "ALTER FUNCTION bddk_meta.retain_active_corpus_generation(pg_catalog.text) STRICT",
        "ALTER TABLE bddk_meta.corpus_generation_seals "
        "DROP CONSTRAINT corpus_generation_seals_relation_count_check; "
        "ALTER TABLE bddk_meta.corpus_generation_seals "
        "ADD CONSTRAINT corpus_generation_seals_relation_count_check "
        "CHECK (relation_count > 0)",
    ],
    ids=(
        "disabled-member-trigger",
        "unlogged-retained-table",
        "enabled-row-security",
        "weakened-guard-function",
        "strict-retention-function",
        "weakened-seal-constraint",
    ),
)
async def test_v7_retention_catalog_tampering_fails_exact_attestation(pg_pool, tamper_sql: str) -> None:
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await connection.execute(tamper_sql)

            report = await inspect_catalog_integrity(connection)
            readiness = await inspect_database_readiness(connection, require_corpus=False)

            expected = "catalog:bddk_retained.v7_exact"
            assert expected in report.failures
            assert expected in readiness.catalog_issues
            assert not readiness.ready
        finally:
            await transaction.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_v7_column_level_grant_fails_exact_acl_attestation(pg_pool) -> None:
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await connection.execute("GRANT SELECT (generation_id) ON bddk_retained.documents TO PUBLIC")

            report = await inspect_catalog_integrity(connection)
            readiness = await inspect_database_readiness(connection, require_corpus=False)

            expected = "acl:bddk_retained.v7_exact"
            assert expected in report.failures
            assert expected in readiness.catalog_issues
            assert not readiness.ready
        finally:
            await transaction.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tamper_sql", "expected"),
    (
        (
            "ALTER FUNCTION bddk_meta.activate_staged_corpus_release(pg_catalog.text) SECURITY INVOKER",
            "routine:bddk_meta.activate_staged_corpus_release(text)",
        ),
        (
            "ALTER TABLE bddk_meta.corpus_release_requests DISABLE TRIGGER reject_corpus_release_request_update_delete",
            "triggers:bddk_meta.corpus_release_exact",
        ),
        (
            "ALTER TABLE bddk_meta.corpus_release_request_activations "
            "DROP CONSTRAINT corpus_release_request_activations_activation_fk",
            "constraints:bddk_meta.corpus_release_exact",
        ),
    ),
    ids=("security-invoker", "disabled-append-only-trigger", "missing-activation-binding"),
)
async def test_v8_staged_release_catalog_tampering_fails_attestation(
    pg_pool,
    tamper_sql: str,
    expected: str,
) -> None:
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await connection.execute(tamper_sql)

            report = await inspect_catalog_integrity(connection)
            readiness = await inspect_database_readiness(connection, require_corpus=False)

            assert expected in report.failures
            assert expected in readiness.catalog_issues
            assert not readiness.ready
        finally:
            await transaction.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_v8_legacy_direct_publication_grant_fails_acl_attestation(pg_pool) -> None:
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await connection.execute(
                """
                DO $publisher$
                BEGIN
                    IF pg_catalog.to_regrole('bddk_release_publisher') IS NULL THEN
                        CREATE ROLE bddk_release_publisher NOLOGIN;
                    END IF;
                END
                $publisher$
                """
            )
            await connection.execute(
                """
                GRANT EXECUTE ON FUNCTION bddk_meta.publish_verified_corpus_release(
                    pg_catalog.text, pg_catalog.text, pg_catalog.text,
                    pg_catalog.int4, pg_catalog.int4, pg_catalog.int4, pg_catalog.text
                ) TO bddk_release_publisher
                """
            )

            report = await inspect_catalog_integrity(connection)

            assert "acl:bddk_meta.staged_corpus_release_exact" in report.failures
        finally:
            await transaction.rollback()


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_v8_request_table_grant_fails_acl_attestation(pg_pool) -> None:
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await connection.execute("GRANT SELECT ON bddk_meta.corpus_release_requests TO PUBLIC")

            report = await inspect_catalog_integrity(connection)

            assert "acl:bddk_meta.staged_corpus_release_exact" in report.failures
        finally:
            await transaction.rollback()
