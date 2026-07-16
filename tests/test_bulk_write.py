"""Set-based corpus-writer and cross-process coordination regressions."""

from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg
import pytest

from bddk_mcp.corpus_coordination import (
    CORPUS_JOB_EXECUTION_ADVISORY_KEY,
    CORPUS_MUTATION_ADVISORY_KEY,
    acquire_corpus_mutation_lock,
)
from bddk_mcp.store.bulk_write import (
    insert_document_chunk_rows,
    insert_document_metadata_rows,
    insert_document_section_rows,
    upsert_decision_cache_rows,
)
from bddk_mcp.store.doc_store import DocumentStore


class _LockConnection:
    def __init__(self, *, active: bool) -> None:
        self.active = active
        self.fetchval = AsyncMock(return_value=None)

    def is_in_transaction(self) -> bool:
        return self.active


@pytest.mark.asyncio
async def test_mutation_lock_fails_closed_outside_explicit_transaction() -> None:
    connection = _LockConnection(active=False)

    with pytest.raises(RuntimeError, match="active explicit transaction"):
        await acquire_corpus_mutation_lock(connection)

    connection.fetchval.assert_not_awaited()


@pytest.mark.asyncio
async def test_mutation_and_job_admission_use_distinct_stable_int8_keys() -> None:
    connection = _LockConnection(active=True)

    await acquire_corpus_mutation_lock(connection)

    assert -(2**63) <= CORPUS_MUTATION_ADVISORY_KEY < 2**63
    assert -(2**63) <= CORPUS_JOB_EXECUTION_ADVISORY_KEY < 2**63
    assert CORPUS_MUTATION_ADVISORY_KEY != CORPUS_JOB_EXECUTION_ADVISORY_KEY
    assert connection.fetchval.await_args.args[1] == CORPUS_MUTATION_ADVISORY_KEY
    assert "pg_advisory_xact_lock" in connection.fetchval.await_args.args[0]


@pytest.mark.asyncio
async def test_decision_cache_bulk_write_uses_zipped_typed_arrays() -> None:
    connection = _LockConnection(active=True)
    connection.fetchval.return_value = 2
    rows = [
        ("d1", "one", "", "", "", "", "", 1.0),
        ("d2", "two", "", "", "", "", "", 1.0),
    ]

    assert await upsert_decision_cache_rows(connection, rows) == 2

    query = connection.fetchval.await_args.args[0]
    assert "FROM ROWS FROM" in query
    assert query.count("pg_catalog.unnest") == 8
    assert "executemany" not in query.lower()
    assert connection.fetchval.await_args.args[1:] == tuple([list(column) for column in zip(*rows, strict=True)])


@pytest.mark.asyncio
async def test_bulk_writer_rejects_ragged_duplicate_and_inaccurate_batches() -> None:
    connection = _LockConnection(active=True)

    with pytest.raises(ValueError, match="exactly 8"):
        await upsert_decision_cache_rows(connection, [("d1",)])
    with pytest.raises(ValueError, match="duplicate logical key"):
        await insert_document_metadata_rows(
            connection,
            [
                ("d1", "", "", "", "", "", 1.0),
                ("d1", "", "", "", "", "", 1.0),
            ],
        )
    connection.fetchval.return_value = 1
    with pytest.raises(RuntimeError, match="unexpected row count"):
        await upsert_decision_cache_rows(
            connection,
            [
                ("d1", "", "", "", "", "", "", 1.0),
                ("d2", "", "", "", "", "", "", 1.0),
            ],
        )


@pytest.mark.asyncio
async def test_empty_bulk_batches_issue_no_statement() -> None:
    connection = _LockConnection(active=True)

    assert await upsert_decision_cache_rows(connection, []) == 0
    assert await insert_document_metadata_rows(connection, []) == 0
    assert await insert_document_section_rows(connection, []) == 0
    assert await insert_document_chunk_rows(connection, []) == 0
    connection.fetchval.assert_not_awaited()


async def _epoch(connection: asyncpg.Connection) -> int:
    return int(await connection.fetchval("SELECT epoch FROM bddk_meta.corpus_state_epoch WHERE singleton_id"))


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_hot_batches_bound_epoch_growth_by_statements_not_rows(pg_pool) -> None:
    connection = await pg_pool.acquire()
    transaction = connection.transaction()
    await transaction.start()
    prefix = f"bulk-epoch-{uuid4().hex}"
    try:
        await acquire_corpus_mutation_lock(connection)
        decision_rows = [
            (f"{prefix}-cache-{index}", f"title {index}", "", "", "", "test", "", 1.0) for index in range(24)
        ]
        before = await _epoch(connection)
        assert await upsert_decision_cache_rows(connection, decision_rows) == len(decision_rows)
        after = await _epoch(connection)
        assert 1 <= after - before <= 2

        document_rows = [(f"{prefix}-doc-{index}", f"title {index}", "test", "", "", "", 1.0) for index in range(24)]
        before = after
        assert await insert_document_metadata_rows(connection, document_rows) == len(document_rows)
        after = await _epoch(connection)
        assert 1 <= after - before <= 2

        section_rows = [
            (
                f"{prefix}-doc-{index}",
                "madde",
                str(index),
                "",
                0,
                5,
                "MADDE",
                f"{index:064x}",
                None,
                None,
                "",
            )
            for index in range(24)
        ]
        before = after
        assert await insert_document_section_rows(connection, section_rows) == len(section_rows)
        after = await _epoch(connection)
        assert 1 <= after - before <= 2

        vector = "[1," + ",".join("0" for _ in range(767)) + "]"
        chunk_rows = [
            (
                f"{prefix}-doc-{index}",
                0,
                f"title {index}",
                "test",
                "",
                "",
                "",
                1,
                1,
                "",
                0,
                5,
                "madde",
                str(index),
                0,
                5,
                f"{index:064x}",
                "MADDE",
                vector,
            )
            for index in range(24)
        ]
        before = after
        assert await insert_document_chunk_rows(connection, chunk_rows) == len(chunk_rows)
        after = await _epoch(connection)
        # One chunk INSERT plus the statement-level publication invalidation;
        # the bound is independent of the number of chunk rows.
        assert 1 <= after - before <= 2

        before = after
        assert await insert_document_chunk_rows(connection, []) == 0
        assert await _epoch(connection) == before
    finally:
        await transaction.rollback()
        await pg_pool.release(connection)


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_repeated_cache_metadata_import_is_a_true_noop(pg_pool) -> None:
    connection = await pg_pool.acquire()
    transaction = connection.transaction()
    await transaction.start()

    class PinnedPool:
        @asynccontextmanager
        async def acquire(self):
            yield connection

    prefix = f"metadata-noop-{uuid4().hex}"
    items = [{"document_id": f"{prefix}-{index}", "title": f"document {index}"} for index in range(16)]
    try:
        before = await _epoch(connection)
        assert await DocumentStore(PinnedPool()).import_from_cache(items) == len(items)  # type: ignore[arg-type]
        after_first = await _epoch(connection)
        assert after_first > before

        assert await DocumentStore(PinnedPool()).import_from_cache(items) == 0  # type: ignore[arg-type]
        assert await _epoch(connection) == after_first
    finally:
        await transaction.rollback()
        await pg_pool.release(connection)


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_actual_publisher_waits_for_sanctioned_writer_lock(pg_pool) -> None:
    """A writer-first schedule cannot let publication observe an interleaving."""

    async with pg_pool.acquire() as setup:
        role = await setup.fetchval("SELECT pg_catalog.to_regrole('bddk_release_publisher')")
        if role is None:
            await setup.execute("CREATE ROLE bddk_release_publisher NOLOGIN")
        current_user = str(await setup.fetchval("SELECT CURRENT_USER"))
        assert re.fullmatch(r"[a-z_][a-z0-9_$]*", current_user)
        await setup.execute(f'GRANT bddk_release_publisher TO "{current_user}"')

    writer = await pg_pool.acquire()
    publisher = await pg_pool.acquire()
    observer = await pg_pool.acquire()
    writer_tx = writer.transaction()
    publisher_tx = publisher.transaction()
    await writer_tx.start()
    await publisher_tx.start()
    document_id = f"writer-order-{uuid4().hex}"
    publisher_task: asyncio.Task | None = None
    try:
        await acquire_corpus_mutation_lock(writer)
        await writer.execute(
            """
            INSERT INTO public.decision_cache (
                document_id, title, content, decision_date, decision_number,
                category, source_url, cached_at
            ) VALUES ($1, '', '', '', '', '', '', 1.0)
            """,
            document_id,
        )
        publisher_pid = int(await publisher.fetchval("SELECT pg_catalog.pg_backend_pid()"))
        publisher_task = asyncio.create_task(
            publisher.fetchrow(
                """
                SELECT *
                FROM bddk_meta.publish_verified_corpus_release(
                    'lock-order-test', $1, $2, 60, 120, 3600, $3
                )
                """,
                "1" * 64,
                "2" * 64,
                "3" * 64,
            )
        )
        wait_event = None
        for _ in range(200):
            wait_event = await observer.fetchval(
                "SELECT wait_event FROM pg_catalog.pg_stat_activity WHERE pid = $1",
                publisher_pid,
            )
            if wait_event == "advisory":
                break
            await asyncio.sleep(0.01)
        assert wait_event == "advisory"
        assert not publisher_task.done()

        await writer_tx.commit()
        with pytest.raises(asyncpg.PostgresError):
            await asyncio.wait_for(publisher_task, timeout=5)
    finally:
        if publisher_task is not None and not publisher_task.done():
            publisher_task.cancel()
            await asyncio.gather(publisher_task, return_exceptions=True)
        if writer.is_in_transaction():
            await writer_tx.rollback()
        if publisher.is_in_transaction():
            await publisher_tx.rollback()
        await pg_pool.release(observer)
        await pg_pool.release(publisher)
        await pg_pool.release(writer)
        async with pg_pool.acquire() as cleanup, cleanup.transaction():
            await acquire_corpus_mutation_lock(cleanup)
            await cleanup.execute(
                "DELETE FROM public.decision_cache WHERE document_id = $1",
                document_id,
            )
