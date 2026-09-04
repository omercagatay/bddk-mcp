"""Strict serving regressions for active-corpus epoch changes."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp.exceptions import ToolError
from mcp.shared.memory import create_connected_server_and_client_session

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.core.exceptions import BddkStorageError
from bddk_mcp.corpus_publication import CorpusPublicationError, CorpusReleaseIdentity
from bddk_mcp.ingest.data_sources import InstitutionDirectory
from bddk_mcp.jobs import OperatorJobManager
from bddk_mcp.tools.registry import LOCAL_CORPUS_PUBLIC_TOOL_NAMES, ToolProfile
from bddk_mcp.tools.search import _search_cache
from tests.in_memory_job_repository import InMemoryJobRepository


def _release(fill: str) -> CorpusReleaseIdentity:
    digest = fill * 64
    return CorpusReleaseIdentity(
        release_id=f"corpus_release_sha256_{digest}",
        manifest_id=f"release-{fill * 3}",
        manifest_sha256=digest,
        signer_key_sha256=digest,
        freshness_policy_result="quantified_measured_signature_verified_pass",
        source_detection_slo_seconds=60,
        publication_slo_seconds=120,
        max_manifest_age_seconds=3600,
        retrieval_profile_sha256=digest,
        corpus_state_sha256=digest,
        completed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def _deps(*, release_id: str | None = None) -> Dependencies:
    client = MagicMock()
    client.load_cache_read_only = AsyncMock(return_value=2)
    doc_store = MagicMock()
    doc_store.get_document_history = AsyncMock(return_value=[])
    return Dependencies(
        pool=MagicMock(),
        doc_store=doc_store,
        client=client,
        http=MagicMock(),
        served_corpus_release_id=release_id,
    )


def test_local_corpus_gate_covers_the_reviewed_public_read_surface():
    assert LOCAL_CORPUS_PUBLIC_TOOL_NAMES == {
        "search_bddk_regulations",
        "search_document_store",
        "get_bddk_document",
        "get_document_history",
        "get_document_section",
        "search_document_sections",
        "resolve_regulation_status",
        "get_amendment_chain",
        "get_cross_references",
        "get_regulatory_digest",
    }


@pytest.fixture(autouse=True)
def _empty_search_cache():
    _search_cache.clear()
    yield
    _search_cache.clear()


@pytest.mark.asyncio
async def test_inactive_release_rejects_local_read_with_stable_safe_error():
    from bddk_mcp.server import create_mcp

    old = _release("a")
    deps = _deps(release_id=old.release_id)
    _search_cache.set("old-query", object())
    server = create_mcp(deps, require_active_corpus_release=True)

    with patch("bddk_mcp.corpus_serving.inspect_active_corpus_release", new=AsyncMock(return_value=None)):
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool("get_document_history", {"document_id": "943"})

    assert result.isError is True
    assert result.content[0].text == (
        "[ERROR:CORPUS_RELEASE_UNAVAILABLE] retryable=true\n"
        "A verified active regulatory corpus release is not available."
    )
    assert deps.served_corpus_release_id is None
    assert _search_cache.get("old-query") is None
    deps.doc_store.get_document_history.assert_not_awaited()
    deps.client.load_cache_read_only.assert_not_awaited()


@pytest.mark.asyncio
async def test_malformed_release_error_is_sanitized_and_clears_epoch():
    from bddk_mcp.server import create_mcp

    sentinel = "PRIVATE-ROW-CONTENT"
    old = _release("a")
    deps = _deps(release_id=old.release_id)
    server = create_mcp(deps, require_active_corpus_release=True)

    with patch(
        "bddk_mcp.corpus_serving.inspect_active_corpus_release",
        new=AsyncMock(side_effect=CorpusPublicationError(sentinel)),
    ):
        with pytest.raises(ToolError) as exc_info:
            await server.call_tool("get_document_history", {"document_id": "943"})

    assert str(exc_info.value).startswith("[ERROR:CORPUS_RELEASE_UNAVAILABLE] retryable=true")
    assert sentinel not in str(exc_info.value)
    assert deps.served_corpus_release_id is None


@pytest.mark.asyncio
async def test_release_replacement_reloads_exact_catalog_and_clears_search_cache():
    from bddk_mcp.server import create_mcp

    old = _release("a")
    new = _release("b")
    deps = _deps(release_id=old.release_id)
    _search_cache.set("old-query", object())
    server = create_mcp(deps, require_active_corpus_release=True)
    inspect = AsyncMock(return_value=new)

    with patch("bddk_mcp.corpus_serving.inspect_active_corpus_release", new=inspect):
        result = await server.call_tool("get_document_history", {"document_id": "943"})

    assert result
    assert deps.served_corpus_release_id == new.release_id
    deps.client.load_cache_read_only.assert_awaited_once_with(require_nonempty=False)
    deps.doc_store.get_document_history.assert_awaited_once_with("943")
    assert _search_cache.get("old-query") is None
    assert inspect.await_count == 3  # pre-check, reload confirmation, post-check


@pytest.mark.asyncio
async def test_release_replacement_reload_failure_never_serves_old_catalog():
    from bddk_mcp.server import create_mcp

    old = _release("a")
    new = _release("b")
    deps = _deps(release_id=old.release_id)
    deps.client.load_cache_read_only = AsyncMock(side_effect=BddkStorageError("PRIVATE-STORAGE-DETAIL"))
    _search_cache.set("old-query", object())
    server = create_mcp(deps, require_active_corpus_release=True)

    with patch("bddk_mcp.corpus_serving.inspect_active_corpus_release", new=AsyncMock(return_value=new)):
        with pytest.raises(ToolError) as exc_info:
            await server.call_tool("get_document_history", {"document_id": "943"})

    assert str(exc_info.value).startswith("[ERROR:CORPUS_RELEASE_UNAVAILABLE] retryable=true")
    assert "PRIVATE-STORAGE-DETAIL" not in str(exc_info.value)
    assert deps.served_corpus_release_id is None
    assert _search_cache.get("old-query") is None
    deps.doc_store.get_document_history.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_calls_cannot_return_old_epoch_after_replacement():
    from bddk_mcp.server import create_mcp

    old = _release("a")
    new = _release("b")
    current_release = old
    first_body_entered = asyncio.Event()
    allow_first_body_to_finish = asyncio.Event()
    body_calls = 0

    async def inspect_release(_pool):
        return current_release

    async def history(_document_id):
        nonlocal body_calls
        body_calls += 1
        if body_calls == 1:
            _search_cache.set("old-body-query", object())
            first_body_entered.set()
            await allow_first_body_to_finish.wait()
        return []

    deps = _deps(release_id=old.release_id)
    deps.doc_store.get_document_history = AsyncMock(side_effect=history)
    server = create_mcp(deps, require_active_corpus_release=True)

    with patch("bddk_mcp.corpus_serving.inspect_active_corpus_release", side_effect=inspect_release):
        old_call = asyncio.create_task(server.call_tool("get_document_history", {"document_id": "943"}))
        await first_body_entered.wait()
        current_release = new
        new_call = asyncio.create_task(server.call_tool("get_document_history", {"document_id": "943"}))
        await asyncio.sleep(0)
        assert body_calls == 1
        allow_first_body_to_finish.set()

        with pytest.raises(ToolError, match="CORPUS_RELEASE_UNAVAILABLE"):
            await old_call
        new_result = await new_call

    assert new_result
    assert body_calls == 2
    assert deps.served_corpus_release_id == new.release_id
    assert _search_cache.get("old-body-query") is None
    deps.client.load_cache_read_only.assert_awaited_once_with(require_nonempty=False)


@pytest.mark.asyncio
async def test_same_epoch_reader_leases_allow_tool_bodies_to_overlap():
    from bddk_mcp.server import create_mcp

    release = _release("a")
    both_entered = asyncio.Event()
    allow_bodies_to_finish = asyncio.Event()
    body_calls = 0

    async def history(_document_id):
        nonlocal body_calls
        body_calls += 1
        if body_calls == 2:
            both_entered.set()
        await allow_bodies_to_finish.wait()
        return []

    deps = _deps(release_id=release.release_id)
    deps.doc_store.get_document_history = AsyncMock(side_effect=history)
    server = create_mcp(deps, require_active_corpus_release=True)

    with patch("bddk_mcp.corpus_serving.inspect_active_corpus_release", new=AsyncMock(return_value=release)):
        first = asyncio.create_task(server.call_tool("get_document_history", {"document_id": "943"}))
        second = asyncio.create_task(server.call_tool("get_document_history", {"document_id": "943"}))
        await asyncio.wait_for(both_entered.wait(), timeout=1)

        assert deps.active_corpus_readers == 2
        assert not deps.active_corpus_idle.is_set()
        deps.client.load_cache_read_only.assert_not_awaited()

        allow_bodies_to_finish.set()
        first_result, second_result = await asyncio.gather(first, second)

    assert first_result
    assert second_result
    assert deps.active_corpus_readers == 0
    assert deps.active_corpus_idle.is_set()


@pytest.mark.asyncio
async def test_epoch_switch_waits_until_every_prior_reader_has_drained():
    from bddk_mcp.server import create_mcp

    old = _release("a")
    new = _release("b")
    current_release = old
    old_bodies_entered = asyncio.Event()
    allow_old_bodies_to_finish = asyncio.Event()
    body_calls = 0

    async def inspect_release(_pool):
        return current_release

    async def history(_document_id):
        nonlocal body_calls
        body_calls += 1
        if body_calls <= 2:
            if body_calls == 2:
                old_bodies_entered.set()
            await allow_old_bodies_to_finish.wait()
        return []

    deps = _deps(release_id=old.release_id)
    deps.doc_store.get_document_history = AsyncMock(side_effect=history)
    server = create_mcp(deps, require_active_corpus_release=True)

    with patch("bddk_mcp.corpus_serving.inspect_active_corpus_release", side_effect=inspect_release):
        old_calls = [
            asyncio.create_task(server.call_tool("get_document_history", {"document_id": str(index)}))
            for index in (1, 2)
        ]
        await asyncio.wait_for(old_bodies_entered.wait(), timeout=1)
        current_release = new
        new_call = asyncio.create_task(server.call_tool("get_document_history", {"document_id": "3"}))
        await asyncio.sleep(0)

        assert body_calls == 2
        deps.client.load_cache_read_only.assert_not_awaited()

        allow_old_bodies_to_finish.set()
        for call in old_calls:
            with pytest.raises(ToolError, match="CORPUS_RELEASE_UNAVAILABLE"):
                await call
        new_result = await new_call

    assert new_result
    assert body_calls == 3
    assert deps.served_corpus_release_id == new.release_id
    deps.client.load_cache_read_only.assert_awaited_once_with(require_nonempty=False)


@pytest.mark.asyncio
async def test_last_stale_reader_write_is_cleared_before_new_epoch_reload():
    from bddk_mcp.server import create_mcp

    old = _release("a")
    new = _release("b")
    current_release = old
    both_entered = asyncio.Event()
    finish_first = asyncio.Event()
    finish_second = asyncio.Event()
    body_calls = 0

    async def inspect_release(_pool):
        return current_release

    async def history(_document_id):
        nonlocal body_calls
        body_calls += 1
        call_number = body_calls
        if body_calls == 2:
            both_entered.set()
        if call_number == 1:
            await finish_first.wait()
        else:
            await finish_second.wait()
            _search_cache.set("late-stale-write", object())
        return []

    deps = _deps(release_id=old.release_id)
    deps.doc_store.get_document_history = AsyncMock(side_effect=history)
    server = create_mcp(deps, require_active_corpus_release=True)

    with patch("bddk_mcp.corpus_serving.inspect_active_corpus_release", side_effect=inspect_release):
        first = asyncio.create_task(server.call_tool("get_document_history", {"document_id": "1"}))
        second = asyncio.create_task(server.call_tool("get_document_history", {"document_id": "2"}))
        await asyncio.wait_for(both_entered.wait(), timeout=1)
        current_release = new

        finish_first.set()
        with pytest.raises(ToolError, match="CORPUS_RELEASE_UNAVAILABLE"):
            await first
        assert deps.active_corpus_readers == 1

        finish_second.set()
        with pytest.raises(ToolError, match="CORPUS_RELEASE_UNAVAILABLE"):
            await second

    assert deps.active_corpus_readers == 0
    assert deps.active_corpus_idle.is_set()
    assert _search_cache.get("late-stale-write") is None


@pytest.mark.asyncio
async def test_cancelled_body_returns_reader_lease_and_keeps_guard_usable():
    from bddk_mcp.corpus_serving import ActiveCorpusGuard

    release = _release("a")
    deps = _deps(release_id=release.release_id)
    guard = ActiveCorpusGuard(deps, required=True)
    body_entered = asyncio.Event()

    async def guarded_body():
        async with guard.tool_call("get_document_history"):
            body_entered.set()
            await asyncio.Event().wait()

    with patch("bddk_mcp.corpus_serving.inspect_active_corpus_release", new=AsyncMock(return_value=release)):
        task = asyncio.create_task(guarded_body())
        await asyncio.wait_for(body_entered.wait(), timeout=1)
        assert deps.active_corpus_readers == 1

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert deps.active_corpus_readers == 0
    assert deps.active_corpus_idle.is_set()


@pytest.mark.asyncio
async def test_non_strict_local_read_does_not_inspect_or_reload_release():
    from bddk_mcp.server import create_mcp

    deps = _deps()
    server = create_mcp(deps, require_active_corpus_release=False)
    inspect = AsyncMock(side_effect=AssertionError("strict inspection must be disabled"))

    with patch("bddk_mcp.corpus_serving.inspect_active_corpus_release", new=inspect):
        result = await server.call_tool("get_document_history", {"document_id": "943"})

    assert result
    inspect.assert_not_awaited()
    deps.client.load_cache_read_only.assert_not_awaited()
    deps.doc_store.get_document_history.assert_awaited_once_with("943")


@pytest.mark.asyncio
async def test_strict_mode_leaves_open_world_and_operator_recovery_calls_usable():
    from bddk_mcp.server import create_mcp

    deps = _deps()
    manager = OperatorJobManager(InMemoryJobRepository())
    deps.job_manager = manager
    server = create_mcp(
        deps,
        profile=ToolProfile.OPERATOR,
        require_active_corpus_release=True,
    )
    inspect = AsyncMock(side_effect=AssertionError("unrelated tools must bypass the corpus gate"))

    with (
        patch("bddk_mcp.corpus_serving.inspect_active_corpus_release", new=inspect),
        patch(
            "bddk_mcp.tools.search.fetch_institutions_with_status",
            new=AsyncMock(
                return_value=InstitutionDirectory(
                    institutions=[
                        {
                            "name": "Örnek Banka",
                            "type": "Banka",
                            "status": "Aktif",
                            "website": "",
                        }
                    ],
                    failed_pages=0,
                    attempted_pages=5,
                )
            ),
        ),
    ):
        institutions = await server.call_tool("search_bddk_institutions", {})
        jobs = await server.call_tool("list_operator_jobs", {})

    assert institutions
    assert jobs
    inspect.assert_not_awaited()
