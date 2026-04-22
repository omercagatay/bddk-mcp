"""Tests for the backfill engine (scan + execute)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from backfill import (
    BackfillCandidate,
    execute_backfill,
    group_by_signature,
    scan_candidates,
)


class _FakePool:
    """Minimal asyncpg.Pool shim that returns preset rows from fetch()."""

    def __init__(self, rows: list[dict]):
        self._rows = rows
        self.last_sql: str | None = None

    async def fetch(self, sql, *args, **kwargs):
        self.last_sql = sql
        return self._rows


def _row(doc_id: str, signature: str = "markitdown_degraded", len_: int = 20000) -> dict:
    return {
        "document_id": doc_id,
        "title": f"Title for {doc_id}",
        "source_url": f"https://mevzuat.gov.tr/{doc_id}",
        "category": "Yönetmelik",
        "decision_date": "",
        "decision_number": "",
        "len": len_,
        "signature": signature,
    }


# ── scan_candidates ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_scan_default_uses_degraded_only_sql():
    pool = _FakePool([_row("mevzuat_10522"), _row("mevzuat_5411")])
    candidates = await scan_candidates(pool)

    assert len(candidates) == 2
    assert isinstance(candidates[0], BackfillCandidate)
    assert "extraction_method = 'markitdown_degraded'" in pool.last_sql
    # The default scan must NOT mix legacy corruption filters in.
    assert "chr(65533)" not in pool.last_sql
    assert "<img" not in pool.last_sql


@pytest.mark.asyncio
async def test_scan_include_legacy_expands_filters():
    pool = _FakePool([_row("mevzuat_1", "ufffd", 15000), _row("mevzuat_2", "leaked_img", 8000)])
    candidates = await scan_candidates(pool, include_legacy_corruption=True)

    assert len(candidates) == 2
    # Legacy scan mode must include all four signatures in the SQL.
    assert "chr(65533)" in pool.last_sql
    assert "<img" in pool.last_sql
    assert "LENGTH(d.markdown_content) < 500" in pool.last_sql
    assert "markitdown_degraded" in pool.last_sql


@pytest.mark.asyncio
async def test_scan_respects_limit():
    rows = [_row(f"mevzuat_{i}") for i in range(50)]
    pool = _FakePool(rows)
    candidates = await scan_candidates(pool, limit=5)

    assert len(candidates) == 5
    assert candidates[0].document_id == "mevzuat_0"
    assert candidates[-1].document_id == "mevzuat_4"


@pytest.mark.asyncio
async def test_scan_coerces_null_fields_to_empty_strings():
    """DB NULLs must not leak into the typed dataclass as None."""
    row = _row("mevzuat_1")
    row["title"] = None
    row["source_url"] = None
    row["category"] = None
    pool = _FakePool([row])
    candidates = await scan_candidates(pool)

    assert candidates[0].title == ""
    assert candidates[0].source_url == ""
    assert candidates[0].category == ""


def test_group_by_signature_counts_correctly():
    candidates = [
        BackfillCandidate("mevzuat_1", "", "", "", "", "", 100, "markitdown_degraded"),
        BackfillCandidate("mevzuat_2", "", "", "", "", "", 100, "markitdown_degraded"),
        BackfillCandidate("mevzuat_3", "", "", "", "", "", 100, "ufffd"),
    ]
    counts = group_by_signature(candidates)
    assert counts == {"markitdown_degraded": 2, "ufffd": 1}


# ── execute_backfill ─────────────────────────────────────────────────────────


def _make_syncer(results: list) -> MagicMock:
    """Return a mock syncer whose sync_document returns each result in order."""
    syncer = MagicMock()
    syncer.sync_document = AsyncMock(side_effect=results)
    return syncer


def _sync_ok(doc_id: str, method: str = "mevzuat_iframe+html_parser", size: int = 70000):
    r = MagicMock()
    r.document_id = doc_id
    r.success = True
    r.method = method
    r.size_bytes = size
    r.error = None
    return r


def _sync_fail(doc_id: str, error: str = "network timeout"):
    r = MagicMock()
    r.document_id = doc_id
    r.success = False
    r.method = None
    r.size_bytes = 0
    r.error = error
    return r


@pytest.mark.asyncio
async def test_execute_backfill_calls_sync_document_with_force_true():
    candidates = [BackfillCandidate("mevzuat_1", "t", "url", "cat", "", "", 1000, "markitdown_degraded")]
    syncer = _make_syncer([_sync_ok("mevzuat_1")])

    await execute_backfill(syncer, candidates, inter_request_delay=0)

    syncer.sync_document.assert_awaited_once()
    kwargs = syncer.sync_document.await_args.kwargs
    assert kwargs["doc_id"] == "mevzuat_1"
    assert kwargs["force"] is True
    assert kwargs["title"] == "t"
    assert kwargs["source_url"] == "url"


@pytest.mark.asyncio
async def test_execute_backfill_partitions_ok_and_failed():
    candidates = [BackfillCandidate(f"mevzuat_{i}", "", "", "", "", "", 1000, "markitdown_degraded") for i in range(3)]
    syncer = _make_syncer([_sync_ok("mevzuat_0"), _sync_fail("mevzuat_1", "boom"), _sync_ok("mevzuat_2")])

    report = await execute_backfill(syncer, candidates, inter_request_delay=0)

    assert report.total == 3
    assert report.ok == ["mevzuat_0", "mevzuat_2"]
    assert report.failed == [("mevzuat_1", "boom")]
    assert report.elapsed_seconds >= 0


@pytest.mark.asyncio
async def test_execute_backfill_captures_exceptions_as_failures():
    """A raised exception must not abort the loop — record it and move on."""
    candidates = [BackfillCandidate(f"mevzuat_{i}", "", "", "", "", "", 1000, "markitdown_degraded") for i in range(2)]
    syncer = MagicMock()
    syncer.sync_document = AsyncMock(side_effect=[RuntimeError("http 500"), _sync_ok("mevzuat_1")])

    report = await execute_backfill(syncer, candidates, inter_request_delay=0)

    assert report.ok == ["mevzuat_1"]
    assert len(report.failed) == 1
    assert report.failed[0][0] == "mevzuat_0"
    assert "RuntimeError" in report.failed[0][1]


@pytest.mark.asyncio
async def test_execute_backfill_invokes_progress_callback_per_doc():
    candidates = [BackfillCandidate(f"mevzuat_{i}", "", "", "", "", "", 1000, "markitdown_degraded") for i in range(3)]
    syncer = _make_syncer([_sync_ok(f"mevzuat_{i}") for i in range(3)])

    seen: list[tuple[int, int, str, bool]] = []

    async def on_progress(index, total, outcome):
        seen.append((index, total, outcome.document_id, outcome.success))

    await execute_backfill(syncer, candidates, inter_request_delay=0, on_progress=on_progress)

    assert seen == [
        (1, 3, "mevzuat_0", True),
        (2, 3, "mevzuat_1", True),
        (3, 3, "mevzuat_2", True),
    ]


@pytest.mark.asyncio
async def test_execute_backfill_on_empty_candidates_is_noop():
    syncer = _make_syncer([])
    report = await execute_backfill(syncer, [], inter_request_delay=0)
    assert report.total == 0
    assert report.ok == []
    assert report.failed == []
    syncer.sync_document.assert_not_awaited()
