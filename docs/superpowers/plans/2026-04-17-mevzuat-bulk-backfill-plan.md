# Mevzuat Bulk Backfill & Vector-Store Re-Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the latent bug where `sync_document` never re-indexes the vector store, then run a one-shot bulk backfill over ~66 corrupted mevzuat rows to replace `\ufffd`/`<img>` leftovers with clean LightOCR extractions.

**Architecture:** Two parts, ordered. Part A (Tasks 1–2) adds an optional `vector_store` dependency to `DocumentSyncer`; on every successful sync, `add_document()` is called so `document_chunks` stays consistent with `documents`. `add_document()` already does DELETE+INSERT in a single transaction, so no separate delete call is needed. Part B (Tasks 3–4) is a one-shot script that scans the live DB for corruption signatures, prompts for confirmation, then re-syncs each match serially with 2s politeness delay to mevzuat.gov.tr.

**Tech Stack:** Python 3.11+, asyncpg, pgvector, httpx (existing), pytest + `unittest.mock.AsyncMock`. No new dependencies.

**Project convention:** per user memory, skip git commit steps. Verification = tests pass + manual smoke check.

---

## File Structure

- **Modify** `doc_sync.py` — `DocumentSyncer.__init__` accepts `vector_store`; `sync_document` calls `vector_store.add_document(...)` after `store_document(...)`.
- **Modify** `tools/sync.py:197,267` — pass `deps.vector_store` into `DocumentSyncer(...)`.
- **Modify** `doc_sync.py:676` (CLI `_cli_sync`) — construct a `VectorStore` alongside `DocumentStore` so CLI force-syncs re-index.
- **Create** `tests/test_doc_sync_reindex.py` — unit test asserting `add_document` is called.
- **Create** `scripts/backfill_mevzuat.py` — one-shot backfill script.

---

## Task 1: Add `vector_store` param to `DocumentSyncer` and re-index on success

**Files:**
- Modify: `doc_sync.py:224-264, 340-361`
- Test: `tests/test_doc_sync_reindex.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_doc_sync_reindex.py` with:

```python
"""DocumentSyncer must re-index the vector store after a successful sync."""

from unittest.mock import AsyncMock

import httpx
import pytest

from doc_sync import DocumentSyncer
from ocr_backends import MarkitdownBackend
from tests.conftest import make_http_response


class _DummyStore:
    """Minimal DocumentStore satisfying DocumentSyncer's interface.

    Mirrors the MemStore used in scripts/resync_corrupted_mevzuat.py — lets
    this test run without a Postgres fixture.
    """

    def __init__(self) -> None:
        self.has: set[str] = set()
        self.stored: dict = {}

    async def has_document(self, doc_id: str) -> bool:
        return doc_id in self.has

    async def store_document(self, doc) -> None:
        self.stored[doc.document_id] = doc
        self.has.add(doc.document_id)

    async def clear_sync_failure(self, doc_id: str) -> None:
        pass

    async def record_sync_failure(self, *args, **kwargs) -> None:
        pass

    async def get_pdf_bytes(self, doc_id: str):
        return None


@pytest.mark.asyncio
async def test_sync_document_calls_add_document_on_success():
    """After a successful extraction, vector_store.add_document must be called."""
    store = _DummyStore()
    vector_store = AsyncMock()
    vector_store.add_document = AsyncMock(return_value=3)

    html = (
        "<html><body><h1>Test Doc</h1>"
        "<p>Madde 1 - Bu belge bir test dokumanidir ve icerikte yeterli karakter "
        "bulunmaktadir cunku extraction minimum uzunluk esigini gecmesi "
        "gerekmektedir. " * 4
        + "</p></body></html>"
    )

    async with DocumentSyncer(
        store,
        ocr_backends=[MarkitdownBackend()],
        vector_store=vector_store,
    ) as syncer:
        syncer._http = AsyncMock(spec=httpx.AsyncClient)
        syncer._http.get = AsyncMock(
            return_value=make_http_response(text=html, content_type="text/html")
        )
        result = await syncer.sync_document(
            doc_id="999999",
            title="Test Doc",
            category="karar",
            source_url="https://example.test/999999",
            decision_date="2026-01-01",
            decision_number="999/1",
            force=True,
        )

    assert result.success, f"sync failed: {result.error}"
    assert "999999" in store.stored, "document was not stored"
    vector_store.add_document.assert_awaited_once()
    call_kwargs = vector_store.add_document.await_args.kwargs
    assert call_kwargs["doc_id"] == "999999"
    assert call_kwargs["title"] == "Test Doc"
    assert call_kwargs["category"] == "karar"
    assert call_kwargs["source_url"] == "https://example.test/999999"
    assert call_kwargs["content"]  # non-empty markdown
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_doc_sync_reindex.py -v`

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'vector_store'`.

- [ ] **Step 3: Add `vector_store` param to `DocumentSyncer.__init__`**

In `doc_sync.py`, update the `__init__` signature (around line 227). Before:

```python
    def __init__(
        self,
        store: DocumentStore,
        request_timeout: float = REQUEST_TIMEOUT,
        ocr_backends: "list[OCRBackend] | None" = None,
        progress_callback: "Callable[[str, int, int], None] | None" = None,
        http: httpx.AsyncClient | None = None,
    ) -> None:
```

After:

```python
    def __init__(
        self,
        store: DocumentStore,
        request_timeout: float = REQUEST_TIMEOUT,
        ocr_backends: "list[OCRBackend] | None" = None,
        progress_callback: "Callable[[str, int, int], None] | None" = None,
        http: httpx.AsyncClient | None = None,
        vector_store: "VectorStore | None" = None,
    ) -> None:
```

Then at the top of the file, add the TYPE_CHECKING import (near other imports around line 34):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vector_store import VectorStore
```

And in `__init__`, after `self._ocr_backends = ...`, add:

```python
        self._vector_store = vector_store
```

- [ ] **Step 4: Call `add_document` after `store_document`**

In `doc_sync.py`, in `sync_document`, locate the success block (around line 340):

```python
        await self._store.store_document(doc)
        await self._store.clear_sync_failure(doc_id)

        return SyncResult(
            document_id=doc_id,
            success=True,
            method=f"{method}+{extraction_method}",
            size_bytes=len(content),
        )
```

Replace with:

```python
        await self._store.store_document(doc)
        await self._store.clear_sync_failure(doc_id)

        if self._vector_store is not None:
            try:
                await self._vector_store.add_document(
                    doc_id=doc_id,
                    title=title or doc_id,
                    content=markdown,
                    category=category,
                    decision_date=decision_date,
                    decision_number=decision_number,
                    source_url=source_url,
                )
            except Exception as e:
                logger.warning(
                    "Re-index failed for %s after successful sync: %s. "
                    "documents table is fresh; chunks are stale — retry with force=True.",
                    doc_id,
                    e,
                )
                return SyncResult(
                    document_id=doc_id,
                    success=False,
                    method=f"{method}+{extraction_method}",
                    error=f"reindex_failed: {e}",
                    size_bytes=len(content),
                )

        return SyncResult(
            document_id=doc_id,
            success=True,
            method=f"{method}+{extraction_method}",
            size_bytes=len(content),
        )
```

Rationale: if re-index fails we mark the SyncResult as failure so the backfill script's summary captures it. `documents` is already updated but `add_document` is idempotent on retry (it deletes+inserts).

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_doc_sync_reindex.py -v`

Expected: PASS.

- [ ] **Step 6: Confirm existing doc_sync tests still pass (they pass no `vector_store`, should skip re-index)**

Run: `uv run pytest tests/test_doc_sync.py -v`

Expected: all existing tests pass (they don't assert on vector_store; the optional param defaults to None and the new branch is skipped).

---

## Task 2: Wire `deps.vector_store` into production call sites

**Files:**
- Modify: `tools/sync.py:197, 267`
- Modify: `doc_sync.py:672-711` (the `_cli_sync` coroutine)

- [ ] **Step 1: Update `tools/sync.py:197`**

Before:

```python
                async with DocumentSyncer(store, http=deps.http) as syncer:
```

After:

```python
                async with DocumentSyncer(
                    store, http=deps.http, vector_store=deps.vector_store
                ) as syncer:
```

- [ ] **Step 2: Update `tools/sync.py:267`**

Before:

```python
        async with DocumentSyncer(store, http=deps.http) as syncer:
```

After:

```python
        async with DocumentSyncer(
            store, http=deps.http, vector_store=deps.vector_store
        ) as syncer:
```

- [ ] **Step 3: Update `_create_pool_and_store` in `doc_sync.py` to also return a VectorStore**

Before (around line 660):

```python
async def _create_pool_and_store(dsn: str | None) -> tuple:
    """Create asyncpg pool and DocumentStore for CLI usage."""
    import asyncpg as _asyncpg

    from config import DATABASE_URL

    pool = await _asyncpg.create_pool(dsn or DATABASE_URL, min_size=1, max_size=5)
    store = DocumentStore(pool)
    await store.initialize()
    return pool, store
```

After:

```python
async def _create_pool_and_store(dsn: str | None) -> tuple:
    """Create asyncpg pool, DocumentStore, and VectorStore for CLI usage."""
    import asyncpg as _asyncpg

    from config import DATABASE_URL
    from vector_store import VectorStore

    pool = await _asyncpg.create_pool(dsn or DATABASE_URL, min_size=1, max_size=5)
    store = DocumentStore(pool)
    await store.initialize()

    vs: VectorStore | None
    try:
        vs = VectorStore(pool)
        await vs.initialize()
    except Exception as e:
        logger.warning("VectorStore init failed (%s) — CLI sync will skip re-index", e)
        vs = None

    return pool, store, vs
```

- [ ] **Step 4: Update the three CLI callers to unpack the new 3-tuple and pass `vs`**

In `_cli_sync` (around line 674), before:

```python
async def _cli_sync(args: argparse.Namespace) -> None:
    """CLI: sync documents."""
    pool, store = await _create_pool_and_store(args.db)
    try:
        async with DocumentSyncer(store) as syncer:
```

After:

```python
async def _cli_sync(args: argparse.Namespace) -> None:
    """CLI: sync documents."""
    pool, store, vs = await _create_pool_and_store(args.db)
    try:
        async with DocumentSyncer(store, vector_store=vs) as syncer:
```

In `_cli_stats` (around line 715), before:

```python
async def _cli_stats(args: argparse.Namespace) -> None:
    """CLI: show store stats."""
    pool, store = await _create_pool_and_store(args.db)
```

After:

```python
async def _cli_stats(args: argparse.Namespace) -> None:
    """CLI: show store stats."""
    pool, store, _vs = await _create_pool_and_store(args.db)
```

In `_cli_import` (around line 735), before:

```python
async def _cli_import(args: argparse.Namespace) -> None:
    """CLI: import metadata from cache without downloading content."""
    pool, store = await _create_pool_and_store(args.db)
```

After:

```python
async def _cli_import(args: argparse.Namespace) -> None:
    """CLI: import metadata from cache without downloading content."""
    pool, store, _vs = await _create_pool_and_store(args.db)
```

- [ ] **Step 5: Verify the test suite still passes after call-site changes**

Run: `uv run pytest tests/ -v --tb=short -x -k "not gpu"`

Expected: all green. (Tests don't invoke the CLI or `tools/sync.py` call sites directly; they pass because the DocumentSyncer signature is backward compatible — `vector_store` is optional.)

- [ ] **Step 6: Lint**

Run: `uv run ruff check doc_sync.py tools/sync.py tests/test_doc_sync_reindex.py`

Expected: no errors. Fix any flagged issues.

---

## Task 3: Write the bulk backfill script

**Files:**
- Create: `scripts/backfill_mevzuat.py`

- [ ] **Step 1: Create the script**

Create `scripts/backfill_mevzuat.py` with exactly this content:

```python
"""One-shot bulk backfill for corrupted mevzuat_* docs in the live DB.

Scans documents for rows matching a corruption signature
(`\\ufffd`, leaked `<img`, or suspiciously short content), prompts for
confirmation, then re-syncs each match serially with a 2s delay between
HTTP fetches. Uses the fixed sync_document path so document_chunks is
refreshed too.

Usage:
    uv run python scripts/backfill_mevzuat.py           # interactive
    uv run python scripts/backfill_mevzuat.py --yes     # skip prompt
    uv run python scripts/backfill_mevzuat.py --limit 1 # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from doc_store import DocumentStore  # noqa: E402
from doc_sync import DocumentSyncer  # noqa: E402
from vector_store import VectorStore  # noqa: E402

LOG_DIR = ROOT / "logs"

SCAN_SQL = """
SELECT
    d.document_id,
    d.source_type,
    d.title,
    d.source_url,
    d.category,
    d.decision_date,
    d.decision_number,
    LENGTH(d.markdown_content) AS len,
    CASE
        WHEN d.markdown_content LIKE '%' || chr(65533) || '%' THEN 'ufffd'
        WHEN d.markdown_content LIKE '%<img%' THEN 'leaked_img'
        WHEN LENGTH(d.markdown_content) < 500 THEN 'too_short'
    END AS signature
FROM documents d
WHERE d.source_type LIKE 'mevzuat_%'
  AND (
    d.markdown_content LIKE '%' || chr(65533) || '%'
    OR d.markdown_content LIKE '%<img%'
    OR LENGTH(d.markdown_content) < 500
  )
ORDER BY d.document_id
"""


def _setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    log_path = LOG_DIR / f"backfill_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return log_path


async def _scan(pool) -> list[dict]:
    rows = await pool.fetch(SCAN_SQL)
    return [dict(r) for r in rows]


def _print_scan_summary(rows: list[dict]) -> None:
    by_sig: dict[str, int] = {}
    for r in rows:
        by_sig[r["signature"]] = by_sig.get(r["signature"], 0) + 1
    print(f"\nFound {len(rows)} candidate docs:")
    for sig, count in sorted(by_sig.items()):
        print(f"  {sig:12s} {count}")
    preview = rows[:10]
    print("\nFirst 10 IDs:")
    for r in preview:
        print(f"  {r['document_id']}  len={r['len']:>6}  sig={r['signature']}")
    if len(rows) > 10:
        print(f"  ... and {len(rows) - 10} more")


def _confirm(n: int) -> bool:
    ans = input(f"\nProceed with re-sync of {n} docs? [y/N] ").strip().lower()
    return ans in ("y", "yes")


async def _run(args: argparse.Namespace) -> int:
    log = logging.getLogger("backfill")
    import asyncpg

    from config import DATABASE_URL

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    try:
        store = DocumentStore(pool)
        await store.initialize()
        vs = VectorStore(pool)
        await vs.initialize()

        rows = await _scan(pool)
        if args.limit:
            rows = rows[: args.limit]

        if not rows:
            log.info("No candidates found — nothing to do.")
            return 0

        _print_scan_summary(rows)

        if not args.yes and not _confirm(len(rows)):
            log.info("Aborted by user.")
            return 1

        ok: list[str] = []
        failed: list[tuple[str, str]] = []

        async with DocumentSyncer(store, vector_store=vs) as syncer:
            for i, r in enumerate(rows, 1):
                doc_id = r["document_id"]
                log.info("[%d/%d] re-sync %s (sig=%s len=%d)", i, len(rows), doc_id, r["signature"], r["len"])
                t0 = time.monotonic()
                try:
                    result = await syncer.sync_document(
                        doc_id=doc_id,
                        title=r["title"] or "",
                        category=r["category"] or "",
                        source_url=r["source_url"] or "",
                        decision_date=r["decision_date"] or "",
                        decision_number=r["decision_number"] or "",
                        force=True,
                    )
                except Exception as e:
                    dt = time.monotonic() - t0
                    failed.append((doc_id, f"exception: {type(e).__name__}: {e}"))
                    log.error("[%d/%d] %s FAILED in %.1fs: %s", i, len(rows), doc_id, dt, e)
                else:
                    dt = time.monotonic() - t0
                    if result.success:
                        ok.append(doc_id)
                        log.info("[%d/%d] %s OK in %.1fs (method=%s, size=%dB)",
                                 i, len(rows), doc_id, dt, result.method, result.size_bytes)
                    else:
                        failed.append((doc_id, result.error or "unknown"))
                        log.warning("[%d/%d] %s FAILED in %.1fs: %s",
                                    i, len(rows), doc_id, dt, result.error)
                # Politeness sleep before next iteration; skip on last.
                if i < len(rows):
                    await asyncio.sleep(2.0)

        print("\n" + "=" * 60)
        print(f"Backfill complete: {len(ok)} ok, {len(failed)} failed.")
        if failed:
            print(f"\nFailed IDs ({len(failed)}):")
            for doc_id, reason in failed:
                print(f"  {doc_id}: {reason}")
        return 0 if not failed else 1
    finally:
        await pool.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk backfill corrupted mevzuat docs")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation")
    parser.add_argument("--limit", type=int, default=0, help="Max docs to process (0 = all)")
    args = parser.parse_args()

    log_path = _setup_logging()
    print(f"Logging to {log_path}")

    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke the script with `--limit 1` (dry run where you say "n" to the prompt)**

Run: `echo "n" | uv run python scripts/backfill_mevzuat.py --limit 1`

Expected: script prints `Found N candidate docs`, prints the first (up to) 1 ID with its signature, prompts `Proceed with re-sync of 1 docs? [y/N]`, receives "n", prints `Aborted by user.` and exits 1. Log file created under `logs/backfill_*.log`.

- [ ] **Step 3: Lint**

Run: `uv run ruff check scripts/backfill_mevzuat.py`

Expected: no errors.

---

## Task 4: End-to-end verification and full backfill run

**Files:** none modified — this is operational.

- [ ] **Step 1: Run the smoke test for real (single doc, `--yes`)**

Run:

```bash
uv run python scripts/backfill_mevzuat.py --limit 1 --yes
```

Expected:
- Log shows `[1/1] re-sync <doc_id>` with a signature (`ufffd`, `leaked_img`, or `too_short`).
- Either `OK in N.Ns` or a recorded failure reason.
- Final line: `Backfill complete: N ok, M failed.`

If OK: verify that doc's `document_chunks` was refreshed:

```bash
uv run python -c "
import asyncio, asyncpg
from config import DATABASE_URL
async def main():
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2)
    doc_id = '<replace-with-the-id-from-the-run>'
    n = await pool.fetchval('SELECT COUNT(*) FROM document_chunks WHERE doc_id = \$1', doc_id)
    has_ufffd = await pool.fetchval(
        'SELECT EXISTS(SELECT 1 FROM document_chunks WHERE doc_id = \$1 AND chunk_text LIKE \$2)',
        doc_id, '%' + chr(65533) + '%'
    )
    print(f'chunks={n}  has_ufffd={has_ufffd}')
    await pool.close()
asyncio.run(main())
"
```

Expected: `chunks>0  has_ufffd=False`.

- [ ] **Step 2: If the running MCP server is serving stale content, restart it**

Check: `pgrep -a -f server.py`

If a process is shown and the last restart predates today, restart it:

```bash
pkill -f "python.*server.py"
# then restart by your usual mechanism (systemd unit, tmux session, etc.)
```

- [ ] **Step 3: Run the full backfill**

```bash
uv run python scripts/backfill_mevzuat.py --yes 2>&1 | tee /tmp/backfill_full.out
```

Expected: every doc either OKs or logs a reason. Final line `Backfill complete: N ok, M failed.` Note failures for follow-up.

- [ ] **Step 4: Verify the scan now returns 0 (or only the known failures)**

```bash
uv run python -c "
import asyncio, asyncpg
from config import DATABASE_URL
async def main():
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2)
    rows = await pool.fetch(\"\"\"
        SELECT document_id FROM documents
        WHERE source_type LIKE 'mevzuat_%'
          AND (markdown_content LIKE '%' || chr(65533) || '%'
               OR markdown_content LIKE '%<img%'
               OR LENGTH(markdown_content) < 500)
        ORDER BY document_id
    \"\"\")
    print(f'Remaining corrupted: {len(rows)}')
    for r in rows[:20]:
        print(f'  {r[\"document_id\"]}')
    await pool.close()
asyncio.run(main())
"
```

Expected: `Remaining corrupted: 0` (or only docs that failed in Step 3, with the same IDs).

- [ ] **Step 5: Spot-check `mevzuat_42628` via the MCP tool**

Call the `get_bddk_document` MCP tool for `mevzuat_42628`, page 5 (EK-2 region). Verify LaTeX/formula markers (e.g. `\Delta R`, `EDD_{i,p}`, `S_{uzun}(t_o)`, or at least `$` / `\frac`) are present in the returned markdown.

Expected: LaTeX markers visible in the output. If still stale, the MCP server has not been restarted — go back to Task 4 Step 2.

- [ ] **Step 6: Mark Task 13 of the parent plan as complete**

Update `docs/superpowers/plans/2026-04-17-mevzuat-formula-extraction-lightocr-plan.md` Task 13 checkboxes, and the `TaskUpdate` entry for Task 13.

---

## Verification summary

- Unit: `uv run pytest tests/test_doc_sync_reindex.py -v` — green.
- Regression: `uv run pytest tests/ -v --tb=short -x -k "not gpu"` — green.
- Lint: `uv run ruff check .` — clean.
- Smoke: `uv run python scripts/backfill_mevzuat.py --limit 1 --yes` — one doc OK, chunks refreshed, `has_ufffd=False`.
- Full: `uv run python scripts/backfill_mevzuat.py --yes` — summary shows N ok / M failed.
- Scan-after: 0 remaining corrupted rows (or only known failures).
- MCP tool: `mevzuat_42628` page 5 returns content with formula markers.
