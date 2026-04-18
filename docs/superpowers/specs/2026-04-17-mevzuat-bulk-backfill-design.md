# Mevzuat Bulk Backfill & Vector-Store Re-Index — Design

**Date:** 2026-04-17
**Scope:** Task 13 of `2026-04-17-mevzuat-formula-extraction-lightocr-plan.md`
**Status:** Approved — ready for implementation plan

## Motivation

Two problems surfaced while validating Task 12 (force-sync of `mevzuat_42628`):

1. **Latent bug — vector store is never refreshed on sync.** `DocumentSyncer.sync_document`
   writes fresh markdown into the `documents` table but never calls
   `vector_store.add_document(...)`, so `document_chunks` keep whatever was written
   at original ingest time. Force-sync silently produces inconsistent DB state —
   the MCP tools (`get_bddk_document`, `search_document_store`) still return the
   stale pre-re-extraction content.

2. **~66 existing mevzuat rows have corrupted extractions** from the pre-LightOCR
   pipeline (`\ufffd` replacement chars, leaked `<img>` tags, or truncated
   content). These need to be re-pulled and re-extracted through the new backend
   chain built in Tasks 1–12.

These are coupled: bulk-resyncing the 66 rows is pointless if
`sync_document` doesn't also refresh `document_chunks`, because the MCP tools
serve from chunks.

## Design

Two ordered parts. Part A lands first with a unit test; Part B depends on it.

### Part A — Fix `sync_document` to always re-index

**File:** `doc_sync.py` (inside the existing `sync_document` method)

**Behavior:** After a successful sync that updates `documents.content_markdown`,
the method must also refresh `document_chunks`:

```python
await self.vector_store.delete_document(doc_id)
await self.vector_store.add_document(doc_id, markdown, metadata)
```

- Runs on **every** successful sync (both polling and force). No flag gating.
- If the extraction returned no new content, the `documents` write is skipped
  and so is the re-index (current behavior preserved).
- The two vector-store calls run through the same asyncpg pool as the
  `documents` update. If we can wrap them in a single transaction without
  major refactor we do; otherwise we accept that a crash between the
  `documents` write and the chunk write leaves the row re-indexable on the
  next sync (idempotent by design).

**Why always, not conditional on content-hash:** keeps the call site simple,
guarantees chunks and document row agree, and re-embedding a handful of docs
per day on polling sync is cheap relative to the HTTP+OCR cost that already
ran.

### Part B — One-shot bulk backfill script

**File:** `scripts/backfill_mevzuat.py`

**Candidate selection (live SQL):**

```sql
SELECT document_id, source_type, LENGTH(content_markdown) AS len
FROM documents
WHERE source_type LIKE 'mevzuat_%'
  AND (
    content_markdown LIKE '%' || chr(65533) || '%'   -- U+FFFD replacement
    OR content_markdown LIKE '%<img%'                -- leaked HTML
    OR LENGTH(content_markdown) < 500                 -- suspiciously short
  )
ORDER BY document_id;
```

**Flow:**

1. Run the query, print `Found N candidates: [id1, id2, ...]` and a count
   breakdown per signature.
2. Prompt `Proceed with re-sync of N docs? [y/N]` — skip prompt if `--yes`
   was passed.
3. For each candidate, in order:
   - Call the **same** `sync_document(doc_id, force=True)` path that Part A
     fixed (no duplicated extraction logic).
   - `time.sleep(2)` between iterations to be polite to mevzuat.gov.tr.
   - On exception: log `ERROR doc_id=... reason=...` and continue.
4. At end, print:
   ```
   Backfill complete: N ok, M failed.
   Failed ids: [...]
   Log: logs/backfill_YYYY-MM-DD-HHMM.log
   ```

**CLI flags:**
- `--yes` — skip interactive confirmation (for replays)
- `--limit N` — cap the candidate count (for smoke testing)

**Logging:**
- One structured log line per doc (start, result, duration)
- Written to both stdout and `logs/backfill_YYYY-MM-DD-HHMM.log`

## Testing

**Part A (unit):**
- Mock the extractor to return a fixed markdown string
- Mock `vector_store.delete_document` and `vector_store.add_document`
- Call `sync_document(doc_id, force=True)`
- Assert both mocks were called exactly once, in order, with the right args

**Part B (smoke, manual):**
- Run `python scripts/backfill_mevzuat.py --limit 1` against the DB where
  `mevzuat_42628` is already re-synced (so it's either not a candidate, or
  it's the only one when forced).
- Verify the script prints a candidate list, prompts, logs cleanly, and
  leaves the DB in the expected state.
- Then run for real (no `--limit`) against the full candidate set.

**Validation after backfill:**
- Re-run the selection query → expect 0 rows (all candidates cleaned up).
- Spot-check `mevzuat_42628` page 5 via MCP tool → formulas present.

## Non-Goals

- No new MCP tool surface — this is a one-shot script.
- No state file / resumability — script is idempotent; re-running it replays
  the failed set.
- No concurrency — LightOCR is GPU-bound and serialized anyway.
- No change to the ingest chain or backend preference order (owned by
  earlier tasks in the parent plan).

## Risks

- **If Part A is wrong, bulk backfill multiplies the damage across 66 rows.**
  Mitigation: unit test for Part A and a `--limit 1` smoke run before the
  full backfill.
- **The MCP server caches connections/state** — a restart is needed for the
  running server to observe the new chunks. This is operational, not a code
  change. Document it in the backfill run-book.
- **mevzuat.gov.tr may rate-limit or serve HTML-only for some IDs.** Those
  fall through to the `<img>` signature next time; the script logs them as
  failures and we handle case-by-case.
