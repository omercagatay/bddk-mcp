# `scripts/patch_doc.py` — Operator Helper for Manual Document Corrections

**Date:** 2026-04-23
**Status:** Approved — ready for implementation plan

## Motivation

Some documents have content the extraction pipeline cannot produce — most
commonly formulas rendered as GIF images in mevzuat HTML that neither
markitdown nor html_parser can recover. On CPU-only Railway deployments the
formula-capable OCR backend (chandra2, `gpu` group) is unavailable, so those
gaps are permanent from the pipeline's side.

The current workaround, exercised for `mevzuat_20029` this session, is a manual
LaTeX insertion into `markdown_content`. That patch is durable only if three
things land together:

1. `documents` row — `markdown_content`, `content_hash`, `extraction_method`
2. `document_chunks` rows — text + hash, embeddings regenerated lazily on search
3. `seed_data/documents.json` and `seed_data/chunks.json` — otherwise the
   startup seed check detects drift and re-imports from the JSON baseline,
   silently reverting the DB patch on the next container restart

Today there is no single operator path that does all three. The session
produced six ad-hoc one-off scripts (`_push_mevzuat_20029.py`,
`_bump_extraction_method.py`, `_refresh_content_hash.py`, etc.). A planned
full `seed.py export` to resync seed_data was blocked as scope escalation —
correctly, because exporting regenerates ~24 MB of JSON with non-deterministic
row ordering, dwarfing the actual correction.

We expect an ongoing stream of similar corrections as operators audit the
corpus. The right shape is one script that performs the full ritual atomically
with a scoped, reviewable diff.

## Design

**File:** `scripts/patch_doc.py`

**Purpose:** Given a doc_id and a corrected markdown file, update the live DB
**and** surgically rewrite only that document's entries in `seed_data/`. Exit
leaves the operator with a small, reviewable `git diff seed_data/` ready to
commit.

### Interface

```
uv run python scripts/patch_doc.py <doc_id> \
    --markdown <path-to-corrected-markdown> \
    [--extraction-method <tag>] \
    [--dry-run]
```

- `<doc_id>` — positional, required. The exact document_id in the `documents`
  table (e.g. `mevzuat_20029`).
- `--markdown` — required path to the full corrected document body. Accepts
  either raw markdown or a docs_dump-style file with a `---` separator
  (`# Title\n- Document ID: ...\n---\n<body>`); the header is stripped if
  present, using the same split logic as `update_mevzuat_20029.py`.
- `--extraction-method` — default `html_parser+manual_latex`. This marker was
  added to `_FORMULA_AWARE_TOKENS` in PR #55 so the MCP formula-unaware banner
  is suppressed on patched docs.
- `--dry-run` — performs validation, header stripping, and chunk regeneration
  but skips the DB UPDATE and JSON file writes. Prints the summary that the
  non-dry path would print, prefixed `[DRY RUN]`.
- `BDDK_DATABASE_URL` env var required (non-dry-run only). Typical production
  invocation:
  ```bash
  railway run --service Postgres -- bash -c \
      'BDDK_DATABASE_URL="$DATABASE_PUBLIC_URL" \
       uv run python scripts/patch_doc.py mevzuat_20029 \
           --markdown mevzuat_20029_updated.md'
  ```

### Flow

1. **Validate inputs.**
   - Markdown file exists and is non-empty.
   - `doc_id` exists in both the live `documents` table and
     `seed_data/documents.json`. Abort with a clear error if either is missing:
     this script patches existing docs, not inserts new ones.
2. **Strip header.** Split on the first `\n---\n`; keep the body. No-op if no
   separator is present.
3. **Pre-compute + sanity check (no writes yet).**
   - Compute `new_hash = SHA-256(body)`.
   - Regenerate chunks via `vector_store._chunk_text(body)` — abort if
     empty (empty-body guard).
   - On `--dry-run`, print the summary (prefixed `[DRY RUN]`) here and exit 0.
4. **DB update.** Reuse existing engine-level methods rather than raw SQL:
   - `doc_store.store_document(StoredDocument(...))` — handles
     `content_hash = SHA-256(body)`, `total_pages` (derived from body
     length), and archives the prior row into `document_versions`
     atomically. The script reads the current DB row first to populate
     title, category, dates, and source_url unchanged; only
     `markdown_content`, `extraction_method`, and the derived
     `content_hash` / `total_pages` / `extracted_at` fields change.
   - `vector_store.add_document(doc_id, title, content, ...)` — deletes old
     chunks and writes fresh ones with embeddings. Cost: one embedding model
     load (~10 s first run, cached thereafter) and N chunk embeds. This
     runs on the operator's machine at patch time; no change to container
     startup cost.
5. **Seed surgery.**
   - Load `seed_data/documents.json`, locate the entry with
     `document_id == target`, update `markdown_content`, `content_hash`,
     `extraction_method`, `extracted_at` (Unix epoch, `time.time()`), and
     `total_pages`. Preserve every other field. Write back with
     `ensure_ascii=False, indent=2` (matching `seed.py export`).
   - Load `seed_data/chunks.json`, filter out all entries with
     `doc_id == target`, regenerate chunks via
     `vector_store._chunk_text(body)`, append fresh entries mirroring the
     schema in `seed.py:70–81` (`doc_id, chunk_index, title, category,
     decision_date, decision_number, source_url, total_chunks,
     total_pages, content_hash, chunk_text`). Write back with the same
     JSON options. This matches the text-only chunk pattern used by both
     `seed.py import_seed` and `scripts/regen_chunks_seed.py`.
6. **Print summary.**
   - Before / after `content_hash`.
   - Char count delta, chunk count delta.
   - Final line: `run 'git diff --stat seed_data/' to review changes`.

### Shared helpers

`update_mevzuat_20029.py` already contains the header-strip logic. Rather than
copy it, extract `_strip_docs_dump_header(text: str) -> str` into `seed.py`
alongside the other seed utilities and have both the new `patch_doc.py` and
any future users of that format import from there. `update_mevzuat_20029.py`
itself is deleted as part of the cleanup (see Follow-up).

### Safety

- **Abort on missing doc.** No auto-create path. Prevents operator typos
  (`mevzuat_20030` vs `mevzuat_20029`) from silently seeding a new row.
- **Dry-run.** Exercises all validation + chunking logic but skips mutations.
  Prints what would change.
- **No git integration.** Script does not stage, commit, or push. Operator
  reviews `git diff seed_data/` manually. Keeps the script's blast radius to
  two JSON files and one DB row.
- **Content hash round-trip assert.** Since the body hash is computed once
  (step 3) and reused in both the `documents.json` entry and every
  `chunks.json` entry for this doc, they cannot diverge. An
  `assert chunk["content_hash"] == doc_hash` on every new chunk is
  belt-and-suspenders protection against future refactoring that
  accidentally introduces per-chunk hashing. Failure is a programming
  error — raise `AssertionError` (exit code 1).
- **Dry-run short-circuits before writes.** `--dry-run` exits after step 3
  with no DB or filesystem mutation. Operator sees the planned chunk count
  and hash delta without side effects.
- **DB writes precede seed writes.** If the DB call fails, seed files are
  untouched and the operator can re-run without risking JSON/DB skew. If
  the seed write fails after a successful DB write, the operator re-runs
  and `store_document`'s ON CONFLICT UPDATE makes the DB side idempotent.

## Testing

**File:** `tests/test_patch_doc.py`

One integration-style test that:

- Creates a `tmp_path / "seed_data"` with minimal `documents.json` and
  `chunks.json` containing two docs (target + a sibling).
- Mocks `DocumentStore.store_document` and `VectorStore.add_document` with
  `AsyncMock` so we do not need a live DB.
- Runs `patch_doc.main(["mevzuat_20029", "--markdown", <tmpfile>])` (or
  equivalent programmatic entry point).
- Asserts:
  1. The mocked `store_document` was awaited once with the correct
     `markdown_content` and `extraction_method`.
  2. The mocked `add_document` was awaited once with the same body.
  3. `seed_data/documents.json` has the target doc's `markdown_content`,
     `content_hash`, and `extraction_method` updated — and the sibling
     doc's entry is byte-identical to before.
  4. `seed_data/chunks.json` has no entries with `doc_id == 'mevzuat_20029'`
     carrying the old content_hash, and every new entry for that doc_id
     carries the new hash — and sibling-doc chunks are byte-identical.
  5. Round-trip assert fires on a seeded hash mismatch (negative case).

Plus a small unit test for `_strip_docs_dump_header` covering both
has-header and no-header inputs.

## Non-goals

- No interactive mode, no config file.
- No automatic git commit or PR creation.
- No support for adding brand-new docs (use `seed.py import` or the sync
  pipeline).
- No `--seed-only` mode that skips the DB. `scripts/regen_chunks_seed.py`
  already covers the "edit documents.json, regen chunks" case for full-corpus
  re-chunking.

## Follow-up cleanup (same PR)

Delete the session's one-off scripts — all superseded by `patch_doc.py`:

- `_push_mevzuat_20029.py`
- `_bump_extraction_method.py`
- `_refresh_content_hash.py`
- `_backup_mevzuat_20029.py`
- `_check_20029.py`
- `_check_hashes.py`
- `update_mevzuat_20029.py` (header-strip helper moves into `seed.py`)

The local working files (`mevzuat_20029_updated.md`,
`backup_mevzuat_20029_*.sql`, `docs_dump/mevzuat_20029.md`) stay untracked —
operator workspace, not repo content.

## Out-of-scope but worth tracking

- **Operator runbook.** A short doc at `docs/runbooks/patch-doc.md` showing
  the end-to-end workflow (patch DB via `railway run`, review seed diff,
  commit, PR, merge, deploy, verify via MCP). Not required for the script
  to work; helpful for the next operator who inherits this.
- **GPU path parity.** If chandra2 becomes available on production (GPU
  upgrade), mevzuat_20029's formula would extract correctly on next sync,
  and the manual override would need to be removed to avoid divergence.
  Detection mechanism (e.g. a nightly diff report) is a separate concern.
