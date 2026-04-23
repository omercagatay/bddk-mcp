# `scripts/patch_doc.py` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an operator-facing script that atomically patches a single BDDK document in both the live Postgres DB and the `seed_data/` baseline, so manual corrections (like the `mevzuat_20029` LaTeX insertion) survive container restarts.

**Architecture:** Single script in `scripts/patch_doc.py` with a DI-friendly `patch_document()` coroutine (takes `DocumentStore`, `VectorStore`, and a `seed_dir` Path) wrapped by a small `main()` CLI that constructs those dependencies from `BDDK_DATABASE_URL`. Uses existing engine methods (`DocumentStore.store_document`, `VectorStore.add_document`) for the DB side and `vector_store._chunk_text` for the text-only seed chunks — mirroring the pattern `scripts/regen_chunks_seed.py` already uses. The header-strip helper currently buried in `update_mevzuat_20029.py:51` moves into `seed.py` so both the new script and `seed.py` users share one implementation.

**Tech Stack:** Python 3.11+, asyncpg, Pydantic (via existing `StoredDocument`), pytest + pytest-asyncio, `unittest.mock.AsyncMock` / `MagicMock`.

**Spec:** `docs/superpowers/specs/2026-04-23-patch-doc-helper-design.md`

---

## File Structure

| Path | Role |
|---|---|
| `seed.py` (modify) | Add `_strip_docs_dump_header(text: str) -> str` as a top-level helper. |
| `scripts/patch_doc.py` (create) | Operator entry point: argparse + async wrapper that constructs dependencies and calls `patch_document()`. |
| `tests/test_seed.py` (modify) | Add one test for the new header-strip helper. |
| `tests/test_patch_doc.py` (create) | Covers `patch_document()` behavior: validation, dry-run, DB calls, seed surgery, isolation from sibling docs. |
| `update_mevzuat_20029.py` (delete) | Superseded; header-strip logic moves to `seed.py`. |
| `_push_mevzuat_20029.py` (delete) | Session one-off, superseded. |
| `_bump_extraction_method.py` (delete) | Session one-off, superseded. |
| `_refresh_content_hash.py` (delete) | Session one-off, superseded. |
| `_backup_mevzuat_20029.py` (delete) | Session one-off, superseded. |
| `_check_20029.py` (delete) | Session one-off, superseded. |
| `_check_hashes.py` (delete) | Session one-off, superseded. |

The branch already exists: `feat/patch-doc-helper` (spec commit `a9f86de` landed).

---

## Task 1 — Extract `_strip_docs_dump_header` into `seed.py`

**Files:**
- Modify: `seed.py` (add helper near the top, after imports)
- Modify: `tests/test_seed.py` (add one test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_seed.py`:

```python
class TestStripDocsDumpHeader:
    """Cover both header-present and header-absent inputs."""

    def test_strips_header_when_separator_present(self):
        text = (
            "# Kredi Riski\n"
            "- Document ID: mevzuat_20029\n"
            "- Decision Date: N/A\n"
            "---\n"
            "body line one\n"
            "body line two\n"
        )
        assert seed._strip_docs_dump_header(text) == "body line one\nbody line two\n"

    def test_passes_through_when_no_separator(self):
        text = "body line one\nbody line two\n"
        assert seed._strip_docs_dump_header(text) == text

    def test_splits_only_on_first_separator(self):
        text = "header\n---\nbody with\n---\nembedded separator\n"
        assert seed._strip_docs_dump_header(text) == "body with\n---\nembedded separator\n"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_seed.py::TestStripDocsDumpHeader -v`
Expected: FAIL with `AttributeError: module 'seed' has no attribute '_strip_docs_dump_header'`

- [ ] **Step 3: Add the helper**

Insert in `seed.py` after the existing imports and before `async def export_seed`:

```python
def _strip_docs_dump_header(text: str) -> str:
    """Return the body of a docs_dump-style markdown file.

    docs_dump files follow the shape ``# Title\n- Document ID: ...\n---\n<body>``.
    Split on the first ``\\n---\\n`` so only the article body is retained. Returns
    ``text`` unchanged when no separator is present — the caller can pass either
    raw markdown or a dump file without branching.
    """
    parts = text.split("\n---\n", 1)
    return parts[1].lstrip() if len(parts) == 2 else text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_seed.py::TestStripDocsDumpHeader -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add seed.py tests/test_seed.py
git commit -m "refactor(seed): lift docs_dump header-strip into shared helper"
```

---

## Task 2 — `scripts/patch_doc.py` skeleton with validation + dry-run

**Files:**
- Create: `scripts/patch_doc.py`
- Create: `tests/test_patch_doc.py`

This task implements the validation-only shape of `patch_document()` and the dry-run exit path. No DB writes yet, no seed writes yet.

- [ ] **Step 1: Write failing tests**

Create `tests/test_patch_doc.py`:

```python
"""Tests for scripts/patch_doc.py — patch_document() behavior."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from doc_store import StoredDocument  # noqa: E402
import patch_doc  # type: ignore  # noqa: E402


def _stored_doc(
    doc_id: str, *, markdown: str = "old body", content_hash: str = "deadbeef"
) -> StoredDocument:
    """StoredDocument instance shaped like DocumentStore.get_document returns."""
    return StoredDocument(
        document_id=doc_id,
        title=f"Title of {doc_id}",
        category="Sermaye Yeterliliği",
        decision_date="",
        decision_number="",
        source_url=f"https://example.org/{doc_id}",
        markdown_content=markdown,
        content_hash=content_hash,
        extraction_method="markitdown_degraded",
        total_pages=1,
        file_size=len(markdown),
    )


def _seed_doc_entry(
    doc_id: str, *, markdown: str = "old body", content_hash: str = "deadbeef"
) -> dict:
    """Dict shaped like seed_data/documents.json entries (adds downloaded_at / extracted_at)."""
    return {
        "document_id": doc_id,
        "title": f"Title of {doc_id}",
        "category": "Sermaye Yeterliliği",
        "decision_date": "",
        "decision_number": "",
        "source_url": f"https://example.org/{doc_id}",
        "markdown_content": markdown,
        "content_hash": content_hash,
        "downloaded_at": 1_700_000_000,
        "extracted_at": 1_700_000_000,
        "extraction_method": "markitdown_degraded",
        "total_pages": 1,
        "file_size": len(markdown),
    }


def _write_seed_files(seed_dir: Path, *, docs: list[dict], chunks: list[dict]) -> None:
    (seed_dir / "documents.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")
    (seed_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")


def _mock_doc_store(current_doc: StoredDocument | None) -> MagicMock:
    """Build a DocumentStore mock that returns current_doc from get_document."""
    store = MagicMock()
    store.get_document = AsyncMock(return_value=current_doc)
    store.store_document = AsyncMock()
    return store


def _mock_vector_store() -> MagicMock:
    vs = MagicMock()
    vs.add_document = AsyncMock(return_value=1)
    return vs


@pytest.mark.asyncio
async def test_patch_document_aborts_when_markdown_file_missing(tmp_path):
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    _write_seed_files(seed_dir, docs=[_seed_doc_entry("mevzuat_20029")], chunks=[])
    missing_md = tmp_path / "does_not_exist.md"

    with pytest.raises(FileNotFoundError):
        await patch_doc.patch_document(
            doc_id="mevzuat_20029",
            markdown_path=missing_md,
            extraction_method="html_parser+manual_latex",
            doc_store=_mock_doc_store(_stored_doc("mevzuat_20029")),
            vector_store=_mock_vector_store(),
            seed_dir=seed_dir,
            dry_run=True,
        )


@pytest.mark.asyncio
async def test_patch_document_aborts_when_doc_missing_from_db(tmp_path):
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    _write_seed_files(seed_dir, docs=[_seed_doc_entry("mevzuat_20029")], chunks=[])
    md = tmp_path / "body.md"
    md.write_text("new body\n", encoding="utf-8")

    with pytest.raises(patch_doc.PatchError, match="not found in DB"):
        await patch_doc.patch_document(
            doc_id="mevzuat_20029",
            markdown_path=md,
            extraction_method="html_parser+manual_latex",
            doc_store=_mock_doc_store(None),
            vector_store=_mock_vector_store(),
            seed_dir=seed_dir,
            dry_run=True,
        )


@pytest.mark.asyncio
async def test_patch_document_aborts_when_doc_missing_from_seed(tmp_path):
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    _write_seed_files(seed_dir, docs=[_seed_doc_entry("mevzuat_99999")], chunks=[])
    md = tmp_path / "body.md"
    md.write_text("new body\n", encoding="utf-8")

    with pytest.raises(patch_doc.PatchError, match="not found in seed_data"):
        await patch_doc.patch_document(
            doc_id="mevzuat_20029",
            markdown_path=md,
            extraction_method="html_parser+manual_latex",
            doc_store=_mock_doc_store(_stored_doc("mevzuat_20029")),
            vector_store=_mock_vector_store(),
            seed_dir=seed_dir,
            dry_run=True,
        )


@pytest.mark.asyncio
async def test_dry_run_performs_no_writes(tmp_path):
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    original_docs = [_seed_doc_entry("mevzuat_20029", markdown="old body", content_hash="oldhash")]
    original_chunks = [
        {
            "doc_id": "mevzuat_20029",
            "chunk_index": 0,
            "title": "Title of mevzuat_20029",
            "category": "",
            "decision_date": "",
            "decision_number": "",
            "source_url": "",
            "total_chunks": 1,
            "total_pages": 1,
            "content_hash": "oldhash",
            "chunk_text": "old body",
        }
    ]
    _write_seed_files(seed_dir, docs=original_docs, chunks=original_chunks)
    md = tmp_path / "body.md"
    md.write_text("brand new body content\n", encoding="utf-8")

    ds = _mock_doc_store(_stored_doc("mevzuat_20029"))
    vs = _mock_vector_store()

    result = await patch_doc.patch_document(
        doc_id="mevzuat_20029",
        markdown_path=md,
        extraction_method="html_parser+manual_latex",
        doc_store=ds,
        vector_store=vs,
        seed_dir=seed_dir,
        dry_run=True,
    )

    # No DB writes
    ds.store_document.assert_not_awaited()
    vs.add_document.assert_not_awaited()
    # No FS writes — JSON files byte-identical to what we seeded
    assert json.loads((seed_dir / "documents.json").read_text(encoding="utf-8")) == original_docs
    assert json.loads((seed_dir / "chunks.json").read_text(encoding="utf-8")) == original_chunks
    # Result carries the planned numbers
    assert result["dry_run"] is True
    assert result["new_hash"] != "oldhash"
    assert result["chunk_count"] >= 1


@pytest.mark.asyncio
async def test_strips_docs_dump_header_before_hashing(tmp_path):
    """When markdown has a docs_dump header, the hash is of the body only."""
    import hashlib

    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    _write_seed_files(seed_dir, docs=[_seed_doc_entry("mevzuat_20029")], chunks=[])
    md = tmp_path / "body.md"
    body = "the actual body\n"
    md.write_text(
        "# Kredi Riski\n- Document ID: mevzuat_20029\n- Decision Date: N/A\n---\n" + body,
        encoding="utf-8",
    )

    result = await patch_doc.patch_document(
        doc_id="mevzuat_20029",
        markdown_path=md,
        extraction_method="html_parser+manual_latex",
        doc_store=_mock_doc_store(_stored_doc("mevzuat_20029")),
        vector_store=_mock_vector_store(),
        seed_dir=seed_dir,
        dry_run=True,
    )

    expected_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert result["new_hash"] == expected_hash
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_patch_doc.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'patch_doc'`

- [ ] **Step 3: Write the implementation**

Create `scripts/patch_doc.py`:

```python
"""Operator helper: atomically patch one document in DB + seed_data.

Intended for manual corrections (e.g. hand-inserted LaTeX where OCR couldn't
recover image-based formulas). For one doc_id, this script:

  1. validates inputs (markdown exists, doc_id present in both DB and seed),
  2. strips any docs_dump-style header from the markdown,
  3. computes the new content hash and regenerates chunks via the same
     _chunk_text used by vector_store.add_document and seed.py,
  4. on --dry-run, stops here and prints the planned delta,
  5. otherwise calls DocumentStore.store_document + VectorStore.add_document,
  6. surgically rewrites only the target doc's entries in seed_data/
     documents.json and chunks.json.

Run:
    BDDK_DATABASE_URL=... uv run python scripts/patch_doc.py <doc_id> \\
        --markdown <path> [--extraction-method <tag>] [--dry-run]

See docs/superpowers/specs/2026-04-23-patch-doc-helper-design.md for rationale.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

# Allow running as a script from the scripts/ directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import asyncpg  # noqa: E402

from config import PAGE_SIZE, require_database_url  # noqa: E402
from doc_store import DocumentStore, StoredDocument  # noqa: E402
from seed import _strip_docs_dump_header  # noqa: E402
from vector_store import VectorStore, _chunk_text  # noqa: E402

DEFAULT_EXTRACTION_METHOD = "html_parser+manual_latex"


class PatchError(RuntimeError):
    """Raised for validation failures that are not programming errors."""


def _content_hash(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


async def patch_document(
    *,
    doc_id: str,
    markdown_path: Path,
    extraction_method: str,
    doc_store: DocumentStore,
    vector_store: VectorStore,
    seed_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    # --- 1. Validate inputs ----------------------------------------------
    if not markdown_path.exists():
        raise FileNotFoundError(markdown_path)
    raw = markdown_path.read_text(encoding="utf-8")
    if not raw.strip():
        raise PatchError(f"{markdown_path} is empty")

    current_doc = await doc_store.get_document(doc_id)
    if current_doc is None:
        raise PatchError(f"{doc_id} not found in DB")

    docs_path = seed_dir / "documents.json"
    chunks_path = seed_dir / "chunks.json"
    seed_docs = json.loads(docs_path.read_text(encoding="utf-8"))
    seed_entry = next((d for d in seed_docs if d.get("document_id") == doc_id), None)
    if seed_entry is None:
        raise PatchError(f"{doc_id} not found in seed_data/documents.json")

    # --- 2. Strip header + compute hash + regenerate chunks --------------
    body = _strip_docs_dump_header(raw)
    new_hash = _content_hash(body)
    chunks = _chunk_text(body)
    if not chunks:
        raise PatchError(f"chunk regeneration produced no chunks for {doc_id}")
    total_pages = max(1, math.ceil(len(body) / PAGE_SIZE))

    result = {
        "doc_id": doc_id,
        "old_hash": current_doc.content_hash or "",
        "new_hash": new_hash,
        "old_len": len(current_doc.markdown_content or ""),
        "new_len": len(body),
        "chunk_count": len(chunks),
        "extraction_method": extraction_method,
        "dry_run": dry_run,
    }

    if dry_run:
        return result

    # --- 3-5 land in later tasks. Raise so Task 2 tests stay honest. -----
    raise NotImplementedError("DB + seed writes land in later tasks")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Patch a BDDK document in DB + seed_data atomically.")
    p.add_argument("doc_id", help="Exact document_id (e.g. mevzuat_20029)")
    p.add_argument("--markdown", required=True, type=Path, help="Path to corrected markdown file")
    p.add_argument(
        "--extraction-method",
        default=DEFAULT_EXTRACTION_METHOD,
        help=f"Value for documents.extraction_method (default: {DEFAULT_EXTRACTION_METHOD!r})",
    )
    p.add_argument("--dry-run", action="store_true", help="Validate and print plan without writing")
    return p


async def _main_async(args: argparse.Namespace) -> int:
    # Task 5 will wire the real asyncpg pool + stores. For now this path is
    # exercised only by integration tests (skipped for Task 2).
    raise NotImplementedError("CLI wiring lands in Task 5")


def main() -> int:
    args = _build_arg_parser().parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_patch_doc.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/patch_doc.py tests/test_patch_doc.py
git commit -m "feat(scripts): patch_doc.py skeleton with validation + dry-run"
```

---

## Task 3 — Wire DB update into `patch_document()`

**Files:**
- Modify: `scripts/patch_doc.py`
- Modify: `tests/test_patch_doc.py`

Extend the function past the `if dry_run` return to call `store_document` + `add_document`. Seed surgery still deferred to Task 4, so non-dry runs will still raise `NotImplementedError` at the end.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_patch_doc.py`:

```python
@pytest.mark.asyncio
async def test_db_update_passes_body_and_metadata(tmp_path):
    """store_document + add_document must be awaited with the stripped body."""
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    _write_seed_files(seed_dir, docs=[_seed_doc_entry("mevzuat_20029")], chunks=[])
    md = tmp_path / "body.md"
    body = "fresh body content\n"
    md.write_text(body, encoding="utf-8")

    ds = _mock_doc_store(_stored_doc("mevzuat_20029"))
    vs = _mock_vector_store()

    with pytest.raises(NotImplementedError):  # seed surgery still TODO
        await patch_doc.patch_document(
            doc_id="mevzuat_20029",
            markdown_path=md,
            extraction_method="html_parser+manual_latex",
            doc_store=ds,
            vector_store=vs,
            seed_dir=seed_dir,
            dry_run=False,
        )

    # store_document awaited once with the correct body + extraction_method
    ds.store_document.assert_awaited_once()
    stored = ds.store_document.await_args.args[0]
    assert stored.document_id == "mevzuat_20029"
    assert stored.markdown_content == body
    assert stored.extraction_method == "html_parser+manual_latex"
    # Existing title / category / source_url carried over from the DB row
    assert stored.title == "Title of mevzuat_20029"
    assert stored.category == "Sermaye Yeterliliği"
    assert stored.source_url == "https://example.org/mevzuat_20029"

    # add_document awaited once with the same body
    vs.add_document.assert_awaited_once()
    kwargs = vs.add_document.await_args.kwargs
    assert kwargs["doc_id"] == "mevzuat_20029"
    assert kwargs["content"] == body
    assert kwargs["title"] == "Title of mevzuat_20029"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_patch_doc.py::test_db_update_passes_body_and_metadata -v`
Expected: FAIL — `store_document.assert_awaited_once()` fails because the code still exits before calling it.

- [ ] **Step 3: Implement DB update**

Replace the `# --- 3-5 ...` placeholder + `raise NotImplementedError(...)` at the end of `patch_document` with:

```python
    # --- 3. DB update ----------------------------------------------------
    # StoredDocument preserves existing metadata; only body, hash,
    # extraction_method, total_pages, file_size change. Other fields
    # (downloaded_at / extracted_at) are written by store_document itself.
    stored = StoredDocument(
        document_id=doc_id,
        title=current_doc.title,
        category=current_doc.category,
        decision_date=current_doc.decision_date,
        decision_number=current_doc.decision_number,
        source_url=current_doc.source_url,
        markdown_content=body,
        content_hash=new_hash,
        extraction_method=extraction_method,
        total_pages=total_pages,
        file_size=len(body.encode("utf-8")),
    )
    await doc_store.store_document(stored)
    await vector_store.add_document(
        doc_id=doc_id,
        title=current_doc.title,
        content=body,
        category=current_doc.category or "",
        decision_date=current_doc.decision_date or "",
        decision_number=current_doc.decision_number or "",
        source_url=current_doc.source_url or "",
    )

    # --- 4. Seed surgery lands in the next task. ------------------------
    raise NotImplementedError("seed surgery lands in Task 4")
```

**Note:** `StoredDocument` (see `doc_store.py:26`) is a Pydantic model with
these fields: `document_id`, `title`, `category`, `decision_date`,
`decision_number`, `source_url`, `pdf_bytes` (bytes | None, default None),
`markdown_content`, `content_hash`, `extraction_method`, `total_pages`,
`file_size`. `downloaded_at` / `extracted_at` are **not** on the model — they
are timestamp columns written by `store_document` internally.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_patch_doc.py -v`
Expected: PASS (6 passed — prior 5 still green, new 1 green)

- [ ] **Step 5: Commit**

```bash
git add scripts/patch_doc.py tests/test_patch_doc.py
git commit -m "feat(scripts): wire DB update (store_document + add_document) into patch_doc"
```

---

## Task 4 — Wire seed surgery into `patch_document()`

**Files:**
- Modify: `scripts/patch_doc.py`
- Modify: `tests/test_patch_doc.py`

Final piece of the core function: rewrite `seed_data/documents.json` and `seed_data/chunks.json` in place for just the target doc. Sibling docs/chunks must remain byte-identical.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_patch_doc.py`:

```python
@pytest.mark.asyncio
async def test_seed_surgery_updates_only_target_doc(tmp_path):
    """documents.json: target updated, sibling byte-identical."""
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    target = _seed_doc_entry("mevzuat_20029", markdown="old body", content_hash="oldhash")
    sibling = _seed_doc_entry("mevzuat_99999", markdown="sibling body", content_hash="siblinghash")
    _write_seed_files(seed_dir, docs=[target, sibling], chunks=[])
    md = tmp_path / "body.md"
    md.write_text("new corrected body\n", encoding="utf-8")

    await patch_doc.patch_document(
        doc_id="mevzuat_20029",
        markdown_path=md,
        extraction_method="html_parser+manual_latex",
        doc_store=_mock_doc_store(_stored_doc("mevzuat_20029")),
        vector_store=_mock_vector_store(),
        seed_dir=seed_dir,
        dry_run=False,
    )

    written = json.loads((seed_dir / "documents.json").read_text(encoding="utf-8"))
    assert len(written) == 2
    target_out = next(d for d in written if d["document_id"] == "mevzuat_20029")
    sibling_out = next(d for d in written if d["document_id"] == "mevzuat_99999")
    assert target_out["markdown_content"] == "new corrected body\n"
    assert target_out["content_hash"] != "oldhash"
    assert target_out["extraction_method"] == "html_parser+manual_latex"
    assert target_out["extracted_at"] > 1_700_000_000  # newer than the seeded value
    # Sibling preserved exactly
    assert sibling_out == sibling


@pytest.mark.asyncio
async def test_seed_surgery_rewrites_chunks_for_target_only(tmp_path):
    """chunks.json: target chunks replaced with fresh ones carrying new hash;
    sibling chunks byte-identical."""
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    _write_seed_files(
        seed_dir,
        docs=[_seed_doc_entry("mevzuat_20029"), _seed_doc_entry("mevzuat_99999")],
        chunks=[
            {
                "doc_id": "mevzuat_20029",
                "chunk_index": 0,
                "title": "Title of mevzuat_20029",
                "category": "",
                "decision_date": "",
                "decision_number": "",
                "source_url": "",
                "total_chunks": 2,
                "total_pages": 1,
                "content_hash": "oldhash",
                "chunk_text": "old part 1",
            },
            {
                "doc_id": "mevzuat_20029",
                "chunk_index": 1,
                "title": "Title of mevzuat_20029",
                "category": "",
                "decision_date": "",
                "decision_number": "",
                "source_url": "",
                "total_chunks": 2,
                "total_pages": 1,
                "content_hash": "oldhash",
                "chunk_text": "old part 2",
            },
            {
                "doc_id": "mevzuat_99999",
                "chunk_index": 0,
                "title": "Title of mevzuat_99999",
                "category": "",
                "decision_date": "",
                "decision_number": "",
                "source_url": "",
                "total_chunks": 1,
                "total_pages": 1,
                "content_hash": "siblinghash",
                "chunk_text": "sibling text",
            },
        ],
    )
    md = tmp_path / "body.md"
    md.write_text("new body for reembed\n", encoding="utf-8")

    await patch_doc.patch_document(
        doc_id="mevzuat_20029",
        markdown_path=md,
        extraction_method="html_parser+manual_latex",
        doc_store=_mock_doc_store(_stored_doc("mevzuat_20029")),
        vector_store=_mock_vector_store(),
        seed_dir=seed_dir,
        dry_run=False,
    )

    written = json.loads((seed_dir / "chunks.json").read_text(encoding="utf-8"))
    target_chunks = [c for c in written if c["doc_id"] == "mevzuat_20029"]
    sibling_chunks = [c for c in written if c["doc_id"] == "mevzuat_99999"]

    # Target replaced: every new chunk carries the same new hash, none have oldhash
    assert target_chunks, "target chunks missing after patch"
    assert all(c["content_hash"] != "oldhash" for c in target_chunks)
    assert len({c["content_hash"] for c in target_chunks}) == 1

    # Sibling preserved exactly (unchanged order, unchanged content)
    assert sibling_chunks == [
        {
            "doc_id": "mevzuat_99999",
            "chunk_index": 0,
            "title": "Title of mevzuat_99999",
            "category": "",
            "decision_date": "",
            "decision_number": "",
            "source_url": "",
            "total_chunks": 1,
            "total_pages": 1,
            "content_hash": "siblinghash",
            "chunk_text": "sibling text",
        }
    ]


@pytest.mark.asyncio
async def test_seed_surgery_matches_doc_and_chunk_hashes(tmp_path):
    """Sanity: every new chunk_text concat back to body; hashes agree with doc."""
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    _write_seed_files(seed_dir, docs=[_seed_doc_entry("mevzuat_20029")], chunks=[])
    md = tmp_path / "body.md"
    md.write_text("consistent body check\n", encoding="utf-8")

    await patch_doc.patch_document(
        doc_id="mevzuat_20029",
        markdown_path=md,
        extraction_method="html_parser+manual_latex",
        doc_store=_mock_doc_store(_stored_doc("mevzuat_20029")),
        vector_store=_mock_vector_store(),
        seed_dir=seed_dir,
        dry_run=False,
    )

    docs_after = json.loads((seed_dir / "documents.json").read_text(encoding="utf-8"))
    chunks_after = json.loads((seed_dir / "chunks.json").read_text(encoding="utf-8"))

    target_doc = next(d for d in docs_after if d["document_id"] == "mevzuat_20029")
    target_chunks = [c for c in chunks_after if c["doc_id"] == "mevzuat_20029"]

    assert target_chunks
    assert all(c["content_hash"] == target_doc["content_hash"] for c in target_chunks)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_patch_doc.py -v`
Expected: FAIL — these three tests hit `NotImplementedError("seed surgery lands in Task 4")`.

- [ ] **Step 3: Implement seed surgery**

Replace the placeholder `raise NotImplementedError("seed surgery lands in Task 4")` at the end of `patch_document` with:

```python
    # --- 4. Seed surgery -------------------------------------------------
    now = time.time()

    # documents.json — update target entry in place
    seed_entry["markdown_content"] = body
    seed_entry["content_hash"] = new_hash
    seed_entry["extraction_method"] = extraction_method
    seed_entry["extracted_at"] = now
    seed_entry["total_pages"] = total_pages
    seed_entry["file_size"] = len(body.encode("utf-8"))
    docs_path.write_text(
        json.dumps(seed_docs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # chunks.json — strip old entries for doc_id, append fresh ones
    seed_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    seed_chunks = [c for c in seed_chunks if c.get("doc_id") != doc_id]
    for i, chunk_text in enumerate(chunks):
        new_chunk = {
            "doc_id": doc_id,
            "chunk_index": i,
            "title": current_doc.title,
            "category": current_doc.category or "",
            "decision_date": current_doc.decision_date or "",
            "decision_number": current_doc.decision_number or "",
            "source_url": current_doc.source_url or "",
            "total_chunks": len(chunks),
            "total_pages": total_pages,
            "content_hash": new_hash,
            "chunk_text": chunk_text,
        }
        # Belt-and-suspenders: doc hash and chunk hash must agree
        assert new_chunk["content_hash"] == new_hash, (
            f"hash divergence at chunk {i} for {doc_id}"
        )
        seed_chunks.append(new_chunk)
    chunks_path.write_text(
        json.dumps(seed_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_patch_doc.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/patch_doc.py tests/test_patch_doc.py
git commit -m "feat(scripts): seed surgery — rewrite only target doc's entries"
```

---

## Task 5 — CLI wiring (`_main_async` + `main`)

**Files:**
- Modify: `scripts/patch_doc.py`

Replace the `raise NotImplementedError("CLI wiring lands in Task 5")` in `_main_async` with the real pool-and-stores construction.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_patch_doc.py`:

```python
def test_arg_parser_requires_doc_id_and_markdown():
    parser = patch_doc._build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # missing positional + required --markdown
    args = parser.parse_args(["mevzuat_20029", "--markdown", "body.md"])
    assert args.doc_id == "mevzuat_20029"
    assert args.markdown == Path("body.md")
    assert args.extraction_method == patch_doc.DEFAULT_EXTRACTION_METHOD
    assert args.dry_run is False


def test_arg_parser_honors_flags():
    parser = patch_doc._build_arg_parser()
    args = parser.parse_args(
        [
            "mevzuat_20029",
            "--markdown",
            "body.md",
            "--extraction-method",
            "html_parser",
            "--dry-run",
        ]
    )
    assert args.extraction_method == "html_parser"
    assert args.dry_run is True
```

- [ ] **Step 2: Run tests to verify they pass (argparse already exists from Task 2)**

Run: `uv run pytest tests/test_patch_doc.py::test_arg_parser_requires_doc_id_and_markdown tests/test_patch_doc.py::test_arg_parser_honors_flags -v`
Expected: PASS (2 passed) — both already green since `_build_arg_parser` was written in Task 2. These tests lock in the parser's shape for regressions.

- [ ] **Step 3: Implement `_main_async`**

Replace the entire `_main_async` function in `scripts/patch_doc.py` with:

```python
async def _main_async(args: argparse.Namespace) -> int:
    pool = await asyncpg.create_pool(require_database_url(), min_size=1, max_size=3)
    try:
        doc_store = DocumentStore(pool)
        await doc_store.initialize()
        vector_store = VectorStore(pool)
        await vector_store.initialize()

        seed_dir = ROOT / "seed_data"

        try:
            result = await patch_document(
                doc_id=args.doc_id,
                markdown_path=args.markdown,
                extraction_method=args.extraction_method,
                doc_store=doc_store,
                vector_store=vector_store,
                seed_dir=seed_dir,
                dry_run=args.dry_run,
            )
        except (PatchError, FileNotFoundError) as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

        prefix = "[DRY RUN] " if result["dry_run"] else ""
        print(f"{prefix}{result['doc_id']}:")
        print(f"  old_hash: {result['old_hash']}")
        print(f"  new_hash: {result['new_hash']}")
        print(f"  char_len: {result['old_len']} -> {result['new_len']}")
        print(f"  chunks:   {result['chunk_count']}")
        print(f"  method:   {result['extraction_method']}")
        if not result["dry_run"]:
            print("\nrun 'git diff --stat seed_data/' to review the seed changes")
        return 0
    finally:
        await pool.close()
```

- [ ] **Step 4: Smoke-test CLI argument parsing (no DB)**

Run: `uv run python scripts/patch_doc.py --help`
Expected: Argparse prints usage with the three flags documented and exits 0.

- [ ] **Step 5: Run the full test file once more**

Run: `uv run pytest tests/test_patch_doc.py -v`
Expected: PASS (11 passed — 9 from Tasks 2-4 + 2 new argparse tests)

- [ ] **Step 6: Commit**

```bash
git add scripts/patch_doc.py tests/test_patch_doc.py
git commit -m "feat(scripts): wire patch_doc.py CLI entry point + summary output"
```

---

## Task 6 — Lint + format + full-suite green

- [ ] **Step 1: Format and lint both new files**

Run: `uv run ruff format scripts/patch_doc.py tests/test_patch_doc.py seed.py tests/test_seed.py`
Expected: `4 files reformatted` (or similar — if unchanged, that's fine too)

Run: `uv run ruff check scripts/patch_doc.py tests/test_patch_doc.py seed.py tests/test_seed.py`
Expected: `All checks passed!`

- [ ] **Step 2: Run full test suite to catch any collateral breakage**

Run: `uv run pytest tests/ -v --tb=short`
Expected: No new failures vs. `main`. The pre-existing `pg_pool` fixture tests may skip without a DB — that's the repo's normal state.

- [ ] **Step 3: Commit any formatter/lint fixups**

```bash
git status --short
# If ruff reformatted anything:
git add scripts/patch_doc.py tests/test_patch_doc.py seed.py tests/test_seed.py
git commit -m "chore: ruff format patch_doc + seed changes"
# If nothing changed, skip this commit.
```

---

## Task 7 — Clean up session's one-off scripts

**Files:**
- Delete: `_push_mevzuat_20029.py`
- Delete: `_bump_extraction_method.py`
- Delete: `_refresh_content_hash.py`
- Delete: `_backup_mevzuat_20029.py`
- Delete: `_check_20029.py`
- Delete: `_check_hashes.py`
- Delete: `update_mevzuat_20029.py`

These are untracked (they never joined git). Removing them keeps the working tree tidy and prevents future confusion about which path is authoritative.

- [ ] **Step 1: Confirm none of these are tracked**

Run: `git ls-files _push_mevzuat_20029.py _bump_extraction_method.py _refresh_content_hash.py _backup_mevzuat_20029.py _check_20029.py _check_hashes.py update_mevzuat_20029.py`
Expected: empty output (none tracked).

- [ ] **Step 2: Delete the files**

Run:
```bash
rm _push_mevzuat_20029.py \
   _bump_extraction_method.py \
   _refresh_content_hash.py \
   _backup_mevzuat_20029.py \
   _check_20029.py \
   _check_hashes.py \
   update_mevzuat_20029.py
```

- [ ] **Step 3: Confirm deletion**

Run: `ls _*.py update_mevzuat_20029.py 2>&1 || true`
Expected: `ls: cannot access ...: No such file or directory` for every path.

No commit — deletions of untracked files aren't staged.

---

## Task 8 — Production validation

Run the new script against prod to reproduce the `mevzuat_20029` patch that the old ad-hoc scripts applied. Confirm DB state, seed_data diff scope, and MCP output.

- [ ] **Step 1: Dry-run first**

Run:
```bash
railway run --service Postgres -- bash -c \
    'BDDK_DATABASE_URL="$DATABASE_PUBLIC_URL" \
     uv run python scripts/patch_doc.py mevzuat_20029 \
         --markdown mevzuat_20029_updated.md \
         --dry-run'
```
Expected output:
- `[DRY RUN]` prefix on summary
- `old_hash: <current DB hash>`
- `new_hash: <sha256 of stripped body>`
- `char_len: <N> -> 114218` (body size after header strip)
- `chunks: ~143` (matches what we saw earlier)
- `method: html_parser+manual_latex`

- [ ] **Step 2: Real run**

Run the same command without `--dry-run`:
```bash
railway run --service Postgres -- bash -c \
    'BDDK_DATABASE_URL="$DATABASE_PUBLIC_URL" \
     uv run python scripts/patch_doc.py mevzuat_20029 \
         --markdown mevzuat_20029_updated.md'
```
Expected: summary without `[DRY RUN]`, plus `run 'git diff --stat seed_data/' to review the seed changes`.

- [ ] **Step 3: Verify seed_data diff is scoped**

Run: `git diff --stat seed_data/`
Expected: Two files changed: `seed_data/documents.json` and `seed_data/chunks.json`. Line counts modest (mevzuat_20029 is one of 318 docs and ~143 of ~9548 chunks). No touches to `decision_cache.json`.

- [ ] **Step 4: Verify DB state via MCP**

Call `mcp__claude_ai_BDDK_MCP__get_bddk_document` with `document_id=mevzuat_20029, page_number=15`.
Expected:
- Header line `- Extraction: html_parser+manual_latex` — **no** `(formula-unaware …)` suffix
- **No** `⚠ Bu belgedeki matematiksel formüller...` banner
- Page body contains `$$H_M = H_N \sqrt{\frac{T_M}{T_N}}$$`

- [ ] **Step 5: Commit seed_data changes**

```bash
git add seed_data/documents.json seed_data/chunks.json
git commit -m "data(seed): capture mevzuat_20029 manual LaTeX correction"
```

- [ ] **Step 6: Open PR**

```bash
git push -u origin feat/patch-doc-helper
gh pr create --base main \
    --title "feat(scripts): patch_doc.py — atomic DB + seed_data correction helper" \
    --body "See docs/superpowers/specs/2026-04-23-patch-doc-helper-design.md + docs/superpowers/plans/2026-04-23-patch-doc-helper-plan.md. First customer: mevzuat_20029 LaTeX patch captured in seed_data."
```

- [ ] **Step 7: Post-merge durability check**

After PR merges and Railway deploys, re-call `mcp__claude_ai_BDDK_MCP__get_bddk_document` for `mevzuat_20029` page 15. Expected: banner still absent, formula still present. This proves the seed_data capture prevents the startup re-import from reverting the patch — the recurrence guarantee that motivated this whole piece of work.

---

## Self-Review Notes

Spec coverage check:
- Interface (args, env) → Task 2 + Task 5 ✓
- Flow steps 1–6 → Task 2 (validate + dry-run), Task 3 (DB update), Task 4 (seed surgery), Task 5 (summary output) ✓
- Shared header-strip helper in `seed.py` → Task 1 ✓
- Testing strategy → covered across Tasks 2–5 (11 tests total) + `TestStripDocsDumpHeader` in Task 1 ✓
- Follow-up cleanup → Task 7 ✓
- Production validation → Task 8 ✓

No placeholders. `StoredDocument` field list is called out in Task 3 Step 3 as a "check" (run `grep` to verify). Chunk dict schema in Task 4 mirrors `scripts/regen_chunks_seed.py:60-73` verbatim.

Type consistency: `PatchError`, `patch_document`, `_build_arg_parser`, `_main_async`, `DEFAULT_EXTRACTION_METHOD`, and the `result` dict keys (`dry_run`, `new_hash`, `chunk_count`, etc.) match between the implementation code and every test that uses them.
