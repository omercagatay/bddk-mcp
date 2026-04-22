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
import time  # noqa: F401  # used in Task 4 seed rewrites (extracted_at stamp)
from pathlib import Path
from typing import Any

# Allow running as a script from the scripts/ directory.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import asyncpg  # noqa: E402,F401  # used in Task 5 CLI wiring

from config import PAGE_SIZE, require_database_url  # noqa: E402,F401  # require_database_url used in Task 5
from doc_store import DocumentStore, StoredDocument  # noqa: E402,F401  # StoredDocument used in Task 3+
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
    chunks_path = seed_dir / "chunks.json"  # noqa: F841  # used in Task 4 seed rewrite
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
    total_pages = max(1, math.ceil(len(body) / PAGE_SIZE))  # noqa: F841  # used in Task 3+ store_document

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
