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
    if not chunks_path.exists():
        raise PatchError(f"seed_data/chunks.json not found at {chunks_path}")

    # --- 2. Strip header + compute hash + regenerate chunks --------------
    body = _strip_docs_dump_header(raw)
    if not body.strip():
        raise PatchError(f"{markdown_path} has no content after stripping docs_dump header")
    new_hash = _content_hash(body)
    chunks = _chunk_text(body)
    if not chunks:
        raise PatchError(f"chunk regeneration produced no chunks for {doc_id}")
    total_pages = max(1, math.ceil(len(body) / PAGE_SIZE))
    result = {
        "doc_id": doc_id,
        "old_hash": current_doc.content_hash,
        "new_hash": new_hash,
        "old_len": len(current_doc.markdown_content),
        "new_len": len(body),
        "chunk_count": len(chunks),
        "extraction_method": extraction_method,
        "dry_run": dry_run,
    }

    if dry_run:
        return result

    # --- 3. DB update ----------------------------------------------------
    # StoredDocument preserves existing metadata; only body, extraction_method,
    # total_pages, and file_size change. content_hash is recomputed from
    # markdown_content by store_document itself, as are downloaded_at /
    # extracted_at. Both DB writes are idempotent on re-run — store_document
    # via ON CONFLICT UPDATE, add_document via DELETE-then-INSERT — so a
    # crash between them is recoverable by re-running the script.
    stored = StoredDocument(
        document_id=doc_id,
        title=current_doc.title,
        category=current_doc.category,
        decision_date=current_doc.decision_date,
        decision_number=current_doc.decision_number,
        source_url=current_doc.source_url,
        markdown_content=body,
        extraction_method=extraction_method,
        total_pages=total_pages,
        file_size=len(body.encode("utf-8")),
    )
    await doc_store.store_document(stored)
    await vector_store.add_document(
        doc_id=doc_id,
        title=current_doc.title,
        content=body,
        category=current_doc.category,
        decision_date=current_doc.decision_date,
        decision_number=current_doc.decision_number,
        source_url=current_doc.source_url,
    )

    # --- 4. Seed surgery -------------------------------------------------
    now = time.time()

    # documents.json — update target entry in place
    seed_entry["markdown_content"] = body
    seed_entry["content_hash"] = new_hash
    seed_entry["extraction_method"] = extraction_method
    seed_entry["extracted_at"] = now
    seed_entry["total_pages"] = total_pages
    seed_entry["file_size"] = len(body.encode("utf-8"))
    docs_tmp = docs_path.with_suffix(".json.new")
    docs_tmp.write_text(
        json.dumps(seed_docs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    docs_tmp.replace(docs_path)

    # chunks.json — strip old entries for doc_id, append fresh ones
    seed_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    seed_chunks = [c for c in seed_chunks if c.get("doc_id") != doc_id]
    for i, chunk_text in enumerate(chunks):
        new_chunk = {
            "doc_id": doc_id,
            "chunk_index": i,
            "title": current_doc.title,
            "category": current_doc.category,
            "decision_date": current_doc.decision_date,
            "decision_number": current_doc.decision_number,
            "source_url": current_doc.source_url,
            "total_chunks": len(chunks),
            "total_pages": total_pages,
            "content_hash": new_hash,
            "chunk_text": chunk_text,
        }
        # Belt-and-suspenders: doc hash and chunk hash must agree
        assert new_chunk["content_hash"] == new_hash, f"hash divergence at chunk {i} for {doc_id}"
        seed_chunks.append(new_chunk)
    chunks_tmp = chunks_path.with_suffix(".json.new")
    chunks_tmp.write_text(
        json.dumps(seed_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    chunks_tmp.replace(chunks_path)

    return result


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


def main() -> int:
    args = _build_arg_parser().parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
