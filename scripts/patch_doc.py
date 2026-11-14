"""Operator helper: source-first, crash-recoverable document correction.

Intended for manual corrections (e.g. hand-inserted LaTeX where OCR couldn't
recover image-based formulas). For one doc_id, this script:

  1. validates inputs (markdown exists, doc_id present in both corpus and seed),
  2. strips any docs_dump-style header from the markdown,
  3. computes the new content hash and previews the section-aware chunk
     count via the same chunker used by vector_store.add_document and seed.py,
  4. on --dry-run, stops here and prints the planned delta,
  5. atomically replaces the seed documents.json (canonical seed format) when a seed directory
     applies: the repository seed for the default corpus, or the one given by --seed-dir; a
     non-default --corpus without --seed-dir leaves every seed file untouched,
  6. updates the canonical corpus row, then republishes its derived vector index.

Filesystem and corpus commits cannot form one atomic transaction.  The
ordering is deliberate: documents.json is the durable reviewed source; the
corpus publication is fail-closed while its canonical row and chunks differ;
and a crash can be recovered by rerunning this idempotent command.  Chunks and
embeddings are regenerated from documents.json by `bddk-mcp build`.

Run:
    uv run python scripts/patch_doc.py <doc_id> --markdown <path> \\
        [--corpus <path>] [--seed-dir <dir>] [--extraction-method <tag>] [--dry-run]

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

from patch_md import validate_latex  # noqa: E402

from bddk_mcp.core.config import CORPUS_PATH, PAGE_SIZE  # noqa: E402
from bddk_mcp.ingest.seed import _strip_docs_dump_header, _write_canonical_json  # noqa: E402
from bddk_mcp.quality.markdown_quality import _STORAGE_PREPARE_MARKERS, prepare_markdown_for_storage  # noqa: E402
from bddk_mcp.store.corpus_db import CorpusDB  # noqa: E402
from bddk_mcp.store.doc_store import DocumentStore, StoredDocument  # noqa: E402
from bddk_mcp.store.vector_store import VectorStore, _chunk_document  # noqa: E402

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
    seed_dir: Path | None,
    dry_run: bool = False,
    skip_latex_check: bool = False,
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

    docs_path = seed_dir / "documents.json" if seed_dir is not None else None
    seed_docs: list[dict[str, Any]] = []
    seed_entry: dict[str, Any] | None = None
    if docs_path is not None:
        seed_docs = json.loads(docs_path.read_text(encoding="utf-8"))
        seed_entry = next((d for d in seed_docs if d.get("document_id") == doc_id), None)
        if seed_entry is None:
            raise PatchError(f"{doc_id} not found in {docs_path}")

    # --- 2. Strip header + validate LaTeX + compute hash + regenerate chunks
    body = _strip_docs_dump_header(raw)
    if not body.strip():
        raise PatchError(f"{markdown_path} has no content after stripping docs_dump header")
    if prepare_markdown_for_storage(body) != body:
        markers = ", ".join(repr(m) for m in _STORAGE_PREPARE_MARKERS if m in body)
        raise PatchError(
            f"{markdown_path} contains storage markers that the corpus would rewrite ({markers}). "
            "Remove them (non-breaking spaces, zero-width characters, form feeds, runs of '****', '----------' "
            "or '__________', 'cid:' and 'data:image' references) so the seed, the stored row and the chunks "
            "carry the same text."
        )
    if not skip_latex_check:
        latex_issues = validate_latex(body)
        if latex_issues:
            formatted = "\n  - ".join(latex_issues)
            raise PatchError(
                f"LaTeX validation failed for {markdown_path}:\n  - {formatted}\n"
                "Fix the markdown or pass --skip-latex-check to proceed anyway."
            )
    new_hash = _content_hash(body)
    tokenizer = vector_store._chunk_tokenizer()
    chunks = _chunk_document(doc_id, body, tokenizer=tokenizer)
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
        "seed_written": docs_path is not None and not dry_run,
    }

    if dry_run:
        return result

    # --- 3. Reviewed-source update ---------------------------------------
    now = time.time()

    # documents.json — update the target entry in place, written in the canonical seed format
    if docs_path is not None and seed_entry is not None:
        seed_entry["markdown_content"] = body
        seed_entry["content_hash"] = new_hash
        seed_entry["extraction_method"] = extraction_method
        seed_entry["extracted_at"] = now
        seed_entry["total_pages"] = total_pages
        seed_entry["file_size"] = len(body.encode("utf-8"))
        docs_tmp = docs_path.with_suffix(".json.new")
        _write_canonical_json(docs_tmp, seed_docs)
        docs_tmp.replace(docs_path)

    # --- 4. Fail-closed corpus publication --------------------------------
    # These two idempotent transactions cannot be combined with filesystem
    # replacement.  If the process stops after the canonical row changes,
    # retrieval publication is hidden by its content-hash contract until a
    # rerun completes add_document.
    # A text repair is not a download: keep the stored download date and stamp the patch time
    # (the same value the seed entry got) as the extraction time.
    current_body = await doc_store.get_current_body(doc_id)
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
        downloaded_at=current_body["downloaded_at"] if current_body is not None else None,
        extracted_at=now,
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

    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Patch reviewed seed source, then fail-closed corpus publication.")
    p.add_argument("doc_id", help="Exact document_id (e.g. mevzuat_20029)")
    p.add_argument("--markdown", required=True, type=Path, help="Path to corrected markdown file")
    p.add_argument(
        "--corpus", default=CORPUS_PATH, help="Corpus file path (default: BDDK_CORPUS_PATH or corpus/bddk.sqlite)"
    )
    p.add_argument(
        "--seed-dir",
        type=Path,
        default=None,
        help="Seed directory whose documents.json is updated (default: the repository seed_data/ for the default "
        "corpus; with another --corpus and no --seed-dir no seed file is written)",
    )
    p.add_argument(
        "--extraction-method",
        default=DEFAULT_EXTRACTION_METHOD,
        help=f"Value for documents.extraction_method (default: {DEFAULT_EXTRACTION_METHOD!r})",
    )
    p.add_argument("--dry-run", action="store_true", help="Validate and print plan without writing")
    p.add_argument(
        "--skip-latex-check",
        action="store_true",
        help="Bypass the LaTeX pre-flight validator (use only when the body legitimately contains unbalanced $-like tokens)",
    )
    return p


def _resolve_seed_dir(seed_dir: Path | None, corpus: str | Path) -> Path | None:
    """The seed directory to update, or None unless --seed-dir is given or the corpus is the repository's own
    corpus/bddk.sqlite (a BDDK_CORPUS_PATH pointing elsewhere never selects the repository seed)."""
    if seed_dir is not None:
        return seed_dir
    if Path(corpus).resolve() == (ROOT / "corpus" / "bddk.sqlite").resolve():
        return ROOT / "seed_data"
    return None


async def _main_async(args: argparse.Namespace) -> int:
    db = CorpusDB.open(args.corpus, create=False)
    try:
        doc_store = DocumentStore(db)
        vector_store = VectorStore(db)

        seed_dir = _resolve_seed_dir(args.seed_dir, args.corpus)

        try:
            result = await patch_document(
                doc_id=args.doc_id,
                markdown_path=args.markdown,
                extraction_method=args.extraction_method,
                doc_store=doc_store,
                vector_store=vector_store,
                seed_dir=seed_dir,
                dry_run=args.dry_run,
                skip_latex_check=args.skip_latex_check,
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
        if seed_dir is None:
            print("  seed:     not written (not the repository corpus and no --seed-dir; repository seed untouched)")
        else:
            print(f"  seed:     {seed_dir / 'documents.json'}{' (dry run, not written)' if result['dry_run'] else ''}")
        if seed_dir is not None and not result["dry_run"]:
            print(f"\nrun 'git diff --stat {seed_dir}' to review the seed changes")
        return 0
    finally:
        db.close()


def main() -> int:
    args = _build_arg_parser().parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    sys.exit(main())
