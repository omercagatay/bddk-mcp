"""Regenerate seed_data/chunks.json from current seed_data/documents.json.

Uses vector_store._chunk_document — same chunker production uses, so post-import
chunks are bit-identical to what add_document() would produce. Embeddings are
NOT computed (seed.py imports text only; embeddings regen on first search).

Required after a corrective edit to documents.json (e.g. encoding resync) since
chunks are derived from markdown_content and otherwise stay stale.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bddk_mcp.core.config import PAGE_SIZE  # noqa: E402
from bddk_mcp.store.vector_store import _chunk_document, _load_embedding_tokenizer  # noqa: E402

DOCS_PATH = ROOT / "seed_data" / "documents.json"
CHUNKS_PATH = ROOT / "seed_data" / "chunks.json"


def build_chunk_records(
    document: dict,
    *,
    tokenizer=None,
    target_tokens: int | None = None,
    token_overlap: int | None = None,
) -> list[dict]:
    content = document.get("markdown_content", "")
    if not content.strip():
        return []

    chunk_kwargs = {"tokenizer": tokenizer}
    if target_tokens is not None:
        chunk_kwargs["target_tokens"] = target_tokens
    if token_overlap is not None:
        chunk_kwargs["token_overlap"] = token_overlap
    chunks = _chunk_document(document["document_id"], content, **chunk_kwargs)
    if not chunks:
        return []

    total_pages = max(1, math.ceil(len(content) / PAGE_SIZE))
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    title = document.get("title", "")
    category = document.get("category", "")
    decision_date = document.get("decision_date", "")
    decision_number = document.get("decision_number", "")
    source_url = document.get("source_url", "")

    return [
        {
            "doc_id": document["document_id"],
            "chunk_index": i,
            "title": title,
            "category": category,
            "decision_date": decision_date,
            "decision_number": decision_number,
            "source_url": source_url,
            "total_chunks": len(chunks),
            "total_pages": total_pages,
            "content_hash": content_hash,
            "chunk_start_char": chunk.start_char,
            "chunk_end_char": chunk.end_char,
            "section_type": chunk.section_type,
            "section_ref": chunk.section_ref,
            "section_start_char": chunk.section_start_char,
            "section_end_char": chunk.section_end_char,
            "section_content_hash": chunk.section_content_hash,
            "chunk_text": chunk.chunk_text,
        }
        for i, chunk in enumerate(chunks)
    ]


def main() -> int:
    docs = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(docs)} docs from {DOCS_PATH}")

    chunks_out: list[dict] = []
    skipped_empty = 0
    total_ufffd_in = 0
    total_ufffd_out = 0
    tokenizer = _load_embedding_tokenizer()

    for d in docs:
        content = d.get("markdown_content", "")
        total_ufffd_in += content.count("\ufffd")
        chunk_records = build_chunk_records(d, tokenizer=tokenizer)
        if not chunk_records:
            skipped_empty += 1
            continue

        for record in chunk_records:
            total_ufffd_out += record["chunk_text"].count("\ufffd")
        chunks_out.extend(chunk_records)

    print(f"Generated {len(chunks_out)} chunks (skipped {skipped_empty} empty docs)")
    print(f"U+FFFD: input={total_ufffd_in} chars in docs, output={total_ufffd_out} chars in chunks")

    new_path = CHUNKS_PATH.with_suffix(".json.new")
    new_path.write_text(json.dumps(chunks_out, ensure_ascii=False, indent=2), encoding="utf-8")
    new_path.replace(CHUNKS_PATH)
    print(f"Wrote {CHUNKS_PATH} ({CHUNKS_PATH.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
