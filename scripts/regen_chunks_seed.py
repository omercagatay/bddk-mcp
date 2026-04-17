"""Regenerate seed_data/chunks.json from current seed_data/documents.json.

Uses vector_store._chunk_text — same chunker production uses, so post-import
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

from config import PAGE_SIZE  # noqa: E402
from vector_store import _chunk_text  # noqa: E402

DOCS_PATH = ROOT / "seed_data" / "documents.json"
CHUNKS_PATH = ROOT / "seed_data" / "chunks.json"


def main() -> int:
    docs = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(docs)} docs from {DOCS_PATH}")

    chunks_out: list[dict] = []
    skipped_empty = 0
    total_ufffd_in = 0
    total_ufffd_out = 0

    for d in docs:
        content = d.get("markdown_content", "")
        total_ufffd_in += content.count("\ufffd")
        if not content.strip():
            skipped_empty += 1
            continue

        chunks = _chunk_text(content)
        if not chunks:
            skipped_empty += 1
            continue

        total_pages = max(1, math.ceil(len(content) / PAGE_SIZE))
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        title = d.get("title", "")
        category = d.get("category", "")
        decision_date = d.get("decision_date", "")
        decision_number = d.get("decision_number", "")
        source_url = d.get("source_url", "")

        for i, chunk in enumerate(chunks):
            total_ufffd_out += chunk.count("\ufffd")
            chunks_out.append(
                {
                    "doc_id": d["document_id"],
                    "chunk_index": i,
                    "title": title,
                    "category": category,
                    "decision_date": decision_date,
                    "decision_number": decision_number,
                    "source_url": source_url,
                    "total_chunks": len(chunks),
                    "total_pages": total_pages,
                    "content_hash": content_hash,
                    "chunk_text": chunk,
                }
            )

    print(f"Generated {len(chunks_out)} chunks (skipped {skipped_empty} empty docs)")
    print(f"U+FFFD: input={total_ufffd_in} chars in docs, output={total_ufffd_out} chars in chunks")

    new_path = CHUNKS_PATH.with_suffix(".json.new")
    new_path.write_text(json.dumps(chunks_out, ensure_ascii=False, indent=2), encoding="utf-8")
    new_path.replace(CHUNKS_PATH)
    print(f"Wrote {CHUNKS_PATH} ({CHUNKS_PATH.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
