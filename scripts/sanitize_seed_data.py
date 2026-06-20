"""Apply storage-safe Markdown cleanup to seed_data documents.

This is a mechanical seed maintenance helper. It keeps document metadata in sync
with sanitized markdown, then operators should run scripts/regen_chunks_seed.py
so seed_data/chunks.json is derived from the cleaned text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bddk_mcp.core.config import PAGE_SIZE  # noqa: E402
from bddk_mcp.quality.markdown_quality import sanitize_markdown_for_storage  # noqa: E402

DEFAULT_DOCS_PATH = ROOT / "seed_data" / "documents.json"


def sanitize_seed_documents(path: Path = DEFAULT_DOCS_PATH, *, write: bool = False) -> dict:
    documents = json.loads(path.read_text(encoding="utf-8"))
    changed: list[str] = []
    now = time.time()

    for document in documents:
        old_markdown = document.get("markdown_content") or ""
        new_markdown = sanitize_markdown_for_storage(old_markdown)
        if new_markdown == old_markdown:
            continue

        document["markdown_content"] = new_markdown
        document["content_hash"] = hashlib.sha256(new_markdown.encode("utf-8")).hexdigest() if new_markdown else ""
        document["total_pages"] = max(1, math.ceil(len(new_markdown) / PAGE_SIZE)) if new_markdown else 1
        document["extracted_at"] = now
        changed.append(document.get("document_id") or "")

    if write:
        path.write_text(json.dumps(documents, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {"documents": len(documents), "changed": len(changed), "changed_doc_ids": changed}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sanitize seed_data documents with the storage markdown sanitizer.")
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCS_PATH)
    parser.add_argument("--write", action="store_true", help="Write sanitized documents.json")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = sanitize_seed_documents(args.documents, write=args.write)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        mode = "updated" if args.write else "would update"
        print(f"{mode} {result['changed']} of {result['documents']} seed documents")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
