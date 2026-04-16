"""Batch resync corrupted mevzuat_* docs in seed_data/documents.json.

Write-then-swap: writes to documents.json.new then atomically renames on success.
Updates markdown_content, content_hash, extracted_at, file_size, extraction_method
for each doc that successfully re-extracts with ufffd=0.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED_PATH = ROOT.parent.parent.parent / "seed_data" / "documents.json"  # /home/cagatay/bddk-mcp/seed_data/documents.json

from doc_store import StoredDocument  # noqa: E402
from doc_sync import DocumentSyncer  # noqa: E402


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MemStore:
    """In-memory store satisfying DocumentSyncer's interface."""

    def __init__(self) -> None:
        self.docs: dict[str, StoredDocument] = {}
        self.failures: dict[str, dict] = {}

    async def has_document(self, doc_id: str) -> bool:
        return doc_id in self.docs

    async def get_document(self, doc_id: str) -> StoredDocument | None:
        return self.docs.get(doc_id)

    async def store_document(self, doc: StoredDocument) -> None:
        self.docs[doc.document_id] = doc

    async def delete_document(self, doc_id: str) -> None:
        self.docs.pop(doc_id, None)

    async def record_sync_failure(self, doc_id, error_msg, category, source_url, retryable) -> None:
        self.failures[doc_id] = {"error": error_msg, "category": category, "retryable": retryable}

    async def clear_sync_failure(self, doc_id: str) -> None:
        self.failures.pop(doc_id, None)

    async def import_from_cache(self, items) -> None:
        pass


async def main() -> int:
    raw = SEED_PATH.read_text(encoding="utf-8")
    data = json.loads(raw)
    docs = data["documents"] if isinstance(data, dict) else data

    corrupted = [d for d in docs if d.get("markdown_content", "").count("\ufffd") > 0]
    print(f"Found {len(corrupted)} corrupted docs in {SEED_PATH}")
    if not corrupted:
        print("Nothing to do.")
        return 0

    # id -> doc-dict for in-place updates
    by_id = {d["document_id"]: d for d in docs}

    store = MemStore()
    ok, failed, still_corrupt = [], [], []

    async with DocumentSyncer(store, prefer_nougat=False) as syncer:
        sem = asyncio.Semaphore(4)  # gentle — mevzuat.gov.tr is slow

        async def resync_one(meta: dict) -> tuple[str, bool, str]:
            did = meta["document_id"]
            async with sem:
                result = await syncer.sync_document(
                    doc_id=did,
                    title=meta.get("title", ""),
                    category=meta.get("category", ""),
                    source_url=meta.get("source_url", ""),
                    decision_date=meta.get("decision_date", ""),
                    decision_number=meta.get("decision_number", ""),
                    force=True,
                )
                if not result.success:
                    return did, False, result.error or "unknown"
                stored = store.docs.get(did)
                if not stored:
                    return did, False, "store miss after sync"
                u = stored.markdown_content.count("\ufffd")
                if u > 0:
                    return did, False, f"still has {u} ufffd"
                return did, True, result.method or ""

        tasks = [resync_one(m) for m in corrupted]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, BaseException):
            failed.append(("?", str(r)))
            continue
        did, success, info = r
        if success:
            ok.append((did, info))
        elif "ufffd" in info:
            still_corrupt.append((did, info))
        else:
            failed.append((did, info))

    print(f"\nResults: ok={len(ok)}  failed={len(failed)}  still_corrupt={len(still_corrupt)}")
    if failed:
        print("\nFailed:")
        for did, info in failed[:20]:
            print(f"  {did}: {info}")
    if still_corrupt:
        print("\nStill corrupted after resync:")
        for did, info in still_corrupt[:20]:
            print(f"  {did}: {info}")

    if not ok:
        print("\nNo successful resyncs — skipping seed write.")
        return 1

    # Merge successes back into seed
    now = time.time()
    total_chars_before = 0
    total_chars_after = 0
    for did, _method in ok:
        stored = store.docs[did]
        meta = by_id[did]
        before = meta.get("markdown_content", "")
        after = stored.markdown_content
        total_chars_before += len(before)
        total_chars_after += len(after)
        meta["markdown_content"] = after
        meta["content_hash"] = _hash(after)
        meta["extracted_at"] = now
        meta["downloaded_at"] = now
        meta["extraction_method"] = stored.extraction_method
        if stored.file_size:
            meta["file_size"] = stored.file_size
        # total_pages stays as-is

    print(f"\nTotal markdown chars: before={total_chars_before}  after={total_chars_after}")

    # Write-then-swap
    new_path = SEED_PATH.with_suffix(".json.new")
    new_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    new_path.replace(SEED_PATH)
    print(f"\nWrote {SEED_PATH}  ({new_path.stat().st_size if new_path.exists() else 'renamed'} bytes)")

    return 0 if not still_corrupt and not failed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
