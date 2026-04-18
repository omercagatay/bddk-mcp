"""Pilot resync: fetch mevzuat_19498 via full DocumentSyncer path, verify clean output."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from doc_store import StoredDocument  # noqa: E402
from doc_sync import DocumentSyncer  # noqa: E402


class MemStore:
    """In-memory store that satisfies DocumentSyncer's interface."""

    def __init__(self) -> None:
        self.docs: dict[str, StoredDocument] = {}
        self.sync_failures: dict[str, dict] = {}

    async def has_document(self, doc_id: str) -> bool:
        return doc_id in self.docs

    async def get_document(self, doc_id: str) -> StoredDocument | None:
        return self.docs.get(doc_id)

    async def get_pdf_bytes(self, doc_id: str) -> bytes | None:
        return None

    async def store_document(self, doc: StoredDocument) -> None:
        self.docs[doc.document_id] = doc

    async def delete_document(self, doc_id: str) -> None:
        self.docs.pop(doc_id, None)

    async def record_sync_failure(self, doc_id, error_msg, category, source_url, retryable) -> None:
        self.sync_failures[doc_id] = {
            "error": error_msg,
            "category": category,
            "source_url": source_url,
            "retryable": retryable,
        }

    async def clear_sync_failure(self, doc_id: str) -> None:
        self.sync_failures.pop(doc_id, None)

    async def import_from_cache(self, items) -> None:  # unused in single-doc path
        pass


async def main() -> int:
    store = MemStore()

    target = {
        "document_id": "mevzuat_19498",
        "title": "Bankaların Likidite Karşılama Oranı Hesaplamasına İlişkin Yönetmelik",
        "category": "Yönetmelik",
        "source_url": (
            "http://www.mevzuat.gov.tr/Metin.Aspx?MevzuatKod=7.5.19498&MevzuatIliski=0&sourceXmlSearch=likidite"
        ),
    }

    async with DocumentSyncer(store) as syncer:
        result = await syncer.sync_document(
            doc_id=target["document_id"],
            title=target["title"],
            category=target["category"],
            source_url=target["source_url"],
            force=True,
        )

    print(f"success={result.success} method={result.method} error={result.error}")

    if not result.success:
        return 1

    doc = store.docs[target["document_id"]]
    content = doc.markdown_content
    ufffd = content.count("\ufffd")

    print(f"extraction_method={doc.extraction_method}")
    print(f"content_length={len(content)}")
    print(f"ufffd_count={ufffd}")
    print(f"file_size={doc.file_size}")
    print("first_300_chars:")
    print(repr(content[:300]))

    if ufffd > 0:
        print("\nFAIL: ufffd still present after resync")
        return 2

    # Spot-check: must contain clean Turkish words
    expected_words = ["LİKİDİTE", "YÖNETMELİK", "BİRİNCİ"]
    missing = [w for w in expected_words if w not in content.upper()]
    if missing:
        print(f"\nFAIL: expected Turkish words missing: {missing}")
        return 3

    print("\nPASS: clean output, Turkish characters intact")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
