"""One-time script to populate cache and sync all BDDK documents."""

import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


async def main():
    from client import BddkApiClient
    from doc_store import DocumentStore
    from doc_sync import DocumentSyncer

    # Step 1: Initialize store
    store = DocumentStore()
    await store.initialize()

    # Step 2: Populate cache by scraping BDDK website
    client = BddkApiClient(doc_store=store)
    print("Fetching document metadata from BDDK website...")
    await client.ensure_cache()
    print(f"Cache populated: {len(client._cache)} documents found")

    # Step 3: Sync all documents
    print("Starting document sync...")
    items = [d.model_dump() for d in client._cache]
    async with DocumentSyncer(store, prefer_nougat=False) as syncer:
        report = await syncer.sync_all(items, concurrency=5, force=False)

    print("\nSync Report:")
    print(f"  Total:      {report.total}")
    print(f"  Downloaded: {report.downloaded}")
    print(f"  Skipped:    {report.skipped}")
    print(f"  Failed:     {report.failed}")
    print(f"  Time:       {report.elapsed_seconds}s")

    if report.errors:
        print("\nFirst 10 errors:")
        for e in report.errors[:10]:
            print(f"  [{e.document_id}] {e.error}")

    st = await store.stats()
    print(f"\nStore now has {st.total_documents} documents ({st.total_size_mb} MB)")

    await client.close()
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
