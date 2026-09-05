"""Validate every page of every seeded doc via MCP + image-marker audit."""

import asyncio
import re
import subprocess
import sys
import time

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

BASE = "http://localhost:8000/mcp"
IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")


async def main():
    # 1. Get (doc_id, total_pages) from chunks table
    q = "SELECT doc_id, MAX(total_pages) AS total_pages FROM document_chunks GROUP BY doc_id ORDER BY doc_id;"
    proc = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "db",
            "psql",
            "-U",
            "bddk_local_admin",
            "-d",
            "bddk",
            "-t",
            "-A",
            "-F",
            "|",
            "-c",
            q,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    doc_pages = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        did, tp = line.split("|", 1)
        doc_pages.append((did, int(tp)))
    if "--first-page-only" in sys.argv:
        doc_pages = [(did, 1) for did, _ in doc_pages]
    total_calls = sum(tp for _, tp in doc_pages)
    print(f"Docs: {len(doc_pages)}   total pages: {total_calls}")

    async with streamable_http_client(BASE) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 3. Per-page validation
            passed = 0
            invalid_page = []  # tool returned an "Invalid page" marker
            empty = []
            airlocked = []
            errored = []
            image_refs_total = 0
            docs_with_images = set()

            start = time.time()
            for index, (doc_id, total_pages) in enumerate(doc_pages):
                for p in range(1, total_pages + 1):
                    try:
                        result = await session.call_tool("get_bddk_document", {"document_id": doc_id, "page_number": p})
                        if result.isError:
                            errored.append((doc_id, p, "MCP tool error"))
                            continue
                        txt = "\n".join(block.text for block in result.content if block.type == "text")
                    except Exception as e:
                        errored.append((doc_id, p, str(e)[:120]))
                        continue
                    if not txt:
                        empty.append((doc_id, p))
                        continue
                    if "airlocked" in txt.lower():
                        airlocked.append((doc_id, p))
                        continue
                    body = txt.split(
                        "Use ONLY the text below. Do not add information not present in this document.\n\n", 1
                    )[-1]
                    if not body.strip():
                        empty.append((doc_id, p))
                        continue
                    if body.lstrip().startswith("Invalid page") or "Invalid page" in body[:200]:
                        invalid_page.append((doc_id, p))
                        continue
                    passed += 1
                    imgs = IMG_RE.findall(body)
                    if imgs:
                        image_refs_total += len(imgs)
                        docs_with_images.add(doc_id)
                if index % 100 == 0:
                    print(f"  doc {doc_id}: {passed} pages OK, elapsed={time.time() - start:.1f}s")

            print(f"\n=== RESULTS ({time.time() - start:.1f}s) ===")
            print(f"pages attempted: {total_calls}")
            print(f"passed:          {passed}")
            print(f"invalid_page:    {len(invalid_page)}  {invalid_page[:10]}")
            print(f"empty:           {len(empty)}  {empty[:10]}")
            print(f"airlocked:       {len(airlocked)}  {airlocked[:10]}")
            print(f"errored:         {len(errored)}  {errored[:5]}")
            print("\n--- image audit ---")
            print(f"docs with markdown image refs: {len(docs_with_images)}")
            print(f"total image refs detected:     {image_refs_total}")
            if docs_with_images:
                print(f"sample docs: {list(sorted(docs_with_images))[:10]}")

            if invalid_page or empty or airlocked or errored:
                raise SystemExit(1)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
