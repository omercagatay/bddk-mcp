"""Fetch the first page of selected documents through the MCP SDK."""

import asyncio
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

BASE = "http://localhost:8000/mcp"


async def main():
    async with streamable_http_client(BASE) as (read, write, _):
        async with ClientSession(read, write) as session:
            print("INIT:", (await session.initialize()).model_dump_json()[:200])
            for doc in sys.argv[1:] or ["mevzuat_42626"]:
                result = await session.call_tool("get_bddk_document", {"document_id": doc, "page_number": 1})
                if result.isError:
                    raise RuntimeError(f"Document fetch failed: {doc}")
                print(f"\n=== doc {doc} ===")
                print(result.model_dump_json()[:1500])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
