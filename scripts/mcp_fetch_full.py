"""Print selected document pages through the MCP SDK."""

import asyncio
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

BASE = "http://localhost:8000/mcp"


async def main():
    doc_id = sys.argv[1] if len(sys.argv) > 1 else "1040"
    pages = [int(page) for page in sys.argv[2:]] or [1]
    async with streamable_http_client(BASE) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for page in pages:
                result = await session.call_tool("get_bddk_document", {"document_id": doc_id, "page_number": page})
                if result.isError:
                    raise RuntimeError(f"Document fetch failed: {doc_id} page {page}")
                print(f"\n========== {doc_id} page {page} ==========")
                print("\n".join(block.text for block in result.content if block.type == "text"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
