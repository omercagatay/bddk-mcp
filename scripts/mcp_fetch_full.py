"""Print selected document pages through the MCP SDK (diagnostic helper for a running server)."""

from __future__ import annotations

import argparse
import asyncio
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

DEFAULT_URL = "http://localhost:8000/mcp"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print pages of one document through a running MCP server.")
    parser.add_argument("document_id", nargs="?", default="1040", help="Document id (default: 1040)")
    parser.add_argument("pages", nargs="*", type=int, help="Page numbers to print (default: page 1)")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Streamable-HTTP MCP endpoint (default: {DEFAULT_URL})")
    return parser


async def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    pages = args.pages or [1]
    async with streamable_http_client(args.url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for page in pages:
                result = await session.call_tool(
                    "get_bddk_document", {"document_id": args.document_id, "page_number": page}
                )
                if result.isError:
                    raise RuntimeError(f"Document fetch failed: {args.document_id} page {page}")
                print(f"\n========== {args.document_id} page {page} ==========")
                print("\n".join(block.text for block in result.content if block.type == "text"))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
