#!/usr/bin/env python3
"""Smoke-test a running BDDK MCP container over Streamable HTTP.

Usage: uv run python scripts/ci_image_smoke.py http://localhost:8000

Connects to <base-url>/mcp, initializes a session, checks that tools/list returns exactly the
reviewed tool surface and that one catalog search answers from the corpus baked into the image.
Talks only to the URL it is given. Exits non-zero with a message on any failure.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from bddk_mcp.tools.registry import PUBLIC_TOOL_NAMES

SEARCH_KEYWORDS = "bankacılık kanunu"


class SmokeFailure(Exception):
    """A smoke check failed."""


async def run_smoke(base_url: str) -> None:
    url = base_url.rstrip("/") + "/mcp"
    async with streamable_http_client(url) as (read_stream, write_stream, _session_id):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            listed = await session.list_tools()
            names = [tool.name for tool in listed.tools]
            if sorted(names) != sorted(PUBLIC_TOOL_NAMES) or len(names) != len(set(names)):
                missing = sorted(set(PUBLIC_TOOL_NAMES) - set(names))
                extra = sorted(set(names) - set(PUBLIC_TOOL_NAMES))
                raise SmokeFailure(f"tools/list differs from PUBLIC_TOOL_NAMES: missing={missing} extra={extra}")

            result = await session.call_tool("search_bddk_regulations", {"keywords": SEARCH_KEYWORDS})
            if result.isError:
                raise SmokeFailure(f"search_bddk_regulations returned an error: {result.content!r}")
            payload = result.structuredContent
            if not isinstance(payload, dict):
                raise SmokeFailure("search_bddk_regulations returned no structured content")
            if payload.get("status") != "ok":
                raise SmokeFailure(f"search_bddk_regulations status is {payload.get('status')!r}, expected 'ok'")
            items = payload.get("results")
            if not isinstance(items, list) or not items:
                raise SmokeFailure(f"search_bddk_regulations found nothing for {SEARCH_KEYWORDS!r}")
            first = items[0]
            if not isinstance(first, dict) or not str(first.get("title") or "").strip():
                raise SmokeFailure("first search result does not name a document (empty title)")
            print(f"first result: {first['title']}")

    print(f"smoke ok: {len(names)} tools, {len(items)} search results")


def _leaf_errors(exc: BaseException) -> list[str]:
    """Flatten (nested) exception groups into one 'Type: message' line per real cause."""
    if isinstance(exc, BaseExceptionGroup):
        return [line for sub in exc.exceptions for line in _leaf_errors(sub)]
    return [f"{type(exc).__name__}: {exc}"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a running BDDK MCP server over HTTP.")
    parser.add_argument("base_url", help="server base URL, for example http://localhost:8000")
    args = parser.parse_args()
    try:
        asyncio.run(run_smoke(args.base_url))
    except SmokeFailure as exc:
        print(f"SMOKE FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # connection, protocol or timeout problems
        for line in _leaf_errors(exc):
            print(f"SMOKE FAILED: {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
