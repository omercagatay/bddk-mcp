# Context Window Measurement Script — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone script that measures how many tokens each BDDK MCP tool consumes and ranks them, so the user can identify context window offenders.

**Architecture:** Direct Python calls against the real PostgreSQL database. Import tool modules, construct `Dependencies`, call each tool function, measure outputs with `tiktoken`. Single file, no production code changes.

**Tech Stack:** Python 3.12, tiktoken, asyncpg, httpx, existing bddk-mcp modules

---

## File Structure

| File | Purpose |
|---|---|
| Create: `measure_context.py` | Main script — setup, call tools, measure, report |
| Create: `tests/test_measure_context.py` | Tests for token counting and report formatting |
| Modify: `pyproject.toml` | Add `tiktoken` to dev dependencies |

---

### Task 1: Add tiktoken dependency

**Files:**
- Modify: `pyproject.toml:19-24`

- [ ] **Step 1: Add tiktoken to dev dependencies**

In `pyproject.toml`, add `tiktoken` to the `[dependency-groups] dev` list:

```toml
[dependency-groups]
dev = [
    "pytest>=9.0.2",
    "pytest-asyncio>=0.24",
    "ruff>=0.8",
    "tiktoken>=0.7",
]
```

- [ ] **Step 2: Install**

Run: `cd /home/cagatay/bddk-mcp && uv sync --dev`
Expected: tiktoken installs successfully, lock file updates.

- [ ] **Step 3: Verify import works**

Run: `cd /home/cagatay/bddk-mcp && uv run python -c "import tiktoken; enc = tiktoken.get_encoding('cl100k_base'); print(enc.encode('test'))"`
Expected: prints `[1985]` (or similar token ID list)

- [ ] **Step 4: Commit**

```bash
cd /home/cagatay/bddk-mcp
git add pyproject.toml uv.lock
git commit -m "chore: add tiktoken dev dependency for context measurement"
```

---

### Task 2: Write token counting and report formatting (TDD)

**Files:**
- Create: `measure_context.py`
- Create: `tests/test_measure_context.py`

- [ ] **Step 1: Write failing tests for `count_tokens` and `format_report`**

Create `tests/test_measure_context.py`:

```python
"""Tests for measure_context token counting and report formatting."""

import tiktoken

from measure_context import count_tokens, format_report


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_simple_english(self):
        result = count_tokens("hello world")
        assert result > 0
        assert isinstance(result, int)

    def test_turkish_text(self):
        result = count_tokens("Türkiye Bankacılık Düzenleme ve Denetleme Kurumu")
        assert result > 0

    def test_long_text_more_tokens(self):
        short = count_tokens("kısa metin")
        long = count_tokens("kısa metin " * 100)
        assert long > short

    def test_consistent_results(self):
        text = "BDDK kararları"
        assert count_tokens(text) == count_tokens(text)


class TestFormatReport:
    def test_empty_results(self):
        report = format_report(baseline_tokens=100, tool_results=[])
        assert "BASELINE" in report
        assert "100" in report

    def test_ranked_by_tokens_descending(self):
        results = [
            {"tool": "tool_a", "chars": 100, "tokens": 30, "error": None},
            {"tool": "tool_b", "chars": 500, "tokens": 150, "error": None},
            {"tool": "tool_c", "chars": 200, "tokens": 60, "error": None},
        ]
        report = format_report(baseline_tokens=200, tool_results=results)
        lines = report.split("\n")
        # Find the ranked lines — tool_b (150) should appear before tool_c (60) before tool_a (30)
        tool_positions = {}
        for i, line in enumerate(lines):
            for r in results:
                if r["tool"] in line and str(r["tokens"]) in line:
                    tool_positions[r["tool"]] = i
        assert tool_positions["tool_b"] < tool_positions["tool_c"] < tool_positions["tool_a"]

    def test_error_tools_shown(self):
        results = [
            {"tool": "broken_tool", "chars": 0, "tokens": 0, "error": "connection refused"},
        ]
        report = format_report(baseline_tokens=100, tool_results=results)
        assert "ERROR" in report
        assert "broken_tool" in report

    def test_summary_section(self):
        results = [
            {"tool": "tool_a", "chars": 1000, "tokens": 250, "error": None},
            {"tool": "tool_b", "chars": 2000, "tokens": 500, "error": None},
        ]
        report = format_report(baseline_tokens=200, tool_results=results)
        assert "SUMMARY" in report
        assert "200,000" in report  # Claude context window reference
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/cagatay/bddk-mcp && uv run pytest tests/test_measure_context.py -v --tb=short`
Expected: FAIL — `ModuleNotFoundError: No module named 'measure_context'`

- [ ] **Step 3: Implement `count_tokens` and `format_report`**

Create `measure_context.py` with the two pure functions first (no async, no DB):

```python
"""Measure context window token usage of BDDK MCP tools.

Usage: uv run python measure_context.py

Requires BDDK_DATABASE_URL to be set (same as server.py).
"""

import tiktoken

_encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding."""
    if not text:
        return 0
    return len(_encoder.encode(text))


def format_report(baseline_tokens: int, tool_results: list[dict]) -> str:
    """Format the measurement results as a ranked report.

    Args:
        baseline_tokens: Token count for the tool definitions.
        tool_results: List of dicts with keys: tool, chars, tokens, error.
    """
    lines: list[str] = []
    lines.append("=== BDDK MCP Context Window Report ===")
    lines.append("")

    # Baseline
    lines.append("BASELINE (sent every conversation turn):")
    lines.append(f"  Tool definitions:  {baseline_tokens:,} tokens")
    lines.append("")

    # Separate successes and errors
    ok = [r for r in tool_results if r["error"] is None]
    errors = [r for r in tool_results if r["error"] is not None]

    # Ranked results
    ok_sorted = sorted(ok, key=lambda r: r["tokens"], reverse=True)

    lines.append("PER-CALL TOKEN USAGE (ranked highest -> lowest):")
    lines.append(f"  {'#':<4} {'Tool':<35} {'Chars':>8} {'Tokens':>8}")
    lines.append(f"  {'':->4} {'':->35} {'':->8} {'':->8}")

    for i, r in enumerate(ok_sorted, 1):
        lines.append(f"  {i:<4} {r['tool']:<35} {r['chars']:>8,} {r['tokens']:>8,}")

    if errors:
        lines.append("")
        lines.append("ERRORS (could not measure):")
        for r in errors:
            lines.append(f"  {r['tool']:<35} ERROR: {r['error']}")

    # Summary
    lines.append("")
    lines.append("SUMMARY:")
    lines.append(f"  Baseline (tool definitions):     {baseline_tokens:>6,} tokens")

    if ok_sorted:
        largest = ok_sorted[0]
        lines.append(f"  Largest single call:             {largest['tokens']:>6,} tokens ({largest['tool']})")

        avg = sum(r["tokens"] for r in ok_sorted) // len(ok_sorted) if ok_sorted else 0
        lines.append(f"  Average per call:                {avg:>6,} tokens")

        worst3 = sum(r["tokens"] for r in ok_sorted[:3])
        lines.append(f"  Worst 3-call sequence:           {worst3:>6,} tokens")

        total = sum(r["tokens"] for r in ok_sorted)
        lines.append(f"  Total if ALL tools called once:  {total:>6,} tokens")

        combined = baseline_tokens + worst3
        pct = combined / 200_000 * 100
        lines.append("")
        lines.append(f"  Claude context window:           200,000 tokens")
        lines.append(f"  Baseline + worst 3 calls:      ~{pct:.1f}% of context")

    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/cagatay/bddk-mcp && uv run pytest tests/test_measure_context.py -v --tb=short`
Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/cagatay/bddk-mcp
git add measure_context.py tests/test_measure_context.py
git commit -m "feat: add token counting and report formatting for context measurement"
```

---

### Task 3: Write baseline measurement (tool definitions)

**Files:**
- Modify: `measure_context.py`
- Modify: `tests/test_measure_context.py`

- [ ] **Step 1: Write failing test for `measure_baseline`**

Append to `tests/test_measure_context.py`:

```python
import asyncio

from measure_context import measure_baseline


class TestMeasureBaseline:
    def test_returns_token_count(self):
        tokens, text = asyncio.run(measure_baseline())
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_text_contains_tool_names(self):
        _, text = asyncio.run(measure_baseline())
        # At minimum, the built-in tools from server.py should be listed
        assert "search_bddk_decisions" in text
        assert "health_check" in text

    def test_text_contains_descriptions(self):
        _, text = asyncio.run(measure_baseline())
        # Tool descriptions should be present
        assert "BDDK" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/cagatay/bddk-mcp && uv run pytest tests/test_measure_context.py::TestMeasureBaseline -v --tb=short`
Expected: FAIL — `cannot import name 'measure_baseline'`

- [ ] **Step 3: Implement `measure_baseline`**

Add to `measure_context.py`, after the `format_report` function:

```python
import asyncio
import json


async def measure_baseline() -> tuple[int, str]:
    """Measure token cost of all tool definitions.

    Registers all tools on a temporary FastMCP instance (no DB needed),
    lists them, serializes to JSON, and counts tokens.

    Returns (token_count, serialized_text).
    """
    from mcp.server.fastmcp import FastMCP

    from deps import Dependencies

    mcp = FastMCP("measure")

    # Dummy deps — tools register their schemas without needing a live DB
    deps = Dependencies(
        pool=None, doc_store=None, client=None, http=None, vector_store=None
    )

    from tools import admin, analytics, bulletin, documents, search, sync

    search.register(mcp, deps)
    documents.register(mcp, deps)
    bulletin.register(mcp, deps)
    analytics.register(mcp, deps)
    sync.register(mcp, deps)
    admin.register(mcp, deps)

    tools = await mcp.list_tools()
    schemas = [json.loads(t.model_dump_json()) for t in tools]
    text = json.dumps(schemas, indent=2)
    return count_tokens(text), text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/cagatay/bddk-mcp && uv run pytest tests/test_measure_context.py::TestMeasureBaseline -v --tb=short`
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/cagatay/bddk-mcp
git add measure_context.py tests/test_measure_context.py
git commit -m "feat: add baseline measurement for tool definition token cost"
```

---

### Task 4: Write per-tool measurement and main entry point

**Files:**
- Modify: `measure_context.py`

- [ ] **Step 1: Add `measure_tools` and `main` to `measure_context.py`**

Append to `measure_context.py`:

```python
import logging
import sys
import time

import asyncpg
import httpx

from client import BddkApiClient
from config import DATABASE_URL, PG_POOL_MAX, PG_POOL_MIN, REQUEST_TIMEOUT
from doc_store import DocumentStore

logger = logging.getLogger(__name__)

# Tools to measure: (function_name, kwargs)
# Functions are looked up from the registered tool closures at runtime.
TOOL_CALLS: list[tuple[str, dict]] = [
    ("search_bddk_decisions", {"keywords": "kredi"}),
    ("search_bddk_institutions", {"keywords": ""}),
    ("search_bddk_announcements", {"keywords": "", "category": "tümü"}),
    ("search_document_store", {"query": "faiz oranı riski"}),
    ("get_bddk_document", {}),  # document_id filled at runtime
    ("get_document_history", {}),  # document_id filled at runtime
    ("document_store_stats", {}),
    ("get_bddk_bulletin", {"metric_id": "1.0.1"}),
    ("get_bddk_bulletin_snapshot", {}),
    ("get_bddk_monthly", {"table_no": 1}),
    ("bddk_cache_status", {}),
    ("analyze_bulletin_trends", {"metric_id": "1.0.1"}),
    ("get_regulatory_digest", {"period": "month"}),
    ("compare_bulletin_metrics", {"metric_ids": "1.0.1,1.0.2,1.0.4"}),
    ("check_bddk_updates", {}),
    ("health_check", {}),
    ("bddk_metrics", {}),
    ("document_health", {}),
]

# These tools mutate state — skip them
SKIP_TOOLS = {"refresh_bddk_cache", "sync_bddk_documents", "trigger_startup_sync"}


async def create_test_deps() -> tuple[Dependencies, list]:
    """Create real Dependencies for measuring tool outputs.

    Returns (deps, cleanup_callbacks) — call cleanup callbacks when done.
    """
    http = httpx.AsyncClient(
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
            "User-Agent": "bddk-mcp-measure/1.0",
        },
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        follow_redirects=True,
    )

    pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=PG_POOL_MIN,
        max_size=PG_POOL_MAX,
        command_timeout=30,
        timeout=10,
    )

    doc_store = DocumentStore(pool)
    await doc_store.initialize()

    client = BddkApiClient(pool=pool, doc_store=doc_store, http=http)
    await client.initialize()

    # Try vector store but don't fail if unavailable
    vector_store = None
    try:
        from vector_store import VectorStore

        vs = VectorStore(pool)
        await vs.initialize()
        vector_store = vs
    except Exception as e:
        logger.warning("VectorStore not available: %s", e)

    deps = Dependencies(
        pool=pool,
        doc_store=doc_store,
        client=client,
        http=http,
        vector_store=vector_store,
    )

    async def cleanup():
        await http.aclose()
        await pool.close()

    return deps, cleanup


async def measure_tools() -> list[dict]:
    """Call each tool with representative inputs and measure output tokens.

    Returns list of dicts: {tool, chars, tokens, error}.
    """
    from mcp.server.fastmcp import FastMCP

    from tools import admin, analytics, bulletin, documents, search, sync

    deps, cleanup = await create_test_deps()

    mcp = FastMCP("measure")
    search.register(mcp, deps)
    documents.register(mcp, deps)
    bulletin.register(mcp, deps)
    analytics.register(mcp, deps)
    sync.register(mcp, deps)
    admin.register(mcp, deps)

    # Build a lookup of tool name -> callable
    # FastMCP stores tools internally; we call them via call_tool
    tool_names = {t.name for t in await mcp.list_tools()}

    # Find a document_id from cache for get_bddk_document / get_document_history
    sample_doc_id = None
    cache_items = deps.client.get_cache_items()
    if cache_items:
        sample_doc_id = cache_items[0].document_id

    results: list[dict] = []

    for tool_name, kwargs in TOOL_CALLS:
        if tool_name in SKIP_TOOLS:
            continue
        if tool_name not in tool_names:
            results.append({"tool": tool_name, "chars": 0, "tokens": 0, "error": "tool not registered"})
            continue

        # Fill in document_id for document tools
        call_kwargs = dict(kwargs)
        if tool_name in ("get_bddk_document", "get_document_history"):
            if sample_doc_id is None:
                results.append({"tool": tool_name, "chars": 0, "tokens": 0, "error": "no documents in cache"})
                continue
            call_kwargs["document_id"] = sample_doc_id

        try:
            output = await mcp.call_tool(tool_name, call_kwargs)
            # call_tool returns list of content objects; extract text
            text = "\n".join(
                part.text for part in output if hasattr(part, "text")
            )
            chars = len(text)
            tokens = count_tokens(text)
            results.append({"tool": tool_name, "chars": chars, "tokens": tokens, "error": None})
            print(f"  {tool_name:<35} {chars:>8,} chars  {tokens:>6,} tokens")
        except Exception as e:
            results.append({"tool": tool_name, "chars": 0, "tokens": 0, "error": str(e)[:120]})
            print(f"  {tool_name:<35} ERROR: {e!s:.80}")

    await cleanup()
    return results


async def async_main() -> None:
    """Run all measurements and print the report."""
    print("Measuring BDDK MCP context window usage...\n")

    # 1. Baseline
    print("Measuring baseline (tool definitions)...")
    baseline_tokens, _ = await measure_baseline()
    print(f"  Tool definitions: {baseline_tokens:,} tokens\n")

    # 2. Per-tool
    print("Measuring per-tool output sizes...")
    tool_results = await measure_tools()
    print()

    # 3. Report
    report = format_report(baseline_tokens, tool_results)
    print(report)


def main() -> None:
    """Entry point."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full test suite to make sure nothing is broken**

Run: `cd /home/cagatay/bddk-mcp && uv run pytest tests/test_measure_context.py -v --tb=short`
Expected: all 12 tests still PASS (new code is async and not tested directly here — it's the integration entry point)

- [ ] **Step 3: Run ruff to check formatting and lint**

Run: `cd /home/cagatay/bddk-mcp && uv run ruff check measure_context.py && uv run ruff format --check measure_context.py`
Expected: no errors. If there are formatting issues, run `uv run ruff format measure_context.py` and re-check.

- [ ] **Step 4: Commit**

```bash
cd /home/cagatay/bddk-mcp
git add measure_context.py
git commit -m "feat: add per-tool measurement and main entry point"
```

---

### Task 5: Run the script against the remote database

**Files:** None (integration test — no code changes)

- [ ] **Step 1: Verify BDDK_DATABASE_URL is set**

Run: `echo $BDDK_DATABASE_URL | sed 's/:[^@]*@/:***@/'`
Expected: prints the database URL with password masked. If empty, set it first.

- [ ] **Step 2: Run the measurement script**

Run: `cd /home/cagatay/bddk-mcp && uv run python measure_context.py`
Expected: the full context window report prints to stdout. Some tools may show ERROR if vector store is unavailable — that's fine.

- [ ] **Step 3: Review the output**

Check:
- Baseline tokens are reasonable (expected ~2000-4000 for 21 tools)
- Tools are ranked from highest to lowest token usage
- Summary math is correct (worst 3 = sum of top 3)
- No unexpected errors

- [ ] **Step 4: Run the full test suite one final time**

Run: `cd /home/cagatay/bddk-mcp && uv run pytest tests/ -v --tb=short`
Expected: all existing tests still PASS, plus the new `test_measure_context.py` tests.
