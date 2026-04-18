# Context Window Token Measurement Script

**Date:** 2026-04-15
**Status:** Approved

## Problem

The BDDK MCP server exposes 21 tools to Claude. Each tool returns variable-length text that consumes context window tokens. There is no visibility into which tools are the biggest context consumers, making it impossible to optimize.

## Solution

A standalone test script (`measure_context.py`) that connects to the remote PostgreSQL database, calls each tool with representative inputs, measures output sizes with `tiktoken`, and prints a ranked report.

## Design

### Script: `measure_context.py`

Single file in the project root. Run with `uv run python measure_context.py`. Requires `BDDK_DATABASE_URL` to be set (same as normal server operation).

### Approach: Direct function calls

Import tool modules directly, create a real `Dependencies` instance (pool, doc_store, client, vector_store), and call each registered tool function. No MCP protocol overhead — direct Python calls against real data.

### Token counting

Use `tiktoken` with `cl100k_base` encoding (closest available approximation to Claude's tokenizer). Added as a dev dependency.

### Two measurement categories

**1. Baseline cost (per-conversation)**

Measure the full tool listing — all 21 tools' names, descriptions, and parameter schemas serialized as JSON. This is sent to Claude on every conversation turn.

**2. Per-call cost (per-invocation)**

Call each tool with representative inputs and measure the output:

| Tool | Test inputs | Notes |
|---|---|---|
| `search_bddk_decisions` | `keywords="kredi"` | Default page size |
| `search_bddk_institutions` | `keywords=""` | All institutions |
| `search_bddk_announcements` | `keywords=""`, `category="tümü"` | All categories |
| `search_document_store` | `query="faiz oranı riski"` | Semantic search |
| `get_bddk_document` | First doc ID from cache, page 1 | 5000-char pages |
| `get_document_history` | Same doc ID | Version list |
| `document_store_stats` | No args | Short summary |
| `get_bddk_bulletin` | `metric_id="1.0.1"` | Default metric |
| `get_bddk_bulletin_snapshot` | No args | All metrics table |
| `get_bddk_monthly` | `table_no=1` | Default table |
| `bddk_cache_status` | No args | Short summary |
| `analyze_bulletin_trends` | `metric_id="1.0.1"` | Default metric |
| `get_regulatory_digest` | `period="month"` | 30-day digest |
| `compare_bulletin_metrics` | `"1.0.1,1.0.2,1.0.4"` | 3 metrics |
| `check_bddk_updates` | No args | Baseline call |
| `health_check` | No args | Short summary |
| `bddk_metrics` | No args | Short summary |
| `document_health` | No args | Can be large |

**Skipped tools** (mutate state, not context offenders):
- `refresh_bddk_cache` — re-scrapes BDDK website
- `sync_bddk_documents` — downloads documents, slow
- `trigger_startup_sync` — triggers migration

### Output format

Printed to stdout. No files written.

```
=== BDDK MCP Context Window Report ===

BASELINE (sent every conversation turn):
  Tool definitions (21 tools):  ~X,XXX tokens

PER-CALL TOKEN USAGE (ranked highest → lowest):
  #  Tool                            Chars    Tokens
  1. tool_name                       X,XXX    X,XXX
  2. ...

SUMMARY:
  Baseline (tool definitions):     X,XXX tokens
  Largest single call:             X,XXX tokens (tool_name)
  Average per call:                  XXX tokens
  Worst 3-call sequence:           X,XXX tokens
  Total if ALL tools called once: XX,XXX tokens

  Claude context window: 200,000 tokens
  Baseline + worst 3 calls:  ~X.X% of context
```

### Dependencies

- `tiktoken` — added to `[project.optional-dependencies]` or `[tool.uv.dev-dependencies]` in `pyproject.toml`
- All existing project dependencies (asyncpg, httpx, etc.)

### Error handling

If a tool call fails (e.g., vector store not initialized, network timeout), the script logs the error and reports `ERROR` for that tool instead of crashing. The report still shows all tools that succeeded.

### What this does NOT do

- Does not modify the production server code
- Does not add runtime instrumentation
- Does not start an MCP server or use MCP protocol
- Does not write files or mutate database state
