# BDDK MCP Benchmark Tool Surface

This benchmark directory keeps tool-calling cases and result reporting separate from runtime deployment mode. Its schemas come from the canonical operator registry, but a benchmark contract is not proof of a live server's configured profile.

The production wheel intentionally contains only `bddk_mcp`; run benchmarks from a source checkout. Install the optional provider-backed grader dependency with:

```bash
uv sync --group benchmark
```

## Tool Profiles

| Profile | Count | Source | Notes |
|---|---:|---|---|
| `runtime-public` | 15 | `bddk-mcp` with `BDDK_ADMIN_TOOLS=false` | Default public deployment. |
| `runtime-admin` | 26 | `bddk-mcp` with `BDDK_ADMIN_TOOLS=true` | Public tools plus document stats, sync, health, metrics, quality, and backfill operator tools. |
| `benchmark-operator-contract` | 26 | `benchmark/tool_schemas.py` | OpenAI-compatible schemas exported from the same canonical operator registry used by the runtime. |

## Runtime Public Tools

- `search_bddk_regulations`
- `search_document_store`
- `search_bddk_institutions`
- `search_bddk_announcements`
- `get_bddk_document`
- `get_document_history`
- `get_document_section`
- `search_document_sections`
- `get_bddk_bulletin`
- `get_bddk_bulletin_snapshot`
- `get_bddk_monthly`
- `analyze_bulletin_trends`
- `get_regulatory_digest`
- `compare_bulletin_metrics`
- `check_bddk_updates`

## Runtime Admin Additions

When `BDDK_ADMIN_TOOLS=true`, the runtime also exposes:

- `document_store_stats`
- `bddk_cache_status`
- `refresh_bddk_cache`
- `sync_bddk_documents`
- `trigger_startup_sync`
- `document_health`
- `health_check`
- `bddk_metrics`
- `backfill_degraded_documents`
- `backfill_status`
- `document_quality_report`

## Benchmark Result Metadata

Phase 2 result JSON records the operator contract exported for the benchmark:

- `exposed_tool_list`
- `deployment_config.tool_count`
- `deployment_config.max_tool_calls`
- `model_id`
- `mcp_server_version`

The schemas are generated from the runtime registry, but the current Phase 2 harness still does not discover them from a live MCP `tools/list` response. Treat `exposed_tool_list` as the reviewed operator contract, not proof that a particular remote process exposed it. Compare it and `deployment_config.tool_count` first; a public-only host is not directly comparable to an operator-contract benchmark run.

For production-style benchmark debugging, set `BDDK_TELEMETRY_ENABLED=true` on the MCP server. Retrieval tools then persist privacy-safe rows in `tool_call_traces` with tool name, args hash/summary, latency, result counts, document IDs, quality labels, relevance stats, optional `BDDK_TELEMETRY_MODEL_ID`, and optional `BDDK_TELEMETRY_SESSION_ID`.
