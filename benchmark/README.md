# BDDK MCP Benchmark Tool Surface

This benchmark directory keeps tool-calling fixtures and result reporting separate from runtime deployment mode. Do not infer the production MCP tool count from the benchmark schema count alone.

## Tool Profiles

| Profile | Count | Source | Notes |
|---|---:|---|---|
| `runtime-public` | 16 | `server.py` with `BDDK_ADMIN_TOOLS=false` | Default read-only public deployment. |
| `runtime-admin` | 26 | `server.py` with `BDDK_ADMIN_TOOLS=true` | Public tools plus document stats, sync, health, metrics, quality, and backfill operator tools. |
| `benchmark-schema-fixture` | 23 | `benchmark/tool_schemas.py` | OpenAI-compatible function schemas used by the local benchmark harness. This fixture is intentionally explicit and may lag or differ from a runtime deployment profile. |

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
- `bddk_cache_status`
- `analyze_bulletin_trends`
- `get_regulatory_digest`
- `compare_bulletin_metrics`
- `check_bddk_updates`

## Runtime Admin Additions

When `BDDK_ADMIN_TOOLS=true`, the runtime also exposes:

- `document_store_stats`
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

Phase 2 result JSON records the exact tool surface used in each run:

- `exposed_tool_list`
- `deployment_config.tool_count`
- `deployment_config.max_tool_calls`
- `model_id`
- `mcp_server_version`

When comparing benchmark runs, compare the recorded `exposed_tool_list` and `deployment_config.tool_count` first. A run against `benchmark-schema-fixture` is not directly comparable to a live `runtime-public` or `runtime-admin` run unless the exposed tools are the same.

For production-style benchmark debugging, set `BDDK_TELEMETRY_ENABLED=true` on the MCP server. Retrieval tools then persist privacy-safe rows in `tool_call_traces` with tool name, args hash/summary, latency, result counts, document IDs, quality labels, relevance stats, optional `BDDK_TELEMETRY_MODEL_ID`, and optional `BDDK_TELEMETRY_SESSION_ID`.
