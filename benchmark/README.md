# BDDK MCP Benchmark Tool Surface

This benchmark directory keeps tool-calling cases and result reporting separate from runtime deployment mode. Its schemas come from the canonical operator registry, but a benchmark contract is not proof of a live server's configured profile.

The production wheel intentionally contains only `bddk_mcp`; run benchmarks from a source checkout. Install the optional provider-backed grader dependency with:

```bash
uv sync --group benchmark
```

## Tool Profiles

| Profile | Count | Source | Notes |
|---|---:|---|---|
| `runtime-public` | 15 | `BDDK_TOOL_PROFILE=public` | Default public process and public database identity. |
| `runtime-operator` | 28 | `BDDK_TOOL_PROFILE=operator` | Separate operator process and DSN; 15 public plus 13 operator tools. |
| `benchmark-operator-contract` | 28 | `benchmark/tool_schemas.py` | OpenAI-compatible schemas exported from the same canonical operator registry used by the runtime. |

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

## Runtime Operator Additions

With `BDDK_TOOL_PROFILE=operator` or `bddk-mcp serve --profile operator`, the runtime also exposes:

- `document_store_stats`
- `bddk_cache_status`
- `refresh_bddk_cache`
- `sync_bddk_documents`
- `trigger_startup_sync`
- `get_operator_job`
- `list_operator_jobs`
- `cancel_operator_job`
- `document_health`
- `health_check`
- `bddk_metrics`
- `backfill_degraded_documents`
- `document_quality_report`

The operator runtime requires `BDDK_OPERATOR_DATABASE_URL`. Job records and
leases are durable PostgreSQL state. The baseline intentionally runs one
operator replica; multi-replica execution remains outside the validated
deployment contract.

## Destructive retrieval-score database guard

`scripts/retrieval_score.py` deletes and recreates its synthetic document IDs.
It therefore requires both a distinct `BDDK_TEST_DATABASE_URL` and a guard
record provisioned only in the approved disposable database. Store only the
SHA-256 digest, never the guard token itself:

```sql
CREATE TABLE bddk_meta.retrieval_benchmark_guard (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    guard_hash text NOT NULL CHECK (guard_hash ~ '^[0-9a-f]{64}$')
);
INSERT INTO bddk_meta.retrieval_benchmark_guard (singleton, guard_hash)
VALUES (true, 'REPLACE_SHA256_OF_RANDOM_TEST_GUARD');
```

Set the original random value as `BDDK_TEST_DATABASE_GUARD` (at least 32
characters). The script verifies the digest before its first write. This table
must not exist in production or shared databases.

## Phase 2 live MCP execution

Phase 2 now uses the official MCP Python client for both supported transports.
It runs `initialize`, discovers the complete live `tools/list` contract, and
executes every tool with `ClientSession.call_tool`. It does not use a custom
HTTP `/call-tool` route.

Streamable HTTP (the URL must include the MCP path):

```bash
python -m benchmark.run --phase 2 \
  --model qwen3.5-9b \
  --mcp-transport streamable-http \
  --mcp-url http://127.0.0.1:8000/mcp
```

For an authenticated endpoint, place the bearer token in
`BDDK_BENCHMARK_MCP_TOKEN`; it is sent as an Authorization header and is never
written to benchmark results.

stdio:

```bash
python -m benchmark.run --phase 2 \
  --model qwen3.5-9b \
  --mcp-transport stdio \
  --mcp-command bddk-mcp \
  --mcp-arg serve --mcp-arg=--profile --mcp-arg public \
  --mcp-arg=--transport --mcp-arg stdio
```

The stdio child receives an exact runtime environment allowlist. LLM/provider
API keys, `ANTHROPIC_API_KEY`, grader configuration, the HTTP benchmark bearer
token, and unrelated parent-process variables are not inherited. Database
identities needed by the selected MCP server profile remain child-runtime
inputs and are never copied into result metadata. The harness also verifies
that the installed MCP SDK's implicit stdio defaults remain a subset of this
reviewed allowlist and fails closed if a future SDK broadens them.

The full case set includes operator cases. A public live profile therefore
marks those cases `LIVE_TOOL_UNAVAILABLE` and not comparable; it never pretends
that the operator contract was exposed. Run a separately secured operator
profile only when those cases are intentionally in scope.

MCP initialization, discovery, transport, malformed model arguments, unknown
tools, `isError=true`, and empty tool results fail closed. They are not passed
back to the model as fabricated tool text. Model-grader absence or failure is
also an explicit not-comparable state; it does not silently reuse the
claim-support metric as a second model score.

Retrieval completion is driven by the structured result `status`: `ok` and
`partial` require at least one structured evidence reference, while
`no_results`, `unavailable`, an unknown status, or a missing structured result
cannot count as successful retrieval. Expected arguments are graded on the
expected tool call, and multi-tool chains must occur in order (unrelated calls
may appear between required steps). Expected document and section references
must occur together on the same structured evidence item.

The deterministic numeric metric is answer-claim support, not recall of every
number present in tool output. Answers without numeric claims are marked
`unscored`; they are never assigned a synthetic `1.0`.

## Optional external model grader

Anthropic grading is external network egress and is disabled even when an API
key happens to exist in the parent environment. Enable it only for an approved
run by setting both variables explicitly:

```bash
export BDDK_BENCHMARK_ALLOW_EXTERNAL_GRADER=true
export ANTHROPIC_API_KEY='...'
python -m benchmark.run --phase 2 --model qwen3.5-9b
```

Tool evidence and answers are redacted, bounded, encoded as JSON data, and
placed inside collision-resistant delimiters before egress. Embedded source
instructions remain untrusted data. If external egress is not approved, omit
the opt-in: routing, arguments, retrieval status, ordered-chain, and structured
source metrics still run, while model-grounding and audit-grade comparison are
reported as unavailable.

## Benchmark Result Metadata

Phase 2 result JSON records the contract discovered from the live process:

- `live_tool_list`
- `live_tool_schema_sha256`
- `deployment_config.tool_count`
- `deployment_config.max_tool_calls`
- `model_id`
- `mcp_transport`
- `mcp_protocol_version`
- `mcp_server_version`
- full redacted final answers and structured tool evidence, plus SHA-256 trace hashes
- external grader model, status, and safe reason code
- Git commit and dirty-state fingerprint
- selected-case dataset hash and case IDs
- declared corpus ID/hash, when supplied, and an observed evidence-reference hash

Compare `live_tool_schema_sha256`, server/protocol version, corpus release, model
identifier, and grader status before comparing scores across runs. A public-only
run is not directly comparable to an operator-profile run.

For a named corpus release, set `BDDK_BENCHMARK_CORPUS_ID` and, when known,
`BDDK_BENCHMARK_CORPUS_SHA256`. These values describe the already-deployed
corpus; the harness also fingerprints document/section/content-hash references
actually observed during the run. Result writing performs a final recursive
credential redaction. Bearer tokens, API keys, passwords, DSNs, cookies, and
credential-bearing URLs must not be persisted.

For production-style benchmark debugging, telemetry requires both
`BDDK_TELEMETRY_ENABLED=true` and a distinct
`BDDK_TELEMETRY_DATABASE_URL` backed by the INSERT-only telemetry role.
Retrieval tools then persist privacy-safe rows in `tool_call_traces` with tool
name, args hash/summary, latency, result counts, document IDs, quality labels,
relevance stats, optional `BDDK_TELEMETRY_MODEL_ID`, and optional
`BDDK_TELEMETRY_SESSION_ID`.
