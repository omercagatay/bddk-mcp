# BDDK MCP Server Restructure

**Date:** 2026-04-11
**Scope:** Restructure server.py monolith into modular architecture with dependency injection, fixing startup timeouts, improving search performance, and establishing proper lifecycle management.
**Execution:** Single big-bang refactor. Business logic unchanged, structure and lifecycle rewritten.

## Problem Statement

`server.py` is 1258 lines containing 20 MCP tool handlers, 6 global mutable singletons, lazy-init connection management, startup/shutdown lifecycle, migration logic, and search caching. This causes:

1. **Railway startup timeouts** — embedding model (~1.1GB) loads synchronously on first request, blocking the server for 20-30 seconds.
2. **Shutdown race conditions** — background sync task uses stores that `_graceful_shutdown()` already closed.
3. **Development friction** — every change to any tool risks breaking unrelated tools due to shared global state.
4. **Silent failures** — lazy-init singletons cache broken state; health check reports stale information.

## Target Architecture

### File Structure

```
server.py           (~80 lines)   entry point, lifecycle, FastMCP setup
deps.py             (~80 lines)   Dependencies dataclass, creation/teardown
tools/
  __init__.py
  search.py         (~150 lines)  search_bddk_decisions, search_institutions,
                                  search_announcements, search_document_store
  documents.py      (~100 lines)  get_bddk_document, get_document_history,
                                  document_store_stats
  bulletin.py       (~150 lines)  get_bddk_bulletin, get_bddk_bulletin_snapshot,
                                  get_bddk_monthly, bddk_cache_status
  analytics.py      (~120 lines)  analyze_bulletin_trends, regulatory_digest,
                                  compare_bulletin_metrics, check_bddk_updates
  sync.py           (~150 lines)  refresh_bddk_cache, sync_bddk_documents,
                                  trigger_startup_sync, startup sync logic,
                                  pgvector migration
  admin.py          (~80 lines)   health_check, bddk_metrics
```

### Dependency Container

```python
@dataclass
class Dependencies:
    pool: asyncpg.Pool
    doc_store: DocumentStore
    client: BddkApiClient
    http: httpx.AsyncClient           # shared across client + syncer
    vector_store: VectorStore | None = None  # None until background init completes
    sync_task: asyncio.Task | None = None
    vector_init_task: asyncio.Task | None = None

    # Health state
    last_sync_time: float | None = None
    last_sync_error: str | None = None
    sync_consecutive_failures: int = 0
    sync_circuit_open: bool = False
    server_start_time: float = field(default_factory=time.time)
```

Each tool module exports `register(mcp: FastMCP, deps: Dependencies)`. Tools access dependencies via closure capture over `deps`.

### Lifecycle

**Startup (server.py):**

1. Create shared `httpx.AsyncClient`
2. Create `asyncpg.Pool` with `min_size`, `max_size`, `command_timeout=30`, `timeout=10`
3. Create `DocumentStore(pool)`, call `await store.initialize()`
4. Create `BddkApiClient(pool, doc_store, http)`
5. Bundle into `Dependencies(pool, doc_store, client, http, vector_store=None)`
6. Register all tool modules: `search.register(mcp, deps)`, etc.
7. Launch background task: load embedding model → `VectorStore(pool)` → `await vs.initialize()` → `deps.vector_store = vs`
8. If `AUTO_SYNC`: launch background task → `deps.sync_task`

Pool creation and doc_store init take ~2 seconds. Server is ready to serve FTS queries immediately. Semantic search returns a "still initializing" message until vector_store is set.

**Shutdown (server.py):**

1. Cancel `deps.vector_init_task` if still running, await it
2. Cancel `deps.sync_task` if still running, await it
3. Close `deps.client` (closes HTTP session)
4. Close `deps.http` (shared httpx client)
5. Close `deps.pool` (closes all PostgreSQL connections)

Ordering guarantees: no task is using connections when the pool closes.

## Performance Improvements

### 1. Concurrent hybrid search

**File:** `vector_store.py` `_hybrid_search()`
**Current:** Sequential `await _vector_search()` then `await _fts_search()`
**Change:** `asyncio.gather(_vector_search(...), _fts_search(...))`
**Impact:** ~2x faster hybrid search latency

### 2. Thread executor for embedding and reranking

**File:** `vector_store.py` `_embed()` and `_rerank()`
**Current:** Synchronous `self._embed_fn.encode()` blocks the asyncio event loop
**Change:** `await loop.run_in_executor(None, self._embed_fn.encode, ...)`
**Impact:** Other requests can be served during CPU-bound embedding/reranking

### 3. Bulk chunk insertion

**File:** `vector_store.py` `add_document()`
**Current:** N individual `INSERT` statements in a for loop
**Change:** `await conn.executemany(sql, args_list)`
**Impact:** ~Nx fewer PostgreSQL round-trips per document indexing

### 4. Targeted page retrieval

**File:** `vector_store.py` `get_document_page()`
**Current:** `get_document()` fetches ALL chunks, reconstructs full text, slices one page
**Change:** Calculate which chunk indices overlap with the requested page range. Fetch only those chunks. Reconstruct only the needed slice.
**Formula:** For page P with PAGE_SIZE chars, step = chunk_size - overlap:
  - start_char = (P-1) * PAGE_SIZE
  - end_char = P * PAGE_SIZE
  - first_chunk = start_char // step
  - last_chunk = end_char // step
  - Fetch chunks [first_chunk, last_chunk] only
**Impact:** For a 50-chunk document, page 1 fetches ~2 chunks instead of 50

### 5. Batch version count query

**File:** `tools/search.py` (extracted from server.py search_bddk_decisions)
**Current:** `get_version_count(doc_id)` called in a loop per search result
**Change:** `get_version_counts(doc_ids: list[str])` — single query:
```sql
SELECT document_id, COUNT(*), MAX(synced_at)
FROM document_versions
WHERE document_id = ANY($1)
GROUP BY document_id
```
**Impact:** 1 query instead of N per search

### 6. Search cache with O(1) eviction

**File:** `tools/search.py`
**Current:** Plain dict, O(n) scan to find oldest entry on eviction
**Change:** `collections.OrderedDict` — move to end on access, pop from front on eviction
**Impact:** O(1) eviction, correct LRU behavior

## Reliability Improvements

### 1. Eager pool creation, fail fast

**Current:** Pool created lazily on first request. If DATABASE_URL is wrong, the first user request gets a connection error.
**Change:** Pool created at startup. If it fails, server exits immediately with a clear error instead of serving broken responses.

### 2. Shared httpx client

**Current:** Three separate `httpx.AsyncClient` instances (BddkApiClient, DocumentSyncer, and implicitly in data_sources).
**Change:** Single `httpx.AsyncClient` created in deps, shared via injection. Fewer TLS handshakes, single connection pool, one place to configure timeouts.
**Constructor changes:** `BddkApiClient.__init__` and `DocumentSyncer.__init__` drop their internal `httpx.AsyncClient` creation and accept `http: httpx.AsyncClient` as a parameter instead. `data_sources.py` functions that make HTTP calls accept the shared client as an argument.

### 3. Nougat model loaded once

**File:** `doc_sync.py` `_extract_with_nougat()`
**Current:** `NougatModel.from_pretrained(...)` called on every extraction — loads ~2GB model each time.
**Change:** Module-level lazy singleton for the model, reused across extractions. Cleared on module unload.

### 4. Cache save without full wipe

**File:** `client.py` `_save_cache_to_db()`
**Current:** `DELETE FROM decision_cache` then N individual INSERTs. Crash mid-save = empty cache.
**Change:** Direct upsert without DELETE. Use `executemany` for bulk insert. Cache is never fully wiped during save.

### 5. Sync circuit breaker

**File:** `tools/sync.py`
**Behavior:**
- Track consecutive sync failures in `deps.sync_consecutive_failures`
- After 10 consecutive failures: `deps.sync_circuit_open = True`, stop retrying
- `health_check` reports: "degraded: sync circuit open after 10 consecutive failures"
- Manual `trigger_startup_sync` resets the circuit
- Total sync timeout: 5 minutes on startup sync. After that, log warning, stop, report partial results.

### 6. pgvector migration: batch existence check + concurrent embedding

**File:** `tools/sync.py` `_migrate_to_pgvector()`
**Current:** Sequential `has_document` per doc, sequential embedding per doc
**Change:**
- Batch existence: `SELECT doc_id FROM document_chunks WHERE doc_id = ANY($1)` for all doc_ids at once
- Process missing docs in batches of 10 with concurrent embedding
- Total migration timeout: 10 minutes

### 7. Pool acquisition timeout

**Current:** No timeout — requests block indefinitely if pool is exhausted.
**Change:** `asyncpg.create_pool(..., timeout=10)` — requests fail after 10 seconds with a clear error instead of hanging.

## Error Handling

### Consistent tool error wrapper

Each tool module uses a consistent pattern. The inner `except` catches domain errors (BddkError, asyncpg.PostgresError, httpx errors) — these produce user-facing messages. The outer `except Exception` is a safety net for unexpected bugs — these log full tracebacks and return a generic message. Tools that access `deps.vector_store` check for `None` first and return a "still initializing" message.

```python
@mcp.tool()
async def some_tool(...) -> str:
    try:
        ...
    except (BddkError, asyncpg.PostgresError) as e:
        metrics.increment("tool_errors", tool="some_tool")
        return f"Error: {e}"
    except Exception as e:
        logger.exception("Unexpected error in some_tool")
        metrics.increment("tool_errors", tool="some_tool")
        return f"Internal error. Please try again."
```

### Health state tracking

`Dependencies` carries health state that `health_check` reads:
- `vector_store is None` → "vector store initializing"
- `sync_circuit_open` → "degraded: sync circuit open"
- `last_sync_error` → reports last error
- `pool.get_size()` / `pool.get_idle_size()` → connection pool utilization

## Testing Strategy

1. **Existing tests** — fix broken imports after refactor, don't rewrite test logic
2. **New tests:**
   - `test_deps.py` — Dependencies creation, teardown ordering, background vector store init
   - `test_lifecycle.py` — startup sequence, shutdown with pending tasks
   - `test_circuit_breaker.py` — sync failure counting, circuit open/close
3. **Integration test** — create real Dependencies with test PostgreSQL, register tools, exercise: search → get document → health check
4. **No database mocking** — use real PostgreSQL via existing conftest.py fixtures

## What Does Not Change

- All 20 MCP tool signatures and return formats
- FastMCP server instructions (grounding rules)
- BDDK scraping logic (parsers, URL templates, Turkish stemming)
- RRF fusion algorithm and scoring
- Document extraction pipeline (Nougat → markitdown → HTML fallback)
- Existing test assertions (only imports change)
- Deployment config (Dockerfile, railway.toml, Procfile)

## Migration

Single big-bang refactor executed as one commit (or small PR). Justified because:
- Solo project, no concurrent PRs to conflict with
- Global → DI transition doesn't support half-measures
- Total scope is reorganization, not rewrite — tool handler logic is copy-pasted into new files
- 17-file test suite validates nothing is broken
