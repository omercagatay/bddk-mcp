# BDDK MCP Server — ChromaDB Rebuild

**Date:** 2026-04-16
**Status:** Draft
**Constraint:** Company requires ChromaDB as the vector database.

## Overview

Rebuild the BDDK MCP server from scratch using ChromaDB (embedded mode) instead of PostgreSQL + pgvector. Flat module architecture, core tools first, with seed data loading and live scraping for ongoing updates.

## Architecture

Flat module design — each file has a single responsibility, no layered abstractions.

```
bddk-mcp/
  server.py          # FastMCP entry point, tool definitions
  scraper.py         # BDDK website scraping
  store.py           # ChromaDB wrapper (add, search, get, delete)
  embeddings.py      # Custom embedding function (multilingual-e5-base)
  seed.py            # Load seed JSON files into ChromaDB
  models.py          # Pydantic schemas for tool inputs/outputs
  config.py          # Settings via env vars
  chroma_data/       # ChromaDB persistent storage (gitignored)
  seed_data/         # JSON seed files (committed)
  tests/             # pytest tests
  pyproject.toml     # Project config, uv managed
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastmcp` | MCP server framework |
| `chromadb` | Vector database (embedded mode) |
| `sentence-transformers` | Embedding model runtime |
| `httpx` | Async HTTP client for scraping |
| `beautifulsoup4` | HTML parsing |
| `pydantic` | Data validation |
| `pytest` + `pytest-asyncio` | Testing |

**Python version:** 3.12

## ChromaDB Store Design

### Collections

**`decisions`** — BDDK decisions (kararlar).

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Decision ID from BDDK |
| `document` | string | Decision text content |
| `metadata.decision_date` | string | Date of the decision |
| `metadata.decision_number` | string | Karar numarasi |
| `metadata.category` | string | Decision category |
| `metadata.institution` | string | Related institution |
| `metadata.source_url` | string | BDDK source URL |

**`documents`** — Chunked regulatory documents (yonetmelikler, tebligler, genelgeler).

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Chunk ID (`{doc_id}_chunk_{n}`) |
| `document` | string | Chunk text |
| `metadata.doc_id` | string | Parent document ID |
| `metadata.title` | string | Document title |
| `metadata.doc_type` | string | yonetmelik, teblig, genelge, etc. |
| `metadata.publish_date` | string | Publication date |
| `metadata.chunk_index` | int | Position in document |
| `metadata.total_chunks` | int | Total chunks for parent doc |
| `metadata.source_url` | string | BDDK source URL |

### Why Two Collections

Decisions and documents have different metadata schemas and search patterns. Decisions are searched by institution/date/keyword. Documents are searched by topic/regulation type. Separate collections keep queries clean and avoid filtering overhead.

### Embedding Function

A custom `ChromaEmbeddingFunction` class in `embeddings.py` that wraps `sentence-transformers` with `intfloat/multilingual-e5-base`. ChromaDB accepts custom embedding functions at collection creation — the model loads once at startup and is reused across all operations.

### Store Operations

`store.py` exposes these functions:

- `init_store(config)` — Create ChromaDB persistent client and get/create both collections with the custom embedding function.
- `add_decisions(decisions)` — Upsert decisions into the `decisions` collection.
- `add_document_chunks(chunks)` — Upsert chunks into the `documents` collection.
- `search(collection, query, n, filters)` — Semantic search with optional metadata `where` filters. Returns results with relevance scores.
- `get(collection, ids)` — Direct retrieval by ID.
- `count(collection)` — Return number of records in a collection.

## MCP Tools (Core Set)

Six tools in the initial release:

### Search Tools

**`search_decisions`**
Search BDDK decisions by query text.
- **Inputs:** `query` (str, required), `date_from` (str, optional), `date_to` (str, optional), `institution` (str, optional), `category` (str, optional), `limit` (int, default 10)
- **Returns:** List of decisions with text, metadata, and relevance scores.
- **Implementation:** Calls `store.search()` on `decisions` collection with metadata `where` filters built from optional inputs.

**`search_documents`**
Search regulatory documents by query text.
- **Inputs:** `query` (str, required), `doc_type` (str, optional), `limit` (int, default 10)
- **Returns:** List of matching chunks with parent document context and relevance scores.
- **Implementation:** Calls `store.search()` on `documents` collection. Groups results by `doc_id` for presentation.

### Retrieval Tools

**`get_decision`**
Retrieve a single decision by ID.
- **Inputs:** `decision_id` (str, required)
- **Returns:** Full decision text and all metadata.

**`get_document`**
Retrieve a full document by ID.
- **Inputs:** `doc_id` (str, required)
- **Returns:** All chunks reassembled in order (sorted by `chunk_index`), plus document metadata.

### Sync Tools

**`sync_decisions`**
Scrape latest decisions from BDDK website and upsert into ChromaDB.
- **Inputs:** `pages` (int, default 1) — number of listing pages to scrape.
- **Returns:** Count of new/updated decisions.

**`sync_documents`**
Download and process new/updated regulatory documents from BDDK.
- **Inputs:** `doc_type` (str, optional) — limit to specific document type.
- **Returns:** Count of new/updated documents.

### Excluded from Core (Future)

- Bulletin/statistical data tools (analytics, trends, comparisons)
- Admin tools (health check, metrics, cache management)
- Regulatory digest / update checking

### Key Design Decision

No separate keyword vs semantic vs hybrid search modes. Every search is semantic via ChromaDB embeddings, with metadata filters layered on top via `where` clauses. This is simpler for the caller and ChromaDB handles the vector math internally.

## Scraper

`scraper.py` handles all BDDK website interaction:

- **`scrape_decisions(page, date_from, date_to)`** — Fetches decision listings from bddk.org.tr. Parses HTML tables, extracts decision metadata and text. Returns list of decision dicts.
- **`scrape_documents(doc_type)`** — Fetches regulatory document listings. Downloads PDF/HTML content, extracts text. Returns raw document content.

**Rate limiting:** 1-second delay between HTTP requests via `httpx`.

**Chunking:** Done at ingestion time in `store.py`, not in the scraper. `chunk_text(text, chunk_size=512, overlap=64)` splits text into overlapping chunks at whitespace boundaries to avoid cutting mid-sentence.

## Seed Loading

`seed.py` handles initial data population:

- **`load_seed_data(seed_dir, store)`** — Reads `decision_cache.json`, `documents.json`, and `chunks.json` from the seed directory. Transforms them into the format expected by store operations. Uses upsert behavior to skip existing records.
- Called automatically at server startup if collections are empty.

## Configuration

`config.py` — Dataclass with environment variable overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `BDDK_CHROMA_PATH` | `./chroma_data` | ChromaDB storage directory |
| `BDDK_EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Embedding model name |
| `BDDK_SEED_DIR` | `./seed_data` | Seed data directory |
| `BDDK_CHUNK_SIZE` | `512` | Characters per chunk |
| `BDDK_CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `BDDK_SCRAPE_DELAY` | `1.0` | Seconds between scrape requests |
| `MCP_TRANSPORT` | `stdio` | `stdio` or `streamable-http` |
| `MCP_HOST` | `0.0.0.0` | HTTP server host |
| `MCP_PORT` | `8000` | HTTP server port |

## Server Entry Point

`server.py` startup flow:

1. Load config from environment
2. Initialize embedding model (once)
3. Initialize ChromaDB persistent client at `BDDK_CHROMA_PATH`
4. Create/get both collections with custom embedding function
5. If collections are empty, run `load_seed_data()`
6. Register 6 MCP tools via FastMCP decorators
7. Start server with configured transport

No dependency injection, no middleware, no lifecycle hooks. Sequential initialization then serve.

## Testing Strategy

```
tests/
  conftest.py          # Shared fixtures (in-memory ChromaDB, sample data)
  test_store.py        # ChromaDB operations (add, search, get, upsert)
  test_embeddings.py   # Embedding function produces correct dimensions
  test_scraper.py      # Scraper parses HTML correctly (mocked HTTP)
  test_seed.py         # Seed loading works, skips duplicates
  test_tools.py        # MCP tools return correct responses
```

**Fixtures:** `conftest.py` creates an in-memory ChromaDB client with a small set of sample decisions and documents (5-10 decisions, 2-3 documents in `tests/fixtures/`).

**Mocking policy:**
- HTTP calls in scraper tests: mocked (no network dependency in tests)
- ChromaDB: never mocked — use real in-memory instance
- Embedding model: real model loaded in tests (cached after first run)
