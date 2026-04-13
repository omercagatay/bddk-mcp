# BDDK MCP Server

MCP server for Turkish banking regulatory intelligence (BDDK) — search decisions, regulations, bulletins, and statistical data. PostgreSQL + pgvector backend, offline-first embeddings.

## Commands

```bash
uv sync --dev                              # Install dependencies
uv run python server.py                    # Run MCP server
uv run python seed.py import               # Seed DB from seed_data/
uv run python seed.py export               # Export DB to seed_data/
uv run pytest tests/ -v --tb=short         # Run all tests
uv run pytest tests/test_client.py -v      # Run single test file
uv run ruff check .                        # Lint
uv run ruff format .                       # Format
```

## Architecture

- **Entry point**: `server.py` — FastMCP server with grounding rules
- **Tools**: `tools/` directory (6 modules, 21 tools)
  - `search.py` — decision search (keyword, semantic, hybrid)
  - `documents.py` — document retrieval and management
  - `bulletin.py` — weekly/monthly statistical bulletins
  - `analytics.py` — trend analysis and comparisons
  - `sync.py` — document synchronization from BDDK website
  - `admin.py` — database health, stats, cache management
- **Core logic**:
  - `client.py` — BDDK website scraper (httpx, BeautifulSoup)
  - `doc_store.py` — PostgreSQL document storage with FTS
  - `vector_store.py` — pgvector semantic search
  - `doc_sync.py` — document download and chunking pipeline
  - `data_sources.py` — bulletin data scrapers
  - `analytics.py` — trend/comparison analytics engine
- **Infrastructure**:
  - `deps.py` — dependency injection container
  - `config.py` — all configuration via `BDDK_*` env vars
  - `models.py` — Pydantic request/response schemas
  - `exceptions.py` — custom exception hierarchy
  - `seed.py` — DB export/import for offline deployment

## Conventions

- Python 3.12+ (CI tests 3.11, 3.12, 3.13), async/await throughout
- Pydantic models for all tool input/output schemas
- Turkish-aware text processing (lowercase with Turkish locale, stemming)
- Raw SQL via asyncpg — no ORM
- All SQL queries live in `doc_store.py` and `vector_store.py`
- Tests mirror source structure: `tests/test_<module>.py`
- Ruff for linting and formatting (line length 120)
- Config via environment variables prefixed with `BDDK_`

## Important Rules

- Never hardcode database credentials — use `BDDK_DATABASE_URL` env var
- Embedding model is offline-first (pre-downloaded via `BDDK_EMBEDDING_MODEL_PATH`)
- All tools must receive dependencies through the `Dependencies` DI container
- Keep `seed_data/` JSON files in sync after schema changes
- All tool functions go in `tools/` modules, registered via `register_tools(mcp, deps)`
