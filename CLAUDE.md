# BDDK MCP Server

MCP server for Turkish banking regulatory intelligence (BDDK) — search decisions, regulations, bulletins, and statistical data. PostgreSQL + pgvector backend, offline-first embeddings.

## Commands

```bash
docker compose up -d db                    # Start PostgreSQL + pgvector locally
uv sync --dev                              # Install runtime + dev dependencies
uv sync --group gpu                        # Add CUDA torch + chandra-ocr (for doc_sync OCR path)
uv run python server.py                    # Run MCP server (needs db up + BDDK_DATABASE_URL)
uv run python seed.py import               # Seed DB from seed_data/
uv run python seed.py export               # Export DB to seed_data/
uv run pytest tests/ -v --tb=short         # Run all tests (skips gpu marker)
uv run pytest tests/test_client.py -v      # Run single test file
uv run ruff check .                        # Lint
uv run ruff format .                       # Format
```

## Architecture

Two-layer pattern: each module under `tools/` is a thin MCP wrapper that calls into a top-level engine module. Edit the engine for logic; edit the tool for tool-shape (args, formatting, grounding text).

- **Entry point**: `server.py` — FastMCP server with grounding rules
- **MCP tool wrappers** (`tools/`, registered via `register(mcp, deps)`):
  - `search.py` → `client.py` + `vector_store.py` (keyword, semantic, hybrid search)
  - `documents.py` → `doc_store.py` (document retrieval and management)
  - `bulletin.py` → `data_sources.py` (weekly/monthly statistical bulletins)
  - `analytics.py` → `analytics.py` (top-level engine; trend/comparison)
  - `sync.py` → `doc_sync.py` (document download, OCR, chunking)
  - `admin.py` (database health, stats, cache management)
- **Engine modules** (top-level):
  - `client.py` — BDDK website scraper (httpx, BeautifulSoup)
  - `doc_store.py` — PostgreSQL document storage with FTS
  - `vector_store.py` — pgvector semantic search
  - `doc_sync.py` — document download → OCR → chunking pipeline
  - `data_sources.py` — bulletin data scrapers
  - `analytics.py` — trend/comparison analytics engine
  - `ocr_backends.py` + `ocr_backends_chandra.py` — pluggable OCR (chandra2 primary, requires `gpu` group)
- **Infrastructure**:
  - `deps.py` — dependency injection container (`Dependencies`)
  - `config.py` — all configuration via `BDDK_*` env vars
  - `models.py` — Pydantic request/response schemas
  - `exceptions.py` — custom exception hierarchy
  - `seed.py` — DB export/import for offline deployment

## Conventions

- Python 3.11+ (`requires-python = ">=3.11,<3.14"`; CI matrix 3.11, 3.12, 3.13), async/await throughout
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
- New tool modules go in `tools/` and expose `register(mcp, deps: Dependencies)` — `server.py` calls each module's `register`
- OCR is part of the doc_sync path; chandra2 needs CUDA (the `gpu` group). For DB-only work the gpu group is optional and tests with `gpu` marker are skipped by default
