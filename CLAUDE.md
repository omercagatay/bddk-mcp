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

Two-layer pattern: each module under `bddk_mcp/tools/` is a thin MCP wrapper that calls into an engine module in `bddk_mcp/`. Edit the engine for logic; edit the tool for tool-shape (args, formatting, grounding text).

- **Entry point**: `server.py` (root shim) → `bddk_mcp/server.py` — FastMCP server with grounding rules
- **MCP tool wrappers** (`bddk_mcp/tools/`, registered via `register(mcp, deps)`):
  - `search.py` → `bddk_mcp/ingest/client.py` + `bddk_mcp/store/vector_store.py` (keyword, semantic, hybrid search)
  - `documents.py` → `bddk_mcp/store/doc_store.py` (document retrieval and management)
  - `bulletin.py` → `bddk_mcp/ingest/data_sources.py` (weekly/monthly statistical bulletins)
  - `analytics.py` → `bddk_mcp/observability/analytics.py` (engine; trend/comparison)
  - `sync.py` → `bddk_mcp/ingest/doc_sync.py` (document download, OCR, chunking)
  - `admin.py` (database health, stats, cache management)
- **Engine modules** (inside `bddk_mcp/` subpackages):
  - `ingest/client.py` — BDDK website scraper (httpx, BeautifulSoup)
  - `store/doc_store.py` — PostgreSQL document storage with FTS
  - `store/vector_store.py` — pgvector semantic search
  - `ingest/doc_sync.py` — document download → OCR → chunking pipeline
  - `ingest/data_sources.py` — bulletin data scrapers
  - `observability/analytics.py` — trend/comparison analytics engine
  - `ocr/base.py` + `ocr/chandra.py` — pluggable OCR (chandra2 primary, requires `gpu` group)
- **Infrastructure** (`bddk_mcp/core/`):
  - `deps.py` — dependency injection container (`Dependencies`)
  - `config.py` — all configuration via `BDDK_*` env vars
  - `models.py` — Pydantic request/response schemas
  - `exceptions.py` — custom exception hierarchy
  - `logging_config.py` — structured logging setup
  - `utils.py` — shared HTTP retry and Turkish-text helpers
- **Quality** (`bddk_mcp/quality/`):
  - `markdown_quality.py` — markdown sanitization and quality labels
  - `quality_scan.py` — DB-side document quality scan engine
- **Seed** (`bddk_mcp/ingest/seed.py` — DB export/import for offline deployment; root `seed.py` is a thin shim)

## Conventions

- Python 3.12+ (`requires-python = ">=3.12,<3.14"`; CI matrix 3.12, 3.13), async/await throughout
- Pydantic models for all tool input/output schemas
- Turkish-aware text processing (lowercase with Turkish locale, stemming)
- Raw SQL via asyncpg — no ORM
- All SQL queries live in `bddk_mcp/store/doc_store.py` and `bddk_mcp/store/vector_store.py`
- Tests mirror source structure: `tests/test_<module>.py`
- Ruff for linting and formatting (line length 120)
- Config via environment variables prefixed with `BDDK_`

## Important Rules

- Never hardcode database credentials — use `BDDK_DATABASE_URL` env var
- Embedding model is offline-first (pre-downloaded via `BDDK_EMBEDDING_MODEL_PATH`)
- All tools must receive dependencies through the `Dependencies` DI container
- Keep `seed_data/` JSON files in sync after schema changes
- New tool modules go in `bddk_mcp/tools/` and expose `register(mcp, deps: Dependencies)` — `bddk_mcp/server.py` calls each module's `register`
- OCR is part of the doc_sync path; chandra2 needs CUDA (the `gpu` group). For DB-only work the gpu group is optional and tests with `gpu` marker are skipped by default
