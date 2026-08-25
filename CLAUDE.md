# BDDK MCP Server

MCP server for Turkish banking regulatory intelligence (BDDK) — search decisions, regulations, bulletins, and statistical data. PostgreSQL + pgvector backend, offline-first embeddings, airlocked serving: retrieval tools answer only from the locally published corpus release; live BDDK/mevzuat access is confined to bulletin/announcement tools and ingest/operator paths.

## Commands

```bash
docker compose up -d db                    # Start PostgreSQL + pgvector locally
uv sync --dev                              # Install runtime + dev dependencies
uv sync --group gpu                        # Add CUDA torch + chandra-ocr (for doc_sync OCR path)
uv run python server.py                    # Run MCP server (root shim; needs db up + BDDK_DATABASE_URL)
uv run bddk-mcp serve                      # Same, via the packaged CLI (also: migrate, bootstrap, admin-ui)
uv run bddk-mcp migrate                    # Create/upgrade the PostgreSQL schema (versioned migrations)
uv run python seed.py import               # Seed DB from seed_data/ (shim for bddk-seed)
uv run python seed.py export               # Export DB to seed_data/
uv run pytest tests/ -v --tb=short         # Run tests (gpu marker skipped by default)
uv run pytest tests/ -m "not postgres and not gpu" -v  # DB-less unit run (matches CI unit job)
uv run pytest tests/test_client.py -v      # Run single test file
uv run ruff check .                        # Lint
uv run ruff format .                       # Format
```

Corpus release lifecycle (operator workflow): `bddk-mcp verify-corpus`, `verify-and-stage-corpus-release`, `activate-corpus-release`, `retain-corpus-generation` — see `docs/CORPUS_GOVERNANCE.md`.

## Architecture

Two-layer pattern: modules under `bddk_mcp/tools/` are thin MCP wrappers over engine modules in `bddk_mcp/` subpackages. Edit the engine for logic; edit the tool for tool-shape (args, formatting, grounding text). Full module map: `docs/REPOSITORY_STRUCTURE.md`; design detail: `docs/ARCHITECTURE.md`.

- **Entry points**: root `server.py` (shim) → `bddk_mcp/server.py` (app wiring, lifespan, HTTP security) on `bddk_mcp/mcp_server.py` (`BddkFastMCP` — privacy-safe tool errors, active-corpus guard). `bddk_mcp/cli.py` backs the packaged `bddk-mcp` CLI.
- **Tool registration**: `bddk_mcp/tools/registry.py` owns the reviewed tool surface — PUBLIC vs OPERATOR profiles, per-tool MCP risk annotations, `extra='forbid'` argument contracts. `register_tool_profile` calls each tool module's `register(mcp, deps)`; `assert_tool_profile` fails startup on any drift from the reviewed name lists.
- **MCP tool wrappers** (`bddk_mcp/tools/`): `search.py`, `documents.py` (incl. formula-aware extraction warnings), `sections.py`, `legal_status.py`, `graph.py` (amendment chains, cross-references), `bulletin.py`, `analytics.py`; operator profile adds `sync.py` + `admin.py`. Shared plumbing: `structured_outputs.py`, `errors.py`, `contract_types.py`, `tool_logging.py`.
- **Engines**:
  - `ingest/` — `client.py` (BDDK scraper: httpx, BeautifulSoup), `html_extractor.py`, `doc_sync.py` (download → OCR → chunking), `backfill.py`, `data_sources.py` (bulletins), `seed.py` (DB export/import; root `seed.py` is a shim)
  - `store/` — `doc_store.py` (documents + FTS), `vector_store.py` (pgvector), `section_index.py` (structural parser for Turkish legal Markdown), `legal_ref.py`, `bulk_write.py`
  - `regulatory/` — abstention-first legal versions/status resolver and the amendment/cross-reference relations graph
  - `quality/` — `markdown_quality.py`, `quality_scan.py`, `quality_failures.yml` (reviewed extraction-failure registry)
  - `ocr/` — pluggable backends (`base.py`, `chandra.py` primary; requires `gpu` group)
  - `observability/` — `analytics.py` (trend/comparison engine), `telemetry.py`, `metrics.py`
- **Corpus governance** (top-level `bddk_mcp/` modules): `corpus_manifest.py`, `corpus_generations.py`, `corpus_publication.py`, `corpus_serving.py` (fail-closed release-epoch guard around local-corpus reads), `catalog_integrity.py`, `citations.py` (versioned, reconstructable citations), `resources.py` (MCP resources)
- **Platform**: `migrations/` (versioned schema modules + `runner.py`), `jobs/` (Postgres-backed operator job manager), `admin/` (loopback Starlette admin console), `operations/recovery.py`, `db_identity.py` / `db_lifecycle.py` / `db_transport.py` / `db_compatibility.py`, `http_security.py`, `transport_tls.py`
- **Infrastructure** (`bddk_mcp/core/`): `deps.py` (DI container `Dependencies`), `config.py` (all config via `BDDK_*` env vars), `models.py`, `exceptions.py`, `logging_config.py`, `outbound_http.py`, `utils.py`

## Conventions

- Python 3.12+ (`requires-python = ">=3.12,<3.14"`; CI matrix 3.12, 3.13), async/await throughout
- Pydantic models for all tool input/output schemas; structured tool results via `tools/structured_outputs.py`
- Turkish-aware text processing (lowercase with Turkish locale, stemming)
- Raw SQL via asyncpg — no ORM. Schema DDL lives only in `bddk_mcp/migrations/`; query SQL lives in the module that owns the table (`store/*.py`, `regulatory/repository.py` + `status_repository.py`, `jobs/postgres.py`)
- Tests mirror source structure (`tests/test_<module>.py`); markers: `gpu` (skipped by default), `postgres` (needs the compose db)
- Ruff for linting and formatting (line length 120)
- Config via environment variables prefixed with `BDDK_`

## Important Rules

- Never hardcode database credentials — use `BDDK_DATABASE_URL` env var
- Embedding model is offline-first (pre-downloaded via `BDDK_EMBEDDING_MODEL_PATH`)
- All tools must receive dependencies through the `Dependencies` DI container
- Adding, renaming, or removing an MCP tool requires updating `tools/registry.py` (PUBLIC/OPERATOR name tuples + `TOOL_ANNOTATIONS`) — startup asserts the registered surface matches the reviewed profile and fails on drift
- Schema changes go through a new `bddk_mcp/migrations/v00NN_*.py` module; keep `seed_data/` JSON files in sync afterwards
- Extraction quality is governed: formula-unaware extraction methods get degraded-content warnings (`tools/documents.py`), known failures are registered in `quality/quality_failures.yml`, triage via `scripts/inventory_dropped_formulas.py` and `scripts/scan_document_quality.py` — see `docs/DOCUMENT_QUALITY.md` before changing extraction or repairing documents
- OCR is part of the doc_sync path; chandra2 needs CUDA (the `gpu` group). For DB-only work the gpu group is optional and tests with `gpu` marker are skipped by default
