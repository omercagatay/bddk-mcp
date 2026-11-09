# BDDK MCP Server — Shared Agent Instructions

This is the canonical project instruction file for Cursor, Codex, Pi, and other coding agents. Keep shared instructions here; `CLAUDE.md` only imports this file.

## Working with the owner

- The owner is not a developer and uses this project for a bank's Inspection Board (Teftiş Kurulu). Explain plans, risks, and results in plain Turkish; retain exact file paths and commands where useful.
- For substantial changes, first present a short plan, scope, verification steps, and rollback approach. Wait for approval before implementing. An approved small change does not need another approval for every edit.
- Work in small, verifiable steps. Reuse existing code and dependencies; avoid unrelated refactors, speculative abstractions, and unnecessary installations.
- Inspect `git status` before editing. Preserve existing modified and untracked work. Do not discard, reset, overwrite, or stash the owner's work without explicit approval.
- Ask for explicit approval before destructive operations, database/schema/corpus mutations, deployment, or `git push`. Approval to edit code is not approval to run migrations or import data. Do not create commits unless requested.
- Use the existing checks appropriate to the change. Never delete, weaken, or skip tests merely to obtain a pass. Report which checks actually ran, their results, and anything unverified; distinguish pre-existing failures from new ones.
- Do not read or expose secret values unnecessarily. Never send credentials, customer data, internal audit findings, or confidential bank documents to external services. Public regulatory text does not make the rest of the workspace public.
- Use applicable installed skills when available, but do not require a particular agent's plugins to work on this repository. Skills do not authorize extra scope, subagent delegation, or external actions; ask before delegating unless already authorized.

## Project scope and current facts

MCP server for Turkish banking regulatory intelligence (BDDK) — search decisions, regulations, bulletins, and statistical data. PostgreSQL + pgvector backend, offline-first embeddings, airlocked serving: retrieval tools answer only from the locally imported corpus; live BDDK/mevzuat access is confined to the bulletin, announcement, and institution-directory tools (plus the live announcement/bulletin half of `get_regulatory_digest`) and the `bddk-mcp sync` ingest path. Live acquisition uses an exact-host HTTPS allowlist for approved BDDK and mevzuat.gov.tr hosts, enforced by `core/outbound_http.py` and, for document streaming, the equivalent bounded path in `ingest/doc_sync.py`.

- Start with [Current Repository Status](docs/STATUS.md) for current repository facts and maturity limits, then [Repository Structure](docs/REPOSITORY_STRUCTURE.md) for the module map. Confirm change-sensitive facts against code and executable contracts; local work may not yet be committed or validated.
- The signed corpus-release workflow, multi-identity database roles, operator job manager, admin console, telemetry, benchmark harness, the legal-status/relations layer (`resolve_regulation_status`, `get_amendment_chain`, `get_cross_references`, Citation v1) and the bank-deployment tooling were removed on 2026-11-08/09 and remain in Git history (tag `archive/pre-simplify-2026-11-09`). Do not reintroduce them piecemeal; restoring any of them is a reviewed decision.
- This is engineering-beta software. Source availability, extraction quality, and legal currentness/applicability are separate claims. Do not describe technical verification as legal approval or bank production acceptance.
- The intended direction is multi-source regulatory research for the Inspection Board, including SPK. SPK support is not established by these instructions: do not assume it exists or widen the outbound allowlist without an approved implementation and tests.

## Commands

These are command references, not permission to mutate an environment. Confirm the target database and obtain approval for lifecycle operations.

```bash
docker compose up -d db                    # PostgreSQL + pgvector; also creates the bddk_test pytest database
docker compose up --build -d bddk-init     # One-shot: migrate + bddk-seed import against the Compose database
uv sync --dev                              # Install runtime + dev dependencies
uv sync --group gpu                        # Add CUDA torch + chandra-ocr (for the doc_sync OCR path)
uv run python server.py                    # Run MCP server (root shim; needs db up + BDDK_DATABASE_URL)
uv run bddk-mcp serve                      # Same, via the packaged CLI (also: migrate, sync)
uv run bddk-mcp migrate                    # Create/upgrade the PostgreSQL schema (approval required)
uv run bddk-mcp sync                       # Download/extract documents listed in the decision cache (approval required)
uv run bddk-seed import                    # Seed DB from seed_data/ (approval required; root seed.py is a shim)
uv run bddk-seed export                    # Export DB to seed_data/ (overwrites local corpus artifacts)
uv run pytest tests/ -m "not postgres and not gpu" -v  # DB-less unit run (matches CI unit job)
BDDK_REQUIRE_TEST_DATABASE=1 uv run pytest tests/ -m "postgres and not gpu" -v  # Dedicated test DB only; fails loudly if absent
uv run pytest tests/test_client.py -v      # Run single test file
uv run ruff check .                        # Lint
uv run ruff format --check .               # Non-mutating format check
uv run ruff format .                       # Format (prefer changed paths to avoid unrelated edits)
uv run python scripts/check_repository_hygiene.py  # Repository surface and documentation links
```

## Code knowledge graph (graphify, optional)

[graphify](https://github.com/Graphify-Labs/graphify) builds a navigable code graph for exploring dependencies and callers. It is an optional developer aid in its own `graph` dependency group; CI and the serving image do not install it.

```bash
uv sync --dev --group graph                  # Install graphify locally
uv run graphify update .                     # Build/refresh graphify-out/ from code (AST only, no LLM, no network)
uv run graphify query "<question>"           # BFS traversal over graphify-out/graph.json
uv run graphify explain "VectorStore"        # A node and its neighbours
uv run graphify affected "<symbol>"          # What depends on a symbol
```

Obsidian vault and agent access (both local, opt-in):

```bash
uv run --group graph python scripts/graphify_sync.py                  # Refresh graph + Obsidian vault in graphify-out/obsidian
uv run --group graph python scripts/graphify_sync.py --install-hooks  # Auto-refresh after commit/merge/checkout (local .git/hooks only)
uv run --group graph python scripts/graphify_sync.py --uninstall-hooks
uv run --group graph python -m graphify.serve graphify-out/graph.json  # Read-only stdio MCP server for an agent (e.g. Hermes)
```

- Use only the local `graphify update` path. Do not run `graphify extract`, `label`, `add`, `clone`, or the assistant `/graphify` skill's semantic step on this repository without explicit approval: they send content to an LLM backend or fetch external URLs.
- `.graphifyignore` keeps `seed_data/`, quality output and environment files out of the graph. Never widen its scope to corpus artifacts, bank documents or secrets.
- `graphify-out/` (including the `obsidian/` vault) is git-ignored local output. Never commit it or point the export at `docs/` or another tracked directory.
- The only approved automation is `scripts/graphify_sync.py`: it strips LLM provider keys from the child environment and its hooks add a marked block without overwriting existing hooks. Do not run `graphify hook install`, `graphify install`, or `graphify <platform> install` (including `hermes install`): they rewrite `AGENTS.md` and install skills that run the LLM extraction step.
- Agents connect through the read-only `graphify.serve` MCP server, configured on the owner's machine, not in this repository. Its answers (code symbols, comments, doc headings) go to whatever model that agent uses, so connect an agent only with the owner's explicit decision about that model. Its `list_prs`/`triage_prs`/`get_pr_impact` tools query GitHub and are not needed for code navigation.
- The graph is a navigation lead, not authority. It goes stale as code changes; confirm every conclusion against the source, callers and tests, and refresh it with `graphify update .` after large changes.

## Architecture

Two-layer pattern: modules under `bddk_mcp/tools/` are thin MCP wrappers over engine modules in `bddk_mcp/` subpackages. Edit the engine for logic; edit the tool for tool-shape (args, formatting, grounding text).

- **Entry points**: root `server.py` (shim) → `bddk_mcp/server.py` (app wiring, lifespan, HTTP security, health routes) on `bddk_mcp/mcp_server.py` (`BddkFastMCP` — privacy-safe tool errors). `bddk_mcp/cli.py` backs the packaged `bddk-mcp` CLI (`serve`, `migrate`, `sync`); `bddk_mcp/ingest/seed.py` backs `bddk-seed` (`import`, `export`).
- **Tool registration**: `bddk_mcp/tools/registry.py` owns the reviewed tool surface — `PUBLIC_TOOL_NAMES`, per-tool MCP risk annotations, `extra='forbid'` argument contracts. `register_tools` calls each tool module's `register(mcp, deps)`; `assert_tool_surface` fails startup on any drift from the reviewed name list.
- **MCP tool wrappers** (`bddk_mcp/tools/`): `search.py`, `documents.py` (incl. formula-aware extraction warnings), `sections.py` (exact sections, section search, character-exact `quotation` check), `bulletin.py`, `analytics.py`. Shared plumbing: `structured_outputs.py`, `errors.py`, `contract_types.py`, `tool_logging.py`.
- **Engines**:
  - `ingest/` — `client.py` (BDDK scraper: httpx, BeautifulSoup), `html_extractor.py`, `doc_sync.py` (download → OCR → chunking), `data_sources.py` (bulletins), `seed.py` (DB export/import; root `seed.py` is a shim)
  - `store/` — `doc_store.py` (documents + FTS), `vector_store.py` (pgvector), `section_index.py` (structural parser for Turkish legal Markdown), `legal_ref.py`, `bulk_write.py`
  - `quality/` — `markdown_quality.py`, `quality_scan.py`, `quality_failures.yml` (reviewed extraction-failure registry)
  - `ocr/` — pluggable backends (`base.py`, `chandra.py` primary; requires `gpu` group)
  - `observability/` — `analytics.py` (trend/comparison engine)
- **Platform**: `migrations/` (versioned schema modules + `runner.py`; the `regulatory_*` tables of v0004/v0006/v0009 stay in the ledger but nothing reads them), `db_lifecycle.py` (migrate + SELECT-only readiness), `db_transport.py` (DSN TLS contract), `db_compatibility.py` (PostgreSQL 17 only), `http_security.py`, `transport_tls.py`, `corpus_coordination.py` (advisory lock keys), `quotations.py` (character-exact quotation check)
- **Infrastructure** (`bddk_mcp/core/`): `deps.py` (DI container `Dependencies`), `config.py` (core config via `BDDK_*` env vars), `models.py`, `exceptions.py`, `logging_config.py`, `outbound_http.py`

## Conventions

- Python 3.12+ (`requires-python = ">=3.12,<3.14"`; CI matrix 3.12, 3.13), async/await throughout
- Pydantic models for all tool input/output schemas; structured tool results via `tools/structured_outputs.py`
- Turkish-aware text processing (lowercase with Turkish locale, stemming)
- Raw PostgreSQL SQL via asyncpg — no ORM. PostgreSQL schema DDL lives in `bddk_mcp/migrations/`; query SQL lives in the module that owns the table (`store/*.py`).
- Tests mirror source structure (`tests/test_<module>.py`); markers: `gpu` (skipped by default), `postgres` (needs the dedicated test database)
- Ruff for linting and formatting (line length 120)
- Application-specific configuration uses `BDDK_*` environment variables; preserve documented transport/platform variables such as `MCP_HOST` and `PORT`.

## Important Rules

- Never hardcode database credentials — use `BDDK_DATABASE_URL` (optionally `BDDK_SCHEMA_OWNER_DATABASE_URL` / `BDDK_INGESTION_DATABASE_URL` for deployments with separate logins).
- Embedding model is offline-first (pre-downloaded via `BDDK_EMBEDDING_MODEL_PATH`).
- All tools must receive dependencies through the `Dependencies` DI container.
- Adding, renaming, or removing an MCP tool requires updating `tools/registry.py` (`PUBLIC_TOOL_NAMES` + `TOOL_ANNOTATIONS`) — startup asserts the registered surface matches the reviewed list and fails on drift.
- PostgreSQL schema changes go through a new `bddk_mcp/migrations/v00NN_*.py` module; keep affected `seed_data/` artifacts in sync. Do not rewrite applied migrations.
- Extraction quality is governed: formula-unaware extraction methods get degraded-content warnings (`tools/documents.py`), known failures are registered in `quality/quality_failures.yml`, triage via `scripts/inventory_dropped_formulas.py` and `scripts/scan_document_quality.py` — read [Document Quality](docs/DOCUMENT_QUALITY.md) before changing extraction or repairing documents.
- OCR is part of the doc_sync path; chandra2 needs CUDA (the `gpu` group). For DB-only work the gpu group is optional and tests with `gpu` marker are skipped by default.
- Adding a root-level file may require updating `ALLOWED_TOP_LEVEL` in `scripts/check_repository_hygiene.py`. Do not relax this check broadly or remove unrelated local files to make it pass.
