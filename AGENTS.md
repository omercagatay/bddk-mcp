# BDDK MCP Server — Shared Agent Instructions

This is the canonical project instruction file for Cursor, Codex, Pi, and other coding agents. Keep shared instructions here; `CLAUDE.md` only imports this file.

## Working with the owner

- The owner is not a developer and uses this project for a bank's Inspection Board (Teftiş Kurulu). Explain plans, risks, and results in plain Turkish; retain exact file paths and commands where useful.
- For substantial changes, first present a short plan, scope, verification steps, and rollback approach. Wait for approval before implementing. An approved small change does not need another approval for every edit.
- Work in small, verifiable steps. Reuse existing code and dependencies; avoid unrelated refactors, speculative abstractions, and unnecessary installations.
- Inspect `git status` before editing. Preserve existing modified and untracked work. Do not discard, reset, overwrite, or stash the owner's work without explicit approval.
- Ask for explicit approval before destructive operations, database/schema/corpus mutations, deployment, or `git push`. Approval to edit code is not approval to run migrations, import data, or activate a release. Do not create commits unless requested.
- Use the existing checks appropriate to the change. Never delete, weaken, or skip tests merely to obtain a pass. Report which checks actually ran, their results, and anything unverified; distinguish pre-existing failures from new ones.
- Do not read or expose secret values unnecessarily. Never send credentials, private signing keys, customer data, internal audit findings, or confidential bank documents to external services. Public regulatory text does not make the rest of the workspace public.
- Use applicable installed skills when available, but do not require a particular agent's plugins to work on this repository. Skills do not authorize extra scope, subagent delegation, or external actions; ask before delegating unless already authorized.

## Project scope and current facts

MCP server for Turkish banking regulatory intelligence (BDDK) — search decisions, regulations, bulletins, and statistical data. PostgreSQL + pgvector backend, offline-first embeddings, airlocked serving: retrieval tools answer only from the locally published corpus release; live BDDK/mevzuat access is confined to the bulletin, announcement, and institution-directory tools (plus the live announcement/bulletin half of `get_regulatory_digest`) and ingest/operator paths. Live acquisition uses an exact-host HTTPS allowlist for approved BDDK and mevzuat.gov.tr hosts, enforced by `core/outbound_http.py` and, for document streaming, the equivalent bounded path in `ingest/doc_sync.py`.

- Start with [Current Repository Status](docs/STATUS.md) for current repository facts and maturity limits, then [Repository Structure](docs/REPOSITORY_STRUCTURE.md) for the module map. Confirm change-sensitive facts against code and executable contracts; local work may not yet be committed or validated.
- [Architecture](docs/ARCHITECTURE.md) includes historical checkpoints. Do not treat its dated counts or capabilities as current when they differ from `docs/STATUS.md` and code.
- This is engineering-beta software. Source availability, extraction quality, a signature, and legal currentness/applicability are separate claims. Do not describe technical verification as legal approval or bank production acceptance.
- The intended direction is multi-source regulatory research for the Inspection Board, including SPK. SPK support is not established by these instructions: do not assume it exists or widen the outbound allowlist without an approved implementation and tests.

## Commands

These are command references, not permission to mutate an environment. Confirm the target database and obtain approval for lifecycle operations. Never substitute real IdP values into documentation.

```bash
# docker compose parses the operator service's required variables even for
# db-only startup; export placeholders once per shell (never real IdP values):
export BDDK_JWT_JWKS_URL=https://placeholder.invalid/jwks BDDK_JWT_ISSUER=https://placeholder.invalid
docker compose up -d bddk-test-db          # PostgreSQL + pgvector + the bddk_test pytest fixture
docker compose up -d db                    # DB only (postgres-marked tests then skip; prefer bddk-test-db)
uv sync --dev                              # Install runtime + dev dependencies
uv sync --group gpu                        # Add CUDA torch + chandra-ocr (for doc_sync OCR path)
uv run python server.py                    # Run MCP server (root shim; needs db up + BDDK_DATABASE_URL)
uv run bddk-mcp serve                      # Same, via the packaged CLI (also: migrate, bootstrap, admin-ui)
uv run bddk-mcp migrate                    # Create/upgrade the PostgreSQL schema (approval required)
uv run python seed.py import               # Seed DB from seed_data/ (approval required; shim for bddk-seed)
uv run python seed.py export               # Export DB to seed_data/ (overwrites local corpus artifacts)
uv run pytest tests/ -m "not postgres and not gpu" -v  # DB-less unit run (matches CI unit job)
BDDK_REQUIRE_TEST_DATABASE=1 uv run pytest tests/ -m "postgres and not gpu" -v  # Dedicated test DB only; fails loudly if absent
uv run pytest tests/test_client.py -v      # Run single test file
uv run ruff check .                        # Lint
uv run ruff format --check .               # Non-mutating format check
uv run ruff format .                       # Format (prefer changed paths to avoid unrelated edits)
uv run python scripts/check_repository_hygiene.py  # Repository surface and documentation links
```

Corpus release lifecycle (operator workflow, explicit approval required): `bddk-mcp verify-corpus`, `verify-and-stage-corpus-release`, `activate-corpus-release`, `retain-corpus-generation` — see [Corpus Governance](docs/CORPUS_GOVERNANCE.md).

## Architecture

Two-layer pattern: modules under `bddk_mcp/tools/` are thin MCP wrappers over engine modules in `bddk_mcp/` subpackages. Edit the engine for logic; edit the tool for tool-shape (args, formatting, grounding text).

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
- **Admin console** (`bddk_mcp/admin/`, `bddk-mcp admin-ui`): defaults to loopback. Remote binds require explicit opt-in and operator-scoped JWT authentication; unauthenticated remote access is rejected. PostgreSQL access remains read-only. Optional SQLite editorial drafts (`drafts.py`), CSRF-protected forms (`csrf.py`), and server-held Ed25519 draft signing are separate from canonical corpus publication. Saving or signing a draft must not mutate, publish, or establish legal approval of the corpus. Keep draft storage and signing keys outside the corpus, with keys outside the draft storage directory. See [Corpus Governance](docs/CORPUS_GOVERNANCE.md#admin-editorial-drafts-and-document-signatures) and [Deployment](docs/DEPLOYMENT.md). Check Git status: working-tree implementations are not proof of merged or deployed features.
- **Platform**: `migrations/` (versioned schema modules + `runner.py`), `jobs/` (Postgres-backed operator job manager), `operations/recovery.py`, `db_identity.py` / `db_lifecycle.py` / `db_transport.py` / `db_compatibility.py`, `http_security.py`, `transport_tls.py`
- **Infrastructure** (`bddk_mcp/core/`): `deps.py` (DI container `Dependencies`), `config.py` (core config via `BDDK_*` env vars), `models.py`, `exceptions.py`, `logging_config.py`, `outbound_http.py`

## Conventions

- Python 3.12+ (`requires-python = ">=3.12,<3.14"`; CI matrix 3.12, 3.13), async/await throughout
- Pydantic models for all tool input/output schemas; structured tool results via `tools/structured_outputs.py`
- Turkish-aware text processing (lowercase with Turkish locale, stemming)
- Raw PostgreSQL SQL via asyncpg — no ORM. PostgreSQL schema DDL lives in `bddk_mcp/migrations/`; query SQL lives in the module that owns the table (`store/*.py`, `regulatory/repository.py` + `status_repository.py`, `jobs/postgres.py`). The isolated admin SQLite draft schema is owned by `bddk_mcp/admin/drafts.py`; it is not a PostgreSQL migration or corpus store.
- Tests mirror source structure (`tests/test_<module>.py`); markers: `gpu` (skipped by default), `postgres` (needs the dedicated test database)
- Ruff for linting and formatting (line length 120)
- Application-specific configuration uses `BDDK_*` environment variables; preserve documented transport/platform variables such as `MCP_HOST` and `PORT`.

## Important Rules

- Never hardcode database credentials — use the documented environment variable for the relevant database identity (`BDDK_DATABASE_URL` for public serving). Preserve role separation.
- Embedding model is offline-first (pre-downloaded via `BDDK_EMBEDDING_MODEL_PATH`).
- All tools must receive dependencies through the `Dependencies` DI container.
- Adding, renaming, or removing an MCP tool requires updating `tools/registry.py` (PUBLIC/OPERATOR name tuples + `TOOL_ANNOTATIONS`) — startup asserts the registered surface matches the reviewed profile and fails on drift.
- PostgreSQL schema changes go through a new `bddk_mcp/migrations/v00NN_*.py` module; keep affected `seed_data/` artifacts and schema/catalog contracts in sync. Do not rewrite applied migrations or hand-edit signed artifacts to bypass verification.
- Extraction quality is governed: formula-unaware extraction methods get degraded-content warnings (`tools/documents.py`), known failures are registered in `quality/quality_failures.yml`, triage via `scripts/inventory_dropped_formulas.py` and `scripts/scan_document_quality.py` — read [Document Quality](docs/DOCUMENT_QUALITY.md) before changing extraction or repairing documents.
- OCR is part of the doc_sync path; chandra2 needs CUDA (the `gpu` group). For DB-only work the gpu group is optional and tests with `gpu` marker are skipped by default.
- Adding a root-level file may require updating `ALLOWED_TOP_LEVEL` in `scripts/check_repository_hygiene.py`. Do not relax this check broadly or remove unrelated local files to make it pass.
