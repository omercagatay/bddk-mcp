# Repository Structure

BDDK MCP separates application code, governed data, operational tooling, deployment assets, tests, and evidence. This map describes the current `main` layout and the placement rules for new work.

## Top-level map

| Path | Purpose |
|---|---|
| `bddk_mcp/` | Installable Python package and all supported runtime behavior. |
| `tests/` | Unit, protocol, PostgreSQL, packaging, deployment, and policy contracts. |
| `benchmark/` | Evaluation datasets, graders, trust policy, reports, and benchmark runners. |
| `scripts/` | Maintainer utilities and repository/deployment validation entry points. |
| `seed_data/` | Governed offline corpus exports, manifest, and signature. |
| `data/` | Static benchmark and evaluation data that is not runtime corpus state. |
| `deploy/` | PostgreSQL, OpenShift, Open WebUI, and trust assets. |
| `supply-chain/` | Scanner manifests, exception policy, and supply-chain documentation. |
| `docs/` | Durable documentation, architecture decisions, and retained evidence. |
| `.github/` | CI, dependency automation, ownership, issue forms, and PR policy. |

## Root files

The root is reserved for project entry points and tools that conventionally live there:

| Path | Purpose |
|---|---|
| `README.md`, `README.en.md` | Project landing page and English operational guide. |
| `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`, `LICENSE` | Repository governance and release history. |
| `pyproject.toml`, `uv.lock`, `MANIFEST.in` | Python metadata, dependency lock, and distribution policy. |
| `server.py`, `seed.py` | Compatibility shims; implementation belongs under `bddk_mcp/`. |
| `Dockerfile`, `docker-compose.yml` | Container builds and the disposable local lifecycle. |
| `railway.toml`, `Procfile` | Hosted deployment entry points. |
| `.env.example`, `.mcp.json` | Sanitized local configuration examples. |

Do not place implementation modules, generated reports, scratch files, one-off plans, or environment-specific secrets in the repository root.

## Python package

| Path | Responsibility |
|---|---|
| `bddk_mcp/core/` | Settings, shared models, dependency wiring, logging, and outbound HTTP controls. |
| `bddk_mcp/tools/` | Thin MCP handlers, schemas, registry, errors, and structured output adapters. |
| `bddk_mcp/store/` | Document, section, vector, reference, and bulk-write storage. |
| `bddk_mcp/ingest/` | Source discovery, extraction, synchronization, seed import/export, and backfill. |
| `bddk_mcp/ocr/` | OCR interfaces and optional GPU implementation. |
| `bddk_mcp/regulatory/` | Canonical legal versions, relations, graph queries, and curation repositories. |
| `bddk_mcp/jobs/` | Durable operator job models, repository, and runner. |
| `bddk_mcp/migrations/` | Append-only schema ledger and migration implementations. |
| `bddk_mcp/observability/` | Privacy-safe analytics, metrics, and telemetry. |
| `bddk_mcp/operations/` | Recovery and other explicit operational workflows. |
| `bddk_mcp/quality/` | Document quality rules and their packaged configuration. |
| Package-root modules | Cross-cutting runtime, corpus, database, citation, and transport boundaries. |

New code should go into the narrowest existing subsystem. Create another package only when it establishes a durable responsibility boundary, not merely to reduce file count.

## Tests and generated artifacts

- Tests normally mirror the source area as `tests/test_<module>.py`. Shared fixtures belong in `tests/conftest.py` or `tests/fixtures/`.
- PostgreSQL and GPU requirements must use the registered `postgres` and `gpu` markers.
- `seed_data/*.json`, `docs/evidence/*.json`, and `uv.lock` are marked generated for GitHub presentation. They remain reviewable and must be regenerated intentionally.
- Binary fixtures are explicitly marked binary in `.gitattributes`.
- Local caches, environments, benchmark results, quality reports, and scratch analysis are ignored by `.gitignore`.

## Documentation placement

- Current user, operator, architecture, and governance guidance belongs in `docs/` and must be linked from [the documentation index](README.md).
- Versioned decisions belong in `docs/decisions/`.
- Reproducible evidence belongs in `docs/evidence/` and must state its environment and limitations.
- Component-specific instructions stay beside the component, such as `benchmark/README.md` or `deploy/postgres/README.md`.
- Historical reviews must name their review date and commit and must not present snapshot findings as current guarantees.

## Branch lifecycle

`main` is the only long-lived branch. Work happens on a short-lived topic branch tied to one pull request. Required checks and review conversations protect `main`; GitHub deletes a topic branch after merge. Closed, superseded, and fully merged branches should not be retained as an archive—pull requests and commit history provide that record.
