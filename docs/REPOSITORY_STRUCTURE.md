# Repository Structure

BDDK MCP separates application code, corpus data, maintainer scripts, deployment assets, tests, and documentation. This map describes the current `main` layout and the placement rules for new work.

## Top-level map

| Path | Purpose |
|---|---|
| `bddk_mcp/` | Installable Python package and all supported runtime behavior. |
| `tests/` | Unit, protocol, packaging, and policy contracts, all over a temporary SQLite corpus file; fixtures under `tests/fixtures/`. |
| `scripts/` | Maintainer utilities (document repair, quality scans, section reindex; each takes `--corpus PATH`) and the repository hygiene check. |
| `seed_data/` | Reviewed corpus source: `documents.json`, `decision_cache.json` (turned into the corpus file by `bddk-mcp build`, written back by `bddk-mcp export`); `corpus_scope.yml` is a retained scope declaration the runtime no longer reads. |
| `corpus/` | Built corpus file (`bddk.sqlite`), git-ignored. Never hand-edit it; rebuild it from `seed_data/`. |
| `deploy/` | Open WebUI integration assets. |
| `docs/` | Durable documentation and its index. |
| `.github/` | CI, dependency automation, ownership, issue forms, and PR policy. |

## Root files

The root is reserved for project entry points and tools that conventionally live there:

| Path | Purpose |
|---|---|
| `README.md`, `README.en.md` | Turkish and English landing pages for bank users. Operator procedure stays in `docs/`. |
| `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`, `LICENSE` | Repository governance and release history. |
| `pyproject.toml`, `uv.lock`, `MANIFEST.in` | Python metadata, dependency lock, and distribution policy. |
| `server.py` | Compatibility shim for `python server.py` and `mcp run server.py`; implementation belongs under `bddk_mcp/`. |
| `Dockerfile`, `docker-compose.yml` | Container build (downloads the embedding model and runs `bddk-mcp build`, so the image carries the corpus file) and the single-service local Compose topology. |
| `railway.toml` | Railway development/preview deployment manifest (not a bank path). |
| `.env.example`, `.mcp.json` | Sanitized local configuration examples. |

Do not place implementation modules, generated reports, scratch files, one-off plans, or environment-specific secrets in the repository root.

## Python package

| Path | Responsibility |
|---|---|
| `bddk_mcp/core/` | Settings, shared models, dependency wiring, logging, and outbound HTTP controls. |
| `bddk_mcp/tools/` | Thin MCP handlers, schemas, registry, errors, and structured output adapters. |
| `bddk_mcp/store/` | The SQLite corpus file (`corpus_db.py`: schema, read-only serving handle, readiness), documents, versions, sections and catalog (`doc_store.py`), chunks, embeddings and hybrid search (`vector_store.py`), the section parser and legal references. |
| `bddk_mcp/ingest/` | Source discovery, extraction, synchronization, and corpus build/export against `seed_data/`. |
| `bddk_mcp/ocr/` | OCR interfaces and optional GPU implementation. |
| `bddk_mcp/observability/` | Bulletin trend/comparison analytics. |
| `bddk_mcp/quality/` | Document quality rules and their packaged configuration. |
| Package-root modules | `server.py`, `cli.py`, `mcp_server.py`, `http_security.py`, `transport_tls.py`, `quotations.py`. |

New code should go into the narrowest existing subsystem. Create another package only when it establishes a durable responsibility boundary, not merely to reduce file count.

## Tests and generated artifacts

- Tests normally mirror the source area as `tests/test_<module>.py`. Shared fixtures belong in `tests/conftest.py` or `tests/fixtures/`.
- GPU requirements must use the registered `gpu` marker. Storage tests use the temporary corpus fixtures in `tests/conftest.py`; nothing may depend on a shared service or on an existing `corpus/bddk.sqlite`.
- `seed_data/*.json` and `uv.lock` are marked generated for GitHub presentation. They remain reviewable and must be regenerated intentionally.
- Binary fixtures are explicitly marked binary in `.gitattributes`.
- Local caches, environments, the built corpus file (`corpus/`), quality reports, and scratch analysis are ignored by `.gitignore`.

## Documentation placement

- Current user, operator, and governance guidance belongs in `docs/` and must be linked from [the documentation index](README.md).
- Component-specific instructions stay beside the component, such as `deploy/open-webui/README.md`.
- Historical reviews must name their review date and commit and must not present snapshot findings as current guarantees.

## Branch lifecycle

`main` is the only long-lived branch. Work happens on a short-lived topic branch tied to one pull request. Required checks and review conversations protect `main`; GitHub deletes a topic branch after merge. Closed, superseded, and fully merged branches should not be retained as an archive—pull requests and commit history provide that record.
