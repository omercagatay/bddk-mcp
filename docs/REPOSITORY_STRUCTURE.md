# Repository Structure

This repository keeps runtime code, MCP tool wrappers, operator scripts, benchmarks, and generated seed data in separate areas.

## Root Files

| Path | Purpose |
|---|---|
| `server.py` | FastMCP entry point, lifecycle setup, and tool registration. |
| `config.py` | `BDDK_*` environment parsing and validation. |
| `deps.py` | Shared dependency container used by tool modules. |
| `client.py` | BDDK catalog/cache client. |
| `doc_store.py` | PostgreSQL document, version, section, and FTS storage. |
| `vector_store.py` | pgvector-backed semantic and hybrid search. |
| `doc_sync.py` | Download, extraction, OCR, chunking, and indexing pipeline. |
| `seed.py` | Import/export between PostgreSQL and `seed_data/`. |
| `quality_failures.yml` | Tracked document quality targets used by quality scripts and tests. |

## Directories

| Path | Purpose |
|---|---|
| `tools/` | MCP tool modules. These should stay thin and delegate to engine modules. |
| `ocr/` | OCR backend interfaces and Chandra OCR integration. |
| `scripts/` | Operator scripts for scans, backfills, patching, smoke tests, and validation. |
| `tests/` | Pytest suite mirroring source modules and operator workflows. |
| `benchmark/` | Tool schema fixtures, gold cases, graders, and benchmark runners. |
| `data/` | Static benchmark and evaluation datasets. |
| `seed_data/` | Generated seed exports used for offline-first deployment. |
| `docs/` | Human documentation, quality notes, reference material, and fix logs. |
| `.github/` | CI, issue templates, PR template, and repository ownership metadata. |

## Generated And Bulky Artifacts

| Path | Notes |
|---|---|
| `seed_data/*.json` | Generated database export. Review intentionally, but avoid hand-editing. |
| `docs/reference/DOCUMENTS.md` | Generated document catalog/reference listing. |
| `docs/fixes/` | Historical fix result logs and maintenance records. |

Generated files are marked in `.gitattributes` so GitHub language statistics and diffs stay focused on source code.
