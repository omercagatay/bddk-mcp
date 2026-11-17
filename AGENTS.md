# BDDK MCP Server — Shared Agent Instructions

This is the canonical project instruction file for Cursor, Codex, Pi, and other coding agents. Keep shared instructions here; `CLAUDE.md` imports this file and adds only the Claude Code-specific rules on subagent models and the advisor. Keep this file to what an agent cannot work out from the code; detail belongs in the linked documents.

## Working with the owner

- The owner is not a developer and uses this project for a bank's Inspection Board (Teftiş Kurulu). Explain plans, risks, and results in plain Turkish and keep English technical terms as they are (store, chunk, embedding, merge, deploy); retain exact file paths and commands where useful.
- For substantial changes, first present a short plan, scope, verification steps, and rollback approach. Wait for approval before implementing. An approved small change does not need another approval for every edit.
- Work in small, verifiable steps. Reuse existing code and dependencies; avoid unrelated refactors, speculative abstractions, and unnecessary installations.
- Inspect `git status` before editing. Preserve existing modified and untracked work. Do not discard, reset, overwrite, or stash the owner's work without explicit approval.
- Ask for explicit approval before destructive operations, corpus-file or schema mutations, deployment, or `git push`. Approval to edit code is not approval to rebuild, sync or export the corpus. Do not create commits unless requested.
- Dependabot pull requests are the one standing exception: the owner delegated them. Review each one, merge it when CI is green and the measurements for the changed packages hold, verify the deployment, and report afterwards. Ask first for a major version of a package on the HTTP or authentication boundary or of the embedding stack, a base-image change beyond a patch release, a failing or flaky CI, and anything that changes search results or needs a corpus rebuild.
- Use the existing checks appropriate to the change. Never delete, weaken, or skip tests merely to obtain a pass. Report which checks actually ran, their results, and anything unverified; distinguish pre-existing failures from new ones.
- Security review is the agent's job; no external review bot does it. Before a pull request is merged, review its diff for security impact (authentication and the HTTP boundary, outbound hosts, input validation, what reaches logs and tool errors, corpus write paths, dependencies, CI and the image) and write the result in the pull request description: what was checked, what was found and fixed, and what was not checked. A finding that is not fixed stays open for the owner's decision; do not merge over it. This also applies to Dependabot pull requests and to ones another session opened.
- Do not read or expose secret values unnecessarily. Never send credentials, customer data, internal audit findings, or confidential bank documents to external services. Public regulatory text does not make the rest of the workspace public.
- Use applicable installed skills when available, but do not require a particular agent's plugins to work on this repository. Skills do not authorize extra scope or external actions; delegation to subagents follows the next section.

## Delegation and model choice

The owner gave the orchestrating agent standing authority to delegate work to subagents and to choose the model for each job. Do not ask the owner which model to use and do not expect a prompt to say it. The condition of that freedom is maximum care: the choice is made per job, deliberately, and the orchestrator answers for every result.

- Choose by what the job needs, in three tiers:
  - **Smallest model**: purely mechanical work whose result a script or a count decides. Running an existing script or harness, counting, tabulating, converting formats, repeating a measurement. No judgement about legal text, no design decision, no code change beyond a literal instruction.
  - **Middle model**: work that needs reading and classification but not design. Scans of document text and metadata, documentation alignment, mechanical code edits against a written specification, test runs with triage.
  - **Strongest subagent model**: judgement-heavy work. Changes to retrieval, ranking, the section parser, quality policy, security boundaries or the corpus build; writing ground truth for an evaluation; root-cause diagnosis; code and security review.
- When two tiers are plausible, take the higher one. A wrong result from a cheaper model costs more than it saves, and anything an inspector could end up citing is never a job for the smallest model.
- The orchestrator's own model is for orchestration, briefs and review. The owner also wants the top tier used where it is available: give it a subagent job only when the job is judgement-heavy and its result is something an inspector could rely on (ground truth, a final or security review), and run one such agent at a time, because it draws on the same usage limit as the orchestrator.
- The orchestrator writes every brief itself: goal, files, what is already known or ruled out, hard rules (read-only or not, what must not be touched), the expected output and what done looks like.
- The orchestrator reviews every result personally before acting on it or reporting it, and verifies it with an independent check of its own (its own query, its own run, the diff read line by line). An agent's statement that it verified its own finding is not verification. Say in the report what was verified independently and what was not.
- Keep the fan-out small, give each set of files one writer at a time, and stop agents that die on a usage limit instead of letting them retry.
- A subagent has no more authority than the orchestrator: everything in "Working with the owner" (approval for push, deployment, corpus mutations, destructive operations; no secrets or bank data to external services) applies to delegated work unchanged.

## Project scope and current facts

MCP server for Turkish banking regulatory intelligence (BDDK): search decisions, regulations, bulletins, and statistical data.

- The whole corpus is one SQLite file (`BDDK_CORPUS_PATH`, default `corpus/bddk.sqlite`, git-ignored) built from `seed_data/` by `bddk-mcp build`, with offline-first embeddings.
- Serving is airlocked: `bddk-mcp serve` opens the file read-only (SQLite `mode=ro`), so a serving process cannot write the corpus, and retrieval tools answer only from it.
- Live BDDK/mevzuat access is confined to the bulletin, announcement, and institution-directory tools (plus the live announcement/bulletin half of `get_regulatory_digest`), the `bddk-mcp sync` ingest path (including `sync --add`) and the read-only `bddk-mcp check-updates`. It uses an exact-host HTTPS allowlist for approved BDDK and mevzuat.gov.tr hosts, enforced by `core/outbound_http.py` and, for document streaming, the equivalent bounded path in `ingest/doc_sync.py`.
- Start with [Current Repository Status](docs/STATUS.md) for current repository facts and maturity limits, then [Repository Structure](docs/REPOSITORY_STRUCTURE.md) for the module map. Confirm change-sensitive facts against code and executable contracts; local work may not yet be committed or validated.
- Removed and kept only in Git history (tag `archive/pre-simplify-2026-11-09`): the signed corpus-release workflow, multi-identity database roles, operator job manager, admin console, telemetry, benchmark harness, the legal-status/relations layer (`resolve_regulation_status`, `get_amendment_chain`, `get_cross_references`, Citation v1) and the bank-deployment tooling. Do not reintroduce them piecemeal; restoring any of them is a reviewed decision.
- There is no PostgreSQL backend; the single corpus file replaced it (schema migration ledger, DSN TLS contract, separate database logins, the `migrate` and seed-import commands, the `postgres` pytest marker and CI job are in Git history at tag `archive/postgres-2026-11-10`). Do not add a database service, a connection string, or an in-place schema upgrade path; a changed schema means a rebuilt corpus file.
- This is engineering-beta software. Source availability, extraction quality, and legal currentness/applicability are separate claims. Do not describe technical verification as legal approval or bank production acceptance.
- The intended direction is multi-source regulatory research for the Inspection Board, including SPK. SPK support is not established by these instructions: do not assume it exists or widen the outbound allowlist without an approved implementation and tests.

## Commands

These are command references, not permission to mutate an environment. Confirm the target corpus file (`BDDK_CORPUS_PATH` or `--corpus`) and obtain approval before `build`, `sync` or `export`. What each corpus command rewrites, reuses and refuses is in [Current Repository Status](docs/STATUS.md#upgrading-an-existing-corpus-file) and [Deployment](docs/DEPLOYMENT.md).

```bash
uv sync --dev                              # Install runtime + dev dependencies
uv sync --group gpu                        # Add CUDA torch + LightOnOCR dependencies
uv run bddk-mcp build                      # seed_data/ -> corpus file; skips when current, reuses seed_data/embeddings/; --force rebuilds and re-embeds every document (approval required)
uv run bddk-mcp export                     # Corpus file -> seed_data/ JSON + embeddings/ cache; overwrites the local seed artifacts (approval required)
uv run bddk-mcp info [--corpus PATH]       # Describe the corpus file read-only; exit 1 when not ready
uv run bddk-mcp check-updates [--corpus PATH] [--json] [--deep]  # Read-only: new, unlisted and changed documents against the live site; --deep downloads every source and compares fingerprints; exit 0 nothing, 3 something to report, 1 error, 2 usage error; reviewed findings in bddk_mcp/ingest/accepted_findings.yml do not count
uv run bddk-mcp sync                       # Download/extract the catalogued documents and re-embed them (approval required; --force, --doc-id, --concurrency, --corpus; --refresh-catalog replaces the curated catalog with everything the pages list)
uv run bddk-mcp sync --add ID [--add ID]   # The way to add documents: adds these catalog entries from the live pages and syncs only them (approval required)
uv run bddk-mcp serve                      # Run MCP server; opens the corpus file read-only (root `server.py` is the import shim)
uv run pytest tests/ -v                    # Whole suite on a temporary corpus file; needs no service (gpu tests are deselected by default; one test marked `network` loads the real tokenizer)
uv run pytest tests/test_client.py -x --lf -q  # Fast loop: one file, stop at the first failure, re-run only the last failures (all tests when none failed)
uv lock --check                            # Lockfile matches pyproject.toml (CI lint job)
uv run ruff check .                        # Lint
uv run ruff format --check .               # Non-mutating format check
uv run ruff format .                       # Format (prefer changed paths to avoid unrelated edits)
uv run python scripts/check_repository_hygiene.py  # Repository surface and documentation links
uv run --group repair python -m scripts.document_repair PDF OUTDIR  # Repair one document from its PDF into OUTDIR; writes no corpus or seed and sends nothing anywhere without --gpu-host (ssh alias); see Document Quality
docker compose up --build -d bddk          # Local container; the image build runs `bddk-mcp build`, so the service needs nothing else
```

The optional code graph (graphify) is described in [Contributing](CONTRIBUTING.md#code-knowledge-graph-graphify-optional); read it before using graphify. Two rules hold regardless: run only the local `graphify update` path and `scripts/graphify_sync.py` (the other graphify commands send content to an LLM backend, fetch external URLs or rewrite `AGENTS.md`), and never commit `graphify-out/`.

## Architecture

Two-layer pattern: modules under `bddk_mcp/tools/` are thin MCP wrappers over engine modules in `bddk_mcp/` subpackages. Edit the engine for logic; edit the tool for tool-shape (args, formatting, grounding text). The module map is in [Repository Structure](docs/REPOSITORY_STRUCTURE.md#python-package).

- `bddk_mcp/tools/registry.py` owns the reviewed tool surface (`PUBLIC_TOOL_NAMES`, per-tool MCP risk annotations, `extra='forbid'` argument contracts); `assert_tool_surface` fails startup on any drift from the reviewed name list.
- `store/corpus_db.py` holds the schema (`CREATE TABLE IF NOT EXISTS`, `SCHEMA_VERSION` in the `meta` table, no migration ledger); serving opens the file with SQLite `mode=ro`, which is the airlock.
- All tools receive their dependencies through the `Dependencies` DI container (`bddk_mcp/core/deps.py`).

## Conventions

- Async/await throughout; Pydantic models for all tool input/output schemas; structured tool results via `tools/structured_outputs.py`.
- Turkish-aware text processing (lowercase with Turkish locale, stemming; FTS5 indexes normalized shadow columns, diacritics stripped in Python).
- Plain `sqlite3` SQL, no ORM. The corpus schema DDL lives in `bddk_mcp/store/corpus_db.py`; query SQL lives in the module that owns the table (`store/*.py`). SQL runs on worker threads through `CorpusDB.run` / `CorpusDB.transaction` so the event loop stays free.
- Tests mirror source structure (`tests/test_<module>.py`). Storage tests use the temporary corpus fixtures in `tests/conftest.py`; nothing depends on a service or on an existing `corpus/bddk.sqlite`. An autouse guard fails a test that opens a non-loopback connection; markers: `gpu` (deselected by default) and `network` (exempt from the guard, only where a test genuinely needs the network).
- Application-specific configuration uses `BDDK_*` environment variables; preserve documented transport/platform variables such as `MCP_HOST` and `PORT`.

## Important Rules

- Never hardcode a corpus location: the file comes from `BDDK_CORPUS_PATH` (default `corpus/bddk.sqlite`) or the `--corpus` option. Serving opens it read-only and must never write it; the only writers are `bddk-mcp build`, `bddk-mcp sync` and the maintainer scripts under `scripts/`.
- Adding, renaming, or removing an MCP tool requires updating `tools/registry.py` (`PUBLIC_TOOL_NAMES` + `TOOL_ANNOTATIONS`); startup asserts the registered surface matches the reviewed list and fails on drift.
- The embedding model is offline-first (pre-downloaded via `BDDK_EMBEDDING_MODEL_PATH`), and every corpus file records its model and dimension in `meta`. A change needs the matching rebuild:
  - Embedding model or dimension: delete the local corpus file and run `bddk-mcp build`, then `bddk-mcp export`. Changing only a variable is not enough, and neither `build` nor `--force` can switch an existing file to another model.
  - Section parser or chunker: run `bddk-mcp build` (the build fingerprint detects it and rewrites every document), then `bddk-mcp export`.
  - Model revision (`BDDK_EMBEDDING_MODEL_REVISION`), the snapshot under `BDDK_EMBEDDING_MODEL_PATH` with the same model name, passage prefix, prefix separator or vector dtype: delete the file and run `build --force`. The embedding cache records no model revision or snapshot, so a plain build refuses these changes and says so.
  - `seed_data/embeddings/` is reused only while its manifest names the configured model and each document's chunk digest; `--force` re-embeds every document and ignores it.
  - Documents removed from the seed are not detected by `build` and need a fresh file. Metadata-only seed edits are detected through the fingerprint.
  - A non-empty `-wal` beside a replaced corpus file is refused on open: move `-wal` and `-shm` away with the old file.
  - `scripts/reindex_document_sections.py` updates sections only and leaves the chunks' section fields stale.
- Schema changes go through `bddk_mcp/store/corpus_db.py` and a rebuilt corpus file (`bddk-mcp build --force`; when a table layout changed, remove the old file first, because `CREATE TABLE IF NOT EXISTS` never alters an existing table). There is no in-place upgrade or migration ledger, and a file of another schema version must be deleted and rebuilt. Keep affected `seed_data/` artifacts in sync. The source fingerprint columns (`documents.source_fingerprint`, `source_fingerprint_at`) are carried by `seed_data/documents.json`, written by `sync` and kept by `scripts/patch_doc.py`; serving never reads them.
- Extraction quality is governed: formula-unaware extraction methods get degraded-content warnings (`tools/documents.py`), known failures are registered in `quality/quality_failures.yml`, triage via `scripts/inventory_dropped_formulas.py` and `scripts/scan_document_quality.py`. Read [Document Quality](docs/DOCUMENT_QUALITY.md) before changing extraction or repairing documents.
- A document is repaired from its PDF with `scripts/document_repair` as [Document Quality](docs/DOCUMENT_QUALITY.md#repairing-a-document-from-its-pdf) describes: accepted formulas are merged, every rejected region and every table is reviewed against the page image with the review logged, and the result is stored with `scripts/patch_doc.py` (approval required). Do not change the guards or the recommended policy without a new measurement against page images, never let a model write prose, and do not add the GPU host to the HTTP outbound allowlist (its path is ssh).
- OCR is part of the doc_sync path. The chain is pdftotext, markitdown, then LightOCR on every host: the PDF text layer is the authority, and LightOCR (CUDA, the `gpu` group) is reached only when both text-layer backends return less than `BDDK_OCR_MIN_CONTENT_LEN` characters (a scanned PDF), because whole-page model OCR changed about 2 % of the words of text-layer pages when it was measured. Do not move a model ahead of the text layer. Mevzuat documents are fetched as HTML first on every host (`BDDK_PREFER_HTML_FOR_MEVZUAT`, default `auto`). LightOCR text carries a model-OCR warning and does not count as formula-aware. For corpus-only work the gpu group is optional.
- Adding a root-level file may require updating `ALLOWED_TOP_LEVEL` in `scripts/check_repository_hygiene.py`. Do not relax this check broadly or remove unrelated local files to make it pass.
