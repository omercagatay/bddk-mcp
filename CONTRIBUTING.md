# Contributing to BDDK MCP

Thanks for improving BDDK MCP. This repository handles regulatory source material and deployment controls, so changes should be small enough to review and explicit about their evidence and operational impact.

## Before you start

- Use a GitHub issue for reproducible bugs, data-quality defects, and larger feature proposals.
- Report vulnerabilities through a [private security advisory](SECURITY.md), never a public issue.
- Read the [repository map](docs/REPOSITORY_STRUCTURE.md) and the area-specific documentation before changing a subsystem.
- Treat the repository as an engineering beta. It is not a source of legal advice or proof that a deployment is production-ready.

## Development setup

Requirements:

- Python 3.12 or 3.13
- [`uv`](https://docs.astral.sh/uv/)
- Docker only if you want to build and run the container image locally

There is no database service. The corpus is one SQLite file (`corpus/bddk.sqlite`, git-ignored) that `uv run bddk-mcp build` produces from `seed_data/`; the test suite never needs it, because every storage test runs on a temporary corpus file created by `tests/conftest.py`.

Install the locked development environment:

```bash
uv sync --frozen --dev
```

Run the fast local checks:

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run python scripts/check_repository_hygiene.py
uv run pytest tests/ -m "not gpu" --strict-markers -q
```

The complete CI matrix also exercises Python 3.12 and 3.13, package artifacts, and container recipes. The whole suite runs without any service; only the `gpu` marker is skipped by default.

## Branch and pull-request workflow

1. Start from current `main` and create a short-lived branch named `type/short-description`, such as `fix/section-pagination` or `docs/deployment-example`.
2. Keep one coherent purpose per branch and pull request. Do not reuse a branch after its PR is merged.
3. Use conventional commit subjects (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
4. Fill in the pull-request template, including exact verification commands and every data, schema, deployment, or security impact.
5. Resolve review conversations and required checks before merging. Merged branches are deleted automatically.

Do not force-push `main`, commit directly around its protection rules, or leave completed work on long-lived topic branches.

## Change rules

### Runtime and MCP contracts

- Keep tool registration declarative in `bddk_mcp/tools/registry.py` and update contract tests when a tool changes.
- Preserve stable error codes, structured outputs, and the text fallback used by clients.
- Add focused tests beside the corresponding `tests/test_<area>.py` coverage.

### Corpus schema and corpus file

- The schema lives in `bddk_mcp/store/corpus_db.py` (`CREATE TABLE IF NOT EXISTS`, `SCHEMA_VERSION` recorded in the `meta` table). There is no migration ledger: a schema change means a rebuilt corpus file (`bddk-mcp build --force`; remove the old file first when a table layout changed), never an in-place upgrade.
- Document the rebuild, re-embedding time and rollout requirements for any schema or retrieval-profile change (embedding model, chunking, text normalization, FTS tokenizer).
- Serving opens the corpus read-only. Keep every write path in `bddk-mcp build`, `bddk-mcp sync`, `bddk-mcp export` and the maintainer scripts under `scripts/`; a tool must never write.
- Query SQL belongs to the module that owns the table (`bddk_mcp/store/*.py`); run it through `CorpusDB.run` / `CorpusDB.transaction`.

### Corpus and generated artifacts

- `seed_data/*.json` and `uv.lock` are generated artifacts. Do not hand-edit them to make a test pass.
- State the generation command and validation result in the PR.
- Never commit private keys, credentials, customer data, restricted source documents, or a built corpus file.
- Use the data-quality issue form for defects that require authoritative source review.

### Deployment and security

- Remote HTTP must remain fail-closed.
- Changes under `deploy/`, the Dockerfile, authentication, or egress need explicit threat/rollout notes.
- Public examples must contain only obvious local fixtures or placeholders; never paste a real token, certificate, or endpoint secret.

### Documentation

- Put durable user and operator guidance in `docs/`; keep the root README as the project entry point.
- Mark historical reviews and evidence as snapshots so they are not mistaken for current guarantees.
- Update [the documentation index](docs/README.md) when adding a top-level document.
- Add user-visible changes to `CHANGELOG.md` under the unreleased section.

## Review standard

A change is ready when its behavior, failure mode, verification evidence, and deployment/data impact are clear to someone who did not author it. Green tests are necessary evidence, but they do not replace source review, legal validation, or environment-specific acceptance where those are required.
