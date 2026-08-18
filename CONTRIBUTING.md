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
- PostgreSQL 17 with `pgvector` and `unaccent` for database-marked tests
- Docker with Compose for the disposable local database workflow

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
uv run pytest tests/ -m "not postgres and not gpu" --strict-markers -q
```

The complete CI matrix also exercises Python 3.12 and 3.13, PostgreSQL roles and migrations, package artifacts, container recipes, OpenShift manifests, and supply-chain evidence. See [Testing and evaluation](docs/TESTING_AND_EVALUATION_STRATEGY.md) for the longer lanes and required environment variables.

## Branch and pull-request workflow

1. Start from current `main` and create a short-lived branch named `type/short-description`, such as `fix/section-pagination` or `docs/deployment-example`.
2. Keep one coherent purpose per branch and pull request. Do not reuse a branch after its PR is merged.
3. Use conventional commit subjects (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
4. Fill in the pull-request template, including exact verification commands and every data, schema, deployment, or security impact.
5. Resolve review conversations and required checks before merging. Merged branches are deleted automatically.

Do not force-push `main`, commit directly around its protection rules, or leave completed work on long-lived topic branches.

## Change rules

### Runtime and MCP contracts

- Public and operator tool profiles are separate security boundaries.
- Keep tool registration declarative in `bddk_mcp/tools/registry.py` and update contract tests when a tool changes.
- Preserve stable error codes, structured outputs, and the text fallback used by clients.
- Add focused tests beside the corresponding `tests/test_<area>.py` coverage.

### Database and migrations

- Migrations in `bddk_mcp/migrations/` are append-only after merge. Add a new numbered migration instead of rewriting history.
- Preserve role separation among schema owner, ingestion, verifier, publisher, public, operator, and telemetry identities.
- Document backup, maintenance-window, backfill, and rollout requirements for any state transition.
- Run the PostgreSQL-marked lanes for schema, privilege, publication, or query changes.

### Corpus and generated artifacts

- `seed_data/*.json`, corpus signatures, evidence JSON, and `uv.lock` are generated or governed artifacts. Do not hand-edit them to make a test pass.
- State the generation command, source corpus/manifest identity, and validation result in the PR.
- Keep trust keys outside the corpus root and never commit private keys, credentials, customer data, or restricted source documents.
- Use the data-quality issue form for defects that require authoritative source review.

### Deployment and security

- Remote public and operator modes must remain fail-closed.
- Changes under `deploy/`, `supply-chain/`, the Dockerfiles, authentication, egress, or database roles need explicit threat/rollout notes.
- Public examples must contain only obvious local fixtures or placeholders; never paste a real DSN, token, certificate, or endpoint secret.

### Documentation

- Put durable user and operator guidance in `docs/`; keep the root README as the project entry point.
- Mark historical reviews and evidence as snapshots so they are not mistaken for current guarantees.
- Update [the documentation index](docs/README.md) when adding a top-level document.
- Add user-visible changes to `CHANGELOG.md` under the unreleased section.

## Review standard

A change is ready when its behavior, failure mode, verification evidence, and deployment/data impact are clear to someone who did not author it. Green tests are necessary evidence, but they do not replace source review, legal validation, or environment-specific acceptance where those are required.
