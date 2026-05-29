# Contributing

## Setup

```bash
uv sync --dev
docker compose up -d db
```

Set `BDDK_DATABASE_URL` when running the server or database-backed scripts:

```bash
export BDDK_DATABASE_URL=postgresql://bddk:bddk@localhost:5432/bddk
```

## Development Loop

```bash
uv run ruff check .
uv run ruff format .
uv run pytest tests/ -v --tb=short
```

Focused checks for common areas:

```bash
uv run pytest tests/test_tools_sections.py tests/test_doc_store.py -k section -v
uv run pytest tests/test_markdown_quality.py tests/test_tools_documents.py -v
uv run pytest tests/test_vector_store.py tests/test_legal_ref.py -v -rs
```

## Repository Conventions

- `server.py` is the FastMCP entry point.
- Top-level engine modules contain business logic.
- `tools/` modules are thin MCP wrappers and should receive dependencies through `Dependencies`.
- `scripts/` contains operator and one-off maintenance commands.
- `seed_data/` is generated from the database and should only change intentionally.
- `docs/reference/` is for generated or bulky reference material.

## Pull Requests

- Keep PRs focused on one behavior or maintenance task.
- Include the verification commands you ran.
- Call out any `seed_data/`, database, or deployment impact.
- Add or update tests when behavior changes.
