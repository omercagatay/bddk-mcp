## Purpose

<!-- What problem does this solve, and why is this the right scope? -->

## Changes

-

## Verification

- [ ] `uv lock --check`
- [ ] `uv run ruff check .`
- [ ] `uv run ruff format --check .`
- [ ] Relevant focused tests pass
- [ ] Full non-GPU suite passes, or the omitted lanes are explained below

Commands and results:

```text

```

## Impact

- [ ] No MCP contract change
- [ ] No corpus schema change (`bddk_mcp/store/corpus_db.py`) and no corpus rebuild required
- [ ] No corpus, seed, embedding, or generated-data change
- [ ] No deployment, authentication, or security-policy change
- [ ] Documentation and changelog are updated where needed

<!-- Explain every unchecked item and include rollout/rebuild notes when relevant. -->

## Review notes

<!-- Call out risky files, generated artifacts, corpus rebuild steps (`bddk-mcp build --force`), or follow-up work. -->
