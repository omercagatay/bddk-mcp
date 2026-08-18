# Current Repository Status

Verified on **2026-08-18** against commit `3a931892a96b4454faf2a48cef6a30c9898829d9` (`main`) before the repository-hygiene change set.

This is the concise source for current repository facts. The longer architecture, review, roadmap, and testing documents preserve dated analysis and may intentionally describe older checkpoints.

## Current contract

| Area | Current repository state |
|---|---|
| Package | `bddk-mcp` 5.0.1 metadata; not yet represented by a `v5.0.1` GitHub release or tag. |
| Python | 3.12 and 3.13. |
| MCP public profile | 17 tools. |
| MCP operator profile | 17 public tools plus 14 operator additions, 31 total. |
| MCP resources/prompts | One resource (`bddk://corpus/active-release`); zero prompts. |
| Database | PostgreSQL 17; append-only migration ledger through schema v10. |
| Corpus | 318 documents and 9,675 chunks in signed manifest `bddk-job-corpus-2026-08-14`. Freshness objectives are quantified; per-document live freshness remains unmeasured. |
| Runtime profiles | Separate public and operator processes, scopes, and database identities. |
| CI | `CI` and `Supply chain evidence` passed on the verified `main` commit. |
| Maturity | Engineering beta. Repository controls do not establish legal advice, bank acceptance, or production readiness. |

The tool counts are derived from `bddk_mcp/tools/registry.py`; schema version is derived from `bddk_mcp/migrations/runner.py`; corpus identity and counts are derived from `seed_data/corpus_scope.yml`. Contract tests pin these facts.

## What is ready at repository level

- Packaged stdio and Streamable HTTP MCP entry points with strict public/operator profiles.
- Protected `main` with required lint, Python, PostgreSQL, packaging, container, and supply-chain checks.
- Fail-closed remote HTTP configuration and separate database lifecycle roles.
- Signed, governed offline corpus artifacts with staged verifier/publisher activation.
- Structured deployment assets for local Compose, PostgreSQL, OpenShift, Keycloak, Open WebUI, Railway, and Hugging Face Spaces.
- Broad automated coverage across runtime, migrations, retrieval, deployment, recovery, and supply-chain policy.

## What is not established

- A tagged or published 5.0.1 release.
- Legal advice or authoritative proof of which rule applies to a real case.
- Measured live freshness for every corpus document.
- Bank-owned identity, CA, network, database, backup/PITR, image-signing, promotion, and operational acceptance.
- Named client/model certification, approved expert judgments, or audit-grade product scores.
- Generation-bound serving and an authorized retained-generation rollback workflow.

See the [gap register](GAP_REGISTER.md), [target architecture](TARGET_ARCHITECTURE.md), and [deployment guide](DEPLOYMENT.md) for the detailed boundaries.

## Maintenance policy

- `main` is the only long-lived branch; merged topic branches are deleted automatically.
- Pull requests must pass the protected checks and state schema, corpus, deployment, and security impact.
- Current facts belong here or in directly executable contracts. Dated reviews remain snapshots and link back here.
- Version metadata, changelog state, and GitHub tags/releases must be reconciled before publishing 5.0.1.
