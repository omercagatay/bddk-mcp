# Current Repository Status

Repository baseline verified on **2026-08-26**. Corpus, schema, and test facts below
were refreshed on **2026-09-07**, based on PR #151 (`fe8d72b`); see the
[corpus refresh evidence](evidence/corpus-v8-release-2026-09-07.md) for execution state.

This is the concise source for current repository facts. The longer architecture, review, roadmap, and testing documents preserve dated analysis and may intentionally describe older checkpoints.

## Current contract

| Area | Current repository state |
|---|---|
| Package | `bddk-mcp` 5.0.1 metadata; not yet represented by a `v5.0.1` GitHub release or tag. |
| Python | 3.12 and 3.13. |
| MCP public profile | 17 tools. |
| MCP operator profile | 17 public tools plus 14 operator additions, 31 total. |
| MCP resources/prompts | One resource (`bddk://corpus/active-release`); zero prompts. |
| Database | PostgreSQL 17 contract coverage; append-only migration ledger through schema v11. |
| Corpus | `bddk-job-corpus-2026-09-07`: 318 unchanged canonical documents / 13,240 regenerated chunks, technically reviewed and owner-delegated Ed25519-signed for parser v8. Freshness remains quantified and unmeasured. Signing does not establish legal currentness; activation/deployment receipts are tracked in the refresh evidence. |
| Runtime profiles | Separate public and operator processes, scopes, and database identities. |
| CI | PR #151 passed all ten required checks. The signed v8 refresh locally passes 1,731 DB-less tests and 222 PostgreSQL tests; skips are not passes. Its own protected PR checks remain required before merge. |
| Maturity | Engineering beta. Repository controls do not establish legal advice, bank acceptance, or production readiness. |

The tool counts are derived from `bddk_mcp/tools/registry.py`; schema version is derived from `bddk_mcp/migrations/runner.py`; corpus identity and counts are derived from `seed_data/corpus_scope.yml`. Contract tests pin these facts.

## What is ready at repository level

- Packaged stdio and Streamable HTTP MCP entry points with strict public/operator profiles.
- Protected `main` with required lint, Python, PostgreSQL, packaging, container, and supply-chain checks.
- Fail-closed remote HTTP configuration and separate database lifecycle roles.
- Signed, governed offline corpus artifacts with staged verifier/publisher activation.
- Structured deployment assets for local Compose, PostgreSQL, OpenShift, and Open WebUI; Railway remains a development/preview profile outside the bank path.
- A loopback-only, read-only operator console (`bddk-mcp admin-ui`, `bddk_mcp/admin/`) that refuses non-loopback binds and ships in no deployment manifest.
- Broad automated coverage across runtime, migrations, retrieval, deployment, recovery, and supply-chain policy.

## What is not established

- A tagged or published 5.0.1 release.
- Independent human/legal approval of the signed technical corpus refresh; signing alone is not that approval.
- Legal advice or authoritative proof of which rule applies to a real case.
- Measured live freshness for every corpus document.
- Bank-owned identity, CA, network, database, backup/PITR, image-signing, promotion, and operational acceptance.
- Named client/model certification, approved expert judgments, or audit-grade product scores.
- Generation-bound serving and an authorized retained-generation rollback workflow.

See the [gap register](GAP_REGISTER.md), [architecture](ARCHITECTURE.md), and [deployment guide](DEPLOYMENT.md) for the detailed boundaries.

## Maintenance policy

- `main` is the only long-lived branch; merged topic branches are deleted automatically.
- Pull requests must pass the protected checks and state schema, corpus, deployment, and security impact.
- Current facts belong here or in directly executable contracts. Dated reviews remain snapshots and link back here.
- Version metadata, changelog state, and GitHub tags/releases must be reconciled before publishing 5.0.1.
