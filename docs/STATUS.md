# Current Repository Status

Repository baseline verified on **2026-08-26**. Corpus and quality-policy facts below
were refreshed on **2026-09-21**, following the merged v8 refresh (PR #152).
See the [repair release evidence](evidence/document-repairs/local-release-v6.json)
for signed artifacts, fresh local staging, and real MCP verification.
The earlier checkpoint was **2026-08-18** (`3a93189`).

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
| Corpus | `bddk-job-corpus-quality-repairs-local-v6-2026-09-21`: 318 documents (50 source repairs), 13,544 chunks, quality policy v5; Ed25519-signed and verified/activated in fresh local staging. All 11 historical extraction failures are retired with the signed corpus. Freshness remains quantified and unmeasured; 310 formula-unaware provenance warnings remain. No production deployment or independent legal approval is claimed. See `docs/evidence/document-repairs/local-release-v6.json`. |
| Runtime profiles | Separate public and operator processes, scopes, and database identities. |
| CI | The quality-repair PR is validated independently of unrelated admin-editor changes. Local unit/integration results and required GitHub checks are recorded in the PR; skips are not passes. Protected checks remain required before merge. |
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
