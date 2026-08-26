# Current Repository Status

Verified on **2026-08-26** on the `bank-delivery-fixes` change set (base `df5bf34`); re-stamp this line with the merge commit when the set lands on `main`. The previous verification was 2026-08-18 against `3a93189`.

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
| Corpus | Drift open (gap register CUR-018): the signed `bddk-job-corpus-2026-08-14` manifest (318 documents / 9,675 chunks) predates the v5 section-parser profile, which regenerates 10,483 chunks. A v5 regeneration and updated manifest (`bddk-job-corpus-2026-08-26`, `signature_status: not_configured`) are staged pending owner review and Ed25519 signature via `scripts/sign_corpus_manifest.py`. Freshness objectives remain quantified and unmeasured. |
| Runtime profiles | Separate public and operator processes, scopes, and database identities. |
| CI | `CI` and `Supply chain evidence` passed on the `main` base commit `df5bf34`; the change set was verified locally (lint, format, hygiene, unit suite, PostgreSQL suite) and must pass both workflows on merge. The committed branch is green, including all 104 corpus-bound contract tests, because it still carries the previously signed corpus. The CUR-018 regeneration is staged **uncommitted** in the working tree; committing it turns 26 of those tests red until the owner signs the manifest. |
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
- An owner-signed corpus manifest for the current v5 retrieval profile (gap register CUR-018; regeneration staged, signature pending).
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
