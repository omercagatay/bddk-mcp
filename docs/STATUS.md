# Current Repository Status

Repository baseline verified on **2026-08-26**. Corpus facts below were refreshed on
**2026-09-29** (KGK TMS 12 addition); quality-policy facts on **2026-09-21**, following the merged v8 refresh (PR #152).
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
| Corpus | `bddk-job-corpus-kgk-tms12-pdf-rebuilds-local-v10-2026-10-02`: 319 documents, 13,558 chunks, quality policy v5; Ed25519-signed by the owner, not yet staged or activated. It supersedes the signed but unactivated v8 manifest (and an owner-signed, uncommitted v9 that lacked two of these repairs). Twenty mevzuat.gov.tr texts are rebuilt from their official PDFs (`scripts/rebuild_mevzuat_from_pdf.py`): sixteen regulations whose extraction scrambled justified lines (`mevzuat_22599`, `16071`, `39171`, `40203`, `40519`, `20645`, `10522`, `38416`, `10731`, `42359`, `38568`, `23228`, `11203`, `10746`, `38797`, `42628`), the communiqué `mevzuat_11201`, and the Bankacılık Kanunu, Sermaye Piyasası Kanunu and Banka Kartları ve Kredi Kartları Kanunu (`mevzuat_5411`, `6362`, `5464`); the communiqué and the laws come from mevzuat.gov.tr's generated PDF of the current consolidated text, and `mevzuat_42628`'s 34 image formulas are transcribed into LaTeX. It adds one KGK accounting standard, TMS 12 Gelir Vergileri (`kgk_tms_12`, category `Muhasebe Standardı`), from a manually downloaded public PDF (`scripts/add_kgk_tms_12.py`); kgk.gov.tr stays off the outbound allowlist, and the standard is indexed as body windows, not numbered paragraphs. The previous v6 release (318 documents, 50 source repairs) was verified/activated in fresh local staging and retired all 11 historical extraction failures; see `docs/evidence/document-repairs/local-release-v6.json`. Freshness remains quantified and unmeasured; 310 documents carry formula-unaware extraction provenance. No production deployment or independent legal approval is claimed. |
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
- An admin console (`bddk-mcp admin-ui`, `bddk_mcp/admin/`) with public-reader PostgreSQL access, optional persistent SQLite editorial drafts and optional server-held Ed25519 document signing. Remote binds require operator JWT authentication. Drafts never mutate/publish the corpus; see [configuration and handoff](CORPUS_GOVERNANCE.md#admin-editorial-drafts-and-document-signatures).
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
