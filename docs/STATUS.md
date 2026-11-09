# Current Repository Status

Repository baseline verified on **2026-11-09** (simplification branch). Corpus facts were last refreshed on **2026-10-16** (quality pass) and **2026-09-29** (KGK TMS 12 addition).

This is the concise source for current repository facts. Dated reviews that used to live beside it were removed on 2026-11-09 and remain in Git history at tag `archive/pre-simplify-2026-11-09`.

## Current contract

| Area | Current repository state |
|---|---|
| Package | `bddk-mcp` 5.0.1 metadata; not yet represented by a `v5.0.1` GitHub release or tag. |
| Python | 3.12 and 3.13. |
| MCP tools | 14 tools in one public surface (`bddk_mcp/tools/registry.py`); no operator profile, resources, or prompts. |
| Database | PostgreSQL 17; append-only migration ledger through schema v11. Serving requires the current ledger version and a published corpus; tables created by the retired governance migrations (v0005–v0011 release, retention and graph-state objects) stay in the schema unused. |
| Corpus | 319 documents, 13,604 chunks, quality policy v7, section parser v10 (`seed_data/documents.json`, `seed_data/decision_cache.json`). `seed_data/corpus_scope.yml` is a retained scope declaration that the runtime no longer reads. Its 2026-10-16 quality pass repaired 297 documents against their official sources; the per-document receipts were archived with the governance layer. Twenty mevzuat.gov.tr texts were rebuilt from their official PDFs and one KGK standard (TMS 12, `kgk_tms_12`) was added from a manually downloaded public PDF. Freshness is not measured; 297 documents carry formula-unaware extraction provenance. No production deployment or independent legal approval is claimed. |
| Legal status | Not modeled. The abstention-first legal-version resolver and relations graph (`resolve_regulation_status`, `get_amendment_chain`, `get_cross_references`, Citation v1) were removed on 2026-11-09 because no production path ever populated their `regulatory_*` tables; those tables remain in the schema (migrations v0004, v0006, v0009) unused. `get_document_section` keeps the character-exact `quotation` check. |
| Runtime | One process, one `BDDK_DATABASE_URL`; optional `BDDK_INGESTION_DATABASE_URL` / `BDDK_SCHEMA_OWNER_DATABASE_URL` for deployments with separate logins. stdio by default; Streamable HTTP with the fail-closed remote policy. |
| CI | Lint, hygiene, unit (3.12/3.13), PostgreSQL (3.12/3.13), packaging and container-static checks. Skips are not passes. |
| Maturity | Engineering beta. Repository controls do not establish legal advice, bank acceptance, or production readiness. |

The tool count is derived from `bddk_mcp/tools/registry.py`; the schema version from `bddk_mcp/migrations/runner.py`. Contract tests pin these facts.

## What is ready at repository level

- Packaged stdio and Streamable HTTP MCP entry points with a strict public tool contract.
- Explicit database preparation: `bddk-mcp migrate`, `bddk-seed import`, `bddk-mcp sync`.
- Fail-closed remote HTTP configuration (exact Host/Origin allowlists, JWT or explicit unauthenticated opt-in).
- Container image and Compose topology for local development; Railway remains a development/preview profile.
- Automated coverage across runtime, migrations, retrieval, extraction quality and the HTTP transport.

## What is not established

- A tagged or published 5.0.1 release.
- Independent human/legal approval of the corpus; a clean extraction is not that approval.
- Legal advice or authoritative proof of which rule applies to a real case.
- Measured live freshness for every corpus document.
- Bank-owned identity, CA, network, database, backup/PITR, image-signing, promotion, and operational acceptance.
- Named client/model certification, approved expert judgments, or audit-grade product scores.
- Any model of legal currentness, amendment chains or cross-references; the assistant must verify those against the official source.

## Maintenance policy

- `main` is the only long-lived branch; merged topic branches are deleted automatically.
- Pull requests must pass the protected checks and state schema, corpus, deployment, and security impact.
- Current facts belong here or in directly executable contracts.
- Version metadata, changelog state, and GitHub tags/releases must be reconciled before publishing 5.0.1.
