# Current Repository Status

Repository baseline verified on **2026-11-10** (SQLite corpus branch). Corpus facts were last refreshed on **2026-10-16** (quality pass) and **2026-09-29** (KGK TMS 12 addition).

This is the concise source for current repository facts. Dated reviews that used to live beside it were removed on 2026-11-09 and remain in Git history at tag `archive/pre-simplify-2026-11-09`. The PostgreSQL backend that the corpus file replaced on 2026-11-10 is at tag `archive/postgres-2026-11-10`.

## Current contract

| Area | Current repository state |
|---|---|
| Package | `bddk-mcp` 5.0.1 metadata; not yet represented by a `v5.0.1` GitHub release or tag. |
| Python | 3.12 and 3.13. |
| MCP tools | 14 tools in one public surface (`bddk_mcp/tools/registry.py`); no operator profile, resources, or prompts. |
| Corpus store | One SQLite file (`BDDK_CORPUS_PATH`, default `corpus/bddk.sqlite`, git-ignored; about 100 MB for the current seed), schema version 1 recorded in its `meta` table together with the embedding model and dimension (`bddk_mcp/store/corpus_db.py`). `bddk-mcp build` creates it from `seed_data/`, reusing the chunk vectors exported to `seed_data/embeddings/` while their digest and model match (an unchanged seed builds in about half a minute without loading the model); serving opens it read-only (SQLite `mode=ro`) and refuses to start unless the tables, documents, chunks and catalog are present, every chunk and section hash is current, and the recorded embedding model matches the configured one. Dense retrieval is exact cosine similarity over all chunk vectors held in memory; keyword and section search are FTS5 BM25 over Turkish-normalized text; dense and keyword candidates are merged with Reciprocal Rank Fusion, and the final order uses a heuristic relevance score. |
| Corpus | 319 documents, 13,721 chunks, quality policy v7 (`seed_data/documents.json`, `seed_data/decision_cache.json`). The 2026-10-16 quality pass repaired 297 documents against their official sources; the per-document receipts were archived with the governance layer. Twenty mevzuat.gov.tr texts were rebuilt from their official PDFs and one KGK standard (TMS 12, `kgk_tms_12`) was added from a manually downloaded public PDF; since 2026-11-12 the section parser indexes its 140 printed paragraph numbers (`29A` as `paragraf 29/a`) instead of body windows, and ends a capped body window at a paragraph, line or word break. The exported vectors in `seed_data/embeddings/` match these chunks, so a build embeds nothing. Freshness is not measured; 297 documents carry formula-unaware extraction provenance. No production deployment or independent legal approval is claimed. |
| Legal status | Not modeled. The abstention-first legal-version resolver and relations graph (`resolve_regulation_status`, `get_amendment_chain`, `get_cross_references`, Citation v1) were removed on 2026-11-09 because no production path ever populated their `regulatory_*` tables; those tables went away with the PostgreSQL schema on 2026-11-10. `get_document_section` keeps the character-exact `quotation` check. |
| Runtime | One process, one corpus file (`BDDK_CORPUS_PATH`); no database service, connection string or separate login. stdio by default; Streamable HTTP with the fail-closed remote policy. |
| CI | Lint, hygiene, unit (3.12/3.13, temporary corpus file, no service), packaging and container-static checks. Skips are not passes. |
| Maturity | Engineering beta. Repository controls do not establish legal advice, bank acceptance, or production readiness. |

The tool count is derived from `bddk_mcp/tools/registry.py`; the corpus schema version from `SCHEMA_VERSION` in `bddk_mcp/store/corpus_db.py`. Contract tests pin these facts.

## What is ready at repository level

- Packaged stdio and Streamable HTTP MCP entry points with a strict public tool contract.
- Explicit corpus preparation: `bddk-mcp build`, `bddk-mcp sync`, `bddk-mcp export`; the serving process never writes the corpus.
- Fail-closed remote HTTP configuration (exact Host/Origin allowlists, JWT or explicit unauthenticated opt-in).
- Container image that carries the built corpus file, and a single-service Compose topology for local development; Railway remains a development/preview profile.
- Automated coverage across runtime, corpus build and readiness, retrieval, extraction quality and the HTTP transport.

## What is not established

- A tagged or published 5.0.1 release.
- Independent human/legal approval of the corpus; a clean extraction is not that approval.
- Legal advice or authoritative proof of which rule applies to a real case.
- Measured live freshness for every corpus document.
- Bank-owned identity, CA, network, corpus-file distribution, backup, image-signing, promotion, and operational acceptance.
- Named client/model certification, approved expert judgments, or audit-grade product scores.
- Any model of legal currentness, amendment chains or cross-references; the assistant must verify those against the official source.
- Retrieval beyond a few hundred thousand chunks: dense search is brute-force cosine over in-memory vectors, which is exact and fast at the current size; an ANN index (hnswlib or sqlite-vec) is the upgrade past that scale.

## Maintenance policy

- `main` is the only long-lived branch; merged topic branches are deleted automatically.
- Pull requests must pass the protected checks and state schema, corpus, deployment, and security impact.
- Current facts belong here or in directly executable contracts.
- Version metadata, changelog state, and GitHub tags/releases must be reconciled before publishing 5.0.1.
