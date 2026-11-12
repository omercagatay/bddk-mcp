# Current Repository Status

Repository baseline verified on **2026-11-12** (SQLite corpus branch). Corpus facts were last refreshed on **2026-10-16** (quality pass), **2026-09-29** (KGK TMS 12 addition) and **2026-11-12** (decision dates of 636/774, re-chunked KGK/TMS documents and their exported vectors).

This is the concise source for current repository facts. Dated reviews that used to live beside it were removed on 2026-11-09 and remain in Git history at tag `archive/pre-simplify-2026-11-09`. The PostgreSQL backend that the corpus file replaced on 2026-11-10 is at tag `archive/postgres-2026-11-10`.

## Current contract

| Area | Current repository state |
|---|---|
| Package | `bddk-mcp` 5.0.1 metadata; not yet represented by a `v5.0.1` GitHub release or tag. |
| Python | 3.12 and 3.13. |
| MCP tools | 14 tools in one public surface (`bddk_mcp/tools/registry.py`); no operator profile, resources, or prompts. |
| Corpus store | One SQLite file (`BDDK_CORPUS_PATH`, default `corpus/bddk.sqlite`, git-ignored; about 100 MB for the current seed), schema version 1 recorded in its `meta` table together with the embedding model and dimension (`bddk_mcp/store/corpus_db.py`). `bddk-mcp build` creates it from `seed_data/`, reusing the chunk vectors exported to `seed_data/embeddings/` while their digest and model match (an unchanged seed builds in about half a minute without loading the model; a build that has anything to chunk loads the tokenizer before its first write, so an offline host fails early and a file the build had just created is removed); serving opens it read-only (SQLite `mode=ro`) and refuses to start unless the tables, documents, chunks and catalog are present, every chunk and section hash is current, and the recorded embedding model matches the configured one. A writer that closes the file leaves it in rollback-journal (`delete`) mode on a best-effort basis, so a read-only open creates no `-wal`/`-shm` files; a file that stayed in WAL mode shows it in `bddk-mcp info`. A writer refuses an existing file of another schema version, a database without a `meta` table and a non-SQLite file, without changing it. `bddk-mcp info` describes the file read-only (package version, a 12-character digest over the stored document texts, counts, embedding model, schema version, size, journal mode; exit 1 when not ready) and `serve` logs the same identity in one `Corpus ready` line once the startup check passes. Dense retrieval is exact cosine similarity over all chunk vectors held in memory; keyword and section search are FTS5 BM25 over Turkish-normalized text; dense and keyword candidates are merged with Reciprocal Rank Fusion, and the final order uses a heuristic relevance score. |
| Corpus | 319 documents, 13,721 chunks, quality policy v8 (any `cid:` marker or double-encoded Turkish text makes a document at least `warning`; no seed document changed label) (`seed_data/documents.json`, `seed_data/decision_cache.json`). The 2026-10-16 quality pass repaired 297 documents against their official sources; the per-document receipts were archived with the governance layer. Twenty mevzuat.gov.tr texts were rebuilt from their official PDFs and one KGK standard (TMS 12, `kgk_tms_12`) was added from a manually downloaded public PDF; since 2026-11-12 the section parser indexes its 140 printed paragraph numbers (`29A` as `paragraf 29/a`) instead of body windows, and ends a capped body window at a paragraph, line or word break. The exported vectors in `seed_data/embeddings/` match these chunks, so a build embeds nothing. Freshness is not measured; 297 documents carry formula-unaware extraction provenance. No production deployment or independent legal approval is claimed. |
| Answers | Section search covers headings and bodies, adds inflection-tolerant matches when strict ones are few, and returns full-text windows of documents that have no article structure (most Kurul Kararları; 204 of 319 documents in the partial corpus checked on 2026-11-12), labelled as such. A capped provision is reported with its true extent and a quotation outside its stored window is `unavailable`, not `not_found`. A document whose BDDK catalog class is Mülga Düzenleme or Düzenleme Taslağı carries a `MÜLGA (repealed)` or `TASLAK (draft)` line and warning; the catalog class is the only source of that mark, and its absence says nothing about being in force. Section answers show the decision date and number and the UTC date the text was downloaded (`unknown` when none is stored); that date does not say the text is current. `get_document_history` is the local text-change log of the corpus file, not amendment history. Catalog date filters report how many undated entries they could not consider. A number in a query that equals a stored document id only scopes the search when that document has a match; the answer says when the scope was applied or dropped. |
| Legal status | Not modeled. The abstention-first legal-version resolver and relations graph (`resolve_regulation_status`, `get_amendment_chain`, `get_cross_references`, Citation v1) were removed on 2026-11-09 because no production path ever populated their `regulatory_*` tables; those tables went away with the PostgreSQL schema on 2026-11-10. `get_document_section` keeps the character-exact `quotation` check. |
| Runtime | One process, one corpus file (`BDDK_CORPUS_PATH`); no database service, connection string or separate login. stdio by default; Streamable HTTP with the fail-closed remote policy. |
| CI | Lint, hygiene, unit (3.12/3.13, temporary corpus file, no service), packaging and container-static checks. Skips are not passes; on 2026-11-12 the suite ran with no skipped tests (tests marked `gpu` are deselected by default). The hygiene check fails on a tracked `.env.*` file other than `.env.example` and expects Dependabot to cover `uv`, `github-actions` and `docker`. |
| Maturity | Engineering beta. Repository controls do not establish legal advice, bank acceptance, or production readiness. |

The tool count is derived from `bddk_mcp/tools/registry.py`; the corpus schema version from `SCHEMA_VERSION` in `bddk_mcp/store/corpus_db.py`. Contract tests pin these facts.

## What is ready at repository level

- Packaged stdio and Streamable HTTP MCP entry points with a strict public tool contract.
- Explicit corpus preparation: `bddk-mcp build`, `bddk-mcp sync`, `bddk-mcp export` (refuses a corpus file that is not ready and then writes nothing to `seed_data/`), and the read-only `bddk-mcp info`; the serving process never writes the corpus. A plain `sync` re-indexes a stored document that lacks current chunks from its stored text instead of downloading it again.
- Fail-closed remote HTTP configuration (exact Host/Origin allowlists, JWT or explicit unauthenticated opt-in).
- Container image that carries the built corpus file, and a single-service Compose topology for local development; Railway remains a development/preview profile.
- Automated coverage across runtime, corpus build and readiness, retrieval, extraction quality and the HTTP transport.

## Upgrading an existing corpus file

Any existing corpus file built with code older than this release (including a file built earlier on 2026-11-12, before the second maintenance round of that day) keeps the old decision dates of documents 636 and 774 (07.02.2013 and 14.02.2020; the seed now has 31.01.2013 and 13.02.2020, as printed in the decision texts) and the old article headings (1,596 headings in 79 documents now come from the title line printed above `MADDE` instead of the first fıkra text). No content hash changed, so a plain `bddk-mcp build` reports that the file already matches the seed. Build a new file with the owner's approval (`bddk-mcp build --corpus <new path>`, which needs the tokenizer from the network or from `BDDK_EMBEDDING_MODEL_PATH`), check it with `bddk-mcp info --corpus <new path>`, then replace the old file while no server has it open; no chunk text changed, so nothing is re-embedded. The Docker image builds a fresh file and is not affected.

## What is not established

- A tagged or published 5.0.1 release.
- Independent human/legal approval of the corpus; a clean extraction is not that approval.
- Legal advice or authoritative proof of which rule applies to a real case.
- Measured live freshness for every corpus document.
- Bank-owned identity, CA, network, corpus-file distribution, backup, image-signing, promotion, and operational acceptance.
- Named client/model certification, approved expert judgments, or audit-grade product scores.
- Any model of legal currentness, amendment chains or cross-references; the assistant must verify those against the official source.
- Live acquisition against mevzuat.gov.tr. On 2026-11-12 the bulletin, monthly, trend, digest, announcement and institution tools and one `sync --force` of a BDDK-hosted decision ran against the live BDDK source from a sandbox (its DNS pre-check was stubbed in the trial script only, because that sandbox has no resolver). mevzuat.gov.tr did not answer from there, so the mevzuat-specific `sync` changes of that day (annex text, error-page fallthrough, charset repair, table limits) were tested with fakes only. The LightOCR path and the Docker image were not exercised.
- The displayed search relevance, its bands and the raw-cosine threshold default (0.80) are calibrated for `intfloat/multilingual-e5-base` on 101 queries with one model revision (see [Document Quality](DOCUMENT_QUALITY.md#displayed-search-relevance)); that is a calibration, not a retrieval evaluation, and any other embedding model gets an uncalibrated identity mapping with a 0.50 threshold. The unit tests use a fake embedder.
- `bddk-mcp build` cannot detect by itself that seed metadata or the section parser changed; it skips by content hash, chunks and catalog count.
- Retrieval beyond a few hundred thousand chunks: dense search is brute-force cosine over in-memory vectors, which is exact and fast at the current size; an ANN index (hnswlib or sqlite-vec) is the upgrade past that scale.

## Maintenance policy

- `main` is the only long-lived branch; merged topic branches are deleted automatically.
- Pull requests must pass the protected checks and state schema, corpus, deployment, and security impact.
- Current facts belong here or in directly executable contracts.
- Version metadata, changelog state, and GitHub tags/releases must be reconciled before publishing 5.0.1.
