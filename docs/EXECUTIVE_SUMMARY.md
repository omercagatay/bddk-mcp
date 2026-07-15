# Executive Summary

Review baseline: commit **5684a34c10e6d90bc22d6ab2a6466944afb6bf81**, reviewed 2026-07-14. Historical findings below describe that commit unless a post-review note says otherwise.

## Bottom line

BDDK MCP is now a coherent engineering beta with a credible MCP, database-safety, acquisition, and deployment foundation. It should still not be relied on to decide which Turkish banking rule is legally applicable, produce audit-grade evidence, or enter bank production until the remaining legal-evidence and bank-acceptance work is completed.

The repository does more than a basic chatbot demo. It collects and normalizes regulatory documents, stores and searches them with keyword and semantic techniques, recognizes Turkish legal section types, protects important publication boundaries, exposes strict tools through official MCP transports, and tests fail-closed remote and database behavior.

The missing pieces are now concentrated in the high-value product and deployment layers: authoritative version history, effective dates, amendment/repeal relationships, exact source-page evidence, whole-corpus rollback, expert Turkish evaluation, accepted bank identity/network/CA/backup controls, and proof with the intended client/model combinations.

## Implementation progress overlay — 2026-07-15

The current working tree has moved beyond the original engineering prototype in several concrete ways:

- **Complete at repository/application level:** a packaged MCP service with official installed stdio and Streamable HTTP tests; one strict registry with 15 public plus 13 additional operator tools; privacy-safe protocol errors; separate public/operator profiles and DSNs; fail-closed Host, HTTPS Origin, asymmetric JWT, scope, body, rate, and concurrency checks; checksum v0001-v0003 migrations; reviewed PostgreSQL role/grant assets; exact target-database, schema-owner, TLS, ACL-provenance, effective-privilege, per-connection identity, and catalog-readiness checks; durable PostgreSQL job records and advisory leases; bounded SSRF/archive acquisition; pinned base/action/model revisions; PostgreSQL/distribution CI; and a non-root OpenShift starter with digest images, stable selectors, exact Secret references, PostgreSQL CA wiring, and default-deny egress.
- **Partial:** six high-value retrieval tools return structured evidence; document/section replacement and current-hash publication fail closed per document; migration v0003 refuses a blocking populated-corpus backfill by default and requires controlled reindexing after approval. Metrics and correlation-safe logging exist. The OpenShift starter still needs bank-specific values and has not run in the bank cluster; there is no immutable whole-corpus generation or rollback.
- **Open:** legal version/effective-state modeling, amendment/repeal lineage, authoritative source-page citations, expert-reviewed Turkish retrieval and answer-grounding evaluation, live named client/model certification, bank-applied identity/CA/egress/LOGIN proof, signed release/SBOM policy, backup/restore drills, numeric SLOs, and validated provision-to-audit-control mappings.

The secure remote application path is now credible for pre-production integration, but this is not production or bank deployment approval. Bank IdP, CA, registry, Route, egress, and network decisions remain unknown. Any older “working-tree” checkpoint sentence in the historical sections below is superseded by this dated overlay.

### Implementation-checkpoint ratings

| Area | Current rating | Plain-language meaning |
|---|---:|---|
| Overall maturity | 3/5 | The project is a coherent engineering beta with clear boundaries, but not yet an audit-grade regulatory knowledge product. |
| Production readiness | 2/5 | Strong repository controls and deployment starters exist; bank integration, recovery, signed delivery, SLOs, and cluster acceptance remain unproved. |
| MCP implementation | 4/5 | Official transports, strict profiles/contracts, stable errors, authentication, and protocol E2E tests are strong; named-client/version evidence remains. |
| Retrieval quality | 3/5 | Hybrid retrieval, structural parsing, current-hash publication guards, and pinned models are credible; legal currentness, authoritative pages, and representative Turkish evaluation remain unsolved. |
| Security | 3/5 | Application, database identity/ACL/TLS, acquisition, job-durability, and starter-platform controls fail closed in important paths; bank-specific acceptance and recovery remain. |
| Testing and evaluation | 3/5 | Unit, PostgreSQL, protocol, package, deployment-contract, and benchmark-contract coverage is broad; expert, live-model, load, recovery, and cluster evidence remains open. |
| Documentation | 4/5 | Architecture, security, deployment, testing, and roadmap boundaries are now extensive; external runbooks and measured acceptance evidence remain. |

## Baseline ratings at the reviewed commit

| Area | Rating | Plain-language meaning |
|---|---:|---|
| Overall maturity | 2/5 | A functional prototype with sound building blocks, not a stable product. |
| Production readiness | 1/5 | Important correctness, security, deployment, and recovery controls are absent. |
| MCP implementation | 2/5 | At the reviewed commit, the right SDK and transports were present but the documented local client command exposed no tools. |
| Retrieval quality | 2/5 | Search technology is promising; legal currentness and audit-grade citations are not solved. |
| Security | 1/5 | Remote HTTP has no built-in identity/permission boundary and operator access is only an on/off flag. |
| Testing and evaluation | 2/5 | Many code tests pass, but real MCP, database, client, citation, and model evaluations are incomplete. |
| Documentation | 2/5 | At the reviewed commit, documentation was helpful and bilingual but several important claims were inaccurate. |

These are the review-baseline scores, not a post-implementation rerating. They measure suitability for high-stakes regulatory use. They do not mean the code is poor; they mean the evidence and controls required for that use are not complete.

## What is already strong

### Retrieval engineering

The project combines semantic search and PostgreSQL full-text search, then fuses the results. It also uses section-aware/token-aware chunking and a multilingual embedding model. This is a strong foundation for Turkish regulatory research.

Evidence: **bddk_mcp/store/vector_store.py:228-390,931-1198**.

### Document acquisition and quality controls

The ingestion code handles several BDDK/Mevzuat page and file patterns, HTML, PDF, legacy document, iframe, ZIP annex, and optional OCR paths. It records hashes and extraction methods and runs deterministic quality checks.

Evidence: **bddk_mcp/ingest/doc_sync.py; html_extractor.py; ocr/base.py; quality/markdown_quality.py; quality/quality_scan.py**.

### Useful regulatory structure

The section parser recognizes articles, temporary articles, principles, paragraphs, annexes, fıkra-like numbered paragraphs, and bent-like lettered paragraphs. Exact document and section tools are more useful than a single generic RAG endpoint.

Evidence: **bddk_mcp/store/section_index.py:24-36,118-177; bddk_mcp/tools/documents.py; sections.py**.

### Good development foundations

The code is modular, parameterizes database queries, uses transactions in important write paths, has a lockfile, passes lint/format checks, and has a broad unit suite. The isolated review run produced **526 passing tests**.

### Grounding intent

The server explicitly tells models to use local documents, cite sources, state when evidence is unavailable, and surface extraction warnings. That is the right product intent, although it must be backed by structured evidence rather than instructions alone.

Evidence: **bddk_mcp/server.py:32-52**.

## What is fragile

### The documented local MCP startup path was broken at the reviewed commit

The reviewed README and checked-in client configuration used **mcp run server.py**. That command imported the server object but skipped the project's startup function, where tools and dependencies were registered. A runtime check found zero tools.

The current working tree closes this launcher defect with an installed subprocess test that covers initialize/list/call, protocol-only stdout, invalid-input recovery, and shutdown. Named client/version compatibility remains broader follow-on evidence.

### Publication is guarded per document, not as one corpus generation

At the reviewed commit, document, section, and vector updates could diverge and retrieval could prefer stale chunks. The current repository replaces document text and parsed sections in one transaction, joins chunks to the current document hash, records retrieval-publication state, invalidates incomplete indexes, and refuses retrieval when integrity is not satisfied. Migration v0003 also refuses its blocking backfill on a populated corpus unless the operator explicitly approves it and then reindexes existing content.

This closes the stale-current-document failure mode, but it is not immutable whole-corpus staging, one atomic release switch, or rollback to a previously validated generation.

### Fresh seed installations lacked exact sections at the reviewed commit

The reviewed seed import loaded documents and chunks but did not populate the `document_sections` table. The current repository builds and validates parser-detectable sections, exercises fresh PostgreSQL bootstrap and reindex paths, and includes exact-reference/alias fixtures. A rehearsal on the bank's actual corpus size and database remains open.

### Packaging and deployment are incomplete

At the reviewed commit the Python package could not be built and the containers mixed startup, database schema changes, seed import, embedding, and serving. The current repository builds and externally installs wheel/sdist artifacts, separates migration/bootstrap from serving, adds fixed health routes, and supplies a non-root OpenShift starter with digest-only application images, stable selectors, exact Secret references, PostgreSQL CA/`verify-full`, and default-deny egress. Bank-specific egress, IdP, Route, CA and registry values; signing/SBOM/vulnerability acceptance; restore evidence; and a real cluster deployment remain open.

### Observability is instrumented but not yet an operational service

Current tool calls update thread-safe metrics and correlation-aware, content-safe logs; telemetry can use a distinct append-only PostgreSQL identity. Metrics are still process-local and are not exported through an accepted Prometheus/OpenTelemetry pipeline. There are no measured release SLOs, bank retention rules, or alerts proven in the target cluster.

## What is unsafe

### Remote HTTP access

The reviewed Streamable HTTP server listened on all interfaces without a caller boundary. The current application fails closed on non-loopback startup unless exact Host/HTTPS Origin, asymmetric JWT/JWKS, profile-scope, body, rate, and concurrency controls are configured. Limits remain process-local, and actual bank TLS/IdP/ingress integration and global enforcement are unproved.

Operator tools now require a distinct process profile, DSN, scope, and explicit remote opt-in. Job records and privacy-safe audit state are durable in PostgreSQL, and advisory leases serialize mutations. The bank must still prove private network reachability, actual principals/scopes, and one-replica operator operation.

### Database privilege

The current repository removes schema, seed, cache-population, and embedding lifecycle writes from serving; supplies separate schema-owner, ingestion, public, operator, and telemetry roles/grants; requires the expected database and schema owner; enforces `verify-full` transport; detects ACL provenance and effective privilege; and validates every pooled public/operator connection. Repository PostgreSQL tests prove the denial/allow matrix. The unsafe unknown is whether the bank's actual LOGINs, memberships, HBA/TLS policy, role names, and restore/upgrade process satisfy that contract.

### Private query logging

At the reviewed commit, normal INFO logs included truncated tool arguments and result previews. Current tool-boundary logs retain metadata only by default, add correlation, and put content preview behind an explicit warned opt-in. Bank log export, retention, access, and incident-response policy remain open.

### Misleading citations

The system labels fixed 5,000-character text windows as pages. Those are not necessarily pages in the official PDF. Section page fields are not populated. An apparently precise page citation can therefore be wrong.

### Legal currentness

Draft, current, old, and repealed material can all be retrieved without a validated effective-date/status model. The system cannot safely answer “what applies today?” or “what applied on this date?”

### Formula and table dependence

Only two of 318 seeded documents were marked with a formula-aware extraction method by the repository's classifier. Capital, liquidity, IRB, interest-rate-risk, and TFRS 9 work can depend on a sign, denominator, threshold, or table cell; those documents require source-level validation before calculation or audit reliance.

## What is missing

- stable identity for a regulation across legal versions;
- official publication/effective/repeal dates and validated current status;
- amendment, replacement, repeal, consolidation, and citation relationships;
- hierarchical provision IDs such as article/fıkra/bent;
- immutable official source artifacts and hashes;
- physical source-page/table/formula provenance;
- an audit-grade citation that a reviewer can reconstruct to an authoritative source page and legal version;
- immutable whole-corpus staging, validation, one-step publication, and rollback;
- application of the repository database-role/identity contract to the bank's actual LOGINs and proof of bank-cluster public/operator isolation;
- bank-integrated TLS, IdP, ingress-global limits, and authorization evidence;
- named MCP host/model compatibility matrix beyond official reference-client E2E;
- representative Turkish regulatory retrieval and citation benchmarks;
- claim-by-claim answer grounding evaluation;
- validated mappings from provisions to obligations, controls, audit steps, and evidence;
- exported metrics/traces, measured SLOs, backups, restore drills, and bank-sized upgrade procedures;
- clear code/data licensing and provenance policy.

## The five most important findings

1. **The system still cannot prove legal applicability.** It has extraction snapshots, not validated official legal versions, effective intervals, or amendment/repeal/consolidation status.
2. **Citations can look more precise than the source evidence.** Character windows are presented as pages, and authoritative source-page/table/formula coordinates are not retained.
3. **Consistency is per document, not per released corpus.** Current-hash publication now fails closed, but there is no immutable validated generation, atomic whole-corpus switch, or rollback.
4. **Bank production acceptance remains unproved.** Repository controls are strong, but actual LOGINs, IdP/CA/Route/egress, signed image delivery, backup/restore, SLOs, and OpenShift cluster behavior have not been accepted.
5. **Evaluation cannot yet support client/model selection or audit-reliability claims.** Phase 2 now uses official MCP stdio/HTTP sessions and a fail-closed grader, but the expert gold data is too small and no named live model/client result is accepted.

## What should be done first

### 1. Run a bank integration and recovery acceptance track

Apply the reviewed role/grant assets to actual bank LOGINs; validate shared-cluster naming, memberships, ACL provenance, HBA/TLS and `verify-full`; fill the OpenShift IdP/CA/Route/registry/egress values; deploy separate public/operator workloads; rehearse v0003 on size-matched data; and prove backup/restore before accepting production.

### 2. Define the legal evidence and version contract

Represent a stable regulation/provision identity, official source artifact and hash, publication/effective/repeal state, amendment/consolidation relations, source pages/tables/formulas, quality status, and reviewer validation. Make “current as of date” unavailable until this data is validated.

### 3. Add immutable whole-corpus publication and rollback

Stage and validate one complete document/section/chunk/model manifest, atomically activate it, bind every query to its generation, retain the prior generation, and prove rollback after simulated ingestion and indexing failures.

### 4. Build expert Turkish evaluation and named compatibility evidence

Expand exact-article, cross-reference, acronym, table/formula, currentness, negative, tool-argument, and claim-grounding cases. Have the owner validate judgments, then run the intended Claude, Codex, GPT/GPT-OSS, and LM Studio host/model versions through the same official MCP benchmark.

### 5. Make operations measurable

Export metrics/traces without query text, define numeric availability/publication-lag/RPO/RTO targets, configure alerts and retention, add signed-image/SBOM/vulnerability gates, and schedule upgrade and restore drills with retained evidence.

## What this project could realistically become

With the roadmap implemented, BDDK MCP could become a strong local or enterprise regulatory evidence service:

- reliable retrieval of Turkish banking provisions;
- reproducible “as of date” research;
- exact, reviewable citations;
- model-independent MCP access;
- controlled comparison of Claude, GPT-family, GPT-OSS, Codex, LM Studio, and other host/model combinations;
- traceable links from provisions to obligations, controls, audit tests, and required evidence;
- a validated regulatory knowledge layer that accumulates reviewed relationships rather than merely adding more chunks.

That outcome does not require an immediate graph database or large distributed platform. PostgreSQL, pgvector, immutable source artifacts, typed relations, strong evaluation, and disciplined publication are enough for the next major version.

## Commercial-use objective

The current MIT license expressly permits commercial use, sublicensing, and sale. No technical control in a locally distributed open-source copy can reliably prevent reuse. Already distributed MIT rights cannot realistically be withdrawn by adding a future runtime check.

The project owner should obtain legal advice before further releases and separate:

- code licensing;
- regulatory-source and derived-data rights;
- validated knowledge/control packs;
- hosted/enterprise access terms.

Authentication, authorization, entitlements, and contracts can control a hosted or private operator service. They cannot make an MIT local copy noncommercial.

## Review limitations

The initial review used an immutable archive of the exact origin/main commit and did not inspect production infrastructure. The owner's initial checkout had pre-existing Git deletion entries for tracked files; the review did not create, restore, or commit those entries, and later implementation used a separate clean worktree. That historical provenance does not mean current source files are deleted.

The current repository has focused PostgreSQL privilege/lifecycle/publication tests, a dedicated actual-LOGIN/ACL CI contract, official MCP stdio/HTTP tests, package-install verification, and mandatory rendered OpenShift contract tests. It still has no evidence from the bank's cluster or bank LOGINs, live upstream production operation, intended named clients/models, GPU workload, size-matched migration, or backup/restore drill. No secret values were inspected or exposed.

Owner clarifications establish that the corpus is an intentional job-specific selection, the owner will validate regulatory content and administer the data layer, source/data use is accepted by the owner, and deployment is expected immediately on a bank's on-premises OpenShift AI environment. Exact client versions, tenant/private-document needs, bank platform values, and measurable “immediate” freshness/availability/recovery targets remain to be defined. Until then, the safe deployment assumption is single-tenant, private network, separate public/operator planes, and no private-document ingestion.

For detailed evidence and the implementation plan:

- [REPOSITORY_REVIEW.md](REPOSITORY_REVIEW.md)
- [GAP_REGISTER.md](GAP_REGISTER.md)
- [ROADMAP.md](ROADMAP.md)
- [TARGET_ARCHITECTURE.md](TARGET_ARCHITECTURE.md)
