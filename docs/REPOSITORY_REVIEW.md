# BDDK MCP Repository Review

Review date: 2026-07-14
Reviewed commit: **5684a34c10e6d90bc22d6ab2a6466944afb6bf81**
Repository: https://github.com/omercagatay/bddk-mcp

## Executive finding

BDDK MCP is a capable pre-production regulatory retrieval prototype with unusually good foundations for its size: a modular FastMCP implementation, multi-source document ingestion, deterministic document-quality checks, section-aware chunking, PostgreSQL full-text search, pgvector semantic search, an extensive unit suite, and explicit grounding instructions.

The reviewed commit is not ready for production regulatory reliance or an Internet-facing enterprise deployment. At that commit, the documented stdio client command started an MCP server with zero registered tools; Streamable HTTP had no authentication or explicit Host/Origin protection; operator functions were separated only by a process-wide environment flag; citation locations were not authoritative source pages; the data model could not determine the applicable version or effective state of a provision; seed and indexing workflows could replace or diverge from canonical content; the package could not be built; and the end-to-end benchmark did not use this server's MCP endpoint.

The appropriate near-term direction is stabilization, not a rewrite. Preserve the ingestion, storage, parsing, retrieval, and modular tool foundations; establish trustworthy runtime, security, versioning, citation, evaluation, and release contracts around them.

## Implementation progress overlay — 2026-07-16

This is the authoritative current-working-tree checkpoint. The detailed review below remains an intentionally preserved snapshot of commit **5684a34c10e6d90bc22d6ab2a6466944afb6bf81**; statements in that historical layer do not override this table. **Complete** means implemented and covered by a focused repository contract, not accepted by a bank. **Partial** means a useful control exists with an explicit residual boundary. **Open** means the stated regulatory or operational capability is absent. Confirmed implementation, reasonable inference, and external unknowns are kept separate.

| Status | Current result | Evidence and residual boundary |
|---|---|---|
| Complete | MCP runtime, capability inventory, and transports | The packaged CLI serves exactly 15 public tools or 15 public plus 14 operator additions—29 total. `resolve_regulation_status` is public; `check_bddk_updates` is operator-only. Both profiles register one path-free resource, `bddk://corpus/active-release`, and zero prompts. Unknown arguments fail closed; stable errors do not expose raw exceptions. Official MCP clients exercise installed stdio and Streamable HTTP initialize/list/resource-read/call flows. Evidence: **bddk_mcp/tools/registry.py:21-71,172-176; bddk_mcp/resources.py:17-75; bddk_mcp/server.py:668-670; tests/test_mcp_stdio_e2e.py; tests/test_mcp_http_runtime.py:35-57**. Named Claude, Codex, GPT, GPT-OSS and LM Studio release certification remains open. |
| Complete | Application HTTP security baseline | A non-loopback listener requires exact Host and HTTPS Origin allowlists, a complete asymmetric JWT/JWKS profile, the correct public/operator scope, and explicit operator exposure. The composed FastMCP application exposes RFC 9728 protected-resource metadata at `/.well-known/oauth-protected-resource/mcp`, and a 401 challenge identifies that URL through `resource_metadata`. Token, body, concurrency, and per-process rate bounds are enforced; fixed liveness/readiness routes disclose no corpus data. Evidence: **bddk_mcp/http_security.py; bddk_mcp/transport_tls.py; bddk_mcp/server.py; tests/test_mcp_http_runtime.py; tests/test_http_security.py; tests/test_server_tls.py**. Bank IdP registration/flows, ingress, CA, and global rate-limit acceptance remain external. |
| Complete | Explicit database lifecycle and repository-level least privilege | Serving performs no DDL, seed import, synchronization, embedding backfill or release publication. The checksum ledger ends at schema v6: v4 supplies eleven owner-only legal-curation tables and the validated-citation view, v5 supplies release/activation/epoch state, and v6 supplies the security-definer legal-status resolver. Exact target/TLS/LOGIN/membership/ACL/catalog checks and per-connection admission fail closed. Schema owner, ingestion, release publisher, public, operator and telemetry are distinct database profiles. The v4 legal subset remains attested at 69 constraints and 21 indexes. Evidence: **bddk_mcp/migrations/runner.py:22-37; bddk_mcp/db_lifecycle.py; bddk_mcp/db_identity.py; bddk_mcp/catalog_integrity.py; deploy/postgres/**. Disposable PG17 contracts are repository evidence; bank-created principals, HBA and DBA execution remain external. |
| Complete | Durable single-replica operator jobs and corpus coordination | Operator jobs persist privacy-safe state in `bddk_operator.operator_jobs`. A session-scoped job-admission lease and a distinct transaction-scoped corpus-mutation lock serve different purposes; sanctioned writers and the publisher share the transaction key. Recovery handles abandoned running and cancel-requested work. Evidence: **bddk_mcp/corpus_coordination.py:1-40; bddk_mcp/jobs/postgres.py; bddk_mcp/jobs/manager.py; tests/test_bulk_write.py:118-231**. Execution still occurs in-process; the supplied OpenShift operator remains one `Recreate` replica and is not a distributed workflow engine. |
| Complete | Bounded outbound acquisition baseline | Regulatory HTTP paths enforce exact approved HTTPS hosts, public-address DNS preflight, redirect revalidation, streamed decoded-body limits, bounded retries, archive limits, sanitized logs/errors, and hardened Office XML parsing. Evidence: **bddk_mcp/core/outbound_http.py; bddk_mcp/ingest/client.py; bddk_mcp/ingest/data_sources.py; bddk_mcp/ingest/doc_sync.py; tests/test_outbound_http.py; tests/test_doc_sync.py**. DNS validation and socket connection are not atomic; platform egress and malware/source-authenticity controls remain required. |
| Complete repository mechanism; governed release blocked | Active-corpus publication and strict serving | V5 persists append-only release and activation evidence, maintains a statement-level mutation epoch over 17 corpus tables, and exposes an active view only when the latest activation epoch still equals current state. The independent release publisher regenerates deterministic derivatives and proves exact database membership before activation. In strict mode, local-corpus tool calls hold same-release reader leases and perform a post-call check; OpenShift enables strict mode while local research defaults it off. Evidence: **bddk_mcp/migrations/v0005_corpus_release_publication.py:15-124,831-1056; bddk_mcp/ingest/seed.py:1207-1300; bddk_mcp/corpus_serving.py:29-35,104-187**. Immutable retained corpus generations and rollback/reactivation remain absent. The tracked manifest declares 8,286 chunks but current-profile regeneration produces 9,675, so strict publication refuses it until regeneration, independent review and re-signing (**seed_data/corpus_scope.yml:31-40; bddk_mcp/ingest/seed.py:383-405,657-698**). |
| Partial | Retrieval consistency, legal resolution and structured evidence | Document text/sections publish transactionally and chunks require current content/profile publication. Six retrieval tools expose typed evidence. Citation v1 reconstructs one exact normalized range only through the validated v4 view. V6 adds a public resolver that returns exactly one validated authoritative version for a requested date or abstains; it never infers currentness from recency or missing repeal. Evidence: **bddk_mcp/citations.py; bddk_mcp/tools/legal_status.py:27-105; bddk_mcp/migrations/v0006_legal_status_resolver.py**. The path is proven with synthetic data, not a real authoritative regulation family. |
| Complete as harness; scores remain exploratory | Reproducible model and Phase 2 baseline | Embedding dimension and default model revisions are pinned. Phase 2 launches the real MCP server, validates the selected manifest, reads the active release through the evaluated session, requires exact manifest ID/SHA equality, makes every tool call on that same session, and rechecks the release before returning. Audit artifacts are bounded and redacted. Evidence: **benchmark/phase2_e2e.py:304-350,531-570,657-668,785-855; benchmark/run.py:95-102; benchmark/report.py:17-31**. Ordinary Phase 1/2 output is explicitly exploratory; no named model/client result is accepted. |
| Partial | Packaging, recovery, OpenShift and operational objectives | Recovery evidence schema v2 covers 29 managed relations and the release activation sequence, rejects all six application DSNs for recovery administration, provisions/verifies six LOGIN profiles, and requires exact active-release and sequence equality (**bddk_mcp/operations/recovery.py:60-106,555-603,1305-1414**). OpenShift separates schema migration, ingestion bootstrap and release publication into three Jobs and distinct ServiceAccounts/Secrets (**deploy/openshift/serviceaccounts.yaml; deploy/openshift/jobs/**). A versioned eight-metric bank-on-prem OpenShift objectives contract exists, but every target/window is null and unapproved and production eligibility fails closed (**docs/decisions/operational-objectives.v1.yml; bddk_mcp/operational_objectives.py:333-501**). No retained bank PITR, cluster, approved-objective or signed-promotion evidence exists. |
| Partial technical capability | Regulatory knowledge and audit-grade legal evidence | V4 models stable instruments, separate source blobs/acquisitions, versions, events/status assertions, provisions, occurrences and append-only review decisions; v6 resolves validated as-of status or abstains. A separate legal-release verifier re-hashes retained authoritative source bytes, acquisition records, page text/mappings and Citation excerpts and verifies signed predecessor history. Development uses a caller-supplied latest hash; signed-policy mode supplies the approved head, and bank-policy mode pins that policy's SHA/version and deployment scope. `PageMappingProof` v2 binds declared reviewer owners to each checkpoint/artifact (**benchmark/legal_release_evidence.py: PageMappingProof and validate_legal_release_evidence; benchmark/evaluation_trust_policy.py**). No real non-fixture family, retained evidence bundle, bank-authenticated curator/reviewer action, bank-owned root/promotion evidence, or provision-to-control/workpaper layer exists. |
| Partial, deliberately non-release | Corpus and expert-evaluation release governance | Release use requires four signed evidence layers: measured corpus manifest, expert dataset, legal-curator attestation over the validated Citation pack, and legal-release checkpoint over retained evidence/history. A separate schema-v2 policy binds five release identities with documented canonical/raw semantics, four distinct declared signer-owner IDs, forward rotation/revocation, deployment scope, and time-bounded/revocable reviewer-owner assertions. Distinct IDs and declared event times do not prove separate human custody or signing time. A passed preflight still states `bank_authorization_verified: false` because bank root/RBAC promotion and human review are external, and `model_scores_authorized: false` because expert-dataset execution is not implemented (**benchmark/expert_evaluation.py; benchmark/legal_release_evidence.py; benchmark/evaluation_trust_policy.py; benchmark/release_preflight.py; benchmark/README.md: Hash and version semantics**). The checked-in corpus is unsigned/unmeasured and chunk-stale; the 20-case set remains draft, unannotated, unadjudicated, unapproved and unmapped to verified real evidence. |

### Implementation-checkpoint ratings

These ratings measure the current working tree against the same high-stakes scale; the historical baseline ratings remain below.

| Dimension | Current rating | Reason for movement or hold |
|---|---:|---|
| Overall repository maturity | 3/5 | The repository is now a coherent engineering beta with explicit lifecycle and security boundaries. It is not yet an audit-grade regulatory knowledge product. |
| Production readiness | 2/5 | Repository controls are materially stronger, but bank-integrated TLS/IdP/egress, signed application delivery, retained backup/PITR evidence, approved objectives, and cluster acceptance remain unproved. |
| MCP implementation | 4/5 | Official transports, strict profiles/contracts, stable errors, authentication, and E2E tests are strong. Named-client/version coverage and fully uniform structured outputs remain. |
| Retrieval quality | 3/5 | Hybrid retrieval, publication guards, a synthetic legal-version pilot, and exact normalized-range Citation v1 are credible; authoritative real-family currentness, retained source bytes/pages, and expert-approved Turkish evaluation are not. |
| Security | 3/5 | Application, database-role/ACL, secure-transport, acquisition, and starter-platform controls now fail closed in important paths; local disposable actual-LOGIN/ACL tests pass. Bank-specific egress, IdP/CA/Route, bank LOGIN/DBA evidence, signed-image, recovery, and formal security testing remain release blockers. |
| Testing and evaluation | 3/5 | Broad unit/PostgreSQL 17/protocol/package/legal-pilot/recovery-policy/supply-chain-contract coverage exists. Retained bank restore evidence, expert-dataset execution, live models, expert judgments, load, and bank OpenShift acceptance are missing. |
| Documentation | 4/5 | The repository now has extensive evidence-based architecture, security, deployment, testing, and roadmap documentation; external runbooks and measured acceptance evidence remain. |

## Scope, method, and evidence rules

The review covered the complete committed repository: source, tests, seed corpus, benchmark code, CI, containers, deployment manifests, configuration, documentation, packaging, and license.

Initial-checkout provenance: before the review began, the owner's original
checkout showed deletion entries for every tracked file. Those pre-existing,
user-owned entries were not created, restored, or committed by the review. The
commit-scoped analysis and tests below therefore ran against an immutable
archive of the reviewed commit at **/tmp/bddk-mcp-review.GHa4FE**. Subsequent
roadmap implementation used a separate clean worktree; the initial deletion
state is historical provenance, not a claim that the current implementation
worktree has deleted source files.

Evidence labels used below:

- **Confirmed**: directly observed in source, runtime introspection, seed data, or an executed check.
- **Inference**: a likely consequence supported by confirmed evidence but not measured in a deployed environment.
- **Unknown**: cannot be answered from the repository and requires deployment, legal, domain-owner, or client testing evidence.

Line references are commit-scoped and use **path:line-line** notation.

### Owner clarifications recorded after the review

The project owner supplied the following deployment and governance decisions on 2026-07-14:

- no external endpoint protection is currently confirmed;
- the project owner is accountable for database administration and for regulatory/extraction validation;
- the 318-document corpus is an intentional job-specific selection, not a claim of exhaustive BDDK coverage;
- source and derived-data usage rights are accepted by the owner as adequate, although this review did not independently verify the legal basis;
- the target environment is a bank's on-premises OpenShift AI platform; exact MCP client/host/model versions are not yet selected;
- multi-tenancy and confidential/private-document scope are undecided, so the safe interim design is single-tenant with no private-corpus capability;
- freshness, availability, and recovery are expected to be “immediate,” but that term must be converted into measurable SLO, RPO, and RTO values before production acceptance.

These clarifications reduce governance ambiguity but do not by themselves
change maturity. At the reviewed commit, absent endpoint protection made the
HTTP/authentication findings P0. The 2026-07-16 overlay records the repository
controls added since then; actual bank ingress, identity, CA, egress, rate and
network enforcement remain release acceptance gates rather than presumed
controls.

## Baseline maturity ratings at the reviewed commit

The scale is fitness for the stated high-stakes regulatory use, not code volume:

1. exploratory or unsafe; 2. functional prototype; 3. coherent beta; 4. production-ready; 5. mature and independently validated.

| Dimension | Rating | Rationale |
|---|---:|---|
| Overall repository maturity | 2/5 | Coherent prototype with strong components, but several core user journeys and correctness contracts are unverified or broken. |
| Production readiness | 1/5 | Remote serving, migration, corpus publication, health, backup, and least-privilege boundaries are not production-safe. |
| MCP implementation | 2/5 | Uses the official SDK and current transports, but documented stdio startup is empty, schemas/outputs are weak, and protocol E2E tests are absent. |
| Retrieval quality | 2/5 | Hybrid retrieval and section-aware chunking are promising; authoritative citations, currentness, version relations, and representative evaluation are missing. |
| Security | 1/5 | No HTTP identity, authorization, inbound throttling, or explicit Origin/Host policy; operator isolation is not an authorization boundary. |
| Testing and evaluation | 2/5 | 526 tests pass locally, but DB/GPU paths skip and the model benchmark is too small and does not exercise the actual MCP transport. |
| Documentation | 2/5 | Bilingual and broad, but materially wrong about startup, tool counts, offline behavior, and licensing. |

## Historical reviewed-commit evidence begins here

The inventory and assessment sections below preserve evidence about commit **5684a34c**. They are not current-state claims unless a paragraph explicitly references the 2026-07-16 implementation overlay. Where an older post-review checkpoint conflicts with that overlay, the overlay controls.

## Phase 1 — Repository inventory

### Main purpose

The project exposes Turkish BDDK/BRSA regulatory catalog, locally stored documents, document sections, bulletins, trends, and update monitoring through MCP tools. Its implementation combines live public-source lookups with a PostgreSQL-backed local regulatory corpus and hybrid retrieval.

Evidence:

- Package description: **pyproject.toml:1-7**
- Server grounding and citation instructions: **bddk_mcp/server.py:32-52**
- Tool modules: **bddk_mcp/tools/search.py**, **documents.py**, **sections.py**, **bulletin.py**, **analytics.py**, **sync.py**, **admin.py**

### Current architecture

~~~mermaid
flowchart LR
    Client[Claude / Codex / GPT host / LM Studio / MCP client]
    Client -->|stdio or Streamable HTTP /mcp| Server[FastMCP server]
    Server --> Registry[Public tool modules]
    Server -->|BDDK_ADMIN_TOOLS=true| Operator[Operator tool modules]
    Registry --> API[BddkApiClient]
    Registry --> Docs[DocumentStore and section index]
    Registry --> Search[VectorStore hybrid retrieval]
    Operator --> Ingest[Sync, extraction, OCR, quality and backfill]
    API --> Sources[BDDK and mevzuat.gov.tr]
    Ingest --> Sources
    Docs --> PG[(PostgreSQL)]
    Search --> PGV[(PostgreSQL + pgvector + unaccent)]
    Ingest --> PGV
    Seed[Bundled seed_data] -->|startup import| PGV
~~~

### Entry points and lifecycle

- **server.py:1-10** is a compatibility shim exporting the package-level FastMCP object and main function.
- **bddk_mcp/server.py:54-60** creates a global FastMCP instance with host 0.0.0.0 and stateless HTTP.
- **bddk_mcp/server.py:63-107** creates the HTTP client, asyncpg pool, document store, and BDDK client.
- **bddk_mcp/server.py:123-133** registers public tools and conditionally registers sync/operator modules.
- **bddk_mcp/server.py:135-152** invokes seed import.
- **bddk_mcp/server.py:170-197** owns startup and teardown.
- **bddk_mcp/server.py:200-247** selects Streamable HTTP or stdio.

**Confirmed defect:** README and .mcp.json use **uv run mcp run server.py** (**README.md:104-109,323-328; .mcp.json:4-9**). That SDK command imports the exported FastMCP object but does not execute the project's startup function, where dependencies and tools are registered. Runtime reproduction returned **0 tools, 0 resources, and 0 prompts**. Direct **uv run python server.py** reaches the custom lifecycle, but no official-client subprocess E2E test proves that route.

### MCP transports, tools, resources, and prompts

Supported by the custom main path:

- stdio
- Streamable HTTP at the SDK default **/mcp** endpoint

Legacy HTTP+SSE is not implemented. It is no longer a required current transport, but compatibility with older client versions is unknown.

Runtime introspection found **15 public tools**:

1. search_bddk_regulations
2. search_bddk_institutions
3. search_bddk_announcements
4. search_document_store
5. get_bddk_document
6. get_document_history
7. get_document_section
8. search_document_sections
9. get_bddk_bulletin
10. get_bddk_bulletin_snapshot
11. get_bddk_monthly
12. analyze_bulletin_trends
13. get_regulatory_digest
14. compare_bulletin_metrics
15. check_bddk_updates

With **BDDK_ADMIN_TOOLS=true**, the total is **26**, adding:

1. document_store_stats
2. bddk_cache_status
3. refresh_bddk_cache
4. sync_bddk_documents
5. trigger_startup_sync
6. document_health
7. health_check
8. bddk_metrics
9. backfill_degraded_documents
10. backfill_status
11. document_quality_report

The README and benchmark documentation claim 16 public plus 10 operator tools and incorrectly classify **bddk_cache_status** as public (**README.md:39-64,258-283; benchmark/README.md:5-45**). Code and tests gate it as operator-only (**bddk_mcp/tools/bulletin.py:156-185; tests/test_tools_bulletin.py:12-38**).

At the reviewed commit, no MCP resources, resource templates, or prompts were registered. This was valid because these capabilities are optional, but canonical regulation resources and reviewed workflow prompts could add product value. The current overlay supersedes this historical finding: one active-release resource and zero prompts are now registered.

### Retrieval pipeline

The primary local retrieval path is:

1. tool validation and cache handling in **bddk_mcp/tools/search.py:304-410**;
2. dense and PostgreSQL FTS queries in **bddk_mcp/store/vector_store.py:931-1067**;
3. reciprocal-rank fusion and optional reranking in **vector_store.py:1071-1198**;
4. best-chunk-per-document result formatting in the tool layer;
5. exact local document or section retrieval through **documents.py** and **sections.py**.

Strengths:

- multilingual-e5 embeddings;
- PostgreSQL FTS with unaccent;
- dense/lexical fusion;
- section- and token-aware chunking;
- optional cross-encoder reranking;
- local-only full-document retrieval;
- quality warnings attached to degraded content.

Limitations:

- optional reranker is disabled by default;
- Turkish query relaxation is hand-written rather than morphologically evaluated;
- thresholds and weights are hand-tuned without a representative regulatory benchmark;
- results collapse to one chunk per document;
- no effective-state or as-of-date filter exists;
- no deterministic structured citation object exists.

Evidence: **bddk_mcp/core/config.py:40-108; bddk_mcp/store/vector_store.py:228-390,505-553,635-692,931-1198; bddk_mcp/tools/documents.py:66-283**.

### Document ingestion and normalization

The ingestion path includes:

- BDDK API/site clients and multiple source strategies: **bddk_mcp/ingest/client.py**, **data_sources.py**
- HTML and embedded-document discovery: **doc_sync.py**, **html_extractor.py**
- MarkItDown PDF/DOCX extraction and degraded-state handling
- optional OCR abstraction and Chandra implementation: **bddk_mcp/ocr/base.py**, **chandra.py**
- deterministic Markdown sanitation and quality checks: **bddk_mcp/quality/markdown_quality.py**, **quality_scan.py**
- seed export/import and embedding backfill: **bddk_mcp/ingest/seed.py:108-416**
- section extraction and legal-reference parsing: **bddk_mcp/store/section_index.py**, **legal_ref.py**

Seed corpus inventory at the reviewed commit:

- 318 documents
- 8,286 chunks
- 318 decision-cache entries
- 249 MarkItDown extractions
- 28 degraded MarkItDown extractions
- 27 HTML-parser extractions
- 313 documents classified clean and 5 warnings by the current quality scorer

The five warning documents were **1043**, **907**, **903**, **mevzuat_16290**, and **1045**. The score was 99.5692/100, but that score measures the implemented anomaly rules, not legal accuracy, completeness, currentness, or citation fidelity.

Section metadata appeared on chunks for 70 of 318 documents. Coverage varies by document type and is incomplete even within structurally relevant categories. Formula-aware extraction metadata appeared on only two documents under the repository classifier.

Two bootstrap/quality disconnects are more serious than those aggregate counts suggest:

- the seed importer loads documents and vector chunks but does not populate **document_sections** (**bddk_mcp/ingest/seed.py:236-349**), and startup does not reindex them; a fresh seeded deployment can therefore return no exact article until the manual reindex script is run;
- **config/quality_failures.yml:1-34** records 11 known failures, while the runtime known-failure set is empty (**bddk_mcp/quality/markdown_quality.py:15,109-138**); the scan script reads but does not apply that registry (**scripts/scan_document_quality.py:163-170**), which is why the implemented scorer can report zero failures.

Search and section responses also omit the quality warnings emitted by full-document retrieval (**tools/search.py:375-395; tools/sections.py:125-312; tools/documents.py:199-217**).

### Canonical data and versioning

The document record stores identity, title, category, decision date/number, source URL, content, hash, extraction method, total pages, and file size (**bddk_mcp/store/doc_store.py:27-43,101-230**).

The repository has a **document_versions** table and archives previous Markdown when the ordinary store path detects a content-hash change (**doc_store.py:294-364**). However:

- the current version is not modeled as an immutable version row;
- versions are synthetic sequence numbers, not source-issued legal versions;
- history exposes hashes, lengths, and timestamps but cannot retrieve or compare historical text (**doc_store.py:789-833; tools/documents.py:285-310**);
- no promulgation/publication date, effective-from/to, repeal, supersession, amendment, consolidated-text, or legal-status relation exists;
- startup seed import directly overwrites documents and chunks without the ordinary archive path (**bddk_mcp/ingest/seed.py:236-358**);
- categories such as draft or repealed are searchable beside current material without a default legal-status guard.

The current model therefore stores document snapshots, not a canonical temporal regulatory record.

### Citation model

The server instructs models to cite document, page, section, and character range (**bddk_mcp/server.py:32-52**), but the available locations are weaker than the wording implies:

- pagination is produced by slicing Markdown into fixed 5,000-character windows (**bddk_mcp/core/config.py:71-79; store/doc_store.py:406-441**);
- vector results map character offsets to those windows (**vector_store.py:798-860**);
- section records have page_start/page_end fields, but the parser does not populate them (**section_index.py:39-52,103-113**);
- section tools mostly expose character ranges rather than official PDF page coordinates;
- search output is human text rather than a typed citation with source URL, version hash, section, page, and excerpt hash.

Calling these windows “pages” can create audit-grade false precision. A valid citation engine must distinguish official source page, normalized-text location, and display window.

### Database and SQL access

PostgreSQL, pgvector, and unaccent are the core persistence/runtime dependencies. asyncpg pools, placeholders, transactions, and query timeouts are used. No caller-supplied SQL tool exists, and no direct SQL-injection path was confirmed.

Representative parameterized paths:

- document lookup: **bddk_mcp/store/doc_store.py:366-404**
- section FTS: **doc_store.py:568-620**
- vector filters/search: **bddk_mcp/store/vector_store.py:931-999**
- transactional vector replacement: **vector_store.py:696-797**

Safety limitations:

- one application role performs serving, DDL, extension creation, ingestion, cache writes, seed overwrite, and telemetry;
- schema setup and ad-hoc migrations run at serving startup;
- no migration ledger, separate migration job, rollback protocol, or database-role tests exist;
- public and operator processes are not structurally separated;
- document, section, and vector replacement span separate transactions, so an index failure can leave fresh canonical text beside stale chunks (**doc_sync.py:487-525**);
- document retrieval prefers vector chunks before the document store (**tools/documents.py:96-119**), making stale-chunk service possible after that failure.

SQL injection risk is presently low; database privilege and publication-consistency risk is high.

### Configuration and environment handling

Configuration is module-level environment parsing in **bddk_mcp/core/config.py**. It covers database URL, HTTP limits/timeouts, embedding/chunking, hybrid search, reranking, admin exposure, telemetry, sync, OCR, and quality controls.

Problems:

- validation is scattered and import-time;
- unknown transport values silently fall back to stdio (**server.py:210-247**);
- the error message suggests copying .env, but application code does not load an env file;
- .mcp.json contains a hard-coded local directory (**.mcp.json:4-9**);
- package version, server initialization version, benchmark profiles, docs, and runtime surface do not share a single source of truth.

### Authentication, authorization, and state

There is no in-repository HTTP authentication or authorization. **BDDK_ADMIN_TOOLS** controls registration, not caller permissions. There are no identities, scopes, tenant boundaries, row policies, quotas, or per-caller audit trails.

Transport HTTP is declared stateless, but application state is global:

- module search cache: **bddk_mcp/tools/search.py:73-88**
- dependency job/health state: **bddk_mcp/core/deps.py:19-47**
- announcement baseline mutated by a public tool: **bddk_mcp/tools/analytics.py:169-200**
- in-memory metrics and caches are replica-local.

One caller can therefore influence another caller's “new updates” result, and state will diverge across workers.

### Logging and observability

Strengths:

- structured JSON logging is available;
- stdio logs use stderr;
- optional database telemetry is off by default and hashes queries by default.

Gaps:

- the standard tool wrapper logs truncated arguments and the first 200 characters of result text at INFO (**bddk_mcp/tools/tool_logging.py:34-85,95-143**);
- query and keyword fields are not classified as sensitive;
- JSON logs explicitly include tool_args, result_preview, and error_message (**bddk_mcp/core/logging_config.py:28-63**);
- metrics record methods exist but are not invoked from the tool boundary (**bddk_mcp/observability/metrics.py:20-43; tools/admin.py:78-104**);
- no OpenTelemetry, Prometheus endpoint, trace propagation, corpus-freshness SLO, or public liveness/readiness endpoint exists;
- the health tool is operator-only and is not a container/orchestrator health route.

### Testing, benchmark, and CI

Confirmed strengths:

- 613 test cases collected, with 526 passing in the isolated default environment;
- unit coverage spans clients, stores, retrieval, ingestion, OCR abstractions, tools, quality, observability, and benchmark helpers;
- Ruff lint and format checks pass;
- CI uses Python 3.12 and 3.13 and provisions pgvector PostgreSQL (**.github/workflows/ci.yml:12-71**).

Confirmed gaps:

- no official MCP ClientSession E2E test over a subprocess or HTTP;
- raw smoke scripts use protocol revision 2024-11-05 and are not CI gates (**scripts/mcp_smoke.py:33-44; mcp_fetch_full.py:38-50**);
- documentation tests assert literal counts rather than runtime registration (**tests/test_docs_tool_surface.py:12-30**);
- no coverage threshold, type checker, package-build gate, image build/run, security scanner, migration-upgrade test, load test, restore test, or client matrix;
- DB fixtures skip when the database is missing, making an accidentally unprovisioned CI job look green.

The benchmark has material validity defects:

- Phase 1 tests static OpenAI-compatible Chat Completions function schemas, not MCP discovery (**benchmark/phase1_tools.py:36-58**).
- At the reviewed commit, the static benchmark defined 23 schemas, unlike the 15/26 runtime profiles. The current tree derives all 29 function schemas from the canonical 15-public/14-operator registry, and Phase 2 discovers the live MCP contract.
- At the reviewed commit, Phase 2 claimed stdio MCP but POSTed to a non-existent **/call-tool** route (**benchmark/phase2_e2e.py:1-10,107-121**). The current runner uses the official `ClientSession` over stdio or **/mcp** and fails closed on protocol/tool errors.
- At the reviewed commit, HTTP errors were converted to strings and could miss transport failure; current benchmark tests reject that behavior.
- source-trace scoring examines tool results rather than final-answer citations.
- the code grader checks numeric/date recall and does not penalize unsupported added claims (**benchmark/graders.py:20-66**).
- at the reviewed commit, absent Anthropic credentials silently changed the grading method to the weak code grader (**graders.py:91-126**); the current runner records an explicit unavailable/non-comparable result instead.
- only three gold cases exist (**benchmark/gold_cases.yml**).
- the NLI set has 30 pairs against a stated target of 500 and lacks document/section/hash provenance (**data/bddk_nli/metadata.json:1-16**).

The transport/grader defects are corrected, but no model-selection conclusion
should be published from the current Phase 2 data: three gold cases and the
small NLI set are not representative, and no external live-model comparison
has been accepted.

### Deployment and operations

Deployment artifacts include:

- standard Dockerfile with bundled seed data;
- Docker Compose with pgvector;
- Railway manifest and Procfile;
- a Hugging Face Spaces Dockerfile.

Confirmed gaps:

- HTTP binds 0.0.0.0 with no auth or explicit Origin/Host policy;
- Compose publishes application and PostgreSQL ports with known development credentials (**docker-compose.yml:1-30**);
- containers run as root and use mutable base/uv tags;
- no Docker health check or application health route;
- no backup/PITR/restore plan;
- no versioned migrations or release job;
- no resource limits, SLOs, alerting, compatibility matrix, rollback or upgrade runbook;
- Railway's /app/data volume is unrelated to the PostgreSQL corpus;
- **Dockerfile.spaces** omits seed_data, defaults to a non-existent db host, and enables auto-sync, while startup sync refuses an empty decision cache (**Dockerfile.spaces:12-28; tools/sync.py:126-153**).

Startup imports seed and can synchronously embed 8,286 chunks before serving. Hash inequality can cause a fresher live corpus to be overwritten with bundled content; the seed path does not archive the replaced version (**ingest/seed.py:142-358**). This is a regulatory-correctness and availability issue, not merely slow startup.

### Packaging and dependency management

Strengths:

- Python 3.12–3.13 support is stated;
- uv.lock pins a reproducible graph;
- runtime and GPU dependency groups are partially separated.

Gaps:

- **uv build** fails because setuptools discovers data, config, bddk_mcp, and seed_data as multiple top-level packages;
- no explicit build backend/package discovery or console script exists;
- the MCP dependency lower bound of 1.0.0 understates APIs the code uses;
- heavy benchmark/OCR/model dependencies remain in the default runtime;
- model and container revisions are not immutably pinned;
- CI lacks supply-chain, SBOM, package installation, and image gates.

### Documentation and licensing

README is bilingual and gives useful setup, corpus, tool, and grounding context. However it:

- documents the broken stdio command;
- reports the wrong public/operator split;
- implies the server is wholly local/offline even though several public tools call live BDDK sources (**README.md:217-224,436-443**);
- says there is no license file (**README.md:461-463**) although **LICENSE** contains MIT.

MIT explicitly permits commercial use, modification, sublicensing, and sale (**LICENSE:1-20**). That directly conflicts with the objective of preventing unauthorized corporate or commercial reuse. Technical restrictions in distributed local code are removable. Future hosted access can be controlled through identity, contracts, quotas, and private datasets, but changing future licensing requires counsel and cannot reliably revoke rights in already distributed MIT copies. Regulatory-document redistribution and derived-dataset rights are also unknown.

### External services and runtime dependencies

Confirmed external dependencies include:

- PostgreSQL with pgvector and unaccent;
- BDDK APIs/pages and mevzuat.gov.tr;
- Hugging Face/model-download infrastructure for multilingual-e5 and optional reranker;
- MarkItDown/PDF extraction stack;
- optional Chandra OCR/CUDA;
- optional Anthropic grader API;
- OpenAI-compatible local model endpoints for benchmark phases;
- Docker/Railway/Hugging Face Spaces deployment surfaces.

There is no evidence that production availability, quotas, terms, model revisions, or upstream change monitoring are contractually managed.

## Phase 2 — Architecture assessment

### MCP specification and compatibility

At the reviewed commit, `uv.lock` pinned MCP SDK **1.27.0**; it understood the 2025-11-25 protocol revision and both standard transports. The current worktree locks **1.28.1**, whose `LATEST_PROTOCOL_VERSION` remains 2025-11-25. Prompts remain absent and optional; the current server now registers one reviewed active-corpus-release resource. The following gaps describe the reviewed commit; the implementation overlay above records the later transport, authorization, resource, schema, annotation, error, and protocol-test controls:

- HTTP Origin validation is required by the current transport specification, but explicit protection is absent and SDK auto-protection is disabled by host 0.0.0.0;
- no HTTP authorization implementation at the reviewed commit;
- no tool annotations for read-only, destructive, idempotent, or open-world behavior;
- all 26 inspected tools lacked annotations;
- runtime input properties lacked property-level descriptions and generally omitted schema-visible bounds/enums;
- outputs are effectively a single result string;
- errors are returned inconsistently as ordinary content, “no results,” or exceptions instead of a stable tool-execution error contract;
- no protocol/client compatibility tests.

Authoritative comparison:

- MCP transport specification: https://modelcontextprotocol.io/specification/2025-11-25/basic/transports
- MCP authorization: https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization
- MCP tools: https://modelcontextprotocol.io/specification/2025-11-25/server/tools

Client implications:

| Client class | Current assessment |
|---|---|
| Claude/Codex stdio using checked-in command | Confirmed broken: zero tools. |
| Direct stdio using project main | Plausible but not E2E-tested. |
| Codex/ChatGPT Streamable HTTP | Protocol shape is supported; remote auth and secure deployment contract are missing. |
| GPT-style function-calling host | Benchmark tests a separate static function contract, not the MCP server. |
| LM Studio/local hosts | Version-specific compatibility unknown; weak runtime schemas likely make small-model selection harder. |
| GPT-OSS or other local model | MCP belongs to the host/orchestrator, not the model; no host/model combination is validated. |
| Older HTTP+SSE clients | Not supported; only add legacy support if a measured client requirement exists. |

Codex currently supports stdio and Streamable HTTP MCP server configuration; repository-specific compatibility remains unproven because its checked-in stdio command is broken. OpenAI reference: https://learn.chatgpt.com/docs/extend/mcp.md

### Retrieval and regulatory quality

The retrieval mechanics are stronger than the legal information model. It can locate likely documents and normalized sections, but cannot reliably answer:

- which legal version applied on a given date;
- whether a provision is currently effective;
- what amended or repealed it;
- whether the returned text is original, amended, consolidated, draft, or repealed;
- the exact official page and immutable source artifact supporting a quotation.

The primary failure modes are:

1. current, repealed, and draft texts competing in search;
2. semantic similarity returning a related but legally inapplicable document;
3. stale vector chunks after partial reindex failure;
4. extraction artifacts, tables, or formula loss;
5. a character window reported as a page;
6. section regex missing unusual Turkish drafting structures;
7. amendment/cross-reference relationships being invisible;
8. query-relaxation changing legal terms without a benchmark;
9. model instructions being treated as sufficient grounding enforcement;
10. unsupported synthesis because tool results lack claim-level structured citations.

Hybrid retrieval should be retained, but validated with Turkish expert-reviewed queries and legal relevance judgments before threshold tuning or infrastructure expansion.

### Data and SQL safety

No arbitrary SQL surface or confirmed SQL injection was found. Parameterization and transactions are generally good. The dominant risks are:

- excessive database privilege in the serving process;
- startup DDL and writes;
- no independent migration lifecycle;
- separate transactions for canonical text and derived indexes;
- error/log detail exposure;
- no tenant boundary if private corpora are added.

### Security and access control

High-risk gaps are unauthenticated HTTP, missing Host/Origin protection, environment-only operator exposure, raw audit-query logging, serving-role overprivilege, and unbounded expensive inputs. Medium risks include second-order SSRF through upstream iframe URLs, archive decompression/memory limits, document prompt injection, mutable supply-chain references, and global state.

No committed production secret was confirmed. Example/default credentials are present for local development.

### Software quality

Modularity and naming are generally good. Tool modules, stores, ingestion, quality, OCR, and observability are separated. The main maintainability problems are:

- global singleton construction and import-time configuration;
- startup mixing serving, DDL, seed publication, embedding, and sync;
- duplicated/static tool definitions in docs and benchmarks;
- string-only cross-layer contracts;
- hand-managed schema changes;
- metrics code not connected to the actual tool boundary;
- operator job lifecycle and naming defects, including **trigger_startup_sync** not actually triggering a sync (**tools/sync.py:266 onward**);
- package discovery not configured.

## Phase 3 — Product and domain review

### Workflow support matrix

| Regulatory workflow | Current support | Evidence-based assessment |
|---|---|---|
| Find an applicable regulation | Partial | Catalog and hybrid search find candidates, but cannot prove applicability/currentness. |
| Retrieve a specific article | Partial to good | Section parsing supports Madde, Geçici Madde, İlke, Paragraf, Ek, fıkra, and bent; coverage and page mapping are incomplete. |
| Compare versions | Weak | History metadata exists; historical text retrieval and semantic/legal diff do not. |
| Identify amendments | Weak | No amendment graph or formal relation model. |
| Determine current effectiveness | Unsupported | No effective-from/to or status derivation. |
| Link provisions to audit procedures | Unsupported | No obligation/control/procedure/evidence model. |
| Generate audit control steps | Model-only | An LLM can synthesize text, but the repository supplies no reviewed workflow, control library, or traceable approval state. |
| Provide traceable legal bases | Partial | Document IDs, URLs, sections, hashes, and character ranges exist, but citation objects and authoritative pages/versions do not. |
| Produce auditor/regulator review evidence | Not yet | Missing immutable evidence pack, as-of state, citation verification, extraction confidence, and reviewer validation. |

### Domain-topic coverage

The corpus contains titles and text relevant to TFRS 9, credit risk, PD/LGD/ECL, IRB, ICAAP/İSEDES, liquidity, capital adequacy, and interest-rate risk. Search can locate that material. The repository does not yet provide:

- validated topic taxonomies and aliases;
- definitions tied to exact provisions and versions;
- obligation, threshold, exception, reporting-frequency, or responsible-party extraction;
- cross-framework mappings;
- bank-specific policy/control mapping;
- approved calculation interpretation;
- audit test procedures and required evidence;
- model-risk validation packs;
- currentness opinions.

These workflows are therefore search-assisted research, not encoded domain capabilities.

### Document search versus regulatory knowledge

Today the project is primarily a sophisticated document-search system:

- it stores normalized texts and chunks;
- parses some structural units;
- offers dense and lexical retrieval;
- exposes local exact retrieval and history metadata;
- records quality signals.

It begins to resemble a knowledge system through section metadata, hashes, legal-reference regexes, quality traces, and document versions. It becomes a genuine regulatory knowledge system only when it can represent and validate:

- immutable source artifacts and normalized versions;
- temporal applicability;
- amendment/repeal/consolidation relationships;
- provision-level stable identities;
- cross-document citations;
- obligations, actors, conditions, exceptions, thresholds, and evidence;
- provenance and human validation state;
- reproducible as-of retrieval and evidence packs.

A graph database is not yet justified. Start with relational relation tables and materialized views; add a graph projection only after real multi-hop queries prove value.

## Findings confirmed at the reviewed commit, inferences, and unknowns

The status overlay near the beginning of this document identifies which commit-scoped findings have since been mitigated in the working tree.

### Confirmed at the reviewed commit

- Documented stdio startup registers zero tools.
- Runtime profiles are 15 public and 26 admin tools; docs say 16/26.
- No resources or prompts exist.
- Remote HTTP has no in-repository auth and no explicit Origin/Host policy.
- All inspected tools are string-returning and annotation-free.
- No caller-supplied SQL exists; SQL is generally parameterized.
- Startup owns DDL, seed import, embedding backfill, and serving.
- Seed drift can replace live content without ordinary version archival.
- Fresh seed import does not populate the exact-section table.
- Eleven configured extraction failures are disconnected from runtime classification, and warnings are not uniform across retrieval paths.
- Canonical documents and derived indexes can diverge.
- “Pages” are normalized-text character windows, not source pages.
- No effective-date/amendment/repeal relation model exists.
- Tool logs include raw/truncated user and result text.
- Metrics are not wired into the tool boundary.
- The package build fails.
- Phase 2 calls a route the server does not expose.
- The current gold/evaluation data is too small and weakly traceable.
- The license is MIT and permits commercial use.

### Reasonable inferences

- At the reviewed commit, small/local models were likely to underperform against the sparse runtime schemas compared with the richer static benchmark schemas. The current tree removes that inventory divergence, strengthens schemas, and same-session-binds Phase 2 to an active release, but named host/model behavior and expert-dataset execution remain unevaluated.
- Multiple HTTP workers will still disagree about process-local caches, metrics, and update baselines. Operator job state and corpus-mutation leases are now PostgreSQL-backed rather than replica-local, but execution remains in process.
- An answer can sound well grounded while citing a pseudo-page or legally obsolete text.
- The synchronous first-start embedding workload can cause deployment startup timeouts.
- Enterprise audit queries may leak through platform log retention.
- Current retrieval score thresholds will not transfer reliably across model/corpus revisions without calibration.

### Remaining unknowns requiring investigation

- Exact bank OpenShift AI topology and accepted values for ingress/Route, IdP,
  application/service/PostgreSQL CAs, registry, egress allows, monitoring,
  shared rate limits, SCC and namespace policy. Repository starter contracts
  exist; bank application and acceptance do not.
- Actual bank PostgreSQL LOGINs, direct memberships, ACL provenance, TLS/HBA,
  backups, restore tests, database size and production data. Repository role,
  target-name, `verify-full`, per-connection identity and catalog contracts
  exist, but the opt-in actual-LOGIN proof has not run in the bank environment.
- Whether the bank deployment actually runs the public and operator profiles as
  separate workloads, as the repository now requires.
- Which Claude, Codex, GPT, LM Studio, and local-host versions are target requirements.
- Real client behavior for structured content, output schemas, annotations, instructions, and older transports.
- Completeness and freshness against the owner's intentional job-specific corpus scope; exhaustive BDDK coverage is not currently intended.
- Validation status of extraction, formulas, tables, section parsing, amendments, and legal applicability; the project owner is the designated reviewer.
- Written provenance/licensing record supporting the owner's acceptance of source and derived-data use.
- Contribution ownership and legal options for future licensing.
- Enterprise tenant model and whether private documents will enter the system; single-tenant/no-private-corpus is the interim assumption.
- Numeric availability, source-detection/publication freshness, RPO, and RTO targets that operationalize “immediate.”

## Historical validation record for commit 5684a34

Commands were non-destructive and ran in the immutable commit archive.

| Command or check | Result |
|---|---|
| git status, log, ls-tree, remote verification | Reviewed commit matched origin/main; the owner's initial checkout had pre-existing deletion entries for all tracked files, so review execution used the immutable archive. This is not the state of the later clean implementation worktree. |
| uv sync --frozen --dev | Completed; isolated environment created. |
| uv run ruff check . | Passed. |
| uv run ruff format --check . | Passed; 138 files already formatted. |
| uv run pytest tests/ --collect-only -q | 610 selected, 3 GPU cases deselected; 613 total discovered. |
| uv run pytest tests/ -q --tb=short | 526 passed, 84 skipped, 3 deselected. |
| pytest re-run with -rs | Same result; skip reasons captured. |
| Runtime FastMCP introspection | 15 public tools, 26 admin tools, 0 resources, 0 prompts, 0 tool annotations. |
| Import root shim as mcp CLI would | 0 tools, 0 resources, 0 prompts. |
| Seed document quality scorer | 99.5692; 313 clean, 5 warnings, 0 failures under implemented rules. |
| uv build | Failed: multiple top-level package discovery. |
| uv pip check | Failed: nvidia-cusparselt-cu13 reported built for another platform; likely environment-specific, unresolved. |
| docker compose config --quiet | Passed. |
| Current MCP/OpenAI documentation review | Completed against official sources. |

Skipped tests:

- 77 PostgreSQL-dependent tests: no disposable PostgreSQL test database was available and creating/mutating one was outside this review.
- 7 Chandra tests: optional Chandra package/GPU stack was absent.
- 3 GPU-marked tests: deselected by repository pytest configuration.

Not executed:

- live BDDK/mevzuat calls;
- real database ingestion, migrations, reindex, or seed import;
- Docker image build/run;
- Railway account or deployment inspection;
- vulnerability/container scanning;
- live model benchmark;
- Claude, Codex, GPT, GPT-OSS, LM Studio, or other client integration.

These require external state, credentials, model downloads, database writes, installed clients, or production access. No secret values were read or printed.

## Review conclusion

At the reviewed commit, the repository was a promising research and local-development system whose advertised MCP path, remote boundary, lifecycle, packaging, and evaluation contracts needed stabilization. That conclusion is preserved as historical evidence, not as a description of the current working tree.

The 2026-07-16 implementation is a coherent engineering beta. In addition to the stabilized MCP/security/database foundation, it has a 29-tool registry, one active-release resource, schema v6, append-only corpus activation with mutation epochs, strict per-tool release binding, recovery-v2 evidence over 29 relations and six identities, separate OpenShift migration/bootstrap/publisher Jobs, a four-layer legal/evaluation verifier, and an eight-metric operational decision contract. It is still not ready for audit-grade legal reliance or bank production acceptance: the tracked 8,286-row chunk artifact regenerates as 9,675 rows and cannot pass strict publication; no real authoritative family or retained legal-release bundle exists; model-score execution and bank authorization remain absent; immutable rollback, approved targets, bank recovery/cluster evidence, signed promotion and named-client/model certification remain open.

The detailed prioritized gaps, target design, test strategy, security controls, and first ten implementation issues are in:

- [GAP_REGISTER.md](GAP_REGISTER.md)
- [TARGET_ARCHITECTURE.md](TARGET_ARCHITECTURE.md)
- [ROADMAP.md](ROADMAP.md)
- [TESTING_AND_EVALUATION_STRATEGY.md](TESTING_AND_EVALUATION_STRATEGY.md)
- [SECURITY_REVIEW.md](SECURITY_REVIEW.md)
