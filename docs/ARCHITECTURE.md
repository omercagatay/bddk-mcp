# Architecture at the Reviewed Commit

This document describes the architecture implemented at commit **5684a34c10e6d90bc22d6ab2a6466944afb6bf81**. It is descriptive, not the target design.

## Implementation progress overlay — 2026-07-16

The current diagram and table in this overlay describe the 2026-07-16 worktree. The detailed architecture after the explicit historical marker preserves the reviewed commit. The current architecture remains a modular monolith, but it now has explicit process, lifecycle, privilege, release-publication and evaluation boundaries:

| Status | Architectural delta | Evidence and remaining boundary |
|---|---|---|
| Complete | MCP and process profiles | The packaged CLI selects one registry per process. Public uses only `BDDK_DATABASE_URL`; operator requires its own DSN, authorization scope and remote opt-in. The registry contains 15 public tools plus 14 operator additions—29 total—with strict input schemas and risk annotations. Both profiles register the one path-free `bddk://corpus/active-release` resource and zero prompts. Evidence: **bddk_mcp/tools/registry.py:21-71,172-176; bddk_mcp/resources.py:17-75; bddk_mcp/server.py:668-670; tests/test_mcp_http_runtime.py:35-57**. |
| Complete | Transport and request boundary | Stdio and stateless JSON Streamable HTTP use the official SDK. Non-loopback HTTP requires Host/Origin allowlists, asymmetric JWT/JWKS verification, profile scope, bounded body/token/concurrency/rate admission, and optional validated server TLS. Body reads now have independent first-byte, inter-chunk, and total deadlines, so a slow unauthenticated sender cannot hold an admission slot indefinitely; deadline expiry returns 408. The composed HTTP application exposes RFC 9728 protected-resource metadata at `/.well-known/oauth-protected-resource/mcp`, and its 401 challenge supplies the same URL through `resource_metadata`. Evidence: **bddk_mcp/http_security.py:93-95,439-458,627-629,763-778; tests/test_http_security.py; tests/test_mcp_stdio_e2e.py; tests/test_mcp_http_runtime.py**. The focused HTTP security lane passed 61 tests. Bank IdP registration, ingress-wide admission, and end-to-end authorization-flow acceptance remain external. |
| Complete | Explicit lifecycle and PostgreSQL authorization | `serve` is read-only with respect to schema/corpus lifecycle. The checksum ledger ends at v8: v4 supplies eleven owner-controlled legal-curation tables and the validated-citation view, v5 supplies corpus release/activation/epoch state, v6 supplies the security-definer legal-status resolver, v7 supplies isolated retained-generation storage, and v8 supplies staged release requests plus one-time activation bindings. Legal-table mutation remains owner-only; v8 gives the release verifier the exact read-only access needed for publication evidence. Exact target/TLS/ACL/catalog/connection identities are checked for schema-owner, ingestion, release-verifier, release-publisher, public, operator and telemetry profiles. The verifier may inspect corpus/legal-version state and stage a bounded request but cannot activate or retain; the publisher activates only by request ID and cannot stage or directly publish a claimed verified release. The v4 legal subset remains attested at 69 constraints and 21 indexes; v7 and v8 have separate exact catalog/ACL contracts. Evidence: **bddk_mcp/migrations/v0008_staged_corpus_releases.py; bddk_mcp/migrations/runner.py; bddk_mcp/db_identity.py; bddk_mcp/catalog_integrity.py; deploy/postgres/**. Focused evidence includes 59 PostgreSQL migration/catalog tests, 117 application/role tests, and two actual-LOGIN tests. Bank issuance and RBAC separation of the seven credentials, HBA, and DBA execution remain external. |
| Complete | Durable single-replica operator control plane and writer coordination | `PostgresJobRepository` persists privacy-safe job state. A session-scoped job-admission lease and a distinct transaction-scoped corpus-mutation lock are separate protocols; every sanctioned writer plus v8 staging and activation use the transaction key. Evidence: **bddk_mcp/corpus_coordination.py:1-40; bddk_mcp/migrations/v0008_staged_corpus_releases.py:209,439; bddk_mcp/jobs/postgres.py; bddk_mcp/jobs/manager.py; tests/test_bulk_write.py:118-231**. Runners remain in the operator process; multi-replica dispatch/failover is unsupported. |
| Complete mechanism; governed release blocked | Active corpus publication, strict serving, and immutable retained snapshots | V5 appends release/activation evidence and increments one statement-level epoch on mutations to 17 tracked corpus tables. V8 closes the repository-level publisher-credential bypass in that original flow: `verify-and-stage-corpus-release` uses the independent release-verifier identity to revalidate and stage an expiring request bound to the database-computed state/epoch/profile and verifier revision/image; `activate-corpus-release` uses the publisher identity and supplies only that one-time request ID. Activation rechecks expiry, use, readiness, state, epoch, and profile atomically. The old direct publication routine is not executable by the publisher. V7 separately adds 17 typed retained relations plus generation, inventory, seal, and release-binding metadata. Physical generation, governed release, seal, request, and activation are distinct identities. Evidence: **bddk_mcp/migrations/v0005_corpus_release_publication.py; bddk_mcp/migrations/v0007_retained_corpus_generations.py; bddk_mcp/migrations/v0008_staged_corpus_releases.py; bddk_mcp/corpus_publication.py; tests/test_migrations.py; tests/test_corpus_publication.py**. A principal that can obtain both verifier and publisher credentials, or schema-owner authority, still collapses the separation; bank Secret/RBAC custody is therefore an external gate. The tracked artifact declares 8,286 chunks but regenerates as 9,675 and is rejected before staging. Retention is not generation-bound serving or rollback; H2-02B remains open. |
| Complete repository boundary; capacity acceptance partial | Administrative retention facade | `bddk-mcp retain-corpus-generation --expected-release-id …` uses only the release-publisher database profile, executes one atomic transaction, and emits content-free receipt and storage evidence. It is absent from both MCP registries. Public, ingestion, verifier, operator, and telemetry profiles have no `bddk_retained` schema usage or direct table privileges; the publisher receives only the active/retention-status views and the narrow activate, retain, and storage-inspection security-definer functions. It has no raw corpus or legal-evidence read access. The CLI reports heap/auxiliary/TOAST/index sizes and, when the post-commit observation is available, a cluster WAL interval labelled non-exclusive; otherwise WAL remains `not_measured`. Backup growth remains `not_measured` until a controlled bank/DBA backup supplies it, and capacity must be approved against unique physical generations rather than release-binding count. Evidence: **bddk_mcp/cli.py; bddk_mcp/corpus_generations.py; bddk_mcp/db_identity.py; deploy/postgres/02_grants.sql; tests/test_cli.py; tests/test_postgres_role_assets.py**. |
| Partial | Retrieval, Citation v1 and legal-status resolution | Transactional document/section publication and current-profile chunk guards remain. Citation v1 reconstructs exact normalized ranges through the validated v4 view. V6 exposes an abstention-first public resolver that returns one validated authoritative legal version for a requested date or no conclusion. A separate legal-release verifier can re-hash retained source bytes, acquisition records, page text/mappings and Citation excerpts and recursively verify signed history. Evidence: **bddk_mcp/citations.py; bddk_mcp/tools/legal_status.py; benchmark/legal_release_evidence.py: PageMappingProof and validate_legal_release_evidence**. Only synthetic legal-family evidence exists; no real retained bundle or curator/source authority has been accepted. |
| Partial | Acquisition, supply chain, observability and objectives | Bounded acquisition, safe logs/metrics and isolated telemetry exist. The supply-chain lane binds reproducible artifacts, image descriptor/config, loaded image and SBOM but does not sign or promote. A versioned bank-on-prem OpenShift objectives contract defines eight metrics, yet every target/window is null and unapproved and alert/evidence integrations remain unverified. Evidence: **bddk_mcp/core/outbound_http.py; .github/workflows/supply-chain.yml; docs/decisions/operational-objectives.v1.yml; bddk_mcp/operational_objectives.py:333-501**. Platform egress, signing/admission, exporters, approved objectives and retention remain external. |
| Partial | OpenShift and recovery topology | The starter separates public/operator workloads and four lifecycle Jobs: schema-owner migration, ingestion bootstrap/reindex, release-verifier verification/staging, and release-publisher activation. Verifier and publisher use different ServiceAccounts, Secrets, DSNs, commands, and database roles; the activation Job receives no corpus PVC or trust key. Retention remains an additional approved one-shot CLI action, not an MCP call or checked-in Job. Recovery fingerprints 53 managed objects, including both v8 request/activation-binding relations, rejects seven runtime DSNs for recovery administration, and verifies seven restored LOGIN profiles. A retained local two-cluster PG17 execution now proves schema v8 source/restore equality for all 53 objects, seven identities, two requests/bindings and two retained generations, including exact logical fingerprint and active-release identity (**docs/evidence/LOCAL_PG17_V8_RECOVERY_DRILL.md; docs/evidence/local-pg17-v8-restore-2026-07-16.json**). Focused OpenShift integration passed 75 tests. This is synthetic repository evidence only; the eight live-environment gates, bank Secret/RBAC separation, CNI enforcement, PITR, TLS/HBA, target-size capacity, and approved RPO/RTO remain open. |
| Partial | Legal and knowledge architecture | V4 models instruments, separate content/acquisition identities, evidence, legal versions, events/status assertions, hierarchical provisions, validated occurrences, family imports and review decisions; v6 resolves validated as-of status or abstains. Ordinary ingestion is deliberately unable to write the legal layer. `PageMappingProof` v2 plus signed-policy schema v2 can bind declared source-review owner IDs. No authoritative real-family import, real bank reviewer/human-action authentication, cross-document relationships, or provision-to-control knowledge exists. |
| Partial, deliberately non-release | Evaluation trust architecture | Phase 2 validates the selected manifest, reads and rechecks the active release on the same evaluated MCP session, and records the attestation. Release validation composes four signed evidence layers plus a separately signed schema-v2 policy: five release identities with documented canonical/raw semantics, four distinct declared operational-owner IDs, forward key rotation/revocation, independent policy head/scope pins, and time-bounded/revocable reviewer-owner assertions for every checkpoint artifact. Distinct IDs and declared event times do not prove separate human custody or signing time. Bank authorization stays false because bank root/RBAC promotion and human-review action are external; model-score authorization stays false because expert-dataset execution is absent. Evidence: **benchmark/phase2_e2e.py; benchmark/expert_evaluation.py; benchmark/evaluation_trust_policy.py; benchmark/release_preflight.py; benchmark/README.md: Hash and version semantics**. |

Current deployment and data flow:

~~~mermaid
flowchart LR
    Client[MCP client or model host]
    Gateway[Bank ingress and identity boundary]
    Public[Public MCP process\n15 tools]
    Operator[Operator MCP process\n29 tools]
    Resource[Active-release MCP resource\n1 resource / 0 prompts]
    Guard[Strict local-corpus read guard\npre/post release check]
    Migrate[Schema-owner migration Job]
    Bootstrap[Ingestion bootstrap Job]
    Verify[Release-verifier verify/stage Job]
    Activate[Release-publisher activation Job]
    Retain[One-shot retain-corpus-generation CLI]
    PG[(Operational PostgreSQL + pgvector)]
    Release[(V5 release/activation/epoch<br/>V8 staged request + binding)]
    Retained[(V7 sealed retained plane<br/>17 typed relations)]
    Legal[(Owner-controlled v0004 legal tables<br/>owner write; verifier read-only)]
    Resolver[V6 legal-status resolver]
    CitationView[Validated citation view]
    Pilot[Offline legal-family importer\nsynthetic pilot only]
    Sources[Approved BDDK and Mevzuat sources]
    Evidence[Current document + sections + published chunks]
    Eval[Phase 2 same-session harness]
    Trust[Four signed evidence layers<br/>schema-v2 policy and reviewers]

    Client --> Gateway
    Gateway --> Public
    Gateway -->|private operator path| Operator
    Client --> Eval
    Eval -->|resource read, tool calls, resource recheck| Public
    Public --> Resource
    Resource --> Release
    Public --> Guard
    Guard -->|reader identity| PG
    Public -->|SELECT only| CitationView
    CitationView --> Legal
    Public --> Resolver
    Resolver --> Legal
    Operator -->|ingestion + job identities| PG
    Operator -. no base-table privilege .-> Legal
    Operator --> Sources
    Migrate -->|schema-owner identity| PG
    Migrate -->|schema-owner identity| Legal
    Bootstrap -->|ingestion identity| PG
    Verify -->|release-verifier: verify exact corpus| PG
    Verify -->|stage expiring state-bound request| Release
    Activate -->|publisher: request ID only| Release
    Retain -->|publisher-only facade; no activation| Retained
    Release -->|release / activation identities| Retained
    PG -->|atomic typed copy + exact v5 hash| Retained
    Sources --> Bootstrap
    Pilot -. future curator identity required .-> Legal
    PG --> Evidence
    Trust -. release evidence only - no model authorization .-> Eval
~~~

The historical system-context and trust-boundary diagrams below must therefore be read as commit evidence, not as diagrams of the implementation overlay.

## Historical baseline architecture begins here

Everything below this heading describes commit **5684a34c** unless a sentence explicitly points back to the 2026-07-16 overlay. Labels such as “current” in the preserved baseline mean current at that reviewed commit.

## System context

~~~mermaid
flowchart TB
    subgraph Hosts[MCP hosts and model environments]
        Claude[Claude host]
        Codex[Codex / ChatGPT host]
        GPT[GPT-compatible orchestrator]
        LM[LM Studio or local host]
    end

    Claude --> MCP
    Codex --> MCP
    GPT --> MCP
    LM --> MCP

    subgraph Runtime[One BDDK MCP process]
        MCP[FastMCP]
        Public[15 public tools]
        Admin[11 optional operator tools]
        Deps[Global Dependencies]
        MCP --> Public
        MCP --> Admin
        Public --> Deps
        Admin --> Deps
    end

    Deps --> HTTP[Shared httpx client]
    Deps --> Store[DocumentStore]
    Deps --> Vector[VectorStore]
    Deps --> Client[BddkApiClient]

    HTTP --> BDDK[BDDK public endpoints]
    HTTP --> Mevzuat[mevzuat.gov.tr]
    Store --> DB[(PostgreSQL)]
    Vector --> PGV[(pgvector and FTS)]
    Client --> DB
    Client --> BDDK
~~~

## Architectural style

The implementation is a single-process modular monolith:

- one global FastMCP server;
- one shared asyncpg pool;
- one shared HTTP client;
- tool modules registered through closures over a global dependency container;
- PostgreSQL as catalog, document, section, vector, trace, cache, sync, and telemetry persistence;
- in-process model loading and embedding;
- optional in-process background sync and backfill jobs.

This was a reasonable shape for local deployment. At the reviewed commit, the same process combined public serving, operator control, schema creation, migrations, seed publication, extraction, embedding, caching, and monitoring, which was too broad for remote or enterprise operation. The current implementation supersedes that behavior with explicit public/operator processes, separate `migrate`, `bootstrap`, and `publish-corpus-release` commands, PostgreSQL identities, an active-release epoch and a durable job ledger; the sequence diagram remains historical evidence only.

## Entry points

| Entry point | Implementation | Behavior |
|---|---|---|
| Root Python shim | **server.py:1-10** | Imports main and mcp; executes main only when invoked as a Python script. |
| Real server | **bddk_mcp/server.py:200-247** | Selects transport, starts lifecycle, and runs uvicorn or stdio. |
| Docker/Procfile/Railway | **Dockerfile:31; Procfile:1; railway.toml** | Uses the Python-script path and reaches the custom lifecycle. |
| Documented MCP CLI | **README.md:104-109,323-328; .mcp.json:4-9** | Imports the FastMCP object without project startup; runtime reproduction found zero registered tools. |
| Seed CLI | **seed.py; bddk_mcp/ingest/seed.py:422 onward** | Exports/imports bundled database content. |
| Benchmark CLI | **benchmark/__main__.py; benchmark/run.py** | Runs static tool, NLI, terminology, and nominal E2E phases. |

At the reviewed commit, the checked-in .mcp.json also embedded a developer-specific absolute path, so it was not portable. The current worktree replaces it with the packaged entry point and no user-specific path.

## Server construction and lifecycle

~~~mermaid
sequenceDiagram
    participant Main as main
    participant Start as startup
    participant DB as PostgreSQL
    participant Seed as seed import
    participant Tools as tool registry
    participant V as vector task
    participant Sync as sync task

    Main->>Start: enter
    Start->>DB: create pool
    Start->>DB: create/alter document schema
    Start->>DB: create client cache schema
    Start->>Seed: initialize schemas and compare seed
    Seed->>DB: optional replace documents/chunks/cache
    Seed->>DB: optional embedding backfill
    Start->>Tools: register public modules
    opt BDDK_ADMIN_TOOLS
        Start->>Tools: register operator modules
    end
    Start->>V: background vector initialization
    opt AUTO_SYNC
        Start->>Sync: background startup sync
    end
    Start-->>Main: serve
~~~

Evidence:

- construction: **bddk_mcp/server.py:54-60**
- dependencies: **server.py:63-107**
- background vector initialization: **server.py:110-120**
- registration: **server.py:123-133**
- seed import: **server.py:135-152**
- lifecycle: **server.py:170-197**

Two consequences follow:

1. importing the FastMCP object is not sufficient to construct a functional server;
2. a public serving process needs broad write/DDL privileges and may spend substantial time mutating the corpus before it becomes ready.

## MCP layer

### Transports

| Transport | Support | Notes |
|---|---|---|
| stdio | Implemented in custom main | Correctly sends logs to stderr; documented MCP CLI path bypasses initialization. |
| Streamable HTTP | Implemented at /mcp | Stateless protocol mode, host 0.0.0.0, no in-repository auth or explicit transport security. |
| Legacy HTTP+SSE | Not implemented | Not required by the current specification; older-client need is unknown. |

An unrecognized MCP_TRANSPORT value falls back to stdio rather than failing validation (**bddk_mcp/server.py:210-247**).

### Capability surface

Runtime discovery:

| Capability | Public profile | Operator profile |
|---|---:|---:|
| Tools | 15 | 26 |
| Resources | 0 | 0 |
| Resource templates | 0 | 0 |
| Prompts | 0 | 0 |

### Tool registration by module

| Module | Profile | Responsibilities |
|---|---|---|
| **tools/search.py** | Public | BDDK catalog, institutions, announcements, hybrid document search |
| **tools/documents.py** | Public plus one operator registration | full local document, extraction history, document-store statistics |
| **tools/sections.py** | Public | exact section retrieval and section FTS |
| **tools/bulletin.py** | Public plus operator cache status/refresh | weekly/monthly bulletins, snapshots, cache controls |
| **tools/analytics.py** | Public | trends, digest, comparisons, update monitoring |
| **tools/sync.py** | Operator | document synchronization and startup-sync controls |
| **tools/admin.py** | Operator | health, metrics, backfill, quality report |

The process-wide BDDK_ADMIN_TOOLS flag decides whether operator functions are registered. It does not authenticate a caller or authorize individual tools.

### Tool contracts

Tool functions generally:

- accept primitive Python parameters;
- validate some inputs manually or through an internal Pydantic object;
- return one human-oriented string;
- use a logging decorator;
- return error-marker strings, ordinary no-result strings, or raised exceptions.

Runtime schema inspection showed:

- no tool annotations;
- no property-level input descriptions;
- few schema-visible numeric/string bounds;
- output schemas equivalent to one string result.

Internal validation can be stronger than the advertised MCP schema. For example **BddkSearchRequest** constrains pagination (**bddk_mcp/core/models.py:4-17**), but the tool's generated input properties do not advertise those constraints because the public signature uses primitive arguments.

### Error and result flow

**bddk_mcp/tools/errors.py:1-18** formats errors as normal text with an embedded code and retryability marker. This makes failure data visible to a model but does not give clients a reliable MCP isError signal.

**bddk_mcp/tools/tool_logging.py:95-143** wraps tool calls, records latency, logs arguments/result previews, and catches exceptions only for logging. There is no common typed response envelope.

## Application state

Despite stateless HTTP protocol mode, application state includes:

- global Dependencies instance: **bddk_mcp/core/deps.py:19-47**
- search LRU/cache: **bddk_mcp/tools/search.py:73-88**
- mutable known-announcement set: **bddk_mcp/tools/analytics.py:169-200**
- vector initialization state;
- sync/backfill task and status state;
- in-memory metrics.

This state is:

- shared by all callers within a process;
- not isolated by session, user, or tenant;
- not shared across replicas;
- reset on restart.

## Data architecture

### PostgreSQL objects

The repository creates and evolves objects programmatically rather than through a migration framework.

Main logical groups:

| Group | Representative objects/purpose |
|---|---|
| Catalog/cache | decision_cache and upstream response state |
| Documents | documents, document_versions |
| Structure | document_sections |
| Retrieval | document_chunks with pgvector embedding, FTS indexes/functions |
| Ingestion | sync metadata, failures, document health/traces |
| Observability | optional tool telemetry |

Evidence: **bddk_mcp/store/doc_store.py:101-230; store/vector_store.py:82-132,602-624; ingest/client.py:277 onward; observability/telemetry.py**.

### Current document entities

~~~mermaid
erDiagram
    DOCUMENT ||--o{ DOCUMENT_VERSION : archives
    DOCUMENT ||--o{ DOCUMENT_SECTION : parsed_into
    DOCUMENT ||--o{ DOCUMENT_CHUNK : indexed_as
    DOCUMENT ||--o{ DOCUMENT_TRACE : processed_by

    DOCUMENT {
        text document_id PK
        text title
        text category
        text decision_date
        text decision_number
        text source_url
        text markdown_content
        text content_hash
        text extraction_method
        int total_pages
    }
    DOCUMENT_VERSION {
        int version_number
        text content_hash
        text markdown_content
        timestamp synced_at
    }
    DOCUMENT_SECTION {
        text section_type
        text section_ref
        int start_char
        int end_char
        int page_start
        int page_end
    }
    DOCUMENT_CHUNK {
        int chunk_index
        text chunk_text
        vector embedding
        text section_type
        text section_ref
        text content_hash
    }
~~~

Missing from this model are stable instruments, immutable source artifacts, official legal versions, hierarchical provision paths, effective periods, legal status, amendments, repeals, consolidation, and validation state.

### Page semantics

The total_pages/page API for normalized documents is based on 5,000-character windows, not retained source-page boundaries:

- configuration: **bddk_mcp/core/config.py:71-79**
- document paging: **bddk_mcp/store/doc_store.py:406-441**
- vector page mapping: **bddk_mcp/store/vector_store.py:798-860**

The section schema contains page fields, but **bddk_mcp/store/section_index.py:103-113** does not populate them. OCR extraction also joins page text without a persistent coordinate model. The architecture therefore has normalized-text offsets but not audit-grade physical-page provenance.

## Ingestion architecture

~~~mermaid
flowchart LR
    Discover[Catalog and list discovery]
    Fetch[HTTP fetch and embedded-source discovery]
    Parse[HTML / MarkItDown / OCR / DOCX]
    Clean[Markdown sanitation]
    Quality[Deterministic quality scan]
    Store[Store current document]
    Sections[Replace section index]
    Chunk[Section/token-aware chunking]
    Embed[Multilingual-e5 embeddings]
    Publish[Replace vector chunks]
    Seed[Export seed JSON]

    Discover --> Fetch --> Parse --> Clean --> Quality
    Quality --> Store --> Sections
    Store --> Chunk --> Embed --> Publish
    Publish --> Seed
~~~

### Source strategies

**bddk_mcp/ingest/doc_sync.py:546-760** handles multiple BDDK/Mevzuat representations, including HTML, PDF, legacy DOC, iframe sources, and ZIP annexes. **html_extractor.py** preserves more regulatory structure than plain text conversion, including headings, lists, and tables. OCR is provider-based and can fall back among LightOCR, pdftotext, MarkItDown, and optional Chandra (**ocr/base.py:29-40,89-325**).

### Quality gates

The quality modules detect common extraction artifacts and attach warning/failure classifications. Full-document retrieval emits warnings. Two disconnects weaken the architecture:

- **config/quality_failures.yml** lists 11 known failures, while the runtime known-failure set is empty (**quality/markdown_quality.py:15,109-138**); the current scorer reported 0 failures.
- search and section tools do not consistently propagate quality status/warnings, unlike full-document retrieval.

### Section indexing

The parser recognizes major Turkish legal structures and some subparagraphs (**store/section_index.py:24-36,118-177**). It does not model parent relationships or a full provision path, so fıkra/bent identities are lossy.

Fresh seed import loads documents and vector chunks but does not populate document_sections (**ingest/seed.py:236-349**). Startup does not reindex them. Consequently exact section tools can return no result on an otherwise successfully seeded deployment until the manual reindex script is run.

### Corpus consistency

Ordinary document storage archives a previous snapshot and commits the current document. Section replacement follows separately. Vector reindexing is another transaction and can fail after current document storage (**ingest/doc_sync.py:487-525**).

Because full-document retrieval prefers vector chunks, a failed reindex may serve stale chunk content instead of the newer document-store content. There is no corpus generation, publish pointer, or consistency assertion at read time.

Startup seed import presents a separate publication path. A hash difference is treated as seed drift; the importer can delete/overwrite cache, documents, and chunks and then backfill embeddings (**ingest/seed.py:142-358**). It bypasses ordinary version archival and may replace a fresher deployed corpus with an older bundled snapshot.

## Retrieval architecture

### Chunking

**bddk_mcp/store/vector_store.py:248-482** supports:

- section-aware boundaries;
- token-aware chunk lengths and overlaps;
- character fallback;
- document and section metadata;
- character-range tracking.

### Dense retrieval

**vector_store.py:635-692,931-999** loads multilingual-e5, applies E5 query/document prefixes, embeds in an executor, and performs pgvector search. Model loading itself occurs in process and can delay first readiness/query.

### Sparse retrieval

**vector_store.py:1003-1067** uses PostgreSQL FTS with the simple dictionary plus unaccent. Hand-written Turkish relaxation and suffix rules live at **vector_store.py:500-553**.

### Fusion and reranking

**vector_store.py:1071-1221**:

- runs dense and sparse retrieval;
- uses reciprocal-rank fusion;
- applies thresholds/exact-match gating;
- optionally reranks;
- deduplicates at document level.

Sparse-only non-exact results can be removed by the semantic threshold, and document-level fusion can preserve a less useful vector payload rather than the strongest lexical provision. These are evidence-based algorithmic risks that need corpus-backed measurement, not speculative replacement.

## Security boundaries

Current boundaries are deployment conventions, not enforced architectural boundaries:

| Intended boundary | Current mechanism | Limitation |
|---|---|---|
| Local versus remote | MCP_TRANSPORT | HTTP defaults to all interfaces; no identity. |
| Public versus operator | BDDK_ADMIN_TOOLS | Global registration flag; no caller scope or separate process. |
| Read versus write DB | None | One role creates schema, serves, ingests, and writes telemetry/cache. |
| User/tenant | None | Shared global state and shared corpus. |
| Trusted code versus upstream data | sanitation/quality heuristics | No explicit prompt-injection trust labeling; some outputs bypass full sanitizer warning path. |
| Approved corpus versus staging | None | Seed/live changes can publish directly. |

## Observability

Current components:

- standard logging and JSON formatter;
- correlation-ID context variable;
- in-memory Metrics class;
- optional PostgreSQL telemetry;
- operator health/metrics tools.

Missing connections:

- no request boundary assigns/propagates correlation IDs;
- tool wrapper does not record Metrics;
- no liveness/readiness HTTP routes;
- no standardized metrics/traces export;
- no source/corpus generation on every tool trace;
- no SLO/alert contract.

## Deployment profiles

### Local stdio

Best fit for the current design because OS/process access can supply the trust boundary. It still requires a PostgreSQL service and a corrected launcher.

### Local Docker Compose

Useful development profile. Both PostgreSQL 5432 and MCP 8000 are published; default credentials and unauthenticated HTTP mean it must not be reused unchanged on a shared or remote host.

### Remote Streamable HTTP

Protocol transport exists, but production controls do not: TLS termination is external, authentication is absent, Origin/Host policy is absent, no rate limits are defined, public/operator planes are not separated, and the database role is overprivileged.

### Railway

The manifest supplies a build/deploy command and restart policy. It has no application health path, migration release phase, rollback contract, or repository-defined auth. Actual platform configuration is unknown.

### Hugging Face Spaces

The alternate Dockerfile is not self-contained: it omits seed data and assumes a db hostname. AUTO_SYNC cannot bootstrap an empty decision cache because startup sync skips that state.

## External dependency map

| Dependency | Purpose | Architectural risk |
|---|---|---|
| BDDK and mevzuat.gov.tr | authoritative discovery/download | upstream format, availability, completeness, and second-order content risks |
| PostgreSQL | canonical operational storage | one broad role, no versioned migration/backup contract |
| pgvector and unaccent | dense/lexical retrieval | extension privilege and version compatibility |
| Hugging Face models | embeddings/reranking | mutable revision and startup/download availability |
| MarkItDown/PDF/DOCX stack | normalization | table/formula/page loss |
| Chandra/CUDA | optional OCR | heavy optional environment and reproducibility |
| Anthropic | optional benchmark grader | silent fallback changes methodology |
| OpenAI-compatible local endpoint | model benchmark | function-calling benchmark is distinct from MCP compatibility |

## Architectural strengths to preserve

- official MCP SDK;
- public tool modules organized by domain;
- dependency injection through registration functions;
- shared async clients/pools with deliberate teardown;
- local-only exact document retrieval;
- layered extraction and OCR provider abstraction;
- deterministic quality scanning and content hashes;
- PostgreSQL/pgvector foundation;
- section-aware chunking and hybrid retrieval;
- graceful semantic-search degradation;
- bilingual documentation and broad unit coverage.

## Architectural constraints to remove

- import-time/global server construction;
- serving startup as migrator, seed importer, and embedding job;
- environment flag as the operator security boundary;
- mutable current-document/index writes without atomic publication;
- string-only tool contracts;
- character windows labeled as pages;
- static duplicated tool schemas/counts;
- replica-local user-visible state;
- unconnected metrics and unsafe default text logging.

The target architecture and an incremental migration path are defined in [TARGET_ARCHITECTURE.md](TARGET_ARCHITECTURE.md).
