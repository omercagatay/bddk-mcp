# Architecture at the Reviewed Commit

This document describes the architecture implemented at commit **5684a34c10e6d90bc22d6ab2a6466944afb6bf81**. It is descriptive, not the target design.

## Implementation progress overlay — 2026-07-15

The diagrams and detailed evidence below intentionally remain a snapshot of the reviewed commit. Unless a later note explicitly says otherwise, everything after this overlay is historical commit evidence. The current architecture is still a modular monolith, but it now has explicit process, lifecycle, privilege, and publication boundaries:

| Status | Architectural delta | Evidence and remaining boundary |
|---|---|---|
| Complete | MCP and process profiles | The packaged CLI selects one registry per process. Public uses only `BDDK_DATABASE_URL`; operator requires its own DSN, authorization scope, and remote opt-in. The registry contains 15 public and 13 additional operator tools with strict input schemas and risk annotations. Evidence: **bddk_mcp/cli.py; bddk_mcp/core/config.py; bddk_mcp/server.py; bddk_mcp/tools/registry.py**. |
| Complete | Transport and request boundary | Stdio and stateless JSON Streamable HTTP use the official SDK. Non-loopback HTTP requires Host/Origin allowlists, asymmetric JWT/JWKS verification, profile scope, bounded body/token/concurrency/rate admission, and optional validated server TLS. Evidence: **bddk_mcp/http_security.py; bddk_mcp/transport_tls.py; tests/test_mcp_stdio_e2e.py; tests/test_mcp_http_runtime.py**. |
| Complete | Explicit lifecycle and PostgreSQL authorization | `serve` is read-only with respect to schema/corpus lifecycle. A checksum ledger owns v0001-v0003 migrations; exact schema-owner identity and expected target database are checked; production transport requires `sslmode=verify-full` and an absolute CA path; ACL provenance and effective capabilities are verified; and every pooled public/operator connection is admitted against its expected identity. Readiness rechecks catalog, identity, operator-job, and telemetry contracts. Migration v0003 refuses the blocking publication backfill on a populated v2 corpus unless an operator explicitly approves it, after which existing documents must be reindexed. Evidence: **bddk_mcp/migrations/**; **bddk_mcp/db_lifecycle.py**; **bddk_mcp/db_transport.py**; **bddk_mcp/db_identity.py**; **bddk_mcp/catalog_integrity.py**; **deploy/postgres/**. Shared-cluster role naming, proof with the bank's actual LOGINs, and a size-matched upgrade rehearsal remain deployment work. |
| Complete | Durable single-replica operator control plane | `PostgresJobRepository` persists privacy-safe job records and PostgreSQL advisory leases serialize corpus mutation. Recovery is explicit. Evidence: **bddk_mcp/jobs/postgres.py; bddk_mcp/jobs/manager.py; bddk_mcp/server.py**. Runners remain in the operator process; multi-replica dispatch/failover is deliberately unsupported. |
| Partial | Retrieval publication | Canonical document and section replacement is transactional. Current-hash joins and `document_retrieval_publications` prevent stale or incomplete chunks from being returned as current, while invalidation, integrity checks, controlled reindexing, and v0003's default populated-corpus refusal protect the migration boundary. Six retrieval tools expose structured evidence. Evidence: **bddk_mcp/store/doc_store.py; bddk_mcp/store/vector_store.py; bddk_mcp/migrations/v0003_retrieval_publication.py; bddk_mcp/tools/structured_outputs.py**. These are per-document guards, not immutable whole-corpus generations, historical rollback, model-profile manifests per generation, or audit-grade pages. |
| Partial | Acquisition, supply chain, and observability | Outbound source requests have exact-host, redirect, public-address DNS, body, retry, archive-member, expansion-ratio, and decoded-size controls; Office XML parsing is hardened. Default embedding/reranker revisions and base images/actions are immutable. Logging is correlation-aware and content-safe by default; metrics are thread-safe; telemetry uses a distinct append-only identity. Evidence: **bddk_mcp/core/outbound_http.py; bddk_mcp/ingest/doc_sync.py; bddk_mcp/core/config.py; Dockerfile; bddk_mcp/core/logging_config.py; bddk_mcp/observability/**. DNS validation is not atomic with socket connection; platform egress, malware/source-authenticity checks, exported metrics/traces, SLOs, and bank retention policy remain open. |
| Partial | OpenShift topology | Separate non-root public/operator workloads, service accounts, TLS Services, a public-only Route, probes, lifecycle Jobs, ingress policies, and explicit telemetry overlay exist. Workloads and Jobs require an application-image digest, version labels stay out of immutable selectors, database DSNs use exact Secret-key references, a separate PostgreSQL CA is mounted for `verify-full`, and egress is default-deny. Checksum-pinned Kustomize rendering is mandatory in CI. Evidence: **deploy/openshift/**; **deploy/openshift-overlays/**; **tests/test_openshift_manifests.py**. Bank-specific egress allows, IdP/Route/CA and registry values, signed image/SBOM/vulnerability policy, restore evidence, and cluster acceptance remain open. |
| Open | Legal and knowledge architecture | No canonical model yet represents legal versions, effective intervals, amendment/repeal/supersession, consolidated provision lineage, authoritative source pages, provision-to-control mappings, or validated audit knowledge. PostgreSQL remains sufficient for the proposed first implementation. |

Current deployment and data flow:

~~~mermaid
flowchart LR
    Client[MCP client or model host]
    Gateway[Bank ingress and identity boundary]
    Public[Public MCP process\n15 tools]
    Operator[Operator MCP process\n28 tools]
    Migrate[Schema-owner migration Job]
    Bootstrap[Ingestion bootstrap Job]
    PG[(PostgreSQL + pgvector)]
    Sources[Approved BDDK and Mevzuat sources]
    Evidence[Current document + sections + published chunks]

    Client --> Gateway --> Public
    Client -->|private operator path| Gateway --> Operator
    Public -->|reader identity| PG
    Operator -->|ingestion + job identities| PG
    Operator --> Sources
    Migrate -->|schema-owner identity| PG
    Bootstrap -->|ingestion identity| PG
    Sources --> Bootstrap
    PG --> Evidence
~~~

The historical system-context and trust-boundary diagrams below must therefore be read as commit evidence, not as diagrams of the implementation overlay.

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

This was a reasonable shape for local deployment. At the reviewed commit, the same process combined public serving, operator control, schema creation, migrations, seed publication, extraction, embedding, caching, and monitoring, which was too broad for remote or enterprise operation. The current implementation supersedes that behavior with explicit public/operator processes, migration/bootstrap commands, PostgreSQL identities, and a durable job ledger; the sequence diagram remains historical evidence only.

## Entry points

| Entry point | Implementation | Behavior |
|---|---|---|
| Root Python shim | **server.py:1-10** | Imports main and mcp; executes main only when invoked as a Python script. |
| Real server | **bddk_mcp/server.py:200-247** | Selects transport, starts lifecycle, and runs uvicorn or stdio. |
| Docker/Procfile/Railway | **Dockerfile:31; Procfile:1; railway.toml** | Uses the Python-script path and reaches the custom lifecycle. |
| Documented MCP CLI | **README.md:104-109,323-328; .mcp.json:4-9** | Imports the FastMCP object without project startup; runtime reproduction found zero registered tools. |
| Seed CLI | **seed.py; bddk_mcp/ingest/seed.py:422 onward** | Exports/imports bundled database content. |
| Benchmark CLI | **benchmark/__main__.py; benchmark/run.py** | Runs static tool, NLI, terminology, and nominal E2E phases. |

At the reviewed commit, the checked-in .mcp.json also embedded a developer-specific absolute path, so it was not portable. The working tree replaces it with the packaged entry point and no user-specific path.

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
