# Target Architecture for the Next Major Version

## Design objective

The next major version should turn the current prototype into a dependable regulatory retrieval service without introducing unnecessary infrastructure. The target is:

- correct and testable MCP behavior;
- a local-first mode that remains easy to run;
- a secure remote mode with an explicit trust boundary;
- immutable, versioned, page-traceable regulatory evidence;
- atomic corpus publication;
- high-quality Turkish hybrid retrieval;
- model-independent structured contracts;
- a measured path from document search to validated regulatory knowledge.

PostgreSQL plus pgvector remains sufficient. A separate graph database, message broker, service mesh, Kubernetes requirement, or distributed vector database is not justified now.

## Implementation progress overlay — 2026-07-15

This checkpoint maps the current working tree to the target below. **Complete** means the repository implementation and a focused automated contract exist; it is not bank deployment acceptance. **Partial** means a useful slice is implemented but at least one target invariant or acceptance gate remains. **Open** means the target capability is not yet adequately implemented.

| Target slice | Status | Current evidence and remaining target work |
|---|---|---|
| MCP factory, registry, and transport | Partial | The installed server factory/lifespan selects exactly the 15-tool public or 28-tool operator profile. Strict generated arguments, risk annotations, stable protocol errors, and official-SDK stdio and Streamable HTTP initialize/list/call/error/shutdown tests are present (**bddk_mcp/server.py; bddk_mcp/mcp_server.py; bddk_mcp/tools/registry.py; tests/test_mcp_stdio_e2e.py; tests/test_mcp_http_runtime.py**). Six retrieval tools now expose validated structured evidence, but invariant 9 remains open for a uniform output schema across every tool and for named-client compatibility. |
| Secure remote HTTP | Complete at the application boundary | Non-loopback startup requires exact Host and HTTPS Origin allowlists plus complete asymmetric JWT/JWKS configuration and profile scopes; request body, rate, and concurrency admission are bounded (**bddk_mcp/http_security.py:320-393,437-488,542-698**). Bank TLS termination, a real IdP/CA, and ingress-wide limits remain deployment responsibilities. |
| Public/operator and database authorization | Complete as a repository boundary; bank acceptance open | Public/operator registries, scopes, DSNs, workloads, service accounts, Secrets, and exposure differ. Versioned DBA assets create NOLOGIN capability roles; startup verifies the exact LOGIN/effective-role membership and effective object privilege inventory, including every new pool connection. Schema-owner migration additionally verifies the exact target database and restricted identity; non-local PostgreSQL DSNs require `sslmode=verify-full` plus an absolute CA path (**bddk_mcp/db_identity.py; bddk_mcp/db_lifecycle.py; bddk_mcp/db_transport.py; deploy/postgres/**). Bank-created LOGINs, IdP mappings, network isolation, and DBA execution evidence remain external acceptance work. |
| Operator jobs | Partial | Mutating operations use durable PostgreSQL job records, hashed idempotency keys, CAS state/progress, crash recovery, and a connection-pinned advisory execution lease (**bddk_mcp/jobs/postgres.py; bddk_mcp/jobs/manager.py; tests/test_postgres_job_repository.py**). The runner task still executes inside one operator process; automatic ownership transfer for an ambiguous crash window, multi-replica failover, and target-bank operational acceptance remain open. |
| Database lifecycle and migration | Partial | A checksum-verified global v0001-v0003 ledger, advisory serialization, catalog attestation, schema-owner/wrong-target checks, role grants, clean/legacy/populated-v2 upgrade tests, and explicit migrate/bootstrap Jobs are implemented (**bddk_mcp/migrations/**; **bddk_mcp/catalog_integrity.py**; **tests/test_migrations.py**; **tests/test_legacy_migration_adoption.py**). A populated v2-to-v3 upgrade fails closed unless `--allow-retrieval-publication-backfill` is explicitly supplied after a stopped-workload, restorable-backup, size-matched rehearsal. Bank restore evidence, shared-cluster role naming, and future large-corpus expand/backfill/contract migrations remain open. |
| Retrieval publication consistency | Partial | Canonical document and section replacement is transactional. Chunks publish only after ordering, count, embedding, source-content, document-content, and retrieval-profile checks; mutation invalidates the publication row and serving joins only the current content/profile (**bddk_mcp/store/doc_store.py; bddk_mcp/store/vector_store.py; bddk_mcp/migrations/v0003_retrieval_publication.py**). This prevents serving stale/incomplete chunks but is not an immutable whole-corpus generation, atomic active pointer, or rollback mechanism. |
| Outbound and document-ingestion safety | Partial | Exact HTTPS destination policy, private/reserved-address rejection, redirect revalidation, bounded streaming/retry behavior, content signatures, and ZIP/DOCX member/count/size/ratio checks now have negative tests (**bddk_mcp/core/outbound_http.py; bddk_mcp/ingest/doc_sync.py; tests/test_outbound_http.py; tests/test_doc_sync.py**). Bank egress allowlists/DNS controls, prompt-injection treatment, parser sandboxing, and live-source acceptance remain open. |
| Observability | Partial | One MCP/tool boundary now propagates privacy-safe correlation IDs and updates thread-safe request/error/latency metrics; health/readiness and a separately scoped telemetry writer exist (**bddk_mcp/mcp_server.py; bddk_mcp/observability/metrics.py; bddk_mcp/observability/telemetry.py; tests/test_metrics.py**). A standard metrics endpoint/exporter, distributed traces, retention/access policy, SLOs, and bank monitoring integration remain open. |
| Evaluation, CI, and release verification | Partial | Phase 2 now uses the official MCP client over stdio or `/mcp`, discovers the live paginated schema, calls tools through `ClientSession`, fails closed on protocol/tool errors, and records audit identities (**benchmark/phase2_e2e.py; benchmark/audit.py; tests/test_benchmark_phase2.py**). CI requires PostgreSQL on Python 3.12/3.13, builds/verifies/install-tests distributions, and checks container recipes. Coverage/type/security/image execution, restore/load gates, expert Turkish cases, citation/claim grading, and named model/client runs remain open. |
| Packaging and OpenShift AI baseline | Partial | Digest-pinned build inputs, offline pinned model assets, distribution-content verification, non-root/read-only workloads, digest-only application image placeholders, exact Secret keys, TLS probes, stable selectors, lifecycle Jobs, a telemetry overlay, and default-deny ingress/egress manifests exist. CI installs checksum-pinned Kustomize and requires the base and telemetry overlay to render; missing tooling fails rather than skips (**Dockerfile; .github/workflows/ci.yml; scripts/verify_distribution.py; deploy/openshift/; deploy/openshift-overlays/; tests/test_openshift_manifests.py**). The bank must still supply signed application-image digests, egress allow rules, CA/IdP/registry policy, backup/restore, and real-cluster acceptance. |
| Regulatory evidence platform | Open | Immutable corpus generations, legal versions and effective state, amendment/repeal relationships, hierarchical provision identity, audit-grade citations, representative Turkish retrieval evaluation, and validated audit/control mappings remain the core next-major-version work. |

The guiding invariants remain acceptance criteria even where a row above is complete for one repository boundary. Application HTTP controls do not prove the enterprise gateway, durable rows do not make the in-process runner a bank-grade workflow engine, and fail-closed per-document retrieval publication is not an immutable corpus-generation switch.

## Guiding invariants

1. A public serving process never performs DDL, corpus ingestion, embedding backfill, or seed replacement.
2. Every returned provision identifies the immutable source artifact and legal version from which it came.
3. An answer about currentness or an as-of date is impossible unless validated temporal data supports it.
4. Search results never mix corpus generations within one request.
5. Public and operator capabilities are different registries, processes, credentials, and database roles.
6. Local stdio remains usable without remote-auth complexity.
7. Remote HTTP refuses insecure startup.
8. Tool schemas, documentation, benchmark schemas, and contract snapshots come from one registry.
9. Tool results are typed data with a useful text fallback.
10. Every published model/retrieval score records corpus, schema, embedding, reranker, host, model, and grader versions.

## Target system context

~~~mermaid
flowchart TB
    subgraph Clients
        Local[Local MCP host]
        Remote[Enterprise or hosted MCP client]
    end

    Local -->|stdio, OS trust| PublicMCP
    Remote -->|HTTPS Streamable HTTP, OAuth| Gateway[Identity / TLS / rate limit]
    Gateway --> PublicMCP[Public MCP serving plane]

    Operator[Operator or scheduled job] -->|private network, admin scope| Admin[Operator control plane]

    PublicMCP --> Registry[Typed public tool registry]
    PublicMCP --> ReadDB[(Serving views, read role)]
    PublicMCP --> Telemetry[Redacted telemetry writer]

    Admin --> Jobs[Ingestion and validation jobs]
    Admin --> WriteDB[(Staging corpus, writer role)]
    Migrator[Versioned migration job] --> Schema[(Schema owner role)]

    Jobs --> Sources[BDDK / Mevzuat / official artifacts]
    Jobs --> Objects[(Immutable source artifacts)]
    Jobs --> WriteDB
    Jobs --> Quality[Quality and domain review gates]
    Quality --> Publish[Atomic corpus publisher]
    Publish --> ReadDB

    Eval[Protocol and retrieval evaluation] --> PublicMCP
    Eval --> ReadDB
    PublicMCP --> Obs[Metrics, traces, logs]
    Admin --> Obs
~~~

The public and operator planes may live in the same repository and even the same image. They should not be the same running server in remote deployments.

## Component design

### 1. Validated configuration

Add one typed Settings model that validates:

- deployment mode: local-stdio, local-http, remote-public, remote-operator;
- transport enum and bind address;
- public and operator profiles;
- allowed hosts and origins;
- issuer, audience, authorization discovery, and required scopes;
- separate database DSNs/roles;
- pool and statement limits;
- query length/result limits;
- embedding/reranker names and immutable revisions;
- active corpus policy;
- telemetry privacy mode;
- egress host allowlist;
- download/archive limits.

Configuration errors must fail before opening a network listener. Remote modes must reject:

- wildcard/all-interface binding without allowed hosts/origins;
- missing authentication;
- operator tools on the public registry;
- a database role that owns the schema;
- mutable/unversioned required model configuration.

Keep environment variables as a deployment interface, but parse them once into immutable validated settings. Support an explicit env-file option for local use rather than implying automatic loading.

### 2. Server factory and lifecycle

Replace the global partially initialized FastMCP object with:

- **create_server(settings, profile)**;
- an SDK lifespan/context that creates dependencies and registers a complete server;
- an installed console script such as **bddk-mcp serve**;
- explicit commands for **migrate**, **ingest**, **seed**, **validate**, and **evaluate**.

Importing the configured server or invoking the console script must produce the same registered contract. No supported entry point may depend on side effects that an MCP CLI bypasses.

### 3. Declarative tool registry

Create one registry entry per tool with:

- stable name and semantic version;
- profile: public or operator;
- required scope;
- input Pydantic model;
- output Pydantic model;
- handler;
- read-only/destructive/idempotent/open-world annotations;
- timeout and cost class;
- privacy/logging class;
- deprecation metadata;
- examples and client notes.

Generate from this registry:

- MCP registrations and JSON Schema 2020-12 contracts;
- exact public/operator contract snapshots;
- README/tool reference;
- benchmark tool definitions;
- compatibility fixtures.

Representative public contract:

~~~json
{
  "status": "ok",
  "data": {
    "query": "TFRS 9 önemli artış",
    "results": []
  },
  "citations": [],
  "warnings": [],
  "meta": {
    "corpus_generation": "2026-07-14.1",
    "retrieval_profile": "hybrid-e5-v2",
    "request_id": "..."
  }
}
~~~

Text content should summarize the same data for clients that do not consume structuredContent. Output schemas must be explicit. Invalid input and execution failures should set MCP tool execution error state while retaining stable error codes, retryability, and safe diagnostic text.

### 4. Public and operator separation

#### Public plane

Contains only read operations:

- catalog and document search;
- exact immutable document/version/provision retrieval;
- bulletin read/analysis;
- safe update queries with caller-owned checkpoints;
- optional canonical MCP resources.

It uses a read-only database role and no general upstream fetch ability for document content. If live public catalog/bulletin fetches remain, give them a narrow egress adapter with bounds and caching.

#### Operator plane

Contains:

- source discovery and synchronization;
- extraction/OCR;
- quality review and quarantine;
- embedding and reindex;
- corpus validation and publication;
- migrations;
- backfill/repair;
- operational diagnostics.

Use a private endpoint/network and explicit scopes. Long operations return durable job IDs; they do not block a normal MCP request. Conflicting jobs use database advisory locks and idempotency keys. Shutdown drains or safely cancels every job.

The repository can retain an all-in-one local development profile, clearly labeled unsafe for remote deployment.

### 5. Authentication and authorization

For stdio, trust derives from the local OS user and process configuration.

For Streamable HTTP:

- terminate TLS at a documented ingress or in the deployment platform;
- implement MCP-compatible OAuth resource-server verification or a rigorously documented bearer-token mode for private installations;
- validate issuer, audience, expiry, signature, and scopes;
- publish/use authorization metadata as required by the selected MCP authorization profile;
- validate Host and Origin on every connection;
- use least privilege scopes such as:
  - bddk.read
  - bddk.monitor
  - bddk.ingest
  - bddk.publish
  - bddk.admin
- rate-limit by principal and IP;
- impose body, query, result, timeout, and concurrent-cost limits;
- log principal identifiers only through stable pseudonymous IDs unless an audited need exists.

Start single-tenant for enterprise deployments. Add row-level tenant isolation only if shared private corpora become a real requirement.

### 6. Canonical regulatory model

Use relational tables with immutable IDs and explicit validation status.

~~~mermaid
erDiagram
    REGULATORY_INSTRUMENT ||--o{ INSTRUMENT_VERSION : has
    SOURCE_ARTIFACT ||--o{ INSTRUMENT_VERSION : proves
    INSTRUMENT_VERSION ||--o{ PROVISION : contains
    PROVISION ||--o{ PROVISION : parent_of
    INSTRUMENT_VERSION ||--o{ REGULATORY_EDGE : source
    INSTRUMENT_VERSION ||--o{ REGULATORY_EDGE : target
    EXTRACTION_RUN ||--o{ PROVISION : produces
    CORPUS_GENERATION ||--o{ GENERATION_MEMBER : contains
    INSTRUMENT_VERSION ||--o{ GENERATION_MEMBER : published_as
    PROVISION ||--o{ CONTROL_MAPPING : supports

    REGULATORY_INSTRUMENT {
        uuid instrument_id PK
        text canonical_title
        text instrument_type
        text issuing_authority
        text canonical_number
    }
    SOURCE_ARTIFACT {
        uuid artifact_id PK
        text sha256
        text official_url
        text mime_type
        timestamp retrieved_at
        text storage_uri
    }
    INSTRUMENT_VERSION {
        uuid version_id PK
        date publication_date
        date effective_from
        date effective_to
        text legal_status
        text normalized_hash
        text validation_status
    }
    PROVISION {
        uuid provision_id PK
        uuid parent_id
        text provision_type
        text reference
        text stable_path
        int source_page_start
        int source_page_end
        text text_hash
    }
    REGULATORY_EDGE {
        text relation_type
        uuid evidence_provision_id
        text validation_status
    }
    CORPUS_GENERATION {
        uuid generation_id PK
        text status
        timestamp published_at
    }
    CONTROL_MAPPING {
        text obligation
        text control_objective
        text test_procedure
        text required_evidence
        text approval_status
    }
~~~

#### Regulatory instrument

Stable identity across amendments, consolidations, source aliases, and extraction runs. Store BDDK, Mevzuat, Resmî Gazete, decision-number, and legacy aliases separately.

#### Immutable source artifact

Retain original bytes locally or in object storage when legally permitted. Record SHA-256, MIME type, official URL, retrieval timestamp, headers, and capture status. When redistribution is not permitted, store a local/private reference and hash, not a public artifact.

#### Instrument version

Distinguish:

- source/legal version;
- normalized extraction revision;
- corpus publication generation.

Store publication, effective-from/to, repeal/supersession status, consolidation status, and validation state. Unknown dates remain explicitly unknown; the system must not infer “current” from latest download time.

#### Hierarchical provision

Represent stable paths such as:

**instrument / version / Madde 9 / fıkra 2 / bent a**

Preserve parent, sequence, title, text, source page/coordinates, normalized offsets, and hashes. Support annexes, tables, footnotes, temporary provisions, paragraphs/principles, ranges, and cross-references.

#### Relations

Use a typed PostgreSQL edge table for:

- amends;
- repeals;
- replaces;
- consolidates;
- implements;
- cites;
- defines;
- exception-to.

Each edge needs an evidence provision, extraction method, confidence, reviewer, and validation status. A graph database should be reconsidered only when measured multi-hop queries become difficult in PostgreSQL.

### 7. Page-, table-, and formula-preserving ingestion

Target pipeline:

~~~mermaid
flowchart LR
    Discover[Discover and inventory]
    Capture[Capture immutable artifact]
    Verify[Verify URL, MIME, size and hash]
    Parse[Page-preserving parse/OCR]
    Normalize[Canonical structured text]
    Structure[Provision/table/formula extraction]
    Quality[Automated quality gate]
    Review[Human review when needed]
    Stage[Stage generation]
    Index[Lexical and dense indexes]
    Validate[Corpus acceptance tests]
    Activate[Atomic active-generation switch]

    Discover --> Capture --> Verify --> Parse --> Normalize --> Structure
    Structure --> Quality
    Quality -->|pass| Stage
    Quality -->|quarantine| Review --> Stage
    Stage --> Index --> Validate --> Activate
~~~

Required controls:

- exact egress host allowlist and redirect revalidation;
- private/reserved IP rejection;
- response size, time, MIME, archive member/count/ratio limits;
- page-preserving PDF/OCR output;
- table structure and formula/token preservation;
- extractor/model/version provenance;
- duplicate/canonical alias detection;
- quarantine rather than silent degraded publication;
- known-failure registry connected to every retrieval surface;
- section index generation during seed/bootstrap;
- deterministic replay from artifact to normalized output.

Prioritize manual validation for TFRS 9, capital, liquidity, IRB, ICAAP/İSEDES, and interest-rate-risk documents where formulas and tables materially affect interpretation.

### 8. Atomic corpus publication

Stop mutating active documents and chunks in place.

For each ingestion run:

1. create a staging generation;
2. attach immutable legal versions and extraction outputs;
3. build sections, lexical index, embeddings, and optional rerank features;
4. validate document/chunk/section counts, hash consistency, vector coverage, source artifacts, quality status, duplicates, and citation round trips;
5. publish by changing one active-generation pointer in a transaction;
6. retain the prior generation for rollback and reproducibility.

Every public query binds to the active generation at request start. Retrieval and exact-fetch paths must verify the same version/generation hash. A failed index build cannot affect active serving.

Seed data becomes an explicit bootstrap artifact with a manifest, corpus generation ID, schema version, model revision, checksums, and legal/provenance metadata. It is imported only by an explicit command into an empty/staging generation.

### 9. Search and retrieval

Use a layered resolver:

1. exact identifiers and canonical aliases;
2. exact provision path/reference;
3. temporal/status/entity filters;
4. lexical retrieval;
5. dense retrieval;
6. reciprocal-rank fusion at provision/chunk level;
7. optional reranking;
8. evidence/citation assembly.

Retain multilingual-e5 initially. Pin the exact model revision and store it with every embedding/index generation. Do not change embedding models based on reputation alone; require benchmark improvement.

Improve Turkish behavior through evaluated, versioned components:

- Unicode/diacritic normalization with original text retained;
- acronym/alias glossary, such as PD/TO, LGD/THK, EAD/BKET, ECL/BKZ, IRB/İDD, ICAAP/İSEDES, SICR;
- phrase and legal-reference parsing;
- inflection-aware expansion;
- negative/currentness/status filters.

Sparse-only strong identifier/provision matches must survive semantic thresholds. Preserve the best evidence payload from each channel rather than always selecting the vector payload. Record component scores for audit and evaluation, but do not expose confusing uncalibrated scores as legal confidence.

Enable reranking only when:

- latency/cost budgets are defined;
- a representative Turkish benchmark shows material gain;
- the revision is pinned;
- fallback behavior is tested;
- results remain reproducible.

### 10. Citation engine

Return a uniform Citation object:

| Field | Meaning |
|---|---|
| citation_id | stable identifier within the response |
| instrument_id / version_id / provision_id | canonical legal identities |
| instrument_title / provision_path | human reference |
| official_url | source location |
| source_artifact_hash | immutable evidence checksum |
| normalized_text_hash | extracted evidence checksum |
| source_page_start/end | physical page when available |
| normalized_start/end | normalized-text offsets |
| excerpt | bounded supporting text |
| effective_from/to and legal_status | only validated temporal state |
| extraction_method and quality_status | evidence quality |
| corpus_generation | reproducibility |
| retrieved_at | response timestamp |

The engine must label location types honestly:

- source_page for official artifact pages;
- normalized_range for text offsets;
- display_window for pagination created only for response sizing.

Citation verification tests should reconstruct every excerpt from the identified artifact/version and fail on a hash, path, page, or excerpt mismatch.

### 11. Regulatory knowledge and audit layer

Add this only after canonical versions/citations work.

First structured entities:

- obligations;
- subjects/actors;
- actions;
- conditions;
- exceptions;
- thresholds;
- frequencies/deadlines;
- reports/evidence;
- applicable entity types;
- validation state.

Audit mappings should connect a validated provision to:

- obligation statement;
- control objective;
- suggested control;
- audit procedure;
- expected evidence;
- owner/frequency;
- reviewer and approval version.

LLM-extracted candidates remain unvalidated until a human domain reviewer approves them. Never present a generated control as a regulatory requirement unless its supporting provision and interpretation are distinguishable.

### 12. SQL and persistence boundaries

Use separate roles:

| Role | Privileges |
|---|---|
| schema_owner/migrator | DDL and extensions; used only in release jobs |
| ingestion_writer | write staging corpus and job state; no role/extension ownership |
| corpus_publisher | validate and switch active generation |
| serving_reader | SELECT on serving views/functions only |
| telemetry_writer | INSERT into narrow telemetry tables, if enabled |

Adopt versioned migrations with:

- schema version ledger;
- clean-install test;
- upgrade from supported previous release;
- advisory lock;
- forward-fix/rollback guidance;
- database privilege integration tests.

Keep parameterized queries and statement timeouts. Add read-only transactions for serving queries and safe, bounded database functions/views only where they simplify privilege enforcement. Do not expose arbitrary SQL.

### 13. Evaluation framework

One evaluation runner should use:

- the official MCP client;
- live-derived tool schemas;
- stdio and Streamable HTTP;
- deterministic corpus generations;
- recorded client/host/model/schema/model-revision metadata;
- fail-closed transport scoring;
- repeated trials and confidence intervals where model nondeterminism matters.

Separate metrics:

- MCP handshake and contract validity;
- tool selection;
- argument validity;
- exact document/provision retrieval;
- ranking quality;
- temporal/currentness correctness;
- citation precision and reconstruction;
- claim-level support and contradiction;
- abstention on unknown/unsupported questions;
- latency, resource use, and failure recovery;
- prompt-injection/tool-abuse resistance.

Model-independent design means evaluating the host-plus-model combination. GPT-OSS, Claude, Codex, LM Studio, or any other model/client is a profile, not an architectural dependency.

### 14. Observability

Add:

- request IDs created at transport boundaries;
- redacted structured logs with no query/result text by default;
- OpenTelemetry spans for MCP call, database query class, upstream fetch, embedding, reranking, and citation assembly;
- Prometheus-compatible metrics or an equivalent standard exporter;
- liveness and readiness routes outside the MCP tool registry;
- corpus generation/age, source freshness, section coverage, vector coverage, quarantine, and failed-sync metrics;
- job status and duration;
- authorization denials and throttling;
- SLOs and alerts.

Do not put raw audit queries or document excerpts into labels, logs, or traces. Define retention, access, export, and deletion policies before enterprise use.

### 15. Client compatibility

Maintain tested profiles rather than blanket “MCP compatible” claims:

| Profile | Required tests |
|---|---|
| Official SDK stdio | initialize, list, call, error, shutdown |
| Official SDK Streamable HTTP | auth, Origin/Host, initialize, list, call, reconnect/error |
| Claude host | checked-in config and representative tool calls |
| Codex/ChatGPT host | stdio/HTTP config and structured-content behavior |
| OpenAI-compatible orchestrator | actual MCP host adapter, not static function schemas alone |
| LM Studio | pinned version, transport, schema limits, local-model tool selection |
| GPT-OSS/local models | host and model pair, constrained schema/argument benchmark |

Test the current protocol revision and one supported prior revision. Record unsupported client behaviors instead of weakening the canonical server contract silently.

## Deployment profiles

### Local development

- Docker Compose or native PostgreSQL;
- explicit migrate and bootstrap commands;
- stdio default;
- optional all-in-one operator profile;
- loopback-only HTTP if requested;
- development text logging opt-in.

### Local enterprise/single host

- public stdio or loopback HTTP;
- separate operator command;
- separate DB roles;
- immutable local artifact storage;
- scheduled backups and restore tests;
- no inbound Internet exposure.

### Remote hosted

- authenticated HTTPS Streamable HTTP;
- public plane only at external ingress;
- operator plane private;
- managed PostgreSQL with PITR;
- object storage for immutable artifacts where permitted;
- versioned release migration and corpus-publish jobs;
- health checks, resource limits, monitoring, and rollback.

### Enterprise private deployment

- customer-controlled identity, network, database, keys, artifact storage, logs, and retention;
- single-tenant corpus by default;
- optional private knowledge/control mappings;
- contractual and technical entitlement at the hosted/operator boundary;
- documented data flow and audit log.

### Bank on-premises OpenShift AI target

The confirmed target is a bank's on-premises OpenShift AI environment. Keep the MCP design platform-neutral, but make this the primary enterprise deployment profile:

- one non-root public MCP Deployment/Service using the serving-reader database role;
- a separate operator Deployment or Job/CronJob, private to the namespace/network, using ingestion/publisher roles;
- bank identity integration and authenticated Streamable HTTP at the Route/ingress boundary;
- explicit Host/Origin policy in the application even when ingress also validates requests;
- NetworkPolicy allowing only required client, PostgreSQL, identity, model-serving, and approved BDDK/Mevzuat egress paths;
- Secrets or the bank's secret-management integration rather than ConfigMaps for credentials;
- readiness/liveness probes, resource requests/limits, disruption policy, audit-safe logs, metrics, and alerts;
- explicit migration and corpus-publication Jobs rather than init-time mutations;
- persistent PostgreSQL/object storage supplied as separately governed bank services;
- single-tenant/no-private-corpus mode until confidentiality and tenancy requirements are decided.

Exact OpenShift AI, MCP client, model-serving, identity, storage, and monitoring versions remain compatibility-test inputs rather than hard-coded application dependencies.

## Component disposition

### Retain

- official MCP Python SDK;
- stdio and Streamable HTTP;
- the installed server factory/lifespan and canonical public/operator tool registry;
- modular register-function organization;
- asyncpg connection pooling and parameterized SQL;
- PostgreSQL, pgvector, and unaccent;
- checksum-verified migrations, catalog attestation, and versioned PostgreSQL role/grant assets;
- exact runtime database-identity and TLS transport verification;
- durable PostgreSQL operator-job records and advisory execution leases;
- BDDK client/source adapters;
- bounded exact-host outbound HTTP and archive validation;
- custom HTML converter;
- OCR provider abstraction;
- deterministic quality checks and content hashes;
- fail-closed per-document retrieval publication as an interim consistency boundary;
- section-aware/token-aware chunking;
- multilingual-e5 baseline and RRF;
- privacy-safe correlation and request/error/latency instrumentation;
- official-client protocol E2E and the real MCP Phase 2 runner;
- verified package artifacts and the hardened OpenShift starter;
- local-only exact document retrieval;
- uv lockfile and Python 3.12/3.13 matrix.

### Refactor

- environment-backed configuration into one immutable validated settings object without weakening current fail-closed guards;
- the remaining tools into uniform typed success/evidence/error outputs;
- current document schema into instrument/version/provision entities;
- the in-process durable-job runner into an explicitly claimable/recoverable worker only if multi-replica operation is required;
- sync into staged corpus generations and atomic publication;
- section parser into hierarchical provision parser;
- quality registry into one runtime/CI source of truth;
- retrieval fusion at provision/chunk level;
- metrics/correlation into standard export, tracing, SLO, and retention contracts;
- the real MCP benchmark into expert-reviewed retrieval, citation, claim-grounding, and model/client evaluation.

### Replace

- mutable in-place active-corpus replacement;
- pseudo-page citations;
- remaining string-only success responses and evidence encoded only in prose;
- extraction snapshots as a substitute for legal versions/currentness;
- ad hoc manual correction provenance with reviewed correction records;
- process-local admission limits with bank ingress-wide policy where remote scale requires it.

### Remove

- unsupported deployment/client claims that have no pinned acceptance evidence;
- any future bypass for database target, TLS, identity, catalog, migration, or retrieval-publication checks;
- direct publication of known-failed or unreviewed extraction.

### Add

- immutable source artifacts and canonical legal versions;
- hierarchical provisions and typed regulatory relations;
- corpus generations and atomic publish/rollback;
- citation/evidence engine;
- expert-reviewed Turkish/domain benchmarks;
- named-client/model compatibility evidence;
- standard metrics/traces, SLOs, backups, restore drills, and operational acceptance evidence;
- validation workflow for obligations/control mappings;
- licensing/data-provenance governance.

## Transition sequence

| Step | Current state on 2026-07-15 | Next acceptance boundary |
|---|---|---|
| 1. Fix launcher and expose one runtime contract | Complete | Retain official-client stdio/HTTP and distribution gates. |
| 2. Secure HTTP and split public/operator processes | Complete at repository boundary | Prove bank IdP, CA, ingress, egress, and scope mappings. |
| 3. Remove DDL, seed, and backfill from serving | Complete | Retain explicit lifecycle identities and fail-closed readiness. |
| 4. Add typed outputs/errors and Citation mapping | Partial | Extend structured evidence from the six retrieval tools to the complete surface; add reconstructable Citation. |
| 5. Add migrations, roles, and corpus publication | Partial | Migrations/roles and per-document retrieval publication are delivered; immutable corpus generations and rollback remain. Rehearse the blocking populated-v2 v3 migration before any real upgrade. |
| 6. Preserve immutable artifacts/pages and hierarchical provisions | Open | Implement and validate against priority formula/table/page fixtures. |
| 7. Add legal version/status and amendment relations | Open | Require official evidence, explicit unknown state, and expert validation. |
| 8. Rebuild retrieval and evaluation on the canonical layer | Partial | Real MCP Phase 2 is delivered; expert Turkish judgments, citation/currentness/claim graders, and model runs remain. |
| 9. Add validated audit knowledge mappings | Open | Start only after temporal, citation, and expert-evaluation foundations pass. |

This sequence lets the project improve without a big-bang rewrite. The existing document/chunk tables can serve as a compatibility view during migration.

## Target acceptance statement

The next major version is architecturally successful when a clean local install and a secure remote deployment can:

- initialize through an official MCP client;
- expose exactly the intended scoped tools;
- retrieve a versioned provision with a reconstructable official-source citation;
- answer an as-of/currentness question only when validated data exists;
- survive failed ingestion without changing active results;
- run serving under a read-only role;
- publish reproducible protocol and retrieval evaluations;
- demonstrate backup/restore and upgrade behavior;
- keep private queries out of logs by default.
