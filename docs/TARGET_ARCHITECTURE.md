# Target Architecture for the Next Major Version

> [!NOTE]
> This document describes a desired end state and includes dated progress overlays; it is not a statement that every target is implemented. Current repository facts are in [Current Repository Status](STATUS.md).

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

## Current implementation checkpoint — 2026-07-16

This table is authoritative over the dated overlay retained below. “Complete” means implemented and contract-tested in the repository; it never means bank acceptance or legally validated source content.

| Target slice | Status | Evidence and remaining boundary |
|---|---|---|
| MCP interface | Repository-complete foundation | One registry exposes 15 public tools and 14 operator additions (29 total), with strict schemas and risk annotations. Official SDK tests cover stdio and Streamable HTTP. One resource, `bddk://corpus/active-release`, is registered; prompts remain absent (**bddk_mcp/tools/registry.py; bddk_mcp/resources.py; tests/test_mcp_stdio_e2e.py; tests/test_mcp_http_runtime.py**). Named Claude/Codex/GPT-OSS/LM Studio certification remains open. |
| Security planes | Repository boundary complete; bank acceptance open | Public, operator, migrator, ingestion, release-verifier, release-publisher, and telemetry responsibilities are separated by configuration, DSN, role assets, workloads, scopes, and fail-closed identity checks. The verifier can stage a release request but cannot activate/retain; the publisher activates by request ID and cannot inspect the corpus or stage. Remote HTTP requires Host/Origin, asymmetric JWT/JWKS policy, and first-byte/inter-chunk/total body deadlines (**bddk_mcp/http_security.py; bddk_mcp/db_identity.py; deploy/postgres/; deploy/openshift/**). Bank IdP, CA, Route, CNI, HBA, seven LOGINs, and verifier/publisher Secret/RBAC separation remain external. |
| Database lifecycle | Current through schema v10 | v4 is the canonical legal model, v5 adds corpus release/activation/epoch state, v6 adds least-privilege status resolution, v7 adds a typed immutable retained-generation plane, and v8 adds staged verification requests with one-time activation bindings. Recovery inventories 53 managed objects, excludes seven application DSNs, and verifies seven restored LOGIN profiles. A retained synthetic two-cluster PG17 execution proves exact schema-v8 equality for all 53 objects, two request/bindings, two retained generations, the logical fingerprint and active release (**bddk_mcp/migrations/; bddk_mcp/operations/recovery.py; docs/evidence/LOCAL_PG17_V8_RECOVERY_DRILL.md**). Focused v8 evidence also includes 59 PG17 migration/catalog tests, 118 application/role tests, and two actual-LOGIN tests. Bank PITR/RPO/RTO, custody, target-size/TLS/HBA/LOGIN execution and controlled backup-growth evidence remain open. |
| Corpus publication | Partial target | Schema v8 splits the supported publication flow: the release verifier independently revalidates and stages an expiring request bound to DB-computed state/epoch/profile and verifier revision/image; the publisher supplies only that one-time request ID to atomic activation. The publisher cannot execute the old direct publication routine. Every corpus mutation increments an epoch that invalidates the active view, and strict local-corpus calls lease and recheck one active release (**bddk_mcp/migrations/v0008_staged_corpus_releases.py; bddk_mcp/corpus_publication.py; bddk_mcp/corpus_serving.py**). This remediates the repository-level publisher-only bypass, but bank custody must keep the two credentials separate. V7 can atomically copy and seal the exact active state across 17 typed retained relations without changing activation. Reads/caches remain v5-table-bound and retained-generation rollback/reactivation remains absent until H2-02B. |
| Retained-generation control plane | Repository mechanism implemented; acceptance partial | Physical generation, governed release, content-derived seal, and activation identities are separate. Generation and seal are derived from exact corpus state plus retrieval profile, so multiple governed releases over that same pair use distinct release bindings to one physical generation/seal. The release-publisher-only `retain-corpus-generation` CLI drives a security-definer facade; it is not an MCP tool, does not serve retained data, and cannot activate/reactivate. A v5 release without a v7 binding is labelled `legacy_v5_unretained`, never backfilled merely from a matching hash. Catalog/ACL/recovery contracts include the v7 plane (**bddk_mcp/migrations/v0007_retained_corpus_generations.py; bddk_mcp/corpus_generations.py; bddk_mcp/cli.py; bddk_mcp/catalog_integrity.py; bddk_mcp/db_identity.py**). Heap/index/TOAST evidence reconciles; WAL, when available, is only an observed cluster interval and otherwise remains `not_measured`; backup growth is `not_measured`, and bank retention/capacity authorization is open. |
| Current tracked corpus | Governable release achieved at the unmeasured level | The manifest (`bddk-job-corpus-2026-08-14`) is non-exhaustive by declaration, Ed25519-signed, owner-quantified, and consistent with the 9,675 chunks the current profile regenerates. Schema v10 admits a closed two-value freshness policy, so the corpus stages and activates as `quantified_unmeasured_signature_verified_pass`; a full local PG17 migrate→bootstrap→stage→activate→serve drill passed. Reaching the measured level still requires a live per-document source-event pipeline (**seed_data/corpus_scope.yml; bddk_mcp/migrations/v0010_corpus_release_freshness_policy.py; docs/GAP_REGISTER.md CUR-001**). |
| Regulatory model and citations | Partial technical pilot | Canonical instruments, versions, provisions, events, status assertions, a public abstention-first resolver, and Citation v1 exist. Legal-release verification can rehash retained source/acquisition/page/excerpt evidence and a signed predecessor chain. A schema-v2 trust policy supports forward operational rotation/revocation and binds each v2 checkpoint/artifact review to a time-bounded, revocable reviewer owner (**bddk_mcp/regulatory/; bddk_mcp/tools/legal_status.py; benchmark/legal_release_evidence.py; benchmark/evaluation_trust_policy.py**). No authoritative real family, bank-issued/root-custodied policy instance, independently authenticated human review action, independently proven page derivation, policy-root lifecycle, or complete historical legal-pack replay exists. |
| Evaluation | Partial, deliberately non-release | Phase 2 binds the selected manifest to the active release before and after calls on the same MCP session. The expert gate composes signed corpus, signed dataset, signed legal-curator pack attestation, and a signed retained-evidence checkpoint with distinct canonical signer fingerprints. Development uses an operator-supplied head; signed-policy mode uses its approved head, and bank-policy mode additionally pins policy SHA/version plus organization/environment/scope and requires v2 reviewer assertions (**benchmark/phase2_e2e.py; benchmark/expert_evaluation.py; benchmark/evaluation_trust_policy.py; benchmark/release_preflight.py**). Bank root/RBAC promotion and human reviewer action remain external, model execution is unimplemented, and all ordinary scores remain exploratory. |
| OpenShift operations | Repository manifests/preflight complete; live gates open | Four separate lifecycle Jobs and identities exist: migrate, bootstrap, verifier-side verify/stage, and publisher-side activation. The activator has neither the corpus PVC nor trust key. Exact Kustomize/render/security/egress/Secret/command contracts passed 75 focused integrated tests (**deploy/openshift/jobs/; deploy/openshift/serviceaccounts.yaml; bddk_mcp/openshift_acceptance.py**). Namespace execution, bank principal/Secret separation, and all other live controls remain external. |
| Objectives and observability | Deliberately non-production | A versioned eight-metric operational contract exists, but every target and rolling window is unset and unapproved (**docs/decisions/operational-objectives.v1.yml; bddk_mcp/operational_objectives.py**). Metrics, alerts, retention, tracing, load, and resilience require bank decisions and execution evidence. |

Maturity remains: overall **3/5**, production readiness **2/5**, MCP **4/5**, retrieval **3/5**, security **3/5**, testing/evaluation **3/5**, documentation **4/5**.

## Prior implementation overlay — 2026-07-15 (superseded where it conflicts)

This checkpoint maps the current working tree to the target below. **Complete** means the repository implementation and a focused automated contract exist; it is not bank deployment acceptance. **Partial** means a useful slice is implemented but at least one target invariant or acceptance gate remains. **Open** means the target capability is not yet adequately implemented.

| Target slice | Status | Current evidence and remaining target work |
|---|---|---|
| MCP factory, registry, and transport | Partial | The installed factory selects exactly the 15-tool public or 29-tool operator profile. Strict generated arguments, risk annotations, stable protocol errors, and official-SDK stdio and Streamable HTTP tests exist (**bddk_mcp/server.py; bddk_mcp/mcp_server.py; bddk_mcp/tools/registry.py; tests/test_mcp_stdio_e2e.py; tests/test_mcp_http_runtime.py**). Six retrieval tools have typed evidence; uniform structured results for the remaining tools and named-client evidence remain open. |
| Secure remote HTTP | Complete at the application boundary | Non-loopback startup requires exact Host and HTTPS Origin allowlists, complete asymmetric JWT/JWKS configuration, and profile scopes. The composed FastMCP application publishes RFC 9728 protected-resource metadata at `/.well-known/oauth-protected-resource/mcp`; unauthenticated 401 responses point clients to it with `resource_metadata`. Request body, rate, and concurrency admission are bounded (**bddk_mcp/http_security.py; tests/test_mcp_http_runtime.py**). This does not prove bank OAuth client registration/flows, TLS termination, IdP/CA integration, shared ingress quotas, or network policy. |
| Public/operator and database authorization | Complete as a repository boundary; bank acceptance open | Public/operator registries, scopes, DSNs, workloads, ServiceAccounts, Secrets, and exposure differ. The v0001-v0008 lifecycle verifies exact target, transport, LOGIN/effective membership, ACL provenance, and catalog shape; disposable PG17 transactional and actual-LOGIN role tests passed locally (**bddk_mcp/db_identity.py; bddk_mcp/db_lifecycle.py; bddk_mcp/db_transport.py; bddk_mcp/catalog_integrity.py; deploy/postgres/**). All eleven v0004 legal-curation tables are owner-controlled and mutation remains owner-only; v8 gives the release verifier the exact read-only exception needed to recompute publication evidence. From that legal layer the public reader can select only `public.regulatory_validated_section_citations`, while ingestion/operator/publisher identities have no direct legal-table privilege. The v0004 catalog digest covers exactly 69 constraints and 21 indexes. Bank-created principals and executed DBA evidence remain external. |
| Operator jobs | Partial | Mutations use durable PostgreSQL job records, idempotency fingerprints, CAS state/progress, startup recovery, and advisory execution leases (**bddk_mcp/jobs/postgres.py; bddk_mcp/jobs/manager.py; tests/test_postgres_job_repository.py**). Execution remains in one operator process; multi-replica ownership transfer and bank operational acceptance are not proved. |
| Database lifecycle and recovery | Partial at the historical checkpoint | At this superseded checkpoint the checksum ledger ended at v0004 and no second-cluster dump had run. The current overlay above supersedes it with a retained schema-v8/53-object/seven-identity synthetic two-cluster execution. Bank backup/PITR, RPO, RTO, custody, TLS/HBA, target-size and target-environment acceptance remain unproved. |
| Retrieval publication and Citation v1 | Partial | Per-document text/section replacement is transactional and chunk serving requires the current content/profile publication (**bddk_mcp/store/doc_store.py; bddk_mcp/store/vector_store.py; bddk_mcp/migrations/v0003_retrieval_publication.py**). Citation v1 can reconstruct an exact normalized range only through the validated v0004 view, rechecks canonical IDs and text hashes, and binds the frozen-whitespace boundary to retained section text. An official MCP session against real PostgreSQL validates the path with synthetic data (**bddk_mcp/citations.py; bddk_mcp/store/doc_store.py; bddk_mcp/tools/sections.py; tests/test_citations.py; tests/test_legal_versions.py**). It remains a technical pilot: retained authoritative artifact bytes, true source pages, real-corpus legal mappings, curator/source authenticity, whole-corpus generations, atomic activation, and rollback are absent. |
| Canonical regulatory model | Partial pilot | v0004 adds instruments, immutable family-import identities, content-addressed `SourceBlob` rows, separately acquisition-addressed `SourceArtifact` rows, evidence declarations, legal versions, events, status assertions, provisions, and validated version-provision occurrences. The importer is transactional, and a synthetic family tests deterministic current/as-of resolution and abstention (**bddk_mcp/migrations/v0004_canonical_legal_versions.py; bddk_mcp/regulatory/; tests/test_legal_versions.py**). It is not wired to ordinary ingestion and does not establish the legal currentness, source-byte authenticity, or curator authority of any real regulation. |
| Acquisition and untrusted-data boundary | Partial | Exact HTTPS destinations, public-address and redirect validation, response/retry/archive bounds, and hardened XML parsing have negative tests. All source-backed tool text now places metadata and bodies inside one escaped untrusted-data envelope, including delimiter-spoof cases (**bddk_mcp/core/outbound_http.py; bddk_mcp/ingest/doc_sync.py; bddk_mcp/tools/structured_outputs.py; tests/test_structured_retrieval_outputs.py**). The code boundary is complete for the present renderers; live-model prompt-injection/tool-escalation evaluation, parser sandboxing, source authenticity, malware controls, and bank egress enforcement remain open. |
| Corpus scope and expert evaluation | Partial, deliberately non-release | The 318-document job-specific selection has a checksummed, machine-readable, non-exhaustive scope manifest, but it is unsigned, has no numeric freshness objectives, and has no per-document measured freshness evidence (**seed_data/corpus_scope.yml; bddk_mcp/corpus_manifest.py**). Bootstrap is now path-bound to that manifest, rejects undeclared reserved filenames, and can enforce quantified/measured/signature gates with a separately mounted key in the same import process. It emits path-free ID/SHA operator evidence but does not yet persist the corpus identity in PostgreSQL (**bddk_mcp/ingest/seed.py; bddk_mcp/cli.py; tests/test_seed.py; tests/test_cli.py**). The 20-case Turkish dataset covers eight domains but every case remains draft. Release still requires three separately verified trust inputs: signed corpus plus measured freshness, separately signed dataset, and separately signed exact validated-Citation export/attestation under a distinct legal-curator key; dataset/legal signer reuse is rejected (**benchmark/expert_evaluation_draft.yml; benchmark/expert_evaluation.py; tests/test_expert_evaluation.py**). |
| Supply-chain evidence | Partial repository lane | A separate workflow uses pinned tools/build inputs, requires reproducible Python distributions, builds containers with Buildx `--provenance=false --load`, binds exact descriptor/manifest/config/loaded-image/Syft identities, produces deterministic CycloneDX SBOMs and unsigned repository SLSA provenance, validates model-manifest/runtime/Dockerfile consistency, scans complete Git history and fresh vulnerability data, and blocks High/Critical or secret findings. An applied pending exception always makes promotion ineligible (**.github/workflows/supply-chain.yml; scripts/supply_chain_evidence.py; supply-chain/; tests/test_supply_chain.py**). It does not sign, push, admit, or promote an image; bank registry, signing identity, exception approval, and promotion policy remain external. |
| OpenShift AI acceptance | Partial repository preflight | The harness requires exactly Kustomize v5.8.1 and the reviewed SHA-256 of the resolved binary, performs a real bounded offline build, and enforces exact rendered-resource, namespace, selector/label, NetworkPolicy, Secret/ConfigMap, workload-shape and restricted-security-context inventories. It renders the exact `bank-bootstrap` overlay, verifies direct strict bootstrap arguments and separate read-only corpus/trust sources, requires approved regulatory-source/proxy HTTPS for both public and operator runtimes, and forbids that reach for lifecycle Jobs (**deploy/openshift-overlays/bank-bootstrap/**; **bddk_mcp/openshift_acceptance.py; scripts/openshift_acceptance.py; tests/test_openshift_acceptance.py**). A passing status is explicitly `preflight_passed_external_gates_pending`; all eight bank/cluster gates remain `not_run`. It is repository contract evidence, not bank acceptance. |
| Observability and product evaluation | Partial | Correlation IDs, privacy-safe tool logging, request/error/latency metrics, readiness, and isolated telemetry exist. The real MCP Phase 2 harness discovers and calls the live contract and fails closed (**bddk_mcp/observability/; benchmark/phase2_e2e.py**). Standard export/tracing, SLOs, retention, load/resilience, live model/client runs, claim grading, and bank monitoring integration remain open. |

The cross-document maturity ratings remain: overall **3/5**, production readiness **2/5**, MCP **4/5**, retrieval **3/5**, security **3/5**, testing/evaluation **3/5**, and documentation **4/5**. The guiding invariants below remain acceptance criteria. A technical pilot is not validated legal knowledge; an offline preflight is not a bank deployment; and repository recovery or supply-chain evidence is not a completed restore, signed promotion, or production approval.

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
    Quality --> Verify[Release verifier stages state-bound request]
    Verify --> Activate[Publisher activates request ID]
    Activate --> ReadDB

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
4. validate document/chunk/section counts, hash consistency, vector coverage, source artifacts, quality status, duplicates, and citation round trips under the verifier identity;
5. stage one expiring request bound to the exact state, epoch, profile, verifier revision, and image;
6. activate only that request ID in a separate publisher transaction after rechecking expiry, reuse, readiness, and drift;
7. retain the prior generation for rollback and reproducibility.

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
| corpus_release_verifier | inspect exact corpus/legal evidence and stage an expiring state-bound request; no activation or retention |
| corpus_publisher | activate a staged request by ID and retain its sealed generation; no raw corpus/evidence reads and no staging |
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
- a separate operator Deployment or Job/CronJob, private to the namespace/network, using only the ingestion/operator roles needed for that workload;
- bank identity integration and authenticated Streamable HTTP at the Route/ingress boundary;
- explicit Host/Origin policy in the application even when ingress also validates requests;
- NetworkPolicy allowing only required client, PostgreSQL, identity, model-serving, and approved BDDK/Mevzuat egress paths;
- Secrets or the bank's secret-management integration rather than ConfigMaps for credentials;
- readiness/liveness probes, resource requests/limits, disruption policy, audit-safe logs, metrics, and alerts;
- four explicit lifecycle Jobs—migration, bootstrap, verify/stage, and activation—with separately custodied verifier/publisher credentials rather than init-time mutations;
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
- exact Kustomize v5.8.1 offline acceptance rendering and namespace checks;
- the guarded recovery/fingerprint/timeout primitives;
- the v7 typed retained-generation tables, exact-v5 fingerprint reproduction,
  immutable seals, publisher-only facade, and recovery/catalog/ACL contracts;
- the v8 verifier/publisher split, expiring state-bound release requests,
  one-time request-ID activation, and exact catalog/ACL/identity contracts;
- the v0004 owner-controlled canonical legal-model pilot and validated-citation view, with owner-only mutation and the v8 verifier read-only exception;
- Citation v1 normalized-range identity and reconstruction rules;
- the non-exhaustive corpus manifest and fail-closed expert-dataset release validator;
- the escaped untrusted-source framing boundary;
- deterministic SBOM/provenance and supply-chain policy code;
- local-only exact document retrieval;
- uv lockfile and Python 3.12/3.13 matrix.

### Refactor

- environment-backed configuration into one immutable validated settings object without weakening current fail-closed guards;
- the remaining tools into uniform typed success/evidence/error outputs;
- bridge the current document/section schema to the isolated v0004 instrument/version/provision pilot without granting ordinary ingestion direct legal-curation writes;
- the synthetic legal-family importer into a separately authenticated curator/reviewer workflow with authoritative retained evidence;
- Citation v1 from normalized-range pilot to artifact-byte/page reconstruction and a uniform citation surface;
- the in-process durable-job runner into an explicitly claimable/recoverable worker only if multi-replica operation is required;
- generation-bind serving and caches to the implemented v7 retained plane,
  then add policy-authorized activation/reactivation without changing the
  historical release or seal identities;
- section parser into hierarchical provision parser;
- quality registry into one runtime/CI source of truth;
- retrieval fusion at provision/chunk level;
- metrics/correlation into standard export, tracing, SLO, and retention contracts;
- the 20-case draft and real MCP harness into signed, independently annotated/adjudicated retrieval, citation, claim-grounding, and model/client evaluation;
- unsigned supply-chain evidence into bank-approved signing, admission, and digest promotion only after the bank selects those mechanisms.

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

- bank issuance, RBAC-separated deployment/promotion, compromise exercises, and root lifecycle/history for the implemented schema-v2 evaluation policy, scope pins, reviewer registry, and rotation/revocation-capable keyring;
- reproducible page derivation plus authenticated reviewer-action/signature evidence beyond the implemented policy-bound owner assertion for retained authoritative source bytes and page/coordinate mappings;
- end-to-end replay of historical legal packs, curator attestations, and retained artifacts;
- real-family legal-version curation, reviewer authority, hierarchical provision coverage, and typed cross-document relations;
- authorized generation-bound serving and rollback/reactivation on top of the
  implemented v7 retained-generation targets; bank retention/capacity approval
  and controlled backup-growth evidence;
- artifact/page-capable citation verification beyond the normalized-range Citation v1 pilot;
- expert-reviewed Turkish/domain benchmarks;
- named-client/model compatibility evidence;
- standard metrics/traces, SLOs, backups, restore drills, and operational acceptance evidence;
- validation workflow for obligations/control mappings;
- licensing/data-provenance governance.

## Transition sequence

| Step | Current state on 2026-07-16 | Next acceptance boundary |
|---|---|---|
| 1. Fix launcher and expose one runtime contract | Complete | Retain official-client stdio/HTTP and distribution gates. |
| 2. Secure HTTP and split public/operator processes | Complete at repository boundary | Prove bank IdP, CA, ingress, egress, and scope mappings. |
| 3. Remove DDL, seed, and backfill from serving | Complete | Retain explicit lifecycle identities and fail-closed readiness. |
| 4. Add typed outputs/errors and Citation mapping | Partial | Six retrieval tools are typed and exact validated sections can carry reconstructable normalized-range Citation v1. Extend the contract to every tool and bind citations to retained authoritative bytes/pages. |
| 5. Add migrations, roles, and corpus publication | Repository boundary implemented through v8; retention acceptance partial | v5 persists releases/activations and invalidates them by epoch; v7 retains and seals the exact 17-relation active state; v8 splits verifier staging from request-ID-only publisher activation. A retained local restore proves the 53-object/seven-identity v8 path. Prove bank credential separation and target-size TLS/HBA/PITR recovery, complete controlled backup/capacity acceptance, then implement generation-bound serving and separately authorized rollback/reactivation in H2-02B. |
| 6. Preserve immutable artifacts/pages and hierarchical provisions | Partial pilot | v0004 declares artifacts and stable provision identities for a synthetic family. Retain authoritative bytes, source pages, tables/formulas, and real-family mappings before regulatory reliance. |
| 7. Add legal version/status and amendment relations | Partial pilot | v4 plus the v6 resolver represent validated claims and abstain on conflict/unknown state. A technical v2 policy-bound reviewer-owner assertion exists; ordinary ingestion, one real authoritative family, bank-authenticated reviewer action/page derivation, and amendment/currentness review remain open. |
| 8. Rebuild retrieval and evaluation on the canonical layer | Partial, non-release | Phase 2 now binds one active release on the same MCP session, and a four-layer cryptographic preflight plus schema-v2 policy/reviewer/rotation verifier exists. Repair/sign the corpus, bank-issue/promote/exercise that policy and its root lifecycle, finish adjudication, implement exact expert-case execution, and run named models before authorizing scores. |
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
