# Security Review

Review basis: commit **5684a34c10e6d90bc22d6ab2a6466944afb6bf81**, reviewed 2026-07-14.

## Implementation progress overlay — 2026-07-16

The findings and severities below remain the reviewed-commit baseline. **Complete** here means an application/repository control and focused automated contract exist; it is not a bank security approval. **Partial** means material residual risk remains. **Open** means the control is not adequately implemented.

| Security slice | Status | Current evidence and residual boundary |
|---|---|---|
| Remote application identity and request boundary | Complete at application boundary | Non-loopback startup fails closed without exact Host and HTTPS Origin allowlists, complete asymmetric JWT/JWKS verification, and the profile scope. The composed FastMCP application publishes RFC 9728 protected-resource metadata at `/.well-known/oauth-protected-resource/mcp`, and an unauthenticated 401 challenge supplies that URL through `resource_metadata`. Body, rate, and concurrency admission are bounded per process (**bddk_mcp/http_security.py; tests/test_mcp_http_runtime.py**). This is MCP authorization discovery at the application boundary, not shared ingress enforcement or proof of bank OAuth client registration/flows, TLS, or IdP integration. |
| Public/operator application separation | Complete at repository boundary | Exactly one strict profile is served per process; public and operator use different registries, scopes, DSNs, workloads, ServiceAccounts, and Secrets. Operator remote exposure requires explicit opt-in (**bddk_mcp/core/config.py; bddk_mcp/server.py; deploy/openshift/**). Whether the bank applies equivalent Route/network/identity separation is an external gate. |
| Error, logging, health, and work admission | Partial overall | Validation/unknown/execution failures are stable and privacy-safe; default tool logging omits request/result/error content; liveness/readiness recheck identity/catalog dependencies. Durable PostgreSQL jobs use idempotency and advisory leases (**bddk_mcp/mcp_server.py; bddk_mcp/jobs/**). Admission remains per process and execution remains a single operator-process task, so shared ingress quotas and multi-replica job ownership are not proved. |
| Database and legal-curation privilege boundary | Complete as repository contract | Exact target, `verify-full` transport, LOGIN/effective membership, ACL provenance, per-connection admission, catalog integrity, and checksum migrations fail closed through v0006. Disposable PG17 transactional allow/deny and actual-LOGIN identity/ACL contracts executed locally and passed. All eleven v0004 legal-curation tables are owner-only, and readiness still attests the exact v4 digest of 69 constraints and 21 indexes; v5 adds attested append-only corpus release/activation state and v6 an attested SECURITY DEFINER abstention-first status resolver. From the legal layer the public reader can select only the security-barrier/definer validated-Citation view and execute the narrowly granted resolver; public, ingestion, and operator identities have no direct legal-table privilege (**bddk_mcp/db_identity.py; bddk_mcp/catalog_integrity.py; bddk_mcp/migrations/; deploy/postgres/**). Bank curator/reviewer identities, bank LOGINs, and DBA execution evidence remain external. |
| Citation and legal-claim integrity | Partial technical pilot | `SourceBlob` provides content identity while `SourceArtifact` separately identifies an acquisition. Citation v1 binds exact normalized ranges to validated non-fixture legal rows. A separate signed legal-release checkpoint can re-hash retained source bytes, acquisition records, page mapping/text, exact excerpts, and every predecessor's retained files. `PageMappingProof` v2 binds each checkpoint/artifact review to an opaque owner in the signed policy's time-bounded, revocable reviewer registry (**bddk_mcp/citations.py; benchmark/legal_release_evidence.py; benchmark/evaluation_trust_policy.py; tests/test_citations.py; tests/test_expert_evaluation.py; tests/test_evaluation_trust_policy.py**). The only complete family remains synthetic. This proves a policy-authorized identity assertion, not a reviewer signature or the human action, and exact excerpt containment does not independently prove raw source/PDF-to-page-text derivation. Historical legal packs are not retained/replayed against predecessor Citations. No real legal currentness, bank reviewer authority, curator authority, or source authenticity is proved. |
| Untrusted-document boundary | Complete in current renderers; live-model risk open | Source-backed text, including titles, metadata, URLs, snippets, and bodies, is escaped and enclosed as untrusted data; delimiter-spoof tests cover official MCP responses (**bddk_mcp/tools/structured_outputs.py; tests/test_structured_retrieval_outputs.py**). This prevents framing ambiguity in server output, not semantic obedience by a live model. No host/model prompt-injection or public-to-operator escalation evaluation has run; parser sandboxing and source authenticity remain open. |
| Acquisition and corpus scope | Partial | Exact approved HTTPS hosts, redirect/public-address validation, bounded response/retry/archive behavior, hardened XML parsing, and per-document publication checks fail closed. Bootstrap verifies manifest-role paths and policy; a distinct publisher revalidates and persists an append-only active release with manifest/retrieval-profile/corpus-state identity, while a mutation epoch invalidates stale activation (**bddk_mcp/ingest/seed.py; bddk_mcp/corpus_publication.py; bddk_mcp/migrations/v0005_corpus_release_publication.py**). The 318-document scope remains unsigned, unquantified, unmeasured, non-exhaustive, and its declared 8,286 chunks drift from 9,675 current-profile rows, so strict publication refuses it. Bank egress, malware/source authenticity, immutable retained generations/rollback, and private-corpus/tenant policy remain open. |
| Recovery and upgrade integrity | Partial repository workflows | A guarded populated-v2-to-current-schema rehearsal proves default refusal, reindex/readiness, actual-content fingerprints, and bounded PostgreSQL subprocess timeout/cleanup. Recovery evidence schema v2 covers 29 managed relations plus activation sequence, rejects all six application DSNs as recovery administration, and verifies six restored LOGIN profiles (**bddk_mcp/operations/recovery.py; tests/test_recovery_workflows.py; docs/RECOVERY_DRILLS.md**). This is a repository workflow/contract; retained bank PITR, backup custody, approved RPO/RTO, and bank restore acceptance remain unproved. |
| OpenShift deployment boundary | Partial repository preflight | The non-root starter separates planes and lifecycle Jobs. The acceptance harness requires exact Kustomize v5.8.1 and the reviewed SHA-256 of the resolved executable, performs a real bounded offline build, and enforces exact resource, namespace, selector/label, NetworkPolicy, Secret/ConfigMap, workload-shape, command/port/volume, and restricted-security-context inventories. The reviewed `bank-bootstrap` overlay passes strict freshness/signature/key gates directly to bootstrap and keeps the read-only approved corpus PVC separate from the read-only corpus-trust Secret. The egress matrix requires narrow regulatory-source/proxy HTTPS for both runtimes—public includes live-source tools—and forbids that reach for lifecycle Jobs (**deploy/openshift-overlays/bank-bootstrap/**; **bddk_mcp/openshift_acceptance.py; tests/test_openshift_acceptance.py**). Its success state still leaves eight live external gates `not_run`; it is repository evidence, not bank acceptance. |
| Supply-chain boundary | Partial repository lane | Pinned build/scanner inputs and reproducible Python artifacts are defined. Containers use Buildx `--provenance=false --load`; evidence then binds exact descriptor/manifest/config/loaded-image/Syft identities, emits unsigned repository SLSA, and verifies model-manifest/runtime/Dockerfile consistency. Complete-history secret scanning, fresh vulnerability data, and High/Critical blocking are fail closed; applied pending exceptions always leave promotion ineligible (**.github/workflows/supply-chain.yml; scripts/supply_chain_evidence.py; supply-chain/**). No artifact is signed or promoted, and no bank registry/admission/exception-approval control is proved. |
| Evaluation integrity | Partial, deliberately non-release | The 20-case/eight-domain Turkish dataset is checksum/corpus-bound, but all cases remain draft and legal currentness/Citation mapping is unverified. Release validation composes four signed layers: measured corpus, expert dataset, Citation pack/legal-curator attestation, and a legal-release checkpoint over retained evidence/history. The preflight now has explicit development and bank-policy modes. Bank-policy mode verifies a separately signed schema-v2 policy, binds five release identities with explicitly documented canonical/raw hash semantics, maps distinct canonical keys and declared owner IDs to four separated roles, enforces validity and effective key/checkpoint revocations, supports policy-approved forward legal-release key rotation, requires v2 reviewer identities for every checkpoint artifact, applies declared-ID reviewer separation/windows/revocation, and pins the current policy SHA/version plus organization/environment/scope (**benchmark/evaluation_trust_policy.py; benchmark/release_preflight.py; benchmark/README.md: Hash and version semantics; tests/test_evaluation_trust_policy.py; tests/test_release_preflight.py**). Event windows use declared signed timestamps and the local clock, not trusted signature timestamps. It still reports bank authorization and model scores false: the repository cannot attest bank ownership/RBAC custody, actual human/team separation, promoted pins, or human reviewer action, and it does not execute the expert dataset. Historical pack replay, reproducible page derivation, currentness/version/amendment scoring, and target-environment stale-policy/compromise exercises remain open. |

Current security maturity remains **3/5**. Cross-document ratings remain overall **3/5**, production readiness **2/5**, MCP **4/5**, retrieval **3/5**, testing/evaluation **3/5**, and documentation **4/5**. Production approval remains blocked by bank-applied identity/network/CA/egress controls, bank LOGIN/DBA and curator evidence, issuance and RBAC-controlled deployment/promotion of a real bank evaluation policy and current head pins, repair and signing of the drifting corpus, a completed isolated restore, signed promotion, real source/page authenticity evidence, live-model injection evaluation, numeric SLO/RPO/RTO, and real cluster validation.

## Baseline security conclusion at the reviewed commit

At the reviewed commit, the repository was suitable only for trusted local experimentation. It was not safe as an Internet-facing MCP endpoint or as a shared enterprise service.

The most important distinction is between safety features and security boundaries:

- Parameterized SQL, local-only full-document retrieval, response page caps, content sanitation, and a default-hidden operator profile are useful safety features.
- At the reviewed commit, none of them authenticated a caller, authorized a tool, isolated a tenant, made the serving database read-only, or protected an HTTP listener. The dated implementation overlay records the later repository controls, including reviewed role/grant assets and runtime identity/ACL/catalog assertions. Whether the bank has applied those assets to its actual LOGINs and accepted the deployed boundaries remains unproved.

No confirmed committed production secret, arbitrary SQL tool, direct SQL injection, path traversal extraction, or active production breach was found. No Critical finding is assigned. The High findings must nevertheless be resolved before remote or regulatory production use.

## Scope

Reviewed:

- MCP transports and session/application state;
- public/operator tools and tool abuse;
- HTTP authentication, authorization, Origin/Host protection, and rate limits;
- SQL construction, transactions, roles, and serving privileges;
- configuration and secrets;
- ingestion egress, redirects, archives, files, and prompt injection;
- logging, telemetry, error leakage, and tenant isolation;
- dependencies, containers, CI, deployment, backup, and recovery;
- licensing and practical control of commercial use.

Not reviewed:

- production infrastructure, gateway, Railway project, DNS, certificates, identity provider, firewall, database grants, or logs;
- current dependency/container vulnerability database;
- formal penetration testing;
- legal advice or source-document rights.

## Assets and security objectives

### Assets

- regulatory source artifacts and normalized text;
- canonical legal/version/quality metadata;
- embeddings, indexes, and search thresholds;
- internal-audit/compliance queries and result context;
- operator credentials and database roles;
- sync/backfill/publish capability;
- evaluation datasets and model-comparison integrity;
- deployment availability and resource budget;
- validation decisions and future control mappings.

### Objectives

| Objective | Required property |
|---|---|
| Regulatory integrity | Active corpus cannot be silently downgraded, partially indexed, or modified by public callers. |
| Evidence traceability | Every citation reconstructs to an immutable source/version. |
| Query confidentiality | Audit/compliance queries and excerpts do not enter unauthorized logs/telemetry. |
| Least privilege | Public serving cannot mutate schema or corpus. |
| Caller control | Every remote request has a verified identity, allowed scope, and bounded cost. |
| Availability | Expensive tools, upstream files, models, and archives cannot exhaust service resources. |
| Evaluation integrity | Benchmark transport, schemas, corpus, graders, and scores are reproducible and fail closed. |
| Tenant isolation | Private content and state cannot cross organizational boundaries. |

## Trust-boundary model

~~~mermaid
flowchart LR
    Internet[Untrusted network/client]
    Local[Trusted local MCP host]
    HTTP[Streamable HTTP listener]
    Stdio[stdio process]
    Public[Public tool plane]
    Admin[Operator tool plane]
    DB[(PostgreSQL)]
    Model[Embedding/reranker code]
    Upstream[BDDK/Mevzuat pages and files]
    Logs[Platform logs/telemetry]

    Internet -->|reviewed commit: no repository auth| HTTP
    Local --> Stdio
    HTTP --> Public
    HTTP -->|when flag enabled| Admin
    Stdio --> Public
    Stdio --> Admin
    Public --> DB
    Admin --> DB
    Admin --> Upstream
    Public --> Upstream
    DB --> Model
    Public --> Logs
    Admin --> Logs
    Upstream -->|untrusted content| DB
~~~

Trust-boundary failures at the reviewed commit:

- the HTTP boundary verifies no identity;
- public and operator tools can share the same listener and process;
- public serving and schema/corpus ownership share one DB identity;
- upstream content becomes model context without a uniform untrusted-data contract;
- logs are a secondary store for user text.

## Confirmed strengths

### SQL construction

No MCP tool accepts arbitrary SQL text. Queries use asyncpg positional placeholders. Dynamic fragments select internal fixed clauses/placeholder positions rather than caller SQL.

Representative evidence:

- document lookup: **bddk_mcp/store/doc_store.py:366-404**
- section FTS and sanitized plainto_tsquery: **doc_store.py:568-620**
- vector filters: **bddk_mcp/store/vector_store.py:931-999**
- transactional vector replacement: **vector_store.py:696-797**

### Transaction use

Multi-row document/version and vector operations generally use transactions:

- **store/doc_store.py:294-364**
- **store/vector_store.py:696-797**

The problem is publication across separate transactions, not a general absence of transactions.

### Default operator visibility

BDDK_ADMIN_TOOLS is false by default (**core/config.py:117-123**). This reduces accidental exposure but is not caller authorization.

### Full-document egress boundary

get_bddk_document reads only the local store and caps a response to five display pages (**tools/documents.py:54-88**). That is a good hallucination and data-source boundary.

### Telemetry defaults

Optional database telemetry is disabled by default and hashes query-like text unless raw storage is explicitly enabled (**core/config.py:125 onward; observability/telemetry.py:48-72,116-170**).

### Secrets scan

No confirmed production credential was found in tracked source. .env is ignored. Docker Compose contains explicit development credentials, which are unsafe only if reused outside isolated local development.

### Owner-supplied deployment facts

The project owner confirms that no external endpoint protection is currently in place, the target is a bank's on-premises OpenShift AI environment, and exact client, tenancy, and private-document requirements are not yet known. The owner will administer the database and validate regulatory content. Accordingly:

- SEC-H1 and SEC-H2 were immediate blockers at the reviewed commit. Repository controls now close their application-level defects, but actual bank ingress, identity, private operator exposure, and shared rate enforcement remain release-acceptance gates;
- the target baseline is separate public/operator workloads and database roles in a single-tenant namespace;
- private-document ingestion remains disabled until confidentiality and tenancy requirements are approved;
- “immediate” availability/freshness/recovery must be converted to numeric SLO, publication-lag, RPO, and RTO acceptance tests.

## Historical High findings at commit 5684a34

The five findings in this section preserve the reviewed-commit threat analysis and evidence, with explicit current-disposition notes where needed. Their current disposition is authoritative only in the 2026-07-15 overlay; historical evidence and remediation text below must not be read as claims that the current code still lacks those controls.

### SEC-H1 — Unauthenticated, under-protected Streamable HTTP

Evidence:

- host 0.0.0.0 and stateless HTTP: **bddk_mcp/server.py:54-60**
- Streamable HTTP app/listener: **server.py:218-234**
- published port: **docker-compose.yml:17-24**
- Docker selects HTTP: **Dockerfile:31**
- runtime inspection: auth=None, transport_security=None

The pinned SDK automatically supplies DNS-rebinding protections only for loopback/localhost defaults. The explicit 0.0.0.0 host prevents that automatic safe default. The current MCP transport specification requires Origin validation for HTTP connections.

Impact:

- any reachable client can enumerate and call public tools;
- DNS rebinding or invalid Origin is not explicitly rejected;
- no identity exists for access logs, quotas, revocation, or scopes;
- semantic search and live-source calls can be abused for cost/availability;
- admin exposure becomes severe if the global flag is enabled.

Required remediation:

1. Bind local HTTP to 127.0.0.1 by default.
2. Configure explicit allowed Hosts and Origins.
3. Require authentication for any remote profile.
4. Validate issuer, audience, signature, expiry, and scopes.
5. Apply body/query/result/time/concurrency/rate limits by principal and IP.
6. Refuse remote startup if required controls are absent.
7. Test valid and invalid Host/Origin, missing/invalid/expired token, wrong audience/scope, and throttling.

References:

- https://modelcontextprotocol.io/specification/2025-11-25/basic/transports
- https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization

### SEC-H2 — Operator exposure is not authorization

Evidence:

- conditional registration: **bddk_mcp/server.py:123-133**
- flag: **core/config.py:117-123**
- synchronization: **tools/sync.py:188 onward**
- backfill/database mutation: **tools/admin.py:108-213**
- cache refresh: **tools/bulletin.py:156-185**

When the flag is true, every caller of the same endpoint can invoke the registered operator tools. There is no role, scope, token verifier, approval, private route, or separate DB role.

Required remediation:

- build separate public and operator registries;
- run operator plane on a private network/process;
- require bddk.ingest, bddk.publish, or bddk.admin scopes;
- use different DB credentials;
- use durable job IDs, exclusive locks, idempotency keys, and safe cancellation;
- refuse to co-host operator tools on a public remote listener.

### SEC-H3 — Public serving holds schema and corpus authority

Evidence:

- one DSN/pool: **server.py:63-107**
- document DDL/migrations: **store/doc_store.py:101-230**
- pgvector extension/schema changes: **store/vector_store.py:602-624**
- cache initialization: **ingest/client.py:277 onward**
- startup seed and embeddings: **ingest/seed.py:108-416**

At the reviewed commit, the serving process could not operate with a read-only account. The current repository removes schema/corpus/cache-population/embedding initialization from `serve`; supplies separate schema-owner, ingestion, public, operator, and telemetry role/grant assets; and verifies exact effective privileges, ACL provenance, database/schema ownership, expected database, secure transport, and every public/operator pool connection. Repository PostgreSQL tests prove the intended denial/allow matrix. The remaining gap is deployment evidence with the bank's actual LOGINs, DBA membership model, TLS/HBA policy, shared-cluster role names, and real upgrade data—not an absent repository role contract.

Required remediation:

| Role | Minimum access |
|---|---|
| schema_owner | migration job only |
| ingestion_writer | staging corpus and jobs |
| corpus_publisher | validate/switch active generation |
| serving_reader | SELECT on serving views |
| telemetry_writer | narrow INSERT only, if used |

Tests must prove serving credentials cannot INSERT, UPDATE, DELETE, CREATE, ALTER, DROP, or create extensions.

### SEC-H4 — Corpus downgrade and stale-index integrity

Security includes integrity. Every startup compares bundled seed hashes and can overwrite differing live content, cache, and chunks without normal version archival (**ingest/seed.py:142-358**). Ordinary document/section/vector publication also spans separate transactions (**store/doc_store.py:294-364; ingest/doc_sync.py:487-525**), and retrieval can prefer stale chunks (**tools/documents.py:96-119**).

Current disposition: serving no longer imports at startup. Schema v5 persists an
append-only active release, any tracked corpus mutation advances an epoch that
invalidates it, and strict local-corpus calls verify one release before and after
the read. Independent publication also rejects a signed chunk artifact that
differs from current-profile regeneration. The present tracked artifact does
differ (8,286 declared versus 9,675 regenerated), so there is no publishable
strict release. Immutable retained corpus generations and a tested prior-release
rollback remain open.

Required remediation:

- explicit seed/bootstrap only;
- immutable source/version rows;
- staging corpus generation;
- validate hashes, sections, quality, vector completeness, citations, and duplicates;
- atomically switch the active generation;
- keep previous generation for rollback;
- bind every query to one generation;
- alert/refuse on document/index version mismatch.

### SEC-H5 — Audit-query text is logged

Evidence:

- arguments and result previews: **tools/tool_logging.py:34-85,95-143**
- JSON fields: **core/logging_config.py:28-63**
- test enshrining raw query behavior: **tests/test_tool_logging.py:45 onward**

The optional telemetry's hashing does not protect normal logs. Enterprise logs often have broader access and longer retention than the source database.

Required default log fields:

- request ID;
- pseudonymous principal;
- tool/version;
- latency and status;
- result count/size;
- corpus generation;
- safe error class/code;
- cost/rate-limit state.

Excluded by default:

- query/keywords;
- document text or excerpts;
- raw structured output;
- tokens/authorization;
- database DSN/details;
- private filenames/content.

Provide an explicit local-debug text mode with warnings, short retention, and tests.

## Historical Medium findings at commit 5684a34

These findings and code references describe the reviewed commit. The 2026-07-15 overlay records the implemented SSRF, archive, immutable-pinning, error, request-admission, logging, and deployment controls and their remaining boundaries.

### Unbounded costly inputs

Limits are inconsistent or absent for semantic queries, lookback periods, days, result sizes, and operator concurrency (**tools/search.py:302 onward; analytics.py:23 onward; sync.py:204 onward; ingest/doc_sync.py:828 onward**).

Controls:

- strict schema limits;
- request body limit;
- per-tool timeout/cost class;
- bounded queues/semaphores independent of caller values;
- rate and concurrent-cost quotas;
- cancellation and backpressure;
- load tests.

### Second-order SSRF

**ingest/doc_sync.py:684-706** can fetch an absolute iframe URL derived from upstream HTML. The shared HTTP client follows redirects (**server.py:67-83**). No exact egress host or private-address restriction is visible.

Controls:

- HTTPS exact-host allowlist;
- DNS and resolved-IP validation;
- private, loopback, link-local, multicast, and metadata IP rejection;
- redirect target revalidation;
- response size/time/content-type caps;
- tests for DNS rebinding and redirect chains.

### Archive/resource exhaustion

ZIP/DOCX annex processing has no archive member/count/ratio/uncompressed-size limits (**ingest/doc_sync.py:307-333,727 onward**). It reads content in memory. No Zip Slip was found because members are not extracted to arbitrary disk paths.

Controls:

- compressed and uncompressed byte limits;
- member-count and expansion-ratio limits;
- MIME/magic checks;
- per-document CPU/time/memory budgets;
- quarantine on violation.

### Document prompt injection

Regulatory/upstream text is untrusted input to a model. Full-document retrieval sanitizes artifacts and attaches some warnings, but search/section snippets have different paths (**tools/search.py:375-395; sections.py:125-312; documents.py:199-217**).

Filtering cannot reliably remove semantic prompt injection. Controls should instead:

- return evidence in a typed data field;
- label content untrusted;
- keep operator tools unavailable to public callers;
- make tools least privilege;
- prevent documents from controlling tool selection/permissions;
- test malicious documents that request secret disclosure, external fetches, or operator calls;
- require claim-level citation verification.

### Global state and tenant isolation

Global caches, jobs, metrics, and update baselines have no user/tenant key (**core/deps.py:19-47; tools/search.py:73-88; tools/analytics.py:169-200**).

Current recommendation: single-tenant deployment. Do not add shared private corpora until identity, per-tenant keys/storage, row policies, cache partitioning, telemetry isolation, deletion/export, and authorization tests exist.

### Error leakage

The tool logging wrapper includes exception messages, and user-visible error strings are inconsistent. No secret was seen, but raw upstream/database exception content can reveal implementation details.

Controls:

- map exceptions to safe stable codes;
- return a request ID;
- keep full stack/detail only in restricted logs;
- redact DSNs, URLs with credentials, SQL details, paths, tokens, and private content;
- test representative asyncpg/httpx/model exceptions.

### Container and supply chain

Evidence:

- root container and mutable bases: **Dockerfile:1-31**
- mutable model download: **Dockerfile:18-21**
- tag-pinned CI actions: **.github/workflows/ci.yml**
- broad default dependencies: **pyproject.toml:6-20**
- no SBOM, vulnerability, secret, or image scanning.

Controls:

- non-root user and read-only filesystem where practical;
- minimal runtime and separate OCR/benchmark images/groups;
- image/action/model digest or immutable revision pins;
- SBOM and signed/provenance-bearing release;
- dependency, secret, license, and image scanning;
- documented patch cadence.

The local environment's uv pip check reported **nvidia-cusparselt-cu13** built for another platform. This may be an isolated environment artifact; it needs reproduction in clean CI rather than a repository security conclusion.

### Development deployment defaults

Compose publishes PostgreSQL and MCP ports and uses known local credentials (**docker-compose.yml:1-30**). This is acceptable only as an explicitly local profile.

Controls:

- bind DB to loopback or do not publish it;
- bind MCP to loopback by default;
- label development credentials;
- prohibit this profile for remote deployment;
- add a separate hardened compose/example.

## Historical SQL safety assessment at commit 5684a34

The table records the reviewed commit. Current database-role, transport, ACL-provenance, per-connection identity, catalog-readiness, and migration-scale controls are summarized in the overlay.

| Question | Finding |
|---|---|
| Arbitrary SQL exposed? | No. |
| Confirmed SQL injection? | No. |
| Parameterization? | Generally yes through asyncpg placeholders. |
| FTS sanitization? | Present; uses internal sanitation/plainto_tsquery. |
| Read-only enforcement? | No; serving role is broadly write/DDL capable. |
| Query allowlist? | SQL is internal fixed code; no caller query allowlist is required today. |
| Statement limits/timeouts? | Some exist, but should be standardized per tool. |
| Connection handling? | Shared asyncpg pool and deliberate teardown are sound. |
| Transaction behavior? | Good within operations; corpus publication across operations is non-atomic. |
| Destructive safeguards? | Operator hidden by default; no caller auth, approval, or role boundary when enabled. |
| Sensitive error control? | Inconsistent; needs stable safe mapping. |

Recommended SQL tests:

- hostile strings across every search/filter/ID/date input;
- statement timeout and cancellation;
- serving-reader privilege denial for every write/DDL class;
- operator writer denial outside staging/job tables;
- migration role isolation;
- concurrent migration/advisory lock;
- corpus generation consistency under failure;
- error redaction.

Do not add an arbitrary SQL MCP tool. If an analytical SQL capability is ever required, expose fixed parameterized report definitions or a parsed read-only AST with a dedicated replica/role, strict statement/resource bounds, and independent threat review.

## Historical secrets and configuration assessment at commit 5684a34

The missing-control list below records the reviewed commit. The current repository now has fail-closed remote configuration, exact OpenShift Secret-key references, PostgreSQL CA mounts, `verify-full` enforcement, separate DSNs, and startup validation; bank-applied values, rotation, and acceptance evidence remain open.

Confirmed:

- database URL is required and fails fast;
- .env is ignored;
- no loaded value was printed during review;
- example/default credentials are present for local development;
- at the reviewed commit, .mcp.json used a hard-coded path and environment substitution; the working tree now uses the portable packaged entry point.

Missing:

- typed secret references/provider support;
- rotation procedure;
- remote auth secrets/keys;
- audience/issuer validation;
- TLS and database SSL policy;
- production-variable contract;
- startup configuration validation.

Requirements:

- never log full DSN/token/key;
- allow file/secret-manager injection;
- define key rotation and token revocation;
- use separate credentials per plane/role/environment;
- require TLS/SSL modes in remote deployments;
- validate settings before listener startup;
- test redaction and missing/invalid configuration.

## Path traversal and unsafe file access

No caller-controlled file-path MCP tool exists. The inspected archive path reads members in memory rather than writing member names to disk, so a direct Zip Slip/path traversal finding was not confirmed.

The current acquisition path additionally bounds archive member count, compressed/uncompressed bytes, expansion ratio, decoded response size, retries, redirects, and approved public-address hosts, and uses hardened Office XML parsing. Residual risk remains in:

- future private/local ingestion features;
- manually invoked repair scripts;
- source authenticity and malware detection;
- the non-atomic interval between DNS validation and socket connection;
- bank platform egress enforcement.

Any future file-ingestion tool must:

- accept artifact IDs or approved roots rather than arbitrary paths;
- resolve/canonicalize and enforce root containment;
- reject symlinks/device files;
- enforce size/type limits;
- use a quarantine directory and non-privileged worker;
- never make it a public remote tool.

## Observability security

Current tool-boundary logging is metadata-only by default and correlation-aware; metrics are thread-safe; optional telemetry has a distinct append-only identity. The remaining operational design must avoid becoming a second data leak:

- generate request IDs at transport ingress;
- avoid query/result text in logs, metrics labels, and traces;
- separate operational logs from auditable operator events;
- record auth decisions and corpus publish events;
- protect/administer log access;
- define retention and deletion;
- avoid high-cardinality principal/document/query labels;
- record corpus/model/schema versions for diagnosis;
- alert on auth failures, throttling, corpus age, vector/section coverage, quality quarantine, sync failure, and privilege denial.

## Backup, recovery, and integrity

Repository upgrade documentation requires a restorable backup and size-matched rehearsal before dangerous migration approval. The current working tree adds a guarded populated-v2-to-current-schema rehearsal and recovery-evidence schema v2, including 29 managed relations, activation sequence, application-DSN exclusion, six LOGIN profiles, content/vector fingerprints, identity/catalog/readiness checks, and bounded PostgreSQL subprocess cleanup. Focused automated tests cover those contracts. Repository capability is not bank-accepted recovery evidence, so the following remain open:

- bank-managed backup/PITR and a retained, accepted bank-like logical restore;
- measured and approved RPO/RTO;
- authoritative source-artifact recovery;
- rollback/reactivation of a prior complete corpus image; active identity and
  invalidation exist, but independently servable immutable generations do not;
- key/config recovery;
- recovery on the bank's actual PostgreSQL, storage, LOGIN, and network controls.

Minimum production control:

1. PostgreSQL automated backups/PITR.
2. Immutable source-artifact backup/versioning.
3. seed/evaluation/config manifests with checksums.
4. quarterly disposable restore drill.
5. documented RPO/RTO and owner.
6. post-restore citation/hash/vector/section integrity suite.
7. active-generation rollback independent of schema rollback.

## Commercial-use and licensing controls

The MIT license continues to permit commercial use, modification, sublicensing, and sale (**LICENSE:1-20**). At the reviewed commit, README's statement that there was no license was incorrect; the current working tree instead documents the code/data/service boundary in **docs/LICENSING_AND_PROVENANCE.md**. That documentation does not revoke rights already granted under MIT or independently establish source-data rights.

Security/technical conclusion:

- a runtime license check in distributed local code can be removed;
- authentication can control a hosted service, not the rights already granted to local MIT copies;
- obfuscation, phone-home, or model/tool checks are not reliable legal controls and would damage local/enterprise trust;
- future private datasets, validated knowledge packs, operator services, and hosted access can have separate technical entitlements if legally structured.

Required next step is legal advice, contributor/source-rights inventory, and a deliberate future code/data/service licensing model. This review is not legal advice.

## Target control matrix

| Threat | Prevent | Detect | Recover |
|---|---|---|---|
| Unauthorized HTTP use | OAuth/bearer verification, scopes, Host/Origin, private network | auth-denial and unusual-rate metrics | revoke token/key, block principal/IP |
| Operator abuse | separate plane/role, private ingress, job locks/idempotency | immutable operator audit events | cancel job, rollback active generation |
| Corpus downgrade | immutable artifacts/versions, staged validation | version/hash/generation assertions | atomic prior-generation switch |
| Stale index | one generation across document/section/vector | coverage/hash consistency metrics | rebuild staging index, do not publish |
| Query leakage | redacted default logs, narrow telemetry | log-policy tests/scans | delete where possible, rotate access |
| SSRF | host/IP/redirect allowlist | denied-egress events | quarantine job, investigate source |
| Archive bomb | byte/member/ratio/time limits | resource-limit alerts | terminate isolated worker |
| Prompt injection | untrusted-data envelope, least privilege, no public admin | injection benchmark/tool-call audit | reject output/action, review corpus |
| SQL abuse | fixed parameterized queries, read-only role | statement/privilege-denial metrics | cancel query, revoke role/session |
| Supply-chain drift | immutable pins, SBOM, signed builds | scheduled scans | rebuild patched image, rollback |
| Tenant leakage | single tenant or identity/RLS/cache isolation | cross-tenant negative tests/audit | revoke, isolate, incident response |

## Security acceptance gates

No remote production release until all pass:

- unauthenticated HTTP returns 401;
- invalid Host/Origin returns 403;
- wrong audience/scope cannot list/call protected tools;
- public endpoint cannot list operator tools under any configuration;
- serving credentials fail all corpus write/DDL attempts;
- production logs contain no query/result excerpts;
- oversized/expensive calls are bounded and throttled;
- prompt-injection cases cannot cause operator calls or secret access;
- SSRF/redirect/private-IP and archive-limit tests pass;
- active corpus remains unchanged after failed ingestion/indexing;
- strict serving and Phase 2 read the same persisted active-release identity,
  and mutation/drift causes fail-closed unavailability rather than silent use;
- citation reconstruction passes for the release corpus;
- evaluation/model claims use a bank-owned trust policy with separated signer
  keys and owners, validity/revocation, canonical release-identity bindings, forward
  legal-release rotation, v2 page-review owner assertions authorized by a
  separate reviewer registry, evidence for the human review action, and
  independently promoted current policy SHA/version and deployment-scope pins;
  development-mode or source-checkout policy validation alone is never treated
  as bank authorization;
- package/image builds and scans pass;
- backup restore and health/readiness drills pass.

## Important unknowns

- exact bank OpenShift AI values and accepted design for ingress/Route, bank IdP, application/service/PostgreSQL CAs, registry, egress allows, shared rate limits, monitoring, SCC, namespace policy, and retention; repository starter contracts exist, but bank application and acceptance do not;
- whether the bank deployment actually keeps public and operator profiles as separate workloads and gives the operator only private, scope-protected reachability, as the repository now requires;
- actual bank PostgreSQL LOGINs, memberships and ACL provenance; TLS/HBA policy; shared-cluster role names; database size; backups; and restore/upgrade results. Repository role, target-database, `verify-full`, per-connection identity, schema-v6 catalog, populated-v2-to-current rehearsal, and recovery-v2 contracts exist, but the opt-in actual-LOGIN and restore/PITR proof have not run in that bank environment;
- OpenShift AI monitoring/log export, persistent services, backup ownership, and incident-response integration;
- whether private documents or sensitive queries are already processed;
- tenant/shared-service plans; use single-tenant and prohibit private-corpus ingestion until decided;
- current locked dependency and built-image vulnerabilities on the bank-approved runner/feed, plus the bank's signing, promotion, admission, malware, and source-authenticity gates. The repository lane produces SBOMs and unsigned provenance and applies a fresh-vulnerability policy, but does not sign or promote;
- a bank-issued instance of the implemented trust-policy schema and approval
  identities for four separated evaluation signers (corpus, expert dataset,
  legal curator, legal-release certifier) and separate legal-source reviewer
  owners/revocations, plus external RBAC custody for the policy root/keyring,
  atomic policy/pin/scope promotion, stale-policy detection, and compromise
  response. The repository can require exact policy SHA/version/scope pins,
  roles/owners, rotation/revocation and approved release identities under the
  documented hash contract,
  but it cannot prove that supplied mounts/pins are bank controlled or discover
  that an externally supplied pin is stale; the tracked corpus/dataset are not
  signed;
- whether real legal evidence will reproducibly derive page text/mappings from
  retained raw source bytes, who authenticates or attests the human behind the
  policy-bound reviewer owner assertion, whether a reviewer signature/action
  record is required, and how historical Citation packs will be retained/replayed
  across checkpoint history;
- numeric corpus detection/publication/maximum-age objectives and the governed
  resolution of the current 8,286-versus-9,675 chunk drift;
- written contributor/source/data provenance supporting the owner's statement that usage rights are acceptable;
- measurable incident response, retention, availability, freshness, RPO, and RTO targets; “immediate” is the business expectation but is not yet a testable objective.

Detailed issue IDs and effort are in [GAP_REGISTER.md](GAP_REGISTER.md). The implementation order is in [ROADMAP.md](ROADMAP.md).
