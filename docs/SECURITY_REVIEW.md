# Security Review

Review basis: commit **5684a34c10e6d90bc22d6ab2a6466944afb6bf81**, reviewed 2026-07-14.

## Post-review security status — 2026-07-14

The working tree narrows several risks without creating a production security boundary:

- local Streamable HTTP now defaults to loopback, while container profiles deliberately bind all interfaces;
- normal serving startup validates existing state without DDL, seed import, live cache population, synchronization, or embedding backfill;
- telemetry is disabled by default, and default tool-boundary logs no longer include argument, result, or error content;
- the project MCP configuration no longer contains a user-specific absolute path.

Application authentication and authorization, explicit remote Host/Origin policy, rate/concurrency controls, separate public/operator processes, proved database roles, tenant isolation, atomic corpus publication, non-root OpenShift deployment controls, and security E2E tests remain absent. A serving-reader role is now architecturally plausible with telemetry disabled, but the repository does not provision that role or prove write denial. The original findings below remain commit-scoped evidence and are not silently removed.

## Security conclusion

The repository is suitable only for trusted local experimentation in its current form. It is not safe as an Internet-facing MCP endpoint or as a shared enterprise service.

The most important distinction is between safety features and security boundaries:

- Parameterized SQL, local-only full-document retrieval, response page caps, content sanitation, and a default-hidden operator profile are useful safety features.
- At the reviewed commit, none of them authenticated a caller, authorized a tool, isolated a tenant, made the serving database read-only, or protected an HTTP listener. The post-review lifecycle split reduces database authority needed by `serve`, but no role or remote security boundary is yet provisioned or proved.

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

    Internet -->|currently no repository auth| HTTP
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

Current trust-boundary failures:

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

- SEC-H1 and SEC-H2 are immediate release blockers, not controls that can be presumed to exist at ingress;
- the target baseline is separate public/operator workloads and database roles in a single-tenant namespace;
- private-document ingestion remains disabled until confidentiality and tenancy requirements are approved;
- “immediate” availability/freshness/recovery must be converted to numeric SLO, publication-lag, RPO, and RTO acceptance tests.

## High findings

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

At the reviewed commit, the serving process could not operate with a read-only account. The working tree removes schema/corpus/cache-population/embedding initialization from `serve`, so a public profile with telemetry disabled is intended to support a serving-reader role. That role has not been provisioned or tested; operator tools and optional telemetry still write, and versioned migration/role boundaries remain required.

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

## Medium findings

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

## SQL safety assessment

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

## Secrets and configuration

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

Risk remains in:

- future private/local ingestion features;
- manually invoked repair scripts;
- large/malformed archives;
- source URL/redirect resolution.

Any future file-ingestion tool must:

- accept artifact IDs or approved roots rather than arbitrary paths;
- resolve/canonicalize and enforce root containment;
- reject symlinks/device files;
- enforce size/type limits;
- use a quarantine directory and non-privileged worker;
- never make it a public remote tool.

## Observability security

The desired observability design must avoid becoming a second data leak:

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

No repository runbook or automated evidence covers:

- managed backup/PITR;
- restore verification;
- recovery time/objective;
- corpus artifact recovery;
- migration rollback/forward fix;
- active-generation rollback;
- key/config recovery.

Minimum production control:

1. PostgreSQL automated backups/PITR.
2. Immutable source-artifact backup/versioning.
3. seed/evaluation/config manifests with checksums.
4. quarterly disposable restore drill.
5. documented RPO/RTO and owner.
6. post-restore citation/hash/vector/section integrity suite.
7. active-generation rollback independent of schema rollback.

## Commercial-use and licensing controls

The current MIT license explicitly permits commercial use, modification, sublicensing, and sale (**LICENSE:1-20**). README's statement that there is no license is incorrect (**README.md:461-463**).

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
- citation reconstruction passes for the release corpus;
- package/image builds and scans pass;
- backup restore and health/readiness drills pass.

## Important unknowns

- the owner confirms no external endpoint protection is currently in place; the exact OpenShift AI ingress/Route, TLS, bank identity provider, NetworkPolicy, and rate-limit design remains undefined;
- actual value of BDDK_ADMIN_TOOLS in deployments;
- database roles, grants, SSL mode, network rules, and backups; the project owner is accountable, but technical evidence is not yet defined;
- OpenShift AI namespace, security context, secrets, image registry, routes, monitoring, persistent services, and logs;
- whether private documents or sensitive queries are already processed;
- tenant/shared-service plans; use single-tenant and prohibit private-corpus ingestion until decided;
- current locked dependency and base-image vulnerabilities;
- written contributor/source/data provenance supporting the owner's statement that usage rights are acceptable;
- measurable incident response, retention, availability, freshness, RPO, and RTO targets; “immediate” is the business expectation but is not yet a testable objective.

Detailed issue IDs and effort are in [GAP_REGISTER.md](GAP_REGISTER.md). The implementation order is in [ROADMAP.md](ROADMAP.md).
