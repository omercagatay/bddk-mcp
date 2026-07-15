# Testing and Evaluation Strategy

## Purpose

The test strategy must answer four different questions:

1. **Does the software behave correctly?**
2. **Does the MCP host/model select and call the right tools correctly?**
3. **Does retrieval return the legally applicable, traceable evidence?**
4. **Does the final answer make only claims supported by that evidence?**

The current repository now has reliable protocol and real-MCP runner coverage. It still does not answer questions 2–4 at a product-evidence level because named host/model runs, representative expert retrieval judgments, legal-currentness fixtures, reconstructable citations, and claim-level grounding remain incomplete.

## Current baseline

Review execution against commit **5684a34c10e6d90bc22d6ab2a6466944afb6bf81**:

| Check | Result |
|---|---|
| Test discovery | 613 total; 610 selected and 3 GPU-marked deselected |
| Default suite | 526 passed, 84 skipped, 3 deselected |
| Ruff lint | Passed |
| Ruff format check | Passed; 138 files formatted |
| Runtime tools | 15 public, 26 operator-enabled |
| Resources/prompts | 0/0 |
| Document quality scorer | 318 total; 313 clean, 5 warning, 0 fail; 99.5692 under current rules |
| Package build | Failed due multiple top-level package discovery |
| Dependency consistency | Failed on local cross-platform nvidia-cusparselt-cu13 artifact; requires clean-CI confirmation |
| Compose configuration | Passed |

Skipped:

- 77 PostgreSQL-dependent tests because no disposable test database was available;
- 7 optional Chandra tests because the package/GPU stack was absent;
- 3 GPU tests deselected by repository configuration.

This review deliberately did not start a database, mutate schemas, access live BDDK services, download/run models, inspect secrets, or use production clients.

### Implementation progress overlay — 2026-07-15

This checkpoint describes the current working tree; the executed-check table above remains the reviewed-commit baseline. **Complete** means a repository gate or focused automated contract exists, not that every target environment has passed it. **Partial** means meaningful coverage exists with important gaps. **Open** means the evaluation outcome remains unproved.

| Test/evaluation slice | Status | Current evidence and remaining gap |
|---|---|---|
| Installed MCP transport E2E | Complete | The official client exercises the installed stdio subprocess through initialize/version/list/call/invalid-extra/recovery/shutdown and checks protocol-only stdout. Streamable HTTP tests cover initialize/list/call, health, Host/Origin, JWT/JWKS, scope, operator opt-in, and shutdown (**tests/test_mcp_stdio_e2e.py; tests/test_mcp_http_runtime.py**). |
| Tool contracts and protocol errors | Partial | The 15-public/28-operator registry owns strict generated argument models and risk annotations; stable privacy-safe error codes are tested. Six retrieval tools validate structured evidence payloads (**tests/test_public_input_contracts.py; tests/test_structured_retrieval_outputs.py**). A uniform output contract and audit-grade Citation remain open across the full surface. |
| PostgreSQL and distribution gates | Complete as repository gates | CI requires PostgreSQL-backed tests on Python 3.12/3.13 and separately builds, inspects, installs outside the checkout, and invokes wheel/sdist CLIs (**.github/workflows/ci.yml; scripts/verify_distribution.py**). This records configured repository gates, not a target-bank runner result. |
| Security and deployment tests | Partial | Host/Origin/JWT/scope/body/rate/concurrency, exact DB identity/TLS/role denial, SSRF/redirect/size, archive-bomb, and static OpenShift security contracts exist (**tests/test_http_security.py; tests/test_db_identity.py; tests/test_db_transport.py; tests/test_outbound_http.py; tests/test_openshift_manifests.py**). Shared ingress controls, prompt-injection elevation, executable image scans, and a real bank-like OpenShift deployment remain unproved. |
| Database lifecycle and recovery | Partial | Checksum migrations, exact schema-owner/target verification, catalog attestation, role/grant assets, durable jobs, clean/legacy/populated-v2 upgrades, and rollback injection are covered. A populated v2 requires the explicit v3 maintenance approval only after backup and size-matched rehearsal (**tests/test_migrations.py; tests/test_legacy_migration_adoption.py; tests/test_postgres_job_repository.py**). Whole-corpus generations, low-downtime large-corpus migration, backup/restore, PITR, and bank DBA evidence remain open. |
| Real MCP Phase 2 runner | Complete as a harness | Phase 2 uses official `ClientSession` transports for stdio and `/mcp`, paginates live discovery, executes actual `call_tool`, fails cases on MCP errors, sanitizes audit artifacts, and records schema/server/protocol/corpus/dataset identities (**benchmark/phase2_e2e.py; benchmark/audit.py; tests/test_benchmark_phase2.py**). No named model/client score or product recommendation follows until an approved corpus and grader suite is run. |
| Observability tests | Partial | Correlation IDs and thread-safe request/error/latency metrics are wired at the MCP/tool boundary and telemetry uses a separate verified writer identity (**tests/test_metrics.py; tests/test_mcp_runtime.py; tests/test_telemetry.py**). Standard exporter/tracing, SLO, retention, load, and bank monitoring tests remain. |
| Retrieval, citation, and answer evaluation | Open | Representative Turkish retrieval judgments, immutable legal-version/currentness cases, citation reconstruction, claim-level grounding, abstention, and unsupported-addition grading remain. |
| Model/client compatibility and operations | Open | No supported named-host/model matrix, reproducible live-model comparison, load/resilience baseline, recovery drill, or real OpenShift deployment acceptance has been produced. |

No aggregate current-suite count is asserted here: the trustworthy current claims are the explicit repository gates and focused contracts above.

## Benchmark defects identified at the reviewed commit and current status

### Phase 1 is function calling, not MCP

**benchmark/phase1_tools.py:36-58** sends static OpenAI-style function schemas to an OpenAI-compatible Chat Completions API. This can be a useful model tool-selection test, but it is not MCP discovery, transport, or host compatibility. It must be labeled accordingly.

### Static schemas drifted from runtime at the reviewed commit

At the reviewed commit, the benchmark defined 23 static schemas while runtime exposed 15 public or 26 operator tools. The working tree now exports all 28 OpenAI-style function schemas from the canonical 15-public/13-operator registry, eliminating that inventory drift. Benchmark Phase 1 remains function calling rather than MCP. Phase 2 now discovers the live MCP schema and therefore no longer shares this defect.

### Phase 2 did not call this MCP server at the reviewed commit; this is corrected

The reviewed-commit implementation POSTed to nonexistent **/call-tool**, described stdio without launching it, and converted HTTP failures into result strings. Current **benchmark/phase2_e2e.py** opens an official SDK session over stdio or Streamable HTTP **/mcp**, paginates `tools/list`, invokes `ClientSession.call_tool`, and raises on transport, protocol, malformed-model, or tool errors. **tests/test_benchmark_phase2.py** verifies those fail-closed paths and checks that the obsolete route is absent.

### Grounding/citation scores are not claim validation

The source-trace score examines tool results, not whether final-answer claims cite and follow the evidence. The code grader checks required numbers/dates and does not penalize unsupported additions (**benchmark/graders.py:20-66; benchmark/scoring.py:191 onward**).

### Silent grader fallback

If the Anthropic key/API is unavailable, model grading silently becomes the weak code score (**benchmark/graders.py:91-126**). A benchmark must never change methodology without failing or making the run non-comparable.

### Data is too small and weakly sourced

- 3 source-grounded gold cases in **benchmark/gold_cases.yml**
- 30 NLI pairs against a stated target of 500 in **data/bddk_nli/metadata.json**
- generic source labels rather than immutable document/version/provision/hash citations
- unclear independent annotation and adjudication

No client/model recommendation should be based on repository presence or harness tests alone. A recommendation requires a recorded successful run on the approved corpus, expert-reviewed cases, pinned host/model/hardware, and the claim/citation methodology below.

## Quality model

Use layered gates. A higher layer never compensates for a failed lower layer.

~~~mermaid
flowchart TB
    Static[Static, unit and schema tests]
    Integration[PostgreSQL, ingestion and migration integration]
    MCP[MCP transport and contract E2E]
    Retrieval[Retrieval and citation evaluation]
    Answer[Claim grounding and abstention]
    Security[Security, abuse and privacy]
    Ops[Performance, deployment and recovery]
    Client[Client/host/model compatibility]

    Static --> Integration --> MCP --> Retrieval --> Answer
    MCP --> Security
    Retrieval --> Security
    MCP --> Ops
    Answer --> Client
~~~

## Test layers

### Layer 0 — Static and repository integrity

Required on every pull request:

- Ruff lint and format;
- strict type checking for production modules;
- import-cycle/dead-code checks where useful;
- lockfile frozen check;
- package metadata and license consistency;
- wheel and sdist build;
- install wheel in a clean environment;
- console-script help/startup validation;
- generated docs/tool schema diff;
- seed manifest/hash/schema validation;
- secret scan;
- dependency/license scan;
- Mermaid/Markdown link checks for durable docs.

Acceptance:

- no warning-only required gate;
- package builds/install/imports;
- checked-in generated artifacts match the runtime registry;
- no unexplained dependency or secret finding.

### Layer 1 — Unit tests

Retain and expand isolated tests for:

- configuration parsing and invalid combinations;
- tool inputs/outputs/errors;
- Turkish normalization and legal-reference parsing;
- section hierarchy;
- pagination/location semantics;
- citation construction;
- query expansion and hybrid scoring;
- quality rules and known-failure registry;
- source URL/redirect/archive policy;
- redaction;
- authorization decisions;
- corpus generation state machine;
- migration helpers;
- model/grader result parsing.

Property-based/fuzz candidates:

- Turkish Unicode, dotted/dotless I, combining marks, punctuation, long strings;
- document IDs and aliases;
- article/fıkra/bent/annex/range expressions;
- FTS and filter inputs;
- malformed MIME/HTML/ZIP metadata;
- citation offset/page round trips;
- schema extra fields/boundaries.

### Layer 2 — PostgreSQL integration

Run against disposable pgvector PostgreSQL in a required CI job. Database absence must fail the job, not skip it.

The current repository covers clean and idempotent migrations, strict legacy adoption, populated-v2 refusal/approved v3 backfill, transactional rollback injection, catalog attestation, durable job concurrency/leases, fail-closed retrieval publication, and role/identity/write-denial contracts. The list below is the full target; supported-release restore, target-bank identities, low-downtime large-corpus migration, whole-corpus generations, and PITR remain residual work.

Test:

- clean migrations;
- upgrade from every supported prior release;
- migration advisory lock and concurrent startup;
- extension setup under migrator role;
- public serving under read-only role;
- ingestion and publisher role grants/denials;
- statement timeouts and cancellation;
- document/version/provision constraints;
- corpus staging, validation, atomic activation, and rollback;
- document/section/chunk/vector hash/generation consistency;
- FTS and dense retrieval;
- telemetry disabled/enabled privacy;
- transaction failure injection;
- backup/restore integrity in scheduled CI.

Database fixtures must use unique schemas/database IDs and guaranteed cleanup. Never point tests at a developer or production database.

### Layer 3 — Ingestion and artifact tests

Use checked-in, legally permitted, minimized fixtures representing:

- normal HTML regulation;
- HTML tables, lists, footnotes, images, and formulas;
- generated/static PDF;
- scanned PDF;
- legacy DOC;
- iframe source;
- ZIP annex with DOCX tables/formulas;
- malformed/oversized/unsupported content;
- upstream page whose content changes;
- duplicate/alias artifacts;
- amendment and repeal notices.

For each fixture assert:

- allowed source/redirect policy;
- artifact hash/MIME/size/retrieval metadata;
- deterministic normalized hash;
- page boundaries;
- table/formula preservation;
- section hierarchy;
- quality/quarantine status;
- no overwrite of a prior valid publication after extraction failure;
- no active-generation change until validation/publish;
- known-failure registry behavior;
- full provenance.

Maintain a small visual/manual gold set for priority formula/table documents. Automated text equality alone cannot validate layout-dependent meaning.

### Layer 4 — MCP protocol and contract

Use the official MCP Python client rather than custom JSON-RPC scripts as the primary test.

#### Stdio

- launch the installed console entry point as a subprocess;
- initialize;
- verify project name/version/capabilities;
- list exact public tools;
- validate input/output schemas and annotations;
- call one catalog, search, exact document, and exact section tool;
- validate structured success/error;
- cancel and shut down cleanly;
- verify stdout contains protocol only;
- repeat with operator profile in an isolated environment.

#### Streamable HTTP

- start on an ephemeral loopback port;
- validate initialize/list/call/error;
- current and one prior supported protocol revision;
- session/stateless behavior;
- valid/invalid Host and Origin;
- missing, invalid, expired, wrong-audience, and wrong-scope credentials;
- public endpoint never lists operator tools;
- body/query/result/rate/concurrency limits;
- disconnect, cancellation, timeout, and graceful shutdown;
- no sensitive response/error leakage.

#### Contract snapshots

Assert:

- exact 15 public and 28 operator names until intentionally versioned;
- every property has description and applicable bounds/enums/formats;
- no unexpected fields;
- every output has stable status/data/citations/warnings/meta;
- errors use isError and stable codes;
- annotations reflect behavior;
- project version is not SDK version;
- docs and benchmark schemas are generated from the same registry.

### Layer 5 — Retrieval evaluation

#### Evaluation unit

Evaluate at multiple levels:

- instrument;
- legal version;
- provision;
- evidence excerpt;
- final answer.

A document-level hit is insufficient when the question names an article, fıkra, bent, exception, effective date, or amendment.

#### Dataset design

Start with at least 100 expert-reviewed queries and grow to 300–500. Stratify by:

- exact document/decision number;
- exact article, temporary article, paragraph, bent, annex, table;
- natural-language concept;
- Turkish inflection and diacritics;
- abbreviation/acronym and Turkish/English synonym;
- cross-document reference;
- amendment/repeal/currentness;
- historical as-of date;
- draft versus effective;
- numeric threshold, formula, or table;
- no-answer/out-of-scope;
- misleading near-match/hard negative;
- duplicate/alias source;
- extraction-degraded case.

Domain quotas should include:

- TFRS 9 and expected credit loss;
- PD, LGD, EAD, SICR;
- IRB/İDD;
- ICAAP/İSEDES;
- credit risk and credit-risk mitigation;
- liquidity;
- capital adequacy;
- interest-rate risk/IRRBB;
- model validation;
- information systems/internal systems where in corpus scope.

Each item must record:

- query and language;
- intent and difficulty;
- applicable as-of date/entity scope;
- accepted instrument/version/provision IDs;
- acceptable alternative evidence;
- hard negatives;
- official source/artifact hash;
- expected legal status;
- expected answer/abstention behavior;
- annotators, adjudicator, approval, and change history.

#### Retrieval metrics

Report:

- instrument Recall@1, @3, @5;
- provision Recall@1, @3, @5;
- MRR and nDCG at provision level;
- exact-reference resolution rate;
- temporal/status filter accuracy;
- hard-negative rejection;
- no-answer precision/recall;
- citation reconstruction rate;
- latency percentiles;
- results by domain, query type, status, extraction quality, and difficulty.

Do not combine these into one headline score without preserving component results.

#### Ablations

Measure:

- lexical only;
- dense only;
- RRF hybrid;
- with/without Turkish glossary;
- with/without morphological expansion;
- reranker off/on;
- alternate chunking/provision units;
- embedding model/revision changes.

Require confidence intervals or repeated paired runs for nondeterministic/model-dependent changes. A change ships only if aggregate improvement does not conceal unacceptable regression in exact references, currentness, or critical domains.

### Layer 6 — Citation correctness

Every citation must pass deterministic checks:

1. instrument/version/provision exists in the recorded corpus generation;
2. source artifact hash matches;
3. normalized text hash matches;
4. excerpt exactly reconstructs from normalized text;
5. source page/coordinates reconstruct when physical-page citation is claimed;
6. display windows are never labeled source pages;
7. legal status/effective dates are validated, not inferred;
8. source URL belongs to the authoritative allowlist;
9. quality warning/quarantine state is present;
10. final answer uses the citation for the claim it follows.

Metrics:

- citation precision;
- citation recall for substantive claims;
- exact page/range rate;
- unsupported citation rate;
- wrong-version rate;
- wrong-status rate;
- orphan citation rate;
- quality-warning omission rate.

### Layer 7 — Final-answer grounding

Build a claim-evidence evaluator:

1. split answer into atomic factual/legal claims;
2. map each claim to cited evidence;
3. classify supported, partially supported, contradicted, unsupported, or non-factual;
4. validate citation location and version;
5. separately score answer completeness, abstention, and language quality.

Critical rules:

- unsupported added claims reduce score;
- correct numbers do not compensate for hallucinated interpretation;
- a tool result containing a source does not count unless the answer cites it correctly;
- a currentness claim fails if temporal data is unknown;
- legal advice/opinion must be distinguished from source-grounded fact;
- grader unavailability fails the run or clearly marks it non-comparable;
- a model grader is calibrated against a human-labeled set and never acts as sole oracle.

Use deterministic checks first. Use model graders only for semantic entailment/coverage where necessary, with pinned grader/prompt/revision and periodic human calibration.

### Layer 8 — Tool-calling and host/model evaluation

Evaluate the whole host-plus-model profile, because models do not independently implement MCP.

For each profile record:

- MCP client/host and version;
- transport and protocol revision;
- actual discovered tool schemas;
- model and revision/quantization;
- system/server instructions passed by host;
- decoding parameters;
- hardware/backend;
- corpus generation;
- embedding/reranker profile;
- trial count/random seeds where available.

Test cases:

- select correct tool among near-neighbors;
- avoid operator tools for read questions;
- exact valid arguments;
- recover from invalid input;
- paginate/follow-up correctly;
- use section before full document when appropriate;
- retrieve and cite multiple provisions;
- abstain when evidence unavailable;
- resist document prompt injection;
- handle quality warnings;
- avoid repeated/unnecessary calls;
- handle timeout/retryable versus non-retryable errors.

Profiles of interest:

| Host/client | Model examples | Purpose |
|---|---|---|
| Claude host | supported Claude models | mainstream MCP client behavior |
| Codex/ChatGPT host | supported OpenAI models | stdio/HTTP and structured content |
| LM Studio | selected local models | local MCP host/schema behavior |
| Generic OpenAI-compatible adapter | GPT-OSS and other local models | function/tool selection behind a real MCP host |
| Official SDK reference client | no model | isolate protocol/server behavior |

Do not claim “GPT-OSS MCP performance” without naming the host/orchestrator and tool adapter.

### Layer 9 — Security testing

Required:

- auth/scope/Host/Origin negatives;
- rate/body/query/result/time/concurrency limits;
- every public tool denied under missing read scope;
- every operator tool denied under read scope;
- serving-reader DB privilege negatives;
- hostile SQL-like input across every parameter;
- prompt-injection documents attempting operator calls, secret retrieval, or external fetch;
- SSRF absolute iframe, redirect, DNS/IP, metadata endpoint, and private network tests;
- archive member/size/ratio/time tests;
- path traversal/symlink tests before any local file-ingestion feature;
- log/trace/metric redaction;
- error-message/DSN/token/path leakage;
- dependency/container/secret scan;
- multi-user/tenant negatives if tenancy is added.

Run static security checks per PR, dynamic abuse tests in CI, and periodic independent penetration review before a public enterprise release.

### Layer 10 — Performance, load, and resilience

Measure separately:

- cold process/startup;
- ready startup with prebuilt active generation;
- cold/warm embedding model;
- lexical, dense, hybrid, reranked searches;
- exact document/section fetch;
- concurrent mixed workload;
- operator ingestion/indexing isolated from serving;
- database failover/latency;
- upstream timeouts/retries/circuit breaker;
- graceful shutdown and job cancellation.

Record p50/p95/p99 latency, throughput, errors, CPU, memory, DB pool/queries, model utilization, result sizes, and cache hit rate.

Initial acceptance budgets should be set from measured local/enterprise targets, not invented here. Required invariant: serving readiness cannot depend on downloading a model, importing seed, running DDL, or embedding the corpus.

### Layer 11 — Deployment and recovery

Test:

- package build/install;
- non-root container build/run;
- OpenShift manifest validation and deployment smoke in a disposable bank-like namespace;
- separate public/operator ServiceAccounts, security contexts, NetworkPolicies, Routes/ingress authentication contract, Secrets, probes, and resource limits;
- liveness/readiness;
- private DB networking;
- missing/invalid configuration fail-fast;
- migration release job;
- upgrade from previous supported version;
- failed migration forward-fix/rollback procedure;
- corpus generation publish/rollback;
- rolling/multiple-replica behavior;
- backup and disposable restore;
- post-restore document/section/vector/citation integrity;
- observability and alert smoke tests.

## Seed and corpus quality gates

Every corpus generation must have a signed/checksummed manifest containing:

- schema and generator version;
- source inventory and scope/exclusions;
- artifact hashes;
- instrument/version/provision counts;
- document/chunk/section counts;
- extraction method/version;
- quality and quarantine counts;
- known-failure registry revision;
- duplicate/alias decisions;
- embedding/reranker model revisions;
- vector coverage;
- citation reconstruction results;
- benchmark results;
- reviewers/approval;
- generated and published timestamps.

Block publication if:

- a known failure appears clean;
- a parser-detectable document lacks required sections;
- document/section/chunk hashes or generation IDs disagree;
- required vectors are missing;
- an official artifact/citation cannot be reconstructed;
- a legal-status assertion lacks validation;
- a priority formula/table document fails review;
- benchmark guardrails regress beyond approved tolerances.

The reviewed-commit quality score of 99.5692 must not be used as an overall corpus-accuracy score. It checked implemented text anomalies only and conflicted with 11 configured failures. The working tree now applies those 11 cases through one packaged fail-closed registry, removing that direct inconsistency; the resulting score still measures only implemented extraction-quality rules, not legal accuracy, completeness, currentness, or citation correctness.

## Reproducibility contract

Every benchmark report must include:

- git commit and dirty-state flag;
- Python/package lock hash;
- MCP SDK/protocol/client versions;
- public/operator registry hash;
- corpus generation and artifact manifest hash;
- embedding and reranker names/revisions;
- database/pgvector version and retrieval settings;
- model/host/backend/quantization;
- prompts and grader revision;
- dataset revision and annotation policy;
- hardware and timing environment;
- seeds/trials/temperature;
- skipped/failed cases and exact reason.

Never:

- silently fall back to a different grader;
- substitute static schemas for live schemas without labeling the run;
- count a transport exception as a tool result;
- compare runs with different corpus/schema/grader configurations as if equivalent;
- omit negative or failed cases from denominator.

## Data governance

Before expanding gold data:

- verify rights to redistribute each source artifact/excerpt;
- use stable artifact/version/provision hashes;
- document annotator qualifications and conflicts;
- require two independent annotations for legal applicability/currentness;
- adjudicate disagreements;
- track corrections and retirement;
- keep organization-specific confidential test sets separate;
- report public and private benchmark results distinctly;
- avoid claiming the NLI/terminology data is expert-validated without evidence.

## Continuous integration design

The pull-request implementation already gates lint/format, unit tests, required PostgreSQL tests on Python 3.12/3.13, a dedicated actual-LOGIN/ACL role contract, official-client MCP tests, distribution build/content/external-install, Dockerfile static checks, and checksum-pinned mandatory Kustomize rendering. Items below that are not in those configured jobs remain target lanes rather than implied delivered coverage.

### Pull-request lane

- static/lint/type;
- unit tests;
- required disposable PostgreSQL integration;
- package build/install;
- generated contract/docs check;
- official-client stdio/HTTP E2E;
- seed/citation integrity on compact fixture;
- security/redaction tests;
- dependency/secret scan;
- container build/smoke.

### Main/release lane

- full PostgreSQL matrix;
- migration upgrade;
- full seed integrity;
- retrieval/citation regression;
- image scan/SBOM;
- deployment smoke;
- release artifact/signature/provenance.

### Scheduled lane

- live-source drift monitor in a controlled non-production environment;
- optional OCR/GPU suite;
- model/client matrix;
- load/resilience;
- vulnerability rescans;
- backup restore drill;
- larger expert benchmark.

Required jobs fail if prerequisites are absent. Optional jobs report an explicit not-run status and never raise the release confidence score.

## Release gates by horizon

### Stable local MCP

- official-client stdio E2E;
- exact runtime tool contract;
- package install;
- section-ready bootstrap;
- known-failure warnings;
- no production-text logging default.

### Remote public beta

- authenticated HTTP;
- Host/Origin/rate-limit tests;
- public/operator separation;
- read-only serving role;
- health/readiness and container hardening;
- atomic corpus generation;
- backup/restore evidence.

### Regulatory retrieval release

- canonical versions and status;
- reconstructable citations;
- 100 or more expert-reviewed queries;
- domain/currentness/formula guardrails;
- claim-level grounding and abstention metrics;
- published reproducibility manifest.

### Regulatory knowledge release

- validated amendment/repeal/citation edges;
- approved obligation/control/audit mappings;
- reviewer workflow and audit trail;
- multi-hop/knowledge evaluation;
- enterprise data-governance and access controls.

## Initial quantitative acceptance targets

Targets should be approved by domain and product owners after a baseline. The following are proposed release gates, not current measurements:

| Metric | Local stable | Regulatory retrieval release |
|---|---:|---:|
| MCP initialize/list/call contract | 100% | 100% |
| Exact legal-reference resolution on gold set | at least 95% | at least 99% |
| Provision Recall@5 | baseline recorded | at least 95% overall and no priority domain below 90% |
| Citation reconstruction | 100% on release corpus | 100% |
| Wrong-version/currentness claims | not supported | 0 on release set |
| Known-failure warning propagation | 100% | 100% |
| Unsupported substantive claim rate | baseline recorded | under 2%, with 0 high-impact cases |
| Auth/scope/Origin negative tests | local N/A | 100% |
| Query/result text in production logs | 0 | 0 |
| Serving-role write/DDL attempts succeeding | 0 | 0 |
| Required CI skips | 0 | 0 |

Any threshold exception must be documented with affected cases, risk owner, expiry date, and compensating control.

## Next evaluation deliverables

The original first two deliverables are complete: Phase 2 uses official MCP stdio/HTTP sessions, and Phase 1 schemas derive from the canonical runtime registry. The next reviewable deliverables are:

1. Build a versioned 25-case protocol/tool-routing smoke set that records the live schema hash and includes recovery, pagination, unnecessary-call, and operator-avoidance cases.
2. Build a 100-query expert-reviewed Turkish retrieval/citation set across the selected regulatory domains.
3. Add exact-reference, legal-currentness, amendment, hard-negative, no-answer, degraded-extraction, table/formula, and prompt-injection cases with immutable source provenance.
4. Implement deterministic citation reconstruction against artifact/version/provision/hash/page-or-range evidence.
5. Implement claim-evidence grading, unsupported-addition penalties, abstention/currentness scoring, and human calibration without silent grader substitution.
6. Add a scheduled size-matched migration/reindex rehearsal that records elapsed time, lock behavior, database/WAL growth, and restore evidence before the v3 maintenance approval can be used outside disposable tests.
7. Publish the first versioned baseline across the official client plus selected Claude, Codex/ChatGPT, LM Studio, and GPT-OSS host/model profiles; record skipped/unavailable profiles rather than imputing success.

The roadmap maps these deliverables into reviewable issues: [ROADMAP.md](ROADMAP.md).
