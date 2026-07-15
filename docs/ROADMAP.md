# Actionable Development Roadmap

## Roadmap principles

- Stabilize the current product before adding knowledge-platform features.
- Use small, reviewable changes; do not rewrite the system.
- Treat regulatory currentness and citation correctness as product functionality.
- Keep PostgreSQL/pgvector unless measured requirements justify more infrastructure.
- Make each deliverable independently testable.
- Do not publish model or retrieval claims until the evaluation path is real and reproducible.
- Keep local stdio easy; make remote and operator modes explicitly secure.

Priorities:

- P0: blocks safe or correct use
- P1: important for the next stable release
- P2: valuable enhancement
- P3: exploratory or long-term

Effort:

- XS: less than one day
- S: one to three days
- M: up to two weeks
- L: two to six weeks
- XL: more than six weeks

## Implementation progress overlay — 2026-07-15

This is a current-working-tree checkpoint; the roadmap tables below remain the original plan and acceptance definitions. **Complete** means the repository implementation and focused automated contract are present, not that the bank has accepted a deployment. **Partial** means a material slice is implemented but one or more acceptance gates remain. **Open** means the planned outcome is not adequately implemented.

| Roadmap item | Status | Current evidence and remaining acceptance |
|---|---|---|
| H0-01 server lifecycle | Complete | The installed stdio subprocess is initialized, listed, called, recovered after invalid input, and shut down by the official client (**tests/test_mcp_stdio_e2e.py**). |
| H0-02 packaging | Complete | CI builds and verifies wheel/sdist content, installs outside the checkout, and exercises the CLI on Python 3.12/3.13 (**.github/workflows/ci.yml; scripts/verify_distribution.py**). |
| H0-03 canonical inventory | Complete | One registry defines 15 public plus 13 operator tools—28 total—and owns profile/risk/schema metadata (**bddk_mcp/tools/registry.py**). The historical 15/26 acceptance text below is superseded. |
| H0-04 seed/section correctness | Complete for the reviewed seed path | Import regenerates sections and chunks from canonical text, validates hashes, supports resumable full reindex, and rejects tampered derived input (**bddk_mcp/ingest/seed.py; tests/test_seed.py**). Canonical legal versions and corpus generations are later-horizon work. |
| H0-05 quality governance | Partial | A fail-closed quality registry and warning propagation have focused coverage; expert resolution/approval, corpus-scope reconciliation, and immutable correction provenance remain. |
| H0-06 privacy-safe logging | Complete at code boundary | Production tool logs avoid query/result text, privacy-safe correlation is propagated, and sentinel/redaction tests exist. Bank retention/access/export policy remains operational acceptance. |
| H0-07 explicit lifecycle | Complete as a repository boundary | Serving is free of DDL/seed/sync/embedding writes. A checksum v0001-v0003 ledger, catalog attestation, schema-owner/wrong-target verification, exact role assets, TLS enforcement, and clean/legacy/populated-v2 tests exist (**bddk_mcp/migrations/; bddk_mcp/db_lifecycle.py; deploy/postgres/**). The blocking populated-v2 v3 backfill requires explicit maintenance approval after backup and size-matched rehearsal. |
| H0-08 operator lifecycle | Partial | Mutations return receipts and use durable PostgreSQL records, hashed idempotency, recovery, CAS progress, cancellation, and a connection-pinned advisory lease (**bddk_mcp/jobs/postgres.py; tests/test_postgres_job_repository.py**). The task runner is still process-resident; multi-replica failover and ambiguous crash ownership remain unaccepted. |
| H0-09 documentation truth | Partial | Launch, profiles, security boundaries, deployment assumptions, and current tool counts are documented. A named-client/version compatibility matrix and bank deployment evidence remain open. |
| H0-10 licensing/provenance | Open | The owner accepted current source/data use for this job, but no legal decision record or technical entitlement design now prevents unauthorized corporate use. |
| H1-01 and H1-03 input/risk contracts | Complete | Generated models reject extras and expose constraints; all 28 tools have reviewed annotations (**bddk_mcp/tools/registry.py:56-184; tests/test_public_input_contracts.py**). |
| H1-02 structured results | Partial | Stable privacy-safe errors and six structured retrieval evidence outputs exist (**bddk_mcp/mcp_server.py; bddk_mcp/tools/structured_outputs.py**). Uniform typed success/citation/warning/meta envelopes remain open across the other tools. |
| H1-04 remote HTTP baseline | Complete at the application boundary | Non-loopback startup requires exact Host and HTTPS Origin allowlists, complete asymmetric JWT/JWKS verification, and profile scope; request body, rate, and concurrency admission are bounded per process (**bddk_mcp/http_security.py:320-393,437-488,542-698**). Shared ingress controls and bank TLS/IdP acceptance remain deployment work. |
| H1-05 public/operator planes | Complete as a repository boundary | One profile is served per process; scopes, DSNs, role inventories, pool-connection identity checks, and OpenShift workloads differ. Bank-created principals, network policy allows, and principal audit integration remain acceptance work. |
| H1-06 migrations and roles | Complete as a repository boundary | Immutable checksums, advisory serialization, role/grant SQL, wrong-target and TLS guards, write-denial tests, catalog attestation, and prior-shape upgrades exist. Shared-cluster naming and bank DBA execution remain external; future large migrations should use expand/backfill/contract phases. |
| H1-07 protocol E2E | Partial | Official stdio subprocess and JSON Streamable HTTP tests cover initialize/list/call, strict input, health, authentication, Origin, scopes, and operator opt-in (**tests/test_mcp_stdio_e2e.py:25-97; tests/test_mcp_http_runtime.py:18-149**). Full named-client/prior-protocol and durable cancellation matrices remain. |
| H1-08 observability | Partial | Liveness/readiness, privacy-safe correlation, and thread-safe request/error/latency metrics are wired and tested. Standard metric export, traces, retention, SLOs, and recovery evidence remain open. |
| H1-09 OpenShift AI | Partial | The starter supplies separate non-root/read-only public/operator workloads, exact Secrets, TLS probes, lifecycle Jobs, stable selectors, telemetry overlay, digest-only image placeholders, and default-deny ingress/egress. Bank image/CA/IdP/registry/egress values and real-cluster validation remain open (**deploy/openshift/; tests/test_openshift_manifests.py**). |
| H2-02 retrieval publication | Partial | Document/sections replace transactionally and chunks are visible only under a current content/profile publication record; mutation invalidates publication. Immutable whole-corpus generations, active-pointer rollback, and per-request generation binding remain open. |
| H3-01 evaluation runner | Complete as a harness | Phase 2 now uses official stdio or `/mcp`, paginates live discovery, calls tools through `ClientSession`, fails closed, and records audit identities (**benchmark/phase2_e2e.py; tests/test_benchmark_phase2.py**). Expert data, claim/citation grading, named model runs, and recommendations remain open. |
| Supporting security and release controls | Partial | SSRF/redirect/response and ZIP/DOCX archive bounds, immutable model/base references, container recipe checks, and distribution verification are delivered. Prompt-injection elevation, executable image/SBOM scans, load/recovery, and bank supply-chain acceptance remain. |
| H1-10, H1-11, H2 legal/evidence work, H3-02 onward, H4 | Open | Backup/restore, named-client evidence, immutable corpus generations, legal version/currentness, page/formula fidelity, audit-grade citations, expert Turkish evaluation, validated regulatory relations, and audit knowledge workflows remain future work. |

Current repository gates now include official stdio and HTTP E2E, required PostgreSQL CI on Python 3.12/3.13, a dedicated actual-LOGIN/ACL role-contract job, distribution verification, and checksum-pinned mandatory Kustomize rendering of the base and telemetry overlay. This overlay does not claim an OpenShift cluster, recovery, retrieval, citation, or live-model acceptance run.

## Horizon 0 — Stabilize and clarify

| Roadmap ID | Deliverable | Problem Solved | Concrete Tasks | Dependencies | Effort | Priority | Acceptance Criteria |
|---|---|---|---|---|---|---|---|
| H0-01 | Functional server factory and stdio lifecycle | Documented mcp run path imports an empty server. | Add create_server/settings/profile factory and SDK lifespan; remove partial global initialization; make supported imports/launchers construct the same registry; validate transport enum. | None | M | P0 | Official MCP subprocess client initializes through the documented command, lists exactly 15 public tools, calls one local tool, and shuts down; invalid transport exits nonzero; import-time test never reports an empty supported server. |
| H0-02 | Buildable package and portable CLI/config | uv build fails; no console entry point; .mcp.json has a developer path. | Configure build backend/package discovery; add bddk-mcp console script; include required data intentionally; build wheel/sdist; replace absolute config with portable examples for stdio/HTTP. | H0-01 | S | P0 | uv build passes; wheel installs in a clean Python 3.12 and 3.13 environment; bddk-mcp serve --help works; checked-in client config contains no user-specific path. |
| H0-03 | Single runtime-derived tool inventory | README says 16 public, runtime is 15, benchmark is 23; annotations/schemas drift. | Add declarative inventory or runtime export; snapshot public/operator names; generate tool reference and benchmark schema input; record project version. | H0-01 | M | P0 | Tests assert exact intended 15/26 profiles; docs and benchmark fixtures are generated/checked from the same registry; initialization reports project version; an intentional tool change requires one reviewed snapshot update. |
| H0-04 | Seed section-index correctness | Fresh seed loads documents/chunks but not document_sections. | Populate sections during explicit bootstrap/import; normalize aliases; validate section/document hashes; make import idempotent; add exact-reference fixtures. | H0-01 | S | P0 | A fresh disposable bootstrap resolves at least 943 İlke 5 and mevzuat_22599 Madde 9; running bootstrap twice produces identical counts/hashes; every parser-detectable required seed document has sections. |
| H0-05 | Unified known-failure and warning registry | Eleven configured failures appear as zero runtime failures; search/section paths omit warnings. | Make one versioned quality registry; load in runtime/CI/seed manifest; propagate quality status through search, section, and full-document outputs; add resolution metadata. | None | M | P0 | All 11 configured cases classify as failed until explicitly resolved; all three retrieval paths expose identical status/warning; CI fails if registry, docs, and seed manifest disagree. |
| H0-06 | Privacy-safe default tool logging | Audit query and result text enters INFO logs. | Replace argument/result previews with sizes/counts/hashes; classify secret/free-text fields; add explicit development-only preview mode; document retention/access; redact safe errors. | None | S | P0 | Tests call every tool family with sentinel text and find no sentinel in production logs/traces; debug mode is off by default and visibly warned; token/DSN redaction tests pass. |
| H0-07 | Explicit bootstrap and migration commands | Serving startup runs DDL, seed overwrite, and embedding backfill. | Add migrate/bootstrap commands; remove automatic seed import and backfill from serve; refuse uninitialized schema with clear readiness/error; document empty/fresh/upgrade flows. | H0-01, H0-02 | M | P0 | Starting the public server performs no DDL, seed document/chunk writes, or embedding backfill; a fresher DB remains byte/hash unchanged after restart; explicit empty bootstrap is repeatable and observable. |
| H0-08 | Correct operator semantics and job lifecycle | trigger_startup_sync does not sync; jobs can overlap or outlive dependencies. | Fix/rename trigger tool; track every task; add single-flight/advisory locks, idempotency key, status, cancellation/drain; cap concurrency. | H0-03 | M | P1 | Tool behavior matches its name; two conflicting jobs yield one active job; all jobs have ID/status; shutdown drains/cancels before closing DB/HTTP; negative/oversized concurrency is rejected. |
| H0-09 | Documentation truth pass | Startup, tool counts, offline behavior, license, and deployment safety claims are wrong. | Correct README/benchmark/deployment docs; distinguish live versus local-only tools; document local-only Compose; add limitations and compatibility table; link review docs. | H0-02, H0-03 | S | P0 | All commands execute in CI smoke; documented counts are generated; live-source tools are marked; MIT statement matches LICENSE; no unsupported production/client claim remains. |
| H0-10 | License and data-provenance decision record | MIT conflicts with non-commercial objective; source/dataset rights are unclear. | Obtain legal review; inventory contributors and source/dataset provenance; decide future code/data/validated-pack/service terms; document what technical entitlement can and cannot control. | Legal counsel | L | P0 | Owner-approved decision record names license for each artifact class, existing MIT implications, source redistribution basis, contributor ownership, and hosted-access policy; README and release checklist match it. |

## Horizon 1 — Reliable MCP foundation

| Roadmap ID | Deliverable | Problem Solved | Concrete Tasks | Dependencies | Effort | Priority | Acceptance Criteria |
|---|---|---|---|---|---|---|---|
| H1-01 | Strict descriptive input contracts | Runtime schemas lack descriptions, bounds, enums, formats, and extra-field policy. | Create Pydantic inputs with descriptions/limits/date and identifier types; forbid extras; centralize cost/time limits; expose JSON Schema 2020-12. | H0-03 | M | P1 | Every public/operator property has description and applicable constraint; boundary/extra-field tests return stable INVALID_INPUT tool errors; schema linter passes. |
| H1-02 | Structured outputs and error envelope | String-only results/errors prevent reliable automation and citation parsing. | Define status/data/citations/warnings/meta models; retain text fallback; use outputSchema/structuredContent; set isError for execution failures; standardize codes/retryability. | H0-03, H1-01 | L | P1 | Every tool validates against its output schema; invalid/upstream/not-found cases have documented stable behavior; clients can parse citations without text scraping; older text-only client smoke still works. |
| H1-03 | Tool risk annotations | Hosts cannot distinguish reads, writes, idempotent, or external-world calls. | Add readOnlyHint, destructiveHint, idempotentHint, openWorldHint from registry; review each tool classification. | H0-03 | S | P1 | All tools have reviewed annotations; snapshot tests match handler behavior; no mutating operator tool is marked read-only/non-destructive. |
| H1-04 | Secure Streamable HTTP baseline | 0.0.0.0 listener lacks explicit Host/Origin, identity, limits, and safe local defaults. | Default local HTTP to loopback; configure Host/Origin allowlists; implement remote token/OAuth validation; audience/scopes; body/rate/concurrency limits; secure startup validation. | H0-01, H1-01 | L | P0 | Local default is loopback; invalid Host/Origin is 403; missing/invalid/expired/wrong-audience token is 401/403; limits produce stable 4xx/tool errors; remote mode refuses insecure config. |
| H1-05 | Separate public and operator planes | Admin flag is not authorization and uses same process/credentials. | Define distinct registries/entry points; private operator endpoint; bddk.read/monitor/ingest/publish/admin scopes; prevent remote co-hosting; add principal audit events. | H0-03, H1-04 | L | P0 | Public endpoint never lists an operator tool under any tested config; read scope cannot invoke operator tools; operator server refuses remote start without auth/private policy; audit events identify action/result without content leakage. |
| H1-06 | Versioned migrations and DB roles | Serving owns schema and corpus. | Adopt migration ledger/tool; schema-owner, ingestion, publisher, reader, telemetry roles; advisory lock; clean/upgrade tests; remove initialization DDL. | H0-07, H1-05 | L | P0 | Clean install and supported prior-version upgrade pass; serving credentials fail all write/DDL attempts; concurrent startup runs no migration; migration job is locked and version recorded. |
| H1-07 | Official MCP protocol E2E suite | Unit tests missed empty launcher and transport contract defects. | Use official client over stdio and HTTP; initialize/list/call/error/cancel/shutdown; public/operator; current and prior protocol; auth/Origin/schema snapshots; CI gate. | H0-01, H1-02, H1-04, H1-05 | M | P0 | All protocol cases pass in CI on Python 3.12/3.13; deliberate empty registry, bad schema, missing auth, or invalid Origin makes the job fail. |
| H1-08 | Health, metrics, traces, and redaction | Existing metrics remain zero; no readiness route or correlation. | Instrument transport/tool boundary; request IDs; standard metric/trace exporter; liveness/readiness routes; corpus/model/schema metadata; redaction tests. | H0-06, H1-02 | M | P1 | Calls change request/error/latency metrics; trace/request IDs correlate safe logs; liveness excludes dependencies and readiness fails on unavailable active corpus/DB; no query/excerpt in labels/logs/traces. |
| H1-09 | Hardened OpenShift AI deployment profile | Root/mutable image, public DB defaults, and no target-platform health/network/release contract. | Build a non-root OpenShift-compatible image; pin bases/models; add Deployment/Service/Route examples, restrictive security context, NetworkPolicy, Secrets contract, probes, resources, SBOM/scans; keep Docker local; deprecate unsupported Railway/Spaces recipes unless separately tested. | H0-02, H1-06, H1-08 | M | P1 | Image builds/runs/scans in CI as non-root; manifests validate; public and operator workloads use separate identities/roles; health gates deployment; DB is private; model revision is immutable; a bank-like disposable OpenShift test namespace passes deployment smoke. |
| H1-10 | Backup, restore, upgrade, and rollback runbooks | No recovery proof or release strategy. | Define RPO/RTO; PostgreSQL/artifact backup; disposable restore test; schema forward-fix/rollback; corpus generation rollback; compatibility/release checklist. | H1-06, H1-09, H2-02 | M | P1 | Scheduled restore recreates a working server and passes hash/section/vector/citation checks; runbook records measured recovery time; prior corpus can be reactivated without schema rollback. |
| H1-11 | Client compatibility matrix | Claude/Codex/GPT/LM Studio/local claims are unverified. | Define pinned client profiles; test configs; structuredContent/instructions/schema behavior; record limitations; decide whether legacy SSE is required from evidence. | H1-07 | L | P1 | Matrix names client/host/version/transport/protocol and pass/fail cases; Claude and Codex checked-in configs call representative tools; LM Studio/local host profile is reproducible; no blanket unsupported claim. |

## Horizon 2 — High-quality regulatory retrieval

| Roadmap ID | Deliverable | Problem Solved | Concrete Tasks | Dependencies | Effort | Priority | Acceptance Criteria |
|---|---|---|---|---|---|---|---|
| H2-01 | Canonical instrument, artifact, and alias model | IDs/hashes do not represent one regulation across sources/versions; duplicate aliases exist. | Add regulatory_instrument, source_artifact, canonical_alias; capture immutable bytes/reference, MIME, URL, SHA-256, retrieval; reconcile BDDK/Mevzuat/RG/decision IDs and duplicate 909/917. | H1-06, H0-10 | L | P1 | Every published version maps to one instrument and artifact; duplicate/alias decisions are explicit; artifact/hash can reconstruct source where legally permitted; uniqueness/alias tests pass. |
| H2-02 | Immutable corpus generations and atomic publish | Active documents/sections/vectors can diverge or be downgraded. | Stage generation; build all derived indexes; validate hashes/counts/vector/section/quality/citations; atomic active pointer; per-request binding; rollback. | H1-06, H2-01 | L | P0 | Injected failure at every ingestion/index stage leaves active results unchanged; one request sees one generation; stale document/chunk mismatch cannot be served; previous generation reactivates transactionally. |
| H2-03 | Legal version and temporal status model | Extraction history cannot answer current/as-of applicability. | Add instrument_version, publication/effective/expiry/repeal/consolidation/status/validation; import official evidence; distinguish extraction revision; status-unknown guard. | H2-01, H2-02 | L | P1 | Gold fixtures return correct version for current and historical dates; unknown status yields explicit abstention; drafts/repealed texts are excluded by current default unless requested; every status has evidence and validation. |
| H2-04 | Hierarchical provision index | Article/fıkra/bent/annex relationships are lossy and page fields empty. | Build parent/path provision model; parser for major/subordinate/range/table/footnote structures; central alias/reference resolver; source page and normalized offsets; migration/reindex. | H2-01, H2-02 | L | P1 | Stable path round-trips for article/fıkra/bent/annex fixtures; exact lookup accepts canonical aliases; all priority parser-detectable documents meet section coverage; parent/path/page/hash constraints pass. |
| H2-05 | Page/table/formula-preserving extraction | Current normalization can lose audit-critical structure. | Page-aware parser/OCR output; structured tables/formulas/images/footnotes; extractor revision provenance; priority-domain manual comparison; quarantine thresholds. | H2-01, H2-04 | L | P1 | Expert-reviewed priority set reproduces all material formulas, thresholds, table rows, and page references; discrepancies quarantine version; extraction is deterministic for pinned runtime. |
| H2-06 | Versioned citation/evidence engine | Current pseudo-pages and string sources are not audit-grade. | Implement Citation object; artifact/version/provision IDs; official URL; artifact/text hashes; true source page or labeled normalized range; excerpt, quality, generation, timestamp; reconstruct verifier. | H1-02, H2-02, H2-03, H2-04 | L | P1 | 100% release-corpus citation reconstruction; no display window labeled source page; wrong hash/version/page/excerpt fails; every substantive retrieval result has citation and quality state. |
| H2-07 | Source coverage, quality, and correction governance | Coverage/exclusions unknown; warnings/failures and manual patches lack approval provenance. | Authoritative inventory/scope manifest; freshness; quarantine/resolution; duplicate detection; correction proposal/reviewer/approval; seed manifest; formula/table review queue. | H0-05, H2-01, H2-05 | L | P1 | Coverage dashboard reconciles included/excluded/missing sources; every correction has source coordinates, before/after hash, reason, reviewer, approval; no quarantined version appears in default search. |
| H2-08 | Calibrated Turkish hybrid retrieval | Handwritten Turkish rules and document-level fusion are unvalidated; sparse-only evidence can drop. | Exact resolver first; glossary/acronyms/morphology; provision-level lexical+dense RRF; preserve best channel payload; model/revision provenance; benchmark threshold tuning. | H2-04, H3-02 | L | P1 | Expert set meets agreed Recall@5/exact-reference guardrails; rare identifier/sparse-only fixtures survive; priority-domain regressions block merge; all scores record retrieval/model/corpus revision. |
| H2-09 | Evidence-based reranking | Reranker is optional/off and uncalibrated. | Evaluate pinned reranker against latency/quality baseline; add deterministic fallback, resource bounds, and profile version; enable only if material gain. | H2-08, H3-02 | M | P2 | Decision record shows paired metrics and latency; enabling meets approved gain and p95 budget with no exact-reference/currentness regression; failure cleanly falls back and is observable. |
| H2-10 | Regulatory relations baseline | Cross-document amendment/repeal/citation relations are absent. | Add typed relational edges with evidence provision, confidence, reviewer/validation; parse candidates; human approval queue; currentness derivation tests. | H2-03, H2-04, H2-06 | L | P2 | Approved fixtures represent amendment, repeal, replacement, implementation, and citation; every edge has reconstructable evidence and reviewer state; unvalidated candidate never changes currentness. |

## Horizon 3 — Evaluation and model benchmarking

| Roadmap ID | Deliverable | Problem Solved | Concrete Tasks | Dependencies | Effort | Priority | Acceptance Criteria |
|---|---|---|---|---|---|---|---|
| H3-01 | Real MCP benchmark runner | Phase 2 calls nonexistent route and static schemas. | Official client stdio/HTTP; live discovery; fail-closed transport/tool calls; record host/protocol/schema/corpus/model; repeat trials; explicit grader availability. | H1-07, H0-03 | M | P0 | A 404/protocol/tool error fails the case/run; same runner works over stdio and /mcp; report includes exact runtime schema hash and never silently substitutes a grader. |
| H3-02 | Expert-reviewed Turkish retrieval set | Three gold cases/30 NLI pairs do not represent domain. | Create at least 100 initial queries across exact, semantic, acronyms, diacritics, as-of, amendment, formula/table, hard-negative, no-answer, and priority domains; dual annotate/adjudicate; source-hash provenance. | H2-03, H2-04, H2-06 | L | P1 | At least 100 approved cases meet domain/type quotas; every positive/negative maps to immutable versions/provisions; agreement/adjudication recorded; dataset validation passes. |
| H3-03 | Citation and claim-grounding graders | Current grader rewards numeric recall and ignores unsupported additions. | Deterministic citation reconstruction; atomic claim extraction; support/contradiction/unsupported classification; abstention/currentness; calibrated model grader only where needed; human audit sample. | H2-06, H3-02 | L | P1 | Hallucinated additions, wrong versions, missing final-answer citations, and unsupported currentness fail calibrated cases; supported concise answers pass; grader agreement reaches approved threshold; methodology cannot silently change. |
| H3-04 | Tool-calling benchmark | Static selection results do not reflect actual MCP host/model behavior. | Cases for routing, arguments, recovery, pagination, warning handling, unnecessary calls, operator avoidance, injection; evaluate host plus model; report confidence intervals. | H3-01, H3-02 | M | P1 | Each profile completes repeated runs; tool/argument/error metrics are separated; actual discovered schemas used; failures included in denominator; report names host and model. |
| H3-05 | Client/model comparison matrix | No evidence across Claude, Codex, GPT-OSS, LM Studio, or GPT hosts. | Select supported pinned profiles; automate where feasible; record versions/hardware/prompts/quantization; compare protocol, retrieval, grounding, latency, and cost without declaring one universal winner. | H1-11, H3-03, H3-04 | L | P2 | Reproducible reports exist for approved profiles; rerun from manifest stays within tolerance; unsupported profiles are clearly marked; GPT-OSS result names its MCP host/adapter. |
| H3-06 | Retrieval/model regression gates | Algorithm/model/corpus changes can silently alter results. | Baseline metrics by domain/query type; paired comparisons; guardrail thresholds; corpus/schema/model manifests; scheduled full suite and PR smoke subset. | H3-02, H3-03 | M | P1 | PR smoke catches seeded regressions; release suite blocks agreed exact/currentness/citation/domain regressions; report contains confidence intervals and case-level diff. |
| H3-07 | Security, load, resilience, and recovery evaluation | No abuse/load/restore evidence. | Auth/rate/SSRF/archive/injection/redaction; cold/warm latency; mixed load; failure injection; DB/upstream/model outage; backup restore; resource budgets/SLOs. | H1-08, H1-09, H1-10, H2-02 | L | P1 | Approved security negatives pass; p95/p99 and capacity baselines meet chosen SLO; failed ingest never changes active corpus; restore passes complete integrity suite. |
| H3-08 | Evaluation governance and publication | Results lack comparable methodology/provenance. | Version reports/datasets/prompts; annotation policy; public/private split; release approval; limitations; retention; model/provider terms. | H3-01 through H3-07 | M | P1 | Every published result has commit, lock, schema, corpus, model, client, grader, dataset, hardware, trials, skips/failures; owner approves limitations; non-comparable runs are not ranked together. |

## Horizon 4 — Regulatory knowledge platform

| Roadmap ID | Deliverable | Problem Solved | Concrete Tasks | Dependencies | Effort | Priority | Acceptance Criteria |
|---|---|---|---|---|---|---|---|
| H4-01 | Validated regulatory relationship layer | Search cannot reason across amendments, citations, definitions, exceptions, or replacements. | Expand approved typed edges; evidence and reviewer workflow; temporal derivation; relation queries and explanations; quality metrics. | H2-10, H3-03 | XL | P2 | Every production edge has evidence/reviewer/version; multi-hop gold questions return all supporting paths; unvalidated edges are visibly excluded; temporal derivations are reproducible. |
| H4-02 | Obligation extraction and validation | Provisions are text, not structured duties/conditions/exceptions. | Extract actor/action/object/condition/exception/threshold/deadline/entity scope as candidates; reviewer UI/workflow; version and provenance; disagreement handling. | H2-03, H2-04, H2-06, H4-01 | XL | P2 | Approved obligation set reaches domain-owner quality target; every field links to exact evidence; model candidates cannot become published without review; amendments invalidate/review affected obligations. |
| H4-03 | Audit control and evidence mappings | No provision-to-control/procedure/evidence lineage. | Define control objective, control, owner/frequency, audit test, sample/evidence, risk, validation; link multiple provisions/versions; organization-specific overlay; approval/audit trail. | H4-02 | XL | P2 | Reviewed pilot for at least two domains traces each control/test step to current provisions; as-of changes show affected controls; generated suggestions are separated from approved mappings. |
| H4-04 | Regulatory evidence packs | Users cannot produce durable reviewer-ready work products. | Export signed/versioned evidence manifest with query, as-of context, provisions, citations, quality, reasoning/assumptions, control mapping, reviewer actions; reproducible rendering. | H3-03, H4-03 | L | P2 | A second reviewer reconstructs every citation and corpus/model version; tampering changes manifest signature/hash; unknowns/interpretations are explicit; pack passes pilot auditor review. |
| H4-05 | Accumulated validated knowledge workflow | Repeated research does not create governed reusable knowledge. | Candidate/review/approve/retire states; reviewer roles; provenance; conflict/staleness alerts; amendment impact; immutable audit history; feedback into benchmark. | H4-01, H4-02, H4-03 | XL | P2 | Every published knowledge item has owner, source, version, review date/status; affected items enter review after source change; retired items remain auditable; benchmark includes validated corrections. |
| H4-06 | Enterprise entitlement and optional tenancy | Shared private deployments need access/data isolation and commercial controls. | Single-tenant packages first; if required, tenant identity, keys, RLS, corpus/cache/job/log isolation, export/delete, entitlements, audit; contractual integration. | H0-10, H1-04, H1-05, H4-05 | XL | P3 | Independent cross-tenant tests find no data/state leak; tenant deletion/export verified; entitlement revocation works at hosted/private boundary; legal/security approval obtained. |
| H4-07 | Graph projection feasibility study | A graph may help multi-hop reasoning but could add needless operations. | Benchmark typed PostgreSQL recursive queries against real workloads; prototype read-only projection only if needed; measure latency, consistency, operations, and failure modes. | H4-01 | M | P3 | Decision uses at least three real multi-hop workloads; graph is adopted only with material measured benefit and a deterministic PostgreSQL source-of-truth projection; otherwise decision records no adoption. |
| H4-08 | Advanced regulatory reasoning evaluation | Basic retrieval metrics do not prove knowledge workflows. | Gold multi-hop, amendment impact, obligation/control lineage, conflicting-source, uncertainty, and reviewer evidence cases; expert adjudication; safety/abstention. | H4-01 through H4-05 | XL | P2 | Expert-approved suite demonstrates evidence-complete paths, correct temporal reasoning, and abstention; no final score hides relation/control/citation failures; release threshold approved by domain owner. |

## Historical first 10 GitHub issues — completed or superseded

This section preserves the original 2026-07-14 implementation sequence as review evidence. Its runtime, packaging, registry, HTTP, process-separation, lifecycle, seed, privacy, and Phase 2 defects have since been completed or materially superseded. Do not open these as new work; the current residual issue set follows the historical release sequence below.

### Issue 1 — Fix FastMCP construction so the documented stdio server exposes tools

**Description:** The checked-in **mcp run server.py** command imports the global FastMCP object without running project startup. Reproduce the zero-tool state, introduce a server factory/lifespan, and make the supported stdio path construct the same public registry as direct execution. Do not redesign tool outputs in this issue.

**Acceptance criteria:**

- official SDK subprocess client initializes through the documented command;
- tools/list returns the intended 15 public names;
- one deterministic tool call reaches its handler using test dependencies;
- supported import/direct launch contracts match;
- shutdown closes dependencies and stdout remains protocol-only;
- regression test fails against the old behavior.

**Labels:** bug, mcp, P0, tests, effort-M  
**Dependencies:** none  
**Roadmap:** H0-01

### Issue 2 — Make the project buildable and add a portable bddk-mcp CLI

**Description:** Configure package discovery/build metadata and a console script. Replace the hard-coded .mcp.json path with portable Claude/Codex examples using the installed/dev CLI.

**Acceptance criteria:**

- uv build creates wheel and sdist;
- both install in clean Python 3.12 and 3.13 environments;
- bddk-mcp serve --help and version work;
- installed stdio server initializes;
- package contents contain only intended modules/data;
- no checked-in user-specific absolute path.

**Labels:** packaging, developer-experience, P0, effort-S  
**Dependencies:** issue 1  
**Roadmap:** H0-02

### Issue 3 — Generate exact public/operator tool contracts from runtime

**Description:** Establish one inventory/export path for tool names, profile, schema, and annotations. Correct 16/26 and 23-schema drift without changing behavior.

**Acceptance criteria:**

- runtime snapshots assert intended 15 public and 26 operator names;
- generated/checked README and benchmark schema input share the snapshot;
- CI fails on unreviewed contract drift;
- project version is present in initialization metadata;
- no test merely checks literal count text.

**Labels:** mcp, contracts, documentation, P0, effort-M  
**Dependencies:** issue 1  
**Roadmap:** H0-03

### Issue 4 — Enforce loopback, Host, and Origin policy for Streamable HTTP

**Description:** Establish the secure transport floor before full identity work. Local HTTP must bind loopback by default; allowed Host/Origin must be explicit; insecure remote settings must fail fast.

**Acceptance criteria:**

- local default binds 127.0.0.1;
- allowed Host/Origin connects;
- invalid Host/Origin returns 403;
- remote/all-interface mode without allowlists fails startup;
- tests use the official Streamable HTTP client and current protocol;
- behavior is documented.

**Labels:** security, mcp, http, P0, effort-S  
**Dependencies:** issue 1  
**Roadmap:** H1-04, first slice

### Issue 5 — Split public and operator tool profiles at the process boundary

**Description:** Replace the single remote server plus BDDK_ADMIN_TOOLS convention with explicit public and operator entry profiles. This issue establishes registry/process isolation; OAuth scope implementation can follow in a focused issue.

**Acceptance criteria:**

- public profile cannot list any of the 11 operator tools under any flag;
- operator profile is a separate command/config and warns/refuses unsafe public binding;
- each profile can use a separate DSN;
- tests cover exact surfaces and negative cross-profile calls;
- deployment docs identify the operator plane as private.

**Labels:** security, architecture, mcp, P0, effort-M  
**Dependencies:** issue 3, issue 4  
**Roadmap:** H1-05, first slice

### Issue 6 — Remove automatic seed import, DDL, and embedding work from serving startup

**Description:** Add explicit migrate/bootstrap commands and make serve read existing state only. Preserve current functionality through commands; do not add the full generation model yet.

**Acceptance criteria:**

- serve executes no CREATE, ALTER, DROP, seed document/chunk writes, or embedding backfill;
- uninitialized state produces a clear non-ready/startup error;
- explicit migrate/bootstrap works against a disposable DB;
- a DB document with a hash different from bundled seed is unchanged after restart;
- startup time no longer scales with corpus embedding.

**Labels:** correctness, database, deployment, P0, effort-M  
**Dependencies:** issue 2  
**Roadmap:** H0-07

### Issue 7 — Populate document_sections during seed bootstrap and validate section readiness

**Description:** Make a fresh bootstrap support exact article retrieval without a separate undocumented/manual repair step.

**Acceptance criteria:**

- bootstrap populates document_sections transactionally/idempotently;
- 943 İlke 5 and mevzuat_22599 Madde 9 exact lookups pass;
- every parser-detectable required seed document has matching section/document hashes;
- alias form 22599 resolves consistently;
- CI seed-integrity job fails when a required section index is absent.

**Labels:** retrieval, ingestion, bug, P0, effort-S  
**Dependencies:** issue 6  
**Roadmap:** H0-04

### Issue 8 — Connect the known-quality registry to every retrieval surface

**Description:** Make config/quality_failures.yml or its replacement authoritative in runtime, CI, seed metadata, search, sections, and full documents.

**Acceptance criteria:**

- all 11 currently configured failures classify as failed until a reviewed resolution;
- search, exact section, and full document return the same quality status/warnings;
- quality report and seed scorer agree;
- resolution requires reason, reviewer/date, and regression fixture;
- a known-failed result cannot appear without warning.

**Labels:** data-quality, retrieval, compliance, P0, effort-M  
**Dependencies:** none  
**Roadmap:** H0-05

### Issue 9 — Stop logging queries and result excerpts by default

**Description:** Redesign the existing logged_tool metadata to preserve operational value without storing internal-audit text. Keep an explicit local-debug preview option.

**Acceptance criteria:**

- sentinel query/result/credential strings never appear in production logs;
- tool name, status, latency, result count/size, and request ID remain;
- error mapping hides DSN, SQL, tokens, paths, and private content;
- debug text mode is opt-in and documented with retention warning;
- tests cover all tool families and JSON logs.

**Labels:** security, privacy, observability, P0, effort-S  
**Dependencies:** none  
**Roadmap:** H0-06

### Issue 10 — Replace benchmark Phase 2 with official MCP calls and fail-closed scoring

**Description:** Remove the nonexistent /call-tool request. Use official MCP clients over stdio and Streamable HTTP, discover actual schemas, and make transport/tool errors fail the case.

**Acceptance criteria:**

- the same case runs through official stdio and /mcp clients;
- tools/list is live-discovered and report stores schema hash;
- 404, initialize failure, tool error, or malformed result sets transport/tool failure and fails the run;
- no exception is graded as an answer;
- grader absence is an explicit failure/not-comparable state;
- regression test demonstrates old /call-tool path would fail.

**Labels:** evaluation, mcp, bug, P0, effort-M  
**Dependencies:** issue 1, issue 3  
**Roadmap:** H3-01

## Historical issue dependency graph

~~~mermaid
flowchart LR
    I1[1 Server lifecycle] --> I2[2 Package CLI]
    I1 --> I3[3 Runtime contracts]
    I1 --> I4[4 Host Origin]
    I3 --> I5[5 Plane split]
    I4 --> I5
    I2 --> I6[6 Explicit bootstrap]
    I6 --> I7[7 Section bootstrap]
    I1 --> I10[10 Real MCP benchmark]
    I3 --> I10
    I8[8 Quality registry]
    I9[9 Private logging]
~~~

## Historical parallel work

Can begin immediately in parallel:

- issue 1: lifecycle;
- issue 8: quality registry;
- issue 9: private logging.

After issue 1:

- issues 2, 3, and 4 can proceed in parallel.

After issue 2:

- issue 6 can proceed.

After issue 3 and 4:

- issue 5 can proceed.

After issue 6:

- issue 7 can proceed.

After issues 1 and 3:

- issue 10 can proceed independently of database/bootstrap work.

Avoid parallel edits to the current global server/registration code in issues 1, 3, and 5 without sequencing or explicit ownership.

## Historical proposed milestones

### Milestone A — v5.0.1 Runtime Truth

Scope:

- issues 1, 2, 3, 7, and documentation corrections;
- launcher, package, exact tool surface, fresh-section bootstrap.

Exit:

- installed official-client stdio path works;
- package artifacts build;
- fresh local bootstrap supports exact sections;
- docs match runtime.

### Milestone B — v5.1 Secure Foundation

Scope:

- issues 4, 5, 6, 8, 9;
- full remote authentication/scope follow-up;
- strict input and output/error contracts;
- protocol E2E.

Exit:

- public serving has no operator/DDL/seed authority;
- remote transport security/auth tests pass;
- queries are private by default;
- known failures are visible everywhere.

### Milestone C — v5.2 Reliable Corpus

Scope:

- canonical artifacts/instruments;
- DB roles/migrations;
- corpus generations;
- legal versions/status;
- provision hierarchy;
- page/formula preservation;
- citation engine.

Exit:

- active corpus publishes atomically;
- as-of/status either resolves from validated evidence or abstains;
- every release citation reconstructs.

### Milestone D — v5.3 Measured Retrieval

Scope:

- issue 10 and complete evaluation redesign;
- 100-query expert set;
- Turkish hybrid calibration;
- citation/claim graders;
- client/model matrix;
- load/security/recovery gates.

Exit:

- reproducible published baseline;
- no silent grader/transport fallback;
- domain/currentness/citation guardrails pass.

### Milestone E — v6.0 Regulatory Knowledge Pilot

Scope:

- validated relation layer;
- obligation extraction/review;
- two-domain audit control/evidence mapping pilot;
- evidence packs and governance.

Exit:

- domain owners approve the pilot;
- every knowledge/control item traces to immutable current/as-of evidence;
- amendment impact and reviewer audit trail work.

## Historical recommended release sequence

1. **v5.0.1 — Runtime correctness patch**  
   Fix lifecycle, packaging, tool-surface truth, seed section readiness, and documentation. Keep local/pre-production label.

2. **v5.1.0 — Secure MCP foundation**  
   Add structured contracts, secure HTTP, public/operator separation, explicit migrations/bootstrap, privacy-safe logs, protocol E2E, and health/metrics. Permit remote beta only after security gates.

3. **v5.2.0 — Versioned regulatory evidence**  
   Add canonical artifacts, corpus generations, legal versions/status, hierarchical provisions, page/formula preservation, citation engine, DB roles, backup/restore. First candidate for controlled regulatory research, not automatic legal conclusions.

4. **v5.3.0 — Evaluated retrieval and compatibility**  
   Publish expert-reviewed retrieval/citation/grounding baseline and client/model matrix. Enable reranking or model changes only when measured.

5. **v6.0.0 — Regulatory knowledge pilot**  
   Add validated relations, obligations, audit mappings, evidence packs, and governed accumulated knowledge for a limited set of domains.

## First 10 residual GitHub issues — 2026-07-15

These issues begin after the completed stabilization batch. They are deliberately bounded residuals; none repeats the already-delivered launcher, packaging, registry, HTTP, durable-ledger, migration framework, SSRF/archive controls, Phase 2 MCP runner, actual-LOGIN CI contract, or mandatory OpenShift rendering.

| Issue | Suggested title and description | Acceptance criteria | Priority / effort | Suggested labels | Dependencies |
|---|---|---|---|---|---|
| R-01 | **Produce a repeatable populated-v2 migration rehearsal report.** Wrap the existing v3 approval gate in a non-production rehearsal workflow that records scale and operational evidence without relaxing timeouts. | Default populated-v2 migration refuses; an explicitly approved disposable restore completes or fails closed; the report records source DB fingerprint, row/relation sizes, elapsed time, lock waits, database/WAL growth, migration checksum, trigger/constraint state, and reindex/readiness results; secrets and corpus text are absent. | P0 / M | `database`, `migration`, `operations`, `p0` | Delivered actual-LOGIN CI gate |
| R-02 | **Add a disposable backup-and-restore integrity drill.** Define the minimum logical restore test while leaving the bank's PITR implementation external. | A scheduled job backs up an approved test database, restores to a new database, reapplies identity/grants, and passes migration-ledger, catalog, document/section/chunk/publication-hash, retrieval, and MCP-readiness checks; measured recovery time and failures are retained as privacy-safe artifacts. | P0 / M | `database`, `recovery`, `testing`, `p0` | Delivered role and catalog contracts |
| R-03 | **Define and test the supported PostgreSQL version contract.** Resolve the current PG17-only CI evidence before choosing the bank database. | Documentation names each supported major version; CI runs the migration, catalog, identity, role, retrieval-publication, and operator-job contract on every supported version; an unsupported major fails startup/preflight with a safe error; the bank-selected version must be inside the tested set. | P1 / M | `database`, `compatibility`, `testing` | Bank PostgreSQL version decision for final acceptance |
| R-04 | **Create a bank OpenShift acceptance harness and evidence bundle.** Turn the starter manifests into a reproducible namespace-level smoke test without checking bank values into Git. | A parameterized, secret-free fixture verifies signed digest-only images, Route/TLS and JWT claim mapping, exact public/operator scopes and Secrets, PostgreSQL CA/LOGIN separation, required egress, NetworkPolicy denies, probes, migration/bootstrap Jobs, telemetry isolation, and rollback; the run records sanitized versions/results and fails on placeholders. | P1 / L | `openshift`, `security`, `deployment`, `bank-acceptance` | R-02, R-03, R-05; bank IdP/CA/registry/egress inputs |
| R-05 | **Generate SBOMs and enforce signed, scanned release artifacts.** Add a reviewable software-supply-chain lane without introducing a new runtime service. | Wheel/sdist and both container variants produce SBOM/provenance artifacts; pinned scanners have an approved severity policy and exception record; application images are signed by the selected bank-compatible mechanism; OpenShift promotion verifies digest, signature, and policy; a vulnerable fixture proves fail-closed behavior. | P1 / M | `supply-chain`, `security`, `release` | Bank registry/signing-policy decision for final promotion |
| R-06 | **Implement a canonical legal-version pilot for one regulation family.** Separate extraction revisions from legal versions before claiming current or historical applicability. | Stable instrument/version/provision identities represent publication, effective, expiry, repeal/supersession, consolidation, validation state, and source evidence; one amendment chain imports deterministically; current/as-of queries return the validated version or explicitly abstain when status is unknown; migration and regression tests pass. | P1 / L | `regulatory-model`, `versioning`, `retrieval` | R-08 and an authoritative status-validation workflow |
| R-07 | **Define Citation v1 and reconstruct one exact-section path.** Start with `get_document_section`; never label normalized offsets as source pages. | Citation schema identifies instrument/legal version, artifact and text hashes, source URL, provision identity, labeled page or normalized range, quality, and retrieval profile; a verifier reconstructs the exact excerpt; wrong hash/range/version fails; text fallback matches structured evidence. | P1 / M | `citations`, `contracts`, `retrieval` | R-06, R-08 |
| R-08 | **Add a machine-readable corpus-scope and freshness manifest.** Represent the owner's selected job corpus without claiming exhaustive BDDK coverage. | A versioned schema records included/excluded source classes, selection owner/purpose, artifact hashes, retrieval/freshness times, known gaps, and signature/checksum; bootstrap and benchmark audit record its hash; stale, missing, or inconsistent entries fail validation; responses expose a concise scope warning. | P1 / M | `data-governance`, `corpus`, `documentation` | None |
| R-09 | **Create the expert-evaluation schema and a 20-case adjudicated pilot.** Establish governance before scaling to the 100-case release set. | Dataset validation requires immutable source/Citation evidence, query class/domain, positives/hard negatives/no-answer, annotator roles, disagreement/adjudication, and version; 20 owner-approved Turkish cases cover at least five selected domains and include exact, semantic, currentness-unknown, table/formula, and abstention cases. | P1 / M | `evaluation`, `turkish`, `domain-review` | R-06, R-07, R-08 |
| R-10 | **Standardize the untrusted-document envelope and prompt-injection negatives.** Mark retrieved and upstream text as evidence, never instructions, across all six structured retrieval tools. | Structured evidence carries an untrusted-source marker and quality warnings; fixtures containing tool-use, credential, data-exfiltration, and policy-override instructions remain inert data; public scope cannot become operator scope; logs/traces do not copy payloads; behavior is tested over official MCP. | P1 / M | `security`, `retrieval`, `prompt-injection` | None |

## Current issue dependencies and parallel work

~~~mermaid
flowchart LR
    R02[R-02 restore drill] --> R04[R-04 bank OpenShift acceptance]
    R03[R-03 PostgreSQL contract] --> R04
    R05[R-05 signed supply chain] --> R04
    R08[R-08 scope manifest] --> R06[R-06 legal-version pilot]
    R06 --> R07[R-07 Citation v1]
    R08 --> R07
    R06 --> R09[R-09 expert pilot]
    R07 --> R09
    R08 --> R09
    R01[R-01 v3 rehearsal]
    R10[R-10 injection envelope]
~~~

R-01, R-02, R-03, R-05, R-08, and R-10 can start in parallel with separate file ownership. R-04 can build its secret-free harness while those run, but bank acceptance cannot close before R-02, R-03, and R-05. R-06 follows the scope contract and authoritative legal-status workflow; R-07 follows that pilot; R-09 follows the legal-version, Citation, and scope contracts. Avoid combining R-01 with a migration rewrite: its purpose is to measure and gate the current v3 path.

## Current milestone structure

| Milestone | Included issues | Exit criteria |
|---|---|---|
| **v5.0.1 Repository Evidence** | R-01, R-02 | A populated-v2 rehearsal and disposable restore prove that the delivered migration, catalog, role, and publication controls survive realistic recovery. This is repository evidence, not bank acceptance. |
| **v5.1 Bank Integration Candidate** | R-03, R-04, R-05, R-10 | Supported PostgreSQL versions are explicit, artifacts are signed/scanned, untrusted evidence is tested, and the bank acceptance harness passes with controlled values. |
| **v5.2 Regulatory Evidence Pilot** | R-06, R-07, R-08, R-09 | One legal-version chain and Citation v1 reconstruct exact evidence, scope/freshness is machine-readable, and the adjudicated pilot dataset validates. Expand to the 100-case release set only after pilot review. |
| **v6.0 Knowledge Platform Pilot** | Immutable corpus generations plus remaining Horizon 2/Horizon 4 items | Atomic corpus rollback, broader legal applicability/currentness, audit-grade citations, expert release evaluation, and validated knowledge workflows pass before this milestone is scheduled. |

## Current recommended release sequence

1. **v5.0.1 — hardened repository candidate.** Keep the delivered MCP/security/database/publication foundations and close R-01 and R-02. Do not call it bank-production-ready.
2. **v5.1 — controlled bank integration candidate.** Close R-03 through R-05 and R-10, then require the bank's IdP, CA, signed-image, egress, monitoring, backup/PITR, and namespace evidence. A passing repository suite alone is insufficient.
3. **v5.2 — regulatory evidence pilot.** Close R-06 through R-09, scale the expert set, and block legal-currentness or audit-grade claims until version/page/citation acceptance passes.
4. **v6.0 — regulatory knowledge pilot.** Introduce immutable corpus generations, legal versions/relations, reviewer workflows, and control mappings incrementally; do not infer completion from document-search quality.

## Decision points

Owner decisions recorded 2026-07-14:

- target platform: bank on-premises OpenShift AI;
- corpus: intentional job-specific selection rather than exhaustive BDDK coverage;
- database administration and regulatory/extraction review owner: project owner;
- source/data rights: accepted by project owner, pending a written provenance record;
- interim tenancy: single-tenant and no private-corpus feature until requirements are known;
- service expectation: “immediate,” pending numeric SLO, freshness, RPO, and RTO definitions;
- current external endpoint protection: none confirmed.

Before v5.1:

- exact bank PostgreSQL major/topology and OpenShift ingress/Route, IdP, CA,
  registry, egress, NetworkPolicy, SCC, monitoring, and namespace values;
- bank signing/SBOM/vulnerability policy and the accepted backup/PITR mechanism;
- future code/data licensing position;
- supported MCP host, client, and model versions.

Before v5.2:

- written rights/provenance and storage policy for original source artifacts;
- effective-date/legal-status authority;
- numeric availability, freshness, RPO, and RTO targets.

Before v6.0:

- first two audit domains;
- approval workflow and liability boundaries;
- whether organization-private control libraries enter scope;
- whether real workloads justify a graph projection or multi-tenancy.

## What not to implement yet

- a separate graph database;
- distributed vector infrastructure;
- multi-tenant private corpora;
- autonomous legal/currentness conclusions;
- automatic publication of LLM-extracted obligations/relations;
- legacy SSE solely for theoretical compatibility;
- a general SQL tool;
- license phone-home checks in local MIT code.

These add operational or trust burden before foundational correctness is solved.
