# Testing and Evaluation Strategy

> [!NOTE]
> Executed counts and implementation overlays in this document are dated evidence, not automatically current results. Current repository facts are in [Current Repository Status](STATUS.md); current test truth comes from the checks attached to the commit under review. Known-stale figures below include the tool counts (15/29 at the overlay date; currently 17/31) and the manifest state (the corpus has been signed and regenerated since).

## Purpose

The test strategy must answer four different questions:

1. **Does the software behave correctly?**
2. **Does the MCP host/model select and call the right tools correctly?**
3. **Does retrieval return the legally applicable, traceable evidence?**
4. **Does the final answer make only claims supported by that evidence?**

The current repository now has reliable protocol and real-MCP runner coverage, a reconstructable normalized-range Citation v1 pilot, and a verifier for separately retained source/acquisition/page/excerpt evidence. It still does not answer questions 2–4 at a product-evidence level because named host/model runs, independently approved retrieval judgments, authoritative legal-currentness fixtures, real non-fixture legal evidence, reproducible source-to-page derivation, and claim-level grounding remain incomplete.

## Current baseline

Review execution against commit **5684a34c10e6d90bc22d6ab2a6466944afb6bf81**:

| Check | Result |
|---|---|
| Test discovery | 613 total; 610 selected and 3 GPU-marked deselected |
| Default suite | 526 passed, 84 skipped, 3 deselected |
| Ruff lint | Passed |
| Ruff format check | Passed; 138 files formatted |
| Runtime tools | 17 public, 31 operator-enabled |
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

### Implementation progress overlay — 2026-07-16

This checkpoint describes the current working tree; the executed-check table above remains the reviewed-commit baseline. **Complete** means a repository gate or focused automated contract exists, not that every target environment has passed it. **Partial** means meaningful coverage exists with important gaps. **Open** means the evaluation outcome remains unproved.

| Test/evaluation slice | Status | Current evidence and remaining gap |
|---|---|---|
| Installed MCP transport E2E | Complete | The official client exercises the installed stdio subprocess through initialize/version/list/call/invalid-extra/recovery/shutdown and checks protocol-only stdout. Streamable HTTP tests cover initialize/list/call, the single `bddk://corpus/active-release` resource, health, Host/Origin, JWT/JWKS, scope, operator opt-in, RFC 9728 protected-resource metadata, the matching `resource_metadata` 401 challenge, and shutdown. No MCP prompts are registered (**bddk_mcp/resources.py; tests/test_mcp_stdio_e2e.py; tests/test_mcp_http_runtime.py**). Bank IdP registration and authorization-flow acceptance remain external. |
| HTTP slow-body admission | Complete at application boundary | Independent first-byte, inter-chunk, and total body-read deadlines are validated, return 408, and release the process admission slot. The focused HTTP-security lane passed 61 tests (**bddk_mcp/http_security.py; tests/test_http_security.py**). Bank ingress/proxy timeout alignment, shared quotas, and slow-client load evidence remain external. |
| Tool contracts and protocol errors | Partial | The 17-public/31-operator registry (17 public plus 14 operator additions) owns strict generated arguments and risk annotations; stable privacy-safe errors are tested. Six retrieval tools validate structured evidence payloads (**tests/test_tool_registry.py; tests/test_public_input_contracts.py; tests/test_structured_retrieval_outputs.py**). The remaining tools do not yet share one structured result contract. |
| PostgreSQL compatibility and lifecycle | Partial | PostgreSQL 17 is the explicit repository contract. Migrations now run through v0008: v4's legal subset still attests exactly 69 constraints/21 indexes, v5 adds release/activation/epoch state, v6 adds the resolver, v7 adds retained generations, and v8 adds staged release requests and one-time activation bindings. The v8 migration/catalog lane passed **59 tests**; application/role coverage passed **118 tests**, and the separate actual-LOGIN lane passed **2 tests**. After correcting the old v2-downgrade fixture, the clean branch-wide v8 PostgreSQL rerun passed **185 tests** with **4 capability-gated skips** and no failures. Prior frozen H2-02A results remain historical (**bddk_mcp/migrations/v0008_staged_corpus_releases.py; bddk_mcp/catalog_integrity.py; bddk_mcp/db_identity.py; tests/test_migrations.py; tests/test_postgres_role_assets.py**). Bank LOGINs, production-size upgrade, failover, and DBA evidence remain external. |
| Staged publication authorization | Repository mechanism covered; bank custody open | Tests cover verifier-only staging, publisher request-ID-only activation, old-direct-routine denial, request expiry/replay, state/epoch/profile drift, readiness, concurrency, atomicity, cross-role denial, exact catalog/ACLs, and actual LOGINs. This remediates the publisher-only bypass in repository code (**bddk_mcp/migrations/v0008_staged_corpus_releases.py; bddk_mcp/corpus_publication.py; tests/test_migrations.py; tests/test_corpus_publication.py; tests/test_db_identity.py**). The bank must still prove that no principal, ServiceAccount, or Secret reader can obtain both credentials and that schema-owner use is separately controlled. |
| Retained-generation contracts | Repository mechanism covered; external acceptance partial | Focused unit/PG17 contracts exercise exact 17-relation typed retention, exact-v5 state-hash reproduction, and retained row/current-state/retained-state hash invariance under changed session `TimeZone`, `DateStyle`, `IntervalStyle`, `bytea_output`, and `extra_float_digits`. They prove two different-state/profile generations coexist, release-idempotent retry, and a differently governed release over the same exact state/profile shares the original physical generation and seal through a distinct binding. They also cover `legacy_v5_unretained`, mutation and truncate refusal, catalog tamper detection, role denial, sanitized errors, storage arithmetic, and absence from both MCP registries (**tests/test_migrations.py:104-126; tests/test_corpus_generations.py; tests/test_corpus_publication.py:580-850; tests/test_catalog_integrity.py; tests/test_cli.py:198-531; tests/test_postgres_role_assets.py**). Parameterized PostgreSQL injection fails each of the 17 member copies plus inventory, seal, and release binding and proves atomic rollback without changing the active release; a fresh current-schema PG17 execution passed all 20 cases on 2026-07-16. CLI tests enforce transaction-local `lock_timeout=30s` and `statement_timeout=30min`; WAL-baseline savepoint or post-commit observation failures degrade to `not_measured` without reversing durable success. Any WAL value remains only an observed non-exclusive cluster interval. Backup growth remains `not_measured`, and no bank capacity/retention approval exists. |
| Recovery workflows | Current local execution complete; bank acceptance open | Tests exercise guarded migration, default refusal, actual-content fingerprints, bounded subprocess cleanup, and recovery-evidence schema v2. Recovery inventories 53 managed objects, including both v8 request/binding relations, rejects seven application DSNs, verifies seven restored LOGIN profiles, rejects orphan retained rows, and binds PostgreSQL 17 locale/collation identities source-to-restore (**bddk_mcp/operations/recovery.py; tests/test_recovery_workflows.py**). A retained synthetic two-cluster PG17 execution proves exact **schema-v8, 53-object, seven-identity** equality with two request/bindings, two retained generations, the same logical fingerprint and active release (**docs/evidence/LOCAL_PG17_V8_RECOVERY_DRILL.md; docs/evidence/local-pg17-v8-restore-2026-07-16.json**). Bank-size backup/PITR, custody, controlled backup growth, approved RPO/RTO, TLS/HBA/LOGIN execution, and bank recovery approval remain unproved. |
| Citation v1 and legal-version pilot | Partial technical evidence | Citation tests cover canonical identity, separate `SourceBlob` content and `SourceArtifact` acquisition identities, frozen-whitespace exact normalized ranges, Unicode/CRLF/astral round trips, excerpt reconstruction, mismatch refusal, and omission for unvalidated/truncated/failed-quality cases. PostgreSQL exposes only validated authoritative non-fixture mappings whose hashes agree. The legal-release verifier additionally re-hashes retained source bytes, acquisition records, page mapping/text, exact excerpts, and every predecessor's retained files. `PageMappingProof` v2 binds each checkpoint/artifact review to an opaque owner in the signed policy's reviewer registry (**bddk_mcp/citations.py; benchmark/legal_release_evidence.py; benchmark/evaluation_trust_policy.py; tests/test_citations.py; tests/test_expert_evaluation.py; tests/test_legal_versions.py**). The only end-to-end family remains synthetic; policy binding does not authenticate the human review action or independently prove source/PDF-to-page-text derivation, and historical Citation packs are not retained/replayed for predecessor checkpoints. |
| Untrusted-document rendering | Complete at current code boundary; live-model evaluation open | Tests cover the escaped untrusted-data envelope and delimiter spoofing across all six retrieval tools and the other source-backed public renderers; official MCP output checks keep malicious metadata and body text inside the data boundary (**bddk_mcp/tools/structured_outputs.py; tests/test_structured_retrieval_outputs.py**). No live host/model prompt-injection or tool-escalation benchmark has run. |
| OpenShift repository preflight | Partial | The acceptance suite requires checksum-pinned Kustomize v5.8.1, performs a bounded offline render, and rejects drift in exact resources, namespace, selectors/labels, NetworkPolicies, Secret/ConfigMap keys, commands, ports, volumes, and restricted security contexts. It now requires four lifecycle Jobs—migrate, bootstrap, verify/stage, activate—and distinct verifier/publisher ServiceAccounts, Secrets and DSNs; the activator has no corpus PVC/trust key. The integrated acceptance/manifest run passed **75 tests** (**deploy/openshift/jobs/; deploy/openshift/serviceaccounts.yaml; bddk_mcp/openshift_acceptance.py; tests/test_openshift_acceptance.py; tests/test_openshift_manifests.py**). It still records eight live external gates as `not_run` and cannot prove bank RBAC Secret-reader separation, CNI, IdP, CA, registry, database, backup, or client/model behavior. |
| Supply-chain lane | Partial | Focused tests cover pinned tool checksums, reproducible distributions, Buildx `--provenance=false --load` descriptor/manifest/config/loaded-image/Syft binding, deterministic SBOM and unsigned repository SLSA, model-manifest/runtime/Dockerfile consistency, complete-history secret policy, vulnerability-database freshness, material-bound exceptions, and separate integrity/release decisions. The always-run `evidence-integrity` contract records High/Critical findings but fails on stale/malformed/suppressive evidence, unexcepted secrets, or invalid exception use. The tag/explicit-manual `release-eligibility` contract revalidates the exact downloaded evidence manifest and fails on unexcepted High/Critical findings or pending external approval (**tests/test_supply_chain.py; .github/workflows/supply-chain.yml; scripts/supply_chain_evidence.py; supply-chain/**). Focused local tests do not by themselves prove the complete hosted linux/amd64 run; the check attached to each commit is its execution evidence. Neither a green repository job nor its unsigned evidence is a signing, admission, or promotion test. |
| Corpus and expert dataset integrity | Partial, deliberately non-release | Strict import and the distinct release verifier check manifest-role bytes and policy, compare the complete regenerated chunk inventory, and stage manifest/retrieval-profile/corpus-state identity; the publisher activates only the resulting request ID, while a mutation epoch invalidates stale activation (**bddk_mcp/ingest/seed.py; bddk_mcp/corpus_publication.py; tests/test_seed.py; tests/test_corpus_publication.py**). The tracked 318-document manifest remains non-exhaustive, unsigned, unquantified, and unmeasured; its declared 8,286 chunks differ from the 9,675 produced by current-profile regeneration, so strict verification/staging refuses it. The 20-case Turkish draft still has pending Citations, `legal_currentness: not_verified`, 40 annotations, 20 adjudications, and approvals. |
| Evaluation release preflight | Partial, deliberately non-release | Release validation requires four separate signed layers—measured corpus, expert dataset, Citation pack/legal-curator attestation, and a legal-release checkpoint over retained evidence/history—and rejects signer reuse across separated roles. Policy-free development retains operator-supplied keys/latest head and can validate v1 page proofs. Any signed policy requires v2 proofs. Bank-policy mode verifies a separately signed schema-v2 policy, five release identities with documented canonical/raw semantics, four roles with distinct declared owner IDs, a separate reviewer registry, validity/effective revocations, forward legal-release rotation, reviewer/artifact/checkpoint/chronology binding, and separately supplied current policy SHA/version and organization/environment/scope pins (**benchmark/evaluation_trust_policy.py; benchmark/legal_release_evidence.py; benchmark/release_preflight.py; benchmark/README.md: Hash and version semantics; tests/test_evaluation_trust_policy.py; tests/test_expert_evaluation.py; tests/test_release_preflight.py**). Declared event windows are not trusted signature timestamps. Both modes keep bank authorization and model-score authorization false. Bank RBAC custody/promotion, actual human/team separation, stale-pin authority, human reviewer action/authentication, historical pack replay, and model execution remain open. |
| Real MCP Phase 2 runner | Complete as a harness | Phase 2 uses official `ClientSession` transports for stdio and `/mcp`, paginates live discovery, reads `bddk://corpus/active-release`, requires its manifest ID/SHA to match the validated local manifest, executes actual `call_tool` on that same session, and rejects a release change on the final same-session read. It sanitizes audit artifacts and records schema/server/protocol/manifest/active-release/dataset identities (**benchmark/phase2_e2e.py; benchmark/audit.py; tests/test_benchmark_phase2.py**). No named model/client score or product recommendation follows from harness correctness. |
| Ordinary benchmark reports | Exploratory only | `benchmark.run` always marks results `exploratory_not_release_evidence` and `model_scores_authorized: false`; console and diagnosis reports refuse deployment advice even if a result JSON is edited. These runners do not execute the expert dataset or invoke the release preflight (**benchmark/run.py; benchmark/report.py; tests/test_benchmark_audit.py**). |
| Observability, load, and client/model operations | Open/Partial | Correlation IDs, privacy-safe request/error/latency metrics, readiness, and isolated telemetry have tests. Standard export/tracing, numeric SLOs, retention, load/resilience, full recovery, and a named Claude/Codex/GPT-OSS/LM Studio matrix remain unproved. |

The pre-v7 local checkpoint on 2026-07-16 passed **1,355** non-GPU,
non-PostgreSQL tests with 37 capability-gated skips and 147 deselections in
52.70 seconds. The PostgreSQL and role-contract results are reported separately
above so skipped capabilities are not hidden inside one inflated aggregate.
`uv lock --check`, Ruff lint and format checks, distribution build/content
verification, isolated wheel import/resource/CLI checks on Python 3.12.13 and
3.13.13, and `git diff --check` also passed. Current testing/evaluation
maturity remains **3/5** because these repository results do not supply expert,
live-model, load, bank-platform, or PITR acceptance; the other calibrated
ratings remain overall **3/5**, production readiness **2/5**, MCP **4/5**,
retrieval **3/5**, security **3/5**, and documentation **4/5**.

The frozen H2-02A branch was then revalidated separately on 2026-07-16:

- **1,411 passed, 34 skipped, 184 deselected** in the complete non-PostgreSQL/non-GPU lane;
- **177 passed, 4 skipped, 1,448 deselected** in the complete required PostgreSQL 17 lane;
- **2 passed** in the separately provisioned PostgreSQL 17 role/actual-LOGIN and ACL-provenance lane;
- **31 passed** in the full migration module, including the real v5 publisher-LOGIN canonical-remediation path and deterministic migration/publication lock race; and
- Ruff lint/format, `git diff --check`, locked dependency verification, and distribution build were rerun on the frozen tree.

These are disposable repository results, not bank backup, capacity, RPO/RTO,
OpenShift, IdP, or approval evidence. GPU/OCR remained capability-gated.

The later schema-v8 integration checkpoint is reported independently rather
than merged into that historical H2-02A aggregate:

- 59 focused PostgreSQL migration/catalog tests passed;
- 118 focused application/role tests and two actual-LOGIN tests passed;
- 75 integrated OpenShift acceptance/manifest tests passed;
- 61 focused HTTP-security tests passed; and
- one local two-cluster PG17 recovery execution passed for schema v8, all 53
  managed objects, seven identities, two request/bindings and two retained
  generations, with exact logical-fingerprint and active-release equality; and
- the clean full non-PostgreSQL v8 lane passed 1,440 tests with 37 skips and no
  failures; and
- after the old v2-downgrade fixture was corrected, the clean full PostgreSQL
  v8 lane passed 185 tests with four capability-gated skips and no failures.

The pre-v7 and frozen H2-02A aggregates above remain historical checkpoints;
the schema-v8 bullets are the current branch-wide repository validation. No
maturity rating is increased merely because the retention or staged-release
mechanisms added tests.

The optional GPU/OCR lane was also probed. Before the lane contract was fixed,
CUDA detection alone started three integration cases in a base development
environment: Chandra was absent and LightOCR could not load its remote-default
model, so all three failed. This was a provisioning-boundary failure, not a
validated OCR regression. The lane now skips only when
`BDDK_REQUIRE_GPU_OCR` is absent; when set to `1`, it fails closed unless CUDA,
the `gpu` dependency group, offline mode, and local LightOCR and Chandra model
directories are present. Both OCR tests use the retained PDF fixture and make
no live source request.

Required invocation shape:

```bash
uv sync --frozen --dev --group gpu
BDDK_REQUIRE_GPU_OCR=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
BDDK_LIGHTOCR_MODEL_PATH=/APPROVED/MODELS/LIGHTOCR \
BDDK_CHANDRA_MODEL=/APPROVED/MODELS/CHANDRA \
  uv run --frozen --group gpu pytest -m gpu -v
```

The current machine does not have those two approved local model directories
or the Chandra optional package, so no OCR-quality pass is claimed.

## Benchmark defects identified at the reviewed commit and current status

### Phase 1 is function calling, not MCP

**benchmark/phase1_tools.py:36-58** sends static OpenAI-style function schemas to an OpenAI-compatible Chat Completions API. This can be a useful model tool-selection test, but it is not MCP discovery, transport, or host compatibility. It must be labeled accordingly.

### Static schemas drifted from runtime at the reviewed commit

At the reviewed commit, the benchmark defined 23 static schemas while runtime exposed 15 public or 26 operator tools. The working tree now exports all 29 OpenAI-style function schemas from the canonical 15-public/14-operator registry, eliminating that inventory drift. Benchmark Phase 1 remains function calling rather than MCP. Phase 2 discovers the live MCP schema and therefore no longer shares this defect.

### Phase 2 did not call this MCP server at the reviewed commit; this is corrected

The reviewed-commit implementation POSTed to nonexistent **/call-tool**, described stdio without launching it, and converted HTTP failures into result strings. Current **benchmark/phase2_e2e.py** opens an official SDK session over stdio or Streamable HTTP **/mcp**, paginates `tools/list`, invokes `ClientSession.call_tool`, and raises on transport, protocol, malformed-model, or tool errors. **tests/test_benchmark_phase2.py** verifies those fail-closed paths and checks that the obsolete route is absent.

### Grounding/citation scores are not claim validation

The source-trace score examines tool results, not whether final-answer claims cite and follow the evidence. The code grader checks required numbers/dates and does not penalize unsupported additions (**benchmark/graders.py:20-66; benchmark/scoring.py:191 onward**).

### Silent grader fallback at the reviewed commit is corrected

At the reviewed commit, an unavailable Anthropic key/API silently substituted the weak code score (**historical benchmark/graders.py:91-126**). Current Phase 2 instead records an explicit unavailable/not-comparable model-grader state while preserving separately labelled deterministic retrieval and numeric-claim metrics. Ordinary human and JSON reports remain exploratory regardless of threshold results (**benchmark/graders.py; benchmark/phase2_e2e.py; benchmark/report.py**).

### Data is too small and weakly sourced

- 3 source-grounded gold cases in **benchmark/gold_cases.yml**
- 30 NLI pairs against a stated target of 500 in **data/bddk_nli/metadata.json**
- generic source labels rather than immutable document/version/provision/hash citations
- unclear independent annotation and adjudication

No client/model recommendation should be based on repository presence or harness tests alone. A recommendation requires a recorded successful run on the approved corpus, expert-reviewed cases, pinned host/model/hardware, and the claim/citation methodology below.

### What the current release preflight does and does not prove

`python -m benchmark.release_preflight` is source-checkout-only: package
discovery and the runtime image include `bddk_mcp`, not `benchmark`
(**pyproject.toml:78-82; Dockerfile:9-16**). It validates the complete current
four-layer trust chain and emits content-free aggregate identities. Canonical raw
Ed25519 fingerprints prevent one signer from appearing independent merely by
changing PEM encoding.

The default `development` mode uses operator-supplied operational keys and a
manual latest-checkpoint SHA-256. It forbids current policy and deployment-scope
pins. If signed-policy inputs are supplied for fixtures, development mode still
does not pin the current policy or compare organization/environment/scope with
independent expectations. This mode is appropriate for cryptographic
consistency checks, not bank authority.

The `bank-policy` mode requires the policy bytes, detached Ed25519 signature,
trusted policy-root key, and exact separately supplied current policy
SHA-256/version plus organization/environment/deployment scope. A validly signed
but different or cross-environment policy is rejected.
The policy supplies the approved latest checkpoint, so a simultaneous manual
latest-head argument is forbidden. It binds five release identities—the
dataset, corpus manifest, legal pack, legal attestation, and legal-release
checkpoint—and authorizes four roles with distinct declared owner IDs:
`corpus_scope_approver`, `expert_dataset_owner`, `legal_curator`, and
`legal_release_certifier`. The root authority cannot be an operational owner or
signer by declared ID/key. Distinct owner strings do not prove different
humans, teams, or custodians; bank governance must establish real separation.
The report exposes bounded policy identity/count evidence, not owner IDs or
labels. Canonical-versus-raw hash and version meanings are defined in the
[benchmark trust contract](../benchmark/README.md#hash-and-version-semantics).

Policy schema v2 also authorizes a separate, canonical reviewer registry.
Every retained artifact in every checkpoint must use `PageMappingProof` v2 and
seal one reviewer owner ID. Tests require exact checkpoint/artifact coverage,
unique reviews, capture ≤ review ≤ checkpoint ≤ validation chronology,
reviewer validity, effective revocation, and reviewer separation from the
issuer and four operational signer owners. Policy-free development can still
validate v1 proofs; any signed-policy authorization rejects them.

Legal-release history can use one primary current key plus explicit predecessor
keys. The latest checkpoint must use the primary key. Policy entries use
`replaces_key_id`; tests reject disconnected/cyclic rotation, reversal to an old
key, use outside the declared event-time window, duplicate canonical signers,
effective key revocation, and effective checkpoint revocation. A retired but
non-revoked key remains valid
only for checkpoint time within its declared authorization window. Policy
validity/approval, corpus review, dataset decision, curator attestation,
checkpoint creation, and page review times are all declared signed fields evaluated against the local process
clock, not independent signature timestamps. Within schema v2, `policy_version`
values after 1 must name the superseded exact policy-byte SHA-256, but the
source-checkout verifier checks only that declaration's shape and does not
retrieve or replay predecessor policy bytes/history. The loader accepts only
policy schema v2, so a signed schema-v1 policy must be reissued.

Both modes deliberately report `bank_authorization_verified: false` and
`model_scores_authorized: false`. Even bank-policy mode cannot prove that the
policy root, policy/key mounts, and current SHA/version/scope pins came from bank
RBAC or an approved promotion. If the externally configured pin is itself stale, the
offline source-checkout verifier has no independent current-policy service from
which to discover that fact. This repository slice therefore does not close the
bank governance issue.

The bank-policy success label is intentionally
`configured_policy_head_preflight_passed`, not a bank-authorization label. Its
four relevant booleans are `configured_root_policy_signature_verified`,
`policy_approved_release_binding_verified`, and
`policy_current_head_pin_verified`; deployment-scope comparison is recorded by
`policy_deployment_scope_pin_verified`. Aggregate reviewer counts/verification
are emitted, but reviewer IDs and labels are not. The reviewer count is the
number of distinct observed reviewer owner IDs, not registry entries; the review
count is the number of checkpoint/artifact pairs. The input provenance remains
`caller_or_deployment_supplied`.

The checkpoint verifier re-hashes source, acquisition, page mapping/text, and
excerpt files for the full predecessor chain. Only the current checkpoint's
legal pack is loaded and compared object-for-object with the dataset; historical
pack bytes/Citation inventories are not retained and replayed. Exact excerpt
containment in retained mapped-page text is verified. V1 supplies only a role
assertion. V2 binds the checkpoint/artifact review to a policy-authorized owner,
but does not authenticate the human action, provide a reviewer signature, or
reproduce raw source/PDF-to-page-text derivation. Because checkpoint bytes are
immutable, a chain containing v1 page proofs needs a new independent genesis in
which every artifact contains v2; that genesis cannot reference the v1-proof
ancestor, so the old chain remains archival without verified continuity.

Most importantly, preflight does not execute the expert cases. The ordinary
Phase 1/2/3 datasets and scores are separate and always exploratory. All tracked
expert evidence remains `legal_currentness: not_verified`; currentness,
version-comparison, and amendment-tracking cases are forced to abstain, and the
preflight lists their score authorization as unsupported. A cryptographic pass
must never be joined to an ordinary benchmark report to manufacture a model
release claim.

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

The current repository covers clean and idempotent migrations through v0008, strict legacy adoption, populated-v2 refusal/approved backfill to the current schema, transactional rollback injection, catalog attestation, durable job concurrency/leases, fail-closed staged retrieval publication, owner-controlled legal tables with owner-only mutation and an exact verifier read-only exception, active-release/epoch state, typed immutable retention of 17 corpus relations, verifier/publisher role separation, and identity/write-denial contracts. The guarded recovery contract inventories 53 managed objects and seven identities, and a retained local two-cluster execution proves that schema-v8 path. The list below is the full target; bank-like target-size/TLS/HBA recovery evidence, target-bank identities, low-downtime large-corpus migration, generation-bound serving/authorized rollback, and PITR remain residual work.

Test:

- clean migrations;
- upgrade from every supported prior release;
- migration advisory lock and concurrent startup;
- extension setup under migrator role;
- public serving under read-only role;
- ingestion and publisher role grants/denials;
- statement timeouts and cancellation;
- document/version/provision constraints;
- corpus staging, validation, atomic activation, typed retention, and separately
  authorized rollback/reactivation;
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

- exact 17 public and 31 operator names (17 plus 14) until intentionally versioned;
- exactly one `bddk://corpus/active-release` resource and no MCP prompts;
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

The tracked bundle currently fails this gate before model benchmarking: its
manifest declares 8,286 chunk rows while current-profile regeneration produced
9,675. The strict release verifier compares the full canonical rows, not only their
count, and requires regeneration, independent review, updated hashes/counts,
and a new signature (**seed_data/corpus_scope.yml:31-40;
bddk_mcp/ingest/seed.py:335-410,1262-1272**).

The reviewed-commit quality score of 99.5692 must not be used as an overall corpus-accuracy score. It checked implemented text anomalies only and conflicted with 11 configured failures. The working tree now applies those 11 cases through one packaged fail-closed registry, removing that direct inconsistency; the resulting score still measures only implemented extraction-quality rules, not legal accuracy, completeness, currentness, or citation correctness.

## Reproducibility contract

Every benchmark report must include:

- git commit and dirty-state flag;
- Python/package lock hash;
- MCP SDK/protocol/client versions;
- public/operator registry hash;
- corpus generation and artifact manifest hash;
- the active-release ID, manifest identity, and retrieval-profile hash read
  before and after Phase 2 on the same MCP session;
- embedding and reranker names/revisions;
- database/pgvector version and retrieval settings;
- model/host/backend/quantization;
- prompts and grader revision;
- dataset revision and annotation policy;
- hardware and timing environment;
- seeds/trials/temperature;
- skipped/failed cases and exact reason.

Every ordinary report must also retain its
`exploratory_not_release_evidence`/`model_scores_authorized: false`
classification. A separate preflight report may prove an operator-supplied
development chain or a configured-policy-bound chain, but until an expert-dataset
execution format and external bank authorization evidence exist it cannot
authorize or sign ordinary model scores.

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

The pull-request implementation already gates lint/format, unit tests, required PostgreSQL tests on Python 3.12/3.13, a dedicated actual-LOGIN/ACL role contract, official-client MCP tests, distribution build/content/external-install, Dockerfile static checks, and checksum-pinned mandatory Kustomize rendering. A separate supply-chain workflow defines pinned-tool, reproducible-build, SBOM, unsigned-provenance, secret-history, and fresh-vulnerability evidence. Its pull-request/`main` `evidence-integrity` decision rejects invalid evidence and unexcepted secrets while reporting High/Critical findings; the stricter `release-eligibility` decision executes only for a `v*` tag or an explicit manual boolean and rejects those unresolved findings and pending external approval. Items below that are not in those configured jobs remain target lanes rather than implied delivered coverage. In particular, both repository decisions are preconditions and do not claim bank signing, admission, risk acceptance, or promotion.

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
- explicitly provisioned, offline OCR/GPU suite (the required lane must fail if
  `BDDK_REQUIRE_GPU_OCR=1` prerequisites are absent);
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
2. Independently annotate, adjudicate, approve, and then expand the tracked 20-case/eight-domain Turkish draft toward a 100-query retrieval/citation set.
3. Add exact-reference, legal-currentness, amendment, hard-negative, no-answer, degraded-extraction, table/formula, and prompt-injection cases with immutable source provenance.
4. Replace the synthetic legal-release fixture with one real authoritative
   family; retain the exact historical packs and reproducibly derive page text
   from raw source bytes (or record an explicitly policy-bound human exception).
5. Implement execution of the exact approved expert dataset, claim-evidence
   grading, unsupported-addition penalties, abstention scoring, and human
   calibration. Keep currentness/version/amendment scoring disabled until real
   authoritative fixtures support it.
6. Execute the recovery-v2 workflow first for current schema v8 and then at
   representative bank-like scale; retain accepted elapsed time, lock,
   database/WAL, 53-object, activation/request-binding, seven-LOGIN,
   live/retained fingerprint, and restore evidence before any bank upgrade.
   Measure generation backup growth with a controlled backup; do not infer it
   from relation size or the non-exclusive retention-command WAL interval.
7. Publish the first versioned baseline across the official client plus selected Claude, Codex/ChatGPT, LM Studio, and GPT-OSS host/model profiles; record skipped/unavailable profiles rather than imputing success.

The roadmap maps these deliverables into reviewable issues: [ROADMAP.md](ROADMAP.md).
