# Executive Summary

Review baseline: commit **5684a34c10e6d90bc22d6ab2a6466944afb6bf81**, reviewed 2026-07-14. The dated implementation overlay and current conclusions below describe the present roadmap worktree; baseline observations are retained only when they are explicitly labelled historical.

## Bottom line

BDDK MCP is now a coherent engineering beta with a credible MCP, database-safety, acquisition, and deployment foundation. It should still not be relied on to decide which Turkish banking rule is legally applicable, produce audit-grade evidence, or enter bank production until the remaining legal-evidence and bank-acceptance work is completed.

The repository does more than a basic chatbot demo. It collects and normalizes regulatory documents, stores and searches them with keyword and semantic techniques, recognizes Turkish legal section types, protects important publication boundaries, exposes strict tools through official MCP transports, and tests fail-closed remote and database behavior.

The missing pieces are now concentrated in the high-value trust and deployment layers: applying the legal model to a real authoritative regulation family, supplying real retained source-byte/page evidence, repairing and approving the tracked corpus, human approval and execution of the Turkish evaluation set, generation-bound serving and authorized corpus rollback, accepted bank identity/network/CA/backup/signing controls, approved operational targets, and proof with the intended client/model combinations. Schema v7 now creates sealed rollback targets, but it does not yet make them servable or activatable.

## Implementation progress overlay — 2026-07-16

The current working tree has moved beyond the original engineering prototype in several concrete ways:

- **Confirmed complete at repository/application level:** the packaged service exposes one canonical registry with **15 public tools plus 14 operator additions, 29 total**, through official stdio and Streamable HTTP paths. It also registers exactly one MCP resource, `bddk://corpus/active-release`, and no prompts. Strict generated inputs, risk annotations, privacy-safe errors, separate public/operator profiles and DSNs, Host/HTTPS-Origin/JWT/scope/body/rate/concurrency controls, RFC 9728 discovery, and official-client tests are present (**bddk_mcp/tools/registry.py:21-71,172-176; bddk_mcp/resources.py:17-75; bddk_mcp/server.py:668-670; tests/test_mcp_http_runtime.py:35-57**).
- **Confirmed database and corpus-publication boundary:** the checksum ledger now ends at **schema v7**: v4 supplies the owner-only canonical legal model and its attested 69-constraint/21-index subset, v5 supplies append-only corpus releases/activations plus a mutation epoch, v6 supplies the least-privilege abstention-first legal-status resolver, and v7 supplies a separately attested retained-generation plane (**bddk_mcp/migrations/runner.py; bddk_mcp/catalog_integrity.py:450-580**). A strict publisher persists release identity, manifest and retrieval-profile hashes, freshness-policy evidence and a corpus-state fingerprint. Any later corpus mutation invalidates the active view; strict local-corpus calls take a same-release reader lease and fail closed if the release changes before completion (**bddk_mcp/migrations/v0005_corpus_release_publication.py; bddk_mcp/corpus_serving.py**). OpenShift enables this strict mode; local research can explicitly leave it disabled.
- **Confirmed immutable target, not rollback:** the publisher-only `bddk-mcp retain-corpus-generation` command can copy the exact active state across **17 typed retained relations**, reproduce its exact v5 state hash, seal it, and create a per-release binding in one transaction. Generation, release, seal, and activation remain different identities. Because generation and seal are derived from exact corpus state plus retrieval profile, differently governed releases over that same state/profile share one physical generation and seal through distinct bindings rather than duplicate storage; old v5 releases without evidence are labelled `legacy_v5_unretained`. The command is not an MCP tool and changes neither serving nor activation (**bddk_mcp/migrations/v0007_retained_corpus_generations.py; bddk_mcp/corpus_generations.py; bddk_mcp/cli.py; tests/test_corpus_publication.py**). It creates a future rollback target, but H2-02B must still bind every read/cache to a generation and add separately authorized activation/reactivation. Backup growth and bank retention/capacity approval remain open.
- **Confirmed lifecycle and recovery boundary:** migration, ingestion bootstrap and release publication use separate identities and OpenShift Jobs; retention is currently a separate approved one-shot CLI operation. The release publisher independently regenerates deterministic derivatives and proves exact database membership before append-only activation (**bddk_mcp/ingest/seed.py; deploy/openshift/jobs/**). Recovery evidence schema v2 inventories **51 managed objects**; that count includes the activation sequence, 17 retained member tables, and their generation/inventory/seal/release-binding/status records. It rejects reuse of all six application DSNs for recovery administration, verifies six restored LOGIN profiles, and compares logical and identity evidence exactly. A retained synthetic two-cluster PostgreSQL 17 run now proves the complete schema-v7 repository path (**bddk_mcp/operations/recovery.py:72-142; tests/test_recovery_workflows.py:472-514; docs/evidence/LOCAL_PG17_V7_RECOVERY_DRILL.md**). This is not bank PITR/RPO/RTO, custody, TLS/HBA, capacity, or recovery-acceptance evidence.
- **Confirmed evaluation controls, but not release evidence:** Phase 2 validates the selected corpus manifest, reads the active release through the evaluated MCP session, requires exact manifest ID/SHA equality, executes all calls on that same session, and rechecks the release at the end (**benchmark/phase2_e2e.py**). Release trust now has four signed evidence layers plus a separately signed schema-v2 policy. The policy binds five release identities with documented canonical/raw hash meanings, four distinct declared signer-owner IDs, forward rotation/revocation, current policy/deployment-scope pins, and a time-bounded/revocable reviewer owner assertion for every v2 checkpoint artifact (**benchmark/expert_evaluation.py; benchmark/legal_release_evidence.py; benchmark/evaluation_trust_policy.py; benchmark/release_preflight.py; benchmark/README.md: Hash and version semantics**). A checkpoint chain containing v1 page proofs needs a new independent all-v2-proof genesis, without a verified continuity claim to the archival chain. Distinct owner IDs and signed event times do not prove separate human custody or signing time. Even a passing preflight reports `bank_authorization_verified: false` because bank root/RBAC promotion and the human review action are external, and `model_scores_authorized: false` because expert-dataset execution is not implemented.
- **Confirmed immediate release blocker:** the tracked manifest declares 318 documents and **8,286 chunks**, while a read-only regeneration under the current pinned profile produced **9,675 chunks**. Strict publication rejects that mismatch, so the artifact must be regenerated, independently reviewed and re-signed before the first governed release (**seed_data/corpus_scope.yml:31-40; bddk_mcp/ingest/seed.py:383-405,657-698,1262-1272**). The manifest also remains unsigned, unmeasured, non-exhaustive and has no approved numeric freshness targets.
- **Operational decisions exist but remain unset:** a versioned bank-on-prem OpenShift contract defines eight metrics—availability, latency, two freshness measures, maximum corpus age, RPO, RTO and evidence retention—but every target/window is null and unapproved, alerts and evidence sources are unverified, and production eligibility fails closed (**docs/decisions/operational-objectives.v1.yml; bddk_mcp/operational_objectives.py:333-501**).
- **Open external or product boundaries:** no real authoritative regulation family, named curator/reviewer evidence, real retained legal-release bundle, bank-owned latest-checkpoint anchor, generation-bound serving or authorized corpus rollback, expert-executed model result, named client/model certification, signed promoted image, or bank-applied IdP/CA/Route/CNI/egress/LOGIN/backup/PITR acceptance exists. V7 copies database fields only; it does not acquire external authoritative files that are absent from PostgreSQL. When available, its WAL number is a non-exclusive observed cluster interval; otherwise WAL is `not_measured`. Backup growth remains `not_measured`, and bank retention/capacity authorization is open. Provision-to-control/workpaper knowledge remains future work.

The secure remote application path is credible for pre-production integration, but this is not production or bank deployment approval. The target is bank on-premises OpenShift AI; its IdP, CA, registry, Route, CNI, egress, PostgreSQL, backup and acceptance details remain unknown. Any older current-state sentence below is superseded by this dated overlay unless it is explicitly labelled historical.

### Implementation-checkpoint ratings

| Area | Current rating | Plain-language meaning |
|---|---:|---|
| Overall maturity | 3/5 | The project is a coherent engineering beta with clear boundaries, but not yet an audit-grade regulatory knowledge product. |
| Production readiness | 2/5 | Strong repository controls and deployment starters exist; bank integration, recovery, signed delivery, SLOs, and cluster acceptance remain unproved. |
| MCP implementation | 4/5 | Official transports, strict profiles/contracts, stable errors, authentication, and protocol E2E tests are strong; named-client/version evidence remains. |
| Retrieval quality | 3/5 | Hybrid retrieval, structural parsing, release-epoch binding, the v6 abstention-first resolver, and exact Citation v1 are credible; the tracked corpus cannot pass strict publication and real legal currentness, authoritative evidence, and expert-approved Turkish evaluation remain unsolved. |
| Security | 3/5 | Application, database identity/ACL/TLS, acquisition, job-durability, and starter-platform controls fail closed in important paths; bank-specific acceptance and recovery remain. |
| Testing and evaluation | 3/5 | Unit, PostgreSQL 17, protocol, package, migration/recovery, deployment-contract, supply-chain-policy, and benchmark-contract coverage is broad, and a synthetic two-cluster v7 restore has now passed. Expert approval, live models, load, bank PITR/restore, and target-cluster evidence remain open. |
| Documentation | 4/5 | Architecture, security, deployment, testing, and roadmap boundaries are now extensive; external runbooks and measured acceptance evidence remain. |

## Baseline ratings at the reviewed commit

| Area | Rating | Plain-language meaning |
|---|---:|---|
| Overall maturity | 2/5 | A functional prototype with sound building blocks, not a stable product. |
| Production readiness | 1/5 | Important correctness, security, deployment, and recovery controls are absent. |
| MCP implementation | 2/5 | At the reviewed commit, the right SDK and transports were present but the documented local client command exposed no tools. |
| Retrieval quality | 2/5 | Search technology is promising; legal currentness and audit-grade citations are not solved. |
| Security | 1/5 | Remote HTTP has no built-in identity/permission boundary and operator access is only an on/off flag. |
| Testing and evaluation | 2/5 | Many code tests pass, but real MCP, database, client, citation, and model evaluations are incomplete. |
| Documentation | 2/5 | At the reviewed commit, documentation was helpful and bilingual but several important claims were inaccurate. |

These are the review-baseline scores, not a post-implementation rerating. They measure suitability for high-stakes regulatory use. They do not mean the code is poor; they mean the evidence and controls required for that use are not complete.

## What is already strong

### Retrieval engineering

The project combines semantic search and PostgreSQL full-text search, then fuses the results. It also uses section-aware/token-aware chunking and a multilingual embedding model. This is a strong foundation for Turkish regulatory research.

Evidence: **bddk_mcp/store/vector_store.py:228-390,931-1198**.

### Document acquisition and quality controls

The ingestion code handles several BDDK/Mevzuat page and file patterns, HTML, PDF, legacy document, iframe, ZIP annex, and optional OCR paths. It records hashes and extraction methods and runs deterministic quality checks.

Evidence: **bddk_mcp/ingest/doc_sync.py; html_extractor.py; ocr/base.py; quality/markdown_quality.py; quality/quality_scan.py**.

### Useful regulatory structure

The section parser recognizes articles, temporary articles, principles, paragraphs, annexes, fıkra-like numbered paragraphs, and bent-like lettered paragraphs. Exact document and section tools are more useful than a single generic RAG endpoint.

Evidence: **bddk_mcp/store/section_index.py:24-36,118-177; bddk_mcp/tools/documents.py; sections.py**.

### Good development foundations

The code is modular, parameterizes database queries, uses transactions in important write paths, has a lockfile, and has a broad unit suite. At the reviewed baseline commit, the isolated run produced **526 passing tests**; that historical count is not a current-suite result.

### Grounding intent

The server explicitly tells models to use local documents, cite sources, state when evidence is unavailable, and surface extraction warnings. That is the right product intent, although it must be backed by structured evidence rather than instructions alone.

Evidence: **bddk_mcp/server.py:32-52**.

## What is fragile

### The documented local MCP startup path was broken at the reviewed commit

The reviewed README and checked-in client configuration used **mcp run server.py**. That command imported the server object but skipped the project's startup function, where tools and dependencies were registered. A runtime check found zero tools.

The current working tree closes this launcher defect with an installed subprocess test that covers initialize/list/call, protocol-only stdout, invalid-input recovery, and shutdown. Named client/version compatibility remains broader follow-on evidence.

### Prior generations are retained, but they cannot yet be served or reactivated

At the reviewed commit, document, section, and vector updates could diverge and retrieval could prefer stale chunks. The current repository additionally persists append-only release/activation evidence, invalidates the active view on any tracked corpus mutation, and pre/post-binds strict local-corpus tool calls to one release ID. This closes both the stale-current-document failure mode and the absence of an atomic active identity.

V7 changes the remaining limitation: releases retained after migration now have
sealed typed copies of all 17 database relations covered by the v5 state hash.
Those copies are immutable and recovery fingerprints include them. However,
public tools still read the mutable v5 serving tables, caches are not
generation-qualified, and there is no tested activation/reactivation event for
a retained generation. This is a sound rollback target, not a working rollback
feature. Pre-v7 releases without a copy are explicitly marked unretained.

### Fresh seed installations lacked exact sections at the reviewed commit

The reviewed seed import loaded documents and chunks but did not populate the `document_sections` table. The current repository builds and validates parser-detectable sections, exercises fresh PostgreSQL bootstrap and reindex paths, and includes exact-reference/alias fixtures. A rehearsal on the bank's actual corpus size and database remains open.

### Packaging and deployment are incomplete

At the reviewed commit the Python package could not be built and the containers mixed startup, database schema changes, seed import, embedding, and serving. The current repository builds and externally installs wheel/sdist artifacts, separates migration/bootstrap from serving, adds fixed health routes, and supplies a non-root OpenShift starter with digest-only application images, stable selectors, exact Secret references, PostgreSQL CA/`verify-full`, and default-deny egress. Bank-specific egress, IdP, Route, CA and registry values; signing/SBOM/vulnerability acceptance; restore evidence; and a real cluster deployment remain open.

### Observability is instrumented but not yet an operational service

Current tool calls update thread-safe metrics and correlation-aware, content-safe logs; telemetry can use a distinct append-only PostgreSQL identity. Metrics are still process-local and are not exported through an accepted Prometheus/OpenTelemetry pipeline. The eight-metric objectives contract exists, but its values/windows are unapproved and no target-cluster alerts, evidence sources or retention registry are verified.

## What is unsafe

### Remote HTTP access

The reviewed Streamable HTTP server listened on all interfaces without a caller boundary. The current application fails closed on non-loopback startup unless exact Host/HTTPS Origin, asymmetric JWT/JWKS, profile-scope, body, rate, and concurrency controls are configured. Limits remain process-local, and actual bank TLS/IdP/ingress integration and global enforcement are unproved.

Operator tools now require a distinct process profile, DSN, scope, and explicit remote opt-in. Job records and privacy-safe audit state are durable in PostgreSQL. A session lease admits an operator job while a separate transaction lock serializes sanctioned corpus writers and publication. The bank must still prove private network reachability, actual principals/scopes, and one-replica operator operation.

### Database privilege

The current repository removes schema, seed, cache-population, embedding and release-publication lifecycle writes from serving; supplies separate schema-owner, ingestion, release-publisher, public, operator and telemetry roles/grants; requires the expected database and schema owner; enforces `verify-full` transport; detects ACL provenance and effective privilege; and validates pooled runtime connections. Disposable PG17 identity/ACL contracts are repository evidence. The unsafe unknown is whether the bank's actual six LOGIN profiles, memberships, HBA/TLS policy, role names, and restore/upgrade process satisfy that contract.

### Private query logging

At the reviewed commit, normal INFO logs included truncated tool arguments and result previews. Current tool-boundary logs retain metadata only by default, add correlation, and put content preview behind an explicit warned opt-in. Bank log export, retention, access, and incident-response policy remain open.

### Citation evidence is technically stronger than the available real corpus

The reviewed commit labelled fixed character windows as pages. Current Citation v1 reconstructs exact normalized ranges through a validated legal view, and the legal-release verifier can re-hash retained source bytes, acquisition records, source-page text/mappings and the rendered Citation excerpt. `PageMappingProof` v2 can bind each review to an owner authorized by the signed policy, but it does not authenticate the human action or reproduce raw-source/PDF-to-page derivation. No real authoritative non-fixture bundle has passed that verifier, and table/formula coordinates remain unresolved. Product output must not imply that a technical capability proves authoritative page evidence for the tracked corpus.

### Legal currentness remains unproved for real regulations

Schema v6 exposes a public resolver that returns one legal version only when exact validated authoritative publication, effective-date and status evidence supports the requested date; otherwise it abstains. The repository fixture proves that behavior synthetically. Ordinary ingestion is not connected to a real curated legal family, so the system still cannot safely answer “what applies today?” for the tracked corpus.

### Formula and table dependence

Only two of 318 seeded documents were marked with a formula-aware extraction method by the repository's classifier. Capital, liquidity, IRB, interest-rate-risk, and TFRS 9 work can depend on a sign, denominator, threshold, or table cell; those documents require source-level validation before calculation or audit reliance.

## What is missing

- application of the implemented legal identities, versions, effective/status claims and hierarchical provisions to a real authoritative regulation family;
- real curator/reviewer authority and source-authenticity evidence;
- a real retained source-byte/acquisition/page/excerpt bundle that passes the implemented legal-release verifier;
- physical table/formula provenance and an audit-grade Citation pack for the real corpus;
- generation-bound serving and tested, separately authorized rollback/reactivation; v7 retained targets and atomic active-release publication are implemented;
- repair, independent review, signature and measured freshness for the tracked 8,286-row chunk artifact, which currently regenerates as 9,675 rows;
- application of the repository database-role/identity contract to the bank's actual LOGINs and proof of bank-cluster public/operator isolation;
- bank-integrated TLS, IdP, ingress-global limits, and authorization evidence;
- named MCP host/model compatibility matrix beyond official reference-client E2E;
- representative Turkish regulatory retrieval and citation benchmarks;
- claim-by-claim answer grounding evaluation;
- validated mappings from provisions to obligations, controls, audit steps, and evidence;
- approval and implementation of the existing eight-metric operational-objectives contract, exported metrics/traces, bank restore/PITR evidence, and bank-sized upgrade procedures;
- clear code/data licensing and provenance policy.

## The five most important findings

1. **The first governed corpus release is blocked.** The tracked 8,286-row chunk artifact regenerates as 9,675 rows, is unsigned and unmeasured, and therefore cannot pass strict publication.
2. **Legal applicability remains unproved for real regulations.** The v4/v6 model and abstention behavior exist, but no authoritative non-fixture family has been curated and accepted.
3. **Immutable rollback targets are implemented, rollback is not.** PostgreSQL persists an atomic active identity and v7 can retain a sealed 17-relation copy, but tools cannot serve that generation and no authorized reactivation event exists.
4. **Bank production acceptance remains unproved.** Repository controls are strong, but actual six-role LOGINs, IdP/CA/Route/CNI/egress, signed image delivery, backup/PITR, approved objectives, and OpenShift behavior have not been accepted.
5. **Evaluation trust checks cannot authorize a model.** Phase 2 has same-session corpus binding and the four-layer cryptographic gate exists, but bank authorization is false, expert-dataset execution is absent, and no named live model/client result is accepted.

## What should be done first

### 1. Repair and publish the first strict corpus release

Regenerate the 9,675 current-profile chunks, independently review the changed artifact, set approved numeric freshness objectives, collect the per-document measurement chain, sign the canonical manifest, keep the trust key separate, bootstrap through the ingestion identity, activate it through the independent release publisher, and retain that active release through the one-shot publisher CLI. Durable release identity and immutable database retention are already implemented; generation-bound rollback is not.

### 2. Set numeric operational targets and repeat the full restore in the bank boundary

Approve values and rolling windows in the delivered eight-metric operational-objectives contract, including availability, latency, freshness, RPO, RTO, alerts, evidence retention, retained-generation count, and capacity. The synthetic repository-scale two-cluster workflow has passed; repeat it with bank TLS/HBA/LOGINs, target-size data, backup custody and PITR, retaining sanitized evidence for all 51 managed objects—including the activation sequence—and six identities. Measure backup growth with a controlled backup; do not substitute the retention command's non-exclusive WAL interval or catalog size.

### 3. Execute the bank integration controls, not just repository preflight

Apply the reviewed roles to the six actual application LOGIN profiles; validate HBA/TLS/`verify-full`; provision the separate corpus PVC and trust Secret; run migration → grants → ingestion bootstrap/reindex → independent release publication → one-shot retention in order; validate RFC 9728 discovery and JWT behavior through the bank Route/IdP; and admit only the signed release-image digest.

### 4. Curate one authoritative real regulation family

Retain authoritative bytes and page evidence, load one bounded amendment chain into the v4 legal model, validate it through the v6 resolver, record effective/repeal/unknown states and curator authority, and require the four-layer release evidence—including a bank-anchored legal-release checkpoint—before permitting current/as-of claims.

### 5. Complete one expert/model vertical slice

Finish two annotations and adjudication for all 20 Turkish pilot cases, bind them to the signed corpus, signed dataset, signed legal-curator Citation pack and signed legal-release checkpoint under separated identities, then implement expert-dataset execution and run one pinned client/host/model through the same-session MCP harness. Expand to Claude, Codex, GPT/GPT-OSS, and LM Studio only after that pilot is trustworthy.

## What this project could realistically become

With the roadmap implemented, BDDK MCP could become a strong local or enterprise regulatory evidence service:

- reliable retrieval of Turkish banking provisions;
- reproducible “as of date” research;
- exact, reviewable citations;
- model-independent MCP access;
- controlled comparison of Claude, GPT-family, GPT-OSS, Codex, LM Studio, and other host/model combinations;
- traceable links from provisions to obligations, controls, audit tests, and required evidence;
- a validated regulatory knowledge layer that accumulates reviewed relationships rather than merely adding more chunks.

That outcome does not require an immediate graph database or large distributed platform. PostgreSQL, pgvector, immutable source artifacts, typed relations, strong evaluation, and disciplined publication are enough for the next major version.

## Commercial-use objective

The current MIT license expressly permits commercial use, sublicensing, and sale. No technical control in a locally distributed open-source copy can reliably prevent reuse. Already distributed MIT rights cannot realistically be withdrawn by adding a future runtime check.

The project owner should obtain legal advice before further releases and separate:

- code licensing;
- regulatory-source and derived-data rights;
- validated knowledge/control packs;
- hosted/enterprise access terms.

Authentication, authorization, entitlements, and contracts can control a hosted or private operator service. They cannot make an MIT local copy noncommercial.

## Review limitations

The initial review used an immutable archive of the exact origin/main commit and did not inspect production infrastructure. The owner's initial checkout had pre-existing Git deletion entries for tracked files; the review did not create, restore, or commit those entries, and later implementation used a separate clean worktree. That historical provenance does not mean current source files are deleted.

The current repository has focused PostgreSQL privilege/lifecycle/publication tests, a dedicated actual-LOGIN/ACL CI contract, official MCP stdio/HTTP tests, package-install verification, recovery-v2 orchestration, mandatory rendered OpenShift contract tests, and a retained synthetic two-cluster schema-v7 restore report. That local report is repository evidence only. No retained report proves a bank cluster, bank LOGINs, live upstream production operation, intended named clients/models, GPU workload, bank-sized migration, or bank-accepted backup/PITR drill. No secret values were inspected or exposed.

Owner clarifications establish that the corpus is an intentional job-specific selection, the owner will validate regulatory content and administer the data layer, source/data use is accepted by the owner, and deployment is expected immediately on a bank's on-premises OpenShift AI environment. Exact client versions, tenant/private-document needs, bank platform values, and measurable “immediate” freshness/availability/recovery targets remain to be defined. Until then, the safe deployment assumption is single-tenant, private network, separate public/operator planes, and no private-document ingestion.

For detailed evidence and the implementation plan:

- [REPOSITORY_REVIEW.md](REPOSITORY_REVIEW.md)
- [GAP_REGISTER.md](GAP_REGISTER.md)
- [ROADMAP.md](ROADMAP.md)
- [TARGET_ARCHITECTURE.md](TARGET_ARCHITECTURE.md)
