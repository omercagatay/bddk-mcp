# Executive Summary

Review baseline: commit **5684a34c10e6d90bc22d6ab2a6466944afb6bf81**, reviewed 2026-07-14. Historical findings below describe that commit unless a post-review note says otherwise.

## Bottom line

BDDK MCP is now a coherent engineering beta with a credible MCP, database-safety, acquisition, and deployment foundation. It should still not be relied on to decide which Turkish banking rule is legally applicable, produce audit-grade evidence, or enter bank production until the remaining legal-evidence and bank-acceptance work is completed.

The repository does more than a basic chatbot demo. It collects and normalizes regulatory documents, stores and searches them with keyword and semantic techniques, recognizes Turkish legal section types, protects important publication boundaries, exposes strict tools through official MCP transports, and tests fail-closed remote and database behavior.

The missing pieces are now concentrated in the high-value trust and deployment layers: applying the new legal-version model to a real authoritative regulation family, retaining and verifying source bytes/pages, human approval of the Turkish evaluation set, whole-corpus rollback, accepted bank identity/network/CA/backup/signing controls, and proof with the intended client/model combinations.

## Implementation progress overlay — 2026-07-15

The current working tree has moved beyond the original engineering prototype in several concrete ways:

- **Complete at repository/application level:** a packaged MCP service with official installed stdio and Streamable HTTP tests; one strict registry with 15 public plus 13 additional operator tools; privacy-safe protocol errors; separate public/operator profiles and DSNs; fail-closed Host, HTTPS Origin, asymmetric JWT, scope, body, rate, and concurrency checks; RFC 9728 protected-resource discovery plus the matching 401 challenge; checksum v0001-v0004 migrations; exact PostgreSQL identity/ACL/TLS/catalog contracts; durable operator jobs; bounded acquisition; PG17 compatibility enforcement; a checksummed corpus-scope manifest; and a hardened OpenShift starter. Bootstrap now reads exact manifest-role paths, rejects undeclared reserved seed files, and can enforce freshness/signature policy with a separately mounted key in the same importing process before opening a database pool. Its completion output includes path-free manifest ID/SHA evidence, though that identity is not yet stored in PostgreSQL. Every source-backed public text result encloses metadata and bodies in one escaped untrusted-data boundary. The offline deployment preflight hashes the exact Kustomize v5.8.1 executable and rejects drift in the reviewed manifest and restricted-security inventories.
- **Complete as technical pilots, not production evidence:** v0004 provides eleven owner-only canonical legal tables. `SourceBlob` represents content identity and `SourceArtifact` separately represents an acquisition; the frozen-whitespace profile binds normalized offsets to exact retained section text. Catalog readiness attests 69 constraints and 21 indexes. Citation v1 reconstruction is proven with synthetic data through an official MCP session and real PostgreSQL. These controls do not retain authoritative source bytes/pages, authenticate a curator or source, or establish any real regulation family's legal currentness.
- **Partial:** guarded migration rehearsal and logical restore workflows exist, but the full second-cluster restore/PITR/RTO/RPO run is external. The OpenShift acceptance harness performs a strict, secret-free repository preflight, not a bank namespace run. It renders and checks a `bank-bootstrap` overlay that passes the strict corpus gates directly, mounts the read-only corpus PVC separately from its verification-key Secret, requires approved live-source/proxy HTTPS for both public and operator, and denies that reach to lifecycle Jobs; all eight live bank/cluster gates remain open. The supply-chain lane builds with Buildx `--provenance=false --load`, binds descriptor/manifest/config/loaded-image/Syft evidence, emits an unsigned repository SLSA envelope, checks model-manifest/runtime/Dockerfile consistency, and enforces secret and High/Critical vulnerability policy. Pending exceptions always leave promotion ineligible; bank signing/admission/promotion remains open. The 20-case Turkish evaluation file is deliberately a draft with all annotations, adjudications, approvals, and Citation mappings pending.
- **Release trust boundary:** expert evaluation can become release evidence only with three separately verified inputs: a signed corpus manifest with per-document measured freshness, a separately signed expert dataset, and a separately signed legal-curator attestation over the exact validated Citation export. Reusing the dataset signer as the legal-curator signer is rejected. The checked-in corpus/dataset fail these gates by design.
- **Open:** a real authoritative legal family and curator/reviewer authority; retained artifact bytes and true source-page evidence; expert-approved Turkish retrieval/grounding results; named client/model certification; bank-applied identity/CA/egress/LOGIN/registry/signing proof; numeric SLOs; whole-corpus generation rollback; and validated provision-to-audit-control mappings.

The secure remote application path is now credible for pre-production integration, but this is not production or bank deployment approval. Bank IdP, CA, registry, Route, egress, and network decisions remain unknown. Any older “working-tree” checkpoint sentence in the historical sections below is superseded by this dated overlay.

### Implementation-checkpoint ratings

| Area | Current rating | Plain-language meaning |
|---|---:|---|
| Overall maturity | 3/5 | The project is a coherent engineering beta with clear boundaries, but not yet an audit-grade regulatory knowledge product. |
| Production readiness | 2/5 | Strong repository controls and deployment starters exist; bank integration, recovery, signed delivery, SLOs, and cluster acceptance remain unproved. |
| MCP implementation | 4/5 | Official transports, strict profiles/contracts, stable errors, authentication, and protocol E2E tests are strong; named-client/version evidence remains. |
| Retrieval quality | 3/5 | Hybrid retrieval, structural parsing, current-hash publication, a synthetic legal-version pilot, and exact Citation v1 are credible; real legal currentness, authoritative pages/bytes, and expert-approved Turkish evaluation remain unsolved. |
| Security | 3/5 | Application, database identity/ACL/TLS, acquisition, job-durability, and starter-platform controls fail closed in important paths; bank-specific acceptance and recovery remain. |
| Testing and evaluation | 3/5 | Unit, PostgreSQL 17, protocol, package, migration/recovery, deployment-contract, supply-chain-policy, and benchmark-contract coverage is broad; expert approval, live models, load, full restore, and cluster evidence remain open. |
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

The code is modular, parameterizes database queries, uses transactions in important write paths, has a lockfile, passes lint/format checks, and has a broad unit suite. The isolated review run produced **526 passing tests**.

### Grounding intent

The server explicitly tells models to use local documents, cite sources, state when evidence is unavailable, and surface extraction warnings. That is the right product intent, although it must be backed by structured evidence rather than instructions alone.

Evidence: **bddk_mcp/server.py:32-52**.

## What is fragile

### The documented local MCP startup path was broken at the reviewed commit

The reviewed README and checked-in client configuration used **mcp run server.py**. That command imported the server object but skipped the project's startup function, where tools and dependencies were registered. A runtime check found zero tools.

The current working tree closes this launcher defect with an installed subprocess test that covers initialize/list/call, protocol-only stdout, invalid-input recovery, and shutdown. Named client/version compatibility remains broader follow-on evidence.

### Publication is guarded per document, not as one corpus generation

At the reviewed commit, document, section, and vector updates could diverge and retrieval could prefer stale chunks. The current repository replaces document text and parsed sections in one transaction, joins chunks to the current document hash, records retrieval-publication state, invalidates incomplete indexes, and refuses retrieval when integrity is not satisfied. Migration v0003 also refuses its blocking backfill on a populated corpus unless the operator explicitly approves it and then reindexes existing content.

This closes the stale-current-document failure mode, but it is not immutable whole-corpus staging, one atomic release switch, or rollback to a previously validated generation.

### Fresh seed installations lacked exact sections at the reviewed commit

The reviewed seed import loaded documents and chunks but did not populate the `document_sections` table. The current repository builds and validates parser-detectable sections, exercises fresh PostgreSQL bootstrap and reindex paths, and includes exact-reference/alias fixtures. A rehearsal on the bank's actual corpus size and database remains open.

### Packaging and deployment are incomplete

At the reviewed commit the Python package could not be built and the containers mixed startup, database schema changes, seed import, embedding, and serving. The current repository builds and externally installs wheel/sdist artifacts, separates migration/bootstrap from serving, adds fixed health routes, and supplies a non-root OpenShift starter with digest-only application images, stable selectors, exact Secret references, PostgreSQL CA/`verify-full`, and default-deny egress. Bank-specific egress, IdP, Route, CA and registry values; signing/SBOM/vulnerability acceptance; restore evidence; and a real cluster deployment remain open.

### Observability is instrumented but not yet an operational service

Current tool calls update thread-safe metrics and correlation-aware, content-safe logs; telemetry can use a distinct append-only PostgreSQL identity. Metrics are still process-local and are not exported through an accepted Prometheus/OpenTelemetry pipeline. There are no measured release SLOs, bank retention rules, or alerts proven in the target cluster.

## What is unsafe

### Remote HTTP access

The reviewed Streamable HTTP server listened on all interfaces without a caller boundary. The current application fails closed on non-loopback startup unless exact Host/HTTPS Origin, asymmetric JWT/JWKS, profile-scope, body, rate, and concurrency controls are configured. Limits remain process-local, and actual bank TLS/IdP/ingress integration and global enforcement are unproved.

Operator tools now require a distinct process profile, DSN, scope, and explicit remote opt-in. Job records and privacy-safe audit state are durable in PostgreSQL, and advisory leases serialize mutations. The bank must still prove private network reachability, actual principals/scopes, and one-replica operator operation.

### Database privilege

The current repository removes schema, seed, cache-population, and embedding lifecycle writes from serving; supplies separate schema-owner, ingestion, public, operator, and telemetry roles/grants; requires the expected database and schema owner; enforces `verify-full` transport; detects ACL provenance and effective privilege; and validates every pooled public/operator connection. The disposable PG17 transactional denial/allow and actual-LOGIN identity/ACL contracts executed locally and passed. The unsafe unknown is whether the bank's actual LOGINs, memberships, HBA/TLS policy, role names, and restore/upgrade process satisfy that contract.

### Private query logging

At the reviewed commit, normal INFO logs included truncated tool arguments and result previews. Current tool-boundary logs retain metadata only by default, add correlation, and put content preview behind an explicit warned opt-in. Bank log export, retention, access, and incident-response policy remain open.

### Misleading citations

The system labels fixed 5,000-character text windows as pages. Those are not necessarily pages in the official PDF. Section page fields are not populated. An apparently precise page citation can therefore be wrong.

### Legal currentness

Draft, current, old, and repealed material can all be retrieved without a validated effective-date/status model. The system cannot safely answer “what applies today?” or “what applied on this date?”

### Formula and table dependence

Only two of 318 seeded documents were marked with a formula-aware extraction method by the repository's classifier. Capital, liquidity, IRB, interest-rate-risk, and TFRS 9 work can depend on a sign, denominator, threshold, or table cell; those documents require source-level validation before calculation or audit reliance.

## What is missing

- stable identity for a regulation across legal versions;
- official publication/effective/repeal dates and validated current status;
- amendment, replacement, repeal, consolidation, and citation relationships;
- hierarchical provision IDs such as article/fıkra/bent;
- immutable official source artifacts and hashes;
- physical source-page/table/formula provenance;
- an audit-grade citation that a reviewer can reconstruct to an authoritative source page and legal version;
- immutable whole-corpus staging, validation, one-step publication, and rollback;
- application of the repository database-role/identity contract to the bank's actual LOGINs and proof of bank-cluster public/operator isolation;
- bank-integrated TLS, IdP, ingress-global limits, and authorization evidence;
- named MCP host/model compatibility matrix beyond official reference-client E2E;
- representative Turkish regulatory retrieval and citation benchmarks;
- claim-by-claim answer grounding evaluation;
- validated mappings from provisions to obligations, controls, audit steps, and evidence;
- exported metrics/traces, measured SLOs, backups, restore drills, and bank-sized upgrade procedures;
- clear code/data licensing and provenance policy.

## The five most important findings

1. **The system still cannot prove legal applicability.** It has extraction snapshots, not validated official legal versions, effective intervals, or amendment/repeal/consolidation status.
2. **Citations can look more precise than the source evidence.** Character windows are presented as pages, and authoritative source-page/table/formula coordinates are not retained.
3. **Consistency is per document, not per released corpus.** Current-hash publication now fails closed, but there is no immutable validated generation, atomic whole-corpus switch, or rollback.
4. **Bank production acceptance remains unproved.** Repository controls are strong, but actual LOGINs, IdP/CA/Route/egress, signed image delivery, backup/restore, SLOs, and OpenShift cluster behavior have not been accepted.
5. **Evaluation cannot yet support client/model selection or audit-reliability claims.** Phase 2 now uses official MCP stdio/HTTP sessions and a fail-closed grader, but the expert gold data is too small and no named live model/client result is accepted.

## What should be done first

### 1. Make a signed corpus release durable and traceable

Persist the verified manifest identity atomically with successful bootstrap, set numeric freshness objectives, collect the per-document measurement chain, sign the canonical manifest, keep the trust key separate, and prove the same strict import on a disposable database. Job output alone is not durable release identity.

### 2. Set numeric operational targets and execute the full restore

Approve availability, latency, freshness, RPO, RTO, alert, and evidence-retention values; then run the delivered `pg_dump`/`pg_restore` workflow between distinct PG17 clusters and retain sanitized measured evidence.

### 3. Execute the bank integration controls, not just repository preflight

Apply the reviewed roles to actual bank LOGINs; validate HBA/TLS/`verify-full`; provision the separate corpus PVC and trust Secret; run migration → grants → strict bootstrap in order; validate RFC 9728 discovery and JWT behavior through the bank Route/IdP; and admit only the signed release-image digest.

### 4. Curate one authoritative real regulation family

Retain authoritative bytes and page evidence, load one bounded amendment chain into the v0004 model, record effective/repeal/unknown states and curator authority, and permit current/as-of answers only when exact Citation evidence validates.

### 5. Complete one expert/model vertical slice

Finish two annotations and adjudication for all 20 Turkish pilot cases, bind them to verified Citations and separately signed corpus/dataset/legal attestations, then run one pinned client/host/model through the real MCP harness. Expand to Claude, Codex, GPT/GPT-OSS, and LM Studio only after that pilot is trustworthy.

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

The current repository has focused PostgreSQL privilege/lifecycle/publication tests, a dedicated actual-LOGIN/ACL CI contract, official MCP stdio/HTTP tests, package-install verification, and mandatory rendered OpenShift contract tests. It still has no evidence from the bank's cluster or bank LOGINs, live upstream production operation, intended named clients/models, GPU workload, size-matched migration, or backup/restore drill. No secret values were inspected or exposed.

Owner clarifications establish that the corpus is an intentional job-specific selection, the owner will validate regulatory content and administer the data layer, source/data use is accepted by the owner, and deployment is expected immediately on a bank's on-premises OpenShift AI environment. Exact client versions, tenant/private-document needs, bank platform values, and measurable “immediate” freshness/availability/recovery targets remain to be defined. Until then, the safe deployment assumption is single-tenant, private network, separate public/operator planes, and no private-document ingestion.

For detailed evidence and the implementation plan:

- [REPOSITORY_REVIEW.md](REPOSITORY_REVIEW.md)
- [GAP_REGISTER.md](GAP_REGISTER.md)
- [ROADMAP.md](ROADMAP.md)
- [TARGET_ARCHITECTURE.md](TARGET_ARCHITECTURE.md)
