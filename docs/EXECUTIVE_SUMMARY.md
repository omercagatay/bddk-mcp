# Executive Summary

Review baseline: commit **5684a34c10e6d90bc22d6ab2a6466944afb6bf81**, reviewed 2026-07-14. Historical findings below describe that commit unless a post-review note says otherwise.

## Bottom line

BDDK MCP is a promising working prototype, but it should not yet be relied on to decide which Turkish banking rule is legally applicable, or be exposed as an unauthenticated Internet service.

The repository already does more than a basic chatbot demo. It collects and normalizes regulatory documents, stores and searches them with both keyword and semantic techniques, recognizes many Turkish legal section types, warns about some extraction problems, and exposes the results through MCP. It also has a large automated test suite.

The missing pieces are the ones that matter most when the answer may be reviewed by an auditor, compliance officer, model validator, or regulator: authoritative version history, effective dates, amendment/repeal relationships, exact source-page evidence, secure remote access, reliable client startup, and a benchmark that proves the system retrieved and cited the right rule.

## Post-review implementation checkpoint

The working tree now fixes several immediate engineering defects found in the baseline review: the importable MCP server exposes the canonical 15-tool public surface; buildable package metadata and portable console commands exist; serving startup is separated from explicit migration/bootstrap work; seed bootstrap builds section indexes and accepts common numeric aliases; the 11-item quality-failure registry is applied at runtime; and default tool-boundary logs omit query, result, and error content. Benchmark function schemas are now derived from the canonical 26-tool operator registry.

These changes are meaningful stabilization, not production approval. Official documented-command subprocess tests, clean installation acceptance, disposable-PostgreSQL bootstrap tests, application authentication and authorization, Host/Origin and rate controls, separate operator/public processes and database roles, atomic corpus publication, legal-version/currentness evidence, audit-grade citations, OpenShift hardening, and representative model/retrieval evaluation remain open.

## Ratings

| Area | Rating | Plain-language meaning |
|---|---:|---|
| Overall maturity | 2/5 | A functional prototype with sound building blocks, not a stable product. |
| Production readiness | 1/5 | Important correctness, security, deployment, and recovery controls are absent. |
| MCP implementation | 2/5 | At the reviewed commit, the right SDK and transports were present but the documented local client command exposed no tools. The working-tree fix still needs subprocess acceptance. |
| Retrieval quality | 2/5 | Search technology is promising; legal currentness and audit-grade citations are not solved. |
| Security | 1/5 | Remote HTTP has no built-in identity/permission boundary and operator access is only an on/off flag. |
| Testing and evaluation | 2/5 | Many code tests pass, but real MCP, database, client, citation, and model evaluations are incomplete. |
| Documentation | 2/5 | At the reviewed commit, documentation was helpful and bilingual but several important claims were inaccurate. A truth pass is now present, with compatibility evidence still incomplete. |

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

The working tree now exports a populated server, provides a packaged CLI, and passes official in-memory client list/call tests. A real documented-command subprocess test, protocol-only stdout check, and installed-server initialization test are still required before closing the finding.

### The corpus is not published atomically

The current document, its sections, and its vector chunks are updated in separate steps. If vector indexing fails after a document changes, the exact document and search index can disagree. Retrieval can prefer the stale index.

At the reviewed commit, startup seed import could also treat any content difference as drift and overwrite a fresher deployed document with the bundled copy without preserving the normal version archive. The working tree removes this import from `serve`, but explicit bootstrap still lacks immutable staging and atomic generation publication.

### Fresh seed installations lacked exact sections at the reviewed commit

The reviewed seed import loaded documents and chunks but did not populate the `document_sections` table. The working tree now builds and validates parser-detectable sections and includes focused exact-reference/alias fixtures. Fresh disposable-PostgreSQL and repeat-bootstrap acceptance remain open.

### Packaging and deployment are incomplete

At the reviewed commit the Python package could not be built and the containers mixed startup, database schema changes, seed import, embedding, and serving. The working tree now builds wheel/sdist artifacts and uses explicit bootstrap before read-only lifecycle startup. The containers still lack an application health route, run as root, use mutable supply-chain references, and have not been accepted on OpenShift.

### Observability gives false confidence

The metrics object exists, but normal tool calls do not update it. The operator metrics tool can therefore report zero activity while the service is being used.

## What is unsafe

### Remote HTTP access

The reviewed Streamable HTTP server listened on all interfaces. The working tree defaults local HTTP to loopback, while container profiles explicitly bind all interfaces. The repository still does not implement caller authentication, tool authorization, inbound rate limits, or an explicit deployment-reviewed Host/Origin allowlist.

Operator tools are hidden by default, but enabling the environment flag exposes sync, refresh, migration-like, and backfill actions to every caller of that server. The flag controls visibility; it is not a permission system.

### Database privilege

The working tree removes schema, seed, cache-population, and embedding lifecycle writes from normal serving startup, and telemetry remains off by default. It has not yet provisioned or proved separate schema-owner, ingestion, serving-reader, operator, and telemetry roles; enabling telemetry or operator tools still requires write authority.

### Private query logging

At the reviewed commit, normal INFO logs included truncated tool arguments and result previews. Working-tree tool-boundary logs now retain metadata only by default, with content preview behind an explicit warned opt-in. Complete tool-family coverage, broader internal/upstream exception review, correlation, and an approved retention/access policy remain open.

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
- a typed citation that a reviewer can reconstruct;
- atomic staging, validation, publication, and rollback of a corpus;
- separate public and operator services and database roles;
- secure remote authentication and authorization;
- official MCP end-to-end tests and client compatibility matrix;
- representative Turkish regulatory retrieval and citation benchmarks;
- claim-by-claim answer grounding evaluation;
- validated mappings from provisions to obligations, controls, audit steps, and evidence;
- health checks, metrics/traces, backups, restore drills, versioned migration history, and upgrade procedures;
- clear code/data licensing and provenance policy.

## The five most important findings

1. **The advertised stdio integration exposed zero tools at the reviewed commit.** The working tree corrects registration and packaging, but subprocess and installed-server acceptance remain open.
2. **Remote serving has no trustworthy access-control boundary.** HTTP is open by repository design, and operator mode is only a global flag.
3. **The system cannot prove legal applicability.** It has extraction snapshots, not official legal versions, effective dates, or amendment/repeal status.
4. **Citations can look more precise than the evidence.** Character windows are presented as pages, and official source-page coordinates are not retained.
5. **Current evaluation cannot support model-selection claims.** The nominal end-to-end benchmark calls a route the MCP server does not expose, uses three gold cases, and can silently weaken its grader.

## What should be done first

### 1. Finish acceptance of the corrected MCP path

The working tree adds an installed entry point and canonical 15/26 registry. Complete official-client subprocess/HTTP tests, installed-wheel initialization, cancellation/shutdown checks, and client compatibility evidence.

### 2. Close the remote/operator security gap

Local HTTP now defaults to loopback and default tool logs omit content. Require Host/Origin policy, identity, scopes, and rate limits for remote use; put operator tools in a separate private process; and approve retention/access policy.

### 3. Stop changing the corpus during serving startup

Explicit migration/bootstrap now owns schema, seed, section, and embedding work. Prove serving-reader write denial, separate operator roles/jobs, and publish a validated corpus generation atomically.

### 4. Define the evidence contract

Create structured outputs and citations that name the document, legal version, provision, official URL, source hash, true page or normalized range, quality state, and corpus generation. Never call a display window a source page.

### 5. Build the legal-version foundation and a trustworthy benchmark

Represent publication/effective/repeal state and amendment relations. Then create expert-reviewed Turkish queries for exact articles, cross-references, acronyms, tables/formulas, currentness, negative cases, and claim-level citation correctness.

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

The review used the exact origin/main commit and did not inspect production infrastructure. Database/GPU integration tests, real clients, live BDDK sources, model runs, and deployment state remain unverified. No secret values were inspected or exposed.

Owner clarifications after the review establish that the corpus is an intentional job-specific selection, the owner will validate regulatory content and administer the data layer, source/data use is accepted by the owner, and the deployment target is a bank's on-premises OpenShift AI environment. Exact client versions, tenant/private-document needs, platform security controls, and measurable “immediate” freshness/availability/recovery targets remain to be defined. Until then, the recommended deployment assumption is single-tenant, private network, separate public/operator planes, and no private-document ingestion.

For detailed evidence and the implementation plan:

- [REPOSITORY_REVIEW.md](REPOSITORY_REVIEW.md)
- [GAP_REGISTER.md](GAP_REGISTER.md)
- [ROADMAP.md](ROADMAP.md)
- [TARGET_ARCHITECTURE.md](TARGET_ARCHITECTURE.md)
