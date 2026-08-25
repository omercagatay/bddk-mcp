# Documentation

This index is the canonical map of BDDK MCP documentation. Documents are grouped by purpose so current operating guidance is not confused with historical reviews or external acceptance evidence.

## Start here

- [Project overview and Turkish quick start](../README.md)
- [English operational guide](../README.en.md)
- [Current repository status](STATUS.md)
- [Repository structure](REPOSITORY_STRUCTURE.md)
- [Contributing](../CONTRIBUTING.md)
- [Security policy](../SECURITY.md)
- [Changelog](../CHANGELOG.md)

## Architecture and direction

- [Executive summary](EXECUTIVE_SUMMARY.md) — dated maturity assessment and implementation overlays.
- [Architecture](ARCHITECTURE.md) — component design and dated implementation checkpoints.
- [Target architecture](TARGET_ARCHITECTURE.md) — desired end state and acceptance invariants.
- [Roadmap](ROADMAP.md) — planned work, historical checkpoints, and sequencing.
- [Gap register](GAP_REGISTER.md) — dated open risks, missing evidence, and ownership boundaries.

## Operations and governance

- [Deployment](DEPLOYMENT.md) — local, container, HTTP, PostgreSQL, and OpenShift operation.
- [Corpus governance](CORPUS_GOVERNANCE.md) — manifest, signature, freshness, publication, and rollback boundaries.
- [Document quality](DOCUMENT_QUALITY.md) — extraction and retrieval quality controls.
- [Document catalog](DOCUMENTS.md) — generated corpus reference.
- [Testing and evaluation](TESTING_AND_EVALUATION_STRATEGY.md) — test lanes, benchmarks, and evidence limits.
- [Recovery drills](RECOVERY_DRILLS.md) — backup/restore procedure and evidence expectations.
- [Legacy database upgrade](LEGACY_DATABASE_UPGRADE.md) — controlled adoption of supported pre-ledger databases.
- [Licensing and provenance](LICENSING_AND_PROVENANCE.md) — source, model, and artifact provenance.
- [Supply-chain evidence](../supply-chain/README.md) — artifact, scanner, and policy workflow.

Deployment-specific guides:

- [PostgreSQL identities and grants](../deploy/postgres/README.md)
- [OpenShift deployment](../deploy/openshift/README.md)
- [Open WebUI integration](../deploy/open-webui/README.md)

## Reviews, decisions, and evidence

These files are dated snapshots or records. Their findings do not override newer code or the current guidance above.

- [Repository review](REPOSITORY_REVIEW.md) — 2026 review plus implementation overlays.
- [Security review](SECURITY_REVIEW.md) — threat review and residual-risk record.
- [Bank migration security checklist](BANK_MIGRATION_SECURITY_CHECKLIST.md) — dated per-question assessment of the bank's 150-item pre-migration application-security questionnaire.
- [`decisions/`](decisions/) — architecture decisions and versioned policy contracts.
- [`evidence/`](evidence/) — retained local test and recovery evidence; not production acceptance.

## Documentation rules

- Add new durable guidance to the closest existing document before creating another top-level file.
- Put architecture decisions in `decisions/` and reproducible evidence in `evidence/`.
- Identify a review date or commit for claims that can become stale.
- Say whether evidence is local, synthetic, deployment-specific, or externally approved.
- Update this index whenever a top-level document is added, renamed, or retired.
