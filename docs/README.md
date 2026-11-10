# Documentation

This index is the canonical map of BDDK MCP documentation.

## Start here

- [Turkish overview for bank users](../README.md)
- [English overview for bank users](../README.en.md)
- [Current repository status](STATUS.md)
- [Repository structure](REPOSITORY_STRUCTURE.md)
- [Contributing](../CONTRIBUTING.md)
- [Security policy](../SECURITY.md)
- [Changelog](../CHANGELOG.md)

## Operations

- [Deployment](DEPLOYMENT.md) — local stdio, container, Streamable HTTP, corpus build and distribution, and the Railway preview profile.
- [Document quality](DOCUMENT_QUALITY.md) — extraction and retrieval quality controls.
- [Document catalog](DOCUMENTS.md) — generated corpus reference.
- [Licensing and provenance](LICENSING_AND_PROVENANCE.md) — source, model, and artifact provenance.

Deployment-specific guides:

- [Open WebUI integration](../deploy/open-webui/README.md)

## Archived material

Architecture checkpoints, roadmap, gap register, security review, testing strategy, corpus-governance guide, legal-answer readiness notes, document-repair receipts and recovery-drill records were removed from the tree on 2026-11-09 and remain in Git history at tag `archive/pre-simplify-2026-11-09`.

## Documentation rules

- Add new durable guidance to the closest existing document before creating another top-level file.
- Identify a review date or commit for claims that can become stale.
- Say whether evidence is local, synthetic, deployment-specific, or externally approved.
- Update this index whenever a top-level document is added, renamed, or retired.
