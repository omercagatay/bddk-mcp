# Security Policy

## Supported code

Security fixes target the current `main` branch. Historical tags and closed branches are not supported release lines. The package metadata may identify an unreleased target version; consult `CHANGELOG.md` and the GitHub release page before treating a tag as a published release.

This repository is an engineering beta. Its application controls do not by themselves establish production, bank, regulatory, or legal acceptance. Deployment-specific identity, TLS, network, database, backup, and operational controls remain the deployer's responsibility.

## Report a vulnerability privately

Use [GitHub's private vulnerability reporting form](https://github.com/omercagatay/bddk-mcp/security/advisories/new).

Do not open a public issue for:

- authentication or authorization bypasses;
- exposed credentials, tokens, keys, database access, or private endpoints;
- cross-tenant, corpus, or document disclosure;
- injection, SSRF, unsafe parsing, or sandbox escape paths;
- a supply-chain finding that is not already public;
- a vulnerability in a deployed BDDK MCP instance.

Include, when available:

- the affected commit, version, component, and deployment mode;
- a minimal reproduction or proof of concept;
- impact and prerequisites;
- sanitized logs or requests;
- a suggested remediation or disclosure constraint.

Never send live credentials, private keys, customer data, or restricted regulatory documents. Replace sensitive values with deterministic placeholders and state what was removed.

## Coordinated handling

The maintainer will assess the report, reproduce it where possible, and coordinate a fix and disclosure plan through the private advisory. Response and remediation timing depend on severity, reproducibility, upstream dependencies, and deployment ownership; no fixed service-level commitment is implied by this public repository.

If the issue belongs to an upstream dependency, deployment platform, or private installation, the maintainer may ask you to coordinate with that owner while keeping the repository advisory private.

## Public hardening information

- [Security review](docs/SECURITY_REVIEW.md) — a dated repository assessment, with explicit residual risks.
- [Deployment guide](docs/DEPLOYMENT.md) — supported process profiles and fail-closed remote configuration.
- [Supply-chain policy](supply-chain/README.md) — scanner evidence, exact exceptions, and release-eligibility boundaries.
- [Corpus governance](docs/CORPUS_GOVERNANCE.md) — corpus identity, signing, freshness, and publication controls.
