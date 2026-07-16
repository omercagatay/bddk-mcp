# Supply-chain evidence lane

This directory defines the repository-controlled part of release supply-chain
assurance. The separate
[`supply-chain.yml`](../.github/workflows/supply-chain.yml) workflow does not
start a runtime service, push an image, request an identity token, or use a
signing credential.

## What the lane proves

For every pull request, `main` push, release tag, or manual run, the workflow:

1. installs Buildx, Syft, Grype, and Gitleaks only from exact official release
   assets whose SHA-256 values are recorded in [`tools.json`](tools.json);
2. uses a digest-pinned BuildKit image and commit-pinned GitHub Actions;
3. scans the complete Git history for secrets with report redaction enabled;
4. builds the wheel and sdist twice with the source commit time as
   `SOURCE_DATE_EPOCH`, canonicalizes setuptools' generated sdist tar/gzip
   metadata, and rejects any byte or filename difference;
5. builds and locally loads both `Dockerfile` and `Dockerfile.spaces`, recording
   Buildx's exact manifest digest, config digest, descriptor, image name, and
   target platform metadata; before either build, a closed consistency check
   requires the declared embedding-model repository and Git commit to match
   both recipes and the runtime default configuration;
6. inspects each loaded image by immutable config ID, requires that ID in the
   raw Syft SBOM, verifies the Buildx descriptor against its manifest digest,
   and emits a CycloneDX JSON SBOM plus unsigned in-toto/SLSA v1 envelope;
7. refreshes Grype's vulnerability database, scans every SBOM with the
   non-suppressive [`grype.yaml`](grype.yaml) configuration, and rejects a
   database older than the limit in [`policy.json`](policy.json);
8. blocks every unexcepted High or Critical vulnerability and every
   unexcepted secret; and
9. uploads the artifacts, SBOMs, scanner reports, Buildx metadata, policy
   result, tool inventory, and a SHA-256 evidence manifest for review.

The workflow deliberately lets scanners finish and then applies one policy
gate. A Gitleaks exit code of `10` means findings were written to the redacted
report; any other non-zero exit is a scanner failure. Findings are never
treated as success: the later policy step rejects them.

## Determinism and time-dependent evidence

The wheel/sdist reproducibility check is byte-for-byte. Setuptools currently
stamps generated sdist directories independently of `SOURCE_DATE_EPOCH`, so a
bounded archive rewrite first sorts members, rejects links/traversal/duplicates,
and fixes owner, mode, timestamp, and gzip metadata without changing file
contents. The normalized sdist is then validated as a distribution. Canonical SBOMs replace
Syft's run timestamp and random serial number with the Git commit time and a
subject-digest-derived UUID, bind their root component to the artifact/image
digest, and sort unordered JSON collections. The generated unsigned SLSA
statements omit run-specific identifiers and are deterministic for the same
subject, source commit, and material files.

Syft does not catalog Python packages directly from a wheel file, so the lane
expands the locally built wheel with member-count, per-file, total-size,
encryption, link, duplicate, and traversal checks. Package SBOM scans disable
file catalogers, preventing temporary extraction paths from entering otherwise
deterministic package inventories.

Vulnerability results are intentionally time-dependent because a current
database is safer than a frozen database. Each Grype report records the tool
and database build metadata; policy rejects missing, malformed, future-dated,
or stale metadata. Raw Buildx result metadata is run evidence, while the
companion normalized, repository-generated SLSA envelope provides the stable
digest/material view. It is not a signed BuildKit attestation and is never
treated as promotion evidence by itself.
The container envelope records the pinned Hugging Face embedding-model Git
commit from [`model-assets.json`](model-assets.json) as an external material
and includes that manifest's own SHA-256. Its license state deliberately
remains `pending_bank_review`; a disconnected bank build must mirror and
approve the same immutable material rather than downloading an unreviewed
moving asset.

## Repository policy and exceptions

The repository default blocks High and Critical vulnerabilities, including
unfixed findings, and blocks all detected secrets. The evidence validator
requires the exact Grype version, a valid database status, `only-fixed: false`,
`only-notfixed: false`, no excluded packages or VEX input, and policy evaluation
of both ordinary and suppressed matches. High/Critical findings therefore
cannot escape through a scanner ignore rule. The checked-in scanner config has
no project ignore rules. It does not silently allowlist fixtures, paths,
packages, or vulnerability families.

An exception is an explicit code change to `policy.json`. Vulnerability
exceptions must match the report target, vulnerability ID, and package name;
secret exceptions must match the Gitleaks fingerprint. Every exception must
have a substantive reason, responsible owner, ISO expiry date, and
`pending_bank_release_review` approval state. Repository policy deliberately
rejects any field that claims bank release approval: that approval belongs to
the bank's external promotion control. Expired, duplicate, partial,
wildcard-like, or malformed entries fail closed. This repository policy is a
safe default, not a claim that the bank has approved its severity, expiry, or
evidence-retention values.

When an exact pending exception matches, the result records separate applied
vulnerability and secret counts, sets `external_approval_required=true`, keeps
`release_promotion_eligible=false`, and uses the explicit status
`repository_policy_passed_external_approvals_pending`. A zero-exception pass is
still `repository_policy_passed_unsigned_evidence` and is also not promotion
eligible. The bank promotion control must reject either result until it has
authenticated its own approval and signed the exact promoted digest.

The current secret exceptions are five exact fingerprints: two detector false
positives on typed Ed25519 private-key parameters, one password generated only
at runtime for a disposable PostgreSQL test role, and two deliberately
synthetic provider-token strings in the benchmark redaction unit test. They do
not allowlist a file, rule, token pattern, current tree, or future commits; they
expire on 2026-10-15 and remain pending bank release review. The owner field
assigns follow-up responsibility; it does not record approval.

## What remains bank-owned

This lane produces **unsigned** evidence and ephemeral local image builds. It
does not establish a promotion identity or prove that a deployed OpenShift
image is the image that was scanned. Before bank promotion, the platform and
security owners still need to select and validate:

- the internal source and image registries and immutable retention policy;
- the signing mechanism, protected key or keyless workload identity, signer
  authorization, certificate/trust roots, and incident revocation process;
- the admission/promotion policy that verifies repository, digest, signature,
  provenance predicate, SBOM, scanner age, and exception approvals;
- approved scanner mirrors and vulnerability feeds for disconnected or
  proxied OpenShift environments;
- evidence retention, vulnerability SLA, and exception approval roles; and
- a bank-controlled runner/base image. `ubuntu-24.04` is version-selected but
  GitHub updates its hosted runner image; the scanners, builder, actions, base
  images, and model revision used by this repository are pinned independently.

No key, token, certificate, registry credential, or assumed bank policy is
stored here. Signing and admission should be added only after those decisions
exist, and their acceptance test must verify the promoted digest rather than a
mutable tag.

## Pin maintenance

Pins originate from the projects' primary release pages:

- [Docker Buildx releases](https://github.com/docker/buildx/releases)
- [Moby BuildKit releases](https://github.com/moby/buildkit/releases)
- [Anchore Syft releases](https://github.com/anchore/syft/releases)
- [Anchore Grype releases](https://github.com/anchore/grype/releases)
- [Gitleaks releases](https://github.com/gitleaks/gitleaks/releases)

To update a tool, review the release notes and upstream provenance, replace the
exact versioned asset URL and SHA-256 in `tools.json`, update the matching
workflow version assertion when applicable, and run `tests/test_supply_chain.py`.
Never replace these pins with `latest`, a branch, a moving action tag, an
unverified install script, or a tag-only container reference.

The formats generated here follow the primary
[CycloneDX JSON specification](https://cyclonedx.org/specification/overview/),
[in-toto Statement v1 specification](https://github.com/in-toto/attestation/tree/main/spec/v1),
and [SLSA provenance v1 specification](https://slsa.dev/spec/v1.0/provenance).
