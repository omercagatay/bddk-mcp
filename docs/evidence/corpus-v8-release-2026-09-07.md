# Corpus v8 refresh — 2026-09-07

## Scope and authority

Owner-delegated technical refresh following PR #151 (`fe8d72b`). This approves
an existing, non-exhaustive stored snapshot for the governed release workflow;
it is **not independent human source validation, legal-currentness certification,
expert-dataset approval, or bank acceptance**. Freshness remains `not_measured`.
Canonical document and decision-cache bytes, source observation/build timestamps,
and the project trust anchor are unchanged. No authoritative acquisitions, legal
mappings, or measured freshness events were invented.

## Artifact identities

| Item | Value |
|---|---|
| Manifest | `bddk-job-corpus-2026-09-07` |
| Canonical manifest SHA-256 | `94fc2737cc673a811a36b23eb7f149b4bb7d1835e54ad4a49ee574fd6cb7fdee` |
| Documents | 318; `4a469fc11bd9848d2acd540296d794126ab9c506dc68d929fb72f580ebe5dba6` |
| Chunks | 13,240; 19,157,379 bytes; `949dcaa12b8377beb7bca364f8aa614ddbdd2b984be4082948a3bb62b68456d9` |
| Decision cache | 318; `9e5677b1a2ffc9fe9d5ed122a90171c896e5f14ee83d958b724e1d2b8b8fd894` |
| Retrieval profile | `bc5acd279fbb521ff5eb45650e7382cdc9b090becd338c20cf7e3fb56d1020e8` |

The existing Ed25519 key, held outside Git, signed the declaration after technical
review. Verification used the unchanged, separately supplied project trust anchor.
A first preparation attempt incorrectly advanced `corpus_built_at` to chunk
regeneration time; the verifier rejected it because the documents artifact fixes
that timestamp. The correction preserves the original canonical build time,
2026-05-12, rather than relabelling old source text as newly acquired.

## Independent technical review

Automated read-only reviewer `403e8fdd-a0b2-4a4` independently regenerated all
13,240 chunks, matching every field and serialized byte. It checked unique,
contiguous inventories, canonical source spans/hashes, and coverage of every
non-whitespace source character. Its regression selection passed **68 tests,
8 deselected**. This reviewer did not access production or verify embeddings.
The parent separately compared all production chunk fields, section identities,
and document hashes in a read-only repeatable-read transaction: exact match.

Historical reproduction with the same tokenizer/chunker explains the delta:

| Parser | Chunks | Difference |
|---|---:|---|
| v5 (`99e13ec`) | 10,483 | Exactly reproduces previous signed artifact |
| v7 (`c4e6314`) | 12,925 | All 318 inventories differ from v5 |
| v8 (`fe8d72b`) | 13,240 | Only 1282, 1291, 934, 935, 943, 947 differ from v7 |

Retained limitations: **18 capped sections** keep explicit truncation notices;
**2,608 govde chunks** use generated window identities, not printed provisions;
**2,012 chunks** have no section identity. Existing formula and extraction-quality
warnings remain. Whole-source coverage does not prove any one section is complete
or semantically faithful.

## Draft evaluation bindings

The unapproved pilot moves to `0.1.0-draft.3`: corpus identities and checksum are
refreshed, including the five abstention-case corpus bindings. Eleven existing
draft section hashes change because trailing-heading trimming shortened their
spans (mevzuat_22599 articles 4/9/10; mevzuat_21192 articles 4/9/25;
mevzuat_21194 articles 4/5/9; mevzuat_42628 articles 4/10). All new hashes reproduce
from canonical source slices. Some removed tails include pre-existing disordered
extraction fragments, retained elsewhere in the complete chunk inventory; this
is **not** a legal validation of those sections.

Queries, proposed answers, evidence document/ref identities, legal-currentness
flags, pending mappings, annotations, adjudications, and owner approvals remain
unchanged. All 20 cases remain draft; all 40 annotations and 20 adjudications remain
pending. No dataset or legal-curator signature was supplied. Test-only evidence
clocks moved beyond the new corpus review while preserving temporal ordering;
**45 expert-evaluation/preflight tests pass**, including fail-closed release gates.

## Verification and activation

The initial production state had no active release. Old signed artifacts were
validly signed but the official staging CLI, in the matching runtime image,
refused their exact chunk/profile mismatch before staging any request.

Verification runtime built from a clean `git archive fe8d72b`:

- Source archive SHA-256:
  `dbcf53e16132031ac889f2b885c088c02ca9d264ef9a60299cff813aaecad6a8`.
- Immutable local verifier image ID:
  `sha256:aff763394c9067356f5b6e1031e6f7d36a4c596a164d142508e13d36e3ca64e2`.
- Isolated publisher image ID:
  `sha256:26b86ada0142d1d6da89e8a2b5846e56423bff61a1d1af268b999e8a2fa96958`.

These are actual local Docker image identities, not claimed registry manifests.
The verifier computes the production retrieval profile without overrides. The
publisher image contains no corpus, trust anchor, or embedding model. Separate
restricted verifier/publisher LOGINs passed application role/catalog checks;
each process receives only its own DSN. Connections to production use verify-full
TLS. This is technical role separation, not evidence of independent bank custody.

The production bootstrap engine passed against a newly created disposable,
loopback-only clone of the PostgreSQL test database: **318 documents, 8,389
sections, 13,240 chunks and 13,240 freshly computed embeddings**. Both strict
quantified-freshness and signature flags were enabled. A second invocation
proved no-op behavior (318 current publications, zero writes/reindex publications).
The harness supplied the fixture pool directly; production LOGIN checks are
separately exercised by the verifier/publisher and PostgreSQL contract tests.

Fresh regression checks: **1,731 DB-less tests passed, 38 skipped**; **222
PostgreSQL tests passed, 5 skipped**; the separate ranking guard passed **3 tests**
covering ten hit@1 queries and eight mechanical ranking cases. Ruff, formatting,
lock consistency, and diff checks passed. GPU tests were excluded.

**Execution checkpoint:** production staging, activation, strict-serving
configuration, and final live verification are not yet complete. No active-release
readiness is claimed at this checkpoint.

## Deployment contract

`railway.toml` now requires quantified freshness and verified signatures in
bootstrap, ensuring exact chunk/profile matching rather than warning-only import.
It explicitly describes bootstrap as non-publishing. The contract regression
failed before the change; all **11 admin-configuration tests** passed afterward.
Strict serving must be enabled only after successful governed activation.
