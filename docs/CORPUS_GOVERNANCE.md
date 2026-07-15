# Corpus Scope, Integrity, and Freshness

## What the manifest proves

`seed_data/corpus_scope.yml` is the version-1 declaration for the reviewed seed
bundle. It records the selection owner and purpose, included and excluded source
classes, known gaps, observation/build/review times, and the identity of every
seed artifact. `bddk_mcp/corpus_manifest.py` validates that closed schema and
binds it to each artifact's SHA-256 digest, byte size, JSON record count, and the
document extraction-time range.

This proves that the declaration and local files agree. It does **not** prove:

- exhaustive BDDK or Official Gazette coverage;
- legal publication, amendment, effective, repeal, supersession, or
  consolidation status;
- authority of a normalized Markdown extraction;
- freshness against an authoritative source at query time; or
- authorship, because the current self-checksum is not a digital signature.

Every structured regulatory retrieval response and its text fallback therefore
carry the same concise scope warning. The warning is part of the contract, not
an optional UI hint.

## Read-only verification

Run the verifier as an optional diagnostic before a bootstrap or benchmark:

```bash
uv run --frozen bddk-mcp verify-corpus --seed-dir seed_data
```

The command reads bounded regular files and emits only manifest identity and
warnings. It does not connect to PostgreSQL, repair data, or print corpus text.
A missing manifest, checksum/size/count mismatch, path escape, malformed JSON,
timestamp inconsistency, or stale quantified manifest exits nonzero.
Its success is not a capability token or trust handoff: a later process can see
different bytes, flags, environment, or trust material.

Production promotion must therefore apply the strict policy in the same
mutating bootstrap process that reads and imports the artifacts:

```bash
BDDK_INGESTION_DATABASE_URL='postgresql://INGESTION:SECRET@HOST:5432/DATABASE?sslmode=verify-full&sslrootcert=%2FAPPROVED%2Fpostgres-ca.crt' \
  uv run --frozen bddk-mcp bootstrap \
  --seed-dir /APPROVED/CORPUS \
  --reindex-existing \
  --require-quantified-freshness \
  --require-measured-freshness \
  --require-verified-signature \
  --trusted-signing-key /APPROVED/TRUST/corpus-signing-public-key.pem
```

For the OpenShift target, the reviewed
`deploy/openshift-overlays/bank-bootstrap` contract invokes that same strict
bootstrap command. It mounts PVC `bddk-mcp-approved-corpus` at
`/var/run/bddk-mcp/corpus` read-only and mounts Secret
`bddk-mcp-corpus-trust` separately at
`/var/run/secrets/bddk-mcp/corpus-trust` read-only, using key
`ed25519-public-key.pem`. Repository preflight renders and exact-inventory-checks
these arguments and sources. It does not provision them, execute the Job, or
approve their bank custody.

The checked-in manifest intentionally fails those additional policy gates:
the owner's “immediate” expectation has not been translated into numeric source
detection, publication, and maximum-age objectives, and no bank-approved
signature mechanism has been selected. Measured freshness additionally requires
per-document authoritative publication, source-detection, download, extraction,
and retrieval-publication timestamps whose observed lags satisfy those numeric
objectives. A future `verified` declaration must
carry a detached Ed25519 signature that validates against this separately
provisioned trust anchor; a self-declared status is rejected. Do not weaken the flags to make a bank
promotion pass; decide and record those controls first.

## Bootstrap and benchmark behavior

A non-empty `bddk-mcp bootstrap` verifies the exact `corpus_scope.yml` and all
declared artifacts before creating a database pool. It loads the documents and
decision cache only from the paths assigned those roles by the manifest, reads
only the declared bounded byte count, and rechecks each hash after validation.
It rejects a present `documents.json`, `chunks.json`, or `decision_cache.json`
when that reserved filename is not declared, closing fallback-filename bypasses.
Production supplies the detached-signature trust key as a separately mounted
file, never as part of the corpus tree or repository.

Successful bootstrap output records a path-free manifest ID, manifest SHA-256,
and scope warnings for operator evidence. That identity is **not yet persisted
in PostgreSQL**, so the output must be retained with the release evidence and
cannot be inferred later from database state alone. Derived sections, chunks,
and embeddings are still regenerated from canonical reviewed document text; a
manifest does not make committed derived rows executable or authoritative.

Phase-2 evaluation verifies the same manifest before opening MCP or model
connections. Its audit metadata records a path-free manifest ID/hash, an
artifact-set hash, build/review times, signature state, and warnings. Set
`BDDK_CORPUS_MANIFEST_PATH` to the reviewed declaration that actually describes
the tested deployment. For a signed declaration, separately set
`BDDK_CORPUS_TRUSTED_SIGNING_KEY` to the approved PEM public-key mount. A
benchmark against a different remote corpus without a
matching declaration is not comparable evidence.

## Reviewed update procedure

1. Acquire and normalize sources through the bounded ingestion path.
2. Review selection scope, known gaps, extraction quality, and source rights.
3. Produce the three JSON artifacts deterministically.
4. Recalculate artifact SHA-256 values, sizes, counts, and observed timestamps.
5. Update `manifest_id`, purpose/scope, and `scope_reviewed_at`; never copy a
   prior checksum into a changed declaration.
6. Recalculate the canonical manifest checksum with
   `canonical_manifest_sha256`; optionally run `bddk-mcp verify-corpus` as a
   read-only diagnostic.
7. Run bootstrap against a disposable database with the same strict
   quantified/measured/signature flags and separately mounted trust key that
   production will use. Retain its path-free manifest ID/SHA output, then run
   exact-section, retrieval-publication, and benchmark regression gates.
8. Once signing is designed, verify the external signature before changing
   `signature_status` to `verified`; the string in the manifest alone is not a
   signature verifier.

Retain the prior manifest and artifacts under the bank's immutable release and
retention controls. The repository does not yet implement whole-corpus
generation activation or rollback, so a validated manifest must not be
described as an atomic corpus-release mechanism.

## Evaluation-release trust boundary

R09 deliberately does not let one checksum or one reviewer establish release
evidence. A release-ready expert evaluation must receive all three of these
separately verified inputs:

1. this corpus manifest, detached-Ed25519 signed against a separately mounted
   trust anchor, with all three numeric objectives and
   `slo_evidence_status: measured` backed by the per-document event chain;
2. the exact expert dataset, independently detached-Ed25519 signed and bound to
   this manifest and its artifact hashes; and
3. an exact export of `public.regulatory_validated_section_citations`, plus a
   detached legal-curator attestation binding the pack hash and complete sorted
   Citation ID inventory to a separately supplied curator trust key.

The validated Citation objects in the dataset must equal the exported objects,
not merely share IDs. Reusing the expert-dataset signing key as the legal
curator key fails release validation. The tracked manifest and 20-case dataset
do not satisfy these gates; they remain governance fixtures, not model-release
or legal-currentness evidence.
