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
- official source authorship: the owner's signature authenticates the corpus
  declaration, not the legal authority of the extracted text.

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

Production import must therefore apply the strict policy in the same mutating
bootstrap process that reads and imports the artifacts:

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

Strict bootstrap is intentionally not the release authority. It returns
`release_publication_required` after import/reindex. Schema v8 then separates
independent verification from activation. A `bddk_release_verifier` identity
must revalidate the same mounted artifacts and trust key, regenerate every
deterministic chunk and embedding, prove exact database membership, and stage a
short-lived request without activating it:

```bash
BDDK_RELEASE_VERIFIER_DATABASE_URL='postgresql://VERIFIER:SECRET@HOST:5432/DATABASE?sslmode=verify-full&sslrootcert=%2FAPPROVED%2Fpostgres-ca.crt' \
BDDK_RELEASE_VERIFIER_REVISION_SHA256='REPLACE_64_LOWERCASE_HEX_REVISION' \
BDDK_RELEASE_VERIFIER_IMAGE_DIGEST='sha256:REPLACE_64_LOWERCASE_HEX_IMAGE_DIGEST' \
BDDK_RELEASE_VERIFICATION_VALIDITY_SECONDS=900 \
  uv run --frozen bddk-mcp verify-and-stage-corpus-release \
    --seed-dir /APPROVED/CORPUS \
    --trusted-signing-key /APPROVED/TRUST/corpus-signing-public-key.pem
```

The verifier outputs a `corpus_release_request_sha256_...` identity together
with `verification_run_sha256` and the resulting
`verification_evidence_sha256`. Recomputing the canonical receipt requires the
reviewed manifest, exact detached-signature SHA, retrieval-profile SHA, exact
verifier revision/image provenance, `verification_run_sha256`, and a governed
export of the append-only staged-request evidence. Retain that complete set
without corpus text in logs; the path-free CLI summary alone is not a complete
audit export. Its
append-only database evidence binds the exact manifest/signature/signer,
freshness policy, retrieval profile, database-computed corpus state and epoch,
per-run verification evidence, verifier LOGIN fingerprint, source revision,
immutable image digest, staging time, and expiry. Revision is exactly 64
lowercase hexadecimal characters; image provenance is a `sha256:` digest. The
validity is accepted only from 60 through 3,600 seconds and defaults to 900.
Treat expiry as a bound on verification freshness, not a scheduling target: if
there is not enough time for reviewed handoff, rerun verification rather than
extending or editing a staged row.

Pass only the exact request ID to a separate `bddk_release_publisher` identity:

```bash
BDDK_RELEASE_PUBLISHER_DATABASE_URL='postgresql://PUBLISHER:SECRET@HOST:5432/DATABASE?sslmode=verify-full&sslrootcert=%2FAPPROVED%2Fpostgres-ca.crt' \
  uv run --frozen bddk-mcp activate-corpus-release \
    --request-id corpus_release_request_sha256_REPLACE_64_LOWERCASE_HEX
```

The publisher process must not receive the corpus PVC/directory, manifest,
detached signature, trust key/Secret, verifier DSN, or verifier role. Its only
release-selection input is the request ID. Activation atomically rejects an
unavailable, expired, previously used, wrong-epoch, changed-state, or non-ready
request before writing the release, activation, and request binding. The old
`publish-corpus-release` CLI always fails closed and is not a fallback.

For the OpenShift target, the reviewed
`deploy/openshift-overlays/bank-bootstrap` contract invokes that same strict
bootstrap command. It mounts PVC `bddk-mcp-approved-corpus` at
`/var/run/bddk-mcp/corpus` read-only and mounts Secret
`bddk-mcp-corpus-trust` separately at
`/var/run/secrets/bddk-mcp/corpus-trust` read-only, using key
`ed25519-public-key.pem`. Repository preflight renders and exact-inventory-checks
these arguments and sources. It does not provision them, execute the Job, or
approve their bank custody.

The same overlay includes `verify-stage-release.yaml` and
`activate-release.yaml`. The verifier Job uses ServiceAccount
`bddk-mcp-release-verifier`, Secret `bddk-mcp-release-verifier-db`, the approved
corpus PVC, and the corpus-trust Secret. The activation Job uses ServiceAccount
`bddk-mcp-release-publisher`, Secret `bddk-mcp-release-publisher-db`, and
`BDDK_RELEASE_REQUEST_ID`; it mounts only PostgreSQL CA and bounded temporary
storage, never corpus or trust material. Repository rendering is still not
evidence that the four Jobs ran in the required `migrate` → `bootstrap` →
verify-and-stage → activate order in a bank namespace.

The checked-in manifest (`bddk-job-corpus-quality-repairs-local-v6-2026-09-21`, reviewed 2026-09-21)
declares owner-quantified objectives (7-day source detection, 14-day
publication, 180-day maximum manifest age) and is Ed25519-signed; the detached
signature `corpus_scope.sig` validates against the project trust anchor
`deploy/trust/corpus-signing-public-key.pem`, whose private key is held outside
Git by the selection owner. A bank deployment must still decide whether that
project key or a bank-issued key is the promoted trust anchor. Measured
freshness additionally requires per-document authoritative publication,
source-detection, download, extraction, and retrieval-publication timestamps
whose observed lags satisfy those numeric objectives. The corpus is a batch
snapshot with no live monitoring pipeline, so `slo_evidence_status` remains
honestly `not_measured`.

Schema v10 makes that an explicitly governable state rather than an
unpublishable one. The release ledger admits exactly
`quantified_measured_signature_verified_pass` and
`quantified_unmeasured_signature_verified_pass`; both require quantified
objectives and a verified signature. The verifier derives the level from
manifest evidence, so `--accept-unmeasured-freshness` permits the weaker level
without ever relabelling an unmeasured corpus as measured, and the level is
fingerprinted into the release and request identities so the two can never be
substituted. An unmeasured corpus also declares exactly the storage fields:
the per-document event columns are required only where freshness is genuinely
measured, and either way the artifact field set is exact. Never fabricate event
timestamps to reach the measured level; build the measurement pipeline
instead.

The historical 2026-09-07 refresh regenerated **13,240 chunks from the same 318 documents**
under section parser v8, section search v6, and quality policy v3. The previous
signed v5 artifact contained 10,483 chunks; all 318 inventories change because
v7's remainder indexing and trailing-heading trimming were also not yet in the
signed artifact. The incremental v7-to-v8 parser delta affects only six documents.
Independent automated technical review reproduced the exact candidate bytes and
verified complete source-character coverage; owner-delegated signing does not
claim independent human source validation. Source observation and canonical
corpus-build timestamps remain unchanged. See the
[refresh evidence](evidence/corpus-v8-release-2026-09-07.md) for exceptions,
verification receipts, deployment state, and retained legal/freshness gaps.

The 2026-09-21 quality repair includes 50 changed documents, **13,544 chunks**,
and quality policy v5. The two native-source rebuilds have genuine new acquisition
and extraction timestamps; older documents retain their original timestamps.
All 11 historical extraction failures are retired only with this signed corpus.
Local verification regenerated every embedding, checked exact membership in a
fresh staging database, and activated through the separate publisher role.
The [repair receipt](evidence/document-repairs/local-release-v6.json) records
318 real MCP document checks. Its `local-v6` identity labels this reviewed
snapshot, not a production deployment or independent legal approval. Existing
formula-unaware provenance, source-scope and freshness cautions remain.

Any future chunker or profile change reopens this review: regenerate, record and
independently review the delta, then update and re-sign the manifest; a checksum
edit alone is not review evidence. Bootstrap still does not activate a release.

## Bootstrap and benchmark behavior

A non-empty `bddk-mcp bootstrap` verifies the exact `corpus_scope.yml` and all
declared artifacts before creating a database pool. It loads the documents and
decision cache only from the paths assigned those roles by the manifest, reads
only the declared bounded byte count, and rechecks each hash after validation.
It rejects a present `documents.json`, `chunks.json`, or `decision_cache.json`
when that reserved filename is not declared, closing fallback-filename bypasses.
Production supplies the detached-signature trust key as a separately mounted
file, never as part of the corpus tree or repository. Verification rejects both
a supplied path and a resolved path inside the corpus root, so a symlink from
the corpus mount to an external key is not a valid separation boundary.

Successful bootstrap output records a path-free manifest ID, manifest SHA-256,
scope warnings, and the need for independent publication. A successful v8
verifier run first persists the append-only staged request and returns its
path-free request/release/state identity, expiry, and bounded verification
summary. A successful
`activate-corpus-release` then uses that request to append the v5 release and
activation plus the v8 one-time request binding. The persisted evidence includes
manifest/signature/signer and verifier provenance, numeric freshness policy,
retrieval-profile hash, exact corpus-state fingerprint, corpus epoch, activation
sequence, and hashed verifier/publisher database actors
(**bddk_mcp/corpus_publication.py;
bddk_mcp/migrations/v0005_corpus_release_publication.py;
bddk_mcp/migrations/v0008_staged_corpus_releases.py**). Neither role can read
the request base tables directly; each can invoke only its own security-definer
facade. Any tracked corpus mutation advances the epoch and makes the active view
unavailable until a new strict verification and activation. The public MCP
resource `bddk://corpus/active-release` exposes only the path-free release,
manifest, and retrieval-profile identity.

Derived sections, chunks, and embeddings are still regenerated from canonical
reviewed document text. Persistence proves which verified state was activated;
it does not make unchecked derived rows or the source declaration authoritative.

Phase-2 evaluation verifies the selected manifest before model execution, then
reads `bddk://corpus/active-release` through the evaluated MCP session. It
requires exact manifest ID/SHA equality, performs all calls on that same session,
and re-reads the resource afterward; an unavailable, malformed, mismatched, or
changed release fails the run (**benchmark/phase2_e2e.py:304-350,531-568,657-668;
tests/test_benchmark_phase2.py:438-493**). Its audit metadata records both the
validated local declaration and active release. Set `BDDK_CORPUS_MANIFEST_PATH`
to the declaration that actually describes the deployment and, for a signed
declaration, set `BDDK_CORPUS_TRUSTED_SIGNING_KEY` to the separately mounted PEM
public key. `BDDK_BENCHMARK_CORPUS_ID` and `BDDK_BENCHMARK_CORPUS_SHA256` remain
labels, not the enforced release binding.

## Admin editorial drafts and document signatures

The admin console (`uv run bddk-mcp admin-ui`) can edit **unpublished editorial
drafts**, never the public PostgreSQL corpus. It retains the public-reader
LOGIN and does not need ingestion, schema-owner, verifier, or publisher grants.
Only admin detail/edit pages overlay drafts; list/search, public retrieval,
chunks, vectors, release state and corpus epoch remain unchanged.

Editing is disabled until `BDDK_ADMIN_DRAFT_DB` names a persistent SQLite file.
Create its private parent directory beforehand; place it outside `BDDK_SEED_DIR`
(the approved corpus directory, or the checkout seed directory by default).
Use a private, backed-up local filesystem volume, not an ephemeral container
layer or network-shared SQLite volume. The file is created owner-only (0600);
an existing more-permissive file is rejected. SQLite transactions serialize
concurrent saves with a five-second lock timeout. No PostgreSQL migration or
public-role grant expansion is involved.

Optional **document signing** configuration (all paths are server-side):

```bash
BDDK_DATABASE_URL='postgresql://PUBLIC_READER:SECRET@HOST:5432/DATABASE?sslmode=verify-full&sslrootcert=%2FAPPROVED%2Fpostgres-ca.crt' \
BDDK_SEED_DIR=/APPROVED/CORPUS \
BDDK_ADMIN_DRAFT_DB=/PRIVATE/EDITORIAL/drafts.sqlite \
BDDK_ADMIN_SIGNING_KEY=/PRIVATE/EDITORIAL-KEYS/ed25519-private.pem \
BDDK_ADMIN_SIGNING_PUBLIC_KEY=/APPROVED/EDITORIAL-TRUST/ed25519-public.pem \
  uv run bddk-mcp admin-ui
```

Provision an editorial Ed25519 PKCS8 PEM private key under separate operator
custody (0600), and independently configure its trusted SubjectPublicKeyInfo PEM
public key. Both files must be outside the corpus and the entire draft-storage
directory. The server checks their correspondence at startup and refuses unsafe,
oversized, non-Ed25519 or mismatched keys; keys are never accepted from forms or
printed. Use a distinct editorial key, not the corpus owner's release key.
Omit both signing variables to retain editing/download with signing clearly
disabled. Supplying only one key fails startup. Key rotation requires a restart;
retain independently trusted historical public keys for previously downloaded
artifacts. Embedded public keys are verification material, **not trust anchors**.

Loopback uses host checking; remote access additionally requires the existing
operator JWT policy (`bddk.operator` and every configured required scope), exact
allowed Host and HTTPS Origin, and `BDDK_ADMIN_REMOTE_ENABLED=true` (see
[deployment](DEPLOYMENT.md)). Both loopback and remote edit/sign POSTs require
an exact same-origin `Origin` header **and** a one-hour, document/browser/operator-
identity-bound CSRF token. Missing/foreign/null origins, absent tokens and cross-
site requests are rejected, including bearer clients. Forms are URL-encoded,
bounded to 12,100,000 wire bytes and a ten-second read deadline. Proxies must
preserve the approved external host/scheme using only trusted proxy forwarding;
an origin mismatch must be fixed in proxy configuration, not bypassed. Restart
invalidates existing form tokens; reload forms. Draft persistence is independent
of tokens. Use one admin process/replica: tokens are process-local (multiple
workers require sticky routing or a separately designed shared token secret).

User workflow:

1. Open a document, choose **Duzenle**, edit and **Taslagi kaydet**. Title is
   limited to 500 characters, content to 1,000,000 nonblank characters; category,
   decision date and number are bounded to 100/32/100 characters. Source URL
   must be absolute HTTPS on the exact BDDK/mevzuat hosts (including reviewed
   `www` aliases), default port only, without credentials. Unknown, duplicate or
   provenance fields are rejected. `extraction_method` cannot be edited; original
   extraction metadata is retained and every manual edit is explicitly unverified.
2. The form binds the original admin-visible canonical content/metadata and PDF
   digest plus an optimistic draft revision. A stale revision/base returns 409
   without overwriting anything. After concurrent editing, reload and reapply
   deliberately. If canonical data changed, download and reconcile offline;
   there is no automatic rebase or force-save button. Base comparison is a
   point-in-time read, not a lock on future corpus publications; operator review
   must recheck the base before applying the artifact.
3. On the detail page choose **Sign saved draft — Imzala**. It signs exactly the
   saved revision, all edited fields, original snapshot, base fingerprint and
   unverified/unpublished markers using Ed25519. Unsaved text is never signed.
   Every subsequent save increments revision and removes the stored signature,
   including a save with identical text. A stale sign cannot sign a newer edit.
4. Download `editorial-draft.json` (also available unsigned). Its `payload` and
   `signature` are canonical JSON; signature is over UTF-8 bytes of
   `json.dumps(payload, ensure_ascii=False, sort_keys=True,
   separators=(",", ":"), allow_nan=False)`. The signature object carries
   `algorithm`, `signature_base64`, `public_key_pem` and the SHA-256 fingerprint
   of raw Ed25519 public bytes. Verify with the **independently trusted** key:

   ```python
   import base64, json
   from pathlib import Path
   from cryptography.hazmat.primitives.serialization import load_pem_public_key

   artifact = json.loads(Path("editorial-draft.json").read_text())
   trusted = load_pem_public_key(Path("approved-editorial-public.pem").read_bytes())
   payload = json.dumps(
       artifact["payload"], ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
   ).encode("utf-8")
   trusted.verify(base64.b64decode(artifact["signature"]["signature_base64"], validate=True), payload)
   ```

These are **editorial artifacts, not deployable releases, regulator signatures,
source validation, or corpus-manifest signatures**. There is no auto-activation,
seed export of drafts, or UI publication endpoint. The SQLite file retains the
latest draft only, not an append-only audit history; download each reviewed
revision into operator-controlled retention before further editing. Loss of a
sidecar backup loses drafts but cannot alter the published corpus.

Publication handoff: independently verify the downloaded signature and expected
revision/base; review edited text against authoritative sources (especially
formulas) and preserve review evidence. Apply only approved changes through the
isolated ingestion/build workflow with original metadata preserved and genuine
extraction/review provenance. Regenerate sections, chunks and embeddings, run
quality/retrieval tests, review the artifact delta, then follow the complete
owner corpus-manifest signing → bootstrap → independent verify-and-stage →
separate activate sequence below. Editorial signing never replaces any step.

## Reviewed update procedure

1. Acquire and normalize sources through the bounded ingestion path.
2. Review selection scope, known gaps, extraction quality, and source rights.
3. Produce the three JSON artifacts deterministically; compare the complete
   chunk artifact to regeneration under the pinned retrieval profile.
4. Recalculate artifact SHA-256 values, sizes, counts, and observed timestamps,
   and retain independent review of every intentional derived-artifact delta.
5. Update `manifest_id`, purpose/scope, and `scope_reviewed_at`; never copy a
   prior checksum into a changed declaration.
6. Recalculate the canonical manifest checksum with
   `canonical_manifest_sha256`; optionally run `bddk-mcp verify-corpus` as a
   read-only diagnostic.
7. Obtain the detached signature through the approved process and verify it
   before changing `signature_status` to `verified`; the string in the manifest
   alone is not a signature verifier.
8. Run bootstrap against a disposable database with the same strict
   quantified/measured/signature flags and separately mounted trust key that
   production will use. Retain its path-free manifest ID/SHA output, then run
   exact-section, retrieval-publication, and benchmark regression gates.
9. With the distinct release-verifier identity and only its verifier DSN,
   corpus/trust mounts, revision SHA-256, immutable image digest, and reviewed
   TTL, run `verify-and-stage-corpus-release`. Retain the request ID, staged and
   expiry timestamps, corpus state/epoch, and path-free verification evidence.
   Do not pass publisher credentials into this process.
10. Before expiry, pass only that exact request ID to
    `activate-corpus-release` through the separate release-publisher identity.
    Do not give the publisher corpus/trust material or verifier credentials.
    Verify the persisted active identity and read it through
    `bddk://corpus/active-release` before permitting strict serving. An expired
    or changed-state request requires a new verifier run, not a retry with
    weakened evidence.
11. If the active release must be retained as an immutable database recovery
    target, run `bddk-mcp retain-corpus-generation --expected-release-id ...`
    through that same publisher identity and retain its content-free receipt.
    The command is administrative CLI only; it is not an MCP tool and does not
    change the active release. A new governed release over an already retained
    exact corpus state/retrieval profile receives its own binding to the existing
    physical generation and seal; it does not create a duplicate copy. The CLI
    bounds its transaction with `lock_timeout=30s` and
    `statement_timeout=30min`; a timeout leaves no partial generation, seal, or
    binding. Resolve contention and confirm the expected release is still active
    before a reviewed retry.

V7 fixes retained-row and both current/retained state hashing to function-local
`TimeZone=UTC`, `DateStyle=ISO, YMD`, `IntervalStyle=postgres`,
`bytea_output=hex`, and `extra_float_digits=3`. Historical release rows are
not rewritten. Before installing v7, the migration recomputes any active v5/v6
release under those canonical settings and refuses the upgrade if it differs.
Preserve that historical evidence; on the unchanged pre-v7 schema (v5 or v6),
independently review/revalidate the exact corpus and use only the separately
approved exact-schema publication-remediation procedure to publish and activate
a new release under the canonical settings. Then retry v7 and continue through
v8. Never manufacture a binding, update the old release hash, or admit serving
or retention during remediation.

V11 is the current schema and ordinary workload admission is v11-only.
Migration `v0011_graph_corpus_state` adds `regulatory_relations` to the mutation
epoch, release fingerprint, verifier/activation locks, and typed retention.
It invalidates every earlier activation and staged request, including an empty
graph: after migrating and applying `deploy/postgres/02_grants.sql`, verify and
stage the corpus again, then activate the new request before resuming strict
serving. The seed artifact format is unchanged.

New retained generations use schema 2 with 18 relations. Existing schema-1
seals remain immutable, reproduce their original fingerprints, and retain their
17-relation inventories; they provide no historical graph evidence.

The v8 additive migration preserves existing v5/v7 release and retention evidence,
creates append-only request/binding relations and two role-separated facades,
and revokes every non-owner grant on the old direct-publication routine. Apply
`deploy/postgres/02_grants.sql` after migration. The code retains exact
v5/v6/v7 publisher catalog/identity checks solely for reviewed migration
remediation, but the current `publish-corpus-release` CLI is disabled and the
direct routine must never be re-granted on v8. A v7 database is therefore an
upgrade source, not a steady-state target for the four-job release flow.

Retain the prior manifest and artifacts under the bank's immutable release and
retention controls. Schema v7 can copy and seal the exact active v5 state across
17 typed retained relations, but serving remains bound to the mutable v5
tables and no tested activation/reactivation workflow exists. A sealed
generation is therefore a rollback target, not product rollback or historical
source-evidence recovery. Legacy v5 releases without a v7 binding remain
`legacy_v5_unretained`.

The retention receipt reports reconciled database storage and, when available,
a non-exclusive observed cluster WAL interval; WAL remains `not_measured` if
that best-effort observation is unavailable. The baseline LSN is attempted in a
savepoint so a measurement-permission/catalog failure does not poison the
retention transaction; the endpoint is best-effort after commit, and unrelated
cluster activity can contribute. Backup growth remains `not_measured` until a
controlled bank/DBA backup, and the bank must approve generation count and
capacity. Count unique state/profile generations for physical storage, while
retaining every governed release binding for traceability. Only fields already
present in PostgreSQL are copied; absent external
authoritative source files and historical legal-release packs are not acquired
by v7 retention.

## Evaluation-release trust boundary

R09 deliberately does not let one checksum or one reviewer establish release
evidence. A release-ready expert evaluation must receive all four of these
separately verified, signed layers:

1. this corpus manifest, detached-Ed25519 signed against a separately mounted
   trust anchor, with all three numeric objectives and
   `slo_evidence_status: measured` backed by the per-document event chain;
2. the canonical expert-dataset payload, independently detached-Ed25519 signed and bound to
   this manifest and its artifact hashes; and
3. an exact export of `public.regulatory_validated_section_citations`, plus a
   detached legal-curator attestation binding the pack hash and complete sorted
   Citation ID inventory to a separately supplied curator trust key; and
4. a separately signed legal-release checkpoint binding that corpus and legal
   pack to retained source bytes, acquisition records, reviewed page text and
   mappings, exact Citation excerpts, and its predecessor checkpoint chain.

The validated Citation objects in the dataset must equal the current exported
objects, not merely share IDs. Canonical Ed25519 fingerprints—not PEM file
bytes—identify signers, and corpus, dataset, curator, and legal-release signers
must all be different. Every historical legal-release signer must also remain
separate from the other three operational roles.

The preflight has two explicit trust modes. Default `development` mode accepts
operator-supplied operational keys and a separately supplied latest-checkpoint
hash; it is a fixture/consistency boundary only. Supplying a signed policy in
development still does not pin its current head or independently compare its
organization/environment/scope. `bank-policy` mode instead
requires an exact detached-signed policy, trusted policy-root key, and separately
configured current policy SHA-256/version plus
organization/environment/deployment-scope pins. The policy supplies the latest
checkpoint, so a manual head hash is rejected in that mode.

The closed schema-v2 policy binds the canonical expert-dataset,
corpus-manifest, legal-pack, legal-attestation, and legal-release-checkpoint
identities. It maps keys and opaque owner identities to
`corpus_scope_approver`, `expert_dataset_owner`,
`legal_curator`, and `legal_release_certifier`, with authorization windows. Keys
are unique across roles, declared owner IDs cannot cross separated roles, and
the policy issuer ID/key cannot also be declared as an operational signer or
owner. Distinct strings do not prove separate human/team custody; bank
governance must verify that. Hash and version meanings are defined in the
[benchmark trust contract](../benchmark/README.md#hash-and-version-semantics).

The policy also carries a separate, canonical legal-source reviewer registry.
Every artifact review in every checkpoint must use `PageMappingProof` v2, bind
the checkpoint/artifact pair to one opaque reviewer owner, occur between source
capture and checkpoint creation, and fall inside that reviewer's authorization
window. Reviewer owners are separate from the policy issuer and all four
operational signer owners; effective reviewer revocation fails closed.

Legal-release checkpoints now support forward key rotation. The latest
checkpoint must verify with the primary current key; older checkpoints can use
explicit predecessor keys. The signed policy connects those keys with
`replaces_key_id` and rejects cycles, disconnected chains, backwards rotation,
use outside the declared event-time window, effective key revocation, and
effective checkpoint revocation.
A retired, non-revoked key can verify only history created within its validity
window as declared in the signed artifact. Policy validity/approval, corpus
review, dataset decision, curator attestation, checkpoint creation, and page
review times are all declared signed fields evaluated against the local process clock—not independent
signature timestamps. Bank promotion or an external timestamp/receipt service
must establish signing time if required. Every predecessor's retained files are
still re-hashed, but historical legal-pack bytes are not retained and replayed
against their historical Citation inventories.

These remain cryptographic/policy-consistency controls, not bank authorization.
The repository does not prove who owns the configured policy root or that the
policy, operational keys, and current SHA/version/scope pins arrived through bank
RBAC and an approved atomic promotion. Organization/environment/scope pins are
also deployment inputs. A signed policy that differs from the
configured current pins fails in bank-policy mode; if the external pins
themselves are stale, the offline verifier cannot discover the newer policy.
Bank deployment must mount these inputs separately, retain the promotion and
approval record, and exercise revocation/compromise handling. This implemented
slice does not by itself complete the bank trust-policy issue.

Page verification proves exact UTF-8 excerpt containment in the retained text
for the attested pages. Policy-free v1 evidence carries only the
`legal_source_reviewer` role assertion. V2 binds each checkpoint/artifact review
to a policy-authorized owner ID, but does not authenticate the human action,
provide a reviewer signature, or independently reproduce page text from the raw
PDF/source bytes. A `LegalReleaseCheckpoint` chain containing v1 page proofs
cannot be mutated into v2. Adoption requires a new independent genesis whose
every artifact contains a v2 proof; it cannot reference a v1-proof ancestor, so
the prior chain remains archival without a verified continuity claim. The
tracked manifest and 20-case dataset do not satisfy these gates; all
evidence is `not_verified` for legal currentness and
currentness/version/amendment score authorization remains unsupported.
`python -m benchmark.release_preflight` validates only this chain, from a source
checkout, and explicitly reports both bank authorization and model score
authorization as false; it does not execute the expert dataset.
