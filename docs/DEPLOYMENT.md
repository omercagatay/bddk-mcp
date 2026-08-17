# Deployment Guide

This guide describes the deployment paths the repository currently supports and the remaining controls for a bank on-premises deployment.

## Runtime Profiles

The server selects exactly one reviewed tool profile per process.

| Mode | Selection | Database identity | Current tool surface |
|---|---|---|---:|
| Public stdio | `bddk-mcp serve --profile public` (default) | `BDDK_DATABASE_URL` | 17 public tools |
| Operator stdio | `bddk-mcp serve --profile operator` | `BDDK_OPERATOR_DATABASE_URL` | 17 public + 14 operator = 31 tools |
| Public Streamable HTTP | Add `--transport streamable-http` | `BDDK_DATABASE_URL` | 17 public tools |
| Operator Streamable HTTP | Operator profile plus explicit remote opt-in when non-loopback | `BDDK_OPERATOR_DATABASE_URL` | 31 total tools |

`BDDK_TOOL_PROFILE=public|operator` is the environment equivalent of `--profile`. The operator profile is not an in-process switch on a public server: it must be a separate process and requires its own DSN. It never falls back to `BDDK_DATABASE_URL`.

Prepare a new dedicated database in separate, least-privileged stages before
either profile serves it. The abbreviated commands below assume that a DBA has
already installed `vector` and `unaccent` in `public` and created the reviewed
LOGIN identities. Replace every uppercase placeholder; `DATABASE` must be the
same exact, independently approved database name in every guard:

```bash
# DBA steps: each SQL file refuses an absent or mistargeted independent guard.
PGOPTIONS='-c bddk.expected_database=DATABASE' \
  psql --single-transaction --set ON_ERROR_STOP=1 DBA_DSN \
  --file deploy/postgres/01_roles.sql

BDDK_EXPECTED_DATABASE_NAME='DATABASE' \
BDDK_SCHEMA_OWNER_DATABASE_URL='postgresql://MIGRATOR:SECRET@HOST:5432/DATABASE?options=-c%20role%3Dbddk_schema_owner&sslmode=verify-full&sslrootcert=%2FAPPROVED%2Fpostgres-ca.crt' \
  uv run --frozen bddk-mcp migrate

PGOPTIONS='-c bddk.expected_database=DATABASE' \
  psql --single-transaction --set ON_ERROR_STOP=1 DBA_DSN \
  --file deploy/postgres/02_grants.sql

# Optional diagnostic only; this result is not a trust handoff to bootstrap.
uv run --frozen bddk-mcp verify-corpus --seed-dir /APPROVED/CORPUS

BDDK_INGESTION_DATABASE_URL='postgresql://INGESTION:SECRET@HOST:5432/DATABASE?sslmode=verify-full&sslrootcert=%2FAPPROVED%2Fpostgres-ca.crt' \
  uv run --frozen bddk-mcp bootstrap \
    --seed-dir /APPROVED/CORPUS \
    --reindex-existing \
    --require-quantified-freshness \
    --require-measured-freshness \
    --require-verified-signature \
    --trusted-signing-key /APPROVED/TRUST/corpus-signing-public-key.pem

BDDK_RELEASE_VERIFIER_DATABASE_URL='postgresql://VERIFIER:SECRET@HOST:5432/DATABASE?sslmode=verify-full&sslrootcert=%2FAPPROVED%2Fpostgres-ca.crt' \
BDDK_RELEASE_VERIFIER_REVISION_SHA256='REPLACE_64_LOWERCASE_HEX_REVISION' \
BDDK_RELEASE_VERIFIER_IMAGE_DIGEST='sha256:REPLACE_64_LOWERCASE_HEX_IMAGE_DIGEST' \
BDDK_RELEASE_VERIFICATION_VALIDITY_SECONDS=900 \
  uv run --frozen bddk-mcp verify-and-stage-corpus-release \
    --seed-dir /APPROVED/CORPUS \
    --trusted-signing-key /APPROVED/TRUST/corpus-signing-public-key.pem

BDDK_RELEASE_PUBLISHER_DATABASE_URL='postgresql://PUBLISHER:SECRET@HOST:5432/DATABASE?sslmode=verify-full&sslrootcert=%2FAPPROVED%2Fpostgres-ca.crt' \
  uv run --frozen bddk-mcp activate-corpus-release \
    --request-id corpus_release_request_sha256_REPLACE_64_LOWERCASE_HEX
```

`bddk-mcp migrate` performs immutable schema work only; it never installs
extensions, grants runtime privileges, or imports corpus data.
`bddk-mcp bootstrap` requires an already migrated and granted schema. It imports
the reviewed seed corpus, builds sections and 768-dimensional embeddings, and
validates readiness; it does **not** migrate. `--reindex-existing` additionally
rebuilds and publishes every canonical document for the active retrieval
profile. It is required after migration v0003 for an existing corpus, which
remains intentionally unsearchable until republished. The legacy `bddk-seed
import|export` helper remains available for reviewed corpus maintenance. The
checked-in seed corpus is job-specific and is not a claim of exhaustive BDDK
coverage. Bootstrap itself binds the reviewed scope declaration to the exact
manifest-declared artifact paths, checksums, byte sizes, record counts, and
extraction timestamps before it opens a database pool. It rejects a present
but undeclared reserved `documents.json`, `chunks.json`, or
`decision_cache.json`; it does not fall back from the manifest to a familiar
filename. A prior `verify-corpus` run is useful diagnostics only and transfers
no trust to this mutating process. Production passes
`--require-quantified-freshness`, `--require-measured-freshness`, and
`--require-verified-signature` directly to bootstrap, with
`--trusted-signing-key` on a separately mounted approved Secret. Numeric
objectives and evidence are different gates: `measured` requires per-document
authoritative-publication, source-detection, download, extraction, and
retrieval-publication events whose calculated lags satisfy all three
objectives. Successful bootstrap output includes the path-free manifest ID and
SHA-256 and reports that publication remains required; bootstrap does not
persist a release candidate. The four lifecycle Jobs/commands are therefore
ordered `migrate` → `bootstrap` → `verify-and-stage-corpus-release` →
`activate-corpus-release`, with DBA grant reconciliation after migration and
before bootstrap.

The verifier's database credential is `BDDK_RELEASE_VERIFIER_DATABASE_URL`,
whose LOGIN must inherit exactly `bddk_release_verifier`. It receives the
approved corpus and separately mounted trust key, repeats strict
manifest/signature/derived-artifact verification,
regenerates embeddings, proves exact database membership while the corpus state
is locked, and stages an append-only request. Its provenance inputs are a
64-lowercase-hex `BDDK_RELEASE_VERIFIER_REVISION_SHA256` and immutable
`sha256:` `BDDK_RELEASE_VERIFIER_IMAGE_DIGEST`. The request lifetime is
`BDDK_RELEASE_VERIFICATION_VALIDITY_SECONDS`, default 900 and accepted only from
60 through 3,600 seconds. The staged evidence binds those values plus the
signature, per-run verification evidence, manifest, retrieval profile,
database-computed state hash and corpus epoch.
The verifier output includes both a random, content-free
`verification_run_sha256` and its derived `verification_evidence_sha256`;
the canonical receipt can be recomputed only with the reviewed manifest, exact
detached-signature SHA, retrieval-profile SHA, exact verifier revision/image
provenance, that verification-run value, and governed append-only staged-request
evidence. The path-free CLI summary alone is not a complete audit export.
The verifier rejects a trust-key path that is supplied inside, or resolves
inside, the corpus root; do not place or symlink the trust anchor into the
approved corpus mount.

The activation process uses only `BDDK_RELEASE_PUBLISHER_DATABASE_URL` and the
opaque `corpus_release_request_sha256_...` request ID. Do not mount or inject the
corpus, manifest, signature, signing key, verifier DSN, or trust material into
that process. In one transaction, activation rejects an unavailable, expired,
previously used, wrong-epoch, changed-state, or non-ready request before
appending the v0005-compatible release and activation plus the v0008 request
binding. The legacy `publish-corpus-release` CLI is deliberately disabled; it
is not a fallback for failed staging. Any later corpus mutation advances the
corpus epoch and invalidates the active view. The current manifest (`bddk-job-corpus-2026-08-14`) is Ed25519-signed,
quantified, and consistent with the 9,675 chunks the current profile
regenerates.

Schema v10 admits exactly two `freshness_policy_result` values:
`quantified_measured_signature_verified_pass` and the explicitly weaker
`quantified_unmeasured_signature_verified_pass`. Quantified objectives and a
verified signature remain unconditional; no unsigned or unquantified release is
admissible at any level. The verifier derives the level from manifest evidence
rather than accepting a caller's claim, and `--accept-unmeasured-freshness` only
*permits* the weaker level — it can never relabel an unmeasured corpus as
measured. Because the policy value is fingerprinted into both the release and
request identities, a measured and an unmeasured release over identical corpus
state can never share an identity or be substituted for one another. Both the
active-release view and the operator `health_check` report the recorded level,
so a bank can see which gate an active release actually passed.

The tracked corpus declares `not_measured` freshness, so it needs
`--accept-unmeasured-freshness` at the verifier stage. The checked-in OpenShift
Jobs and the `bank-bootstrap` overlay deliberately keep the strict measured
flags: admitting an unmeasured corpus into a bank namespace must be an explicit,
separately approved deviation that drops `--require-measured-freshness` from the
bootstrap Job and adds `--accept-unmeasured-freshness` to the verifier Job.
Reaching the measured level still requires a live pipeline that records
per-document authoritative-publication, detection, download, extraction, and
retrieval-publication events.

The ordinary migration refuses an unmanaged pre-ledger schema. The explicit
`bddk-mcp migrate --adopt-legacy` path supports only the exact verified legacy
shape and is never a clean-install flag or a general repair mechanism. Stop all
workloads, prove a restorable backup, and follow
[`docs/LEGACY_DATABASE_UPGRADE.md`](LEGACY_DATABASE_UPGRADE.md) before using it.

Migration v0003 also refuses a populated v0002 database before its blocking
section-hash/FK backfill unless
`--allow-retrieval-publication-backfill` is explicit. Use that flag only after
stopping serving and ingestion, proving a restorable backup, and rehearsing the
upgrade against a size-matched restore. It is not needed or permitted as a
routine clean-install shortcut. After migration, run ingestion bootstrap with
`--reindex-existing` before serving so every approved document receives a
validated retrieval publication.

The current immutable ledger ends at schema v10: v9 adds the regulatory
cross-reference edge graph, and v10 widens the release freshness policy to the
closed two-value set above, replacing the v8 staging routine with its
policy-aware signature. Current runtime, verifier, and publisher admission
requires v10; a database left at v8 or v9 is an upgrade source, not a compatible
steady-state target. Reapply `deploy/postgres/02_grants.sql` after the v10
migration so the new staging signature receives the reviewed verifier grant. An
ordinary v7 → v8 migration
is additive: it creates append-only staged-request and activation-binding
relations plus the verifier/activation facades, and revokes every non-owner
`EXECUTE` grant on the v5 direct-publication routine. It does not rewrite an
existing release, activation, retained generation, seal, or release binding.
Reapply `deploy/postgres/02_grants.sql` immediately after migration so the exact
verifier/publisher ACL split is reconciled. Current runtime, verifier, and
publisher identity/catalog admission requires v10; a database left at v7, v8, or
v9 is an upgrade source, not a compatible steady-state target for this release.
Migration-time catalog attestation deliberately accepts an owner-only routine
ACL shape before grant reconciliation. That state is not deployment-ready:
exact verifier/publisher workload-identity checks continue to fail closed until
`02_grants.sql` installs the reviewed grants.

The earlier v7 canonical-hash guard remains a separate remediation boundary. If
migration from v5/v6 refuses because the active pre-v7 release hash is not the
canonical recomputation, keep the database unchanged, preserve the historical
row, independently review/revalidate the corpus, and use only the exact
publication-only v5/v6 compatibility procedure before retrying v7 and then v8.
The code retains exact v5/v6/v7 publication identity/catalog checks solely for
reviewed migration remediation, but the current `publish-corpus-release` CLI is
disabled and schema v8 revokes its database capability. Do not manufacture a
release row or retention binding, do not grant the v5 routine back on v8, and do
not admit serving or retention while remediation is in progress. Follow the
versioned [legacy upgrade runbook](LEGACY_DATABASE_UPGRADE.md) and require a
separately approved remediation artifact/procedure rather than treating the old
alias as an operational shortcut.

### Database target and TLS contract

The two DBA SQL files compare the separately supplied
`bddk.expected_database` setting with `current_database()` before changing
cluster roles or database privileges. The migration command independently
requires `BDDK_EXPECTED_DATABASE_NAME` and performs the same exact-name check.
A mistargeted DSN therefore cannot authorize a lifecycle operation by itself.

All application PostgreSQL DSNs fail closed by default unless they are
PostgreSQL URIs with a hostname, exactly `sslmode=verify-full`, and exactly one
absolute `sslrootcert` path. The CA must be an approved PostgreSQL trust root;
it is not the OpenShift service CA used for MCP application sockets.
`BDDK_ALLOW_INSECURE_DATABASE=true` bypasses this application check only for an
isolated local-development database. Never set it in a shared, remote, bank,
staging or production workload.

This release supports **PostgreSQL major 17 only**. Migration, readiness,
schema-owner, public, ingestion, release-verifier, release-publisher, operator,
and telemetry connection boundaries
verify `server_version_num` and refuse any other or unverifiable major before
performing application work. Expanding the supported set requires the same
mandatory full migration/catalog/role/publication/job suite on the additional
major; “likely compatible” is not an admission rule.

### Database identity contract

The repository SQL creates NOLOGIN group roles; the bank must create and rotate
the authenticating LOGIN roles outside Git. The required effective memberships
are:

| Identity | Required group-role memberships | Used for |
|---|---|---|
| Schema owner | `bddk_schema_owner`, entered through `SET ROLE` | `bddk-mcp migrate` only |
| Ingestion | exactly `bddk_ingestion` | `bootstrap`, synchronization, reindex and corpus backfill |
| Release verifier | exactly `bddk_release_verifier` | strict corpus/trust verification and short-lived request staging only |
| Release publisher | exactly `bddk_release_publisher` | request-ID-only activation and the separate v7 retention facades; no corpus/trust access |
| Public runtime | exactly `bddk_public_reader` | public MCP process |
| Operator runtime | exactly `bddk_public_reader`, `bddk_ingestion`, `bddk_operator_runtime` | operator MCP process and its current in-process runners |
| Telemetry writer | exactly `bddk_telemetry_writer` | optional column-scoped trace inserts only |

Public, operator, ingestion, verifier, and publisher entry points inspect the
real PostgreSQL session identity, memberships, database/schema/object
privileges, and unexpected managed objects; a differently written DSN for the
same over-privileged LOGIN does not bypass the check. Verifier and publisher
LOGINs must be distinct, inherit neither one another's role nor any runtime or
ingestion role, and must not share DSNs. Telemetry performs a separate exact
INSERT-only identity check. Every newly opened physical connection in these
pools is checked against its selected contract, so pool growth or a changed
database endpoint cannot introduce an unchecked session. The application never
grants these roles itself. See
[`deploy/postgres/README.md`](../deploy/postgres/README.md) for the authoritative
matrix and DBA apply order.

The migration connection is checked separately: its effective `current_user`
must be exactly `bddk_schema_owner`, entered through `SET ROLE` from a distinct
restricted LOGIN with the exact direct membership. Administrative attributes,
unexpected memberships, database ownership and the wrong database name all
cause refusal.

Database readiness includes a live, `SELECT`-only catalog attestation of
critical constraints, trigger definitions and enablement, function
bodies/security/configuration, FTS indexes and HNSW options. Missing or drifted
objects make startup fail with bounded remediation guidance even when the
database accepts `SELECT 1`. Repair drift through a reviewed migration or clean
restore; never edit the checksum ledger or weaken readiness.

## Local stdio

For the disposable loopback development topology, let Compose execute the same
role → identity/extension → migration → grants → bootstrap sequence:

```bash
uv sync --frozen
export BDDK_JWT_ISSUER=https://idp.invalid
export BDDK_JWT_RESOURCE=https://localhost:8000/mcp
export BDDK_JWT_JWKS_URL=https://idp.invalid/jwks
export BDDK_JWT_AUDIENCE=bddk-mcp-local
docker compose up --build -d bddk-bootstrap
docker compose wait bddk-bootstrap
BDDK_ALLOW_INSECURE_DATABASE=true \
BDDK_DATABASE_URL=postgresql://bddk_local_public:local-only-public@localhost:5432/bddk \
  uv run --frozen bddk-mcp serve --profile public
```

The fixed local credentials are public test fixtures and the PostgreSQL port is
bound to host loopback. The reserved `.invalid` JWT values only satisfy Compose
interpolation for the unused HTTP runtime definition; this lifecycle target
does not start that server, and those values must never be used to serve HTTP.
Do not copy the fixed credentials into a remote environment. A pre-ledger
Compose volume is not automatically adopted; preserve it and use the legacy
upgrade runbook, or intentionally recreate only disposable data.

The default transport is stdio. Use the repository [`.mcp.json`](../.mcp.json) with clients that support that project configuration format. Codex uses `config.toml`; see the examples in the main [README](../README.md#codex-configuration).

A local operator process requires a separately provisioned database role:

```bash
BDDK_ALLOW_INSECURE_DATABASE=true \
BDDK_OPERATOR_DATABASE_URL=postgresql://bddk_local_operator:local-only-operator@localhost:5432/bddk \
  uv run --frozen bddk-mcp serve --profile operator
```

The insecure bypass and DSNs above are local, loopback-only test fixtures.
Store real credentials in platform Secrets and retain the fail-closed
`verify-full` contract remotely.

## Streamable HTTP Contract

Select HTTP with `MCP_TRANSPORT=streamable-http` or `--transport streamable-http`. The MCP endpoint is `/mcp`. FastMCP is configured for stateless JSON responses rather than durable server sessions.

Loopback (`127.0.0.1`) is the default bind and is the appropriate development default. A non-loopback bind, including `0.0.0.0` in a container, fails startup unless all remote security settings are complete (or the explicit unauthenticated opt-in described below replaces the bearer-authentication settings):

| Variable | Remote requirement |
|---|---|
| `MCP_HOST`, `PORT` | Explicit bind address and port |
| `BDDK_HTTP_ALLOWED_HOSTS` | Comma-separated exact authorities; no wildcard, scheme, path, or credentials |
| `BDDK_HTTP_ALLOWED_ORIGINS` | Comma-separated exact HTTPS origins; no wildcard, credentials, path, query, or fragment |
| `BDDK_JWT_ISSUER` | Bank-approved absolute HTTPS issuer URL |
| `BDDK_JWT_RESOURCE` | Bank-approved absolute HTTPS protected-resource URL used for MCP metadata and authorization flow |
| `BDDK_JWT_JWKS_URL` | Bank-approved absolute HTTPS JWKS URL |
| `BDDK_JWT_AUDIENCE` | Exact access-token audience mapped by the IdP to this MCP resource server; do not use the calling client's ID-token audience |
| `BDDK_JWT_REQUIRED_SCOPES` | Must include `bddk.read` for public or `bddk.operator` for operator |
| `BDDK_JWT_ALGORITHMS` | Approved asymmetric algorithms only; symmetric JWT algorithms are rejected |
| `BDDK_JWT_ACCESS_TOKEN_TYPES` | Allowed JOSE `typ` values; fail-closed default is RFC 9068 `at+jwt`; generic `JWT` requires explicit Keycloak compatibility opt-in |
| `BDDK_TLS_CERT_FILE`, `BDDK_TLS_KEY_FILE` | Optional paired PEM paths for HTTPS at the application socket; one without the other, unreadable material, encrypted keys, or a mismatched pair fails startup |
| `BDDK_OPERATOR_REMOTE_ENABLED` | Must be explicitly `true` for a non-loopback operator profile |

The issuer, resource, JWKS, audience, token type, and scope values are deployment-specific. Obtain them from the bank-approved identity design; do not copy values from repository tests. `BDDK_JWT_RESOURCE` identifies this protected resource to MCP clients; the composed application publishes RFC 9728 metadata at `/.well-known/oauth-protected-resource/mcp`, and its unauthenticated 401 challenge supplies the same URL through `resource_metadata`. Access-token binding is enforced through `BDDK_JWT_AUDIENCE`. Configure a dedicated API/resource-server audience distinct from interactive OAuth client audiences so a Keycloak-style ID token cannot be confused with an access token. Keep the fail-closed `at+jwt` default when the issuer supports RFC 9068. Set `BDDK_JWT_ACCESS_TOKEN_TYPES=at+jwt,JWT` only when generic `JWT` is required by the approved Keycloak/IdP profile. This discovery contract is application evidence only; client registration and the complete bank IdP authorization flow still require live acceptance.

Host, Origin, content type, duplicate headers, and request-body size are validated before bearer authentication. JWT verification then bounds token length and checks the asymmetric signature, key ID/algorithm, access-token type, issuer, exact resource-server audience, expiry, optional not-before time, and subject. A custom token `resource` claim is neither required nor trusted. Missing or invalid authentication returns 401; a cryptographically valid token without the profile's required scope returns 403.

`BDDK_HTTP_ALLOW_UNAUTHENTICATED` is a supported explicit opt-in that serves a non-loopback bind without bearer authentication. It is unset by default; while it is unset, every requirement in the table above applies unchanged. When set:

- it cannot be combined with any configured `BDDK_JWT_`-prefixed setting — startup refuses and names the offending variables;
- it is refused outright for a non-loopback operator profile, regardless of `BDDK_OPERATOR_REMOTE_ENABLED`: operator tools stay authenticated or loopback-only by construction;
- the server advertises no OAuth discovery: there is no `WWW-Authenticate` challenge, and both `/.well-known/oauth-protected-resource/mcp` and `/.well-known/oauth-authorization-server` return 404;
- exact `BDDK_HTTP_ALLOWED_HOSTS`/`BDDK_HTTP_ALLOWED_ORIGINS` values, the body-size and body-deadline limits, the concurrency cap, and the per-minute rate limit all still apply.

Without bearer authentication the per-process rate limiter is the primary abuse control, so its client key must be correct. Behind a reverse proxy, set `BDDK_HTTP_TRUSTED_PROXY_HOPS` (below) to the real, operator-controlled hop count; a wrong value makes the limiter either shared across all callers or spoofable.

The application has coarse per-process overload controls:

| Variable | Default | Behavior |
|---|---:|---|
| `BDDK_HTTP_MAX_BODY_BYTES` | 1,048,576 | Rejects oversized POST bodies with 413 |
| `BDDK_HTTP_BODY_FIRST_BYTE_TIMEOUT_SECONDS` | 5 | Requires the first non-empty request-body byte within `(0, 120]` seconds; otherwise returns 408 |
| `BDDK_HTTP_BODY_CHUNK_TIMEOUT_SECONDS` | 5 | Requires each later non-empty body chunk within `(0, 120]` seconds; otherwise returns 408 |
| `BDDK_HTTP_BODY_TOTAL_TIMEOUT_SECONDS` | 30 | Bounds the complete body read to `(0, 300]` seconds, including continuous drip traffic; otherwise returns 408 |
| `BDDK_HTTP_MAX_CONCURRENCY` | 32 | Immediately rejects excess in-process work with 503 and `Retry-After` |
| `BDDK_HTTP_RATE_LIMIT_PER_MINUTE` | 120 | Fixed-window limit per rate-limit client key, returning 429 and `Retry-After` |
| `BDDK_HTTP_TRUSTED_PROXY_HOPS` | 0 | Trusted reverse proxies in front of the bind; `0` keys the rate limiter on the ASGI socket peer and ignores `X-Forwarded-For` entirely |
| `BDDK_JWT_MAX_TOKEN_LENGTH` | 16,384 | Rejects oversized bearer tokens before verification |

The three body deadlines are independent; an empty ASGI request event does not
count as the first byte, and the total deadline still applies when every chunk
arrives within the chunk deadline. Align ingress/proxy limits explicitly so the
intended layer returns the timeout and releases its capacity slot. The rate and
concurrency state is bounded, process-local and non-durable. It is not shared
across workers or replicas and does not provide a global ingress limit. Uvicorn
proxy-header trust is disabled. At the default `BDDK_HTTP_TRUSTED_PROXY_HOPS=0`
the rate limiter keys on the ASGI socket peer and does not accept
client-supplied `Forwarded` or `X-Forwarded-For` values, which is correct for a
directly exposed bind. Behind `n` operator-controlled reverse proxies, set the
real hop count; the limiter then keys on the `n`-th entry from the right of the
combined forwarded list — the last hop the operator vouches for — and anything
unusable degrades to a shared `unknown` bucket rather than falling back to the
socket peer, so a spoofed header cannot merge distinct callers. Count only hops
you control that append to the header; a wrong value makes the limiter either
shared or spoofable. A bank deployment still needs shared
ingress request/rate policy, an approved end-to-end TLS topology, audit events,
and abuse controls at the approved gateway.

### Fixed health endpoints

Only these exact, content-free GET routes are provided for orchestration:

| Endpoint | Meaning |
|---|---|
| `GET /health/live` | The application process is alive; returns `{"status":"alive"}` |
| `GET /health/ready` | Dependencies exist and the cached five-second schema/catalog/identity/job/telemetry attestation passes; otherwise returns 503 |

These two routes intentionally bypass MCP bearer authentication and Host/Origin admission so Kubernetes/OpenShift probes can use a pod-IP Host header. They return no corpus, configuration, identity, or job data. Do not treat the `health_check` MCP operator tool as an HTTP probe.

The readiness probe bounds each refresh to five seconds and caches only the
boolean outcome for five seconds to avoid running catalog inventories on every
probe. Each refresh checks database readiness and the complete runtime identity;
operator and enabled telemetry state are checked through their own exact
contracts. Liveness deliberately does not contact dependencies.

## Durable Operator Jobs and Current Limitation

The operator profile adds 14 tools to the 17 public tools. Mutating tools such as update checks, cache refresh, synchronization, startup synchronization, and executed backfill return an immediate job receipt. Use `get_operator_job`, `list_operator_jobs`, and `cancel_operator_job` to inspect or request cancellation.

The server uses `PostgresJobRepository` in the operator profile. The global v2
migration creates `bddk_operator.operator_jobs`; the repository persists job
state, argument fingerprints, hashed idempotency keys, numeric progress, bounded
result metrics, safe error codes, and retained terminal history. It deliberately
does not persist raw tool arguments, document text, or exception messages. A
session-level PostgreSQL advisory lease controls cross-process runner admission.
A separate transaction-level advisory lock serializes every sanctioned corpus
writer transaction and release publication. The operator pool therefore
requires `BDDK_PG_POOL_MAX >= 2`: the job-admission lease pins one connection
while state/progress writes and writer transactions use another.

This is durable state and cross-process single-flight coordination, but it is
not a distributed workflow engine:

- on startup, an abandoned `running` job is marked `interrupted` only after the
  process obtains the repository lease, proving no live holder remains;
- a persisted `cancel_requested` job is finalized as `cancelled` during recovery;
- a `queued` job is never guessed stale because another process may be between
  persistence and lease acquisition; retrying the same idempotency key can
  resume it, or it can be cancelled explicitly;
- runner code and task scheduling remain inside the operator process; there is
  no external dispatcher, retry scheduler, or bank-approved workflow SLA;
- graceful shutdown drains and then cancels local tasks, but abrupt process,
  node, or database failure still requires operational reconciliation.

The supplied OpenShift operator Deployment deliberately remains one replica
with `Recreate`. Do not scale it or represent it as a bank-grade queue/system of
record until overlapping-pod, queued-recovery, cancellation, database-failover,
and multi-replica acceptance tests pass in the target bank environment.

## Docker and Container Contract

Both Dockerfiles install and run the packaged `bddk-mcp` entry point. Their base
and `uv` images are digest-pinned. The standard and Spaces images explicitly
copy `seed_data/` and download `intfloat/multilingual-e5-base` at full commit
`d13f1b27baf31030b7fd040960d60d909913633f`, then save it at
`/app/embedding_model` for offline runtime loading.

The immutable schema stores `public.vector(768)`, and configuration rejects any
`BDDK_EMBEDDING_DIM` other than `768`. A remote custom embedding model requires
an immutable full commit in `BDDK_EMBEDDING_MODEL_REVISION`; the built-in
optional reranker `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` defaults to full
commit `1427fd652930e4ba29e8149678df786c240d8825`. A local model path is trusted as
a deployment artifact and should be content-addressed/scanned by the bank.
Changing an embedding model, tokenizer, prefixes, or chunk settings requires a
controlled full re-embedding and retrieval regression run. The v3 migration
adds a retrieval-publication record keyed by document, content hash, and active
retrieval-profile hash. Chunk mutation invalidates that record; ingestion
publishes again only after chunk count, ordering, hashes, embeddings, and totals
pass integrity checks. Retrieval joins the current publication and therefore
fails closed instead of serving incomplete, stale, or differently profiled
chunks. After this migration, the mandatory ingestion `bootstrap`/reindex stage
must republish every approved document before serving.

Serving startup does not perform schema, seed, synchronization, or embedding-backfill writes. It validates the database, loads the decision cache with `SELECT`, and constructs search clients. An incomplete database fails startup with remediation guidance. Telemetry can write when enabled, and the operator profile intentionally performs writes; provision database roles accordingly.

The runtime wheel/sdist exclude the repository-only benchmark harness, tests,
deployment SQL/manifests, `.env` files, and `seed_data/`; the code package does
not silently distribute the reviewed corpus. A source checkout or the provided
container supplies `seed_data/`. Another wheel installation must mount an
approved corpus and pass `--seed-dir` or `BDDK_SEED_DIR` to `bootstrap`. Run
evaluation workflows from a source checkout with `uv sync --group benchmark`.

The Compose and image definitions bind HTTP on `0.0.0.0`. Under the fail-closed HTTP policy, a container must receive the exact Host/Origin settings above and either the complete JWT settings or the explicit `BDDK_HTTP_ALLOW_UNAUTHENTICATED` opt-in, or startup will be refused. Development credentials and database ports from Compose must never be reused or exposed outside a developer environment.

The repository supply-chain lane invokes Buildx with
`--provenance=false --load`, then binds the exact manifest descriptor/digest,
image-config digest,
loaded local image identity, and Syft SBOM before producing canonical evidence.
It separately creates an **unsigned** repository SLSA provenance envelope; that
envelope is not Buildx or bank-signed attestation. The lane also requires the
model manifest's immutable Git commit to agree with runtime configuration and
both Dockerfiles.

The workflow has two deliberately different repository decisions. The
always-run `evidence-integrity` job builds and scans everything, verifies
fresh/non-suppressive evidence, fails on unexcepted secrets, and records
unresolved High/Critical vulnerabilities without making them a pull-request
failure by themselves. The `release-eligibility` job runs only on a `v*` tag or
when a manual workflow invocation explicitly sets
`evaluate_release_policy=true` **on `main`**; a feature-branch manual run cannot
produce that job. For a `v*` push, the checked-out tag commit must be in
`origin/main` history; a tag on an unmerged feature commit fails closed. It
requires the same run's integrity job to
succeed, downloads the artifact bound to that run ID and attempt, and exactly
re-hashes that run's complete evidence manifest before it binds
the standard and Spaces scan reports to their respective Dockerfile SHA-256,
and then fails on any unexcepted High/Critical finding. It also fails whenever
an applied pending vulnerability or secret exception reports
`external_approval_required=true`.

Despite its name, a green `release-eligibility` job is only a repository
precondition. It cannot authenticate bank risk acceptance, sign an artifact,
prove internal-registry custody, verify the digest selected by an OpenShift
deployment, apply an admission policy, or authorize promotion. The repository
result therefore keeps `release_promotion_eligible=false`; the bank-controlled
promotion path must independently authenticate approvals and verify the exact
image digest, signature, provenance, SBOM, scanner age, and retained evidence.
The GitHub workflow itself does not push, sign, admit, or promote an image.
Protecting the workflow, policy, and release tags with repository rulesets and
CODEOWNERS remains a separate GitHub governance requirement.

### Outbound regulatory HTTP

Live catalog, document-sync, institution, announcement, and bulletin paths use
code-owned outbound controls: exact HTTPS hosts for BDDK/mevzuat, no embedded
credentials or non-default ports, public-address DNS preflight, bounded redirect
hops with destination/DNS revalidation, streamed decoded-body limits, and
privacy-safe retry logs that omit URLs, query strings, and exception messages.
Catalog/API responses are capped at 8 MiB; document-sync limits vary by artifact
class up to 128 MiB. These are application memory safeguards, not source
authenticity, malware-scanning, or network-isolation guarantees.

DNS validation and the subsequent socket connection are not atomic. A bank
deployment must enforce the same destinations with OpenShift egress
NetworkPolicy, an approved egress proxy/firewall, DNS policy, and TLS inspection
rules where required. Public institution, announcement, bulletin, and update
tools can call live BDDK sources, so both public and operator runtimes require
narrow TCP 443 egress to an approved regulatory-source destination or enterprise
proxy. Lifecycle Jobs require DNS/PostgreSQL only and must not inherit this
source reach. The repository starter default-denies egress for all BDDK MCP
pods but deliberately supplies no bank addresses or peer selectors. Without
the bank-specific exact allow policies, connectivity is expected to fail
closed.

## OpenShift AI Starter

[`deploy/openshift`](../deploy/openshift/README.md) now provides a conservative
starter: separate public/operator Deployments, Services and service accounts;
OpenShift service-serving certificates at both application sockets; a
public-only re-encrypt TLS Route; an internal HTTPS operator Service; exact
HTTPS probes; non-root/read-only security contexts; resource bounds; restricted
ingress plus default-deny egress; exact Secret references; a mounted PostgreSQL
CA contract; and separately applied migration, bootstrap, verifier-stage, and
publisher-activation Jobs. Every workload
uses a fail-closed image-digest placeholder, the image uses a digest-pinned `uv`
source, the embedding-model revision is pinned, and the default non-root UID is
compatible with OpenShift's arbitrary-UID model. Version labels are excluded
from immutable selectors. The offline preflight executes exact standalone
Kustomize v5.8.1 and binds the actual executable SHA-256 to the reviewed release
input. It requires exact rendered-object, selector/label/namespace,
NetworkPolicy, Secret/ConfigMap-key, command/port, volume/mount, pod-container,
and restricted security-context inventories; omissions, additions, sidecars,
init/ephemeral containers, host namespace sharing, command overrides, and
broadened ingress/egress fail closed.

The production-import repository contract is the separately reviewed
[`deploy/openshift-overlays/bank-bootstrap`](../deploy/openshift-overlays/bank-bootstrap/)
overlay. It renders the runtime and lifecycle inventory while patching the
bootstrap Job to use the read-only `bddk-mcp-approved-corpus` PVC and the
separate read-only `bddk-mcp-corpus-trust` Secret. The exact command passes
`--require-quantified-freshness`, `--require-measured-freshness`,
`--require-verified-signature`, and `--trusted-signing-key` directly to the
mutating bootstrap process. The offline preflight renders and checks this
overlay; it does not prove that either bank-managed source exists or that the
Job ran successfully. Because the activation resource must contain the actual
unexpired staged request ID, the complete final-release preflight runs after the
first three lifecycle Jobs have completed through a separately reviewed gate;
never use a dummy ID or apply the combined overlay to impose Job order.

This is repository-level implementation evidence, not bank acceptance or a production-ready platform configuration,
including for OpenShift AI. No target
bank cluster, SCC, namespace policy, storage class, PostgreSQL service, IdP,
internal registry, or enterprise proxy was available for this review. Every
`REPLACE_*` value must be
resolved through the bank design, and Secret examples must never contain real
values in Git. The manifests still require validation against the bank's SCC,
trust-bundle injection, registry/image policy, ingress, IdP, namespace labels,
capacity model, backup/restore, disaster recovery, and release controls. The
included policy default-denies egress; environment-specific allow rules are not
supplied. Operator job records and advisory leases are durable, but the
in-process runner/failover behavior remains unaccepted, so the starter
deliberately uses one `Recreate` operator replica.

The automated database lifecycle and role-isolation lane exercises the complete
declared compatibility set, currently PostgreSQL 17 only. A bank-selected
version outside that set is rejected and requires a separately reviewed release
that adds equivalent mandatory evidence before deployment.

For the target bank deployment:

1. Provision a dedicated external PostgreSQL database, install `vector` and
   `unaccent` in `public`, and create distinct schema-owner, ingestion,
   release-verifier, release-publisher, public, operator, and optional telemetry
   LOGIN identities using the reviewed group roles. Apply both DBA SQL files
   with the independent target-database guard and run migration with
   `BDDK_EXPECTED_DATABASE_NAME`. For a v5/v6 upgrade,
   treat any pre-v7 release hash as historical: v7 canonicalizes retained-row
   and current/retained state hashing with function-local `TimeZone=UTC`,
   `DateStyle=ISO, YMD`, `IntervalStyle=postgres`, `bytea_output=hex`, and
   `extra_float_digits=3`. The v7 migration refuses if the active release does
   not match that canonical recomputation. Keep the old ledger row immutable;
   on the unchanged pre-v7 schema (v5 or v6), use only the separately approved
   exact-schema publication-remediation procedure to append/activate a reviewed
   canonical release, then retry v7 and continue immediately through v8. The
   current `publish-corpus-release` CLI is disabled, v8 revokes direct
   publication from non-owners, and current workload admission is v10-only. Continue through v9 and v10, then reapply the grants so the verifier receives the v10 staging signature.
2. Replace the application image placeholder with the scanned immutable digest,
   create `bddk-mcp-postgres-ca` with key `ca.crt`, and keep every workload DSN
   at `sslmode=verify-full` with the mounted absolute CA path.
3. Provision the approved corpus as read-only PVC
   `bddk-mcp-approved-corpus` and its Ed25519 public verification key as key
   `ed25519-public-key.pem` in the separate read-only Secret
   `bddk-mcp-corpus-trust`. Use the reviewed `bank-bootstrap` overlay contract,
   which passes all strict freshness/signature flags and the separately mounted
   `--trusted-signing-key` directly to bootstrap. Preserve migration → grants →
   bootstrap/reindex → release-verifier verification/staging → request-ID-only
   release-publisher activation → approved one-shot `retain-corpus-generation`
   ordering when the release must be an immutable recovery target. The verifier
   Job must receive its own DSN, the corpus/trust mounts, immutable verifier
   revision/image provenance, and a 60–3,600 second request TTL. The activation
   Job must receive only its distinct publisher DSN and exact staged request ID,
   with no corpus PVC, trust Secret, verifier DSN, manifest, or signing material.
   Do not start lifecycle Jobs concurrently merely because the overlay renders
   them together. Retain the bootstrap and verifier's path-free evidence, the
   activation receipt, and the content-free retention receipt. The
   retention command currently has no checked-in OpenShift Job and must use the
   release-publisher identity; it is not an MCP operation and does not activate
   or serve the retained generation. It uses transaction-local
   `lock_timeout=30s` and `statement_timeout=30min`; a timeout is a rolled-back
   operation, not partial success, and requires active-release revalidation
   before retry. If a separately governed release has the
   same exact corpus state and retrieval profile as an existing retained target,
   the command creates a new per-release binding and reuses that physical
   generation and seal. The base `jobs/bootstrap.yaml` remains a
   development/baseline Job and is not a production trust gate.
4. Run public and operator profiles as separate workloads; inject `BDDK_DATABASE_URL` only into public and `BDDK_OPERATOR_DATABASE_URL` only into operator.
5. Keep remote operator disabled unless a private operator Route, `bddk.operator` authorization, and explicit `BDDK_OPERATOR_REMOTE_ENABLED=true` have been approved.
6. Add bank-specific least-privilege egress for DNS, PostgreSQL, IdP/JWKS,
   and TCP 443 to the approved BDDK/Mevzuat source or enterprise proxy before
   starting either public or operator runtime; retain the checked-in default
   deny. Keep lifecycle Jobs limited to DNS/PostgreSQL, without regulatory-source
   or proxy reach.
7. Review and adapt the starter, then validate the exact `/health/live` and `/health/ready` probes in a disposable bank-like namespace.
8. Validate the re-encrypt Route-to-pod handshake and operator Service TLS against the injected OpenShift service CA; retain the application's JWT checks and place shared request limits and audit events at the approved ingress boundary.
9. Prove backup, point-in-time/selected restore, schema upgrade, legacy refusal
   or adoption where relevant, and rollback/cutover procedures in an isolated
   environment. The retained
   [local PostgreSQL 17 v8 drill](evidence/LOCAL_PG17_V8_RECOVERY_DRILL.md)
   passed across two disposable clusters for all 53 managed objects, staged
   request/binding state, and seven LOGIN profiles including the verifier. That
   synthetic repository-scale run is not bank backup/PITR, TLS/HBA, custody,
   target-size RPO/RTO, or capacity evidence; repeat the current v8 contract at
   the bank boundary. Measure backup growth on a target with the same database encoding,
   collation/character-classification names, locale provider/locale, ICU rules,
   and collation version as the source; recovery evidence records and requires
   those values exactly and rejects stored/actual version drift. A mismatch is a fail-closed restore, not an
   equivalent target. Use a controlled backup; the retention CLI's observed WAL
   interval is cluster-wide and non-exclusive. Its baseline is attempted inside
   a savepoint and its endpoint after commit; either failure leaves WAL
   `not_measured` without undoing a durable seal. V7 retained targets are not
   application rollback until H2-02B generation-bound serving/reactivation is
   implemented. Size capacity against unique retained state/profile generations,
   not governed-release binding count, and keep bank retention/capacity approval
   as a release gate.
10. Run release-specific MCP discovery, schema, authentication, tool-call,
   structured-output, timeout, and citation tests for every actual client/model
   combination used by the bank.

The repository has not completed this bank-cluster acceptance, backup/restore
drill, or a release compatibility matrix for Claude, Codex, GPT-based clients,
LM Studio, GPT-OSS, or other local clients. Example configurations demonstrate
configuration shape; they are not compatibility certification.

OpenShift rotates service-serving certificates by updating the generated Secret. Uvicorn does not hot-reload that material, so the bank must use an approved restart/reloader mechanism and prove both routine rotation and CA rollover in a disposable namespace before promotion.

## Railway and Spaces

`railway.toml` builds the standard Dockerfile. Railway must inject a
`verify-full` PostgreSQL DSN with an available CA path and the complete
non-loopback HTTP policy. The `/app/data` volume does not back up an external
PostgreSQL database.

`Dockerfile.spaces` uses port `7860` and includes the reviewed seed corpus, but
the database must be bootstrapped separately. It requires the same secure
database transport, non-loopback HTTP and JWT policies. Neither target supplies
bank-specific identity or global ingress controls by itself.

## Secrets and Logs

- Keep DSNs, tokens, and credentials in platform Secrets, never in Git, image layers, committed MCP configuration, or command-line examples with real values.
- Keep `BDDK_RELEASE_VERIFIER_DATABASE_URL` and
  `BDDK_RELEASE_PUBLISHER_DATABASE_URL` in different Secrets consumed by
  different principals. The verifier may receive approved corpus/trust mounts;
  the activator must receive neither and accepts only the staged request ID.
- Mount the approved PostgreSQL CA separately from credentials; keep
  `sslmode=verify-full` and the absolute `sslrootcert` path in every remote DSN.
  Never enable `BDDK_ALLOW_INSECURE_DATABASE` outside isolated local
  development.
- Keep `BDDK_TOOL_LOG_CONTENT=false` in shared and production environments.
- Keep telemetry disabled unless approved; when enabled, inject a distinct
  `BDDK_TELEMETRY_DATABASE_URL` whose LOGIN inherits only
  `bddk_telemetry_writer`. Never reuse the public or operator DSN.
- Keep `BDDK_TELEMETRY_STORE_TEXT=false` unless storage of raw queries has been explicitly approved.
- Treat bank queries as confidential even when the regulatory corpus itself is public.
