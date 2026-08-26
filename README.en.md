# BDDK MCP Server — English Operational Guide

[Bilingual project README](README.md) | [Current status](docs/STATUS.md) | [Documentation index](docs/README.md) | [Deployment guide](docs/DEPLOYMENT.md) | [Contributing](CONTRIBUTING.md) | [Security](SECURITY.md)

BDDK MCP Server is an offline-first Model Context Protocol server for searching, retrieving, and analyzing Turkish banking regulation data from BDDK and mevzuat.gov.tr. This page summarizes the current runtime contract; the bilingual README contains the broader feature and development guide.

## Process Profiles and Tool Surface

The server exposes exactly one reviewed profile per process:

| Profile | Selection | Database identity | MCP tools |
|---|---|---|---:|
| Public | `BDDK_TOOL_PROFILE=public` or `bddk-mcp serve --profile public` | `BDDK_DATABASE_URL` | 17 public tools |
| Operator | `BDDK_TOOL_PROFILE=operator` or `bddk-mcp serve --profile operator` | `BDDK_OPERATOR_DATABASE_URL` | 17 public tools plus 14 operator tools, 31 tools total |

The operator profile requires its own DSN and does not fall back to `BDDK_DATABASE_URL`. Run public and operator profiles as separate processes, database roles, service accounts, and network boundaries.

### Public tools (17)

- `search_bddk_regulations`
- `search_document_store`
- `search_bddk_institutions`
- `search_bddk_announcements`
- `get_bddk_document`
- `get_document_history`
- `get_document_section`
- `search_document_sections`
- `resolve_regulation_status`
- `get_amendment_chain`
- `get_cross_references`
- `get_bddk_bulletin`
- `get_bddk_bulletin_snapshot`
- `get_bddk_monthly`
- `analyze_bulletin_trends`
- `get_regulatory_digest`
- `compare_bulletin_metrics`

### Operator additions (14)

- `check_bddk_updates`
- `document_store_stats`
- `bddk_cache_status`
- `refresh_bddk_cache`
- `sync_bddk_documents`
- `trigger_startup_sync`
- `get_operator_job`
- `list_operator_jobs`
- `cancel_operator_job`
- `document_health`
- `health_check`
- `bddk_metrics`
- `backfill_degraded_documents`
- `document_quality_report`

Mutating operator tools return a job receipt immediately. Records, hashed
idempotency keys, numeric progress, and bounded result metrics are durable in
PostgreSQL table `bddk_operator.operator_jobs`. A session-scoped job-admission
lease prevents concurrent runner ownership across processes; a distinct
transaction-scoped corpus-mutation lock serializes sanctioned writer
transactions and the release publisher. Runner tasks still execute inside
the operator process: abandoned `running` work is marked `interrupted` only
after recovery obtains the lease, while `queued` work is never guessed stale
and must be resumed with the same idempotency key or cancelled explicitly. The
OpenShift starter remains one `Recreate` replica until multi-replica and
failover behavior is bank-accepted; this is not a general workflow queue.

## Local Start

For disposable loopback development, let Compose run role/extension setup,
schema-owner migration, DBA grants, and ingestion bootstrap as separate stages,
then start the default public stdio profile:

```bash
export BDDK_JWT_ISSUER=https://idp.invalid
export BDDK_JWT_RESOURCE=https://localhost:8000/mcp
export BDDK_JWT_JWKS_URL=https://idp.invalid/jwks
export BDDK_JWT_AUDIENCE=bddk-mcp-local
docker compose up --build -d bddk-bootstrap
docker compose wait bddk-bootstrap
export BDDK_DATABASE_URL=postgresql://bddk_local_public:local-only-public@localhost:5432/bddk
uv run --frozen bddk-mcp serve --profile public
```

Start a local operator process only with a separately provisioned operator role:

```bash
export BDDK_OPERATOR_DATABASE_URL=postgresql://bddk_local_operator:local-only-operator@localhost:5432/bddk
uv run --frozen bddk-mcp serve --profile operator
```

The reserved `.invalid` JWT values only allow Compose to parse an unused HTTP
service definition; the lifecycle target starts no HTTP server, and those
values are not valid server configuration. The fixed local passwords are
public test fixtures, not production credential recommendations. Put real DSNs
in the platform secret store.

For a bank database, a DBA must first install `vector` and `unaccent` in
`public`, apply `deploy/postgres/01_roles.sql`, and provision distinct LOGIN
identities. Run `bddk-mcp migrate` only with the schema-owner connection that
executes `SET ROLE bddk_schema_owner`, apply `deploy/postgres/02_grants.sql` as
DBA, and only then run `bddk-mcp bootstrap` with the exact ingestion identity.
`bootstrap` imports corpus data, sections, and embeddings into an already
migrated schema; it never migrates.

`verify-corpus` is an optional read-only preflight that opens no database
connection. It does not transfer trust to a later import. A production
promotion passes the numeric-objective and detached-signature gates directly to
`bootstrap`, plus the per-document measurement gate wherever the corpus is
actually monitored (schema v10 also admits an explicitly recorded
quantified-unmeasured release for a batch-snapshot corpus):

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

Before opening a database pool, bootstrap validates the exact
`corpus_scope.yml` and reads document/cache input only from manifest-declared
paths with the declared byte length and hash. It rejects a present but
undeclared `documents.json`, `chunks.json`, or `decision_cache.json`. The trust
key must be delivered through a mount/Secret separate from the corpus. On
completion, bootstrap emits the path-free manifest ID and SHA-256 as operator
evidence and reports that release publication is required; it does not persist
a candidate. Release admission is a two-credential flow:
`verify-and-stage-corpus-release` uses the distinct `bddk_release_verifier`
identity, externally mounted trust key, pinned verifier revision/image digest,
and a 60–3,600 second TTL; `activate-corpus-release` gives the
`bddk_release_publisher` only the resulting opaque request ID. Activation
fails closed after expiry, reuse, or corpus state/epoch/profile drift. The old
`publish-corpus-release` alias is disabled.

The tracked 318-document selection is deliberately non-exhaustive. It is
Ed25519-signed and declares quantified freshness objectives, but its freshness
is not measured against per-document source events, so it activates at the
explicitly weaker `quantified_unmeasured_signature_verified_pass` level. It is a
reviewed development corpus, not a production-freshness claim.

Ordinary migration refuses a pre-ledger unmanaged schema. The explicit
`bddk-mcp migrate --adopt-legacy` option accepts only the exact supported shape
after a proven backup and the
[`docs/LEGACY_DATABASE_UPGRADE.md`](docs/LEGACY_DATABASE_UPGRADE.md) procedure;
it is not a clean-install or general repair flag.

A populated version-2 database also refuses migration 3 by default because its
retrieval-publication backfill takes blocking locks and validates large foreign
keys. Use `--allow-retrieval-publication-backfill` only during a reviewed
maintenance window after stopping workloads, proving a restorable backup, and
rehearsing against a size-matched restore. `BDDK_EXPECTED_DATABASE_NAME` and
the independent DBA-script target setting must match the active database.
Outside the isolated local Compose profile, every PostgreSQL DSN must use
`sslmode=verify-full` and an absolute `sslrootcert` path.

The required automated PostgreSQL and role-contract lanes currently prove
PostgreSQL 17 only. Treat other major versions as unsupported until the
compatibility matrix is explicitly expanded and the bank-selected version
passes the full contract.

## Database and Retrieval Identity

The required NOLOGIN memberships are exact: public gets only
`bddk_public_reader`; ingestion gets only `bddk_ingestion`; release verification
gets only `bddk_release_verifier`; activation/retention gets only
`bddk_release_publisher`; operator gets
`bddk_public_reader`, `bddk_ingestion`, and `bddk_operator_runtime`; optional
telemetry gets only `bddk_telemetry_writer`. Public, ingestion, and operator
startup verify the actual PostgreSQL session, memberships, and effective
object privileges on every physical pool connection. They also reject
privileges sourced from `PUBLIC` or direct LOGIN ACLs instead of the reviewed
group roles. Telemetry separately verifies column-scoped INSERT-only
access and refuses trace reads/changes or broader membership. Differently
spelled DSNs do not bypass these checks.

The default embedding model is pinned to full commit
`d13f1b27baf31030b7fd040960d60d909913633f`; the container saves it locally for
offline runtime use. The immutable database schema is `public.vector(768)` and
rejects every other configured dimension. The optional built-in reranker is
pinned to `1427fd652930e4ba29e8149678df786c240d8825`. Changing the model,
tokenizer, prefixes, or chunk settings requires controlled full re-embedding
and retrieval regression testing.

Retrieval requires a publication record matching the document content hash and
active retrieval-profile hash. Chunk mutation invalidates that record, and
ingestion republishes only after chunk/embedding integrity checks pass. Run the
required bootstrap or controlled reindex after a schema/model-profile upgrade;
unpublished or stale chunks fail closed instead of entering results.

Migration v0005 persists independently revalidated corpus releases and atomic
activations. Mutations advance an epoch that invalidates the active view, and
strict local-corpus calls verify one release before and after execution. This
is the active-identity boundary. Migration v0007 additionally lets the distinct
release-publisher identity run
`bddk-mcp retain-corpus-generation --expected-release-id ...` to copy and seal
the exact active state across 17 typed retained relations. The physical
generation and seal are derived from the corpus state and retrieval profile:
separately governed releases over that
same exact state/profile receive distinct release bindings to the same retained
generation and seal, rather than duplicate storage. V7 makes retained-row and
both current/retained state hashing independent of session formatting by fixing
function-local `TimeZone=UTC`, `DateStyle=ISO, YMD`,
`IntervalStyle=postgres`, `bytea_output=hex`, and `extra_float_digits=3`.
A release published before v7 under different settings makes the v7 migration
fail closed before the hash function changes. The current binary disables the
old direct-publication CLI. An unchanged v5/v6 database that hits this exact
guard therefore requires the separately approved, digest-pinned pre-v8
remediation described in the legacy-upgrade runbook, or the blue-green
data-only path; never manufacture a historical release row or binding. After
v8, the verifier stages short-lived evidence and the publisher activates only
its one-time request ID. These administrative CLIs are not MCP tools.
Generation-bound serving and authorized rollback remain H2-02B. Backup
growth is still `not_measured`, and bank retention/capacity approval remains
open. The tracked corpus manifest is being re-signed: the v5 section parser regenerates 10,483 chunks, so the previously signed 9,675-chunk artifact is superseded and the staged manifest is unsigned pending owner review (gap register CUR-018). Migration v0010 admits exactly two freshness policy levels,
both requiring quantified objectives and a verified signature; the tracked
corpus activates at the explicitly unmeasured level.

Migration v0004 supplies the eleven-table canonical legal pilot; v0006 adds the
public abstention-first legal-status resolver. The synthetic PostgreSQL proof
does not establish a real regulation family's currentness.

The expert set is non-release. Its gate requires four signed layers: measured
corpus, expert dataset, exact-Citation legal-curator attestation, and a retained
source/acquisition/page/excerpt legal-release checkpoint. Canonical signer
fingerprints must be distinct. Current preflight uses operator-supplied anchors
and therefore authorizes neither bank use nor model scores; key rotation, named
reviewer policy, adjudication, and expert-case execution remain open.

## Streamable HTTP Contract

The HTTP transport is selected with `MCP_TRANSPORT=streamable-http`. It serves MCP at `/mcp` using stateless JSON responses. Loopback is the default bind.

Only two fixed, content-free HTTP probe routes bypass MCP authentication and Host/Origin admission so an orchestrator can use a pod-IP Host header:

- `GET /health/live` returns process liveness.
- `GET /health/ready` periodically re-attests migrations, critical catalog objects, corpus readiness, workload ACLs, and optional operator/telemetry storage; it returns 503 on drift or unavailability.

Both unauthenticated probe routes remain subject to the process rate and
concurrency admission limits.

A non-loopback bind fails at startup unless all of the following are configured:

- exact `BDDK_HTTP_ALLOWED_HOSTS` values, without wildcards;
- exact HTTPS `BDDK_HTTP_ALLOWED_ORIGINS` values, without wildcards;
- complete `BDDK_JWT_ISSUER`, `BDDK_JWT_RESOURCE`, `BDDK_JWT_JWKS_URL`, and `BDDK_JWT_AUDIENCE` settings;
- asymmetric `BDDK_JWT_ALGORITHMS`, approved `BDDK_JWT_ACCESS_TOKEN_TYPES`, and at least one `BDDK_JWT_REQUIRED_SCOPES` value;
- the profile scope: `bddk.read` for public or `bddk.operator` for operator;
- `BDDK_OPERATOR_REMOTE_ENABLED=true` for a non-loopback operator process.

`BDDK_HTTP_ALLOW_UNAUTHENTICATED` is a supported explicit opt-in for serving a
non-loopback bind without bearer authentication. It is unset by default, and
while it is unset the fail-closed requirements above are unchanged. When set,
it cannot be combined with any configured `BDDK_JWT_`-prefixed setting —
startup refuses and names the offending variables — and it is refused outright
for a non-loopback operator profile regardless of
`BDDK_OPERATOR_REMOTE_ENABLED`: operator tools stay authenticated or
loopback-only by construction. An unauthenticated server advertises no OAuth
discovery: there is no `WWW-Authenticate` challenge, and both the
protected-resource and authorization-server well-known routes return 404.
Host, Origin, body-size, concurrency, and rate limits still apply.

`BDDK_TLS_CERT_FILE` and `BDDK_TLS_KEY_FILE` optionally enable HTTPS at the
application socket. They are an inseparable PEM pair and are validated before
the listener opens. The OpenShift starter supplies them with service-serving
certificates and uses a re-encrypt Route; certificate rotation currently
requires a controlled pod restart.

Do not copy example identity-provider URLs from tests into a deployment. Obtain issuer, resource, JWKS, audience, token-type, and scope values from the bank-approved identity design. The composed remote application publishes RFC 9728 metadata at `/.well-known/oauth-protected-resource/mcp`, and its 401 challenge identifies that URL through `resource_metadata`. Tokens are bound to the API through the exact configured audience. This is application-level MCP authorization discovery, not bank IdP client registration or end-to-end flow acceptance. Use a dedicated resource-server audience rather than an interactive client's ID-token audience.

Host, Origin, content type, and request-body size are checked before bearer authentication. Valid JWT access tokens are verified against JWKS, issuer, exact resource-server audience, approved JOSE type, required/optional time claims, algorithm, and scope authorization. Tokens need not carry a custom `resource` claim or `nbf`; when `nbf` is present it is validated. The fail-closed default accepts only RFC 9068 `at+jwt`; generic Keycloak-style `JWT` requires the explicit `BDDK_JWT_ACCESS_TOKEN_TYPES=at+jwt,JWT` compatibility opt-in and a dedicated API audience.

`BDDK_HTTP_MAX_BODY_BYTES`, `BDDK_HTTP_MAX_CONCURRENCY`, and `BDDK_HTTP_RATE_LIMIT_PER_MINUTE` provide coarse protection inside each application process. Rate identity comes from the ASGI peer address; untrusted forwarding headers are not accepted by default. `BDDK_HTTP_TRUSTED_PROXY_HOPS` (default `0`) controls that rate-limit client key: at `0` the limiter keys on the socket peer and ignores `X-Forwarded-For` entirely, which is correct for a directly exposed bind. Behind `n` operator-controlled reverse proxies, set the real hop count; the key is then the `n`-th entry from the right of the combined forwarded list, and anything unusable degrades to a shared `unknown` bucket rather than falling back to the socket peer, so a spoofed header cannot merge distinct callers. A wrong hop count makes the limiter either shared or spoofable, and when authentication is disabled through the explicit opt-in the rate limiter is the primary abuse control. These controls are neither shared across replicas nor a global ingress limit. OpenShift deployments still need an approved end-to-end TLS topology, identity-aware ingress, shared request/rate policy, audit events, and NetworkPolicy controls.

Live regulatory fetch paths separately enforce exact BDDK/mevzuat HTTPS hosts,
public-address DNS checks, destination revalidation on bounded redirects,
streamed response limits by artifact class, and retry logs without raw URLs,
queries, or exception messages. DNS validation and socket connection are not
atomic, so the target OpenShift environment must enforce matching egress with
NetworkPolicy and/or an approved proxy/firewall. Public institution,
announcement, bulletin, and update tools can call live BDDK sources; therefore
both public and operator runtimes require narrowly approved regulatory-source
or proxy TCP 443 egress. Lifecycle Jobs require DNS/PostgreSQL only and must not
receive that source egress.

The runtime wheel/sdist exclude `seed_data`, tests, benchmark code, and
deployment assets. The provided container explicitly includes the reviewed
seed corpus; a wheel deployment must mount an approved corpus and pass
`--seed-dir` or `BDDK_SEED_DIR` to `bootstrap`.

The repository has not yet been accepted on the bank's OpenShift AI cluster,
proven through the bank backup/restore process, or certified across a
release-specific Claude, Codex, GPT, LM Studio, GPT-OSS, and local-model client
matrix. The included client configurations and OpenShift manifests are starter
evidence, not certification.

Repository preflight binds the exact checksum-verified Kustomize v5.8.1 binary
and strict rendered/security inventories. It renders the exact
`deploy/openshift-overlays/bank-bootstrap` contract, including direct strict
bootstrap flags, a read-only approved-corpus PVC, and a separate read-only
corpus-trust Secret, but deliberately leaves actual bank provisioning and all
eight live gates unrun. Supply-chain evidence builds with Buildx `--provenance=false --load`,
binds descriptor/manifest/config/loaded-image/Syft identities, and emits an
unsigned repository SLSA envelope. Pending exceptions never make a result
promotion eligible; signing, admission, and registry promotion remain bank
controls.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) before enabling any remote profile.
