# Deployment Guide

This guide describes the deployment paths the repository currently supports and the remaining controls for a bank on-premises deployment.

## Runtime Profiles

The server selects exactly one reviewed tool profile per process.

| Mode | Selection | Database identity | Current tool surface |
|---|---|---|---:|
| Public stdio | `bddk-mcp serve --profile public` (default) | `BDDK_DATABASE_URL` | 15 public tools |
| Operator stdio | `bddk-mcp serve --profile operator` | `BDDK_OPERATOR_DATABASE_URL` | 15 public + 13 operator = 28 tools |
| Public Streamable HTTP | Add `--transport streamable-http` | `BDDK_DATABASE_URL` | 15 public tools |
| Operator Streamable HTTP | Operator profile plus explicit remote opt-in when non-loopback | `BDDK_OPERATOR_DATABASE_URL` | 28 total tools |

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

BDDK_INGESTION_DATABASE_URL='postgresql://INGESTION:SECRET@HOST:5432/DATABASE?sslmode=verify-full&sslrootcert=%2FAPPROVED%2Fpostgres-ca.crt' \
  uv run --frozen bddk-mcp bootstrap --reindex-existing
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
coverage.

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

### Database identity contract

The repository SQL creates NOLOGIN group roles; the bank must create and rotate
the authenticating LOGIN roles outside Git. The required effective memberships
are:

| Identity | Required group-role memberships | Used for |
|---|---|---|
| Schema owner | `bddk_schema_owner`, entered through `SET ROLE` | `bddk-mcp migrate` only |
| Ingestion | exactly `bddk_ingestion` | `bootstrap`, synchronization, reindex and corpus backfill |
| Public runtime | exactly `bddk_public_reader` | public MCP process |
| Operator runtime | exactly `bddk_public_reader`, `bddk_ingestion`, `bddk_operator_runtime` | operator MCP process and its current in-process runners |
| Telemetry writer | exactly `bddk_telemetry_writer` | optional column-scoped trace inserts only |

Public, operator, and ingestion startup inspect the real PostgreSQL session
identity, memberships, database/schema/object privileges, and unexpected
managed objects; a differently written DSN for the same over-privileged LOGIN
does not bypass the check. Telemetry performs a separate exact INSERT-only
identity check. Public/operator pools also run the bounded identity assertion on
every newly opened connection, so pool growth or a changed database endpoint
cannot introduce an unchecked session. The application never grants these
roles itself. See
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

Loopback (`127.0.0.1`) is the default bind and is the appropriate development default. A non-loopback bind, including `0.0.0.0` in a container, fails startup unless all remote security settings are complete:

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

The issuer, resource, JWKS, audience, token type, and scope values are deployment-specific. Obtain them from the bank-approved identity design; do not copy values from repository tests. `BDDK_JWT_RESOURCE` identifies this protected resource to MCP clients; access-token binding is enforced through `BDDK_JWT_AUDIENCE`. Configure a dedicated API/resource-server audience distinct from interactive OAuth client audiences so a Keycloak-style ID token cannot be confused with an access token. Keep the fail-closed `at+jwt` default when the issuer supports RFC 9068. Set `BDDK_JWT_ACCESS_TOKEN_TYPES=at+jwt,JWT` only when generic `JWT` is required by the approved Keycloak/IdP profile.

Host, Origin, content type, duplicate headers, and request-body size are validated before bearer authentication. JWT verification then bounds token length and checks the asymmetric signature, key ID/algorithm, access-token type, issuer, exact resource-server audience, expiry, optional not-before time, and subject. A custom token `resource` claim is neither required nor trusted. Missing or invalid authentication returns 401; a cryptographically valid token without the profile's required scope returns 403.

The application has coarse per-process overload controls:

| Variable | Default | Behavior |
|---|---:|---|
| `BDDK_HTTP_MAX_BODY_BYTES` | 1,048,576 | Rejects oversized POST bodies with 413 |
| `BDDK_HTTP_MAX_CONCURRENCY` | 32 | Immediately rejects excess in-process work with 503 and `Retry-After` |
| `BDDK_HTTP_RATE_LIMIT_PER_MINUTE` | 120 | Fixed-window limit per ASGI peer, returning 429 and `Retry-After` |
| `BDDK_JWT_MAX_TOKEN_LENGTH` | 16,384 | Rejects oversized bearer tokens before verification |

The rate and concurrency state is bounded, process-local and non-durable. It is not shared across workers or replicas and does not provide a global ingress limit. Uvicorn proxy-header trust is disabled, and the limiter does not accept client-supplied `Forwarded` or `X-Forwarded-For` values. A bank deployment still needs shared ingress request/rate policy, an approved end-to-end TLS topology, audit events, and abuse controls at the approved gateway.

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

The operator profile adds 13 tools to the 15 public tools. Mutating tools such as cache refresh, synchronization, startup synchronization, and executed backfill return an immediate job receipt. Use `get_operator_job`, `list_operator_jobs`, and `cancel_operator_job` to inspect or request cancellation.

The server uses `PostgresJobRepository` in the operator profile. The global v2
migration creates `bddk_operator.operator_jobs`; the repository persists job
state, argument fingerprints, hashed idempotency keys, numeric progress, bounded
result metrics, safe error codes, and retained terminal history. It deliberately
does not persist raw tool arguments, document text, or exception messages. A
session-level PostgreSQL advisory lease named for the corpus-mutation resource
serializes the current mutating runners across processes. The operator pool
therefore requires `BDDK_PG_POOL_MAX >= 2`: the lease pins one connection while
state/progress writes use another.

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

The Compose and image definitions bind HTTP on `0.0.0.0`. Under the fail-closed HTTP policy, a container must receive the exact Host/Origin and complete JWT settings above or startup will be refused. Development credentials and database ports from Compose must never be reused or exposed outside a developer environment.

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
rules where required. The repository starter default-denies egress for all BDDK
MCP pods but deliberately supplies no environment-specific allowlist. Without a
bank overlay allowing the required destinations, lifecycle and runtime
connectivity is expected to fail closed.

## OpenShift AI Starter

[`deploy/openshift`](../deploy/openshift/README.md) now provides a conservative
starter: separate public/operator Deployments, Services and service accounts;
OpenShift service-serving certificates at both application sockets; a
public-only re-encrypt TLS Route; an internal HTTPS operator Service; exact
HTTPS probes; non-root/read-only security contexts; resource bounds; restricted
ingress plus default-deny egress; exact Secret references; a mounted PostgreSQL
CA contract; and separately applied migration/bootstrap Jobs. Every workload
uses a fail-closed image-digest placeholder, the image uses a digest-pinned `uv`
source, the embedding-model revision is pinned, and the default non-root UID is
compatible with OpenShift's arbitrary-UID model. Version labels are excluded
from immutable selectors.

This is repository-level implementation evidence, not bank acceptance or a production-ready platform configuration,
including for OpenShift AI. No target
bank cluster, SCC, namespace policy, storage class, PostgreSQL service, IdP,
internal registry, or enterprise proxy was available for this review. Every `REPLACE_*` value must be
resolved through the bank design, and Secret examples must never contain real
values in Git. The manifests still require validation against the bank's SCC,
trust-bundle injection, registry/image policy, ingress, IdP, namespace labels,
capacity model, backup/restore, disaster recovery, and release controls. The
included policy default-denies egress; environment-specific allow rules are not
supplied. Operator job records and advisory leases are durable, but the
in-process runner/failover behavior remains unaccepted, so the starter
deliberately uses one `Recreate` operator replica.

The automated database lifecycle and role-isolation lane currently exercises
PostgreSQL 17. Other PostgreSQL versions may work, but remain unaccepted until
the same migration, extension, privilege, ingestion, and rollback checks pass.

For the target bank deployment:

1. Provision a dedicated external PostgreSQL database, install `vector` and
   `unaccent` in `public`, and create distinct schema-owner, ingestion, public,
   operator, and optional telemetry LOGIN identities using the reviewed group
   roles. Apply both DBA SQL files with the independent target-database guard
   and run migration with `BDDK_EXPECTED_DATABASE_NAME`.
2. Replace the application image placeholder with the scanned immutable digest,
   create `bddk-mcp-postgres-ca` with key `ca.crt`, and keep every workload DSN
   at `sslmode=verify-full` with the mounted absolute CA path.
3. Run public and operator profiles as separate workloads; inject `BDDK_DATABASE_URL` only into public and `BDDK_OPERATOR_DATABASE_URL` only into operator.
4. Keep remote operator disabled unless a private operator Route, `bddk.operator` authorization, and explicit `BDDK_OPERATOR_REMOTE_ENABLED=true` have been approved.
5. Add bank-specific least-privilege egress for DNS, PostgreSQL, IdP/JWKS,
   approved BDDK/Mevzuat hosts and the enterprise proxy before starting selected
   pods; retain the checked-in default deny.
6. Review and adapt the starter, then validate the exact `/health/live` and `/health/ready` probes in a disposable bank-like namespace.
7. Validate the re-encrypt Route-to-pod handshake and operator Service TLS against the injected OpenShift service CA; retain the application's JWT checks and place shared request limits and audit events at the approved ingress boundary.
8. Prove backup, point-in-time/selected restore, schema upgrade, legacy refusal
   or adoption where relevant, and rollback/cutover procedures in an isolated
   environment.
9. Run release-specific MCP discovery, schema, authentication, tool-call,
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
