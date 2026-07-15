# OpenShift deployment starter

This directory is a security-oriented starter, not a bank-approved production
configuration. It deliberately contains `REPLACE_*` values and references
Secrets and a PostgreSQL CA ConfigMap that are not part of
`kustomization.yaml`; the application will fail closed until the bank supplies
its IdP, Route, registry, CA trust, database name and database roles. Every
workload and lifecycle Job uses the fail-closed image reference
`REPLACE_IMAGE_REGISTRY/bddk-mcp@sha256:REPLACE_64_HEX_IMAGE_DIGEST`; replace it
with the exact digest of the scanned release image, never a tag.

Release order:

1. Build and scan the image, replace its digest in every manifest, and resolve
   every other `REPLACE_*` value, including the exact
   `REPLACE_DATABASE_NAME` used by the migration target guard.
2. Provision a dedicated PostgreSQL database and approved extensions. Apply
   `deploy/postgres/01_roles.sql` with
   `PGOPTIONS='-c bddk.expected_database=DATABASE'`, then bind bank-managed
   LOGIN identities to the reviewed NOLOGIN group roles in
   `deploy/postgres/README.md`.
3. Provision the four required Secret names shown in `secrets.example.yaml`
   through the approved secret manager. Every PostgreSQL DSN must retain
   `sslmode=verify-full` and the encoded absolute `sslrootcert` path shown in
   the example. Never apply that example with real values in Git.
4. Create a ConfigMap named `bddk-mcp-postgres-ca` with the approved PostgreSQL
   trust root under key `ca.crt`. This is distinct from the OpenShift service CA
   used for the MCP application sockets.
5. Before creating a selected pod, apply a bank-specific egress NetworkPolicy
   that permits only the required DNS, PostgreSQL, IdP/JWKS, approved
   BDDK/Mevzuat and enterprise-proxy destinations. The checked-in base adds a
   default-deny egress policy but cannot supply environment-specific allows.
6. Apply `jobs/migrate.yaml` and wait for successful completion. Then apply
   `deploy/postgres/02_grants.sql` as the database administrator with the same
   independent `bddk.expected_database` target guard.
7. Apply `jobs/bootstrap.yaml` and wait for successful completion. Its
   `--reindex-existing` argument rebuilds and publishes every canonical
   document under the active retrieval profile, including an existing corpus
   made unpublished by migration v0003.
8. Apply the runtime resources with `oc apply -k deploy/openshift`.
9. Wait for the service-serving certificate controller to create
   `bddk-mcp-public-tls` and `bddk-mcp-operator-tls` and for both Deployments to
   become ready.
10. Verify the public Route's re-encrypt handshake, the internal operator
   Service's certificate chain, `/health/live`, `/health/ready`, JWT
   rejection/acceptance, public tool discovery, and denial of operator tools
   through the public Route.

The supplied migration Job runs ordinary `bddk-mcp migrate`; this is the only
correct mode for a clean database. If a pre-ledger database must be preserved,
do not add `--adopt-legacy` casually or edit the migration ledger. Stop every
workload, prove a restorable bank backup, follow
[`docs/LEGACY_DATABASE_UPGRADE.md`](../../docs/LEGACY_DATABASE_UPGRADE.md), and
run a separately reviewed one-time Job with the explicit flag. Refusal requires
the documented blue-green data-only path.

A populated v0002 database also requires a separately reviewed one-time Job:
the checked-in clean-install Job deliberately omits
`--allow-retrieval-publication-backfill`. Stop serving and ingestion, prove a
restorable backup, rehearse against a size-matched restore, and only then add
that flag for the v0003 migration. The standard bootstrap Job subsequently
runs `--reindex-existing` so validated retrieval publications exist before
serving.

The migration Job receives `BDDK_SCHEMA_OWNER_DATABASE_URL` and the independent
`BDDK_EXPECTED_DATABASE_NAME`; the bootstrap Job receives only
`BDDK_INGESTION_DATABASE_URL`. Runtime Deployments receive their own
public/operator DSN and never receive either lifecycle DSN. Baseline workloads
do not reference the optional `bddk-mcp-telemetry-db` Secret and keep telemetry
disabled. Enabling it requires the explicit
`deploy/openshift-overlays/telemetry` Kustomize overlay and a distinct LOGIN
that inherits only `bddk_telemetry_writer`; a missing approved telemetry Secret then fails
deployment rather than silently changing the data flow. This starter does not
create or rotate any bank credential. Render or apply that profile with
`oc apply -k deploy/openshift-overlays/telemetry`.

The schema-owner connection must authenticate as a separate restricted LOGIN
and enter exactly `bddk_schema_owner` through `SET ROLE`. Migration refuses the
wrong database name, an administrative or multiply affiliated identity, and
an unexpected ownership/privilege shape. Runtime readiness also attests the
critical constraints, functions, triggers, FTS indexes and HNSW options; a
successful `SELECT 1` alone cannot make a drifted catalog ready.

Both application sockets use OpenShift service-serving certificates. The
public Route uses `reencrypt`, so client-to-router and router-to-pod traffic is
encrypted. The operator Service has no Route and listens through HTTPS on
service port 443. `service-ca.yaml` receives the namespace's injected CA bundle
for validation jobs. An approved operator client in another namespace must
create its own CA-injected ConfigMap, mount `service-ca.crt`, validate the
Service DNS name, and send a correctly scoped bearer token; ConfigMaps cannot
be shared across namespaces.

The service certificate operator rotates the generated Secret, but the current
Uvicorn process does not hot-reload certificates. Configure the bank-approved
restart/reloader mechanism and acceptance-test normal certificate rotation and
service-CA rollover. A successful pod probe alone does not prove that the
router or an operator client validates the intended certificate chain.

The operator Service intentionally has no Route and one replica. Job records
are durable in PostgreSQL and a session advisory lease serializes corpus
mutation. A restart marks abandoned running work interrupted; a persisted
queued job is never guessed stale and can be resumed by retrying the same
idempotency key or cancelled explicitly. Keep `Recreate`/one replica until the
bank acceptance suite covers overlapping-pod and multi-replica failover.

The included NetworkPolicies restrict ingress and default-deny all egress for
pods labeled `app.kubernetes.io/name=bddk-mcp`. No generic egress allowlist is
safe to ship because bank addresses, namespace labels and proxy topology are
environment-specific. Without the bank-specific allow policy, lifecycle Jobs
cannot reach PostgreSQL and runtime pods cannot reach DNS, PostgreSQL or the
IdP/JWKS endpoint; readiness failure is expected. Cluster ingress must also
provide global/client-aware rate limiting because the application limiter sees
the router peer and is only a per-process guard.

Regulatory outbound code accepts only exact BDDK/mevzuat HTTPS hosts, performs
public-address DNS checks, revalidates every bounded redirect, streams into
artifact-specific limits (up to 128 MiB for approved document classes), and
omits URL/query/exception text from retry logs. This does not eliminate the
DNS-to-connect race or replace platform egress enforcement, malware scanning,
or source-validation controls.

The image explicitly includes the repository's reviewed `seed_data` and the
embedding model downloaded at full commit
`d13f1b27baf31030b7fd040960d60d909913633f`. The immutable database dimension is
768; another configured dimension fails startup. The optional default reranker
revision is `1427fd652930e4ba29e8149678df786c240d8825`. A model or chunk-profile
change requires controlled re-embedding and retrieval regression testing.

OpenShift assigns namespace-specific UID ranges. The image has a non-root
default UID and group-0-compatible read permissions, while the manifests avoid
pinning an out-of-range UID. Validate the manifests against the bank's SCC,
NetworkPolicy implementation, trust bundle injection, image policy and
OpenShift AI conventions before promotion. Release labels intentionally do not
enter immutable Deployment, Service or NetworkPolicy selectors.

No bank OpenShift AI cluster, bank PostgreSQL backup/restore workflow, or
release-specific MCP client/model compatibility matrix was available for this
starter. Prove those controls—including restore, rollback, certificate
rotation, database failover, client discovery/authentication/tool calls, and
citation output—in an isolated bank-like namespace before promotion.

Platform references: [OpenShift service-serving certificate and injected CA
configuration](https://docs.redhat.com/en/documentation/openshift_container_platform/4.22/html/security_and_compliance/configuring-certificates)
and [OpenShift re-encrypt Route behavior](https://docs.redhat.com/en/documentation/openshift_container_platform/4.20/html/ingress_and_load_balancing/routes).
