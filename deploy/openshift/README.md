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

The checked-in lifecycle inventory is exactly four Jobs, and production must
execute them serially: `bddk-mcp-migrate-v5-0-1` →
`bddk-mcp-bootstrap-v5-0-1` → `bddk-mcp-verify-stage-release-v5-0-1` →
`bddk-mcp-activate-release-v5-0-1`. DBA grant reconciliation runs after the
migration Job and before bootstrap; it is not a Kubernetes Job in this starter.

1. Build and scan the image, replace its digest in every manifest, and resolve
   every other value available before staging, including the exact
   `REPLACE_DATABASE_NAME` used by the migration target guard. Do not invent
   `REPLACE_RELEASE_REQUEST_ID`; resolve it only from the verifier output in
   step 8.
2. Provision a dedicated PostgreSQL database and approved extensions. Apply
   `deploy/postgres/01_roles.sql` with
   `PGOPTIONS='-c bddk.expected_database=DATABASE'`, then bind bank-managed
   LOGIN identities to the reviewed NOLOGIN group roles in
   `deploy/postgres/README.md`.
3. Provision the six required database Secret names shown in
   `secrets.example.yaml`
   through the approved secret manager. Every PostgreSQL DSN must retain
   `sslmode=verify-full` and the encoded absolute `sslrootcert` path shown in
   the example. Never apply that example with real values in Git.
4. Create a ConfigMap named `bddk-mcp-postgres-ca` with the approved PostgreSQL
   trust root under key `ca.crt`. This is distinct from the OpenShift service CA
   used for the MCP application sockets.
5. Before creating a selected pod, apply the bank-specific exact egress
   NetworkPolicies. Public and operator require DNS, PostgreSQL, and TCP 443 to
   the approved regulatory-source destination or enterprise proxy; public
   access is required because institution, announcement, bulletin, and update
   tools can call live BDDK services. Only the operator runtime additionally
   requires IdP/JWKS egress — the public profile is unauthenticated and
   verifies no tokens. Lifecycle Jobs require DNS and
   PostgreSQL only and must not receive regulatory-source/proxy egress. The
   checked-in base adds default deny but cannot supply bank addresses or peer
   selectors.
6. Apply `jobs/migrate.yaml` and wait for successful completion. Then apply
   `deploy/postgres/02_grants.sql` as the database administrator with the same
   independent `bddk.expected_database` target guard.
7. Provision PVC `bddk-mcp-approved-corpus` and Secret
   `bddk-mcp-corpus-trust` with key `ed25519-public-key.pem` through approved
   bank controls. Use the exact
   `deploy/openshift-overlays/bank-bootstrap` contract: it mounts the PVC and
   Secret separately and read-only, and passes `--require-quantified-freshness`,
   `--require-measured-freshness`, `--require-verified-signature`, and the
   mounted `--trusted-signing-key` directly to bootstrap. After migration and
   grants have succeeded, apply the accepted strict bootstrap Job through a
   lifecycle mechanism that preserves that order, wait for completion, and
   retain its path-free manifest ID/SHA output. The overlay renders runtime and
   all four lifecycle resources together for exact preflight; that is
   not authorization to start lifecycle Jobs concurrently. The base
   `jobs/bootstrap.yaml` remains a development/baseline Job and is not a
   production trust gate.
8. Apply `jobs/verify-stage-release.yaml` only after strict bootstrap succeeds.
   Its `bddk-mcp-release-verifier` ServiceAccount and
   `bddk-mcp-release-verifier-db` Secret (key
   `BDDK_RELEASE_VERIFIER_DATABASE_URL`) are distinct from the publisher. Supply
   the exact image/source provenance through
   `BDDK_RELEASE_VERIFIER_IMAGE_DIGEST` and
   `BDDK_RELEASE_VERIFIER_REVISION_SHA256`; keep
   `BDDK_RELEASE_VERIFICATION_VALIDITY_SECONDS` between 60 and 3,600 seconds
   (the manifest default is 900). It mounts the approved corpus and trust Secret
   read-only, proves exact database membership/state/epoch, and emits a
   `corpus_release_request_sha256_...` request ID plus its expiry.
   Keep those as separate volume sources and paths: verification rejects a
   trust-key path supplied or resolved beneath the corpus root.
9. Before that request expires, set the exact ID as
   `release.release_request_id` in the secret-free acceptance input and as
   the resolved value of `BDDK_RELEASE_REQUEST_ID` (replacing
   `REPLACE_RELEASE_REQUEST_ID`) in `jobs/activate-release.yaml`; rerun offline
   preflight for the exact resolved activation manifest, then apply only that
   Job. It uses the separate `bddk-mcp-release-publisher` ServiceAccount and
   `bddk-mcp-release-publisher-db` Secret (key
   `BDDK_RELEASE_PUBLISHER_DATABASE_URL`). The activation pod must receive only
   its publisher DSN, PostgreSQL CA, bounded temporary storage, and request ID—no
   corpus PVC, trust Secret/key, manifest/signature, verifier DSN, or verifier
   role. An expired, used, or changed-state request requires a fresh verifier
   Job. Never substitute the disabled `publish-corpus-release` alias.
10. Apply the runtime resources with `oc apply -k deploy/openshift`.
11. Wait for the service-serving certificate controller to create
   `bddk-mcp-public-tls` and `bddk-mcp-operator-tls` and for both Deployments to
   become ready.
12. Verify the public Route's re-encrypt handshake, the internal operator
   Service's certificate chain, `/health/live`, `/health/ready`, operator JWT
   rejection/acceptance, public tool discovery, and denial of operator tools
   through the public Route. The public profile serves the department's
   LDAP-authenticated Open WebUI frontend with
   `BDDK_HTTP_ALLOW_UNAUTHENTICATED=true`, so also prove that the Route is
   reachable only from the approved bank network segment; that network
   boundary replaces per-request bearer authentication for read-only tools.

Before activation or runtime apply, copy `acceptance.example.yaml` and
`acceptance-egress.example.yaml` to a secret-free release workspace, resolve
every placeholder (including the installed binary's SHA-256 and the actual
unexpired verifier request ID), install the checksum-verified standalone
Kustomize v5.8.1 used by CI, and run:

```console
uv run python scripts/openshift_acceptance.py --config /path/to/acceptance.yaml
```

A complete release preflight cannot exist before verification has staged the
request ID. Execute migrate, grant reconciliation, strict bootstrap, and the
verifier Job through their separately reviewed lifecycle gate first; then run
preflight against the exact final activation/runtime release workspace. Never
use a dummy request ID to obtain a passing report, and never apply the combined
overlay as a way to impose ordering.

The offline harness hashes the resolved Kustomize executable and requires that
digest to equal `release.kustomize_binary_sha256` in addition to requiring the
exact v5.8.1 version. It validates immutable image references, release and
rollback metadata, Route/TLS and JWT claim inputs, exact rendered-object and
NetworkPolicy inventories, exact selectors/labels/namespaces, exact
runtime/lifecycle database Secret and ConfigMap key boundaries, PostgreSQL CA
mounts, commands, ports, probes, volumes, and restrictive pod/container
security contexts. Sidecars, init/ephemeral containers, command overrides,
host namespace sharing, extra Secret/ConfigMap injection, broadened ingress or
egress, and omitted Kustomize resources fail closed. It prints only
digests, hashed environment identity and named check results. A successful
result is deliberately named `preflight_passed_external_gates_pending`; it
always records the live cluster, IdP, PostgreSQL, network, backup/restore and
client/model exercises as `not_run`. Do not treat repository preflight evidence
as bank acceptance.

The command builds a private temporary copy of the reviewed bank-bootstrap
overlay with the pinned Kustomize renderer, adds the declared egress resources,
applies the configured namespace, and never writes into the checkout or
applies to a cluster. It rejects a missing runtime resource, unresolved
placeholder, wrong renderer
version, or operator Service DNS that does not match the namespace. Build the
same overlay from the exact accepted release values, run preflight against that
release checkout, and retain the emitted input and rendered manifest hashes
with the release evidence.

The acceptance inventory recognizes the exact base and four-Job lifecycle
Kustomizations plus `deploy/openshift-overlays/bank-bootstrap`. It fails closed
if the strict bootstrap/verifier arguments, approved-corpus PVC, corpus-trust
Secret, read-only mounts, verifier provenance/TTL, role-specific DSN Secret,
request ID, or separate volume sources drift. It also rejects any corpus/trust
mount or verifier credential on the activation Job. The focused
acceptance/manifest contract uses checksum-pinned Kustomize v5.8.1. This
proves the repository render and policy contract only: the PVC and Secret must
still be provisioned, and the lifecycle Job must run successfully in an
isolated bank-like namespace before production use.

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

The current ledger ends at schema v8. An ordinary stopped-workload v7 → v8
migration preserves existing releases and retained-generation evidence, adds
the append-only request/binding relations and role-separated facades, and
revokes non-owner execution of the v5 direct-publication routine. Reapply
`02_grants.sql` before any verifier, publisher, or runtime starts. Current
workload identity/catalog admission is v8-only; a v7 database is an upgrade
source, not a compatible runtime target.

If the earlier v7 canonical-hash guard refuses an active v5/v6 release, keep the
pre-v7 database unchanged and follow the separately approved exact-schema
publication remediation before retrying v7 and then v8. The current
`publish-corpus-release` CLI is disabled even though exact v5/v6/v7 compatibility
checks remain in code for reviewed migration remediation. Do not re-grant the
direct routine on v8, update historical release rows, synthesize retention
bindings, or run serving/retention during remediation.

The migration Job receives `BDDK_SCHEMA_OWNER_DATABASE_URL` and the independent
`BDDK_EXPECTED_DATABASE_NAME`; the bootstrap Job receives only
`BDDK_INGESTION_DATABASE_URL`; the verifier's environment receives its verifier
DSN plus non-secret revision/image/TTL provenance; and the activator's
environment receives its publisher DSN plus the staged request ID. Runtime
Deployments receive their own
public/operator DSN and never receive a lifecycle DSN. The verifier and
publisher must use distinct LOGINs, Secrets, ServiceAccounts, and custodians;
neither role may inherit the other. Baseline workloads
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
are durable in PostgreSQL; a session advisory lease controls runner admission,
while a distinct transaction advisory lock serializes sanctioned corpus writers
and corpus verification/activation/retention state changes. A restart marks
abandoned running work interrupted; a persisted queued job is never guessed
stale and can be resumed by retrying the same idempotency key or cancelled
explicitly. Keep `Recreate`/one replica until the bank acceptance suite covers
overlapping-pod and multi-replica failover.

The included NetworkPolicies restrict ingress and default-deny all egress for
pods labeled `app.kubernetes.io/name=bddk-mcp`. No generic egress allowlist is
safe to ship because bank addresses, namespace labels and proxy topology are
environment-specific. The acceptance matrix requires DNS/PostgreSQL for every
component, IdP/JWKS for the operator runtime only, and approved
regulatory-source or proxy TCP 443 for both public and operator; it forbids
giving that source reach to lifecycle Jobs and forbids IdP/JWKS egress on the
unauthenticated public runtime. Without the bank-specific allow policies,
connectivity and readiness failure are expected. Because the public profile
carries no bearer authentication, ingress restriction is a security control,
not only hygiene: keep the public Route reachable solely from the approved
department/frontend segment. Cluster ingress must also
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

The retained
[local PostgreSQL 17 v8 recovery drill](../../docs/evidence/LOCAL_PG17_V8_RECOVERY_DRILL.md)
passed across two disposable clusters for 53 managed objects and seven LOGIN
profiles including the verifier. It is synthetic repository evidence, not bank
backup/PITR, TLS/HBA, custody, target-size RPO/RTO, or capacity acceptance. No
bank OpenShift AI cluster, bank PostgreSQL backup/restore workflow, or
release-specific MCP client/model compatibility matrix was available for this
starter. Prove those controls—including bank-boundary v8 restore, rollback,
certificate rotation, database failover, client
discovery/authentication/tool calls, and citation output—in an isolated
bank-like namespace before promotion.

Platform references: [OpenShift service-serving certificate and injected CA
configuration](https://docs.redhat.com/en/documentation/openshift_container_platform/4.22/html/security_and_compliance/configuring-certificates)
and [OpenShift re-encrypt Route behavior](https://docs.redhat.com/en/documentation/openshift_container_platform/4.20/html/ingress_and_load_balancing/routes).
