# PostgreSQL least-privilege roles

These SQL assets define database authorization boundaries for a **dedicated
BDDK MCP database**. They do not create LOGIN roles, passwords, Secrets, or a
database. Bank IAM/database administrators remain responsible for identities,
credential rotation, TLS, PostgreSQL host-based access rules and database
provisioning.

The six `bddk_*` group-role names are cluster-global. Production therefore
requires either a dedicated PostgreSQL cluster/service for this installation
or a formal DBA reservation proving those exact names belong only to this
deployment. A dedicated database inside an otherwise shared cluster is not by
itself sufficient. Installation-qualified role mapping is future work.

The required repository integration and actual-LOGIN contract lanes currently
prove PostgreSQL 17 only. Do not assume another major version is supported
until it is added to the compatibility matrix and accepted for the bank target.

## Mandatory database target and transport guards

Both DBA SQL files require an independently supplied target name before any
database-wide or cluster-role mutation. Set the custom PostgreSQL parameter on
the `psql` process and use the exact database name, for example:

```bash
PGOPTIONS='-c bddk.expected_database=DATABASE' \
  psql --single-transaction --set ON_ERROR_STOP=1 DBA_DSN \
  --file deploy/postgres/01_roles.sql
```

The scripts compare `bddk.expected_database` with `current_database()` and
refuse an absent or mismatched value. The value must be supplied separately
from the DSN so a mistargeted URL alone cannot authorize the operation. Apply
the same guard to `02_grants.sql`.

The application independently requires `BDDK_EXPECTED_DATABASE_NAME` for
`bddk-mcp migrate`. Its schema-owner check requires `current_user` to be
exactly `bddk_schema_owner`, reached from a distinct restricted `session_user`
through direct membership and `SET ROLE`; it also refuses administrative role
attributes, unexpected memberships, database ownership, and an incorrect
database name.

Outside isolated local development, every application PostgreSQL URL must use
a PostgreSQL URI with a hostname, exactly `sslmode=verify-full`, and exactly one
absolute `sslrootcert` path. For example, append
`&sslmode=verify-full&sslrootcert=%2Fapproved%2Fpostgres-ca.crt` after the
schema-owner `options` parameter, or use `?sslmode=...` when it is the first
query parameter. `BDDK_ALLOW_INSECURE_DATABASE=true` bypasses this application
guard and is permitted only in the disposable loopback Compose topology; never
set it in a shared, remote, bank, staging or production workload.

## Apply order

Use an approved database-owner or administrator session and stop on the first
SQL error. Apply each file as one transaction, for example with
`psql --single-transaction --set ON_ERROR_STOP=1`.

1. Provision the dedicated database and the approved `vector` and `unaccent`
   extensions.
2. Apply `01_roles.sql` with the independent
   `PGOPTIONS='-c bddk.expected_database=DATABASE'` guard described above.
3. Create bank-managed LOGIN identities outside this repository and grant them
   only the group-role memberships shown below.
4. Set `BDDK_EXPECTED_DATABASE_NAME` to that exact dedicated database name and
   run `bddk-mcp migrate` through a connection that executes
   `SET ROLE bddk_schema_owner`. The group role, rather than the authenticating
   identity, must own every application object. Its database-level `CREATE`
   privilege is required only so the immutable migrations can create the
   `bddk_meta` and `bddk_operator` schemas. For the current asyncpg-based job,
   add the URL-encoded startup parameter
   `options=-c%20role%3Dbddk_schema_owner` to the migration DSN (use `?` for the
   first query parameter or `&` after an existing TLS parameter), or configure
   an equivalent database-scoped role default through the DBA.
5. Apply `02_grants.sql` with the same independent `PGOPTIONS` target guard. It
   intentionally fails if an expected migrated table or sequence is absent.
6. Run strict bootstrap with the ingestion identity and a trust key mounted
   separately from the corpus:

   ```bash
   bddk-mcp bootstrap \
     --seed-dir /APPROVED/CORPUS \
     --reindex-existing \
     --require-quantified-freshness \
     --require-measured-freshness \
     --require-verified-signature \
     --trusted-signing-key /APPROVED/TRUST/corpus-signing-public-key.pem
   ```

   Bootstrap revalidates the exact manifest-declared paths and rejects
   undeclared reserved seed filenames before opening a database pool; a prior
   `verify-corpus` result is not a trust handoff. Retain its path-free manifest
   ID/SHA completion output. Bootstrap reports that publication is required but
   does not persist a release candidate.
7. Run `bddk-mcp publish-corpus-release` through the distinct
   `bddk-mcp-release-publisher-db` Secret
   (`BDDK_RELEASE_PUBLISHER_DATABASE_URL`). The release-publisher revalidates
   the imported corpus and signature before atomically persisting the v0005
   content-addressed release and activation;
   it must inherit exactly `bddk_release_publisher` and no ingestion or runtime
   role.
8. For a release that must become an immutable recovery/rollback target, run
   the one-shot administrative command through the same distinct release
   publisher identity:

   ```bash
   bddk-mcp retain-corpus-generation \
     --expected-release-id corpus_release_sha256_APPROVED_RELEASE_HASH
   ```

   `BDDK_RELEASE_PUBLISHER_DATABASE_URL` is required unless an explicitly
   supplied `--db` passes the same transport and exact-identity checks. Run this
   only after publication has made the expected release active. The CLI applies
   transaction-local `lock_timeout=30s` and `statement_timeout=30min`; either
   timeout rolls back the retention transaction, so resolve contention and
   recheck the expected active release before a reviewed retry. The command is
   deliberately absent from both MCP registries and does not activate,
   reactivate, or serve retained rows. Retain its content-free JSON receipt and
   storage evidence under the approved operational-evidence policy.
9. Start the public and operator workloads with their separate identities.
   Reindexing and publication are mandatory when migration v0003 has made a
   pre-existing corpus fail closed until republished.
10. Reapply and test `02_grants.sql` after every schema migration. A migration
   that adds a relation must add an explicit grant here in the same release.

The ordinary migration is the clean-install/default path and refuses managed
objects without a valid global ledger. `bddk-mcp migrate --adopt-legacy` is a
one-time, fail-closed path for only the exact final pre-ledger schema; it is not
a repair mode. Before using it, stop all readers/writers, prove a restorable
encrypted backup, and follow
[`docs/LEGACY_DATABASE_UPGRADE.md`](../../docs/LEGACY_DATABASE_UPGRADE.md). If
the structural verifier refuses, use the documented blue-green data-only path;
never insert migration rows manually or restore legacy DDL over a clean schema.

A populated v0002 database is a separate controlled-upgrade case. Migration
v0003 refuses before its blocking section-hash/FK backfill unless the operator
has stopped serving and ingestion, proved a restorable backup, rehearsed a
size-matched restore, and then explicitly supplies
`--allow-retrieval-publication-backfill`. After the migration and grants,
`bootstrap --reindex-existing` must publish every approved document before
serving. Do not put the approval flag in the normal clean-install command.

`01_roles.sql` revokes database `CONNECT`/`CREATE`/`TEMPORARY` and object privileges
from PostgreSQL's `PUBLIC` pseudo-role. Establish the approved memberships
before recycling application connections. Do not apply these assets to a
shared database: the `public` schema hardening and ownership reconciliation are
database-wide by design.

## Reviewed role matrix

| NOLOGIN group role | Exact purpose |
|---|---|
| `bddk_schema_owner` | Owns the `public`, `bddk_meta`, `bddk_operator`, and `bddk_retained` schemas and managed objects; runs migrations through `SET ROLE` |
| `bddk_public_reader` | Read-only access to the six public corpus relations, validated-citation and active-release views, migration ledger, and narrowly granted public functions |
| `bddk_ingestion` | Corpus and sync-state `SELECT`/`INSERT`/`UPDATE`/`DELETE`, three corpus ID sequences, and read-only global migration ledger |
| `bddk_release_publisher` | Revalidates the imported corpus, atomically persists its release/activation, and can invoke the narrow v7 retention/status facades; cannot mutate live corpus content, access retained tables directly, or run application tools |
| `bddk_operator_runtime` | Job-ledger read/write/prune in `bddk_operator` and read-only global migration ledger |
| `bddk_telemetry_writer` | Column-scoped `INSERT` on `tool_call_traces` and `USAGE` on its sequence; no trace reads or changes |

Migration v0004's eleven `regulatory_*` base tables are an owner-only canonical
legal-version validation workspace. No public, ingestion, operator, or
telemetry runtime role has any privilege on those tables. The public reader
gets `SELECT` only on the owner-executed, security-barrier
`regulatory_validated_section_citations` view; the operator sees that view only
through its separate `bddk_public_reader` membership. The view exposes only
validated, authoritative, non-fixture occurrences whose source, normalized
document, section, legal-version, provision, and evidence hashes agree. Do not
grant ad hoc base-table access or broaden the view outside a reviewed migration.

Readiness attests the complete v0004 catalog, not just selected object names:
the canonical digest must cover exactly 69 constraints and 21 indexes across
those eleven tables. An added, omitted, renamed, or definition-drifted object
fails readiness. Repository PostgreSQL tests also call `get_document_section`
through an official MCP session against the real validated view using a
synthetic legal family. That integration proves the SQL/MCP/contract path; it
does not authenticate a real BDDK source or curator.

The view also recomputes `documents.content_hash` from the retained normalized
Markdown and `document_sections.content_hash` from the exact retained section
text with schema-qualified PostgreSQL expressions. The document expression is
`pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(document.markdown_content, 'UTF8')), 'hex')`;
the section expression applies the same construction to `section.content`.
Stale document or section text cannot retain a citation merely because its
stored hash was not updated. The view also binds the section offsets back to
the exact normalized-document substring using the frozen Citation v1 boundary
whitespace transform. At bundle-mapping time, the declared content SHA is
checked against its content-derived `SourceBlob` identity, while each
`SourceArtifact` identity is derived separately from that blob identity plus
its canonical URI and acquisition timestamp. Migration v0004 does **not**,
however, retain or reconstruct source-artifact bytes in a form that can be
rehashed against the blob claim. Citation v1 is therefore a partial pilot for
validated normalized text and acquisition identity, not proof of
source-artifact authenticity.

## Schema v7 retained-generation boundary

Migration v0007 creates `bddk_retained` with 17 typed, generation-qualified
mirror relations for the exact tables covered by the v5 corpus-state epoch. It
also creates generation, per-relation inventory, seal, and retained-release
binding records plus the security-barrier
`bddk_meta.corpus_release_retention_status` view. Physical generation,
governed release, content-derived seal, and activation sequence are distinct
identities. Generation identity is derived from the exact corpus-state and
retrieval-profile hashes, and one generation has one seal. Multiple governed
releases over that same exact state/profile therefore have distinct per-release
bindings to the shared generation and seal; release count is not physical-copy
count. A pre-v7 release without a binding is reported as
`legacy_v5_unretained`; it is never backfilled from a matching current hash.

V7 canonicalizes `current_corpus_state_sha256`, retained-state hashing, and
retained-row hashing with function-local `TimeZone=UTC`,
`DateStyle=ISO, YMD`, `IntervalStyle=postgres`, `bytea_output=hex`, and
`extra_float_digits=3`; caller session formatting cannot change those results.
Migration does not rewrite historical v5 release hashes. Before changing the
hash function, v7 recomputes any active v5/v6 release canonically and refuses
the migration on mismatch. Do not edit the release ledger or synthesize a
retention binding: on the unchanged pre-v7 schema (v5 or v6), independently
review and revalidate the unchanged corpus, then use the exact
publication-only `publish-corpus-release` compatibility path to append and
activate a canonical release. Retry v7 before retention. Serving, retention,
and every other workload remain v7-only.

`bddk_meta.retain_active_corpus_generation(text)` acquires the schema-migration
lock before the corpus-mutation lock, locks the live 17-relation state, copies
typed rows in one transaction, reproduces the exact v5 state hash, and seals
the result. For a new exact state/profile, any failure rolls back the
generation, inventory, seal, and release binding together. When the physical
generation already exists, the function adds only the new release binding; a
binding failure leaves that shared sealed generation unchanged. Both paths
leave the v5 active release unchanged. Sealed members and
metadata reject update/delete/truncate. Catalog readiness attests the exact v7
schema, constraints, indexes, triggers, routines, view, and ownership.

All application profiles have no direct table privilege or schema usage on
`bddk_retained`. The release publisher receives only `SELECT` on the retention
status view and `EXECUTE` on the retention and storage-inspection facades. Do
not add ad hoc grants, expose these operations through MCP, or treat the
publisher role as a serving identity.

The receipt's heap/auxiliary/TOAST/index totals are PostgreSQL catalog storage
for the retained store and reconcile arithmetically. When available, the
best-effort WAL value is only the cluster-wide interval observed around the
command and is labelled `observed_cluster_interval_not_exclusive`; unrelated
activity can contribute. Its pre-retention LSN baseline is attempted inside a
savepoint in the durable transaction; failure rolls back only that measurement
attempt and retention proceeds. The endpoint is best-effort after commit. If
either observation is unavailable or invalid, WAL remains `not_measured` and
the already committed retention is not misreported as failed.

Backup growth remains `not_measured` until a controlled bank/DBA backup is
performed. These figures do not select a retention count or authorize capacity.
Capacity planning must count unique retained state/profile generations, not
the governed release bindings that share them.

V7 creates immutable database rollback targets, but it does not implement
generation-bound serving, activation, reactivation, or rollback. It also
retains only fields already in PostgreSQL. A stored `documents.pdf_blob` is
copied, but missing external authoritative files/evidence packs are not
acquired; `regulatory_source_blobs` remains a content-identity record. H2-02B
and bank source/backup policy remain required before claiming product rollback
or complete authoritative-source preservation.

The SQL deliberately does **not** create a role named `bddk_operator`; that
identifier is the operator schema and using it for both would make deployment
reviews needlessly ambiguous.

## Bank-managed LOGIN memberships

The identity names below are illustrative; provision the real LOGIN roles and
credentials through the bank's approved controls.

| Workload / OpenShift Secret | Required memberships |
|---|---|
| Migration / `bddk-mcp-schema-owner-db` (`BDDK_SCHEMA_OWNER_DATABASE_URL`) | `bddk_schema_owner` (connection must `SET ROLE` it) |
| Bootstrap and ingestion / `bddk-mcp-ingestion-db` (`BDDK_INGESTION_DATABASE_URL`) | `bddk_ingestion` |
| Release publication / `bddk-mcp-release-publisher-db` (`BDDK_RELEASE_PUBLISHER_DATABASE_URL`) | `bddk_release_publisher` |
| Public MCP / `bddk-mcp-public-db` | `bddk_public_reader` |
| Operator MCP / `bddk-mcp-operator-db` | `bddk_public_reader`, `bddk_ingestion`, `bddk_operator_runtime` |

The operator currently executes synchronization/backfill runners in its own
process and pool, so it needs both corpus ingestion and job-ledger memberships.
If execution later moves to a separate worker, remove `bddk_ingestion` from the
operator web identity and grant it only to that worker.

Telemetry is disabled unless explicitly configured. Enabling it requires a
separate `BDDK_TELEMETRY_DATABASE_URL` whose LOGIN inherits only
`bddk_telemetry_writer`; startup verifies the exact column-level INSERT-only
contract and refuses a role that can read or modify trace rows. Never grant
`bddk_telemetry_writer` to the public or operator workload LOGIN.

At runtime, public, ingestion, and operator entry points verify the authenticating
session, exact direct/inherited application-role closure, database/schema/table/
sequence/routine privileges, and absence of unreviewed managed objects. The
telemetry path independently verifies its exact column-scoped INSERT-only
contract. DSN string inequality is not treated as an authorization boundary;
an over-privileged or multiply affiliated LOGIN is refused even through a
differently written connection URL. Public/operator pools repeat a bounded
identity assertion for every connection they open, not only the first startup
connection.

Readiness also performs a live, `SELECT`-only catalog attestation of critical
constraints, trigger definitions and enablement, function bodies/security and
configuration, full-text indexes, and HNSW options. Missing or drifted catalog
objects make the database not ready even when its tables exist and `SELECT 1`
succeeds. Repair catalog drift through a reviewed migration or clean restore;
never patch the checksum ledger or weaken the attestation.

The repository's fixed-password identities under `local-dev/` are disposable
fixtures used only by the loopback-bound Compose topology. They are not a
template for bank-managed LOGIN roles and must never be applied to a remote,
shared, staging, enterprise, or production database.

Do not grant one NOLOGIN group role to another. Grant the reviewed combination
directly to each bank-managed LOGIN identity so effective privileges remain
obvious in database audits.

## Verification expectations

After deployment, test effective privileges using the actual workload LOGINs,
not the administrator account:

- public can search/read corpus tables but cannot insert, update, delete, read
  `sync_failures`, access operator jobs, or create schema objects;
- ingestion can mutate corpus/sync tables but cannot access traces or operator
  jobs and cannot create tables;
- operator runtime can mutate `operator_jobs` and read the global schema ledger but
  its operator-only role alone cannot access the corpus;
- telemetry can execute the application's column-scoped trace insert but cannot
  select, update, delete, override `id`/`created_at`, or create objects;
- every runtime identity is denied access to the eleven owner-only `regulatory_*`
  base tables; public/operator citation reads succeed only through the
  attested `regulatory_validated_section_citations` view;
- an unprivileged database identity inheriting none of these roles cannot
  connect after the `PUBLIC` revocation.
- disabling a publication-invalidation trigger, replacing its function, or
  dropping a required FTS index makes catalog readiness fail closed.
- changing the validated-citation view's owner, security options, dependencies,
  projection, joins, or validation predicates makes catalog readiness fail closed.

`tests/test_postgres_role_assets.py` enforces the static privilege contract and
contains an opt-in transactional PostgreSQL denial test for a dedicated test
database.
