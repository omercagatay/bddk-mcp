# Guarded Migration and Restore Drills

## Scope

`scripts/recovery_drill.py` provides two explicit, non-serving workflows:

- `migration-rehearsal` proves the populated-v2 default refusal, then applies
  the explicitly approved migration and republishes retrieval data on one
  marked disposable restore;
- `restore-drill` takes a read-only exported snapshot, creates a new database on
  a different PostgreSQL cluster, restores it, reapplies reviewed roles/grants,
  and verifies logical identity, catalog, readiness, and least-privilege LOGINs.

Neither workflow is a production backup scheduler or PITR implementation. The
bank remains responsible for encrypted backup storage, retention, PITR/WAL,
key management, scheduling, alerts, RPO/RTO, and recovery authorization.

## Non-negotiable safety boundary

The workflows refuse before mutation unless all applicable guards agree:

- the exact acknowledgement is
  `I_UNDERSTAND_THIS_MUTATES_ONLY_A_DISPOSABLE_RECOVERY_TARGET`;
- a migration target matches `bddk_v2_rehearsal_*`, or a restore target matches
  `bddk_restore_drill_*`;
- the database contains the independently provisioned SHA-256 guard in
  `bddk.recovery_drill_guard`;
- restore administration connects to a dedicated `bddk_recovery_admin*`
  database with a superuser/CREATEDB identity and the matching guard;
- source and restore clusters have different PostgreSQL system identifiers;
- the restore target does not already exist; and
- the restore administrator DSN is not any configured application runtime DSN.

The restore workflow never issues `DROP DATABASE`. It retains a created target
after success or failure for explicit investigation and cleanup by the isolated
environment owner. Do not rename a real database to satisfy the name pattern,
set the guard on a production cluster, or reuse this workflow as a general DBA
utility.

## Prerequisites

- PostgreSQL 17 on the supported release contract;
- matching, review-controlled `pg_dump` and `pg_restore` binaries on the runner;
- a reviewed, ready source snapshot and a separately isolated disposable
  restore cluster;
- `deploy/postgres/01_roles.sql` and `02_grants.sql` from the same commit;
- approved source-read and recovery-admin credentials delivered by a secret
  manager, never command-line arguments or Git; and
- sufficient temporary encrypted storage for a custom-format logical dump.

The command accepts no DSN, password, or guard secret as an argument. Inject
the following names through the approved job Secret mechanism:

| Workflow | Required environment names |
|---|---|
| Both | `BDDK_RECOVERY_GUARD_TOKEN`, `BDDK_RECOVERY_ACKNOWLEDGEMENT` |
| Migration rehearsal | `BDDK_SCHEMA_OWNER_DATABASE_URL`, `BDDK_INGESTION_DATABASE_URL` |
| Restore drill | `BDDK_RECOVERY_SOURCE_DATABASE_URL`, `BDDK_RECOVERY_ADMIN_DATABASE_URL` |

`pg_dump` and `pg_restore` are bounded to 1,800 seconds each by default. Set
`BDDK_RECOVERY_PG_TOOL_TIMEOUT_SECONDS` to a reviewed whole number from 30
through 21,600 when measured corpus scale justifies a different bound. An
invalid value fails before process launch. On timeout the runner sends
terminate, waits ten seconds, then kills and reaps the child; retained evidence
contains only the stable `pg_tool_timed_out` code, never arguments or process
environment values.

Example shapes, with identifiers only:

```bash
uv run --frozen python scripts/recovery_drill.py migration-rehearsal \
  --expected-target bddk_v2_rehearsal_RELEASE \
  --report /APPROVED/EVIDENCE/migration.json

uv run --frozen python scripts/recovery_drill.py restore-drill \
  --expected-source APPROVED_SOURCE \
  --expected-admin bddk_recovery_admin_RELEASE \
  --target bddk_restore_drill_RELEASE \
  --report /APPROVED/EVIDENCE/restore.json
```

Reports are created once with mode `0600`; an existing path is never
overwritten. Database URLs and secrets must be injected out of band.

## Evidence and acceptance

The deterministic report contains only bounded operational evidence: workflow
status, hashed target identity, timings, logical fingerprint, migration
version/checksum, row/relation/database/dump sizes, catalog/readiness outcomes,
WAL growth, lock samples/waiters, and reindex counts. It excludes DSNs,
credentials, target names, document identifiers, and corpus text.

The logical fingerprint hashes actual retained document Markdown, section text,
chunk text, source PDF bytes, decision-cache/document-version bodies, and the
pgvector binary serialization with PostgreSQL 17's SHA-256 functions. Stored
content hashes, byte lengths, or embedding presence alone are not accepted as
proof: same-length text or vector corruption changes the logical fingerprint.
Operational JSON/error fields are likewise reduced with SHA-256 before they
enter the outer deterministic fingerprint; no MD5 evidence remains.

For schema v0004 and later, relation counts and the logical fingerprint cover
all eleven canonical legal-version `regulatory_*` base tables in
parent-before-child order and the derived
`regulatory_validated_section_citations` security-barrier view. Family-import
evidence binds the predecessor bundle hash, exact member manifest, declared
importer, and effective PostgreSQL current/session users. The fingerprint
includes those values, `SourceBlob` content identity, separate
`SourceArtifact` acquisition identity, claim hashes, validation provenance,
and the view's validated citation rows, but the report never emits claim rows,
reviewer identifiers, source URIs, or legal text. Catalog readiness separately
requires the exact v0004 digest for 69 constraints and 21 indexes. The view has
zero storage bytes in the relation evidence because it is not materialized. A
restore with omitted or changed legal-version state therefore fails the same
equality gate as a changed retrieval corpus.

Repository evidence is complete only when:

1. default populated-v2 migration refusal is observed without schema change;
2. the approved rehearsal reaches the current checksum and a ready published
   corpus;
3. an isolated logical restore has the same logical fingerprint and row counts;
4. schema-owner, public, ingestion, operator, and telemetry LOGIN contracts pass;
5. a representative MCP read/retrieval and release-specific citation check pass;
6. the measured duration satisfies the bank's numeric RTO and the backup point
   satisfies its RPO; and
7. the evidence artifact is retained under approved access and retention rules.

The local repository run on 2026-07-15 proved the guarded, rollback-only
populated-v2 migration path on disposable PostgreSQL 17. A full
`pg_dump`→isolated-cluster→`pg_restore` run was not possible because compatible
host client binaries and a second isolated cluster were unavailable. The
workflow fails closed when either binary is absent. This is an explicit open
acceptance gate, not a skipped success.
