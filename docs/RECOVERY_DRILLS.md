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

For schema v0007, the same recovery fingerprint additionally covers all 17
typed `bddk_retained` member relations, ordered by a per-row SHA-256 digest,
plus generation, relation-inventory, seal, retained-release binding, and
retention-status state. Relation evidence and restore ordering therefore
inventory **51 managed objects**. A restored generation must preserve the exact
typed database rows and its generation/release/seal/activation relationships;
one state/profile-derived generation has one seal, while multiple governed
release bindings may legitimately reference that same pair. Restore comparison
must preserve those many-to-one bindings rather than infer a physical copy per
release.
Catalog readiness and database-identity checks separately verify the v7
triggers, routines, constraints, view, ownership, and ACL boundary. The report
still emits hashes/counts/sizes rather than corpus text or principals.

For schema v0008, recovery additionally inventories
`bddk_meta.corpus_release_requests` and
`bddk_meta.corpus_release_request_activations`. The current restore ordering and
fingerprint contract therefore cover **53 managed objects**. Identity recovery
also adds the independent `release_verifier` profile, so the current matrix has
**seven** application LOGIN profiles. Readiness verifies that the verifier can
stage but cannot activate/retain, while the publisher can activate a one-time
request ID and retain but cannot stage or call the old direct publication
routine. These current code/test contracts are now also exercised by the local
two-cluster v8 report described below; that synthetic report is still not bank
production acceptance evidence.

V7 retention preserves fields already stored in PostgreSQL, including a
`documents.pdf_blob` when present. It does not make missing external
authoritative source files part of the database backup, and
`regulatory_source_blobs` still stores content identity rather than external
artifact bytes. Recovery evidence must not be described as source-authenticity
or off-database evidence-pack recovery.

Recovery evidence schema v2 now inventories 53 objects: the v5 release
state/view objects, v7 retained state, the activation sequence, and the two v8
request/activation-binding relations. It rejects reuse of any of the seven
application DSNs for recovery administration and verifies seven restored LOGIN
profiles, including the independent release verifier and release publisher.
Active release identity, staged-request/activation-binding state,
activation-sequence identity/ownership,
database encoding, collation/character-classification names, locale provider,
provider locale, ICU rules, stored/actual collation versions, row counts, and
logical fingerprints must match exactly. A stored/actual collation-version
mismatch fails the snapshot before comparison. The versioned v5 fingerprint orders
textual identities under the database default collation: function-local
formatting settings make hashes session-independent, but do not make a
cross-collation restore equivalent. A differently collated or encoded target
therefore fails closed and must not be promoted. This repository
contract does not substitute for retained bank PITR, backup-custody, or numeric
RPO/RTO acceptance evidence.

Repository evidence is complete only when:

1. default populated-v2 migration refusal is observed without schema change;
2. the approved rehearsal reaches the current checksum and a ready published
   corpus;
3. an isolated logical restore has the same logical fingerprint and row counts;
4. schema-owner, public, ingestion, release-verifier, release-publisher,
   operator, and telemetry LOGIN contracts pass;
5. a representative MCP read/retrieval and release-specific citation check pass;
6. the measured duration satisfies the bank's numeric RTO and the backup point
   satisfies its RPO; and
7. the evidence artifact is retained under approved access and retention rules.

The local repository run on 2026-07-15 proved the guarded, rollback-only
populated-v2 migration path on disposable PostgreSQL 17. On 2026-07-16, the
then-current schema-v7 workflow also completed a full
`pg_dump`→isolated-cluster→`pg_restore` run against two disposable PostgreSQL 17
clusters. The retained report inventories all 51 managed objects, including
two sealed generations across all 17 retained member relations, and proves
equal logical fingerprints, active-release identity, locale evidence,
catalog/readiness state, activation-sequence state, and six restored LOGIN
profiles (**docs/evidence/LOCAL_PG17_V7_RECOVERY_DRILL.md**). This is a
synthetic, **historical schema-v7/51-object/six-identity proof only**. It must
not be cited as current-schema evidence.

Later on 2026-07-16, the current schema-v8 workflow completed a separate
`pg_dump`→isolated-cluster→`pg_restore` run on PostgreSQL 17.10. Its report
preserves all 53 managed objects, two staged requests and their activation
bindings, two retained generations, the exact active release and logical
fingerprint, catalog/readiness state, activation-sequence state, and all seven
restored LOGIN-profile contracts
(**docs/evidence/LOCAL_PG17_V8_RECOVERY_DRILL.md**). This proves execution of
the repository's current synthetic v8 restore contract. Bank backup custody,
PITR, TLS/HBA, bank-sized capacity and elapsed-time evidence, representative
MCP/citation smoke checks, approved RPO/RTO, and bank DBA acceptance remain
explicit external gates.

When available, the `retain-corpus-generation` CLI's WAL field is a cluster-wide
observed interval around its transaction, not exact WAL attributable to one
generation. The pre-retention LSN baseline is attempted inside a savepoint so
its failure rolls back only the measurement attempt; the endpoint is
best-effort after the durable transaction commits. If either observation is
unavailable or invalid, WAL remains `not_measured` without converting a
committed seal into a reported failure.
Its backup-growth field remains `not_measured`; only a controlled backup before
and after retention can supply that evidence. Neither metric authorizes a
retention window or capacity allocation without the bank owner/DBA decision.

Finally, a successfully restored sealed generation is a rollback *target*, not
an application rollback. V7 does not route serving queries to retained tables
or append activation/reactivation events. H2-02B must implement and test those
semantics before an operator runbook can claim prior-generation rollback.
