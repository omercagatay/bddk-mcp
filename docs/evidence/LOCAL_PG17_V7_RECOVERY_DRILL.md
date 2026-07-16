# Local PostgreSQL 17 schema-v7 recovery evidence

## Result

On 2026-07-16, commit `af275eb420ee6b36e7b59e46f69fe7a03419e8a5`
completed the repository's guarded logical-backup/restore workflow against two
different disposable PostgreSQL 17 clusters. The source contained one
synthetic, current-profile corpus with an active governed release, canonical
legal-version rows, and two sealed retained generations. The restored database
matched the source's logical fingerprint and active-release identity.

The retained machine-readable report is
[`local-pg17-v7-restore-2026-07-16.json`](local-pg17-v7-restore-2026-07-16.json).
It contains no DSN, password, guard token, target database name, document
identifier, legal text, reviewer identity, or database LOGIN name.

| Check | Observed result |
|---|---:|
| Workflow status | `passed` |
| Source and restored migration version | `7` |
| Managed objects inventoried on each side | `51` |
| Retained typed member relations exercised | `17` |
| Catalog valid on both sides | yes |
| Readiness ready on both sides | yes |
| Active release identity equal | yes |
| Logical fingerprint equal | yes |
| Six restored LOGIN profiles verified | yes |
| Dump bytes | `322806` |
| Backup subprocess time | `243 ms` |
| Restore subprocess time | `280 ms` |
| End-to-end workflow time | `1373 ms` |

## Reproducibility boundary

- Source and target used the exact image
  `pgvector/pgvector:pg17@sha256:d2ef61f42ef767baa5a1475393303cc235bcd92febd9d7014eddb48b41f3bad0`.
- The matching `pg_dump` and `pg_restore` executables ran from that same image.
- Source and target were separate containers with different PostgreSQL system
  identifiers and loopback-only published ports.
- Authentication used PostgreSQL `trust` only inside this disposable,
  loopback-bound test. No credential was created or retained.
- The source was built through the reviewed roles, schema-v7 migrations,
  grants, publication, and retention routines. An initial publication under a
  deliberately non-current test profile failed the recovery readiness gate;
  the fixture was republished and retained under the actual pinned retrieval
  profile before the passing run.
- The workflow retained its restored target for inspection as designed. Both
  disposable containers were removed after evidence validation.

The report can be checked without reading corpus content:

```bash
jq -e '
  .schema_version == 2 and
  .status == "passed" and
  .identities_verified == true and
  .source.migration_version == 7 and
  .restored.migration_version == 7 and
  (.source.relations | length) == 51 and
  (.restored.relations | length) == 51 and
  .source.logical_fingerprint_sha256 == .restored.logical_fingerprint_sha256 and
  .source.active_corpus_release_id == .restored.active_corpus_release_id
' docs/evidence/local-pg17-v7-restore-2026-07-16.json
```

## What this does not prove

This is repository-scale synthetic evidence, not bank acceptance. It does not
prove production backup custody, encryption, PITR/WAL replay, bank TLS/HBA or
LOGIN policy, OpenShift execution, bank-size duration or capacity, source-file
recovery outside PostgreSQL, controlled backup growth attributable to corpus
retention, or compliance with an approved numeric RPO/RTO. Those gates remain
open and must be evidenced in the target bank environment.
