# Legacy database upgrade

This runbook covers upgrading an existing, already-ledgered database through
ordinary migration. The `--adopt-legacy` path for databases created by the
pre-ledger initializers was removed: no supported deployment still carries that
shape. A database whose managed objects exist without
`bddk_meta.schema_migrations` fails ordinary migration and rolls back; move its
data with the blue-green data-only path below.

An already-ledgered v5/v6/v7 database uses ordinary migration. V7 makes
retained-row and v5/retained state hashing
session-independent with function-local `TimeZone=UTC`,
`DateStyle=ISO, YMD`, `IntervalStyle=postgres`, `bytea_output=hex`, and
`extra_float_digits=3`. It deliberately does not rewrite an existing release
hash. If a pre-v7 release was published under different session formatting,
the v7 migration fails before replacing the hash function when the canonical
current-state hash differs from the recorded hash. Preserve the old ledger
evidence. The current binary deliberately disables `publish-corpus-release`
and must not be used to manufacture a historical row or binding. If an
unchanged v5/v6 database hits this exact guard, stop the upgrade and use only a
separately approved, digest-pinned pre-v8 remediation build under an isolated
maintenance procedure, or take the blue-green data-only path below. Revalidate
and canonically republish on that unchanged pre-v7 schema, then remove the
remediation capability before retrying migrations through v8. This repository
does not automate that break-glass decision in the current CLI. Serving,
retention, verification, activation, and every other steady-state workload
remain latest-schema-only.

The ordinary command remains fail-closed:

```bash
BDDK_EXPECTED_DATABASE_NAME='DATABASE' \
BDDK_SCHEMA_OWNER_DATABASE_URL='postgresql://MIGRATOR:SECRET@HOST:5432/DATABASE?options=-c%20role%3Dbddk_schema_owner&sslmode=verify-full&sslrootcert=%2FAPPROVED%2Fpostgres-ca.crt' \
  bddk-mcp migrate
```

It never guesses that existing objects belong to this project. Do not create
ledger rows manually.

Both DBA SQL assets additionally require
`PGOPTIONS='-c bddk.expected_database=DATABASE'`, and the application migration
requires the independent exact `BDDK_EXPECTED_DATABASE_NAME` shown above.
Outside isolated local development, every application DSN must retain exactly
`sslmode=verify-full` and one absolute `sslrootcert` path. Do not bypass these
target and transport guards to make an old database pass.

The `bddk_meta.legacy_schema_adoptions` table created by migration v0003 stays
in the schema as audit history; nothing writes to it any more.

## Blue-green data-only path

Use this when an old database cannot be migrated in place. Do not patch the
ledger, rename objects by trial and error, or weaken a verifier. Leave the
source database unchanged and use a new database:

1. Keep the old database read-only and preserve the verified backup.
2. Create a new dedicated database; install `vector` and `unaccent` in
   `public`; apply `01_roles.sql` with the independent target guard; and run
   ordinary `bddk-mcp migrate` with the schema-owner identity, exact
   `BDDK_EXPECTED_DATABASE_NAME`, and `verify-full` database transport.
3. Produce a bank-approved encrypted **data-only** export of the old
   application tables. Do not restore old schemas, functions, triggers,
   indexes, privileges, owners, or extension objects into the new database.
4. Load into a staging database first, using explicit column lists and reviewed
   mappings for `decision_cache`, `documents`, `document_sections`,
   `document_versions`, `document_chunks`, `sync_metadata`, `sync_failures`,
   and—only when retention policy permits—`tool_call_traces`. Preserve
   `pdf_blob` and embeddings if they are required. Reconcile identity sequences
   beyond the imported maximum IDs.
5. Rebuild sections/chunks and publish every imported canonical document under
   the active retrieval profile with the reviewed ingestion workflow. Do not
   copy `document_retrieval_publications` from another database or manufacture
   publication rows.
6. Validate per-table counts, nullability, uniqueness, document content hashes,
   section/chunk relationships, embedding dimensions and null counts, FTS
   population, live catalog attestation, sample exact-article retrieval,
   citation reconstruction, and application readiness. Review differences;
   never silently drop rejected rows.
7. Apply `02_grants.sql` with the independent target guard, run
   public/operator write-denial tests and MCP smoke tests, then switch clients
   to the new database through the normal bank change process. Keep the old
   database read-only until acceptance completes.

`bddk-seed export` is useful for a reviewed search-corpus transfer, but it is
not a lossless database backup: it omits PDF blobs, embeddings, document-version
history, sync state, telemetry, and sequence state. Do not use it as the sole
preservation mechanism for a regulated production upgrade.
