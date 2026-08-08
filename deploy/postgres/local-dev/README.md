# Disposable local PostgreSQL identities

This directory is used only by the loopback-bound `docker-compose.yml`
development topology. `01_identities.sql` creates fixed-password LOGIN roles
after the repository's NOLOGIN group-role baseline has been applied. It also
enables the `vector` and `unaccent` extensions in the disposable local
database.

Never apply this file to a shared, remote, staging, bank, or production
database. The passwords are public test fixtures, not secrets. Enterprise
deployments must provision extensions and LOGIN identities through approved
database administration and secret-management controls, following the parent
directory's role matrix.

Compose deliberately executes schema migration, post-migration grants, and
seed bootstrap as separate one-shot services with separate identities. An
existing `pgdata` volume is preserved across runs; migrations and grants are
rechecked on each startup. Removing the volume destroys the local corpus and
is appropriate only when intentionally resetting this disposable environment.

The role and grant services set
`PGOPTIONS='-c bddk.expected_database=bddk'`, while the migration service sets
`BDDK_EXPECTED_DATABASE_NAME=bddk`. These are independent wrong-target guards,
not credentials. The application services also set
`BDDK_ALLOW_INSECURE_DATABASE=true` because this fixture is confined to a
loopback-published, disposable PostgreSQL service without TLS. That bypass must
never be copied to a shared, remote, staging, bank or production deployment;
those environments require `sslmode=verify-full` and an absolute
`sslrootcert` path.

`bddk-bootstrap` imports into an already migrated schema; it does not migrate.
Compose invokes it with `--reindex-existing`, so every canonical document is
rebuilt and published under the active retrieval profile after migration. This
is required for an existing corpus after migration v0003; until publication is
complete, retrieval intentionally fails closed rather than serving stale or
partially indexed chunks.

An old pre-ledger `pgdata` volume will fail the ordinary migration instead of
being guessed/adopted. Preserve any data you need and follow
[`docs/LEGACY_DATABASE_UPGRADE.md`](../../../docs/LEGACY_DATABASE_UPGRADE.md),
or deliberately recreate only a disposable volume. Never add
`--adopt-legacy` to the normal Compose lifecycle.

A populated v0002 disposable volume likewise refuses migration v0003 until a
human explicitly approves its blocking backfill. Either intentionally recreate
the disposable volume or follow the stopped-workload, backup and size-matched
rehearsal procedure before a one-time
`--allow-retrieval-publication-backfill` run. Never add that approval flag to
the normal Compose lifecycle.
