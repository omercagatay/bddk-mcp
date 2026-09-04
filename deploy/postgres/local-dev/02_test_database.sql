-- LOCAL DEVELOPMENT ONLY.
--
-- Create the disposable pytest database fixture that mirrors the CI
-- `postgres-test` service container: LOGIN role `bddk` (password `bddk`) and
-- database `bddk_test` owned by it. tests/conftest.py targets
-- postgresql://bddk:bddk@localhost:5432/bddk_test by default; without this
-- fixture every `postgres`-marked test silently skips on a local run.
-- Never apply this file to a shared, remote, enterprise, bank, staging, or
-- production cluster.

DO $test_role$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'bddk') THEN
        EXECUTE 'CREATE ROLE bddk LOGIN';
    END IF;
END
$test_role$;

-- CI grants the test role the service-container superuser; the disposable
-- loopback fixture mirrors that so migration/recovery tests behave the same.
ALTER ROLE bddk
    LOGIN SUPERUSER
    PASSWORD 'bddk';

-- CREATE DATABASE cannot run inside a transaction block, so this file must be
-- executed without --single-transaction; \gexec keeps it idempotent.
SELECT 'CREATE DATABASE bddk_test OWNER bddk'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'bddk_test')
\gexec
