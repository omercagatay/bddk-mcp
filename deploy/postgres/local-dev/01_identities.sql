-- LOCAL DEVELOPMENT ONLY.
--
-- These fixed-password LOGIN roles are disposable fixtures for the
-- loopback-bound docker-compose.yml topology. Never apply this file to a
-- shared, remote, enterprise, bank, staging, or production database.

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;
CREATE EXTENSION IF NOT EXISTS unaccent WITH SCHEMA public;

DO $roles$
DECLARE
    role_name text;
BEGIN
    FOREACH role_name IN ARRAY ARRAY[
        'bddk_local_migrator',
        'bddk_local_ingestion',
        'bddk_local_public',
        'bddk_local_operator',
        'bddk_local_telemetry'
    ]
    LOOP
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = role_name) THEN
            EXECUTE format('CREATE ROLE %I LOGIN', role_name);
        END IF;
    END LOOP;
END
$roles$;

ALTER ROLE bddk_local_migrator
    LOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS
    PASSWORD 'local-only-migrator';
ALTER ROLE bddk_local_ingestion
    LOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS
    PASSWORD 'local-only-ingestion';
ALTER ROLE bddk_local_public
    LOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS
    PASSWORD 'local-only-public';
ALTER ROLE bddk_local_operator
    LOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS
    PASSWORD 'local-only-operator';
ALTER ROLE bddk_local_telemetry
    LOGIN INHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS
    PASSWORD 'local-only-telemetry';

GRANT bddk_schema_owner TO bddk_local_migrator;
GRANT bddk_ingestion TO bddk_local_ingestion;
GRANT bddk_public_reader, bddk_ingestion, bddk_operator_runtime
    TO bddk_local_operator;
GRANT bddk_public_reader TO bddk_local_public;
GRANT bddk_telemetry_writer TO bddk_local_telemetry;
