-- PostgreSQL group roles and baseline database hardening for bddk-mcp.
--
-- Run this file as the database owner or an approved database administrator,
-- before the application migrations.  It intentionally creates NOLOGIN group
-- roles only; bank-managed LOGIN roles and credentials remain outside Git.

-- Bind this destructive, database-wide script to a value supplied separately
-- from the connection URL: PGOPTIONS='-c bddk.expected_database=DATABASE'.
-- Refuse before creating or altering any cluster role when the independent
-- deployment target is absent or does not match the active database.
DO $target_database$
DECLARE
    expected_database text := current_setting('bddk.expected_database', true);
BEGIN
    IF expected_database IS NULL OR btrim(expected_database) = '' THEN
        RAISE EXCEPTION 'bddk.expected_database must be set before role bootstrap';
    END IF;
    IF current_database() <> expected_database THEN
        RAISE EXCEPTION 'role bootstrap target database does not match the approved database';
    END IF;
END
$target_database$;

DO $roles$
DECLARE
    role_name text;
BEGIN
    FOREACH role_name IN ARRAY ARRAY[
        'bddk_schema_owner',
        'bddk_public_reader',
        'bddk_ingestion',
        'bddk_release_publisher',
        'bddk_operator_runtime',
        'bddk_telemetry_writer'
    ]
    LOOP
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = role_name) THEN
            EXECUTE format('CREATE ROLE %I', role_name);
        END IF;
    END LOOP;
END
$roles$;

-- Reconcile security attributes on every run.  None of these group roles may
-- authenticate, administer PostgreSQL, bypass RLS, or create databases/roles.
ALTER ROLE bddk_schema_owner
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE bddk_public_reader
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE bddk_ingestion
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE bddk_release_publisher
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE bddk_operator_runtime
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE bddk_telemetry_writer
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;

-- The deployment assumes a dedicated application database.  CONNECT and
-- temporary-object creation must be explicitly inherited from an approved
-- group role instead of arriving through PostgreSQL's PUBLIC pseudo-role.
DO $database_privileges$
BEGIN
    EXECUTE format(
        'REVOKE CONNECT, CREATE, TEMPORARY ON DATABASE %I FROM PUBLIC',
        current_database()
    );
    EXECUTE format(
        'GRANT CONNECT, CREATE ON DATABASE %I TO bddk_schema_owner',
        current_database()
    );
    EXECUTE format(
        'GRANT CONNECT ON DATABASE %I TO bddk_public_reader, bddk_ingestion, '
        'bddk_release_publisher, bddk_operator_runtime, bddk_telemetry_writer',
        current_database()
    );
END
$database_privileges$;

-- Migrations must SET ROLE bddk_schema_owner so new objects have one stable,
-- non-authenticating owner.  The migration runner itself creates bddk_meta and
-- bddk_operator; pre-creating them here would make its immutable first version
-- fail.  Do not create a role named bddk_operator: that identifier is reserved
-- for the post-migration schema.
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
ALTER SCHEMA public OWNER TO bddk_schema_owner;
GRANT USAGE, CREATE ON SCHEMA public TO bddk_schema_owner;

-- Remove PostgreSQL's broad object defaults from both current and future
-- application objects.  Positive application grants are deliberately listed
-- object-by-object in 02_grants.sql after migrations complete.
REVOKE ALL PRIVILEGES ON ALL TABLES IN SCHEMA public FROM PUBLIC;
REVOKE ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public FROM PUBLIC;

ALTER DEFAULT PRIVILEGES FOR ROLE bddk_schema_owner IN SCHEMA public
    REVOKE ALL PRIVILEGES ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE bddk_schema_owner IN SCHEMA public
    REVOKE ALL PRIVILEGES ON SEQUENCES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES FOR ROLE bddk_schema_owner IN SCHEMA public
    REVOKE ALL PRIVILEGES ON FUNCTIONS FROM PUBLIC;
