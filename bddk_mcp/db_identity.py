"""Fail-closed PostgreSQL workload-identity verification.

The deployment SQL grants capabilities to small NOLOGIN group roles.  A DSN
string is not an authorization boundary: two differently written DSNs can
still authenticate as the same PostgreSQL LOGIN.  This module therefore
checks the effective identity and every application-object privilege before a
public, operator, ingestion, or release-publication workload is allowed to run.

The contract intentionally assumes the dedicated database topology described
by ``deploy/postgres/README.md``.  Unexpected application schemas or objects
are rejected so a newly introduced relation cannot silently inherit an
unreviewed runtime privilege.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import asyncpg

from bddk_mcp.db_compatibility import PostgreSQLCompatibilityError, assert_supported_postgresql

DatabaseIdentityProfile = Literal["public", "operator", "ingestion", "release-publisher"]

_TABLE_PRIVILEGES = frozenset(
    {
        "SELECT",
        "INSERT",
        "UPDATE",
        "DELETE",
        "TRUNCATE",
        "REFERENCES",
        "TRIGGER",
        "MAINTAIN",
        "SELECT_COLUMNS",
        "INSERT_COLUMNS",
        "UPDATE_COLUMNS",
        "REFERENCES_COLUMNS",
        "SELECT_GRANT",
        "INSERT_GRANT",
        "UPDATE_GRANT",
        "DELETE_GRANT",
        "TRUNCATE_GRANT",
        "REFERENCES_GRANT",
        "TRIGGER_GRANT",
        "MAINTAIN_GRANT",
    }
)
_CORPUS_TABLES = frozenset(
    {
        "public.decision_cache",
        "public.documents",
        "public.document_sections",
        "public.document_versions",
        "public.document_chunks",
        "public.document_retrieval_publications",
    }
)
_INGESTION_TABLES = _CORPUS_TABLES | {
    "public.sync_metadata",
    "public.sync_failures",
}
_REGULATORY_VERSION_TABLES = frozenset(
    {
        "public.regulatory_evidence",
        "public.regulatory_family_imports",
        "public.regulatory_instruments",
        "public.regulatory_legal_events",
        "public.regulatory_legal_status_assertions",
        "public.regulatory_legal_version_artifacts",
        "public.regulatory_legal_version_provisions",
        "public.regulatory_legal_versions",
        "public.regulatory_provisions",
        "public.regulatory_source_blobs",
        "public.regulatory_source_artifacts",
    }
)
_REGULATORY_PUBLIC_VIEWS = frozenset({"public.regulatory_validated_section_citations"})
_CORPUS_RELEASE_VIEWS = frozenset({"bddk_meta.active_corpus_release"})
_ALL_TABLES = (
    _INGESTION_TABLES
    | {
        "public.tool_call_traces",
        "bddk_meta.schema_migrations",
        "bddk_meta.legacy_schema_adoptions",
        "bddk_meta.corpus_releases",
        "bddk_meta.corpus_release_activations",
        "bddk_operator.operator_jobs",
    }
    | _REGULATORY_VERSION_TABLES
    | _REGULATORY_PUBLIC_VIEWS
    | _CORPUS_RELEASE_VIEWS
)
_ALL_SEQUENCES = frozenset(
    {
        "public.document_sections_id_seq",
        "public.document_versions_id_seq",
        "public.document_chunks_id_seq",
        "public.tool_call_traces_id_seq",
        "bddk_meta.corpus_release_activations_activation_sequence_seq",
    }
)
_INGESTION_SEQUENCES = frozenset(
    {
        "public.document_sections_id_seq",
        "public.document_versions_id_seq",
        "public.document_chunks_id_seq",
    }
)
_ALL_ROUTINES = frozenset(
    {
        "public.immutable_unaccent(text)",
        "public.documents_tsv_trigger()",
        "public.document_sections_tsv_trigger()",
        "public.chunks_tsv_trigger()",
        "public.invalidate_retrieval_publication()",
        "bddk_meta.corpus_fingerprint_frame(text)",
        "bddk_meta.current_corpus_state_sha256(text)",
        "bddk_meta.corpus_retrieval_ready(text)",
        "bddk_meta.reject_corpus_release_mutation()",
        "bddk_meta.publish_verified_corpus_release(text, text, text, integer, integer, integer, text)",
        "bddk_meta.resolve_regulation_status(text, date)",
    }
)


@dataclass(frozen=True)
class _IdentityContract:
    memberships: frozenset[str]
    schemas: Mapping[str, frozenset[str]]
    tables: Mapping[str, frozenset[str]]
    sequences: Mapping[str, frozenset[str]]
    routines: Mapping[str, frozenset[str]]


@dataclass(frozen=True)
class DatabaseIdentityInspection:
    """Read-only snapshot of one workload LOGIN's effective capabilities."""

    current_user: str
    session_user: str
    session_can_login: bool
    session_inherits: bool
    direct_memberships: frozenset[str]
    inherited_memberships: frozenset[str]
    unsafe_roles: frozenset[str]
    membership_admin: bool
    public_acl_leakage: bool
    direct_login_acl: bool
    database_privileges: frozenset[str]
    schemas: Mapping[str, frozenset[str]]
    tables: Mapping[str, frozenset[str]]
    sequences: Mapping[str, frozenset[str]]
    routines: Mapping[str, frozenset[str]]


class DatabaseIdentityError(RuntimeError):
    """The configured database LOGIN violates its workload contract."""


def _object_contract(
    names: frozenset[str],
    privileges: Mapping[str, frozenset[str]],
) -> Mapping[str, frozenset[str]]:
    return MappingProxyType({name: privileges.get(name, frozenset()) for name in sorted(names)})


def _build_contracts() -> Mapping[str, _IdentityContract]:
    read_tables = {
        name: frozenset({"SELECT"}) for name in _CORPUS_TABLES | _REGULATORY_PUBLIC_VIEWS | _CORPUS_RELEASE_VIEWS
    }
    read_tables["bddk_meta.schema_migrations"] = frozenset({"SELECT"})

    ingestion_tables = {name: frozenset({"SELECT", "INSERT", "UPDATE", "DELETE"}) for name in _INGESTION_TABLES}
    ingestion_tables["bddk_meta.schema_migrations"] = frozenset({"SELECT"})
    ingestion_tables["bddk_meta.active_corpus_release"] = frozenset({"SELECT"})

    publisher_tables = {name: frozenset({"SELECT"}) for name in _CORPUS_TABLES | _REGULATORY_VERSION_TABLES}
    publisher_tables["bddk_meta.schema_migrations"] = frozenset({"SELECT"})
    publisher_tables["bddk_meta.active_corpus_release"] = frozenset({"SELECT"})

    operator_tables = dict(ingestion_tables)
    operator_tables["bddk_operator.operator_jobs"] = frozenset({"SELECT", "INSERT", "UPDATE", "DELETE"})
    for name in _REGULATORY_PUBLIC_VIEWS:
        operator_tables[name] = frozenset({"SELECT"})

    public_schemas = MappingProxyType(
        {
            "public": frozenset({"USAGE"}),
            "bddk_meta": frozenset({"USAGE"}),
            "bddk_operator": frozenset(),
        }
    )
    operator_schemas = MappingProxyType(
        {
            "public": frozenset({"USAGE"}),
            "bddk_meta": frozenset({"USAGE"}),
            "bddk_operator": frozenset({"USAGE"}),
        }
    )
    no_sequences = _object_contract(_ALL_SEQUENCES, {})
    ingestion_sequences = _object_contract(
        _ALL_SEQUENCES,
        {name: frozenset({"USAGE"}) for name in _INGESTION_SEQUENCES},
    )
    public_routines = _object_contract(
        _ALL_ROUTINES,
        {
            "public.immutable_unaccent(text)": frozenset({"EXECUTE"}),
            "bddk_meta.current_corpus_state_sha256(text)": frozenset({"EXECUTE"}),
            "bddk_meta.corpus_retrieval_ready(text)": frozenset({"EXECUTE"}),
            "bddk_meta.resolve_regulation_status(text, date)": frozenset({"EXECUTE"}),
        },
    )
    ingestion_routines = _object_contract(
        _ALL_ROUTINES,
        {
            "public.immutable_unaccent(text)": frozenset({"EXECUTE"}),
            "bddk_meta.current_corpus_state_sha256(text)": frozenset({"EXECUTE"}),
            "bddk_meta.corpus_retrieval_ready(text)": frozenset({"EXECUTE"}),
        },
    )
    publisher_routines = _object_contract(
        _ALL_ROUTINES,
        {
            "bddk_meta.current_corpus_state_sha256(text)": frozenset({"EXECUTE"}),
            "bddk_meta.corpus_retrieval_ready(text)": frozenset({"EXECUTE"}),
            "bddk_meta.publish_verified_corpus_release(text, text, text, integer, integer, integer, text)": (
                frozenset({"EXECUTE"})
            ),
        },
    )
    operator_routines = _object_contract(
        _ALL_ROUTINES,
        {
            "public.immutable_unaccent(text)": frozenset({"EXECUTE"}),
            "bddk_meta.current_corpus_state_sha256(text)": frozenset({"EXECUTE"}),
            "bddk_meta.corpus_retrieval_ready(text)": frozenset({"EXECUTE"}),
            "bddk_meta.resolve_regulation_status(text, date)": frozenset({"EXECUTE"}),
        },
    )

    return MappingProxyType(
        {
            "public": _IdentityContract(
                memberships=frozenset({"bddk_public_reader"}),
                schemas=public_schemas,
                tables=_object_contract(_ALL_TABLES, read_tables),
                sequences=no_sequences,
                routines=public_routines,
            ),
            "ingestion": _IdentityContract(
                memberships=frozenset({"bddk_ingestion"}),
                schemas=public_schemas,
                tables=_object_contract(_ALL_TABLES, ingestion_tables),
                sequences=ingestion_sequences,
                routines=ingestion_routines,
            ),
            "release-publisher": _IdentityContract(
                memberships=frozenset({"bddk_release_publisher"}),
                schemas=public_schemas,
                tables=_object_contract(_ALL_TABLES, publisher_tables),
                sequences=no_sequences,
                routines=publisher_routines,
            ),
            "operator": _IdentityContract(
                memberships=frozenset(
                    {
                        "bddk_public_reader",
                        "bddk_ingestion",
                        "bddk_operator_runtime",
                    }
                ),
                schemas=operator_schemas,
                tables=_object_contract(_ALL_TABLES, operator_tables),
                sequences=ingestion_sequences,
                routines=operator_routines,
            ),
        }
    )


_CONTRACTS = _build_contracts()

_IDENTITY_SQL = """
WITH RECURSIVE session_role AS (
    SELECT role.oid,
           role.rolname,
           role.rolcanlogin,
           role.rolinherit,
           role.rolsuper,
           role.rolcreaterole,
           role.rolcreatedb,
           role.rolreplication,
           role.rolbypassrls
    FROM pg_catalog.pg_roles AS role
    WHERE role.rolname = session_user
), role_closure AS (
    SELECT * FROM session_role
    UNION
    SELECT inherited_role.oid,
           inherited_role.rolname,
           inherited_role.rolcanlogin,
           inherited_role.rolinherit,
           inherited_role.rolsuper,
           inherited_role.rolcreaterole,
           inherited_role.rolcreatedb,
           inherited_role.rolreplication,
           inherited_role.rolbypassrls
    FROM role_closure AS member_role
    JOIN pg_catalog.pg_auth_members AS membership
      ON membership.member = member_role.oid
    JOIN pg_catalog.pg_roles AS inherited_role
      ON inherited_role.oid = membership.roleid
), direct_memberships AS (
    SELECT inherited_role.rolname
    FROM session_role
    JOIN pg_catalog.pg_auth_members AS membership
      ON membership.member = session_role.oid
    JOIN pg_catalog.pg_roles AS inherited_role
      ON inherited_role.oid = membership.roleid
)
SELECT current_user::pg_catalog.text AS current_user_name,
       session_user::pg_catalog.text AS session_user_name,
       COALESCE((SELECT rolcanlogin FROM session_role), false) AS session_can_login,
       COALESCE((SELECT rolinherit FROM session_role), false) AS session_inherits,
       COALESCE(
           ARRAY(SELECT rolname FROM direct_memberships ORDER BY rolname),
           ARRAY[]::pg_catalog.text[]
       ) AS direct_memberships,
       COALESCE(
           ARRAY(
               SELECT rolname
               FROM role_closure
               WHERE rolname <> session_user
               ORDER BY rolname
           ),
           ARRAY[]::pg_catalog.text[]
       ) AS inherited_memberships,
       COALESCE(
           ARRAY(
               SELECT rolname
               FROM role_closure
               WHERE rolsuper
                  OR rolcreaterole
                  OR rolcreatedb
                  OR rolreplication
                  OR rolbypassrls
                  OR (rolname <> session_user AND rolcanlogin)
               ORDER BY rolname
           ),
           ARRAY[]::pg_catalog.text[]
       ) AS unsafe_roles,
       EXISTS (
           SELECT 1
           FROM role_closure AS member_role
           JOIN pg_catalog.pg_auth_members AS membership
             ON membership.member = member_role.oid
           WHERE membership.admin_option
       ) AS membership_admin,
       pg_catalog.array_remove(ARRAY[
           CASE WHEN pg_catalog.has_database_privilege(
               current_user, current_database(), 'CONNECT'
           ) THEN 'CONNECT' END,
           CASE WHEN pg_catalog.has_database_privilege(
               current_user, current_database(), 'CREATE'
           ) THEN 'CREATE' END,
           CASE WHEN pg_catalog.has_database_privilege(
               current_user, current_database(), 'TEMPORARY'
           ) THEN 'TEMPORARY' END,
           CASE WHEN pg_catalog.has_database_privilege(
               current_user, current_database(), 'CONNECT WITH GRANT OPTION'
           ) THEN 'CONNECT_GRANT' END,
           CASE WHEN pg_catalog.has_database_privilege(
               current_user, current_database(), 'CREATE WITH GRANT OPTION'
           ) THEN 'CREATE_GRANT' END,
           CASE WHEN pg_catalog.has_database_privilege(
               current_user, current_database(), 'TEMPORARY WITH GRANT OPTION'
           ) THEN 'TEMPORARY_GRANT' END
       ], NULL) AS database_privileges
"""

_SCHEMAS_SQL = """
SELECT namespace.nspname AS object_name,
       pg_catalog.array_remove(ARRAY[
           CASE WHEN pg_catalog.has_schema_privilege(
               current_user, namespace.oid, 'USAGE'
           ) THEN 'USAGE' END,
           CASE WHEN pg_catalog.has_schema_privilege(
               current_user, namespace.oid, 'CREATE'
           ) THEN 'CREATE' END,
           CASE WHEN pg_catalog.has_schema_privilege(
               current_user, namespace.oid, 'USAGE WITH GRANT OPTION'
           ) THEN 'USAGE_GRANT' END,
           CASE WHEN pg_catalog.has_schema_privilege(
               current_user, namespace.oid, 'CREATE WITH GRANT OPTION'
           ) THEN 'CREATE_GRANT' END
       ], NULL) AS privileges
FROM pg_catalog.pg_namespace AS namespace
WHERE namespace.nspname NOT IN ('pg_catalog', 'information_schema')
  AND namespace.nspname NOT LIKE 'pg_toast%'
  AND namespace.nspname NOT LIKE 'pg_temp_%'
ORDER BY namespace.nspname
"""

_RELATIONS_SQL = """
SELECT namespace.nspname || '.' || relation.relname AS object_name,
       relation.relkind,
       pg_catalog.array_remove(ARRAY[
           CASE WHEN pg_catalog.has_table_privilege(current_user, relation.oid, 'SELECT')
               THEN 'SELECT' END,
           CASE WHEN pg_catalog.has_table_privilege(current_user, relation.oid, 'INSERT')
               THEN 'INSERT' END,
           CASE WHEN pg_catalog.has_table_privilege(current_user, relation.oid, 'UPDATE')
               THEN 'UPDATE' END,
           CASE WHEN pg_catalog.has_table_privilege(current_user, relation.oid, 'DELETE')
               THEN 'DELETE' END,
           CASE WHEN pg_catalog.has_table_privilege(current_user, relation.oid, 'TRUNCATE')
               THEN 'TRUNCATE' END,
           CASE WHEN pg_catalog.has_table_privilege(current_user, relation.oid, 'REFERENCES')
               THEN 'REFERENCES' END,
           CASE WHEN pg_catalog.has_table_privilege(current_user, relation.oid, 'TRIGGER')
               THEN 'TRIGGER' END,
           CASE WHEN pg_catalog.current_setting('server_version_num')::pg_catalog.int4 >= 170000
                  AND pg_catalog.has_table_privilege(current_user, relation.oid, 'MAINTAIN')
               THEN 'MAINTAIN' END,
           CASE WHEN NOT pg_catalog.has_table_privilege(current_user, relation.oid, 'SELECT')
                  AND pg_catalog.has_any_column_privilege(current_user, relation.oid, 'SELECT')
               THEN 'SELECT_COLUMNS' END,
           CASE WHEN NOT pg_catalog.has_table_privilege(current_user, relation.oid, 'INSERT')
                  AND pg_catalog.has_any_column_privilege(current_user, relation.oid, 'INSERT')
               THEN 'INSERT_COLUMNS' END,
           CASE WHEN NOT pg_catalog.has_table_privilege(current_user, relation.oid, 'UPDATE')
                  AND pg_catalog.has_any_column_privilege(current_user, relation.oid, 'UPDATE')
               THEN 'UPDATE_COLUMNS' END,
           CASE WHEN NOT pg_catalog.has_table_privilege(current_user, relation.oid, 'REFERENCES')
                  AND pg_catalog.has_any_column_privilege(current_user, relation.oid, 'REFERENCES')
               THEN 'REFERENCES_COLUMNS' END,
           CASE WHEN pg_catalog.has_table_privilege(
               current_user, relation.oid, 'SELECT WITH GRANT OPTION'
           ) THEN 'SELECT_GRANT' END,
           CASE WHEN pg_catalog.has_table_privilege(
               current_user, relation.oid, 'INSERT WITH GRANT OPTION'
           ) THEN 'INSERT_GRANT' END,
           CASE WHEN pg_catalog.has_table_privilege(
               current_user, relation.oid, 'UPDATE WITH GRANT OPTION'
           ) THEN 'UPDATE_GRANT' END,
           CASE WHEN pg_catalog.has_table_privilege(
               current_user, relation.oid, 'DELETE WITH GRANT OPTION'
           ) THEN 'DELETE_GRANT' END,
           CASE WHEN pg_catalog.has_table_privilege(
               current_user, relation.oid, 'TRUNCATE WITH GRANT OPTION'
           ) THEN 'TRUNCATE_GRANT' END,
           CASE WHEN pg_catalog.has_table_privilege(
               current_user, relation.oid, 'REFERENCES WITH GRANT OPTION'
           ) THEN 'REFERENCES_GRANT' END,
           CASE WHEN pg_catalog.has_table_privilege(
               current_user, relation.oid, 'TRIGGER WITH GRANT OPTION'
           ) THEN 'TRIGGER_GRANT' END,
           CASE WHEN pg_catalog.current_setting('server_version_num')::pg_catalog.int4 >= 170000
                  AND pg_catalog.has_table_privilege(
                      current_user, relation.oid, 'MAINTAIN WITH GRANT OPTION'
                  ) THEN 'MAINTAIN_GRANT' END
       ], NULL) AS privileges
FROM pg_catalog.pg_class AS relation
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = relation.relnamespace
WHERE namespace.nspname NOT IN ('pg_catalog', 'information_schema')
  AND namespace.nspname NOT LIKE 'pg_toast%'
  AND namespace.nspname NOT LIKE 'pg_temp_%'
  AND relation.relkind IN ('r', 'p', 'v', 'm', 'f')
ORDER BY namespace.nspname, relation.relname
"""

_SEQUENCES_SQL = """
SELECT namespace.nspname || '.' || relation.relname AS object_name,
       pg_catalog.array_remove(ARRAY[
           CASE WHEN pg_catalog.has_sequence_privilege(current_user, relation.oid, 'USAGE')
               THEN 'USAGE' END,
           CASE WHEN pg_catalog.has_sequence_privilege(current_user, relation.oid, 'SELECT')
               THEN 'SELECT' END,
           CASE WHEN pg_catalog.has_sequence_privilege(current_user, relation.oid, 'UPDATE')
               THEN 'UPDATE' END,
           CASE WHEN pg_catalog.has_sequence_privilege(
               current_user, relation.oid, 'USAGE WITH GRANT OPTION'
           ) THEN 'USAGE_GRANT' END,
           CASE WHEN pg_catalog.has_sequence_privilege(
               current_user, relation.oid, 'SELECT WITH GRANT OPTION'
           ) THEN 'SELECT_GRANT' END,
           CASE WHEN pg_catalog.has_sequence_privilege(
               current_user, relation.oid, 'UPDATE WITH GRANT OPTION'
           ) THEN 'UPDATE_GRANT' END
       ], NULL) AS privileges
FROM pg_catalog.pg_class AS relation
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = relation.relnamespace
WHERE namespace.nspname NOT IN ('pg_catalog', 'information_schema')
  AND namespace.nspname NOT LIKE 'pg_toast%'
  AND namespace.nspname NOT LIKE 'pg_temp_%'
  AND relation.relkind = 'S'
ORDER BY namespace.nspname, relation.relname
"""

_ROUTINES_SQL = """
SELECT namespace.nspname || '.' || routine.proname || '('
           || pg_catalog.oidvectortypes(routine.proargtypes) || ')' AS object_name,
       pg_catalog.array_remove(ARRAY[
           CASE WHEN pg_catalog.has_function_privilege(current_user, routine.oid, 'EXECUTE')
               THEN 'EXECUTE' END,
           CASE WHEN pg_catalog.has_function_privilege(
               current_user, routine.oid, 'EXECUTE WITH GRANT OPTION'
           ) THEN 'EXECUTE_GRANT' END
       ], NULL) AS privileges
FROM pg_catalog.pg_proc AS routine
JOIN pg_catalog.pg_namespace AS namespace
  ON namespace.oid = routine.pronamespace
WHERE namespace.nspname NOT IN ('pg_catalog', 'information_schema')
  AND namespace.nspname NOT LIKE 'pg_toast%'
  AND namespace.nspname NOT LIKE 'pg_temp_%'
  AND routine.prokind IN ('f', 'p')
  AND NOT EXISTS (
      SELECT 1
      FROM pg_catalog.pg_depend AS dependency
      WHERE dependency.classid = 'pg_catalog.pg_proc'::pg_catalog.regclass
        AND dependency.objid = routine.oid
        AND dependency.refclassid = 'pg_catalog.pg_extension'::pg_catalog.regclass
        AND dependency.deptype = 'e'
  )
ORDER BY namespace.nspname, routine.proname,
         pg_catalog.oidvectortypes(routine.proargtypes)
"""

_ACL_PROVENANCE_SQL = """
WITH session_role AS (
    SELECT role.oid
    FROM pg_catalog.pg_roles AS role
    WHERE role.rolname = session_user
), database_acl AS (
    SELECT acl.grantee
    FROM pg_catalog.pg_database AS database_record
    CROSS JOIN LATERAL pg_catalog.aclexplode(
        COALESCE(
            database_record.datacl,
            pg_catalog.acldefault('d'::"char", database_record.datdba)
        )
    ) AS acl
    WHERE database_record.datname = current_database()
), schema_acl AS (
    SELECT acl.grantee
    FROM pg_catalog.pg_namespace AS namespace
    CROSS JOIN LATERAL pg_catalog.aclexplode(
        COALESCE(
            namespace.nspacl,
            pg_catalog.acldefault('n'::"char", namespace.nspowner)
        )
    ) AS acl
    WHERE namespace.nspname IN ('public', 'bddk_meta', 'bddk_operator')
      AND NOT (
          namespace.nspname = 'public'
          AND acl.grantee = 0
          AND acl.privilege_type = 'USAGE'
      )
), relation_acl AS (
    SELECT acl.grantee
    FROM pg_catalog.pg_class AS relation
    JOIN pg_catalog.pg_namespace AS namespace
      ON namespace.oid = relation.relnamespace
    CROSS JOIN LATERAL pg_catalog.aclexplode(
        COALESCE(
            relation.relacl,
            pg_catalog.acldefault(
                CASE WHEN relation.relkind = 'S' THEN 's'::"char" ELSE 'r'::"char" END,
                relation.relowner
            )
        )
    ) AS acl
    WHERE namespace.nspname IN ('public', 'bddk_meta', 'bddk_operator')
      AND relation.relkind IN ('r', 'p', 'v', 'm', 'f', 'S')
), routine_acl AS (
    SELECT acl.grantee
    FROM pg_catalog.pg_proc AS routine
    JOIN pg_catalog.pg_namespace AS namespace
      ON namespace.oid = routine.pronamespace
    CROSS JOIN LATERAL pg_catalog.aclexplode(
        COALESCE(
            routine.proacl,
            pg_catalog.acldefault('f'::"char", routine.proowner)
        )
    ) AS acl
    WHERE namespace.nspname IN ('public', 'bddk_meta', 'bddk_operator')
      AND NOT EXISTS (
          SELECT 1
          FROM pg_catalog.pg_depend AS dependency
          WHERE dependency.classid = 'pg_catalog.pg_proc'::pg_catalog.regclass
            AND dependency.objid = routine.oid
            AND dependency.refclassid = 'pg_catalog.pg_extension'::pg_catalog.regclass
            AND dependency.deptype = 'e'
      )
), acl_entries AS (
    SELECT grantee FROM database_acl
    UNION ALL
    SELECT grantee FROM schema_acl
    UNION ALL
    SELECT grantee FROM relation_acl
    UNION ALL
    SELECT grantee FROM routine_acl
)
SELECT EXISTS (
           SELECT 1 FROM acl_entries WHERE grantee = 0
       ) AS public_acl_leakage,
       EXISTS (
           SELECT 1
           FROM acl_entries
           WHERE grantee = (SELECT oid FROM session_role)
       ) AS direct_login_acl
"""


def _row_value(row: Any, key: str, default: Any = None) -> Any:
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def _privilege_map(rows: list[Any], *, table_rows: bool = False) -> Mapping[str, frozenset[str]]:
    result: dict[str, frozenset[str]] = {}
    for row in rows:
        name = str(_row_value(row, "object_name", ""))
        if not name or name in result:
            raise ValueError("database identity inspection returned an invalid object inventory")
        privileges = frozenset(str(value) for value in (_row_value(row, "privileges", ()) or ()))
        if table_rows:
            unknown = privileges - _TABLE_PRIVILEGES
            if unknown:
                raise ValueError("database identity inspection returned invalid table privileges")
        result[name] = privileges
    return MappingProxyType(result)


async def inspect_database_identity(pool: asyncpg.Pool) -> DatabaseIdentityInspection:
    """Inspect one pool's effective LOGIN and application-object privileges."""

    async with pool.acquire() as connection:
        return await inspect_database_connection_identity(connection)


async def inspect_database_connection_identity(
    connection: asyncpg.Connection,
) -> DatabaseIdentityInspection:
    """Inspect one physical connection before it is admitted to a pool."""

    await assert_supported_postgresql(connection)
    identity = await connection.fetchrow(_IDENTITY_SQL)
    schema_rows = await connection.fetch(_SCHEMAS_SQL)
    relation_rows = await connection.fetch(_RELATIONS_SQL)
    sequence_rows = await connection.fetch(_SEQUENCES_SQL)
    routine_rows = await connection.fetch(_ROUTINES_SQL)
    acl_provenance = await connection.fetchrow(_ACL_PROVENANCE_SQL)

    if identity is None:
        raise ValueError("database identity inspection returned no identity")
    if acl_provenance is None:
        raise ValueError("database identity inspection returned no ACL provenance")
    current_user = str(_row_value(identity, "current_user_name", ""))
    session_user = str(_row_value(identity, "session_user_name", ""))
    if not current_user or not session_user:
        raise ValueError("database identity inspection returned an invalid identity")

    return DatabaseIdentityInspection(
        current_user=current_user,
        session_user=session_user,
        session_can_login=bool(_row_value(identity, "session_can_login", False)),
        session_inherits=bool(_row_value(identity, "session_inherits", False)),
        direct_memberships=frozenset(_row_value(identity, "direct_memberships", ()) or ()),
        inherited_memberships=frozenset(_row_value(identity, "inherited_memberships", ()) or ()),
        unsafe_roles=frozenset(_row_value(identity, "unsafe_roles", ()) or ()),
        membership_admin=bool(_row_value(identity, "membership_admin", False)),
        public_acl_leakage=bool(_row_value(acl_provenance, "public_acl_leakage", True)),
        direct_login_acl=bool(_row_value(acl_provenance, "direct_login_acl", True)),
        database_privileges=frozenset(_row_value(identity, "database_privileges", ()) or ()),
        schemas=_privilege_map(schema_rows),
        tables=_privilege_map(relation_rows, table_rows=True),
        sequences=_privilege_map(sequence_rows),
        routines=_privilege_map(routine_rows),
    )


def identity_contract_failures(
    inspection: DatabaseIdentityInspection,
    profile: DatabaseIdentityProfile,
) -> tuple[str, ...]:
    """Return bounded contract labels for tests and internal diagnostics."""

    contract = _CONTRACTS.get(profile)
    if contract is None:
        return ("unknown_profile",)

    failures: list[str] = []
    if inspection.current_user != inspection.session_user:
        failures.append("session_role_changed")
    if not inspection.session_can_login or not inspection.session_inherits:
        failures.append("login_attributes")
    if inspection.unsafe_roles:
        failures.append("unsafe_role_attributes")
    if inspection.membership_admin:
        failures.append("membership_admin")
    if inspection.public_acl_leakage:
        failures.append("public_acl_leakage")
    if inspection.direct_login_acl:
        failures.append("direct_login_acl")
    if inspection.direct_memberships != contract.memberships:
        failures.append("direct_memberships")
    if inspection.inherited_memberships != contract.memberships:
        failures.append("inherited_memberships")
    if inspection.database_privileges != frozenset({"CONNECT"}):
        failures.append("database_privileges")
    if inspection.schemas != contract.schemas:
        failures.append("schema_privileges")
    if inspection.tables != contract.tables:
        failures.append("table_privileges")
    if inspection.sequences != contract.sequences:
        failures.append("sequence_privileges")
    if inspection.routines != contract.routines:
        failures.append("routine_privileges")
    return tuple(failures)


async def assert_database_identity(
    pool: asyncpg.Pool,
    profile: DatabaseIdentityProfile,
) -> None:
    """Require the exact least-privilege identity for ``profile``.

    Error text deliberately excludes role names, connection details, and ACLs.
    Detailed differences are available only through the pure
    :func:`identity_contract_failures` helper used by tests.
    """

    if profile not in _CONTRACTS:
        raise DatabaseIdentityError("Unknown database workload identity profile.")
    try:
        inspection = await inspect_database_identity(pool)
        if identity_contract_failures(inspection, profile):
            raise DatabaseIdentityError(
                f"The configured {profile} database identity does not satisfy its exact least-privilege contract."
            )
    except DatabaseIdentityError:
        raise
    except PostgreSQLCompatibilityError as exc:
        raise DatabaseIdentityError(str(exc)) from None
    except Exception:
        raise DatabaseIdentityError(
            f"The configured {profile} database identity could not be verified against its least-privilege contract."
        ) from None


async def assert_database_connection_identity(
    connection: asyncpg.Connection,
    *,
    profile: DatabaseIdentityProfile,
) -> None:
    """Reject a newly opened physical connection unless its full contract matches.

    This function is suitable for ``asyncpg.create_pool(init=...)``.  It is
    intentionally independent of a pool so proxies, failover, or credential
    remapping cannot introduce a differently privileged connection after the
    one-time startup attestation has passed.
    """

    if profile not in _CONTRACTS:
        raise DatabaseIdentityError("Unknown database workload identity profile.")
    try:
        inspection = await inspect_database_connection_identity(connection)
        if identity_contract_failures(inspection, profile):
            raise DatabaseIdentityError(
                f"A {profile} database connection does not satisfy its exact least-privilege contract."
            )
    except DatabaseIdentityError:
        raise
    except PostgreSQLCompatibilityError as exc:
        raise DatabaseIdentityError(str(exc)) from None
    except Exception:
        raise DatabaseIdentityError(
            f"A {profile} database connection could not be verified against its least-privilege contract."
        ) from None
