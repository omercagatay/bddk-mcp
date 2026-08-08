"""Global immutable PostgreSQL migration API."""

from bddk_mcp.migrations.legacy import LegacyAdoptionError
from bddk_mcp.migrations.model import Migration
from bddk_mcp.migrations.runner import (
    LATEST_SCHEMA_VERSION,
    MIGRATION_LOCK_TIMEOUT,
    MIGRATION_STATEMENT_TIMEOUT,
    MIGRATIONS,
    MigrationCompatibilityError,
    MigrationError,
    MigrationHistoryError,
    MigrationLockTimeoutError,
    MigrationNotReadyError,
    MigrationPrerequisiteError,
    MigrationScaleError,
    MigrationState,
    MigrationStatementTimeoutError,
    assert_migrations_current,
    inspect_migration_state,
    inspect_migration_state_connection,
    migrate,
    validate_migration_history,
)

__all__ = (
    "LATEST_SCHEMA_VERSION",
    "MIGRATIONS",
    "MIGRATION_LOCK_TIMEOUT",
    "MIGRATION_STATEMENT_TIMEOUT",
    "Migration",
    "LegacyAdoptionError",
    "MigrationCompatibilityError",
    "MigrationError",
    "MigrationHistoryError",
    "MigrationLockTimeoutError",
    "MigrationNotReadyError",
    "MigrationPrerequisiteError",
    "MigrationScaleError",
    "MigrationState",
    "MigrationStatementTimeoutError",
    "assert_migrations_current",
    "inspect_migration_state",
    "inspect_migration_state_connection",
    "migrate",
    "validate_migration_history",
)
