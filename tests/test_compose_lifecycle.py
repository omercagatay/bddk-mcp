"""Static checks for the disposable local Compose lifecycle."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

import yaml

ROOT = Path(__file__).parents[1]
COMPOSE_PATH = ROOT / "docker-compose.yml"


def _compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def _dependency(service: dict) -> str:
    dependencies = service["depends_on"]
    assert len(dependencies) == 1
    name, policy = next(iter(dependencies.items()))
    assert policy["condition"] in {"service_healthy", "service_completed_successfully"}
    return name


def test_compose_has_explicit_ordered_database_lifecycle():
    services = _compose()["services"]
    assert _dependency(services["bddk-db-roles"]) == "db"
    assert _dependency(services["bddk-local-identities"]) == "bddk-db-roles"
    assert _dependency(services["bddk-migrate"]) == "bddk-local-identities"
    assert _dependency(services["bddk-db-grants"]) == "bddk-migrate"
    assert _dependency(services["bddk-bootstrap"]) == "bddk-db-grants"
    assert _dependency(services["bddk"]) == "bddk-bootstrap"
    assert _dependency(services["bddk-operator"]) == "bddk-bootstrap"

    assert services["bddk-migrate"]["command"][-1] == "migrate"
    assert services["bddk-bootstrap"]["command"][:2] == [".venv/bin/bddk-mcp", "bootstrap"]
    assert "--reindex-existing" in services["bddk-bootstrap"]["command"]
    assert services["bddk-db-roles"]["command"][-1] == "--file=/sql/01_roles.sql"
    assert services["bddk-db-grants"]["command"][-1] == "--file=/sql/02_grants.sql"
    assert services["bddk-db-roles"]["environment"]["PGOPTIONS"] == "-c bddk.expected_database=bddk"
    assert services["bddk-db-grants"]["environment"]["PGOPTIONS"] == "-c bddk.expected_database=bddk"
    assert services["bddk-migrate"]["environment"]["BDDK_EXPECTED_DATABASE_NAME"] == "bddk"
    for service_name in ("bddk-migrate", "bddk-bootstrap", "bddk", "bddk-operator"):
        assert services[service_name]["environment"]["BDDK_ALLOW_INSECURE_DATABASE"] == "true"


def test_compose_database_identities_are_distinct_and_role_scoped():
    services = _compose()["services"]
    assignments = {
        "schema-owner": services["bddk-migrate"]["environment"]["BDDK_SCHEMA_OWNER_DATABASE_URL"],
        "ingestion": services["bddk-bootstrap"]["environment"]["BDDK_INGESTION_DATABASE_URL"],
        "public": services["bddk"]["environment"]["BDDK_DATABASE_URL"],
        "operator": services["bddk-operator"]["environment"]["BDDK_OPERATOR_DATABASE_URL"],
        "telemetry": services["bddk"]["environment"]["BDDK_TELEMETRY_DATABASE_URL"],
    }
    usernames = {name: urlsplit(dsn).username for name, dsn in assignments.items()}
    assert usernames == {
        "schema-owner": "bddk_local_migrator",
        "ingestion": "bddk_local_ingestion",
        "public": "bddk_local_public",
        "operator": "bddk_local_operator",
        "telemetry": "bddk_local_telemetry",
    }
    assert len(set(assignments.values())) == len(assignments)
    assert "role%3Dbddk_schema_owner" in assignments["schema-owner"]
    assert services["bddk-operator"]["environment"]["BDDK_TELEMETRY_DATABASE_URL"] == assignments["telemetry"]


def test_compose_remote_surfaces_are_loopback_published_and_fail_closed():
    services = _compose()["services"]
    assert services["db"]["ports"] == ["127.0.0.1:5432:5432"]
    assert services["bddk"]["ports"] == ["127.0.0.1:8000:8000"]
    assert services["bddk-operator"]["ports"] == ["127.0.0.1:8001:8000"]
    assert services["bddk-operator"]["profiles"] == ["operator"]
    assert services["bddk"]["environment"]["BDDK_TELEMETRY_ENABLED"] == "${BDDK_TELEMETRY_ENABLED:-false}"
    assert services["bddk-operator"]["environment"]["BDDK_TELEMETRY_ENABLED"] == ("${BDDK_TELEMETRY_ENABLED:-false}")


def test_local_login_fixture_is_explicitly_non_production_and_least_privileged():
    sql = (ROOT / "deploy" / "postgres" / "local-dev" / "01_identities.sql").read_text(encoding="utf-8")
    assert "LOCAL DEVELOPMENT ONLY" in sql
    assert "Never apply this file" in sql
    assert sql.count("NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS") == 5
    assert "GRANT bddk_schema_owner TO bddk_local_migrator" in sql
    assert "GRANT bddk_ingestion TO bddk_local_ingestion" in sql
    assert "GRANT bddk_public_reader TO bddk_local_public" in sql
    assert "GRANT bddk_telemetry_writer TO bddk_local_telemetry" in sql
