"""Regression checks for documented MCP tool-surface and deployment facts."""

import json
import tomllib
from pathlib import Path

from bddk_mcp.tools.registry import PUBLIC_TOOL_NAMES

ROOT = Path(__file__).resolve().parents[1]

_RETIRED_STRINGS = (
    "BDDK_ADMIN_TOOLS",
    "BDDK_TOOL_PROFILE",
    "BDDK_OPERATOR_DATABASE_URL",
    "runtime-admin",
    "backfill_status",
    "publish-corpus-release",
    "bddk-mcp bootstrap",
    "admin-ui",
    "get_operator_job",
    "BDDK_DATABASE_URL",
    "bddk-seed",
    "bddk-mcp migrate",
    "pgvector",
)


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_readmes_list_every_reviewed_tool_and_no_retired_surface():
    assert len(PUBLIC_TOOL_NAMES) == 14

    for path in ("README.md", "README.en.md"):
        readme = _read(path)
        for tool_name in PUBLIC_TOOL_NAMES:
            assert f"`{tool_name}`" in readme, (path, tool_name)
        for retired in _RETIRED_STRINGS:
            assert retired not in readme, (path, retired)


def test_project_mcp_config_is_portable_and_uses_packaged_entry_point():
    raw = _read(".mcp.json")
    config = json.loads(raw)["mcpServers"]["bddk"]

    assert config["command"] == "uv"
    assert config["args"] == ["run", "--frozen", "bddk-mcp"]
    assert config["env"]["MCP_TRANSPORT"] == "stdio"
    assert "/home/" not in raw
    assert "server.py" not in raw


def test_container_and_deployment_docs_use_packaged_entry_point():
    content = _read("Dockerfile")
    assert "bddk-mcp" in content
    assert "python server.py" not in content
    assert "deploy/trust" not in content

    deployment = _read("docs/DEPLOYMENT.md")
    assert "## Bank deployment" in deployment
    assert "not bank acceptance or a production-ready platform configuration" in deployment
    assert "14 public tools" in deployment
    assert "stateless JSON responses" in deployment
    assert "GET /health/live" in deployment
    assert "GET /health/ready" in deployment
    assert "global ingress limit" in deployment
    assert "process-local and non-durable" in deployment
    assert "bddk-mcp build" in deployment
    for retired in _RETIRED_STRINGS:
        assert retired not in deployment, retired
    for variable in (
        "BDDK_HTTP_ALLOWED_HOSTS",
        "BDDK_HTTP_ALLOWED_ORIGINS",
        "BDDK_JWT_ISSUER",
        "BDDK_JWT_RESOURCE",
        "BDDK_JWT_JWKS_URL",
        "BDDK_JWT_AUDIENCE",
        "BDDK_JWT_REQUIRED_SCOPES",
        "BDDK_JWT_MAX_TOKEN_LENGTH",
        "BDDK_JWT_ALGORITHMS",
        "BDDK_JWT_ACCESS_TOKEN_TYPES",
        "BDDK_TLS_CERT_FILE",
        "BDDK_TLS_KEY_FILE",
        "BDDK_HTTP_MAX_BODY_BYTES",
        "BDDK_HTTP_MAX_CONCURRENCY",
        "BDDK_HTTP_RATE_LIMIT_PER_MINUTE",
    ):
        assert variable in deployment


def test_environment_example_records_the_http_boundary():
    example = _read(".env.example")

    for variable in (
        "BDDK_CORPUS_PATH",
        "BDDK_HTTP_ALLOWED_HOSTS",
        "BDDK_HTTP_ALLOWED_ORIGINS",
        "BDDK_JWT_ISSUER",
        "BDDK_JWT_RESOURCE",
        "BDDK_JWT_JWKS_URL",
        "BDDK_JWT_AUDIENCE",
        "BDDK_JWT_REQUIRED_SCOPES",
        "BDDK_JWT_ACCESS_TOKEN_TYPES",
        "BDDK_TLS_CERT_FILE",
        "BDDK_TLS_KEY_FILE",
        "BDDK_HTTP_MAX_BODY_BYTES",
        "BDDK_HTTP_MAX_CONCURRENCY",
        "BDDK_HTTP_RATE_LIMIT_PER_MINUTE",
    ):
        assert variable in example
    assert "/health/live" in example
    assert "/health/ready" in example
    assert "process-local" in example
    for retired in _RETIRED_STRINGS:
        assert retired not in example, retired


def test_runtime_distribution_is_the_package_only():
    project = tomllib.loads(_read("pyproject.toml"))

    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == ["bddk_mcp*"]
    assert "anthropic" not in " ".join(project["project"]["dependencies"])
    assert "jinja2" not in " ".join(project["project"]["dependencies"])
    assert "benchmark" not in project["dependency-groups"]


def test_status_document_pins_current_facts():
    status = _read("docs/STATUS.md")

    assert "14 tools" in status
    assert "One SQLite file" in status
    assert "corpus/bddk.sqlite" in status
    assert "schema version 1" in status
    assert "13,604 chunks" in status
    assert "archive/pre-simplify-2026-11-09" in status
    for retired in _RETIRED_STRINGS:
        assert retired not in status, retired
