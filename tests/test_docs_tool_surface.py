"""Regression checks for documented MCP tool-surface counts."""

import json
import tomllib
from pathlib import Path

from bddk_mcp.tools.registry import OPERATOR_TOOL_NAMES, PUBLIC_TOOL_NAMES

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_readmes_distinguish_public_and_operator_tool_counts():
    assert len(PUBLIC_TOOL_NAMES) == 15
    assert len(OPERATOR_TOOL_NAMES) == 13
    assert {
        "get_operator_job",
        "list_operator_jobs",
        "cancel_operator_job",
    } <= set(OPERATOR_TOOL_NAMES)
    assert "backfill_status" not in OPERATOR_TOOL_NAMES

    for path in ("README.md", "README.en.md"):
        readme = _read(path)
        assert "BDDK_TOOL_PROFILE=public" in readme
        assert "BDDK_TOOL_PROFILE=operator" in readme
        assert "15 public tools plus 13 operator tools" in readme
        assert "28 tools total" in readme
        assert "BDDK_OPERATOR_DATABASE_URL" in readme
        for tool_name in PUBLIC_TOOL_NAMES + OPERATOR_TOOL_NAMES:
            assert f"`{tool_name}`" in readme

    assert "canonical operator registry" in _read("README.md")


def test_benchmark_docs_record_exposed_tool_profiles():
    benchmark_readme = _read("benchmark/README.md")

    assert "runtime-public" in benchmark_readme
    assert "runtime-operator" in benchmark_readme
    assert "benchmark-operator-contract" in benchmark_readme
    assert "| `runtime-public` | 15 |" in benchmark_readme
    assert "| `runtime-operator` | 28 |" in benchmark_readme
    assert "| `benchmark-operator-contract` | 28 |" in benchmark_readme
    assert "live_tool_list" in benchmark_readme
    assert "live_tool_schema_sha256" in benchmark_readme
    assert "official MCP Python client" in benchmark_readme
    assert "does not use a custom" in benchmark_readme
    assert "/call-tool" in benchmark_readme
    for tool_name in PUBLIC_TOOL_NAMES + OPERATOR_TOOL_NAMES:
        assert f"`{tool_name}`" in benchmark_readme


def test_operational_docs_do_not_recommend_legacy_combined_admin_profile():
    for path in ("README.md", "README.en.md", "docs/DEPLOYMENT.md", ".env.example", "benchmark/README.md"):
        content = _read(path)
        assert "BDDK_ADMIN_TOOLS" not in content
        assert "runtime-admin" not in content
        assert "backfill_status" not in content


def test_project_mcp_config_is_portable_and_uses_packaged_entry_point():
    raw = _read(".mcp.json")
    config = json.loads(raw)["mcpServers"]["bddk"]

    assert config["command"] == "uv"
    assert config["args"] == ["run", "--frozen", "bddk-mcp"]
    assert config["env"]["MCP_TRANSPORT"] == "stdio"
    assert "/home/" not in raw
    assert "server.py" not in raw


def test_container_and_deployment_docs_use_packaged_entry_point():
    for path in ("Dockerfile", "Dockerfile.spaces", "Procfile"):
        content = _read(path)
        assert "bddk-mcp" in content
        assert "python server.py" not in content

    deployment = _read("docs/DEPLOYMENT.md")
    assert "OpenShift AI Starter" in deployment
    assert "deploy/openshift" in deployment
    assert "not bank acceptance or a production-ready platform configuration" in deployment
    assert "15 public tools" in deployment
    assert "28 total tools" in deployment
    assert "stateless JSON responses" in deployment
    assert "GET /health/live" in deployment
    assert "GET /health/ready" in deployment
    assert "BDDK_OPERATOR_DATABASE_URL" in deployment
    assert "BDDK_OPERATOR_REMOTE_ENABLED" in deployment
    assert "global ingress limit" in deployment
    assert "process-local and non-durable" in deployment
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


def test_environment_example_records_profile_http_and_job_boundaries():
    example = _read(".env.example")

    for variable in (
        "BDDK_TOOL_PROFILE",
        "BDDK_OPERATOR_DATABASE_URL",
        "BDDK_OPERATOR_REMOTE_ENABLED",
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
        "BDDK_OPERATOR_JOB_HISTORY",
        "BDDK_OPERATOR_JOB_DRAIN_TIMEOUT",
    ):
        assert variable in example
    assert "/health/live" in example
    assert "/health/ready" in example
    assert "process-local" in example
    assert "lost on restart" in example


def test_runtime_distribution_excludes_repository_only_benchmark():
    project = tomllib.loads(_read("pyproject.toml"))

    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == ["bddk_mcp*"]
    assert "anthropic>=0.40" not in project["project"]["dependencies"]
    assert "anthropic>=0.40" in project["dependency-groups"]["benchmark"]
    assert "uv sync --group benchmark" in _read("benchmark/README.md")
