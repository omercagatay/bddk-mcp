"""Regression checks for documented MCP tool-surface counts."""

import json
import tomllib
from pathlib import Path

from bddk_mcp.tools.registry import OPERATOR_TOOL_NAMES, PUBLIC_TOOL_NAMES

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_readme_distinguishes_public_and_admin_tool_counts():
    readme = _read("README.md")

    assert "BDDK_ADMIN_TOOLS=false" in readme
    assert "15 public tools" in readme
    assert "BDDK_ADMIN_TOOLS=true" in readme
    assert "15 public tools plus 11 operator tools" in readme
    assert "26 tools total" in readme
    assert "canonical operator registry" in readme
    for tool_name in PUBLIC_TOOL_NAMES + OPERATOR_TOOL_NAMES:
        assert f"`{tool_name}`" in readme


def test_benchmark_docs_record_exposed_tool_profiles():
    benchmark_readme = _read("benchmark/README.md")

    assert "runtime-public" in benchmark_readme
    assert "runtime-admin" in benchmark_readme
    assert "benchmark-operator-contract" in benchmark_readme
    assert "| `runtime-public` | 15 |" in benchmark_readme
    assert "26" in benchmark_readme
    assert "| `benchmark-operator-contract` | 26 |" in benchmark_readme
    assert "exposed_tool_list" in benchmark_readme
    assert "does not discover them from a live MCP `tools/list` response" in benchmark_readme


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
    assert "OpenShift AI Status" in deployment
    assert "does **not** yet contain production-ready OpenShift manifests" in deployment
    assert "no application-level authentication or rate limiting" in deployment
    assert "15 public tools" in deployment
    assert "26 total tools" in deployment


def test_runtime_distribution_excludes_repository_only_benchmark():
    project = tomllib.loads(_read("pyproject.toml"))

    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == ["bddk_mcp*"]
    assert "anthropic>=0.40" not in project["project"]["dependencies"]
    assert "anthropic>=0.40" in project["dependency-groups"]["benchmark"]
    assert "uv sync --group benchmark" in _read("benchmark/README.md")
