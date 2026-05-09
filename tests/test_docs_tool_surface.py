"""Regression checks for documented MCP tool-surface counts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_readme_distinguishes_public_and_admin_tool_counts():
    readme = _read("README.md")

    assert "BDDK_ADMIN_TOOLS=false" in readme
    assert "16 read-only tools" in readme
    assert "BDDK_ADMIN_TOOLS=true" in readme
    assert "26 tools" in readme
    assert "Total possible MCP tools" in readme


def test_benchmark_docs_record_exposed_tool_profiles():
    benchmark_readme = _read("benchmark/README.md")

    assert "runtime-public" in benchmark_readme
    assert "runtime-admin" in benchmark_readme
    assert "benchmark-schema-fixture" in benchmark_readme
    assert "16" in benchmark_readme
    assert "26" in benchmark_readme
    assert "23" in benchmark_readme
    assert "exposed_tool_list" in benchmark_readme
