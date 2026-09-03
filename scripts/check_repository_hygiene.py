#!/usr/bin/env python3
"""Validate the repository surface and documentation map."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote

import yaml

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = {
    ".gitattributes",
    ".github/CODEOWNERS",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/dependabot.yml",
    ".github/ISSUE_TEMPLATE/bug.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/ISSUE_TEMPLATE/data-quality.yml",
    ".github/ISSUE_TEMPLATE/feature.yml",
    ".github/ISSUE_TEMPLATE/question.yml",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "docs/README.md",
    "docs/REPOSITORY_STRUCTURE.md",
    "docs/STATUS.md",
}

ALLOWED_TOP_LEVEL = {
    ".claude",
    ".dockerignore",
    ".env.example",
    ".gitattributes",
    ".github",
    ".gitignore",
    ".mcp.json",
    "CHANGELOG.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    "Dockerfile",
    "LICENSE",
    "MANIFEST.in",
    "Procfile",
    "README.en.md",
    "README.md",
    "SECURITY.md",
    "bddk_mcp",
    "benchmark",
    "data",
    "deploy",
    "docker-compose.yml",
    "docs",
    "pyproject.toml",
    "railway.toml",
    "scripts",
    "seed.py",
    "seed_data",
    "server.py",
    "supply-chain",
    "tests",
    "uv.lock",
}

JUNK_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "node_modules"}
JUNK_NAMES = {".DS_Store", ".env"}
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)\n]+)\)")
VERSION_ASSIGNMENT = re.compile(r'^__version__\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)


def _repository_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def _markdown_files(tracked: list[str]) -> list[Path]:
    return [ROOT / path for path in tracked if path.endswith(".md")]


def _relative_link_target(markdown: Path, raw_target: str) -> Path | None:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = target.split(maxsplit=1)[0]

    if not target or target.startswith(("#", "/", "http://", "https://", "mailto:", "data:")):
        return None

    path_text = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not path_text:
        return None
    return (markdown.parent / path_text).resolve()


def _check_issue_forms(errors: list[str]) -> None:
    forms = sorted((ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.yml"))
    for form in forms:
        try:
            payload = yaml.safe_load(form.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            errors.append(f"{form.relative_to(ROOT)} is not valid YAML: {exc}")
            continue

        if not isinstance(payload, dict):
            errors.append(f"{form.relative_to(ROOT)} must contain a YAML mapping")
            continue
        if form.name == "config.yml":
            if not isinstance(payload.get("blank_issues_enabled"), bool):
                errors.append(".github/ISSUE_TEMPLATE/config.yml must set blank_issues_enabled explicitly")
            continue

        for key in ("name", "description", "body"):
            if not payload.get(key):
                errors.append(f"{form.relative_to(ROOT)} is missing {key}")
        ids = [item.get("id") for item in payload.get("body", []) if isinstance(item, dict) and item.get("id")]
        if len(ids) != len(set(ids)):
            errors.append(f"{form.relative_to(ROOT)} contains duplicate field IDs")


def _check_dependabot(errors: list[str]) -> None:
    path = ROOT / ".github" / "dependabot.yml"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        errors.append(f".github/dependabot.yml is not valid YAML: {exc}")
        return

    if not isinstance(payload, dict) or payload.get("version") != 2:
        errors.append(".github/dependabot.yml must use configuration version 2")
        return
    updates = payload.get("updates")
    if not isinstance(updates, list):
        errors.append(".github/dependabot.yml must define an updates list")
        return
    ecosystems = {entry.get("package-ecosystem") for entry in updates if isinstance(entry, dict)}
    if ecosystems != {"uv", "github-actions"}:
        errors.append(".github/dependabot.yml must cover exactly uv and github-actions")


def main() -> int:
    errors: list[str] = []
    repository_files = _repository_files()
    repository_file_set = set(repository_files)

    missing = sorted(REQUIRED_FILES - repository_file_set)
    errors.extend(f"missing required repository file: {path}" for path in missing)

    unexpected = sorted({Path(path).parts[0] for path in repository_files} - ALLOWED_TOP_LEVEL)
    errors.extend(f"unexpected top-level entry: {path}" for path in unexpected)

    for path_text in repository_files:
        path = Path(path_text)
        if path.name in JUNK_NAMES or path.suffix == ".pyc" or JUNK_PARTS.intersection(path.parts):
            errors.append(f"tracked local/build artifact: {path_text}")

    for markdown in _markdown_files(repository_files):
        text = markdown.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = _relative_link_target(markdown, raw_target)
            if target is not None and not target.exists():
                errors.append(f"broken relative link in {markdown.relative_to(ROOT)}: {raw_target.strip()}")

    docs_index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    for document in sorted((ROOT / "docs").glob("*.md")):
        if document.name != "README.md" and f"({document.name})" not in docs_index:
            errors.append(f"docs/README.md does not index docs/{document.name}")

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_version = pyproject["project"]["version"]
    init_text = (ROOT / "bddk_mcp" / "__init__.py").read_text(encoding="utf-8")
    version_match = VERSION_ASSIGNMENT.search(init_text)
    if version_match is None or version_match.group(1) != package_version:
        errors.append("pyproject.toml and bddk_mcp.__version__ disagree")

    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    for marker in ("seed_data/*.json", "docs/evidence/*.json", "uv.lock", "tests/fixtures/*.pdf"):
        if marker not in attributes:
            errors.append(f".gitattributes does not classify {marker}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for link in ("docs/README.md", "CONTRIBUTING.md", "SECURITY.md"):
        if f"]({link})" not in readme:
            errors.append(f"README.md does not link to {link}")

    _check_issue_forms(errors)
    _check_dependabot(errors)

    if errors:
        print("Repository hygiene check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    summary = {
        "documentation_files": len(_markdown_files(repository_files)),
        "issue_forms": len(list((ROOT / ".github" / "ISSUE_TEMPLATE").glob("*.yml"))) - 1,
        "repository_files": len(repository_files),
    }
    print(f"Repository hygiene check passed: {json.dumps(summary, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
