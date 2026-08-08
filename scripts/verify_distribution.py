"""Fail CI when Python distributions contain or omit security-relevant files."""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path


def _wheel_members(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return set(archive.namelist())


def _sdist_members(path: Path) -> set[str]:
    with tarfile.open(path, "r:gz") as archive:
        return set(archive.getnames())


def _assert_safe(members: set[str], *, artifact: Path) -> None:
    normalized = {member.removeprefix("./") for member in members}
    forbidden_parts = {".env", "seed_data", "tests", "__pycache__"}
    leaked = sorted(
        member
        for member in normalized
        if forbidden_parts.intersection(Path(member).parts) or member.endswith((".pyc", ".pyo"))
    )
    if leaked:
        raise SystemExit(f"{artifact.name} contains forbidden paths: {leaked[:10]}")


def main(directory: Path) -> None:
    wheels = sorted(directory.glob("*.whl"))
    sdists = sorted(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit(f"expected one wheel and one sdist in {directory}")

    wheel_members = _wheel_members(wheels[0])
    _assert_safe(wheel_members, artifact=wheels[0])
    expected = {
        "bddk_mcp/__init__.py",
        "bddk_mcp/server.py",
        "bddk_mcp/quality/quality_failures.yml",
    }
    missing = sorted(expected - wheel_members)
    if missing:
        raise SystemExit(f"wheel is missing packaged runtime files: {missing}")

    _assert_safe(_sdist_members(sdists[0]), artifact=sdists[0])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_distribution.py DIST_DIRECTORY")
    main(Path(sys.argv[1]))
