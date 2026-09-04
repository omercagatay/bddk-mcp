"""Regression test: the admin console's templates and static assets must ship
in the built wheel.

`bddk_mcp/admin/app.py` resolves its Jinja and StaticFiles directories
relative to `Path(__file__).resolve().parent`. In an editable install that
path always exists because it *is* the source tree, so a test that only
checks `Path(bddk_mcp.admin.__file__).parent / "templates"` would pass even
if `[tool.setuptools.package-data]` stripped those files from a real,
non-editable install. This test instead builds an actual wheel and inspects
its contents, the way a non-editable `pip install bddk-mcp` would see them.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_ADMIN_FILES = {
    "bddk_mcp/admin/templates/base.html",
    "bddk_mcp/admin/templates/not_found.html",
    "bddk_mcp/admin/templates/documents/list.html",
    "bddk_mcp/admin/templates/documents/detail.html",
    "bddk_mcp/admin/templates/documents/search.html",
    "bddk_mcp/admin/templates/governance/status.html",
    "bddk_mcp/admin/templates/login.html",
}


@pytest.mark.skipif(shutil.which("uv") is None, reason="requires the uv CLI to build a real wheel")
def test_wheel_contains_admin_templates_and_static_files(tmp_path: Path) -> None:
    # A stale bddk_mcp.egg-info/SOURCES.txt at the repo root would seed
    # setuptools' manifest_files cache with paths from a *previous* build,
    # letting this test pass even when [tool.setuptools.package-data] is
    # broken. Force a from-scratch manifest so the wheel reflects only the
    # current pyproject.toml configuration.
    shutil.rmtree(REPO_ROOT / "bddk_mcp.egg-info", ignore_errors=True)

    out_dir = tmp_path / "dist"
    result = subprocess.run(
        ["uv", "build", "--wheel", "--force-pep517", "-o", str(out_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, f"uv build failed:\nstdout={result.stdout}\nstderr={result.stderr}"

    wheels = list(out_dir.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, found {wheels}"

    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())

    missing = sorted(EXPECTED_ADMIN_FILES - names)
    assert not missing, (
        f"built wheel is missing admin template/static files: {missing}. "
        "Check [tool.setuptools.package-data] for 'bddk_mcp.admin' in pyproject.toml."
    )


def test_python_source_only_check_is_not_sufficient_evidence() -> None:
    """Documents why the wheel-based test above exists: the source-tree
    check that this test performs passes unconditionally in this editable
    dev environment, regardless of package-data configuration, and must
    never be treated as proof the files ship in a real install."""
    import bddk_mcp.admin.app as admin_app

    package_root = Path(admin_app.__file__).resolve().parent
    assert (package_root / "templates" / "documents" / "list.html").exists()
