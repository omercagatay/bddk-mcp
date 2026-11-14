"""Tests for the pinned-version checks of scripts/check_repository_hygiene.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "check_repository_hygiene", ROOT / "scripts" / "check_repository_hygiene.py"
)
hygiene = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hygiene)

REVISION = "d13f1b27baf31030b7fd040960d60d909913633f"
OTHER_REVISION = "0" * 40


def _tree(tmp_path: Path, *, uv_docker="0.12.3", uv_ci="0.12.3", rev_docker=REVISION, rev_config=REVISION) -> Path:
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / "bddk_mcp" / "core").mkdir(parents=True)
    (tmp_path / "Dockerfile").write_text(
        f"COPY --from=ghcr.io/astral-sh/uv:{uv_docker}@sha256:{'a' * 64} /uv /usr/local/bin/uv\n"
        f"RUN python -c \"M('intfloat/multilingual-e5-base', revision='{rev_docker}')\"\n",
        encoding="utf-8",
    )
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text(f'env:\n  UV_VERSION: "{uv_ci}"\n', encoding="utf-8")
    (tmp_path / "bddk_mcp" / "core" / "config.py").write_text(
        f'_DEFAULT_EMBEDDING_MODEL_REVISION = "{rev_config}"\n', encoding="utf-8"
    )
    return tmp_path


def _errors(root: Path) -> list[str]:
    errors: list[str] = []
    hygiene._check_pinned_versions(errors, root)
    return errors


def test_matching_pins_pass(tmp_path):
    assert _errors(_tree(tmp_path)) == []


def test_uv_version_mismatch_fails(tmp_path):
    errors = _errors(_tree(tmp_path, uv_ci="0.11.14"))
    assert len(errors) == 1 and "uv version differs" in errors[0]


def test_model_revision_mismatch_fails(tmp_path):
    errors = _errors(_tree(tmp_path, rev_config=OTHER_REVISION))
    assert len(errors) == 1 and "embedding model revision differs" in errors[0]


@pytest.mark.parametrize("kwargs", [{"uv_docker": "x"}, {"rev_docker": "main"}])
def test_missing_pin_in_dockerfile_fails(tmp_path, kwargs):
    assert _errors(_tree(tmp_path, **kwargs))


def test_repository_pins_agree():
    assert _errors(ROOT) == []
