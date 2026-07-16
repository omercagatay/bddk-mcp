"""Regression checks for the single, explicit database-migration boundary."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_APPLICATION_DATABASE_PATHS = (
    "bddk_mcp/ingest/client.py",
    "bddk_mcp/store/doc_store.py",
    "bddk_mcp/store/vector_store.py",
    "scripts/retrieval_score.py",
    "scripts/reindex_document_sections.py",
    "scripts/patch_doc.py",
)

_SETUP_PATHS_WITHOUT_INITIALIZER_CALLS = (
    "tests/conftest.py",
    "tests/test_improvements.py",
    "tests/test_integration.py",
    "tests/test_f1_score.py",
)

_DDL = re.compile(
    r"\b(?:create|alter|drop)\s+(?:table|index|schema|extension|function|trigger)\b",
    re.IGNORECASE,
)


@pytest.mark.parametrize("relative_path", _APPLICATION_DATABASE_PATHS)
def test_runtime_and_operational_paths_contain_no_schema_ddl(relative_path: str):
    source = (ROOT / relative_path).read_text(encoding="utf-8")

    assert _DDL.search(source) is None, f"schema DDL must live in bddk_mcp/migrations, not {relative_path}"


@pytest.mark.parametrize("relative_path", _SETUP_PATHS_WITHOUT_INITIALIZER_CALLS)
def test_test_setup_does_not_depend_on_legacy_initializers(relative_path: str):
    source = (ROOT / relative_path).read_text(encoding="utf-8")

    assert ".initialize()" not in source


def test_write_scripts_select_explicit_non_public_database_identities():
    retrieval_score = (ROOT / "scripts/retrieval_score.py").read_text(encoding="utf-8")
    reindex = (ROOT / "scripts/reindex_document_sections.py").read_text(encoding="utf-8")
    patch_doc = (ROOT / "scripts/patch_doc.py").read_text(encoding="utf-8")

    assert "BDDK_TEST_DATABASE_URL" in retrieval_score
    assert 'require_database_url("ingestion")' in reindex
    assert 'require_database_url("ingestion")' in patch_doc
    assert "assert_database_ready" in retrieval_score
    assert "assert_database_ready" in reindex
    assert "assert_database_ready" in patch_doc
