"""Tests for scripts/backfill_quality_failures.py."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from backfill_quality_failures import load_fail_documents, main  # noqa: E402


def test_load_fail_documents_from_quality_failures_yml():
    candidates = load_fail_documents(ROOT / "config" / "quality_failures.yml")
    doc_ids = {candidate.document_id for candidate in candidates}

    assert {
        "1043",
        "1045",
        "1305",
        "1313",
        "1314",
        "1334",
        "903",
        "905",
        "907",
        "mevzuat_16290",
        "mevzuat_21192",
    } <= doc_ids


def test_load_fail_documents_from_quality_findings_csv(tmp_path):
    csv_path = tmp_path / "quality_findings.csv"
    csv_path.write_text(
        "document_id,label,flags,sample\nmevzuat_21192,fail,data_uri_image,blob\n943,warning,control_char,text\n",
        encoding="utf-8",
    )

    candidates = load_fail_documents(csv_path)

    assert [candidate.document_id for candidate in candidates] == ["mevzuat_21192"]
    assert candidates[0].reason == "data_uri_image"


def test_backfill_quality_failures_dry_run_lists_known_failures(capsys):
    code = main(["--dry-run", "--config", str(ROOT / "config" / "quality_failures.yml")])
    out = capsys.readouterr().out

    assert code == 0
    assert "Quality failure backfill candidates: 11" in out
    assert "mevzuat_21192" in out
    assert "1314" in out
    assert "Dry run" in out


def test_backfill_quality_failures_doc_id_filters_one_candidate(capsys):
    code = main(["--dry-run", "--config", str(ROOT / "config" / "quality_failures.yml"), "--doc-id", "mevzuat_21192"])
    out = capsys.readouterr().out

    assert code == 0
    assert "Quality failure backfill candidates: 1" in out
    assert "mevzuat_21192" in out
    assert "1314" not in out
