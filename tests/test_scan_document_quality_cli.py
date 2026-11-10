"""Tests for scripts/scan_document_quality.py."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from scan_document_quality import _build_parser, load_quality_failures, main  # noqa: E402

from bddk_mcp.core.config import CORPUS_PATH  # noqa: E402
from bddk_mcp.store.corpus_db import CorpusDB  # noqa: E402
from bddk_mcp.store.doc_store import DocumentStore, StoredDocument  # noqa: E402


def test_load_quality_failures_reads_known_fail_list():
    failures = load_quality_failures(ROOT / "tests/fixtures/historical-quality-failures.yml")

    doc_ids = {item["document_id"] for item in failures}
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


def test_quality_cli_scans_md_dir_and_writes_outputs(tmp_path):
    docs = tmp_path / "docs"
    out = tmp_path / "reports"
    docs.mkdir()
    (docs / "mevzuat_21192.md").write_text("<img src='data:image/x-wmf;base64,AAA'>", encoding="utf-8")

    code = main(["--md-dir", str(docs), "--out-dir", str(out), "--allow-failures"])

    assert code == 0
    assert (out / "quality_report.md").exists()
    assert (out / "quality_findings.csv").exists()
    assert (out / "quality_findings.json").exists()
    assert (out / "suspicious_snippets.md").exists()

    report_json = json.loads((out / "quality_findings.json").read_text(encoding="utf-8"))
    assert report_json["document_findings"][0]["document_id"] == "mevzuat_21192"
    assert report_json["document_findings"][0]["label"] == "fail"
    assert "data_uri_image" in report_json["document_findings"][0]["flags"]


def test_quality_cli_scans_corpus_file_and_writes_outputs(tmp_path):
    corpus = tmp_path / "c.sqlite"
    out = tmp_path / "reports"
    db = CorpusDB.open(corpus)
    doc = StoredDocument(
        document_id="warn",
        title="Formül",
        markdown_content="Bu metinde aşağıdaki formül kullanılır.",
        extraction_method="html_parser",
    )
    asyncio.run(DocumentStore(db).store_document(doc))
    db.close()

    code = main(["--db", "--corpus", str(corpus), "--out-dir", str(out), "--allow-failures"])

    assert code == 0
    report_json = json.loads((out / "quality_findings.json").read_text(encoding="utf-8"))
    assert report_json["total_documents"] == 1
    assert report_json["methods"] == [{"method": "html_parser", "doc_count": 1, "avg_chars": len(doc.markdown_content)}]
    assert report_json["document_findings"][0]["document_id"] == "warn"
    assert "formula_ref_without_latex_or_image" in report_json["document_findings"][0]["flags"]
    assert "**BDDK Document Quality Report**" in (out / "quality_report.md").read_text(encoding="utf-8")


def test_quality_cli_exits_nonzero_for_failures_by_default(tmp_path):
    docs = tmp_path / "docs"
    out = tmp_path / "reports"
    docs.mkdir()
    (docs / "bad.md").write_text("<img src='data:image/x-wmf;base64,AAA'>", encoding="utf-8")

    code = main(["--md-dir", str(docs), "--out-dir", str(out)])

    assert code == 1


def test_quality_cli_fail_on_specific_signal(tmp_path):
    docs = tmp_path / "docs"
    out = tmp_path / "reports"
    docs.mkdir()
    (docs / "warn.md").write_text("Bu metinde aşağıdaki formül kullanılır.", encoding="utf-8")

    code = main(
        [
            "--md-dir",
            str(docs),
            "--out-dir",
            str(out),
            "--allow-failures",
            "--fail-on",
            "formula_ref_without_latex_or_image",
        ]
    )

    assert code == 1


def test_parser_defaults_to_the_configured_corpus_path():
    args = _build_parser().parse_args(["--db", "--out-dir", "reports"])
    assert args.corpus == CORPUS_PATH
