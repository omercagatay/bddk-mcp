"""--fail-on validation of scripts/scan_document_quality.py per scan mode."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from scan_document_quality import main  # noqa: E402

from bddk_mcp.store.corpus_db import CorpusDB  # noqa: E402
from bddk_mcp.store.doc_store import DocumentStore, StoredDocument  # noqa: E402


@pytest.mark.parametrize("name", ["short_content", "formula_ref_without_formula", "leaked_img_tag"])
def test_md_dir_fail_on_rejects_names_this_mode_can_never_produce(tmp_path, capsys, name):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Kısa.", encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        main(["--md-dir", str(docs), "--out-dir", str(tmp_path / "r"), "--fail-on", f"cid_marker,{name}"])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert name in err and "--md-dir" in err
    assert "formula_ref_without_latex_or_image" in err  # the valid names of this mode are listed
    assert not (tmp_path / "r").exists()


def test_md_dir_fail_on_still_accepts_assessment_flag_names(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("Temiz metin.", encoding="utf-8")

    assert main(["--md-dir", str(docs), "--out-dir", str(tmp_path / "r"), "--fail-on", "cid_marker,mojibake"]) == 0


def test_db_mode_still_accepts_scan_anomaly_names(tmp_path):
    corpus = tmp_path / "c.sqlite"
    db = CorpusDB.open(corpus)
    markdown = "Madde 1 - " + "Bankalar risk yönetimi kurar. " * 30
    doc = StoredDocument(document_id="d1", title="T", markdown_content=markdown, extraction_method="html_parser")
    asyncio.run(DocumentStore(db).store_document(doc))
    db.close()
    out = tmp_path / "reports"

    base = ["--db", "--corpus", str(corpus), "--out-dir", str(out)]
    assert main([*base, "--fail-on", "short_content,formula_ref_without_formula"]) == 0
