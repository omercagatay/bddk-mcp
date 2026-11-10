"""Tests for scripts/backfill_quality_failures.py."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import backfill_quality_failures  # noqa: E402
from backfill_quality_failures import load_fail_documents, main  # noqa: E402

from bddk_mcp.core.config import CORPUS_PATH  # noqa: E402
from bddk_mcp.ingest.doc_sync import SyncResult  # noqa: E402
from bddk_mcp.store.corpus_db import CorpusDB  # noqa: E402
from bddk_mcp.store.doc_store import DocumentStore, StoredDocument  # noqa: E402
from bddk_mcp.store.vector_store import VectorStore  # noqa: E402

HISTORICAL_REGISTRY = ROOT / "tests/fixtures/historical-quality-failures.yml"


def test_load_fail_documents_from_quality_failures_yml():
    candidates = load_fail_documents(HISTORICAL_REGISTRY)
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
    code = main(["--dry-run", "--config", str(HISTORICAL_REGISTRY)])
    out = capsys.readouterr().out

    assert code == 0
    assert "Quality failure backfill candidates: 11" in out
    assert "mevzuat_21192" in out
    assert "1314" in out
    assert "Dry run" in out


def test_backfill_quality_failures_doc_id_filters_one_candidate(capsys):
    code = main(
        [
            "--dry-run",
            "--config",
            str(HISTORICAL_REGISTRY),
            "--doc-id",
            "mevzuat_21192",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "Quality failure backfill candidates: 1" in out
    assert "mevzuat_21192" in out
    assert "1314" not in out


def test_backfill_quality_failures_execute_reextracts_stored_documents(tmp_path, capsys, monkeypatch):
    """--execute re-syncs every flagged document that exists in the corpus file (with force)."""
    corpus = tmp_path / "c.sqlite"
    db = CorpusDB.open(corpus)
    doc = StoredDocument(
        document_id="mevzuat_21192",
        title="Eski",
        category="Yönetmelik",
        source_url="https://www.mevzuat.gov.tr/mevzuat?MevzuatNo=21192",
        markdown_content="eski metin",
        extraction_method="markitdown",
    )
    asyncio.run(DocumentStore(db).store_document(doc))
    db.close()
    csv_path = tmp_path / "quality_findings.csv"
    csv_path.write_text(
        "document_id,label,flags,sample\nmevzuat_21192,fail,data_uri_image,blob\n1314,fail,c1_control_char,x\n",
        encoding="utf-8",
    )

    calls: list[dict] = []
    wiring: dict = {}

    class FakeSyncer:
        def __init__(self, store, **kwargs):
            wiring.update(store=store, **kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def sync_document(self, **kwargs):
            calls.append(kwargs)
            return SyncResult(document_id=kwargs["doc_id"], success=True, method="html_parser", size_bytes=42)

    monkeypatch.setattr(backfill_quality_failures, "DocumentSyncer", FakeSyncer)

    code = main(["--execute", "--config", str(csv_path), "--corpus", str(corpus)])
    out = capsys.readouterr().out

    assert code == 0
    assert "[1/2] mevzuat_21192: re-extracting" in out
    assert "ok: method=html_parser size=42" in out
    assert "[2/2] 1314: re-extracting" in out
    assert "skipped: not found in document store" in out
    # Only the stored document reaches the syncer, with its catalog metadata and force=True
    assert [call["doc_id"] for call in calls] == ["mevzuat_21192"]
    assert calls[0]["force"] is True
    assert calls[0]["title"] == "Eski"
    assert calls[0]["source_url"] == doc.source_url
    # The syncer is wired to the corpus file's stores so re-extraction republishes chunks too
    assert isinstance(wiring["store"], DocumentStore)
    assert isinstance(wiring["vector_store"], VectorStore)


def test_parser_defaults_to_the_configured_corpus_path():
    assert backfill_quality_failures._build_parser().parse_args([]).corpus == CORPUS_PATH
