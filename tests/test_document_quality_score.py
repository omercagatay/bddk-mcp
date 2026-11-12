import asyncio
import json

from bddk_mcp.core.config import CORPUS_PATH
from bddk_mcp.quality.quality_scan import AnomalyCount, DocumentFinding, MethodBreakdown, QualityReport
from bddk_mcp.store.corpus_db import CorpusDB
from bddk_mcp.store.doc_store import DocumentStore, StoredDocument
from scripts.document_quality_score import _build_parser, main, quality_score


def _report(**overrides):
    base = {
        "total_documents": 10,
        "methods": [MethodBreakdown(method="markitdown", doc_count=10, avg_chars=1000)],
        "anomalies": [],
        "orphan_chunks": 0,
        "docs_without_chunks": 0,
        "document_findings": [],
    }
    base.update(overrides)
    return QualityReport(**base)


def test_quality_score_is_100_for_clean_corpus():
    result = quality_score(_report())

    assert result["score"] == 100
    assert result["clean_documents"] == 10
    assert result["warning_documents"] == 0
    assert result["fail_documents"] == 0


def test_quality_score_penalizes_failures_warnings_anomalies_and_integrity():
    result = quality_score(
        _report(
            document_findings=[
                DocumentFinding(document_id="fail", label="fail", flags=["data_uri_image"]),
                DocumentFinding(document_id="warn", label="warning", flags=["formula_ref_without_latex_or_image"]),
            ],
            anomalies=[
                AnomalyCount(name="data_uri_image", docs_flagged=1, description="blob"),
                AnomalyCount(name="formula_ref_without_formula", docs_flagged=2, description="missing formula"),
            ],
            orphan_chunks=1,
            docs_without_chunks=1,
        )
    )

    assert result["score"] == 87.75
    assert result["clean_documents"] == 8
    assert result["warning_documents"] == 1
    assert result["fail_documents"] == 1
    assert result["anomaly_hits"] == 3


def test_main_scores_the_corpus_file(tmp_path, capsys):
    corpus = tmp_path / "c.sqlite"
    db = CorpusDB.open(corpus)
    doc = StoredDocument(
        document_id="clean",
        title="Temiz",
        markdown_content="Temiz belge çğıöşü " * 40,  # >500 chars and never embedded
        extraction_method="html_parser",
    )
    asyncio.run(DocumentStore(db).store_document(doc))
    db.close()

    assert main(["--db", "--corpus", str(corpus), "--json"]) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["total_documents"] == 1
    assert result["clean_documents"] == 1
    assert result["fail_documents"] == 0
    assert result["docs_without_chunks"] == 1
    assert 0 < result["score"] < 100


def test_parser_defaults_to_the_configured_corpus_path():
    assert _build_parser().parse_args(["--db"]).corpus == CORPUS_PATH
