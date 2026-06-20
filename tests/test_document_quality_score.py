import json

from bddk_mcp.quality.quality_scan import AnomalyCount, DocumentFinding, MethodBreakdown, QualityReport
from scripts.document_quality_score import quality_score, scan_seed_dir


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


def test_scan_seed_dir_scores_documents_and_chunk_integrity(tmp_path):
    seed_dir = tmp_path / "seed_data"
    seed_dir.mkdir()
    (seed_dir / "documents.json").write_text(
        json.dumps(
            [
                {
                    "document_id": "clean",
                    "markdown_content": "Temiz belge çğıöşü " * 40,
                    "extraction_method": "html_parser",
                },
                {
                    "document_id": "bad",
                    "markdown_content": "<img src='data:image/x-wmf;base64,AAA'>",
                    "extraction_method": "markitdown",
                },
                {
                    "document_id": "missing_chunks",
                    "markdown_content": "Uzun belge " * 80,
                    "extraction_method": "markitdown",
                },
            ]
        ),
        encoding="utf-8",
    )
    (seed_dir / "chunks.json").write_text(
        json.dumps(
            [
                {"doc_id": "clean"},
                {"doc_id": "bad"},
                {"doc_id": "orphan"},
            ]
        ),
        encoding="utf-8",
    )

    report = scan_seed_dir(seed_dir)

    assert report.total_documents == 3
    assert {method.method for method in report.methods} == {"html_parser", "markitdown"}
    assert report.orphan_chunks == 1
    assert report.docs_without_chunks == 1
    assert report.document_findings[0].document_id == "bad"
