"""Tests for the quality_scan engine and its admin tool wrapper."""

from __future__ import annotations

import pytest

from bddk_mcp.quality.quality_scan import (
    AnomalyCount,
    DocumentFinding,
    MethodBreakdown,
    QualityReport,
    format_report,
    format_report_csv,
    format_report_json,
    scan_quality,
)

# -- format_report: pure unit tests --------------------------------------------


def _make_report(**overrides) -> QualityReport:
    base = dict(
        total_documents=100,
        methods=[MethodBreakdown(method="markitdown", doc_count=90, avg_chars=17000)],
        anomalies=[
            AnomalyCount(
                name="camelcase_concat",
                docs_flagged=0,
                description="Adjacent lowercase+uppercase with no separator",
                sample_doc_ids=[],
            )
        ],
        orphan_chunks=0,
        docs_without_chunks=0,
        document_findings=[],
    )
    base.update(overrides)
    return QualityReport(**base)


def test_format_report_clean_corpus():
    out = format_report(_make_report())
    assert "Corpus: **100 documents**" in out
    assert "markitdown" in out
    assert "All anomaly signals are clean." in out


def test_format_report_highlights_firing_signals():
    report = _make_report(
        anomalies=[
            AnomalyCount(
                name="camelcase_concat",
                docs_flagged=30,
                description="Adjacent lowercase+uppercase with no separator",
                sample_doc_ids=["mevzuat_40520", "mevzuat_10522"],
            ),
            AnomalyCount(
                name="replacement_char",
                docs_flagged=0,
                description="U+FFFD",
                sample_doc_ids=[],
            ),
        ]
    )
    out = format_report(report)
    assert "1 anomaly signal(s) firing" in out
    assert "mevzuat_40520" in out
    assert "samples:" in out


def test_format_report_shows_chunk_integrity():
    report = _make_report(orphan_chunks=3, docs_without_chunks=7)
    out = format_report(report)
    assert "Orphan chunks (no parent doc): 3" in out
    assert "Docs >500 chars missing chunks: 7" in out


def test_format_report_shows_document_level_findings():
    report = _make_report(
        document_findings=[
            DocumentFinding(
                document_id="mevzuat_21192",
                label="fail",
                flags=["data_uri_image", "wmf_data_uri"],
                sample="raw image blob",
            ),
            DocumentFinding(
                document_id="943",
                label="warning",
                flags=["control_char"],
                sample="TFRS 9",
            ),
        ]
    )

    out = format_report(report)

    assert "**Document findings**" in out
    assert "mevzuat_21192" in out
    assert "fail" in out
    assert "data_uri_image" in out
    assert "943" in out


def test_format_report_json_and_csv_include_findings():
    report = _make_report(
        document_findings=[
            DocumentFinding(
                document_id="mevzuat_21192",
                label="fail",
                flags=["data_uri_image"],
                counts={"data_uri_image": 1},
                sample="blob",
            )
        ]
    )

    as_json = format_report_json(report)
    as_csv = format_report_csv(report)

    assert as_json["document_findings"][0]["document_id"] == "mevzuat_21192"
    assert as_json["document_findings"][0]["label"] == "fail"
    assert "document_id,label,flags,sample" in as_csv
    assert "mevzuat_21192,fail,data_uri_image,blob" in as_csv


# -- scan_quality: integration against a temporary corpus file ----------------


@pytest.fixture
async def seeded_quality_db(corpus_db):
    """Seed a throwaway set of docs with one orphan chunk and one parsed section."""
    docs = [
        ("doc_clean", "Normal document with çğıöşü letters " * 120, "markitdown"),
        ("doc_camelcase", "BÖLÜMBaşlangıç HükümleriAmaç " + "çğıöşü " * 120, "html_parser"),
        ("doc_replacement", "Some text \ufffd with replacement " + "çğıöşü " * 120, "markitdown"),
        ("doc_c1", "Fiyat \x96 aralığı " + "çğıöşü " * 120, "html_parser"),
        ("doc_imgtag", "Some <img src='x.png'> leaked " + "çğıöşü " * 120, "markitdown"),
        ("doc_data_uri", "Some <img src='data:image/x-wmf;base64,AAA'> leaked " + "çğıöşü " * 120, "markitdown"),
        ("doc_cid", ("cid:12 " * 25) + "çğıöşü " * 120, "markitdown"),
        ("doc_short", "tiny", "markitdown"),
        ("doc_dots", "TOC entry .......... page 3 " + "çğıöşü " * 120, "markitdown"),
        (
            "doc_formula_missing",
            "Article references aşağıdaki formül ama formül yok. " + "çğıöşü " * 120,
            "markitdown",
        ),
        (
            "doc_formula_ok",
            "Article with formula $$x=y$$ and formül ref. " + "çğıöşü " * 120,
            "chandra2",
        ),
        ("doc_no_diacritics", "ASCII only text here no turkish letters at all " * 40, "glm_ocr"),
    ]

    def _seed(conn):
        conn.executemany(
            "INSERT INTO documents (document_id, title, markdown_content, extraction_method, content_hash)"
            " VALUES (?, ?, ?, ?, 'x')",
            [(doc_id, doc_id, content, method) for doc_id, content, method in docs],
        )
        conn.executemany(
            "INSERT INTO chunks (doc_id, chunk_index, chunk_text, start_char, end_char, content_hash, embedding)"
            " VALUES (?, 0, 'chunk', 0, 5, 'x', X'00')",
            [("doc_clean",), ("missing_parent_doc",)],
        )
        conn.execute(
            "INSERT INTO document_sections (doc_id, section_type, section_ref, start_char, end_char, content,"
            " content_hash, source_content_hash) VALUES ('doc_clean', 'madde', '1', 0, 14, 'Hüküm içeriği.', 'h', 'x')"
        )

    await corpus_db.transaction(_seed)
    return corpus_db


@pytest.mark.asyncio
async def test_scan_quality_detects_all_seeded_anomalies(seeded_quality_db):
    report = await scan_quality(seeded_quality_db)

    assert report.total_documents == 12
    method_names = {m.method for m in report.methods}
    assert {"markitdown", "html_parser", "chandra2", "glm_ocr"} <= method_names

    signals = {a.name: a for a in report.anomalies}

    assert signals["camelcase_concat"].docs_flagged >= 1
    assert "doc_camelcase" in signals["camelcase_concat"].sample_doc_ids

    assert signals["replacement_char"].docs_flagged >= 1
    assert "doc_replacement" in signals["replacement_char"].sample_doc_ids

    assert signals["c1_control_char"].sample_doc_ids == ["doc_c1"]

    assert signals["leaked_img_tag"].docs_flagged >= 1
    assert "doc_imgtag" in signals["leaked_img_tag"].sample_doc_ids

    assert signals["data_uri_image"].docs_flagged >= 1
    assert "doc_data_uri" in signals["data_uri_image"].sample_doc_ids

    assert signals["cid_marker"].docs_flagged >= 1
    assert "doc_cid" in signals["cid_marker"].sample_doc_ids

    assert signals["short_content"].docs_flagged >= 1
    assert "doc_short" in signals["short_content"].sample_doc_ids

    assert signals["long_dot_run"].docs_flagged >= 1
    assert "doc_dots" in signals["long_dot_run"].sample_doc_ids

    assert signals["formula_ref_without_formula"].docs_flagged >= 1
    assert "doc_formula_missing" in signals["formula_ref_without_formula"].sample_doc_ids
    assert "doc_formula_ok" not in signals["formula_ref_without_formula"].sample_doc_ids

    assert signals["diacritic_outlier"].docs_flagged >= 1
    assert "doc_no_diacritics" in signals["diacritic_outlier"].sample_doc_ids

    assert signals["zero_sections_despite_content"].docs_flagged >= 1
    assert "doc_cid" in signals["zero_sections_despite_content"].sample_doc_ids
    assert "doc_clean" not in signals["zero_sections_despite_content"].sample_doc_ids

    assert report.orphan_chunks == 1

    findings = {f.document_id: f for f in report.document_findings}
    assert findings["doc_data_uri"].label == "fail"
    assert "data_uri_image" in findings["doc_data_uri"].flags
    assert findings["doc_cid"].label == "fail"
    assert "cid_marker" in findings["doc_cid"].flags
    assert findings["doc_formula_missing"].label == "warning"
    assert "formula_ref_without_latex_or_image" in findings["doc_formula_missing"].flags
    assert "doc_clean" not in findings
