"""Tests for the quality_scan engine and its admin tool wrapper."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.server.fastmcp.exceptions import ToolError

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


# -- scan_quality: integration against a single-connection pool --------------


@pytest.fixture
async def seeded_quality_pool(pg_pool):
    """Seed a throwaway set of docs into an isolated schema.

    Uses a single pinned connection so schema / search_path are consistent
    across every query scan_quality issues (a real pool could reuse different
    connections that don't see the temp schema).
    """
    from tests.conftest import SingleConnPool

    schema = "quality_scan_test"
    conn = await pg_pool.acquire()
    try:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.execute(f"CREATE SCHEMA {schema}")
        await conn.execute(f"SET search_path TO {schema}, public")
        await conn.execute(
            """
            CREATE TABLE documents (
                document_id text PRIMARY KEY,
                title text NOT NULL DEFAULT '',
                markdown_content text NOT NULL DEFAULT '',
                extraction_method text DEFAULT 'markitdown'
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE document_chunks (
                id serial PRIMARY KEY,
                doc_id text NOT NULL
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE document_sections (
                id serial PRIMARY KEY,
                doc_id text NOT NULL,
                section_type text NOT NULL,
                section_ref text NOT NULL,
                content text NOT NULL
            )
            """
        )
        docs = [
            ("doc_clean", "Normal document with çğıöşü letters " * 120, "markitdown"),
            ("doc_camelcase", "BÖLÜMBaşlangıç HükümleriAmaç " + "çğıöşü " * 120, "html_parser"),
            ("doc_replacement", "Some text � with replacement " + "çğıöşü " * 120, "markitdown"),
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
        for doc_id, content, method in docs:
            await conn.execute(
                "INSERT INTO documents (document_id, title, markdown_content, extraction_method)"
                " VALUES ($1, $2, $3, $4)",
                doc_id,
                doc_id,
                content,
                method,
            )
        await conn.execute("INSERT INTO document_chunks (doc_id) VALUES ('doc_clean'), ('missing_parent_doc')")
        await conn.execute(
            "INSERT INTO document_sections (doc_id, section_type, section_ref, content)"
            " VALUES ('doc_clean', 'madde', '1', 'Hüküm içeriği.')"
        )

        yield SingleConnPool(conn)
    finally:
        try:
            await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            await conn.execute("SET search_path TO public")
        finally:
            await pg_pool.release(conn)


@pytest.mark.asyncio
async def test_scan_quality_detects_all_seeded_anomalies(seeded_quality_pool):
    report = await scan_quality(seeded_quality_pool)

    assert report.total_documents == 11
    method_names = {m.method for m in report.methods}
    assert {"markitdown", "html_parser", "chandra2", "glm_ocr"} <= method_names

    signals = {a.name: a for a in report.anomalies}

    assert signals["camelcase_concat"].docs_flagged >= 1
    assert "doc_camelcase" in signals["camelcase_concat"].sample_doc_ids

    assert signals["replacement_char"].docs_flagged >= 1
    assert "doc_replacement" in signals["replacement_char"].sample_doc_ids

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


# -- admin tool wrapper -------------------------------------------------------


@pytest.mark.asyncio
async def test_document_quality_report_without_pool_returns_message():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.admin import register

    mcp = MagicMock()
    captured: dict[str, object] = {}

    def capture_tool():
        def inner(fn):
            captured[fn.__name__] = fn
            return fn

        return inner

    mcp.tool = capture_tool
    deps = Dependencies(
        pool=None,
        doc_store=None,
        client=None,
        http=None,
        server_start_time=time.time(),
    )
    register(mcp, deps)

    fn = captured["document_quality_report"]
    out = await fn()
    assert "DB pool not initialized" in out


@pytest.mark.asyncio
async def test_document_quality_report_does_not_expose_database_error_text():
    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.tools.admin import register

    mcp = MagicMock()
    captured: dict[str, object] = {}

    def capture_tool():
        def inner(fn):
            captured[fn.__name__] = fn
            return fn

        return inner

    mcp.tool = capture_tool
    deps = Dependencies(pool=MagicMock(), doc_store=None, client=None, http=None)
    sentinel = "postgresql://user:secret@internal.example/regulations"
    with patch("bddk_mcp.tools.admin.scan_quality", new=AsyncMock(side_effect=RuntimeError(sentinel))):
        register(mcp, deps)
        with pytest.raises(ToolError) as error:
            await captured["document_quality_report"]()

    assert "[ERROR:QUALITY_SCAN_FAILED]" in str(error.value)
    assert sentinel not in str(error.value)
