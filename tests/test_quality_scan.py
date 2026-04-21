"""Tests for the quality_scan engine and its admin tool wrapper."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from quality_scan import (
    AnomalyCount,
    MethodBreakdown,
    QualityReport,
    format_report,
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
        docs = [
            ("doc_clean", "Normal document with çğıöşü letters " * 120, "markitdown"),
            ("doc_camelcase", "BÖLÜMBaşlangıç HükümleriAmaç " + "çğıöşü " * 120, "html_parser"),
            ("doc_replacement", "Some text � with replacement " + "çğıöşü " * 120, "markitdown"),
            ("doc_imgtag", "Some <img src='x.png'> leaked " + "çğıöşü " * 120, "markitdown"),
            ("doc_short", "tiny", "markitdown"),
            ("doc_dots", "TOC entry .......... page 3 " + "çğıöşü " * 120, "markitdown"),
            (
                "doc_formula_missing",
                "Article references formül ama formül yok. " + "çğıöşü " * 120,
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

    assert report.total_documents == 9
    method_names = {m.method for m in report.methods}
    assert {"markitdown", "html_parser", "chandra2", "glm_ocr"} <= method_names

    signals = {a.name: a for a in report.anomalies}

    assert signals["camelcase_concat"].docs_flagged >= 1
    assert "doc_camelcase" in signals["camelcase_concat"].sample_doc_ids

    assert signals["replacement_char"].docs_flagged >= 1
    assert "doc_replacement" in signals["replacement_char"].sample_doc_ids

    assert signals["leaked_img_tag"].docs_flagged >= 1
    assert "doc_imgtag" in signals["leaked_img_tag"].sample_doc_ids

    assert signals["short_content"].docs_flagged >= 1
    assert "doc_short" in signals["short_content"].sample_doc_ids

    assert signals["long_dot_run"].docs_flagged >= 1
    assert "doc_dots" in signals["long_dot_run"].sample_doc_ids

    assert signals["formula_ref_without_formula"].docs_flagged >= 1
    assert "doc_formula_missing" in signals["formula_ref_without_formula"].sample_doc_ids
    assert "doc_formula_ok" not in signals["formula_ref_without_formula"].sample_doc_ids

    assert signals["diacritic_outlier"].docs_flagged >= 1
    assert "doc_no_diacritics" in signals["diacritic_outlier"].sample_doc_ids

    assert report.orphan_chunks == 1


# -- admin tool wrapper -------------------------------------------------------


@pytest.mark.asyncio
async def test_document_quality_report_without_pool_returns_message():
    from deps import Dependencies
    from tools.admin import register

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
