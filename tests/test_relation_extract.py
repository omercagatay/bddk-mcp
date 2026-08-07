"""Deterministic extraction of cross-reference candidates from Turkish legal text."""

from __future__ import annotations

import pytest

from bddk_mcp.regulatory import relation_extract
from bddk_mcp.regulatory.relation_extract import (
    extract_candidate_relations,
    extract_relations_batch,
)

_AMEND = (
    "MADDE 1 – 5411 sayılı Bankacılık Kanununun 93 üncü maddesi aşağıdaki şekilde değiştirilmiştir."
)
_REPEAL = "MADDE 2 – Aynı Yönetmeliğin 12 nci maddesi yürürlükten kaldırılmıştır."
_CITE = "Bu Yönetmelik, 5411 sayılı Bankacılık Kanununun 93 üncü maddesine dayanılarak hazırlanmıştır."
_EXCEPTION = "Ancak 9 uncu madde hükümleri saklıdır."
_PLAIN = "Kredi riski, borçlunun yükümlülüklerini yerine getirememe olasılığıdır."


def _types(text):
    return [c.relation_type for c in extract_candidate_relations(text)]


def test_amendment_clause_detected():
    candidates = extract_candidate_relations(_AMEND)
    amends = [c for c in candidates if c.relation_type == "amends"]
    assert len(amends) == 1
    assert amends[0].target_article == "93"
    assert "5411" in amends[0].target_mention
    assert amends[0].confidence >= 0.8


def test_repeal_clause_detected():
    repeals = [c for c in extract_candidate_relations(_REPEAL) if c.relation_type == "repeals"]
    assert len(repeals) == 1
    assert repeals[0].target_article == "12"


def test_citation_with_dayanilarak_is_implements():
    kinds = _types(_CITE)
    assert "implements" in kinds
    assert "amends" not in kinds


def test_exception_clause_detected():
    exceptions = [c for c in extract_candidate_relations(_EXCEPTION) if c.relation_type == "exception_to"]
    assert len(exceptions) == 1
    assert exceptions[0].target_article == "9"


def test_plain_text_yields_nothing():
    assert extract_candidate_relations(_PLAIN) == []


def test_spans_point_at_matched_clause():
    candidate = extract_candidate_relations(_REPEAL)[0]
    start, end = candidate.span
    assert "yürürlükten kaldırılmıştır" in _REPEAL[start:end]


def test_suffixed_article_number():
    text = "Aynı Kanunun 26/A maddesi yürürlükten kaldırılmıştır."
    repeals = [c for c in extract_candidate_relations(text) if c.relation_type == "repeals"]
    assert repeals and repeals[0].target_article == "26/A"


@pytest.mark.asyncio
async def test_batch_skips_failing_documents(monkeypatch):
    """One bad document must not abort the batch (spec §6). Pure orchestration test — no DB."""
    calls = []

    async def _flaky(pool, *, doc_id, artifact_id, candidates, imported_by):
        if doc_id == "boom-doc":
            raise RuntimeError("synthetic per-document failure")
        calls.append(doc_id)
        return (1, 2)

    monkeypatch.setattr(relation_extract, "resolve_and_import", _flaky)
    summary = await extract_relations_batch(
        pool=None,
        items=[
            ("boom-doc", "art-x", _REPEAL),
            ("943", "art-1", _REPEAL),
        ],
        imported_by="test-suite",
    )
    assert summary == {"processed": 1, "failed": 1, "resolved": 1, "external": 2}
    assert calls == ["943"]
