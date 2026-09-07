"""Full-seed regressions for the reported BKZ/backtesting retrieval failures.

These are retrieval checks, not certification of source currency or legal applicability.
Run with BDDK_REQUIRE_TEST_DATABASE=1 uv run pytest tests/test_credit_loss_retrieval.py -s.
"""

import hashlib
import json
from pathlib import Path

import pytest

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.quality.markdown_quality import sanitize_markdown_for_context
from bddk_mcp.store.doc_store import StoredDocument
from bddk_mcp.store.section_index import extract_document_sections
from tests.test_tools_sections import _capture_tool

SEED = Path(__file__).resolve().parents[1] / "seed_data" / "documents.json"


def test_guideline_paragraphs_and_principle_boundaries_on_real_seed():
    docs = {d["document_id"]: d for d in json.loads(SEED.read_text())}
    for doc_id, principle, last_paragraph, next_heading, target in (
        ("943", "7", "52.", "ÜÇÜNCÜ KISIM", "43"),
        ("935", "6", "25.", "2.2.", "29"),
    ):
        text = docs[doc_id]["markdown_content"]
        sections = extract_document_sections(doc_id, text)
        exact = next(s for s in sections if s.section_type == "ilke" and s.section_ref == principle)
        assert last_paragraph in exact.content
        assert next_heading not in exact.content
        paragraph = next(s for s in sections if s.section_type == "paragraf" and s.section_ref == target)
        assert paragraph.content.startswith(f"{target}.")
        assert text[paragraph.start_char : paragraph.end_char].strip() == paragraph.content


@pytest.mark.asyncio
async def test_credit_loss_queries_against_full_seed(doc_store):
    for doc in json.loads(SEED.read_text()):
        await doc_store.store_document(StoredDocument(**doc))
    tool = _capture_tool(
        Dependencies(pool=None, doc_store=doc_store, client=None, http=None), "search_document_sections"
    )
    cases = (
        (
            "gerçekleşen kayıp oranı geriye dönük test LGD tarihsel veri stres testi",
            None,
            {
                ("1040", "paragraf", "80"),
                ("935", "paragraf", "29"),
                ("935", "paragraf", "30"),
                ("946", "paragraf", "120"),
            },
            1,
        ),
        (
            "beklenen kredi zararı geriye dönük test tarihsel kayıp deneyimi ileriye yönelik bilgi stres senaryosu",
            "1040",
            {("1040", "paragraf", "80"), ("1040", "paragraf", "134")},
            1,
        ),
        (
            "tahmin edilen zarar karşılıkları gerçekleşen zararları geriye dönük test",
            "1040",
            {("1040", "paragraf", "134")},
            1,
        ),
        (
            "BKZ model validasyonu",
            "943",
            {("943", "ilke", "5"), ("943", "paragraf", "43")},
            1,
        ),
        ("LGD geriye dönük test", "935", {("935", "paragraf", "30")}, 1),
        ("THK geriye-dönük test", "935", {("935", "paragraf", "30")}, 1),
        (
            "tarihsel zarar deneyimi geleceğe yönelik makroekonomik",
            "943",
            {("943", "paragraf", "44"), ("943", "paragraf", "45"), ("943", "ilke", "6")},
            1,
        ),
        (
            "model çıktıları performans eşikleri yeniden düzenlenmesi",
            "943",
            {("943", "paragraf", "43"), ("943", "ilke", "5")},
            1,
        ),
        ("teminat değerlemelerinin geriye dönük testi", "1040", {("1040", "paragraf", "139")}, 1),
        ("gayrimenkul nakit akışı olumsuz ekonomik koşullar", "1040", {("1040", "paragraf", "157")}, 1),
    )
    outcomes = []
    for query, doc_id, relevant, cutoff in cases:
        result = await tool(query, document_id=doc_id, limit=10)
        keys = [(s["document_id"], s["section_type"], s["section_ref"]) for s in result.structuredContent["results"]]
        passed = bool(relevant.intersection(keys[:cutoff]))
        outcomes.append(passed)
        print(json.dumps({"query": query, "top": keys, "hit_at": cutoff, "passed": passed}, ensure_ascii=False))
    assert all(outcomes)

    exact = _capture_tool(Dependencies(pool=None, doc_store=doc_store, client=None, http=None), "get_document_section")
    for doc_id, ref in (
        ("943", "43"),
        ("1040", "80"),
        ("1040", "134"),
        ("935", "29"),
        ("946", "120"),
        ("1040", "139"),
        ("1040", "157"),
    ):
        result = await exact(doc_id, section_type="paragraf", section_ref=ref)
        assert len(result.structuredContent["results"]) == 1
        item = result.structuredContent["results"][0]
        document = await doc_store.get_document(doc_id)
        source = document.markdown_content[item["start_char"] : item["end_char"]].strip()
        assert item["section_type"] == "paragraf" and item["section_ref"] == ref
        assert item["content_hash"] == hashlib.sha256(source.encode()).hexdigest()
        assert item["content"] == sanitize_markdown_for_context(source)
        assert not item["content_truncated"]
        evidence = result.structuredContent["evidence"][0]
        assert evidence["untrusted_source"] is True
        assert evidence["quality"]["label"] in {"clean", "warning", "fail"}
        assert evidence["title"] and evidence["title"] in result.text
        assert evidence["source_url"].endswith(f"/DokumanGetir/{doc_id}")
        assert evidence["source_url"] in result.text
        assert evidence["extraction_method"]
        assert "citation" not in evidence  # Seed text alone is not independently validated authority.
        assert "citation_v1_unavailable_no_validated_mapping" in result.text
        assert "do not establish current legal applicability" in result.text

    result = await tool("943 paragraf 43", limit=1)
    assert result.structuredContent["exact_reference_detected"]
    assert result.structuredContent["results"][0]["section_ref"] == "43"
