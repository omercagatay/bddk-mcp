"""Lexical gate of the section-search audit set (tests/fixtures/section_search_audit.json).

Every answerable audit question runs through search_document_sections on the full seed with the seed catalog and no
vector store (so without the semantic slot). A question recorded as missed on this path is a strict xfail: a change
that finds it fails as loudly as one that loses a recorded find; update the record in the data file then.
The real-model outcomes are gated by scripts/eval_section_search.py and in CI by scripts/ci_image_smoke.py.
"""

import asyncio
import json
from pathlib import Path

import pytest

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.ingest.client import BddkApiClient
from bddk_mcp.store.corpus_db import CorpusDB
from bddk_mcp.store.doc_store import DocumentStore, StoredDocument
from scripts.eval_section_search import DATA, difference, gated_questions, is_found
from tests.test_tools_sections import _capture_tool

SEED = Path(__file__).resolve().parents[1] / "seed_data"
QUESTIONS = gated_questions()


async def _run_audit(path: Path) -> dict:
    db = CorpusDB.open(path)
    client = None
    try:
        store = DocumentStore(db)
        for doc in json.loads((SEED / "documents.json").read_text(encoding="utf-8")):
            await store.store_document(StoredDocument(**doc))
        await store.replace_catalog(json.loads((SEED / "decision_cache.json").read_text(encoding="utf-8")))
        client = BddkApiClient(store=store, allow_live_population=False)
        await client.load_cache_read_only()
        tool = _capture_tool(
            Dependencies(db=None, doc_store=store, client=client, http=None), "search_document_sections"
        )
        outcomes = {}
        for q in QUESTIONS:
            results = (await tool(q["query"])).structuredContent["results"]
            outcomes[q["id"]] = (is_found(results, q["gold"]), results)
        stored_spans = {}
        for q in QUESTIONS:
            for g in q["gold"]:
                key = (g["document_id"], g["section_type"], g["section_ref"])
                rows = await store.get_document_section(key[0], section_type=key[1], section_ref=key[2])
                stored_spans[key] = [(r.start_char, r.end_char) for r in rows]
        return {"outcomes": outcomes, "spans": stored_spans}
    finally:
        if client is not None:
            await client.close()
        db.close()


@pytest.fixture(scope="module")
def audit(tmp_path_factory):
    """One seed corpus and every audit query, once per module (about 10 s)."""
    return asyncio.run(_run_audit(tmp_path_factory.mktemp("audit") / "corpus.sqlite"))


def _case(q: dict):
    marks = []
    if q.get("platform_dependent"):
        marks.append(pytest.mark.xfail(strict=False, reason=f"platform dependent: {q['note']}"))
    elif q["lexical"] == "missed":
        marks.append(pytest.mark.xfail(strict=True, raises=AssertionError, reason=f"recorded missed: {q['query']}"))
    return pytest.param(q, id=q["id"], marks=marks)


@pytest.mark.parametrize("question", [_case(q) for q in QUESTIONS])
def test_section_search_finds_gold_provision(audit, question):
    found, results = audit["outcomes"][question["id"]]
    assert found, difference(question, found, "lexical", results)


def test_audit_data_is_well_formed(audit):
    data = json.loads(DATA.read_text(encoding="utf-8"))
    questions = data["questions"]
    assert {"authors", "criterion", "measurements"} <= data["header"].keys()
    assert len({q["id"] for q in questions}) == len(questions)
    assert [q["id"] for q in questions if q.get("debatable_gold")] == ["A03"]
    for q in questions:
        outcomes = (q["real_model"], q["lexical"])
        if q["category"] == "absent":
            assert q["gold"] == [] and outcomes == (None, None), q["id"]
            continue
        assert q["gold"] and set(outcomes) <= {"found", "missed"}, q["id"]
        assert not q.get("platform_dependent") or q.get("note"), q["id"]  # an ungated question says why
        for g in q["gold"]:
            key = (g["document_id"], g["section_type"], g["section_ref"])
            assert audit["spans"][key] == [(g["start_char"], g["end_char"])], (q["id"], key)
