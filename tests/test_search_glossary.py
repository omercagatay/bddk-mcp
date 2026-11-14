"""The abbreviation glossary of search_document_store: every entry is printed as a pair by the documents it cites."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from bddk_mcp.store.doc_store import normalize_search_text
from bddk_mcp.store.search_glossary import GLOSSARY, glossary_entry

_SEED = Path(__file__).resolve().parents[1] / "seed_data" / "documents.json"


@pytest.fixture(scope="module")
def seed_texts() -> dict[str, str]:
    documents = json.loads(_SEED.read_text(encoding="utf-8"))
    return {
        document["document_id"]: normalize_search_text(document["markdown_content"] or "") for document in documents
    }


def _pair_pattern(abbreviation: str, expansion: str) -> re.Pattern[str]:
    """'<expansion> (<ABBR>)' or '<ABBR> (<expansion>)' on the folded text; the last word may be inflected
    ("oranını", "süreci'ne") and a company suffix ("A.Ş.") may stand before the parenthesis."""
    words = [re.escape(word) for word in normalize_search_text(expansion).split()]
    words[-1] = words[-1][:-1] + r"\w*"
    spelled = r"\s+".join(words) + r"[’'\w]*(?:\s+a\.s\.)?"
    abbr = re.escape(normalize_search_text(abbreviation))
    return re.compile(rf"(?<!\w){spelled}\s*\(\s*{abbr}(?!\w)|(?<!\w){abbr}\s*\(\s*{spelled}")


def test_every_entry_cites_one_to_three_stored_documents_that_print_the_pair(seed_texts):
    assert GLOSSARY
    for entry in GLOSSARY:
        assert 1 <= len(entry.evidence_document_ids) <= 3, entry
        pattern = _pair_pattern(entry.abbreviation, entry.expansion)
        for document_id in entry.evidence_document_ids:
            assert document_id in seed_texts, (entry.abbreviation, document_id)
            assert pattern.search(seed_texts[document_id]), (entry.abbreviation, document_id)


def test_entries_are_unique_and_at_least_three_letters_long():
    abbreviations = [normalize_search_text(entry.abbreviation) for entry in GLOSSARY]
    assert len(abbreviations) == len(set(abbreviations))
    assert all(len(abbreviation) >= 3 for abbreviation in abbreviations)


def test_lookup_ignores_case_and_turkish_letters():
    assert glossary_entry("LKO").expansion == "likidite karşılama oranı"
    assert glossary_entry("lko") is glossary_entry("LKO")
    assert glossary_entry("ISEDES") is glossary_entry("İSEDES")
    assert glossary_entry("XYZQ") is None
    assert glossary_entry("SYR") is None
