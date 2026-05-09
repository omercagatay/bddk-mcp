"""Gold-set regulatory QA cases for retrieval-source grading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from benchmark.test_cases import TestCase

GOLD_CASES_PATH = Path(__file__).with_name("gold_cases.yml")


@dataclass(frozen=True)
class GoldCase:
    """A benchmark case with explicit expected source evidence."""

    id: str
    query: str
    expected_documents: list[str] = field(default_factory=list)
    expected_sections: list[dict] = field(default_factory=list)
    expected_terms: list[str] = field(default_factory=list)


def load_gold_cases(path: str | Path = GOLD_CASES_PATH) -> list[GoldCase]:
    """Load gold-set retrieval cases from YAML."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    if isinstance(raw, dict):
        raw = raw.get("gold_cases", [])

    cases: list[GoldCase] = []
    for item in raw:
        cases.append(
            GoldCase(
                id=str(item["id"]),
                query=str(item["query"]),
                expected_documents=[str(doc_id) for doc_id in item.get("expected_documents", [])],
                expected_sections=[
                    {"type": str(section.get("type", "")), "ref": str(section.get("ref", ""))}
                    for section in item.get("expected_sections", [])
                ],
                expected_terms=[str(term) for term in item.get("expected_terms", [])],
            )
        )
    return cases


def gold_cases_as_test_cases(path: str | Path = GOLD_CASES_PATH) -> list[TestCase]:
    """Convert gold-set cases into Phase 2 TestCase instances."""
    return [_gold_case_to_test_case(case) for case in load_gold_cases(path)]


def _gold_case_to_test_case(case: GoldCase) -> TestCase:
    return TestCase(
        id=f"gold:{case.id}",
        question=case.query,
        expected_tool="",
        category="gold",
        expected_documents=case.expected_documents,
        expected_sections=case.expected_sections,
        expected_terms=case.expected_terms,
    )
