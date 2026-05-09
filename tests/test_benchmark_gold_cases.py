"""Tests for benchmark gold-set regulatory retrieval cases."""

from pathlib import Path

from benchmark.gold_cases import gold_cases_as_test_cases, load_gold_cases
from benchmark.scoring import audit_grade_metrics, source_correctness_metrics


def test_load_gold_cases_from_yaml():
    cases = load_gold_cases()
    by_id = {case.id: case for case in cases}

    assert {"tfrs9_ilke5", "karsilik_madde9", "sicr_criteria"} <= set(by_id)
    assert by_id["tfrs9_ilke5"].expected_documents == ["943"]
    assert by_id["tfrs9_ilke5"].expected_sections == [{"type": "ilke", "ref": "5"}]
    assert by_id["karsilik_madde9"].expected_documents == ["mevzuat_22599"]
    assert by_id["sicr_criteria"].expected_terms == [
        "30 günden fazla gecikme",
        "kredi derecesi",
        "makroekonomik görünüm",
    ]


def test_load_gold_cases_accepts_explicit_path(tmp_path):
    path = tmp_path / "gold.yml"
    path.write_text(
        """
- id: custom
  query: "Özel soru"
  expected_documents: ["943"]
""".strip(),
        encoding="utf-8",
    )

    cases = load_gold_cases(path)

    assert len(cases) == 1
    assert cases[0].id == "custom"
    assert cases[0].query == "Özel soru"


def test_gold_cases_convert_to_phase2_test_cases():
    cases = gold_cases_as_test_cases()
    by_id = {case.id: case for case in cases}

    assert by_id["gold:tfrs9_ilke5"].category == "gold"
    assert by_id["gold:tfrs9_ilke5"].question == "943 numaralı rehberde İlke 5 model validasyonu ne diyor?"
    assert by_id["gold:tfrs9_ilke5"].expected_documents == ["943"]
    assert by_id["gold:tfrs9_ilke5"].expected_sections == [{"type": "ilke", "ref": "5"}]
    assert by_id["gold:tfrs9_ilke5"].expected_terms == []


def test_source_correctness_scores_expected_docs_sections_and_terms():
    case = gold_cases_as_test_cases()[0]
    trace = {
        "tool_calls": [
            {
                "name": "get_document_section",
                "args": {"document_id": "943", "section_type": "ilke", "section_ref": "5"},
            }
        ],
        "tool_results": ["Document 943\nİlke 5 - Model validasyonu\nModel validasyonu bağımsız yapılır."],
    }

    metrics = source_correctness_metrics(case, trace)

    assert metrics["expected_source_checks"] == 2
    assert metrics["matched_source_checks"] == 2
    assert metrics["retrieval_source_correctness_score"] == 1.0
    assert metrics["retrieval_source_correctness_success"] is True


def test_source_correctness_catches_missing_expected_terms():
    case = gold_cases_as_test_cases()[2]
    trace = {
        "tool_calls": [{"name": "search_document_sections", "args": {"query": case.question}}],
        "tool_results": ["Document 943\nKredi riskinde önemli artış için kredi derecesi dikkate alınır."],
    }

    metrics = source_correctness_metrics(case, trace)

    assert metrics["retrieval_source_correctness_score"] < 1.0
    assert metrics["retrieval_source_correctness_success"] is False
    assert "30 günden fazla gecikme" in metrics["missing_expected_terms"]


def test_audit_grade_metrics_include_gold_source_correctness():
    case = gold_cases_as_test_cases()[0]
    trace = {
        "tool_calls": [
            {
                "name": "get_document_section",
                "args": {"document_id": "943", "section_type": "ilke", "section_ref": "5"},
            }
        ],
        "tool_results": ["Document 943\nİlke 5 - Model validasyonu\nModel validasyonu bağımsız yapılır."],
        "final_answer": "943 numaralı rehberde İlke 5, model validasyonunun bağımsız yapılmasını düzenler.",
    }

    metrics = audit_grade_metrics(case, trace, code_score=0.8, model_score=0.8)

    assert metrics["retrieval_source_correctness_score"] == 1.0
    assert metrics["retrieval_source_correctness_success"] is True
    assert metrics["audit_grade_success"] is True


def test_gold_cases_file_is_repo_relative():
    assert Path("benchmark/gold_cases.yml").exists()
