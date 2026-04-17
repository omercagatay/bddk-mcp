"""Tests for benchmark graders."""

from benchmark.graders import code_grader, extract_facts


class TestExtractFacts:
    def test_extracts_numbers(self):
        text = "Toplam krediler: 15,234,567.89 TL, haftalık değişim: %2.34"
        facts = extract_facts(text)
        assert "15,234,567.89" in facts or "15234567.89" in facts
        assert "2.34" in facts

    def test_extracts_dates(self):
        text = "Tarih: 15.03.2025, dönem: 2024/12"
        facts = extract_facts(text)
        assert "15.03.2025" in facts

    def test_empty_text(self):
        assert extract_facts("") == []


class TestCodeGrader:
    def test_fully_grounded(self):
        tool_result = "Toplam krediler: 15,234,567 TL. Haftalık değişim: %2.34"
        answer = "Toplam krediler 15,234,567 TL tutarında olup haftalık %2.34 artış göstermiştir."
        score = code_grader(tool_result, answer)
        assert score >= 0.8

    def test_hallucinated(self):
        tool_result = "Toplam krediler: 15,234,567 TL"
        answer = "Toplam krediler 20,000,000 TL tutarındadır ve %5 artış göstermiştir."
        score = code_grader(tool_result, answer)
        assert score < 0.5

    def test_empty_answer(self):
        assert code_grader("some data", "") == 0.0
