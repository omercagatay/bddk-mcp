"""Tests for benchmark configuration."""

from benchmark.config import MODELS, OLLAMA_BASE_URL, PHASE1_THRESHOLDS


def test_models_not_empty():
    assert len(MODELS) >= 5


def test_each_model_has_required_fields():
    required = {"name", "ollama_tag", "active_params", "quantization"}
    for model in MODELS:
        missing = required - set(model.keys())
        assert not missing, f"Model {model.get('name', '?')} missing: {missing}"


def test_thresholds_are_fractions():
    assert 0 < PHASE1_THRESHOLDS["tool_selection"] <= 1.0
    assert 0 < PHASE1_THRESHOLDS["nli_macro_f1"] <= 1.0
    assert 0 < PHASE1_THRESHOLDS["terminology"] <= 1.0


def test_ollama_base_url():
    assert OLLAMA_BASE_URL.startswith("http")
