"""Tests for benchmark configuration."""

from benchmark.config import LLM_BASE_URL, MODELS, PHASE1_THRESHOLDS


def test_models_not_empty():
    assert len(MODELS) >= 5


def test_each_model_has_required_fields():
    required = {"name", "model_id", "active_params", "quantization"}
    for model in MODELS:
        missing = required - set(model.keys())
        assert not missing, f"Model {model.get('name', '?')} missing: {missing}"


def test_thresholds_are_fractions():
    assert 0 < PHASE1_THRESHOLDS["tool_selection"] <= 1.0
    assert 0 < PHASE1_THRESHOLDS["nli_macro_f1"] <= 1.0
    assert 0 < PHASE1_THRESHOLDS["terminology"] <= 1.0


def test_ollama_base_url():
    assert LLM_BASE_URL.startswith("http")
