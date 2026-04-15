"""Model list, LLM server settings, and evaluation thresholds."""

import os

# -- LLM Server (Ollama or LM Studio — both expose OpenAI-compatible API) -----

LLM_BASE_URL = os.environ.get("BDDK_LLM_URL", "http://localhost:1234")
LLM_TIMEOUT = float(os.environ.get("BDDK_LLM_TIMEOUT", "120.0"))

MODELS = [
    {
        "name": "gemma-4-e2b-it",
        "model_id": "gemma-4-e2b-it",
        "active_params": "2B",
        "quantization": "FP16",
        "notes": "Baseline/small",
    },
    {
        "name": "gemma-4-e4b-it",
        "model_id": "gemma-4-e4b-it",
        "active_params": "4B",
        "quantization": "FP16",
        "notes": "Mid-small",
    },
    {
        "name": "gemma-4-26b-a4b-it",
        "model_id": "gemma-4-26b-a4b-it",
        "active_params": "4B (MoE)",
        "quantization": "Q4",
        "notes": "Best efficiency candidate",
    },
    {
        "name": "qwen3.5-9b",
        "model_id": "qwen3.5:9b",
        "active_params": "9B",
        "quantization": "Q8",
        "notes": "Mid-range",
    },
    {
        "name": "qwen3.5-27b",
        "model_id": "qwen3.5:27b",
        "active_params": "27B",
        "quantization": "Q4",
        "notes": "Large dense",
    },
    {
        "name": "qwen3.5-35b-a3b",
        "model_id": "qwen3.5:35b-a3b",
        "active_params": "3B (MoE)",
        "quantization": "Q4",
        "notes": "Large-efficient",
    },
    {
        "name": "qwen3.5-9b-uncensored",
        "model_id": "qwen3.5:9b-uncensored",
        "active_params": "9B",
        "quantization": "Q8",
        "notes": "Uncensored variant",
    },
]

TRIALS_PER_CASE = 3
MAX_TOOL_CALLS = 5

PHASE1_THRESHOLDS = {
    "tool_selection": 0.50,
    "nli_macro_f1": 0.40,
    "terminology": 0.60,
}

GRADER_MODEL = os.environ.get("BDDK_GRADER_MODEL", "claude-opus-4-6")
