"""Model list, LLM server settings, and evaluation thresholds."""

import os

# -- LLM Server (Ollama default; LM Studio / other OpenAI-compatible backends -
# can also be used by overriding BDDK_LLM_URL) --------------------------------

LLM_BASE_URL = os.environ.get("BDDK_LLM_URL", "http://localhost:11434")
LLM_TIMEOUT = float(os.environ.get("BDDK_LLM_TIMEOUT", "120.0"))

MODELS = [
    {
        "name": "gemma-4-e2b-it",
        "model_id": "gemma4:e2b-it-q8_0",
        "active_params": "2B",
        "quantization": "Q8",
        "notes": "Baseline/small",
    },
    {
        "name": "gemma-4-e4b-it",
        "model_id": "gemma4:e4b-it-q8_0",
        "active_params": "4B",
        "quantization": "Q8",
        "notes": "Mid-small",
    },
    {
        "name": "gemma-4-26b-a4b-it",
        "model_id": "gemma4:26b-a4b-it-q4_K_M",
        "active_params": "4B (MoE)",
        "quantization": "Q4_K_M",
        "notes": "Best efficiency candidate",
    },
    {
        "name": "qwen3.5-9b",
        "model_id": "qwen3.5:9b-q8_0",
        "active_params": "9B",
        "quantization": "Q8",
        "notes": "Mid-range — 3.6 has no 9B variant, keep 3.5",
    },
    {
        "name": "qwen3.6-27b",
        "model_id": "qwen3.6:27b-q4_K_M",
        "active_params": "27B",
        "quantization": "Q4_K_M",
        "notes": "Large dense (upgraded from 3.5)",
    },
    {
        "name": "qwen3.6-35b-a3b",
        "model_id": "qwen3.6:35b-a3b-q4_K_M",
        "active_params": "3B (MoE)",
        "quantization": "Q4_K_M",
        "notes": "Large-efficient (upgraded from 3.5)",
    },
    {
        "name": "qwen3.5-9b-uncensored",
        "model_id": "huihui_ai/qwen3.5-abliterated:9b-q8_0",
        "active_params": "9B",
        "quantization": "Q8",
        "notes": "Uncensored variant — huihui abliterated",
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
