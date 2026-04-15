# benchmark/phase1_nli.py
"""Phase 1b: Banking NLI evaluator.

Sends BDDK-NLI premise-hypothesis pairs to the model and asks it
to classify the relationship as entailment, contradiction, or neutral.
"""

from __future__ import annotations

import logging
import time

import httpx

from benchmark.config import OLLAMA_BASE_URL, OLLAMA_TIMEOUT
from benchmark.nli_dataset import load_pairs
from benchmark.scoring import nli_metrics

logger = logging.getLogger(__name__)

NLI_SYSTEM_PROMPT = (
    "Sen bir Türk bankacılık mevzuatı uzmanısın. Sana iki metin verilecek: "
    "bir öncül (premise) ve bir hipotez (hypothesis). "
    "Bu iki metin arasındaki ilişkiyi belirle.\n\n"
    "Yanıtın SADECE şu üç kelimeden biri olmalı:\n"
    "- entailment (öncül hipotezi destekliyor)\n"
    "- contradiction (öncül hipotezle çelişiyor)\n"
    "- neutral (ilişki belirsiz veya dolaylı)\n\n"
    "Başka hiçbir şey yazma, sadece etiket."
)


def _build_nli_prompt(premise: str, hypothesis: str) -> str:
    """Build the user prompt for NLI classification."""
    return f"Öncül: {premise}\nHipotez: {hypothesis}\nİlişki:"


def _parse_nli_label(text: str) -> str:
    """Extract NLI label from model response text."""
    text = text.strip().lower()
    for label in ("entailment", "contradiction", "neutral"):
        if label in text:
            return label
    return "unknown"


async def _classify_pair(
    client: httpx.AsyncClient,
    model: str,
    premise: str,
    hypothesis: str,
) -> tuple[str, float]:
    """Classify a single NLI pair. Returns (predicted_label, latency_s)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": NLI_SYSTEM_PROMPT},
            {"role": "user", "content": _build_nli_prompt(premise, hypothesis)},
        ],
        "stream": False,
        "temperature": 0.0,
    }

    start = time.time()
    resp = await client.post(
        f"{OLLAMA_BASE_URL}/v1/chat/completions",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    latency = time.time() - start
    resp.raise_for_status()

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    label = _parse_nli_label(content)
    return label, latency


async def run_phase1b(model_tag: str) -> dict:
    """Run Phase 1b NLI evaluation for a single model."""
    pairs = load_pairs()
    if not pairs:
        return {"phase": "1b", "model": model_tag, "error": "No NLI pairs found"}

    true_labels = []
    pred_labels = []
    details = []

    async with httpx.AsyncClient() as client:
        for pair in pairs:
            logger.info("Phase 1b: model=%s pair=%d", model_tag, pair.id)
            try:
                pred, latency = await _classify_pair(
                    client, model_tag, pair.premise, pair.hypothesis
                )
            except Exception as e:
                logger.warning("NLI pair %d failed: %s", pair.id, e)
                pred = "unknown"
                latency = 0.0

            true_labels.append(pair.label)
            pred_labels.append(pred)
            details.append({
                "pair_id": pair.id,
                "true_label": pair.label,
                "pred_label": pred,
                "correct": pair.label == pred,
                "latency_s": latency,
            })

    metrics = nli_metrics(true_labels, pred_labels)
    unknown_count = sum(1 for d in details if d["pred_label"] == "unknown")

    return {
        "phase": "1b",
        "model": model_tag,
        "total_pairs": len(pairs),
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class": metrics["per_class"],
        "unknown_responses": unknown_count,
        "avg_latency_s": sum(d["latency_s"] for d in details) / len(details) if details else 0.0,
        "details": details,
    }
