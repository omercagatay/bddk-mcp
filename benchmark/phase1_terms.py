# benchmark/phase1_terms.py
"""Phase 1c: Turkish BDDK terminology evaluator.

Presents each term as a multiple-choice question (1 correct + 2 distractors).
Measures whether the model understands domain-specific banking terms.
"""

from __future__ import annotations

import logging
import random
import time

import httpx

from benchmark.config import OLLAMA_BASE_URL, OLLAMA_TIMEOUT
from benchmark.terminology import TERMINOLOGY, TermQuestion

logger = logging.getLogger(__name__)

TERM_SYSTEM_PROMPT = (
    "Sen bir Türk bankacılık terimleri uzmanısın. "
    "Sana bir bankacılık terimi ve üç tanım seçeneği verilecek. "
    "Doğru tanımın harfini (A, B veya C) yaz. "
    "Sadece tek bir harf yaz, başka hiçbir şey yazma."
)


def _build_term_prompt(term: TermQuestion, seed: int) -> tuple[str, str]:
    """Build a multiple-choice prompt. Returns (prompt, correct_letter).

    Shuffles options deterministically using the seed so results are
    reproducible but the correct answer isn't always 'A'.
    """
    options = [term.correct_definition] + term.distractors
    rng = random.Random(seed)
    rng.shuffle(options)

    correct_idx = options.index(term.correct_definition)
    correct_letter = chr(ord("A") + correct_idx)

    letters = ["A", "B", "C"]
    lines = [f"Terim: {term.term}\n"]
    for letter, option in zip(letters, options, strict=True):
        lines.append(f"{letter}) {option}")

    return "\n".join(lines), correct_letter


def _parse_letter(text: str) -> str:
    """Extract answer letter from model response."""
    text = text.strip().upper()
    for ch in text:
        if ch in "ABC":
            return ch
    return ""


async def _ask_term(
    client: httpx.AsyncClient,
    model: str,
    prompt: str,
) -> tuple[str, float]:
    """Send a terminology question. Returns (answer_letter, latency_s)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TERM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
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
    return _parse_letter(content), latency


async def run_phase1c(model_tag: str) -> dict:
    """Run Phase 1c terminology evaluation for a single model."""
    details = []

    async with httpx.AsyncClient() as client:
        for i, term in enumerate(TERMINOLOGY):
            logger.info("Phase 1c: model=%s term=%d/%d", model_tag, i + 1, len(TERMINOLOGY))
            prompt, correct_letter = _build_term_prompt(term, seed=i)

            try:
                answer, latency = await _ask_term(client, model_tag, prompt)
            except Exception as e:
                logger.warning("Term %d failed: %s", i + 1, e)
                answer = ""
                latency = 0.0

            details.append(
                {
                    "term": term.term,
                    "correct_letter": correct_letter,
                    "answer": answer,
                    "correct": answer == correct_letter,
                    "latency_s": latency,
                }
            )

    n = len(details)
    correct_count = sum(1 for d in details if d["correct"])
    accuracy = correct_count / n if n else 0.0
    no_answer = sum(1 for d in details if not d["answer"])

    return {
        "phase": "1c",
        "model": model_tag,
        "total_terms": n,
        "correct": correct_count,
        "accuracy": accuracy,
        "no_answer_count": no_answer,
        "avg_latency_s": sum(d["latency_s"] for d in details) / n if n else 0.0,
        "details": details,
    }
