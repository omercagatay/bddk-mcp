# benchmark/nli_dataset.py
"""BDDK-NLI dataset loader and builder.

Loads NLI pairs from data/bddk_nli/pairs.json. Provides utilities
for building new pairs from the document store.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "bddk_nli"


@dataclass
class NLIPair:
    """A single NLI premise-hypothesis pair."""

    id: int
    premise: str
    hypothesis: str
    label: str  # entailment | contradiction | neutral
    source: str


def load_pairs() -> list[NLIPair]:
    """Load NLI pairs from pairs.json."""
    pairs_file = DATA_DIR / "pairs.json"
    if not pairs_file.exists():
        return []

    with open(pairs_file, encoding="utf-8") as f:
        raw = json.load(f)

    return [
        NLIPair(
            id=item["id"],
            premise=item["premise"],
            hypothesis=item["hypothesis"],
            label=item["label"],
            source=item.get("source", ""),
        )
        for item in raw
    ]


def load_metadata() -> dict:
    """Load dataset metadata."""
    meta_file = DATA_DIR / "metadata.json"
    if not meta_file.exists():
        return {}

    with open(meta_file, encoding="utf-8") as f:
        return json.load(f)


def save_pairs(pairs: list[NLIPair]) -> None:
    """Save NLI pairs to pairs.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "id": p.id,
            "premise": p.premise,
            "hypothesis": p.hypothesis,
            "label": p.label,
            "source": p.source,
        }
        for p in pairs
    ]
    with open(DATA_DIR / "pairs.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
