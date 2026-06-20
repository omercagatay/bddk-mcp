"""Print a numeric retrieval-quality score for autoresearch.

Composite score:
    MRR * 40 + Hit@1 * 40 + F1@3 * 20

The script reuses the golden corpus from tests/test_f1_score.py so the metric
and regression tests measure the same retrieval behavior.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import asyncpg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bddk_mcp.store.vector_store import VectorStore  # noqa: E402

DEFAULT_DSN = "postgresql://bddk:bddk@localhost:5432/bddk_test"


def _load_f1_module():
    spec = importlib.util.spec_from_file_location("bddk_f1_score_dataset", ROOT / "tests" / "test_f1_score.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load tests/test_f1_score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def composite_score(avg: dict[str, float]) -> float:
    return (avg["mrr"] * 40) + (avg["hit_at_1"] * 40) + (avg["f1_at_3"] * 20)


def _json_safe(value):
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


async def _populate(store: VectorStore, corpus: list[dict]) -> None:
    for doc in corpus:
        await store.delete_document(doc["doc_id"])
    for doc in corpus:
        await store.add_document(
            doc_id=doc["doc_id"],
            title=doc["title"],
            content=doc["content"],
            category=doc["category"],
        )


async def _cleanup(store: VectorStore, corpus: list[dict]) -> None:
    for doc in corpus:
        await store.delete_document(doc["doc_id"])


async def run_score(dsn: str, *, verbose: bool = False) -> dict:
    f1 = _load_f1_module()
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5, timeout=5)
    try:
        await pool.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await pool.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
        store = VectorStore(pool)
        await store.initialize()
        store._ensure_embeddings()

        await _populate(store, f1.CORPUS)
        try:
            evaluation = await f1._run_evaluation(store, store.search, "Default mode")
        finally:
            await _cleanup(store, f1.CORPUS)

        score = composite_score(evaluation["avg"])
        result = {
            "score": round(score, 4),
            "mrr": evaluation["avg"]["mrr"],
            "hit_at_1": evaluation["avg"]["hit_at_1"],
            "f1_at_3": evaluation["avg"]["f1_at_3"],
            "f1_at_5": evaluation["avg"]["f1_at_5"],
            "queries": len(f1.GROUND_TRUTH),
        }
        if verbose:
            result["results"] = _json_safe(evaluation["results"])
        return result
    finally:
        await pool.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute BDDK retrieval-quality score.")
    parser.add_argument("--dsn", default=os.environ.get("BDDK_TEST_DATABASE_URL", DEFAULT_DSN))
    parser.add_argument("--json", action="store_true", help="Print JSON instead of only the numeric score")
    parser.add_argument("--verbose", action="store_true", help="Include per-query details in JSON output")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    with redirect_stdout(sys.stderr):
        result = asyncio.run(run_score(args.dsn, verbose=args.verbose))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"{result['score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
