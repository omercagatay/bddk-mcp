"""Print a numeric retrieval-quality score for autoresearch.

Composite score:
    MRR * 40 + Hit@1 * 40 + F1@3 * 20

The script reuses the golden corpus from tests/test_f1_score.py so the metric
and regression tests measure the same retrieval behavior.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import importlib.util
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import asyncpg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bddk_mcp.db_lifecycle import assert_database_ready  # noqa: E402
from bddk_mcp.store.doc_store import DocumentStore, StoredDocument  # noqa: E402
from bddk_mcp.store.vector_store import VectorStore  # noqa: E402

_TEST_GUARD_SQL = """
SELECT current_database()::pg_catalog.text AS database_name, guard_hash
FROM bddk_meta.retrieval_benchmark_guard
WHERE singleton = true
"""


def _assert_test_database_identity(dsn: str) -> str:
    """Reject missing or known runtime/lifecycle identities for destructive scoring."""
    normalized = dsn.strip()
    if not normalized:
        raise RuntimeError("The retrieval benchmark requires a non-empty test database URL.")
    reused = next(
        (
            variable
            for variable in (
                "BDDK_DATABASE_URL",
                "BDDK_OPERATOR_DATABASE_URL",
                "BDDK_SCHEMA_OWNER_DATABASE_URL",
                "BDDK_INGESTION_DATABASE_URL",
                "BDDK_TELEMETRY_DATABASE_URL",
            )
            if os.environ.get(variable, "").strip() == normalized
        ),
        None,
    )
    if reused is not None:
        raise RuntimeError(
            f"The retrieval benchmark database must not reuse {reused}; provision a disposable test identity."
        )
    return normalized


def _require_test_database_url() -> str:
    """Return the explicit test-only database identity used by the benchmark."""
    dsn = os.environ.get("BDDK_TEST_DATABASE_URL", "").strip()
    if not dsn:
        raise RuntimeError(
            "BDDK_TEST_DATABASE_URL is required. Point it at a migrated, disposable test database; "
            "the retrieval benchmark deletes and inserts document chunks."
        )
    return _assert_test_database_identity(dsn)


def _require_test_database_guard() -> str:
    token = os.environ.get("BDDK_TEST_DATABASE_GUARD", "").strip()
    if len(token) < 32:
        raise RuntimeError("BDDK_TEST_DATABASE_GUARD must be a separately provisioned value of at least 32 characters.")
    return token


def _validate_test_database_guard(row, token: str) -> None:
    expected_hash = hashlib.sha256(token.encode()).hexdigest()
    try:
        database_name = str(row["database_name"])
        stored_hash = str(row["guard_hash"])
    except (KeyError, TypeError):
        database_name = stored_hash = ""
    if not database_name or not hmac.compare_digest(stored_hash, expected_hash):
        raise RuntimeError(
            "The retrieval benchmark target is not the approved disposable test database; no rows were changed."
        )


async def _assert_test_database_guard(pool, token: str) -> None:
    try:
        row = await pool.fetchrow(_TEST_GUARD_SQL)
    except Exception:
        raise RuntimeError(
            "The retrieval benchmark target guard could not be verified; no rows were changed."
        ) from None
    _validate_test_database_guard(row, token)


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


async def _populate(store: VectorStore, document_store: DocumentStore, corpus: list[dict]) -> None:
    for doc in corpus:
        await store.delete_document(doc["doc_id"])
        await document_store.delete_document(doc["doc_id"])
    for doc in corpus:
        await document_store.store_document(
            StoredDocument(
                document_id=doc["doc_id"],
                title=doc["title"],
                markdown_content=doc["content"],
                category=doc["category"],
            )
        )
        await store.add_document(
            doc_id=doc["doc_id"],
            title=doc["title"],
            content=doc["content"],
            category=doc["category"],
        )


async def _cleanup(store: VectorStore, document_store: DocumentStore, corpus: list[dict]) -> None:
    for doc in corpus:
        await store.delete_document(doc["doc_id"])
        await document_store.delete_document(doc["doc_id"])


async def run_score(dsn: str, *, verbose: bool = False) -> dict:
    dsn = _assert_test_database_identity(dsn)
    guard_token = _require_test_database_guard()
    f1 = _load_f1_module()
    pool = await asyncpg.create_pool(dsn, min_size=1, max_size=5, timeout=5)
    try:
        await _assert_test_database_guard(pool, guard_token)
        await assert_database_ready(pool=pool, require_corpus=False)
        store = VectorStore(pool)
        document_store = DocumentStore(pool)
        store._ensure_embeddings()

        await _populate(store, document_store, f1.CORPUS)
        try:
            evaluation = await f1._run_evaluation(store, store.search, "Default mode")
        finally:
            await _cleanup(store, document_store, f1.CORPUS)

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
    parser.add_argument(
        "--dsn",
        default=None,
        help="Override the required BDDK_TEST_DATABASE_URL test identity",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of only the numeric score")
    parser.add_argument("--verbose", action="store_true", help="Include per-query details in JSON output")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    dsn = args.dsn or _require_test_database_url()
    with redirect_stdout(sys.stderr):
        result = asyncio.run(run_score(dsn, verbose=args.verbose))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"{result['score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
