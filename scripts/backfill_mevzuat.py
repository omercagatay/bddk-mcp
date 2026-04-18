"""One-shot bulk backfill for corrupted mevzuat_* docs in the live DB.

Scans documents for rows matching a corruption signature
(U+FFFD replacement chars, leaked ``<img`` tags, or suspiciously short
content), prompts for confirmation, then re-syncs each match serially
with a 2s delay between HTTP fetches. Uses the fixed sync_document path
so document_chunks is refreshed too.

Usage:
    uv run python scripts/backfill_mevzuat.py           # interactive
    uv run python scripts/backfill_mevzuat.py --yes     # skip prompt
    uv run python scripts/backfill_mevzuat.py --limit 1 # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from doc_store import DocumentStore  # noqa: E402
from doc_sync import DocumentSyncer  # noqa: E402
from ocr_backends import get_default_backends  # noqa: E402
from vector_store import VectorStore  # noqa: E402

LOG_DIR = ROOT / "logs"

SCAN_SQL = """
SELECT
    d.document_id,
    d.title,
    d.source_url,
    d.category,
    d.decision_date,
    d.decision_number,
    LENGTH(d.markdown_content) AS len,
    CASE
        WHEN d.markdown_content LIKE '%' || chr(65533) || '%' THEN 'ufffd'
        WHEN d.markdown_content LIKE '%<img%' THEN 'leaked_img'
        WHEN LENGTH(d.markdown_content) < 500 THEN 'too_short'
        WHEN d.extraction_method = 'markitdown_degraded' THEN 'degraded_pdf'
    END AS signature
FROM documents d
WHERE d.document_id LIKE 'mevzuat_%'
  AND (
    d.markdown_content LIKE '%' || chr(65533) || '%'
    OR d.markdown_content LIKE '%<img%'
    OR LENGTH(d.markdown_content) < 500
    OR d.extraction_method = 'markitdown_degraded'
  )
ORDER BY d.document_id
"""


def _setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    log_path = LOG_DIR / f"backfill_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return log_path


async def _scan(pool) -> list[dict]:
    rows = await pool.fetch(SCAN_SQL)
    return [dict(r) for r in rows]


FETCH_BY_IDS_SQL = """
SELECT
    d.document_id,
    d.title,
    d.source_url,
    d.category,
    d.decision_date,
    d.decision_number,
    LENGTH(d.markdown_content) AS len,
    'forced' AS signature
FROM documents d
WHERE d.document_id = ANY($1::text[])
ORDER BY d.document_id
"""


async def _fetch_by_ids(pool, ids: list[str]) -> list[dict]:
    rows = await pool.fetch(FETCH_BY_IDS_SQL, ids)
    return [dict(r) for r in rows]


def _print_scan_summary(rows: list[dict]) -> None:
    by_sig: dict[str, int] = {}
    for r in rows:
        by_sig[r["signature"]] = by_sig.get(r["signature"], 0) + 1
    print(f"\nFound {len(rows)} candidate docs:")
    for sig, count in sorted(by_sig.items()):
        print(f"  {sig:12s} {count}")
    preview = rows[:10]
    print("\nFirst 10 IDs:")
    for r in preview:
        print(f"  {r['document_id']}  len={r['len']:>6}  sig={r['signature']}")
    if len(rows) > 10:
        print(f"  ... and {len(rows) - 10} more")


def _confirm(n: int) -> bool:
    ans = input(f"\nProceed with re-sync of {n} docs? [y/N] ").strip().lower()
    return ans in ("y", "yes")


async def _run(args: argparse.Namespace) -> int:
    log = logging.getLogger("backfill")
    import asyncpg

    from config import DATABASE_URL

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    try:
        store = DocumentStore(pool)
        await store.initialize()
        vs = VectorStore(pool)
        await vs.initialize()

        if args.ids_file:
            ids_path = Path(args.ids_file)
            ids = [
                line.strip()
                for line in ids_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            rows = await _fetch_by_ids(pool, ids)
            missing = set(ids) - {r["document_id"] for r in rows}
            if missing:
                log.warning("IDs not found in DB: %s", sorted(missing))
        else:
            rows = await _scan(pool)
        if args.limit:
            rows = rows[: args.limit]

        if not rows:
            log.info("No candidates found — nothing to do.")
            return 0

        _print_scan_summary(rows)

        if not args.yes and not _confirm(len(rows)):
            log.info("Aborted by user.")
            return 1

        ok: list[str] = []
        failed: list[tuple[str, str]] = []

        backends = get_default_backends(include_chandra=args.use_chandra)

        async with DocumentSyncer(store, vector_store=vs, ocr_backends=backends) as syncer:
            for i, r in enumerate(rows, 1):
                doc_id = r["document_id"]
                log.info(
                    "[%d/%d] re-sync %s (sig=%s len=%d)",
                    i,
                    len(rows),
                    doc_id,
                    r["signature"],
                    r["len"],
                )
                t0 = time.monotonic()
                try:
                    result = await syncer.sync_document(
                        doc_id=doc_id,
                        title=r["title"] or "",
                        category=r["category"] or "",
                        source_url=r["source_url"] or "",
                        decision_date=r["decision_date"] or "",
                        decision_number=r["decision_number"] or "",
                        force=True,
                    )
                except Exception as e:
                    dt = time.monotonic() - t0
                    failed.append((doc_id, f"exception: {type(e).__name__}: {e}"))
                    log.error(
                        "[%d/%d] %s FAILED in %.1fs: %s",
                        i,
                        len(rows),
                        doc_id,
                        dt,
                        e,
                    )
                else:
                    dt = time.monotonic() - t0
                    if result.success:
                        ok.append(doc_id)
                        log.info(
                            "[%d/%d] %s OK in %.1fs (method=%s, size=%dB)",
                            i,
                            len(rows),
                            doc_id,
                            dt,
                            result.method,
                            result.size_bytes,
                        )
                    else:
                        failed.append((doc_id, result.error or "unknown"))
                        log.warning(
                            "[%d/%d] %s FAILED in %.1fs: %s",
                            i,
                            len(rows),
                            doc_id,
                            dt,
                            result.error,
                        )
                if i < len(rows):
                    await asyncio.sleep(2.0)

        print("\n" + "=" * 60)
        print(f"Backfill complete: {len(ok)} ok, {len(failed)} failed.")
        if failed:
            print(f"\nFailed IDs ({len(failed)}):")
            for doc_id, reason in failed:
                print(f"  {doc_id}: {reason}")
        return 0 if not failed else 1
    finally:
        await pool.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Bulk backfill corrupted mevzuat docs")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation")
    parser.add_argument("--limit", type=int, default=0, help="Max docs to process (0 = all)")
    parser.add_argument(
        "--ids-file",
        type=str,
        default="",
        help="Path to file with one document_id per line (skips SCAN_SQL)",
    )
    parser.add_argument(
        "--use-chandra",
        action="store_true",
        help="Run Chandra2 as primary OCR backend (in-process HF inference).",
    )
    args = parser.parse_args()

    log_path = _setup_logging()
    print(f"Logging to {log_path}")

    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
