"""Dump current markdown_content for a set of document_ids as JSON.

Used to take a pre-Chandra2 baseline before the backfill overwrites rows.

Usage:
    uv run python scripts/snapshot_mevzuat_markdown.py \
        --ids-file scripts/backfill_ids_20260418.txt \
        --out logs/lightocr_baseline_20260418.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SNAPSHOT_SQL = """
SELECT document_id, extraction_method, markdown_content,
       LENGTH(markdown_content) AS chars
FROM documents
WHERE document_id = ANY($1::text[])
ORDER BY document_id
"""


async def _run(ids_file: Path, out_path: Path) -> int:
    import asyncpg

    from config import DATABASE_URL

    ids = [
        line.strip()
        for line in ids_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not ids:
        print(f"No IDs found in {ids_file}", file=sys.stderr)
        return 2

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=2)
    try:
        rows = await pool.fetch(SNAPSHOT_SQL, ids)
    finally:
        await pool.close()

    payload = [
        {
            "document_id": r["document_id"],
            "extraction_method": r["extraction_method"],
            "markdown_content": r["markdown_content"] or "",
            "chars": int(r["chars"] or 0),
        }
        for r in rows
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    missing = set(ids) - {r["document_id"] for r in rows}
    if missing:
        print(f"WARNING: {len(missing)} IDs not found: {sorted(missing)}", file=sys.stderr)
    print(f"Wrote {len(payload)} rows to {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Snapshot markdown_content for a set of doc_ids")
    p.add_argument("--ids-file", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    return asyncio.run(_run(args.ids_file, args.out))


if __name__ == "__main__":
    sys.exit(main())
