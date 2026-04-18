"""Pull a single mevzuat PDF from the cached DB for smoke testing.

Usage: uv run python scripts/extract_fixture_pdf.py
"""
import asyncio
import os
from pathlib import Path

import asyncpg


DOCUMENT_ID = "mevzuat_42628"
OUT_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / f"{DOCUMENT_ID}_sample.pdf"


async def main() -> None:
    dsn = os.environ.get("BDDK_DATABASE_URL")
    if not dsn:
        raise SystemExit("BDDK_DATABASE_URL not set")

    conn = await asyncpg.connect(dsn)
    try:
        row = await conn.fetchrow(
            "SELECT pdf_blob, file_size FROM documents WHERE document_id = $1",
            DOCUMENT_ID,
        )
    finally:
        await conn.close()

    if row is None or row["pdf_blob"] is None:
        raise SystemExit(f"No pdf_blob for {DOCUMENT_ID}")

    pdf_bytes = bytes(row["pdf_blob"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_bytes(pdf_bytes)
    print(f"wrote {len(pdf_bytes)} bytes to {OUT_PATH} (file_size={row['file_size']})")


if __name__ == "__main__":
    asyncio.run(main())
