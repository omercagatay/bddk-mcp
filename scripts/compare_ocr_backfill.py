"""Generate a before/after validation report for the Chandra2 backfill.

Reads the baseline JSON produced by scripts/snapshot_mevzuat_markdown.py,
queries the live DB for current markdown, fetches cached PDF bytes for each
doc, and emits a per-doc metrics report (markdown + csv).

Usage:
    uv run python scripts/compare_ocr_backfill.py \
        --baseline logs/lightocr_baseline_20260418.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LOG_DIR = ROOT / "logs"

_EMPTY_FORM_RE = re.compile(r"<form>\s*</form>", re.IGNORECASE)
_LATEX_PATTERNS = [
    r"_\{[^}]+\}",
    r"\^\{[^}]+\}",
    r"\\frac",
    r"\\sum",
    r"\\Delta",
    r"\\cdot",
    r"\\sqrt",
    r"\\min",
    r"\\max",
]
_LATEX_RE = re.compile("|".join(_LATEX_PATTERNS))
_MD_IMG_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_HTML_IMG_RE = re.compile(r"<img\s+[^>]*src=[^>]+>", re.IGNORECASE)


def count_form_drops(text: str) -> int:
    return len(_EMPTY_FORM_RE.findall(text or ""))


def count_latex_markers(text: str) -> int:
    return len(_LATEX_RE.findall(text or ""))


def count_md_image_refs(text: str) -> int:
    return len(_MD_IMG_RE.findall(text or "")) + len(_HTML_IMG_RE.findall(text or ""))


def regression_flag(before: dict, after: dict) -> bool:
    if after["form_drops"] > before["form_drops"]:
        return True
    if after["latex_markers"] < before["latex_markers"]:
        return True
    return False


def is_silent_drop_candidate(row: dict) -> bool:
    return row.get("pdf_image_count", 0) > row.get("md_image_refs", 0) + row.get("latex_markers", 0)


def _pdf_image_count(pdf_bytes: bytes) -> int:
    if not pdf_bytes:
        return 0
    import pdfplumber

    total = 0
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            total += len(page.images or [])
    return total


CURRENT_SQL = """
SELECT document_id, extraction_method, markdown_content,
       LENGTH(markdown_content) AS chars
FROM documents
WHERE document_id = ANY($1::text[])
ORDER BY document_id
"""


async def _build_rows(baseline: list[dict]) -> list[dict]:
    import asyncpg

    from config import DATABASE_URL
    from doc_store import DocumentStore

    ids = [b["document_id"] for b in baseline]
    by_id_before = {b["document_id"]: b for b in baseline}

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=3)
    try:
        current = await pool.fetch(CURRENT_SQL, ids)
        store = DocumentStore(pool)
        out: list[dict] = []
        for row in current:
            doc_id = row["document_id"]
            before = by_id_before.get(doc_id, {"markdown_content": "", "chars": 0})
            after_md = row["markdown_content"] or ""
            before_md = before.get("markdown_content", "") or ""
            pdf_bytes = await store.get_pdf_bytes(doc_id)
            pdf_imgs = _pdf_image_count(pdf_bytes) if pdf_bytes else 0
            before_metrics = {
                "chars": before.get("chars", 0),
                "form_drops": count_form_drops(before_md),
                "latex_markers": count_latex_markers(before_md),
                "md_image_refs": count_md_image_refs(before_md),
            }
            after_metrics = {
                "chars": int(row["chars"] or 0),
                "form_drops": count_form_drops(after_md),
                "latex_markers": count_latex_markers(after_md),
                "md_image_refs": count_md_image_refs(after_md),
            }
            entry = {
                "document_id": doc_id,
                "extraction_method_before": before.get("extraction_method"),
                "extraction_method_after": row["extraction_method"],
                "pdf_image_count": pdf_imgs,
                "before": before_metrics,
                "after": after_metrics,
                "regression": regression_flag(before_metrics, after_metrics),
                "silent_drop_candidate": is_silent_drop_candidate(
                    {
                        "pdf_image_count": pdf_imgs,
                        "md_image_refs": after_metrics["md_image_refs"],
                        "latex_markers": after_metrics["latex_markers"],
                    }
                ),
            }
            out.append(entry)
        return out
    finally:
        await pool.close()


def _render_markdown(rows: list[dict]) -> str:
    lines = ["# OCR backfill comparison report", ""]
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## Per-doc metrics")
    lines.append("")
    lines.append(
        "| doc_id | method_before → after | chars Δ | form_drops Δ | "
        "latex Δ | md_imgs Δ | pdf_imgs | regression | silent? |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        b, a = r["before"], r["after"]
        lines.append(
            f"| {r['document_id']} | {r['extraction_method_before']} → {r['extraction_method_after']} "
            f"| {b['chars']} → {a['chars']} "
            f"| {b['form_drops']} → {a['form_drops']} "
            f"| {b['latex_markers']} → {a['latex_markers']} "
            f"| {b['md_image_refs']} → {a['md_image_refs']} "
            f"| {r['pdf_image_count']} "
            f"| {'REGRESSION' if r['regression'] else 'ok'} "
            f"| {'YES' if r['silent_drop_candidate'] else '-'} |"
        )
    regressions = [r["document_id"] for r in rows if r["regression"]]
    silent = [r["document_id"] for r in rows if r["silent_drop_candidate"]]
    lines.append("")
    lines.append(f"**Regressions:** {regressions or 'none'}")
    lines.append(f"**Silent drop candidates:** {silent or 'none'}")
    return "\n".join(lines) + "\n"


def _render_csv(rows: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "document_id",
        "extraction_method_before",
        "extraction_method_after",
        "chars_before",
        "chars_after",
        "form_drops_before",
        "form_drops_after",
        "latex_markers_before",
        "latex_markers_after",
        "md_image_refs_before",
        "md_image_refs_after",
        "pdf_image_count",
        "regression",
        "silent_drop_candidate",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "document_id": r["document_id"],
                    "extraction_method_before": r["extraction_method_before"],
                    "extraction_method_after": r["extraction_method_after"],
                    "chars_before": r["before"]["chars"],
                    "chars_after": r["after"]["chars"],
                    "form_drops_before": r["before"]["form_drops"],
                    "form_drops_after": r["after"]["form_drops"],
                    "latex_markers_before": r["before"]["latex_markers"],
                    "latex_markers_after": r["after"]["latex_markers"],
                    "md_image_refs_before": r["before"]["md_image_refs"],
                    "md_image_refs_after": r["after"]["md_image_refs"],
                    "pdf_image_count": r["pdf_image_count"],
                    "regression": r["regression"],
                    "silent_drop_candidate": r["silent_drop_candidate"],
                }
            )


async def _run(baseline_path: Path) -> int:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    rows = await _build_rows(baseline)

    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
    md_path = LOG_DIR / f"compare_{stamp}.md"
    csv_path = LOG_DIR / f"compare_{stamp}.csv"
    md_path.write_text(_render_markdown(rows), encoding="utf-8")
    _render_csv(rows, csv_path)
    print(f"Wrote {md_path} and {csv_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Compare OCR backfill before/after")
    p.add_argument("--baseline", required=True, type=Path)
    args = p.parse_args()
    return asyncio.run(_run(args.baseline))


if __name__ == "__main__":
    sys.exit(main())
