"""One-shot seed cleanup: drop items excluded by the scope filter.

Rules (matches _is_in_scope in client.py + page removals):
- category == "Faizsiz Bankacılık"  → drop
- category == "Finansal Kiralama ve Faktoring"  → drop (page 52 no longer scraped)
- title contains "6361 sayılı" (case-insensitive)  → drop
- category == "Kurul Kararı" AND title matches firm-specific pattern  → drop
  (page 55 faaliyet/kuruluş izni noise; policy decisions like "27.04.2023 #10585
  zamanaşımına uğrayan mevduat" are kept.)

Cascades to documents.json and chunks.json by document_id.

Usage:
    uv run python scripts/clean_seed_scope.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

SEED_DIR = Path(__file__).resolve().parent.parent / "seed_data"

EXCLUDED_CATEGORIES = {"Faizsiz Bankacılık", "Finansal Kiralama ve Faktoring"}
EXCLUDED_TITLE_SUBSTRINGS = ("6361 sayılı",)
# Firm-specific Kurul Kararı marker — excludes faaliyet/kuruluş izni items from page 55.
# Boundary uses a negative lookahead on Turkish letters so "A.Ş'ye" / "A.Ş'nin" also match
# (the original trailing-dot form `A\.Ş\.` missed apostrophe suffixes).
FIRM_PAT = re.compile(
    r"(?:A\.Ş|Ş\.P\.Ç)(?![a-zA-ZçğıöşüÇĞİÖŞÜ])|Finansal Kiralama|Faktoring",
    re.IGNORECASE,
)


def should_drop(item: dict) -> bool:
    cat = item.get("category", "") or ""
    title = item.get("title", "") or ""
    if cat in EXCLUDED_CATEGORIES:
        return True
    tl = title.lower()
    if any(s.lower() in tl for s in EXCLUDED_TITLE_SUBSTRINGS):
        return True
    if cat == "Kurul Kararı" and FIRM_PAT.search(title):
        return True
    return False


def main() -> None:
    cache_path = SEED_DIR / "decision_cache.json"
    docs_path = SEED_DIR / "documents.json"
    chunks_path = SEED_DIR / "chunks.json"

    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    docs = json.loads(docs_path.read_text(encoding="utf-8"))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    print(f"Before: cache={len(cache)} docs={len(docs)} chunks={len(chunks)}")

    dropped_ids: set[str] = set()
    kept_cache = []
    for item in cache:
        if should_drop(item):
            did = item.get("document_id")
            if did:
                dropped_ids.add(did)
        else:
            kept_cache.append(item)

    kept_docs = [d for d in docs if d.get("document_id") not in dropped_ids]
    kept_chunks = [c for c in chunks if c.get("doc_id") not in dropped_ids]

    print(f"Dropped {len(dropped_ids)} document_ids.")
    print(f"After:  cache={len(kept_cache)} docs={len(kept_docs)} chunks={len(kept_chunks)}")

    # Breakdown by category of what's dropped
    dropped_items = [i for i in cache if i.get("document_id") in dropped_ids]
    from collections import Counter

    cats = Counter(i.get("category", "") for i in dropped_items)
    print("\nDropped by category:")
    for cat, n in cats.most_common():
        print(f"  {cat}: {n}")

    cache_path.write_text(json.dumps(kept_cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    docs_path.write_text(json.dumps(kept_docs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    chunks_path.write_text(json.dumps(kept_chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\nWritten: decision_cache.json, documents.json, chunks.json")


if __name__ == "__main__":
    main()
