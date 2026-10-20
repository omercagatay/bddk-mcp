"""Check evidence bytes and exact passages; this does NOT validate legal applicability.

Run with the project's existing environment: .venv/bin/python <this-file>
"""

import gzip
import hashlib
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

root = Path(__file__).resolve().parent
sources = json.loads((root / "sources.json").read_text())
review = json.loads((root / "claims.json").read_text())
texts = {}
for source in sources:
    raw = gzip.decompress((root / source["raw_gzip"]).read_bytes())
    assert len(raw) == source["bytes"], source["id"]
    assert hashlib.sha256(raw).hexdigest() == source["sha256"], source["id"]
    if "text_path" not in source:
        continue
    soup = BeautifulSoup(raw, "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    paragraphs = [re.sub(r"\s+", " ", p.get_text("", strip=False)).strip() for p in soup.find_all("p")]
    text = "\n".join(p for p in paragraphs if p)
    assert text == (root / source["text_path"]).read_text(), source["id"]
    assert hashlib.sha256(text.encode()).hexdigest() == source["text_sha256"], source["id"]
    texts[source["id"]] = text

passages = 0
for claim in review["claims"]:
    for evidence in claim["evidence"]:
        text = texts[evidence["source_id"]]
        assert text[evidence["start_char"] : evidence["end_char"]] == evidence["quotation"], claim["id"]
        passages += 1

assert review["independent_legal_review"] is False
assert review["database_mutation"] is False
print(f"PASS: {len(sources)} source hashes, {len(texts)} reproduced extractions, {passages} exact passages.")
print("This checks file integrity and quotation locations, NOT legal approval or date-specific applicability.")
