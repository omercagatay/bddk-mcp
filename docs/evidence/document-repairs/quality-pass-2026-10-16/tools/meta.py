"""Refresh metadata of changed documents in a documents.json: meta.py SP IN.json OUT.json
content_hash = sha256(markdown); total_pages = ceil(len/5000) (>=1); extraction_method from methods.json when listed."""

import hashlib
import json
import math
import sys
from pathlib import Path

SP = Path(sys.argv[1])
docs = json.loads(Path(sys.argv[2]).read_text())
head = {d["document_id"]: d for d in json.loads((SP / "head_docs.json").read_text())}
methods = json.loads((SP / "methods.json").read_text())
n = 0
for d in docs:
    h = head[d["document_id"]]
    if d["markdown_content"] == h["markdown_content"]:
        continue
    d["content_hash"] = hashlib.sha256(d["markdown_content"].encode()).hexdigest()
    d["total_pages"] = max(1, math.ceil(len(d["markdown_content"]) / 5000))
    if d["document_id"] in methods:
        d["extraction_method"] = methods[d["document_id"]]
    n += 1
Path(sys.argv[3]).write_text(json.dumps(docs, ensure_ascii=False) + "\n")
print("changed docs", n)
