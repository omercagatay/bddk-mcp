"""Residual layout-defect census: residual.py DOCUMENTS.json [ids...]"""

import json
import re
import sys

docs = json.load(open(sys.argv[1]))
only = set(sys.argv[2:])
LIST = re.compile(r"^\s*(?:[a-zçğıöşü]{1,3}[.)]|\(?[ivx]{1,4}[.)]|\d{1,2}[.)-]|[•▪➢✓→—-])\s")
rows = []
for d in docs:
    did = d["document_id"]
    if only and did not in only:
        continue
    paras = [p for p in d["markdown_content"].split("\n\n")]
    pageno = sum(1 for p in paras if re.fullmatch(r"\s*(?:-\s*)?\d{1,3}(?:\s*-)?\s*", p))
    mid = []
    for a, b in zip(paras, paras[1:], strict=False):
        a2, b2 = a.rstrip(), b.lstrip()
        if not a2 or not b2 or a2.startswith("|") or b2.startswith(("|", "$", "#")) or a2.endswith("$"):
            continue
        if (
            re.search(r"[a-zçğıöşü,]$", a2)
            and re.match(r"[a-zçğıöşü]", b2)
            and not LIST.match(b2)
            and not re.match(
                r"(karar verilmiştir|ifade eder|uygun görülmüştür|kredi kartı talebi toplamaları uygun)", b2
            )
        ):
            mid.append((a2[-40:], b2[:40]))
    dbl = len(re.findall(r"\S  +\S", d["markdown_content"]))
    rows.append((did, d["extraction_method"], pageno, len(mid), dbl, mid[:3]))
for r in sorted(rows, key=lambda r: -(r[2] + r[3])):
    if r[2] or r[3]:
        print(r[0], r[1], f"pageno={r[2]} midbreak={r[3]} dblspace={r[4]}", r[5][:2])
print("docs with pageno", sum(1 for r in rows if r[2]), "with midbreak", sum(1 for r in rows if r[3]))
