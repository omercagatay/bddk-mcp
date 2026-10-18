"""Real mid-sentence paragraph-break candidates: realbreak.py DOCUMENTS.json [ids...] [--show]
A ends with a lowercase word (no comma/punctuation); B starts lowercase; excluded: list labels, a short closing line
printed after a list ('şarttır.', 'krediler sınıflandırılır.'), table/LaTeX blocks."""

import json
import re
import sys
from collections import Counter

docs = json.load(open(sys.argv[1]))
only = {a for a in sys.argv[2:] if not a.startswith("--")}
LIST = re.compile(r"^\s*(?:[a-zçğıöşü]{1,3} ?[.)]|\(?[ivx]{1,4}[.)]|\d{1,2}[.)-]|[•▪➢✓→—-])\s")
cnt = Counter()
for d in docs:
    did = d["document_id"]
    if only and did not in only:
        continue
    paras = d["markdown_content"].split("\n\n")
    for a, b in zip(paras, paras[1:], strict=False):
        a2, b2 = a.rstrip(), b.lstrip()
        if not a2 or not b2 or a2.startswith("|") or b2.startswith(("|", "$", "#")) or a2.endswith("$"):
            continue
        a_item = bool(LIST.match(a2))
        if (
            not (re.search(r"[a-zçğıöşü]-?$", a2) or (a2.endswith(",") and not a_item))
            or not re.match(r"[a-zçğıöşü]", b2)
            or LIST.match(b2)
        ):
            continue
        first_b = b2.split("\n")[0]
        closing = "\n" not in b2 and len(b2) <= 70 and re.search(r"[.:;,]$", b2)
        if closing and a_item:
            continue
        if closing and re.match(r"(karar verilmiştir|ifade eder|uygun görülmüştür)", b2):
            continue
        cnt[did] += 1
        if "--show" in sys.argv:
            print(f"{did}: …{a2[-60:]!r} ⏎⏎ {first_b[:60]!r}")
print(sorted(cnt.items(), key=lambda x: -x[1]))
print("total", sum(cnt.values()), "docs", len(cnt))
