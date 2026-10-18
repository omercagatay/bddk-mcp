"""Flattened-table census: runs of >= 4 consecutive short paragraphs (<= 30 chars, no list label, not headings).
usage: flatcensus.py DOCUMENTS.json [--show]"""

import json
import re
import sys

docs = json.load(open(sys.argv[1]))
LIST = re.compile(r"^\s*(?:[a-zçğıöşü]{1,3} ?[.)]|\(?[ivx]{1,4}[.)]|\(\d{1,2}\)|[•▪➢✓→—-])\s")
tot = 0
for d in docs:
    ps = d["markdown_content"].split("\n\n")
    runs, cur = [], []
    for i, p in enumerate(ps):
        s = p.strip()
        short = (
            0 < len(s) <= 30
            and not s.startswith(("|", "$", "#", "MADDE", "Madde"))
            and not LIST.match(s)
            and "\n" not in s
        )
        if short:
            cur.append(i)
        else:
            if len(cur) >= 4:
                runs.append(cur)
            cur = []
    if len(cur) >= 4:
        runs.append(cur)
    if runs:
        tot += len(runs)
        print(d["document_id"], d["extraction_method"], "runs", len(runs), "paras", sum(len(r) for r in runs))
        if "--show" in sys.argv:
            for r in runs[:3]:
                print("    ", [ps[i].strip()[:25] for i in r[:10]])
print("total runs", tot)
