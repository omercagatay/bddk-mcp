"""For each mid-sentence paragraph break, locate it in the PDF layout text: breakwhy.py SP [ids...]"""

import json
import re
import sys
from pathlib import Path

SP = Path(sys.argv[1])
docs = {d["document_id"]: d for d in json.loads((SP / "work" / "documents.json").read_text())}
ids = sys.argv[2:]
LIST = re.compile(r"^\s*(?:[a-zçğıöşü]{1,3}[.)]|\(?[ivx]{1,4}[.)]|\d{1,2}[.)-]|[•▪➢✓→—-])\s")
CLOSE = re.compile(r"(karar verilmiştir|ifade eder|uygun görülmüştür|kredi kartı talebi toplamaları uygun)")
for did in ids or docs:
    lay_p = SP / "lay" / f"{did}.txt"
    if not lay_p.exists():
        continue
    pages = lay_p.read_text(errors="replace").split("\f")
    paras = docs[did]["markdown_content"].split("\n\n")
    for a, b in zip(paras, paras[1:], strict=False):
        a2, b2 = a.rstrip(), b.lstrip()
        if not a2 or not b2 or a2.startswith("|") or b2.startswith(("|", "$", "#")) or a2.endswith("$"):
            continue
        if not (
            re.search(r"[a-zçğıöşü,]$", a2)
            and re.match(r"[a-zçğıöşü]", b2)
            and not LIST.match(b2)
            and not CLOSE.match(b2)
        ):
            continue
        last = a2.split("\n")[-1].strip()
        first = b2.split("\n")[0].strip()
        where = "?"
        for pno, pg in enumerate(pages, 1):
            lines = pg.split("\n")
            norm = [re.sub(r"\s+", " ", ln).strip() for ln in lines]
            for i, ln in enumerate(norm):
                if ln.endswith(last[-30:]) and last[-30:]:
                    # next nonblank line
                    j = i + 1
                    gap = 0
                    while j < len(lines) and not norm[j]:
                        j += 1
                        gap += 1
                    if j < len(lines) and norm[j].startswith(first[:20]):
                        ind_a = len(lines[i]) - len(lines[i].lstrip())
                        ind_b = len(lines[j]) - len(lines[j].lstrip())
                        where = f"p{pno} same-page gap={gap} indent {ind_a}->{ind_b} lenA={len(lines[i].rstrip())}"
                    elif j >= len(lines) or all(not x for x in norm[i + 1 :]):
                        where = f"p{pno} END-OF-PAGE"
                    else:
                        where = f"p{pno} next={norm[j][:30]!r}"
                    break
            if where != "?":
                break
        print(f"{did}: …{last[-35:]!r} ⏎⏎ {first[:35]!r}  [{where}]")
