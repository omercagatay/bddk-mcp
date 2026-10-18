"""Rejoin a paragraph split by a page-bottom footnote block: footmove.py SP [--apply]

A (no closing punctuation) + footnote paragraphs ("$^{n}$ ..." or the "_______" rule) + C (starts lowercase)
-> A + "\n" + C, followed by the footnote paragraphs. Token multiset is unchanged.
"""

import json
import re
import sys
from pathlib import Path

SP = Path(sys.argv[1])
APPLY = "--apply" in sys.argv
SKIP = {"1334", "1043", "1045", "1305", "1313", "1314", "903", "905", "907", "943", "1296"}
W = SP / "work" / "documents.json"
docs = json.loads(W.read_text())
FOOT = re.compile(r"^(\$\^\{\d{1,2}\}\$ \S|_______$)")
total = 0
for d in docs:
    did = d["document_id"]
    if did in SKIP:
        continue
    paras = d["markdown_content"].split("\n\n")
    out = []
    i = 0
    n = 0
    while i < len(paras):
        a = paras[i]
        j = i + 1
        while j < len(paras) and FOOT.match(paras[j].strip()):
            j += 1
        if (
            j > i + 1
            and j < len(paras)
            and re.search(r"[\w,(\-]$", a.rstrip())
            and not a.lstrip().startswith("|")
            and re.match(r"[a-zçğıöşü(]", paras[j].lstrip())
        ):
            out.append(a.rstrip() + "\n" + paras[j].lstrip())
            out.extend(paras[i + 1 : j])
            n += 1
            i = j + 1
            continue
        out.append(a)
        i += 1
    if n:
        total += n
        new = "\n\n".join(out)
        print(did, n)
        if APPLY:
            old = d["markdown_content"]
            d["markdown_content"] = new
            log = SP / "manual" / f"{did}.json"
            prior = json.loads(log.read_text()) if log.exists() else []
            log.write_text(
                json.dumps(prior + [{"old": old, "new": new, "footnote_rejoin": n}], ensure_ascii=False, indent=1)
            )
print("total", total)
if APPLY:
    W.write_text(json.dumps(docs, ensure_ascii=False) + "\n")
