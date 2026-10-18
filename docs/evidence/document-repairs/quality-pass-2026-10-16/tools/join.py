"""Rejoin reviewed mid-sentence paragraph breaks in the work copy: join.py SP JOINS.json [--apply]
JOINS: [[did, a_tail, b_head], ...]; the paragraph ending with a_tail and the next one starting with b_head are
joined with a single line break. Each doc's change is logged to manual/{did}.json with key "joins"."""

import json
import sys
from collections import defaultdict
from pathlib import Path

SP = Path(sys.argv[1])
joins = json.loads(Path(sys.argv[2]).read_text())
APPLY = "--apply" in sys.argv
W = SP / "work" / "documents.json"
docs = json.loads(W.read_text())
byid = {d["document_id"]: d for d in docs}
per = defaultdict(list)
for did, a, b in joins:
    per[did].append((a, b))
for did, pairs in per.items():
    old = byid[did]["markdown_content"]
    paras = old.split("\n\n")
    done = 0
    for a, b in pairs:
        a_n, b_n = " ".join(a.split())[-40:], " ".join(b.split())[:30]
        hits = [
            i
            for i in range(len(paras) - 1)
            if " ".join(paras[i].split()).endswith(a_n) and " ".join(paras[i + 1].split()).startswith(b_n)
        ]
        if not hits:
            print(did, "NOT FOUND", a_n, "|", b_n)
            continue
        for i in reversed(hits):  # identical repeated pairs are all joined
            paras[i : i + 2] = [paras[i].rstrip() + "\n" + paras[i + 1].lstrip()]
            done += 1
    new = "\n\n".join(paras)
    print(did, "joined", done)
    if APPLY and new != old:
        byid[did]["markdown_content"] = new
        log = SP / "manual" / f"{did}.json"
        prior = json.loads(log.read_text()) if log.exists() else []
        log.write_text(json.dumps(prior + [{"old": old, "new": new, "joins": done}], ensure_ascii=False, indent=1))
if APPLY:
    W.write_text(json.dumps(docs, ensure_ascii=False) + "\n")
