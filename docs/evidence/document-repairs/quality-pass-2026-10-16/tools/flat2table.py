"""Rebuild a flattened table (one cell per paragraph) as Markdown: flat2table.py SP DOC_ID SPECS.json [--apply]
SPECS: [{"start": first cell, "after": text that precedes it (to locate), "ncol": n, "end": paragraph after the table,
         "header_rows": [[...]] optional explicit header rows replacing the first cells}]"""

import json
import sys
from pathlib import Path

SP = Path(sys.argv[1])
DID = sys.argv[2]
specs = json.loads(Path(sys.argv[3]).read_text())
APPLY = "--apply" in sys.argv
W = SP / "work" / "documents.json"
docs = json.loads(W.read_text())
d = next(x for x in docs if x["document_id"] == DID)
old = d["markdown_content"]
t = old
for s in specs:
    base = t.index(s["after"]) if s.get("after") else 0
    a = t.index("\n\n" + s["start"] + "\n\n", base) + 2
    b = t.index("\n\n" + s["end"], a)
    cells = t[a:b].split("\n\n")
    n = s["ncol"]
    skip = s.get("skip_cells", 0)
    head = s.get("header_rows")
    body = cells[skip:]
    assert len(body) % n == 0, (s["start"], len(body), n)
    rows = [body[i : i + n] for i in range(0, len(body), n)]
    if head:
        rows = head + rows
    md = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * n] + ["| " + " | ".join(r) + " |" for r in rows[1:]]
    new = "\n".join(md)
    print(f"{DID}: {s['start']!r} {len(rows)} rows x {n}\n{new}\n")
    t = t[:a] + new + t[b:]
if APPLY and t != old:
    d["markdown_content"] = t
    W.write_text(json.dumps(docs, ensure_ascii=False) + "\n")
    log = SP / "manual" / f"{DID}.json"
    prior = json.loads(log.read_text()) if log.exists() else []
    log.write_text(
        json.dumps(prior + [{"old": old, "new": t, "flat_tables": len(specs)}], ensure_ascii=False, indent=1)
    )
