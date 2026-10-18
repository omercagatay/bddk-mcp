"""Turn the flattened closing Resmî Gazete list of html_parser regulations into the reviewed table layout.
usage: rgflat.py SP [--apply]"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

SP = Path(sys.argv[1])
APPLY = "--apply" in sys.argv
sys.path.insert(0, str(SP))
W = SP / "work" / "documents.json"
docs = json.loads(W.read_text())
DATE = re.compile(r"\d{1,2}/\d{1,2}/\d{4}$")
NUM = re.compile(r"\d{4,5}(?: ?\(?\d?\.? ?Mükerrer\)?| Mükerrer)?$")
LABEL = re.compile(r"\d{1,2}\.?$")


def T(s):
    return Counter(w.lower() for w in re.findall(r"\w+", s))


def table(head, orig, amend_head, rows):
    lines = [f"|  | {head} | {head} |", "|---|---|---|", "|  | Tarihi | Sayısı |", f"|  | {orig[0]} | {orig[1]} |"]
    if amend_head:
        lines += [f"|  | {amend_head} | {amend_head} |", "|  | Tarihi | Sayısı |"]
        lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(lines)


for d in docs:
    t = d["markdown_content"]
    m = re.search(r"(?:^|\n\n)(\S+ Yayımlandığı Resmî ?Gazete[’']?nin)\n\nTarihi\n\nSayısı\n\n", t)
    if not m:
        continue
    start = m.start(1)
    paras = t[start:].split("\n\n")
    head = paras[0]
    i = 3
    if not (DATE.match(paras[i]) and NUM.match(paras[i + 1])):
        print(d["document_id"], "unexpected original row", paras[i : i + 2])
        continue
    orig = (paras[i], paras[i + 1])
    i += 2
    amend, rows = None, []
    if i + 2 < len(paras) and "Değişiklik Yapan" in paras[i] and paras[i + 1] == "Tarihi" and paras[i + 2] == "Sayısı":
        amend = paras[i]
        i += 3
        while i < len(paras) and LABEL.match(paras[i]):
            label = paras[i]
            i += 1
            date = num = ""
            if i < len(paras) and DATE.match(paras[i]):
                date = paras[i]
                i += 1
                if i < len(paras) and NUM.match(paras[i]):
                    num = paras[i]
                    i += 1
            rows.append((label, date, num))
    old = "\n\n".join(paras[:i])
    new = table(head, orig, amend, rows)
    # token check: the table repeats merged header labels only
    lost = T(old) - T(new)
    if lost:
        print(d["document_id"], "LOST", lost)
        continue
    print(d["document_id"], "rows", len(rows), "next:", repr(paras[i][:40]) if i < len(paras) else "<end>")
    if APPLY:
        newt = t[:start] + new + t[start + len(old) :]
        log = SP / "manual" / f"{d['document_id']}.json"
        prior = json.loads(log.read_text()) if log.exists() else []
        log.write_text(json.dumps(prior + [{"old": old, "new": new, "rg_table": True}], ensure_ascii=False, indent=1))
        d["markdown_content"] = newt
if APPLY:
    W.write_text(json.dumps(docs, ensure_ascii=False) + "\n")
