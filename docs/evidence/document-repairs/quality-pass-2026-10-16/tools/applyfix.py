"""Apply reviewed fixes to generated regions: applyfix.py SP DID FIXES.json
FIXES: {"page","top","replace":[[old,new],...]} | {"page","top","md"} |
       {"merge_from":[page,top], "into":[page,top], "continues": bool}: the first table of `merge_from` is the
       continuation of the last table of `into` (rows padded to its width; with `continues` the first row's cells
       finish the previous row's cells); trailing empty rows of `into` are dropped."""

import json
import sys

SP, DID, fx = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
p = f"{SP}/specs/{DID}.json"
s = json.load(open(p))


def reg(page, top):
    return next(r for r in s["regions"] if r["page"] == page and abs(r["top"] - top) < 1)


def rows_of(block):
    rows = [ln for ln in block.split("\n") if not ln.startswith("|---")]
    return [[c.strip() for c in ln.strip()[1:-1].split(" | ")] for ln in rows]


def md_of(rows):
    n = len(rows[0])
    return "\n".join(
        ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * n] + ["| " + " | ".join(r) + " |" for r in rows[1:]]
    )


for f in fx:
    if "merge_from" in f:
        src, dst = reg(*f["merge_from"]), reg(*f["into"])
        sb = src["md"].split("\n\n")
        db = dst["md"].split("\n\n")
        si = next(i for i, b in enumerate(sb) if b.startswith("|"))
        di = max(i for i, b in enumerate(db) if b.startswith("|"))
        add, base = rows_of(sb[si]), rows_of(db[di])
        while base and not any(base[-1]):
            base.pop()
        n = len(base[0])
        if f.get("continues"):
            first = add.pop(0)
            for j, c in enumerate(first):
                if c:
                    base[-1][j] = (base[-1][j] + " " + c).strip()
        for r in add:
            base.append(r + [""] * (n - len(r)))
        db[di] = md_of(base)
        dst["md"] = "\n\n".join(db)
        src["md"] = "\n\n".join(sb[:si] + sb[si + 1 :])
        continue
    r = reg(f["page"], f["top"])
    if "tables" in f:  # replace the region's table blocks, in order, with reviewed transcriptions
        bl = r["md"].split("\n\n")
        ti = [i for i, b in enumerate(bl) if b.startswith("|")]
        assert len(ti) == len(f["tables"]), (f["page"], f["top"], len(ti))
        for i, md in zip(ti, f["tables"], strict=True):
            bl[i] = md
        r["md"] = "\n\n".join(bl)
    if "md" in f:
        r["md"] = f["md"]
    for old, new in f.get("replace", []):
        assert r["md"].count(old) == 1, (f["page"], f["top"], old[:60], r["md"].count(old))
        r["md"] = r["md"].replace(old, new)
json.dump(s, open(p, "w"), ensure_ascii=False, indent=1)
print("fixes", len(fx))
