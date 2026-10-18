"""Write superscript footnote markers found by supers.py as $^{n}$ (and the matching footnote bodies).

usage: g2.py SP [--apply]
A marker is located by a cursor that walks the document in the census order: the previous word followed
directly (or after one space) by the marker digits. The footnote body is a paragraph "n Text" after the marker.
"""

import json
import re
import sys
from pathlib import Path

SP = Path(sys.argv[1])
APPLY = "--apply" in sys.argv
SKIP = {
    "1287",
    "1334",
    "1043",
    "1045",
    "1305",
    "1313",
    "1314",
    "903",
    "905",
    "907",
    "943",
    "1084",
    "1126",
    "1164",
    "1327",
    "915",
    "934",
    "786",
    "787",
    "789",
    "793",
    "798",
    "1296",
}
sup = json.loads((SP / "supers.json").read_text())
W = SP / "work" / "documents.json"
docs = json.loads(W.read_text())
byid = {d["document_id"]: d for d in docs}
report = {}
for did, hits in sup.items():
    if did in SKIP or not hits or did not in byid:
        continue
    t = byid[did]["markdown_content"]
    cur = 0
    edits = []
    missing = []
    nums = []
    if did == "1041":  # "seviyesinin34 45": markers 3 and 4, then the number 4 with marker 5 (as in 1084)
        hits = [h for h in hits if not (h[1] == "seviyesinin" and h[2] == "34") and not (h[1] == "4" and h[2] == "5")]
        t = t.replace("seviyesinin34 45\n", "seviyesinin$^{3}$$^{4}$ 4$^{5}$\n", 1)
        nums_extra = [("3", t.find("$^{3}$$^{4}$")), ("4", t.find("$^{3}$$^{4}$")), ("5", t.find("4$^{5}$"))]
    else:
        nums_extra = []
    for _page, prev, mark, _glued, _nxt in hits:
        if not re.fullmatch(r"\d{1,2}", mark):
            if re.fullmatch(r"\d{2}", mark) is None:
                missing.append((prev, mark, "non-numeric"))
                continue
        pat = re.compile(re.escape(prev) + r"( ?)" + re.escape(mark) + r"(?!\d)")
        m = pat.search(t, cur)
        if not m:
            missing.append((prev, mark))
            continue
        a = m.start() + len(prev)
        edits.append((a, m.end(), f"$^{{{mark}}}$"))
        cur = m.end()
        nums.append((mark, m.end()))
    # footnote bodies: paragraph "n Text" after its marker
    for mark, pos in sorted(nums + nums_extra, key=lambda x: int(x[0])):
        fm = re.compile(r"(?:^|\n\n)(" + re.escape(mark) + r") (?=[A-ZÇĞİÖŞÜa-zçğıöşü“\"(])").search(t, pos)
        if fm:
            edits.append((fm.start(1), fm.end(1) + 1, f"$^{{{mark}}}$ "))
    edits.sort()
    # drop overlaps
    clean = []
    for e in edits:
        if clean and e[0] < clean[-1][1]:
            continue
        clean.append(e)
    new = t
    t = byid[did]["markdown_content"]
    for a, b, rep in reversed(clean):
        new = new[:a] + rep + new[b:]
    report[did] = {"markers": len(nums), "bodies": len(clean) - len(nums), "missing": missing}
    if APPLY and new != t:
        byid[did]["markdown_content"] = new
        log = SP / "manual" / f"{did}.json"
        prior = json.loads(log.read_text()) if log.exists() else []
        log.write_text(
            json.dumps(prior + [{"old": t, "new": new, "superscripts": len(nums)}], ensure_ascii=False, indent=1)
        )
    print(did, report[did])
if APPLY:
    W.write_text(json.dumps(docs, ensure_ascii=False) + "\n")
