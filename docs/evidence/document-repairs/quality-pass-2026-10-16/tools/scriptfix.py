"""Write glued formula-variable sub/superscripts as LaTeX: scriptfix.py SP [--write]

Every token below is printed in the source PDF with a smaller, lowered (sub) or raised (sup) script part
(scriptdetail.py: glyph size and baseline). Only prose is touched (LaTeX spans are skipped). `skip` lists the
0-based prose occurrences the PDF prints plain (checked on the page image); `expect` is the number replaced.
"""

import json
import re
import sys
from pathlib import Path

SP = Path(sys.argv[1])
B, A = r"(?<![A-Za-zÇĞİÖŞÜçğıöşü0-9])", r"(?![A-Za-zÇĞİÖŞÜçğıöşü0-9])"


def subs(base, script, expect, skip=(), kind="_", left=B):
    return {"base": base, "script": script, "expect": expect, "skip": list(skip), "kind": kind, "left": left}


SPEC = {
    "903": [
        subs("GKD", "1", 6),
        subs("GKD", "1", 1, left=r"(?<=85x)"),
        subs("GKD", "2", 4),
        subs("GKD", "3", 3),
        subs("KTO", "1", 1),
        subs("KTO", "2", 1),
        subs("KTO", "3", 1),
        subs("İTKD", "1", 4),
        subs("İTKD", "2", 2),
        subs("İTKD", "3", 1),
        subs("İpotekTutarı", "1", 5),
        subs("İpotekTutarı", "2", 4),
        subs("İpotekTutarı", "3", 2),
    ],
    "1305": [
        subs("BK", "ET", 5),
        subs("H", "K", 2),
        subs("H", "T", 1),
        subs("K", "U", 2),
        subs("NA", "t", 1),
        subs("T", "VA", 3),
        subs("T", "VAi", 1),
        subs("THK", "S", 1),
        subs("THK", "T", 3),
        subs("THK", "Ti", 1),
        subs("THK", "U", 5),
    ],
    # p1 'ğ) BKET:' (definition) is printed plain; pp.5-16 print BK with subscript ET
    "mevzuat_21194": [subs("BK", "ET", 5, skip=[0]), subs("TO", "kt", 1), subs("NA", "t", 1)],
    # p22 '(Di)' (kredi türevi) and p14 'sözleşme vadesi (Vi)' are printed plain
    "1313": [
        subs("AN", "i", 1),
        subs("BA", "i", 6),
        subs("BI", "i", 5),
        subs("BN", "i", 1),
        subs("BT", "i", 1),
        subs("CPD", "NG", 1),
        subs("CTD", "MS", 1),
        subs("D", "i", 4, skip=[2]),
        subs("D", "j1", 1),
        subs("D", "j2", 1),
        subs("D", "j3", 1),
        subs("DOD", "i", 2),
        subs("Eklenti", "toplam", 2, kind="^"),
        subs("H", "portföy", 2),
        subs("K", "i", 1),
        subs("K", "s", 2),
        subs("PF", "i", 2),
        subs("RAÖRT", "i", 1),
        subs("RT", "i", 9),
        subs("S", "MKT", 4),
        subs("S", "i", 2),
        subs("T", "A", 1),
        subs("T", "B", 1),
        subs("T", "t", 2),
        subs("TF", "MKT", 2),
        subs("TF", "i", 4),
        subs("TF", "ÜK", 2),
        subs("UF", "i", 1),
        subs("V", "i", 2, skip=[0]),
        subs("VF", "i", 3),
        subs("VO", "i", 7),
        subs("d", "i", 4),
        subs("f", "A", 1),
        subs("f", "B", 1),
        subs("f", "portföy", 1),
        subs("f", "s", 1),
        subs("f", "t", 1),
        subs("δ", "i", 2),
        subs("ρ", "i", 3),
        subs("σ", "i", 1),
    ],
}
# 1285: 'CO2eq' is printed with a subscript 2 in 8 places (pp.2-14), with subscript '2eq' twice (p17), plain elsewhere;
# indices are the document-order occurrences of 'CO2eq' (also inside 'CO2eqg/ÜYK'), aligned by page and context
CO2 = {
    0: "CO$_{2}$eq",
    1: "CO$_{2}$eq",
    2: "CO$_{2}$eq",
    3: "CO$_{2}$eq",
    4: "CO$_{2}$eq",
    6: "CO$_{2}$eq",
    7: "CO$_{2}$eq",
    12: "CO$_{2}$eq",
    18: "CO$_{2eq}$",
    19: "CO$_{2eq}$",
}

W = SP / "work" / "documents.json"
docs = json.loads(W.read_text())
by_id = {d["document_id"]: d for d in docs}
changes = {}
for did, items in SPEC.items():
    t = by_id[did]["markdown_content"]
    n_doc = 0
    for it in items:
        prose = re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$", lambda m: "\0" * len(m.group(0)), t, flags=re.S)
        tok = it["base"] + it["script"]
        hits = [m.start() for m in re.finditer(it["left"] + re.escape(tok) + A, prose)]
        keep = [h for i, h in enumerate(hits) if i not in it["skip"]]
        assert len(keep) == it["expect"], (did, tok, len(hits), it)
        rep = f"{it['base']}${it['kind']}{{{it['script']}}}$"
        for h in reversed(keep):
            t = t[:h] + rep + t[h + len(tok) :]
        n_doc += len(keep)
    changes[did] = (by_id[did]["markdown_content"], t, n_doc)
t = by_id["1285"]["markdown_content"]
hits = [m.start() for m in re.finditer("CO2eq", t)]
assert len(hits) == 20, len(hits)
for i in sorted(CO2, reverse=True):
    t = t[: hits[i]] + CO2[i] + t[hits[i] + 5 :]
changes["1285"] = (by_id["1285"]["markdown_content"], t, len(CO2))
for did, (_old, _new, n) in changes.items():
    print(did, n)
if "--write" in sys.argv:
    for did, (old, new, n) in changes.items():
        by_id[did]["markdown_content"] = new
        log = SP / "manual" / f"{did}.json"
        prior = json.loads(log.read_text()) if log.exists() else []
        log.write_text(json.dumps(prior + [{"old": old, "new": new, "scripts": n}], ensure_ascii=False, indent=1))
    W.write_text(json.dumps(docs, ensure_ascii=False) + "\n")
    print("written")
