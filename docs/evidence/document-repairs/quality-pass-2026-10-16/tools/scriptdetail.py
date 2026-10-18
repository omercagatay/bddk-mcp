"""Per-document script census: scriptdetail.py SP DOC_ID -> (base word, script text, kind) counts from the PDF."""

import sys
from collections import Counter

import pdfplumber

SP, DID = sys.argv[1], sys.argv[2]
c = Counter()
with pdfplumber.open(f"{SP}/pdf/{DID}.bin") as pdf:
    for pno, pg in enumerate(pdf.pages, 1):
        ch = pg.chars
        k = 1
        while k < len(ch):
            a, b = ch[k - 1], ch[k]
            dy = b["top"] - a["top"]
            if (
                b["text"].isalnum()
                and a["text"].isalpha()
                and b["size"] < a["size"] - 0.4
                and b["x0"] - a["x1"] < 2
                and (1.0 <= dy < a["size"] * 0.8 or -a["size"] * 0.8 < dy <= -1.0)
            ):
                j, w = k - 1, ""
                while (
                    j >= 0
                    and ch[j]["text"].isalpha()
                    and abs(ch[j]["size"] - a["size"]) < 0.5
                    and abs(ch[j]["top"] - a["top"]) < 0.5
                ):
                    w = ch[j]["text"] + w
                    j -= 1
                s, m = "", k
                while (
                    m < len(ch)
                    and ch[m]["text"].strip()
                    and abs(ch[m]["size"] - b["size"]) < 0.3
                    and abs(ch[m]["top"] - b["top"]) < 0.5
                ):
                    s += ch[m]["text"]
                    m += 1
                c[(w, s, "sub" if dy > 0 else "sup", pno)] += 1
                k = m
                continue
            k += 1
agg = Counter()
for (w, s, kind, _pno), n in c.items():
    agg[(w, s, kind)] += n
for (w, s, kind), n in sorted(agg.items(), key=lambda x: (-x[1], x[0])):
    pages = sorted({p for (w2, s2, k2, p) in c if (w2, s2, k2) == (w, s, kind)})
    print(f"{n:4d} {kind} {w!r} {s!r} pages={pages}")
