"""Ruled rows whose label cell holds several sub-labels separated by blank space (no rule between them): split
into one Markdown row per sub-label, each column's lines assigned by y. usage: splitrows.py SP DID PAGES"""

import re
import sys

import pdfplumber

SP, DID = sys.argv[1], sys.argv[2]
PAGES = [int(x) for x in sys.argv[3].split(",")]


def clamp(b, pg):
    return (max(b[0], 0), max(b[1], 0), min(b[2], pg.width), min(b[3], pg.height))


out = []
with pdfplumber.open(f"{SP}/pdf/{DID}.bin") as pdf:
    for pno in PAGES:
        pg = pdf.pages[pno - 1]
        for t in pg.find_tables():
            for r in t.rows:
                if not r.cells or not r.cells[0] or len([c for c in r.cells if c]) < 3:
                    continue
                lines = pg.crop(clamp(r.cells[0], pg)).extract_text_lines()
                if len(lines) < 2:
                    continue
                steps = [b["top"] - a["top"] for a, b in zip(lines, lines[1:], strict=False)]
                step = min(steps)
                starts = [lines[0]["top"]] + [
                    b["top"] for a, b in zip(lines, lines[1:], strict=False) if b["top"] - a["top"] > 1.8 * step
                ]
                if len(starts) < 2:
                    continue
                bounds = starts[1:] + [r.bbox[3] + 1]
                rows = []
                for g0, g1 in zip(starts, bounds, strict=False):
                    row = []
                    for c in r.cells:
                        if not c:
                            continue
                        ls = pg.crop(clamp(c, pg)).extract_text_lines()
                        txt = " ".join(ln["text"] for ln in ls if g0 - 2 <= ln["top"] < g1 - 2)
                        row.append(re.sub(r"\s+", " ", txt).strip())
                    rows.append(row)
                out.append((pno, round(r.bbox[1], 1), rows))
for pno, y, rows in out:
    print(f"--- p{pno} y{y}")
    for row in rows:
        print("| " + " | ".join(row) + " |")
