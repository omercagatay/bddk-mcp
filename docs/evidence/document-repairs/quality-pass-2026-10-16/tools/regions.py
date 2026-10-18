"""Ruled tables -> relay `regions` (page area + Markdown): regions.py SP DOC_ID [pages] [--write]

Every ruled table pdfplumber finds becomes one region. Cells merged across rows or columns repeat their text in
each grid position (the relay convention); line breaks inside a cell become spaces, bullet lines '<br/>• '.
A one-column table (a boxed text frame) becomes plain paragraphs, one per row.
"""

import json
import re
import sys
from pathlib import Path

import pdfplumber

SP = Path(sys.argv[1])
DID = sys.argv[2]
PAGES = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else None
PMAP = {"Đ": "İ", "ﬁ": "fi", "ﬂ": "fl", ";": ";", "": "•", "": "▪", "": "➢", "": "✓", "": "→", "": "—", "­": ""}


def clean(text):
    for k, v in PMAP.items():
        text = text.replace(k, v)
    out = ""
    for ln in (re.sub(r"\s+", " ", x).strip() for x in text.split("\n")):
        if not ln:
            continue
        if re.match(r"^[•▪➢✓→]", ln):
            out += ("<br/>" if out else "") + ln
        else:
            out += (" " if out else "") + ln
    return out.replace("|", "\\|")


def snap(vals, tol=1.5):
    out = []
    for v in sorted(vals):
        if not out or v - out[-1] > tol:
            out.append(v)
    return out


def idx(edges, v, tol=1.5):
    return min(range(len(edges)), key=lambda i: abs(edges[i] - v))


regions = []
with pdfplumber.open(SP / "pdf" / f"{DID}.bin") as pdf:
    for pno, page in enumerate(pdf.pages, 1):
        if PAGES and pno not in PAGES:
            continue
        for t in page.find_tables():
            cells = [c for c in t.cells]
            xs = snap([c[0] for c in cells] + [c[2] for c in cells])
            ys = snap([c[1] for c in cells] + [c[3] for c in cells])
            grid = [["" for _ in xs[:-1]] for _ in ys[:-1]]
            for x0, top, x1, bottom in cells:
                c0, c1, r0, r1 = idx(xs, x0), idx(xs, x1), idx(ys, top), idx(ys, bottom)
                txt = clean(
                    page.within_bbox(
                        (max(x0, 0), max(top, 0), min(x1, page.width), min(bottom, page.height))
                    ).extract_text()
                    or ""
                )
                for r in range(r0, r1):
                    for c in range(c0, c1):
                        grid[r][c] = txt
            # drop columns/rows that are empty everywhere (ruling artefacts)
            keep_c = [c for c in range(len(xs) - 1) if any(grid[r][c] for r in range(len(grid)))]
            grid = [[row[c] for c in keep_c] for row in grid]
            grid = [row for row in grid if any(row)]
            if not grid:
                continue
            ncol = len(grid[0])
            if ncol == 1:
                md = "\n\n".join(row[0] for row in grid)
            else:
                lines = ["| " + " | ".join(grid[0]) + " |", "|" + "---|" * ncol]
                lines += ["| " + " | ".join(row) + " |" for row in grid[1:]]
                md = "\n".join(lines)
            regions.append(
                {
                    "page": pno,
                    "top": round(max(t.bbox[1], 0), 1),
                    "bottom": round(min(t.bbox[3], page.height), 1),
                    "md": md,
                }
            )
for r in regions:
    print(f"--- p{r['page']} y{r['top']}-{r['bottom']}\n{r['md']}\n")
if "--write" in sys.argv:
    sp = SP / "specs" / f"{DID}.json"
    spec = json.loads(sp.read_text()) if sp.exists() else {}
    keep = [r for r in spec.get("regions", []) if PAGES and r["page"] not in PAGES]
    spec["regions"] = sorted(keep + regions, key=lambda r: (r["page"], r["top"]))
    sp.write_text(json.dumps(spec, ensure_ascii=False, indent=1))
    print("wrote", sp, len(spec["regions"]))
