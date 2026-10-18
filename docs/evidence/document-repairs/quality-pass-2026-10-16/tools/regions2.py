"""Ruled tables -> relay `regions` (page area + Markdown): regions2.py SP DOC_ID [pages] [--write]

Cell text comes from pdfplumber's own table extraction; a grid position covered by a merged cell repeats that
cell's text (relay convention). A row whose positions all belong to one cell (a full-width frame row such as
'Amaç: …', 'Tanımlar …') becomes paragraphs: its lines are split where the vertical gap is wider than the normal
line spacing or where a line starts with a 'Label:' cue. Consecutive grid rows become one Markdown table.
"""

import json
import re
import statistics
import sys
from pathlib import Path

import pdfplumber

SP = Path(sys.argv[1])
DID = sys.argv[2]
PAGES = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else None
PMAP = {"Đ": "İ", "ﬁ": "fi", "ﬂ": "fl", ";": ";", "": "•", "": "▪", "": "➢", "": "✓", "": "→", "": "—", "­": ""}
PMAP["\x00"] = "•"  # a glyph the PDF font lacks (printed as the .notdef box); used as list bullets


def norm_text(x):
    for k, v in PMAP.items():
        x = x.replace(k, v)
    return x


CUE = re.compile(
    r"^(Amaç|Kapsam|İçerik|Sıklık|Format|İlave açıklama|Tanımlar|Açıklamalar|Satırlar|Sütunlar|"
    r"Şablonlar arası bağlantılar|Not)\b.{0,40}:"
)


LIST = re.compile(r"^(?:\(?[a-zçğıöşü]{1,2}\)|\(?[ivx]{1,4}[.)]|\d{1,2}[.)]|[•▪➢✓→☐-])\s")


def flat(text):
    out = ""
    for ln in (re.sub(r"\s+", " ", x).strip() for x in norm_text(text or "").split("\n")):
        if not ln:
            continue
        out += ("<br/>" if out and re.match(r"^[•▪➢✓→]", ln) else (" " if out else "")) + ln
    return out.replace("|", "\\|")


def paragraphs(page, bbox, cell_text=None):
    x0, y0, x1, y1 = bbox

    def inside(o):  # a character belongs to the cell its centre lies in (no clipped or doubled boundary lines)
        if o.get("object_type") != "char":
            return False
        cx, cy = (o["x0"] + o["x1"]) / 2, (o["top"] + o["bottom"]) / 2
        return x0 <= cx <= x1 and y0 <= cy <= y1

    lines = page.filter(inside).extract_text_lines(strip=True)
    for ln in lines:
        ln["text"] = norm_text(ln["text"])
    if cell_text is not None:  # keep only lines pdfplumber assigned to this cell (its box can reach into the grid)
        own = re.sub(r"\s+", "", cell_text)
        lines = [ln for ln in lines if re.sub(r"\s+", "", ln["text"]) in own]
    if not lines:
        return []
    gaps = [b["top"] - a["top"] for a, b in zip(lines, lines[1:], strict=False) if b["top"] > a["top"]]
    step = statistics.median(gaps) if gaps else 0
    paras, cur = [], []
    for k, ln in enumerate(lines):
        prev = lines[k - 1] if k > 0 else None
        short_end = prev is not None and prev["x1"] < x1 - 30 and re.search(r"[.:;]$", prev["text"].strip())
        head = prev is not None and re.fullmatch(
            r"(Açıklamalar|Tanımlar|Satırlar|Sütunlar|Şablonlar arası(ndaki)? (bağlantılar|ilişki))\s*:?",
            prev["text"].strip(),
        )
        new = k > 0 and (
            CUE.match(ln["text"])
            or LIST.match(ln["text"])
            or short_end
            or head
            or (step and ln["top"] - prev["top"] > 1.45 * step)
        )
        if new and cur:
            paras.append(" ".join(cur))
            cur = []
        cur.append(re.sub(r"\s+", " ", norm_text(ln["text"])).strip())
    if cur:
        paras.append(" ".join(cur))
    return paras


def center_in(box, x, y):
    return box[0] - 0.5 <= x <= box[2] + 0.5 and box[1] - 0.5 <= y <= box[3] + 0.5


regions = []
with pdfplumber.open(SP / "pdf" / f"{DID}.bin") as pdf:
    for pno, page in enumerate(pdf.pages, 1):
        if PAGES and pno not in PAGES:
            continue
        for t in page.find_tables():
            data = t.extract()
            rows = t.rows
            ncol = max(len(r.cells) for r in rows)
            colx = []
            for j in range(ncol):
                xs = [r.cells[j] for r in rows if j < len(r.cells) and r.cells[j]]
                colx.append(((min(c[0] for c in xs) + max(c[2] for c in xs)) / 2) if xs else None)
            width = t.bbox[2] - t.bbox[0]
            items = []  # (y, kind, payload)
            for i, r in enumerate(rows):
                yc = (r.bbox[1] + r.bbox[3]) / 2
                texts = [j for j in range(ncol) if j < len(r.cells) and r.cells[j]]
                if len(texts) == 1 and r.cells[texts[0]][2] - r.cells[texts[0]][0] >= 0.9 * width:
                    cb = list(r.cells[texts[0]])
                    if i + 1 < len(rows):  # the frame row ends where the next row starts
                        cb[3] = min(cb[3], rows[i + 1].bbox[1] + 1)
                    items.append((r.bbox[1], "paras", paragraphs(page, cb)))
                    continue
                vals = []
                for j in range(ncol):
                    c = r.cells[j] if j < len(r.cells) else None
                    if c:
                        vals.append(flat(data[i][j]))
                        continue
                    x = colx[j]
                    # a merged cell covering this position: the nearest real cell to the left in this row, or above
                    # in this column, whose box contains the position's centre
                    left = next((k for k in range(j - 1, -1, -1) if k < len(r.cells) and r.cells[k]), None)
                    if left is not None and x is not None and center_in(r.cells[left], x, yc):
                        vals.append(flat(data[i][left]))
                        continue
                    up = next((k for k in range(i - 1, -1, -1) if j < len(rows[k].cells) and rows[k].cells[j]), None)
                    if up is not None and x is not None and center_in(rows[up].cells[j], x, yc):
                        vals.append(flat(data[up][j]))
                        continue
                    vals.append("")
                items.append((r.bbox[1], "row", vals))
            # text printed inside the table box but in no cell (a heading or note between two ruled blocks)
            cells_all = [c for rr in rows for c in rr.cells if c]
            box = (max(t.bbox[0], 0), max(t.bbox[1], 0), min(t.bbox[2], page.width), min(t.bbox[3], page.height))
            for ln in page.crop(box).extract_text_lines(strip=True):
                cx, cy = (ln["x0"] + ln["x1"]) / 2, (ln["top"] + ln["bottom"]) / 2
                if not any(center_in(c, cx, cy) for c in cells_all):
                    items.append((ln["top"], "paras", [re.sub(r"\s+", " ", norm_text(ln["text"])).strip()]))
            items.sort(key=lambda x: x[0])
            blocks, grid = [], []
            for _y, kind, val in items:
                if kind == "row":
                    grid.append(val)
                    continue
                if grid:
                    blocks.append(("table", grid))
                    grid = []
                blocks.append(("paras", val))
            if grid:
                blocks.append(("table", grid))
            out, seen_full = [], set()
            for kind, val in blocks:
                if kind == "paras":
                    key = tuple(val)
                    if key in seen_full:  # a merged frame cell spans several rows: write it once
                        continue
                    seen_full.add(key)
                    out += [p for p in val if p]
                else:
                    keep = [j for j in range(len(val[0])) if any(row[j] for row in val)]
                    val = [[row[j] for j in keep] for row in val]
                    n = len(val[0])
                    out.append(
                        "\n".join(
                            ["| " + " | ".join(val[0]) + " |", "|" + "---|" * n]
                            + ["| " + " | ".join(row) + " |" for row in val[1:]]
                        )
                    )
            regions.append(
                {
                    "page": pno,
                    "top": round(max(t.bbox[1], 0), 1),
                    "bottom": round(min(t.bbox[3], page.height), 1),
                    "md": "\n\n".join(out),
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
