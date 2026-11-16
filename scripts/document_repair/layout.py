"""Page geometry from the PDF text layer: chars (unmapped glyphs recovered), text bands, and formula / ruled-table
regions snapped to whole bands.

A band is a maximal group of chars whose (slightly shrunk) vertical extents overlap: one prose line, or a display
formula with its fraction levels, indices and limits. Regions are unions of whole bands, so a region edge never cuts
a glyph.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

import numpy as np

MATH_FONT = ("math", "symbol")  # Wingdings is a bullet font, not math
CID = re.compile(r"^\(cid:(\d+)\)$")


def nfkc_math(s):
    """Mathematical alphanumerics (U+1D400-1D7FF) to plain letters/digits; everything else unchanged."""
    return "".join(unicodedata.normalize("NFKC", ch) if 0x1D400 <= ord(ch) <= 0x1D7FF or ch == "ℎ" else ch for ch in s)


def is_math_char(c):
    if any(m in c["fontname"].lower() for m in MATH_FONT):
        return True
    ch = c["text"]
    if len(ch) != 1:
        return False
    o = ord(ch)
    return 0x1D400 <= o <= 0x1D7FF or 0x2200 <= o <= 0x22FF or 0x0391 <= o <= 0x03C9


def load(pdf_path):
    import pdfplumber

    from .glyphmap import install

    pdf = pdfplumber.open(pdf_path)
    install(pdf)
    pages = []
    for pno, pg in enumerate(pdf.pages, 1):
        chars = []
        for c in pg.chars:
            t = c["text"]
            m = CID.match(t)
            chars.append(
                {
                    "t": t,
                    "cid": bool(m),
                    "recovered": False,
                    "font": c["fontname"].split("+")[-1],
                    "size": c["size"],
                    "x0": c["x0"],
                    "x1": c["x1"],
                    "top": c["top"],
                    "bottom": c["bottom"],
                    "base": pg.height - c["matrix"][5],
                    "math": is_math_char(c),
                }
            )
        sizes = Counter(round(c["size"], 1) for c in chars if c["t"].strip())
        body = sizes.most_common(1)[0][0] if sizes else 12.0
        pages.append({"page": pno, "w": pg.width, "h": pg.height, "body": body, "chars": chars, "_pg": pg})
    return pdf, pages


def bands(page):
    """Group visible chars into bands by overlapping (15 % shrunk) vertical extents."""
    cs = sorted((c for c in page["chars"] if c["t"].strip()), key=lambda c: c["top"])
    out = []
    for c in cs:
        h = c["bottom"] - c["top"]
        top, bot = c["top"] + 0.15 * h, c["bottom"] - 0.15 * h
        if out and top < out[-1]["bot"]:
            b = out[-1]
            b["chars"].append(c)
            b["bot"] = max(b["bot"], bot)
        else:
            out.append({"chars": [c], "top_s": top, "bot": bot})
    for b in out:
        b["top"] = min(c["top"] for c in b["chars"])
        b["bottom"] = max(c["bottom"] for c in b["chars"])
        b["x0"] = min(c["x0"] for c in b["chars"])
        b["x1"] = max(c["x1"] for c in b["chars"])
        n = len(b["chars"])
        b["math_ratio"] = sum(c["math"] or c["cid"] for c in b["chars"]) / n
        b["text"] = "".join(c["t"] for c in sorted(b["chars"], key=lambda c: (round(c["base"]), c["x0"])))
    return out


def visible_rulings(pg, bbox, img=None):
    """Share of the table's ruling segments that are drawn dark on the rendered page (invisible rulings exist)."""
    x0, top, x1, bottom = bbox
    segs = [
        e
        for e in pg.edges
        if e["x0"] >= x0 - 2 and e["x1"] <= x1 + 2 and e["top"] >= top - 2 and e["bottom"] <= bottom + 2
    ]
    if not segs or img is None:
        return 0.0
    s = img.width / pg.width
    px = img.load()
    dark = total = 0
    for e in segs:
        n = max(2, int(max(e["x1"] - e["x0"], e["bottom"] - e["top"]) * s / 3))
        for i in range(n + 1):
            x = (e["x0"] + (e["x1"] - e["x0"]) * i / n) * s
            y = (e["top"] + (e["bottom"] - e["top"]) * i / n) * s
            best = 255
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    xx, yy = min(img.width - 1, max(0, int(x) + dx)), min(img.height - 1, max(0, int(y) + dy))
                    best = min(best, px[xx, yy])
            total += 1
            dark += best < 160
    return dark / max(1, total)


def regions(page, bnds, img=None, force=()):
    """Formula regions (bands with >= 30 % math or unmapped glyphs, merged when closer than one body line) and
    ruled tables with visible rulings; each grown to whole bands. Returns sorted list of dicts."""
    body = page["body"]
    pg = page["_pg"]
    cand = []
    for t in pg.find_tables():
        vis = visible_rulings(pg, t.bbox, img)
        if vis >= 0.6:
            cand.append(
                {"kind": "table", "bbox": list(t.bbox), "rulings_visible": round(vis, 2), "pdf_grid": t.extract()}
            )
    for bi, b in enumerate(bnds):
        if b["math_ratio"] >= 0.3 or bi in force:
            if any(
                r["bbox"][1] - 1 <= b["top"] and b["bottom"] <= r["bbox"][3] + 1 for r in cand if r["kind"] == "table"
            ):
                continue
            if cand and cand[-1]["kind"] == "formula" and b["top"] - cand[-1]["bbox"][3] < 1.0 * body:
                cand[-1]["bbox"][3] = max(cand[-1]["bbox"][3], b["bottom"])
                cand[-1]["bbox"][0] = min(cand[-1]["bbox"][0], b["x0"])
                cand[-1]["bbox"][2] = max(cand[-1]["bbox"][2], b["x1"])
            else:
                cand.append({"kind": "formula", "bbox": [b["x0"], b["top"], b["x1"], b["bottom"]]})
    if not cand:
        return []
    # Ink blocks: runs of rendered rows (over the text column) that carry any dark pixel. Char boxes come from the
    # font size, but big operators, radicals and rules ink outside them, so cuts are placed in blank pixel rows.
    col0 = min(b["x0"] for b in bnds)
    col1 = max(b["x1"] for b in bnds)
    s = img.width / page["w"]
    a = np.asarray(img)
    dark = (a[:, max(0, int((col0 - 2) * s)) : int((col1 + 2) * s) + 1] < 160).any(axis=1)
    # a gap of fewer than 3 blank rows (150 DPI) is not a safe cut: close it
    gap = 0
    for yy in range(len(dark)):
        if dark[yy]:
            if 0 < gap < 3 and yy - gap - 1 >= 0 and dark[yy - gap - 1]:
                dark[yy - gap : yy] = True
            gap = 0
        else:
            gap += 1
    blocks, y = [], 0
    while y < len(dark):
        if dark[y]:
            y0 = y
            while y < len(dark) and dark[y]:
                y += 1
            blocks.append([y0, y])
        y += 1
    row_block = np.full(len(dark), -1)
    for k, (y0, y1) in enumerate(blocks):
        row_block[y0:y1] = k
    owner = [set() for _ in blocks]
    for i, b in enumerate(bnds):
        for c in b["chars"]:
            # glyph middle from the baseline: some producers' char boxes are offset from the ink (1336)
            k = row_block[min(len(dark) - 1, max(0, int((c["base"] - 0.3 * c["size"]) * s)))]
            if k >= 0:
                owner[k].add(i)
    owned = [k for k in range(len(blocks)) if owner[k]]
    attach = {}  # char-less block (rule, vinculum, radical top) -> nearest block with chars
    for k in range(len(blocks)):
        if not owner[k] and owned:
            attach[k] = min(owned, key=lambda j, k=k: max(blocks[j][0] - blocks[k][1], blocks[k][0] - blocks[j][1]))

    def blocks_of(bandset):
        ks = {k for k in range(len(blocks)) if owner[k] & bandset}
        return ks | {k for k, j in attach.items() if j in ks}

    for r in cand:
        r["bands"] = {
            i for i, b in enumerate(bnds) if b["bottom"] > r["bbox"][1] + 0.5 and b["top"] < r["bbox"][3] - 0.5
        }
    changed = True
    while changed:
        changed = False
        for r in cand:
            grown = set(r["bands"])
            for k in blocks_of(r["bands"]):
                grown |= owner[k]
            if grown != r["bands"]:
                r["bands"], changed = grown, True
        cand.sort(key=lambda r: min(r["bands"]) if r["bands"] else 0)
        merged = []
        for r in cand:
            if merged and (r["bands"] & merged[-1]["bands"] or min(r["bands"]) <= max(merged[-1]["bands"])):
                m = merged[-1]
                m["bands"] |= r["bands"]
                if r["kind"] == "table":
                    m["kind"] = "table"
                    m["pdf_grid"] = r.get("pdf_grid")
                    m["rulings_visible"] = r.get("rulings_visible")
                changed = True
            else:
                merged.append(r)
        cand = merged
    for r in cand:
        bs = [bnds[i] for i in sorted(r["bands"])]
        r["bbox"] = [
            min(b["x0"] for b in bs),
            min(b["top"] for b in bs),
            max(b["x1"] for b in bs),
            max(b["bottom"] for b in bs),
        ]
        ks = sorted(blocks_of(r["bands"]))
        # Kept as measured on 2026-11-15: this parses as `[ink edge] if ks else ([] + [char-box edge])`, so a region
        # with ink takes the ink edge alone (the merge regression depends on it).
        top_px = min([blocks[ks[0]][0]] if ks else [] + [int(r["bbox"][1] * s)])
        bot_px = max([blocks[ks[-1]][1]] if ks else [] + [int(r["bbox"][3] * s) + 1])
        prev_end = max((y1 for y0, y1 in blocks if y1 <= top_px), default=None)
        next_start = min((y0 for y0, y1 in blocks if y0 >= bot_px), default=None)
        cut_top = (prev_end + top_px) / 2 if prev_end is not None else max(0, top_px - 6)
        cut_bot = (bot_px + next_start) / 2 if next_start is not None else min(len(dark), bot_px + 6)
        r["cut"] = [min(col0, r["bbox"][0]) - 4, cut_top / s, max(col1, r["bbox"][2]) + 4, cut_bot / s]
        r["snap"] = "ink" if ks else "char-box fallback"
        # context variant: one more ink block above and below (a second recognition that sees whole neighbour lines)
        above = [b for b in blocks if b[1] <= top_px]
        below = [b for b in blocks if b[0] >= bot_px]
        ctop = cut_top
        if above:
            pe = (
                max(y1 for y0, y1 in blocks if y1 <= above[-1][0])
                if any(y1 <= above[-1][0] for y0, y1 in blocks)
                else None
            )
            ctop = (pe + above[-1][0]) / 2 if pe is not None else max(0, above[-1][0] - 6)
        cbot = cut_bot
        if below:
            ns = (
                min(y0 for y0, y1 in blocks if y0 >= below[0][1])
                if any(y0 >= below[0][1] for y0, y1 in blocks)
                else None
            )
            cbot = (below[0][1] + ns) / 2 if ns is not None else min(len(dark), below[0][1] + 6)
        r["cut_ctx"] = [r["cut"][0], ctop / s, r["cut"][2], cbot / s]
        r["band_ids"] = sorted(r.pop("bands"))
        inside = [c for i in r["band_ids"] for c in bnds[i]["chars"]]
        r["n_chars"] = sum(1 for c in inside if c["t"].strip())
        r["n_cid"] = sum(c["cid"] for c in inside)
        r["n_cid_unrecovered"] = sum(c["cid"] and not c["recovered"] for c in inside)
    return cand
