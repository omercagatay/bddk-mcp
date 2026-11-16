"""Stage 3a: masked full-page images per region (pypdfium2 + PIL), the only data the GPU step sends.

<doc>_<id>_a.png  4B input: 400 DPI, longest side 2048, everything outside the region cut box white
<doc>_<id>_b.png  1B input: 200 DPI, longest side 1540, same mask
<doc>_<id>_c.png  4B second rendering: as a, but the kept region shifted by a non-patch-aligned offset
                  (all vision patches change, glyph pixels and resolution stay as on the model card)
<doc>_<id>_d.png  formulas only, 4B: as a, but the kept box grows by one whole ink line above and below
list_a.txt ... list_d.txt; edges.json: regions whose cut rows are not white at 400 DPI
"""

from __future__ import annotations

import json
from pathlib import Path

PROFILES = {"a": (400, 2048), "b": (200, 1540), "c": (400, 2048), "d": (400, 2048)}
SHIFT = (11, 37)  # px at 2048; 4B patch size 16 (merge 2): 11/37 are not multiples of either


def page_image(doc, pno, dpi, edge):
    from PIL import Image

    img = doc[pno - 1].render(scale=dpi / 72).to_pil().convert("RGB")
    img.thumbnail((edge, edge), Image.LANCZOS)
    return img


def masked(img, cut, page_w, page_h, shift=(0, 0)):
    from PIL import Image

    s = img.width / page_w
    box = [
        max(0, round(cut[0] * s)),
        max(0, round(cut[1] * s)),
        min(img.width, round(cut[2] * s)),
        min(img.height, round(cut[3] * s)),
    ]
    out = Image.new("RGB", img.size, "white")
    dx, dy = shift
    if box[1] + dy < 0 or box[3] + dy > img.height:
        dy = -dy
    out.paste(img.crop(box), (box[0] + dx, box[1] + dy))
    return out


def render(pdf_path, regs, outdir) -> dict[str, list[str]]:
    """Write the masked images of ``regs``; returns the image names (without .png) per variant."""
    import pypdfium2 as pdfium

    pdf_path, outdir = Path(pdf_path), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    lists = {k: [] for k in PROFILES}
    cache = {}
    edges = {}
    for r in regs:
        for k, (dpi, edge) in PROFILES.items():
            if k == "d" and r["kind"] != "formula":
                continue
            key = (r["page"], dpi, edge)
            if key not in cache:
                if len(cache) > 4:
                    cache.clear()
                cache[key] = page_image(doc, r["page"], dpi, edge)
            cut = r["cut_ctx"] if k == "d" else r["cut"]
            img = masked(cache[key], cut, r["page_w"], r["page_h"], SHIFT if k == "c" else (0, 0))
            if k == "a":
                full = cache[key].convert("L")
                sc = full.width / r["page_w"]
                x0, x1 = max(0, round(cut[0] * sc)), min(full.width, round(cut[2] * sc))
                for y in (round(cut[1] * sc), min(full.height - 1, round(cut[3] * sc))):
                    if min(full.getpixel((x, y)) for x in range(x0, x1)) < 160:
                        edges.setdefault(r["id"], []).append(y)
            name = f"{pdf_path.stem}_{r['id']}_{k}"
            img.save(outdir / f"{name}.png")
            lists[k].append(name)
    (outdir / "edges.json").write_text(json.dumps(edges), encoding="utf-8")
    for k, v in lists.items():
        (outdir / f"list_{k}.txt").write_text("\n".join(v) + "\n", encoding="utf-8")
    print(pdf_path.stem, len(regs), "regions rendered; cut rows with ink:", edges or "none")
    return lists
