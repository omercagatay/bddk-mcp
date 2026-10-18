"""Crop each break candidate (A end + B start) from its PDF and stack them 4 per montage.
usage: cropall.py SP CANDIDATES.json OUTPREFIX  (candidates: [[did, a_tail, b_head], ...])"""

import json
import re
import sys
from pathlib import Path

import pdfplumber
from PIL import Image, ImageDraw

SP = Path(sys.argv[1])
cands = json.loads(Path(sys.argv[2]).read_text())
out = sys.argv[3]
PMAP = {"Đ": "İ"}
crops = []
for k, (did, a_tail, b_head) in enumerate(cands):
    aw = re.findall(r"\S+", a_tail)[-3:]
    bw = re.findall(r"\S+", b_head)[:2]
    pdf = pdfplumber.open(SP / "pdf" / f"{did}.bin")
    found = None
    for pno, page in enumerate(pdf.pages, 1):
        ws = page.extract_words()
        texts = [w["text"].replace("Đ", "İ") for w in ws]
        for i in range(len(ws) - len(aw) - len(bw) + 1):
            if texts[i : i + len(aw)] == aw and texts[i + len(aw) : i + len(aw) + len(bw)] == bw:
                top, bottom = ws[i]["top"], ws[i + len(aw) + len(bw) - 1]["bottom"]
                found = (
                    pno,
                    page.crop((15, max(0, top - 45), page.width - 15, min(page.height, bottom + 25)))
                    .to_image(resolution=90)
                    .original,
                )
                break
        if found:
            break
    if not found:
        print(k, did, "NOT FOUND", aw, bw)
        continue
    im = found[1].convert("RGB")
    d = ImageDraw.Draw(im)
    d.text((2, 2), f"[{k}] {did} p{found[0]}", fill=(255, 0, 0))
    crops.append(im)
for m in range(0, len(crops), 4):
    group = crops[m : m + 4]
    w = max(i.width for i in group)
    h = sum(i.height + 6 for i in group)
    canvas = Image.new("RGB", (w, h), "white")
    y = 0
    for i in group:
        canvas.paste(i, (0, y))
        y += i.height + 6
    canvas.save(f"{out}{m // 4}.png")
print("crops", len(crops), "montages", (len(crops) + 3) // 4)
