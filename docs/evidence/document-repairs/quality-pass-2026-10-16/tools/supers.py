"""Census of superscript words in BDDK PDFs (bbox height/baseline): supers.py SP [ids...]"""

import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

SP = Path(sys.argv[1])
ids = sys.argv[2:] or sorted(p.stem for p in (SP / "pdf").glob("*.bin"))
WORD = re.compile(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>')
res = {}
for did in ids:
    out = subprocess.run(
        ["pdftotext", "-bbox-layout", str(SP / "pdf" / f"{did}.bin"), "-"], capture_output=True, text=True
    ).stdout
    hits = []
    for page_no, page in enumerate(out.split("<page ")[1:], 1):
        for line in re.findall(r"<line[^>]*>(.*?)</line>", page, re.S):
            ws = [(float(a), float(b), float(c), float(d), t) for a, b, c, d, t in WORD.findall(line)]
            if len(ws) < 2:
                continue
            hs = [d - b for a, b, c, d, t in ws]
            med = statistics.median(hs)
            base = statistics.median(d for a, b, c, d, t in ws)
            for k, (x0, y0, _x1, y1, t) in enumerate(ws):
                h = y1 - y0
                if re.fullmatch(r"[\d*,]{1,5}", t) and h < 0.78 * med and y1 < base - 0.15 * med and k > 0:
                    prev = ws[k - 1]
                    glued = x0 - prev[2] < 1.5
                    hits.append((page_no, prev[4], t, glued, ws[k + 1][4] if k + 1 < len(ws) else ""))
    res[did] = hits
json.dump(res, open(SP / "supers.json", "w"), ensure_ascii=False)
tot = sum(len(v) for v in res.values())
print("docs with superscripts", sum(1 for v in res.values() if v), "total", tot)
for did, v in res.items():
    if v:
        print(did, len(v), v[:4])
