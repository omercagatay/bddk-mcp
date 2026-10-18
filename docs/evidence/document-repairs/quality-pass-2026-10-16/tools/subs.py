"""Census of subscripts (smaller, lowered digits/letters right after a letter) in source PDFs: subs.py SP"""

import glob
import json
import os
import re
import sys
from collections import Counter

import pdfplumber

SP = sys.argv[1]
docs = {d["document_id"]: d["markdown_content"] for d in json.load(open("seed_data/documents.json"))}
out = {}
for f in sorted(glob.glob(f"{SP}/pdf/*.bin")):
    did = os.path.basename(f)[:-4]
    if did not in docs:
        continue
    c = Counter()
    try:
        with pdfplumber.open(f) as pdf:
            for pg in pdf.pages:
                ch = pg.chars
                for k in range(1, len(ch)):
                    a, b = ch[k - 1], ch[k]
                    if (
                        b["text"].isalnum()
                        and a["text"].isalpha()
                        and b["size"] < a["size"] - 0.4
                        and 1.0 <= b["top"] - a["top"] < a["size"] * 0.8
                        and b["x0"] - a["x1"] < 2
                    ):
                        j, w = k - 1, ""
                        while j >= 0 and ch[j]["text"].isalpha() and abs(ch[j]["size"] - a["size"]) < 0.5:
                            w = ch[j]["text"] + w
                            j -= 1
                        sub = ""
                        m = k
                        while m < len(ch) and ch[m]["text"].isalnum() and abs(ch[m]["size"] - b["size"]) < 0.5:
                            sub += ch[m]["text"]
                            m += 1
                        c[(w[-15:], sub)] += 1
    except Exception as e:
        print(did, "ERR", e)
        continue
    if c:
        t = re.sub(r"\$\$.*?\$\$", "", docs[did], flags=re.S)
        glued = {f"{w}{s}": n for (w, s), n in c.items() if len(w) >= 2 and f"{w}{s}" in t}
        out[did] = {"pdf": sum(c.values()), "glued_in_md": glued}
json.dump(out, open(f"{SP}/subs.json", "w"), ensure_ascii=False, indent=1)
for did, v in out.items():
    if v["glued_in_md"]:
        print(did, v["pdf"], dict(list(v["glued_in_md"].items())[:12]))
