"""Classify word-diff ops stored vs rebuild: wclass.py SP DOC_ID [--ws] -> whitespace-only ops (count) + real ops"""

import difflib
import json
import re
import sys
from pathlib import Path

SP = Path(sys.argv[1])
DID = sys.argv[2]
SHOW_WS = "--ws" in sys.argv
old = next(d for d in json.loads((SP / "work" / "documents.json").read_text()) if d["document_id"] == DID)[
    "markdown_content"
]
new = (SP / "rebuild" / f"{DID}.md").read_text()


def norm(w):
    return w.strip("*_#")


A = [norm(w) for w in re.findall(r"\S+", old)]
B = [norm(w) for w in re.findall(r"\S+", new)]
A2 = [w for w in A if w]
B2 = [w for w in B if w]
sm = difflib.SequenceMatcher(None, A2, B2, autojunk=False)
ws, real = [], []
for t, i1, i2, j1, j2 in sm.get_opcodes():
    if t == "equal":
        continue
    item = (" ".join(A2[max(0, i1 - 5) : i1]), " ".join(A2[i1:i2]), " ".join(B2[j1:j2]), " ".join(A2[i2 : i2 + 5]))
    (ws if "".join(A2[i1:i2]) == "".join(B2[j1:j2]) else real).append(item)
print(
    f"{DID}: whitespace-only ops {len(ws)} (stored has more tokens in {sum(1 for w in ws if len(w[1].split()) > len(w[2].split()))}); real ops {len(real)}"
)
for pre, a, b, post in ws if SHOW_WS else []:
    print(f"   WS …{pre[-40:]} ⟦{a}⟧ → ⟦{b}⟧ {post[:40]}…")
for pre, a, b, post in real:
    print(f"   RE …{pre[-50:]} ⟦{a[:300]}⟧ → ⟦{b[:300]}⟧ {post[:50]}…")
