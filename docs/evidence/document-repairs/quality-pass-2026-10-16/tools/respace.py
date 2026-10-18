"""Take word boundaries from the stored text where stored and rebuild differ only in spacing.

usage: respace.py SP DOC_ID [--keep=i,j] [--write]
A whitespace-only op has the same characters on both sides (ignoring *_# emphasis marks); the stored spacing is
used unless its index is in --keep. Every op is printed with its index for review.
"""

import difflib
import json
import re
import sys
from pathlib import Path

SP = Path(sys.argv[1])
DID = sys.argv[2]
KEEP = {int(x) for a in sys.argv if a.startswith("--keep=") for x in a[7:].split(",") if x}
old = next(d for d in json.loads((SP / "work" / "documents.json").read_text()) if d["document_id"] == DID)[
    "markdown_content"
]
path = SP / "rebuild" / f"{DID}.md"
new = path.read_text()


def norm(w):
    return w.strip("*_#")


A = [norm(m.group()) for m in re.finditer(r"\S+", old)]
A = [w for w in A if w]
BM = [m for m in re.finditer(r"\S+", new) if norm(m.group())]
B = [norm(m.group()) for m in BM]
sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
edits = []
k = 0
for t, i1, i2, j1, j2 in sm.get_opcodes():
    if t == "equal" or "".join(A[i1:i2]) != "".join(B[j1:j2]):
        continue
    use_stored = k not in KEEP
    print(
        f"[{k}] {'STORED' if use_stored else 'keep  '} ⟦{' '.join(A[i1:i2])}⟧ vs rebuilt ⟦{' '.join(B[j1:j2])}⟧ "
        f"…{' '.join(A[i2 : i2 + 4])}"
    )
    if use_stored:
        edits.append((BM[j1].start(), BM[j2 - 1].end(), " ".join(A[i1:i2])))
    k += 1
for a, b, rep in reversed(edits):
    new = new[:a] + rep + new[b:]
print(f"{DID}: {len(edits)} respaced")
if "--write" in sys.argv:
    path.write_text(new)
