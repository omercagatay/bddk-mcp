"""Short token differences (punctuation included) stored vs rebuild, outside tables/LaTeX: puncdiff.py STORED REBUILT"""

import difflib
import re
import sys


def toks(p):
    t = open(p).read()
    keep = [ln for ln in t.split("\n") if not ln.startswith("|") and "$" not in ln]
    return re.findall(r"\w+|[^\w\s]", " ".join(keep))


a, b = toks(sys.argv[1]), toks(sys.argv[2])
sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
for op, i1, i2, j1, j2 in sm.get_opcodes():
    if op != "equal" and (i2 - i1) <= 8 and (j2 - j1) <= 8:
        print(op, " ".join(a[max(0, i1 - 5) : i2 + 3]), " || ", " ".join(b[max(0, j1 - 5) : j2 + 3]))
