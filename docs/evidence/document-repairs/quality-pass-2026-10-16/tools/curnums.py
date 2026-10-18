"""Numbers (rates, dates, RG issue numbers) on one side only: corpus text vs today's mevzuat.gov.tr PDF.
usage: curnums.py SP"""

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

SP = Path(sys.argv[1])
docs = {d["document_id"]: d for d in json.loads(Path("seed_data/documents.json").read_text())}


def nums(t):
    t = re.sub(r"\$\$.*?\$\$", " ", t, flags=re.S)
    t = re.sub(r"\$\^\{[^}]*\}\$|\$_\{[^}]*\}\$", " ", t)  # footnote markers / subscripts written as LaTeX
    return Counter(re.findall(r"\d+(?:[.,/]\d+)*", t))


for f in sorted((SP / "cur").glob("mevzuat_*.bin")):
    did = f.stem
    pdf = subprocess.run(["pdftotext", "-raw", str(f), "-"], capture_output=True, text=True).stdout
    pdf = re.sub(r"(?m)^\s*\d+\s*$", " ", pdf)  # page numbers on their own line
    a, b = nums(docs[did]["markdown_content"]), nums(pdf)
    od, op = a - b, b - a
    if od or op:
        print(f"{did}: corpus-only {dict(od.most_common(12))} | pdf-only {dict(op.most_common(12))}")
