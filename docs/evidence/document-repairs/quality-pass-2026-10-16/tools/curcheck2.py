"""Second stage of curcheck: a word seen on one side only counts as a difference only if it does not occur in the
other side's whitespace-free letter stream (the GeneratePdf text layer glues words). usage: curcheck2.py SP"""

import json
import re
import subprocess
import sys
from pathlib import Path

SP = Path(sys.argv[1])
rep = json.loads((SP / "curcheck.json").read_text())
docs = {d["document_id"]: d for d in json.loads(Path("seed_data/documents.json").read_text())}


def letters(t):
    t = re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$", " ", t, flags=re.S).replace("İ", "i").replace("I", "ı").lower()
    return re.sub(r"[^a-zçğıöşüâîû]", "", t)


out = {}
for did, r in rep.items():
    if r["status"] != "compared":
        continue
    pdf = subprocess.run(
        ["pdftotext", "-raw", str(SP / "cur" / f"{did}.bin"), "-"], capture_output=True, text=True
    ).stdout
    ld, lp = letters(docs[did]["markdown_content"]), letters(pdf)
    miss_pdf = {w: n for w, n in r["only_pdf"].items() if w not in ld}  # printed today, absent from the corpus text
    miss_doc = {w: n for w, n in r["only_doc"].items() if w not in lp}  # in the corpus text, absent from today's PDF
    if miss_pdf or miss_doc:
        out[did] = {"pdf_only": miss_pdf, "doc_only": miss_doc}
        print(did, "| today's PDF only:", list(miss_pdf.items())[:15], "| corpus only:", list(miss_doc.items())[:15])
print("documents with real word differences:", len(out))
(SP / "curcheck2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
