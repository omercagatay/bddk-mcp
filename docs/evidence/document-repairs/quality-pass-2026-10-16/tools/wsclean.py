"""Whitespace-only cleanup of justified-text spacing: wsclean.py SP DOC_ID... [--write]

Outside fenced code, display LaTeX and table rows: leading indentation is removed (4+ spaces would render a
paragraph as a code block), runs of spaces inside a line become one space (inline $…$ LaTeX is left as written),
trailing spaces go. The non-whitespace characters and the paragraph structure must stay identical.
"""

import json
import re
import sys
from pathlib import Path

SP = Path(sys.argv[1])
IDS = [a for a in sys.argv[2:] if not a.startswith("--")]


def clean(text):
    out, fence, math = [], False, False
    for ln in text.split("\n"):
        s = ln.strip()
        if s.startswith("```"):
            fence = not fence
            out.append(ln)
            continue
        if fence:
            out.append(ln)
            continue
        if math:  # inside a multi-line $$ block, until the line that closes it
            out.append(ln)
            math = not s.endswith("$$")
            continue
        if s.startswith("$$"):
            math = not (len(s) > 2 and s.endswith("$$"))
            out.append(ln)
            continue
        if s.startswith("|"):
            out.append(ln)
            continue
        if s.count("$") % 2 == 0:  # collapse outside inline math; a line with an unpaired '$' is left alone
            s = "".join(p if p.startswith("$") else re.sub(r" {2,}", " ", p) for p in re.split(r"(\$[^$]*\$)", s))
        out.append(s)
    return "\n".join(out)


W = SP / "work" / "documents.json"
docs = json.loads(W.read_text())
for d in docs:
    if d["document_id"] not in IDS:
        continue
    old = d["markdown_content"]
    new = clean(old)
    split_xml = "    </Kalem>\n```\n\n```xml\n    .\n"  # 907: one XML example split by a page break into two fences
    merged = d["document_id"] == "907" and split_xml in new
    if merged:
        new = new.replace(split_xml, "    </Kalem>\n    .\n")
    assert re.sub(r"\s+", "", old).replace("``````xml", "") == re.sub(r"\s+", "", new), d["document_id"]
    assert len(re.split(r"\n\s*\n", old.strip())) - merged == len(re.split(r"\n\s*\n", new.strip()))
    lead = sum(1 for p in re.split(r"\n\s*\n", new) if re.match(r" {4,}\S", p))
    print(d["document_id"], len(old), "->", len(new), "indented paragraphs left (code)", lead)
    if "--write" in sys.argv:
        log = SP / "manual" / f"{d['document_id']}.json"
        prior = json.loads(log.read_text()) if log.exists() else []
        log.write_text(json.dumps(prior + [{"old": old, "new": new, "whitespace": True}], ensure_ascii=False, indent=1))
        d["markdown_content"] = new
if "--write" in sys.argv:
    W.write_text(json.dumps(docs, ensure_ascii=False) + "\n")
    print("written")
