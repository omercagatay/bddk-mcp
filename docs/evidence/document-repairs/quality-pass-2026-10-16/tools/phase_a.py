"""Phase A: deterministic glyph repairs on the work copy (each mapping source-verified)."""

import json
import sys
from pathlib import Path

W = Path(sys.argv[1]) / "work" / "documents.json"
MAP = {
    "\x96": "–",
    "\x92": "’",
    "\x93": "“",
    "\x94": "”",  # cp1254 punctuation decoded as C1 controls (html_parser)
    "Đ": "İ",  # PDF text-layer mapping error; glyph prints İ (verified 168 p1)
    ";": ";",  # GREEK QUESTION MARK, canonically equal to ';'
    "ﬁ": "fi",
    "ﬂ": "fl",  # ligature glyphs (1310)
}
docs = json.loads(W.read_text())
log = {}
for d in docs:
    t = d["markdown_content"]
    counts = {k: t.count(k) for k in MAP if k in t}
    if counts:
        for k, v in MAP.items():
            t = t.replace(k, v)
        d["markdown_content"] = t
        log[d["document_id"]] = {repr(k): n for k, n in counts.items()}
W.write_text(json.dumps(docs, ensure_ascii=False) + "\n")
json.dump(log, open(Path(sys.argv[1]) / "phase_a_log.json", "w"), ensure_ascii=False, indent=1)
print(len(log), "docs changed")
