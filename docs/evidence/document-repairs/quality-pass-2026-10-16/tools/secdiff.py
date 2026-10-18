"""Corpus-wide section diff for a candidate heading regex: secdiff.py SP"""

import json
import logging
import re
import sys

sys.path.insert(0, ".")
logging.disable(logging.WARNING)
import bddk_mcp.store.section_index as si  # noqa: E402

SP = sys.argv[1]
docs = json.load(open(f"{SP}/work/documents.json"))


def run():
    return {
        d["document_id"]: [
            (s.section_type, s.section_ref, s.start_char, s.end_char)
            for s in si.extract_document_sections(d["document_id"], d["markdown_content"])
        ]
        for d in docs
    }


CAND = re.compile(
    r"^(?P<prefix>\s*(?:#{1,6}\s+)?(?:\*{2,3}\s*)?)"
    r"(?:(?P<gecici>geçici\s+madde)\s+(?P<gecici_ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?)"
    r"|(?P<madde>madde)\s+(?P<madde_ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?)"
    r"|(?P<ilke>ilke)\s+(?P<ilke_ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?)"
    r"|(?P<paragraf>paragraf)\s+(?P<paragraf_ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?)"
    r"|(?P<ek>ek)[-\s–—\x96\x97]*(?P<ek_ref>\d+[A-Za-zÇĞİÖŞÜçğıöşü]?))"
    r"(?:\$\^\{[^}]*\}\$)*"
    r"(?:\s*\*{1,3})?"
    r"\s*(?:[-:–—\x96\x97]\s*(?P<title>.*?))?"
    r"\s*\*{0,3}\s*$",
    re.IGNORECASE,
)
base = run()
orig = si._HEADING_RE
si._HEADING_RE = CAND
new = run()
si._HEADING_RE = orig
changed = sorted(k for k in base if base[k] != new[k])
print("docs changed:", len(changed))
for k in changed:
    b, n = set(base[k]), set(new[k])
    add = [(a, r, e - s) for a, r, s, e in new[k] if (a, r, s, e) not in b and a != "govde"]
    rem = [(a, r, e - s) for a, r, s, e in base[k] if (a, r, s, e) not in n and a != "govde"]
    print(k, "| new:", add[:10], "| old:", rem[:10])
