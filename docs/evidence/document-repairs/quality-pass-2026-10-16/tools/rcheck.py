"""Check a rebuild against the stored text: rcheck.py SP DOC_ID  (token multiset, sections, engine label)"""

import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
logging.disable(logging.WARNING)
from bddk_mcp.quality.markdown_quality import assess_markdown_quality  # noqa: E402
from bddk_mcp.store.section_index import extract_document_sections  # noqa: E402

SP = Path(sys.argv[1])
DID = sys.argv[2]
old = next(d for d in json.loads((SP / "work" / "documents.json").read_text()) if d["document_id"] == DID)[
    "markdown_content"
]
new = (SP / "rebuild" / f"{DID}.md").read_text()
T = lambda s: Counter(w.lower() for w in re.findall(r"(?<![<\\\w])\w+", s))  # noqa: E731


def S(s):
    return Counter(
        f"{x.section_type} {x.section_ref}" for x in extract_document_sections(DID, s) if x.section_type != "govde"
    )  # noqa: E731


print(f"{DID}: stored-only {sorted((T(old) - T(new)).elements())[:40]}")
print(f"{' ' * len(DID)}  rebuilt-only {sorted((T(new) - T(old)).elements())[:40]}")
so, sn = S(old), S(new)
print(f"  sections lost {sorted((so - sn).elements())} gained {sorted((sn - so).elements())}")
q = assess_markdown_quality(new, document_id=DID)
print(
    f"  engine {q.label} {q.flags}  paras {old.count(chr(10) * 2)}->{new.count(chr(10) * 2)}  dbl-space {old.count('  ')}->{new.count('  ')}"
)
