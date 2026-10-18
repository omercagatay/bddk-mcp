"""Use the relay rebuild as a layout oracle for stored break candidates: oracle.py SP DOC_ID OUT.json
Prints join/keep/unknown per candidate; JOIN = the rebuild prints A's end and B's start in one paragraph."""

import json
import re
import subprocess
import sys

SP, did, out = sys.argv[1], sys.argv[2], sys.argv[3]
r = open(f"{SP}/rebuild/{did}.md").read()
rn = re.sub(r"[ \t]+", " ", r)
res = subprocess.run(
    [sys.executable, f"{SP}/realbreak.py", f"{SP}/work/documents.json", did, "--show"], capture_output=True, text=True
).stdout
cands = []
for line in res.splitlines():
    m = re.match(r"(\S+): …(['\"])(.*)\2 ⏎⏎ (['\"])(.*)\4$", line)
    if m:
        cands.append([m.group(1), m.group(3).replace("\\n", "\n"), m.group(5)])
verdict = {"join": [], "keep": [], "unknown": []}
for c in cands:
    a = " ".join(c[1].split())[-25:]
    b = " ".join(c[2].split())[:20]
    pat = re.escape(a).replace("\\ ", r"\s+") + r"(\s+)" + re.escape(b).replace("\\ ", r"\s+")
    ms = list(re.finditer(pat, rn))
    if len(ms) != 1:
        verdict["unknown"].append(c)
    elif "\n\n" in ms[0].group(1):
        verdict["keep"].append(c)
    else:
        verdict["join"].append(c)
json.dump(verdict, open(out, "w"), ensure_ascii=False, indent=0)
print(did, {k: len(v) for k, v in verdict.items()})
