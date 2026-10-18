"""Gate a relay rebuild against the stored text before applying it.

usage: gate.py SP [ids...]   (default: every BDDK doc not excluded)
Writes SP/gate.json: per doc {status, reasons, suspicious:[...], stored_only, rebuilt_only, sections_equal}
status: pass (apply), review (suspicious spots to judge), fail (do not apply)
"""

import json
import re
import subprocess
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from bddk_mcp.store.section_index import extract_document_sections  # noqa: E402

SP = Path(sys.argv[1])
PY = sys.executable
EXCLUDE = {"1334", "1043", "1045", "1305", "1313", "1314", "903", "905", "907", "943", "1296"}
TOK = re.compile(r"(?<![<\\\w])\w+")
args = [a for a in sys.argv[2:] if not a.startswith("--")]
BASE = next((a[11:] for a in sys.argv if a.startswith("--baseline=")), str(SP / "work" / "documents.json"))
docs = {d["document_id"]: d for d in json.loads(Path(BASE).read_text())}
done = set(json.loads((SP / "methods.json").read_text()))
if "--list" in " ".join(sys.argv):
    args = (SP / "apply_list.txt").read_text().split()
ids = args or sorted(p.stem for p in (SP / "pdf").glob("*.bin"))
ids = [i for i in ids if i not in EXCLUDE and (args or i not in done)]


def real_tables(did):
    n = []
    with pdfplumber.open(SP / "pdf" / f"{did}.bin") as pdf:
        for pno, page in enumerate(pdf.pages, 1):
            for t in page.find_tables():
                rows = t.extract()
                cols = max(sum(1 for c in r if c not in (None, "")) for r in rows) if rows else 0
                if len(rows) >= 2 and cols >= 2:
                    n.append(pno)
    return n


def toks(s):
    return [w.lower() for w in TOK.findall(s.replace("Đ", "İ"))]


def secs(did, text):
    return [(s.section_type, s.section_ref) for s in extract_document_sections(did, text) if s.section_type != "govde"]


def breaks(text):
    """token index -> True where a paragraph break precedes that token."""
    out = set()
    tl = list(TOK.finditer(text))
    starts = [m.start() for m in tl]
    for m in re.finditer(r"\n[ \t]*\n", text):
        # index of first token after this break
        lo, hi = 0, len(starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if starts[mid] < m.end():
                lo = mid + 1
            else:
                hi = mid
        out.add(lo)
    return out, tl


res = {}
for did in ids:
    d = docs[did]
    s = d["markdown_content"]
    reasons = []
    if "$" in s or "|" in s:
        reasons.append("stored has LaTeX/table markup")
    r = subprocess.run([PY, str(SP / "relay.py"), str(SP), did, "--write"], capture_output=True, text=True)
    if r.returncode:
        res[did] = {"status": "fail", "reasons": ["relay error: " + r.stderr[-300:]]}
        continue
    new = (SP / "rebuild" / f"{did}.md").read_text()
    a, b = Counter(toks(s)), Counter(toks(new))
    so, ro = a - b, b - a
    npages = len((SP / "lay" / f"{did}.txt").read_text(errors="replace").split("\f"))
    dropped = json.loads((SP / "rebuild" / f"{did}.dropped.json").read_text())
    furniture = Counter(w for line in dropped for w in toks(line))
    so_np = so - furniture  # stored-only tokens not explained by dropped page furniture
    pagey = all(re.fullmatch(r"\d{1,3}", w) and int(w) <= npages for w in so_np.elements())
    if ro:
        reasons.append(f"rebuilt-only tokens {sorted(ro.elements())[:20]}")
    if so_np and not pagey:
        reasons.append(f"stored-only non-page tokens {sorted(so_np.elements())[:20]}")
    if sum(so_np.values()) > 2 * npages:
        reasons.append("too many stored-only page tokens")
    import unicodedata

    odd = sorted({hex(ord(ch)) for ch in new if unicodedata.category(ch) in ("Co", "Cc", "Cs") and ch not in "\n\t"})
    if odd:
        reasons.append(f"rebuilt-only private/control chars {odd}")

    def P(x):
        return Counter(ch for ch in x if not ch.isspace() and not ch.isalnum() and ch not in "|-_$")

    pd = (P(s) - P(new)) + (P(new) - P(s))
    if sum(pd.values()) > max(4, 2 * npages):
        reasons.append(f"punctuation differs {dict(pd)}")
    se = secs(did, s) == secs(did, new)
    if not se:
        reasons.append("section list differs")
    tabs = real_tables(did)
    if tabs:
        reasons.append(f"ruled tables on pages {sorted(set(tabs))}")
    # paragraph-break diff on the common token sequence (ignore stored-only page numbers)
    bs, ts = breaks(s)
    bn, tn = breaks(new)
    # drop stored tokens the rebuild removed (page numbers, footer lines) before comparing order
    rm = Counter(so)
    keep_s = []
    for i, m in enumerate(ts):
        w = m.group().lower()
        if rm.get(w):
            rm[w] -= 1
            continue
        keep_s.append(i)
    # map common tokens by order: build sequences of (lowercase token, break-before)
    seq_s = [(ts[i].group().lower(), i in bs) for i in keep_s]
    seq_n = [(m.group().lower(), i in bn) for i, m in enumerate(tn)]
    susp = []
    reordered = False
    if [w for w, _ in seq_s] == [w for w, _ in seq_n]:  # same order: compare paragraph breaks
        for k, ((w, sb), (_, nb)) in enumerate(zip(seq_s, seq_n, strict=False)):
            if sb == nb:
                continue
            ctx_i = ts[keep_s[k]].start()
            before = s[max(0, ctx_i - 70) : ctx_i].rstrip()
            after = s[ctx_i : ctx_i + 50]
            if sb and not nb:  # stored break removed by relay
                if re.search(r"[.:;!?)”\"]$", before) and w[:1].isalpha() and s[ctx_i : ctx_i + 1].isupper():
                    susp.append(("merged", before[-60:], after))
            else:  # relay added a break inside a stored paragraph
                susp.append(("split", before[-60:], after))
    else:
        # stored order differs from the rebuild (a scramble the rebuild fixes, or content the rebuild moved):
        # accept only if the rebuild's order matches the PDF content-stream order (pdftotext -raw), an
        # ordering independent of the layout analysis the rebuild used
        raw = (SP / "raw" / f"{did}.txt").read_text(errors="replace")
        rt = [w for w in toks(raw)]
        nt = [w for w in toks(new)]
        sm = SequenceMatcher(None, nt, rt, autojunk=False)
        da, db = Counter(), Counter()
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                da.update(nt[i1:i2])
                db.update(rt[j1:j2])
        moved = sum((da & db).values())
        if moved:
            reasons.append(f"rebuild order disagrees with content-stream order ({moved} moved tokens)")
        else:
            reordered = True
    status = (
        "fail"
        if any(x.startswith(("rebuilt-only", "stored-only", "section", "relay", "too many")) for x in reasons)
        else ("review" if reasons or susp else "pass")
    )
    res[did] = {
        "status": status,
        "reasons": reasons,
        "suspicious": susp[:40],
        "n_susp": len(susp),
        "reordered": reordered,
        "stored_only": sorted(so.elements())[:30],
        "pages": npages,
    }
    print(did, status, len(susp), "; ".join(reasons)[:200], flush=True)
prev = json.loads((SP / "gate.json").read_text()) if (SP / "gate.json").exists() else {}
prev.update(res)
(SP / "gate.json").write_text(json.dumps(prev, ensure_ascii=False, indent=1))
print(Counter(v["status"] for v in res.values()))
