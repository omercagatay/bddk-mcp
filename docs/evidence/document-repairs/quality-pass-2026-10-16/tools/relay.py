"""Rebuild a BDDK document from its PDF text layer (pdftotext -layout) with reviewed table blocks.

usage: relay.py SP DOC_ID [--write]
spec:  SP/specs/{id}.json  {"tables": [{"start": "<line prefix>", "end": "<line prefix>|null", "md": "..."}],
                            "drop": ["<exact stripped line>", ...], "indent": 3, "starts": ["<line prefix>"],
                            "conts": ["<line prefix>"]}
- characters come from the text layer (Đ etc. mapped); runs of spaces collapse to one;
- a line indented >= body margin + `indent` starts a paragraph, a flush line continues it (also across pages);
  `starts`/`conts` override the rule for specific lines;
- a bare page number on a page's last line is dropped, so are `drop` lines (running headers);
- each table block replaces the layout lines from `start` up to (not including) `end`;
  each `regions` entry {"page", "top", "bottom", "md"} replaces the lines printed in that page area (points from the
  page top, as pdftotext -bbox and pdfplumber report them).
Prints a token check against the stored text. --write saves SP/rebuild/{id}.md
"""

import html
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

SP = Path(sys.argv[1])
DID = sys.argv[2]
DEBUG = next((a[8:] for a in sys.argv if a.startswith("--debug=")), "")
PMAP = {"Đ": "İ", "ﬁ": "fi", "ﬂ": "fl", ";": ";", "": "•", "": "▪", "": "➢", "": "✓", "": "→", "": "—", "­": ""}
spec_path = SP / "specs" / f"{DID}.json"
spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
IND = spec.get("indent", 3)
lay = (SP / "lay" / f"{DID}.txt").read_text()
for k, v in PMAP.items():
    lay = lay.replace(k, v)
pages = lay.split("\f")
if pages and not pages[-1].strip():
    pages = pages[:-1]


bbox_html = subprocess.run(
    ["pdftotext", "-bbox-layout", str(SP / "pdf" / f"{DID}.bin"), "-"], capture_output=True, text=True
).stdout
for k, v in PMAP.items():
    bbox_html = bbox_html.replace(k, v)
WORD_RE = re.compile(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>')
word_y = []  # per page: word text -> yMin of its occurrences (used only to place `regions`)
for ph in bbox_html.split("<page ")[1:]:
    wy = defaultdict(list)
    for w in WORD_RE.findall(ph):
        wy[html.unescape(w[4])].append((float(w[1]), float(w[3])))
    word_y.append(wy)
page_lines = []  # per page: normalized line text -> list of xMax (physical right edge, points)
page_right = []
page_gap = []  # median baseline-to-baseline distance of consecutive lines
for ph in bbox_html.split("<page ")[1:]:
    d = defaultdict(list)
    spans = []
    for ln in re.findall(r"<line[^>]*>(.*?)</line>", ph, re.S):
        ws = WORD_RE.findall(ln)
        if not ws:
            continue
        txt = " ".join(html.unescape(w[4]) for w in ws)
        x0, x1 = float(ws[0][0]), float(ws[-1][2])
        y0, y1 = min(float(w[1]) for w in ws), max(float(w[3]) for w in ws)
        d[re.sub(r"\s+", " ", txt).strip()].append((x1, y0, y1))
        spans.append((x0, x1))
    page_lines.append(d)
    ys = sorted({round(v[1], 1) for vs in d.values() for v in vs})
    gaps = sorted(b - a for a, b in zip(ys, ys[1:], strict=False) if 0 < b - a < 40)
    page_gap.append(gaps[len(gaps) // 2] if gaps else 0.0)
    xs = sorted(x1 for x0, x1 in spans)
    page_right.append(xs[len(xs) * 9 // 10] if xs else 0.0)

# BDDK letterhead footer printed at the bottom of every page (page furniture, like page numbers)
FOOTER = re.compile(
    r"^(Atatürk Bulvarı No:? ?191 Kavaklıdere 06680 ANKARA.*|Tel\.?: \(312\) 4\d\d \d\d \d\d Faks: \(312\) 4\d\d \d\d \d\d|İnternet adresi: www\.bddk\.org\.tr)$"
)
MARKER = re.compile(
    r"^(\{\d{1,3}\}\s|[a-zçğıöşü]\.\s|\(?[a-zçğıöşü]{1,3}\)|\(?[A-ZÇĞİÖŞÜ]\)(?=\s)|\(\d{1,3}\)|\d{1,3}(\.\d{1,3}){0,3}[.\-)]\s|[ivxIVX]+[.)]\s"
    r"|[•▪➢✓→]|- |MADDE|GEÇİCİ MADDE|EK MADDE|EK[- ]?\d)(?!\s*(bend|fıkra|numaralı|alt bend|sayılı|madde))"
)
_pm = []
for _pg in pages:
    _ind = sorted(len(ln) - len(ln.lstrip()) for ln in _pg.split("\n") if len(ln.strip()) > 50)
    if len(_ind) >= 5:
        _pm.append(_ind[0])
DOC_MARGIN = sorted(_pm)[len(_pm) // 2] if _pm else 0
dropped = []
items = []  # (kind, text, indent_flag)
last_short = True  # the previous line (also across pages) ended a paragraph: short and closed by punctuation
for page_no, pg in enumerate(pages):
    lines = pg.split("\n")
    # running headers/footers listed in the spec go first, so a page number above a footer is still the last line
    lines = ["" if re.sub(r"\s+", " ", ln).strip() in spec.get("drop", []) else ln for ln in lines]
    nz = [i for i, ln in enumerate(lines) if ln.strip()]
    for i in reversed(nz[-4:]):  # footer lines at the page bottom (page number may sit among them)
        t = re.sub(r"\s+", " ", lines[i]).strip()
        if FOOTER.match(t):
            dropped.append(t)
            lines[i] = ""
    nz = [i for i, ln in enumerate(lines) if ln.strip()]
    if nz and re.fullmatch(r"\s*(?:Sayfa\s*:?\s*)?-?\s*\d{1,3}\s*(?:/\s*\d{1,3}\s*)?-?\s*", lines[nz[-1]]):
        lines[nz[-1]] = ""
    body = [ln for ln in lines if ln.strip()]
    if not body:
        continue
    ind = sorted(len(ln) - len(ln.lstrip()) for ln in body if len(ln.strip()) > 50)
    # left edge of the text block: the least-indented long line; a page with few long lines uses the document's
    margin = ind[0] if len(ind) >= 5 else DOC_MARGIN
    pl = page_lines[page_no] if page_no < len(page_lines) else {}
    right = page_right[page_no] if page_no < len(page_right) else 0.0
    gap = page_gap[page_no] if page_no < len(page_gap) else 0.0
    prev_y = None
    region_y = 0.0  # last placed line on this page (for `regions` only)
    used = {}
    prev_upper = False
    prev_blank = False  # leading blank lines of a page do not break a paragraph
    prev_indent = margin
    seen_text = False
    force_cont = False
    hang = None
    first_on_page = True
    for ln in lines:
        if not ln.strip():
            prev_blank = seen_text
            continue
        seen_text = True
        s = re.sub(r"\s+", " ", ln).strip()
        s = re.sub(r"^(\d{2,3}\.)(?=[A-ZÇĞİÖŞÜ])", r"\1 ", s)
        if re.fullmatch(r"_{5,}", s):
            s = "_______"  # the printed footnote rule; its length is layout (as in the mevzuat rebuilds)  # "100.Senaryolar": the tab after a 3-digit number is lost
        if s in spec.get("drop", []):
            continue
        indent = len(ln) - len(ln.lstrip())
        marker = MARKER.match(s)
        prev_short = last_short
        xs = pl.get(s)
        k = used.get(s, 0)  # same text twice on a page: take its occurrences in order
        used[s] = k + 1
        here = xs[k] if xs and k < len(xs) else None
        if here is None:  # the bbox layout may split a printed line; fall back to a unique prefix match
            pre = [
                v
                for t_, vs in pl.items()
                if len(s) > 12 and (t_.startswith(s[:25]) or s.startswith(t_[:25]))
                for v in vs
            ]
            here = pre[0] if len(pre) == 1 else None
        wide_gap = bool(here and prev_y is not None and gap and here[1] - prev_y > 1.45 * gap)
        dbg_prev_y = prev_y
        # a blank layout line is a paragraph break only if the page really leaves extra vertical space there
        blank_break = prev_blank and (here is None or prev_y is None or not gap or here[1] - prev_y > 1.2 * gap)
        start = (
            blank_break
            or bool(marker)
            or prev_short
            or wide_gap
            or (
                indent >= margin + IND and indent > prev_indent + 2 and prev_indent < margin + IND
            )  # indented first line
            or indent <= margin - IND
        )  # hanging (outdented) start
        if hang is not None and not prev_blank and not marker and abs(indent - hang) <= 2:
            start = False  # hanging indent: the line aligns with the text after the previous item's marker
        if start:
            hang = None
            if marker:
                rest = ln[indent + marker.end() :]
                hang = indent + marker.end() + (len(rest) - len(rest.lstrip()))
        letters = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü]", "", s)
        is_upper = len(letters) > 3 and letters == letters.upper()
        if is_upper and prev_upper and not prev_blank and not marker and not wide_gap:
            start = False  # a multi-line capitalised title
        prev_upper = is_upper and not re.search(r"[.:;]$", s)
        if first_on_page and not marker and not re.fullmatch(r"\d{1,3}", s) and s != s.upper():
            last = next((it[1] for it in reversed(items) if it[0] == "line"), "")
            if last and not re.search(r"[.:;!?)”\"]$", last):
                start = False  # a sentence running over the page break
        first_on_page = False
        if force_cont:
            start = False
        force_cont = bool(re.fullmatch(r"\d{1,3}", s))  # a footnote number on its own line: start, text follows
        if force_cont:
            start = True
        prev_indent = indent
        x_right = here[0] if here else right  # unknown/ambiguous: assume full width
        prev_y = here[1] if here else None
        last_short = x_right < right - 20 and bool(re.search(r"[.:;!?)”]$", s))
        if any(s.startswith(x) for x in spec.get("starts", [])):
            start = True
        if any(s.startswith(x) for x in spec.get("conts", [])):
            start = False
        y_line = (here[1] + here[2]) / 2 if here else None  # vertical centre of the printed line
        if y_line is None and spec.get("regions"):  # a layout line joining several table cells: use its first word
            w0 = s.split(" ")[0]
            wy_ = word_y[page_no] if page_no < len(word_y) else {}
            ws_ = (
                wy_.get(w0)
                or wy_.get(w0.rstrip("0123456789"))  # a superscript digit is its own word
                or wy_.get(re.sub(r"\(\d{1,2}\)$", "", w0), [])
            )  # so is a raised '(n)' marker
            after = [y for y in ws_ if y[0] >= region_y - 6]
            pick = min(after) if after else None
            if pick is not None:
                ws_.remove(pick)
            y_line = (pick[0] + pick[1]) / 2 if pick else None
        if y_line is not None:
            region_y = y_line
        region = next(
            (
                r
                for r in spec.get("regions", [])
                if r["page"] == page_no + 1 and y_line is not None and r["top"] <= y_line <= r["bottom"]
            ),
            None,
        )
        if region is not None:  # a reviewed table drawn in this page area replaces its layout lines
            if not region.get("_done"):
                region["_done"] = True
                items.append(("table", region["md"], True))
            last_short = True
            continue
        m = re.fullmatch(r"(Karar Sayısı ?: ?\S+) (Karar Tarihi ?: ?\S+)", s)
        if m:  # the decision header prints number and date on one line, left and right
            items += [("line", m.group(1), True), ("line", m.group(2), True)]
        else:
            if DEBUG and DEBUG in s:
                print(
                    "DBG",
                    repr(s[:40]),
                    dict(
                        prev_blank=prev_blank,
                        marker=bool(marker),
                        prev_short=prev_short,
                        indent=indent,
                        margin=margin,
                        hang=hang,
                        start=start,
                        wide_gap=wide_gap,
                        gap=gap,
                        here=here,
                        py=dbg_prev_y,
                    ),
                    file=sys.stderr,
                )
            items.append(("line", s, start))
        prev_blank = False
    items.append(("page", "", False))

# table blocks
out_items = []
i = 0
tables = list(spec.get("tables", []))
used = set()
seen = {}
while i < len(items):
    kind, s, st = items[i]
    hit = None
    for n, t in enumerate(tables):
        if n not in used and kind == "line" and s.startswith(t["start"]):
            seen[n] = seen.get(n, 0) + 1
            if seen[n] < t.get("nth", 1):
                continue
            hit = n
            break
    if hit is None:
        out_items.append(items[i])
        i += 1
        continue
    used.add(hit)
    t = tables[hit]
    j = i
    while j < len(items) and not (t["end"] and items[j][0] == "line" and items[j][1].startswith(t["end"]) and j > i):
        j += 1
    out_items.append(("table", t["md"], True))
    i = j
missing = [t["start"] for n, t in enumerate(tables) if n not in used]
missing += [f"region p{r['page']} y{r['top']:.0f}" for r in spec.get("regions", []) if not r.get("_done")]

paras, cur = [], []
for kind, s, st in out_items:
    if kind == "page":
        continue
    if kind == "table":
        if cur:
            paras.append("\n".join(cur))
            cur = []
        if s:  # an empty block removes its lines (e.g. a table body already written with its header)
            paras.append(s)
        continue
    if st and cur:
        paras.append("\n".join(cur))
        cur = []
    cur.append(s)
if cur:
    paras.append("\n".join(cur))
merged = []
for p_ in paras:  # a footnote number printed on its own line belongs to the footnote text after it
    if merged and re.fullmatch(r"\d{1,3}", merged[-1]) and not p_.startswith("|"):
        merged[-1] = merged[-1] + " " + p_
    else:
        merged.append(p_)
paras = [re.sub(r"^(\d{1,3})\n", r"\1 ", p_) for p_ in merged]
doc = "\n\n".join(paras)
for old_s, new_s, count in spec.get("replace", []):
    assert doc.count(old_s) == count, (old_s, doc.count(old_s))
    doc = doc.replace(old_s, new_s)

W = json.loads((SP / "work" / "documents.json").read_text())
old = next(d for d in W if d["document_id"] == DID)["markdown_content"]


def T(x):
    return Counter(w.lower() for w in re.findall(r"(?<![<\\\w])\w+", x))


a, b = T(old), T(doc)
print(f"{DID}: paras={len(paras)} tables={len(used)}/{len(tables)} missing_tables={missing}")
print("  stored-only:", sorted((a - b).elements())[:80])
print("  rebuilt-only:", sorted((b - a).elements())[:80])
if "--write" in sys.argv:
    (SP / "rebuild").mkdir(exist_ok=True)
    (SP / "rebuild" / f"{DID}.md").write_text(doc)
    (SP / "rebuild" / f"{DID}.dropped.json").write_text(json.dumps(dropped, ensure_ascii=False))
    print("  wrote", SP / "rebuild" / f"{DID}.md", len(doc))
