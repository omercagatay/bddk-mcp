"""Stage 4: automatic acceptance signals per region. Each guard returns (ok, detail); a region is accepted only if
every guard of the policy passes (``pipeline.accept``).

G0 trunc     recognition ended with end-of-sequence, output not empty, no unrecovered glyph in the region
G1 words     every text-layer word/number of the region's prose lines is present; no word outside LaTeX that the
             region's text layer lacks (the bake-off guard)
G2 rows      tables: every text-layer line with >= 2 numbers is one output row with those numbers in order + label
G3 glyphs    multiset of glyphs the output renders == multiset of the region's text-layer glyphs (unmapped glyphs
             recovered); any LaTeX command outside the whitelist rejects
G4 scripts   glyphs set small and raised/lowered in the PDF == glyphs inside ^{}/_{} (sup and sub separately)
G5 order     each PDF baseline row, left to right, is a subsequence of the output glyph stream
G6 rules     formulas: thin horizontal rules in the region (fraction bars, radical vinculums) == count of
             \\frac + \\sqrt + \\overline (+ \\underline); the overline-extender groups the reference counts
             (``extenders``) are recorded but not compared
G7 grid      tables: a vertically merged cell with text in pdfplumber's ruled grid (an independent reader) must
             not come back split into the text plus empty cells below it
G8 agree     a second recognition (other model / other rendering) renders the same canonical glyph stream
"""

from __future__ import annotations

import re
from collections import Counter

from .latexglyphs import norm_glyphs, parse
from .layout import nfkc_math
from .mdparse import cell_norm, delatex, is_num, model_table_rows, model_tables, split_stream, strip_model, tokens

# ---------------------------------------------------------------- reference from the text layer
BIG_OPS = set("∑∏∫⋃⋂")


def classify_scripts(chars, body, rules=()):
    """sup/sub per char: set smaller than a nearby larger char to its left (its parent) and raised/lowered against
    that parent's baseline. Parents are searched only inside the same fraction part (chars over or under the same
    fraction rule), so numerators set in a smaller size are not mistaken for superscripts. A script of a script
    inherits the outer script's position, as LaTeX nesting does."""

    def ctx(c):
        best, dist = None, None
        cx = (c["x0"] + c["x1"]) / 2
        for k, r in enumerate(rules):
            if not (r["x0"] - 1 <= cx <= r["x1"] + 1) or k in vinculums:
                continue
            # baselines, not char boxes: some producers' font boxes are offset from the glyphs (1336)
            up = r["top"] - c["base"]
            down = c["base"] - r["bottom"]
            if -0.5 <= up <= 0.9 * body:
                d, side = up, "num"
            elif 0 <= down <= 1.5 * body:
                d, side = down, "den"
            else:
                continue
            if dist is None or d < dist:
                best, dist = (k, side), d
        return best

    vis = [c for c in chars if c["t"].strip()]
    # a rule that starts where a radical sign ends is the radical's vinculum, not a fraction bar
    vinculums = {k for k, r in enumerate(rules) if any(c["t"] == "√" and abs(c["x1"] - r["x0"]) < 3 for c in vis)}
    for c in vis:
        c["ctx"] = ctx(c)
    for c in chars:
        c["pos"] = None
    for c in sorted(vis, key=lambda c: -c["size"]):
        cand = [
            n
            for n in vis
            if n["ctx"] == c["ctx"]
            and n["size"] > c["size"] * 1.05
            and n["x0"] < c["x0"]
            and abs(n["base"] - c["base"]) < 0.6 * n["size"]
        ]
        if not cand:
            # a script on a tall closing delimiter, e.g. (...)^{-1} around a fraction
            left = [n for n in vis if n["x0"] < c["x0"] and n["size"] > c["size"] * 1.05]
            near = max(left, key=lambda n: n["x1"]) if left else None
            if (
                near is not None
                and near["t"] in ")]}|"
                and c["x0"] - near["x1"] < 0.5 * near["size"]
                and c["base"] < near["base"] - 0.15 * near["size"]
            ):
                c["pos"] = "sup"
                continue
            # a raised small mark at the start of a line (footnote number before the footnote text)
            right = [
                n
                for n in vis
                if n["ctx"] == c["ctx"]
                and n["size"] > c["size"] * 1.05
                and n["x0"] > c["x0"]
                and abs(n["base"] - c["base"]) < 0.6 * n["size"]
            ]
            if right:
                nb = min(right, key=lambda n: n["x0"])
                if c["base"] < nb["base"] - 0.15 * nb["size"]:
                    c["pos"] = "sup"
                continue
            # limits under / over a big operator: centred on it, further from its baseline
            cx = (c["x0"] + c["x1"]) / 2
            ops = [
                n
                for n in vis
                if n is not c
                and n["t"] in BIG_OPS
                and n["ctx"] == c["ctx"]
                and c["size"] < 0.85 * n["size"]
                and 0.3 * n["size"] < abs(n["base"] - c["base"]) < 2 * n["size"]
                and abs((n["x0"] + n["x1"]) / 2 - cx) < 3 * n["size"]
            ]
            if ops:
                op = min(ops, key=lambda n: abs((n["x0"] + n["x1"]) / 2 - cx))
                c["pos"] = op["pos"] or ("sub" if c["base"] > op["base"] else "sup")
            continue
        par = max(cand, key=lambda n: n["x0"])
        if par["pos"]:
            c["pos"] = par["pos"]
        elif c["base"] < par["base"] - 0.15 * par["size"]:
            c["pos"] = "sup"
        elif c["base"] > par["base"] + 0.05 * par["size"]:
            c["pos"] = "sub"


def words_of(chars):
    """Split a band's chars into words: first into text lines by baseline (a band can hold several lines of a
    table cell), scripts kept with their line, then at spaces, gaps wider than 0.3 em and script changes.
    (Bug fix of 2026-11-15: the first version sorted a whole band by x and split at 0.15 em, which interleaved
    multi-line table cells and cut words at fallback-font letters.)"""
    vis = sorted(chars, key=lambda c: c["base"])
    lines = []
    for c in vis:
        tol = 0.4 * max(c["size"], 8)
        for ln in lines:
            if abs(ln[0] - c["base"]) <= tol or (c.get("pos") and abs(ln[0] - c["base"]) <= 0.8 * c["size"] + tol):
                ln[1].append(c)
                break
        else:
            lines.append([c["base"], [c]])
    out = []
    # producers that write space glyphs mark word ends themselves: then only a wide gap (> 0.6 em) splits, so
    # fallback-font letters inside a word (1336) do not; without space glyphs a 0.25 em gap is a word end
    gap = 0.6 if any(not c["t"].strip() for c in chars) else 0.25
    for _, lc in sorted(lines, key=lambda ln: ln[0]):
        cur, last = [], None
        for c in sorted(lc, key=lambda c: c["x0"]):
            if not c["t"].strip():
                if cur:
                    out.append(cur)
                cur, last = [], None
                continue
            if cur and (c["x0"] - last["x1"] > gap * c["size"] or (c.get("pos") or "") != (last.get("pos") or "")):
                out.append(cur)
                cur = []
            cur.append(c)
            last = c
        if cur:
            out.append(cur)
    return out


def region_ref(page, bnds, region, pg):
    spaces = [c for c in page["chars"] if not c["t"].strip()]

    def with_spaces(b):
        """the band's glyphs plus the space glyphs on its lines (bands are built from visible glyphs only)"""
        lo = min(c["base"] for c in b["chars"]) - 1
        hi = max(c["base"] for c in b["chars"]) + 1
        return b["chars"] + [c for c in spaces if lo <= c["base"] <= hi and b["x0"] - 2 <= c["x0"] <= b["x1"] + 2]

    ids = region["band_ids"]
    rb = [bnds[i] for i in ids]
    chars = [c for b in rb for c in b["chars"]]
    cut = region["cut"]
    rules = [
        o
        for o in pg.rects + pg.lines
        if o["height"] <= 1.5 and o["width"] >= 3 and o["top"] >= cut[1] and o["bottom"] <= cut[3]
    ]
    classify_scripts(chars, page["body"], rules if region["kind"] == "formula" else ())
    vis = [c for c in chars if c["t"].strip()]
    lo, hi = region["cut"][1], region["cut"][3]
    mathf = [c for c in page["chars"] if c["math"] and not c["t"].strip() and lo <= (c["top"] + c["bottom"]) / 2 <= hi]
    extenders = [c for c in mathf if c["x1"] - c["x0"] > 0.5]
    prose_words, all_words = [], []
    for b in rb:
        for w in words_of(with_spaces(b)):
            toks = tokens("".join(c["t"] for c in w))
            all_words.extend(toks)
            if b["math_ratio"] < 0.3 and not any(c["math"] or c["cid"] for c in w):
                prose_words.extend(toks)
    rows = {}
    for c in sorted(vis, key=lambda c: c["base"]):
        key = next((k for k in rows if abs(k - c["base"]) < 0.8), c["base"])
        rows.setdefault(key, []).append(c)
    row_glyphs = [
        [g for c in sorted(r, key=lambda c: c["x0"]) for g in norm_glyphs(c["t"])] for _, r in sorted(rows.items())
    ]
    groups = []  # overline extender glyphs (math-font glyphs mapped to a space) joined per baseline run
    for c in sorted(extenders, key=lambda c: (round(c["base"]), c["x0"])):
        if groups and abs(groups[-1][-1]["base"] - c["base"]) < 0.5 and c["x0"] - groups[-1][-1]["x1"] < 1.0:
            groups[-1].append(c)
        else:
            groups.append([c])
    num_lines = []
    for b in rb:
        toks = [t for w in words_of(with_spaces(b)) for t in tokens("".join(c["t"] for c in w))]
        nums = [t for t in toks if is_num(t)]
        if len(nums) >= 2:
            first = toks.index(nums[0])
            num_lines.append({"nums": nums, "label": toks[first - 1] if first > 0 else None, "text": " ".join(toks)})
    return {
        "kind": region["kind"],
        "glyphs": Counter(g for c in vis for g in norm_glyphs(c["t"])),
        "sup": Counter(g for c in vis if c["pos"] == "sup" for g in norm_glyphs(c["t"])),
        "sub": Counter(g for c in vis if c["pos"] == "sub" for g in norm_glyphs(c["t"])),
        "rows": [r for r in row_glyphs if r],
        "prose_words": Counter(prose_words),
        "all_words": Counter(all_words),
        "unrecovered": sum(c["cid"] and not c["recovered"] for c in chars),
        "rules": len(rules) if region["kind"] == "formula" else None,
        "extenders": len(groups),
        "num_lines": num_lines,
        "pdf_grid": region.get("pdf_grid"),
    }


# ---------------------------------------------------------------- output side
def canonicalise(md, ref):
    """Deterministic notation fixes applied before the guards (recorded): \\max for a text-layer 'maks'."""
    fixes = []
    if re.search(r"<su[bp]>", md, flags=re.I):
        md = re.sub(r"<sub>(.*?)</sub>", r"$_{\1}$", md, flags=re.S | re.I)
        md = re.sub(r"<sup>(.*?)</sup>", r"$^{\1}$", md, flags=re.S | re.I)
        fixes.append("html sub/sup -> latex scripts")
    flat = "".join(g for r in ref["rows"] for g in r)
    if "\\max" in md and "maks" in flat and "max" not in flat:
        md = md.replace("\\max", "\\operatorname{maks}")
        fixes.append("\\max -> maks")
    return md, fixes


def render(md):
    """Glyph stream of an output: [(glyph, script)], unknown commands, rule count, overline count."""
    out, unknown, rules, overlines = [], [], 0, 0
    for is_math, chunk in split_stream(md):
        if is_math:
            p = parse(chunk)
            out.extend(p.out)
            unknown.extend(p.unknown)
            rules += p.rules
            overlines += p.overlines
        else:
            out.extend((g, None) for g in norm_glyphs(chunk))
    return out, unknown, rules, overlines


def canonical(md):
    stream, unknown, rules, _ = render(md)
    return (
        "".join(g + ("₊" if s == "sup" else "₋" if s == "sub" else "") for g, s in stream),
        tuple(sorted(unknown)),
        rules,
    )


def table_rows(md):
    return [[cell_norm(c) for c in row] for t in model_tables(md) for row in t]


# ---------------------------------------------------------------- guards
def g0(ref, meta, md):
    if meta.get("truncated") or not meta.get("eos", True):
        return False, "truncated"
    if not md.strip():
        return False, "empty output"
    if ref["unrecovered"]:
        return False, f"{ref['unrecovered']} unrecovered glyphs in the text layer"
    return True, ""


def g1(ref, md):
    plain, math = strip_model(md)
    have = Counter(tokens(plain)) + Counter(tokens(" ".join(delatex(s) for s in math)))
    miss = ref["prose_words"] - have
    extra = Counter(tokens(plain)) - ref["all_words"]
    ok = not miss and not extra
    return ok, "" if ok else f"missing {dict(miss)} extra {dict(extra)}"


def g2(ref, md):
    if ref["kind"] != "table" or not ref["num_lines"]:
        return True, "n/a"
    rows = [" ".join(r).split() for r in table_rows(md)]
    fails = []
    for ln in ref["num_lines"]:
        ok = False
        for rt in rows:
            it = iter(rt)
            if all(n in it for n in ln["nums"]) and (ln["label"] is None or ln["label"] in rt):
                ok = True
                break
        if not ok:
            fails.append(ln["text"])
    return not fails, "" if not fails else f"{len(fails)} rows: {fails[:2]}"


def g3(ref, stream, unknown):
    if unknown:
        return False, f"unknown commands {sorted(set(unknown))}"
    out = Counter(g for g, _ in stream)
    miss, extra = ref["glyphs"] - out, out - ref["glyphs"]
    ok = not miss and not extra
    return ok, "" if ok else f"missing {''.join(sorted(miss.elements()))!r} extra {''.join(sorted(extra.elements()))!r}"


def g4(ref, stream):
    sup = Counter(g for g, s in stream if s == "sup")
    sub = Counter(g for g, s in stream if s == "sub")
    bad = []
    for name, a, b in (("sup", ref["sup"], sup), ("sub", ref["sub"], sub)):
        if a != b:
            bad.append(
                f"{name} pdf-only {''.join(sorted((a - b).elements()))!r} out-only {''.join(sorted((b - a).elements()))!r}"
            )
    return not bad, "; ".join(bad)


def g5(ref, stream):
    seq = [g for g, _ in stream]
    bad = []
    for row in ref["rows"]:
        it = iter(seq)
        if not all(g in it for g in row):
            bad.append("".join(row)[:60])
    return not bad, "" if not bad else f"{len(bad)} rows out of order: {bad[:2]}"


def g6(ref, rules, overlines):
    if ref["kind"] != "formula":
        return True, "n/a"
    msgs = []
    if ref["rules"] != rules:
        msgs.append(f"pdf rules {ref['rules']} vs latex frac/sqrt/overline {rules}")
    return not msgs, "; ".join(msgs)


def _pdf_cells(grid):
    """pdfplumber grid (None = covered by a merged cell) -> {(r, c): (rowspan, colspan, text)} for cell origins."""
    out = {}
    nr = len(grid)
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v is None:
                continue
            cs = 1
            while c + cs < len(row) and row[c + cs] is None:
                cs += 1
            rs = 1
            while r + rs < nr and c < len(grid[r + rs]) and grid[r + rs][c] is None:
                rs += 1
            out[(r, c)] = (rs, cs, cell_norm(nfkc_math(v)))
    return out


def _model_cells(rows):
    """parsed HTML/pipe rows [(rowspan, colspan, text)] -> {(r, c): (rowspan, colspan, text)}; rows that hold only
    empty single cells (invented empty header rows) are dropped first."""
    rows = [row for row in rows if any(cell_norm(t) or rs > 1 or cs > 1 for rs, cs, t in row)]
    occ, out = set(), {}
    for r, row in enumerate(rows):
        c = 0
        for rs, cs, t in row:
            while (r, c) in occ:
                c += 1
            out[(r, c)] = (rs, cs, cell_norm(nfkc_math(t)))
            for dr in range(rs):
                for dc in range(cs):
                    occ.add((r + dr, c + dc))
            c += cs
    return out


def g7(ref, md):
    """Tables: a vertically merged cell that carries text in pdfplumber's ruled grid (an independent reader) must not
    come back split into the text plus empty cells below it (the rows below would read as having no value).
    Cell positions and texts of the grid are not compared: its row/column splits are unreliable (bake-off and
    2026-11-15 repair measurements)."""
    if ref["kind"] != "table" or not ref["pdf_grid"]:
        return True, "n/a"
    P = _pdf_cells(ref["pdf_grid"])
    best, best_score = None, None
    for rows in model_table_rows(md):
        M = _model_cells(rows)
        score = sum(1 for k, v in P.items() if k in M and M[k][2] == v[2] and v[2])
        if best_score is None or score > best_score:
            best, best_score = M, score
    if best is None:
        return False, "no table in output"
    bad = []
    for (r, c), (rs, _cs, t) in P.items():
        if not t or rs < 2:
            continue
        m = best.get((r, c))
        if m is None or m[2] != t:
            continue
        split = [(r + k, c) for k in range(1, rs) if (r + k, c) in best and not best[(r + k, c)][2]]
        if split and m[0] < rs:
            bad.append(
                f"merged cell {t[:30]!r} spans {rs} rows in the PDF, output {m[0]} row(s) + empty cells {split[:3]}"
            )
    return not bad, "; ".join(bad)


def g8(md, others):
    """Agreement with each second recognition on the canonical glyph stream (tables: normalised rows)."""
    res = {}
    mine = canonical(md)[0]
    for name, other in others.items():
        if other is None:
            res[name] = None
            continue
        res[name] = canonical(other)[0] == mine
    return res


def check(ref, meta, md, others=None):
    md, fixes = canonicalise(md, ref)
    stream, unknown, rules, overlines = render(md)
    out = {"fixes": fixes}
    out["G0"] = g0(ref, meta, md)
    out["G1"] = g1(ref, md)
    out["G2"] = g2(ref, md)
    out["G3"] = g3(ref, stream, unknown)
    out["G4"] = g4(ref, stream)
    out["G5"] = g5(ref, stream)
    out["G6"] = g6(ref, rules, overlines)
    out["G7"] = g7(ref, md)
    if others is not None:
        ag = g8(md, {k: canonicalise(v, ref)[0] if v is not None else None for k, v in others.items()})
        out["G8"] = {k: (v, "" if v else "disagrees") for k, v in ag.items()}
    out["md"] = md
    return out
