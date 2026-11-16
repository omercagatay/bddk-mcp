"""Stage 1 + 5: page text from the text layer, model-free repairs, accepted regions merged in.

assemble(pdf_path, pages, regions, accepted) -> (markdown, report)
  pages     layout.load() pages (chars carry 'pos' after guards.classify_scripts)
  regions   detect() regions (cut boxes)
  accepted  {region id: markdown to put in place of the region's text-layer lines}

Model-free repairs of the text layer: raised or lowered small glyphs (footnote markers glued to figures, flattened
indices) are wrapped as $^{..}$ / $_{..}$; footnotes (from a line whose first glyph is a raised small number at the
left text edge to the page bottom, continued on the next page up to the first sentence end) are moved after the body.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter

from . import textlayer
from .guards import classify_scripts
from .layout import bands, nfkc_math
from .mdparse import model_tables
from .textlayer import SENT_END, join_pages, normalise


def line_extent(b):
    """Vertical extent of a band from baselines (char boxes of some producers are offset from the ink)."""
    return (min(c["base"] - 0.8 * c["size"] for c in b["chars"]), max(c["base"] + 0.25 * c["size"] for c in b["chars"]))


def footnotes(pages_bands):
    """Per page (first footnote band index or None, number of top bands continuing the previous page's footnote)."""
    res, open_fn = [], False
    for _page, bnds in pages_bands:
        col0 = min((b["x0"] for b in bnds), default=0)
        start = None
        for i, b in enumerate(bnds):
            vis = sorted((c for c in b["chars"] if c["t"].strip()), key=lambda c: c["x0"])
            if i == 0 or not vis or vis[0]["x0"] > col0 + 3:
                continue
            if vis[0].get("pos") == "sup" and vis[0]["t"].isdigit() and any(not c.get("pos") for c in vis):
                start = i
                break
        cont = 0
        if open_fn:
            limit = start if start is not None else len(bnds)
            for i in range(limit):
                cont = i + 1
                if SENT_END.search(bnds[i]["text"]):
                    break
        if start is not None:
            open_fn = not SENT_END.search(bnds[-1]["text"])
        elif cont:
            open_fn = cont == len(bnds) and not SENT_END.search(bnds[-1]["text"])
        else:
            open_fn = False
        res.append((start, cont))
    return res


def to_pipe(md):
    """Model table (HTML or pipe) -> Markdown pipe table; a spanned cell's text is repeated in every cell it covers
    (as in the existing seed tables), so no cell reads as empty."""
    out = []
    for t in model_tables(md, fill_spans=True):
        rows = [[re.sub(r"\s+", " ", c).replace("|", "\\|").strip() for c in r] for r in t]
        w = max(len(r) for r in rows)
        rows = [r + [""] * (w - len(r)) for r in rows]
        # a row that is one cell spanning the whole width (legend, note) becomes a paragraph after the table
        notes = [r[0] for r in rows if w > 1 and len(set(r)) == 1 and r[0]]
        rows = [r for r in rows if not (w > 1 and len(set(r)) == 1 and r[0])]
        if not rows:
            out.append("\n\n".join(notes))
            continue
        tab = "\n".join(
            ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * w] + ["| " + " | ".join(r) + " |" for r in rows[1:]]
        )
        # round trip: the stored pipe table must parse back to exactly the model's (span-filled) grid
        back = [[c.replace("\\|", "|") for c in r] for r in (model_tables(tab) or [[]])[0]]
        want = [[c.replace("\\|", "|") for c in r] for r in rows]
        if back != want:
            raise ValueError("pipe table round trip failed")
        out.append(tab + ("\n\n" + "\n\n".join(notes) if notes else ""))
    return out


def region_markdown(md, kind):
    """Accepted output as stored: tables as pipe tables (text around them kept), formulas as given ($$...$$)."""
    if kind == "table" and ("<table" in md or md.lstrip().startswith("|")):
        # keep HTML sub/superscripts as LaTeX scripts (the HTML table parser keeps only text)
        md = re.sub(r"<sub>(.*?)</sub>", r"$_{\1}$", md, flags=re.S | re.I)
        md = re.sub(r"<sup>(.*?)</sup>", r"$^{\1}$", md, flags=re.S | re.I)
        tables = to_pipe(md)
        rest = re.sub(r"<table.*?</table>", "\0", md, flags=re.S)
        rest = re.sub(r"(?m)^\s*\|.*\|\s*$\n?", "", rest)
        parts = rest.split("\0")
        merged = parts[0]
        for k, t in enumerate(tables):
            merged += "\n\n" + t + "\n\n" + (parts[k + 1] if k + 1 < len(parts) else "")
        return merged.strip()
    return md.strip()


def align(text, chars):
    """text index -> pdfplumber char (or None), by aligning the non-space characters of both."""
    t_idx = [i for i, ch in enumerate(text) if not ch.isspace()]
    t_ns = "".join(text[i] for i in t_idx)
    owners = []
    for c in chars:
        owners.extend([c] * len(nfkc_math(c["t"])))
    p_ns = "".join(nfkc_math(c["t"]) for c in chars)
    sm = difflib.SequenceMatcher(None, t_ns, p_ns, autojunk=False)
    out = [None] * len(text)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            out[t_idx[a + k]] = owners[b + k]
    return out


def wrap_scripts(line, amap, start, ok=lambda c: True):
    """$^{..}$ / $_{..}$ around runs of characters aligned to raised/lowered small glyphs (only glyphs ok() admits:
    body / footnote glyphs, never glyphs of a region that shares the line)."""
    out, i, runs = [], 0, 0
    while i < len(line):
        c = amap[start + i]
        kind = c.get("pos") if c is not None and not line[i].isspace() and ok(c) else None
        if kind:
            j = i
            while (
                j + 1 < len(line)
                and amap[start + j + 1] is not None
                and amap[start + j + 1].get("pos") == kind
                and ok(amap[start + j + 1])
            ):
                j += 1
            if i > 0 and line[i - 1].isalpha() and j + 1 < len(line) and line[j + 1].isalpha():
                # a "script" inside a word (a letter in a fallback font): not an index, leave the text alone
                out.append(line[i : j + 1])
                i = j + 1
                continue
            out.append(("$^{" if kind == "sup" else "$_{") + line[i : j + 1] + "}$")
            runs += 1
            i = j + 1
        else:
            out.append(line[i])
            i += 1
    return "".join(out), runs


SCRIPT_RE = re.compile(r".{0,25}\$[\^_]\{[^}]*\}\$.{0,10}")


def assemble(pdf_path, pages, regions, accepted):
    """Page text = the text layer's own lines, each assigned to body / footnote / region by aligning its characters
    to the PDF's glyphs; a region's lines are replaced by the accepted output only when no line mixes region and
    non-region glyphs and the region's lines are contiguous. Returns (markdown, report)."""
    pages_bands = []
    for p in pages:
        bn = bands(p)
        classify_scripts([c for b in bn for c in b["chars"]], p["body"])
        pages_bands.append((p, bn))
    fn = footnotes(pages_bands)
    body_pages, foot = [], []
    report = {"pages": [], "script_runs": 0, "script_examples": [], "regions": {}}
    full = textlayer.pdftotext_pages(pdf_path)
    kinds = {r["id"]: r["kind"] for r in regions}
    for (p, bn), (fstart, fcont), text in zip(pages_bands, fn, full, strict=False):
        label = {i: "foot" if (fstart is not None and i >= fstart) or i < fcont else "body" for i in range(len(bn))}
        for r in regions:
            if r["page"] == p["page"]:
                for i in r["band_ids"]:
                    label[i] = r["id"]
        chars = []
        for i, b in enumerate(bn):
            for c in sorted(b["chars"], key=lambda c: c["x0"]):
                if c["t"].strip():
                    c["band"] = i
                    chars.append(c)
        amap = align(text, chars)
        lines, pos = [], 0
        for ln in text.split("\n"):
            votes = Counter(label[amap[k]["band"]] for k in range(pos, pos + len(ln)) if amap[k] is not None)
            lines.append([ln, pos, votes.most_common(1)[0][0] if votes else None, votes])
            pos += len(ln) + 1
        # unlabelled lines (blank or unaligned) take the label of their neighbours
        for k, L in enumerate(lines):
            if L[2] is None:
                prev = next((lines[j][2] for j in range(k - 1, -1, -1) if lines[j][2]), None)
                nxt = next((lines[j][2] for j in range(k + 1, len(lines)) if lines[j][2]), None)
                if prev == nxt or nxt is None:
                    L[2] = prev or "body"
                elif not L[0].strip():
                    L[2] = prev if prev in ("body", "foot") else nxt
                else:
                    L[2] = nxt if prev in ("body", "foot") else prev
        # a line where body text and one region sit side by side (pdftotext -layout puts "1. aşama:" and the
        # formula's middle row on one line): split it at the boundary when the two do not interleave
        for L in lines:
            regs_here = [lab for lab in L[3] if lab not in ("body", "foot")]
            plain_here = [lab for lab in L[3] if lab in ("body", "foot")]
            if len(regs_here) != 1 or len(plain_here) != 1:
                continue
            rid, plab = regs_here[0], plain_here[0]
            rpos = [k for k in range(L[1], L[1] + len(L[0])) if amap[k] is not None and label[amap[k]["band"]] == rid]
            ppos = [k for k in range(L[1], L[1] + len(L[0])) if amap[k] is not None and label[amap[k]["band"]] == plab]
            if max(ppos) < min(rpos):
                L.append(("prefix", L[0][: min(rpos) - L[1]].rstrip(), plab))
            elif min(ppos) > max(rpos):
                L.append(("suffix", L[0][max(rpos) - L[1] + 1 :].strip(), plab))
            else:
                continue
            L[2] = rid
            L[3] = Counter({rid: len(rpos)})
        for r in regions:
            if r["page"] != p["page"]:
                continue
            rid = r["id"]
            leak_in = sum(n for L in lines if L[2] == rid for lab, n in L[3].items() if lab != rid)
            leak_out = sum(L[3].get(rid, 0) for L in lines if L[2] != rid)
            idx = [k for k, L in enumerate(lines) if L[2] == rid]
            contiguous = bool(idx) and idx == list(range(idx[0], idx[-1] + 1))
            report["regions"][rid] = {
                "lines": len(idx),
                "leak_in": leak_in,
                "leak_out": leak_out,
                "contiguous": contiguous,
                "clean": contiguous and not leak_in and not leak_out,
            }
        # safety: a region whose text-layer lines cannot be located (none, or not contiguous) may have had its lines
        # absorbed by a neighbour; then no region next to it is replaced, so nothing of it can be deleted
        page_regs = [r for r in regions if r["page"] == p["page"]]
        for r in page_regs:
            rep_r = report["regions"][r["id"]]
            if rep_r["lines"] and rep_r["contiguous"]:
                continue
            for q in page_regs:
                if (
                    q is not r
                    and q["cut"][1] - r["cut"][3] < 1.5 * p["body"]
                    and r["cut"][1] - q["cut"][3] < 1.5 * p["body"]
                ):
                    report["regions"][q["id"]]["clean"] = False
                    report["regions"][q["id"]]["blocked_by_unlocatable_neighbour"] = r["id"]
        # regions whose rows pdftotext interleaves on shared lines (stacked formulas): merged as one block when every
        # region of the group is accepted and the block's lines hold no body text
        parent = {}

        def find(x, parent=parent):
            while parent.get(x, x) != x:
                x = parent[x]
            return x

        for L in lines:
            regs_here = sorted({lab for lab in L[3] if lab not in ("body", "foot")} | ({L[2]} - {"body", "foot"}))
            for a, b in zip(regs_here, regs_here[1:], strict=False):
                parent[find(a)] = find(b)
        groups = {}
        for r in regions:
            if r["page"] == p["page"]:
                groups.setdefault(find(r["id"]), []).append(r)
        block_of = {}
        for g in groups.values():
            if len(g) < 2:
                continue
            ids = {r["id"] for r in g}
            idx = [k for k, L in enumerate(lines) if L[2] in ids or any(lab in ids for lab in L[3])]
            plain = sum(n for k in idx for lab, n in lines[k][3].items() if lab in ("body", "foot"))
            outside = sum(n for k, L in enumerate(lines) if k not in idx for lab, n in L[3].items() if lab in ids)
            ok = bool(idx) and idx == list(range(idx[0], idx[-1] + 1)) and not plain and not outside
            for r in g:
                report["regions"][r["id"]]["group"] = sorted(ids)
                report["regions"][r["id"]]["group_separable"] = ok
            if ok and all(i in accepted for i in ids):
                order = [r["id"] for r in sorted(g, key=lambda r: r["cut"][1])]
                for k in idx:
                    block_of[k] = order
                for i in ids:
                    report["regions"][i]["clean"] = True
        # a region is replaced only if every region stacked right above or below it (within 1.5 lines) is replaced
        # too: with garbled text-layer lines a neighbour's line can be mis-assigned and would vanish (1336 p8 D_b)
        ok_ids = {r["id"] for r in page_regs if r["id"] in accepted and report["regions"][r["id"]]["clean"]}
        changed = True
        while changed:
            changed = False
            for r in page_regs:
                if r["id"] not in ok_ids:
                    continue
                for q in page_regs:
                    if (
                        q is not r
                        and q["id"] not in ok_ids
                        and q["cut"][1] - r["cut"][3] < 1.5 * p["body"]
                        and r["cut"][1] - q["cut"][3] < 1.5 * p["body"]
                    ):
                        ok_ids.discard(r["id"])
                        report["regions"][r["id"]]["clean"] = False
                        report["regions"][r["id"]]["blocked_by_unmerged_neighbour"] = q["id"]
                        changed = True
                        break
        block_of = {k: v for k, v in block_of.items() if all(i in ok_ids for i in v)}
        parts, done, fparts = [], set(), []
        for k, L in enumerate(lines):
            ln, start, lab = L[0], L[1], L[2]
            if k in block_of:
                order = block_of[k]
                if order[0] not in done:
                    try:
                        stored = [region_markdown(accepted[i], kinds[i]) for i in order]
                    except ValueError:
                        stored = None
                    if stored is not None:
                        parts += [""] + [x for st in stored for x in (st, "")]
                        done.update(order)
                        continue
                else:
                    continue
            if lab not in ("body", "foot"):
                if lab in accepted and report["regions"][lab]["clean"] and lab not in done:
                    try:
                        stored = region_markdown(accepted[lab], kinds[lab])
                    except ValueError as e:  # the region keeps its text-layer lines
                        report["regions"][lab]["clean"] = False
                        report["regions"][lab]["merge_error"] = str(e)
                    else:
                        mine = [M for M in lines if M[2] == lab and len(M) > 4]
                        pre = [M[4][1] for M in mine if M[4][0] == "prefix" and M[4][1]]
                        suf = [M[4][1] for M in mine if M[4][0] == "suffix" and M[4][1]]
                        parts += pre + ["", stored, ""] + suf
                        if mine:
                            report["regions"][lab]["split_lines"] = len(mine)
                        done.add(lab)
                        continue
                if lab in done:
                    continue
                parts.append(ln)
                continue
            piece, n = wrap_scripts(ln, amap, start, lambda c, label=label: label[c["band"]] in ("body", "foot"))
            report["script_runs"] += n
            if n and len(report["script_examples"]) < 80:
                report["script_examples"] += SCRIPT_RE.findall(piece)[:3]
            (fparts if lab == "foot" else parts).append(piece)
        report["pages"].append({"page": p["page"], "footnote_start_band": fstart, "footnote_continuation_bands": fcont})
        body_pages.append("\n".join(parts))
        foot.append("\n".join(fparts))
    text = join_pages(body_pages)
    foot_text = join_pages(foot)
    if foot_text.strip():
        text = text.rstrip() + "\n\n" + foot_text.strip("\n")
    return normalise(text), report
