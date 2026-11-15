"""Repair one document from its PDF: text stage, regions, optional recognition, guards, policy, merge.

  1 text layer + model-free repairs      (this machine: pdftotext -layout, pdfplumber, Cambria Math glyph match)
  2 formula / ruled-table regions        (this machine)
  3 recognition of masked page images    (optional, GPU host over ssh: LightOnOCR-3-4B greedy as a, c, d; 3-1B as b)
  4 acceptance guards per region         (this machine)
  5 merge accepted regions               (this machine)

Outputs in OUTDIR: repaired.md, regions.json (verdict + reason per region: the review list), counts.json,
guards.json, regions_detected.json and masked/ (the page images the reviewer reads, and the GPU input).
"""

from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from pathlib import Path

from . import glyphmap, gpu, masks
from .assemble import assemble
from .detect import detect
from .guards import canonical, check, region_ref
from .layout import bands

GUARDS = ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7"]
AGREE = ["1b", "shift", "ctx"]
POLICIES = ("recommended", "all", "base")
# A recognition output without its meta record is treated as truncated (fail closed).
NO_META = {"eos": False, "truncated": True, "meta_missing": True}


def load_outputs(doc, regs, outdir, mapping=None):
    """Recognition outputs per region and variant: from ``mapping`` (--outputs), else from ``outdir`` (this run's
    fetch); ``outdir`` None means none."""
    meta = {}
    for f in [outdir / "meta.jsonl"] if outdir is not None and outdir.exists() else []:
        for ln in f.read_text(encoding="utf-8").splitlines():
            r = json.loads(ln)
            meta[r["name"]] = r
    res = {}
    for r in regs:
        res[r["id"]] = {}
        for v in ("a", "b", "c", "d"):
            name = f"{doc}_{r['id']}_{v}"
            f = Path(mapping[name]) if mapping and name in mapping else outdir / f"{name}.md" if outdir else None
            res[r["id"]][v] = f.read_text(encoding="utf-8") if f is not None and f.exists() else None
            m = mapping.get(name + ".meta") if mapping else meta.get(name)
            res[r["id"]][v + "_meta"] = m or NO_META
    return res


def verdicts(pages, regs, outs):
    rows = []
    by_page = {}
    for r in regs:
        p = pages[r["page"] - 1]
        if r["page"] not in by_page:
            by_page[r["page"]] = bands(p)
        bn = by_page[r["page"]]
        ref = region_ref(p, bn, r, p["_pg"])
        o = outs[r["id"]]
        if o["a"] is None:
            rows.append({"id": r["id"], "kind": r["kind"], "page": r["page"], "missing_output": True})
            continue
        others = {"1b": o["b"], "shift": o["c"]}
        res = check(ref, o["a_meta"], o["a"], others)
        # context variant: the region's canonical stream must occur inside the context run's stream
        if r["kind"] == "formula" and o["d"] is not None:
            res["G8"]["ctx"] = (canonical(o["a"])[0] in canonical(o["d"])[0], "")
        elif r["kind"] == "formula":
            res["G8"]["ctx"] = (None, "no output")
        for k, meta_k in (("1b", "b_meta"), ("shift", "c_meta"), ("ctx", "d_meta")):
            if o.get(meta_k, {}).get("truncated") and res["G8"].get(k):
                res["G8"][k] = (False, "second run truncated")
        row = {"id": r["id"], "kind": r["kind"], "page": r["page"], "fixes": res["fixes"], "md": res["md"]}
        for g in GUARDS:
            row[g] = [bool(res[g][0]), res[g][1]]
        for k in AGREE:
            v = res["G8"].get(k)
            row["G8_" + k] = [v[0] if v else None, v[1] if v else "n/a"]
        rows.append(row)
    return rows


def accept(row, policy):
    """policy 'recommended' (2026-11-15 measurement): formulas need G0-G3, G5, G6; tables are never stored
    without review (one table error in 25 was caught only by a guard designed after seeing it, or by the 1B
    agreement that also rejects a fifth of the correct tables). 'all': every guard incl. the three agreements.
    'base': the bake-off guards only."""
    if row.get("missing_output"):
        return False, ["no recognition output"]
    if policy == "recommended":
        if row["kind"] == "table":
            return False, ["tables: review"]
        need = ["G0", "G1", "G2", "G3", "G5", "G6"]
    else:
        need = {"base": ["G0", "G1", "G2"], "all": GUARDS + ["G8_" + k for k in AGREE]}[policy]
    failed = [g for g in need if row.get(g) and row[g][0] is False]
    if policy == "all":
        # a second recognition that is missing is not agreement (the context run exists for formulas only)
        failed += [g for g in ("G8_1b", "G8_shift") if row.get(g) and row[g][0] is None]
        if row["kind"] == "formula" and row.get("G8_ctx") and row["G8_ctx"][0] is None:
            failed.append("G8_ctx")
    return not failed, failed


def run(
    pdf,
    od,
    *,
    policy="recommended",
    outputs=None,
    gpu_host=None,
    gpu_workdir="bddk-repair",
    gpu_python="python3",
    math_font=None,
):
    """Repair ``pdf`` into the directory ``od``; returns the counts written to counts.json."""
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; one of {', '.join(POLICIES)}")
    pdf, od = Path(pdf), Path(od)
    od.mkdir(parents=True, exist_ok=True)
    font = glyphmap.configure(math_font)
    if font is None:
        print(
            "Cambria Math font not found (--math-font / BDDK_REPAIR_MATH_FONT): unmapped math glyphs stay "
            "unrecovered, and every region that contains one is rejected (guard G0) and goes to review.",
            flush=True,
        )
    doc = pdf.stem
    t0 = time.time()
    _, pages, regs = detect(pdf)
    (od / "regions_detected.json").write_text(json.dumps(regs, ensure_ascii=False, indent=1), encoding="utf-8")
    gpu_s, rec = None, None
    if regs:
        lists = masks.render(pdf, regs, od / "masked")
        if gpu_host:
            # only this run's recognitions count: outputs left in a reused directory are removed first
            rec = od / "rec"
            if rec.exists():
                shutil.rmtree(rec)
            gpu_s = gpu.run(gpu_host, doc, od / "masked", lists, rec, gpu_workdir, gpu_python)
        elif not outputs:
            print(f"No GPU host: {len(regs)} regions detected, none recognised; every one is listed for review.")
    mapping = json.loads(Path(outputs).read_text(encoding="utf-8")) if outputs else None
    outs = load_outputs(doc, regs, rec, mapping) if regs else {}
    rows = verdicts(pages, regs, outs) if regs else []
    accepted, reg_out = {}, []
    for row, r in zip(rows, regs, strict=True):
        ok, failed = accept(row, policy)
        if ok:
            accepted[r["id"]] = row["md"]
        reg_out.append(
            {
                "id": r["id"],
                "page": r["page"],
                "kind": r["kind"],
                "cut": r["cut"],
                "accepted": ok,
                "failed_guards": failed,
                "reasons": {g: row[g][1] for g in failed if g in row},
            }
        )
    md, rep = assemble(pdf, pages, regs, accepted)
    merged = {rid for rid in accepted if rep["regions"].get(rid, {}).get("clean")}
    for x in reg_out:
        x["merged"] = x["id"] in merged
        if x["accepted"] and not x["merged"]:
            x["reasons"]["merge"] = "text-layer lines of the region are not separable from neighbouring text"
    (od / "repaired.md").write_text(md, encoding="utf-8")
    (od / "regions.json").write_text(json.dumps(reg_out, ensure_ascii=False, indent=1), encoding="utf-8")
    (od / "guards.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    fail_reasons = Counter(g for x in reg_out if not x["accepted"] for g in x["failed_guards"])
    counts = {
        "doc": doc,
        "pages": len(pages),
        "regions": len(regs),
        "by_kind": dict(Counter(r["kind"] for r in regs)),
        "accepted": len(accepted),
        "merged": len(merged),
        "rejected": len(regs) - len(accepted),
        "rejected_by_guard": dict(fail_reasons),
        "first_failing_guard": dict(Counter(x["failed_guards"][0] for x in reg_out if x["failed_guards"])),
        "script_markup_runs": rep["script_runs"],
        "gpu_seconds": gpu_s,
        "total_seconds": round(time.time() - t0, 1),
        "policy": policy,
    }
    (od / "counts.json").write_text(json.dumps(counts, ensure_ascii=False, indent=1), encoding="utf-8")
    return counts
