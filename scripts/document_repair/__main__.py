"""python -m scripts.document_repair PDF OUTDIR [--policy P] [--gpu-host ALIAS] [--outputs MAP] [--math-font PATH]

Without a GPU host (the default) the tool runs the text stage and region detection only, sends nothing anywhere
and lists every region for review. See docs/DOCUMENT_QUALITY.md, "Repairing a document from its PDF".
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .pipeline import POLICIES, run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.document_repair", description=__doc__.split("\n\n")[1])
    ap.add_argument("pdf", help="the document's PDF; its file name (without .pdf) names the outputs")
    ap.add_argument("outdir", help="output directory (repaired.md, regions.json, counts.json, guards.json, masked/)")
    ap.add_argument(
        "--policy",
        choices=POLICIES,
        default="recommended",
        help="acceptance policy; 'recommended' (default): formulas need G0, G1, G2, G3, G5, G6; tables always review",
    )
    ap.add_argument(
        "--gpu-host",
        default=os.environ.get("BDDK_REPAIR_GPU_HOST") or None,
        help="ssh host alias of the GPU machine (default: BDDK_REPAIR_GPU_HOST; unset: no recognition)",
    )
    ap.add_argument("--gpu-workdir", default="bddk-repair", help="remote directory holding recognize.py")
    ap.add_argument("--gpu-python", default="python3", help="remote Python with torch and transformers")
    ap.add_argument("--outputs", help="JSON map of recognition outputs already computed (name -> file, name.meta)")
    ap.add_argument(
        "--math-font",
        default=os.environ.get("BDDK_REPAIR_MATH_FONT") or None,
        help="full Cambria Math font (cambria.ttc); default BDDK_REPAIR_MATH_FONT, else %%WINDIR%%\\Fonts on Windows",
    )
    a = ap.parse_args(argv)
    if a.gpu_host and a.outputs:
        ap.error("--gpu-host and --outputs exclude each other")
    counts = run(
        a.pdf,
        a.outdir,
        policy=a.policy,
        outputs=a.outputs,
        gpu_host=a.gpu_host,
        gpu_workdir=a.gpu_workdir,
        gpu_python=a.gpu_python,
        math_font=a.math_font,
    )
    print(json.dumps(counts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
