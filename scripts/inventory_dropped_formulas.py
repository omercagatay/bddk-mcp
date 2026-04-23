"""Scan seed_data/documents.json for documents with likely-dropped formulas.

Offline, read-only: reads the committed seed directly — no DB required. Produces
a ranked markdown report for triage.

Heuristics (per document):

  S1  Extraction-method flag. Counts only when other signals fire — by itself
      absence of `manual_latex` means nothing (most docs have no formulas).

  S2  "Intro → Formülde gap" — the classic dropped-formula signature: an
      introducer phrase (`aşağıdaki formül`, `formül ile hesaplanır`,
      `aşağıdaki formüle göre`) followed within 250 chars by `Formülde,` or
      `Formülde;` with no LaTeX block between them. Matches the real failure
      mode where the formula image was stripped but the surrounding prose
      (intro + variable-definition clause) remained.

  S3  Residual broken image references (`_dosyalar/imageNNN.(gif|png)`) still
      present in markdown — rare (the extractor usually strips them) but
      definitive when present.

  S4  "Dangling definition" pattern: a line that starts with 5+ spaces then
      `:` then text — the signature of an inline variable glyph that was
      stripped, leaving the definition without its subject (e.g. `
      : tk gelecek zamanında`).

  S5  `Formülde,` directly followed by bullet-list with no LaTeX immediately
      preceding. Variant of S2 that catches cases where the introducer phrase
      itself was trimmed but the variable-list remained.

Score = weighted sum of signal hits. Output is ranked descending.

Usage:
    uv run python scripts/inventory_dropped_formulas.py
    uv run python scripts/inventory_dropped_formulas.py --min-score 2
    uv run python scripts/inventory_dropped_formulas.py --out logs/report.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED_DOCS = ROOT / "seed_data" / "documents.json"

INTRO_PATTERNS = [
    r"aşağıdaki formül",
    r"formül ile hesaplanır",
    r"formülü ile hesaplanır",
    r"aşağıdaki formüle göre",
    r"aşağıdaki formül yoluyla",
    r"aşağıda yer alan korelasyon formülü",
]
INTRO_RE = re.compile("|".join(INTRO_PATTERNS), re.IGNORECASE)

LATEX_BLOCK_RE = re.compile(r"\$\$[^$]{1,600}\$\$")
LATEX_INLINE_RE = re.compile(r"(?<!\$)\$[^$\n]{1,200}\$(?!\$)")
BROKEN_IMG_RE = re.compile(r"_dosyalar/image\d+\.(?:gif|png)")
DANGLING_DEF_RE = re.compile(r"^\s{5,}:\s+\S", re.MULTILINE)
FORMULDE_BULLET_RE = re.compile(
    r"Formülde[,:]\s*\n+\s*[-•]\s",
    re.MULTILINE,
)

# Signal weights. S3 (broken image ref) is definitive. S2 and S5 are the
# classic dropped-formula signatures. S1 is weak on its own (counted only
# as a multiplier/modifier, not stand-alone evidence).
WEIGHTS = {"S1": 1, "S2": 3, "S3": 5, "S4": 1, "S5": 3}


@dataclass
class DocReport:
    doc_id: str
    title: str
    extraction_method: str
    char_len: int
    signals: dict[str, int] = field(default_factory=dict)
    score: int = 0
    excerpts: list[str] = field(default_factory=list)


def _has_latex_within(body: str, start: int, window: int = 500) -> bool:
    slice_ = body[start : start + window]
    return bool(LATEX_BLOCK_RE.search(slice_) or LATEX_INLINE_RE.search(slice_))


def _formulde_bullet_without_latex(body: str) -> int:
    """Count Formülde→bullet patterns with no LaTeX block immediately before
    the Formülde marker (within 200 chars back) — indicates formula was
    dropped, leaving the variable definitions stranded."""
    hits = 0
    for m in FORMULDE_BULLET_RE.finditer(body):
        back = body[max(0, m.start() - 200) : m.start()]
        if not LATEX_BLOCK_RE.search(back):
            hits += 1
    return hits


FORMULDE_RE = re.compile(r"Formülde[,;:]", re.IGNORECASE)


def _intro_to_formulde_gap_without_latex(body: str) -> tuple[int, list[str]]:
    """S2: intro phrase followed by `Formülde,` within 250 chars, with no
    LaTeX block between them. This is the precise signature of a dropped
    formula — surrounding prose remains but the math itself is gone."""
    hits = 0
    excerpts: list[str] = []
    for m in INTRO_RE.finditer(body):
        window_start = m.end()
        window_end = min(len(body), window_start + 250)
        gap = body[window_start:window_end]
        formulde = FORMULDE_RE.search(gap)
        if not formulde:
            continue
        between = gap[: formulde.start()]
        if LATEX_BLOCK_RE.search(between) or LATEX_INLINE_RE.search(between):
            continue
        hits += 1
        if len(excerpts) < 2:
            snippet = body[max(0, m.start() - 40) : window_start + formulde.end() + 40]
            snippet = re.sub(r"\s+", " ", snippet).strip()
            excerpts.append(f"…{snippet}…")
    return hits, excerpts


def analyze(doc: dict) -> DocReport:
    body = doc["markdown_content"]
    method = doc.get("extraction_method", "") or ""

    signals: dict[str, int] = {}
    excerpts: list[str] = []

    # S2 — intro-to-Formülde gap (classic dropped formula signature)
    s2_hits, s2_excerpts = _intro_to_formulde_gap_without_latex(body)
    if s2_hits:
        signals["S2"] = s2_hits
        excerpts.extend(s2_excerpts)

    # S3 — residual broken image refs
    broken = len(BROKEN_IMG_RE.findall(body))
    if broken:
        signals["S3"] = broken

    # S4 — dangling definition lines (stripped inline variables)
    dangling = len(DANGLING_DEF_RE.findall(body))
    if dangling:
        signals["S4"] = dangling

    # S5 — Formülde immediately followed by bullet, no LaTeX before
    s5 = _formulde_bullet_without_latex(body)
    if s5:
        signals["S5"] = s5

    # S1 — weak modifier: counted ONLY if other signals already fire. An
    # extraction_method lacking manual_latex is benign if the doc has no
    # formula content.
    if signals and "manual_latex" not in method:
        signals["S1"] = 1

    score = sum(WEIGHTS[sig] * count for sig, count in signals.items())
    return DocReport(
        doc_id=doc["document_id"],
        title=doc.get("title", ""),
        extraction_method=method,
        char_len=len(body),
        signals=signals,
        score=score,
        excerpts=excerpts,
    )


def render_report(reports: list[DocReport], min_score: int) -> str:
    ranked = sorted(
        (r for r in reports if r.score >= min_score),
        key=lambda r: r.score,
        reverse=True,
    )
    lines = [
        "# Dropped-formula inventory — seed_data scan",
        "",
        f"Scanned {len(reports)} docs, flagged {len(ranked)} with score ≥ {min_score}.",
        "",
        "Signal legend (weights in parentheses):",
        "- **S1** (×1) — extraction_method lacks `manual_latex`",
        "- **S2** (×2) — formula-introducer phrase without LaTeX within 500 chars",
        "- **S3** (×5) — residual `_dosyalar/imageNNN.gif` broken image ref",
        "- **S4** (×1) — dangling `   : definition` line (stripped inline variable)",
        "- **S5** (×3) — `Formülde,` → bullet list with no LaTeX immediately before",
        "",
        "| Rank | doc_id | Score | S1 | S2 | S3 | S4 | S5 | Extraction method | Title |",
        "|------|--------|------:|---:|---:|---:|---:|---:|-------------------|-------|",
    ]
    for i, r in enumerate(ranked, 1):
        title = (r.title[:60] + "…") if len(r.title) > 60 else r.title
        row = (
            f"| {i} | `{r.doc_id}` | {r.score} | "
            f"{r.signals.get('S1', 0)} | {r.signals.get('S2', 0)} | "
            f"{r.signals.get('S3', 0)} | {r.signals.get('S4', 0)} | "
            f"{r.signals.get('S5', 0)} | "
            f"`{r.extraction_method or '—'}` | {title} |"
        )
        lines.append(row)
    lines.append("")
    lines.append("## Excerpts (top 20 flagged docs)")
    for r in ranked[:20]:
        if not r.excerpts:
            continue
        lines.append(f"\n### `{r.doc_id}` — score {r.score}")
        for e in r.excerpts:
            lines.append(f"> {e}")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--min-score", type=int, default=1, help="minimum score to include in report (default 1)")
    p.add_argument("--out", type=Path, default=ROOT / "logs" / "dropped_formulas_report.md")
    args = p.parse_args()

    with SEED_DOCS.open(encoding="utf-8") as f:
        docs = json.load(f)

    reports = [analyze(d) for d in docs]
    out_text = render_report(reports, args.min_score)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out_text, encoding="utf-8")

    flagged = sum(1 for r in reports if r.score >= args.min_score)
    print(f"scanned {len(reports)} docs, flagged {flagged}")
    print(f"report written to {args.out}")

    # Top 5 to stdout for quick glance
    ranked = sorted(reports, key=lambda r: r.score, reverse=True)
    print("\ntop 5 by score:")
    for r in ranked[:5]:
        print(f"  {r.score:3d}  {r.doc_id:24s}  {r.extraction_method}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
