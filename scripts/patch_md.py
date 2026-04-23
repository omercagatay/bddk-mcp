r"""Small DSL for surgically editing a markdown body at known anchor points.

Intended use: formula-recovery workflows (see commit history for SYSTEMIC-7).
Edits are expressed as a list of `Insert(anchor, content)` / `Replace(old, new)`
operations. `apply_ops` enforces that every anchor is unique within the body
at the moment it's applied — ambiguous or missing anchors raise early with
helpful context instead of silently producing wrong output.

A companion `validate_latex` catches the common LaTeX typos that would
otherwise reach the database unnoticed (unbalanced `$$`, mismatched
`\begin/\end`, stray `$`).

Usage:
    from patch_md import Insert, Replace, apply_ops, validate_latex

    body = seed_document["markdown_content"]
    ops = [
        Insert("2) Gelişmiş İDD Yaklaşımının kullanıldığı durumda RA,",
               r"$$RA = \max\{0;\ 12{,}5 \times (THK - BKET)\}$$"),
        Replace("RA =\n\ncase1\ncase2",
                r"$$RA = \begin{cases}…\end{cases}$$"),
    ]
    new_body = apply_ops(body, ops)
    issues = validate_latex(new_body)
    assert not issues, issues
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class AnchorError(ValueError):
    """Raised when an Insert/Replace op's anchor is not uniquely resolvable."""


class Op(Protocol):
    """Common interface — apply returns the mutated body or raises AnchorError."""

    def apply(self, body: str) -> str: ...


@dataclass(frozen=True)
class Insert:
    """Insert `content` as a new block immediately after `anchor`.

    The existing whitespace between `anchor` and the next non-whitespace
    character is collapsed; `\\n\\n<content>\\n\\n` is inserted in its place,
    so the content becomes a properly-separated markdown block.

    The `anchor` must appear exactly once in `body` at apply time. A mismatch
    (0 or 2+ occurrences) raises AnchorError with the match locations /
    nearest partial-match suggestion.
    """

    anchor: str
    content: str

    def apply(self, body: str) -> str:
        _require_unique(body, self.anchor, op_kind="Insert", op_detail=self.anchor[:60])
        pos = body.index(self.anchor) + len(self.anchor)
        end = pos
        while end < len(body) and body[end] in " \t\n":
            end += 1
        return body[:pos] + "\n\n" + self.content + "\n\n" + body[end:]


@dataclass(frozen=True)
class Replace:
    """Replace exactly one occurrence of `old` with `new`.

    `old` must appear exactly once in `body` at apply time.
    """

    old: str
    new: str

    def apply(self, body: str) -> str:
        _require_unique(body, self.old, op_kind="Replace", op_detail=self.old[:60])
        return body.replace(self.old, self.new, 1)


def apply_ops(body: str, ops: list[Op]) -> str:
    """Apply ops sequentially. Each op sees the mutated body from previous ops."""
    for i, op in enumerate(ops, 1):
        try:
            body = op.apply(body)
        except AnchorError as e:
            # Re-raise with op index for easier debugging in long op lists.
            raise AnchorError(f"op #{i} ({type(op).__name__}): {e}") from None
    return body


# --- anchor resolution ---------------------------------------------------


def _line_number(body: str, pos: int) -> int:
    return body.count("\n", 0, pos) + 1


def _find_partial_matches(body: str, anchor: str, *, limit: int = 3) -> list[tuple[int, str]]:
    """Best-effort nearest matches when the full anchor isn't found.

    Strategy: take the first 30 chars of the anchor and search for them
    as a substring. Returns up to `limit` hits with surrounding context.
    """
    if not anchor.strip():
        return []
    probe = anchor[:30]
    hits: list[tuple[int, str]] = []
    start = 0
    while len(hits) < limit:
        idx = body.find(probe, start)
        if idx < 0:
            break
        snippet = body[idx : idx + max(len(anchor), 80)]
        snippet = re.sub(r"\s+", " ", snippet).strip()
        hits.append((idx, snippet))
        start = idx + 1
    return hits


def _require_unique(body: str, needle: str, *, op_kind: str, op_detail: str) -> None:
    n = body.count(needle)
    if n == 1:
        return
    if n == 0:
        hints = _find_partial_matches(body, needle)
        if hints:
            hint_lines = "\n".join(f"    line {_line_number(body, idx)}: {snippet[:120]}…" for idx, snippet in hints)
            raise AnchorError(
                f"anchor not found (0 matches) for {op_kind}({op_detail!r}).\n"
                f"  Nearest partial matches of the first 30 chars:\n{hint_lines}"
            )
        raise AnchorError(
            f"anchor not found (0 matches) for {op_kind}({op_detail!r}) — "
            "no partial matches either; double-check the anchor text against the current body."
        )
    # n >= 2 — show all occurrences with line numbers.
    locations = []
    start = 0
    for _ in range(min(n, 5)):
        idx = body.find(needle, start)
        if idx < 0:
            break
        locations.append(f"    line {_line_number(body, idx)} (char {idx})")
        start = idx + 1
    raise AnchorError(
        f"anchor is not unique ({n} matches) for {op_kind}({op_detail!r}).\n"
        f"  Occurrences:\n" + "\n".join(locations) + "\n"
        "  Extend the anchor until it matches exactly one location."
    )


# --- LaTeX validation ----------------------------------------------------


_BLOCK_DELIM_RE = re.compile(r"\$\$")
_BEGIN_RE = re.compile(r"\\begin\{([^}]+)\}")
_END_RE = re.compile(r"\\end\{([^}]+)\}")
# Inline $...$ — a `$` not preceded by `$` or `\`, not part of `$$`.
_INLINE_DOLLAR_RE = re.compile(r"(?<![\\$])\$(?!\$)")


def validate_latex(body: str) -> list[str]:
    """Return a list of LaTeX issues. Empty list = pass.

    Checks:
      1. `$$` count is even (block delimiters balanced).
      2. `\\begin{X}` and `\\end{X}` counts match, per X.
      3. Inline `$` count (excluding `$$` and `\\$`) is even.
    """
    issues: list[str] = []

    block_delims = len(_BLOCK_DELIM_RE.findall(body))
    if block_delims % 2 != 0:
        issues.append(f"odd number of `$$` delimiters ({block_delims}) — a block formula is unclosed")

    begin_counts: dict[str, int] = {}
    for m in _BEGIN_RE.finditer(body):
        begin_counts[m.group(1)] = begin_counts.get(m.group(1), 0) + 1
    end_counts: dict[str, int] = {}
    for m in _END_RE.finditer(body):
        end_counts[m.group(1)] = end_counts.get(m.group(1), 0) + 1
    for env in sorted(set(begin_counts) | set(end_counts)):
        b, e = begin_counts.get(env, 0), end_counts.get(env, 0)
        if b != e:
            issues.append(f"`\\begin{{{env}}}` × {b} vs `\\end{{{env}}}` × {e} — environment unbalanced")

    # Inline $ count: strip $$ blocks first so we only see genuine inline delims.
    stripped = _BLOCK_DELIM_RE.sub("", body)
    inline_count = len(_INLINE_DOLLAR_RE.findall(stripped))
    if inline_count % 2 != 0:
        issues.append(f"odd number of inline `$` delimiters ({inline_count}) — an inline formula is unclosed")

    return issues
