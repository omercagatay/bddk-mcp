"""Tests for scripts/patch_md.py — Insert / Replace / apply_ops / validate_latex."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from patch_md import AnchorError, Insert, Replace, apply_ops, validate_latex  # noqa: E402

# ---------- Insert ----------


def test_insert_places_content_block_after_anchor():
    body = "first paragraph.\n\nsecond paragraph.\n\nthird paragraph."
    op = Insert("first paragraph.", "$$x = 1$$")
    out = apply_ops(body, [op])
    assert "first paragraph.\n\n$$x = 1$$\n\nsecond paragraph." in out


def test_insert_collapses_existing_whitespace_after_anchor():
    """Inserted content replaces whatever whitespace existed between the anchor
    and the next non-whitespace character — two blank lines, four blank
    lines, tabs, should all end up with exactly one blank line before and
    one blank line after the inserted content."""
    body = "anchor line.\n\n\n\n\tnext content."
    out = apply_ops(body, [Insert("anchor line.", "$$inserted$$")])
    assert out == "anchor line.\n\n$$inserted$$\n\nnext content."


def test_insert_fails_when_anchor_missing_and_suggests_partial():
    body = "The 5411 sayılı Bankacılık Kanunu is the reference law.\nOther content here.\n"
    op = Insert("The 5411 sayılı Bankacılık Kanunununu typo", "$$formula$$")
    with pytest.raises(AnchorError) as exc:
        apply_ops(body, [op])
    msg = str(exc.value)
    assert "0 matches" in msg
    # First 30 chars of the anchor match the real text — should surface as hint.
    assert "line 1" in msg
    assert "5411 sayılı Bankacılık Kanunu" in msg


def test_insert_partial_match_tolerates_typo_near_anchor_start():
    """If the typo lands in the first 30 chars, the head-of-anchor probe
    misses but the tail-of-anchor probe should still surface the real line."""
    body = "2) Gelişmiş İDD Yaklaşımının kullanıldığı durumda RA, formülü uygulanır.\n" + "filler\n" * 200
    # Typo in "Geliş" → "Geliştiş" (first 30 chars broken, tail still correct).
    op = Insert("2) Geliştiş İDD Yaklaşımının kullanıldığı durumda RA,", "$$RA = …$$")
    with pytest.raises(AnchorError) as exc:
        apply_ops(body, [op])
    msg = str(exc.value)
    assert "0 matches" in msg
    # Head probe fails, tail probe ("kullanıldığı durumda RA,") succeeds.
    assert "line 1" in msg
    assert "kullanıldığı durumda RA" in msg


def test_insert_fails_when_anchor_matches_twice_and_lists_locations():
    body = "c) 0 < TO < 1 için.\n\nblah blah.\n\nc) 0 < TO < 1 için.\n"
    op = Insert("c) 0 < TO < 1 için.", "$$formula$$")
    with pytest.raises(AnchorError) as exc:
        apply_ops(body, [op])
    msg = str(exc.value)
    assert "2 matches" in msg
    assert "line 1" in msg
    assert "line 5" in msg


# ---------- Replace ----------


def test_replace_swaps_old_for_new_exactly_once():
    body = "before\nRA = old\nafter"
    out = apply_ops(body, [Replace("RA = old", "$$RA = \\text{new}$$")])
    assert "RA = old" not in out
    assert "$$RA = \\text{new}$$" in out


def test_replace_fails_on_non_unique_old():
    body = "foo\nfoo\n"
    with pytest.raises(AnchorError) as exc:
        apply_ops(body, [Replace("foo\n", "bar\n")])
    assert "2 matches" in str(exc.value)


# ---------- apply_ops sequencing ----------


def test_apply_ops_runs_sequentially_with_mutated_body():
    """Second op should see the body AFTER the first op. If ops mutate the
    same region, the second anchor must match against the mutated state."""
    body = "Para 1.\n\nPara 2.\n"
    ops = [
        Insert("Para 1.", "INSERTED A"),
        Insert("INSERTED A", "INSERTED B"),
    ]
    out = apply_ops(body, ops)
    # After op 1: Para 1. | INSERTED A | Para 2.
    # After op 2: Para 1. | INSERTED A | INSERTED B | Para 2.
    assert out.count("INSERTED A") == 1
    assert out.count("INSERTED B") == 1
    assert out.index("INSERTED A") < out.index("INSERTED B")


def test_apply_ops_error_includes_op_index():
    body = "text"
    ops = [Insert("text", "A"), Insert("missing anchor", "B")]
    with pytest.raises(AnchorError) as exc:
        apply_ops(body, ops)
    assert "op #2" in str(exc.value)


# ---------- validate_latex ----------


def test_validate_latex_accepts_balanced_body():
    body = (
        "Here is a formula: $$RA = \\max\\{0;\\ 12,5 \\times THK\\}$$\n\n"
        "And inline $x + y = z$ too. Also:\n\n"
        "$$RA = \\begin{cases} \\%300 & \\text{listed} \\\\ \\%400 & \\text{other} \\end{cases}$$"
    )
    assert validate_latex(body) == []


def test_validate_latex_detects_unbalanced_block_delims():
    body = "opens but never closes: $$ R = 0,12 × TO "
    issues = validate_latex(body)
    assert any("$$" in i and "unclosed" in i for i in issues)


def test_validate_latex_detects_unbalanced_begin_end():
    body = "$$RA = \\begin{cases} x \\end{cases}$$\n$$\\begin{matrix} 1 \\\\ 2 $$"
    issues = validate_latex(body)
    assert any("matrix" in i for i in issues)


def test_validate_latex_detects_odd_inline_dollar():
    body = "inline starts $ x + y but never closes, and then normal text."
    issues = validate_latex(body)
    assert any("inline" in i for i in issues)


def test_validate_latex_ignores_escaped_dollar():
    """`\\$` (a literal dollar in LaTeX output) must not count as an inline
    delimiter."""
    body = "price tag says \\$100 USD. $$x = y$$"
    assert validate_latex(body) == []


def test_validate_latex_accepts_nested_inline_inside_block():
    """Nested `$…$` inside `$$…$$` blocks is OK — what we really care about
    is that the outer $$ pairs are balanced and inline $ outside blocks are
    also balanced."""
    body = "$$V = \\frac{\\sum_t t \\cdot NA_t}{\\sum_t NA_t}$$\n\nInline var $s_{t_k}$ stands alone."
    assert validate_latex(body) == []


# ---------- dogfooding: mevzuat_21194-style usage ----------


def test_dogfood_realistic_formula_insert_sequence():
    """Simulated version of the mevzuat_21194 fix flow — three anchors,
    three formulas, applied in order, validated."""
    body = (
        "2) Gelişmiş İDD Yaklaşımının kullanıldığı durumda RA,\n\n"
        "formülü ile hesaplanır. Formülde, BKET gösterir.\n\n"
        "c) 0 <TO< 1 için RA aşağıdaki formül ile hesaplanır.\n\n"
        "Formülde,\n\n"
        "-          R, aşağıdaki formül ile hesaplanan korelasyonu,\n\n"
        "ifade eder.\n"
    )
    ops = [
        Insert(
            "2) Gelişmiş İDD Yaklaşımının kullanıldığı durumda RA,",
            r"$$RA = \max\{0;\ 12{,}5 \times (THK - BKET)\}$$",
        ),
        Insert(
            "c) 0 <TO< 1 için RA aşağıdaki formül ile hesaplanır.",
            r"$$RA = \left[THK \cdot N(\ldots)\right] \cdot 12{,}5$$",
        ),
        Insert(
            "-          R, aşağıdaki formül ile hesaplanan korelasyonu,",
            r"$$R = 0{,}12 \times \frac{1 - e^{-50 \cdot TO}}{1 - e^{-50}}$$",
        ),
    ]
    out = apply_ops(body, ops)
    assert out.count("$$") == 6
    assert validate_latex(out) == []
