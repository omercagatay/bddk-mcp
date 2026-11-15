"""LaTeX -> the glyphs it renders, in reading order, each with its script position (None / 'sub' / 'sup').

Only whitelisted commands are understood; any other command is reported so the guard can reject the region instead
of silently dropping what the command would draw (the bake-off's delatex() deleted \\lambda, \\dot and overlines).
"""

from __future__ import annotations

import re
import unicodedata

from .layout import nfkc_math

GREEK = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "θ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "rho": "ρ",
    "varrho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "varphi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
}
SYMBOLS = {
    "sum": "∑",
    "prod": "∏",
    "int": "∫",
    "times": "×",
    "cdot": "·",
    "ast": "*",
    "pm": "±",
    "mp": "∓",
    "leq": "≤",
    "le": "≤",
    "geq": "≥",
    "ge": "≥",
    "neq": "≠",
    "ne": "≠",
    "approx": "≈",
    "equiv": "≡",
    "in": "∈",
    "notin": "∉",
    "infty": "∞",
    "partial": "∂",
    "to": "→",
    "rightarrow": "→",
    "Rightarrow": "⇒",
    "ldots": "...",
    "dots": "...",
    "cdots": "⋯",
    "%": "%",
    "$": "$",
    "&": "&",
    "#": "#",
    "_": "_",
    "{": "{",
    "}": "}",
    "lbrace": "{",
    "rbrace": "}",
    "langle": "⟨",
    "rangle": "⟩",
    "vert": "|",
    "lvert": "|",
    "rvert": "|",
    "mid": "|",
    "|": "‖",
    "Vert": "‖",
    "lfloor": "⌊",
    "rfloor": "⌋",
    "lceil": "⌈",
    "rceil": "⌉",
    "prime": "′",
    "circ": "∘",
    "lbrack": "[",
    "rbrack": "]",
    "backslash": "\\",
    "div": "÷",
    "setminus": "∖",
    "cup": "∪",
    "cap": "∩",
    "subset": "⊂",
    "forall": "∀",
    "exists": "∃",
    "neg": "¬",
    "bullet": "•",
    "star": "⋆",
    "colon": ":",
}
WORDS = {
    "min": "min",
    "max": "max",
    "exp": "exp",
    "ln": "ln",
    "log": "log",
    "lim": "lim",
    "sin": "sin",
    "cos": "cos",
    "det": "det",
    "sup": "sup",
    "inf": "inf",
    "arg": "arg",
}
ACCENTS = {
    "dot": "˙",
    "ddot": "¨",
    "hat": "^",
    "widehat": "^",
    "bar": "¯",
    "tilde": "~",
    "widetilde": "~",
    "vec": "→",
    "check": "ˇ",
    "breve": "˘",
    "acute": "´",
    "grave": "`",
}
RULED = {"frac", "dfrac", "tfrac", "cfrac", "overline", "underline"}  # draw a rule, no glyph; args parsed
TEXTLIKE = {"text", "textrm", "textit", "textbf", "textnormal", "mbox", "operatorname"}
FONTLIKE = {"mathrm", "mathit", "mathbf", "mathsf", "mathtt", "boldsymbol", "bm", "mathbb", "mathcal", "pmb"}
SILENT = {
    "left",
    "right",
    "big",
    "Big",
    "bigg",
    "Bigg",
    "bigl",
    "bigr",
    "Bigl",
    "Bigr",
    "biggl",
    "biggr",
    "Biggl",
    "Biggr",
    "displaystyle",
    "textstyle",
    "scriptstyle",
    "limits",
    "nolimits",
    ",",
    ";",
    ":",
    "!",
    " ",
    "quad",
    "qquad",
    "enspace",
    "thinspace",
    "medspace",
    "thickspace",
    "\\",
    "nonumber",
    "middle",
}
SKIPARG = {"hspace", "vspace", "phantom", "hphantom", "vphantom", "label", "tag", "begin", "end"}

TOK = re.compile(r"\\[A-Za-z]+\*?|\\.|\{|\}|\^|_|&|\s+|.", re.S)
NORM = str.maketrans(
    {
        "−": "-",
        "–": "-",
        "—": "-",
        "∗": "*",
        "⋅": "·",
        "∙": "·",
        "ϕ": "φ",
        "ϵ": "ε",
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "\u00a0": " ",
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "…": "...",
        "∶": ":",
    }
)


def norm_glyphs(s):
    """Canonical glyph string: NFKC for mathematical alphanumerics only, then the NORM table, no whitespace."""
    s = unicodedata.normalize("NFC", nfkc_math(s)).translate(NORM)
    return [ch for ch in s if not ch.isspace()]


class Parser:
    def __init__(self, src):
        self.toks = TOK.findall(src)
        self.i = 0
        self.out = []  # (glyph, script)
        self.unknown = []
        self.rules = 0  # \frac, \overline, \sqrt vinculum ...
        self.overlines = 0

    def peek(self):
        while self.i < len(self.toks) and self.toks[self.i].isspace():
            self.i += 1
        return self.toks[self.i] if self.i < len(self.toks) else None

    def emit(self, s, script):
        for g in norm_glyphs(s):
            self.out.append((g, script))

    def group_raw(self):
        """Raw text of the next {...} group (for \\text), or the next token."""
        t = self.peek()
        if t != "{":
            self.i += 1
            return t or ""
        depth, buf = 0, []
        while self.i < len(self.toks):
            t = self.toks[self.i]
            self.i += 1
            if t == "{":
                depth += 1
                if depth == 1:
                    continue
            elif t == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(t)
        return "".join(buf)

    def atom(self, script):
        t = self.peek()
        if t is None:
            return
        if t == "{":
            self.i += 1
            self.seq(script, until="}")
            return
        self.i += 1
        self.token(t, script)

    def seq(self, script, until=None):
        while True:
            t = self.peek()
            if t is None:
                return
            self.i += 1
            if until and t == until:
                return
            self.token(t, script)

    def token(self, t, script):
        if t == "{":
            self.seq(script, until="}")
        elif t == "}":
            return
        elif t in ("^", "_"):
            self.atom(script or ("sup" if t == "^" else "sub"))
        elif t == "&" or t == "~":
            return
        elif t.startswith("\\"):
            name = t[1:].rstrip("*")
            if name in GREEK:
                self.emit(GREEK[name], script)
            elif name in SYMBOLS:
                self.emit(SYMBOLS[name], script)
            elif name in WORDS:
                self.emit(WORDS[name], script)
            elif name in ACCENTS:
                self.emit(ACCENTS[name], script)
                self.atom(script)
            elif name in RULED:
                self.rules += 1
                self.overlines += name == "overline"
                self.atom(script)
                if name.endswith("frac"):
                    self.atom(script)
            elif name == "sqrt":
                self.rules += 1
                self.emit("√", script)
                if self.peek() == "[":
                    self.i += 1
                    while self.peek() not in (None, "]"):
                        t2 = self.toks[self.i]
                        self.i += 1
                        self.token(t2, "sup")
                    self.i += 1
                self.atom(script)
            elif name in TEXTLIKE:
                self.emit(self.group_raw().replace("\\ ", " ").replace("{,}", ","), script)
            elif name in FONTLIKE:
                self.atom(script)
            elif name in SILENT:
                if name in (
                    "left",
                    "right",
                    "middle",
                    "bigl",
                    "bigr",
                    "Bigl",
                    "Bigr",
                    "big",
                    "Big",
                    "bigg",
                    "Bigg",
                    "biggl",
                    "biggr",
                    "Biggl",
                    "Biggr",
                ):
                    nxt = self.peek()
                    if nxt == ".":
                        self.i += 1
            elif name in SKIPARG:
                self.group_raw()
            else:
                self.unknown.append(t)
        else:
            if t.isspace():
                return
            self.emit(t, script)


def parse(src):
    src = src.replace("{,}", ",").replace("{.}", ".")
    p = Parser(src)
    p.seq(None)
    return p
