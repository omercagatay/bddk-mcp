"""Model-output parsing shared by the guards and the merge (adapted from the 2026-11-15 bake-off score.py/tables.py)."""

from __future__ import annotations

import html
import re
import unicodedata
from html.parser import HTMLParser

APOS = str.maketrans(
    {
        "’": "'",
        "‘": "'",
        "`": "'",
        "´": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        "\u00a0": " ",
        "\u2009": " ",
        "\u202f": " ",
        "\u00ad": "",
    }
)
SUPER = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
EDGE = "\"'.,;:()[]{}!?«»*•·…|/\\<>"
LETTER_DIGIT = re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_][,;:])(?=\d)")
MATH_PATTERNS = [
    r"```latex.*?```",
    r"\$\$.*?\$\$",
    r"\\\[.*?\\\]",
    r"\\\(.*?\\\)",
    r"\\begin\{(equation|align|gather|eqnarray)\*?\}.*?\\end\{\1\*?\}",
    r"(?<!\\)\$[^$\n]+?(?<!\\)\$",
]


def tokens(text):
    text = unicodedata.normalize("NFC", text).translate(APOS)
    out = []
    for raw in text.split():
        for part in LETTER_DIGIT.split(raw):
            t = part.strip(EDGE + "-")
            if t and any(ch.isalnum() for ch in t):
                out.append(t)
    return out


def is_num(t):
    return any(c.isdigit() for c in t) and not any(c.isalpha() for c in t)


def delatex(seg):
    s = seg.replace("{,}", ",").replace("{.}", ".").replace("\\%", "%").replace("\\ ", " ").replace("\\,", " ")
    s = re.sub(r"\\(text|mathrm|textrm|mathit|operatorname|mathbf)\s*\{([^{}]*)\}", r" \2 ", s)
    s = re.sub(r"\\[A-Za-z]+", " ", s)
    return re.sub(r"[{}^_$\\&]", " ", s)


def strip_model(text):
    """(plain text without markup, list of LaTeX segments)."""
    math = []

    def grab(m):
        math.append(m.group(0))
        return " "

    for pat in MATH_PATTERNS:
        text = re.sub(pat, grab, text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"<sup>(.*?)</sup>", r" \1", text, flags=re.S | re.I)
    text = re.sub(r"<sub>(.*?)</sub>", r" \1", text, flags=re.S | re.I)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"</?[A-Za-z][^<>]*>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+", lambda m: " " + m.group(0).translate(SUPER) + " ", text)
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*\|?\s*:?-{3,}.*$", " ", text, flags=re.M)
    text = text.replace("**", " ").replace("__", " ").replace("|", " ")
    text = re.sub(r"(?<!\w)[*_](?=\S)|(?<=\S)[*_](?!\w)", " ", text)
    return text, math


def split_stream(text):
    """Output in reading order as [(is_math, chunk)] with markup removed from the plain chunks."""
    spans = []
    for pat in MATH_PATTERNS:
        for m in re.finditer(pat, text, flags=re.S):
            spans.append((m.start(), m.end()))
    spans.sort()
    keep, last = [], -1
    for a, b in spans:
        if a >= last:
            keep.append((a, b))
            last = b
    out, pos = [], 0
    for a, b in keep:
        if a > pos:
            out.append((False, text[pos:a]))
        out.append((True, text[a:b]))
        pos = b
    if pos < len(text):
        out.append((False, text[pos:]))
    res = []
    for is_math, chunk in out:
        if is_math:
            res.append((True, re.sub(r"^```latex|```$|^\$+|\$+$|^\\\[|\\\]$|^\\\(|\\\)$", "", chunk.strip())))
        else:
            res.append((False, strip_model(chunk)[0]))
    return res


class _TP(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables, self.stack = [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self.stack.append({"rows": []})
        elif self.stack and tag == "tr":
            self.stack[-1]["rows"].append([])
        elif self.stack and tag in ("td", "th"):
            t = self.stack[-1]
            if not t["rows"]:
                t["rows"].append([])
            t["rows"][-1].append([int(a.get("rowspan") or 1), int(a.get("colspan") or 1), ""])
        elif self.stack and tag == "br":
            self.handle_data(" ")

    def handle_endtag(self, tag):
        if tag == "table" and self.stack:
            self.tables.append(self.stack.pop())

    def handle_data(self, data):
        if self.stack and self.stack[-1]["rows"] and self.stack[-1]["rows"][-1]:
            self.stack[-1]["rows"][-1][-1][2] += data


def grid(rows, fill_spans=False):
    """(rowspan, colspan, text) rows -> rectangular grid; spanned cells '' (or the span's text if fill_spans)."""
    occ = {}
    for r, row in enumerate(rows):
        c = 0
        for rs, cs, text in row:
            while (r, c) in occ:
                c += 1
            for dr in range(rs):
                for dc in range(cs):
                    occ[(r + dr, c + dc)] = text if (dr, dc) == (0, 0) or fill_spans else ""
            c += cs
    if not occ:
        return []
    nr = max(k[0] for k in occ) + 1
    nc = max(k[1] for k in occ) + 1
    return [[occ.get((r, c), "") for c in range(nc)] for r in range(nr)]


def model_tables(md, fill_spans=False):
    """HTML and pipe tables of a model output as grids of raw cell strings."""
    p = _TP()
    p.feed(md)
    tabs = [grid(t["rows"], fill_spans) for t in p.tables]
    block = []
    for line in md.splitlines() + [""]:
        if line.strip().startswith("|"):
            block.append(line)
        elif block:
            rows = [
                [c.strip() for c in ln.strip().strip("|").split("|")]
                for ln in block
                if not re.match(r"^\s*\|?\s*:?-{2,}", ln)
            ]
            tabs.append(rows)
            block = []
    return [t for t in tabs if t]


def cell_norm(cell):
    if not cell:
        return ""
    s = re.sub(r"\$+([^$]*)\$+", lambda m: " " + delatex(m.group(1)) + " ", cell)
    return " ".join(tokens(strip_model(s)[0]))


def model_table_rows(md):
    """Tables of a model output as rows of (rowspan, colspan, text) cells (pipe tables: spans 1)."""
    p = _TP()
    p.feed(md)
    out = [[[(rs, cs, t) for rs, cs, t in row] for row in t["rows"]] for t in p.tables]
    block = []
    for line in md.splitlines() + [""]:
        if line.strip().startswith("|"):
            block.append(line)
        elif block:
            out.append(
                [
                    [(1, 1, c.strip()) for c in ln.strip().strip("|").split("|")]
                    for ln in block
                    if not re.match(r"^\s*\|?\s*:?-{2,}", ln)
                ]
            )
            block = []
    return [t for t in out if t]
