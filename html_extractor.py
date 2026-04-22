"""Rich HTML → markdown converter for mevzuat.gov.tr iframe content.

Replaces the legacy `get_text(separator=" ")` extractor. Preserves:
  - Bold / italic inline runs (including Word-style `style="font-weight:700"`)
  - Formula images (as markdown image refs with original relative paths)
  - Real GFM tables built from <table> (colspan/rowspan flattened)
  - Heading promotion for `<hN>` and BÖLÜM / EK markers
  - List items (`<li>` → `- item`)

Entry point:
    html_to_markdown(html_str) -> str

Public helpers (rendering primitives) are exposed for tests.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, NavigableString, Tag

INLINE_TAGS = {"span", "font", "a", "u", "sub", "sup"}
BOLD_TAGS = {"b", "strong"}
ITALIC_TAGS = {"i", "em"}

# Table-cell deduplication threshold. A handful of mevzuat pages copy the
# entire article body into 3+ sibling <td> cells (hidden from the browser
# view by a CSS layout trick, but surfaced verbatim by any non-rendering
# extractor). Within a single document we suppress any cell whose content
# duplicates a previously-seen cell *and* is at least this many characters
# long. The length gate keeps short labels ("1", "A", "MADDE 1") legitimately
# repeating across rows.
_DEDUP_MIN_CELL_CHARS = 200


def _cell_fingerprint(text: str) -> str:
    """Alphanumeric-only signature used to detect duplicate cells.

    Two cells whose visible prose is the same but which differ in bold
    marker placement, stray whitespace, or trailing punctuation will share
    a fingerprint and dedupe correctly. Short / empty cells collapse to
    a short fingerprint that the caller further gates on length.
    """
    return "".join(c for c in text if c.isalnum())


# Whitespace class includes zero-width space (U+200B) commonly emitted by MS
# Word and the narrow no-break space (U+202F) seen in mevzuat exports.
WS_RE = re.compile(r"[ \t\r\n​ \xa0]+")

HEADING_LEVEL = {
    "h1": "#",
    "h2": "##",
    "h3": "###",
    "h4": "####",
    "h5": "#####",
    "h6": "######",
}

HEADING_RE = re.compile(
    r"^(BİRİNCİ|İKİNCİ|ÜÇÜNCÜ|DÖRDÜNCÜ|BEŞİNCİ|ALTINCI|YEDİNCİ|SEKİZİNCİ|DOKUZUNCU|ONUNCU)\s+BÖLÜM\b",
    re.IGNORECASE,
)
EK_RE = re.compile(r"^EK[-–—]?\s*\d+", re.IGNORECASE)


def _clean(text: str) -> str:
    return WS_RE.sub(" ", text).strip()


def _is_bold(node: Tag) -> bool:
    if node.name in BOLD_TAGS:
        return True
    style = (node.get("style") or "").lower().replace(" ", "")
    return "font-weight:700" in style or "font-weight:bold" in style


def _is_italic(node: Tag) -> bool:
    if node.name in ITALIC_TAGS:
        return True
    style = (node.get("style") or "").lower().replace(" ", "")
    return "font-style:italic" in style


def _render_inline(node, bold: bool = False, italic: bool = False) -> str:
    """Render inline content preserving bold/italic/images/links."""
    if isinstance(node, NavigableString):
        txt = WS_RE.sub(" ", str(node))
        if not txt.strip():
            return " " if txt else ""
        if bold and italic:
            return f"***{txt}***"
        if bold:
            return f"**{txt}**"
        if italic:
            return f"*{txt}*"
        return txt

    if not isinstance(node, Tag):
        return ""

    if node.name == "br":
        return " "

    if node.name == "img":
        src = node.get("src", "")
        alt = node.get("alt", "") or ""
        return f"![{alt}]({src})" if src else ""

    sub_bold = bold or _is_bold(node)
    sub_italic = italic or _is_italic(node)

    pieces: list[tuple[bool, str]] = []  # (from_tag, text)
    for child in node.children:
        txt = _render_inline(child, sub_bold, sub_italic)
        if not txt:
            continue
        is_tag = isinstance(child, Tag) and child.name not in ("br",)
        pieces.append((is_tag, txt))

    parts: list[str] = []
    for i, (is_tag, txt) in enumerate(pieces):
        if i > 0:
            prev_is_tag, prev_txt = pieces[i - 1]
            # Word-origin HTML often splits a line across sibling <span>s without
            # explicit whitespace between them. Insert a space when both sides
            # are tag-rendered and neither boundary already has whitespace —
            # this fixes "YÖNETMELİKBİRİNCİ" concat without inserting spaces
            # inside bold/italic runs that already have proper whitespace.
            if prev_is_tag and is_tag and prev_txt and txt and not prev_txt[-1].isspace() and not txt[0].isspace():
                parts.append(" ")
        parts.append(txt)

    out = "".join(parts)

    if node.name == "a":
        href = node.get("href")
        if href and out.strip():
            return f"[{out.strip()}]({href})"
    return out


def _render_paragraph(p: Tag) -> str:
    text = _render_inline(p)
    text = text.replace("****", "")
    text = WS_RE.sub(" ", text).strip()
    # Move whitespace out of invalid emphasis markers while preserving valid
    # ``**text**``. GFM rejects leading/trailing whitespace inside bold; flip
    # to outside. The `(^|\s)` / `(\s|$)` anchors keep valid runs untouched.
    text = re.sub(r"(^|\s)\*\*(\s+)", r"\1\2**", text)
    text = re.sub(r"(\s+)\*\*(\s|$)", r"**\1\2", text)
    text = re.sub(r"\*\*\s+\*\*", " ", text)
    return WS_RE.sub(" ", text).strip()


def _cell_text(td: Tag) -> str:
    """Render a single table cell to a one-line markdown string."""
    chunks: list[str] = []
    for child in td.children:
        if isinstance(child, NavigableString):
            t = WS_RE.sub(" ", str(child))
            if t.strip():
                chunks.append(t.strip())
        elif isinstance(child, Tag):
            if child.name in ("p", "div"):
                chunks.append(_render_paragraph(child))
            else:
                chunks.append(_render_inline(child))
    text = " ".join(c for c in chunks if c and c.strip())
    text = WS_RE.sub(" ", text).strip()
    return text.replace("|", "\\|")


def _render_table(table: Tag, seen_cells: set[str] | None = None) -> str:
    """Build a markdown table with colspan/rowspan flattened.

    `seen_cells` is a document-scoped set of large cell contents already
    emitted. Any cell whose text is at least `_DEDUP_MIN_CELL_CHARS` long
    and matches a prior cell is blanked so the duplication doesn't reach
    the reader. If dedup leaves the table entirely empty, the whole table
    is skipped.
    """
    rows_raw = table.find_all("tr")
    if not rows_raw:
        return ""

    grid: list[list[str]] = []
    pending: dict[tuple[int, int], str] = {}
    max_cols = 0

    for r, tr in enumerate(rows_raw):
        row: list[str] = []
        c = 0
        for cell in tr.find_all(["td", "th"]):
            while (r, c) in pending:
                row.append(pending.pop((r, c)))
                c += 1
            text = _cell_text(cell)
            colspan = int(cell.get("colspan", 1) or 1)
            rowspan = int(cell.get("rowspan", 1) or 1)
            for i in range(colspan):
                row.append(text if i == 0 else "")
                for rs in range(1, rowspan):
                    pending[(r + rs, c + i)] = text if i == 0 else ""
                c += 1
        while (r, c) in pending:
            row.append(pending.pop((r, c)))
            c += 1
        grid.append(row)
        max_cols = max(max_cols, len(row))

    for row in grid:
        while len(row) < max_cols:
            row.append("")

    if not grid or max_cols == 0:
        return ""

    # Dedup pass: compare cells by alphanumeric fingerprint so cosmetic
    # differences (bold marker placement, whitespace, trailing punctuation)
    # don't defeat the dedup. Blank any cell whose fingerprint has already
    # been emitted in this document.
    if seen_cells is not None:
        for row in grid:
            for i, cell_text in enumerate(row):
                if len(cell_text) < _DEDUP_MIN_CELL_CHARS:
                    continue
                fp = _cell_fingerprint(cell_text)
                if len(fp) < _DEDUP_MIN_CELL_CHARS:
                    continue
                if fp in seen_cells:
                    row[i] = ""
                else:
                    seen_cells.add(fp)
        # If every cell ended up blank after dedup, the table carries no
        # new information — skip it entirely so the reader doesn't see
        # an empty grid.
        if not any(c.strip() for row in grid for c in row):
            return ""

    header_idx = 0
    for r, tr in enumerate(rows_raw):
        if tr.find(["th"]) or tr.find(["b", "strong"]):
            header_idx = r
            break

    if header_idx > 0:
        header = [" / ".join(filter(None, (grid[i][col] for i in range(header_idx + 1)))) for col in range(max_cols)]
        body = grid[header_idx + 1 :]
    else:
        header = grid[0]
        body = grid[1:]

    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * max_cols) + "|"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _walk(container: Tag, out: list[str], seen_cells: set[str] | None = None) -> None:
    for child in container.children:
        if isinstance(child, NavigableString):
            t = str(child).strip()
            if t:
                out.append(t)
            continue
        if not isinstance(child, Tag):
            continue

        name = child.name

        if name in ("script", "style"):
            continue

        if name == "table":
            md = _render_table(child, seen_cells)
            if md:
                out.append(md)
            continue

        if name in HEADING_LEVEL:
            text = _render_paragraph(child)
            if text:
                out.append(f"{HEADING_LEVEL[name]} {text}")
            continue

        if name == "p":
            text = _render_paragraph(child)
            if not text:
                continue
            plain = re.sub(r"[*_]", "", text).strip()
            if HEADING_RE.match(plain) or EK_RE.match(plain):
                out.append(f"## {plain}")
            else:
                out.append(text)
            continue

        if name == "li":
            text = _render_paragraph(child)
            if text:
                out.append(f"- {text}")
            continue

        if name in ("ul", "ol"):
            for li in child.find_all("li", recursive=False):
                txt = _render_paragraph(li)
                if txt:
                    out.append(f"- {txt}")
            continue

        if name in ("div", "section", "article", "body", "main", "header", "footer"):
            _walk(child, out, seen_cells)
            continue

        # Fallback: inline-render anything else as a paragraph.
        txt = _render_inline(child)
        txt = WS_RE.sub(" ", txt).strip()
        if txt:
            out.append(txt)


def html_to_markdown(html: str) -> str:
    """Convert an HTML string to markdown.

    Handles mevzuat.gov.tr iframe content (Word-origin <p>/<span> structure,
    colspan tables, formula images). Returns "" on empty/garbage input.
    """
    if not html or not html.strip():
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()
    body = soup.body or soup
    out: list[str] = []
    seen_cells: set[str] = set()
    _walk(body, out, seen_cells)
    result = "\n\n".join(p for p in (x.strip() for x in out) if p)
    return re.sub(r"\n{3,}", "\n\n", result).strip()
