"""One-off source-bound rebuild of 16290 and 21192; writes review candidates only.

Use native cell coordinates, not PDF border guesses. Never deduplicate source text.
Formula replacements are explicitly reviewed data, never inferred from prose.
"""

import hashlib
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

from bddk_mcp.ingest.html_extractor import _render_table

ROOT = Path(__file__).resolve().parents[4]
AUDIT = ROOT / "quality_reports/repair-audit"
SOURCES = AUDIT / "sources/remaining-review"
FORMULAS = json.loads((SOURCES / "formula-transcriptions.json").read_text())


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def render(path, encoding="utf-8"):
    soup = BeautifulSoup(path.read_text(encoding=encoding), "html.parser")
    body = soup.body
    assert body is not None
    consumed = set()
    tables, images, relocated = [], [], []

    def inline(node):
        if isinstance(node, Comment):
            return ""
        if isinstance(node, NavigableString):
            consumed.add(id(node))
            return str(node)
        if not isinstance(node, Tag):
            return ""
        if node.name == "img":
            src = node.get("src")
            assert src in FORMULAS, (path, src)
            images.append(src)
            formula = FORMULAS[src]
            return "$" + formula + "$" if formula else ""
        if node.name == "br":
            return " "
        text = "".join(inline(child) for child in node.children)
        if node.name in {"sub", "sup"} and clean(text):
            # Keep source indices/powers visible; do not glue them to base numbers.
            operator = "_" if node.name == "sub" else "^"
            return "$" + operator + "{" + clean(text) + "}$"
        return text

    def table(node):
        rows = node.find_all("tr")
        assert all(r.find_parent("table") is node for r in rows), "Nested source grid requires separate review"
        slots, cells = {}, []
        for row_index, row in enumerate(rows):
            col = 0
            for cell in row.find_all(["td", "th"], recursive=False):
                while (row_index, col) in slots:
                    col += 1
                height, width = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
                assert 1 <= height <= len(rows) - row_index and 1 <= width <= 40
                value = clean(inline(cell))
                cells.append({"row": row_index, "column": col, "rowspan": height, "colspan": width, "text": value})
                for r in range(row_index, row_index + height):
                    for c in range(col, col + width):
                        assert (r, c) not in slots, "Overlapping native cells"
                        slots[r, c] = value
                col += width
        width = max(c for _, c in slots) + 1
        grid = [[slots.get((r, c), "") for c in range(width)] for r in range(len(rows))]
        # Reuse the existing Markdown renderer on a fully expanded grid. Merged
        # labels repeat across their actual source span; genuinely blank cells stay blank.
        normalized = BeautifulSoup("<table></table>", "html.parser")
        for row in grid:
            tr = normalized.new_tag("tr")
            for value in row:
                td = normalized.new_tag("td")
                td.string = value
                tr.append(td)
            normalized.table.append(tr)
        md = _render_table(normalized.table)  # No document-wide deduplication.
        tables.append(
            {
                "table": len(tables),
                "rows": len(rows),
                "columns": width,
                "cells": cells,
                "grid": grid,
                "markdown_sha256": sha(md),
            }
        )
        return md

    def walk(node):
        output = []
        for child in node.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                text = clean(inline(child))
                if text:
                    output.append(text)
            elif child.name in {"script", "style"}:
                continue
            elif child.name == "table":
                output.append(table(child))
            elif child.name in {"p", "h1", "h2", "h3", "h4", "h5", "h6"}:
                text = clean(inline(child))
                if not text:
                    continue
                if path.stem == "EK-2" and text == "min(1 yıl;vade)":
                    # Source places the upper bound in the paragraph immediately above the image.
                    relocated.append(text)
                    continue
                if child.name.startswith("h"):
                    text = "## " + text
                elif len(child.find_all("img")) == 1 and not clean(child.get_text()):
                    text = "$$\n" + text.strip("$") + "\n$$"
                output.append(text)
            elif child.name in {"ol", "ul"}:
                start = int(child.get("start", 1))
                for offset, li in enumerate(child.find_all("li", recursive=False)):
                    number = int(li.get("value", start + offset))
                    typ = child.get("type", "1")
                    if child.name == "ul":
                        marker = "-"
                    elif typ in {"a", "A"}:
                        assert 1 <= number <= 26
                        marker = chr(ord(typ) + number - 1) + "."
                    elif typ == "i":
                        marker = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x")[number - 1] + "."
                    else:
                        marker = str(number) + "."
                    parts = walk(li)
                    if parts:
                        parts[0] = marker + " " + parts[0]
                        output.extend(parts)
            else:
                output.extend(walk(child))
        return output

    blocks = walk(body)
    missing = [
        str(s)
        for s in body.descendants
        if isinstance(s, NavigableString)
        and not isinstance(s, Comment)
        and str(s).strip()
        and id(s) not in consumed
        and not s.find_parent(["script", "style"])
    ]
    assert not missing, (path, missing[:5])
    assert len(images) == len(body.find_all("img"))
    assert len(tables) == len(body.find_all("table"))
    md = "\n\n".join(blocks).strip() + "\n"
    # Adjacent inline index/power runs are one math span, not an accidental $$ block.
    md = re.sub(r"(?<=\})\$\$(?=[_^])", "", md)
    field_codes = []
    if path.stem == "EK-2":
        # Two Word QUOTE field instructions have no displayed result. The HTML
        # exporter leaks their instruction math with control delimiters; the DOCX
        # fldChar begin/end structure and PDF pages 38/39 confirm they are not body text.
        for field in (r"$THK_i^{\text{toplam}}$", r"$M_i^{\text{koruma}} B_i$"):
            old = "\x07" + field + " \x03\x08"
            assert md.count(old) == 1
            md = md.replace(old, "")
            field_codes.append(field)
        for old, new in (
            (
                "D$_{j}$ = | D$_{j1}$ | + | D$_{j2 |}$ + | D$_{j3 |}$",
                r"$$" + "\n" + r"D_j = \lvert D_{j1}\rvert + \lvert D_{j2}\rvert + \lvert D_{j3}\rvert" + "\n" + r"$$",
            ),
            (
                "Eklenti$_{j}^{(DK)}$ = | Dj$^{(DK)}$ | * DOVF$_{j}^{(DK)}$",
                r"$$" + "\n" + r"\text{Eklenti}_j^{(DK)} = \lvert Dj^{(DK)}\rvert * DOVF_j^{(DK)}" + "\n" + r"$$",
            ),
        ):
            assert md.count(old) == 1
            md = md.replace(old, new)
    # Preserve source footnote rules literally, not as underscore-run artifacts.
    md = re.sub(r"(?m)^_{10,}$", lambda m: "\\_" * len(m.group()), md)
    return md, {
        "source_path": str(path.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "paragraph_blocks": len(blocks) - len(tables),
        "tables": tables,
        "image_replacements": images,
        "non_displayed_quote_fields": field_codes,
        "relocated_formula_bounds": relocated,
        "unconsumed_source_text_nodes": len(missing),
        "markdown_sha256": sha(md),
    }


if __name__ == "__main__":
    out = AUDIT / "remaining-candidates"
    out.mkdir(exist_ok=True)
    md, evidence = render(SOURCES / "16290.doc", "utf-16")
    assert len(evidence["tables"]) == 112 and not evidence["image_replacements"]
    (out / "mevzuat_16290.md").write_text(md)
    (out / "16290-source-coverage.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    sections, receipts = [], []
    for part in ("21192", "EK-1", "EK-2", "EK-3", "EK-4"):
        md, receipt = render(SOURCES / "html" / f"{part}.html")
        sections.append(md)
        receipts.append(receipt)
        (out / f"21192-{part}.md").write_text(md)
    md = "\n\n".join(sections)
    (out / "mevzuat_21192.md").write_text(md)
    (out / "21192-source-coverage.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n")
    print("Rebuilt native sources:", len(receipts), "21192 parts;", len(evidence["tables"]), "16290 tables.")
