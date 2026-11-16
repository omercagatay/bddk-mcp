"""Stage 2: detect formula and ruled-table regions per page."""

from __future__ import annotations

from . import textlayer
from .assemble import align
from .layout import bands, load, regions


def lost_math_bands(page, bn, text):
    """Bands holding math glyphs that the text layer (pdftotext) does not reproduce: inline formulas that would
    otherwise stay garbled in the stored text."""
    chars = [c for b in bn for c in sorted(b["chars"], key=lambda c: c["x0"]) if c["t"].strip()]
    aligned = {id(c) for c in align(text, chars) if c is not None}
    return {
        i
        for i, b in enumerate(bn)
        if any((c["math"] or c["cid"]) and c["t"].strip() and id(c) not in aligned for c in b["chars"])
    }


def detect(pdf_path):
    import pypdfium2 as pdfium

    pdf, pages = load(pdf_path)
    doc = pdfium.PdfDocument(str(pdf_path))
    out = []
    texts = textlayer.pdftotext_pages(pdf_path)
    for p, text in zip(pages, texts, strict=False):
        bn = bands(p)
        img = doc[p["page"] - 1].render(scale=150 / 72).to_pil().convert("L")
        for k, r in enumerate(regions(p, bn, img, lost_math_bands(p, bn, text))):
            r.update(page=p["page"], id=f"p{p['page']:03d}_{k}", page_w=p["w"], page_h=p["h"], body=p["body"])
            r["bbox"] = [round(v, 2) for v in r["bbox"]]
            r["cut"] = [round(v, 2) for v in r["cut"]]
            r["cut_ctx"] = [round(v, 2) for v in r["cut_ctx"]]
            out.append(r)
    return pdf, pages, out
