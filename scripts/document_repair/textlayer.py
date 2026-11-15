"""Stage 1: the text layer as the sync path reads it (``pdftotext -layout``), split per page, and the seed convention.

Poppler writes U+1D400.. math letters as UTF-16 surrogate halves encoded one by one (CESU-8); decoding them as pairs
restores the letters that a plain UTF-8 decode turns into U+FFFD. Mathematical alphanumerics are then folded to plain
letters (NFKC on U+1D400-1D7FF only).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import unicodedata
from pathlib import Path

from bddk_mcp.ocr.base import normalize_pdftotext_layout

from .layout import nfkc_math

SENT_END = re.compile(r"[.:;!?)”\"’]\s*$")


def decode_poppler(raw: bytes) -> str:
    s = raw.decode("utf-8", "surrogatepass")
    s = s.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    return nfkc_math(s.replace("\r\n", "\n"))


def pdftotext_pages(pdf) -> list[str]:
    """The text layer exactly as the sync path makes it (pdftotext -layout over the whole file), split per page."""
    if shutil.which("pdftotext") is None:
        raise RuntimeError("pdftotext (poppler-utils) is not on PATH")
    # an absolute path never starts with "-", so a file name cannot be read as an option
    path = str(Path(pdf).resolve())
    out = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", path, "-"], capture_output=True, check=True, timeout=600
    ).stdout
    pages = decode_poppler(out).split("\f")
    return pages[:-1] if pages and not pages[-1].strip() else pages


def normalise(text: str) -> str:
    """The seed convention of ``pdftotext_layout`` texts (the sync backend's), NFC, one final newline."""
    return unicodedata.normalize("NFC", normalize_pdftotext_layout(text)) + "\n"


def join_pages(page_texts):
    """Join page bodies; a page break inside a sentence (previous page ends without sentence punctuation, next page
    starts lowercase) becomes a plain line break instead of a paragraph break."""
    out = ""
    for t in page_texts:
        t = t.strip("\n")
        if not t:
            continue
        if not out:
            out = t
            continue
        prev_last = out.rstrip().split("\n")[-1].strip()
        first = t.lstrip().split("\n")[0].strip()
        if prev_last and first and not SENT_END.search(prev_last) and first[:1].islower():
            out = out.rstrip() + "\n" + t.lstrip("\n")
        else:
            out = out.rstrip() + "\n\n" + t
    return out
