"""Source-verified repair for BDDK guide 943.

markitdown inserted the running header and printed page numbers into the
body and swapped words at page breaks. The official PDF token multiset matches
the stored Markdown. This rebuilds reading order from that PDF, removing only
the 21 running headers and page numbers 1-21. Footnotes stay. Not a corpus
signature or a Railway publish.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_PATH = ROOT / "seed_data" / "documents.json"
EVIDENCE_PATH = ROOT / "docs" / "evidence" / "document-repairs" / "943-page-furniture.json"
HEADER = "Bankacılık Düzenleme ve Denetleme Kurulu"
NEW_METHOD = "markitdown+structural_repair"
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
FOOTNOTES = (
    ("tanımlamaktadır1", "1"),
    ("kapsamalıdır2", "2"),
)


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def pdf_layout(pdf_path: Path) -> str:
    return subprocess.check_output(
        ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), "-"],
        text=True,
    )


def _strip_page(page: str) -> tuple[str, str | None]:
    lines = [line for line in page.splitlines() if line.strip() != HEADER]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and re.fullmatch(r"\d{1,3}", lines[-1].strip()):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()
    footnote = None
    for index in range(len(lines) - 1, -1, -1):
        if not re.fullmatch(r"\d{1,2}", lines[index].strip()):
            continue
        note = " ".join(line.strip() for line in lines[index + 1 :] if line.strip())
        body_before = any(line.strip() for line in lines[:index])
        if note and body_before:
            footnote = f"{lines[index].strip()} {note}"
            lines = lines[:index]
        break
    body = "\n".join(re.sub(r" {2,}", " ", line.strip()) for line in lines)
    body = re.sub(r"\n{3,}", "\n\n", body).strip("\n")
    return body, footnote


def _ends_sentence(text: str) -> bool:
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in ".!?:" or stripped.endswith(("”", "’"))


def build_candidate(layout: str) -> str:
    pages = [page for page in layout.split("\f") if page.strip()]
    bodies: list[str] = []
    footnotes: dict[str, str] = {}
    for page in pages:
        body, footnote = _strip_page(page)
        if footnote:
            footnotes[footnote.split(" ", 1)[0]] = footnote
        if body:
            bodies.append(body)
    if not bodies:
        raise SystemExit("PDF produced no body")
    text = bodies[0]
    for body in bodies[1:]:
        first = body.splitlines()[0]
        if first[:1].islower() and not _ends_sentence(text):
            text = text.rstrip() + " " + body.lstrip()
        else:
            text = text.rstrip() + "\n\n" + body.lstrip()
    for marker, number in FOOTNOTES:
        footnote = footnotes.get(number)
        if footnote is None or marker not in text:
            raise SystemExit(f"missing footnote {number} or marker {marker}")
        at = text.find(marker)
        end = text.find("\n\n", at)
        end = len(text) if end < 0 else end
        text = text[:end] + "\n\n" + footnote + text[end:]
    return text if text.endswith("\n") else text + "\n"


def expected_removal() -> Counter[str]:
    removed: Counter[str] = Counter()
    for word in HEADER.split():
        removed[word] += 21
    for number in range(1, 22):
        removed[str(number)] += 1
    return removed


def main() -> int:
    pdf_path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/bddk-943/download.bin")
    apply = "--apply" in sys.argv
    layout = pdf_layout(pdf_path)
    pdf_sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    raw = DOCS_PATH.read_text(encoding="utf-8")
    docs = json.loads(raw)
    if json.dumps(docs, ensure_ascii=False) + "\n" != raw:
        raise SystemExit("documents.json round-trip identity failed")
    document = next(item for item in docs if item["document_id"] == "943")
    original = document["markdown_content"]
    if Counter(tokens(layout)) != Counter(tokens(original)):
        raise SystemExit("PDF tokens differ from stored 943; refusing to repair")
    candidate = build_candidate(layout)
    removed = Counter(tokens(original)) - Counter(tokens(candidate))
    added = Counter(tokens(candidate)) - Counter(tokens(original))
    if added or removed != expected_removal():
        raise SystemExit(f"token delta is not only page furniture: added={added} removed={removed}")

    from bddk_mcp.store.section_index import extract_document_sections

    sections = extract_document_sections("943", candidate)
    paragraphs = [section.section_ref for section in sections if section.section_type == "paragraf"]
    principles = [section.section_ref for section in sections if section.section_type == "ilke"]
    ilke6 = next(section for section in sections if section.section_type == "ilke" and section.section_ref == "6")
    if principles != [str(n) for n in range(1, 8)] or paragraphs != [str(n) for n in range(1, 105)]:
        raise SystemExit(f"section identity changed: ilke={principles} paragraf={paragraphs}")
    if HEADER in ilke6.content or "\n10\n" in ilke6.content:
        raise SystemExit("ilke 6 still contains page furniture")
    if "Uzman görüşü kullanımı" not in ilke6.content:
        raise SystemExit("ilke 6 page break was not joined")

    original_sha = hashlib.sha256(original.encode()).hexdigest()
    candidate_sha = hashlib.sha256(candidate.encode()).hexdigest()
    print(f"943 {original_sha[:12]} -> {candidate_sha[:12]} headers=21 page_numbers=1-21")
    if not apply:
        print("dry-run only; pass --apply to write seed")
        return 0
    document["markdown_content"] = candidate
    document["content_hash"] = candidate_sha
    document["extraction_method"] = NEW_METHOD
    document["total_pages"] = 21
    tmp = DOCS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(docs, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(DOCS_PATH)
    evidence = {
        "document_id": "943",
        "status": "applied_to_local_seed_signature_pending",
        "repair_class": "page_furniture_reading_order",
        "source_url": document["source_url"],
        "source_pdf_sha256": pdf_sha,
        "source_pdf_pages": 21,
        "original_markdown_sha256": original_sha,
        "candidate_markdown_sha256": candidate_sha,
        "extraction_method_after": NEW_METHOD,
        "removed": "21 running headers and printed page numbers 1-21",
        "preserved": "All other tokens, including both footnotes",
        "publication": "Local seed only. Corpus signature is now stale. Railway was not changed.",
    }
    EVIDENCE_PATH.write_text(json.dumps(evidence, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {DOCS_PATH} and {EVIDENCE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
