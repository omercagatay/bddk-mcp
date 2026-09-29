"""Add KGK's "TMS 12 Gelir Vergileri" to the seed corpus from a manually downloaded PDF.

kgk.gov.tr is deliberately not on the outbound allowlist. The owner downloads the
PDF once; this script checks its SHA-256, extracts it with `pdftotext -layout`,
drops page numbers and the running "TMS 12" header, appends one document to the
three seed artifacts, and refreshes the manifest's artifact facts. The manifest
is left unsigned (`not_configured`); sign it with scripts/sign_corpus_manifest.py
after reviewing the delta. Edit the manifest's prose (id, scope) before running.

Paragraph numbers ("24 İndirilebilir ...") carry no trailing dot, so the section
parser indexes the body as govde windows. No text is invented to change that.

Usage:
    curl -o /tmp/TMS_12.pdf <SOURCE_URL>
    uv run python scripts/add_kgk_tms_12.py /tmp/TMS_12.pdf
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402
from regen_chunks_seed import build_chunk_records  # noqa: E402

from bddk_mcp.core.config import PAGE_SIZE  # noqa: E402
from bddk_mcp.corpus_manifest import canonical_manifest_sha256  # noqa: E402
from bddk_mcp.store.vector_store import _load_embedding_tokenizer  # noqa: E402

SEED = ROOT / "seed_data"
DOCUMENT_ID = "kgk_tms_12"
TITLE = "TMS 12 Gelir Vergileri"
CATEGORY = "Muhasebe Standardı"
SOURCE_URL = "https://www.kgk.gov.tr/Portalv2Uploads/files/Duyurular/v2/TMS_TFRS_Setleri/2026/Mavi_Kitap/TMS/TMS_12.pdf"
PDF_SHA256 = "8fde38087f966cef8f9f556d4b210d26aa87ac81d59ef2a1d504173a2cc5c190"
ACQUIRED_AT = datetime(2026, 9, 29, 13, 56, 2, tzinfo=UTC)
_PLACEHOLDER_SHA = "0" * 64


def extract_markdown(pdf: Path) -> str:
    layout = subprocess.run(
        ["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), "-"], check=True, capture_output=True, text=True
    ).stdout
    lines: list[str] = []
    for line in layout.replace("\f", "\n").splitlines():
        text = line.strip()
        # Page numbers alternate left/right; every page repeats the "TMS 12" header.
        if re.fullmatch(r"\d{1,3}", text) or (text == "TMS 12" and any(lines)):
            continue
        lines.append(re.sub(r" {2,}", " ", text))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def _replace_once(text: str, pattern: str, replacement: str) -> str:
    new, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"manifest pattern not found exactly once: {pattern}")
    return new


def refresh_manifest(path: Path, built_at: str) -> str:
    text = path.read_text(encoding="utf-8")
    text = _replace_once(text, r'^(  source_observed_end: )"[^"]+"$', rf'\g<1>"{ACQUIRED_AT:%Y-%m-%dT%H:%M:%SZ}"')
    text = _replace_once(text, r'^(  corpus_built_at: )"[^"]+"$', rf'\g<1>"{built_at}"')
    for role in ("documents", "chunks", "decision_cache"):
        data = (SEED / f"{role}.json").read_bytes()
        records = len(json.loads(data))
        text = _replace_once(
            text,
            rf"^(  - role: {role}\n    path: {role}\.json\n    sha256: )[0-9a-f]{{64}}(\n    bytes: )\d+(\n    records: )\d+$",
            rf"\g<1>{hashlib.sha256(data).hexdigest()}\g<2>{len(data)}\g<3>{records}",
        )
    unsigned = f'integrity:\n  manifest_sha256: "{_PLACEHOLDER_SHA}"\n  signature_status: not_configured\n'
    text = _replace_once(text, r"^integrity:\n(?:  .+\n?)+", unsigned)
    manifest_sha = canonical_manifest_sha256(yaml.safe_load(text))
    path.write_text(text.replace(_PLACEHOLDER_SHA, manifest_sha), encoding="utf-8")
    return manifest_sha


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    pdf = Path(sys.argv[1])
    pdf_bytes = pdf.read_bytes()
    if hashlib.sha256(pdf_bytes).hexdigest() != PDF_SHA256:
        print("refused: PDF differs from the reviewed KGK file", file=sys.stderr)
        return 1

    documents = json.loads((SEED / "documents.json").read_text(encoding="utf-8"))
    if any(document["document_id"] == DOCUMENT_ID for document in documents):
        print(f"refused: {DOCUMENT_ID} is already in the corpus", file=sys.stderr)
        return 1

    markdown = extract_markdown(pdf)
    document = {
        "document_id": DOCUMENT_ID,
        "title": TITLE,
        "category": CATEGORY,
        "decision_date": "",
        "decision_number": "",
        "source_url": SOURCE_URL,
        "markdown_content": markdown,
        "content_hash": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
        "downloaded_at": ACQUIRED_AT.timestamp(),
        "extracted_at": time.time(),
        "extraction_method": "pdftotext_layout",
        "total_pages": max(1, math.ceil(len(markdown) / PAGE_SIZE)),
        "file_size": len(pdf_bytes),
    }
    chunk_records = build_chunk_records(document, tokenizer=_load_embedding_tokenizer())
    cache = json.loads((SEED / "decision_cache.json").read_text(encoding="utf-8"))
    chunks = json.loads((SEED / "chunks.json").read_text(encoding="utf-8"))

    documents.append(document)
    cache.append(
        {
            "document_id": DOCUMENT_ID,
            "title": TITLE,
            "content": TITLE,
            "decision_date": "",
            "decision_number": "",
            "category": CATEGORY,
            "source_url": SOURCE_URL,
        }
    )
    chunks.extend(chunk_records)
    # Match each artifact's existing serialization byte-for-byte.
    (SEED / "documents.json").write_text(json.dumps(documents, ensure_ascii=False) + "\n", encoding="utf-8")
    (SEED / "decision_cache.json").write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    (SEED / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    # The manifest loader requires corpus_built_at to equal the newest extracted_at.
    built_at = datetime.fromtimestamp(document["extracted_at"], UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    manifest_sha = refresh_manifest(SEED / "corpus_scope.yml", built_at)
    print(f"added {DOCUMENT_ID}: {len(markdown)} chars, {len(chunk_records)} chunks; unsigned manifest {manifest_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
