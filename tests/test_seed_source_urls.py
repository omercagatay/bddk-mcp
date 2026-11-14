"""Reviewed seed source_url values: https only, and consistent between the two seed files."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

SEED_DIR = Path(__file__).resolve().parents[1] / "seed_data"

# Bakanlar Kurulu Kararı record (Tur 3): the canonical mevzuat.gov.tr address cannot be derived offline.
HTTP_EXCEPTIONS = {"mevzuat_20145885"}


def _load(name: str) -> dict[str, dict]:
    records = json.loads((SEED_DIR / name).read_text(encoding="utf-8"))
    return {record["document_id"]: record for record in records}


def test_no_plain_http_source_url_except_the_listed_exception():
    for name in ("documents.json", "decision_cache.json"):
        offenders = {
            doc_id for doc_id, record in _load(name).items() if (record.get("source_url") or "").startswith("http://")
        }
        assert offenders == HTTP_EXCEPTIONS, name


def test_mevzuat_urls_name_their_document_number():
    for name in ("documents.json", "decision_cache.json"):
        for doc_id, record in _load(name).items():
            url = record.get("source_url") or ""
            parsed = urlparse(url)
            if not doc_id.startswith("mevzuat_") or not parsed.netloc.endswith("mevzuat.gov.tr"):
                continue
            params = parse_qs(parsed.query)
            number = params.get("MevzuatNo", [""])[0] or params.get("MevzuatKod", [""])[0].split(".")[-1]
            assert doc_id == f"mevzuat_{number}", (name, doc_id, url)


def test_seed_files_agree_on_source_url():
    documents = _load("documents.json")
    catalog = _load("decision_cache.json")
    assert documents.keys() == catalog.keys()
    mismatched = [doc_id for doc_id in documents if documents[doc_id]["source_url"] != catalog[doc_id]["source_url"]]
    assert mismatched == []
