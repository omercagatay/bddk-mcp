"""KGK TMS 12 seed record written by scripts/add_kgk_tms_12.py."""

import hashlib
import json
import re
from pathlib import Path

from pydantic import TypeAdapter

from bddk_mcp.tools.contract_types import REGULATION_CATEGORIES, RegulationCategory

SEED = Path(__file__).parents[1] / "seed_data"


def _tms_12() -> dict:
    documents = json.loads((SEED / "documents.json").read_text(encoding="utf-8"))
    return next(document for document in documents if document["document_id"] == "kgk_tms_12")


def test_tms_12_is_a_filterable_kgk_accounting_standard() -> None:
    document = _tms_12()
    assert document["category"] == "Muhasebe Standardı"
    assert document["category"] in REGULATION_CATEGORIES
    assert TypeAdapter(RegulationCategory).validate_python("muhasebe standardi") == "Muhasebe Standardı"
    assert document["source_url"].startswith("https://www.kgk.gov.tr/")
    assert document["content_hash"] == hashlib.sha256(document["markdown_content"].encode("utf-8")).hexdigest()


def test_tms_12_keeps_paragraph_numbers_and_drops_page_furniture() -> None:
    text = _tms_12()["markdown_content"]
    assert text.startswith("TMS 12\nGELİR VERGİLERİ\n")
    assert re.search(r"^24 İndirilebilir bütün geçici farklar için", text, re.MULTILINE)
    assert re.search(r"^4A Bu Standart, Ekonomik İşbirliği", text, re.MULTILINE)
    assert len(re.findall(r"^TMS 12$", text, re.MULTILINE)) == 1
    assert not re.search(r"^\d{1,3}$", text, re.MULTILINE)
