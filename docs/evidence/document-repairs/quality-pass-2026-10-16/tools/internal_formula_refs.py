"""Record the reviewed internal formula cross-references of quality policy v7.

For every phrase in markdown_quality._REVIEWED_INTERNAL_FORMULA_REFS: each occurrence in the
stored text, the article that holds it, and where the referenced formula sits in the same text.
Run from the repository root:
    .venv/bin/python docs/evidence/document-repairs/quality-pass-2026-10-16/tools/internal_formula_refs.py
"""

import json
from pathlib import Path

from bddk_mcp.quality.markdown_quality import _REVIEWED_INTERNAL_FORMULA_REFS, QUALITY_ASSESSMENT_POLICY_VERSION
from bddk_mcp.store.section_index import extract_document_sections

OUT = Path(__file__).resolve().parents[1] / "policy" / "internal-formula-references.json"
DRAFT_NOTE = (
    "printed reference; in this draft the formula is madde 38 fıkra 1 (the final Tebliğ mevzuat_20029 has it at 40/1)"
)
# (document, phrase) -> referenced provision as printed, and per host article (or "*") the
# (type, ref) of the article that holds the referenced formula plus an anchor of that formula
REFERENTS = {
    ("1312", "40 ıncı maddenin birinci fıkrasında yer alan formüle"): (
        DRAFT_NOTE,
        {"*": ("madde", "38", "TVA = T x (1 – HT – HFX)")},
    ),
    ("1313", "Black-Scholes formülü\nkullanılarak"): (
        "madde 23",
        {"*": ("madde", "23", r"\ln (\mathrm{PF}_i/\mathrm{UF}_i)")},
    ),
    ("1313", "opsiyon formülü\nkullanılarak"): (
        "madde 23",
        {"*": ("madde", "23", r"\ln (\mathrm{PF}_i/\mathrm{UF}_i)")},
    ),
    ("1313", "ikinci fıkrada yer alan formüle"): (
        "fıkra 2 of the same article",
        {
            "30": ("madde", "30", r"D_i^{(\mathrm{FO})} ="),
            "31": ("madde", "31", r"D_j^{(\mathrm{DK})} ="),
            "33": ("madde", "33", r"D_i^{(\mathrm{HS})} ="),
        },
    ),
    ("1313", "dördüncü fıkrada yer alan formüle"): (
        "fıkra 4 of the same article",
        {"59": ("madde", "59", r"\mathrm{SMKT} = \sum \mathrm{RTi} * \mathrm{RA} * \%8")},
    ),
    ("mevzuat_19498", "Ek-3’te yer alan formül"): (
        "EK-3",
        {"*": ("ek", "3", r"\text{Maksimum}\,[\text{Düzeltilmiş 2B Kalite Likit Varlıklar}")},
    ),
    ("mevzuat_20029", "44 üncü maddenin üçüncü fıkrasında yer alan formül"): (
        "madde 44 fıkra 3",
        {"*": ("madde", "44", r"H_M = H_N \sqrt{\frac{T_M}{T_N}}")},
    ),
    ("mevzuat_20029", "40 ıncı maddenin birinci fıkrasında yer alan formüle"): (
        "madde 40 fıkra 1",
        {"*": ("madde", "40", "TVA = T x (1 – HT – HFX)")},
    ),
}


def _host(sections, offset):
    return next(
        (s for s in sections if s.section_type in ("madde", "ek") and s.start_char <= offset < s.end_char),
        None,
    )


def main() -> None:
    docs = {d["document_id"]: d for d in json.loads(Path("seed_data/documents.json").read_text(encoding="utf-8"))}
    records = []
    for doc_id, phrases in _REVIEWED_INTERNAL_FORMULA_REFS.items():
        text = docs[doc_id]["markdown_content"]
        sections = extract_document_sections(doc_id, text)
        for phrase in phrases:
            referenced, anchors = REFERENTS[(doc_id, phrase)]
            start = text.find(phrase)
            assert start >= 0, (doc_id, phrase)
            while start >= 0:
                host = _host(sections, start)
                formula_type, formula_ref, anchor = anchors.get(host.section_ref if host else "", anchors.get("*"))
                formula_article = next(
                    s for s in sections if (s.section_type, s.section_ref) == (formula_type, formula_ref)
                )
                formula_at = text.find(anchor, formula_article.start_char, formula_article.end_char)
                assert formula_at >= 0, (doc_id, phrase, start)
                formula_host = _host(sections, formula_at)
                records.append(
                    {
                        "document_id": doc_id,
                        "document_content_hash": docs[doc_id]["content_hash"],
                        "phrase": phrase,
                        "offset": start,
                        "host": f"{host.section_type} {host.section_ref}" if host else None,
                        "referenced_provision": referenced,
                        "formula_anchor": anchor,
                        "formula_offset": formula_at,
                        "formula_host": f"{formula_host.section_type} {formula_host.section_ref}"
                        if formula_host
                        else None,
                    }
                )
                start = text.find(phrase, start + 1)
    OUT.write_text(
        json.dumps(
            {
                "policy_version": QUALITY_ASSESSMENT_POLICY_VERSION,
                "status": "reviewed_by_assistant_approved_by_owner_for_policy",
                "review_scope": "Each referenced formula is present in the same stored text; legal correctness of "
                "the source's own cross-reference numbering is not assessed.",
                "occurrences": records,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({len(records)} occurrences)")


if __name__ == "__main__":
    main()
