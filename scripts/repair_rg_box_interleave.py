"""Source-verified structural repair for RG box-interleaved documents.

Fixes the 'RG atıf kutusu cümleye karışması' extraction artifact in
mevzuat_10746, mevzuat_10731 and mevzuat_42359 by reordering the stored
fragments into the verified legal reading order. Reading order verified
against official public sources (resmigazete.gov.tr 2019/02/13 issue for
10746; RG 10/12/2022-32039 amendment text via alomaliye.com for 10731;
resmigazete.gov.tr 2025/04/09 + lexpera consolidated text for 42359).

Read-only except seed_data/documents.json, which is replaced atomically
after count==1 verification of every replacement. chunks.json is rebuilt
afterwards via scripts/regen_chunks_seed.py (derived artifact).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_PATH = ROOT / "seed_data" / "documents.json"
NEW_METHOD = "markitdown_degraded+structural_repair"

# (old, new) per document. Every old string must occur exactly once.
REPAIRS: dict[str, list[tuple[str, str]]] = {
    "mevzuat_10746": [
        (
            "İlişkin  Yönetmeliğin  19  uncu  maddesi\nİşlemlerine\nGazete’de  yayımlanan  Bankaların  Kredi\n",
            "Gazete’de  yayımlanan  Bankaların  Kredi  İşlemlerine  İlişkin  Yönetmeliğin  19  uncu  maddesi\n",
        ),
    ],
    "mevzuat_10731": [
        (
            "5/9/2013 tarihli ve 28756 sayılı Resmî\ngöre\nBankaların Özkaynaklarına İlişkin\n\nYönetmeliğe\n\nyayımlanan\n\nGazete’de\nhesaplanan konsolide özkaynağı,",
            "5/9/2013 tarihli ve 28756 sayılı Resmî\nGazete’de yayımlanan Bankaların Özkaynaklarına İlişkin Yönetmeliğe göre hesaplanan konsolide özkaynağı,",
        ),
        (
            "i)  Özkaynak:  Bankaların\nYönetmeliğe göre hesaplanan özkaynağı,\n\n(Değişik\n\nibare:RG-10/12/2022-32039)(1)  Özkaynaklarına İlişkin\n",
            "i)  Özkaynak:  Bankaların  (Değişik  ibare:RG-10/12/2022-32039)(1)  Özkaynaklarına İlişkin\nYönetmeliğe göre hesaplanan özkaynağı,\n",
        ),
    ],
    "mevzuat_42359": [
        (
            "g) Nitelikli pay: Kanunun 3 üncü maddesinde tanımlanan nitelikli payı,\nve\nğ) Özkaynak: 5/9/2013 tarihli\n\n28756\n\nsayılı\nbelirlenen\n\nResmî\nusul\n\nGazete’de\nesaslar\n\nve\n\nyayımlanan\nçerçevesinde\n\nBankaların Özkaynaklarına İlişkin\nhesaplanan özkaynağı,\nifade eder.\n",
            "g) Nitelikli pay: Kanunun 3 üncü maddesinde tanımlanan nitelikli payı,\nğ) Özkaynak: 5/9/2013 tarihli ve 28756 sayılı Resmî Gazete’de yayımlanan Bankaların Özkaynaklarına İlişkin Yönetmelikte belirlenen usul ve esaslar çerçevesinde hesaplanan özkaynağı,\nifade eder.\n",
        ),
    ],
}


def main() -> int:
    raw = DOCS_PATH.read_text(encoding="utf-8")
    docs = json.loads(raw)

    # Round-trip identity check before touching anything.
    if json.dumps(docs, ensure_ascii=False) + "\n" != raw:
        print("HATA: documents.json round-trip kimliği sağlanamadı; serileştirme biçimi farklı.")
        return 2

    originals = {d["document_id"]: d for d in docs if d["document_id"] in REPAIRS}
    if set(originals) != set(REPAIRS):
        print("HATA: beklenen belgelerin bir kısmı bulunamadı:", set(REPAIRS) - set(originals))
        return 2

    evidence: list[dict] = []
    for did, reps in REPAIRS.items():
        d = originals[did]
        md = d["markdown_content"]
        original_sha = hashlib.sha256(md.encode()).hexdigest()
        applied = []
        for old, new in reps:
            count = md.count(old)
            if count != 1:
                print(f"HATA {did}: hedef bölge {count} kez bulundu (1 bekleniyor): {old[:70]!r}...")
                return 2
            applied.append({"old": old, "new": new})
            md = md.replace(old, new)
        d["markdown_content"] = md
        d["content_hash"] = hashlib.sha256(md.encode()).hexdigest()
        d["extraction_method"] = NEW_METHOD
        evidence.append(
            {
                "document_id": did,
                "original_markdown_sha256": original_sha,
                "candidate_markdown_sha256": d["content_hash"],
                "replacements": applied,
            }
        )
        print(
            f"{did}: {len(reps)} bölge onarıldı, {original_sha[:12]} -> {d['content_hash'][:12]}, method={NEW_METHOD}"
        )

    tmp = DOCS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(docs, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(DOCS_PATH)
    print("documents.json atomik olarak değiştirildi.")

    # Persist exact diffs for the evidence records.
    for item in evidence:
        d = next(x for x in docs if x["document_id"] == item["document_id"])
        item["source_url"] = d["source_url"]
        item["extraction_method"] = d["extraction_method"]
    Path("/tmp/bddk-qscan/repair-diffs.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
