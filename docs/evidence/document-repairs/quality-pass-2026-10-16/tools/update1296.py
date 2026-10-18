"""Bring 1296 to its current official PDF (bddk.org.tr/Mevzuat/DokumanGetir/1296, created 31/8/2026): the token diff
against the PDF text layer has exactly three content changes (Kurul Kararı 20/8/2026-11559 amends item 1; the date is
printed 29/05/2025). Source metadata follows the new download. usage: update1296.py SP PDF [--write]"""

import json
import sys
import time
from pathlib import Path

SP = Path(sys.argv[1])
PDF = Path(sys.argv[2])
EDITS = [
    ("29/5/2025 tarihli ve 11216 sayılı Kurul Kararı", "29/05/2025 tarihli ve 11216 sayılı Kurul Kararı"),
    (
        "1. Katılma Hesabı Kaynaklı Pozisyonların Dikkate Alınma Oranı (Madde 4)\n\n",
        "1. Katılma Hesabı Kaynaklı Pozisyonların Dikkate Alınma Oranı (Madde 4)\n"
        "(Değişik: Kurul Kararı-20/8/2026-11559)\n\n",
    ),
    (
        "katılma hesabı kaynaklı pozisyonlar %100 oranında dikkate alınır.",
        "katılma hesabı kaynaklı pozisyonlar %80 oranında dikkate alınır.",
    ),
]
W = SP / "work" / "documents.json"
docs = json.loads(W.read_text())
doc = next(d for d in docs if d["document_id"] == "1296")
old = doc["markdown_content"]
new = old
for a, b in EDITS:
    assert new.count(a) == 1, a
    new = new.replace(a, b)
meta = {
    "downloaded_at": PDF.stat().st_mtime,
    "file_size": PDF.stat().st_size,
    "extracted_at": time.time(),
    "title": " ".join(doc["title"].split()),
}  # the listing title carries a double space
print({k: doc[k] for k in meta}, "->", meta)
if "--write" in sys.argv:
    log = SP / "manual" / "1296.json"
    prior = json.loads(log.read_text()) if log.exists() else []
    log.write_text(
        json.dumps(prior + [{"old": old, "new": new, "official_update": True}], ensure_ascii=False, indent=1)
    )
    doc["markdown_content"] = new
    doc.update(meta)
    W.write_text(json.dumps(docs, ensure_ascii=False) + "\n")
    print("written")
