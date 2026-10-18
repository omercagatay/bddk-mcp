"""Scan Resmî Gazete daily indexes for acts that amend or repeal a corpus regulation: rgscan.py SP

For every mevzuat_* document the core of its title (without 'Hakkında/İlişkin Yönetmelik/Tebliğ') is matched
against every RG item title of the scanned days. Hits are printed with the issue date for review; the scan also
lists every item that mentions banking/BDDK terms, so an amendment published under a changed name is not missed.
"""

import html
import json
import re
import sys
from pathlib import Path

SP = Path(sys.argv[1])
items = []  # (day, title, href)
for f in sorted((SP / "rgidx").glob("*.htm")):
    t = f.read_text(encoding="cp1254", errors="replace").replace("\r", " ").replace("\n", " ")
    for m in re.finditer(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', t, re.I | re.S):
        s = html.unescape(re.sub(r"<[^>]+>", "", m.group(2)))
        s = re.sub(r"\s+", " ", s).strip(" —–- ")
        if len(s) > 20:
            items.append((f.stem, s, m.group(1)))


def norm(s):
    s = s.replace("İ", "i").replace("I", "ı").lower()
    s = re.sub(r"[^a-zçğıöşüâîû0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


STOP = re.compile(r"\b(hakkında|ilişkin|dair|yönetmelik|yönetmeliği|tebliğ|tebliği|kanunu|kanun|ile|ve|usul|esaslar)\b")
docs = [d for d in json.loads(Path("seed_data/documents.json").read_text()) if d["document_id"].startswith("mevzuat_")]
hits = []
for d in docs:
    core = [w for w in norm(STOP.sub(" ", norm(d["title"]))).split() if len(w) > 3]
    for day, title, href in items:
        nt = norm(title)
        if sum(w in nt for w in core) >= max(3, int(0.8 * len(core))) and re.search(
            r"değişiklik|yürürlükten|kaldırıl", nt
        ):
            hits.append((d["document_id"], day, title, href))
for h in hits:
    print("HIT", *h)
TERMS = re.compile(
    r"bank|bddk|kredi|finansal|faktoring|kiralama|tasarruf finansman|sermaye yeterlili|likidite|"
    r"kaldıraç|katılım|ödeme hizmet|elektronik para|mevduat",
    re.I,
)
print("--- banking-related RG items in the scanned period ---")
for day, title, href in items:
    if TERMS.search(title):
        print(day, "|", title, "|", href)
print("days scanned:", len({i[0][:8] for i in items}), "items:", len(items))
