"""Census of official mevzuat annex archives (-ek.zip) for every mevzuat_* document: ekcensus.py SP"""

import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

SP = Path(sys.argv[1])
OUT = SP / "ek"
OUT.mkdir(exist_ok=True)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
SEG = {"1": "kanun", "7": "yonetmelik", "9": "teblig", "3": "yonetmelik", "2": "kanunhukmundekararname"}
res = {}
for d in json.loads(Path("seed_data/documents.json").read_text()):
    did = d["document_id"]
    if not did.startswith("mevzuat_"):
        continue
    n = did.split("_", 1)[1]
    m = re.search(r"MevzuatKod=(\d+)\.(\d+)\.", d["source_url"]) or re.search(
        r"MevzuatTur=(\d+)&MevzuatTertip=(\d+)", d["source_url"]
    )
    tur, tertip = m.group(1), m.group(2)
    url = f"https://www.mevzuat.gov.tr/MevzuatMetin/{SEG.get(tur, 'yonetmelik')}/{tur}.{tertip}.{n}-ek.zip"
    path = OUT / f"{did}-ek.zip"
    code = "cached"
    if not path.exists():
        for _ in range(4):
            code = subprocess.run(
                ["curl", "-s", "-A", UA, "--max-time", "60", "-o", str(path), "-w", "%{http_code}", url],
                capture_output=True,
                text=True,
            ).stdout
            if code in ("200", "404", "302"):
                break
            time.sleep(3)
    members = []
    if path.exists() and path.read_bytes()[:2] == b"PK":
        with zipfile.ZipFile(path) as zf:
            members = [i.filename for i in zf.infolist() if not i.is_dir()]
    elif path.exists():
        path.unlink()
    in_corpus = "## Ekler" in d["markdown_content"]
    res[did] = {"http": code, "members": members, "corpus_has_ekler_section": in_corpus}
    if members:
        print(did, members, "corpus Ekler:", in_corpus)
    time.sleep(1)
(SP / "ekcensus.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
print("with annex archive:", sum(1 for r in res.values() if r["members"]), "of", len(res))
