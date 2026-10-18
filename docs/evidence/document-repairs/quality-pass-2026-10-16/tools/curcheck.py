"""Currency check of every mevzuat.gov.tr document against today's official PDF: curcheck.py SP [--fetch] [ids...]

Downloads mevzuat.gov.tr's generated PDF of the current consolidated text (static MevzuatMetin PDF for 3.5.*) to
SP/cur/{id}.bin and compares word multisets (Turkish-lowercased, LaTeX/table markup and page furniture ignored).
A document whose words differ beyond extraction noise is listed with the differing words for review.
"""

import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

SP = Path(sys.argv[1])
IDS = [a for a in sys.argv[2:] if not a.startswith("--")]
OUT = SP / "cur"
OUT.mkdir(exist_ok=True)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"  # the project ingest client's browser User-Agent (doc_sync.py); curl's default is dropped
TUR = {"1": "Kanun", "7": "KurumVeKurulusYonetmeligi", "9": "Teblig"}
docs = [d for d in json.loads(Path("seed_data/documents.json").read_text()) if d["document_id"].startswith("mevzuat_")]
if IDS:
    docs = [d for d in docs if d["document_id"] in IDS]


def url_for(d):
    n = d["document_id"].split("_", 1)[1]
    m = re.search(r"MevzuatKod=(\d+)\.(\d+)\.", d["source_url"]) or re.search(
        r"MevzuatTur=(\d+)&MevzuatTertip=(\d+)", d["source_url"]
    )
    tur, tertip = m.group(1), m.group(2)
    if tur == "3":
        return f"https://www.mevzuat.gov.tr/MevzuatMetin/{tur}.{tertip}.{n}.pdf"
    return f"https://www.mevzuat.gov.tr/File/GeneratePdf?mevzuatNo={n}&mevzuatTur={TUR[tur]}&mevzuatTertip={tertip}"


def fetch(d):
    path = OUT / f"{d['document_id']}.bin"
    if path.exists() and path.stat().st_size > 1000:
        return path
    for _ in range(6):
        r = subprocess.run(["curl", "-s", "-L", "-A", UA, "--max-time", "90", "-o", str(path), url_for(d)], check=False)
        if r.returncode == 0 and path.exists() and path.read_bytes()[:4] == b"%PDF":
            return path
        time.sleep(5)
    return None


def words(text):
    text = re.sub(r"\$\$.*?\$\$|\$[^$\n]*\$", " ", text, flags=re.S)
    text = text.replace("İ", "i").replace("I", "ı").lower()
    return Counter(w for w in re.findall(r"[a-zçğıöşüâîû]{3,}", text))


report = {}
for d in docs:
    did = d["document_id"]
    path = fetch(d) if "--fetch" in sys.argv else OUT / f"{did}.bin"
    if not path or not path.exists():
        report[did] = {"status": "not_fetched"}
        print(did, "NOT FETCHED")
        continue
    pdf_text = subprocess.run(["pdftotext", "-raw", str(path), "-"], capture_output=True, text=True).stdout
    a, b = words(d["markdown_content"]), words(pdf_text)
    only_doc, only_pdf = a - b, b - a
    report[did] = {
        "status": "compared",
        "only_doc": dict(only_doc.most_common()),
        "only_pdf": dict(only_pdf.most_common()),
    }
    print(
        f"{did}: only_doc={sum(only_doc.values())} only_pdf={sum(only_pdf.values())}  "
        f"doc:{list(only_doc)[:8]} pdf:{list(only_pdf)[:8]}"
    )
(SP / "curcheck.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
