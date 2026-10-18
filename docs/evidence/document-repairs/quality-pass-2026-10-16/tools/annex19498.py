"""mevzuat_19498 annexes from the official archive mevzuat.gov.tr/MevzuatMetin/yonetmelik/7.5.19498-ek.zip:
EK-1 and EK-2 (.xlsx, the LCR reporting forms whose rates Madde 9 refers to) were never extracted because the ingest
annex reader handles .docx only; EK-3 (.docx) was. The forms are written as Markdown tables, one per block of the
sheet (header rows: Dikkate Alınma Oranları / Çıkış Oranı / Giriş Oranı); the rate cells carry Excel's percent
format (numFmt 9, '0%') and are written in the corpus' Turkish style (%100); text cells are kept as typed.
EK-3's three formulas are written as LaTeX (the docx prints them as text). usage: annex19498.py SP EK.zip [--write]
"""

import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path

from defusedxml import ElementTree as ET  # the archive is downloaded input

SP = Path(sys.argv[1])
ARCHIVE = Path(sys.argv[2]).read_bytes()
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
M = f"{{{NS['m']}}}"


def col(ref):
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def cells(xlsx):
    zf = zipfile.ZipFile(io.BytesIO(xlsx))
    shared = [
        "".join(t.text or "" for t in si.iter(M + "t"))
        for si in ET.fromstring(zf.read("xl/sharedStrings.xml")).findall("m:si", NS)
    ]
    xfs = [int(x.get("numFmtId", 0)) for x in ET.fromstring(zf.read("xl/styles.xml")).find("m:cellXfs", NS)]
    rows = {}
    for c in ET.fromstring(zf.read("xl/worksheets/sheet1.xml")).iter(M + "c"):
        v = c.find("m:v", NS)
        if v is None:
            continue
        if c.get("t") == "s":
            val = shared[int(v.text)].strip()
        else:
            assert xfs[int(c.get("s", 0))] == 9, c.get("r")  # every number on these forms is a '0%' rate
            pct = float(v.text) * 100
            assert abs(pct - round(pct)) < 1e-9, (c.get("r"), v.text)
            val = f"%{round(pct)}"
        if val:
            rows.setdefault(int(re.search(r"\d+", c.get("r")).group(0)), {})[col(c.get("r"))] = re.sub(r"\s+", " ", val)
    return rows


def esc(s):
    return s.replace("|", "\\|")


def annex_md(xlsx):
    rows = cells(xlsx)
    out, table, cols = [], [], None
    for r in sorted(rows):
        c = rows[r]
        header = c.get(2) in ("Dikkate Alınma Oranları", "Çıkış Oranı") or c.get(3) == "Giriş Oranı"
        if header:
            if table:
                out.append("\n".join(table))
            rate_cols = [k for k in (2, 3) if k in c]
            cols = [0, 1, *rate_cols, 4, 5]
            names = [c.get(0, ""), c.get(1, ""), *[c[k] for k in rate_cols], c.get(4, "YP"), c.get(5, "TP+YP")]
            if names[-2:] != ["YP", "TP+YP"]:
                names[-2:] = ["YP", "TP+YP"]
            table = ["| " + " | ".join(esc(n) for n in names) + " |", "|" + "---|" * len(names)]
            continue
        footnote = len(c) == 1 and next(iter(c.values())).startswith("*")
        if cols is None or footnote:  # preamble rows, the closing footnote
            if table:
                out.append("\n".join(table))
                table = []
            out.append("\n".join(esc(c[k]) for k in sorted(c)))
            continue
        vals = [c.get(k, "") for k in cols]
        if len(cols) == 5 and not vals[2] and 3 in c:  # single-rate block whose rate sits in column D
            vals[2] = c[3]
        table.append("| " + " | ".join(esc(v) for v in vals) + " |")
    if table:
        out.append("\n".join(table))
    return "\n\n".join(out)


with zipfile.ZipFile(io.BytesIO(ARCHIVE)) as zf:
    ek1 = annex_md(zf.read("EK 1.xlsx"))
    ek2 = annex_md(zf.read("EK 2.xlsx"))

T = lambda s: "\\text{" + s.replace("%", "\\%") + "}"  # noqa: E731
EK3 = "\n\n".join(
    [  # the docx prints each left-hand side on its own '(a) … =' line, then the right-hand side
        "EK:3",
        "Yüksek kaliteli likit varlık stokunun hesaplamasında aşağıdaki formüller kullanılır.",
        "(a) 2B Kalite Likit Varlıklar için %15 Üst Sınır Aşım Tutarı =",
        "$$"
        + T("Maksimum")
        + "\\,["
        + T("Düzeltilmiş 2B Kalite Likit Varlıklar")
        + " - 15/85 \\times ("
        + T("Düzeltilmiş Birinci Kalite Likit Varlıklar")
        + " + "
        + T("Düzeltilmiş 2A Kalite Likit Varlıklar")
        + "),\\ "
        + T("Düzeltilmiş 2B Kalite Likit Varlıklar")
        + " - 15/60 \\times "
        + T("Düzeltilmiş Birinci Kalite Likit Varlıklar")
        + ",\\ 0]$$",
        "(b) İkinci Kalite Likit Varlıklar İçin %40 Üst Sınır Aşım Tutarı =",
        "$$"
        + T("Maksimum")
        + "\\,["
        + T("Düzeltilmiş 2A Kalite Likit Varlıklar")
        + " + "
        + T("Düzeltilmiş 2B Kalite Likit Varlıklar")
        + " - "
        + T("2B Kalite Likit Varlıklar İçin %15 Üst Sınır Aşım Tutarı")
        + " - 2/3 \\times "
        + T("Düzeltilmiş Birinci Kalite Likit Varlıklar")
        + ",\\ 0]$$",
        "(c) Yüksek Kaliteli Likit Varlık Stoku =",
        "$$"
        + T("Birinci Kalite Likit Varlıklar")
        + " + "
        + T("2A Kalite Likit Varlıklar")
        + " + "
        + T("2B Kalite Likit Varlıklar")
        + " - "
        + T("2B Kalite Likit Varlıklar için %15 Üst Sınır Aşım Tutarı")
        + " - "
        + T("İkinci Kalite Likit Varlıklar İçin %40 Üst Sınır Aşım Tutarı")
        + "$$",
    ]
)
# plain "EK-n" lines: the section index opens an `ek` section on them (a "### EK-n" heading is not one, and the
# preceding MADDE 34 would run on over the annexes)
new_annexes = f"## Ekler\n\nEK-1\n\n{ek1}\n\nEK-2\n\n{ek2}\n\nEK-3\n\n{EK3}"

W = SP / "work" / "documents.json"
docs = json.loads(W.read_text())
doc = next(d for d in docs if d["document_id"] == "mevzuat_19498")
old = doc["markdown_content"]
i = old.index("## Ekler\n\n### EK-3.docx\n\n")
tail = old[i:]
assert tail.rstrip().endswith("İkinci Kalite Likit Varlıklar İçin %40 Üst Sınır Aşım Tutarı"), tail[-200:]
new = old[:i] + new_annexes
print(
    "archive sha256",
    hashlib.sha256(ARCHIVE).hexdigest(),
    "| EK-1 table rows",
    ek1.count("\n| "),
    "| EK-2",
    ek2.count("\n| "),
)
(SP / "annex19498.md").write_text(new_annexes)
if "--write" in sys.argv:
    log = SP / "manual" / "mevzuat_19498.json"
    prior = json.loads(log.read_text()) if log.exists() else []
    log.write_text(json.dumps(prior + [{"old": old, "new": new, "annex_xlsx": True}], ensure_ascii=False, indent=1))
    doc["markdown_content"] = new
    W.write_text(json.dumps(docs, ensure_ascii=False) + "\n")
    print("written")
