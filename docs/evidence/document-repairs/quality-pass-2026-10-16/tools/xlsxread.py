"""Read an .xlsx sheet with the standard library (cell values, merged ranges): xlsxread.py FILE.xlsx [ZIP_MEMBER_OF_ARCHIVE]"""

import io
import re
import sys
import zipfile
from xml.etree import ElementTree as ET

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def col(ref):
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def read(data):
    zf = zipfile.ZipFile(io.BytesIO(data))
    shared = []
    if "xl/sharedStrings.xml" in zf.namelist():
        for si in ET.fromstring(zf.read("xl/sharedStrings.xml")).findall("m:si", NS):
            shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
    sheets = sorted(n for n in zf.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml", n))
    out = []
    for sh in sheets:
        root = ET.fromstring(zf.read(sh))
        rows = {}
        for c in root.iter(f"{{{NS['m']}}}c"):
            r = int(re.search(r"\d+", c.get("r")).group(0))
            k = col(c.get("r"))
            v = c.find("m:v", NS)
            t = c.get("t")
            if t == "s" and v is not None:
                val = shared[int(v.text)]
            elif t == "inlineStr":
                val = "".join(x.text or "" for x in c.iter(f"{{{NS['m']}}}t"))
            else:
                val = v.text if v is not None else ""
            if val != "":
                rows.setdefault(r, {})[k] = val
        merges = [m.get("ref") for m in root.iter(f"{{{NS['m']}}}mergeCell")]
        out.append((sh, rows, merges))
    return out


if __name__ == "__main__":
    data = open(sys.argv[1], "rb").read()
    if len(sys.argv) > 2:
        data = zipfile.ZipFile(io.BytesIO(data)).read(sys.argv[2])
    for sh, rows, merges in read(data):
        print("==", sh, "merges:", merges)
        for r in sorted(rows):
            print(r, {k: rows[r][k] for k in sorted(rows[r])})
