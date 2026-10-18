"""Text formulas written as LaTeX, each compared with the official source line (page images in the receipt):
textformulas.py SP [--write]. The source prints these as plain text (no sub/superscripts); the LaTeX keeps the printed
symbols and operators ('*' stays '*', the printed multiplication 'x'/'×' becomes \\times, decimal commas {,})."""

import json
import sys
from pathlib import Path

SP = Path(sys.argv[1])
EDITS = {
    "1288": [  # BDDK DokumanGetir/1288, pp. 13, 16, 17
        (
            "\n\nRisk Ağırlıklı Tutar = Ortalama Risk Ağırlığı × Kaldıraç Oranı × Risk Tutarı\n\n",
            "\n\n$$\\text{Risk Ağırlıklı Tutar} = \\text{Ortalama Risk Ağırlığı} \\times \\text{Kaldıraç Oranı} \\times \\text{Risk Tutarı}$$\n\n",
        ),
        ("\n\nA = ((%55 * G) - B)\n\n", "\n\n$$A = ((\\%55 * G) - B)$$\n\n"),
        ("\n\nA = ((%55 * G)- B)- ((%55 * G- C) * D)\n\n", "\n\n$$A = ((\\%55 * G) - B) - ((\\%55 * G - C) * D)$$\n\n"),
    ],
    "mevzuat_21195": [  # GeneratePdf p. 4
        (
            "\n\nRA* = [RA(MKP) x (v – 0,25)/(V – 0,25)] + [RA(A) x (V – v)/(V – 0,25)]\n\n",
            "\n\n$$RA* = [RA(MKP) \\times (v - 0{,}25)/(V - 0{,}25)] + [RA(A) \\times (V - v)/(V - 0{,}25)]$$\n\n",
        ),
    ],
    "mevzuat_39158": [  # GeneratePdf p. 1, definition (ll)
        ("Süreklilik Yüzdesi: MTBF/(MTBF+MTTR) formülü ile", "Süreklilik Yüzdesi: $MTBF/(MTBF+MTTR)$ formülü ile"),
    ],
}
W = SP / "work" / "documents.json"
docs = json.loads(W.read_text())
for d in docs:
    if d["document_id"] not in EDITS:
        continue
    old = new = d["markdown_content"]
    for a, b in EDITS[d["document_id"]]:
        assert new.count(a) == 1, (d["document_id"], a)
        new = new.replace(a, b)
    print(d["document_id"], len(EDITS[d["document_id"]]), "formula(s)")
    if "--write" in sys.argv:
        log = SP / "manual" / f"{d['document_id']}.json"
        prior = json.loads(log.read_text()) if log.exists() else []
        log.write_text(
            json.dumps(
                prior + [{"old": old, "new": new, "text_formulas": len(EDITS[d["document_id"]])}],
                ensure_ascii=False,
                indent=1,
            )
        )
        d["markdown_content"] = new
if "--write" in sys.argv:
    W.write_text(json.dumps(docs, ensure_ascii=False) + "\n")
    print("written")
