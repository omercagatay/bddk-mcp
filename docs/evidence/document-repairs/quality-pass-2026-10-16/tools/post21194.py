"""21194 after relay + respace: the GeneratePdf output prints every formula of EK-1/EK-2 as an empty box (no glyphs,
no images), so the reviewed LaTeX of the stored text is grafted at the box positions (checked on the page images).
The stored text is the baseline (pre-pass) document from baseline-documents.json.gz.
usage: post21194.py SP BASELINE.json.gz"""

import gzip
import json
import sys
from pathlib import Path

SP = Path(sys.argv[1])
p = SP / "rebuild" / "mevzuat_21194.md"
t = p.read_text()
base = json.loads(gzip.decompress(Path(sys.argv[2]).read_bytes()))
S = next(d for d in base if d["document_id"] == "mevzuat_21194")["markdown_content"].split("\n\n")


def stored(start):
    hits = [x for x in S if x.startswith(start)]
    assert len(hits) >= 1, start
    return hits[0]


RA_TO1 = stored("$$RA = \\max\\{0;")
RA_IDD = stored(
    "$$RA = \\left[THK \\cdot N\\!\\left(\\frac{G(TO)}{\\sqrt{1-R}} + \\sqrt{\\frac{R}{1-R}} \\cdot G(0{,}999)\\right) - THK \\cdot TO\\right] \\cdot \\frac"
)
RA_PER = stored(
    "$$RA = \\left[THK \\cdot N\\!\\left(\\frac{G(TO)}{\\sqrt{1-R}} + \\sqrt{\\frac{R}{1-R}} \\cdot G(0{,}999)\\right) - THK \\cdot TO\\right] \\cdot 12"
)
R_KUR = stored(
    "$$R = 0{,}12 \\times \\frac{1 - e^{-50 \\cdot TO}}{1 - e^{-50}} + 0{,}24 \\times \\left[1 - \\frac{1 - e^{-50 \\cdot TO}}{1 - e^{-50}}\\right]$$"
)
# paragraph 5 is marked '(Değişik:RG-14/3/2018-30360)'; the stored transcription (max[S, 5], 40, /36) is not that
# formula. Transcribed from the amending Tebliğ's own image (RG 14/3/2018-30360, 20180314-8_dosyalar/image002.gif).
assert stored(
    "$$R = 0{,}12 \\times \\frac{1 - e^{-50 \\cdot TO}}{1 - e^{-50}} + 0{,}24 \\times \\left[1 - \\frac{1 - e^{-50 \\cdot TO}}{1 - e^{-50}}\\right] - 0{,}04"
)
R_KOBI = (
    "$$R = 0{,}12 \\times \\frac{1 - e^{-50 \\cdot TO}}{1 - e^{-50}} + 0{,}24 \\times \\left[1 - \\frac{1 - e^{-50 \\cdot TO}}"
    "{1 - e^{-50}}\\right] - 0{,}04 \\times \\left(1 - \\frac{\\min\\{\\max\\{(C/10),\\ S\\};\\ C\\} - (C/10)}{9C/10}\\right)$$"
)
R_VYT = stored("$$R = 0{,}12 \\times \\frac{1 - e^{-50 \\cdot TO}}{1 - e^{-50}} + 0{,}30")
R_PER = stored("$$R = 0{,}03")
B = stored("$$b = ")
RA_HS = stored("$$RA = \\begin{cases}")
V_EF = stored("$$V = \\frac{\\sum_t t")
V_IMY = stored("$$V = \\min")
LEGEND = [
    stored("$s_{t_k}$ : "),
    stored("$df_{t_k}$ : "),
    stored("$BRT_{t_k}$ : "),
    stored("$EBRT_{t_k}$ : "),
    "$\\Delta t_k$ : $t_k - t_{k-1}$’dir.",
]  # printed as 'Δtk : tk – tk-1’dir.'

AFTER = [  # (paragraph printed before the empty formula box, formula)
    ("2) Gelişmiş İDD Yaklaşımının kullanıldığı durumda RA,", RA_TO1),
    ("c) 0 <TO< 1 için RA aşağıdaki formül ile hesaplanır.", RA_IDD),
    ("- R, aşağıdaki formül ile hesaplanan korelasyonu,", R_KUR),
    ("- “b” aşağıdaki formül ile hesaplanan vade ayarlamasını", B),
    ("hesaplanmasında, aşağıda yer alan korelasyon\nformülü kullanılır.", R_KOBI),
    (
        "6. Volatilitesi yüksek ticari gayrimenkul finansmanı kredileri için ikinci fıkra uyarınca risk ağırlıklarının "
        "hesaplanmasında, aşağıda yer alan korelasyon formülü kullanılır.",
        R_VYT,
    ),
    ("a) TO = 1 için", RA_TO1),
    ("b) 0 <TO< 1 için", RA_PER),
    ("- “R” aşağıdaki formül ile hesaplanan korelasyonu", R_PER),
]
# the bold '-' after 'GEÇİCİ MADDE 1' is overprinted on '(1)' (x 136.7 vs 135.8); the page prints 'GEÇİCİ MADDE 1-(1)'
assert t.count("GEÇİCİ MADDE 1 (1) - 4 üncü") == 1
t = t.replace("GEÇİCİ MADDE 1 (1) - 4 üncü", "GEÇİCİ MADDE 1 - (1) 4 üncü")
paras = t.split("\n\n")
for anchor, f in AFTER:
    hits = [i for i, x in enumerate(paras) if x == anchor or (x.endswith(anchor) and "\n" in anchor)]
    assert len(hits) == 1, (anchor, hits)
    paras.insert(hits[0] + 1, f)


def one(pred, what):
    hits = [i for i, x in enumerate(paras) if pred(x)]
    assert len(hits) == 1, (what, hits)
    return hits[0]


i = one(lambda x: x.startswith("12. Gelişmiş İDD Yaklaşımını kullanan bankalarca nakit akışları belirli"), "V_EF")
paras.insert(i + 1, V_EF)
i = one(
    lambda x: (
        x == "Teşkilatlanmış bir borsada işlem gören hisse senedi yatırımları için %300\n"
        "RA = Diğer tüm hisse senedi yatırımları için %400"
    ),
    "RA_HS",
)
paras[i] = RA_HS
i = one(lambda x: x.startswith("Risk Ağırlıklı Tutar = RA × Risk Tutarı ×\nFormülde, “TOkt”"), "TOkt")
head, rest = paras[i].split("\n", 1)
paras[i : i + 1] = [head + " $[0{,}15 + (160 \\times TO_{kt})]$", rest]
i = one(lambda x: x.startswith("17. Yönetmeliğe ekli Ek-2’nin İkinci Kısmının Üçüncü Bölümünde"), "V_IMY")
# the symbol boxes form a ruled 'table' whose region (p10 y338-365, two legend lines) is dropped: the whole legend
# comes from the stored text (its lines checked against page 10)
assert paras[i + 1] == "Formülde," and paras[i + 5].startswith("∆tk : "), paras[i + 1 : i + 6]
paras[i + 1 : i + 6] = [V_IMY, "Formülde,"] + LEGEND
p.write_text("\n\n".join(paras))
print("grafted", len(AFTER) + 4, "formulas, legend", len(LEGEND))
