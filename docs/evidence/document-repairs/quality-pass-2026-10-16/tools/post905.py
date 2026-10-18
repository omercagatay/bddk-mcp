"""905 after relay: footnote markers/bodies as Unicode superscripts (the reviewed form of its tables) and
paragraphs split by a page-bottom footnote block rejoined. usage: post905.py SP"""

import re
import sys
from pathlib import Path

SP = Path(sys.argv[1])
p = SP / "rebuild" / "905.md"
t = p.read_text()
SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
MARKERS = [  # printed word/number with its glued marker digits -> superscript marker
    ("alacaklar1,", "alacaklar¹,"),
    ("alacaklar2”", "alacaklar²”"),
    ("gereği3", "gereği³"),
    ("sağlayan6", "sağlayan⁶"),
    ("sınıflarından7", "sınıflarından⁷"),
    ("%3508 =", "%350⁸ ="),
    ("%1.2509 =", "%1.250⁹ ="),
    ("1,9511 değeri", "1,95¹¹ değeri"),
    ("ağırlığı12 %75", "ağırlığı¹² %75"),
    ("1,4613 değeri", "1,46¹³ değeri"),
    ("51.118.421,0514 değeri", "51.118.421,05¹⁴ değeri"),
    ("edilsin.15", "edilsin.¹⁵"),
]
for old, new in MARKERS:
    assert t.count(old) == 1, (old, t.count(old))
    t = t.replace(old, new)
BODIES = [
    "Ülkemizin üniter",
    "Bankaların Sermaye Yeterliliğinin Ölçülmesine ve Değerlendirilmesine İlişkin Yönetmelik EK-1",
    "Yönetmelik’te yer alan tanım",
    "12/7/2012 tarih ve 4834 sayılı Kurul Kararı uyarınca, Türkiye’de",
    "12/7/2012 tarih ve 4834 sayılı Kurul Kararı uyarınca yurtdışında",
    "Yönetmelik 6 ncı maddenin ikinci fıkrasındaki koşulları",
    "KOBİ Perakende Alacaklar veya",
    "Tebliğin 16’ncı maddesinin birinci fıkrasında yer alan risk ağırlığı tablosuna göre, 4",
    "Tebliğin 16’ncı maddesinin birinci fıkrasında yer alan risk ağırlığı tablosuna göre, 5",
    "Menkul kıymetleştirilecek alacakların tamamının",
    "Küsuratlar dikkate",
    "Menkul kıymetleştirilecek alacakların tamamının",
    "Küsuratlar dikkate",
    "Söz konusu değer küsuratların",
    "“Ağırlıklı Ortalama Risk Ağırlığı” hesabı",
]
for n, start in enumerate(BODIES, 1):
    old = f"\n\n{n} {start}"
    assert old in t, old
    t = t.replace(old, f"\n\n{str(n).translate(SUP)} {start}", 1)
# footnote 7's body picked up the next page's first line; it continues the paragraph before footnote 6
t = t.replace(
    "\n\n⁷ KOBİ Perakende Alacaklar veya Diğer Perakende Alacaklar\nkurumsal alacaklar için,",
    "\n\n⁷ KOBİ Perakende Alacaklar veya Diğer Perakende Alacaklar\n\nkurumsal alacaklar için,",
    1,
)
FOOT = re.compile(r"^[⁰¹²³⁴⁵⁶⁷⁸⁹]+ ")
paras = t.split("\n\n")
out, i, joined = [], 0, 0
while i < len(paras):
    a = paras[i]
    j = i + 1
    while j < len(paras) and FOOT.match(paras[j]):
        j += 1
    if (
        j > i + 1
        and j < len(paras)
        and re.search(r"[\w,]$", a.rstrip())
        and not a.startswith("|")
        and re.match(r"[a-zçğıöşü]", paras[j])
    ):
        out.append(a.rstrip() + "\n" + paras[j])
        out.extend(paras[i + 1 : j])
        joined += 1
        i = j + 1
        continue
    out.append(a)
    i += 1
p.write_text("\n\n".join(out))
print("markers", len(MARKERS), "bodies", len(BODIES), "rejoined", joined)
