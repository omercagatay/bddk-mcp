"""mevzuat_21194 Ek-1 fıkra 5: the consolidated text marks the paragraph '(Değişik:RG-14/3/2018-30360)' but still
prints the legend of the 2016 amendment (RG 20/1/2016-29599, net satış hâsılatı, no 'C'). RG 30360 MADDE 3 replaced
the whole paragraph; its legend is taken verbatim from the amending Tebliğ's page.
usage: legend21194.py SP RG30360.htm [--write]"""

import html
import json
import re
import sys
from pathlib import Path

SP = Path(sys.argv[1])
page = html.unescape(re.sub(r"<[^>]+>", " ", Path(sys.argv[2]).read_text(encoding="cp1254")))
page = re.sub(r"\s+", " ", page)
m = re.search(r"MADDE 3 – .*?(Formülde, “C” milyon TL.*?korelasyon formülü kullanılır\.)”", page)
new_legend = m.group(1)
assert new_legend.count("“C”") == 1 and "yıllık cirosunu" in new_legend, new_legend

W = SP / "work" / "documents.json"
docs = json.loads(W.read_text())
doc = next(d for d in docs if d["document_id"] == "mevzuat_21194")
old = doc["markdown_content"]
paras = old.split("\n\n")
hits = [
    i
    for i, p in enumerate(paras)
    if p.startswith("Formülde, “S” milyon TL olarak, KOBİ’nin içinde yer aldığı risk grubunun")
]
assert len(hits) == 1, hits
old_legend = re.sub(r"\s+", " ", paras[hits[0]])
assert old_legend.endswith("ikinci fıkrada belirtilen korelasyon formülü kullanılır."), old_legend
assert "net satış hâsılatını" in old_legend and "“C”" not in old_legend
paras[hits[0]] = new_legend
new = "\n\n".join(paras)
print("old:", old_legend)
print("new:", new_legend)
if "--write" in sys.argv:
    log = SP / "manual" / "mevzuat_21194.json"
    prior = json.loads(log.read_text()) if log.exists() else []
    log.write_text(json.dumps(prior + [{"old": old, "new": new, "rg_legend": True}], ensure_ascii=False, indent=1))
    doc["markdown_content"] = new
    W.write_text(json.dumps(docs, ensure_ascii=False) + "\n")
    print("written")
