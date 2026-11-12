"""Measure what search_document_store would show for a fixed query set (maintainer tool, not run in CI).

Loads the real embedding model, opens the corpus file read-only and runs 101
queries in five groups: Turkish in-domain questions with expected documents,
English questions on corpus topics, near-domain questions about instruments
outside the corpus, off-topic questions and gibberish strings. For each group it
prints the raw top-1 cosine, the displayed top-1 relevance, the share of
strong/moderate/weak/keyword/no-result answers and, where documents are
expected, the rank-1 and top-5 counts. Displayed values and bands come from
``VectorStore.search``, so they are what the server would show with the current
code and configuration. Nothing is written except the optional ``--out`` JSON.

This is a calibration on a small query set, not a retrieval evaluation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IN_DOMAIN: list[tuple[str, list[str]]] = [
    ("ikinci grup kredilerin sınıflandırılma koşulları nelerdir", ["mevzuat_22599", "779"]),
    ("donuk alacak olarak sınıflandırma için gecikme gün sayısı", ["mevzuat_22599", "775"]),
    ("TFRS 9 beklenen kredi zararı karşılığı nasıl hesaplanır", ["943", "mevzuat_22599"]),
    ("bankalarda sızma testi kim tarafından ve ne sıklıkla yapılır", ["915", "mevzuat_34360"]),
    ("likidite karşılama oranı yüksek kaliteli likit varlıklar", ["mevzuat_19498"]),
    ("net istikrarlı fonlama oranı asgari oran", ["mevzuat_40203", "rg_32202_20230526_6"]),
    ("kaldıraç oranı nasıl hesaplanır", ["mevzuat_18990"]),
    ("sermaye koruma tamponu ve döngüsel sermaye tamponu oranları", ["mevzuat_18991"]),
    ("bankaların destek hizmeti alabileceği alanlar ve şartlar", ["mevzuat_15481"]),
    ("uzaktan kimlik tespiti görüntülü görüşme şartları", ["mevzuat_38568", "1171"]),
    ("dijital bankaların müşteri türleri ve kredi sınırlamaları", ["mevzuat_39158"]),
    ("sır niteliğindeki müşteri bilgilerinin üçüncü kişilerle paylaşılması", ["mevzuat_39171", "1135"]),
    ("büyük kredi sınırı risk grubu özkaynağın yüzde yirmi beşi", ["mevzuat_40519", "mevzuat_5411"]),
    ("iç denetim birimi ve denetim komitesinin görevleri", ["mevzuat_19864", "mevzuat_10750"]),
    ("İSEDES raporu içeriği", ["951", "mevzuat_19864"]),
    ("bankacılık hesaplarından kaynaklanan faiz oranı riski standart şok", ["mevzuat_42628", "1291", "1296"]),
    ("operasyonel risk kayıp veri tabanı", ["948", "mevzuat_20031"]),
    ("sorunlu alacakların çözümlenmesi stratejisi", ["1040"]),
    ("kredi tahsis sürecinde gelir doğrulama", ["1041", "949"]),
    ("iklimle bağlantılı finansal risklerin yönetimi", ["1282"]),
    ("yeşil varlık oranı hesaplaması", ["mevzuat_42363"]),
    ("varlık yönetim şirketlerinin kuruluş şartları", ["mevzuat_38797"]),
    ("bankaların bağımsız denetimi denetçi rotasyonu", ["mevzuat_20645"]),
    ("konut kredilerinde kredi değer oranı", ["1164"]),
    ("bankalarda ücretlendirme politikası değişken ücret", ["953", "mevzuat_10750"]),
    ("karşı taraf kredi riski yönetimi", ["950"]),
    ("yoğunlaşma riski sektörel yoğunlaşma", ["947"]),
    ("stres testi senaryoları sermaye planlaması", ["946"]),
    ("gerçeğe uygun değer ölçümü değerleme düzeltmeleri", ["945"]),
    ("içsel derecelendirmeye dayalı yaklaşım temerrüt olasılığı tahmini", ["mevzuat_21194", "935", "934"]),
    ("kredi riski azaltım teknikleri finansal teminatlar", ["mevzuat_20029"]),
    ("menkul kıymetleştirme pozisyonları risk ağırlığı", ["mevzuat_21195"]),
    ("bankaların özkaynak hesaplaması çekirdek sermaye indirimleri", ["mevzuat_18799"]),
    ("zamanaşımına uğrayan mevduat", ["mevzuat_10735", "1176"]),
    ("banka çağrı merkezi hizmet seviyesi", ["mevzuat_34535"]),
    ("bankaların yapabileceği bağış ve yardım sınırı", ["mevzuat_10739"]),
    ("sistemik önemli bankalar önlem planı", ["mevzuat_38416", "1167", "mevzuat_21425"]),
    ("ertelenmiş vergi varlığı muhasebeleştirme", ["kgk_tms_12"]),
    ("bankalarda kurumsal yönetim ilkeleri yönetim kurulu yapısı", ["mevzuat_10750"]),
    ("finansal holding şirketleri konsolide denetim", ["mevzuat_10732"]),
]
ENGLISH: list[tuple[str, list[str]]] = [
    ("capital adequacy ratio calculation for banks", ["mevzuat_21192"]),
    ("liquidity coverage ratio high quality liquid assets", ["mevzuat_19498"]),
    ("loan classification and provisioning rules", ["mevzuat_22599"]),
    ("penetration testing requirements for bank information systems", ["915", "mevzuat_34360"]),
    ("interest rate risk in the banking book", ["mevzuat_42628", "1291", "1296"]),
    ("expected credit loss IFRS 9 guideline", ["943"]),
]
NEAR_DOMAIN: list[str] = [
    "sigorta şirketlerinin teknik karşılıkları nasıl hesaplanır",
    "gelir vergisi beyannamesi ne zaman verilir",
    "kira geliri istisna tutarı",
    "anonim şirket genel kurul toplantı nisabı",
    "KVKK açık rıza şartları nelerdir",
    "bireysel emeklilik sisteminde devlet katkısı oranı",
    "kripto varlık hizmet sağlayıcıları lisans şartları",
    "iş kanununa göre kıdem tazminatı hesaplama",
    "gümrük vergisi oranları",
    "kat mülkiyeti kanunu aidat borcu",
]
OFF_TOPIC: list[str] = [
    "dün akşamki futbol maçının sonucu",
    "çikolatalı kek tarifi için gereken malzemeler",
    "İstanbul'da en iyi balık restoranları hangileridir",
    "the quick brown fox jumps over the lazy dog",
    "haftalık hava durumu tahmini Ankara",
    "Python'da liste nasıl sıralanır",
    "how to train a neural network with pytorch",
    "Osmanlı İmparatorluğu ne zaman kuruldu",
    "diş ağrısına ne iyi gelir",
    "en iyi bilim kurgu filmleri",
    "kedi maması önerisi",
    "araba lastiği ne zaman değiştirilir",
    "yoga yapmanın faydaları",
    "Mars gezegenine yolculuk ne kadar sürer",
    "bahçede domates nasıl yetiştirilir",
    "Eurovision şarkı yarışmasını kim kazandı",
    "best hiking trails in the Alps",
    "recipe for sourdough bread",
    "who won the world cup in 2018",
    "symptoms of the common cold",
    "gitar akorları yeni başlayanlar için",
    "üniversite sınavına nasıl hazırlanılır",
    "bebeklerde uyku düzeni",
    "satranç açılışları Sicilya savunması",
    "fotosentez nasıl gerçekleşir",
    "ikinci dünya savaşı ne zaman bitti",
    "en yakın eczane nerede",
    "telefonumun şarjı çabuk bitiyor",
    "düğün davetiyesi metni örnekleri",
    "how to change a flat tire",
    "javascript promise async await explained",
    "balkon çiçekleri bakımı",
    "maraton antrenman programı",
    "kahve demleme yöntemleri",
    "köpek eğitimi otur komutu",
    "Türkiye'nin en yüksek dağı hangisidir",
    "pizza hamuru nasıl yapılır",
    "ayın evreleri nelerdir",
    "ehliyet sınavı soruları",
    "minecraft redstone circuits tutorial",
]
GIBBERISH: list[str] = [
    "asdf qwer zxcv uiop hjkl",
    "lorem ipsum dolor sit amet",
    "xxxxx yyyyy zzzzz",
    "1234 5678 9012",
    "aaaa bbbb cccc dddd",
]

GROUPS: list[tuple[str, list[tuple[str, list[str]]]]] = [
    ("in_domain", IN_DOMAIN),
    ("english", ENGLISH),
    ("near_domain", [(q, []) for q in NEAR_DOMAIN]),
    ("off_topic", [(q, []) for q in OFF_TOPIC]),
    ("gibberish", [(q, []) for q in GIBBERISH]),
]
BANDS = ("strong", "moderate", "weak", "keyword", "no_result")
_CONFIDENCE_BANDS = {"high": "strong", "medium": "moderate", "low": "weak"}
SHOWN_LIMIT = 5


def percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolation percentile (``fraction`` in [0, 1]) of a non-empty list."""
    if not values:
        raise ValueError("percentile of an empty list")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def top_band(hits: list[dict]) -> str:
    """The band the client would see for the first hit, as search_document_store labels it."""
    if not hits:
        return "no_result"
    first = hits[0]
    if first.get("match_type") == "fts_exact":
        return "keyword"
    return _CONFIDENCE_BANDS[str(first.get("confidence"))]


def band_counts(bands: list[str]) -> dict[str, int]:
    """Count answers per band, every band present (zero when unused)."""
    unknown = set(bands) - set(BANDS)
    if unknown:
        raise ValueError(f"unknown band(s): {sorted(unknown)}")
    counter = Counter(bands)
    return {band: counter.get(band, 0) for band in BANDS}


def _distribution(values: list[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"min={min(values):.3f} p10={percentile(values, 0.1):.3f} med={statistics.median(values):.3f} "
        f"p90={percentile(values, 0.9):.3f} max={max(values):.3f}"
    )


def summarize(rows: list[dict]) -> list[str]:
    lines: list[str] = []
    for group, _ in GROUPS:
        group_rows = [row for row in rows if row["group"] == group]
        if not group_rows:
            continue
        raw = [row["raw_top1"] for row in group_rows if row["raw_top1"] is not None]
        shown = [row["shown_top1"] for row in group_rows if row["shown_top1"] is not None]
        counts = band_counts([row["band"] for row in group_rows])
        total = len(group_rows)
        lines.append(f"{group} (n={total})")
        lines.append(f"  raw top-1 cosine:     {_distribution(raw)}")
        lines.append(f"  displayed top-1:      {_distribution(shown)}")
        lines.append(
            "  top-1 band:           "
            + ", ".join(f"{band} {count}/{total} ({count / total:.0%})" for band, count in counts.items())
        )
        lines.append(f"  avg shown hits:       {sum(row['shown_n'] for row in group_rows) / total:.1f}")
        if any(row["expected"] for row in group_rows):
            rank1 = sum(1 for row in group_rows if row["expected_raw_rank"] == 1)
            top5 = sum(1 for row in group_rows if row["expected_raw_rank"] and row["expected_raw_rank"] <= 5)
            in_shown = sum(1 for row in group_rows if row["expected_in_shown"])
            expected_raw = [row["expected_raw"] for row in group_rows if row["expected_raw"] is not None]
            lines.append(f"  expected doc raw:     {_distribution(expected_raw)}")
            lines.append(
                f"  expected doc rank:    rank-1 {rank1}/{total}, top-5 {top5}/{total} (dense); "
                f"in shown results {in_shown}/{total}"
            )
    return lines


def _calibration_line(embedding_model: str) -> str:
    from bddk_mcp.store import vector_store

    calibration_of = getattr(vector_store, "relevance_calibration", None)
    threshold_of = getattr(vector_store, "semantic_threshold", None)
    if calibration_of is None or threshold_of is None:
        threshold = vector_store.SEMANTIC_RELEVANCE_THRESHOLD
        return f"model {embedding_model}: no calibration in this code; semantic threshold {threshold}"
    calibration = calibration_of(embedding_model)
    return (
        f"model {embedding_model}: floor {calibration.floor}, ceiling {calibration.ceiling}, "
        f"raw-cosine threshold {threshold_of(embedding_model)}"
    )


async def run(corpus: Path) -> list[dict]:
    from bddk_mcp.core.config import EMBEDDING_MODEL_NAME
    from bddk_mcp.store.corpus_db import CorpusDB
    from bddk_mcp.store.vector_store import VectorStore

    db = CorpusDB.open(corpus, readonly=True)
    rows: list[dict] = []
    try:
        store = VectorStore(db)
        print(_calibration_line(EMBEDDING_MODEL_NAME))
        for group, queries in GROUPS:
            for query, expected in queries:
                dense = await store._vector_search(query, limit=50, fetch_limit=100)
                shown = await store.search(query, limit=SHOWN_LIMIT)
                ranks = [rank for rank, hit in enumerate(dense, 1) if hit["doc_id"] in expected]
                rows.append(
                    {
                        "group": group,
                        "query": query,
                        "expected": expected,
                        "raw_top1": float(dense[0]["semantic_relevance"]) if dense else None,
                        "expected_raw": max(
                            (float(hit["semantic_relevance"]) for hit in dense if hit["doc_id"] in expected),
                            default=None,
                        ),
                        "expected_raw_rank": ranks[0] if ranks else None,
                        "shown_top1": float(shown[0]["relevance"]) if shown else None,
                        "shown_n": len(shown),
                        "band": top_band(shown),
                        "expected_in_shown": any(hit["doc_id"] in expected for hit in shown),
                        "shown": [
                            {
                                "doc_id": hit["doc_id"],
                                "relevance": hit["relevance"],
                                "semantic_relevance": hit.get("semantic_relevance"),
                                "match_type": hit.get("match_type"),
                            }
                            for hit in shown
                        ],
                    }
                )
    finally:
        db.close()
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True, help="Corpus file to open read-only.")
    parser.add_argument("--out", type=Path, help="Optional JSON file for the per-query results.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.corpus.is_file():
        print(f"Corpus file not found: {args.corpus}", file=sys.stderr)
        return 2
    logging.disable(logging.WARNING)
    rows = asyncio.run(run(args.corpus))
    print("\n".join(summarize(rows)))
    if args.out is not None:
        args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
