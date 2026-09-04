"""Mechanical rank-1 metric for loose section ranking. Prints one integer, exits 0."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from bddk_mcp.store.doc_store import StoredDocumentSection
from bddk_mcp.tools.sections import _search_sections_loose


def _sec(
    doc_id: str,
    section_type: str,
    section_ref: str,
    content: str,
    *,
    start_char: int,
    heading: str = "",
) -> StoredDocumentSection:
    return StoredDocumentSection(
        doc_id=doc_id,
        section_type=section_type,
        section_ref=section_ref,
        heading=heading,
        start_char=start_char,
        end_char=start_char + max(len(content), 1),
        content=content,
        content_hash=f"{doc_id}:{section_type}:{section_ref}:{start_char}",
    )


LYO13 = _sec(
    "mevzuat_10749",
    "madde",
    "13",
    "MADDE 13 Asgari likidite yeterliliği oranı. Toplam likidite yeterlilik oranı yüzde yüzden, "
    "yabancı para likidite yeterlilik oranı yüzde seksenden az olamaz.",
    start_char=9412,
    heading="Asgari likidite yeterliliği oranı",
)
LYO15 = _sec(
    "mevzuat_10749",
    "madde",
    "15",
    "MADDE 15 Oransal sınırlara uyumsuzluk. İkinci vade diliminde arka arkaya iki kez uyumsuzluk "
    "gerçekleşemez. Birinci vade diliminde uyumsuzluk iki hafta içinde giderilir.",
    start_char=10916,
)
LYO_GECICI = _sec(
    "mevzuat_10749",
    "gecici_madde",
    "1",
    "GEÇİCİ MADDE 1 likidite yeterlilik oranı haftalık basit aritmetik ortalaması yüzde beşten az olamaz. "
    "Giderilen uyumsuzluklar dahil altı defadan fazla uyumsuzluk gerçekleştirilemez.",
    start_char=14006,
)
LYO_GOVDE = _sec(
    "mevzuat_10749",
    "govde",
    "2",
    "ÜÇÜNCÜ BÖLÜM Asgari Likidite Yeterlilik Oranı ve Bildirim",
    start_char=9316,
    heading="yapısal başlık yok — gövde/dipnot kalanı",
)
LYO5 = _sec(
    "mevzuat_10749",
    "madde",
    "5",
    "MADDE 5 Toplam likidite yeterlilik oranı hesaplamasında kasa, efektif deposu ve bankalar dikkate alınır.",
    start_char=1686,
)
IPC5464 = _sec(
    "mevzuat_5464",
    "madde",
    "35",
    "MADDE 35 Kurul kararıyla idari para cezası uygulanır. Aykırılık teşkil eden tutarın yüzde biri oranına kadar.",
    start_char=100,
)
IRRBB4 = _sec(
    "mevzuat_42628",
    "madde",
    "4",
    "MADDE 4 Bankacılık hesaplarından kaynaklanan faiz oranı riski standart rasyosu, ekonomik değer "
    "değişimi risk tutarının ana sermayeye bölünmesi suretiyle hesaplanır. Standart rasyo %15’i aşamaz.",
    start_char=3208,
)
IRRBB7 = _sec(
    "mevzuat_42628",
    "madde",
    "7",
    "MADDE 7 Standartlaştırılmaya uyumlu olmayan pozisyonlar, vadesiz mevduat ile erken kapanma riski.",
    start_char=7482,
)
IRRBB11 = _sec(
    "mevzuat_42628",
    "madde",
    "11",
    "MADDE 11 Azami oranlarda aşım oluşması halinde son dönem aşım tutarı özkaynaktan indirilir. "
    "Asgari oranların sağlanamaması halinde nedenler Kuruma bildirilir. Uyumsuzluklar ayrı ayrı değerlendirilir.",
    start_char=18977,
)
KREDI6 = _sec(
    "mevzuat_40519",
    "madde",
    "6",
    "MADDE 6 Bir gerçek ya da tüzel kişiye veya bir risk grubuna kullandırılabilecek kredilerin risk "
    "tutarları toplamı ana sermayenin ve özkaynağın yüzde yirmi beşini aşamaz.",
    start_char=11678,
)
KREDI16 = _sec(
    "mevzuat_40519",
    "madde",
    "16",
    "MADDE 16 Kredi sınırlarında aşım oluşması halinde bankalar nedenlerini derhal Kuruma bildirir. "
    "Özkaynakta düşüş nedeniyle aşım altı ay içinde giderilir.",
    start_char=35322,
)
KREDI5 = _sec(
    "mevzuat_40519",
    "madde",
    "5",
    "MADDE 5 Risk tutarı bankanın ana sermayesinin yüzde beşini aşan risk grupları belirlenirken ekonomik bağımlılık analiz edilir.",
    start_char=8958,
)
NSFR4 = _sec(
    "mevzuat_40203",
    "madde",
    "4",
    "MADDE 4 Net istikrarlı fonlama oranı mevcut istikrarlı fon tutarının gerekli istikrarlı fon tutarına "
    "bölünmesi suretiyle hesaplanır. Üç aylık basit aritmetik ortalaması yüzde yüzden az olamaz. "
    "Kurul asgari bir oran tesis etmeye yetkilidir.",
    start_char=2881,
)
NSFR27 = _sec(
    "mevzuat_40203",
    "madde",
    "27",
    "MADDE 27 Net istikrarlı fonlama oranının asgari oranın altına düşmesi halinde uyumsuzluk giderilir. "
    "Asgari oranların sağlanamaması halinde nedenler Kuruma bildirilir.",
    start_char=9000,
)
NSFR_GOVDE = _sec(
    "mevzuat_40203",
    "govde",
    "1",
    "BEŞİNCİ BÖLÜM Net İstikrarlı Fonlama Oranına İlişkin Raporlama ve Oransal Sınırlara Uyumsuzluk",
    start_char=8500,
)

CASES: list[tuple[str, tuple[str, str, str], list[StoredDocumentSection]]] = [
    (
        "likidite yeterlilik oranı asgari oran yaptırım idari para cezası",
        ("mevzuat_10749", "madde", "13"),
        [LYO_GECICI, LYO_GOVDE, LYO5, IPC5464, LYO13, LYO15],
    ),
    (
        "asgari likidite yeterlilik oranı yüzde uyumsuzluk",
        ("mevzuat_10749", "madde", "15"),
        [LYO_GECICI, LYO_GOVDE, LYO13, LYO15],
    ),
    (
        "standart şok faiz oranı riski oranı yüzde özkaynak aşım yaptırım",
        ("mevzuat_42628", "madde", "4"),
        [IRRBB7, IRRBB11, IRRBB4],
    ),
    (
        "faiz oranı riski standart rasyosu aşım uyumsuzluk",
        ("mevzuat_42628", "madde", "11"),
        [IRRBB7, IRRBB4, IRRBB11],
    ),
    (
        "kredi sınırı yüzde ana sermaye aşım uyumsuzluk idari para cezası",
        ("mevzuat_40519", "madde", "6"),
        [KREDI16, KREDI5, IPC5464, KREDI6],
    ),
    (
        "kredi sınırlarında aşım özkaynak",
        ("mevzuat_40519", "madde", "16"),
        [KREDI6, KREDI5, KREDI16],
    ),
    (
        "net istikrarlı fonlama oranı asgari yüzde",
        ("mevzuat_40203", "madde", "4"),
        [NSFR27, NSFR_GOVDE, NSFR4],
    ),
    (
        "geçici madde likidite yeterlilik",
        ("mevzuat_10749", "gecici_madde", "1"),
        [LYO13, LYO15, LYO_GOVDE, LYO_GECICI],
    ),
]


async def _rank1(query: str, pool: list[StoredDocumentSection]) -> StoredDocumentSection | None:
    async def search(search_query, *, document_id=None, section_type=None, limit=10):
        if search_query == query:
            return []
        return list(pool)

    deps = SimpleNamespace(doc_store=SimpleNamespace(search_document_sections=search))
    hits = await _search_sections_loose(deps, query, document_id=None, section_type=None, limit=4)
    return hits[0] if hits else None


def rank_gold_score() -> int:
    score = 0
    for query, expected, pool in CASES:
        top = asyncio.run(_rank1(query, pool))
        if top is not None and (top.doc_id, top.section_type, top.section_ref) == expected:
            score += 1
    return score


def test_rank_gold_score() -> None:
    assert rank_gold_score() == 8


def main() -> None:
    print(rank_gold_score())


if __name__ == "__main__":
    main()
