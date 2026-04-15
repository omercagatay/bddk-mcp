# benchmark/test_cases.py
"""30 tool-calling test cases for Phase 1a evaluation.

Written from an internal auditor's perspective. Each case has a Turkish
natural-language question and the expected tool + parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TestCase:
    """A single tool-calling test case."""

    id: int
    question: str
    expected_tool: str
    expected_params: dict = field(default_factory=dict)
    category: str = ""
    is_multi_tool: bool = False
    expected_chain: list[str] = field(default_factory=list)


TEST_CASES: list[TestCase] = [
    # -- Search & Regulatory (1-6) --------------------------------------------
    TestCase(
        id=1,
        question="Sermaye yeterliliği rasyosu hesaplamasına ilişkin güncel düzenlemeler nelerdir?",
        expected_tool="search_bddk_decisions",
        expected_params={"keywords": "sermaye yeterliliği"},
        category="search",
    ),
    TestCase(
        id=2,
        question="Takipteki alacaklara ilişkin karşılık oranları hakkında BDDK kararları var mı?",
        expected_tool="search_bddk_decisions",
        expected_params={"keywords": "takipteki alacak karşılık"},
        category="search",
    ),
    TestCase(
        id=3,
        question="TMSF mevduat sigortası teminat limitiyle ilgili son düzenlemeler?",
        expected_tool="search_document_store",
        expected_params={"query": "TMSF mevduat sigortası teminat limiti"},
        category="search",
    ),
    TestCase(
        id=4,
        question="Katılım bankalarına yönelik faizsiz bankacılık standartları nelerdir?",
        expected_tool="search_document_store",
        expected_params={"query": "katılım bankası faizsiz bankacılık"},
        category="search",
    ),
    TestCase(
        id=5,
        question="BDDK'ya kayıtlı aktif leasing şirketlerini listeler misin?",
        expected_tool="search_bddk_institutions",
        expected_params={"keywords": "leasing", "institution_type": "leasing", "active_only": True},
        category="search",
    ),
    TestCase(
        id=6,
        question="Son dönemde yayımlanan mevzuat duyurularını görebilir miyim?",
        expected_tool="search_bddk_announcements",
        expected_params={"category": "mevzuat"},
        category="search",
    ),
    # -- Bulletin & Data (7-10) -----------------------------------------------
    TestCase(
        id=7,
        question="Bankacılık sektörünün toplam kredi hacmi son bir haftada nasıl değişti?",
        expected_tool="get_bddk_bulletin",
        expected_params={"metric_id": "1.0.1", "currency": "TRY"},
        category="bulletin",
    ),
    TestCase(
        id=8,
        question="Yabancı para mevduat tutarını USD bazında görebilir miyim?",
        expected_tool="get_bddk_bulletin",
        expected_params={"currency": "USD"},
        category="bulletin",
    ),
    TestCase(
        id=9,
        question="Sektörün güncel bilançosunun anlık görüntüsünü alabilir miyim?",
        expected_tool="get_bddk_bulletin_snapshot",
        expected_params={},
        category="bulletin",
    ),
    TestCase(
        id=10,
        question="Aylık detaylı verilerde özkaynak tablosunu 2024 Aralık için göster",
        expected_tool="get_bddk_monthly",
        expected_params={"year": 2024, "month": 12},
        category="bulletin",
    ),
    # -- Analytics (11-14) ----------------------------------------------------
    TestCase(
        id=11,
        question="Son 12 haftada toplam krediler ve toplam mevduat nasıl bir trend izledi?",
        expected_tool="compare_bulletin_metrics",
        expected_params={"metric_ids": "1.0.1,1.0.2"},
        category="analytics",
    ),
    TestCase(
        id=12,
        question="Haftalık bülten verilerinde kredi büyümesindeki eğilim nedir?",
        expected_tool="analyze_bulletin_trends",
        expected_params={"metric_id": "1.0.1"},
        category="analytics",
    ),
    TestCase(
        id=13,
        question="Son bir aydaki düzenleyici değişikliklerin özetini çıkarabilir misin?",
        expected_tool="get_regulatory_digest",
        expected_params={"period": "month"},
        category="analytics",
    ),
    TestCase(
        id=14,
        question="Son kontrolümüzden bu yana yeni BDDK duyurusu var mı?",
        expected_tool="check_bddk_updates",
        expected_params={},
        category="analytics",
    ),
    # -- Document retrieval (15-17) -------------------------------------------
    TestCase(
        id=15,
        question="Şu dökümanın tam metnini incelemek istiyorum: 1291",
        expected_tool="get_bddk_document",
        expected_params={"document_id": "1291"},
        category="document",
    ),
    TestCase(
        id=16,
        question="Bu düzenlemenin geçmiş versiyonlarını görebilir miyim? (1280)",
        expected_tool="get_document_history",
        expected_params={"document_id": "1280"},
        category="document",
    ),
    TestCase(
        id=17,
        question="Doküman deposunda kaç belge var, genel durumu nedir?",
        expected_tool="document_store_stats",
        expected_params={},
        category="document",
    ),
    # -- Admin/health (18-20) -------------------------------------------------
    TestCase(
        id=18,
        question="Sistem sağlık durumunu kontrol edelim",
        expected_tool="health_check",
        expected_params={},
        category="admin",
    ),
    TestCase(
        id=19,
        question="Son dönemde araçların performans metrikleri nasıl?",
        expected_tool="bddk_metrics",
        expected_params={},
        category="admin",
    ),
    TestCase(
        id=20,
        question="Önbelleği yenileyebilir misin?",
        expected_tool="refresh_bddk_cache",
        expected_params={},
        category="admin",
    ),
    # -- Multi-tool scenarios (21-23) -----------------------------------------
    TestCase(
        id=21,
        question="Kredi kartı taksit düzenlemelerini bul ve ilgili dökümanın tam metnini getir",
        expected_tool="search_bddk_decisions",
        expected_params={"keywords": "kredi kartı taksit"},
        category="multi_tool",
        is_multi_tool=True,
        expected_chain=["search_bddk_decisions", "get_bddk_document"],
    ),
    TestCase(
        id=22,
        question="Toplam kredi büyüme trendini analiz et, sonra bu alandaki son düzenlemeleri ara",
        expected_tool="analyze_bulletin_trends",
        expected_params={},
        category="multi_tool",
        is_multi_tool=True,
        expected_chain=["analyze_bulletin_trends", "search_bddk_decisions"],
    ),
    TestCase(
        id=23,
        question="Sektör bilançosunun anlık görüntüsünü al ve mevduat-kredi oranını yorumla",
        expected_tool="get_bddk_bulletin_snapshot",
        expected_params={},
        category="multi_tool",
        is_multi_tool=True,
        expected_chain=["get_bddk_bulletin_snapshot"],
    ),
    # -- Adapted from test_f1_score.py ground truth (24-30) -------------------
    TestCase(
        id=24,
        question="Sermaye yeterliliği rasyosu nasıl hesaplanır?",
        expected_tool="search_document_store",
        expected_params={"query": "sermaye yeterliliği rasyosu hesaplama"},
        category="semantic_search",
    ),
    TestCase(
        id=25,
        question="Faiz oranı riski yönetimi ve stres testi",
        expected_tool="search_document_store",
        expected_params={"query": "faiz oranı riski yönetimi stres testi"},
        category="semantic_search",
    ),
    TestCase(
        id=26,
        question="Kredi kartı limit ve taksit kuralları",
        expected_tool="search_document_store",
        expected_params={"query": "kredi kartı limit taksit kuralları"},
        category="semantic_search",
    ),
    TestCase(
        id=27,
        question="Takipteki alacakların sınıflandırılması ve karşılık oranları",
        expected_tool="search_document_store",
        expected_params={"query": "takipteki alacak sınıflandırma karşılık oranları"},
        category="semantic_search",
    ),
    TestCase(
        id=28,
        question="Katılım bankası faizsiz finans ürünleri murabaha sukuk",
        expected_tool="search_document_store",
        expected_params={"query": "katılım bankası faizsiz finans murabaha sukuk"},
        category="semantic_search",
    ),
    TestCase(
        id=29,
        question="Mevduat sigortası TMSF teminat limiti",
        expected_tool="search_document_store",
        expected_params={"query": "mevduat sigortası TMSF teminat limiti"},
        category="semantic_search",
    ),
    TestCase(
        id=30,
        question="Basel III likidite karşılama oranı",
        expected_tool="search_document_store",
        expected_params={"query": "Basel III likidite karşılama oranı"},
        category="semantic_search",
    ),
]
