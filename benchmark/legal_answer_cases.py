"""Development-only legal-answer rubrics, not approved expert ground truth."""

from benchmark.test_cases import TestCase

LEGAL_ANSWER_CASES = [
    TestCase(
        id="legal-local-quotation",
        question="1040 sayılı belgenin 134. paragrafını alıntıla ve neyi karşılaştırdığını açıkla. "
        "Yalnızca yerel metni soruyorum; güncel yürürlük görüşü istemiyorum.",
        expected_tool="get_document_section",
        expected_params={"document_id": "1040", "section_type": "paragraf", "section_ref": "134"},
        expected_documents=["1040"],
        expected_sections=[{"document_id": "1040", "type": "paragraf", "ref": "134"}],
        required_answer_points=[
            "1040 paragraf 134 kimliğiyle doğru ve kesilmemiş doğrudan alıntı verilir.",
            "Tahmin edilen zarar karşılıkları ile gerçekleşen zararlar karşılaştırılır; zorunlu LGD formülü eklenmez.",
            "Yerel metnin içeriği ile güncel hukuki uygulanabilirlik birbirinden ayrılır.",
        ],
        expected_abstention=False,
    ),
    TestCase(
        id="legal-validation-completeness",
        question="943 belgesinin 43. paragrafındaki model validasyonu unsurlarının tamamını, "
        "dış hizmet alımı ve bağımsız gözden geçirme dahil, yerel metne göre açıkla.",
        expected_tool="get_document_section",
        expected_params={"document_id": "943", "section_type": "paragraf", "section_ref": "43"},
        expected_documents=["943"],
        expected_sections=[{"document_id": "943", "type": "paragraf", "ref": "43"}],
        required_answer_points=[
            "Görevler, uzmanlık, geliştirmeden bağımsız doğrulama/onay ve bulguların gecikmeksizin raporlanması.",
            "Dış hizmet alımı bankanın etkinlik ve rehbere uyum sorumluluğunu kaldırmaz.",
            "Kapsam ve metodoloji: krediye uygunluk, güvenilirlik, tutarlılık, doğruluk ve model kısıtları.",
            "Girdiler: geçmiş/güncel/gelecek bilgilerin kalitesi ve güvenilirliği; portföye uygun, doğru ve eksiksiz veri.",
            "Tasarım: teorinin kavramsal uygunluğu ve ekonomik/kredi ortamı ile borçludaki değişiklikleri dikkate alması.",
            "Çıktı: performans standartları oluşturulur; eşiğin önemli ölçüde aşılmasında yeniden düzenleme/geliştirme önlemi.",
            "Belgelendirme süreç/metot değişikliklerini, veri aralığını, sonuçları ve iyileştirici önlemleri içerir.",
            "Belgelendirme düzenli olarak gözden geçirilir ve güncellenir.",
            "Etkinlik ve bağımsızlık düzenli gözden geçirilir; tespitler üst yönetim ve denetim komitesine bildirilir.",
            "943 paragraf 43 doğru kaynak gösterilir; düzenli ifadesi sabit yıllık/aylık sıklığa dönüştürülmez.",
        ],
        expected_abstention=False,
    ),
    TestCase(
        id="legal-accounting-capital-scope",
        question="935 paragraf 30 ile 943 paragraf 43'ü birlikte oku. İDD/THK geriye dönük testi ile "
        "TFRS 9 BKZ model validasyonunu ayır; ilkini tüm muhasebe modellerine otomatik uygulama.",
        expected_tool="get_document_section",
        expected_documents=["935", "943"],
        expected_sections=[
            {"document_id": doc, "type": "paragraf", "ref": ref} for doc, ref in (("935", "30"), ("943", "43"))
        ],
        required_answer_points=[
            "935 paragraf 30 İDD tahminleri TO, THK ve dönüştürme oranını gerçekleşen deneyimlerle karşılaştırır.",
            "943 paragraf 43 BKZ modelinin girdi, tasarım ve çıktı/performans validasyonunu düzenler.",
            "İDD sermaye yaklaşımı muhasebe BKZ kapsamıyla karıştırılmaz; otomatik uygulanabilirlik çıkarılmaz.",
            "İki ayrı iddia kendi belge ve paragrafına bağlanır; eşik, süre veya formül uydurulmaz.",
        ],
        expected_abstention=False,
    ),
    TestCase(
        id="legal-invented-specifics",
        question="943 paragraf 43 ve 1040 paragraf 134, LGD için yıllık test, en az 5 yıllık veri, "
        "%10 kabul eşiği ve zorunlu bir formül belirliyor mu? Yalnızca bu iki hükme dayan.",
        expected_tool="get_document_section",
        expected_documents=["943", "1040"],
        expected_sections=[
            {"document_id": doc, "type": "paragraf", "ref": ref} for doc, ref in (("943", "43"), ("1040", "134"))
        ],
        required_answer_points=[
            "Yıllık sabit sıklık, en az 5 yıllık veri, %10 kabul eşiği ve zorunlu LGD formülü bu hükümlerden doğrulanmaz.",
            "943 paragraf 43 validasyon unsurları ile 1040 paragraf 134 karşılık/gerçekleşen zarar testi ayrı kaynaklanır.",
            "Desteksiz ayrıntılar verilmez; bu iki hükümde yokluğu tüm mevzuatta yokluk olarak genellenmez.",
        ],
        expected_abstention=True,
    ),
    TestCase(
        id="legal-dated-applicability",
        question="943 paragraf 43'ün yerel metni 30 Haziran 2024 tarihinde yürürlükte olan sürüm mü? "
        "Tarihli sürüm kanıtını kontrol et; alıntı eşleşmesini tek başına yeterli sayma.",
        expected_tool="get_document_section",
        expected_params={"document_id": "943", "section_type": "paragraf", "section_ref": "43", "as_of": "2024-06-30"},
        expected_documents=["943"],
        required_answer_points=[
            "Yerel metin/alınabilir alıntı ile 2024-06-30 tarihli doğrulanmış sürüm ayrılır.",
            "Yürürlük ancak gerçek alıntıyla bağlı dated_version kanıtı varsa doğrulanır; aksi halde açıkça belirsizdir.",
            "Eksik kanıt, farklı sürüm veya belirsiz durum açıkça belirtilir; belge URL'si yürürlük kanıtı yapılmaz.",
        ],
        answer_as_of="2024-06-30",
    ),
    TestCase(
        id="legal-missing-provision",
        question="943 belgesinin 999. paragrafını aynen alıntıla. Bulunamazsa komşu hükümle doldurma.",
        expected_tool="get_document_section",
        expected_params={"document_id": "943", "section_type": "paragraf", "section_ref": "999"},
        required_answer_points=[
            "İstenen kesin paragrafın bulunamadığı belirtilir.",
            "Komşu hüküm veya uydurma alıntı 999. paragraf olarak sunulmaz.",
        ],
        expected_abstention=True,
    ),
]
