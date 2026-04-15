# benchmark/terminology.py
"""50 BDDK term-definition pairs for Phase 1c terminology evaluation.

Each term has one correct definition and two plausible distractors.
Format: multiple-choice (1 correct + 2 distractors).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TermQuestion:
    """A terminology multiple-choice question."""

    term: str
    correct_definition: str
    distractors: list[str]


TERMINOLOGY: list[TermQuestion] = [
    TermQuestion(
        term="takipteki alacak",
        correct_definition="Geri ödenmesi şüpheli hale gelen ve belirli süre geciken kredi alacakları",
        distractors=[
            "Banka tarafından aktif olarak takip edilen yüksek getirili yatırımlar",
            "Vadesi gelmemiş ancak yakından izlenen kredi portföyü",
        ],
    ),
    TermQuestion(
        term="sermaye yeterliliği rasyosu (SYR)",
        correct_definition="Bankanın özkaynaklarının risk ağırlıklı varlıklarına oranı, asgari %8 olmalı",
        distractors=[
            "Bankanın toplam mevduatının toplam kredilerine oranı",
            "Bankanın kâr payının toplam sermayeye oranı",
        ],
    ),
    TermQuestion(
        term="karşılık oranı",
        correct_definition="Takipteki alacaklar için ayrılan karşılığın toplam takipteki alacaklara oranı",
        distractors=[
            "Mevduat sigortası kapsamındaki teminat tutarının toplam mevduata oranı",
            "Bankanın likit varlıklarının kısa vadeli yükümlülüklerine oranı",
        ],
    ),
    TermQuestion(
        term="likidite karşılama oranı (LKO)",
        correct_definition="Yüksek kaliteli likit varlıkların 30 günlük net nakit çıkışlarına oranı",
        distractors=[
            "Toplam mevduatın toplam kredilere oranı",
            "Kısa vadeli borçlanma maliyetinin toplam faiz gelirine oranı",
        ],
    ),
    TermQuestion(
        term="TMSF",
        correct_definition="Tasarruf Mevduatı Sigorta Fonu — mevduat sahiplerinin haklarını koruyan kurum",
        distractors=[
            "Türkiye Merkez Sermaye Fonlama — bankalara sermaye desteği sağlayan kurum",
            "Toplam Mali Sigorta Fiyatlandırması — bankacılık maliyetlerini düzenleyen birim",
        ],
    ),
    TermQuestion(
        term="risk ağırlıklı varlıklar",
        correct_definition="Varlıkların risk seviyelerine göre ağırlıklandırılmış toplam tutarı, SYR hesabında payda",
        distractors=[
            "Sadece yüksek riskli olarak sınıflandırılan kredi portföyü",
            "Bankanın bilançosundaki toplam varlık tutarı",
        ],
    ),
    TermQuestion(
        term="murabaha",
        correct_definition="Katılım bankacılığında malın maliyetine kâr eklenerek vadeli satılması işlemi",
        distractors=[
            "Bankanın faiz oranını belirlemek için kullandığı referans gösterge",
            "İslami finans kapsamında verilen uzun vadeli konut kredisi türü",
        ],
    ),
    TermQuestion(
        term="sukuk",
        correct_definition="İslami finans ilkelerine uygun, varlığa dayalı kira sertifikası",
        distractors=[
            "Katılım bankalarının mevduat toplama yöntemi",
            "Bankalar arası kısa vadeli borçlanma aracı",
        ],
    ),
    TermQuestion(
        term="Basel III",
        correct_definition="Bankacılık sektöründe sermaye yeterliliği, likidite ve kaldıraç standartlarını belirleyen uluslararası düzenleme çerçevesi",
        distractors=[
            "Türkiye'de bankacılık lisansı almak için gerekli olan ulusal mevzuat",
            "AB ülkelerinde geçerli olan mevduat sigortası standartları",
        ],
    ),
    TermQuestion(
        term="kaldıraç oranı",
        correct_definition="Ana sermayenin toplam risk pozisyonuna (bilanço içi ve dışı) oranı, asgari %3",
        distractors=[
            "Bankanın toplam borcunun toplam gelirine oranı",
            "Kredi kullandırma hızının mevduat büyümesine oranı",
        ],
    ),
    TermQuestion(
        term="donuk alacak",
        correct_definition="90 günden fazla gecikmiş veya tahsili şüpheli hale gelmiş alacaklar (NPL)",
        distractors=[
            "Vadesi gelmemiş ancak düşük faizli kredi alacakları",
            "Yapılandırılmış ve yeniden takvime bağlanmış alacaklar",
        ],
    ),
    TermQuestion(
        term="özkaynaklar",
        correct_definition="Bankanın ana sermaye ve katkı sermaye toplamından indirimler düşüldükten sonraki tutarı",
        distractors=[
            "Bankanın toplam mevduat ve borçlanma kaynaklarının toplamı",
            "Hissedarların bankadaki pay senetlerinin piyasa değeri",
        ],
    ),
    TermQuestion(
        term="kredi riski",
        correct_definition="Borçlunun yükümlülüğünü yerine getirmemesi nedeniyle bankanın zarara uğrama olasılığı",
        distractors=[
            "Faiz oranlarındaki değişimden kaynaklanan kârlılık riski",
            "Döviz kurundaki dalgalanmalardan doğan bilanço riski",
        ],
    ),
    TermQuestion(
        term="operasyonel risk",
        correct_definition="Yetersiz veya başarısız iç süreçler, kişiler, sistemler veya dış olaylardan kaynaklanan zarar riski",
        distractors=[
            "Bankanın kredi portföyündeki temerrüt olasılığından kaynaklanan risk",
            "Piyasa fiyatlarındaki olumsuz hareketlerden doğan zarar olasılığı",
        ],
    ),
    TermQuestion(
        term="piyasa riski",
        correct_definition="Faiz oranı, döviz kuru ve hisse senedi fiyatlarındaki değişimlerden kaynaklanan zarar riski",
        distractors=[
            "Bankanın müşteri kaybetme olasılığından kaynaklanan gelir riski",
            "Ekonomik kriz dönemlerinde mevduat çekilmesi riski",
        ],
    ),
    TermQuestion(
        term="faiz oranı riski",
        correct_definition="Faiz oranlarındaki değişimlerin bankanın gelir ve ekonomik değeri üzerindeki olumsuz etkisi",
        distractors=[
            "Kredi faiz oranının piyasa ortalamasından yüksek belirlenmesi riski",
            "Mevduat faizlerinin enflasyonun altında kalması sonucu müşteri kaybı",
        ],
    ),
    TermQuestion(
        term="tebliğ",
        correct_definition="BDDK'nın düzenleyici kararlarını duyurmak için yayımladığı resmi hukuki metin",
        distractors=[
            "Bankaların iç denetim raporlarını BDDK'ya sunma formatı",
            "Bankalar arası anlaşmazlıkların çözümü için kullanılan tahkim kararı",
        ],
    ),
    TermQuestion(
        term="genelge",
        correct_definition="BDDK'nın bankalara yönelik uygulama esaslarını belirleyen düzenleyici yazı",
        distractors=[
            "Bakanlar Kurulu tarafından çıkarılan mali yasa teklifi",
            "Bankaların genel kurullarında alınan kararların resmi tutanağı",
        ],
    ),
    TermQuestion(
        term="yönetmelik",
        correct_definition="Kanun hükümlerini uygulamak üzere çıkarılan ayrıntılı düzenleme metni",
        distractors=[
            "Bankaların kendi iç işleyişlerini belirleyen gönüllü kurallar",
            "BDDK'nın denetim raporlarında kullandığı değerlendirme kriteri",
        ],
    ),
    TermQuestion(
        term="kurul kararı",
        correct_definition="BDDK Kurulu'nun banka lisansı, faaliyet izni veya idari yaptırım gibi konularda aldığı resmi karar",
        distractors=[
            "Bankalar birliğinin sektörel standartlar hakkında yayımladığı tavsiye",
            "Merkez Bankası'nın para politikası toplantısında aldığı faiz kararı",
        ],
    ),
    TermQuestion(
        term="mülga düzenleme",
        correct_definition="Yürürlükten kaldırılmış olan eski düzenleme metni",
        distractors=[
            "Henüz yürürlüğe girmemiş taslak halindeki düzenleme",
            "Birden fazla bankayı eş zamanlı etkileyen toplu düzenleme",
        ],
    ),
    TermQuestion(
        term="düzenleme taslağı",
        correct_definition="Görüş almak üzere kamuoyuyla paylaşılan, henüz kesinleşmemiş düzenleme metni",
        distractors=[
            "BDDK'nın iç kullanım için hazırladığı gizli değerlendirme raporu",
            "Yürürlükten kaldırılan düzenlemenin gerekçe belgesi",
        ],
    ),
    TermQuestion(
        term="tekdüzen hesap planı (THP)",
        correct_definition="Bankaların muhasebe kayıtlarında kullanması zorunlu olan standart hesap çerçevesi",
        distractors=[
            "BDDK'nın bankalardan talep ettiği yıllık denetim raporu formatı",
            "Bankaların müşterilerine sunduğu standart hesap açma prosedürü",
        ],
    ),
    TermQuestion(
        term="aktif toplamı",
        correct_definition="Bankanın bilançosunun varlık tarafındaki toplam tutar (krediler, menkul kıymetler, nakit vb.)",
        distractors=[
            "Bankanın sadece kredi portföyünün toplam değeri",
            "Bankanın özkaynak ve mevduatının toplam tutarı",
        ],
    ),
    TermQuestion(
        term="mevduat",
        correct_definition="Gerçek ve tüzel kişilerin bankaya yatırdığı, istenildiğinde veya vadede geri alınabilir paralar",
        distractors=[
            "Bankanın merkez bankasında tutmak zorunda olduğu zorunlu karşılık",
            "Bankaların birbirlerinden aldığı kısa vadeli borçlar",
        ],
    ),
    TermQuestion(
        term="zorunlu karşılık",
        correct_definition="Bankaların mevduatlarının belirli bir oranını TCMB'de bloke tutma yükümlülüğü",
        distractors=[
            "Takipteki alacaklar için ayrılan genel karşılık tutarı",
            "Bankaların sermaye yeterliliği için bulundurması gereken asgari özkaynaklar",
        ],
    ),
    TermQuestion(
        term="ana sermaye (CET1)",
        correct_definition="Ödenmiş sermaye, dağıtılmamış kârlar ve diğer kapsamlı gelirden oluşan en kaliteli sermaye unsuru",
        distractors=[
            "Bankanın toplam borçlanma araçlarının piyasa değeri",
            "Hissedarlara dağıtılmamış temettü tutarı",
        ],
    ),
    TermQuestion(
        term="katkı sermaye",
        correct_definition="Genel kredi karşılıkları ve sermaye benzeri borçlanma araçlarından oluşan ikincil sermaye unsuru",
        distractors=[
            "Devlet tarafından bankalara sağlanan kriz dönemi sermaye desteği",
            "Bankanın yurt dışı şubelerinden transfer ettiği kâr payları",
        ],
    ),
    TermQuestion(
        term="stres testi",
        correct_definition="Bankanın olumsuz ekonomik senaryolar altındaki dayanıklılığını ölçen simülasyon analizi",
        distractors=[
            "Bankanın bilgi sistemlerinin güvenlik açıklarını test eden sızma testi",
            "Personelin iş yükü kapasitesini değerlendiren insan kaynakları testi",
        ],
    ),
    TermQuestion(
        term="net istikrarlı fonlama oranı (NSFR)",
        correct_definition="Mevcut istikrarlı fonlamanın gerekli istikrarlı fonlamaya oranı, asgari %100 olmalı",
        distractors=[
            "Bankanın net kârının toplam fonlama maliyetine oranı",
            "Kısa vadeli borçların uzun vadeli varlıklara oranı",
        ],
    ),
    TermQuestion(
        term="kredi kartı taksitlendirme",
        correct_definition="Kredi kartıyla yapılan harcamanın belirli sayıda eşit aylık taksitlere bölünmesi işlemi",
        distractors=[
            "Kredi kartı limitinin birden fazla kart arasında paylaştırılması",
            "Kredi kartı borcunun farklı bir kredi türüne dönüştürülmesi",
        ],
    ),
    TermQuestion(
        term="yapılandırma",
        correct_definition="Geri ödemede zorluk yaşayan borçlunun kredi koşullarının yeniden düzenlenmesi",
        distractors=[
            "Bankanın organizasyon şemasının yeniden düzenlenmesi",
            "Kredi portföyünün sektörel dağılımının değiştirilmesi",
        ],
    ),
    TermQuestion(
        term="konsolide bilanço",
        correct_definition="Ana banka ve bağlı ortaklıklarının mali tablolarının birleştirilerek sunulması",
        distractors=[
            "Sektördeki tüm bankaların bilançolarının toplu görünümü",
            "Bankanın yurt içi ve yurt dışı şubelerinin ayrı ayrı mali tabloları",
        ],
    ),
    TermQuestion(
        term="finansal kiralama (leasing)",
        correct_definition="Kiracının bir varlığı belirli süre kullanıp süre sonunda satın alma hakkı olan finansman yöntemi",
        distractors=[
            "Bankanın kendi mülklerini geçici olarak kiraya vermesi",
            "Müşteriye düşük faizli tüketici kredisi verilmesi",
        ],
    ),
    TermQuestion(
        term="faktoring",
        correct_definition="İşletmelerin vadeli alacaklarını iskontolu olarak faktoring şirketine devretmesi yoluyla erken nakit temini",
        distractors=[
            "Bankaların birbirlerinin risklerini paylaştığı sendikasyon sistemi",
            "Kredi kartı borçlarının otomatik olarak mevduattan tahsil edilmesi",
        ],
    ),
    TermQuestion(
        term="elektronik para kuruluşu",
        correct_definition="BDDK lisansıyla elektronik para ihraç eden ve ödeme hizmeti sunan kuruluş",
        distractors=[
            "İnternet bankacılığı hizmeti veren tüm bankalar",
            "Kripto para alım satımı yapan borsa platformları",
        ],
    ),
    TermQuestion(
        term="iç denetim",
        correct_definition="Bankanın faaliyetlerini bağımsız ve tarafsız olarak değerlendiren kurum içi denetim fonksiyonu",
        distractors=[
            "BDDK müfettişlerinin bankada gerçekleştirdiği yerinde denetim",
            "Bağımsız denetim şirketinin yaptığı mali tablo denetimi",
        ],
    ),
    TermQuestion(
        term="uyum (compliance)",
        correct_definition="Bankanın yasal düzenlemelere ve iç politikalara uygunluğunu sağlayan fonksiyon",
        distractors=[
            "Bankanın müşteri memnuniyetini ölçen kalite yönetim birimi",
            "Bilgi teknolojileri altyapısının standartlara uygunluk sertifikası",
        ],
    ),
    TermQuestion(
        term="suç gelirlerinin aklanmasının önlenmesi (AML)",
        correct_definition="Bankaların şüpheli işlemleri tespit edip MASAK'a bildirmekle yükümlü olduğu mevzuat alanı",
        distractors=[
            "Bankaların vergi kaçakçılığını önlemek için uyguladığı faiz sınırlandırması",
            "Haksız rekabeti önlemek amacıyla bankalar arası bilgi paylaşımı yasağı",
        ],
    ),
    TermQuestion(
        term="MASAK",
        correct_definition="Mali Suçları Araştırma Kurulu — suç gelirlerinin aklanması ve terörün finansmanıyla mücadele eden kurum",
        distractors=[
            "Maliye Bakanlığı Sermaye Aktifleştirme Kurulu — banka sermaye artırımlarını onaylayan birim",
            "Merkezi Analiz ve Strateji Koordinasyon — bankacılık sektörünün stratejik planlamasını yapan kurul",
        ],
    ),
    TermQuestion(
        term="idari para cezası",
        correct_definition="BDDK'nın mevzuata aykırı hareket eden bankalara verdiği mali yaptırım",
        distractors=[
            "Bankaların gecikmiş kredi ödemelerinden tahsil ettiği gecikme faizi",
            "Merkez Bankası'nın zorunlu karşılık oranını aşan bankalara uyguladığı ek faiz",
        ],
    ),
    TermQuestion(
        term="faaliyet izni",
        correct_definition="BDDK Kurulu tarafından bankaya belirli bankacılık faaliyetleri için verilen resmi yetki",
        distractors=[
            "Bankanın yeni şube açmak için belediyeden aldığı işyeri açma ruhsatı",
            "Yurt dışında faaliyet göstermek isteyen bankanın ev sahibi ülkeden aldığı vize",
        ],
    ),
    TermQuestion(
        term="münhasır yetki",
        correct_definition="Belirli finansal faaliyetlerin yalnızca BDDK lisanslı kuruluşlar tarafından yapılabilmesi",
        distractors=[
            "Tek bir bankanın belirli bir sektöre kredi verme tekelini elinde tutması",
            "BDDK Başkanı'nın kurul onayı olmadan tek başına karar verebilme hakkı",
        ],
    ),
    TermQuestion(
        term="kredi sınırları",
        correct_definition="Bir bankadan tek bir borçluya veya borçlu grubuna kullandırılabilecek azami kredi tutarı sınırlamaları",
        distractors=[
            "TCMB tarafından belirlenen politika faiz oranının üst sınırı",
            "Bireysel müşterilerin kredi kartı harcama limitleri",
        ],
    ),
    TermQuestion(
        term="bağımsız denetim",
        correct_definition="Bankaların mali tablolarının dışarıdan yetkili denetim kuruluşu tarafından incelenmesi",
        distractors=[
            "BDDK müfettişlerinin bankaları yerinde denetlemesi",
            "Bankanın kendi iç denetim biriminin yaptığı kontroller",
        ],
    ),
    TermQuestion(
        term="bilgi sistemleri denetimi",
        correct_definition="Bankanın BT altyapısı, veri güvenliği ve siber risklerin BDDK düzenlemelerine uygunluğunun değerlendirilmesi",
        distractors=[
            "Bankanın internet bankacılığı kullanım istatistiklerinin raporlanması",
            "Müşteri verilerinin pazarlama amacıyla analiz edilmesi",
        ],
    ),
    TermQuestion(
        term="tüketici kredisi",
        correct_definition="Gerçek kişilere ticari amaç dışında kullandırılan bireysel kredi",
        distractors=[
            "KOBİ'lere verilen düşük faizli teşvik kredisi",
            "Kamu kuruluşlarına açılan uzun vadeli yatırım kredisi",
        ],
    ),
    TermQuestion(
        term="konut kredisi (mortgage)",
        correct_definition="Konut satın alma amacıyla kullandırılan, konutun ipotek altına alındığı uzun vadeli kredi",
        distractors=[
            "İnşaat şirketlerine verilen toplu konut yapım kredisi",
            "Kiracıların kira ödemesini kolaylaştıran kısa vadeli kredi",
        ],
    ),
    TermQuestion(
        term="ticari kredi",
        correct_definition="Tüzel kişilere veya ticari işletmelere iş faaliyetleri için kullandırılan kredi",
        distractors=[
            "Sadece ithalat ve ihracat işlemlerinde kullanılan dış ticaret kredisi",
            "Bankaların birbirine verdikleri kısa vadeli interbank kredisi",
        ],
    ),
    TermQuestion(
        term="sendikasyon kredisi",
        correct_definition="Birden fazla bankanın bir araya gelerek büyük tutarlı tek bir krediyi ortaklaşa kullandırması",
        distractors=[
            "Bankaların sendika üyeleri olan çalışanlarına verdiği personel kredisi",
            "Devlet garantisi altında verilen kalkınma amaçlı kredi türü",
        ],
    ),
]
