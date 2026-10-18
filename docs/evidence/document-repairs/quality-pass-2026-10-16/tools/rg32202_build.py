"""Rebuild rg_32202_20230526_6 from RG 26/5/2023 sayı 32202 (htm prose + visually transcribed table images)."""

from pathlib import Path

B = "■"  # a cell printed solid black (closed to data entry) in the source template
SAT = "Kalan Vadesine Göre, Dikkate Alma Oranı Uygulanmamış Tutar"
TOP = "Dikkate Alma Oranı Uygulanmış Toplam Tutar"
W = ""  # white (open) cell


def template(period, pct):
    rows = [
        f"| {period} | {period} | a | b | c | ç | d |",
        "|---|---|---|---|---|---|---|",
        f"|  |  | {SAT} | {SAT} | {SAT} | {SAT} | {TOP} |",
        f"|  |  | Vadesiz* | 6 Aydan Kısa Vadeli | 6 Ay ile 6 Aydan Uzun 1 Yıldan Kısa Vadeli | 1 Yıl ve 1 Yıldan Uzun Vadeli | {TOP} |",
        "| **Mevcut İstikrarlı Fon** | **Mevcut İstikrarlı Fon** |  |  |  |  |  |",
    ]
    items = [
        (1, "Özkaynak Unsurları", "....."),
        (2, "Ana sermaye ve katkı sermaye", "....."),
        (3, "Diğer özkaynak unsurları", "....."),
        (4, "Gerçek kişi ve perakende müşteri mevduatı/katılım fonu", "....."),
        (5, "İstikrarlı mevduat/katılım fonu", "....."),
        (6, "Düşük istikrarlı mevduat/katılım fonu", "....."),
        (7, "Diğer kişilere borçlar", "....."),
        (8, "Operasyonel mevduat/katılım fonu", "....."),
        (9, "Diğer borçlar", "....."),
        (10, "Birbirlerine bağlı varlıklara eşdeğer yükümlülükler", "#####"),
        (11, "Diğer yükümlülükler", "....."),
        (12, "Türev yükümlülükler", "#...#"),
        (13, "Yukarıda yer almayan diğer özkaynak unsurları ve yükümlülükler", "....."),
        (14, "**Mevcut İstikrarlı Fon**", "####."),
        ("GIF", None, None),
        (15, "Yüksek kaliteli likit varlıklar", "####."),
        (16, "Kredi kuruluşları veya finansal kuruluşlara depo edilen operasyonel mevduat/katılım fonu", "....."),
        (17, "Canlı alacaklar", "....."),
        (
            18,
            "Teminatı birinci kalite likit varlık olan, kredi kuruluşları veya finansal kuruluşlardan alacaklar",
            ".....",
        ),
        (
            19,
            "Kredi kuruluşları veya finansal kuruluşlardan teminatsız veya teminatı birinci kalite likit varlık olmayan teminatlı alacaklar",
            ".....",
        ),
        (
            20,
            "Kredi kuruluşları veya finansal kuruluşlar dışındaki kurumsal müşteriler, kuruluşlar, gerçek kişi ve perakende müşteriler, merkezi yönetimler, merkez bankaları ile kamu kuruluşlarından olan alacaklar",
            ".....",
        ),
        (21, f"*{pct} ya da daha düşük risk ağırlığına tabi alacaklar*", "....."),
        (22, "İkamet amaçlı gayrimenkul ipoteği ile teminatlandırılan alacaklar", "....."),
        (23, f"*{pct} ya da daha düşük risk ağırlığına tabi alacaklar*", "....."),
        (
            24,
            "Yüksek kaliteli likit varlık niteliğini haiz olmayan, borsada işlem gören hisse senetleri ile borçlanma araçları",
            ".....",
        ),
        (25, "Birbirlerine bağlı yükümlülüklere eşdeğer varlıklar", "#####"),
        (26, "Diğer varlıklar", "....."),
        (27, "Altın dahil fiziki teslimatlı emtia", ".###."),
        (28, "Türev sözleşmelerin başlangıç teminatı veya merkezi karşı tarafa verilen garanti fonu", "#...."),
        (29, "Türev varlıklar", "#...."),
        (30, "Türev yükümlülüklerin değişim teminatı düşülmeden önceki tutarı", "#...."),
        (31, "Yukarıda yer almayan diğer varlıklar", "....."),
        (32, "Bilanço dışı borçlar", "#...."),
        (33, "**Gerekli İstikrarlı Fon**", "####."),
        (34, "**Net İstikrarlı Fonlama Oranı (%)**", "####."),
    ]
    for n, label, cells in items:
        if n == "GIF":
            rows.append("| **Gerekli İstikrarlı Fon** | **Gerekli İstikrarlı Fon** |  |  |  |  |  |")
            continue
        rows.append("| " + " | ".join([str(n), label] + [B if c == "#" else W for c in cells]) + " |")
    return "\n".join(r.replace("|  |", "|  |") for r in rows)


EXPL = [
    (1, "Özkaynak unsurları, 2 ve 3 üncü satırların toplamı", ""),
    (
        2,
        "Bankaların Özkaynaklarına İlişkin Yönetmeliğin 8 inci maddesinin dokuzuncu fıkrası ile 9 uncu maddesinde belirtilen indirimler uygulanmamış ana sermaye ve katkı sermaye",
        "7/1(a), 7/2, 7/3, 10/1(ç), 11/1(c)",
    ),
    (3, "2 nci satırda raporlanmayan özkaynak unsurları", "7/1(b), 10/1(ç), 11/1(e)"),
    (
        4,
        "Bankaların Likidite Karşılama Oranı Hesaplamasına İlişkin Yönetmeliğin 13 üncü maddesinin birinci ve üçüncü fıkraları ile 14 üncü maddesinin ikinci ve dördüncü fıkralarında belirtilen gerçek kişi ve perakende müşterilere ait mevduat/katılım fonu, 5 ve 6 ncı satırların toplamı",
        "",
    ),
    (
        5,
        "Bankaların Likidite Karşılama Oranı Hesaplamasına İlişkin Yönetmeliğin 13 üncü maddesinin birinci fıkrası ile 14 üncü maddesinin ikinci fıkrasında belirtilen gerçek kişi ve perakende müşterilere ait istikrarlı mevduat/katılım fonu",
        "5/6, 7/1(c), 8",
    ),
    (
        6,
        "Bankaların Likidite Karşılama Oranı Hesaplamasına İlişkin Yönetmeliğin 13 üncü maddesinin birinci ve üçüncü fıkraları ile 14 üncü maddesinin dördüncü fıkrasında belirtilen gerçek kişi ve perakende müşterilere ait düşük istikrarlı mevduat/katılım fonu",
        "5/6, 7/1(c), 9",
    ),
    (7, "Diğer kişilere borçlar, 8 ve 9 uncu satırların toplamı", ""),
    (
        8,
        "Bankaların Likidite Karşılama Oranı Hesaplamasına İlişkin Yönetmeliğin 15 inci maddesinde belirtilen operasyonel mevduat/katılım fonu",
        "5/6, 7/1(c), 10/1(b), 11/1(a)",
    ),
    (
        9,
        "Kredi kuruluşları veya finansal kuruluşlar dışındaki kurumsal müşterilere, merkezi yönetimler, kamu kuruluşları, çok taraflı kalkınma bankaları ve ulusal kalkınma bankaları, kredi kuruluşları veya finansal kuruluşlar ile merkez bankalarına olan diğer teminatlı ve teminatsız borçlar",
        "5/6, 7/1(c), 10/1(a), 10/1(c), 10/1(ç), 11/1(a)",
    ),
    (10, "Birbirlerine bağlı varlıklara eşdeğer yükümlülükler", B),
    (11, "Diğer yükümlülükler, 12 ve 13 üncü satırların toplamı", ""),
    (
        12,
        "Dikkate Alma Oranı Uygulanmamış sütunlarda; vadelerine bakılmaksızın Bankaların Net İstikrarlı Fonlama Oranı Hesaplamasına İlişkin Yönetmeliğin 6 ncı maddesi uyarınca hesaplanan türev yükümlülükler<br/>(Dikkate Anma Oranı Uygulanmış sütunu; Bankaların Net İstikrarlı Fonlama Oranı Hesaplamasına İlişkin Yönetmeliğin 6 ncı maddesi uyarınca hesaplanan türev yükümlülüklerin, 15 inci maddesi uyarınca hesaplanan türev varlıklardan fazla olan kısmı %0 dikkate alınma oranına tabi tutulduğundan veri girişine kapalıdır.)",
        "6, 11/1(c)",
    ),
    (
        13,
        "Yukarıda yer almayan diğer özkaynak unsurları ve yükümlülükler",
        "11/1(a), 11/1(b), 11/1(ç), 11/1(d), 11/1(e), 11/2",
    ),
    (14, "Mevcut istikrarlı fon tutarı; dikkate alınma oranı uygulanmış 1, 4, 7, 10 ve 11 inci satırların toplamı", ""),
    (
        15,
        "Bankaların Net İstikrarlı Fonlama Oranı Hesaplamasına İlişkin Yönetmeliğin 12 nci maddesinin onuncu fıkrasında belirtildiği şekliyle toplam yüksek kaliteli likit varlıklar (teminata konu edilen ve teminata konu edilmemiş)<br/>(a) Teminata konu edilen varlıklar (teminatlı menkul kıymetler veya varlığa dayalı menkul kıymetler dahil)<br/>(b) Teminata konu edilmeyen varlıklar (varlıkların kullanımını, satışını, transfer edilmesini, tasfiye edilmesini engelleyen yasal, düzenleyici, sözleşmeye dayanan veya bunların dışında herhangi bir kısıtlama bulunmamasını ifade eder.)",
        "12/10, 13, 16/1(a), 16/1(b), 17, 19/1(a), 20/1(a), 20/1(d), 22/1(a), 23/1(a)",
    ),
    (
        16,
        "Bankaların Likidite Karşılama Oranı Hesaplamasına İlişkin Yönetmeliğin 15 inci maddesinde belirtilen, kredi kuruluşları veya finansal kuruluşlara depo edilen operasyonel mevduat/katılım fonu",
        "13, 20/1(c), 23/1(a), 23/1(ç)",
    ),
    (17, "Canlı alacaklar; 18, 19, 20, 22 ve 24 üncü satırların toplamı", ""),
    (
        18,
        "Kredi kuruluşları veya finansal kuruluşlardan olan Bankaların Likidite Karşılama Oranı Hesaplamasına İlişkin Yönetmeliğin 6 ncı maddesinde belirtilen birinci kalite likit varlıklarla teminatlandırılmış canlı alacaklar",
        "13, 18, 20/1(b), 20/1(d), 23/1(a), 23/1(ç)",
    ),
    (
        19,
        "Kredi kuruluşları veya finansal kuruluşlardan teminatsız veya teminatı birinci kalite likit varlık olmayan teminatlı alacaklardan canlı olanlar",
        "13, 19/1(b), 20/1(b), 20/1(d), 23/1(a), 23/1(ç)",
    ),
    (
        20,
        "Kredi kuruluşları veya finansal kuruluşlar dışındaki kurumsal müşteriler, çok taraflı kalkınma bankaları ve ulusal kalkınma bankaları, gerçek kişi ve perakende müşteriler, merkezi yönetimler, merkez bankaları ile kamu kuruluşlarından olan canlı alacaklar",
        "13, 16/1(c), 20/1/(b), 20/1(ç), 21/1(b), 21/1(c), 22/1(b), 22/1(d), 23/1(a), 23/1(ç)",
    ),
    (
        21,
        "Kredi kuruluşları veya finansal kuruluşlar dışındaki kurumsal müşteriler, gerçek kişi ve perakende müşteriler, merkezi yönetimler, merkez bankaları ile kamu kuruluşlarından olan %35 ya da daha düşük risk ağırlığına tabi canlı alacaklar",
        "13, 16/1(c), 20/1(ç), 21/1(b), 21/1(c), 23/1(a), 23/1(ç)",
    ),
    (
        22,
        "İkamet amaçlı gayrimenkul ipoteği ile teminatlandırılan canlı alacaklar",
        "13, 20/1(ç), 21/1(a), 21/1(c), 22/1(b), 22/1(d), 23/1(a)",
    ),
    (
        23,
        "İkamet amaçlı gayrimenkul ipoteği ile teminatlandırılan % 35 ya da daha düşük risk ağırlığına tabi canlı alacaklar",
        "13, 20/1(ç), 21/1(a), 21/1(c), 23/1(a)",
    ),
    (
        24,
        "Yüksek kaliteli likit varlık niteliğini haiz olmayan, borsada işlem gören hisse senetleri ile borçlanma araçlarından canlı alacak niteliğini haiz olanlar",
        "13, 20/1(ç), 22/1(c), 22/1(d), 23/1(a)",
    ),
    (25, "Birbirlerine bağlı yükümlülüklere eşdeğer varlıklar", B),
    (26, "Diğer varlıklar; 27 ilâ 31 inci satırların toplamı", ""),
    (27, "Altın dahil fiziki teslimatlı emtia", "12/11, 13, 22/1(ç), 22/1(d), 23/1(a)"),
    (
        28,
        "Türev sözleşmelerin başlangıç teminatı olarak verilen nakit, menkul kıymet ve diğer varlıklar veya merkezi karşı tarafa verilen garanti fonu",
        "22/1(a), 22/4",
    ),
    (
        29,
        "Dikkate Alma Oranı Uygulanmamış sütununda, vadelerine bakılmaksızın Bankaların Net İstikrarlı Fonlama Oranı Hesaplamasına İlişkin Yönetmeliğin 15 inci maddesi uyarınca hesaplanan türev varlıklar,<br/>Dikkate Alma Oranı Uygulanmış sütunu, Bankaların Net İstikrarlı Fonlama Oranı Hesaplamasına İlişkin Yönetmeliğin 15 inci madde uyarınca hesaplanan türev varlıkların, 6 ncı madde uyarınca hesaplanan türev yükümlülüklerden fazla olan kısmı",
        "15, 23/1(a)",
    ),
    (
        30,
        "Dikkate Alma Oranı Uygulanmamış sütunlarda; Bankaların Net İstikrarlı Fonlama Oranı Hesaplamasına İlişkin Yönetmeliğin 6 ncı maddesi uyarınca hesaplanan türev yükümlülüklerin vade ayrımı yapılmaksızın değişim teminatı düşülmeden önceki tutarı<br/>Dikkate Alma Oranı Uygulanmış sütununda; Dikkate Alma Oranı Uygulanmamış sütunlarda raporlanan tutarın % 10’u (% 100 dikkate alınma oranı uygulanmış)",
        "6, 23/1(b)",
    ),
    (31, "Yukarıda yer almayan diğer varlıklar", "16/1(ç), 23/1(c), 25"),
    (32, "Bilanço dışı borçlar", "24"),
    (
        33,
        "Dikkate alma oranı uygulanmış 15, 16, 17, 25, 26 ve 32 nci satırların toplamından oluşan gerekli istikrarlı fon tutarı",
        "",
    ),
    (34, "Net istikrarlı fonlama oranı (%)", "4/1"),
]
EH = "Net istikrarlı fonlama oranı şablonundaki satırların açıklaması"
RH = "Bankaların Net İstikrarlı Fonlama Oranı Hesaplamasına İlişkin Yönetmeliğin ilgili hükümleri"
expl = [f"| **{EH}** | **{EH}** | **{EH}** |", "|---|---|---|", f"|  | **Açıklama** | **{RH}** |"]
expl += [f"| {n} | {a} | {r} |" if r else f"| {n} | {a} |  |" for n, a, r in EXPL]

doc = f"""26 Mayıs 2023 CUMA

Resmî Gazete

Sayı : 32202

TEBLİĞ

Bankacılık Düzenleme ve Denetleme Kurumundan:

BANKALARCA KAMUYA AÇIKLANACAK FİNANSAL TABLOLAR İLE BUNLARA İLİŞKİN AÇIKLAMA VE DİPNOTLAR HAKKINDA TEBLİĞDE DEĞİŞİKLİK YAPILMASINA DAİR TEBLİĞ

MADDE 1- 28/6/2012 tarihli ve 28337 sayılı Resmî Gazete’de yayımlanan Bankalarca Kamuya Açıklanacak Finansal Tablolar ile Bunlara İlişkin Açıklama ve Dipnotlar Hakkında Tebliğin 13 üncü maddesinin başlığı “Likidite riski yönetimi, likidite karşılama oranı ve net istikrarlı fonlama oranı” şeklinde değiştirilmiş ve aynı maddeye aşağıdaki fıkralar eklenmiştir.

“(4) 26/5/2023 tarihli ve 32202 sayılı Resmî Gazete’de yayımlanan Bankaların Net İstikrarlı Fonlama Oranı Hesaplamasına İlişkin Yönetmelik uyarınca hesaplanan net istikrarlı fonlama oranı ve bu orana ilişkin kalemler cari ve önceki dönem sonu için aşağıdaki formatta kamuya açıklanır. Raporlama dönemi dahil son üç aya ilişkin net istikrarlı fonlama oranlarının üç aylık basit aritmetik ortalaması bir önceki dönem üç aylık basit aritmetik ortalaması ile birlikte ayrıca açıklanır.

Net istikrarlı fonlama oranı şablonu

{template("Cari Dönem", "%35")}

{template("Önceki Dönem", "% 35")}

*Vadesiz sütununda raporlanan kalemlerin belirli bir vadesi bulunmamaktadır. Sayılanlarla sınırlı olmamak üzere bunlar belirli bir vadesi olmayan özkaynak unsurlarını, vadesiz mevduatı, kısa pozisyonları, vadesi belli olmayan pozisyonları, yüksek kaliteli likit varlık olmayan hisse senetlerini ve fiziki teslimatlı emtiayı içerir.

**Net istikrarlı fonlama oranı şablonunun doldurulmasında genel hususlar**

Şablondaki satırlar sabittir ve bütün bankalar tarafından doldurulması zorunludur. Şablon hakkındaki önemli hususlar:

Koyu gri satırlar, net istikrarlı fonlama oranı şablonunun bölümlerini gösterir.

Açık gri satırlar, net istikrarlı fonlama oranı şablonunun ilgili bölümlerinin alt bileşenlerini gösterir.

Boyanmamış satırlar, mevcut istikrarlı fon ile gerekli istikrarlı fon altındaki alt bileşenleri gösterir.$^{{1}}$

Siyah satırlara herhangi bir veri girişi yapılmayacaktır.

Şablona girilecek veriler dönem sonları itibarı ile hesaplanacak tutarlardır.

Gerekli istikrarlı fon satırlarındaki veriler hem teminata konu edilen hem de teminata konu edilmeyen tutarları içerir. a ile ç arasındaki sütunlardaki tutar hesaplanırken işlemin vadesi, d sütunundaki dikkate alma oranı uygulanmış toplam tutar hesaplanırken teminata konu edilen varlıklar için teminata konu edilme süresi de dikkate alınarak hesaplama yapılır.

Kalan Vadesine Göre Dikkate Alma Oranı Uygulanmamış Tutar sütununda raporlanan tutarlar kalan vadelerine göre ve Bankaların Net İstikrarlı Fonlama Oranı Hesaplamasına İlişkin Yönetmelik uyarınca belirlenir.

_______

$^{{1}}$ İstisna olarak, 21 ve 23 üncü satırlardaki tutarlar sırasıyla 20 ve 22 inci satırların alt bileşenleridir. 17 nci satır ise 18, 19, 20, 22 ve 24 üncü satırların toplamıdır.

Net istikrarlı fonlama oranı şablonuna ilişkin açıklamalar

{chr(10).join(expl)}

(5) Bankalar, sonuçların ve verinin anlaşılmasını teminen net istikrarlı fonlama oranına ilişkin sayısal açıklamalarda bulunur. Bankalarca önemlilik durumu göz önünde bulundurularak, asgari olarak net istikrarlı fonlama oranının ana kalemlerini etkileyen önemli unsurlar ile dönemler arasında ve zaman içerisinde oluşan verilerdeki değişiklikler (strateji değişiklikleri, fonlama yapısı, değişen durumlar) ve bu değişikliklerin nedenleri açıklanır.”

MADDE 2- Aynı Tebliğin 25 inci maddesinin ikinci fıkrasının (b) bendinde yer alan “likidite riski yönetimine ve likidite karşılama oranınave” ibaresi “likidite riski yönetimine, likidite karşılama oranına, net istikrarlı fonlama oranına ve” şeklinde değiştirilmiştir.

MADDE 3- Bu Tebliğ 1/1/2024 tarihinde yürürlüğe girer.

MADDE 4- Bu Tebliğ hükümlerini Bankacılık Düzenleme ve Denetleme Kurumu Başkanı yürütür.
"""
Path(__file__).with_name("rebuilt.md").write_text(doc, encoding="utf-8")
print(len(doc))
