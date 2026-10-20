# İnsan incelemesine sunulan tarihli tespit

**Durum: Onay bekliyor.** Bu belge, inceleme yapıldığı veya sonucun onaylandığı
anlamına gelmez. Yazılım testi, corpus imzası ve geliştirme planına verilen onay,
aşağıdaki kaynak/hukuki sürüm incelemesinin yerine geçmez.

## İncelenecek dar kapsam

**Bankaların Özkaynaklarına İlişkin Yönetmelik (`mevzuat_18799`), 9'uncu madde,
yalnız 30.06.2026 tarihi.** Önerilen kayıt başka bir tarih için sonuç üretmez.
Banka/J25 uygulamasının doğruluğunu, mahsup hakkını, hesaplama matrahını veya
bankaya özel Kurul kararlarının bulunmadığını onaylamaz.

Kaynak dosyaları ve alıntılar [ana raporda](README.md), makinece kontrol edilebilir
kaynak özetleri [sources.json](sources.json) içindedir. Önce ilgili resmî kaynakları
ve [alıntıları](claims.json) inceleyin; aşağıdaki tespitleri otomatik olarak kabul etmeyin.

## Teyit veya düzeltme gereken tespitler

1. **Metin eşleştirmesi:** Kaydedilmiş belgenin tamamı, resmî konsolidasyonun
   paragraf metniyle karşılaştırılmıştır. Boşluk, açıklanan eski karakter kodlaması
   kaynaklı noktalama farkları ve tek editoryal ayıraç dışında metin eşleşmektedir.
   [Karşılaştırma kaydı](whole-document-comparison.json). Bu farklarla madde 9'un
   kaynak eşleştirmesini uygun buluyor musunuz? Kayıp görsel/formül veya başka
   anlam değişikliği görürseniz belirtin.
2. **İlk metin ile mevcut sürüm ayrımı:** İlk düzenlemenin yayımı 05.09.2013,
   yürürlüğü 01.01.2014'tür. 2015 ve Ocak 2016 değişiklikleri 31.03.2016'da
   yürürlüğe girmiştir. Taslaktaki güncel normalleştirilmiş sürüm için 2018
   değişikliği esas alınarak yayım **14.03.2018**, ilgili hükümler için geçerli
   tarih **01.01.2018** önerilmiştir. Bu, ilk düzenlemenin tarihlerini değiştirmez.
   Bu sürüm-olay ilişkisinin doğru olduğuna özellikle bakın; uygun değilse kayıt
   düzeltilmeden onaylanmamalıdır.
3. **2019/2021 zinciri:** 2018 değişiklik yönetmeliğinin 2'nci maddesine ilişkin
   erteleme ve yürürlükten kaldırma, ana Yönetmeliğin 9'uncu maddesinin veya Geçici
   5'in kaldırılması gibi yorumlanmamıştır. Kaynakları bu yönden kontrol edin.
4. **30.06.2026 durum tespiti:** Sunulan resmî kaynak ve değişiklik zincirinin,
   eşleştirilen metni bu tarih için yürürlükteki sürüm olarak kabul etmeye yeterli
   olup olmadığını değerlendirin. Bu cümle, resmî kaynakta aynen yazan tarihli bir
   durum cümlesi değil, incelemeye sunulan bir çıkarımdır. Yalnızca aksi yönde sonuç
   bulunamamasına dayanarak onaylamayın; eksik resmî kaynak veya çelişki varsa
   belirtin ve kaydı beklemede bırakın.
5. **Kapsam dışı:** J25 tutarının geçici fark niteliği, Geçici 5/5'e uygunluk,
   bankanın TFRS 9 başlangıç/tercih bilgisi, mahsup koşulları ve sayısal hesaplar
   bu kaydın sonucu değildir. Bağımsız bankacılık/hukuk sertifikasyonu veya üretim
   kabulü iddiası oluşturulmayacaktır.

## Kaydedilecek gerçek inceleme bilgisi

İnceleme gerçekten tamamlandığında şu bilgileri verin:

- İnceleyen kişi veya izlenebilir yetkili rol/kimlik;
- İnceleme tarihi ve saati (saat dilimiyle);
- İncelenen kaynaklar ve yöntem;
- Sonuç: uygun / düzeltme gerekli / kanıt yetersiz;
- Varsa düzeltmeler ve değerlendirme sınırları.

Onay gelene kadar [taslak paket](legal_evidence.candidate.json) içindeki bütün
`validation` alanları **`in_review`** kalır. İnceleyen, tarih, yöntem veya onay
kaydı uydurulmaz. Onaylanan/düzeltilen kayıt ayrı bir artefakt olarak hazırlanır;
bu onaysız taslak geriye dönük olarak değiştirilip onaylanmış gibi gösterilmez.
