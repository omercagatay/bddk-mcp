# Değişiklik Günlüğü

Bu dosya BDDK MCP projesindeki önemli değişiklikleri Türkçe olarak kaydeder.
Biçim, [Keep a Changelog](https://keepachangelog.com/tr-TR/1.1.0/) yaklaşımına
yakındır; ancak repository'nin tarihsel sürümleme düzensizliklerini gizlemez.
Sürüm numaraları [Anlamsal Sürümleme](https://semver.org/lang/tr/) ilkeleriyle
yorumlanmalıdır.

## Sürümleme hakkında önemli not

Uygulama/paket sürümü, `v0001`–`v0011` adlı eski PostgreSQL migration
sürümlerinden bağımsızdır. PostgreSQL backend'i ve migration'ları 2026-11-10'da
kaldırıldı (Git etiketi `archive/postgres-2026-11-10`); güncel corpus sözleşmesi
`meta` tablosundaki `SCHEMA_VERSION = 1` değeridir ve migration kaydı yoktur. Sürüm geçmişi:

- `1.0.0` ve `2.0.0`, Git'te gerçekten `v1.0.0` ve `v2.0.0` etiketiyle
  işaretlenmiş tek sürümlerdir. `v1.0.0` etiketindeki `pyproject.toml` paket
  metadata'sı hâlâ `0.1.0` olduğundan ilk etikette de sürüm kaynağı tam uyumlu
  değildir; `v2.0.0` noktasında paket metadata'sı `2.0.0` olur.
- `3.0.0` ve `4.0.0`, `pyproject.toml` içindeki paket sürümünü değiştiren
  commit dönemleridir; bunlara karşılık gelen Git etiketi yoktur.
- Commit mesajlarında geçen “v5” SQLite/FTS5 özelliğini, “v6” ise ChromaDB
  özelliğini adlandırır. Bu iki commit sırasında paket sürümü hâlâ `4.0.0` idi;
  dolayısıyla bunlar resmî `5.0.0` veya `6.0.0` sürümleri değildir.
- Gerçek paket `5.0.0`, 9 Nisan 2026 tarihli PostgreSQL/pgvector ağırlıklı
  refactor ile başladı; bu sürüm için de Git etiketi oluşturulmadı.
- Mevcut geliştirme dalı paket metadata'sını `5.0.1` olarak taşır, ancak henüz
  yayımlanmış veya etiketlenmiş bir release değildir.
- `v0001`–`v0011` adları Python paket sürümü değil, ileri yönlü PostgreSQL
  migration sürümleridir. Bir uygulama release'i birden fazla migration
  içerebilirdi; bu migration'lar artık kodda yoktur, yalnızca Git geçmişindedir.

## [Yayınlanmadı] — hedef paket sürümü 5.0.1

### Değişti — ikinci bakım turu: bağımsız denetim ve tasarım incelemesi (2026-11-12)

Aynı gün ikinci tur: bağımsız denetimin ve dört bakış açılı tasarım
incelemesinin bulguları uygulandı, ardından çapraz inceleme yapıldı.
Değişiklikler önce ağ erişimi olmayan bir ortamda sahte yanıtlarla test edildi;
sonra gerçek tokenizer ve embedding modeliyle tam bir corpus kuruldu ve tool'lar
BDDK canlı kaynağına karşı denendi. Neyin doğrulandığı ve neyin doğrulanamadığı
en sondaki "Doğrulama" başlığındadır.

**Yükseltme notu.** Bu turdan önce üretilmiş bir corpus dosyası, 636 ve 774
numaralı belgelerin eski karar tarihlerini (07.02.2013 ve 14.02.2020; seed'de
artık karar metninde yazan 31.01.2013 ve 13.02.2020 var) ve eski madde
başlıklarını korur. Başlık değişikliği 79 belgede 1.596 başlıktır: `MADDE`
satırında başlık yoksa başlık artık üstündeki başlık satırından gelir,
önceden ilk fıkra metni olurdu. Hiçbir içerik özeti değişmediği için düz bir
`bddk-mcp build` "dosya seed ile zaten uyumlu" der ve eski değerleri bırakır.
Sahibin onayıyla yeni bir dosya üretin (`bddk-mcp build --corpus <yeni yol>`;
yeni dosyada her belge yeniden bölündüğü için tokenizer ağdan ya da
`BDDK_EMBEDDING_MODEL_PATH` üzerinden yüklenebilmelidir), `bddk-mcp info --corpus
<yeni yol>` ile kontrol edin, sonra hiçbir sunucu açık tutmazken eski dosyanın
yerine koyun; gerekirse ardından `bddk-mcp export`. Chunk metinleri değişmediği için yeniden embedding gerekmez.
Docker imajı her seferinde yeni dosya üretir, bundan etkilenmez.

HTTP:

- `/mcp` üzerinde `POST` dışındaki her yöntem (`PUT`, `PATCH`, `OPTIONS`,
  `HEAD`, `TRACE` dahil) artık admission'dan önce, gövde okunmadan 405 ve
  `Allow: POST` ile reddedilir. Önceden yalnızca `GET` ve `DELETE` böyleydi;
  diğerleri concurrency slot'u tutarak SDK'ya geçiyordu.
- `alg` üyesi olmayan bir JWK artık anahtar türüne göre denetlenir (RS*/PS* için
  RSA, ES* için eşleşen eğrili EC). `BDDK_JWT_ALGORITHMS` izin veriyorsa bu
  anahtarla PS256, RS384 ve RS512 de kabul edilir. Anahtar türü uymayan token
  reddedilir.
- `BDDK_HTTP_MAX_BODY_BYTES` 4 MiB üstünde de (en çok 16 MiB) geçerli olur.
  Önceden SDK'nın kendi 4 MiB sınırı ayardan bağımsız 413 veriyordu.
- İstek gövdesi geri verildikten sonra uygulamanın `receive()` çağrıları gerçek
  bağlantıyı bekler; istemcinin bağlantıyı kapatması artık görülür.

Tool çıktıları, istemciye görünen sözleşme (yalnızca ekleme; hiçbir alan
yeniden adlandırılmadı ya da silinmedi):

- Yeni isteğe bağlı argümanlar: `get_document_section` için `start_char` (aynı
  numaralı birkaç bölümden birini başlangıç konumuyla seçen, `0` veya daha büyük
  tamsayı; yalnızca o corpus dosyası için geçerlidir, atıf değildir).
  `search_bddk_regulations` için `keywords` artık isteğe bağlı (varsayılan
  boş): kategori veya `date_from`/`date_to` ile birlikte boş bırakılırsa süzülmüş
  kataloğu en yeniden başlayarak sayfalar; ne süzgeç ne anahtar kelime yoksa
  `INVALID_INPUT` döner.
- Yeni isteğe bağlı çıktı alanları: `SectionSearchResponse` için
  `inferred_document_filter_applied`, `inferred_document_filter_dropped` ve
  `parsed_section_ref`; `RegulationCatalogResponse` için `undated_excluded`; `DocumentSearchItem` için `match_type`, `section_type`,
  `section_ref`, `source_url` ve `decision_number`; `SectionItem` için
  `stem_match`, `parent_section_type`, `parent_section_ref`,
  `stored_span_capped`, `original_end_char`, `continuation_of_section_type`,
  `continuation_of_section_ref`, `continuation_of_section_end_char` ve
  `document_structureless`; `EvidenceReference` için `source_downloaded_on`;
  `DocumentHistoryResponse` için `current_body` ve `resolved_document_id`.
- `match_strength` yeni bir değer aldı: `keyword`. Sorgudaki hukuki atıfla (madde
  numarası, karar numarası, tarih) birebir eşleşen FTS sonuçları artık
  `[keyword match on the legal reference in the query; no semantic score]`
  etiketiyle gelir; `relevance` 0.0 düşük skor değildir ve bu sonuçlar zayıf
  eşleşme sayılmaz. Sunucu talimatının 7. kuralı bu sonuçları %50 kuralından
  muaf tutar, onları doğrulanacak ipucu olarak tanımlar.
- Yeni hata kodu `INSUFFICIENT_DATA`: `analyze_bulletin_trends` ikiden az gözlem
  olduğunda yeniden denenemeyen bu hatayı döner. `compare_bulletin_metrics`
  tablosu aynı durumda `HATA` yerine nedenle birlikte `YETERSİZ VERİ` yazar.
- `get_document_history` kayıtlı olmayan bir belge numarası için boş sonuç
  yerine `NOT_FOUND` döner. `get_bddk_document` ve `get_document_history`
  `document_id` için diğer tool'larla aynı katı kalıbı kullanır (harf, rakam, `_`
  ve `-`, en çok 64 karakter); `.` veya `:` içeren numara `INVALID_INPUT` olur.
  `page_number` ve `max_pages` katı tamsayıdır: `true` ve `"2"` reddedilir.
  Çıplak numaralar `mevzuat_` veya `bddk_` önekiyle çözülür.
- `get_document_history` her cevabın başına, veri çerçevesinin dışında sabit bir
  uyarı koyar: bu corpus dosyasının yerel metin değişikliği kaydıdır, tadil
  tarihçesi değildir. Mevcut gövde "Current stored body (not a version)" satırıyla
  ayrı bildirilir. Sürüm ve katalog zamanları artık UTC yazılır (` UTC` son eki);
  önceden sunucunun yerel saatiydi ve son ek yoktu.
- Belge durumu: BDDK katalog sınıfı Mülga Düzenleme veya Düzenleme Taslağı olan
  belgelerde `Document status: ... MÜLGA (repealed)` ya da `TASLAK (draft)` satırı
  ve aynı metinde bir uyarı çıkar (`get_document_section`,
  `search_document_sections`, `get_bddk_document`, `search_document_store`,
  `search_bddk_regulations`). Sunucu talimatına 22. kural eklendi: bu işaretli
  belge güncel hukuk gibi sunulmaz; işaretin olmaması yürürlükte olduğunu
  göstermez. `get_bddk_document`, `search_document_store` ve
  `search_bddk_regulations` section tool'larındaki hukuki uygulanabilirlik
  uyarısını da taşır.
- Kaynak tarihleri: section çıktıları `Decision date / number: T / N` ve `Text
  obtained from the source on: YYYY-MM-DD (UTC download date; not a statement
  that the text is current)` satırlarını yazar; indirme tarihi kayıtlı değilse
  `unknown`. `get_bddk_document` başlığı kaynak tarihi satırını alır, kanıtı
  `content_hash` ve `source_downloaded_on` taşır. `search_bddk_regulations` artık
  `Versions: N (latest: X)` yerine "Replaced bodies in this corpus file: N ...; a
  local text-change log, not amendment history" yazar ve her sonuca "Catalog
  source (not a validated citation)" satırı ekler.
- `get_bddk_document` başlığı artık "sarmalama yok" demez: sayfalar
  `PAGE_SIZE` karakterlik sabit pencerelerdir ve bir kelimeyi, tablo satırını
  veya formülü kesebilir; çok sayfalı çıktıda `---` ve `### Page N/T` satırları
  tool ayırıcısıdır. Çift sayıda olmayan `$$` işaretiyle ya da tablo satırı
  ortasında biten, son olmayan sayfa metinde ve `warnings[]` içinde uyarı alır.
  Depolama hatası artık "belge yok" sayılmaz; temizlenmiş `TOOL_EXECUTION_FAILED`
  olarak çıkar.
- Alıntı kontrolü ve kısaltılmış hükümler: kayıtlı pencerede bulunmayan alıntı
  artık `not_found` değil `unavailable` döner (neden `reference_truncated`,
  boşluk `quotation_reference_truncated`); pencerenin tamamına eşit alıntının
  kapsamı `excerpt` olur. Kısaltılmış bir hüküm gerçek uzunluğunu yazar
  ("Stored window capped at 20,000 of N characters ..."). Hükmün devamını
  taşıyan `govde` satırları "continuation of <tür> <no>" etiketini alır, hükmün
  gerçek bitişini aşan satırlar bunu söyler; bunlar için "yasal hüküm olarak
  atıf yapmayın" uyarısı yerine o hükme atıf yapıp `get_bddk_document` ile
  doğrulama uyarısı gelir. Bu eski uyarı yalnızca yapılı belgelerin gerçek artık
  satırlarında kalır.
- Madde yapısı olmayan belgeler: `search_document_sections` artık bu
  belgelerin (çoğu Kurul Kararı; 319 belgenin 204'ü) `govde`
  satırlarını varsayılan olarak da döndürür ve "full-text window (this document has no
  article structure)" diye etiketler; belgeye başlık ve karar tarihiyle atıf
  yapılması istenir. `search_document_store` aynı etiketi ve aynı karar satırını
  kullanır; bu satır artık `Decision date / number: T / N` biçimindedir. Bu
  satırlarda `SectionItem.heading` boştur (önceden genel `govde` başlığıydı).
  Yapılı belgelerin `govde` satırları `section_type=govde` verilmedikçe yine
  gelmez.
- fıkra ve bent satırları tekil adres değildir: çıktı artık "Parent: madde N"
  (ya da `gecici_madde`/`ek_madde`) satırı yazar ve ipucu "bu satırlar üst madde
  taşımaz" demez; tam hüküm için üst madde istenir.
- `search_bddk_regulations`: bağlaç sözcükleri (sayılı, hakkında, ilişkin, dair,
  göre, ile, ve, veya, no, nolu, numaralı, tarihli, hk) yok sayılır; belge türü
  sözcükleri (yönetmelik, tebliğ, genelge, rehber, karar, kanun, taslak; her
  çekimiyle, ASCII ya da Türkçe harfle) hiçbir zaman zorunlu değildir, yalnızca
  başlığında ya da kategorisinde o türü taşıyan belgeleri öne alır. İçerik
  sözcükleri başlıkta en az 5 karakterlik ortak kökle de eşleşir; tam başlık
  eşleşmesi kök eşleşmesinin, karar numarası eşleşmesi ikisinin önündedir.
  "kanuni takip" kanun belgelerini, "rehberlik" rehberleri getirmez. Özet başlığı
  tekrar ediyorsa (319 kayıttan 318'i) özet satırı yazılmaz. Sonuç yoksa metin
  "bu corpus kataloğunda (N belge)" eşleşme olmadığını ve yokluğun enstrümanın
  var olmadığını göstermediğini söyler; tool açıklaması BDDK sitesinde değil yerel
  katalogda aradığını belirtir.
- `search_bddk_regulations` tarih süzgeci: karar tarihi okunamayan katalog
  kayıtları süzgece alınamaz; metin, uyarılar ve `undated_excluded` kaç kaydın
  atlandığını söyler ("K catalog entries have no decision date and were not
  considered by the date filter"). Sonuç boşsa metin artık "corpus'ta
  olmayabilir" demek yerine tarih sınırsız yeniden denemeyi önerir. Tarih
  argümanlarının açıklaması bu uyarıyı taşır.
- `search_bddk_announcements` kategorilerin tümünde en yeniden başlayarak
  listeler, tarihsizler sonda; her satır `[kategori]` yazar. 20'den fazlaysa "showing
  the 20 most recent of N" der ve "... and K more" ile biter.
- `search_document_store` ile `search_bddk_regulations` önbelleği corpus yazımından
  sonra eski cevap vermez (önbellek anahtarı corpus değişim belirtecini içerir).
  Yalnızca gevşetilmiş OR geri dönüşüyle bulunan sonuçlar `fts_exact` olmaz,
  normal anlamsal eşik ve boşluk süzgecine tabidir.
- `DocumentSearchItem.relevance` açıklaması artık sıralamayı yanlış anlatmaz
  (önce hybrid sonuçlar skora göre, sonra `fts_exact` BM25 sırasıyla, sonra
  kalan vector/FTS sonuçları skora göre). Çıktı şema açıklamaları sayfa ve
  bölüm içeriğini "exact stored characters (no sanitization)" diye tanımlar;
  `DocumentResponse.quality` tüm kayıtlı gövdeyi kapsar.

Bölüm araması:

- Sorgudaki bir sayı kayıtlı bir belge numarasına eşitse (`kredi 90 gün` →
  belge 90) artık yalnızca tam, tam kelime ve çekim toleranslı geçişlerde o
  belgeye kapsam çizer. Orada eşleşme yoksa kapsam bırakılır ve önce katı geçiş,
  sonra gevşek geri dönüş tüm belgelerde çalışır. Önceden gevşek geri dönüş de
  o tek belgede kalıyor ve ilgisiz sonuç ya da boş cevap veriyordu. Kapsam
  uygulandığında metin "the search was limited to document N because of the number
  N in the query ..." der ve `inferred_document_filter_applied` true olur;
  bırakıldığında not sayıyı ve nedeni yazar. Tanımlı olmayan bir numara hiç kapsam
  sayılmaz. Açıkça verilen `document_id` hâlâ kesin süzgeçtir.
- TFRS, TMS, UFRS, UMS, IFRS, IAS, BDS, TFRYK, UFRYK, IFRIC, BOBİ FRS ve Basel
  sözcüklerinden sonra gelen sayı, COVID-19 ve "yıl..." ile biten 19xx/20xx yılı
  artık sorgudan belge numarası diye okunmaz.
- Belge belirlenmeden verilen bir bölüm göndermesi ("madde 9") artık sessizce
  düşmez: metin ve `warnings` bunun süzgeç olarak uygulanmadığını ve
  `search_bddk_regulations` ardından `get_document_section` çağrılacağını söyler;
  `section_ref` boş kalır, ayrıştırılan değer `parsed_section_ref` içindedir.
  "9. madde", "9 uncu madde", "10'uncu maddesinde" ve "ek/geçici" önekli biçimler
  madde olarak ayrıştırılır, sayı belge numarası sayılmaz ("5411 sayılı Kanunun 10
  uncu maddesi" → belge 5411, madde 10).
- Kesin eşleşme azsa aynı kapsamda çekim toleranslı (ön ek kök) eşleşmeler
  tamamlanır; `stem_match` true, sıra boş ve "inflection-tolerant match" satırı
  vardır. Gevşek geri dönüş yalnızca iki geçiş de boşsa çalışır. Tam kelime
  sonucu olmadığında etiket "exact-word hits'ten sonra listelendi" demez.
  Zaten tam referansla bulunmuş bir sonuç `stem_match` sayılmaz.
- NO RESULTS metni belge veya bölüm türü süzgecini kaldırmayı yalnızca çağıran o
  argümanı verdiyse önerir.
- Madde başlığı artık `MADDE` satırında başlık yoksa üstündeki başlık satırından
  alınır ("Tanımlar"; önceden "(1) Bu Yönetmelikte yer alan;"), başlık satırındaki
  değişiklik notu da başlığa girer (`mevzuat_38416` madde 14). Böylece
  `get_document_section(heading="Tanımlar")` ve başlık ağırlıklı arama bulur.
  Bölüm türü, numara, konum, içerik, hash ve sıra 319 belgenin hiçbirinde
  değişmedi; chunk sınırları ve seed vektörleri geçerli kalır. Etki için yukarıdaki
  yükseltme notuna bakın.
- Kısaltma notunun sözcükleri (`karakterden kısaltıldı`, `get_bddk_document` ...)
  bu tur sonrası üretilen corpus dosyalarında bölüm FTS dizininde yer almaz;
  eski dosyalarda dizin yeniden üretilene kadar kalır.

Corpus, seed ve CLI:

- Yeni komut `bddk-mcp info [--corpus YOL]`: corpus dosyasını salt okunur açıp
  paket sürümünü, corpus özetini (saklanan belge metinleri üzerinden 12
  karakterlik özet), belge, chunk ve katalog sayılarını, embedding modelini,
  şema sürümünü, dosya boyutunu ve journal modunu yazar; hazır değilse sorunları
  sıralar ve 1 ile çıkar. `serve` başlangıç denetimi geçince aynı kimliği tek
  `Corpus ready: ...` satırıyla loglar. `CorpusReadiness` iki isteğe bağlı alan
  aldı (`schema_version`, `digest`); `/health` cevapları değişmedi.
- `bddk-mcp export` hazır olmayan bir corpus dosyasını reddeder (tek satır hata,
  çıkış kodu 1) ve seed dizinine hiçbir şey yazmaz.
- Yazıcı (build, sync, betikler) kapanırken dosyayı rollback-journal (`delete`)
  kipinde bırakır; salt okunur açılış artık yanında `-wal`/`-shm` üretmez. Bu en
  iyi çaba esasıdır: bu turdan önce üretilmiş ya da geçişi başarısız olmuş dosya WAL
  kipinde kalır (`info` bunu gösterir); `bddk-mcp build` çalıştırmak yazıcı açıp
  kapattığı için dosyayı dönüştürür.
- Okuma izni, salt okunur dizin ya da "açılamıyor" hatası artık "geçerli bir
  corpus değil, build çalıştırın" yerine izin sorunu olarak bildirilir. Hazır
  değil ipucu yalnızca `bddk-mcp build` der; `sync` eşdeğer çare olarak
  önerilmez.
- Yazıcı, başka şema sürümündeki dosyayı, `meta` tablosu olmayan bir SQLite
  veritabanını ve SQLite olmayan dosyayı değiştirmeden reddeder.
- Boş gövdeli seed belgesi `sha256('')` özetiyle saklanır; build tekrarında
  atlar ve export edilmiş seed yeniden build olur. Boş gövdeyi dolduran güncelleme
  sürüm arşivi oluşturmaz.
- `bddk-mcp build` chunk'lanacak bir şey varsa tokenizer'ı ilk yazımdan önce
  yükler; çevrimdışı bir makinede erken ve temiz hata verir, yeni oluşturduğu
  dosyayı siler. "Seed verisi yok" hatası okunan dizini söyler.
- 636 ve 774 numaralı Kurul Kararı kayıtlarının karar tarihi, karar metninde
  yazan Karar Tarihi'ne düzeltildi (31.01.2013, 13.02.2020).

Sync:

- Düz `bddk-mcp sync` (`--force` olmadan), metni kayıtlı ama güncel chunk'ı
  olmayan belgeyi indirmeden, kayıtlı metinden yeniden indeksler; metin ve
  `extraction_method` değişmez. Sonuç yöntemi `reindexed`, başarısızlık
  `reindex_failed` olarak kaydedilir. Özet satırı artık `Sync: T total, D
  downloaded, R reindexed, S skipped, F failed, Xs`; `SyncReport` isteğe bağlı
  `reindexed` alanı aldı.
- Mevzuat `*-ek.zip` içindeki ek metni artık saklanan markdown'a girer ve ek
  `<pre>` satırları ayrı satır kalır.
- İç içe HTML tablosu dış hücresinin içinde bir kez, `;` ile birleştirilmiş düz
  metin olarak çıkar; hayalet satır oluşmaz. `colspan`/`rowspan` 1–100 aralığına
  sıkıştırılır, 0, negatif ya da sayı olmayan değer 1 sayılır. Eksik `</tr>` ya da
  `</td>` yüzünden iç içe düşen satır ve hücreler artık atılmaz.
- Çıktı ızgarası 200.000 hücreyi aşacak bir tablo ızgara yerine düz hücre
  metni olarak yazılır ve uyarı loglanır (çok geniş `colspan` ile şişirilmiş
  tablolar bellek ve süre tüketiyordu). Bekleyen `rowspan` hücreleri de bu
  sayıya girer. Bir belgenin bütün tablolarının ızgaraları birlikte 2.000.000
  hücreyi geçecekse kalan tablolar da düz metin olarak yazılır.
- Mevzuat hata sayfası dönen bir iframe ya da div artık indirmeyi bitirmez;
  GeneratePdf, statik PDF, `.htm` ve diğer adaylar denenir.
- Çıkarma sırasında oluşan bir istisna sync'ten kaçmaz: gerçek belge numarasıyla
  `sync_failures` içine yazılır ve CLI `[<belge no>] Extraction failed: ...`
  yazar (önceden `[unknown]`). Eski OLE `.doc` indirmesi "legacy .doc is not
  supported (OLE Word file)" nedeniyle başarısız olur. HTML ayrıştırma hatası
  markitdown'a düşer, markitdown istisnası başarısız çıkarma sayılır.
- HTML çözümleme sırası: katı UTF-8; olmazsa, en az bir geçerli çok baytlı
  karakter çözülebiliyorsa başıboş baytlar windows-1254 okunan UTF-8; olmazsa
  bildirilen karakter kümesi (HTTP ya da `<meta>`); en son windows-1254.
  Bildirilen tek baytlık küme geçerli UTF-8 baytların önüne geçmez. Metin
  kodlaması olmayan bildirimler (`zlib`, `hex`, `base64`, `rot13`, `bz2`, `uu`)
  yok sayılır ve çıkarma bunlar yüzünden başarısız olmaz.
- LightOCR: `max_new_tokens`'a EOS olmadan ulaşan sayfa backend'i başarısız
  sayar ("page N of M" loglanır); zincir uyarılı `_degraded` backend'lere düşer.
  Boş sayfalar loglanır ve sayılır. Başarısızlık sınıfları artık `mevzuat_14036`
  gibi belge numaralarındaki 403/404'ü HTTP durumu diye okumaz.

Bülten ve analitik:

- `get_bddk_monthly` hücreleri BDDK'nın gönderdiği gibi geçirir: JSON sayısı
  en-US biçimiyle (`1,500`, `1,234.50`), metin hücreleri olduğu gibi yazılır;
  tablo bunu söyler ve eksik hücre `-` olur. Beklenmeyen cevap biçimi artık "veri
  yok" değil, yeniden denenebilir üst akış hatasıdır.
- Önceki değer 0 olunca trend ve karşılaştırma tool'ları `0%` yerine yüzdenin
  hesaplanamadığını ("önceki değer 0") yazar; negatif tabanlarda `abs(önceki)`
  kullanılır. Trend etiketi son 4 noktanın net hareketine bakar (4'ten az
  noktada son adım); anlatı kaç nokta kullanıldığını söyler.
- Kurum rehberinde Türkçe büyük/küçük harfli iptal başlıkları (`Iptal`, `IPTAL`,
  noktalı büyük İ) tanınır, `active_only` bunları dışarıda bırakır.
- Özet penceresi ve aylık geriye yürüme, sunucunun saat diliminden bağımsız
  İstanbul (UTC+3) tarihini kullanır. Haftalık seri tarihler okunabiliyorsa
  eskiden yeniye sıralanır; boş haftalık dönem yeniden denenebilir üst akış hatası
  yerine "No data returned" yazar.

Kalite ve betikler:

- Kalite politikası `markdown-quality-assessment-v8`: herhangi bir `cid:` işareti
  ya da çift kodlanmış Türkçe metin (mojibake) belgeyi en az `warning` yapar.
  Seed'de hiçbir belgenin etiketi değişmedi. Mojibake desenleri Latin-1 ve
  ISO-8859-9 biçimlerini de kapsar.
- Depolama temizleyicisi yalnızca gözden geçirilmiş dört antet sözcüğünü onarır
  (`BBANKACILIK`, `DDÜZENLEME`, `DDENETLEME`, `KKURUMU`); `AAOIFI`, `LLOYD` ya da
  `XXXXX` gibi gerçek kısaltmalar ve yer tutucular korunur.
- `scripts/scan_document_quality.py`: `--fail-on` raporda görünen sinyal adlarını
  kabul eder, etiketi temiz belgelerde de çalışır; bilinmeyen ad 2 ile çıkar.
  Rapor, chunk bütünlüğü sayıları sıfır değilken artık "tüm sinyaller temiz"
  demez; yeni `mojibake` anomali denetimi ve `QualityReport.signal_docs` alanı
  eklendi, rapor "Assessment flags" bloğunu yazar.
- `scripts/patch_doc.py`: corpus'un depolarken yeniden yazacağı metni reddeder
  (işaretleri adlandırır, `--dry-run`'da da); seed'i kanonik biçimde yazar. Yeni
  `--seed-dir`; varsayılan olmayan `--corpus` ile `--seed-dir` verilmezse seed
  yazılmaz ve bunu söyler.
- `scripts/inventory_dropped_formulas.py` açıklaması düzeltildi, raporları
  varsayılan olarak `quality_reports/` altına yazar. `scripts/mcp_fetch_full.py`
  `argparse` kullanır (`--help`, `--url`, belge numarası, sayfa numaraları).

Testler, ayarlar ve depo:

- Hygiene denetimi `.env.example` dışında izlenen her `.env.*` dosyasında ve
  Dependabot `uv`, `github-actions`, `docker` ekosistemlerinden birini
  kapsamıyorsa başarısız olur. Dependabot artık Dockerfile temel imajları için de
  haftalık güncelleme PR'ı açar.
- Tokenizer testi yüklenemediğinde nedeni ve çareyi söyleyen bir mesajla
  başarısız olur (hâlâ başarısızlıktır). Test paketi bu ortamda atlanan test
  içermeden çalıştı; `gpu` işaretli testler varsayılan olarak seçilmez.
- `.env.example` kodun okuduğu her değişkeni varsayılanıyla listeler ve hangilerinin
  yalnızca sync ya da OCR için olduğunu söyler. Üstteki not `docker compose`'un
  yalnızca `BDDK_HTTP_ALLOWED_HOSTS` ve `BDDK_HTTP_ALLOWED_ORIGINS` değişkenlerini
  aktardığını belirtir. `docs/DEPLOYMENT.md` sync'te yeniden indekslemeyi,
  `BDDK_PREFER_HTML_FOR_MEVZUAT` ayarını, izin mesajını ve yükseltme notunu anlatır.

Kaldırılanlar:

- `BDDK_EMBEDDING_DIM`, `BDDK_EMBEDDING_CHUNK_TARGET_TOKENS` ve
  `BDDK_EMBEDDING_CHUNK_TOKEN_OVERLAP` artık okunmaz; ayarlanırlarsa hata vermeden
  yok sayılır. Boyut (768), chunk boyutu (400 token) ve örtüşme (40) kod sabitidir.
- `BDDK_CHANDRA_MODEL` ve Chandra OCR backend'i (hiçbir sync yolu onu
  kurmuyordu), ilgili bağımlılık ve testler.
- `scripts/sanitize_seed_data.py` ve `scripts/mcp_smoke.py`.
  `scan_document_quality.py` için `--quality-failures` ve `--manifest`
  seçenekleri (`--manifest` zaten yok sayılıyordu).

Arama skoru:

- `search_document_store` artık gösterdiği `relevance` değerini embedding
  modeline göre kalibre eder: sıralama skoru (ham cosine ile kelime ve ifade
  katkıları; sıralama ve skor aralığı filtresi değişmedi)
  `intfloat/multilingual-e5-base` için 0.82–0.96 aralığından 0–1'e taşınır;
  hiçbir anahtar kelime eşleşmezse yalnızca gösterilen değer düşürülür.
  `semantic_relevance` ham cosine'dir; eleme eşiği ham cosine'e uygulanır ve
  varsayılanı bu model için 0.80, başka modeller için 0.50'dir
  (`BDDK_SEMANTIC_THRESHOLD` hâlâ geçersiz kılar). Gerçek modelle 101 soruda
  ölçüldü: 40 alan içi sorunun 40'ı yine "strong"; 40 alan dışı sorunun 34'ü
  önceden "strong" iken artık 38'i "weak" ya da sonuçsuz. Ölçüm betiği
  `scripts/calibrate_relevance.py`; ayrıntı ve sınırlar
  [Document Quality](docs/DOCUMENT_QUALITY.md#displayed-search-relevance).

Doğrulama (2026-11-12):

Gerçek tokenizer ve embedding modeliyle (`intfloat/multilingual-e5-base`,
sabitlenmiş revizyon):

- Seed'den tam `bddk-mcp build`: 319 belge, 8.556 bölüm, 13.721 chunk. Chunk'ların
  hepsi seed önbelleğinden geldi, hiçbiri yeniden gömülmedi; yani bu turdaki
  ayrıştırıcı değişiklikleri hiçbir chunk metnini değiştirmedi.
- Bu dosyadan alınan `bddk-mcp export`, `seed_data/` içeriğini bayt bayt geri
  verdi (iki JSON dosyası, manifest ve 319 vektör dosyası).
- Test paketi atlanan ya da başarısız test olmadan geçti; gerçek tokenizer testi
  (`test_real_token_chunker_produces_exact_document_slices_for_every_document`)
  dahil.
- Tool'lar gerçek sunucu nesnesi üzerinden, tam corpus ve gerçek modelle
  çağrıldı: anlamsal arama, bölüm araması, başlıkla madde bulma
  (`heading="Tanımlar"`).

BDDK canlı kaynağına karşı:

- Haftalık bülten, aylık tablo, trend, karşılaştırma, özet, duyuru ve kurum
  rehberi tool'ları gerçek cevaplarla çalıştı. Aylık tablo dönem verilmeden
  Eylül 2026'ya gitti; iptal edilmiş banka kurum rehberinde işaretli geldi;
  duyurular en yeniden başlayıp "514'ün en yeni 20'si" dedi.
- Bir Kurul Kararı (belge 779) geçici bir corpus kopyasına `--force` ile yeniden
  senkronize edildi: belge indirildi, CPU zinciriyle çıkarıldı
  (`pdftotext_degraded`), yeniden indekslendi, eski gövde sürüm olarak
  arşivlendi ve dosya hazır kaldı.
- Deneme ortamında DNS çözümleyici yoktu; bu yüzden yalnızca deneme betiğinde
  DNS ön kontrolü devre dışı bırakıldı. Tam host ve yönlendirme kontrolleri
  etkindi; depodaki kod değişmedi.

Doğrulanamayanlar:

- mevzuat.gov.tr deneme ortamından cevap vermedi (bağlantı zaman aşımı). Sync'in
  mevzuat'a özgü değişiklikleri (ek metni, hata sayfası geçişi, karakter kümesi,
  tablo sınırları) yalnızca sahte yanıtlarla test edildi. Gerçek mevzuat
  HTML'sinde eksik `</tr>`/`</td>` ne sıklıkta görülüyor, ölçülmedi.
- LightOCR (GPU) yolu çalıştırılmadı; kesilme denetimi sahte model ile test
  edildi.
- Docker imajı kurulmadı.

### Değişti — bakım turu: denetimden çıkan düzeltmeler (2026-11-12)

Kod tabanının tamamı gözden geçirildi; doğrulanan 80 bulgunun hiçbiri yüksek
veya kritik değildi. Bu turda düzeltilenler aşağıdadır.

HTTP:

- `/mcp` üzerinde `GET` ve `DELETE` artık Host/Origin kontrolünden sonra,
  admission'dan önce 405 ve `Allow: POST` ile reddedilir. Sunucu stateless
  JSON çalışır ve server-initiated stream sunmaz; buna rağmen SDK kimliği
  doğrulanmamış bir `GET /mcp` isteğini SSE stream olarak açık tutuyor ve bir
  concurrency slot'u harcıyordu. Birkaç açık stream health probe'lar dahil
  bütün istekleri kilitleyebiliyordu.
- uvicorn `limit_concurrency` kaldırıldı. Bu ayar yalnızca açık duran
  bağlantıları sayıp `Retry-After` içermeyen bir 503 döndürüyordu; tek
  concurrency sınırı artık middleware limiter'dır. IPv6 istemcilerin
  rate-limit anahtarı /64 ağıdır.
- JSON log formatı retry alanlarını (`attempt`, `max_retries`, `max_attempts`,
  `http_method`, `upstream_boundary`, `status_code`) yazar; hiçbir kodun
  üretmediği `doc_id`, `query`, `result_count` anahtarları çıkarıldı.
  `AuthSettings.validate_token_resource=False` açıkça ayarlandı (audience
  kontrolünü `JwtTokenVerifier` yapar; davranış değişmedi).

Tool çıktıları:

- `get_bddk_monthly`: dönem verilmezse son 3 tamamlanmış ay yeniden eskiye
  doğru denenir ve çıktı kullanılan dönemi yazar. BDDK bir ayı haftalar sonra
  yayımladığı için "son tamamlanmış ay" çoğu zaman henüz yoktur; eski varsayılan
  ise sabit 12/2025 idi. BDDK'nın "bu döneme ait veri yok" cevabı
  (`success=false` ve "veri bulunma…" mesajı) veri yok olarak döner ve
  BDDK'nın mesajı çıktıya yazılır; başka her `success=false` cevabı ve
  beklenmeyen cevap biçimi hata sayılır.
  `year` ve `month` birlikte verilir ya da hiç verilmez.
- `compare_bulletin_metrics` en çok 4 metrik kabul eder; beşinci metrik artık
  sessizce atılmaz, şema reddeder. `get_bddk_bulletin` en yeni 60 gözlemi yazar
  ve liste kesildiyse "Son 60 / N gözlem gösteriliyor." der.
- `analyze_bulletin_trends` düşüşleri de sayar (sabit seri artık "yatay"),
  anlatı kullanılan son 4 veri noktasını söyler, boş başlık yerine metrik
  kimliği yazılır.
- `get_regulatory_digest` gerçek toplamları ve kesilen listeler için
  "... ve N daha" satırını yazar, kararların yerel corpus kataloğundan
  geldiğini söyler. Pencere bugünü içeren takvim günleridir (day=1, week=7).
- `get_bddk_document`: var olan bir belgenin aralık dışındaki sayfası artık
  `NOT_FOUND` değil, sayfa sayısını veren `INVALID_INPUT` hatasıdır.
- `get_document_section` quotation kontrolü önce saklanan hash'i doğrular,
  sonra alıntıyı kısaltma notu çıkarılmış yasal metinle karşılaştırır; kısaltma
  notu artık alıntı olarak "eşleşmez". fikra/bent satırlarının tekil adres
  olmadığı ve üst maddenin getirilmesi gerektiği çıktıda yazar.
- Section aramasındaki tek kelimelik `+400` özel durumu silindi. `relevance`
  alanı "heuristic sıralama skoru" olarak tanımlandı (olasılık ya da cosine
  değil).

Retrieval:

- Exact legal-reference sorgularında sıra: önce hem FTS hem dense ile bulunan
  (hybrid) sonuçlar, sonra yalnız FTS ile bulunan exact eşleşmeler (BM25
  sırasıyla), en sonda yalnız vector sonuçları. Önceki sırada FTS-only
  eşleşmeler hybrid sonucu limit dışına itebiliyordu.
- Belge quality değerlendirmesi artık yalnızca dönen sonuçlar için, bir kez ve
  event loop dışında yapılır (önceden her aday için ana thread'de çalışıyordu).
- Silindi: reranker, karakter tabanlı chunker ve hiçbir şeyi değiştirmeyen
  ayarlar (`BDDK_RERANKER*`, `BDDK_HYBRID_SEARCH`, `BDDK_RRF_K`,
  `BDDK_FTS_THRESHOLD`, `BDDK_EMBEDDING_CHUNK_SIZE`,
  `BDDK_EMBEDDING_CHUNK_OVERLAP`, `BDDK_EMBEDDING_CHUNK_MODE`,
  `BDDK_SYNC_CONCURRENCY`, `BDDK_STALE_CACHE_FALLBACK`). Token chunker
  değişmedi.

Corpus, seed ve CLI:

- Section parser, paragraf numarasını yalın basan KGK standartlarını
  (`kgk_tms_12`: 6 govde penceresi yerine 140 paragraf) indeksler ve kısaltılan
  govde pencerelerini kelime ortasında değil paragraf, satır ya da kelime
  sınırında bitirir (#181'in taşınmış hâli). 14 belgenin chunk'ları değişti:
  13.604 yerine 13.721 chunk. `seed_data/embeddings/` bu 14 belge için
  yenilendi; build yine hiçbir chunk gömmeden tamamlanır.
- `bddk-mcp export` iki seed JSON dosyasını tek kanonik biçimde yazar (id'ye
  göre sıralı, 2 boşluk girinti, LF, sonda yeni satır) ve `downloaded_at` ile
  `extracted_at` değerlerini korur; `build` ardından `export` seed dosyalarını
  byte byte geri verir. Seed dosyaları bir kez bu biçimde yeniden yazıldı;
  içerik değişmedi.
- On Kurul Kararı kaydının karar tarihi, sayısı ve başlığı düzeltildi (1121,
  1157, 1166, 1207, 1208, 1214, 1216, 1270, 751, 773). Scraper, BDDK'nın
  "(25.10.2018 - 8062 ) başlık" biçimindeki satırlarını da ayrıştırır.
- CLI: `--concurrency` pozitif tamsayı ister; boş ya da eksik seed ile `build`
  ve belge hatası olan `sync` exit code 1 verir; çalışma hataları tek satır
  yazılır. Okunamayan bir corpus dosyası not-ready hatası verir; başka modelle
  kurulmuş dosyaya build "dosyayı silip yeniden build edin" der.

Sync:

- Metni kaydedilmiş ama indekslemesi başarısız kalmış belge sonraki `sync`'te
  yeniden denenir. Yalnızca `_degraded` backend'lerin (pdftotext, markitdown)
  bulunduğu makinelerde mevzuat belgeleri önce HTML olarak alınır. HTML
  extractor yorum, doctype, CDATA ve processing instruction düğümlerini metne
  katmaz. `pdftotext` Windows'ta da çalışır.

Temizlik:

- Kaldırılan tasarımlardan kalan dosyalar silindi: `.gitlab-ci.yml`,
  `scripts/validate_all_pages.py`, `docs/DOCUMENTS.md`,
  `seed_data/corpus_scope.yml`, `document_quality_score.py --seed-dir` modu,
  kullanılmayan store metotları ve test yardımcıları.
- Dockerfile'da model indirme katmanı `COPY seed_data/` satırının üstüne
  alındı; seed değişikliği artık model katmanını geçersiz kılmaz.
- Windows'ta başarısız olan iki test (hook izin biti, `bddk-mcp.exe` adı)
  düzeltildi; dokümanlar kodla hizalandı.

### Eklendi — `seed_data/embeddings/` vektör cache'i (2026-11-10)

- `bddk-mcp export` artık chunk vektörlerini de yazar: belge başına bir float16
  `.npy` dosyası ve modelin adını, boyutunu ve her belgenin chunk metinlerinin
  SHA-256 özetini tutan `manifest.json` (`seed_data/embeddings/`, 319 belge
  için 21 MB). `bddk-mcp build` bir belgenin vektörlerini yalnızca yeni
  üretilen chunk metinlerinin özeti ve yapılandırılmış model manifest ile
  birebir eşleşirse cache'ten alır; aksi hâlde belgeyi yeniden gömer.
  `--force` cache'i yok sayar. Değişmemiş seed ile build, modeli hiç
  yüklemeden dizüstü CPU'da 816 saniye yerine 35 saniye sürdü ve birebir aynı
  chunk ve vektörleri üretti; Docker imaj build'indeki corpus adımı da aynı
  oranda kısalır. CLI çıktısı kaç chunk'ın gömüldüğünü, kaçının cache'ten
  geldiğini yazar.
- `search_document_sections` açıklamasındaki PostgreSQL kalıntısı
  (`ts_rank_cd`) SQLite FTS5 BM25 ifadesiyle düzeltildi; `uv.lock`'ta
  `setuptools` 84.1.0'a yükseltildi (Dependabot güvenlik uyarısı: 84.1.0
  öncesinde Windows yol kaçışı açığı).

### Değiştirildi — PostgreSQL + pgvector yerine tek SQLite corpus dosyası (2026-11-10)

- Corpus artık tek bir SQLite dosyası (`BDDK_CORPUS_PATH`, varsayılan
  `corpus/bddk.sqlite`): documents, document_versions, document_sections,
  chunks (float16 embedding BLOB), decision_cache, sync_failures ve meta
  tabloları; section ve chunk metni için FTS5 indeksleri
  (`bddk_mcp/store/corpus_db.py`). Serve süreci dosyayı salt okunur açar
  (`mode=ro`); airlock artık dosya kipinden gelir.
- Arama: dense retrieval numpy ile tüm chunk vektörleri üzerinde tam kosinüs
  (yaklaşık değil, kesin sonuç; birkaç yüz bin chunk'a kadar milisaniyeler),
  keyword retrieval FTS5 BM25 (Türkçe küçük harf ve diacritic normalizasyonu
  Python'da), ikisi RRF ile birleşir; reranker isteğe bağlı kaldı (2026-11-12'de reranker silindi). Section
  search de FTS5 BM25. 319 belgelik seed üzerindeki BKZ/backtesting regresyon
  sorgularının hepsi aynı section'ı ilk sırada buldu.
- Komutlar: `bddk-mcp build` (seed_data → corpus dosyası; section, chunk ve
  embedding üretir; güncelse atlar, `--force` yeniden üretir), `bddk-mcp export`
  (corpus → seed_data JSON), `bddk-mcp sync` (`--corpus`), `bddk-mcp serve`.
  `bddk-mcp migrate`, `bddk-seed` ve kök `seed.py` kaldırıldı.
- Docker imajı corpus dosyasını build aşamasında üretir; Railway'de veritabanı
  servisi, `preDeployCommand` ve DSN değişkenleri yok. `docker-compose.yml`
  tek servise indi. Script'ler `--corpus PATH` alır.
- Kaldırılanlar (son hâlleri `archive/postgres-2026-11-10` etiketinde):
  `bddk_mcp/migrations/` (11 modül + runner, 5.498 satır), `db_lifecycle.py`,
  `db_transport.py`, `db_compatibility.py`, `corpus_coordination.py`,
  `store/bulk_write.py`, retrieval profile hash katmanı, `deploy/postgres-local/`,
  `asyncpg` bağımlılığı, `BDDK_DATABASE_URL` / `BDDK_INGESTION_DATABASE_URL` /
  `BDDK_SCHEMA_OWNER_DATABASE_URL` / `BDDK_PG_POOL_*` / `BDDK_ALLOW_INSECURE_DATABASE`
  / `BDDK_TEST_DATABASE_URL` değişkenleri, pytest `postgres` işareti ve CI'daki
  Postgres işi. Test paketi artık hiçbir servis olmadan, geçici bir corpus
  dosyasıyla çalışır.
- Yeni bağımlılık: `numpy` (zaten sentence-transformers ile geliyordu).
- Model imzası: corpus dosyası hangi embedding modeli ve boyutla üretildiğini
  `meta` tablosunda tutar; farklı model yapılandırılırsa readiness reddeder.

### Kaldırıldı — yayın töreni, kimlik katmanı, iş kuyruğu, admin konsolu, telemetri (2026-11-09)

- 319 belgelik korpus için gereksiz olan katmanlar kaldırıldı; hepsi Git
  geçmişinde duruyor (son hâlleri `9ed84a0` commit'inde,
  `archive/pre-simplify-2026-11-09` etiketi):
  - İmzalı korpus yayın töreni: manifest doğrulama, Ed25519 imza, aşamalı
    doğrulama/aktivasyon, nesil saklama, katalog bütünlük denetimi ve aktif
    yayın muhafızı (`corpus_manifest.py`, `corpus_publication.py`,
    `corpus_generations.py`, `corpus_serving.py`, `catalog_integrity.py`,
    `release_yaml.py`, `resources.py`, `scripts/sign_corpus_manifest.py`,
    `deploy/trust/`). `bootstrap`, `verify-corpus`,
    `verify-and-stage-corpus-release`, `activate-corpus-release`,
    `retain-corpus-generation`, `publish-corpus-release` ve `admit-next-upload`
    komutları gitti; kurulum artık `bddk-mcp migrate` + `bddk-seed import`.
    `bddk-seed import` imza ve manifest okumaz; parçaları ve gömmeleri belge
    metninden yeniden üretir; türetilmiş `seed_data/chunks.json` (19 MB) ve
    `corpus_scope.sig` silindi, `scripts/regen_chunks_seed.py`, tek seferlik
    `repair_943_page_furniture.py` / `repair_rg_box_interleave.py` scriptleri
    ve F1 değerlendirme testi (`tests/test_f1_score.py`) kaldırıldı.
    Uygulanmış v0005–v0011 migration'ları değişmedi; oluşturdukları yönetişim
    tabloları şemada kullanılmadan kalıyor.
  - Çoklu PostgreSQL kimliği sözleşmesi (`db_identity.py`, `deploy/postgres/`
    rol/grant SQL'i, CI'daki rol matrisi). Tek `BDDK_DATABASE_URL` yeter;
    `BDDK_SCHEMA_OWNER_DATABASE_URL` ve `BDDK_INGESTION_DATABASE_URL` ayrı
    login kullanan kurulumlar için isteğe bağlı kaldı ve ona düşer.
  - Operatör profili ve 14 operatör aracı ile Postgres destekli iş yöneticisi
    (`bddk_mcp/jobs/`, `tools/sync.py`, `tools/admin.py`, `ingest/backfill.py`).
    Belge indirme artık senkron `bddk-mcp sync` komutu: kataloğu
    `decision_cache` tablosundan okur, `--refresh-catalog` ile önce BDDK'dan
    yeniler; eski `.cache.json` yolu, `doc_sync.py`'nin kendi komut satırı ve
    hiçbir aracın okumadığı `cache_status` istatistikleri kaldırıldı.
  - Kaldırılan katmanların geride bıraktığı artık kod: istemcideki canlı belge
    çekme yolu (`get_document_markdown`), mağaza istatistikleri ile liste/içe
    aktarma yardımcıları, karakter tabanlı eski parçalayıcı, `check_updates`,
    korelasyon kimliği yardımcıları ve `initialize` uyumluluk sarmalayıcıları.
  - Admin konsolu (`bddk_mcp/admin/`, `bddk-mcp admin-ui`, Railway admin
    servisi) ve `jinja2` bağımlılığı.
  - Telemetri ve sayaçlar (`observability/telemetry.py`, `metrics.py`,
    `BDDK_TELEMETRY_*`).
  - Benchmark/uzman değerlendirme paketi (`benchmark/`, `data/`, `anthropic`
    bağımlılık grubu) ve ilgili testler.
  - `docs/evidence/` altındaki onarım makbuzları, kurtarma kayıtları ve
    tek kullanımlık onarım scriptleri (`rebuild_mevzuat_from_pdf.py`,
    `add_kgk_tms_12.py`); testlerin kullandığı iki dosya `tests/fixtures/`
    altına taşındı. Tarihli planlama belgeleri (ROADMAP, GAP_REGISTER,
    SECURITY_REVIEW, ARCHITECTURE, TESTING_AND_EVALUATION_STRATEGY,
    CORPUS_GOVERNANCE, LEGACY_DATABASE_UPGRADE, ADMIN_UPLOAD_ADMISSION*)
    kaldırıldı.
  - Hukuki durum katmanı: `resolve_regulation_status`, `get_amendment_chain`
    ve `get_cross_references` araçları, `bddk_mcp/regulatory/`, `citations.py`
    (Citation v1), `get_document_section`'ın `as_of` parametresi ve
    `search_document_sections`'ın `expand_references` parametresi. Bu
    tabloları dolduran hiçbir üretim yolu yoktu; `seed_data/` ile kurulan
    veritabanında araçlar her soruya "doğrulanmış veri yok" dönüyordu.
    `quotation` (alıntıyı kayıtlı metinde harfi harfine doğrulama) kaldı.
    `regulatory_*` tabloları ve `regulatory_validated_*` görünümleri şemada
    (v0004, v0006, v0009) kullanılmadan duruyor; `document_sections` sorguları
    artık atıf görünümüne katılmıyor.
- Yerel Compose topolojisi `db` + `bddk-init` + `bddk` servislerine indi;
  Railway ön-dağıtım komutu `migrate && bddk-seed import --reindex-existing`.
- Değişmeyen: 14 genel araç, HTTP/JWT taşıma sözleşmesi, migration kayıt
  defteri, korpus artefaktları.

### Kaldırıldı — banka üretim hazırlığı katmanı (2026-11-08)

- Yakın vadede banka kurulumu planlanmadığı için şu parçalar kaldırıldı; hepsi
  Git geçmişinde duruyor (son hâlleri `e858590` commit'inde):
  - OpenShift başlangıç manifestleri ve çevrimdışı kabul kontrolü
    (`deploy/openshift/`, `deploy/openshift-overlays/`,
    `bddk_mcp/openshift_acceptance.py`) ile CI'daki Kustomize adımı.
  - Yedekten geri yükleme tatbikatı (`bddk_mcp/operations/`,
    `scripts/recovery_drill.py`, `docs/RECOVERY_DRILLS.md`). Graf testinin
    kullandığı saklı nesil mühür sorgusu test dosyasına taşındı. Geçmiş tatbikat
    kanıtları `docs/evidence/` altında kalır.
  - Operasyonel hedefler sözleşmesi (`bddk_mcp/operational_objectives.py`,
    `docs/decisions/`).
  - Supply-chain kanıt hattı (`supply-chain/`,
    `.github/workflows/supply-chain.yml` ve scriptleri). Depo artık SBOM,
    imaj taraması ve provenance kanıtı üretmez.

### Kaldırıldı — eski şema kabul yolu (`--adopt-legacy`)

- Kayıt defteri olmayan ön-v0001 veritabanlarını sisteme alan
  `bddk-mcp migrate --adopt-legacy` bayrağı ve `bddk_mcp/migrations/legacy.py`
  kaldırıldı; bu şekilde bir veritabanı artık kullanılmıyor. Kayıt defteri
  olmayan bir şema normal migration'da reddedilip geri alınmaya devam eder
  (yeni test: `test_migrate_refuses_an_unmanaged_schema_without_a_ledger`).
  v0003'teki `bddk_meta.legacy_schema_adoptions` tablosu denetim geçmişi olarak
  kalır.

### Güvenlik — bağımlılık ve temel imaj

- `fsspec` 2026.2.0 → 2026.9.0 (`uv.lock`; GHSA-27vj-qcqg-25rc 2026.6.0'da kapanıyor). Yalnızca bu paket yükseltildi.
- Dockerfile temel imajı aynı etiketin (`python:3.13.16-slim-trixie`) güncel derlemesine sabitlendi (`sha256:70729b46…`); iki imaj arasındaki tek paket farkı `libpcre2-8-0` 10.46-1~deb13u2 → deb13u3 (CVE-2026-103111 düzeltmesi).
- İmaj sertleştirildi: sunucunun hiç kullanmadığı terminal, oturum açma, mount ve Perl paketleri (`util-linux`, `mount`, `login`, `bsdutils`, `ncurses`, `readline`, `bash`, `perl-base` ve bağlı kütüphaneler) Dockerfile'da kaldırıldı. `sh`, coreutils ve findutils kaldı (Railway başlatma komutunu `sh -c` ile sarar). CI'daki embedding duman testi, aynı kaldırma adımı uygulanmış yerel bir arm64 test imajında iki kullanıcı kimliğiyle de geçti. Bu imaj önceki temel imaj ve kilit dosyasıyla kuruldu; son Dockerfile'ın uçtan uca doğrulaması CI'daki amd64 derlemesinde yapılacak.
- Güvenlik istisnaları 53'ten 13'e indi. 42'si kaldırılan paketlerle, biri temel imaj güncellemesiyle ortadan kalktı. Kalan 13 bulgu Debian'ın henüz düzeltmediği çekirdek kütüphanelerde: glibc, GCC 14 çalışma kütüphaneleri (yeni CVE-2026-95619 dahil), `libacl1` ve `zlib1g`. Bu 13 istisna yeni Dockerfile hash'ine bağlandı, sahibin kararıyla 15/11/2026'ya kadar (30 gün) yenilendi ve banka incelemesini bekliyor. amd64 temel imaj taraması (Grype 0.115.0, veritabanı 16/10/2026) bu 13 kayıtla birebir eşleşiyor. Python bağımlılıklarında High/Critical bulgu yok.
- Sır istisnaları 101 kayıt (README'deki 99 sayısı, belge 943'ün iki `version_key` özetini saymıyordu). Her satır kayıtlı commit'inde yeniden okundu ve gizli bilgi olmadığı doğrulandı; dalların artık erişmediği beş commit GitHub'dan SHA ile çekildi. Yerel Gitleaks 8.30.1 tüm-tarihçe taraması 96 bulgu buldu; hepsi istisna kapsamında. Bu geçişin değişiklikleri commit edilmiş gibi tarandığında yeni bulgu çıkmadı. Süreler 15/11/2026'ya kadar bir kez (30 gün) yenilendi.

### Düzeltildi — 2026-10-16 belge kalite geçişi (297 belge)

- Motor etiketleri zaten 319/319 "clean" idi; asıl hatalar sınıf taramasıyla bulundu ve resmî kaynağa (PDF/RG) göre düzeltildi: cp1254 C1 karakterleri, `Đ`→`İ` gibi yanlış kod çözümleri, iki yana yaslı satırlarda karışan kelime sırası, düzleşmiş tablolar, sayfa numarası/antet artıkları, cümle ortasında bölünmüş paragraflar, yapışık dipnot işaretleri ve formül değişkenlerinin alt/üst simgeleri (ör. `VFi` → `VF$_{i}$`), eksik maddeler (`mevzuat_21425` MADDE 10–12) ve kod bloğu gibi görünen girintili paragraflar.
- `mevzuat_21194` (İDD Tebliği): KOBİ korelasyon formülü, fıkrayı değiştiren 14/3/2018 tarihli RG 30360 Tebliği'nin kendi görüntüsünden aktarıldı; kayıtlı formül (`max[S, 5]`, `/36`) ne 2015 aslı (`max{4, S}`, `−4`, `/36`) ne de değişiklikti. Konsolide metin aynı fıkranın açıklamasında 2016 (RG 29599) metnini bırakmıştı; açıklama da RG 30360'tan birebir alındı. RG 29599, 30360 ve 30742 ile değişen diğer bütün hükümler konsolide metinde doğru bulundu.
- `1296` (2025/1 sayılı Genelge) aynı adresteki güncel resmî PDF'e getirildi: Kurul Kararı 20/8/2026-11559 ile katılma hesabı kaynaklı pozisyonlar %100 → %80, tarih "29/05/2025". Tek boşluklu metindeki "Yönetmelik EK-2’de yer alan formüllerde" ifadesi başka bir yönetmeliğin formüllerine atıftır; sahip onayıyla kalite politikası `markdown-quality-assessment-v6` bu incelenmiş ifadeyi formül-atıf kuralının dışında tutar (politika kimliği değiştiği için sürüm yayımında yeniden indeksleme gerekir).
- `mevzuat_19498` (Likidite Karşılama Oranı Yönetmeliği): Madde 9'un atıf yaptığı Ek-1 ve Ek-2 bildirim cetvelleri (dikkate alınma, çıkış ve giriş oranları) korpusta yoktu; içe aktarıcı resmî ek arşivindeki yalnızca .docx dosyasını okuyordu. İki Excel eki resmî arşivden (`7.5.19498-ek.zip`) tablo olarak eklendi; 628 ve 634 hücrenin tamamı sırasıyla doğrulandı. Ek-2'deki konsolide oranlar Ek-1'den farklıdır (ör. zorunlu karşılıklar %50). 69 mevzuat belgesinin ek arşivleri tarandı; eksik ek başka belgede yok.
- Düz metin formüller (1288, `mevzuat_21195`, `mevzuat_39158`, 19498 Ek-3) resmî kaynak satırlarıyla karşılaştırılarak LaTeX'e çevrildi; formülden söz edip formül-farkında olmayan belge kalmadı.
- Bölüm ayrıştırıcısı `turkish-regulatory-sections-v9`: uzun tireli (`EK–1`) ve Markdown başlıklı (`## EK-1`) ek başlıklarını tanır; 1305, `mevzuat_20029`, `mevzuat_21192` ve `mevzuat_21194`'te son madde artık ekleri yutmuyor. Noktalı biçim (`İlke 6.`, `EK 1.`) bilerek kapsam dışı: genel olarak tanınması 945/946'da numaralı paragraf bölümlerini siliyordu.
- Güncellik kontrolü (16/10/2026): 69 mevzuat.gov.tr belgesi bugünkü resmî PDF'lerle sözcük ve sayı düzeyinde karşılaştırıldı; 8 Nisan–16 Ekim 2026 arasındaki 192 günlük Resmî Gazete fihristi (4 mükerrer dahil) korpus mevzuatını değiştiren düzenleme için tarandı. Bulunan tek değişiklik (RG 30/4/2026-33239, `mevzuat_16290`) korpusta zaten işli. Kayıt: `docs/evidence/document-repairs/quality-pass-2026-10-16/currency/check-2026-10-16.json`.
- Belge başına SHA-256 makbuzları, temel (baseline) belge seti, araçlar ve spec'ler: `docs/evidence/document-repairs/quality-pass-2026-10-16/`. Yeni corpus kapsam kimliği `bddk-job-corpus-quality-pass-local-v11-2026-10-16` bölüm ayrıştırıcısı v10 ile chunk dosyası yeniden üretildikten sonra sahip tarafından 18/10/2026'da yeniden imzalandı (manifest sha `99997032…`); benchmark veri seti `0.1.0-draft.11` olarak bu manifeste bağlandı. #178 ile birleştirildi ve 18/10/2026'da Railway geliştirme/önizleme servisine dağıtıldı (tek seferlik `--force`, #179 ile geri alındı); yönetişimli staging ve aktivasyon yapılmadı. İçe aktarmadan önce Railway veritabanında 943'ün hukuki sürüm paketine ait 111 doğrulanmış hükmün v10 bölüm bağlantıları, sahip onayıyla ve yedeklenerek temizlendi; paket v11 metnine göre yeniden üretilmeden bağlantılar geri kurulmamalı.
- Kalite taraması: `c1_control_char` SQL sinyali eklendi; `camelcase_concat` ve formül-atıf sinyalleri artık incelenmiş tanımlayıcıları bilen belge değerlendirmesinden sayılıyor.

### Düzeltildi — bölüm ayrıştırıcısı v10 ve kalite politikası v7 (17/10/2026)

- Formül sınırı: bölüm sonundaki formül ve denklem satırları (`$$`, LaTeX komutları, ` = ` içeren satırlar) artık "sonraki maddenin başlığı" sanılıp kırpılmıyor. 7 belgede 44 bölüm formülünü geri aldı: 1312 madde 39'un formülü hiçbir bölümde değildi, 1313'te bazı fıkralar yarım kalmış bir `$$` ile bitiyordu.
- Kalite politikası `markdown-quality-assessment-v7` (sahip onaylı): aynı belgedeki başka bir madde, fıkra veya eke yapılan 10 formül atfı (1312, 1313, `mevzuat_19498`, `mevzuat_20029`) belge bazında incelenmiş atıf olarak formül-atıf kuralının dışında tutulur. Her atfın gösterdiği formülün belgede bulunduğu `docs/evidence/document-repairs/quality-pass-2026-10-16/policy/internal-formula-references.json` kaydındadır; 1312'deki atıf taslağın kendi numaralandırmasıyla 40. maddeyi gösterir, formül taslağın 38. maddesindedir. Bölüm düzeyindeki formül notu 27'den 1'e indi; kalan tek not 20.000 karakter sınırında kesilen `mevzuat_21192` EK-4'tür ve kesilme notuyla gelir.
- Araya eklenen maddeler kendi bölümüdür: `Madde 66/A` kayıtta `madde 66/a` olur (10 belgede 32 madde; `mevzuat_6362`'de 13). Önceden üst madde (ör. `mevzuat_5411` madde 66) eklenen maddenin metnini de taşıyordu. `EK MADDE N` yeni `ek_madde` türüdür (`mevzuat_5411`, `mevzuat_38568`). 1230'un `MADDE N. –` biçimi (27 madde, 3 geçici madde), `GEÇİÇİ MADDE` (`mevzuat_19864`), `GEÇİCİMADDE` (`mevzuat_10735`), tiresiz `MADDE 16 (1)` (1313) ve 947'nin noktalı `İlke N.`/`EK N.` başlıkları tanınır. Noktalı biçim yalnızca belgede iki noktalı bir ilke başlığı da varsa sayılır; 945/946 gibi numaralı paragraf gövdeli rehberler değişmez.
- Kimlik almayan sınırlar: `MADDE 146 ila 149` notu (`mevzuat_6362`), kanun ekleri `(1) SAYILI LİSTE` / `(I) SAYILI CETVEL` (`mevzuat_6362`, `mevzuat_5411`) ve taslakların `GENEL GEREKÇE` bölümü önceki maddeyi kapatır ve gövde olarak aranır. Önceden `mevzuat_5411` madde 171 cetveli yutup 20.000 karakterde kesiliyordu; 10 taslakta son madde gerekçeyi hüküm metni gibi taşıyordu.
- Sahte kimlikler kaldırıldı (23 bent, 6 fıkra, 1 madde): satır kaymasıyla yeni satıra düşen atıflar ("… 7 nci maddenin⏎(2) numaralı fıkrası"), "(c) bentleri", "(g) sütunu", "(a) ile (e)" gibi açıklamalar, Roma rakamlı "(i)/(ii)" alt listeleri, kanun eki başlıkları ve 1230'da başka bir kanundan alıntılanan "Madde 44". 1230 madde 20 son cümlesini geri aldı; mülga tek satırlık maddeler (ör. `mevzuat_5411` madde 72) artık sonraki maddenin başlığını taşımıyor; EK tablolarının son satırları ve kısaltma listeleri eklerinde kalıyor.
- Sorguda `madde 66/A`, `madde 66A` ve `66/A. madde` harfli maddeye; `geçici madde N` `gecici_madde`'ye, `ek madde N` `ek_madde`'ye çözülür. Önceden üçü de sessizce başka bir maddeyi döndürüyordu. `section_ref` artık harfli maddeyi (`66/A`) ve tek harfli bendi (`a`, `ç`) kabul eder; ayrıştırıcı, girdi sözleşmesi ve sorgu aynı `canonical_section_ref` biçimini kullanır. Hukuki atıf eşleşme profili `turkish-legal-reference-bypass-v2` oldu. Davranış notu: `madde 7/a` artık araya eklenen 7/A maddesi olarak okunur; belgede böyle bir madde yoksa yanıt "istenen hüküm bulunamadı" uyarısıyla döner, madde 7 sessizce verilmez.
- CI: hazır Kustomize'ı daha yeni olan GitHub çalıştırıcılarında OpenShift kabul testleri artık yalnızca sabit Kustomize v5.8.1 varken çalışır; zorunlu kol (`BDDK_REQUIRE_KUSTOMIZE=1`) hiç atlamaz.
- Etki (v9'a göre): 35 belgenin bölümleri değişti; bölüm sayısı 8.350 → 8.415, chunk sayısı 13.552 → 13.604. 60 madde, 14 ilke, 5 geçici madde, 2 ek madde ve 2 ek yeni bölüm oldu; benchmark'ın dayandığı 18 bölümün içeriği değişmedi. PostgreSQL şema değişikliği yoktur. Ayrıştırıcı, kalite ve atıf eşleşme profilleri değiştiği için yayında yeniden indeksleme gerekir.

### Değiştirildi — arama gecikmesi, test süresi ve imaj boyutu

- Belge kalite değerlendirmesi (`assess_markdown_quality`) metin, belge kimliği ve kayıt defteri
  girdisine göre bellekte önbelleğe alınır; hibrit aramanın yerel medyan gecikmesi 389 ms'den
  yaklaşık 50 ms'ye indi. Bölüm sorguları belgenin tam gövdesini satır başına taşımaz; gevşek
  bölüm araması kaynak aralığını yalnızca son isabetler için getirir. Sonuçlar ve retrieval
  profil hash'i değişmez.
- Gömme modeli süreç başına bir kez yüklenir; Postgres test süiti yerelde 10,5 dakikadan
  2 dakikaya indi, CI'da aynı etki beklenir.
- Dockerfile sahiplik ve izin ayarlarını dosyaları oluşturan katmanların içinde uygular; ayrı
  `chgrp/chmod` katmanı tüm ağacı kopyalayarak imajı ikiye katlıyordu (14,2 GB → 7,23 GB,
  izin eşliği doğrulandı). `supply-chain/policy.json` içindeki Dockerfile hash'i buna göre
  güncellendi.

### Değiştirildi — lisans

- Lisans MIT'den [PolyForm Noncommercial 1.0.0](LICENSE)'a değiştirildi (SPDX `PolyForm-Noncommercial-1.0.0`; `pyproject.toml` ve NLI veri seti üst verisi güncellendi). Bankalar, finansal kuruluşlar ve diğer şirketler dahil her türlü ticari kullanım, telif hakkı sahibinden ayrı yazılı lisans alınmadan yasaktır. README'ye ticari lisans iletişim satırı eklendi. GitHub deposu 2 Ekim 2026'da private yapıldı.
- `aa62301` commit'i ve öncesi sürümler MIT kapsamında yayımlanmıştır; bu izin yalnızca o sürümler için geçerliliğini korur. Hukuki inceleme henüz yapılmadı (`docs/LICENSING_AND_PROVENANCE.md`, GAP LIC-001).

### Düzeltildi — resmî PDF'ten yeniden kurulan yirmi mevzuat metni

- Metin çıkarımı (markitdown) iki yana yaslı satırlardaki kelimelerin sırasını bozmuş, tablo hücrelerini dağıtmış ve resim olarak basılmış formülleri düşürmüştü. Yirmi mevzuat.gov.tr metni resmî PDF'lerinden baştan kuruldu ve her sayfası PDF ile gözle karşılaştırıldı: on altı yönetmelik (`mevzuat_22599`, `16071`, `39171`, `40203`, `40519`, `20645`, `10522`, `38416`, `10731`, `42359`, `38568`, `23228`, `11203`, `10746`, `38797`, `42628`), bir tebliğ (`mevzuat_11201`) ve üç kanun (Bankacılık Kanunu `mevzuat_5411`, Sermaye Piyasası Kanunu `6362`, Banka Kartları ve Kredi Kartları Kanunu `5464`). Örnek: Karşılık Yönetmeliği Madde 6/5'in ortası kayıptı.
- Araç: `scripts/rebuild_mevzuat_from_pdf.py`. Tablolar ve formüller dışındaki metin PDF metin katmanıyla harfi harfine aynıdır; her tablo, yerini aldığı PDF satırlarının kelimeleriyle denetlenir. Her belgenin değişen bölümleri `docs/evidence/document-repairs/mevzuat_<id>.json` içinde listelenir; resmî metindeki yazım tuhaflıkları düzeltilmeden korunur ve aynı yerde kayıtlıdır.
- BHFOR Yönetmeliği'nin (`mevzuat_42628`) 34 formül resmi gözle okunarak LaTeX'e çevrildi (`native-sources/mevzuat_42628-formulas.json`). Bu bir asistan okumasıdır; resimlerin metin katmanı olmadığından makineyle doğrulanamaz.
- Kanunlar ve tebliğ için mevzuat.gov.tr sabit PDF sunmuyor; kaynak, her istekte yeniden üretilen güncel konsolide metin PDF'idir. İncelenen kopyalar `native-sources/<n>-generated.pdf.gz` olarak saklandı.
- Bölüm dizini artık madde numarasından hemen sonra gelen dipnot işaretini tanıyor (`MADDE 7$^{(1)}$ –`); bu maddeler dizinden düşmüyor.
- Corpus kapsam beyanı yeni bir kimlikle (`…-pdf-rebuilds-local-v10-2026-10-02`) güncellendi. Yayına alınmadı (staging/aktivasyon yok). Bu bir teknik doğrulamadır, hukuki onay değildir.

### Eklendi — KGK TMS 12

- Corpus'a KGK'nın "TMS 12 Gelir Vergileri" standardı (`kgk_tms_12`) eklendi. Kaynak, KGK'nın 2026 Mavi Kitap PDF'idir; bir kez elle indirildi ve `scripts/add_kgk_tms_12.py` ile SHA-256 kontrolüyle çıkarıldı. kgk.gov.tr canlı erişim izin listesine eklenmedi.
- Yeni kategori: `Muhasebe Standardı`. Arama araçlarında bu kategoriyle filtrelenebilir.
- Standart paragraf numaralarıyla değil metin pencereleriyle dizinlenir; "paragraf 24'ü getir" biçimli bölüm çağrısı henüz yoktur. Yürürlük durumu modellenmez.
- Corpus kapsam beyanı KGK kaynak sınıfını içerecek şekilde güncellendi ve yeniden imzalandı.

### Değiştirildi — giriş sayfaları

- `README.md` ve `README.en.md` genel banka kullanıcısı için baştan yazıldı. Aynı iskelet; biri Türkçe, biri İngilizce. Operatör prosedürü `docs/DEPLOYMENT.md` içinde kalır.
- `docs/DEPLOYMENT.md` içindeki Ağustos 2026 yeniden imzalama notu kaldırıldı. Güncel corpus kimliği `docs/STATUS.md` dosyasına bırakıldı.
- Belge indeksi ve depo yapısı, İngilizce dosyayı artık ayrı işletme kılavuzu diye tanıtmaz.

### Düzeltildi — yetkilendirme, corpus bütünlüğü ve veri işleme

- Uzak yönetim konsolu bearer token ve oturum girişinde yapılandırılmış tüm
  JWT scope'larını zorunlu tutar. `/health/ready` veritabanına erişilemiyorsa
  veya kontrol zaman aşımına uğrarsa `503` döndürür.
- Şema v11, ilişki grafını corpus epoch'una, release fingerprint'ine, yayın
  kilitlerine, typed retention'a ve recovery envanterine ekler. Migration ve
  `deploy/postgres/02_grants.sql` sonrasında corpus yeniden doğrulanıp stage
  edilmeli ve yeni istek aktive edilmelidir; önceki aktivasyonlar ve bekleyen
  istekler geçersizleşir. Yeni retained generation'lar 18 ilişki içerir;
  eski 17 ilişkili mühürler korunur. Seed artifact biçimi değişmez.
- Gzip HTTP yanıtlarının ikinci kez açılması önlendi; açılmış içerik için
  boyut sınırı korunur. OCR event loop dışında, syncer başına tek işçiyle
  çalışır; iptal edilen iş, işçi tamamlanmadan lease'i bırakmaz.
- Haftalık bülten trend ve karşılaştırmalarında eksik veya geçersiz gözlemler
  sıfıra çevrilmez; yetersiz seri hata olarak bildirilir.

### Sadeleştirildi — kullanılmayan kod ve istemci tekrarları

- Kullanılmayan task/backfill alanları ve sync metadata metotları kaldırıldı.
  MCP kontrol script'leri mevcut resmi SDK'yı, indirme kontrolleri ortak HTTP
  doğrulamasını, dosya hash'leri ve sıralı tekilleştirme standart kütüphaneyi
  kullanır; yeni bağımlılık eklenmedi.

### Eklendi — uzaktan yönetim konsolu (JWT)

- `bddk-mcp admin-ui` loopback dışı bind'i `BDDK_ADMIN_REMOTE_ENABLED=true` ve
  tam JWT (`bddk.operator` scope, Host/Origin allowlist) ile kabul eder.
  Tarayıcı `/login` üzerinden HttpOnly çerez koyar; `Authorization: Bearer`
  de geçerlidir. Kimlik doğrulamasız uzak bind ve `BDDK_HTTP_ALLOW_UNAUTHENTICATED`
  reddedilir. Mağaza hataları artık istisna metnini yüzeye çıkarmaz.

### Kaldırıldı — kullanılmayan ağırlık

- `uvloop` ve doğrudan `pillow` bağımlılıkları; Hugging Face Spaces Dockerfile'ı;
  tarihli `TARGET_ARCHITECTURE` / `REPOSITORY_REVIEW` / `EXECUTIVE_SUMMARY`
  örtüleri; üretim paketinden `InMemoryJobRepository`; `BddkVectorStoreError`.
- `BDDK_AUTO_SYNC` artık yapılandırma sabiti değil; `true` serving'de hâlâ reddedilir.

### Düzeltildi — section search kesikleri ve sızan madde başlıkları

- `search_document_sections` artık 220 karakterlik baştan kesik önizleme yerine
  sorguya hizalı 2000 karakterlik alıntıyı metin olarak döndürür; limit rakamları
  (`%15` vb.) kesilmez.
- Filtresiz section araması iç içe `fikra`/`bent` satırlarını atlar; tam madde
  metni öne çıkar. `section_type=fikra` hâlâ çalışır.
- Section parser `turkish-regulatory-sections-v6`: sonraki maddenin kısa başlığı
  ve PDF satır artıkları önceki maddenin gövdesinden budanır. Mevcut
  `document_sections` satırları için `scripts/reindex_document_sections.py --execute`
  gerekir (`bootstrap --reindex-existing` vektörde zaten olan belgeleri atlar).

Bu bölüm `main` üzerindeki `5684a34` tabanından başlayıp
`codex/roadmap-v5.0.1` dalında commitlenmiş çalışmaları, 16 Temmuz 2026 tarihli
şema-v8 kod doğrulama noktası `7cd6242` ve operasyon/recovery belge noktası
`83d31a4` dâhil olmak üzere kapsar. Buradaki maddeler yayımlanmış sürüm
taahhüdü veya banka üretim kabulü değildir.

### Eklendi — section parser v5 ve loopback yönetim konsolu (2026-08)

- Section parser `turkish-regulatory-sections-v5`'e yükseltildi (#135, #136):
  tire ile numaralanmış paragraflar (`paragraf` section tipi) ve geç annex
  başlıkları arkasındaki gövdeler artık indekslenir; numaralı-paragraf Rehber
  ve Genelge belgeleri bölüm bazında erişilebilir. Parser sürümü retrieval
  profile kimliğinin parçası olduğundan bu değişiklik imzalı corpus'un
  yeniden üretilip yeniden imzalanmasını gerektirir (gap register CUR-018).
- Salt-okunur, yalnız loopback'e bağlanan operatör konsolu eklendi
  (`bddk-mcp admin-ui`, `bddk_mcp/admin/`; #125). Loopback dışı bind
  konfigürasyon hatası olarak reddedilir; konsol hiçbir deployment
  manifest'inin parçası değildir.

### Düzeltildi — banka teslimat denetimi bulguları (2026-08-26)

- Kurum dizini, duyuru ve haftalık bülten snapshot araçları ağ/upstream
  hatasını artık boş sonuç ("NO RESULTS") olarak sunmuyor; toplam başarısızlık
  `[ERROR:UPSTREAM_FETCH_FAILED] retryable=true` olarak yüzeye çıkar, kısmi
  başarısızlık uyarı satırı ekler, `get_regulatory_digest` erişilemeyen
  bölümleri açıkça işaretler ve `check_bddk_updates` boş (yanlış) baseline
  kaydetmez. Erişilemeyen upstream'de seri sayfa/kategori döngüleri hızlı
  başarısız olur.
- Egress allowlist'ini atlayan kullanılmayan `core/utils.py` HTTP yardımcıları
  silindi; geriye kalan tüm giden istekler ya `core/outbound_http.py` sınırından
  ya da `ingest/doc_sync.py` içindeki eşdeğer sınırlı akış yolundan geçer
  (exact-host allowlist, redirect yeniden doğrulama, boyut sınırları).
- Model varlıkları startup'ta doğrulanır: `HF_HUB_OFFLINE` altında
  `BDDK_EMBEDDING_MODEL_PATH` zorunludur ve `BDDK_RERANKER=true` yerel model
  yolu olmadan başlatmayı reddeder (ilk aramada indirme hatası yerine).
  `BddkApiClient` artık varsayılan olarak canlı cache doldurmayı reddeder
  (`allow_live_population=False`).
- `pdf2image` yalnız GPU OCR yolunda kullanıldığından `gpu` bağımlılık
  grubuna taşındı; serving imajı poppler gerektiren bağımlılık taşımaz.
- `railway.toml` onarıldı (gerekli HTTP-policy değişkenleri belgelendi,
  `/health/ready` healthcheck ve `--require-verified-signature` eklendi,
  kullanılmayan volume kaldırıldı) ve ölü `Procfile` silindi; Railway/Spaces
  banka yolu dışında geliştirme/önizleme profilleri olarak işaretlendi.
- `deploy/openshift/README.md` şema anlatımı v8'den güncel v10'a düzeltildi ve
  runbook'un başına fail-closed tuzak listesi eklendi: varsayılan sıfır-egress
  render'ı, canlı araçlar için banka proxy'sinde hostname bazlı
  bddk.org.tr/mevzuat.gov.tr whitelist gerekliliği ve bank-bootstrap
  overlay'inin measured-freshness kapısı.
- Compose'a CI eşleniği `bddk_test` fixture'ını kuran `bddk-test-db` servisi
  eklendi (lokal `postgres` işaretli testler sessizce skip olmuyor) ve
  teslim edilen kimlik doğrulamasız uzak public profili uçtan uca çağıran
  resmi-istemci testi eklendi.
- Corpus manifest'ini sonuçlandırıp imzalayan sahip yardımcı script'i eklendi
  (`scripts/sign_corpus_manifest.py`): yanlış anahtar ve artifact drift'inde
  imzalamayı reddeder, imzayı yazmadan önce yeniden doğrular.

### Değiştirildi — banka sunucusu migrasyonu ve Keycloak'ın kaldırılması

- MCP sunucusu banka sunucularına taşınırken kullanıcıya dönük Keycloak/OAuth
  katmanı kaldırıldı: departman kullanıcıları (yaklaşık 25 kişi) Open WebUI
  önyüzüne Microsoft (Active Directory) hesaplarıyla LDAP üzerinden giriş
  yapar; önyüz MCP public profiline banka ağı içinde doğrudan,
  `BDDK_HTTP_ALLOW_UNAUTHENTICATED=true` ile bearer token olmadan bağlanır.
  Erişim denetimi ağ izolasyonuna (Route/ingress erişim kısıtları, exact
  Host/Origin allowlist'leri ve NetworkPolicy'ler) devredildi.
- `deploy/keycloak/` (realm, rebuild ve probe araçları) silindi.
- `deploy/open-webui/` sadeleştirildi: MCP OAuth token-refresh yaması, yamalı
  imaj build'i ve ilgili testler kaldırıldı; compose artık digest ile
  sabitlenmiş upstream imajı LDAP yapılandırması ve kapalı self-signup ile
  çalıştırır.
- OpenShift starter'ında public ConfigMap'ten tüm `BDDK_JWT_*` anahtarları
  kaldırılıp `BDDK_HTTP_ALLOW_UNAUTHENTICATED: "true"` eklendi; JWT sözleşmesi
  yalnız operator profilinde kaldı (uygulama, operator profilinin loopback
  dışı kimlik doğrulamasız çalışmasını zaten reddeder). Offline preflight
  sözleşmesi, kabul örnek girdileri ve testler buna göre güncellendi; public
  runtime için IdP/JWKS egress'i artık zorunlu değil, aksine yasak.

### Eklendi — repository bakımı

- GitHub issue formları, pull-request şablonu, CODEOWNERS, `SECURITY.md` ve
  `CONTRIBUTING.md` eklendi.
- Dokümantasyon için tek bir indeks, güncel repository durum sayfası ve mevcut
  dizin yerleşimini açıklayan yapı rehberi eklendi; tarihsel incelemeler güncel
  çalışma sözleşmesinden açıkça ayrıldı.
- `seed_data`, evidence JSON, lock file ve binary fixture sunumu
  `.gitattributes` ile sınıflandırıldı.
- Repository kökü, doküman linkleri, issue-form YAML'ları, sürüm eşleşmesi ve
  governance dosyalarını doğrulayan otomatik hygiene kontrolü CI'a eklendi.
- Haftalık Python ve GitHub Actions güncellemelerini gruplayan Dependabot
  politikası eklendi; merge sonrası branch silme repository ayarıyla branch
  birikiminin tekrar oluşması önlendi.
- Açık dependency uyarılarını gidermek için kilitli `setuptools` 84.0.0 ve
  `torch` 2.13.0 sürümlerine yükseltildi; minimum build/GPU gereksinimleri bu
  güvenli tabanlara çekildi.

### Eklendi — MCP ve uygulama yaşam döngüsü

- Tek kaynaklı MCP araç kayıt sistemi eklendi. Güncel sözleşme 17 public araç
  ve operator profilinde bunlara eklenen 14 araç olmak üzere toplam 31 aracı
  açıkça tanımlar.
- Public ve operator çalışma profilleri ayrıldı; operator araçları public
  süreçte kaydedilmez.
- Araçların `readOnlyHint`, `destructiveHint`, `idempotentHint` ve
  `openWorldHint` risk açıklamaları merkezi kayıt sistemine bağlandı.
- FastMCP'nin varsayılan olarak görmezden geldiği bilinmeyen girdi alanları
  `extra="forbid"` ile reddedilir hâle getirildi.
- Resmî MCP istemcisiyle stdio ve Streamable HTTP üzerinde initialize,
  `tools/list`, çağrı, hata ve kapanış akışlarını sınayan sözleşme testleri
  eklendi.
- `bddk://corpus/active-release` adlı tek MCP resource'u eklendi. Bu geliştirme
  noktasında MCP prompt tanımı yoktur.
- Kurulu paketten çalıştırılabilen `bddk-mcp` ve `bddk-seed` konsol girişleri,
  ayrı migrate/bootstrap komutları ve açık uygulama yaşam döngüsü eklendi.
- Liveness ve readiness uçları ile veritabanı hazırlık kontrolleri eklendi;
  serving süreci artık migration, seed import veya embedding backfill yapmaz.
- Public local-corpus çağrılarının aynı aktif release kimliği altında başlayıp
  bitmesini doğrulayan corpus-serving lease/guard katmanı eklendi.
- Public araçların yalnız seçilmiş corpus kapsamını temsil ettiği ve eksik
  sonucun “mevzuat yoktur” anlamına gelmediği kullanım sınırları dokümante
  edildi.

### Eklendi — düzenleyici çapraz referans grafı

- `regulatory_relations` tablosu eklendi (migration `v0009`): sekiz onaylı
  ilişki türü (`amends`, `repeals`, `replaces`, `consolidates`, `implements`,
  `cites`, `defines`, `exception_to`), türetilmiş `rel_sha256` kimliği, kanıt
  referansı ve inceleme (validation) kolonları ile immutable-or-identical
  yazım disiplini.
- Serving profillerinin okuduğu üç güvenlik-bariyerli görünüm eklendi:
  `regulatory_validated_relations`, `regulatory_validated_legal_versions`,
  `regulatory_validated_legal_events`. Doğrulanmamış veya fixture-artifact'lı
  satırlar hiçbir çalışma kimliğine sunulmaz.
- Türkçe mevzuat metninden deterministik (regex tabanlı, LLM'siz) aday kenar
  çıkarımı eklendi; belirsiz hedefler fuzzy eşleme yerine
  `target_external_ref` olarak saklanır ve makine çıkarımı adaylar insan
  incelemesine kadar `unvalidated` kalır.
- İki yeni public MCP aracı eklendi: `get_amendment_chain` (doğrulanmış sürüm
  zinciri ve gelen değişiklik kenarları) ve `get_cross_references`
  (doğrulanmış çapraz referans komşuluğu). Public sözleşme 17 araca, operator
  toplamı 31 araca çıktı.
- `search_document_sections` aracına opsiyonel `expand_references` bayrağı
  eklendi: doğrulanmış kenarlar bir adım takip edilip ilişkili bölüm
  işaretçileri eklenir; içerik hiçbir zaman inline edilmez ve genişletme
  hatası düz aramaya geri düşer.

### Eklendi — güvenlik ve yetki ayrımı

- PostgreSQL için schema owner, ingestion, release verifier, release publisher,
  public reader, operator runtime ve telemetry writer sorumlulukları ayrı NOLOGIN rollerine
  ayrıldı; çalışma kimliğinin beklenen LOGIN/membership/effective privilege
  sözleşmesine uyduğu bağlantı açılışında doğrulanır.
- PostgreSQL bağlantılarında beklenen veritabanı/şema sahibi, ACL kaynağı,
  katalog bütünlüğü ve `sslmode=verify-full` koşullarını fail-closed doğrulayan
  kontroller eklendi.
- Uzak Streamable HTTP açılışı için exact Host, HTTPS Origin, asimetrik
  JWT/JWKS, profil-scope, istek gövdesi, hız ve eşzamanlılık sınırları eklendi.
  Güvenli uzak yapılandırma eksikse non-loopback açılış reddedilir.
- İlk gövde byte'ı, iki chunk arası ve toplam gövde okuma süresi ayrı
  sınırlandı; boş ASGI event'i ilk byte sayılmaz, aşırı uzun `Content-Length`
  güvenli biçimde reddedilir ve timeout sonrasında concurrency slot'u bırakılır.
- RFC 9728 protected-resource metadata ve uyumlu `WWW-Authenticate`
  challenge desteği eklendi.
- Dış HTTP erişimi için izinli hedef, redirect, DNS/IP, boyut ve zaman aşımı
  denetimleri içeren merkezi outbound katmanı eklendi.
- Normal loglarda araç girdi ve cevap metni kaldırıldı; korelasyon ve privacy-
  safe metadata varsayılan hâle getirildi. İçerik önizlemesi yalnız açık yerel
  debug tercihiyle kullanılabilir.
- Operator işlemleri privacy-safe, idempotent ve kalıcı PostgreSQL job
  kayıtlarına bağlandı; iptal, yeniden başlatma ve eşzamanlı corpus yazarı
  koordinasyonu eklendi.
- Migration, corpus mutation, release publication ve retention yollarına
  ayrı advisory-lock sözleşmeleri eklendi.
- Paket/image/model supply-chain kanıtı için secret tarama, SBOM, zafiyet
  politikası ve digest denetimi üreten CI yardımcıları eklendi.

### Eklendi — mevzuat modeli, arama ve atıf

- Canonical mevzuat pilot modeli eklendi: düzenleyici belge/instrument,
  hukuki sürüm, kaynak blob/artifact, kanıt, hukuki olay, durum iddiası,
  provision ve sürüm-provision ilişkileri ayrı varlıklar olarak tutulabilir.
- Belge indirme/extraction geçmişinden farklı, kanıta dayalı hukuki sürümleme
  modeli oluşturuldu; extraction snapshot'ı hukuki yürürlük kanıtı olarak
  yorumlanmaz.
- Doğrulanmış publication/effective-date/status kanıtı yeterli değilse cevap
  uydurmak yerine abstain eden `resolve_regulation_status` aracı ve en dar
  yetkili SECURITY DEFINER resolver eklendi.
- `Citation v1` eklendi. Normalized metindeki exact aralık, excerpt, artifact
  ve metin hashleri, canonical hukuki kimlik ve corpus release bağlarıyla
  yeniden kurulabilir atıf üretilebilir.
- Retained source byte, acquisition kaydı, sayfa metni/mapping ve Citation
  excerpt zincirini yeniden hashleyerek doğrulayan legal-release evidence
  kontrolleri eklendi.
- Exact kanun/madde referanslarını koruyan arama, belge alias çözümleme,
  section-aware/token-aware chunking ve hybrid sparse/semantic retrieval
  sözleşmeleri sertleştirildi.
- Retrieval profile; embedding modeli/revision, chunking ve arama ayarlarını
  hashlenebilir bir kimlikte topladı. Active release ile profile uyuşmazlığı
  fail-closed hâle getirildi.
- Belge kalite uyarıları ve structured retrieval cevapları genişletildi;
  model yönlendirmesi kanıt yokluğunu açıkça belirtmeye odaklandı.

### Eklendi — corpus yayınlama ve değişmez generation'lar

- İmzalı ve kapsamı tanımlı corpus manifesti; artifact üyeliği, hash,
  freshness politikası ve retrieval profile doğrulaması eklendi.
- Corpus'un 17 izlenen tablosu için deterministik state fingerprint ve mutation
  epoch oluşturuldu. İzlenen herhangi bir değişiklik aktif release görünümünü
  geçersiz kılar.
- Append-only corpus release ve activation kayıtları eklendi; başarısız veya
  yarım kalan yayın aktif kimliği değiştirmez.
- Şema v8 ile imza/corpus üyeliği doğrulaması `bddk_release_verifier`,
  aktivasyon ise `bddk_release_publisher` rolüne ayrıldı. Verifier manifest,
  doğrulanmış aynı imza byte'larının hash'i, signer, ölçülmüş freshness,
  retrieval profile, exact corpus state/epoch, verifier revision/image ve
  60–3.600 saniyelik TTL'yi append-only request'e bağlar.
- Publisher yalnız opaque, tek kullanımlık request ID alır; corpus, imza veya
  trust key görmeden expiry, reuse, readiness, epoch ve state yeniden
  doğrulandıktan sonra aktivasyon yapabilir. Eski direct-publication yetkisi
  v8'de bütün non-owner rollerden geri alınır.
- Trust key'in verilen veya resolve edilen yolu corpus root içinde kalıyorsa
  doğrulama reddedilir. İmza ikinci kez açılmaz; stage kanıtı Ed25519 ile
  doğrulanan exact byte dizisini kullanır. Uzun chunk/embedding işlemlerinden
  sonra freshness, stage transaction commit etmeden hemen önce tekrar sınanır.
- Verifier çıktısı request ve evidence hashleriyle birlikte content-free
  `verification_run_sha256` değerini de verir. Canonical receipt ancak reviewed
  manifest, exact detached-signature SHA, retrieval-profile SHA, verifier
  revision/image provenance'i, staged request kanıtı ve verification-run değeri
  birlikte saklanırsa yeniden hesaplanabilir; path-free CLI özeti tek başına
  tam audit export'u değildir.
- Şema v7 ile aktif release'in exact 17 tablo durumunu ayrı typed retained
  ilişkilere atomik olarak kopyalayan, envanterleyen ve mühürleyen immutable
  generation katmanı eklendi.
- Aynı state/profile'a ait farklı yönetişim release'leri için fiziksel veriyi
  çoğaltmadan ayrı release-binding oluşturulması sağlandı.
- Retained generation, release, seal ve activation kimlikleri birbirinden
  ayrıldı; v7 öncesi kanıtsız release'ler `legacy_v5_unretained` olarak açıkça
  etiketlendi.
- `bddk-mcp retain-corpus-generation` yalnız publisher kimliğiyle çalışan,
  MCP'ye açılmayan ve serving/activation durumunu değiştirmeyen yönetim komutu
  olarak eklendi.

### Eklendi — değerlendirme ve güven zinciri

- Phase 2 değerlendirmesi resmî MCP oturumuna ve oturum boyunca değişmeyen
  active corpus release/manifest kimliğine bağlandı.
- Tool-calling, answer-grounding, Citation, abstention ve corpus uyumluluğu için
  fail-closed benchmark kontrolleri genişletildi.
- Corpus, expert dataset, legal-curator attestation ve legal-release checkpoint
  olmak üzere dört ayrı imzalı kanıt katmanı tanımlandı.
- Beş release kimliğini, dört ayrı signer-owner kimliğini, rotation/revocation
  kurallarını, deployment scope'u ve zaman sınırlı reviewer-owner yetkisini
  bağlayan imzalı evaluation trust policy şema v2 eklendi.
- Legal checkpoint predecessor zinciri, exact signer fingerprint karşılaştırma,
  sealed evidence'in tek kez parse edilmesi ve retained page/excerpt kanıtı
  kontrolleri eklendi.
- Evaluation release preflight eklendi. Banka yetkilendirmesi ve insan
  incelemesi repository dışında kaldığında sonuç açıkça yetkisiz olarak işaretlenir.
- Eksik release kanıtıyla üretilen benchmark raporlarının “release sonucu”
  gibi görünmesini önlemek için `exploratory` etiketi eklendi.

### Eklendi — OpenShift, operasyon ve kurtarma

- Banka on-premises OpenShift hedefi için non-root public/operator workload,
  ayrı ServiceAccount/Secret, digest-only image, probe, resource,
  NetworkPolicy, service, Route ve service-CA başlangıç manifestleri eklendi.
- Migration, bootstrap, verify/stage ve request-ID activation dört ayrı Job'a
  ve veritabanı kimliğine ayrıldı. Activator'a corpus PVC veya trust Secret
  bağlanmaz. Bankaya özgü IdP, CA, registry, Route ve CNI değerleri
  örnek manifestlerde bilinçli olarak doldurulmadı.
- OpenShift kabul denetleyicisi; exact image/Secret/PVC/env/command envanteri,
  TLS/JWT/egress ve görev ayrımı kontrolleriyle eklendi.
- Availability, latency, source detection, publication freshness, corpus yaşı,
  RPO, RTO ve evidence retention ölçülerini içeren sürümlü operational-
  objectives sözleşmesi eklendi. Hedefler onaylanmadığında production
  eligibility fail-closed kalır.
- Backup, restore ve upgrade için guard token, disposable-target koruması,
  PostgreSQL araç zaman aşımı, cleanup ve source/restore fingerprint
  karşılaştırması içeren recovery workflow eklendi.
- Şema v7 için iki ayrı disposable PostgreSQL 17 cluster üzerinde gerçek
  `pg_dump`/`pg_restore` çalıştırıldı ve sansürlenmiş makine-okunur kanıt
  repository'de tutuldu.
- Güncel şema v8 için ikinci bir iki-cluster PostgreSQL 17.10 tatbikatı
  çalıştırıldı: kaynak ve restore edilen hedefte 53 yönetilen nesne, yedi LOGIN
  profili, iki staged request/binding, iki retained generation, aynı logical
  fingerprint ve aynı active release doğrulandı.

### Değiştirildi

- Paket sürümü `5.0.0`'dan `5.0.1`'e, desteklenen Python aralığı
  `>=3.12,<3.14` olarak güncellendi.
- Repository paket hiyerarşisi ve runtime bootstrap akışı tek canonical
  `bddk_mcp` paketi altında toplandı.
- Wheel/sdist üretimi, paket dışından kurulum testi ve package-data envanteri
  düzeltildi.
- Ana README Türkçe, `README.en.md` İngilizce kullanım belgesi olarak
  ayrıştırıldı; araç sayısı ve public/operator sınırı canonical registry ile
  eşleştirildi.
- Serving, migration, ingestion, publication, retention ve recovery adımları
  birbirinden ayrılmış lifecycle operasyonlarına dönüştürüldü.
- Corpus yazma yolları ortak bulk-write/transaction katmanında batch edilerek
  N+1 ve yarım yayın riskleri azaltıldı.
- Retrieval cevaplarının bir bölümü typed structured output ve ortak güvenli
  hata sözleşmesine geçirildi; henüz bütün 29 araca yayılmış değildir.
- Test ve dokümantasyon yüzeyi; architecture, target architecture, security,
  testing/evaluation, gap register, roadmap, deployment, recovery ve corpus
  governance belgeleriyle genişletildi.

### Düzeltildi

- Dokümante edilmiş stdio başlatıcısının tools olmadan ayağa kalkabilmesi
  giderildi; runtime tool registry bootstrap'a bağlandı.
- PostgreSQL pool bulunmadığında `document_health` içinde oluşan
  `UnboundLocalError` giderildi ve resmî MCP istemci regresyon testi eklendi.
- Strict local-corpus çağrısında release'in çağrı sırasında değişmesi veya
  corpus mutation sonrası eski release'in aktif görünmesi engellendi.
- Evaluation signer kimliklerinde canonical fingerprint karşılaştırması ve
  checkpoint predecessor doğrulaması düzeltildi.
- Legal evidence byte'larının birden çok kez farklı parse edilmesi ve excerpt
  kanıtının retained page ile bağ kurmaması giderildi.
- Benchmark ulaşım/kanıt hatalarının normal model cevabı gibi puanlanması
  fail-closed hâle getirildi.
- Migration/checksum/catalog readiness ile legacy şema adoption sırası
  sertleştirildi.
- Corpus-controlled trust anchor, doğrulama/hash arasındaki imza TOCTOU yarışı,
  stage öncesi freshness aşımı, slow-body ilk-byte bypass'ı ve aşırı uzun
  `Content-Length` dönüşümü adversarial inceleme sonrasında kapatıldı.

### PostgreSQL migration özeti

| Şema | Migration adı | Sağladığı sınır |
|---|---|---|
| `v0001` | `core_document_retrieval_schema` | Belge, section, sürüm, chunk, cache ve temel retrieval şeması |
| `v0002` | `durable_operator_jobs` | Privacy-safe, kalıcı operator job ledger |
| `v0003` | `retrieval_publication_integrity` | Retrieval publication bütünlüğü ve kontrollü legacy adoption |
| `v0004` | `canonical_legal_version_pilot` | Canonical instrument, hukuki sürüm, kanıt, olay, provision ve Citation modeli |
| `v0005` | `verified_corpus_release_publication` | Append-only release/activation, corpus epoch ve exact active-release görünümü |
| `v0006` | `validated_legal_status_resolver` | En dar yetkili, kanıt yetersizse abstain eden hukuki durum resolver'ı |
| `v0007` | `retained_corpus_generations` | 17 corpus ilişkisinin typed, immutable ve sealed generation kopyası |
| `v0008` | `staged_corpus_releases` | Ayrı verifier request'i, TTL/state/epoch bağı ve request-ID-only tek kullanımlık aktivasyon |
| `v0009` | `regulatory_relation_edges` | Kanıt ve inceleme kaydı taşıyan typed çapraz referans kenarları ve yalnızca doğrulanmış satırları sunan üç güvenlik-bariyerli görünüm |
| `v0010` | `corpus_release_freshness_policy` | Kapalı iki değerli freshness politika seti (`quantified_measured/unmeasured_signature_verified_pass`) ve politika-farkındalıklı staging imzası |

### Şema v8 — tamamlanan repository sınırı (tarihsel not; güncel ledger v10'da biter)

`v0008_staged_corpus_releases.py`, uygulama komutları, exact rol/ACL
sözleşmeleri, OpenShift ayrımı ve recovery envanteri commitlendi. Migration;
append-only `corpus_release_requests` ile request/activation binding tablosunu,
SECURITY DEFINER stage/activate facade'larını, immutable trigger/constraint
setini ve legacy grant temizliğini ekler. Negatif rol, çift-role, replay,
expiry, mutation, yanlış profile, eşzamanlı aktivasyon, rogue grant, katalog
tampering, gerçek LOGIN, OpenShift mount/Secret ve iki-cluster restore testleri
geçti.

Bu “repository mekanizması tamamlandı” ifadesidir; bankanın iki credential'a
aynı principal'ın erişemediğini, gerçek Secret/RBAC/rotation/audit ayrımını veya
üretim corpus'unun yayınlanabilirliğini kanıtlamaz.

### Doğrulama kanıtı

16 Temmuz 2026 tarihli güncel şema-v8 çalışma ağacında kaydedilen sonuçlar:

- PostgreSQL gerektirmeyen tam test hattı: 1.440 başarılı, 37 atlandı,
  192 seçim dışı; hata yok.
- Zorunlu PostgreSQL 17 tam test hattı: 185 başarılı, 4 ortam-kabiliyeti testi
  atlandı, 1.480 seçim dışı; hata yok.
- Migration/katalog odaklı PostgreSQL hattı: 59 başarılı.
- Uygulama/rol sözleşmesi hattı: 118 başarılı; ayrıca gerçek LOGIN/ACL hattı:
  2 başarılı.
- HTTP güvenlik hattı: ilk-byte/chunk/toplam gövde süreleri dâhil 61 başarılı.
- OpenShift manifest ve kabul hattı: dört ayrı lifecycle Job'u dâhil
  75 başarılı.
- Dokümantasyon sözleşmesi hattı: 13 başarılı.
- Ruff lint ve format kontrolleri, `uv lock --check`, `git diff --check` ve
  ayrı geçici dizine wheel/sdist build'i başarılı.
- İki ayrı disposable PostgreSQL 17.10 cluster üzerinde güncel şema-v8
  `pg_dump`/`pg_restore` tatbikatı: 53 managed object, yedi LOGIN profili, iki
  staged request/binding, iki retained generation, aynı active release ve aynı
  logical fingerprint; workflow sonucu `passed`.

Güncel kalıcı recovery kanıtı:
`docs/evidence/LOCAL_PG17_V8_RECOVERY_DRILL.md` (arşiv: `archive/pre-simplify-2026-11-09`)
ve
`docs/evidence/local-pg17-v8-restore-2026-07-16.json` (arşiv: `archive/pre-simplify-2026-11-09`).

Önceki şema-v7 kontrol noktası ayrıca korunur: PostgreSQL gerektirmeyen hatta
1.411 başarılı/34 atlandı, PostgreSQL 17 hattında 177 başarılı/4 atlandı ve
iki-cluster kurtarma tatbikatında 51 managed object ile altı LOGIN profili
doğrulandı. Tarihsel kanıt:
`docs/evidence/LOCAL_PG17_V7_RECOVERY_DRILL.md` (arşiv: `archive/pre-simplify-2026-11-09`)
ve
`docs/evidence/local-pg17-v7-restore-2026-07-16.json` (arşiv: `archive/pre-simplify-2026-11-09`).

Bu yerel sonuçlar banka üretim ortamı, gerçek corpus hukuki onayı, PITR,
hedef-hacim RPO/RTO veya OpenShift kabulü yerine geçmez.

### Bilinen sınırlar ve release engelleri

- Tracked manifest 318 belge ve 8.286 chunk bildirirken güncel pinned profile
  read-only regeneration'da 9.675 chunk üretir. Bu nedenle tracked corpus
  yeniden üretilip bağımsız incelenmeden strict publication geçmez.
- Şema v7 ile eklenen katman generation'ları saklar fakat public serving hâlâ mutable current
  tablolara bağlıdır. Generation-bound serving ve yetkili reactivation/rollback
  henüz yoktur.
- Repository'de gerçek ve yetkili bir mevzuat ailesi için tamamlanmış retained
  source/page/reviewer kanıt paketi yoktur. Mevcut canonical legal model pilot
  ve fixture kanıtıyla sınırlıdır.
- Sayfa, tablo ve formül korunumu bütün corpus için audit-grade doğrulanmış
  değildir; teknik Citation kabiliyeti kaynak otoritesini tek başına kanıtlamaz.
- Expert evaluation veri seti hâlâ draft'tır; bağımsız uzman onayı ve canlı
  model sonuçlarının release yetkilendirmesi yoktur.
- Banka IdP/CA/Route/CNI/egress, gerçek LOGIN/HBA/TLS, registry, signed image,
  backup custody, PITR, hedef veri boyutu, RPO/RTO ve OpenShift kabul kanıtı
  repository'de yoktur.
- Rate limiting ve bazı metrikler process-local'dır; multi-replica/global kota
  ve kabul edilmiş Prometheus/OpenTelemetry hattı tamamlanmamıştır.
- `MIT` lisansı ticari kullanım, değiştirme ve yeniden dağıtım izni verir.
  Dağıtılmış açık kaynak kopyalarda yetkisiz kurumsal kullanımı teknik olarak
  engellemez; farklı gelecek lisansları için hukuki değerlendirme gerekir.
- Bu sürümleme noktasında 29 aracın tamamı aynı typed result envelope'a
  geçirilmemiştir.

## 5.0.0 sonrası geliştirme dönemi — 9 Nisan–20 Haziran 2026

Bu dönem `5.0.0` paket metadata'sını koruyan çok sayıda commit içerir, ancak
ayrı patch/minor sürümleri veya Git etiketleriyle yayımlanmamıştır.

### Eklendi

- Runtime'da her açılışta BDDK scraping yapmak yerine PostgreSQL'den serving ve
  sıfır-request deployment için seed export/import akışı eklendi.
- PostgreSQL pgvector schema, seed migration ve semantic embedding backfill
  yolları eklendi; ilk kurulum/migration sırası için düzeltmeler yapıldı.
- Monolitik server; dependency injection, ayrı araç modülleri, ingest, store,
  OCR, kalite ve gözlemlenebilirlik paketlerine bölündü.
- Streamable HTTP ve stdio yaşam döngüsü, Railway shutdown tanıları, Docker,
  Compose ve Hugging Face Spaces başlangıç desteği geliştirildi.
- MCP tool-calling, NLI, BDDK terminolojisi ve canlı uçtan uca model
  değerlendirmesi için benchmark paketi eklendi. İlk veri setleri 30 tool-
  calling vakası, 30 NLI çifti ve 50 terminoloji sorusu içeriyordu.
- BDDK belge health/quality araçları, kalite tarayıcıları, bilinen hata
  etiketleri ve düzeltme/backfill komutları eklendi.
- Chandra2 OCR, CPU için zengin HTML-first extraction, PDF/DOC magic-byte ve
  HTML hata-sayfası doğrulaması, encoding corruption taraması ve formula-loss
  envanteri eklendi.
- Madde, geçici madde, ek ve alt bölüm ayrıştırma; section storage/reindex,
  exact section retrieval ve section search MCP araçları eklendi.
- Section-aware ve token-aware chunking, exact legal-reference koruma, hybrid
  arama score tanıları, gold-set vakaları ve audit-grade benchmark puanlama
  eklendi.
- Bounded document page retrieval, loose section fallback ve isteğe bağlı
  retrieval telemetry eklendi.
- MIT lisansı 21 Nisan 2026'da repository'ye eklendi.

### Değiştirildi

- Desteklenen Python tabanı `>=3.12,<3.14` olarak daraltıldı.
- Runtime canlı fetch davranışı sınırlandırılarak local corpus “airlock” yaklaşımı
  güçlendirildi.
- Corpus kapsamı geleneksel bankacılık başlıklarıyla sınırlandırıldı ve README
  Türkçe/İngilizce olarak yenilendi.
- Proje dosyaları `bddk_mcp` Python paket hiyerarşisine taşındı; kök config ve
  dokümantasyon yüzeyi sadeleştirildi.
- Hybrid arama sıralaması, exact phrase ağırlığı, semantic gap ve lexical tie-
  breaker davranışları düzenlendi.
- CI lint/container smoke testleri ve least-privilege GitHub token ayarları
  eklendi; bağımlılık güvenlik güncellemeleri uygulandı.

### Düzeltildi

- Seed import sırasında eksik `document_chunks` tablosu, migration sırası,
  singleton retry ve pgvector fallback sorunları giderildi.
- Seed UPSERT/drift detection ve eski chunk temizliği düzeltildi; import sonrası
  embedding backfill eklenerek boş semantic search engellendi.
- 51 encoding-bozuk mevzuat belgesi yeniden senkronize edildi; U+FFFD, NUL,
  HTML whitespace ve duplicate-cell extraction sorunları düzeltildi.
- Mevzuat indirmede belge türü adayları, GeneratePdf yolu, iframe/PDF/DOC
  fallback'leri ve büyük/encoded HTML hata sayfası tespiti düzeltildi.
- Bazı mevzuat ekleri ve düşmüş formüller official kaynaklardan elle
  birleştirildi/transcribe edildi; atomik patch yardımcıları eklendi.
- Search N+1 sorgusu, shutdown race, stdio background task yaşam döngüsü ve
  transport environment parsing sorunları giderildi.
- Numeric belge kimliklerinin `mevzuat_`/`bddk_` alias'larına çözülmesi ve integer
  section reference kabulü düzeltildi.
- Section indexing blind spot'ları ve section relevance görünürlüğü düzeltildi.
- Orta seviye code-review ve dependency CVE bulguları ele alındı.

## [5.0.0] — 9 Nisan 2026 — etiketlenmemiş paket sürümü

### Değiştirildi

- Paket sürümü ilk kez gerçekten `5.0.0` yapıldı.
- SQLite/ChromaDB ağırlıklı önceki tasarım PostgreSQL ve pgvector merkezli bir
  mimariye refactor edildi.
- Vector store büyük ölçüde yeniden yazıldı; full-text ve semantic sonuçları
  birleştiren retrieval yaklaşımı sadeleştirildi.
- Client, document store, sync, server, config ve exception katmanları
  sadeleştirilip test edilebilirlikleri artırıldı.
- Bağımlılık seti ve lockfile önemli ölçüde küçültüldü.
- Docker Compose ile yerel PostgreSQL geliştirme ortamı eklendi.

### Eklendi

- Arama kalitesini ölçmek için F1 evaluation testleri ve genişletilmiş test
  fixture'ları eklendi.

### Kaldırıldı

- Ayrı `sync_all.py` script'i kaldırıldı; senkronizasyon akışı uygulama
  lifecycle'ına taşındı.

## “v6” özellik dönemi — 6 Nisan 2026 — release değildir

Bu ad yalnız `95b74cc` commit mesajından gelir; o committe paket sürümü
`4.0.0` olarak kalmıştır. Repository'de `v6.0.0` etiketi yoktur.

### Eklendi

- ChromaDB tabanlı vector store ve `multilingual-e5-base` embedding modeli
  eklendi.
- 1.093 belge ve 9.377 chunk için semantic index oluşturulduğu commit
  açıklamasında kaydedildi.
- Belge getirme sırası ChromaDB → SQLite → live fetch fallback zinciri olarak
  değiştirildi.
- Railway'de SQLite sync sonrasında ChromaDB'nin otomatik oluşturulması eklendi.

### Bilinen tarihsel sınır

- Commit açıklamasındaki 1–4 ms ID retrieval ve 25–95 ms semantic search
  değerleri tarihsel geliştirici ölçümleridir; yeniden üretilebilir benchmark
  raporu veya güncel performans taahhüdü değildir.

## “v5” özellik dönemi — 6 Nisan 2026 — release değildir

Bu ad yalnız `69d0707` commit mesajından gelir; o committe paket sürümü
`4.0.0` olarak kalmıştır. Repository'de bu özelliğe karşılık gelen `v5.0.0`
etiketi yoktur.

### Eklendi

- SQLite + FTS5 tabanlı offline-first local document store eklendi.
- Mevzuat.gov.tr için HTML → PDF → iframe → legacy DOC şeklinde dört katmanlı
  indirme fallback'i eklendi.
- Store-first belge getirme ve live fetch sonucunu otomatik cache etme eklendi.
- `sync_bddk_documents`, `search_document_store` ve `document_store_stats`
  MCP araçları eklendi.
- Railway persistent volume ve ilk deployment'ta arka planda auto-sync desteği
  eklendi.

### Tarihsel veri notu

- Commit açıklaması o anda 1.093 belgenin, yaklaşık 117 MB veriyle ve sıfır
  hata ile senkronize edildiğini bildirir. Bu sayı güncel corpus kapsamı veya
  hukuki eksiksizlik garantisi değildir.

## [4.0.0] — 5 Nisan 2026 — etiketlenmemiş paket dönemi

### Eklendi

- Haftalık bülten trendlerini ve haftadan haftaya değişimleri analiz eden
  `analyze_bulletin_trends` aracı eklendi.
- Kararlar ve duyuruları bir araya getiren `get_regulatory_digest` eklendi.
- Metrikleri yan yana karşılaştıran `compare_bulletin_metrics` eklendi.
- Yeni duyuruları izleyen `check_bddk_updates` eklendi.
- 17 tabloluk aylık bankacılık istatistiklerini getiren `get_bddk_monthly`
  eklendi.

### Düzeltildi

- Aylık bültenin doğru `BasitRaporGetir` endpoint'ini kullanması sağlandı.
- Client analytics katmanı için cache erişimi açıldı.

### Test

- Commit açıklaması 56 mevcut ve 12 yeni analytics testi olmak üzere toplam
  68 testin geçtiğini bildirir.

### Yayın durumu

- `pyproject.toml` bu committe `4.0.0` oldu; `v4.0.0` Git etiketi yoktur.

## [3.0.0] — 5 Nisan 2026 — etiketlenmemiş paket dönemi

### Eklendi

- Banka, finansal kiralama, faktoring, finansman ve varlık yönetimi dâhil 340'tan
  fazla kuruluşu kapsayan `search_bddk_institutions` eklendi.
- CSRF token akışını ele alan haftalık sektör bülteni aracı
  `get_bddk_bulletin` eklendi.
- 22 bankacılık metriğinin en güncel görünümünü sunan
  `get_bddk_bulletin_snapshot` eklendi.
- Basın açıklaması ve düzenleme duyurusu arayan
  `search_bddk_announcements` eklendi.
- Bankalar için accordion, diğer kuruluşlar için tab-pane kullanan iki farklı
  HTML yapısının ayrıştırılması eklendi.

### Test

- Commit açıklaması 14 yeni data-source testiyle toplam 56 test bulunduğunu
  bildirir.

### Yayın durumu

- `pyproject.toml` bu committe `3.0.0` oldu; `v3.0.0` Git etiketi yoktur.

## [2.0.0] — 5 Nisan 2026

Git etiketi: `v2.0.0`.

### Eklendi

- BDDK mevzuat kapsamı üç sayfadan dokuz sayfaya genişletildi; commit açıklaması
  yaklaşık 1.094 belge bildirir.
- Kanun, Banka Kartları, Finansal Kiralama ve Faktoring, BDDK Düzenlemesi,
  Düzenleme Taslağı ve Mülga Düzenleme kategorileri eklendi.
- Mevzuat.gov.tr için kanun, tüzük, kararname ve diğer belge türleri
  genişletildi.
- `date_from`/`date_to` tarih aralığı filtresi eklendi.
- Başlık > stem > substring ağırlıklı relevance sıralaması eklendi.
- Yaklaşık 50 ek kalıbı kullanan temel Türkçe morphology/stemming desteği
  eklendi.
- Restart sonrasında yaşayan disk cache (`.cache.json`) ve
  `bddk_cache_status` tanı aracı eklendi.

### Değiştirildi

- HTTP çağrılarına üç denemeli exponential backoff eklendi.
- Belge fetch/PDF conversion hataları ve sayfa hataları daha açıklayıcı ve
  kontrollü hâle getirildi.

### Test

- Pytest/pytest-asyncio kurulumu ve helper/search mantığını kapsayan 42 unit test
  eklendi.

## [1.0.0] — 5 Nisan 2026

Git etiketi: `v1.0.0`.
Etiket altındaki paket metadata'sı tarihsel olarak `0.1.0` değerini taşır.

### Eklendi

- BDDK Kurul Kararları ile yönetmelik, genelge, tebliğ ve rehberleri arayıp
  getiren ilk MCP server eklendi.
- Bilgi Sistemleri, Sermaye Yeterliliği ve Tekdüzen Hesap Planı başlıkları için
  kategori filtresi ve Türkçe-aware anahtar kelime araması eklendi.
- BDDK ve mevzuat.gov.tr kaynaklarından PDF belge getirme eklendi.
- Bir saat TTL'li in-memory cache eklendi.

## Git kanıtları ve karşılaştırma bağlantıları

- [`v1.0.0` etiketi](https://github.com/omercagatay/bddk-mcp/tree/v1.0.0)
  — commit `526abee`.
- [`v2.0.0` etiketi](https://github.com/omercagatay/bddk-mcp/tree/v2.0.0)
  — commit `c64eed4`.
- [`3.0.0` paket dönemi](https://github.com/omercagatay/bddk-mcp/commit/717b1da099dbd1f507878e8fa825a51a27b29a3d)
  — etiketlenmemiş commit `717b1da`.
- [`4.0.0` paket dönemi](https://github.com/omercagatay/bddk-mcp/commit/347715e31583df3be58ae4dba69bfd7bcef66a6c)
  — etiketlenmemiş commit `347715e`.
- [“v5” SQLite özellik commit'i](https://github.com/omercagatay/bddk-mcp/commit/69d0707e98c1b478b51fdc71be1446753452170d)
  — paket sürümü bu noktada `4.0.0`.
- [“v6” ChromaDB özellik commit'i](https://github.com/omercagatay/bddk-mcp/commit/95b74cc22bd28cc5c9894f5412858effd19d1f3e)
  — paket sürümü bu noktada `4.0.0`.
- [Gerçek `5.0.0` paket refactor'u](https://github.com/omercagatay/bddk-mcp/commit/1364318abbeeaeb9da1eb639c4a8cfb7e1d3aa33)
  — etiketlenmemiş commit `1364318`.
- [`5.0.1` dalının başlangıç commit'i](https://github.com/omercagatay/bddk-mcp/commit/316325f4b68e0602a9585ee7b8c503434e0545ff)
  — MCP runtime/bootstrap stabilizasyonu.
- [`5.0.1` şema-v7 doğrulama noktası](https://github.com/omercagatay/bddk-mcp/commit/461d4e48f4baf3afe534e2a80200d33ce13ff811)
  — iki-cluster recovery kanıtı.
- `5.0.1` şema-v8 kod doğrulama noktası — bu çalışma dalındaki commit
  `7cd6242`; staged verifier/publisher sınırı ve adversarial düzeltmeler.
- `5.0.1` şema-v8 operasyon/recovery belge noktası — bu çalışma dalındaki
  commit `83d31a4`; güncel iki-cluster kanıtı ve işletim belgeleri.

[Yayınlanmadı]: https://github.com/omercagatay/bddk-mcp/compare/v2.0.0...HEAD
[5.0.0]: https://github.com/omercagatay/bddk-mcp/commit/1364318abbeeaeb9da1eb639c4a8cfb7e1d3aa33
[4.0.0]: https://github.com/omercagatay/bddk-mcp/commit/347715e31583df3be58ae4dba69bfd7bcef66a6c
[3.0.0]: https://github.com/omercagatay/bddk-mcp/commit/717b1da099dbd1f507878e8fa825a51a27b29a3d
[2.0.0]: https://github.com/omercagatay/bddk-mcp/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/omercagatay/bddk-mcp/tree/v1.0.0
