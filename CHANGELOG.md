# Değişiklik Günlüğü

Bu dosya BDDK MCP projesindeki önemli değişiklikleri Türkçe olarak kaydeder.
Biçim, [Keep a Changelog](https://keepachangelog.com/tr-TR/1.1.0/) yaklaşımına
yakındır; ancak repository'nin tarihsel sürümleme düzensizliklerini gizlemez.
Sürüm numaraları [Anlamsal Sürümleme](https://semver.org/lang/tr/) ilkeleriyle
yorumlanmalıdır.

## Sürümleme hakkında önemli not

Uygulama/paket sürümü ile PostgreSQL şema sürümü birbirinden bağımsızdır:

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
- `v0001`–`v0010` adları Python paket sürümü değil, ileri yönlü PostgreSQL
  migration sürümleridir. Bir uygulama release'i birden fazla migration
  içerebilir.

## [Yayınlanmadı] — hedef paket sürümü 5.0.1

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

### Şema v8 — tamamlanan repository sınırı

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
[`docs/evidence/LOCAL_PG17_V8_RECOVERY_DRILL.md`](docs/evidence/LOCAL_PG17_V8_RECOVERY_DRILL.md)
ve
[`docs/evidence/local-pg17-v8-restore-2026-07-16.json`](docs/evidence/local-pg17-v8-restore-2026-07-16.json).

Önceki şema-v7 kontrol noktası ayrıca korunur: PostgreSQL gerektirmeyen hatta
1.411 başarılı/34 atlandı, PostgreSQL 17 hattında 177 başarılı/4 atlandı ve
iki-cluster kurtarma tatbikatında 51 managed object ile altı LOGIN profili
doğrulandı. Tarihsel kanıt:
[`docs/evidence/LOCAL_PG17_V7_RECOVERY_DRILL.md`](docs/evidence/LOCAL_PG17_V7_RECOVERY_DRILL.md)
ve
[`docs/evidence/local-pg17-v7-restore-2026-07-16.json`](docs/evidence/local-pg17-v7-restore-2026-07-16.json).

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
