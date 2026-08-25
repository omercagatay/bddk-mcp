# Banka Geçişi Uygulama Güvenliği Kontrol Listesi Değerlendirmesi

Banka BT güvenlik ekibinin geçiş öncesi ilettiği 150 maddelik "Ana Kontrol /
Kontrol Soruları" seti, bu depodaki güncel duruma karşı madde madde
değerlendirilmiştir.

- **Değerlendirme tarihi:** 2026-08-25
- **Değerlendirme temeli:** `main` / commit `99e13ec` (paket 5.0.1 hedefi,
  şema v10, imzalı corpus `bddk-job-corpus-2026-08-14`)
- **Yöntem:** Depodaki güvenlik dokümanları ([güvenlik incelemesi](SECURITY_REVIEW.md),
  [gap register](GAP_REGISTER.md), [dağıtım rehberi](DEPLOYMENT.md),
  [tedarik zinciri politikası](../supply-chain/README.md)) esas alınmış; kritik
  iddialar kod üzerinde ayrıca doğrulanmıştır (dış URL taraması, `verify=False` /
  `subprocess` / `eval` / f-string SQL taramaları, Git yazarlık trailer'ları,
  CI/policy dosyaları, admin konsolu bağlama kuralları).
- **Sınır:** Bu doküman depo sınırındaki kanıtı raporlar. `SECURITY.md`'deki
  ilkeyle uyumlu olarak, depo kontrolleri tek başına banka/üretim kabulü
  oluşturmaz; "🏦" işaretli maddeler banka tarafında tamamlanmadan geçiş
  onayı verilmemelidir.

## Durum göstergeleri

| İşaret | Anlamı |
|---|---|
| ✅ | Depo sınırında karşılanıyor; kanıt dosyaları belirtildi |
| 🟡 | Kısmen karşılanıyor; mekanizma var, sınır/artık risk açık |
| ❌ | Karşılanmıyor; geçiş öncesi aksiyon gerekli |
| 🏦 | Banka tarafı kontrol; depo değil, dağıtım/süreç sahibi banka tamamlar |
| ➖ | Bu uygulama için uygulanamaz (gerekçesi satırda) |

## Yönetici özeti

Depo, bu soru setinin sorduğu kontrollerin büyük bölümünü **bilinçli ve
kanıtlı** biçimde uygulamıştır: fail-closed uzak yapılandırma, yedi ayrık
PostgreSQL kimliği, imzalı ve iki-rollü (verifier/publisher) corpus yayın
protokolü, exact-host dış erişim allowlist'i, tam-geçmiş secret taraması,
SBOM + zafiyet taraması, dijest-pinli tedarik zinciri, non-root/read-only
konteyner ve ~1.485 otomatik testlik bir kanıt tabanı. Sık rastlanan
uygulama açıkları (hard-coded credential, SQL injection, doğrulanmamış TLS,
kontrolsüz egress, debug modu) için depo düzeyinde olumsuz bulgu yoktur.

Geçiş kararını etkileyecek dört ana bulgu:

1. **Public MCP ucunda istek-başına kimlik doğrulama yok (tasarım kararı).**
   Banka mimarisinde Keycloak katmanı kaldırılmış; ~25 kullanıcı Open WebUI'ye
   AD/LDAP ile girer, önyüz MCP public profiline banka ağı içinde
   `BDDK_HTTP_ALLOW_UNAUTHENTICATED=true` ile bearer'sız bağlanır. Erişim
   denetimi ağ izolasyonuna (Route kısıtları, exact Host/Origin, NetworkPolicy)
   devredilmiştir. Sonuç: MCP katmanı son kullanıcıyı tanımaz; iz kayıtları
   kullanıcıya bağlanamaz (madde 64, 71–72, 88, 129). Bu mimari sapmanın banka
   risk onayı ve Open WebUI logları ile korelasyon tasarımı şarttır.
2. **Dört-göz ilkesi depo tarafında yok.** Tek maintainer modeli: PR + korumalı
   `main` + zorunlu CI var, ancak bağımsız ikinci onaycı yoktur (madde 32–33,
   108, 114). Banka Git/change sürecinde bağımsız onay zorunlu kılınmalıdır.
3. **Bağımsız sızma testi ve DAST yapılmamıştır** (madde 101, 136–142).
   Odaklı otomatik güvenlik testleri (61 HTTP güvenlik testi, rol deny-matrisi,
   FTS sanitizasyon, girdi sözleşmeleri) kısmi telafidir; banka pentest'i
   geçiş şartı olmalıdır.
4. **Banka tarafı kabul kanıtları açık.** Yedek/PITR ve restore tatbikatının
   banka ortamında tekrarı, imaj imzalama/registry admission, 60 işletim
   sistemi CVE istisnasının banka onayı, SIEM entegrasyonu, sayısal SLA/RPO/RTO
   hedefleri (gap register: CUR-002, CUR-008…CUR-010, CUR-016, CUR-017).

Madde bazlı durum dağılımı (150 satır): **✅ 67 · 🟡 48 · ❌ 12 · 🏦 17 · ➖ 6**.

---

## 1. Yönetişim, Sahiplik ve Envanter (1–7)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 1 | Veri envanteri | 🟡 | İşlenen veri: kamuya açık BDDK/mevzuat metinleri (318 belge, 9.675 chunk; makine-okur imzalı kapsam: `seed_data/corpus_scope.yml`), kullanıcı sorguları (gizli kabul edilir — `docs/DEPLOYMENT.md` "Secrets and Logs") ve opsiyonel, hash'lenmiş telemetri. Müşteri/kişisel veri işlenmez. Aksiyon: bu envanterin banka veri-envanteri şablonuna resmî aktarımı. |
| 2 | Bilgi güvenliği sınıfı (C-I-A ayrı ayrı) | 🟡 | C-I-A ihtiyaçları `docs/SECURITY_REVIEW.md` "Assets and security objectives" tablosunda ayrıştırılmış (gizlilik: sorgular/telemetri; bütünlük: corpus/citation; erişilebilirlik: `docs/decisions/operational-objectives.v1.yml`). Resmî sınıflandırma etiketi banka şablonunda atanmalı. |
| 3 | İş sahibi + teknik sahip | 🏦 | Depo tek maintainer'lı (`.github/CODEOWNERS`). Bankada iş sahibi ve teknik uygulama sahibi resmen atanmalı. |
| 4 | Desteklenen iş süreçleri ve amaç dokümantasyonu | ✅ | Amaç ve kullanım sınırları dokümante: `README.md`, `docs/EXECUTIVE_SUMMARY.md`, `docs/STATUS.md` (iç denetim/uyum için mevzuat arama-analiz; hukuki mütalaa olmadığı açıkça yazılı). |
| 5 | Üretime alma resmî karar/onayı | 🏦 | Depo bunu bilinçli olarak açık bırakır: "bank acceptance open" (`docs/STATUS.md`). Bu soru setinin sonucu üzerinden banka onay süreci işletilmeli. |
| 6 | Mimari/veri akışı/entegrasyon/bağımlılık diyagramları | ✅ | `docs/ARCHITECTURE.md`, `docs/TARGET_ARCHITECTURE.md`, `docs/REPOSITORY_STRUCTURE.md`; güven sınırı diyagramı `docs/SECURITY_REVIEW.md` içinde. Güncel sözleşme `docs/STATUS.md`'de sürüm/commit bağlı tutuluyor. |
| 7 | Canlıya geçiş öncesi BS risk analizi | 🟡 | Tehdit, etki, önem derecesi ve aksiyonlar `docs/SECURITY_REVIEW.md` + `docs/GAP_REGISTER.md`'de (High bulgular, efor, bağımlılıklar). Banka BS risk metodolojisi formatında yeniden onaylanmalı; bu doküman girdi olarak kullanılabilir. |

## 2. Risk Değerlendirmesi (8–12)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 8 | AI/Claude ile geliştirme riskleri ayrıca değerlendirildi mi | 🟡 | AI'a özgü riskler bu değerlendirmenin 13–20. maddelerinde kanıtlarıyla ele alındı; `docs/SECURITY_REVIEW.md` prompt-injection ve untrusted-document sınırlarını ayrıca inceler. Banka risk kaydına ayrı kalem olarak aktarılmalı. |
| 9 | Claude'a aktarılan bilgi/kod kapsamı | ✅ | Depoda yalnız kamuya açık mevzuat verisi ve açık kaynak kod vardır; sır, erişim anahtarı, üretim/müşteri verisi yoktur (tam-geçmiş Gitleaks taraması; 8 secret istisnasının tümü false-positive/sentetik — `supply-chain/README.md`). `.env` git dışı (`.gitignore`). Banka içi bilgiyle çalışma başlarsa banka AI kullanım politikası uygulanmalı. |
| 10 | Claude kodu insan incelemesiz üretime çıkamaz | ✅ | Tüm değişiklikler PR + korumalı `main` + zorunlu CI'dan geçer (`docs/STATUS.md`, `CONTRIBUTING.md`); depodan üretime otomatik dağıtım (CD) yoktur — dağıtım bankada elle/kontrollü yürütülür. Nüans: incelemeyi tek maintainer yapar (bkz. 13, 33). |
| 11 | Kabul edilen riskler için yönetim onayı | 🟡 | İstisna mekanizması açık kayıtlı: `supply-chain/policy.json` her istisnada gerekçe + sorumlu + son tarih + `pending_bank_release_review` durumu taşır ve banka onayı olmadan `release_promotion_eligible=false` kalır. Banka yönetim onayı fiilen alınmalı (CUR-002/016). |
| 12 | Claude üretimi kod bölümleri izlenebilir mi | ✅ | Git geçmişinde `Co-Authored-By: Claude …` trailer'ları commit bazında mevcut (ör. `99e13ec`, `7392d1e`, `644f542`); her değişiklik PR numarasına bağlı. AI katkısı commit/PR granülaritesinde izlenebilir. |

## 3. Claude / AI Destekli Kod Geliştirmeye Özgü Kontroller (13–20)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 13 | AI kodun tamamı teknik incelemeden geçti mi | 🟡 | Her PR maintainer incelemesi + zorunlu CI (lint, iki Python sürümü, PostgreSQL, rol sözleşmesi, paket, konteyner, supply-chain) ile birleşir. Bağımsız ikinci insan gözü yoktur; banka geçişinde ikinci gözden geçiren zorunlu kılınmalı (bkz. 33). |
| 14 | Önerilen paket/kütüphanelerin gerçekliği ve güvenilirliği | ✅ | Tüm bağımlılıklar `uv.lock` içinde hash'li olarak PyPI'dan çözülür; CI `--frozen` kurar; var olmayan/uydurma paket derlemeyi kırar. Haftalık Dependabot güncellemesi (`.github/dependabot.yml`). |
| 15 | Bilinmeyen dış servis/URL/IP/telemetry bağlantısı | ✅ | Kod taramasıyla doğrulandı: çalışma zamanı dış erişim yalnız `bddk.org.tr` ve `mevzuat.gov.tr` (exact-host allowlist, `bddk_mcp/core/outbound_http.py:23-24`). Telemetri yalnız bankanın kendi PostgreSQL'ine, varsayılan kapalı. Phone-home/üçüncü taraf telemetry yok. Build aşamasında embedding modeli sabit commit'le Hugging Face'ten iner; banka için mirror/onay kaydı `supply-chain/model-assets.json` (`pending_bank_review`). |
| 16 | Banka dışına veri sızıntısının ağ seviyesinde testi | 🟡 | Uygulama katmanı egress'i sınırlar; OpenShift starter'ı tüm pod'lar için default-deny egress uygular (`deploy/openshift/networkpolicies.yaml`). Bankada fiilî ağ testi (izinli hedefler dışında bağlantı denemesi) henüz yapılmadı; kabul testi olarak eklenmeli. |
| 17 | Lisans/fikri mülkiyet riski | 🟡 | Kod MIT lisanslı; kaynak/veri/model kökeni `docs/LICENSING_AND_PROVENANCE.md`'de; üçüncü taraf envanteri SBOM'da. Bankanın kullanımına engel yok; ticari strateji/veri hakları açık kalemi (`docs/GAP_REGISTER.md` CUR-015) banka hukuk incelemesine iletilmeli. |
| 18 | Claude kullanımına kurum içi onay/kurallar | 🟡 | Depo düzeyinde kurallar mevcut: `CLAUDE.md` (çalışma kuralları, credential yasağı), `.claude/settings.json` (commit öncesi zorunlu ruff kontrolü hook'ları), `CONTRIBUTING.md`. Banka içi kabul edilebilir kullanım politikası banka tarafında çıkarılmalı. |
| 19 | Prompt'lara credential/sır aktarımının önlenmesi | 🟡 | Teknik taraf: depoda sır yok, `.env` git dışı, tam-geçmiş Gitleaks her PR'da çalışır; `CONTRIBUTING.md` gerçek DSN/token yapıştırmayı yasaklar. Prompt tarafı süreç kontrolüdür: banka politikası + kullanıcı bilgilendirmesi gerekli. |
| 20 | Kaynak kodda backdoor amaçlı gizli kullanıcı/sabit parola | ✅ | Kodda kullanıcı tablosu, sabit parola veya master-password mekanizması yoktur; kimlik tamamen dış IdP (operator JWT) ve PostgreSQL LOGIN'lerindedir. Gitleaks tam-geçmiş taraması + bu değerlendirme kapsamındaki desen taramaları olumsuz bulgu vermedi. |

## 4. Backdoor / Zararlı veya Yetkisiz Kod Kontrolleri (21–29)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 21 | Hard-coded credential/anahtar/connection string | ✅ | Gitleaks tam tarih taraması (her PR); 8 istisnanın tümü false-positive/sentetik fixture (`supply-chain/README.md`). Compose kimlikleri loopback'e bağlı, "disposable local fixture" etiketli (`docker-compose.yml:1-2`). `.env.example` yalnız placeholder içerir. |
| 22 | Görünmeyen gizli URL/endpoint/yönetici ekranı/debug fonksiyonu | ✅ | HTTP yüzeyi sabit ve sayılıdır: `/mcp`, `/health/live`, `/health/ready` (+ JWT modunda RFC 9728 well-known metadata; unauthenticated modda bu uçlar 404). Yönetim konsolu ayrı süreçtir, **loopback dışı bind'ı yapısal olarak reddeder** ve yalnız GET (okuma) rotaları içerir (`bddk_mcp/admin/config.py`, `bddk_mcp/admin/views/documents.py`). İçerikli debug logu explicit opt-in olup üretimde yasaktır. Öneri: admin konsolunu banka imaj/namespace'ine hiç dağıtmamak. |
| 23 | Auth/authz atlatan özel kullanıcı/header/parametre | ✅ | Bilinen bypass parametresi yoktur; Host/Origin/JWT doğrulaması fail-closed'dur ve tek istisna (`BDDK_HTTP_ALLOW_UNAUTHENTICATED`) gizli değil, startup'ta global ve dokümante bir yapılandırmadır (`docs/DEPLOYMENT.md`). 61 odaklı HTTP güvenlik testi negatif senaryoları içerir (`tests/test_http_security.py`). |
| 24 | Tarih/kullanıcı/IP/özel değerle kontrolleri devre dışı bırakan kod | ✅ | 2026-07 güvenlik incelemesi ve bu değerlendirme kapsamındaki taramalarda bulunamadı (`docs/SECURITY_REVIEW.md`: "No confirmed … backdoor"). Yokluk kanıtı sınırlıdır; bağımsız kod incelemesi/pentest ile teyit önerilir (madde 136). |
| 25 | Otomatik çalışan yetkisiz script/scheduled task/startup hook | ✅ | Serving süreci başlangıçta hiçbir şema/corpus/senkronizasyon yazımı yapmaz; `BDDK_AUTO_SYNC=true` serving'de reddedilir (`.env.example`, `docs/DEPLOYMENT.md`). Zamanlanmış görev yoktur; yaşam döngüsü işleri ayrı, elle tetiklenen Job'lardır. Depodaki `.claude/settings.json` hook'ları yalnız geliştirme ortamı araçlarıdır, çalışma zamanına dahil değildir. |
| 26 | Dış internet erişimi iş ihtiyacıyla sınırlı / egress filtering | ✅ | Uygulama katmanı: exact HTTPS host allowlist (yalnız BDDK/mevzuat), 443 dışı port ve credential'lı URL reddi, public-DNS doğrulaması, ≤3 redirect'te yeniden doğrulama, akış boyut sınırları (`bddk_mcp/core/outbound_http.py`). Dağıtım katmanı: default-deny egress NetworkPolicy + dokümante dar allow matrisi (`deploy/openshift/networkpolicies.yaml`, `docs/DEPLOYMENT.md`). Banka değerleriyle fiilî uygulanması 16. maddeye bağlı. |
| 27 | Deploy edilen artifact'ın onaylı kaynaktan üretildiğinin doğrulanması | 🟡 | Supply-chain hattı her build'de imaj manifest/config dijestlerini, SBOM'u ve Dockerfile SHA-256 bağlarını üretir; yeniden üretilebilir wheel/sdist byte-denk doğrulanır (`.github/workflows/supply-chain.yml`, `supply-chain/README.md`). Kanıt **imzasızdır**: imaj imzalama + admission bankanın promotion kontrolüdür (CUR-016). |
| 28 | Production artifact üzerinde zararlı kod/malware taraması | 🟡 | Grype zafiyet taraması + Gitleaks secret taraması vardır; klasik antivirüs/malware taraması yoktur. Banka registry'sinin imaj tarama/admission politikası devreye alınmalı. |
| 29 | Kaynak kod merkezî kurumsal repository'de mi | ✅ | GitHub `omercagatay/bddk-mcp`, korumalı `main`, tek uzun ömürlü dal politikası (`docs/STATUS.md`). Geçişte banka içi Git'e aynalama ve orada dört-göz zorunluluğu önerilir (bkz. 33). |

## 5. Kaynak Kod ve Yazılım Tedarik Zinciri Güvenliği (30–40)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 30 | Repo erişimi rol bazlı / asgari yetki | 🟡 | Tek maintainer + CODEOWNERS; dış katkı PR zorunlu. Banka Git'inde rol bazlı erişim matrisi kurulmalı. |
| 31 | Main'e doğrudan commit engeli | ✅ | Korumalı `main`: zorunlu lint/test/PostgreSQL/paket/konteyner/supply-chain kontrolleri (`docs/STATUS.md` "Protected main with required checks"); geçmişteki tüm değişiklikler PR squash merge'leridir (git log `(#N)` referansları). |
| 32 | PR + bağımsız code review zorunlu | 🟡 | PR ve şablonu zorunludur (`.github/PULL_REQUEST_TEMPLATE.md` doğrulama komutları + etki beyanı ister). "Bağımsız" inceleme tek-maintainer modelde yoktur → bankada zorunlu ikinci onaycı. |
| 33 | Geliştiricinin kendi değişikliğini tek başına üretime taşıma engeli | ❌ | Depo gerçeği: maintainer kendi PR'ını merge edebilir. Telafi: zorunlu CI + üretime otomatik dağıtım olmaması. Aksiyon: banka Git/change sürecinde dört-göz (bağımsız onay) teknik olarak zorunlu kılınmalı; release tarafında verifier/publisher Secret'ları farklı kişilere verilmeli (CUR-017). |
| 34 | Commit/PR/merge/build/deploy'un tekil kullanıcıya loglanması | ✅ | Git commit kimlikleri + PR numaraları + GitHub Actions run kayıtları kişi bazında izlenebilir; Claude katkısı trailer'larla ayrışır (madde 12). Banka dağıtım adımları banka araçlarında loglanmalı. |
| 35 | Açık kaynak envanteri / SBOM | ✅ | Her PR/push/tag'de CycloneDX SBOM (Syft) üretilir ve kanıt manifestiyle arşivlenir; `uv.lock` tam sürüm+hash envanteridir (`.github/workflows/supply-chain.yml`). |
| 36 | Kullanılmayan/eski bağımlılıkların kaldırılması | 🟡 | Bağımlılık grupları ayrık (dev/benchmark/gpu üretim imajına girmez; imaj `--no-dev` — `Dockerfile`); Dependabot güncel tutar. Periyodik "kullanılmayan bağımlılık" temizlik turu tanımlı değil; yıllık gözden geçirme önerilir. |
| 37 | Paketlerin güvenilir kaynaklardan alınması | ✅ | PyPI + `uv.lock` hash'leri; taban imajlar ve `uv` dijest-pinli; GitHub Actions commit-pinli; tarayıcı araçları SHA-256 doğrulamalı indirilir (`supply-chain/tools.json`); embedding modeli sabit Git commit'i. |
| 38 | Dependency confusion/typosquatting değerlendirmesi | 🟡 | Hash-pinning + tek kayıtlı indeks + Dependabot bu riskleri büyük ölçüde kapatır; ayrıca yazılı değerlendirme yoktur. Bankada iç mirror/proxy (ör. Artifactory) üzerinden kurulum önerilir. |
| 39 | Bağımlılık bütünlüğü: hash/lock/imza | ✅ | `uv.lock` hash doğrulaması + CI/`.mcp.json`/imajda `--frozen`; corpus artefaktları Ed25519 imzalı; imajlar dijest-pinli. |
| 40 | (39 ile aynı soru — listede mükerrer) | ✅ | 39'daki kanıt geçerlidir. |

## 6. OWASP Top 10 — Uygulama Güvenliği Testleri (41–71)

### Broken Access Control (41–44)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 41 | URL/ID değiştirerek başka kullanıcının kaydını görme | ➖ | Kullanıcıya özel kayıt/tenant verisi yoktur; tüm belge içeriği kamuya açık mevzuattır ve ID'ler zaten herkese aynı içeriği döner. Özel/tenant verisi eklenirse yeniden değerlendirilmeli (`docs/GAP_REGISTER.md` TEN-001 bunu açıkça yasaklar). |
| 42 | Yetkisiz kullanıcının yetkili fonksiyonlara doğrudan erişimi | ✅ | Operator araçları public süreçte hiç kayıtlı değildir (ayrı registry/process/DSN); HTTP'de operator profili `bddk.operator` scope'lu JWT gerektirir ve kimliksiz uzak operator yapısal olarak reddedilir (`bddk_mcp/tools/registry.py`, `docs/DEPLOYMENT.md`, `tests/test_mcp_http_runtime.py`). |
| 43 | IDOR/BOLA testleri | 🟡 | Otomatik sözleşme/negatif testler mevcut; bağımsız IDOR/BOLA test raporu yok. Banka pentest kapsamına alınmalı (madde 136–141). |
| 44 | Yönetici fonksiyonlarının standart kullanıcıdan çağrılamaması | ✅ | 42 ile aynı mekanizma; public `tools/list` operator araçlarını hiç göstermez, testlerle sabitlenmiştir. Bankada canlı ortam doğrulaması kabul testine eklenmeli. |

### Security Misconfiguration (45–48)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 45 | Debug/development modu üretimde kapalı | ✅ | Debug modu yoktur; içerikli araç logu (`BDDK_TOOL_LOG_CONTENT`) varsayılan kapalıdır ve üretimde yasaklanmıştır (`docs/DEPLOYMENT.md`); imaj dev bağımlılıklarını içermez (`Dockerfile` `--no-dev`). |
| 46 | Kullanılmayan servis/port/fonksiyon kapalı | ✅ | Konteynerde tek süreç, tek port (8000); sabit route seti; yerel PostgreSQL portu loopback'e bağlı (`docker-compose.yml`). |
| 47 | Hata mesajları teknoloji/dizin/SQL detayı sızdırmıyor | ✅ | Hatalar kararlı, gizlilik-korumalı kodlara eşlenir; exception mesajları, DSN'ler ve yollar loglara/iş kayıtlarına yazılmaz (`docs/SECURITY_REVIEW.md` overlay; `tests/test_internal_log_privacy.py`; operator job kayıtları ham hata metni saklamaz). |
| 48 | Güvenlik header'ları | 🟡 | Host/Origin doğrulaması, 401/403/408/413/429 kontratları ve boyut/deadline sınırları var; klasik tarayıcı header'ları (HSTS, `X-Content-Type-Options`, CSP) uygulama katmanında set edilmiyor — JSON API için etki sınırlı, admin konsolu loopback+GET-only. Öneri: header'ları OpenShift Route/ingress'te eklemek. |

### Software Supply Chain Failures (49–52)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 49 | Tüm bağımlılıklar biliniyor ve taranıyor | ✅ | SBOM + Grype her PR/push/tag'de; tarayıcı bastırması (ignore/VEX) politika gereği yasak ve tespit edilirse fail (`supply-chain/README.md`). |
| 50 | Kritik/yüksek CVE içeren dependency | 🟡 | Uygulama (Python) bağımlılıklarında açık High/Critical yoktur; mevcut 60 istisna kaydının tamamı (30 benzersiz CVE/paket × 2 imaj) Debian "wont-fix" veya yalnız CPython 3.15'te düzeltilen **taban imaj OS paketleridir**; tümü gerekçeli, 2026-10-15 son tarihli ve `pending_bank_release_review` durumundadır (`supply-chain/policy.json`). Aksiyon: bu listenin banka zafiyet yönetimi onayından geçirilmesi. |
| 51 | CI/CD değişiklik yetkileri kısıtlı | 🟡 | Workflow'lar `permissions: contents: read`; actions commit-pinli; CODEOWNERS tanımlı. Workflow/policy/release-tag'lerin ruleset korumasının repository ayarında doğrulanması ayrı governance kalemi olarak dokümante edilmiştir (`supply-chain/README.md`). |
| 52 | Build pipeline bütünlüğü | ✅ | Byte-denk yeniden üretilebilir wheel/sdist; dijest-pinli BuildKit; kanıt dosyaları SHA-256 manifestiyle bağlanır ve release job'ı aynı run'ın kanıtını yeniden hash'ler; tag'in `main` atası olması zorunlu (`.github/workflows/supply-chain.yml`). |

### Cryptographic Failures (53–56)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 53 | Aktarımda güncel TLS | ✅ | PostgreSQL DSN'lerinde `sslmode=verify-full` + onaylı CA zorunlu, fail-closed (`bddk_mcp/db_transport.py`, `docs/DEPLOYMENT.md`); dış istekler yalnız HTTPS; uygulama soketinde opsiyonel TLS + OpenShift re-encrypt Route (service CA). |
| 54 | Saklamada şifreleme | 🏦 | Uygulama verisi kamuya açık mevzuat korpusudur; gizlilik ihtiyacı sorgu/telemetri katmanındadır (varsayılan: hiç saklanmaz). Disk/DB-at-rest şifreleme banka PostgreSQL/storage standardıyla sağlanmalı. |
| 55 | Şifreleme anahtarları kaynak kodda mı | ✅ | Hayır. Depoda yalnız Ed25519 **public** doğrulama anahtarı vardır (`deploy/trust/corpus-signing-public-key.pem`); imza özel anahtarı ve tüm credential'lar depo dışında/platform Secret'ında tutulur. |
| 56 | Algoritmalar güncel ve güvenilir | ✅ | JWT'de yalnız asimetrik algoritmalar (varsayılan RS256, simetrik red); corpus imzası Ed25519; bütünlük SHA-256. |

### Injection (57–61)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 57 | Command/OS injection | ✅ | Kullanıcı girdisinin ulaştığı hiçbir shell/subprocess yolu yoktur; sınırlı `subprocess` kullanımı sabit argv + timeout'ludur (OCR `pdftotext`, kurtarma aracı, Kustomize preflight) ve `shell=True` hiç kullanılmaz (kod taramasıyla doğrulandı). |
| 58 | LDAP/NoSQL/Template injection | ➖ | LDAP yalnız Open WebUI/AD tarafındadır (banka kapsamı); NoSQL yoktur; Jinja2 yalnız loopback admin konsolunda, autoescape'li ve GET-only kullanımdadır. |
| 59 | SQL injection / girdilerin sorguya eklenmesi | ✅ | Tüm sorgular asyncpg pozisyonel parametrelidir; MCP'de serbest SQL aracı yoktur; FTS girdisi sanitize edilip `plainto_tsquery`/websearch yoluna gider (`docs/SECURITY_REVIEW.md` "SQL construction"; `tests/test_fts_sanitization.py`). F-string SQL taraması yalnız iç sabitleri (timeout/kilit değerleri, sabit predicate listeleri) gösterdi. |
| 60 | Parametreli sorgu/prepared statement | ✅ | 59 ile aynı kanıt. |
| 61 | Dosya adı/arama/filtre/URL parametre manipülasyon testleri | 🟡 | Girdi sözleşme testleri + FTS sanitizasyon + hostile girdi senaryoları depo testlerinde; bağımsız manipülasyon testi banka pentest kapsamına (madde 136–141). |

### Insecure Design (62–63)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 62 | Kötüye kullanım senaryoları / threat modeling | ✅ | Güven sınırı modeli, varlık/hedef tablosu, tehdit-önle/tespit/kurtar matrisi ve kabul kapıları `docs/SECURITY_REVIEW.md`'dedir; kalan riskler `docs/GAP_REGISTER.md`'de kayıtlıdır. |
| 63 | Beklenen iş akışını atlayarak işlem engeli | ✅ | Örnek mekanizmalar: iki-rollü staged release (verifier stage'ler, publisher yalnız tek-kullanımlık request-ID ile aktive eder; süresi/epoch'u/state'i değişmiş istek reddedilir), corpus mutasyonunda epoch geçersizleştirme, fail-closed startup zinciri (`bddk_mcp/corpus_publication.py`, migration v0008). |

### Authentication Failures (64–69)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 64 | Bankanın merkezî kimlik altyapısı kullanılıyor mu | 🟡 | **Ana mimari bulgu.** Kullanıcılar Open WebUI'ye Microsoft AD/LDAP ile girer (banka merkezî kimliği ✓). Ancak MCP public ucu banka ağı içinde `BDDK_HTTP_ALLOW_UNAUTHENTICATED=true` ile bearer'sız çalışır; erişim denetimi ağ izolasyonuna devredilmiştir (`CHANGELOG.md` "Keycloak'ın kaldırılması", `docs/DEPLOYMENT.md`). Operator profili her istekte JWT zorunludur ve kimliksiz uzak operator yapısal olarak reddedilir. Aksiyon: bu sapmanın banka risk onayı; public Route'un yalnız onaylı segment/ön yüzden erişilebildiğinin kanıtlanması. |
| 65 | MFA gereksinimi değerlendirmesi | 🏦 | AD/Open WebUI oturumu banka politikasına tabidir; MCP katmanında kullanıcı oturumu yoktur. |
| 66 | Brute-force'a karşı kilitleme/rate limiting | 🟡 | MCP: istemci başına 120 istek/dk + 32 eşzamanlılık + gövde/deadline sınırları (süreç-lokal; `docs/DEPLOYMENT.md`). Login brute-force yüzeyi AD/Open WebUI tarafındadır; paylaşımlı ingress limiti banka gateway'inde tanımlanmalı. |
| 67 | Oturum zaman aşımı | ➖ | MCP HTTP'si stateless'tır (kalıcı oturum/cookie yok); operator JWT'lerinde `exp` doğrulanır. Kullanıcı oturum politikası Open WebUI/AD tarafında (banka). |
| 68 | Logout sonrası session/token geçersizliği | 🏦 | Ön yüz oturumu ve token yaşam döngüsü IdP/Open WebUI'dedir. MCP token revocation listesi tutmaz; operator erişiminde kısa ömürlü token kullanımı önerilir. |
| 69 | Session fixation/hijacking testleri | ➖ | MCP'de oturum çerezi/sunucu oturumu yoktur; bu test sınıfı Open WebUI için banka pentest'inde koşulmalı. |

### Mishandling of Exceptional Conditions (70–71)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 70 | Fail-open yerine fail-secure/fail-closed | ✅ | Sistematik tasarım ilkesi: eksik uzak güvenlik ayarında startup reddi, `verify-full` sağlanamayan DB bağlantısının reddi, katalog/kimlik doğrulaması geçmeyen readiness'ın 503 dönmesi, imzasız/sürüklenmiş corpus'un yayınlanamaması (`docs/DEPLOYMENT.md`, `bddk_mcp/db_identity.py`, `bddk_mcp/catalog_integrity.py`). |
| 71 | Her kullanıcının benzersiz hesapla erişimi | ❌ | Kullanıcılar Open WebUI'de benzersiz AD hesabıyla oturum açar, ancak MCP katmanına kullanıcı kimliği taşınmaz: tüm çağrılar tek servis bağlamında gelir ve MCP izleri kullanıcıya bağlanamaz (bkz. 64, 88). Aksiyon: Open WebUI erişim/istek loglarının MCP korelasyon kimlikleriyle eşlenmesi tasarlanmalı veya ileride istek-başına kimlik yeniden değerlendirilmeli. |

## 7. Kimlik ve Erişim Yönetimi (72–79)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 72 | Ortak/generic kullanıcı hesaplarının engellenmesi | 🟡 | İnsan hesapları AD'de bireyseldir (banka); servis hesapları işlev-başına ayrıktır (yedi DB kimliği). Ancak MCP public trafiği fiilen tek servis kimliği altında akar → 71'deki izlenebilirlik bulgusu burada da geçerli. |
| 73 | Yetkilerin rol/profil üzerinden verilmesi | ✅ | Yedi NOLOGIN grup rolü + LOGIN'lerin tam-eşleşme üyelik sözleşmesi; startup gerçek oturum kimliğini, üyelikleri ve etkin ayrıcalıkları doğrular, fazlasını reddeder (`deploy/postgres/01_roles.sql`, `02_grants.sql`, `bddk_mcp/db_identity.py`). |
| 74 | Talep/onay/uygulama görev ayrılığı | 🏦 | İnsan-yetki süreçleri banka IAM'indedir. Depo, release tarafında örnek görev ayrılığını teknik olarak kurar: verifier ile publisher rolleri birbirinin üyeliğini alamaz (madde 63). |
| 75 | Ayrıcalıklı/admin kullanıcıların ayrıca belirlenmesi | ✅ | Ayrıcalıklı düzlemler açık: schema-owner (yalnız migrate), ingestion, verifier, publisher, operator; her biri ayrı DSN/Secret/ServiceAccount ister (`docs/DEPLOYMENT.md` kimlik tablosu). Banka LOGIN'lerinin bu matrise göre açılması gerekir. |
| 76 | Admin işlemlerinin ayrıntılı loglanması ve gözden geçirilmesi | 🟡 | Operator işlemleri kalıcı PostgreSQL job kayıtları üretir (durum, parmak izi, ilerleme, güvenli hata kodu; ham içerik saklanmaz — `bddk_mcp/jobs/`). DB düzeyinde DDL/grant denetimi (ör. pgaudit) ve düzenli gözden geçirme banka DBA sürecinde kurulmalı. |
| 77 | Servis hesaplarının son kullanıcıca kullanılamaması | ✅ | DSN'ler yalnız platform Secret'larında yaşar; kullanıcı akışı hiçbir DB kimliğine dokunmaz. Yanlış/fazla yetkili LOGIN ile açılan her bağlantı sözleşme kontrolünde reddedilir (`bddk_mcp/db_identity.py`). |
| 78 | Başarısız girişte hesap kilitleme | 🏦 | AD/Open WebUI politikası. MCP tarafında karşılığı: 401 + rate limit (madde 66). |
| 79 | Tüm veri alanları ve sınıfları belirlenmiş mi | 🟡 | Şema, migration'larda sürümlü ve katalog-attestasyonlu tanımlıdır; veri sınıfları madde 1–2'deki çerçevededir. Resmî alan-bazlı sınıflandırma banka şablonuna aktarılmalı. |

## 8. Veri Güvenliği ve Gizlilik (80–87)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 80 | Hassas veri erişiminin yetkiyle sınırlanması | ✅ | Korpus kamuya açıktır; hassas olan kullanıcı sorgularıdır: loglar varsayılan metadata-only'dir, telemetri varsayılan kapalıdır ve açılırsa yalnız INSERT yetkili ayrı kimlikle yazar; sorgu metni ancak açık opt-in ile saklanır (`.env.example`, `bddk_mcp/observability/telemetry.py`). |
| 81 | Aktarımda şifreleme | ✅ | Madde 53 ile aynı kanıt. |
| 82 | Saklamada şifreleme | 🏦 | Madde 54 ile aynı: banka DB/storage standardı. |
| 83 | Üretim verisinin dev/test'te kullanılmaması | ✅ | Müşteri/üretim verisi yoktur; korpus zaten kamuya açık kaynaklardan derlenir; testler sentetik/fixture veri kullanır (`tests/fixtures`, sentetik legal aile). |
| 84 | Hassas verinin loglara açık yazılmaması | ✅ | Varsayılan log alanları yalnız metadata + korelasyon kimliğidir; sorgu/yanıt önizlemesi ancak `BDDK_TOOL_LOG_CONTENT=true` ile mümkündür ve üretimde yasaktır; DSN/token asla loglanmaz; redaksiyon testlidir (`bddk_mcp/core/logging_config.py`, `tests/test_internal_log_privacy.py`). |
| 85 | Hassas verinin local storage/temp/cache'te tutulması | 🟡 | Sunucu tarafında kontrolsüz alan yoktur (cache'ler DB/süreç içi). Tarayıcı tarafı (Open WebUI) ve MCP yanıtlarının LLM istemci bağlamına gitmesi banka istemci politikasında değerlendirilmeli. |
| 86 | Export/download için yetkilendirme ve loglama | 🟡 | Toplu export aracı yoktur; tam belge çağrısı yanıt başına 5 sayfa ile sınırlıdır ve araç çağrıları loglanır. Kullanıcı-bazlı yetki/iz mümkün değildir (madde 71'e bağlı). |
| 87 | Dosya yükleme kontrolleri | ➖ | Kullanıcı dosya yüklemesi yoktur. Sistemin kendi indirmeleri (ingestion) tip/boyut/arşiv-üye/oran sınırlarına ve onaylı-host koşuluna tabidir (`bddk_mcp/ingest/doc_sync.py`, `docs/SECURITY_REVIEW.md` "Path traversal" bölümü). |

## 9. İz Kayıtları ve Denetlenebilirlik (88–94)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 88 | Log alanları: tarih, saat/zaman dilimi, olay, kullanıcı, değişiklik | 🟡 | Yapılandırılmış JSON: timestamp, seviye, logger, mesaj, correlation_id, operation, süre, araç adı/durumu, boyut/sayı alanları (`bddk_mcp/core/logging_config.py`). Eksikler: kullanıcı kimliği alanı public profilde yoktur (madde 71) ve zaman damgası açıkça UTC işaretli değildir → dağıtımda UTC/timezone-explicit format ve toplayıcı normalizasyonu yapılandırılmalı. |
| 89 | Login/logout ve başarısız girişlerin kaydı | 🏦 | Kullanıcı girişleri AD/Open WebUI'de loglanır (banka). MCP operator tarafında 401/403 kararları loglanır. |
| 90 | Yetki verme/değiştirme/kaldırma kaydı | 🟡 | Yetkiler sürümlü SQL dosyalarıyla (git geçmişi) değişir ve çalışma zamanı ACL kaynağı doğrulanır; DB'de canlı grant değişikliklerinin denetimi (pgaudit vb.) banka DBA'inde kurulmalı. |
| 91 | Hassas kayıt görüntüleme/indirme kaydı | 🟡 | Araç çağrıları (belge kimlikleriyle, telemetri açıksa) izlenebilir; kullanıcı-bazlı görüntüleme izi yoktur (madde 71). İçerik kamuya açık olduğundan artık risk sınırlıdır; sorgu gizliliği esas korunandır. |
| 92 | Logların kullanıcı/yöneticilerce değiştirilememesi | 🟡 | Loglar stdout üzerinden platform log sistemine akar (uygulama içinde silme/değiştirme yüzeyi yok); telemetri kimliği yalnız kolon-kapsamlı INSERT yetkilidir, okuma/güncelleme/silme reddedilir (`.env.example`). Merkezî log deposunun değiştirilemezliği banka platform kontrolüdür. |
| 93 | Log bütünlüğünün teknik korunması | 🏦 | Banka log platformu (WORM/imza/retention) sağlar; depo tarafında ön koşul olan yapılandırılmış, içeriksiz log formatı hazırdır. |
| 94 | Logların merkezî SIEM'e aktarımı ve senaryolarla izlenmesi | ❌ | SIEM entegrasyonu/forwarding tanımlı değildir. Aksiyon: OpenShift log forwarding → banka SIEM; önerilen alarm başlıkları depoda hazır listelidir (auth reddi, throttle, corpus yaşı, kapsam/kalite, sync hatası, ayrıcalık reddi — `docs/SECURITY_REVIEW.md` "Observability security"). |

## 10. Sistem Geliştirme Yaşam Döngüsü — SDLC (95–106)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 95 | Gereksinimlerde güvenlik/loglama/gizlilik/performans tanımı | ✅ | Kabul değişmezleri `docs/TARGET_ARCHITECTURE.md`'de; ölçüm/kanıt/sahiplik sözleşmesi `docs/decisions/operational-objectives.v1.yml`'de (sekiz metrik; sayısal hedefler bilinçli olarak banka onayına bırakılmış — bkz. 124). |
| 96 | Dev, test/UAT ve production ortam ayrımı | ✅ | Yerel geliştirme (loopback Compose, atılabilir kimlikler) ile banka hedefi (OpenShift overlay'leri, Secret'lar) fiziksel ve yapılandırma olarak ayrıktır; dev kimliklerinin uzakta kullanımı yasak ve etiketlidir. Banka kendi dev/UAT/prod namespace ayrımını kurmalı. |
| 97 | Geliştiricinin üretime sürekli erişiminin olmaması | 🏦 | Üretim ortamı ve erişim modeli bankadadır; depo hiçbir üretim credential'ı içermez. |
| 98 | Zorunlu üretim erişiminin süreli/onaylı/loglu olması | 🏦 | Banka PAM/erişim süreci. Depodan destek: ayrıcalıklı işler kısa ömürlü Job'lar ve süreli staged-request (60–3.600 sn TTL) olarak tasarlanmıştır. |
| 99 | Kaynak kod incelemesi | 🟡 | PR-bazlı maintainer incelemesi + dış güvenlik incelemesi kaydı (2026-07, `docs/SECURITY_REVIEW.md`); bağımsız ikinci gözle sürekli inceleme yok (madde 32–33). |
| 100 | SAST | 🟡 | Ruff (E,F,W,I,UP,B) + kapsamlı güvenlik odaklı birim/integrasyon testleri koşuyor; adanmış SAST aracı (ör. Bandit/Semgrep/CodeQL) yok. Aksiyon: CI'a SAST adımı eklemek veya banka SAST hattından geçirmek. |
| 101 | DAST | ❌ | Dinamik tarama yapılmamıştır. Kısmi telafi: canlı sunucuya karşı koşan 61 HTTP güvenlik testi ve resmî MCP istemci E2E'leri. Aksiyon: banka DAST/pentest (madde 136). |
| 102 | SCA/dependency taraması | ✅ | Grype (SBOM üzerinden, güncel DB şartı ≤72 saat, bastırma yasak) her PR/push/tag'de + haftalık Dependabot (`supply-chain/`, `.github/dependabot.yml`). |
| 103 | Secret scanning | ✅ | Gitleaks, **tam Git geçmişi** üzerinde, her çalıştırmada; istisnalar tekil parmak izi bazlı ve gerekçeli (`supply-chain/README.md`). |
| 104 | Critical/High bulguların üretim öncesi giderilmesi | ✅ | Mekanizma fail-closed: `release-eligibility` job'ı istisnasız her High/Critical bulguda ve onay bekleyen istisna varlığında başarısız olur; `release_promotion_eligible` hiçbir zaman depo kararıyla true olmaz. Mevcut açık kalemler madde 50'deki OS istisnalarıdır. |
| 105 | İstisna bırakılan açıklar için risk kabulü + telafi edici kontrol | ✅ | Her istisna: hedef+CVE+paket+Dockerfile-SHA eşleşmesi, gerekçe, sorumlu, son kullanma tarihi, `pending_bank_release_review`; süresi geçen/kullanılmayan/geniş istisna fail (`supply-chain/policy.json`, `supply-chain/README.md`). Banka onay adımı fiilen işletilmeli (madde 11). |
| 106 | Her üretim değişikliğinin kayıtlı change talebi | 🟡 | Depo tarafında her değişiklik PR kaydına bağlıdır (etki beyanı zorunlu). Banka üretimindeki dağıtım/konfig değişiklikleri banka change yönetimine bağlanmalı. |

## 11. Değişiklik ve Release Yönetimi (107–115)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 107 | Risk ve etki analizi | 🟡 | PR şablonu her değişiklikte veri/şema/dağıtım/güvenlik etki beyanını zorunlu kılar (`.github/PULL_REQUEST_TEMPLATE.md`); banka değişiklik risk analizi ayrıca yapılmalı. |
| 108 | Değişikliğin bağımsız kişilerce onayı | ❌ | Madde 33 ile aynı bulgu: tek maintainer. Banka change board/ikinci onaycı zorunlu. |
| 109 | Test ortamında test | ✅ | CI her PR'da tam matris koşar (iki Python sürümü, gerçek PostgreSQL 17 + pgvector servisi, rol allow/deny matrisi, paket kurulumu, konteyner/manifest statik doğrulaması). |
| 110 | UAT/kullanıcı kabulü | 🏦 | Departman kullanıcılarıyla banka UAT'si planlanmalı; depo istemci/model uyum matrisini açık iş olarak listeler (CUR-012). |
| 111 | Üretime aktarılan sürüm = test edilen sürüm (hash) | ✅ | Mekanizma: imaj dijest bağlama (manifest/config dijestleri kanıtta), OpenShift manifestlerinde dijest-pinli imaj zorunluluğu, tarama raporlarının Dockerfile SHA-256'ya bağlanması. Banka admission'ın aynı dijesti doğrulaması gerekir (CUR-016). |
| 112 | Deployment öncesi rollback planı | 🟡 | İmaj dijestiyle geri dönüş + append-only şema/migration disiplini + restore drill runbook'u var. Sınır: corpus generation'ları mühürlü saklanır ancak otomatik "önceki release'e dön" akışı yoktur (H2-02B açık, CUR-007) ve DB PITR bankadadır. Rollback prosedürü banka runbook'una yazılmalı. |
| 113 | Deployment sonrası smoke test | ✅ | İçeriksiz `/health/live` + bağımlılık/katalog/kimlik doğrulayan `/health/ready`; OpenShift kabul harness'i; `scripts/mcp_smoke.py`. Banka kabulünde canlı smoke adımı tanımlanmalı. |
| 114 | Release'te tek kişinin hazırla+onayla+taşı yasağı | ❌ | Depo tarafında tek kişi mümkündür (madde 33/108). Teknik altyapı görev ayrılığına hazırdır: verify/stage ve activate ayrı roller, ayrı Secret/ServiceAccount'lardır; banka bu Secret'ları **farklı kişilere** vermelidir (CUR-017). |
| 115 | Sunucu/container/platform güvenli baseline | ✅ | Non-root (UID 10001), read-only rootfs, restricted güvenlik bağlamı, resource sınırları, dijest-pinli taban imaj, arbitrary-UID uyumu; preflight sapmaları fail-closed yakalar (`Dockerfile`, `deploy/openshift/`, `bddk_mcp/openshift_acceptance.py`). |

## 12. Güvenlik Konfigürasyonu ve Sunucu Güvenliği (116–121)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 116 | Kullanılmayan port ve servisler kapalı | ✅ | Madde 46 ile aynı kanıt. |
| 117 | Üretimde development/debug aracı yok | ✅ | İmaj `--no-dev` kurulur; test/benchmark/deploy varlıkları wheel'e dahil edilmez (`docs/DEPLOYMENT.md`); debug modu yok (madde 45). |
| 118 | Asgari işletim sistemi/dosya sistemi yetkisi | ✅ | Non-root, grup-0 izin modeli, read-only kök dosya sistemi, yazılabilir alan yalnız geçici dizinler (`Dockerfile`, OpenShift security context). |
| 119 | Konfig değişikliklerinin change yönetimiyle yapılması | 🟡 | Tüm konfig git'te sürümlüdür (kustomize/manifest/env sözleşmeleri) → GitOps'a uygun. Bankadaki canlı konfig değişikliklerinin banka change sürecine bağlanması gerekir. |
| 120 | Dosya bütünlük kontrolü | 🟡 | Uygulama dosyaları: immutable dijest-pinli imaj + read-only FS (drift'e kapalı). Veritabanı nesneleri: readiness'ta canlı katalog attestasyonu (constraint/trigger/fonksiyon/indeks tanımları, checksum'lu migration defteri — `bddk_mcp/catalog_integrity.py`). Klasik host-FIM banka platform standardıdır. |
| 121 | Periyodik vulnerability scan kapsamı | 🟡 | Tarama her PR/push/tag'de + haftalık Dependabot tetiklemeleriyle koşar; **zamanlanmış (cron) bağımsız tarama yoktur**. Aksiyon: workflow'a schedule eklemek ve/veya banka registry'sinin periyodik imaj taramasına almak. |

## 13. Güvenlik Açığı ve Yama Yönetimi (122–125)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 122 | OS/framework/runtime/dependency yama takibi | 🟡 | Python bağımlılıkları + GitHub Actions haftalık Dependabot'ta; OS yamaları taban imaj rebuild'iyle gelir ancak **taban imaj dijest güncellemesi manueldir** (Dependabot docker ekosistemi tanımlı değil). Aksiyon: Dockerfile dijestleri için güncelleme rutini (Dependabot docker desteği veya aylık manuel bump). |
| 123 | Yamaların üretim öncesi testi | ✅ | Her bağımlılık güncellemesi PR olarak tam CI matrisinden geçer (git geçmişindeki dependabot PR'ları kanıttır). |
| 124 | Kritik açık giderme için risk bazlı SLA | 🟡 | İstisnalarda zorunlu son tarih mekanizması vardır (mevcutlar 2026-10-15); sayısal, onaylı SLA hedefleri bilinçli olarak açıktır (CUR-008). Aksiyon: banka zafiyet SLA'sının bu depoya uygulanacak şekilde onaylanması. |
| 125 | Ağ segmenti ayrımının güvenlik seviyesine uygunluğu | 🏦 | Depo default-deny + bileşen-bazlı dar ingress şablonu sağlar (madde 126); segment tasarımı ve uygunluk kararı bankadadır. |

## 14. Ağ ve API Güvenliği (126–132)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 126 | Firewall erişimlerinin kaynak/hedef/port bazında sınırlanması | ✅ | Starter: tüm pod'lara default-deny ingress+egress; public'e yalnız router'dan 8000/TCP; operator'a yalnız `bddk.bank/operator-client` etiketli namespace'lerden 8000/TCP (`deploy/openshift/networkpolicies.yaml`). Banka exact değerleri ekleyip uygulamalı. |
| 127 | Doğrudan internete açık mı | ✅ | Hayır — tasarım banka içi ağdır; internete açılma gereksinimi yoktur ve mevcut kimliksiz-public yapılandırmasıyla **açılmamalıdır** (unauthenticated opt-in yalnız ağ-izolasyonlu senaryo için tanımlı). |
| 128 | Dışa internet erişiminin allowlist ile sınırlanması | ✅ | Madde 26 ile aynı kanıt (uygulama allowlist'i + default-deny egress + dokümante dar allow matrisi; lifecycle Job'larına kaynak erişimi yasak). |
| 129 | API'lerde her istekte authentication + authorization | 🟡 | Operator: ✅ her istekte JWT (imza, issuer, audience, `typ`, expiry, scope). Public banka profili: ❌ bearer yok (`BDDK_HTTP_ALLOW_UNAUTHENTICATED`) — telafi kontrolleri Host/Origin allowlist, NetworkPolicy, rate/gövde sınırları. Madde 64'teki risk onayına bağlı. |
| 130 | API girdilerinde schema/input validation | ✅ | Tüm araç girdileri strict Pydantic modelleridir; bilinmeyen alan `extra="forbid"` ile reddedilir; sınır/enum/format şemalarda görünür (`bddk_mcp/tools/registry.py`, `tests/test_public_input_contracts.py`). |
| 131 | Toplu veri çekme/scraping'e karşı kontroller | ✅ | 120/dk rate limit + 32 eşzamanlılık + yanıt başına 5 sayfa belge sınırı + sonuç sayısı sınırları. (İçerik kamuya açık mevzuat olduğundan kalan risk maliyet/kaynak tüketimidir; paylaşımlı ingress limiti bankada.) |
| 132 | Girdi tipi/format/uzunluk/zorunluluk kontrolü | ✅ | Madde 130 ile aynı kanıt; ayrıca HTTP gövde boyutu (1 MiB) ve token uzunluğu (16.384) sınırları. |

## 15. Uygulama Kontrolleri (133–136)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 133 | Geçersiz/mantıksız veri girişinin engellenmesi | ✅ | Şema doğrulama + sınırlar + kararlı doğrulama hataları (madde 130). |
| 134 | Veri işlemede bütünlük kontrolleri | ✅ | Güçlü alan: içerik hash'leri, chunk sayısı/sıra/hash/embedding bütünlüğü geçmeden yayın yok; retrieval yalnız güncel doğrulanmış yayını okur, aksi halde fail-closed; DB katalog attestasyonu; imzalı manifest (`docs/DEPLOYMENT.md`, `bddk_mcp/corpus_publication.py`). |
| 135 | Kritik işlemlerde doğrulama/uyarı | ✅ | Yıkıcı/mutasyon araçları merkezi kayıtta `destructiveHint/readOnlyHint` ile işaretlidir (MCP istemcileri onay akışını buna göre uygular); mutasyonlar iş makbuzu döner, izlenebilir ve iptal edilebilirdir; release aktivasyonu iki-aşamalı ve tek-kullanımlıktır. |
| 136 | Üretim öncesi bağımsız güvenlik/sızma testi | ❌ | Yapılmadı (`docs/SECURITY_REVIEW.md` kapsam-dışı listesinde açıkça belirtilir). Geçiş şartı olarak banka pentest'i planlanmalı; kapsam önerisi 137–142'de. |

## 16. Sızma Testi (137–142)

Bağımsız sızma testi henüz yapılmamıştır; aşağıdaki satırlar mevcut otomatik
test kanıtını ve banka pentest kapsamına önerilen başlıkları verir.

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 137 | Authentication/authorization bypass testleri | ❌ | Otomatik negatif testler var (401/403/scope/Host/Origin, operator izolasyonu); bağımsız test yok. Pentest odağı: kimliksiz public mimarisinde ağ-izolasyon varsayımının delinip delinemeyeceği; operator düzlemine public'ten yükselme. |
| 138 | SQL/command/injection testleri | ❌ | Otomatik: parametrizasyon + FTS sanitizasyon + hostile girdi testleri. Bağımsız injection testi pentest kapsamına. |
| 139 | Session management testleri | ➖ | MCP stateless (madde 67/69); bu sınıf Open WebUI/AD oturumu için banka pentest'inde koşulmalı. |
| 140 | File upload/download güvenlik testleri | ❌ | Kullanıcı yüklemesi yok; test odağı ingestion indirme hattı (arşiv sınırları, redirect/DNS doğrulaması) ve 5-sayfa yanıt sınırı olmalı. |
| 141 | API güvenlik testleri | ❌ | MCP endpoint'i (schema zorlaması, rate/gövde sınırları, hata sızıntısı) bağımsız test edilmeli. |
| 142 | Kapsamlı değişiklik sonrası güvenlik testi tekrarı | 🟡 | Otomatik güvenlik regresyonları her PR'da koşar (sürekli); bağımsız pentest tekrarı banka politikasına bağlanmalı (ör. majör mimari değişiklikte). |

## 17. Yedekleme, Süreklilik ve Kurtarma (143–146)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 143 | Uygulama/konfig/veri yedekleme kapsamı | 🟡 | Depo: runbook + korumalı restore-drill aracı + iki-küme PG17 tatbikat kanıtı (53 yönetilen nesne, 7 LOGIN profili — `docs/RECOVERY_DRILLS.md`, `docs/evidence/`). Konfig git'te sürümlüdür. **Banka yedek/PITR kurulumu ve banka ortamında tatbikat açıktır** (CUR-009); geçiş şartı yapılmalı. |
| 144 | Yedeklerin yetkisiz erişime karşı korunması | 🏦 | Banka yedek altyapısı (şifreleme, custody, retention). Depo, uygulama DSN'lerinin restore yönetimi olarak kullanılmasını reddeder (yedek erişimi uygulama kimliklerinden ayrıktır). |
| 145 | Kritik BS servisi ise süreklilik planına dahil | 🏦 | BS kritiklik sınıflandırması ve süreklilik planı bankadadır. Depo kısıtı şeffaftır: operator tek replika (`Recreate`), RPO/RTO hedefleri onaysız (CUR-008/013) — plan bu kısıtları dikkate almalı. |
| 146 | Performans ve erişilebilirlik izleme | 🟡 | Health probe'ları + sekiz-metrik ölçüm sözleşmesi hazır, ancak hedef değerler onaysız ve alert'ler `not_implemented` (`docs/decisions/operational-objectives.v1.yml`); standart metrik exporter'ı yok. Banka izleme entegrasyonu + hedef onayı gerekli. |

## 18. Operasyon ve Güvenlik İzleme (147–150)

| # | Kontrol | Durum | Değerlendirme / Kanıt / Aksiyon |
|---|---|---|---|
| 147 | Hataların merkezî monitoring/loglamaya aktarımı | 🟡 | Yapılandırılmış JSON log + korelasyon kimliği stdout'a hazır; merkezî sisteme aktarım OpenShift log forwarding ile bankada kurulmalı (madde 94 ile birlikte). |
| 148 | Olağandışı davranış tespit senaryoları | ❌ | Tanımlı senaryo yoktur. Başlangıç seti depoda hazırdır: auth reddi artışı, throttle, corpus yaş/drift, vektör/bölüm kapsama düşüşü, kalite karantinası, sync hatası, ayrıcalık reddi (`docs/SECURITY_REVIEW.md`). Banka SOC ile senaryolaştırılmalı. |
| 149 | Güvenlik olaylarının SOME/SOC süreçlerine aktarımı | 🏦 | Banka olay yönetimi süreci; depo tarafında dış bildirim kanalı tanımlıdır (`SECURITY.md` özel zafiyet bildirimi). Log/alarm akışı 94/148'e bağlıdır. |
| 150 | Önemli olayların kök neden analizi ve tekrar önleme | 🟡 | Mühendislik pratiği kanıtlı: bulgular gap register'da kimlik/kanıt/efor ile izlenir, kapanışlar tarihli kanıtla kaydedilir (ör. CUR-001 kapanışı, v8 remediasyonu). Banka olay/problem yönetimi entegrasyonu kurulmalı. |

---

## Geçiş öncesi öncelikli aksiyon listesi

Sorumluluk: **B** = banka, **R** = repo değişikliği, **B+R** = ortak.

1. **B — Kimlik mimarisi risk onayı ve iz korelasyonu (64, 71–72, 88, 129).**
   Kimliksiz public MCP + ağ izolasyonu kararının banka risk kabulü; public
   Route'un yalnız onaylı segment/ön yüzden erişilebildiğinin testi; Open WebUI
   kullanıcı loglarının MCP korelasyon kimlikleriyle eşlenmesi.
2. **B — Dört-göz ve release görev ayrılığı (32–33, 108, 114; CUR-017).**
   Banka Git'inde bağımsız PR onayı zorunlu; verifier ve publisher
   Secret'larının farklı kişilere verildiğinin ve çapraz okumaya kapalı
   olduğunun kanıtlanması.
3. **B — Bağımsız sızma testi + DAST + egress/sızıntı testi (16, 101, 136–142).**
4. **B+R — Zafiyet yönetimi:** 60 OS-CVE istisnasının banka onayı (50, 104–105);
   zamanlanmış tarama (121, R: workflow'a `schedule`); taban imaj dijest
   güncelleme rutini (122); banka SLA onayı (124).
5. **R — SAST adımı:** CI'a Bandit/Semgrep benzeri adanmış SAST eklenmesi (100).
6. **B — İzleme/SIEM:** log forwarding + UTC zaman damgası yapılandırması +
   anomali senaryoları + SOC entegrasyonu (88, 94, 147–149).
7. **B — Yedek/PITR ve tatbikatın banka ortamında tekrarı (143–145; CUR-009);
   RPO/RTO ve operasyonel hedef değerlerinin onayı (124, 146; CUR-008).**
8. **B — İmaj imzalama, registry admission ve dijest doğrulama (27, 111;
   CUR-016); embedding modeli mirror onayı (15).**
9. **B — Resmî onay katmanı:** veri envanteri/sınıflandırma aktarımı (1–2, 79),
   sahiplik ataması (3), banka formatında BS risk analizi ve üretime geçiş
   onayı (5, 7, 11).
10. **R (öneri) — Route/ingress'te güvenlik header'ları (48); admin konsolunun
    banka dağıtımına dahil edilmemesi (22).**

## İlgili depo kaynakları

- [Güvenlik incelemesi](SECURITY_REVIEW.md) — tehdit modeli, bulgular, kabul kapıları
- [Gap register](GAP_REGISTER.md) — açık risklerin kimlikli kaydı (CUR-*)
- [Güncel repository durumu](STATUS.md) — doğrulanmış güncel sözleşme
- [Dağıtım rehberi](DEPLOYMENT.md) — profiller, kimlik matrisi, banka adımları
- [Tedarik zinciri politikası](../supply-chain/README.md) — tarama/istisna/release kuralları
- [Kurtarma tatbikatları](RECOVERY_DRILLS.md) ve [kanıtlar](evidence/)
