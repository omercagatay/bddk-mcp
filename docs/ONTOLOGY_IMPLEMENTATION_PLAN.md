# Ontoloji pilotu — Adım 0 uygulama planı (mevzuat_22599)

Tarih: 2026-10-28. Durum: **onaylandı (2026-10-28, sahip).** Üst plan: [ONTOLOGY_PLAN.md](ONTOLOGY_PLAN.md).
Bu sayfa Adım 1–5'i dosyalara bağlar; kod, veritabanı ve korpus bu adımda değişmedi.

## Okumadan çıkan temel bulgular

- **Şema değişikliği gerekmiyor.** Pilotun ihtiyaç duyduğu tüm tablolar var: `regulatory_instruments`,
  `regulatory_legal_versions`, `regulatory_legal_events`, `regulatory_legal_status_assertions`,
  `regulatory_provisions` (v0004) ve `regulatory_relations` (v0009). Yeni migration, yeni MCP aracı ve
  `tools/registry.py` değişikliği yok.
- **Mevcut yazma yolları yeniden kullanılır:** `import_legal_version_bundle` (`bddk_mcp/regulatory/repository.py`)
  ve `import_relations` (`bddk_mcp/regulatory/relations.py`). İkisi de "bir kez yaz, aynıysa geç, farklıysa iptal"
  kuralıyla çalışır; yanlış bir satır sonradan düzeltilemez. Bu yüzden her şey önce dosyaya, sonra `--dry-run`, en
  son onayla veritabanına gider. `relation_extract.resolve_and_import` kullanılmaz.
- **Araçlar yalnızca `validated` satırları görür** (`regulatory_validated_*` görünümleri). Onaylanmamış teklif hiçbir
  araçta görünmez; bu, kural 4 ve 5'i veritabanı düzeyinde zaten sağlıyor.
- **Statü çözücü** (`bddk_meta.resolve_regulation_status`, v0006) bir tarihte yalnızca şunlar varsa sonuç verir:
  doğrulanmış `publication` olayı ≤ tarih, doğrulanmış `effective` olayı ≤ tarih, tarihi kapsayan tam bir
  `effective` statü iddiası, sonlandırıcı olay yok. 1/1/2018 öncesi için `effective` olayı gelecekte kaldığı için
  `status_not_validated_for_date` ile çekimser kalır; penceremin dışı için de aynı. İstenen davranış ek kod
  gerektirmez.
- **Değişiklik zinciri aracı** (`get_amendment_chain`) hedefi bizim düzenleme olan `amends/repeals/replaces`
  kenarlarını listeler. Kenarın `source_instrument_id` alanı zorunlu bir yabancı anahtardır: 7 değiştirici
  yönetmelik `regulatory_instruments` tablosunda **asgari satır** olarak bulunmak zorunda (plan Adım 4 ile uyumlu).
- **Kaynak metin yerelde var:** `seed_data/documents.json` içinde `mevzuat_22599`, `content_hash`
  `c7a59f78…` , yöntem `manual_pdf`, resmî PDF sha256 `a5b278dd…` (`scripts/rebuild_mevzuat_from_pdf.py`,
  `SPECS["mevzuat_22599"]`), indirme zamanı 2026-04-21T15:52:19Z. Kayıtlı `source_url` http'li ve boşluk içeriyor;
  demet için HTTPS kanonik adres gerekir (karar E).

### Metinden doğrulanan gerçekler

| İddia | Sonuç |
|---|---|
| RG tablosu: asıl 22/6/2016–29750 ve 7 değişiklik | Doğru: 29918, 30373, 30510, 30569, 30961, 31533, 31845 |
| Satır içi etiketler | 24 etiket: `Ek` 11, `Değişik` 8, `Mülga` 3, `Değişik ibare` 1, `Başlığı ile Birlikte Değişik` 1; maddeler 5, 6, 7, 8, 13, 16, 18, 19, Geçici 1 ve 23 |
| Etiketlerin tablo ile çapraz doğrulaması | 24/24 etiketin RG tarihi ve sayısı tabloda var → hepsi `cross_confirmed` adayı |
| 5 haneli olmayan RG sayısı | Bu metinde yok (kontrol yine de yazılacak) |
| Madde 22 | 1/11/2006–26333 sayılı yönetmeliği yürürlükten kaldırır; hedef korpusta yok → `target_external_ref` |
| Madde 23 | "1/1/2018 tarihinde yürürlüğe girer"; kendisi RG 29918 ile değişmiş |
| Dikkat | Etiketlere yapışık dipnot işaretleri `$^{(1)}$` var; dipnotlar değişikliklerin yürürlük tarihlerini verir |

## Adım → dosya eşlemesi

Yeni kod `bddk_mcp/ontology/` alt paketinde toplanır; incelenmiş `regulatory/` motoru yalnızca küçük bir
yardımcıyla genişler. Her adımda `uv run ruff check .`, `uv run ruff format --check .`,
`uv run pytest tests/ -m "not postgres and not gpu"` ve `uv run python scripts/check_repository_hygiene.py` koşar.

| Adım | Yeni / değişen dosyalar | Not |
|---|---|---|
| 1 Spec — **tamamlandı 2026-10-28** | `ontology/ontology.yml` (yeni), `bddk_mcp/ontology/__init__.py`, `bddk_mcp/ontology/spec.py` (YAML yükleyici, kapalı şema), `tests/test_ontology_spec.py`, `scripts/check_repository_hygiene.py` (`ALLOWED_TOP_LEVEL` listesine yalnızca `ontology` eklenir) | Sürüklenme testi: spec'teki `storage: regulatory_relations` bağ türleri == `relations.RELATION_TYPES` == v0009 CHECK listesi. `contains` ve `creates` spec'te "henüz saklanmıyor" olarak işaretlenir. |
| 2 Kural çıkarıcı — **tamamlandı 2026-10-28** | `bddk_mcp/ontology/proposals.py` (Teklif/Karar modelleri), `bddk_mcp/ontology/rule_extract.py` (RG tablosu, satır içi etiket, Madde 22/23), `scripts/build_22599_link_proposals.py`, `docs/evidence/legal-versions/22599.proposals.json`, `tests/test_ontology_rule_extract.py` | Girdi `seed_data/documents.json`; `content_hash` dosyaya yazılır, ofsetler bu hash'e bağlıdır. Sonuç 34 teklif: 7 tablo satırı + yayım olayı + 24 etiket + Madde 22 (`repeals`) + Madde 23 yürürlük olayı; 31 `cross_confirmed`, 3 `needs_review`. Veritabanı açılmaz. |
| 3 İnceleme ekranı | `bddk_mcp/ontology/review_store.py` (ayrı SQLite, 0600, korpus dışı, yalnızca ekleme → denetim izi), `bddk_mcp/admin/services/proposals.py`, `bddk_mcp/admin/views/proposals.py`, `bddk_mcp/admin/templates/proposals/*.html`, `bddk_mcp/admin/config.py` (+`BDDK_ONTOLOGY_PROPOSALS`, `BDDK_ONTOLOGY_REVIEW_DB`), `bddk_mcp/admin/app.py` (rota kaydı), `tests/test_admin_proposals_view.py`, `tests/test_ontology_review_store.py`, `tests/test_admin_package_data.py` (yeni şablonlar) | CSRF: mevcut `FormSecurity`. Otomatik kabul: `cross_confirmed` kural teklifleri, teklif kimliğinden türetilen deterministik 1/5 örnek dışında `approved/auto:cross_confirmed` yazılır. Kısayollar A/R/N, sayaç. Sayfa PostgreSQL'e dokunmaz. |
| 4 Demet + dry-run | `bddk_mcp/ontology/bundle_builder.py`, `scripts/build_22599_link_bundle.py`, `docs/evidence/legal-versions/22599.decisions.json` (karar dışa aktarımı), `22599.json` (LegalVersionBundle), `22599.relations.json` (7 asgari enstrüman + kenarlar), `bddk_mcp/regulatory/repository.py` (+`import_instruments`), `bddk_mcp/ontology/importer.py`, `bddk_mcp/cli.py` (+`import-legal-bundle --bundle … --relations … --dry-run`), `tests/test_22599_link_bundle.py`, `tests/test_cli.py` | Yalnızca `approved` kararlar. Sürüm: `version_key` = PDF sha, `legal_text_sha256` = `c7a59f78…`, yayım 2016-06-22 (kanıt: RG tablosu satırı), yürürlük 2018-01-01 (kanıt: Madde 23), statü iddiası = verdiğin pencere, konsolidasyon `unknown`. Kenar kanıtı: `locator=normalized/chars=S-E`, `statement_sha256` = alıntının hash'i. `--dry-run` dosyaları doğrular ve tablo tablo satırları yazar, bağlantı açmaz. |
| 5 İçe aktarım + doğrulama | `tests/test_22599_graph_pilot.py` (`postgres` işaretli: gerçek demeti test havuzuna alır, 3 aracı çağırır), `benchmark/legal_answer_cases.py` (3 vaka: zincir 7 kenar, Madde 22 `repeals`, statü pencere içi çözülür / 2017-12-31 çekimser), `docs/STATUS.md` ve `docs/GAP_REGISTER.md` CUR-004 notu | Komut: `uv run bddk-mcp import-legal-bundle --db postgresql://bddk:bddk@localhost:5432/bddk_test …` (dry-run'sız) **yalnızca açık onayla, yalnızca yerel `bddk_test` veritabanı**. `--db` varsayılanı `migrate` ile aynı: `BDDK_SCHEMA_OWNER_DATABASE_URL`. |

## Riskler ve geri alma

| Risk | Önlem / geri alma |
|---|---|
| Satırlar bir kez yazılır; yanlış kenar düzeltilemez | Dosya → dry-run → onay sırası; önce yalnızca yerel test veritabanı. Geri alma: `bddk_test` veritabanını düşür, `docker compose up bddk-test-db` ile yeniden oluştur (betik idempotent), `uv run bddk-mcp migrate`. Paylaşılan veritabanına pilot yazılmaz. |
| Yazma kimliği: `deploy/postgres/02_grants.sql` tüm uygulama rollerinden (`bddk_ingestion` dâhil) `regulatory_*` tablolarına yazma yetkisini alıyor; yalnızca şema sahibi yazabilir | İçe aktarım komutu `migrate` gibi şema-sahibi DSN'i ile çalışır. Yerel `bddk_test` fikstürü süper kullanıcı `bddk` rolünü kullanır; bankadaki rol modeli pilot kapsamında değişmez. |
| Her `regulatory_*` yazımı korpus epoch'unu artırır (v0005/v0011 tetikleyicileri) → etkin sürüm geçersizleşir, `get_amendment_chain` dâhil yerel-korpus araçları yeniden verify/stage/activate yapılmadan `CORPUS_RELEASE_UNAVAILABLE` verir | Adım 5 doğrulaması pytest üzerinden (araç fonksiyonları doğrudan, korunma dışı). Çalışan sunucuda görmek istersen test veritabanında sürüm yeniden etkinleştirilir; bu ayrı onay ister. |
| Ofsetler ve kanıt hash'leri `content_hash c7a59f78…` metnine bağlı; metin değişirse (yeni kalite geçişi) teklifler eskir | Teklif dosyası hash'i taşır; uyuşmazsa çıkarıcı ve dry-run reddeder (943 demetinin v11'de bağlarının kopması emsali). |
| Kimlik seçimleri (identity_key, kanonik URI, enstrüman türü) içe aktarımdan sonra değişmez | Aşağıdaki kararlar Adım 4'ten önce verilir; dry-run çıktısında görünür. |
| Araç çıktısı değiştirici yönetmelikleri yalnızca `inst_sha256_…` ile gösterir, başlık yok | Pilot için kabul; başlık/identity_key gösterimi backlog. |
| Bölüm düzeyi çapraz referans (`section_type=madde, section_ref=22`) için `document_section_id` eşlemesi gerekir | Pilotta belge düzeyi `get_cross_references` yeter; eşleme Adım 5'te isteğe bağlı (test veritabanı aynı metni taşıyor). |
| Hijyen/paketleme kontrolleri: yeni kök dizin, yeni dokümanlar, yeni şablonlar | `ALLOWED_TOP_LEVEL` +1, `docs/README.md` dizini, `test_admin_package_data.py` güncellemesi aynı adımda yapılır. |
| Adım 1–4 geri alma | Yalnızca dosya: `git checkout -- <yol>` / yeni dosyayı sil; inceleme deposu sidecar SQLite dosyasıdır, silinmesi korpusu etkilemez. |

## Kararlar (önerilen varsayılanlarla 2026-10-28'de onaylandı)

- **A. Ana enstrüman `identity_key`:** `22599` (öneri; `_resolve_target_instrument` rakam-sınırı eşlemesiyle ve belge
  kimliğiyle uyumlu). Alternatif: `7.5.22599`.
- **B. 7 değiştirici yönetmelik:** `identity_key` `rg-29918` biçimi, başlık "RG 14/12/2016-29918 ile değişiklik
  yönetmeliği", tür `yonetmelik` (öneri). Resmî başlıkları korpusta yok; kaynak metin tabloyu veriyor.
- **C. Satır içi `Ek/Değişik/Mülga` etiketleri:** bizim maddeyi hedefleyen `amends` kenarı
  (`target_provision_id` = `madde/N`), değişiklik türü kanıt açıklamasında; `repeals` yalnızca Madde 22 (öneri).
- **D. Kural kanıtlarının otorite düzeyi:** `authoritative` (resmî konsolide metinden birebir alıntı; öneri). Adım 6
  AI teklifleri `secondary`.
- **E. Kanonik HTTPS kaynak:** `https://www.mevzuat.gov.tr/MevzuatMetin/yonetmelik/7.5.22599.pdf` (rebuild
  betiğinin kullandığı statik PDF; sha `a5b278dd…`). `retrieved_at` 2026-04-21T15:52:19Z.
- **F. Statü penceresi:** `valid_from` / `valid_through` tarihlerini sen verirsin; 1/1/2018'den önce başlayamaz. **Açık:** tarihler henüz verilmedi; Adım 4 başlamadan önce istenecek.
- **G. Otomatik kabul edilen tekliflerin veritabanındaki `validated_by` değeri:** `project_owner` (demeti sen
  onaylıyorsun); karar kaydında `auto:cross_confirmed` izi kalır (öneri).
- **H. Spec konumu:** kökte `ontology/` (planındaki gibi; hijyen listesine bir giriş).

## Bu oturumda yapılmayanlar

Kod, test, veritabanı, korpus veya `seed_data/` değişikliği yok. `uv sync` çalıştırılmadı; hijyen betiği
yerelde PyYAML gerektirdiği için koşulmadı (dizin ve bağlantılar elle kontrol edildi); PR #185 CI'da lint, hijyen,
birim, PostgreSQL ve paketleme kontrolleri yeşil geçti. Dokümanlar sahibin stop hook'u üzerine
`ccr-f9d2104c-5x206h` dalına gönderildi; `main` dalına birleştirme sahibin kararıdır.
