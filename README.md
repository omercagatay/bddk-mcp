# BDDK MCP Server

[Türkçe](#turkce) | [English](#english) | [English-only operational guide](README.en.md)

BDDK MCP Server is an offline-first Model Context Protocol server for searching, retrieving, and analyzing Turkish banking regulation data from BDDK and mevzuat.gov.tr. It combines catalog search, document retrieval, section-level legal lookup, semantic search, bulletin analytics, document quality checks, and operator backfill workflows.

---

<a id="turkce"></a>

## Türkçe

### Ne İşe Yarar?

Bu proje, BDDK karar ve düzenlemeleri için güvenli ve izlenebilir bir MCP sunucusu oluşturmayı hedefler. Amaç, modelin kendi bilgisinden cevap üretmesi yerine yerel veri deposundaki BDDK kaynaklarına dayanmasıdır. Mevcut üretim güvenliği sınırları için [deployment belgesine](docs/DEPLOYMENT.md) bakın.

Temel kullanım alanları:

- BDDK düzenleme kataloğunda arama
- Doküman gövdesinde semantik ve tam metin arama
- Belirli doküman sayfalarını Markdown olarak getirme
- `Madde`, `İlke`, `Paragraf`, `Ek` gibi bölümleri doğrudan getirme
- Haftalık ve aylık bankacılık bülteni verilerini sorgulama
- Regülatör değişiklikleri, duyurular ve trendler için özet üretme
- Doküman kalitesi, OCR/formül riski ve extraction hatalarını izleme

### Öne Çıkan Özellikler

- **MCP SDK tabanlı araçlar:** stdio ve Streamable HTTP için Claude/Codex yapılandırma örnekleri vardır; bunlar release-spesifik istemci uyumluluk sertifikası değildir.
- **Offline-first doküman retrieval:** Düzenleme metinleri ve bölümleri PostgreSQL/pgvector üzerinden servis edilir; kurum, duyuru ve bülten araçları upstream erişimi gerektirebilir.
- **Katalog ve gövde araması ayrımı:** `search_bddk_regulations` sadece başlık/metadata arar; `search_document_store` doküman gövdesinde semantik arama yapar.
- **Bölüm bazlı erişim:** `get_document_section` ve `search_document_sections` ile `943 İlke 5` veya `mevzuat_22599 Madde 9` gibi referanslar doğrudan bulunur.
- **Exact legal-reference koruması:** `Madde 9` gibi lexical eşleşmeler, semantik skor düşük olsa bile korunur.
- **Kalite etiketleri:** Doküman çıktıları `clean`, `warning`, `fail` sinyalleri ve kalite bayraklarıyla işaretlenir.
- **Doküman context sanitization:** `get_bddk_document`, Data URI, raw HTML/OCR artefact ve uzun satırları model context'ine verilmeden temizler.
- **Operatör scriptleri:** kalite tarama, kalite backfill ve `document_sections` reindex akışları mevcuttur.
- **PostgreSQL + pgvector:** dokümanlar, bölümler, FTS ve vektör arama tek veritabanı üzerinde çalışır.

### Araç Yüzeyi

Varsayılan `public` process profili (`BDDK_TOOL_PROFILE=public` veya `bddk-mcp serve --profile public`) yalnızca 15 public araç expose eder.

| Modül | Araçlar |
|---|---|
| Arama | `search_bddk_regulations`, `search_document_store`, `search_bddk_institutions`, `search_bddk_announcements` |
| Doküman | `get_bddk_document`, `get_document_history` |
| Bölümler ve hukuki durum | `get_document_section`, `search_document_sections`, `resolve_regulation_status` |
| Bülten | `get_bddk_bulletin`, `get_bddk_bulletin_snapshot`, `get_bddk_monthly` |
| Analitik | `analyze_bulletin_trends`, `get_regulatory_digest`, `compare_bulletin_metrics` |

Ayrı `operator` process profili (`BDDK_TOOL_PROFILE=operator` veya `bddk-mcp serve --profile operator`) 15 public araca 14 operatör aracı ekler ve toplam 29 araç expose eder. Bu profil ayrı, yazma yetkili `BDDK_OPERATOR_DATABASE_URL` ister; public DSN'e geri dönmez.

- `check_bddk_updates`
- `document_store_stats`
- `bddk_cache_status`
- `refresh_bddk_cache`
- `sync_bddk_documents`
- `trigger_startup_sync`
- `get_operator_job`
- `list_operator_jobs`
- `cancel_operator_job`
- `document_health`
- `health_check`
- `bddk_metrics`
- `backfill_degraded_documents`
- `document_quality_report`

Geçerli runtime için canonical operator registry 15 public ve 14 operatör aracı, yani toplam 29 MCP aracı içerir. Mutating operatör araçları hemen bir job receipt döner; durum `get_operator_job`, `list_operator_jobs` ve `cancel_operator_job` ile izlenir. Job kayıtları, hash'lenmiş idempotency anahtarları, sayısal progress ve sınırlı sonuç metrikleri PostgreSQL'deki `bddk_operator.operator_jobs` tablosunda durable olarak tutulur. Session-level job-admission lease aynı runner'ın process'ler arasında eşzamanlı sahiplenilmesini önler; ayrı transaction-level corpus-mutation lock ise sanctioned writer transaction'larını ve release publisher'ı serialize eder. Runner task'ları hâlâ operator process'indedir; stale `queued` işler otomatik tahmin edilmez ve çok-replica failover banka ortamında doğrulanmamıştır. Bu nedenle OpenShift starter tek `Recreate` replica kullanır ve sistem bank-grade workflow queue olarak sunulmamalıdır. Benchmark şemaları aynı canonical operatör registry'sinden üretilir; benchmark koşuları yine de kullandıkları exact tool listesini ve profili kaydetmelidir. Bkz. [benchmark/README.md](benchmark/README.md).

### Hızlı Başlangıç

Gereksinimler:

- Python 3.12 veya 3.13
- `uv`
- PostgreSQL 17, `pgvector` ve `unaccent` (bu release test edilmemiş major sürümleri fail-closed reddeder)
- Opsiyonel: Docker Compose

Kurulum:

```bash
git clone https://github.com/omercagatay/bddk-mcp.git
cd bddk-mcp
uv sync
```

Disposable lokal PostgreSQL lifecycle:

```bash
export BDDK_JWT_ISSUER=https://idp.invalid
export BDDK_JWT_RESOURCE=https://localhost:8000/mcp
export BDDK_JWT_JWKS_URL=https://idp.invalid/jwks
export BDDK_JWT_AUDIENCE=bddk-mcp-local
docker compose up --build -d bddk-bootstrap
docker compose wait bddk-bootstrap
export BDDK_DATABASE_URL=postgresql://bddk_local_public:local-only-public@localhost:5432/bddk
```

Compose yalnızca loopback geliştirme ortamında DBA role/extension hazırlığı → schema-owner `migrate` → DBA grants → ingestion `bootstrap` sırasını çalıştırır. `.invalid` JWT değerleri Compose'un kullanılmayan HTTP service tanımını parse etmesi içindir; bu lifecycle komutu HTTP server başlatmaz ve bu değerler server çalıştırmak için geçerli değildir. Sabit şifreler public test fixture'larıdır ve remote ortamda kullanılmamalıdır. Üretimde `bddk-mcp migrate` yalnız schema işidir; `bddk-mcp bootstrap` önceden migrate edilmiş ve grant uygulanmış şemaya reviewed seed, section ve 768-boyutlu embedding yazar, migration çalıştırmaz. Ayrı kimlik ve tam sıra için [deployment belgesine](docs/DEPLOYMENT.md) bakın.

Corpus kapsamını ve üç seed artifact'ını DB bağlantısı kurmadan incelemek için
isteğe bağlı read-only preflight çalıştırın:

```bash
uv run --frozen bddk-mcp verify-corpus
```

Bu komut checksum, boyut, kayıt sayısı ve freshness zamanlarını doğrular; ancak
sonraki bir process'e güven aktarmaz. Production import aynı strict policy'leri
doğrudan mutating `bootstrap` invocation'ında yeniden uygulamalıdır:

```bash
BDDK_INGESTION_DATABASE_URL='postgresql://INGESTION:SECRET@HOST:5432/DATABASE?sslmode=verify-full&sslrootcert=%2FAPPROVED%2Fpostgres-ca.crt' \
  uv run --frozen bddk-mcp bootstrap \
    --seed-dir /APPROVED/CORPUS \
    --reindex-existing \
    --require-quantified-freshness \
    --require-measured-freshness \
    --require-verified-signature \
    --trusted-signing-key /APPROVED/TRUST/corpus-signing-public-key.pem
```

`bootstrap`, DB pool açmadan önce exact `corpus_scope.yml` ve manifest'te
tanımlanan artifact path/bytes/hash'lerini doğrular; mevcut fakat manifest'te
tanımsız `documents.json`, `chunks.json` veya `decision_cache.json` dosyasını
reddeder. Trust key corpus'tan ayrı bir Secret/mount ile gelmelidir. Numeric
hedef tanımlamak tek başına ölçüm değildir: `measured` durumunda her doküman
için authoritative publication → source detection → download → extraction →
retrieval publication zaman zinciri ve hesaplanan gecikmeler hedefler içinde
olmalıdır. Mevcut reviewed manifest imzasızdır, numeric hedefleri yoktur ve
`slo_evidence_status: not_measured` kabul edilir; bu production bootstrap'ını
bilinçli olarak geçmez. Başarılı bootstrap, operator evidence için path-free
manifest ID ve SHA-256 döndürür ve ayrı publication gerektiğini bildirir.
`bddk-mcp publish-corpus-release`, bağımsız `bddk_release_publisher` identity'si
ile corpus'u yeniden doğrular; release ve activation kayıtlarını aynı
transaction içinde persist eder.

Ledger öncesi eski bir veritabanı ordinary migration tarafından fail-closed reddedilir. `--adopt-legacy` yalnız exact desteklenen shape için, doğrulanmış backup ve [legacy upgrade runbook'u](docs/LEGACY_DATABASE_UPGRADE.md) ile kullanılan açık bir seçimdir; clean install veya genel repair flag'i değildir.

Dolu bir version-2 veritabanında migration 3 de blocking retrieval-publication backfill ve foreign-key validation nedeniyle varsayılan olarak reddedilir. `--allow-retrieval-publication-backfill` yalnız workload'lar durdurulduktan, geri yüklenebilir backup kanıtlandıktan ve aynı boyuttaki restore üzerinde prova yapıldıktan sonra kontrollü bakım penceresinde kullanılmalıdır. `BDDK_EXPECTED_DATABASE_NAME` ve DBA script'lerinin bağımsız hedef ayarı aktif veritabanıyla eşleşmelidir. İzole lokal Compose profili dışında PostgreSQL DSN'leri `sslmode=verify-full` ve absolute `sslrootcert` kullanmak zorundadır.

Test:

```bash
uv run pytest tests/test_tools_sections.py tests/test_doc_store.py -k section -v
uv run ruff check .
```

MCP stdio çalıştırma:

```bash
BDDK_DATABASE_URL=postgresql://bddk:bddk@localhost:5432/bddk \
uv run --frozen bddk-mcp serve
```

HTTP transport:

```bash
BDDK_DATABASE_URL=postgresql://bddk:bddk@localhost:5432/bddk \
MCP_TRANSPORT=streamable-http \
PORT=8000 \
uv run --frozen bddk-mcp serve
```

Streamable HTTP MCP endpoint'i `http://localhost:8000/mcp` olur ve server stateless JSON response modunda çalışır. Remote uygulama RFC 9728 protected-resource metadata'yı `/.well-known/oauth-protected-resource/mcp` yolunda yayınlar; 401 challenge aynı URL'yi `resource_metadata` ile bildirir. Bu application-level MCP authorization discovery'dir; bank IdP client registration/flow acceptance kanıtı değildir. Sabit, content-free probe endpoint'leri `GET /health/live` ve `GET /health/ready`'dir; readiness migration, kritik catalog objeleri, corpus publication ve workload ACL'lerini periyodik olarak yeniden doğrular. Probe'lar authentication/Host kontrolü dışında olsa da process rate ve concurrency limitlerine tabidir. Loopback dışı bind fail-closed davranır: exact Host/HTTPS Origin allowlist'leri ve tam JWT/JWKS ayarları zorunludur; public profil `bddk.read`, operator profil `bddk.operator` scope'u ister. Remote operator ayrıca `BDDK_OPERATOR_REMOTE_ENABLED=true` ile açık opt-in ister. Body, concurrency ve dakikalık rate limitleri uygulama process'i içinde uygulanır; replica'lar arasında global ingress limiti sağlamaz. Ayrıntılar için [deployment belgesine](docs/DEPLOYMENT.md) bakın.

Eski seed import/export yardımcı komutu da korunur; yeni deployment'larda doğrulama içeren `bddk-mcp bootstrap` tercih edilir:

```bash
BDDK_INGESTION_DATABASE_URL=postgresql://bddk_local_ingestion:local-only-ingestion@localhost:5432/bddk \
uv run --frozen bddk-seed import
```

### Claude Yapılandırması

Repository kökündeki [`.mcp.json`](.mcp.json), repository kökünü çalışma dizini olarak kullanan `.mcp.json` uyumlu istemciler için taşınabilir bir stdio örneğidir:

```json
{
  "mcpServers": {
    "bddk": {
      "command": "uv",
      "args": ["run", "--frozen", "bddk-mcp"],
      "env": {
        "MCP_TRANSPORT": "stdio",
        "BDDK_DATABASE_URL": "${BDDK_DATABASE_URL}"
      }
    }
  }
}
```

### Codex Yapılandırması

Codex CLI ve IDE extension aynı Codex MCP ayarını kullanır. `~/.codex/config.toml` veya güvenilen bir repository içindeki `.codex/config.toml` dosyasına şunu ekleyin; `cwd` değerini kendi checkout yolunuza göre değiştirin:

```toml
[mcp_servers.bddk]
command = "uv"
args = ["run", "--frozen", "bddk-mcp"]
cwd = "/absolute/path/to/bddk-mcp"
env_vars = ["BDDK_DATABASE_URL"]
startup_timeout_sec = 30
tool_timeout_sec = 60
```

`codex mcp list` veya Codex içinde `/mcp` ile bağlantıyı doğrulayın.

Docker, Railway ve OpenShift AI sınırları için [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) belgesine bakın.

### Örnek Sorgular

```text
search_bddk_regulations(keywords="kredilerin sınıflandırılması")
search_document_store(query="TFRS 9 kredi riskinde önemli artış")
get_bddk_document(document_id="mevzuat_22599", page_number=1)
get_document_section(document_id="943", section_type="ilke", section_ref="5")
search_document_sections(query="Karşılık Yönetmeliği Madde 9 TFRS 9")
get_bddk_bulletin(metric_id="1.0.1", currency="TRY", days=90)
analyze_bulletin_trends(metric_id="1.0.1", lookback_weeks=12)
get_regulatory_digest(period="week")
```

### Operatör Workflowleri

Kalite taraması:

```bash
uv run python scripts/scan_document_quality.py --db --out-dir quality_reports --allow-failures
```

Kalite problemi olan dokümanları dry-run:

```bash
uv run python scripts/backfill_quality_failures.py --dry-run
```

Belirli bir kalite failure dokümanını yeniden çekme:

```bash
uv run python scripts/backfill_quality_failures.py --doc-id mevzuat_21192 --execute
```

Mevcut dokümanlardan `document_sections` tablosunu ingestion kimliğiyle yeniden oluşturma:

```bash
BDDK_INGESTION_DATABASE_URL=postgresql://INGESTION:SECRET@HOST:5432/DB \
  uv run python scripts/reindex_document_sections.py --execute
```

Execution yapan quality backfill, sync ve reindex scriptleri de `BDDK_INGESTION_DATABASE_URL` ister ve exact `bddk_ingestion` privilege contract'ını doğrular; public veya operator DSN'iyle çalıştırılmamalıdır.

Opsiyonel retrieval telemetry:

```bash
BDDK_DATABASE_URL=postgresql://PUBLIC:SECRET@HOST:5432/DB \
BDDK_TELEMETRY_ENABLED=true \
BDDK_TELEMETRY_DATABASE_URL=postgresql://TELEMETRY:SECRET@HOST:5432/DB \
  uv run --frozen bddk-mcp serve --profile public
```

Telemetry varsayılan olarak kapalıdır. Açıldığında ayrı LOGIN yalnız `bddk_telemetry_writer` rolünü inherit etmelidir; startup column-scoped INSERT-only yetkiyi doğrular ve trace okuma/değiştirme veya geniş membership'i reddeder. `tool_call_traces` tablosuna latency, result count, doc ID, kalite etiketi ve relevance özeti yazar; query/prompt metni hash/uzunluk özeti olarak saklanır. Raw metin yalnızca `BDDK_TELEMETRY_STORE_TEXT=true` açıkça set edilirse yazılır.

### Mimari

```text
server.py                 Kök shim → bddk_mcp/server.py
seed.py                   Kök shim → bddk_mcp/ingest/seed.py
bddk_mcp/                 Ana paket
  server.py               FastMCP giriş noktası ve lifecycle
  core/                   config, DB identity, outbound HTTP, logging ve modeller
  migrations/             Immutable global PostgreSQL migration ledger
  jobs/                   Durable operator job modelleri ve PostgreSQL repository
  store/                  doc_store, vector_store, section_index, legal_ref
  ingest/                 client, data_sources, doc_sync, html_extractor, backfill, seed
  quality/                markdown_quality, quality_scan
  observability/          analytics, telemetry, metrics
  tools/                  MCP tool modülleri
  ocr/                    base, chandra (pluggable OCR)
scripts/                  Operatör ve backfill scriptleri
benchmark/                Tool schema ve benchmark altyapısı
```

### Veri Kalitesi ve Güvenlik Notları

- Tam düzenleme dokümanı ve bölüm retrieval cevapları lokal store'dan gelir; bu iki akış runtime'da doküman live-fetch yapmaz.
- Katalog cache yenileme, kurum/duyuru araması ve bülten araçları yapılandırmaya ve cache durumuna göre BDDK upstream servislerine erişebilir.
- Live regulatory HTTP path'leri exact BDDK/mevzuat HTTPS host'ları, redirect/DNS revalidation ve artifact tipine göre code-owned streaming limitleri uygular; URL/query/exception metni retry loglarına yazılmaz. Public kurum/duyuru/bülten/update araçları da live BDDK erişimi yapabildiği için OpenShift egress kontratı hem public hem operator runtime'a yalnız approved regulatory-source veya proxy için TCP 443 vermeli; lifecycle Job'larına bu erişim verilmemelidir. DNS-to-connect yarışı nedeniyle NetworkPolicy veya approved proxy/firewall zorunludur.
- Default embedding modeli full commit `d13f1b27baf31030b7fd040960d60d909913633f`, opsiyonel default reranker `1427fd652930e4ba29e8149678df786c240d8825` ile pinlenmiştir; immutable şema yalnız `vector(768)` kabul eder. Model/chunk ayarı değişikliği kontrollü full re-embedding ve retrieval regression gerektirir.
- Retrieval publication kaydı yalnız chunk bütünlüğü, güncel content hash'i ve aktif retrieval profile doğrulandıktan sonra yazılır; eksik veya stale index arama sonuçlarına sessizce karışmaz.
- Bootstrap, reviewed corpus'u manifest'in exact artifact path'lerine bağlar ve reserved seed filename bypass'ını reddeder. Ayrı `verify-corpus` koşusu yalnız diagnostic preflight'tır; production güven gate'leri doğrudan aynı `bootstrap` komutuna ve ayrı mount edilen trust key'e verilmelidir. `deploy/openshift-overlays/bank-bootstrap` bu exact komutu, read-only approved-corpus PVC'yi ve ayrı read-only corpus-trust Secret'ını repository preflight'ında doğrular; gerçek bank PVC/Secret provision ve Job koşusu hâlâ dış gate'tir.
- v0005 strict publisher'ın bağımsız doğruladığı manifest/retrieval kimliğini append-only release ve activation olarak PostgreSQL'e kaydeder. On yedi corpus tablosundaki her mutation singleton epoch'i artırıp active view'ı geçersiz kılar; strict local-corpus tool çağrıları aynı release'i çağrı öncesi/sonrası doğrular. v0007 ayrıca ayrı release-publisher kimliğinin `bddk-mcp retain-corpus-generation --expected-release-id ...` ile exact active state'i 17 typed retained relation'a kopyalayıp mühürlemesini sağlar. Physical generation ve seal corpus state/retrieval profile'dan türetilir; aynı exact state/profile'ı yöneten farklı release'ler depolamayı çoğaltmak yerine aynı generation/seal'e ayrı release binding'leri alır. v7 retained-row ile current/retained state hashlerini function-local `TimeZone=UTC`, `DateStyle=ISO, YMD`, `IntervalStyle=postgres`, `bytea_output=hex` ve `extra_float_digits=3` ayarlarıyla session formatından bağımsızlaştırır. v7 öncesinde farklı ayarlar altında publish edilmiş active release canonical recomputation ile uyuşmazsa v7 migration hash fonksiyonunu değiştirmeden fail closed olur; historical release satırı değiştirilmeden veya backfill edilmeden, değişmeden kalan v7-öncesi şemada (v5 veya v6) yalnız publication için tanımlı exact `publish-corpus-release` uyumluluk yolu corpus'u yeniden review ederek canonical release'i publish/activate etmeli ve ardından v7 tekrar denenmelidir. Serving, retention ve diğer workload admission yolları v7-only kalır. Bu administrative CLI bir MCP tool'u değildir; retained veriyi serve, activate, reactivate veya rollback etmez. Generation-bound serving ve authorized rollback H2-02B'de açıktır; backup growth `not_measured` ve banka retention/capacity onayı eksiktir. Tracked manifest'teki 8.286 chunk, mevcut profile-v2'nin ürettiği 9.675 satırla uyuşmadığı için tracked corpus şu anda strict publish edilemez.
- v0004'ün 11 owner-only legal-curation tablosu `SourceBlob` içerik kimliğini `SourceArtifact` acquisition kimliğinden ayırır; v0006 public `resolve_regulation_status` fonksiyon/tool yolunu conflict veya eksik kanıtta abstain edecek şekilde ekler. Synthetic real-PostgreSQL kanıtı gerçek regulation family/currentness kanıtı değildir.
- Evaluation gate dört imzalı katman ister: ölçülmüş corpus, expert dataset, exact Citation pack için legal-curator attestation ve retained source/acquisition/page/excerpt zincirini bağlayan legal-release checkpoint. Canonical corpus/dataset/curator/release signer fingerprint'leri ayrı olmalıdır. Mevcut preflight yalnız operator-supplied anchor'lar altında cryptographic consistency kanıtlar; bank authorization ve model-score authorization her zaman false'tur. Tracked 20-case dataset draft'tır; key rotation, named reviewer policy ve expert-case execution henüz yoktur.
- Supply-chain lane container'ları Buildx `--provenance=false --load` ile lokal olarak üretir; manifest descriptor/digest, config digest, loaded image ve Syft SBOM aynı image'a fail-closed bağlanır. Repository ayrıca unsigned SLSA provenance üretir ve model manifest/runtime/Dockerfile pinlerinin uyumunu doğrular. Pending exception kullanılan sonuç hiçbir zaman promotion-eligible değildir; bank signing, admission ve registry promotion yine dış gate'tir.
- Runtime wheel/sdist `seed_data`, benchmark ve deployment asset'lerini içermez; sağlanan container reviewed seed'i açıkça içerir. Wheel kurulumu approved corpus mount etmeli ve bootstrap'a `--seed-dir` veya `BDDK_SEED_DIR` vermelidir.
- Kalitesi düşük extraction çıktıları `warning` veya `fail` olarak işaretlenir.
- Formül ağır veya OCR bozuk dokümanlarda kaynak PDF incelemesi gerekebilir.
- `get_bddk_document` cevaplarında data URI, raw HTML ve bazı OCR artefact'ları temizlenir.
- Model cevap verirken sadece tool çıktısına dayanmalıdır; karar numarası, tarih veya hukuki sonuç uydurulmamalıdır.
- Bilinen extraction sorunları, fail doküman listesi ve backfill komutları için [docs/DOCUMENT_QUALITY.md](docs/DOCUMENT_QUALITY.md) sayfasına bakın.
- Bankanın OpenShift AI cluster'ı, backup/restore akışı ve Claude/Codex/GPT/GPT-OSS/LM Studio/local model client matrix'i henüz bu repository ile acceptance testinden geçmemiştir.

---

<a id="english"></a>

## English

### What Is This?

BDDK MCP Server aims to provide a safe, auditable Model Context Protocol interface for Turkish banking regulation data. It is designed to ground LLM answers in local BDDK data instead of relying on the model's prior knowledge. See the [deployment guide](docs/DEPLOYMENT.md) for current production-security boundaries.

Common use cases:

- Search the BDDK regulation catalog
- Search inside document bodies with semantic and full-text retrieval
- Retrieve paginated Markdown documents
- Retrieve exact legal sections such as `Madde`, `Ilke`, `Paragraf`, and `Ek`
- Query weekly and monthly banking bulletin data
- Produce regulatory digests and trend summaries
- Monitor document quality, OCR/formula risk, and extraction failures

### Highlights

- **MCP SDK tool surface:** stdio and Streamable HTTP examples are provided for Claude/Codex; examples are not release-specific compatibility certification.
- **Offline-first document retrieval:** regulation text and sections are served from PostgreSQL/pgvector; institution, announcement, and bulletin tools may require upstream access.
- **Catalog/body separation:** `search_bddk_regulations` searches metadata; `search_document_store` searches document bodies.
- **Section-level retrieval:** `get_document_section` and `search_document_sections` support references like `943 Ilke 5` and `mevzuat_22599 Madde 9`.
- **Exact legal-reference preservation:** lexical hits such as `Madde 9` survive dense relevance filtering.
- **Quality labels:** document outputs include `clean`, `warning`, or `fail` metadata and quality flags.
- **Document-context sanitization:** `get_bddk_document` removes data URIs, raw HTML/OCR artifacts, and pathological long lines before model context.
- **Operator scripts:** quality scan, quality backfill, and `document_sections` reindex workflows are included.
- **PostgreSQL + pgvector:** documents, sections, FTS, and vector search share one database.

### Tool Surface

The default `public` process profile (`BDDK_TOOL_PROFILE=public` or `bddk-mcp serve --profile public`) exposes only 15 public tools.

| Module | Tools |
|---|---|
| Search | `search_bddk_regulations`, `search_document_store`, `search_bddk_institutions`, `search_bddk_announcements` |
| Documents | `get_bddk_document`, `get_document_history` |
| Sections and legal status | `get_document_section`, `search_document_sections`, `resolve_regulation_status` |
| Bulletin | `get_bddk_bulletin`, `get_bddk_bulletin_snapshot`, `get_bddk_monthly` |
| Analytics | `analyze_bulletin_trends`, `get_regulatory_digest`, `compare_bulletin_metrics` |

The separate `operator` process profile (`BDDK_TOOL_PROFILE=operator` or `bddk-mcp serve --profile operator`) adds 14 operator tools to the 15 public tools and exposes 29 tools total. It requires a separate, write-capable `BDDK_OPERATOR_DATABASE_URL` and never falls back to the public DSN.

- `check_bddk_updates`
- `document_store_stats`
- `bddk_cache_status`
- `refresh_bddk_cache`
- `sync_bddk_documents`
- `trigger_startup_sync`
- `get_operator_job`
- `list_operator_jobs`
- `cancel_operator_job`
- `document_health`
- `health_check`
- `bddk_metrics`
- `backfill_degraded_documents`
- `document_quality_report`

The canonical operator registry contains 15 public tools plus 14 operator tools, or 29 MCP tools total. Mutating operator tools return an immediate job receipt; use `get_operator_job`, `list_operator_jobs`, and `cancel_operator_job` to observe them. Job records, hashed idempotency keys, numeric progress, and bounded result metrics are durable in PostgreSQL table `bddk_operator.operator_jobs`. A session-scoped job-admission lease prevents concurrent ownership of the same runner across processes; a distinct transaction-scoped corpus-mutation lock serializes sanctioned writer transactions and the release publisher. Runner tasks still live in the operator process, stale `queued` work is never guessed automatically, and multi-replica failover has not been accepted in a bank environment. The OpenShift starter therefore uses one `Recreate` replica, and this is not represented as a bank-grade workflow queue. Benchmark schemas are exported from the same canonical operator registry; benchmark runs should still record the exact tool list and profile they used. See [benchmark/README.md](benchmark/README.md).

### Quick Start

Requirements:

- Python 3.12 or 3.13
- `uv`
- PostgreSQL 17 with `pgvector` and `unaccent` (this release fails closed on untested major versions)
- Optional: Docker Compose

Install:

```bash
git clone https://github.com/omercagatay/bddk-mcp.git
cd bddk-mcp
uv sync
```

Disposable local PostgreSQL lifecycle:

```bash
export BDDK_JWT_ISSUER=https://idp.invalid
export BDDK_JWT_RESOURCE=https://localhost:8000/mcp
export BDDK_JWT_JWKS_URL=https://idp.invalid/jwks
export BDDK_JWT_AUDIENCE=bddk-mcp-local
docker compose up --build -d bddk-bootstrap
docker compose wait bddk-bootstrap
export BDDK_DATABASE_URL=postgresql://bddk_local_public:local-only-public@localhost:5432/bddk
```

For loopback development only, Compose runs DBA role/extension setup → schema-owner `migrate` → DBA grants → ingestion `bootstrap`. The reserved `.invalid` JWT values only let Compose parse an unused HTTP service definition; this lifecycle target starts no HTTP server, and those values are not valid server configuration. Its fixed passwords are public test fixtures and must not be copied remotely. In production, `bddk-mcp migrate` performs schema work only. `bddk-mcp bootstrap` requires an already migrated and granted schema, then imports the reviewed seed, sections, and 768-dimensional embeddings; it does not migrate. See the [deployment guide](docs/DEPLOYMENT.md) for the complete identity and apply order.

Use the optional read-only preflight to inspect the corpus declaration and all
three seed artifacts without opening a database connection:

```bash
uv run --frozen bddk-mcp verify-corpus
```

The command checks checksums, sizes, record counts, and freshness timestamps,
but it does not transfer trust to a later process. The production import must
reapply the strict policies directly in the mutating `bootstrap` invocation:

```bash
BDDK_INGESTION_DATABASE_URL='postgresql://INGESTION:SECRET@HOST:5432/DATABASE?sslmode=verify-full&sslrootcert=%2FAPPROVED%2Fpostgres-ca.crt' \
  uv run --frozen bddk-mcp bootstrap \
    --seed-dir /APPROVED/CORPUS \
    --reindex-existing \
    --require-quantified-freshness \
    --require-measured-freshness \
    --require-verified-signature \
    --trusted-signing-key /APPROVED/TRUST/corpus-signing-public-key.pem
```

Before opening a database pool, `bootstrap` validates the exact
`corpus_scope.yml` and the manifest-declared artifact paths, bytes, and hashes;
it rejects a present but undeclared `documents.json`, `chunks.json`, or
`decision_cache.json`. Supply the trust key from a Secret/mount separate from
the corpus. Declaring numeric objectives is not measurement: `measured` status
requires a per-document authoritative-publication → source-detection → download
→ extraction → retrieval-publication timeline and calculated lags within those
objectives. The current reviewed manifest is unsigned, has no numeric
objectives, and is treated as `slo_evidence_status: not_measured`; it
intentionally fails this production bootstrap. Successful bootstrap output
includes the path-free manifest ID and SHA-256 for operator evidence and marks
publication as required; it does not persist a candidate. The separate
`bddk-mcp publish-corpus-release` step revalidates the corpus, then persists the
content-addressed release and its activation atomically through the
`bddk_release_publisher` identity.

Ordinary migration fails closed on a pre-ledger unmanaged database. `--adopt-legacy` is an explicit option only for the exact supported shape after a proven backup and the [legacy upgrade runbook](docs/LEGACY_DATABASE_UPGRADE.md); it is not a clean-install or general repair flag.

A populated version-2 database also refuses migration 3 by default because the retrieval-publication backfill takes blocking locks and validates foreign keys. Use `--allow-retrieval-publication-backfill` only in a controlled maintenance window after stopping workloads, proving a restorable backup, and rehearsing a size-matched restore. `BDDK_EXPECTED_DATABASE_NAME` and the independent DBA-script target must match the active database. Outside isolated local Compose, PostgreSQL DSNs must use `sslmode=verify-full` and an absolute `sslrootcert` path.

Test:

```bash
uv run pytest tests/test_tools_sections.py tests/test_doc_store.py -k section -v
uv run ruff check .
```

Run MCP over stdio:

```bash
BDDK_DATABASE_URL=postgresql://bddk:bddk@localhost:5432/bddk \
uv run --frozen bddk-mcp serve
```

Run streamable HTTP:

```bash
BDDK_DATABASE_URL=postgresql://bddk:bddk@localhost:5432/bddk \
MCP_TRANSPORT=streamable-http \
PORT=8000 \
uv run --frozen bddk-mcp serve
```

The Streamable HTTP MCP endpoint is `http://localhost:8000/mcp`, configured for stateless JSON responses. The remote application publishes RFC 9728 protected-resource metadata at `/.well-known/oauth-protected-resource/mcp`, and its 401 challenge identifies the same URL through `resource_metadata`. This is application-level MCP authorization discovery, not proof of bank IdP client registration or flow acceptance. The fixed, content-free probe endpoints are `GET /health/live` and `GET /health/ready`; readiness periodically re-attests migrations, critical catalog objects, corpus publication, and workload ACLs. Probes bypass authentication/Host checks but remain subject to process rate and concurrency admission. A non-loopback bind fails closed unless exact Host/HTTPS Origin allowlists and the complete JWT/JWKS configuration are supplied; the public profile requires `bddk.read`, while the operator profile requires `bddk.operator`. Remote operator HTTP also requires the explicit `BDDK_OPERATOR_REMOTE_ENABLED=true` opt-in. Body, concurrency, and per-minute rate controls are local to one application process and are not a shared ingress rate limit across replicas. See the [deployment guide](docs/DEPLOYMENT.md) for the complete contract.

The legacy seed import/export helper remains available; prefer `bddk-mcp bootstrap` for new deployments because it includes readiness validation:

```bash
BDDK_INGESTION_DATABASE_URL=postgresql://bddk_local_ingestion:local-only-ingestion@localhost:5432/bddk \
uv run --frozen bddk-seed import
```

### Claude Configuration

The repository [`.mcp.json`](.mcp.json) is a portable stdio example for `.mcp.json`-compatible clients that launch it with the repository root as the working directory:

```json
{
  "mcpServers": {
    "bddk": {
      "command": "uv",
      "args": ["run", "--frozen", "bddk-mcp"],
      "env": {
        "MCP_TRANSPORT": "stdio",
        "BDDK_DATABASE_URL": "${BDDK_DATABASE_URL}"
      }
    }
  }
}
```

### Codex Configuration

Codex CLI and the IDE extension share the Codex MCP configuration. Add the following to `~/.codex/config.toml` or `.codex/config.toml` in a trusted repository, replacing `cwd` with your checkout path:

```toml
[mcp_servers.bddk]
command = "uv"
args = ["run", "--frozen", "bddk-mcp"]
cwd = "/absolute/path/to/bddk-mcp"
env_vars = ["BDDK_DATABASE_URL"]
startup_timeout_sec = 30
tool_timeout_sec = 60
```

Verify the connection with `codex mcp list` or `/mcp` inside Codex.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for Docker, Railway, and OpenShift AI boundaries.

### Example Queries

```text
search_bddk_regulations(keywords="kredilerin siniflandirilmasi")
search_document_store(query="TFRS 9 kredi riskinde onemli artis")
get_bddk_document(document_id="mevzuat_22599", page_number=1)
get_document_section(document_id="943", section_type="ilke", section_ref="5")
search_document_sections(query="Karsilik Yonetmeligi Madde 9 TFRS 9")
get_bddk_bulletin(metric_id="1.0.1", currency="TRY", days=90)
analyze_bulletin_trends(metric_id="1.0.1", lookback_weeks=12)
get_regulatory_digest(period="week")
```

### Operator Workflows

Run a document quality scan:

```bash
uv run python scripts/scan_document_quality.py --db --out-dir quality_reports --allow-failures
```

Dry-run backfill for known quality failures:

```bash
uv run python scripts/backfill_quality_failures.py --dry-run
```

Re-extract one known failed document:

```bash
uv run python scripts/backfill_quality_failures.py --doc-id mevzuat_21192 --execute
```

Rebuild `document_sections` for existing stored documents with the ingestion identity:

```bash
BDDK_INGESTION_DATABASE_URL=postgresql://INGESTION:SECRET@HOST:5432/DB \
  uv run python scripts/reindex_document_sections.py --execute
```

Executed quality backfill, synchronization, and reindex scripts likewise require `BDDK_INGESTION_DATABASE_URL` and verify the exact `bddk_ingestion` privilege contract. Do not run them with the public or operator DSN.

Optional retrieval telemetry:

```bash
BDDK_DATABASE_URL=postgresql://PUBLIC:SECRET@HOST:5432/DB \
BDDK_TELEMETRY_ENABLED=true \
BDDK_TELEMETRY_DATABASE_URL=postgresql://TELEMETRY:SECRET@HOST:5432/DB \
  uv run --frozen bddk-mcp serve --profile public
```

Telemetry is disabled by default. When enabled, its distinct LOGIN must inherit only `bddk_telemetry_writer`; startup verifies the exact column-scoped INSERT-only contract and rejects trace reads/changes or broader membership. The server writes latency, result counts, document IDs, quality labels, and relevance summaries to `tool_call_traces`; query/prompt text is stored as a hash and length summary. Raw text is only stored when `BDDK_TELEMETRY_STORE_TEXT=true` is explicitly set.

### Architecture

```text
server.py                 Root shim → bddk_mcp/server.py
seed.py                   Root shim → bddk_mcp/ingest/seed.py
bddk_mcp/                 Main package
  server.py               FastMCP entry point and lifecycle
  core/                   configuration, DB identity, outbound HTTP, logging, models
  migrations/             Immutable global PostgreSQL migration ledger
  jobs/                   Durable operator-job models and PostgreSQL repository
  store/                  doc_store, vector_store, section_index, legal_ref
  ingest/                 client, data_sources, doc_sync, html_extractor, backfill, seed
  quality/                markdown_quality, quality_scan
  observability/          analytics, telemetry, metrics
  tools/                  MCP tool modules
  ocr/                    base, chandra (pluggable OCR)
scripts/                  Operator and backfill scripts
benchmark/                Tool schemas and benchmark infrastructure
```

### Data Quality And Safety Notes

- Full regulation-document and section retrieval responses are served from the local store; those paths do not live-fetch documents at runtime.
- Catalog refresh, institution/announcement search, and bulletin tools can access BDDK upstream services depending on configuration and cache state.
- Live regulatory HTTP paths enforce exact BDDK/mevzuat HTTPS hosts, redirect/DNS revalidation, and code-owned streaming limits by artifact type; retry logs omit URLs, query strings, and exception text. Because public institution, announcement, bulletin, and update tools can also call live BDDK sources, the OpenShift egress contract must grant approved regulatory-source or proxy TCP 443 to both public and operator runtimes, but not to lifecycle Jobs. NetworkPolicy or an approved proxy/firewall remains required because DNS validation cannot eliminate the DNS-to-connect race.
- The default embedding model is pinned to full commit `d13f1b27baf31030b7fd040960d60d909913633f`, the optional default reranker to `1427fd652930e4ba29e8149678df786c240d8825`, and the immutable schema accepts only `vector(768)`. A model/chunk-setting change requires controlled full re-embedding and retrieval regression testing.
- A retrieval publication record is written only after chunk integrity, the current content hash, and the active retrieval profile pass validation; incomplete or stale indexes are not silently mixed into search results.
- Bootstrap binds the reviewed corpus to the manifest's exact artifact paths and rejects reserved seed-filename bypasses. A separate `verify-corpus` run is diagnostic preflight only; production trust gates must be passed directly to the same `bootstrap` invocation with a separately mounted trust key. `deploy/openshift-overlays/bank-bootstrap` exact-inventory-checks that command, a read-only approved-corpus PVC, and a separate read-only corpus-trust Secret in repository preflight; actual bank provisioning and Job execution remain external gates.
- Migration v0005 persists an independently revalidated manifest/retrieval identity as an append-only release and activation. Mutation of any of 17 corpus tables advances a singleton epoch and invalidates the active view; strict local-corpus calls verify one release before and after execution. Migration v0007 additionally lets the distinct release-publisher identity run `bddk-mcp retain-corpus-generation --expected-release-id ...` to copy and seal the exact active state across 17 typed retained relations. Physical generation and seal identities are derived from corpus state and retrieval profile, so differently governed releases over the same exact state/profile receive distinct release bindings to the same retained generation and seal instead of duplicating storage. V7 makes retained-row and current/retained state hashing session-independent with function-local `TimeZone=UTC`, `DateStyle=ISO, YMD`, `IntervalStyle=postgres`, `bytea_output=hex`, and `extra_float_digits=3`. If an active pre-v7 release does not match the canonical recomputation, the v7 migration fails before replacing the hash function. On the unchanged pre-v7 schema (v5 or v6), the exact publication-only `publish-corpus-release` compatibility path reviews, publishes, and activates a new canonical release instead of rewriting or backfilling its historical row; then retry v7. Serving, retention, and every other workload remain v7-only. This administrative CLI is not an MCP tool and does not serve, activate, reactivate, or roll back retained data. Generation-bound serving and authorized rollback remain H2-02B; backup growth is `not_measured`, and bank retention/capacity approval is open. The tracked 8,286-chunk artifact currently differs from the 9,675 rows generated by profile v2, so the tracked corpus cannot pass strict publication.
- Migration v0004's 11 owner-only legal-curation tables separate source-content and acquisition identity; v0006 adds the public abstention-first `resolve_regulation_status` path. Synthetic real-PostgreSQL evidence does not establish a real regulation family's currentness.
- The evaluation gate requires four signed layers: measured corpus, expert dataset, legal-curator attestation over the exact Citation pack, and a legal-release checkpoint over retained source/acquisition/page/excerpt history. Canonical corpus/dataset/curator/release signer fingerprints must differ. Current preflight proves only cryptographic consistency under operator-supplied anchors; bank authorization and model-score authorization remain false. The 20-case set is draft, and key rotation, named-reviewer policy, and expert-case execution remain open.
- The supply-chain lane builds containers locally with Buildx `--provenance=false --load`; it fail-closed binds the manifest descriptor/digest, config digest, loaded image, and Syft SBOM to the same image. The repository separately creates unsigned SLSA provenance and verifies model-manifest/runtime/Dockerfile pin consistency. Any result that applies a pending exception is never promotion-eligible; bank signing, admission, and registry promotion remain external gates.
- Runtime wheels/sdists exclude `seed_data`, benchmark code, and deployment assets; the provided container explicitly includes the reviewed seed. A wheel deployment must mount an approved corpus and pass `--seed-dir` or `BDDK_SEED_DIR` to bootstrap.
- Low-quality extractions are marked as `warning` or `fail`.
- Formula-heavy or OCR-corrupted documents may require source PDF review.
- `get_bddk_document` removes data URIs, raw HTML, and selected OCR artifacts before model context.
- The model should answer only from tool output. It should not invent decision numbers, dates, or legal conclusions.
- See [docs/DOCUMENT_QUALITY.md](docs/DOCUMENT_QUALITY.md) for known extraction issues, the tracked fail list, and backfill commands.
- The target bank's OpenShift AI cluster, backup/restore process, and Claude/Codex/GPT/GPT-OSS/LM Studio/local-model client matrix have not yet completed acceptance testing with this repository.

### Development Commands

```bash
uv run pytest tests/ -v --tb=short
uv run ruff check .
uv run ruff format .
```

Focused checks used often in this project:

```bash
uv run pytest tests/test_markdown_quality.py tests/test_tools_documents.py -v
uv run pytest tests/test_legal_ref.py tests/test_section_index.py tests/test_tools_sections.py -v
uv run pytest tests/test_vector_store.py tests/test_legal_ref.py -v -rs
```

### License

The source code is distributed under the [MIT License](LICENSE). Regulatory-source documents and other third-party data may have separate provenance or reuse conditions; the code license does not grant additional rights over those materials. The confirmed boundary, unresolved decisions, and release gate are recorded in [Licensing and Provenance](docs/LICENSING_AND_PROVENANCE.md).
