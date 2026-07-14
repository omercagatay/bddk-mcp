# BDDK MCP Server

[Türkçe](#turkce) | [English](#english)

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

- **MCP uyumlu araçlar:** Claude, Codex ve MCP destekleyen istemcilerle çalışır.
- **Offline-first doküman retrieval:** Düzenleme metinleri ve bölümleri PostgreSQL/pgvector üzerinden servis edilir; kurum, duyuru ve bülten araçları upstream erişimi gerektirebilir.
- **Katalog ve gövde araması ayrımı:** `search_bddk_regulations` sadece başlık/metadata arar; `search_document_store` doküman gövdesinde semantik arama yapar.
- **Bölüm bazlı erişim:** `get_document_section` ve `search_document_sections` ile `943 İlke 5` veya `mevzuat_22599 Madde 9` gibi referanslar doğrudan bulunur.
- **Exact legal-reference koruması:** `Madde 9` gibi lexical eşleşmeler, semantik skor düşük olsa bile korunur.
- **Kalite etiketleri:** Doküman çıktıları `clean`, `warning`, `fail` sinyalleri ve kalite bayraklarıyla işaretlenir.
- **Doküman context sanitization:** `get_bddk_document`, Data URI, raw HTML/OCR artefact ve uzun satırları model context'ine verilmeden temizler.
- **Operatör scriptleri:** kalite tarama, kalite backfill ve `document_sections` reindex akışları mevcuttur.
- **PostgreSQL + pgvector:** dokümanlar, bölümler, FTS ve vektör arama tek veritabanı üzerinde çalışır.

### Araç Yüzeyi

Varsayılan public deployment `BDDK_ADMIN_TOOLS=false` ile 15 public araç expose eder.

| Modül | Araçlar |
|---|---|
| Arama | `search_bddk_regulations`, `search_document_store`, `search_bddk_institutions`, `search_bddk_announcements` |
| Doküman | `get_bddk_document`, `get_document_history` |
| Bölümler | `get_document_section`, `search_document_sections` |
| Bülten | `get_bddk_bulletin`, `get_bddk_bulletin_snapshot`, `get_bddk_monthly` |
| Analitik | `analyze_bulletin_trends`, `get_regulatory_digest`, `compare_bulletin_metrics`, `check_bddk_updates` |

`BDDK_ADMIN_TOOLS=true` ile 11 ek operatör aracı açılır. Admin/operator deployment toplam 26 araç expose eder: 15 public araç + 11 operatör aracı.

- `document_store_stats`
- `bddk_cache_status`
- `refresh_bddk_cache`
- `sync_bddk_documents`
- `trigger_startup_sync`
- `document_health`
- `health_check`
- `bddk_metrics`
- `backfill_degraded_documents`
- `backfill_status`
- `document_quality_report`

Geçerli runtime için toplam olası MCP araç sayısı 26'dır. Benchmark şemaları aynı canonical operatör registry'sinden üretilir; benchmark koşuları yine de kullandıkları exact tool listesini ve profili kaydetmelidir. Bkz. [benchmark/README.md](benchmark/README.md).

### Hızlı Başlangıç

Gereksinimler:

- Python 3.12 veya 3.13
- `uv`
- PostgreSQL 14+ ve `pgvector`
- Opsiyonel: Docker Compose

Kurulum:

```bash
git clone https://github.com/omercagatay/bddk-mcp.git
cd bddk-mcp
uv sync
```

Lokal PostgreSQL:

```bash
docker compose up -d db
export BDDK_DATABASE_URL=postgresql://bddk:bddk@localhost:5432/bddk
uv run --frozen bddk-mcp bootstrap
```

`bootstrap`, schema migration, reviewed seed import, section index ve embedding backfill işlemlerini açık bir operatör adımı olarak çalıştırır. `serve` bu işlemleri otomatik yapmaz ve hazır olmayan veritabanında açıklayıcı bir hata ile durur.

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

Streamable HTTP MCP endpoint'i `http://localhost:8000/mcp` olur. Mevcut sunucuda uygulama-seviyesi kimlik doğrulama veya rate limiting yoktur; güvenlik katmanı eklenmeden güvenilmeyen bir ağa açmayın.

Eski seed import/export yardımcı komutu da korunur; yeni deployment'larda doğrulama içeren `bddk-mcp bootstrap` tercih edilir:

```bash
BDDK_DATABASE_URL=postgresql://bddk:bddk@localhost:5432/bddk \
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

Mevcut dokümanlardan `document_sections` tablosunu yeniden oluşturma:

```bash
uv run python scripts/reindex_document_sections.py --execute
```

Railway production üzerinde tek seferlik section reindex:

```bash
railway run --service Postgres --environment production \
  sh -c 'uv run python scripts/reindex_document_sections.py --database-url "$DATABASE_PUBLIC_URL" --execute'
```

Opsiyonel retrieval telemetry:

```bash
BDDK_TELEMETRY_ENABLED=true uv run --frozen bddk-mcp
```

Telemetry varsayılan olarak kapalıdır. Açıldığında `tool_call_traces` tablosuna latency, result count, doc ID, kalite etiketi ve relevance özeti yazar; query/prompt metni hash/uzunluk özeti olarak saklanır. Raw metin yalnızca `BDDK_TELEMETRY_STORE_TEXT=true` açıkça set edilirse yazılır.

### Mimari

```text
server.py                 Kök shim → bddk_mcp/server.py
seed.py                   Kök shim → bddk_mcp/ingest/seed.py
bddk_mcp/                 Ana paket
  server.py               FastMCP giriş noktası ve lifecycle
  core/                   config, deps, exceptions, logging_config, models, utils
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
- Kalitesi düşük extraction çıktıları `warning` veya `fail` olarak işaretlenir.
- Formül ağır veya OCR bozuk dokümanlarda kaynak PDF incelemesi gerekebilir.
- `get_bddk_document` cevaplarında data URI, raw HTML ve bazı OCR artefact'ları temizlenir.
- Model cevap verirken sadece tool çıktısına dayanmalıdır; karar numarası, tarih veya hukuki sonuç uydurulmamalıdır.
- Bilinen extraction sorunları, fail doküman listesi ve backfill komutları için [docs/DOCUMENT_QUALITY.md](docs/DOCUMENT_QUALITY.md) sayfasına bakın.

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

- **MCP-compatible tools:** works with Claude, Codex, and other MCP clients.
- **Offline-first document retrieval:** regulation text and sections are served from PostgreSQL/pgvector; institution, announcement, and bulletin tools may require upstream access.
- **Catalog/body separation:** `search_bddk_regulations` searches metadata; `search_document_store` searches document bodies.
- **Section-level retrieval:** `get_document_section` and `search_document_sections` support references like `943 Ilke 5` and `mevzuat_22599 Madde 9`.
- **Exact legal-reference preservation:** lexical hits such as `Madde 9` survive dense relevance filtering.
- **Quality labels:** document outputs include `clean`, `warning`, or `fail` metadata and quality flags.
- **Document-context sanitization:** `get_bddk_document` removes data URIs, raw HTML/OCR artifacts, and pathological long lines before model context.
- **Operator scripts:** quality scan, quality backfill, and `document_sections` reindex workflows are included.
- **PostgreSQL + pgvector:** documents, sections, FTS, and vector search share one database.

### Tool Surface

The default public deployment with `BDDK_ADMIN_TOOLS=false` exposes 15 public tools.

| Module | Tools |
|---|---|
| Search | `search_bddk_regulations`, `search_document_store`, `search_bddk_institutions`, `search_bddk_announcements` |
| Documents | `get_bddk_document`, `get_document_history` |
| Sections | `get_document_section`, `search_document_sections` |
| Bulletin | `get_bddk_bulletin`, `get_bddk_bulletin_snapshot`, `get_bddk_monthly` |
| Analytics | `analyze_bulletin_trends`, `get_regulatory_digest`, `compare_bulletin_metrics`, `check_bddk_updates` |

With `BDDK_ADMIN_TOOLS=true`, 11 additional operator tools are exposed. The admin/operator deployment exposes 26 tools total: 15 public tools plus 11 operator tools.

- `document_store_stats`
- `bddk_cache_status`
- `refresh_bddk_cache`
- `sync_bddk_documents`
- `trigger_startup_sync`
- `document_health`
- `health_check`
- `bddk_metrics`
- `backfill_degraded_documents`
- `backfill_status`
- `document_quality_report`

The current runtime has 26 possible MCP tools. Benchmark schemas are exported from the same canonical operator registry; benchmark runs should still record the exact tool list and profile they used. See [benchmark/README.md](benchmark/README.md).

### Quick Start

Requirements:

- Python 3.12 or 3.13
- `uv`
- PostgreSQL 14+ with `pgvector`
- Optional: Docker Compose

Install:

```bash
git clone https://github.com/omercagatay/bddk-mcp.git
cd bddk-mcp
uv sync
```

Local PostgreSQL:

```bash
docker compose up -d db
export BDDK_DATABASE_URL=postgresql://bddk:bddk@localhost:5432/bddk
uv run --frozen bddk-mcp bootstrap
```

`bootstrap` explicitly runs schema migration, reviewed seed import, section indexing, and embedding backfill. `serve` never performs these lifecycle writes and exits with an actionable error when the database is not ready.

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

The Streamable HTTP MCP endpoint is `http://localhost:8000/mcp`. The current server has no application-level authentication or rate limiting; do not expose it to an untrusted network until a security layer is added.

The legacy seed import/export helper remains available; prefer `bddk-mcp bootstrap` for new deployments because it includes readiness validation:

```bash
BDDK_DATABASE_URL=postgresql://bddk:bddk@localhost:5432/bddk \
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

Rebuild `document_sections` for existing stored documents:

```bash
uv run python scripts/reindex_document_sections.py --execute
```

One-off section reindex on Railway production:

```bash
railway run --service Postgres --environment production \
  sh -c 'uv run python scripts/reindex_document_sections.py --database-url "$DATABASE_PUBLIC_URL" --execute'
```

Optional retrieval telemetry:

```bash
BDDK_TELEMETRY_ENABLED=true uv run --frozen bddk-mcp
```

Telemetry is disabled by default. When enabled, the server writes latency, result counts, document IDs, quality labels, and relevance summaries to `tool_call_traces`; query/prompt text is stored as a hash and length summary. Raw text is only stored when `BDDK_TELEMETRY_STORE_TEXT=true` is explicitly set.

### Architecture

```text
server.py                 Root shim → bddk_mcp/server.py
seed.py                   Root shim → bddk_mcp/ingest/seed.py
bddk_mcp/                 Main package
  server.py               FastMCP entry point and lifecycle
  core/                   config, deps, exceptions, logging_config, models, utils
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
- Low-quality extractions are marked as `warning` or `fail`.
- Formula-heavy or OCR-corrupted documents may require source PDF review.
- `get_bddk_document` removes data URIs, raw HTML, and selected OCR artifacts before model context.
- The model should answer only from tool output. It should not invent decision numbers, dates, or legal conclusions.
- See [docs/DOCUMENT_QUALITY.md](docs/DOCUMENT_QUALITY.md) for known extraction issues, the tracked fail list, and backfill commands.

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

The source code is distributed under the [MIT License](LICENSE). Regulatory-source documents and other third-party data may have separate provenance or reuse conditions; the code license does not grant additional rights over those materials.
