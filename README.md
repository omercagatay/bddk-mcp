# BDDK MCP Server

[Türkçe](#tr) | [English](#en)

---

<a name="tr"></a>

## TR — Türkçe

BDDK (Bankacılık Düzenleme ve Denetleme Kurumu) karar ve düzenlemelerini arama, erişim ve analiz için [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) sunucusu.

Banka ortamları için **çevrimdışı öncelikli** tasarım — harici API bağımlılığı yok, tüm veriler doğrudan BDDK'dan çekiliyor, embedding modeli derleme zamanında önceden indiriliyor.

### Özellikler

**Arama ve Erişim**
- 1000+ BDDK dokümanında Türkçe destekli anahtar kelime araması ve basit kök bulma
- pgvector ile semantik arama (multilingual-e5-base, hibrit yoğun+seyrek RRF füzyonu)
- 12 kategoriye göre filtreleme (kapsam filtresi sonrası), tarih aralığı filtreleme
- Sayfalanmış Markdown olarak doküman erişimi (BDDK ve mevzuat.gov.tr)
- Doküman versiyonlama — düzenleme değişikliklerini takip

**Veri ve Analitik**
- 340+ kuruluş rehberi (bankalar, leasing, faktoring, finansman, varlık yönetimi)
- Haftalık bülten verileri — zaman serisi ve son durum
- Aylık istatistikler — 17 tabloda detaylı bankacılık verileri
- Trend analizi — haftalık değişimler, Türkçe anlatım
- Düzenleyici özet — karar, duyuru ve bülten verilerini birleştiren yönetici raporu
- Metrik karşılaştırma, güncelleme tespiti, basın ve düzenleme duyuruları

**Altyapı**
- PostgreSQL + pgvector — tek veritabanında FTS5, vektör araması ve doküman depolama
- Modüler mimari — 21 MCP aracı, 6 araç modülüne ayrılmış, bağımlılık enjeksiyonu
- Devre kesici ve senkronizasyon hata takibi — kalıcı `sync_failures` tablosu
- Çevrimdışı öncelikli — embedding modeli derleme zamanında indiriliyor
- 2 katmanlı doküman çıkarma (LightOnOCR-2-1B GPU birincil, markitdown CPU yedek); `scripts/backfill_mevzuat.py --use-chandra` ile backfill yolunda Chandra2 öne alınabilir
- mevzuat.gov.tr için 4 katmanlı indirme (.htm > .pdf > iframe > .doc)
- Otomatik senkronizasyon, arkaplan vektör deposu başlatma, devre kesici
- GitHub Actions CI (lint + test), 305 test

### Kapsam (Konvansiyonel Banka Odağı)

Katalog, konvansiyonel (mevduat) banka kullanımı için varsayılan olarak daraltılmıştır. Aşağıdakiler hem canlı scrape'te hem de seed yüklemesinde dışlanır:

- `Faizsiz Bankacılık` kategorisi (katılım bankacılığı düzenlemeleri)
- `Finansal Kiralama ve Faktoring` kategorisi (Mevzuat Liste/52) — finansal kiralama, faktoring ve finansman şirketleri banka kanunu kapsamı dışındadır
- Başlığında "6361 sayılı" geçen kararlar (Finansal Kiralama / Faktoring / Finansman Kanunu)
- Mevzuat Liste/55 firma-özel Kurul Kararları (faaliyet/kuruluş izni gürültüsü); esas politika kararları Liste/56'da yer alır ve korunur (örn. zamanaşımına uğrayan mevduat, SYR düzenlemeleri)

Kuruluş rehberi (bankalar + leasing/faktoring/finansman/varlık yönetim şirketleri) bu filtrenin dışındadır — bir bankanın karşı taraflarını tanıması gerektiği için kuruluş araması aynen erişilebilir.

Filtre `client.py` içindeki `_is_in_scope` fonksiyonunda uygulanır ve idempotent'tir; temiz bir katalog için yalnızca üretim DB'sini reseed etmek yeterlidir.

### Araçlar

| Modül | Araçlar |
|---|---|
| **Arama** | `search_bddk_decisions`, `search_document_store`, `search_bddk_institutions`, `search_bddk_announcements` |
| **Doküman** | `get_bddk_document`, `get_document_history`, `document_store_stats` |
| **Bülten** | `get_bddk_bulletin`, `get_bddk_bulletin_snapshot`, `get_bddk_monthly`, `compare_bulletin_metrics` |
| **Analitik** | `analyze_bulletin_trends`, `get_regulatory_digest`, `check_bddk_updates`, `bddk_cache_status` |
| **Senkronizasyon** | `refresh_bddk_cache`, `sync_bddk_documents`, `trigger_startup_sync`, `document_health` |
| **Yönetim** | `health_check`, `bddk_metrics` |

### Kurulum

**Gereksinimler:** Python 3.11–3.13, [uv](https://docs.astral.sh/uv/), PostgreSQL (pgvector eklentisi ile)

**Claude Code** — `~/.claude/settings.json` dosyasına ekleyin:

```json
{
  "mcpServers": {
    "bddk": {
      "command": "uv",
      "args": [
        "run", "--directory", "/path/to/bddk-mcp",
        "--python", "3.12", "mcp", "run", "server.py"
      ]
    }
  }
}
```

**Claude Desktop** — `claude_desktop_config.json` dosyasına aynı yapılandırmayı ekleyin.

**Railway** — Repo'yu bağlayın, otomatik derleme. Dockerfile embedding modelini derleme zamanında indirir (~1GB), `TRANSFORMERS_OFFLINE=1` ile çalışma zamanında ağ bağımlılığı yoktur.

### Kullanım Örnekleri

```
# Sermaye yeterliliği düzenlemelerini ara
search_bddk_decisions(keywords="sermaye yeterliliği")

# Semantik arama
search_document_store(query="faiz oranı riski nasıl hesaplanır")

# Belge getir
get_bddk_document(document_id="1296")

# Haftalık bülten
get_bddk_bulletin(metric_id="1.0.1", currency="TRY", days=90)

# Trend analizi
analyze_bulletin_trends(metric_id="1.0.1", lookback_weeks=12)

# Düzenleyici özet (period: "day" | "week" | "month" | "quarter")
get_regulatory_digest(period="day")

# Senkronizasyon durumu
document_health()
```

---

<a name="en"></a>

## EN — English

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server for searching, retrieving, and analyzing decisions and regulations from **BDDK** (Banking Regulation and Supervision Agency of Turkey).

Designed for **offline-first** operation in bank environments — no external API dependencies, all data scraped directly from BDDK, embedding model pre-downloaded at build time.

### Features

**Search & Retrieval**
- Search across 1000+ BDDK documents with Turkish-aware keyword matching and basic stemming
- Semantic search via pgvector with multilingual-e5-base embeddings (hybrid dense+sparse RRF fusion)
- Category filtering (12 categories, post-scope-filter), date range filtering
- Document retrieval as paginated Markdown (BDDK and mevzuat.gov.tr)
- Document versioning — track regulation changes over time

**Data & Analytics**
- Institution directory — 340+ institutions (banks, leasing, factoring, finance, asset management)
- Weekly bulletin data — banking sector metrics with time-series and latest snapshot
- Monthly statistics — detailed banking sector data across 17 tables
- Trend analysis — week-over-week changes with Turkish-language narratives
- Regulatory digest — executive summary combining decisions, announcements, and bulletin data
- Metric comparison, update detection, press releases and regulation announcements

**Infrastructure**
- PostgreSQL + pgvector — single database for FTS, vector search, and document storage
- Modular architecture — 21 MCP tools split across 6 tool modules with dependency injection
- Circuit breaker and sync failure tracking — persistent `sync_failures` table with error categorization
- Offline-first — embedding model pre-downloaded at build time, no runtime network dependency for ML
- 2-layer document extraction (LightOnOCR-2-1B GPU primary, markitdown CPU fallback); Chandra2 optionally prepended on the backfill path via `scripts/backfill_mevzuat.py --use-chandra`
- 4-layer mevzuat.gov.tr download fallback (.htm > .pdf > iframe > .doc)
- Auto-sync, background vector store init, circuit breaker for sync reliability
- GitHub Actions CI (lint + test), 305 tests

### Scope (Conventional-Bank Default)

The catalog is scoped by default to conventional (deposit-taking) bank use. The following are excluded from both live scrape and seed load:

- `Faizsiz Bankacılık` category (participation / Islamic-banking regulations)
- `Finansal Kiralama ve Faktoring` category (Mevzuat Liste/52) — leasing, factoring, and financing companies fall outside the Banking Law
- Decisions whose title contains "6361 sayılı" (Financial Leasing / Factoring / Financing Law)
- Mevzuat Liste/55 firm-specific Kurul Kararları (activity/establishment-permit noise); substantive policy decisions live on Liste/56 and are kept in full (e.g. deposit statute-of-limitations, capital-adequacy rulings)

The institution directory (banks + leasing/factoring/financing/asset-management firms) is **not** filtered — a bank still needs to look up counterparties, so institution search works unchanged.

The filter lives in `_is_in_scope` (`client.py`) and is idempotent; reseeding production is enough to land a clean catalog.

### Tools

| Module | Tools |
|---|---|
| **Search** | `search_bddk_decisions`, `search_document_store`, `search_bddk_institutions`, `search_bddk_announcements` |
| **Documents** | `get_bddk_document`, `get_document_history`, `document_store_stats` |
| **Bulletin** | `get_bddk_bulletin`, `get_bddk_bulletin_snapshot`, `get_bddk_monthly`, `compare_bulletin_metrics` |
| **Analytics** | `analyze_bulletin_trends`, `get_regulatory_digest`, `check_bddk_updates`, `bddk_cache_status` |
| **Sync** | `refresh_bddk_cache`, `sync_bddk_documents`, `trigger_startup_sync`, `document_health` |
| **Admin** | `health_check`, `bddk_metrics` |

### Architecture

```
server.py (FastMCP entry point, lifecycle management)
  ├── deps.py (Dependencies dataclass — shared state via DI)
  ├── config.py (unified configuration, env-var overrides, validators)
  ├── tools/
  │   ├── search.py (keyword + semantic + institution + announcement search)
  │   ├── documents.py (document retrieval, history, stats)
  │   ├── bulletin.py (weekly/monthly bulletin, metric comparison)
  │   ├── analytics.py (trends, digest, updates, cache status)
  │   ├── sync.py (document sync, cache refresh, health check)
  │   └── admin.py (health check, performance metrics)
  ├── client.py (HTTP scraper, PostgreSQL cache, Turkish NLP, search ranking)
  ├── data_sources.py (institutions, bulletins, announcements)
  ├── analytics.py (trend analysis, digest, comparison)
  ├── doc_store.py (PostgreSQL document store + FTS + versioning + sync failures)
  ├── vector_store.py (pgvector + multilingual-e5-base, hybrid search, re-ranking)
  ├── doc_sync.py (download + extraction pipeline with error categorization)
  ├── seed.py (seed data import for fresh deployments)
  ├── exceptions.py (custom exception hierarchy)
  ├── logging_config.py (structured JSON logging)
  └── metrics.py (request/latency/cache tracking)
```

### Storage

- **PostgreSQL + pgvector** — single database for document storage, full-text search (tsvector), vector embeddings (HNSW index), decision cache, and sync failure tracking
- **In-memory LRU cache** — O(1) eviction for search results (configurable TTL)
- **Stale cache fallback** — serves expired cache when BDDK is unreachable

### Extraction Pipeline

Documents are downloaded and converted to Markdown via an ordered backend chain (`ocr_backends.get_default_backends`):

1. **LightOnOCR-2-1B** (GPU, primary) — formula-aware, requires CUDA (`gpu` group)
2. **markitdown** (CPU fallback) — lightweight PDF/DOCX extraction, no formulas

For the backfill path (`scripts/backfill_mevzuat.py`), pass `--use-chandra` to prepend Chandra2 (also GPU) as the primary when the heavier model is available.

For mevzuat.gov.tr, a 4-layer download fallback is used: `.htm` > `.pdf` > iframe > `.doc`

Sync failures are tracked persistently with error categorization (robots_txt, timeout, extraction, download, connection) and retryability flags.

### Setup

#### Prerequisites

- Python 3.11–3.13
- [uv](https://docs.astral.sh/uv/)
- PostgreSQL with pgvector extension

#### Claude Code

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "bddk": {
      "command": "uv",
      "args": [
        "run", "--directory", "/path/to/bddk-mcp",
        "--python", "3.12", "mcp", "run", "server.py"
      ]
    }
  }
}
```

#### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or equivalent:

```json
{
  "mcpServers": {
    "bddk": {
      "command": "uv",
      "args": [
        "run", "--directory", "/path/to/bddk-mcp",
        "--python", "3.12", "mcp", "run", "server.py"
      ]
    }
  }
}
```

Replace `/path/to/bddk-mcp` with the actual path to this repository.

#### Railway Deployment

1. Connect this repo to a Railway project
2. Add a PostgreSQL service with pgvector
3. Railway auto-builds the Docker image (downloads the ~1GB embedding model at build time)
4. At runtime, the container is fully self-contained — no network dependency for ML

The Dockerfile sets `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1` to prevent any runtime model downloads.

### Environment Variables

All configuration is centralized in `config.py` and overridable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `MCP_TRANSPORT` | `stdio` | Transport: `stdio` (local) or `streamable-http` (Railway) |
| `PORT` | `8000` | Server port (streamable-http only) |
| `BDDK_DATABASE_URL` | _(required)_ | PostgreSQL connection string — server/seed/doc_sync refuse to start if unset |
| `BDDK_PG_POOL_MIN` | `2` | Minimum pool connections |
| `BDDK_PG_POOL_MAX` | `10` | Maximum pool connections |
| `BDDK_AUTO_SYNC` | `false` | Auto-sync documents on startup |
| `BDDK_EMBEDDING_MODEL_PATH` | `""` | Pre-downloaded model path (air-gapped) |
| `BDDK_EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Embedding model name |
| `BDDK_EMBEDDING_DIM` | `768` | Embedding dimension |
| `BDDK_PAGE_SIZE` | `5000` | Characters per page in paginated output |
| `BDDK_EMBEDDING_CHUNK_SIZE` | `1000` | Characters per embedding chunk |
| `BDDK_EMBEDDING_CHUNK_OVERLAP` | `200` | Overlap between embedding chunks |
| `BDDK_CACHE_TTL` | `3600` | Decision list cache TTL (seconds) |
| `BDDK_SEARCH_CACHE_TTL` | `300` | Search result cache TTL (seconds) |
| `BDDK_SEARCH_CACHE_MAX` | `200` | Maximum cached search results |
| `BDDK_STALE_CACHE_FALLBACK` | `true` | Serve expired cache when BDDK unreachable |
| `BDDK_HYBRID_SEARCH` | `true` | Enable hybrid dense+sparse search |
| `BDDK_RRF_K` | `60` | RRF fusion constant |
| `BDDK_RERANKER` | `false` | Enable cross-encoder re-ranking |
| `BDDK_REQUEST_TIMEOUT` | `60` | HTTP request timeout (seconds) |
| `BDDK_MAX_RETRIES` | `3` | Maximum HTTP retry attempts |
| `BDDK_SYNC_CONCURRENCY` | `5` | Parallel document downloads |
| `BDDK_LIGHTOCR_MODEL_PATH` | `""` | Pre-downloaded LightOnOCR model path (air-gapped) |
| `BDDK_LIGHTOCR_MODEL` | `lightonai/LightOnOCR-2-1B` | Primary OCR model name |
| `BDDK_LIGHTOCR_DEVICE` | `auto` | OCR device: `auto`, `cuda`, or `cpu` |
| `BDDK_OCR_MIN_CONTENT_LEN` | `500` | Minimum characters required for OCR output to count as successful |
| `BDDK_CHANDRA_MODEL` | `datalab-to/chandra-ocr-2` | Chandra2 model name (backfill opt-in) |

### Install & Test

```bash
uv sync
uv run pytest tests/ -v
```

### Available Categories

Kapsam filtresi sonrası katalog (318 karar). `Faizsiz Bankacılık` ve `Finansal Kiralama ve Faktoring` kategorileri [Scope](#scope-conventional-bank-default) bölümünde açıklandığı gibi dışlandı.

| Kategori | Açıklama | Sayı |
|---|---|---|
| Kurul Kararı | Kurul Kararları (Liste/56, esas politika kararları) | 190 |
| Yönetmelik | Yönetmelikler | 39 |
| Rehber | Rehberler | 19 |
| Genelge | Genelgeler | 13 |
| Düzenleme Taslağı | Düzenleme Taslakları | 11 |
| Sermaye Yeterliliği | Sermaye Yeterliliği Tebliğ ve Rehberleri | 10 |
| Bilgi Sistemleri | BT ve İş Süreç Düzenlemeleri | 8 |
| BDDK Düzenlemesi | BDDK İç Düzenlemeleri | 7 |
| Mülga Düzenleme | Yürürlükten Kaldırılmış Düzenlemeler | 7 |
| Tebliğ | Tebliğler | 7 |
| Tekdüzen Hesap Planı | Tekdüzen Hesap Planları | 4 |
| Kanun | Kanunlar | 3 |

### Data Sources

#### Regulations (Mevzuat)

| Sayfa | URL | İçerik | Durum |
|---|---|---|---|
| 49 | bddk.org.tr/Mevzuat/Liste/49 | Kanunlar | ✓ |
| 50 | bddk.org.tr/Mevzuat/Liste/50 | Bankacılık Kanunu düzenlemeleri | ✓ |
| 51 | bddk.org.tr/Mevzuat/Liste/51 | Banka ve Kredi Kartı düzenlemeleri | ✓ |
| 52 | bddk.org.tr/Mevzuat/Liste/52 | Kiralama, Faktoring, Finansman düzenlemeleri | ✗ kapsam dışı |
| 54 | bddk.org.tr/Mevzuat/Liste/54 | BDDK iç düzenlemeleri | ✓ |
| 55 | bddk.org.tr/Mevzuat/Liste/55 | Kurul Kararları (firma-özel faaliyet/kuruluş izni) | ✗ kapsam dışı |
| 56 | bddk.org.tr/Mevzuat/Liste/56 | Kurul Kararları (esas politika kararları) | ✓ |
| 58 | bddk.org.tr/Mevzuat/Liste/58 | Düzenleme taslakları | ✓ |
| 63 | bddk.org.tr/Mevzuat/Liste/63 | Mülga düzenlemeler | ✓ |

#### Institutions (Kuruluşlar)

| Sayfa | URL | İçerik |
|---|---|---|
| 77 | bddk.org.tr/Kurulus/Liste/77 | Bankalar (67) |
| 78 | bddk.org.tr/Kurulus/Liste/78 | Finansal Kiralama Şirketleri (86) |
| 79 | bddk.org.tr/Kurulus/Liste/79 | Faktoring Şirketleri (118) |
| 80 | bddk.org.tr/Kurulus/Liste/80 | Finansman Şirketleri (29) |
| 82 | bddk.org.tr/Kurulus/Liste/82 | Varlık Yönetim Şirketleri (44) |

#### Bulletins (Bültenler)

| Kaynak | URL | İçerik |
|---|---|---|
| Haftalık Bülten | bddk.org.tr/bultenhaftalik | Bankacılık sektörü metrikleri |
| Aylık Bülten | bddk.org.tr/BultenAylik | Detaylı aylık istatistikler (17 tablo) |

## License

MIT
