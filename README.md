# BDDK MCP Server

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server for searching and retrieving decisions and regulations from **BDDK** (Banking Regulation and Supervision Agency of Turkey).

## Features

### Search & Retrieval
- **Search** across 1000+ BDDK documents with Turkish-aware keyword matching and basic stemming
- **Semantic search** via ChromaDB with multilingual-e5-base embeddings (~30ms for 1000+ docs)
- **Category filtering** by regulation type (14 categories)
- **Date range filtering** for board decisions
- **Relevance ranking** (title match > stem match > substring match)
- **Document retrieval** as paginated Markdown (BDDK and mevzuat.gov.tr)
- **Document versioning** — track regulation changes over time

### Data & Analytics
- **Institution directory** — search 340+ institutions (banks, leasing, factoring, finance, asset management)
- **Weekly bulletin data** — banking sector metrics with time-series and latest snapshot
- **Monthly statistics** — detailed banking sector data by table, period, and bank group
- **Trend analysis** — week-over-week changes with Turkish-language narratives
- **Regulatory digest** — executive summary combining decisions, announcements, and bulletin data
- **Metric comparison** — side-by-side comparison of multiple banking metrics
- **Update detection** — monitor for new BDDK announcements
- **Announcements** — press releases, regulation notices, HR and data publications

### Infrastructure
- **Dual storage** — SQLite + FTS5 for full-text search, ChromaDB for semantic search
- **Document sync** — bulk download with 3 extraction methods (Nougat GPU, markitdown, HTML)
- **Incremental sync** — etag/last-modified tracking per document
- **Persistent caching** to disk with 1-hour TTL
- **Structured JSON logging** with correlation IDs
- **Health monitoring** — server health check and performance metrics
- **Graceful shutdown** — flush WAL, close stores on SIGTERM
- **Rate limiting** — semaphore-based throttling for outbound requests
- **CI/CD** — GitHub Actions with lint + test matrix (Python 3.11/3.12)
- **Retry with backoff** for resilient HTTP fetching
- **Custom exception hierarchy** with specific error types

### Available Categories

| Category | Description | Count |
|---|---|---|
| Kurul Karari | Board Decisions (published & unpublished) | ~957 |
| Yonetmelik | Regulations | 39 |
| Rehber | Guidelines | 19 |
| Genelge | Circulars | 13 |
| Duzenleme Taslagi | Regulation Drafts | 11 |
| Sermaye Yeterliligi | Capital Adequacy Communiques & Guidelines | 10 |
| Bilgi Sistemleri | IT & Business Process Regulations | 8 |
| Finansal Kiralama ve Faktoring | Leasing & Factoring Regulations | 7 |
| BDDK Duzenlemesi | BDDK Internal Regulations | 7 |
| Mulga Duzenleme | Repealed Regulations | 7 |
| Teblig | Communiques | 6 |
| Kanun | Laws | 4 |
| Tekduzen Hesap Plani | Uniform Chart of Accounts | 4 |
| Faizsiz Bankacilik | Islamic Banking Regulations | 2 |

## Tools

### Search & Retrieval

#### `search_bddk_decisions`

Search for BDDK decisions and regulations by keyword.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `keywords` | `str` | required | Search terms in Turkish |
| `page` | `int` | `1` | Page number |
| `page_size` | `int` | `10` | Results per page (max 50) |
| `category` | `str \| None` | `None` | Category filter |
| `date_from` | `str \| None` | `None` | Start date (DD.MM.YYYY) |
| `date_to` | `str \| None` | `None` | End date (DD.MM.YYYY) |

#### `search_document_store`

Semantic search across all BDDK documents using vector embeddings.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `query` | `str` | required | Natural language query in Turkish |
| `category` | `str \| None` | `None` | Optional category filter |
| `limit` | `int` | `10` | Maximum results to return |

#### `get_bddk_document`

Retrieve a BDDK document as paginated Markdown. Uses ChromaDB first, falls back to SQLite, then live fetch.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `document_id` | `str` | required | Document ID from search results |
| `page_number` | `int` | `1` | Page of the Markdown output (5000 chars/page) |

#### `get_document_history`

Get version history for a document — shows all previous versions with timestamps.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `document_id` | `str` | required | Document ID |

#### `bddk_cache_status`

Show cache statistics: total items, age, categories, and any page errors.

### Institutions

#### `search_bddk_institutions`

Search the BDDK institution directory (banks, leasing, factoring, etc.).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `keywords` | `str` | `""` | Search terms (e.g. "Ziraat", "Garanti") |
| `institution_type` | `str \| None` | `None` | Filter by type: Banka, Finansal Kiralama Sirketi, Faktoring Sirketi, Finansman Sirketi, Varlik Yonetim Sirketi |
| `active_only` | `bool` | `True` | Only show active institutions |

### Bulletin & Analytics

#### `get_bddk_bulletin`

Get weekly banking sector bulletin time-series data.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metric_id` | `str` | `"1.0.1"` | Metric ID (e.g. 1.0.1=Total Loans, 1.0.2=Consumer Loans) |
| `currency` | `str` | `"TRY"` | TRY or USD |
| `column` | `str` | `"1"` | 1=TP (TL), 2=YP (Foreign Currency), 3=Total |
| `date` | `str` | `""` | Specific date (DD.MM.YYYY), empty for latest |
| `days` | `int` | `90` | Number of days of history |

#### `get_bddk_bulletin_snapshot`

Get the latest weekly bulletin snapshot — all metrics with current TP/YP values.

#### `get_bddk_monthly`

Get detailed monthly banking sector statistics.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `table_no` | `int` | `1` | Table number (1-17). 1=Assets, 2=Loans, 4=Deposits, 9=Capital Adequacy, 11=Income |
| `year` | `int` | `2025` | Year |
| `month` | `int` | `12` | Month (1-12) |
| `currency` | `str` | `"TL"` | TL or USD |
| `party_code` | `str` | `"10001"` | Bank group code. 10001=Sector, 10002=Deposit Banks, 10004=Participation Banks |

#### `analyze_bulletin_trends`

Analyze trends in weekly bulletin data with week-over-week changes and Turkish-language narrative.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metric_id` | `str` | `"1.0.1"` | Metric ID |
| `currency` | `str` | `"TRY"` | TRY or USD |
| `column` | `str` | `"1"` | 1=TP, 2=YP, 3=Total |
| `lookback_weeks` | `int` | `12` | Number of weeks to analyze |

#### `compare_bulletin_metrics`

Compare multiple bulletin metrics side-by-side.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `metric_ids` | `str` | `"1.0.1,1.0.2"` | Comma-separated metric IDs |
| `currency` | `str` | `"TRY"` | TRY or USD |
| `column` | `str` | `"1"` | 1=TP, 2=YP, 3=Total |
| `days` | `int` | `90` | Days of history |

#### `get_regulatory_digest`

Get a digest of recent BDDK regulatory changes (decisions + announcements + bulletin).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `period` | `str` | `"month"` | Time period: week, month, quarter |

### Announcements & Monitoring

#### `search_bddk_announcements`

Search BDDK announcements and press releases.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `keywords` | `str` | `""` | Search terms in Turkish |
| `category` | `str` | `"basin"` | basin (press), mevzuat (regulation), insan kaynaklari (HR), veri (data) |

#### `check_bddk_updates`

Check for new BDDK announcements since last check.

### Document Management

#### `sync_bddk_documents`

Sync BDDK documents to local storage (download, extract, store).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `force` | `bool` | `False` | Re-download all documents |
| `document_id` | `str \| None` | `None` | Sync a single document by ID |
| `concurrency` | `int` | `5` | Number of parallel downloads |

#### `document_store_stats`

Show document store statistics for both SQLite and ChromaDB stores.

#### `trigger_startup_sync`

Manually trigger document sync if auto-sync was skipped.

### Server Operations

#### `health_check`

Check server health status (uptime, cache, store stats, sync status).

#### `bddk_metrics`

Show server performance metrics (request counts, latency per tool, cache hit rate).

## Architecture

```
server.py (FastMCP, 19 tools)
  ├── client.py (HTTP scraper, cache, Turkish NLP, search ranking)
  ├── data_sources.py (institutions, bulletins, announcements)
  ├── analytics.py (trends, digest, comparison, monitoring)
  ├── doc_store.py (SQLite + FTS5 + document versioning)
  ├── vector_store.py (ChromaDB + multilingual-e5-base)
  ├── doc_sync.py (download + extraction pipeline)
  ├── exceptions.py (custom exception hierarchy)
  ├── logging_config.py (structured JSON logging)
  └── metrics.py (request/latency/cache tracking)
```

### Storage

- **SQLite + FTS5** — persistent document store with full-text search, document versioning, and incremental sync metadata
- **ChromaDB** — vector store with multilingual embeddings for semantic search (~30ms for 1000+ docs)
- **JSON cache** — in-memory + disk cache for document metadata (1-hour TTL)

### Extraction Pipeline

Documents are downloaded and converted to Markdown via a 3-layer fallback:

1. **Nougat** (GPU) — best quality for academic PDFs with LaTeX/formulas (requires CUDA)
2. **markitdown** (CPU) — lightweight PDF/DOCX extraction (default for Railway)
3. **HTML parser** — last resort for HTML-embedded content

For mevzuat.gov.tr, a 4-layer download fallback is used: `.htm` > `.pdf` > iframe > `.doc`

## Setup

### Prerequisites

- Python 3.11 - 3.13
- [uv](https://docs.astral.sh/uv/)

### Claude Code

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "bddk": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/path/to/bddk-mcp",
        "--python", "3.12",
        "mcp", "run", "server.py"
      ]
    }
  }
}
```

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "bddk": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/path/to/bddk-mcp",
        "--python", "3.12",
        "mcp", "run", "server.py"
      ]
    }
  }
}
```

Replace `/path/to/bddk-mcp` with the actual path to this repository.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MCP_TRANSPORT` | `stdio` | Transport: `stdio` (local) or `streamable-http` (Railway) |
| `PORT` | `8000` | Server port (streamable-http only) |
| `BDDK_DB_PATH` | `./bddk_docs.db` | Path to SQLite database |
| `BDDK_CHROMA_PATH` | `./chroma_db` | Path to ChromaDB directory |
| `BDDK_AUTO_SYNC` | `""` | Set to `1`/`true` to auto-sync on startup |

### Install & Test

```bash
uv sync
uv run pytest tests/ -v
```

### Lint & Format

```bash
uv run ruff check .
uv run ruff format --check .
```

## Development

### Project Structure

```
bddk-mcp/
├── server.py              # FastMCP server with 19 tool definitions
├── client.py              # HTTP scraper, cache, Turkish NLP
├── data_sources.py        # Institution, bulletin, announcement fetchers
├── analytics.py           # Trend analysis, digest, comparison
├── doc_store.py           # SQLite + FTS5 document store
├── vector_store.py        # ChromaDB vector store
├── doc_sync.py            # Document download & extraction pipeline
├── models.py              # Pydantic request/response models
├── exceptions.py          # Custom exception hierarchy
├── logging_config.py      # Structured JSON logging
├── metrics.py             # Performance metrics tracking
├── __init__.py            # Package exports
├── pyproject.toml         # Dependencies and tool config
├── Dockerfile             # Railway deployment image
├── Procfile               # Railway process config
├── .github/workflows/
│   └── ci.yml             # GitHub Actions CI (lint + test)
└── tests/
    ├── conftest.py         # Shared fixtures and helpers
    ├── test_helpers.py     # Turkish NLP helper tests
    ├── test_search.py      # Search ranking logic tests
    ├── test_client.py      # HTTP scraping and cache tests
    ├── test_data_sources.py # Data fetcher tests
    ├── test_doc_store.py   # SQLite store tests
    ├── test_doc_sync.py    # Extraction pipeline tests
    ├── test_vector_store.py # ChromaDB store tests
    ├── test_analytics.py   # Analytics computation tests
    ├── test_exceptions.py  # Exception hierarchy + logging tests
    └── test_fts_sanitization.py # FTS5 injection prevention tests
```

### Testing

153 tests covering all modules:

```bash
uv run pytest tests/ -v              # Run all tests
uv run pytest tests/ -k "not vector" # Skip vector store tests (need model)
```

### CI/CD

GitHub Actions runs on every push to `main` and on pull requests:
- **Lint**: `ruff check` + `ruff format --check`
- **Test**: `pytest` on Python 3.11 and 3.12

## Usage Examples

```
> Search for regulations about capital adequacy
search_bddk_decisions(keywords="sermaye yeterliliği")

> Filter by category
search_bddk_decisions(keywords="banka", category="Yönetmelik")

> Filter by date range
search_bddk_decisions(keywords="banka", date_from="01.01.2024", date_to="31.12.2024")

> Semantic search (understands meaning, not just keywords)
search_document_store(query="faiz oranı riski nasıl hesaplanır")

> Get a specific document
get_bddk_document(document_id="1296")

> Get a mevzuat.gov.tr regulation
get_bddk_document(document_id="mevzuat_42628")

> Check document version history
get_document_history(document_id="mevzuat_42628")

> Check cache status
bddk_cache_status()

> Search for banks
search_bddk_institutions(keywords="Ziraat")

> List all factoring companies
search_bddk_institutions(institution_type="Faktoring Şirketi")

> Get latest banking sector data
get_bddk_bulletin_snapshot()

> Get total loans time-series
get_bddk_bulletin(metric_id="1.0.1", currency="TRY", days=90)

> Analyze trends with narrative
analyze_bulletin_trends(metric_id="1.0.1", lookback_weeks=12)

> Compare multiple metrics
compare_bulletin_metrics(metric_ids="1.0.1,1.0.2,1.0.4")

> Get regulatory digest
get_regulatory_digest(period="month")

> Search press releases
search_bddk_announcements(keywords="dolandırıcılık", category="basın")

> Check for new announcements
check_bddk_updates()

> Sync documents to local storage
sync_bddk_documents(concurrency=5)

> Check server health
health_check()

> View performance metrics
bddk_metrics()
```

## Data Sources

### Regulations (Mevzuat)

| Source | URL | Content |
|---|---|---|
| Page 49 | bddk.org.tr/Mevzuat/Liste/49 | Laws |
| Page 50 | bddk.org.tr/Mevzuat/Liste/50 | Banking Law regulations |
| Page 51 | bddk.org.tr/Mevzuat/Liste/51 | Bank & Credit Card regulations |
| Page 52 | bddk.org.tr/Mevzuat/Liste/52 | Leasing, Factoring, Finance regulations |
| Page 54 | bddk.org.tr/Mevzuat/Liste/54 | BDDK internal regulations |
| Page 55 | bddk.org.tr/Mevzuat/Liste/55 | Board Decisions (published) |
| Page 56 | bddk.org.tr/Mevzuat/Liste/56 | Board Decisions (unpublished) |
| Page 58 | bddk.org.tr/Mevzuat/Liste/58 | Regulation drafts |
| Page 63 | bddk.org.tr/Mevzuat/Liste/63 | Repealed regulations |

### Institutions

| Source | URL | Content |
|---|---|---|
| Page 77 | bddk.org.tr/Kurulus/Liste/77 | Banks (67) |
| Page 78 | bddk.org.tr/Kurulus/Liste/78 | Leasing Companies (86) |
| Page 79 | bddk.org.tr/Kurulus/Liste/79 | Factoring Companies (118) |
| Page 80 | bddk.org.tr/Kurulus/Liste/80 | Finance Companies (29) |
| Page 82 | bddk.org.tr/Kurulus/Liste/82 | Asset Management Companies (44) |

### Other Data

| Source | URL | Content |
|---|---|---|
| Weekly Bulletin | bddk.org.tr/bultenhaftalik | Banking sector metrics (loans, deposits, etc.) |
| Monthly Bulletin | bddk.org.tr/BultenAylik | Detailed monthly statistics (17 tables) |
| Announcements | bddk.org.tr/Duyuru/Liste/39-48 | Press releases, regulation notices |

## License

MIT

---

# BDDK MCP Sunucusu

BDDK (Bankacilik Duzenleme ve Denetleme Kurumu) karar ve duzenlemelerini aramak ve getirmek icin [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) sunucusu.

## Ozellikler

### Arama ve Erisim
- 1000'den fazla BDDK dokumani arasinda **Turkce destekli arama** (temel kok bulma/stemming dahil)
- ChromaDB ile **semantik arama** — anlam tabanli, sadece anahtar kelime degil (~30ms)
- 14 kategoride **kategori filtreleme**
- Kurul kararlari icin **tarih araligi filtreleme**
- **Ilgi siralama** (baslik eslesmesi > kok eslesmesi > alt dize eslesmesi)
- Hem BDDK hem mevzuat.gov.tr'den **dokuman getirme** (sayfalanmis Markdown)
- **Dokuman versiyonlama** — duzenleme degisikliklerini zaman icinde takip edin

### Veri ve Analitik
- **Kurulus rehberi** — 340+ kurulus arama (banka, kiralama, faktoring, finansman, varlik yonetim)
- **Haftalik bulten verisi** — bankacilik sektoru metrikleri, zaman serisi ve guncel snapshot
- **Aylik istatistikler** — detayli bankacilik sektoru verileri
- **Trend analizi** — haftalik degisim orani ve Turkce anlatim
- **Duzenleyici ozet** — karar, duyuru ve bulten verilerini birlestiren yonetici ozeti
- **Metrik karsilastirma** — birden fazla bankacilik metrigini yan yana karsilastirin
- **Guncelleme tespiti** — yeni BDDK duyurularini izleyin
- **Duyurular** — basin duyurulari, mevzuat duyurulari, IK ve veri yayimlama duyurulari

### Altyapi
- **Cift depolama** — tam metin aramasi icin SQLite + FTS5, semantik arama icin ChromaDB
- **Dokuman senkronizasyonu** — 3 cikarma yontemiyle toplu indirme (Nougat GPU, markitdown, HTML)
- **Artimsal senkronizasyon** — dokuman basina etag/last-modified takibi
- Disk uzerinde **kalici onbellekleme** (1 saat TTL)
- Korelasyon ID'leri ile **yapilandirilmis JSON log**
- **Saglik izleme** — sunucu saglik kontrolu ve performans metrikleri
- **Zarif kapatma** — SIGTERM'de WAL flush, depolari kapat
- **Hiz sinirlamasi** — giden istekler icin semafor tabanli kisitlama
- **CI/CD** — GitHub Actions ile lint + test matrisi (Python 3.11/3.12)
- **Tekrar deneme** ile dayanikli HTTP istekleri
- **Ozel istisna hiyerarsisi** ile belirli hata turleri

## Kurulum

### Gereksinimler

- Python 3.11 - 3.13
- [uv](https://docs.astral.sh/uv/)

### Claude Code

`~/.claude/settings.json` dosyasina ekleyin:

```json
{
  "mcpServers": {
    "bddk": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/repo/yolu/bddk-mcp",
        "--python", "3.12",
        "mcp", "run", "server.py"
      ]
    }
  }
}
```

### Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json` dosyasina ekleyin:

```json
{
  "mcpServers": {
    "bddk": {
      "command": "uv",
      "args": [
        "run",
        "--directory", "/repo/yolu/bddk-mcp",
        "--python", "3.12",
        "mcp", "run", "server.py"
      ]
    }
  }
}
```

`/repo/yolu/bddk-mcp` kismini bu reponun gercek yolu ile degistirin.

### Ortam Degiskenleri

| Degisken | Varsayilan | Aciklama |
|---|---|---|
| `MCP_TRANSPORT` | `stdio` | Aktarim: `stdio` (yerel) veya `streamable-http` (Railway) |
| `PORT` | `8000` | Sunucu portu (sadece streamable-http) |
| `BDDK_DB_PATH` | `./bddk_docs.db` | SQLite veritabani yolu |
| `BDDK_CHROMA_PATH` | `./chroma_db` | ChromaDB dizin yolu |
| `BDDK_AUTO_SYNC` | `""` | Baslangiçta otomatik senkronizasyon icin `1`/`true` yapin |

### Yukle ve Test Et

```bash
uv sync
uv run pytest tests/ -v
```

## Kullanim Ornekleri

```
> Sermaye yeterliligi hakkinda duzenleme ara
search_bddk_decisions(keywords="sermaye yeterliliği")

> Semantik arama (anlam tabanli)
search_document_store(query="faiz oranı riski nasıl hesaplanır")

> Kategoriye gore filtrele
search_bddk_decisions(keywords="banka", category="Yönetmelik")

> Tarih araligi ile filtrele
search_bddk_decisions(keywords="banka", date_from="01.01.2024", date_to="31.12.2024")

> Belirli bir dokumani getir
get_bddk_document(document_id="1296")

> Dokuman versiyon gecmisi
get_document_history(document_id="mevzuat_42628")

> Banka ara
search_bddk_institutions(keywords="Ziraat")

> Guncel bankacilik sektoru verisi
get_bddk_bulletin_snapshot()

> Trend analizi
analyze_bulletin_trends(metric_id="1.0.1", lookback_weeks=12)

> Metrik karsilastirma
compare_bulletin_metrics(metric_ids="1.0.1,1.0.2,1.0.4")

> Duzenleyici ozet
get_regulatory_digest(period="month")

> Yeni duyurulari kontrol et
check_bddk_updates()

> Sunucu saglik kontrolu
health_check()

> Performans metrikleri
bddk_metrics()
```

## Lisans

MIT
