# BDDK MCP Server

[Türkçe](#turkce) | [English](#english)

BDDK MCP Server is an offline-first Model Context Protocol server for searching, retrieving, and analyzing Turkish banking regulation data from BDDK and mevzuat.gov.tr. It combines catalog search, document retrieval, section-level legal lookup, semantic search, bulletin analytics, document quality checks, and operator backfill workflows.

---

<a id="turkce"></a>

## Türkçe

### Ne İşe Yarar?

Bu proje, BDDK karar ve düzenlemelerini LLM araçları için güvenli ve izlenebilir bir MCP sunucusu olarak sunar. Amaç, modelin kendi bilgisinden cevap üretmesi yerine yerel veri deposundaki BDDK kaynaklarına dayanmasıdır.

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
- **Offline-first çalışma:** Veriler PostgreSQL/pgvector üzerinden yerel veya deployment veritabanından servis edilir.
- **Katalog ve gövde araması ayrımı:** `search_bddk_regulations` sadece başlık/metadata arar; `search_document_store` doküman gövdesinde semantik arama yapar.
- **Bölüm bazlı erişim:** `get_document_section` ve `search_document_sections` ile `943 İlke 5` veya `mevzuat_22599 Madde 9` gibi referanslar doğrudan bulunur.
- **Exact legal-reference koruması:** `Madde 9` gibi lexical eşleşmeler, semantik skor düşük olsa bile korunur.
- **Kalite etiketleri:** Doküman çıktıları `clean`, `warning`, `fail` sinyalleri ve kalite bayraklarıyla işaretlenir.
- **Güvenli Markdown:** Data URI, raw HTML/OCR artefact ve uzun satırlar context'e verilmeden temizlenir.
- **Operatör scriptleri:** kalite tarama, kalite backfill ve `document_sections` reindex akışları mevcuttur.
- **PostgreSQL + pgvector:** dokümanlar, bölümler, FTS ve vektör arama tek veritabanı üzerinde çalışır.

### Araç Yüzeyi

Varsayılan public deployment `BDDK_ADMIN_TOOLS=false` ile 16 read-only araç expose eder.

| Modül | Araçlar |
|---|---|
| Arama | `search_bddk_regulations`, `search_document_store`, `search_bddk_institutions`, `search_bddk_announcements` |
| Doküman | `get_bddk_document`, `get_document_history` |
| Bölümler | `get_document_section`, `search_document_sections` |
| Bülten | `get_bddk_bulletin`, `get_bddk_bulletin_snapshot`, `get_bddk_monthly`, `bddk_cache_status` |
| Analitik | `analyze_bulletin_trends`, `get_regulatory_digest`, `compare_bulletin_metrics`, `check_bddk_updates` |

`BDDK_ADMIN_TOOLS=true` ile ek operatör araçları açılır. Admin/operator deployment toplam 26 tools olarak belgelenir: 16 public araç + 10 operatör aracı.

- `document_store_stats`
- `refresh_bddk_cache`
- `sync_bddk_documents`
- `trigger_startup_sync`
- `document_health`
- `health_check`
- `bddk_metrics`
- `backfill_degraded_documents`
- `backfill_status`
- `document_quality_report`

Geçerli runtime için toplam olası MCP araç sayısı 26 tools olarak belgelenir. Benchmark schema fixture sayısı runtime deployment sayısından farklı olabilir; benchmark koşuları kullandıkları exact tool listesini kaydetmelidir. Bkz. [benchmark/README.md](benchmark/README.md).

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
uv run python -c 'import asyncio, asyncpg
async def main():
    conn = await asyncpg.connect("postgresql://bddk:bddk@localhost:5432/bddk")
    exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", "bddk_test")
    if not exists:
        await conn.execute("CREATE DATABASE bddk_test")
    await conn.close()
asyncio.run(main())'
```

Test:

```bash
uv run pytest tests/test_tools_sections.py tests/test_doc_store.py -k section -v
uv run ruff check .
```

MCP stdio çalıştırma:

```bash
BDDK_DATABASE_URL=postgresql://bddk:bddk@localhost:5432/bddk \
uv run mcp run server.py
```

HTTP transport:

```bash
BDDK_DATABASE_URL=postgresql://bddk:bddk@localhost:5432/bddk \
MCP_TRANSPORT=streamable-http \
PORT=8000 \
uv run python server.py
```

### Claude / Codex Yapılandırması

Örnek MCP config:

```json
{
  "mcpServers": {
    "bddk": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/bddk-mcp",
        "mcp",
        "run",
        "server.py"
      ],
      "env": {
        "BDDK_DATABASE_URL": "postgresql://bddk:bddk@localhost:5432/bddk"
      }
    }
  }
}
```

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

### Mimari

```text
server.py                 FastMCP giriş noktası ve lifecycle
deps.py                   Paylaşılan bağımlılık container'ı
config.py                 Ortam değişkenleri ve validasyon
client.py                 BDDK katalog/cache istemcisi
doc_store.py              PostgreSQL doküman, versiyon, FTS ve bölüm deposu
vector_store.py           pgvector + hibrit semantik/FTS arama
section_index.py          Madde/İlke/Paragraf/Ek parser'ı
legal_ref.py              Türkçe legal-reference parser'ı
markdown_quality.py       Markdown sanitization ve kalite etiketleri
doc_sync.py               İndirme ve extraction pipeline'ı
tools/                    MCP tool modülleri
scripts/                  Operatör ve backfill scriptleri
benchmark/                Tool schema ve benchmark altyapısı
```

### Veri Kalitesi ve Güvenlik Notları

- Tool cevapları sadece lokal store'dan gelir; runtime'da doküman live-fetch yapılmaz.
- Kalitesi düşük extraction çıktıları `warning` veya `fail` olarak işaretlenir.
- Formül ağır veya OCR bozuk dokümanlarda kaynak PDF incelemesi gerekebilir.
- Tool cevaplarında data URI, raw HTML ve bazı OCR artefact'ları temizlenir.
- Model cevap verirken sadece tool çıktısına dayanmalıdır; karar numarası, tarih veya hukuki sonuç uydurulmamalıdır.

---

<a id="english"></a>

## English

### What Is This?

BDDK MCP Server exposes Turkish banking regulation data as a safe, auditable Model Context Protocol server. It is designed to ground LLM answers in local BDDK data instead of relying on the model's prior knowledge.

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
- **Offline-first runtime:** data is served from PostgreSQL/pgvector rather than live web fetches.
- **Catalog/body separation:** `search_bddk_regulations` searches metadata; `search_document_store` searches document bodies.
- **Section-level retrieval:** `get_document_section` and `search_document_sections` support references like `943 Ilke 5` and `mevzuat_22599 Madde 9`.
- **Exact legal-reference preservation:** lexical hits such as `Madde 9` survive dense relevance filtering.
- **Quality labels:** document outputs include `clean`, `warning`, or `fail` metadata and quality flags.
- **Safe Markdown:** data URIs, raw HTML/OCR artifacts, and pathological long lines are sanitized before model context.
- **Operator scripts:** quality scan, quality backfill, and `document_sections` reindex workflows are included.
- **PostgreSQL + pgvector:** documents, sections, FTS, and vector search share one database.

### Tool Surface

The default public deployment with `BDDK_ADMIN_TOOLS=false` exposes 16 read-only tools.

| Module | Tools |
|---|---|
| Search | `search_bddk_regulations`, `search_document_store`, `search_bddk_institutions`, `search_bddk_announcements` |
| Documents | `get_bddk_document`, `get_document_history` |
| Sections | `get_document_section`, `search_document_sections` |
| Bulletin | `get_bddk_bulletin`, `get_bddk_bulletin_snapshot`, `get_bddk_monthly`, `bddk_cache_status` |
| Analytics | `analyze_bulletin_trends`, `get_regulatory_digest`, `compare_bulletin_metrics`, `check_bddk_updates` |

With `BDDK_ADMIN_TOOLS=true`, operator tools are also exposed. The admin/operator deployment exposes 26 tools total: 16 public tools plus 10 operator tools.

- `document_store_stats`
- `refresh_bddk_cache`
- `sync_bddk_documents`
- `trigger_startup_sync`
- `document_health`
- `health_check`
- `bddk_metrics`
- `backfill_degraded_documents`
- `backfill_status`
- `document_quality_report`

Total possible MCP tools in the current runtime is 26. Benchmark schema fixture counts can differ from runtime deployment counts; benchmark runs should record the exact exposed tool list they used. See [benchmark/README.md](benchmark/README.md).

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
uv run python -c 'import asyncio, asyncpg
async def main():
    conn = await asyncpg.connect("postgresql://bddk:bddk@localhost:5432/bddk")
    exists = await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", "bddk_test")
    if not exists:
        await conn.execute("CREATE DATABASE bddk_test")
    await conn.close()
asyncio.run(main())'
```

Test:

```bash
uv run pytest tests/test_tools_sections.py tests/test_doc_store.py -k section -v
uv run ruff check .
```

Run MCP over stdio:

```bash
BDDK_DATABASE_URL=postgresql://bddk:bddk@localhost:5432/bddk \
uv run mcp run server.py
```

Run streamable HTTP:

```bash
BDDK_DATABASE_URL=postgresql://bddk:bddk@localhost:5432/bddk \
MCP_TRANSPORT=streamable-http \
PORT=8000 \
uv run python server.py
```

### Claude / Codex Configuration

Example MCP config:

```json
{
  "mcpServers": {
    "bddk": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/bddk-mcp",
        "mcp",
        "run",
        "server.py"
      ],
      "env": {
        "BDDK_DATABASE_URL": "postgresql://bddk:bddk@localhost:5432/bddk"
      }
    }
  }
}
```

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

### Architecture

```text
server.py                 FastMCP entry point and lifecycle
deps.py                   Shared dependency container
config.py                 Environment configuration and validation
client.py                 BDDK catalog/cache client
doc_store.py              PostgreSQL document, version, FTS, and section store
vector_store.py           pgvector + hybrid semantic/FTS search
section_index.py          Madde/Ilke/Paragraf/Ek parser
legal_ref.py              Turkish legal-reference parser
markdown_quality.py       Markdown sanitization and quality labels
doc_sync.py               Download and extraction pipeline
tools/                    MCP tool modules
scripts/                  Operator and backfill scripts
benchmark/                Tool schemas and benchmark infrastructure
```

### Data Quality And Safety Notes

- Tool responses are served from the local store; documents are not live-fetched at runtime.
- Low-quality extractions are marked as `warning` or `fail`.
- Formula-heavy or OCR-corrupted documents may require source PDF review.
- Data URIs, raw HTML, and selected OCR artifacts are removed before model context.
- The model should answer only from tool output. It should not invent decision numbers, dates, or legal conclusions.

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

No license file is currently included. Treat reuse rights as unspecified until a license is added.
