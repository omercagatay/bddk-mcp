# BDDK MCP Server

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server for searching and retrieving decisions and regulations from **BDDK** (Banking Regulation and Supervision Agency of Turkey).

## Features

- **Search** across 1000+ BDDK documents with Turkish-aware keyword matching and basic stemming
- **Category filtering** by regulation type (14 categories)
- **Date range filtering** for board decisions
- **Relevance ranking** (title match > stem match > substring match)
- **Document retrieval** as paginated Markdown (BDDK and mevzuat.gov.tr)
- **Persistent caching** to disk with 1-hour TTL
- **Retry with backoff** for resilient HTTP fetching
- **Error handling** with graceful fallbacks

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

### `search_bddk_decisions`

Search for BDDK decisions and regulations by keyword.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `keywords` | `str` | required | Search terms in Turkish |
| `page` | `int` | `1` | Page number |
| `page_size` | `int` | `10` | Results per page (max 50) |
| `category` | `str \| None` | `None` | Category filter |
| `date_from` | `str \| None` | `None` | Start date (DD.MM.YYYY) |
| `date_to` | `str \| None` | `None` | End date (DD.MM.YYYY) |

### `get_bddk_document`

Retrieve a BDDK document as paginated Markdown.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `document_id` | `str` | required | Document ID from search results |
| `page_number` | `int` | `1` | Page of the Markdown output (5000 chars/page) |

### `bddk_cache_status`

Show cache statistics: total items, age, categories, and any page errors.

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

### Install & Test

```bash
uv sync
uv run pytest tests/ -v
```

## Usage Examples

```
> Search for regulations about capital adequacy
search_bddk_decisions(keywords="sermaye yeterliliği")

> Filter by category
search_bddk_decisions(keywords="banka", category="Yönetmelik")

> Filter by date range
search_bddk_decisions(keywords="banka", date_from="01.01.2024", date_to="31.12.2024")

> Get a specific document
get_bddk_document(document_id="1296")

> Get a mevzuat.gov.tr regulation
get_bddk_document(document_id="mevzuat_42628")

> Check cache status
bddk_cache_status()
```

## Data Sources

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

## License

MIT

---

# BDDK MCP Sunucusu

BDDK (Bankacilik Duzenleme ve Denetleme Kurumu) karar ve duzenlemelerini aramak ve getirmek icin [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) sunucusu.

## Ozellikler

- 1000'den fazla BDDK dokumani arasinda **Turkce destekli arama** (temel kok bulma/stemming dahil)
- 14 kategoride **kategori filtreleme**
- Kurul kararlari icin **tarih araligi filtreleme**
- **Ilgi siralama** (baslik eslesmesi > kok eslesmesi > alt dize eslesmesi)
- Hem BDDK hem mevzuat.gov.tr'den **dokuman getirme** (sayfalanmis Markdown)
- Disk uzerinde **kalici onbellekleme** (1 saat TTL)
- **Tekrar deneme** ile dayanikli HTTP istekleri
- **Hata yonetimi** ve geri donus mekanizmalari

### Mevcut Kategoriler

| Kategori | Aciklama | Adet |
|---|---|---|
| Kurul Karari | Kurul Kararlari (yayimlanmis ve yayimlanmamis) | ~957 |
| Yonetmelik | Yonetmelikler | 39 |
| Rehber | Rehberler | 19 |
| Genelge | Genelgeler | 13 |
| Duzenleme Taslagi | Duzenleme Taslaklari | 11 |
| Sermaye Yeterliligi | Sermaye Yeterliligi Tebligleri ve Rehberleri | 10 |
| Bilgi Sistemleri | Bilgi Sistemleri ve Is Sureclerine Iliskin Duzenlemeler | 8 |
| Finansal Kiralama ve Faktoring | Finansal Kiralama ve Faktoring Duzenlemeleri | 7 |
| BDDK Duzenlemesi | BDDK'ya Iliskin Duzenlemeler | 7 |
| Mulga Duzenleme | Mulga Duzenlemeler | 7 |
| Teblig | Tebligler | 6 |
| Kanun | Kanunlar | 4 |
| Tekduzen Hesap Plani | Tekduzen Hesap Plani | 4 |
| Faizsiz Bankacilik | Faizsiz Bankacıliga Iliskin Duzenlemeler | 2 |

## Araclar

### `search_bddk_decisions`

BDDK karar ve duzenlemelerini anahtar kelimeyle arayin.

| Parametre | Tip | Varsayilan | Aciklama |
|---|---|---|---|
| `keywords` | `str` | zorunlu | Turkce arama terimleri |
| `page` | `int` | `1` | Sayfa numarasi |
| `page_size` | `int` | `10` | Sayfa basina sonuc (maks 50) |
| `category` | `str \| None` | `None` | Kategori filtresi |
| `date_from` | `str \| None` | `None` | Baslangic tarihi (GG.AA.YYYY) |
| `date_to` | `str \| None` | `None` | Bitis tarihi (GG.AA.YYYY) |

### `get_bddk_document`

Bir BDDK dokumanini sayfalanmis Markdown olarak getirin.

| Parametre | Tip | Varsayilan | Aciklama |
|---|---|---|---|
| `document_id` | `str` | zorunlu | Arama sonuclarindan dokuman ID'si |
| `page_number` | `int` | `1` | Markdown ciktisinin sayfasi (sayfa basina 5000 karakter) |

### `bddk_cache_status`

Onbellek istatistiklerini gosterin: toplam oge, yas, kategoriler ve sayfa hatalari.

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

### Yukle ve Test Et

```bash
uv sync
uv run pytest tests/ -v
```

## Kullanim Ornekleri

```
> Sermaye yeterliligi hakkinda duzenleme ara
search_bddk_decisions(keywords="sermaye yeterliliği")

> Kategoriye gore filtrele
search_bddk_decisions(keywords="banka", category="Yönetmelik")

> Tarih araligi ile filtrele
search_bddk_decisions(keywords="banka", date_from="01.01.2024", date_to="31.12.2024")

> Belirli bir dokumani getir
get_bddk_document(document_id="1296")

> mevzuat.gov.tr'den yonetmelik getir
get_bddk_document(document_id="mevzuat_42628")

> Onbellek durumunu kontrol et
bddk_cache_status()
```

## Veri Kaynaklari

| Kaynak | URL | Icerik |
|---|---|---|
| Sayfa 49 | bddk.org.tr/Mevzuat/Liste/49 | Kanunlar |
| Sayfa 50 | bddk.org.tr/Mevzuat/Liste/50 | Bankacilik Kanununa iliskin duzenlemeler |
| Sayfa 51 | bddk.org.tr/Mevzuat/Liste/51 | Banka Kartlari ve Kredi Kartlari duzenlemeleri |
| Sayfa 52 | bddk.org.tr/Mevzuat/Liste/52 | Finansal Kiralama, Faktoring duzenlemeleri |
| Sayfa 54 | bddk.org.tr/Mevzuat/Liste/54 | BDDK'ya iliskin duzenlemeler |
| Sayfa 55 | bddk.org.tr/Mevzuat/Liste/55 | Resmi Gazetede yayimlanan Kurul Kararlari |
| Sayfa 56 | bddk.org.tr/Mevzuat/Liste/56 | Resmi Gazetede yayimlanmayan Kurul Kararlari |
| Sayfa 58 | bddk.org.tr/Mevzuat/Liste/58 | Duzenleme taslaklari |
| Sayfa 63 | bddk.org.tr/Mevzuat/Liste/63 | Mulga duzenlemeler |

## Lisans

MIT
