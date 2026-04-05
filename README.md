# BDDK MCP Server

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) server for searching and retrieving decisions and regulations from **BDDK** (Banking Regulation and Supervision Agency of Turkey).

## Features

- **Search** across 1000+ BDDK documents with Turkish-aware keyword matching
- **Category filtering** by regulation type
- **Document retrieval** as paginated Markdown (supports both BDDK-hosted and mevzuat.gov.tr documents)
- **In-memory caching** with 1-hour TTL for fast repeated queries

### Available Categories

| Category | Description | Count |
|---|---|---|
| Kurul Karari | Board Decisions (published & unpublished) | ~957 |
| Yonetmelik | Regulations | 39 |
| Rehber | Guidelines | 19 |
| Genelge | Circulars | 13 |
| Sermaye Yeterliligi | Capital Adequacy Communiques & Guidelines | 10 |
| Bilgi Sistemleri | IT & Business Process Regulations | 8 |
| Teblig | Communiques | 6 |
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
| `category` | `str \| None` | `None` | Optional category filter |

### `get_bddk_document`

Retrieve a BDDK document as paginated Markdown.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `document_id` | `str` | required | Document ID from search results |
| `page_number` | `int` | `1` | Page of the Markdown output (5000 chars/page) |

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

### Install Dependencies

```bash
uv sync
```

## Usage Examples

```
> Search for regulations about capital adequacy
search_bddk_decisions(keywords="sermaye yeterliliği")

> Filter by category
search_bddk_decisions(keywords="banka", category="Yönetmelik")

> Get a specific document
get_bddk_document(document_id="1296")

> Get a mevzuat.gov.tr regulation
get_bddk_document(document_id="mevzuat_42628")
```

## Data Sources

| Source | URL | Content |
|---|---|---|
| Page 50 | bddk.org.tr/Mevzuat/Liste/50 | Banking Law regulations (all categories above except Board Decisions) |
| Page 55 | bddk.org.tr/Mevzuat/Liste/55 | Board Decisions (published in Official Gazette) |
| Page 56 | bddk.org.tr/Mevzuat/Liste/56 | Board Decisions (unpublished) |

## License

MIT

---

# BDDK MCP Sunucusu

BDDK (Bankacilik Duzenleme ve Denetleme Kurumu) karar ve duzenlemelerini aramak ve getirmek icin [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) sunucusu.

## Ozellikler

- 1000'den fazla BDDK dokumani arasinda **Turkce destekli anahtar kelime aramasi**
- Duzenleme turune gore **kategori filtreleme**
- Hem BDDK hem mevzuat.gov.tr'den **dokuman getirme** (sayfalanmis Markdown olarak)
- Hizli tekrar sorgular icin **bellek ici onbellekleme** (1 saat TTL)

### Mevcut Kategoriler

| Kategori | Aciklama | Adet |
|---|---|---|
| Kurul Karari | Kurul Kararlari (yayimlanmis ve yayimlanmamis) | ~957 |
| Yonetmelik | Yonetmelikler | 39 |
| Rehber | Rehberler | 19 |
| Genelge | Genelgeler | 13 |
| Sermaye Yeterliligi | Sermaye Yeterliligi Tebligleri ve Rehberleri | 10 |
| Bilgi Sistemleri | Bilgi Sistemleri ve Is Sureclerine Iliskin Duzenlemeler | 8 |
| Teblig | Tebligler | 6 |
| Tekduzen Hesap Plani | Tekduzen Hesap Plani | 4 |
| Faizsiz Bankacilik | Faizsiz Bankacıliga Iliskin Duzenlemeler | 2 |

## Araclar

### `search_bddk_decisions`

BDDK karar ve duzenlemelerini anahtar kelimeyle arayın.

| Parametre | Tip | Varsayilan | Aciklama |
|---|---|---|---|
| `keywords` | `str` | zorunlu | Turkce arama terimleri |
| `page` | `int` | `1` | Sayfa numarasi |
| `page_size` | `int` | `10` | Sayfa basina sonuc (maks 50) |
| `category` | `str \| None` | `None` | Opsiyonel kategori filtresi |

### `get_bddk_document`

Bir BDDK dokumanini sayfalanmis Markdown olarak getirin.

| Parametre | Tip | Varsayilan | Aciklama |
|---|---|---|---|
| `document_id` | `str` | zorunlu | Arama sonuclarindan dokuman ID'si |
| `page_number` | `int` | `1` | Markdown ciktisinin sayfasi (sayfa basina 5000 karakter) |

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

### Bagimliliklari Yukleyin

```bash
uv sync
```

## Kullanim Ornekleri

```
> Sermaye yeterliligi hakkinda duzenleme ara
search_bddk_decisions(keywords="sermaye yeterliliği")

> Kategoriye gore filtrele
search_bddk_decisions(keywords="banka", category="Yönetmelik")

> Belirli bir dokumani getir
get_bddk_document(document_id="1296")

> mevzuat.gov.tr'den yonetmelik getir
get_bddk_document(document_id="mevzuat_42628")
```

## Veri Kaynaklari

| Kaynak | URL | Icerik |
|---|---|---|
| Sayfa 50 | bddk.org.tr/Mevzuat/Liste/50 | Bankacilik Kanununa iliskin duzenlemeler |
| Sayfa 55 | bddk.org.tr/Mevzuat/Liste/55 | Resmi Gazetede yayimlanan Kurul Kararlari |
| Sayfa 56 | bddk.org.tr/Mevzuat/Liste/56 | Resmi Gazetede yayimlanmayan Kurul Kararlari |

## Lisans

MIT
