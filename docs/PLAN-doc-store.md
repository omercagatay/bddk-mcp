# BDDK Lokal Doküman Deposu — Uygulama Planı

## Context

BDDK MCP Server'da (`/Users/omercagataytasdeviren/bddk_mcp_module/`) dokümanlar her seferinde uzak sunuculardan çekiliyor. Sorunlar:
- **mevzuat.gov.tr** PDF'leri programatik erişimi engelliyor (HTTP 403)
- Matematiksel formüller (faiz oranı şokları vb.) düzgün çıkarılamıyor
- Her doküman çağrısı ağ bağımlı ve yavaş
- Mevcut cache sadece metadata'yı tutuyor, doküman içeriğini değil

**Hedef**: Tüm mevzuat dokümanlarını lokal SQLite veritabanında tut, formülleri LaTeX olarak düzgün çıkar, hem CLI hem MCP tool ile senkronize et.

---

## Karar Analizi

### 1. Depolama: SQLite + FTS5 (Vector DB değil)

**Neden SQLite + FTS5:**
- ~200 doküman, küçük bir korpus — vector DB overkill
- Railway'de tek dosya ile deploy, ek servis gerekmez
- FTS5 ile Türkçe full-text arama (mevcut `_turkish_stem` fonksiyonları yeniden kullanılabilir)
- Hem metadata hem doküman içeriği tek DB'de

**Neden Vector DB değil:**
- Railway ephemeral filesystem → ChromaDB/Qdrant persistent storage sorunu
- Embedding modeli gerekir (~500MB+ container size artışı veya harici API maliyeti)
- Türkçe hukuki metinlerde genel amaçlı embedding kalitesi düşük
- Regulatory search keyword-based çalışıyor, semantic search ek değer katmaz

> **İleride** semantic search eklemek istenirse, SQLite üzerine `sqlite-vec` eklentisi veya ayrı bir embedding pipeline kolayca entegre edilebilir.

### 2. Formül Çıkarma: Nougat (GPU — RTX 5080 lokal makine)

**Nougat** = Meta'nın academic PDF parser'ı, en yüksek kalite LaTeX formül çıkarma:
- RTX 5080 lokal makinede çalışır (extraction aşamasında)
- Railway'de GPU gerekmez — önceden çıkarılmış markdown DB'den sunulur
- Akademik düzeyde LaTeX formül çıkarma
- Tablo yapısını mükemmel koruma
- Sayfa başına ~5-15 saniye (GPU ile)

**Split mimari:**
```
[Lokal RTX 5080]                    [Railway (CPU only)]
doc_sync.py                         server.py
  ↓ Nougat ile PDF → LaTeX/MD         ↓ SQLite'dan oku
  ↓ SQLite'a yaz                       ↓ FTS5 ile ara
  ↓ DB dosyasını Railway'e gönder      ↓ Markdown dön
```

| Özellik | markitdown (mevcut) | pymupdf4llm | **Nougat (seçilen)** |
|---------|-------------------|-------------|--------|
| GPU | Hayır | Hayır | **Evet (lokal)** |
| Formül | Yok | İyi (heuristik) | **Mükemmel (ML)** |
| Tablo | Zayıf | İyi | **Mükemmel** |
| Hız | Yavaş | Hızlı | Yavaş (ama tek seferlik) |
| Container etkisi | ~5MB | ~15MB | **0 (Railway'de yok)** |

### 3. mevzuat.gov.tr Erişimi: Katmanlı Fallback

**Neden mevcut yöntemler yetersiz:**
- `curl`/`httpx` ile doğrudan PDF → 403 Forbidden
- Playwright/Selenium → 100MB+ container, Railway'de Chromium gerekli

**Önerilen yaklaşım — 3 katmanlı fallback:**

```
1. Word (.doc) indirme ← EN HAFİF, büyük olasılıkla çalışır
   URL: https://mevzuat.gov.tr/MevzuatMetin/yonetmelik/7.5.42628.doc

2. Session-based PDF ← Cookie + Referer ile
   a. HTML sayfayı GET → cookie al
   b. PDF'i aynı session ile GET + Referer header

3. HTML iframe içeriği ← Son çare
   HTML sayfadaki iframe src'sini parse et, iframe'in HTML'ini çek, markdown'a çevir
```

**Ek**: BDDK-hosted dokümanlar (`/Mevzuat/DokumanGetir/{id}`) sorunsuz çalışıyor, değişiklik gerekmez.

### 4. Güncelleme: CLI + MCP Tool

- `python doc_sync.py sync` → Manuel CLI komutu
- `sync_bddk_documents` → MCP tool olarak Claude'dan tetikleme
- İkisi de sadece yeni/değişen dokümanları indirir (content hash karşılaştırma)

---

## Uygulama Planı

### Adım 1: `doc_store.py` — SQLite Depo Katmanı (yeni dosya)

```python
# /Users/omercagataytasdeviren/bddk_mcp_module/doc_store.py

class DocumentStore:
    """SQLite + FTS5 doküman deposu."""

    def __init__(self, db_path: Path)
    async def get_document(self, doc_id: str) -> Optional[StoredDocument]
    async def get_document_page(self, doc_id: str, page: int) -> Optional[DocumentPage]
    async def store_document(self, doc: StoredDocument) -> None
    async def search_content(self, query: str, limit: int = 20) -> list[SearchHit]
    async def needs_refresh(self, doc_id: str, max_age_days: int = 30) -> bool
    async def import_from_cache(self, cache_items: list[dict]) -> int
    def stats(self) -> dict
```

**SQLite şeması:**
```sql
CREATE TABLE documents (
    document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT,
    decision_date TEXT,
    decision_number TEXT,
    source_url TEXT,
    pdf_blob BLOB,              -- ham PDF (yeniden işleme için)
    markdown_content TEXT,       -- çıkarılmış markdown
    content_hash TEXT,           -- SHA256 (değişim tespiti)
    downloaded_at REAL,
    extracted_at REAL,
    extraction_method TEXT,      -- 'pymupdf4llm', 'markitdown', 'manual'
    total_pages INTEGER DEFAULT 1,
    file_size INTEGER
);

CREATE VIRTUAL TABLE documents_fts USING fts5(
    title, markdown_content, category,
    content='documents',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE decision_index (
    document_id TEXT PRIMARY KEY,
    title TEXT, content TEXT, category TEXT,
    decision_date TEXT, decision_number TEXT,
    source_url TEXT, indexed_at REAL
);
```

### Adım 2: `doc_sync.py` — İndirme/Senkronizasyon (yeni dosya)

```python
# /Users/omercagataytasdeviren/bddk_mcp_module/doc_sync.py

class DocumentSyncer:
    async def sync_document(self, doc_id: str, source_url: str) -> SyncResult
    async def sync_all(self, decision_index: list[dict], concurrency: int = 5) -> SyncReport
    async def _download_mevzuat(self, mevzuat_no, tur, tertip) -> DownloadResult
    async def _download_bddk(self, document_id: str) -> DownloadResult

# CLI interface
if __name__ == "__main__":
    # python doc_sync.py sync [--force] [--doc-id X]
```

**Mevzuat indirme fallback zinciri:**
1. `.doc` URL dene → `markitdown` ile markdown'a çevir
2. Session-based `.pdf` dene → Nougat ile çevir (lokal GPU) / markitdown (Railway)
3. HTML iframe parse → BeautifulSoup ile çevir
4. Başarısız olursa logla, sonraki sync'te tekrar dene

**Nougat entegrasyonu (doc_sync.py):**
```python
def _extract_with_nougat(self, pdf_path: Path) -> str:
    """Nougat ile PDF → LaTeX/Markdown çıkarma (GPU gerekli)."""
    try:
        from nougat import NougatModel
        model = NougatModel.from_pretrained("facebook/nougat-base")
        # GPU'ya taşı
        model = model.to("cuda")
        result = model.predict(str(pdf_path))
        return result  # LaTeX + Markdown
    except ImportError:
        # Nougat yoksa markitdown ile fallback
        return self._extract_with_markitdown(pdf_path)
```

### Adım 3: `client.py` — Mevcut `get_document_markdown` Refaktör

**Değişiklik:**
```
Eski akış: URL çöz → HTTP GET → markitdown → paginate → dön
Yeni akış: DocumentStore'dan kontrol → varsa dön → yoksa indir → pymupdf4llm → sakla → paginate → dön
```

- `BddkApiClient.__init__` → `DocumentStore` inject et
- `get_document_markdown` → store-first lookup + fallback to live fetch
- `_mevzuat_to_pdf_url` yanına `_mevzuat_to_doc_url` ekle
- PDF extraction: `markitdown[pdf]` → `nougat-ocr` (lokal) / `markitdown` fallback (Railway)

### Adım 4: `models.py` — Yeni Modeller

```python
class StoredDocument(BaseModel):
    document_id: str
    title: str
    category: str = ""
    source_url: str = ""
    pdf_bytes: Optional[bytes] = None
    markdown_content: str = ""
    content_hash: str = ""
    extraction_method: str = "pymupdf4llm"

class SyncResult(BaseModel):
    document_id: str
    success: bool
    method: str = ""     # 'doc', 'pdf_session', 'html_iframe'
    error: str = ""

class SyncReport(BaseModel):
    total: int
    downloaded: int
    skipped: int
    failed: int
    errors: list[SyncResult]
```

### Adım 5: `server.py` — Yeni MCP Tool

```python
@mcp.tool()
async def sync_bddk_documents(
    force: bool = False,
    document_id: str | None = None,
) -> str:
    """Sync BDDK documents to local storage."""
```

### Adım 6: `pyproject.toml` — Bağımlılıklar

```toml
dependencies = [
    "mcp[cli]>=1.0.0",
    "httpx>=0.27",
    "pydantic>=2.0",
    "markitdown>=0.1",          # [pdf] kaldırıldı, .doc/.html + Railway fallback
    "beautifulsoup4>=4.12",
    "aiosqlite>=0.20",          # YENİ: async SQLite
]

[dependency-groups]
dev = [
    "pytest>=9.0.2",
    "pytest-asyncio>=1.3.0",
]
gpu = [
    "nougat-ocr>=0.1.17",      # YENİ: Nougat (RTX 5080 lokal makine)
    "torch>=2.0",               # PyTorch GPU backend
]
```

**Not**: `nougat-ocr` sadece lokal RTX 5080 makinesine kurulur (`uv sync --group gpu`).
Railway container'ına eklenmez — önceden çıkarılmış markdown SQLite DB'de zaten mevcut.

### Adım 7: `Dockerfile` + `railway.toml` — Deployment

```dockerfile
RUN mkdir -p /app/data
ENV BDDK_DB_PATH=/app/data/bddk_docs.db
```

Railway dashboard'dan `/app/data` persistent volume mount edilecek.

---

## Dosya Değişiklikleri Özeti

| Dosya | İşlem | Açıklama |
|-------|-------|----------|
| `doc_store.py` | YENİ | SQLite + FTS5 depo katmanı |
| `doc_sync.py` | YENİ | İndirme/senkronizasyon + CLI |
| `client.py` | DEĞİŞTİR | Store-first lookup, pymupdf4llm entegrasyonu |
| `models.py` | DEĞİŞTİR | StoredDocument, SyncResult modelleri |
| `server.py` | DEĞİŞTİR | sync_bddk_documents MCP tool |
| `pyproject.toml` | DEĞİŞTİR | pymupdf4llm, aiosqlite bağımlılıkları |
| `Dockerfile` | DEĞİŞTİR | /app/data dizini, BDDK_DB_PATH |
| `railway.toml` | DEĞİŞTİR | Volume mount |
| `.gitignore` | DEĞİŞTİR | `bddk_docs.db` ekle |

## Doğrulama

1. **Lokal (RTX 5080):** `uv sync --group gpu && python doc_sync.py sync` → tüm dokümanlar indirilmeli
2. `get_bddk_document("1291")` → Rehber dokümanı SQLite'dan gelmeli
3. `get_bddk_document("mevzuat_42628")` → mevzuat.gov.tr dokümanı .doc fallback ile gelmeli
4. Faiz oranı formülleri LaTeX formatında çıkmalı (Nougat ile `\frac{...}{...}`, `\sum` vb.)
5. `sync_bddk_documents` MCP tool'u Claude'dan çağrılabilmeli
6. Railway deploy sonrası DB'den dokümanlar okunmalı (Nougat gerekmeden)
7. **Formül testi:** mevzuat_42628 (faiz oranı riski yönetmeliği) MADDE 9'daki iskonto formülü `İO0,p(to) = exp(-F0,p(to)·to)` doğru LaTeX olarak çıkmalı

## İş Akışı (Süreç)

```
[İlk Kurulum — Lokal RTX 5080]
1. uv sync --group gpu          ← Nougat + PyTorch kur
2. python doc_sync.py sync      ← Tüm dokümanları indir + Nougat ile işle
3. bddk_docs.db oluşur          ← ~200+ doküman, markdown + PDF blob

[Railway Deploy]
4. bddk_docs.db'yi Railway volume'a kopyala (scp/rsync/git-lfs)
5. MCP server SQLite'dan okur   ← GPU gerekmez

[Güncelleme (yeni mevzuat yayınlandığında)]
6. python doc_sync.py sync      ← Sadece yeni dokümanları indir
7. DB'yi Railway'e tekrar gönder
   VEYA
   sync_bddk_documents MCP tool ile Claude'dan tetikle (Railway'de markitdown fallback)
```
