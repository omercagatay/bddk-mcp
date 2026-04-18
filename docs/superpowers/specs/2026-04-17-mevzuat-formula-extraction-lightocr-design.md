# Mevzuat Formül Çıkarma — LightOnOCR-2-1B Geçişi

**Tarih:** 2026-04-17
**Durum:** Design — user onayı bekleniyor
**Etkilenen bileşenler:** `doc_sync.py`, `config.py`, `pyproject.toml`, yeni `ocr_backends.py`

## Sorun

BDDK mevzuat belgeleri (ör. `mevzuat_42628` — Bankacılık Hesaplarından Kaynaklanan Faiz Oranı Riskinin Standart Yaklaşımla Ölçülmesine İlişkin Yönetmelik) mevzuat.gov.tr üzerindeki HTML sayfasında formülleri `<img>` olarak gömüyor. Mevcut `_extract_html_to_markdown` yalnızca `h1-h5, p, li, td, th` tag'lerinden `get_text()` ile yazı çekiyor — `<img>` sessizce düşüyor. Sonuç: EK-2 gibi formül bölümleri tamamen kayboluyor; denetçi bu eksiği farkedemiyor.

Aynı sorun şu an `_extract_with_nougat` kullanılan PDF belgelerinde de kısmen var: Nougat 2023'ten beri güncellenmemiş, halüsinasyon eğilimli, yavaş (RTX 5080 üzerinde sayfa başı ~5-10s), ve 2026-Q1 benchmark'ında (arXiv 2512.09874) artık top-20 parser listesinde bile yok.

## Çözüm Özeti

1. **Extraction backend:** Nougat → **LightOnOCR-2-1B** (HF: `lightonai/LightOnOCR-2-1B`) birincil; **PP-StructureV3** fallback; **markitdown** son çare.
2. **Download sıralaması (mevzuat):** HTML önce yerine **PDF önce** (GeneratePdf → pdf_direct → htm → iframe/div → doc).
3. **Ortam:** Sync lokal'de (RTX 5080) koşar; Railway sadece seed'den okur (mevcut pattern).
4. **Rollout:** Önce 42628 tek sync + manuel doğrulama → 5 örnek mevzuat → tüm 1094 belge gece batch.

**Acceptance kriteri:** `mevzuat_42628` sayfa 5'te EK-2 formül sembolleri (`$`, `\frac`, `t_0`, vb.) görünür, `extraction_method='lightocr'` DB'de kayıtlı, Türkçe karakterler bozulmamış.

## Seçilen Yaklaşım

2026-Q1 benchmark (arXiv 2512.09874) lokal GPU + açık kaynak + yüksek formül skoru kriterleriyle filtrelendi:

| Model | Tablo | Formül | 100 sayfa | VRAM | Karar |
|-------|-------|--------|-----------|------|-------|
| **LightOnOCR-2-1B** | **9.08** | **9.57** | 30 dk | ~4 GB | **Birincil** — kombine skorda lokal lider |
| dots.ocr | 8.73 | 9.55 | 20 dk | ~7 GB | Değerlendirilmedi (LightOCR daha küçük) |
| **PP-StructureV3** | 6.86 | **9.59** | **3 dk** | küçük | **Fallback** — çok hızlı, formülleri iyi |
| Chandra | 8.43 | 9.45 | 4 saat | — | Yavaş |
| Nougat (eski) | — | — | yavaş | — | Kaldırılıyor |
| Marker | — | — | — | — | Benchmark'ta yok, geride kalmış |

## Mimari

### Yeni bileşen: `OCRBackend` protocol

```python
class OCRBackend(Protocol):
    name: str
    def is_available(self) -> bool    # lazy CUDA + model dosyası kontrolü
    def extract(self, pdf_bytes: bytes) -> str | None
```

Üç implementasyon (`ocr_backends.py` — yeni dosya):
- `LightOCRBackend` — GPU; `name="lightocr"`; model lifecycle syncer boyunca sıcak tutulur
- `PPStructureBackend` — GPU fallback; `name="pp_structure"`
- `MarkitdownBackend` — CPU son çare (mevcut `_extract_with_markitdown` sarmalı); `name="markitdown_degraded"` (PDF yolunda formül kaybı olduğu için explicit olarak "degraded" olarak işaretlenir; HTML yolunda zaten `html_parser` kullanılır, değişmez)

### Değişen modüller

**`doc_sync.py`:**
- `_extract_with_nougat` silinir
- `_extract_structured` artık `ocr_backends` listesi üzerinden gezer; ilk başarılı olan (MIN_CONTENT_LEN ≥ 500) kazanır
- `_download_mevzuat` katman sırası: GeneratePdf → pdf_direct → htm → iframe/div → doc (tur auto-detection korunur)
- `DocumentSyncer.__init__` kwarg'ı `prefer_nougat` → `ocr_backends: list[OCRBackend] | None`
- Concurrency default 5 → **1** (tek GPU; LightOCR içsel batch_size=4)

**`config.py` — yeni env var'lar:**
- `BDDK_OCR_BACKEND` = `lightocr | pp_structure | markitdown` (default `lightocr`)
- `BDDK_LIGHTOCR_MODEL_PATH` (offline-first, lokal model path)
- `BDDK_LIGHTOCR_BATCH_SIZE` (default 4)
- `BDDK_LIGHTOCR_DEVICE` (default `auto` → CUDA varsa cuda, yoksa cpu)
- `BDDK_OCR_MIN_CONTENT_LEN` (default 500 karakter)

**`pyproject.toml`:**
- Ekle (default deps): `transformers>=4.50`, `accelerate`, `pillow`, `pdf2image`, `paddleocr`, `paddlepaddle-gpu`
- Kaldır: `nougat-ocr`

Fallback zinciri (LightOCR → PPStructure → markitdown) için ikisi de default olmalı; opsiyonel extra yapmak fallback'i devre dışı bırakır.

**`doc_store.py` — yeni helper:**
- `get_pdf_bytes(doc_id) -> bytes | None` — DB'deki `pdf_bytes` kolonundan okur, re-extraction'da bandwidth tasarrufu

## Veri Akışı

### Tek belge sync (ör. `mevzuat_42628`, force=True)

1. `DocumentSyncer.sync_document(doc_id="mevzuat_42628", force=True)`
2. force=True → `has_document()` kontrolü atlanır
3. `_download_mevzuat()` — yeni sıralamayla:
   - Layer 1: Main page visit (cookie) → GeneratePdf → `%PDF-` magic
   - Layer 2: Direct static .pdf
   - Layer 3: .htm (fallback — formül kayıp riski)
   - Layer 4: iframe/div
   - Layer 5: .doc
4. `content (PDF bytes), ext=".pdf"` → `_extract_structured()`:
   ```
   for backend in [LightOCR, PPStructure, Markitdown]:
       if backend.is_available():
           result = backend.extract(pdf_bytes)
           if result and len(result) >= MIN_CONTENT_LEN:
               return ExtractionResult(content=result, method=backend.name)
   ```
5. Markdown dolu → `StoredDocument` (pdf_bytes ham olarak saklanır, markdown_content LaTeX ile, extraction_method="lightocr")
6. `DocumentStore.store_document()` → pg UPSERT; chunk'lar ve vector embed'ler yeniden hesaplanır
7. `clear_sync_failure(doc_id)`
8. `SyncResult(success=True, method="mevzuat_generate_pdf+lightocr")`

### Toplu sync

- `concurrency=1` (GPU tek iş yapar)
- LightOCR içinde batch_size=4 ile sayfalar GPU'ya yüklenir
- Başarısızlar `sync_failures` tablosuna retry için yazılır

## Hata Yönetimi

| Senaryo | Tespit | Davranış |
|---------|--------|----------|
| CUDA yok | `torch.cuda.is_available()==False` | LightOCR `is_available()=False` → PPStructure dener → yoksa markitdown |
| Model bozuk | `transformers` `OSError` | Hata logla, backend devre dışı, sonrakine geç |
| GPU OOM | `torch.cuda.OutOfMemoryError` | `empty_cache()`, batch_size yarıya indir, max 2 retry; sonra PPStructure |
| PDF bozuk | `pdf2image` `PDFPageCountError` | `retryable=False`, sync_failure kaydı |
| Halüsinasyon/boş | `len(md) < MIN_CONTENT_LEN` or `_is_error_page(md)` | Bu backend'i atla, sonrakine geç |
| Tüm backend'ler fail | — | `retryable=False`, extraction_method='failed' |

### Veri kaybı güvenliği (kritik)

**Mevcut bug:** `_extract` başarısız + `force=True` → `delete_document()` çağrılıyor, eski markdown kaybolur.

**Yeni davranış:** Yeni extraction başarılıysa üstüne yaz; başarısızsa eski kayıt dokunulmaz, sadece `sync_failure` kaydı eklenir.

### Degraded mode tracking

`extraction_method` kolonu artık `lightocr | pp_structure | markitdown_degraded | failed` değerlerini alır. Denetçi sorgusu:

```sql
SELECT document_id, title FROM documents WHERE extraction_method LIKE '%degraded%' OR extraction_method='failed';
```

## Test Stratejisi

### Birim testleri (GPU'suz, CI)

`tests/test_ocr_backends.py` — yeni:
- `test_lightocr_unavailable_without_cuda`
- `test_markitdown_backend_extracts_simple_pdf`
- `test_backend_chain_fallback` — [fail, fail, success] → üçüncüyü döner
- `test_empty_output_triggers_fallback` — MIN_CONTENT_LEN altı → sonrakine geç
- `test_data_loss_protection_on_force_reextract` — yeni fail + force=True → eski kayıt duruyor

`tests/test_doc_sync.py` — güncellenenler:
- `test_mevzuat_download_order` — httpx mock, GeneratePdf önce
- `test_extraction_method_logged_correctly`

### Entegrasyon testi (GPU gerekli, `pytest.mark.gpu`)

`tests/test_integration_lightocr.py`:
- `test_42628_ek2_formulas_extracted` — gerçek 42628 PDF → LightOCR → çıktıda `$`, `\frac`, `t_0`, veya "Paralel yukarı"+formül-benzeri sembol var mı
- `test_lightocr_turkish_chars` — ç, ğ, ı, ş korunuyor mu

### Manuel doğrulama

1. `uv run python doc_sync.py sync --doc-id mevzuat_42628 --force`
2. MCP: `get_bddk_document(document_id="mevzuat_42628", page=5)`
3. EK-2'de formül sembolleri gözle kontrol
4. `SELECT extraction_method, file_size FROM documents WHERE document_id='mevzuat_42628';` → `lightocr`

### Rollout

Kapsam aşamalı:

**Faz 1 (bug fix — mevzuat_*, html-extracted olanlar):**
1. 42628 manuel doğrulama ✓
2. 5 örnek formül içeren mevzuat (sermaye yeterliliği, likidite, vb.) tek tek → EK bölümleri kontrol
3. Regresyon spot check: 10 rastgele başarılı belge → hala düzgün
4. Hedef set:
   ```sql
   SELECT document_id FROM documents
   WHERE document_id LIKE 'mevzuat_%'
     AND extraction_method IN ('html_parser', 'markitdown', 'mevzuat_htm');
   ```
5. Bu set üzerinde `sync --force` (tahminen ~200-400 mevzuat × ortalama ~2 dk/belge = 7-14 saat; hafta sonu iş)
6. `seed.py export` → seed_data lokal olarak güncel (git commit yok)

**Faz 2 (opsiyonel kalite iyileştirme — BDDK kararları + kalan mevzuat):**
- Bug fix değil, kalite iyileştirme. Ayrı karar; bu spec kapsamı dışında.
- BDDK karar belgeleri zaten PDF-based; Nougat yerine LightOCR kullanmak ek doğruluk getirir ama aciliyeti yok.

**Süre tahmini doğrulaması (Faz 1 başlarken):** İlk 10 belge sync süresi ölçülür, kalan tahmin buna göre güncellenir.

## Acceptance Kriteri

- [ ] 42628 sayfa 5'te EK-2 formül sembolleri (`$`, `\frac`, `t_0`, veya benzeri) görünür
- [ ] `extraction_method='lightocr'` DB kayıtlı
- [ ] Türkçe karakterler (ç, ğ, ı, ş, ö, ü) bozulmamış
- [ ] En az 5 diğer mevzuat belgesinde EK bölümlerinde formül/tablo görünürlüğü artışı gözle doğrulandı
- [ ] Regresyon yok: 10 random başarılı belgede önceki içerik korundu
- [ ] `extraction_method LIKE '%degraded%'` sorgusu denetçiye düşük kaliteli belgeleri listeler

## Kapsam Dışı

- Railway ortamında GPU inference (ayrı proje, şu an gerek yok)
- BDDK karar belgeleri (doc_id digit-only) için toplu re-extraction (Faz 2, ayrı karar)
- rg_* OCR bug'ı (memory'deki ayrı bug)
- mevzuat_* encoding korupsiyon bug'ı (ayrı bug; `_decode_html` fix'i scope dışı)

**Not:** Yeni pipeline tüm belge tiplerine otomatik uygulanır (sync çağrıldığında Nougat yerine LightOCR çalışır). "Kapsam dışı" olan yalnızca **toplu geri-dolum**: BDDK kararları yeniden sync'lenmez, mevcut Nougat-extracted markdown'ları korunur; sadece yeni gelen/force edilenler LightOCR'dan geçer.

## Referanslar

- Benchmark: arXiv 2512.09874 — *Benchmarking Document Parsers on Mathematical Formula Extraction from PDFs* (Aralık 2025)
- Model: `lightonai/LightOnOCR-2-1B` (Apache 2.0)
- Fallback: `PaddlePaddle/PP-StructureV3`
- Sorun geçmişi: memory/project_bddk_encoding_bug.md (2026-04-16)
