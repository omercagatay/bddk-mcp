# Mevzuat Formül Çıkarma — LightOnOCR-2-1B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **User preference (memory/feedback_no_git_internal.md):** Skip git add/commit/branch steps. Write code and run tests only.

**Goal:** BDDK mevzuat belgelerinde `<img>` olarak gömülü formüllerin (ör. `mevzuat_42628` EK-2) kayboldan düzgün extract edilmesi.

**Architecture:** Nougat yerine LightOnOCR-2-1B (birincil, GPU), PP-StructureV3 (fallback), markitdown (son çare). `OCRBackend` protocol ile pluggable backend zinciri. Mevzuat download sırası PDF-önce olacak. Sync lokal'de (RTX 5080) koşar.

**Tech Stack:** Python 3.12, asyncio, PostgreSQL + asyncpg, transformers (LightOCR), paddleocr (PP-StructureV3), pdf2image, markitdown.

**Referans spec:** `docs/superpowers/specs/2026-04-17-mevzuat-formula-extraction-lightocr-design.md`

---

## File Structure

**Create:**
- `ocr_backends.py` — OCRBackend Protocol + 3 implementasyon + chain runner
- `tests/test_ocr_backends.py` — birim testler
- `tests/test_integration_lightocr.py` — GPU integration testi (pytest.mark.gpu)

**Modify:**
- `pyproject.toml` — transformers, accelerate, pdf2image, paddleocr; nougat-ocr kaldırılır
- `config.py` — `BDDK_OCR_*` env var'ları
- `doc_sync.py` — Nougat extraction yolu silinir, `_extract_structured` backend zincirini kullanır, `_download_mevzuat` sıralaması değişir, force+fail veri-kaybı koruması
- `doc_store.py` — `get_pdf_bytes(doc_id)` helper
- `tests/test_doc_sync.py` — mevcut testler güncellenir

---

## Task 1: Dependencies ve Config

**Files:**
- Modify: `pyproject.toml`
- Modify: `config.py:1-60` (yeni env var bloğu eklenir)

- [ ] **Step 1: `pyproject.toml` — Nougat kaldır, yeni depler ekle**

`pyproject.toml` içindeki `[project].dependencies` listesinden `nougat-ocr` satırını sil. Yerine şunları ekle:

```toml
"transformers>=4.50",
"accelerate>=1.0",
"pdf2image>=1.17",
"pillow>=10.0",
"paddleocr>=2.10",
"paddlepaddle-gpu>=2.6; platform_system!='Darwin'",
```

- [ ] **Step 2: `uv sync --dev` çalıştır**

Komut:
```bash
cd /home/cagatay/bddk-mcp && uv sync --dev
```
Beklenen: tüm bağımlılıklar install edilir, hata yok.

- [ ] **Step 3: `config.py`'e yeni env var bloğu ekle**

`# -- Embedding model` bölümünden sonra, `# -- pgvector` bölümünden önce şu bloğu ekle:

```python
# -- OCR extraction backends -------------------------------------------------

# Primary backend for PDF → markdown extraction.
# Options: lightocr, pp_structure, markitdown
OCR_BACKEND = os.environ.get("BDDK_OCR_BACKEND", "lightocr")

# LightOnOCR-2-1B model path (offline-first; empty = download from HF)
LIGHTOCR_MODEL_PATH = os.environ.get("BDDK_LIGHTOCR_MODEL_PATH", "")
LIGHTOCR_MODEL_NAME = os.environ.get("BDDK_LIGHTOCR_MODEL", "lightonai/LightOnOCR-2-1B")

# Batch size for page-level inference (RTX 5080: 4 is safe)
LIGHTOCR_BATCH_SIZE = int(os.environ.get("BDDK_LIGHTOCR_BATCH_SIZE", "4"))

# Device: auto | cuda | cpu
LIGHTOCR_DEVICE = os.environ.get("BDDK_LIGHTOCR_DEVICE", "auto")

# Minimum extracted character count to accept a backend's output
OCR_MIN_CONTENT_LEN = int(os.environ.get("BDDK_OCR_MIN_CONTENT_LEN", "500"))
```

- [ ] **Step 4: Config import testi**

```bash
cd /home/cagatay/bddk-mcp && uv run python -c "from config import OCR_BACKEND, LIGHTOCR_MODEL_NAME, OCR_MIN_CONTENT_LEN; print(OCR_BACKEND, LIGHTOCR_MODEL_NAME, OCR_MIN_CONTENT_LEN)"
```
Beklenen çıktı: `lightocr lightonai/LightOnOCR-2-1B 500`

---

## Task 2: OCRBackend Protocol + MarkitdownBackend

**Files:**
- Create: `ocr_backends.py`
- Create: `tests/test_ocr_backends.py`

- [ ] **Step 1: Failing test yaz — `tests/test_ocr_backends.py`**

```python
"""Unit tests for OCR backend implementations."""

from unittest.mock import patch

import pytest

from ocr_backends import ExtractionAttempt, MarkitdownBackend, OCRBackend


class TestMarkitdownBackend:
    def test_is_available_returns_true(self):
        backend = MarkitdownBackend()
        assert backend.is_available() is True

    def test_name_is_markitdown_degraded(self):
        backend = MarkitdownBackend()
        assert backend.name == "markitdown_degraded"

    def test_extract_returns_none_on_empty_pdf(self):
        backend = MarkitdownBackend()
        result = backend.extract(b"")
        assert result is None

    def test_extract_wraps_markitdown(self):
        backend = MarkitdownBackend()
        with patch("ocr_backends._run_markitdown") as mock:
            mock.return_value = "extracted text"
            result = backend.extract(b"%PDF-1.4\nfake")
            assert result == "extracted text"
            mock.assert_called_once()


class TestProtocol:
    def test_backend_has_required_methods(self):
        backend: OCRBackend = MarkitdownBackend()
        assert hasattr(backend, "name")
        assert hasattr(backend, "is_available")
        assert hasattr(backend, "extract")

    def test_extraction_attempt_model(self):
        attempt = ExtractionAttempt(backend="markitdown_degraded", content="x", error="")
        assert attempt.backend == "markitdown_degraded"
        assert attempt.content == "x"
```

- [ ] **Step 2: Test'i çalıştır, fail doğrula**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/test_ocr_backends.py -v
```
Beklenen: `ModuleNotFoundError: No module named 'ocr_backends'`

- [ ] **Step 3: `ocr_backends.py` — Protocol + MarkitdownBackend implementasyonu**

```python
"""
OCR backend implementations for BDDK document extraction.

Pluggable Protocol-based backends in preference order:
    1. LightOCRBackend       (GPU, formula-aware, primary)
    2. PPStructureBackend    (GPU fallback, fast)
    3. MarkitdownBackend     (CPU last resort, no formulas)

Each backend owns its own model lifecycle and availability check.
"""

from __future__ import annotations

import io
import logging
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ExtractionAttempt(BaseModel):
    """Result of a single backend extraction attempt."""

    backend: str
    content: str = ""
    error: str = ""


@runtime_checkable
class OCRBackend(Protocol):
    """Protocol for PDF → markdown extraction backends."""

    name: str

    def is_available(self) -> bool:
        """Check if backend can run (model loaded, CUDA available, etc.)."""
        ...

    def extract(self, pdf_bytes: bytes) -> str | None:
        """Extract markdown from PDF bytes. Returns None on failure."""
        ...


# ─── Markitdown backend (CPU, no formula support) ──────────────────────────


def _run_markitdown(pdf_bytes: bytes) -> str | None:
    """Invoke markitdown on raw PDF bytes. Isolated for mocking in tests."""
    try:
        from markitdown import MarkItDown

        md = MarkItDown()
        result = md.convert_stream(io.BytesIO(pdf_bytes), file_extension=".pdf")
        text = result.text_content.strip()
        return text if text else None
    except (ValueError, OSError, UnicodeDecodeError) as e:
        logger.warning("markitdown extraction failed: %s", e)
        return None


class MarkitdownBackend:
    """CPU-only, text-focused PDF extraction. No formula support."""

    name = "markitdown_degraded"

    def is_available(self) -> bool:
        try:
            import markitdown  # noqa: F401

            return True
        except ImportError:
            return False

    def extract(self, pdf_bytes: bytes) -> str | None:
        if not pdf_bytes:
            return None
        return _run_markitdown(pdf_bytes)
```

- [ ] **Step 4: Test'i tekrar çalıştır, PASS doğrula**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/test_ocr_backends.py -v
```
Beklenen: 5 test PASS.

---

## Task 3: Backend zincir runner + MIN_CONTENT_LEN

**Files:**
- Modify: `ocr_backends.py` (yeni `run_extraction_chain` fonksiyonu)
- Modify: `tests/test_ocr_backends.py` (zincir testleri)

- [ ] **Step 1: `tests/test_ocr_backends.py` sonuna failing testler ekle**

```python
from ocr_backends import run_extraction_chain


class _FakeBackend:
    """Test double implementing OCRBackend protocol."""

    def __init__(self, name: str, output: str | None, available: bool = True):
        self.name = name
        self._output = output
        self._available = available
        self.call_count = 0

    def is_available(self) -> bool:
        return self._available

    def extract(self, pdf_bytes: bytes) -> str | None:
        self.call_count += 1
        return self._output


class TestExtractionChain:
    def test_first_successful_backend_wins(self):
        b1 = _FakeBackend("first", "x" * 600)
        b2 = _FakeBackend("second", "y" * 600)
        attempt = run_extraction_chain(b"pdf", [b1, b2], min_len=500)
        assert attempt.backend == "first"
        assert attempt.content == "x" * 600
        assert b1.call_count == 1
        assert b2.call_count == 0

    def test_unavailable_backend_skipped(self):
        b1 = _FakeBackend("first", "x" * 600, available=False)
        b2 = _FakeBackend("second", "y" * 600)
        attempt = run_extraction_chain(b"pdf", [b1, b2], min_len=500)
        assert attempt.backend == "second"
        assert b1.call_count == 0

    def test_short_output_triggers_fallback(self):
        b1 = _FakeBackend("first", "tiny")
        b2 = _FakeBackend("second", "x" * 600)
        attempt = run_extraction_chain(b"pdf", [b1, b2], min_len=500)
        assert attempt.backend == "second"

    def test_none_output_triggers_fallback(self):
        b1 = _FakeBackend("first", None)
        b2 = _FakeBackend("second", "x" * 600)
        attempt = run_extraction_chain(b"pdf", [b1, b2], min_len=500)
        assert attempt.backend == "second"

    def test_all_backends_fail_returns_error(self):
        b1 = _FakeBackend("first", None)
        b2 = _FakeBackend("second", "tiny")
        attempt = run_extraction_chain(b"pdf", [b1, b2], min_len=500)
        assert attempt.backend == "failed"
        assert "all backends" in attempt.error.lower()
```

- [ ] **Step 2: Test'leri çalıştır, fail doğrula**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/test_ocr_backends.py::TestExtractionChain -v
```
Beklenen: 5 test FAIL — `run_extraction_chain` import edilemiyor.

- [ ] **Step 3: `ocr_backends.py` sonuna `run_extraction_chain` ekle**

```python
def run_extraction_chain(
    pdf_bytes: bytes,
    backends: list[OCRBackend],
    min_len: int,
) -> ExtractionAttempt:
    """Try each backend in order. First result with len >= min_len wins.

    Returns ExtractionAttempt(backend="failed", error=...) if all fail.
    """
    errors: list[str] = []
    for backend in backends:
        if not backend.is_available():
            logger.debug("backend=%s unavailable, skipping", backend.name)
            errors.append(f"{backend.name}: unavailable")
            continue
        try:
            result = backend.extract(pdf_bytes)
        except Exception as e:
            logger.warning("backend=%s raised: %s", backend.name, e)
            errors.append(f"{backend.name}: {type(e).__name__}: {e}")
            continue

        if result is None:
            errors.append(f"{backend.name}: returned None")
            continue
        if len(result) < min_len:
            errors.append(f"{backend.name}: output too short ({len(result)} < {min_len})")
            continue

        logger.info("backend=%s succeeded, chars=%d", backend.name, len(result))
        return ExtractionAttempt(backend=backend.name, content=result)

    return ExtractionAttempt(
        backend="failed",
        error=f"all backends failed: {'; '.join(errors)}",
    )
```

- [ ] **Step 4: Test'leri çalıştır, PASS doğrula**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/test_ocr_backends.py -v
```
Beklenen: 10 test PASS (5 önceki + 5 yeni).

---

## Task 4: LightOCRBackend

**Files:**
- Modify: `ocr_backends.py` (LightOCRBackend sınıfı)
- Modify: `tests/test_ocr_backends.py` (LightOCR testleri)

- [ ] **Step 1: Failing testler ekle — `tests/test_ocr_backends.py`**

```python
from unittest.mock import MagicMock, patch

from ocr_backends import LightOCRBackend


class TestLightOCRBackend:
    def test_name(self):
        backend = LightOCRBackend()
        assert backend.name == "lightocr"

    def test_is_available_false_when_no_cuda(self):
        backend = LightOCRBackend()
        with patch("ocr_backends._cuda_available", return_value=False):
            assert backend.is_available() is False

    def test_is_available_false_when_transformers_missing(self):
        backend = LightOCRBackend()
        with patch("ocr_backends._cuda_available", return_value=True):
            with patch("ocr_backends._transformers_available", return_value=False):
                assert backend.is_available() is False

    def test_is_available_true_when_all_present(self):
        backend = LightOCRBackend()
        with patch("ocr_backends._cuda_available", return_value=True):
            with patch("ocr_backends._transformers_available", return_value=True):
                assert backend.is_available() is True

    def test_extract_returns_none_on_empty_pdf(self):
        backend = LightOCRBackend()
        assert backend.extract(b"") is None

    def test_extract_loads_model_lazily(self):
        backend = LightOCRBackend()
        assert backend._model is None  # not loaded at init
        with patch("ocr_backends._cuda_available", return_value=True):
            with patch("ocr_backends._transformers_available", return_value=True):
                mock_model = MagicMock()
                mock_model.generate_markdown.return_value = "extracted"
                with patch.object(backend, "_load_model", return_value=mock_model):
                    result = backend.extract(b"%PDF-1.4\nfake")
                    assert result == "extracted"
```

- [ ] **Step 2: Testleri çalıştır, fail doğrula**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/test_ocr_backends.py::TestLightOCRBackend -v
```
Beklenen: `ImportError` veya `AttributeError: module 'ocr_backends' has no attribute 'LightOCRBackend'`.

- [ ] **Step 3: `ocr_backends.py`'e LightOCRBackend ekle**

Dosyanın üstündeki import'ların altına helper'ları ekle:

```python
def _cuda_available() -> bool:
    """Isolated for test mocking."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _transformers_available() -> bool:
    """Isolated for test mocking."""
    try:
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False
```

MarkitdownBackend sınıfından sonra LightOCRBackend ekle. Not: aşağıda `model_obj.eval()` PyTorch modelini evaluation moduna alır (arbitrary-code execution değil, inference için standart çağrı):

```python
# ─── LightOnOCR-2-1B backend (GPU, primary) ─────────────────────────────────


class LightOCRBackend:
    """Primary extractor: LightOnOCR-2-1B on CUDA.

    Loads model lazily on first extract() call. Model stays warm for the
    lifetime of the backend instance.
    """

    name = "lightocr"

    def __init__(
        self,
        model_name: str = "",
        model_path: str = "",
        batch_size: int = 4,
        device: str = "auto",
    ) -> None:
        from config import (
            LIGHTOCR_BATCH_SIZE,
            LIGHTOCR_DEVICE,
            LIGHTOCR_MODEL_NAME,
            LIGHTOCR_MODEL_PATH,
        )

        self._model_name = model_name or LIGHTOCR_MODEL_NAME
        self._model_path = model_path or LIGHTOCR_MODEL_PATH
        self._batch_size = batch_size or LIGHTOCR_BATCH_SIZE
        self._device_pref = device or LIGHTOCR_DEVICE
        self._model = None

    def is_available(self) -> bool:
        if not _cuda_available() and self._device_pref != "cpu":
            return False
        if not _transformers_available():
            return False
        return True

    def _load_model(self):
        """Load LightOnOCR-2-1B. Called once on first extract()."""
        import torch
        from transformers import AutoModel, AutoProcessor

        source = self._model_path or self._model_name
        device = "cuda" if self._device_pref == "auto" and torch.cuda.is_available() else self._device_pref

        logger.info("Loading LightOnOCR from %s onto %s", source, device)
        processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
        model_obj = AutoModel.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model_obj = model_obj.to(device)
        # Put model in inference mode (standard PyTorch call, not code execution)
        model_obj.eval()

        class _Wrapper:
            def __init__(self, model, processor, batch_size, device):
                self.model = model
                self.processor = processor
                self.batch_size = batch_size
                self.device = device

            def generate_markdown(self, pdf_bytes: bytes) -> str | None:
                from pdf2image import convert_from_bytes

                images = convert_from_bytes(pdf_bytes, dpi=200)
                if not images:
                    return None
                pages: list[str] = []
                for i in range(0, len(images), self.batch_size):
                    batch = images[i : i + self.batch_size]
                    inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        outputs = self.model.generate(**inputs, max_new_tokens=4096)
                    decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)
                    pages.extend(decoded)
                return "\n\n".join(p.strip() for p in pages if p.strip())

        logger.info("LightOnOCR loaded, VRAM cached: %.1f GB", torch.cuda.memory_allocated() / 1e9)
        return _Wrapper(model_obj, processor, self._batch_size, device)

    def extract(self, pdf_bytes: bytes) -> str | None:
        if not pdf_bytes:
            return None
        try:
            if self._model is None:
                self._model = self._load_model()
            return self._model.generate_markdown(pdf_bytes)
        except Exception as e:
            logger.warning("LightOCR extraction failed: %s", e)
            return None
```

- [ ] **Step 4: Testleri çalıştır, PASS doğrula**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/test_ocr_backends.py -v
```
Beklenen: 16 test PASS.

---

## Task 5: PPStructureBackend (fallback)

**Files:**
- Modify: `ocr_backends.py` (PPStructureBackend sınıfı)
- Modify: `tests/test_ocr_backends.py` (PPStructure testleri)

- [ ] **Step 1: Failing testler ekle**

```python
from ocr_backends import PPStructureBackend


class TestPPStructureBackend:
    def test_name(self):
        backend = PPStructureBackend()
        assert backend.name == "pp_structure"

    def test_is_available_false_when_paddleocr_missing(self):
        backend = PPStructureBackend()
        with patch("ocr_backends._paddleocr_available", return_value=False):
            assert backend.is_available() is False

    def test_extract_returns_none_on_empty(self):
        backend = PPStructureBackend()
        assert backend.extract(b"") is None
```

- [ ] **Step 2: Test'leri çalıştır, fail doğrula**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/test_ocr_backends.py::TestPPStructureBackend -v
```

- [ ] **Step 3: `ocr_backends.py`'e PPStructureBackend ekle**

Helper eklemesi (diğer `_*_available` fonksiyonlarının yanına):

```python
def _paddleocr_available() -> bool:
    try:
        import paddleocr  # noqa: F401

        return True
    except ImportError:
        return False
```

LightOCRBackend sınıfından sonra:

```python
# ─── PP-StructureV3 backend (GPU, fallback) ─────────────────────────────────


class PPStructureBackend:
    """Fallback extractor: PaddleOCR PP-StructureV3. Fast, strong on formulas."""

    name = "pp_structure"

    def __init__(self) -> None:
        self._engine = None

    def is_available(self) -> bool:
        return _paddleocr_available()

    def _load_engine(self):
        from paddleocr import PPStructureV3

        logger.info("Loading PP-StructureV3")
        return PPStructureV3(use_gpu=_cuda_available())

    def extract(self, pdf_bytes: bytes) -> str | None:
        if not pdf_bytes:
            return None
        try:
            from pdf2image import convert_from_bytes

            if self._engine is None:
                self._engine = self._load_engine()
            images = convert_from_bytes(pdf_bytes, dpi=200)
            pages: list[str] = []
            for img in images:
                result = self._engine.predict(img)
                # PP-StructureV3 predict returns list[dict] with 'markdown' key
                for item in result or []:
                    md = item.get("markdown") if isinstance(item, dict) else None
                    if md:
                        pages.append(md)
            return "\n\n".join(pages) if pages else None
        except Exception as e:
            logger.warning("PPStructure extraction failed: %s", e)
            return None
```

- [ ] **Step 4: Testleri çalıştır, PASS doğrula**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/test_ocr_backends.py -v
```
Beklenen: 19 test PASS.

---

## Task 6: Default backend factory

**Files:**
- Modify: `ocr_backends.py` (`get_default_backends` fonksiyonu)
- Modify: `tests/test_ocr_backends.py`

- [ ] **Step 1: Failing test ekle**

```python
from ocr_backends import get_default_backends


class TestDefaultBackends:
    def test_default_order_is_lightocr_pp_markitdown(self):
        backends = get_default_backends()
        names = [b.name for b in backends]
        assert names == ["lightocr", "pp_structure", "markitdown_degraded"]
```

- [ ] **Step 2: Test'i çalıştır, fail doğrula**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/test_ocr_backends.py::TestDefaultBackends -v
```

- [ ] **Step 3: `ocr_backends.py` sonuna ekle**

```python
def get_default_backends() -> list[OCRBackend]:
    """Return backend chain in preference order (lightocr → pp → markitdown)."""
    return [LightOCRBackend(), PPStructureBackend(), MarkitdownBackend()]
```

- [ ] **Step 4: Test'i çalıştır, PASS doğrula**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/test_ocr_backends.py -v
```
Beklenen: 20 test PASS.

---

## Task 7: DocumentSyncer'ı backend zincirine entegre et

**Files:**
- Modify: `doc_sync.py:94-138` (Nougat extraction yolu silinir)
- Modify: `doc_sync.py:284-324` (`DocumentSyncer.__init__`)
- Modify: `doc_sync.py:549-610` (`_extract`, `_extract_structured`)
- Modify: `tests/test_doc_sync.py` (mevcut testler güncellenir)

- [ ] **Step 1: `doc_sync.py` üstüne import ekle**

Dosyanın üstündeki import bloğuna:

```python
from config import BASE_DIR, OCR_MIN_CONTENT_LEN, REQUEST_TIMEOUT
from ocr_backends import OCRBackend, get_default_backends, run_extraction_chain
```

(Mevcut `from config import BASE_DIR, REQUEST_TIMEOUT` satırını yenisiyle değiştir.)

- [ ] **Step 2: Nougat extraction fonksiyonunu tamamen sil**

`doc_sync.py:94-138` aralığındaki `def _extract_with_nougat(...)` fonksiyonunu sil.

- [ ] **Step 3: `DocumentSyncer.__init__` imzasını güncelle**

```python
def __init__(
    self,
    store: DocumentStore,
    request_timeout: float = REQUEST_TIMEOUT,
    ocr_backends: "list[OCRBackend] | None" = None,
    progress_callback: "Callable[[str, int, int], None] | None" = None,
    http: httpx.AsyncClient | None = None,
) -> None:
    self._store = store
    self._ocr_backends = ocr_backends if ocr_backends is not None else get_default_backends()
    self._progress_callback = progress_callback
    self._owns_http = http is None
    # ... (mevcut http kurulum kodu aynı kalır)
```

`prefer_nougat` parametresini tamamen kaldır.

- [ ] **Step 4: `_extract_structured` fonksiyonunu sadeleştir**

Eski implementasyon (`doc_sync.py:564-610`) yerine:

```python
def _extract_structured(self, content: bytes, ext: str) -> ExtractionResult:
    """Extract markdown. PDFs → OCR backend chain. HTML → existing HTML parser."""
    if ext == ".pdf":
        attempt = run_extraction_chain(content, self._ocr_backends, min_len=OCR_MIN_CONTENT_LEN)
        if attempt.backend != "failed":
            return ExtractionResult(content=attempt.content, method=attempt.backend)
        return ExtractionResult(method="failed", error=attempt.error, retryable=False)

    if ext in (".html", ".htm"):
        html_str = _decode_html(content)
        result = _extract_html_to_markdown(html_str)
        if result and not _is_error_page(result):
            return ExtractionResult(content=result, method="html_parser")
        # Fallback: try markitdown on HTML
        try:
            from markitdown import MarkItDown

            md = MarkItDown()
            html_result = md.convert_stream(io.BytesIO(content), file_extension=".html").text_content.strip()
            if html_result and not _is_error_page(html_result):
                return ExtractionResult(content=html_result, method="markitdown")
        except (ValueError, OSError, UnicodeDecodeError):
            pass
        return ExtractionResult(method="failed", error="html_parser and markitdown both failed", retryable=True)

    if ext == ".doc":
        # .doc files still go through markitdown (no OCR backend handles DOC directly)
        from ocr_backends import _run_markitdown

        result = _run_markitdown(content)
        if result and len(result) >= OCR_MIN_CONTENT_LEN:
            return ExtractionResult(content=result, method="markitdown")
        return ExtractionResult(method="failed", error="markitdown failed on .doc", retryable=True)

    return ExtractionResult(method="failed", error=f"Unsupported extension: {ext}", retryable=False)
```

Not: `_extract_with_markitdown` fonksiyonu doc_sync.py:141-152 arasında bulunuyor. PDF yolu artık ocr_backends'ten geçtiği için bu helper fonksiyon artık kullanılmıyor — silebilirsin.

- [ ] **Step 5: Mevcut testleri çalıştır**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/test_doc_sync.py -v
```
Beklenen: bazı testler başarısız olabilir (prefer_nougat kwarg'ı kaldırıldı). Aşağıdaki adımda düzelt.

- [ ] **Step 6: `tests/test_doc_sync.py` — `prefer_nougat` referanslarını temizle**

Dosyada `prefer_nougat=False` veya `prefer_nougat=True` geçen yerleri Grep ile bul. Her birini `ocr_backends=[]` (boş liste — sadece markitdown fallback) veya test senaryosuna uygun bir fake backend listesi ile değiştir.

Örnek:
```python
# Öncesi:
async with DocumentSyncer(store, prefer_nougat=False) as syncer:

# Sonrası:
from ocr_backends import MarkitdownBackend
async with DocumentSyncer(store, ocr_backends=[MarkitdownBackend()]) as syncer:
```

- [ ] **Step 7: CLI `--no-nougat` bayrağını kaldır**

`doc_sync.py:804` civarında:

```python
sync_p.add_argument("--no-nougat", action="store_true", help="Skip Nougat, use markitdown")
```

satırını sil. `_cli_sync`'te `prefer_nougat=not args.no_nougat` referansını da kaldır. Yerine default backend zincirini kullan (parametre geçme).

- [ ] **Step 8: Tüm testleri çalıştır**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/ -v --tb=short
```
Beklenen: tümü PASS.

---

## Task 8: Force re-extract'te veri kaybı koruması

**Files:**
- Modify: `doc_sync.py:376-389` (sync_document'teki delete_document çağrısı)
- Modify: `tests/test_doc_sync.py` (yeni regression testi)

- [ ] **Step 1: Failing test ekle — `tests/test_doc_sync.py`**

```python
import pytest

from doc_sync import DocumentSyncer
from ocr_backends import MarkitdownBackend


@pytest.mark.asyncio
async def test_force_reextract_failure_preserves_old_content(doc_store):
    """When force=True and new extraction fails, old markdown must remain in DB."""
    # Seed DB with a working document
    from doc_store import StoredDocument

    original = StoredDocument(
        document_id="42628",
        title="Test doc",
        markdown_content="ORIGINAL CONTENT",
        extraction_method="lightocr",
    )
    await doc_store.store_document(original)

    # Create a syncer whose backend always returns None (simulates extraction failure)
    class _AlwaysFailBackend:
        name = "test_fail"

        def is_available(self) -> bool:
            return True

        def extract(self, pdf_bytes: bytes) -> str | None:
            return None

    async with DocumentSyncer(doc_store, ocr_backends=[_AlwaysFailBackend()]) as syncer:
        result = await syncer.sync_document(doc_id="42628", force=True)

    assert result.success is False
    # Critical: original content must still be in DB
    stored = await doc_store.get_document("42628")
    assert stored is not None
    assert stored.markdown_content == "ORIGINAL CONTENT"
```

- [ ] **Step 2: Test'i çalıştır, fail doğrula**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/test_doc_sync.py::test_force_reextract_failure_preserves_old_content -v
```
Beklenen: FAIL (şu an `delete_document` çağrılıyor — eski içerik silinir).

- [ ] **Step 3: `doc_sync.py:376-389` — `delete_document` çağrısını kaldır**

Mevcut kod (376-389 civarı):

```python
if not markdown:
    error_msg = f"Extraction failed (method={extraction_method})"
    cat, retryable = _categorize_error(error_msg)
    await self._store.record_sync_failure(doc_id, error_msg, cat, source_url, retryable)
    # Clear corrupted content so has_document() returns False on next sync
    if force:
        await self._store.delete_document(doc_id)
        logger.info("Cleared corrupted content for %s", doc_id)
    return SyncResult(
        document_id=doc_id,
        success=False,
        error=error_msg,
    )
```

Yerine:

```python
if not markdown:
    error_msg = f"Extraction failed (method={extraction_method})"
    cat, retryable = _categorize_error(error_msg)
    await self._store.record_sync_failure(doc_id, error_msg, cat, source_url, retryable)
    # DO NOT delete old content on failed force re-extract — data loss protection
    logger.warning(
        "Extraction failed for %s; preserving old content (force=%s)", doc_id, force
    )
    return SyncResult(
        document_id=doc_id,
        success=False,
        error=error_msg,
    )
```

- [ ] **Step 4: Test'i çalıştır, PASS doğrula**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/test_doc_sync.py::test_force_reextract_failure_preserves_old_content -v
```

---

## Task 9: Mevzuat download sıralamasını değiştir (PDF önce)

**Files:**
- Modify: `doc_sync.py:424-545` (`_download_mevzuat`)
- Modify: `tests/test_doc_sync.py` (sıralama testi)

- [ ] **Step 1: Failing test ekle — `tests/test_doc_sync.py`**

```python
import httpx
import pytest
from pytest_httpx import HTTPXMock


@pytest.mark.asyncio
async def test_mevzuat_download_tries_pdf_before_htm(httpx_mock: HTTPXMock, doc_store):
    """_download_mevzuat must attempt PDF paths before HTML to preserve formulas."""
    # Main page visit (required for GeneratePdf cookies)
    httpx_mock.add_response(
        url="https://www.mevzuat.gov.tr/mevzuat?MevzuatNo=42628&MevzuatTur=7&MevzuatTertip=5",
        content=b"<html><body>main page</body></html>",
    )
    # GeneratePdf returns a PDF first
    httpx_mock.add_response(
        url="https://www.mevzuat.gov.tr/File/GeneratePdf?mevzuatNo=42628&mevzuatTur=Yonetmelik&mevzuatTertip=5",
        content=b"%PDF-1.4\n" + b"x" * 1000,
    )

    async with DocumentSyncer(doc_store, ocr_backends=[MarkitdownBackend()]) as syncer:
        content, method, ext = await syncer._download_mevzuat("mevzuat_42628")

    assert ext == ".pdf"
    assert method == "mevzuat_generate_pdf"
    assert content.startswith(b"%PDF-")
```

- [ ] **Step 2: Test'i çalıştır, fail doğrula**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/test_doc_sync.py::test_mevzuat_download_tries_pdf_before_htm -v
```
Beklenen: FAIL (şu an .htm önce deneniyor).

- [ ] **Step 3: `_download_mevzuat` — katmanları yeniden sırala**

`doc_sync.py:455-545` arasındaki for loop'u yeniden düzenle. Yeni sıra:

1. Main page visit (cookie için — her zaman ilk)
2. GeneratePdf API
3. Direct .pdf
4. .htm (fallback — formül kayıp riski)
5. iframe/div extraction
6. .doc

Replace block (for loop içinde):

```python
for candidate_tur in tur_candidates:
    segment = MEVZUAT_TUR_MAP.get(candidate_tur, "yonetmelik")
    base = f"{candidate_tur}.{tertip}.{mevzuat_no}"
    layer_timeout = httpx.Timeout(30.0, connect=10.0)

    # Layer 1: Main page visit — establishes session cookies for GeneratePdf
    main_url = f"https://www.mevzuat.gov.tr/mevzuat?MevzuatNo={mevzuat_no}&MevzuatTur={candidate_tur}&MevzuatTertip={tertip}"
    main_page_visited = False
    main_page_html = ""
    try:
        resp = await self._http.get(main_url, timeout=layer_timeout)
        if resp.status_code == 200:
            main_page_visited = True
            main_page_html = resp.text
    except Exception as e:
        logger.debug("mevzuat %s: main page visit failed (tur=%s): %s", doc_id, candidate_tur, e)

    # Layer 2: GeneratePdf API (preferred — server-rendered PDF with all formulas as images)
    if main_page_visited:
        try:
            gen_pdf_url = _mevzuat_generate_pdf_url(mevzuat_no, candidate_tur, tertip)
            if gen_pdf_url:
                resp = await self._http.get(gen_pdf_url, timeout=layer_timeout)
                if resp.status_code == 200 and len(resp.content) > 500 and resp.content[:5] == b"%PDF-":
                    logger.info("mevzuat %s: downloaded via GeneratePdf (tur=%s)", doc_id, candidate_tur)
                    return resp.content, "mevzuat_generate_pdf", ".pdf"
        except Exception as e:
            logger.debug("mevzuat %s: GeneratePdf failed (tur=%s): %s", doc_id, candidate_tur, e)

    # Layer 3: Direct static .pdf
    try:
        pdf_url = _mevzuat_pdf_url(mevzuat_no, candidate_tur, tertip)
        if pdf_url:
            resp = await self._http.get(pdf_url, timeout=layer_timeout)
            if resp.status_code == 200 and len(resp.content) > 500 and resp.content[:5] == b"%PDF-":
                logger.info("mevzuat %s: downloaded via .pdf (tur=%s)", doc_id, candidate_tur)
                return resp.content, "mevzuat_pdf", ".pdf"
    except Exception as e:
        logger.debug("mevzuat %s: .pdf failed (tur=%s): %s", doc_id, candidate_tur, e)

    # Layer 4: .htm (fallback — formulas may be lost)
    try:
        htm_url = f"https://www.mevzuat.gov.tr/mevzuatmetin/{segment}/{base}.htm"
        resp = await self._http.get(htm_url, timeout=layer_timeout)
        if resp.status_code == 200 and len(resp.content) > 200 and not _is_error_page(resp.text):
            logger.warning(
                "mevzuat %s: falling back to .htm (tur=%s) — formulas may be lost",
                doc_id, candidate_tur,
            )
            return resp.content, "mevzuat_htm", ".html"
    except Exception as e:
        logger.debug("mevzuat %s: .htm failed (tur=%s): %s", doc_id, candidate_tur, e)

    # Layer 5: iframe/div from already-fetched main page
    if main_page_visited and main_page_html:
        try:
            soup = BeautifulSoup(main_page_html, "html.parser")
            iframe = soup.find("iframe", src=True)
            if iframe:
                iframe_url = iframe["src"]
                if not iframe_url.startswith("http"):
                    iframe_url = f"https://www.mevzuat.gov.tr{iframe_url}"
                iframe_resp = await self._http.get(iframe_url, timeout=layer_timeout)
                if iframe_resp.status_code == 200 and len(iframe_resp.content) > 200:
                    logger.warning(
                        "mevzuat %s: falling back to iframe (tur=%s)", doc_id, candidate_tur
                    )
                    return iframe_resp.content, "mevzuat_iframe", ".html"
            div = soup.find("div", id="divMevzuatMetni")
            if div and len(div.get_text(strip=True)) > 100:
                logger.warning(
                    "mevzuat %s: falling back to main page div (tur=%s)", doc_id, candidate_tur
                )
                return str(div).encode("utf-8"), "mevzuat_div", ".html"
        except Exception as e:
            logger.debug("mevzuat %s: iframe/div parse failed (tur=%s): %s", doc_id, candidate_tur, e)

    # Layer 6: .doc (heaviest, only for first/default tur)
    if candidate_tur == tur:
        try:
            doc_url = _mevzuat_doc_url(mevzuat_no, candidate_tur, tertip)
            resp = await self._http.get(doc_url, timeout=httpx.Timeout(90.0, connect=15.0))
            if (
                resp.status_code == 200
                and len(resp.content) > 100
                and resp.content[:4] in (b"\xd0\xcf\x11\xe0", b"PK\x03\x04")
            ):
                logger.info("mevzuat %s: downloaded via .doc (tur=%s)", doc_id, candidate_tur)
                return resp.content, "mevzuat_doc", ".doc"
        except Exception as e:
            logger.debug("mevzuat %s: .doc failed (tur=%s): %s", doc_id, candidate_tur, e)

    if candidate_tur != tur_candidates[-1]:
        logger.debug("mevzuat %s: tur=%s failed, trying next candidate", doc_id, candidate_tur)
```

- [ ] **Step 4: Test'i çalıştır, PASS doğrula**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/test_doc_sync.py::test_mevzuat_download_tries_pdf_before_htm -v
```

- [ ] **Step 5: Tam test suitini çalıştır (regresyon kontrolü)**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/ -v --tb=short
```
Beklenen: tümü PASS.

---

## Task 10: `DocumentStore.get_pdf_bytes` helper

**Files:**
- Modify: `doc_store.py` (`get_pdf_bytes` metodu)
- Modify: `doc_sync.py` (sync_document başında cached PDF kontrolü)
- Modify: `tests/test_doc_store.py` (yeni test)

- [ ] **Step 1: Failing test ekle — `tests/test_doc_store.py`**

```python
import pytest


@pytest.mark.asyncio
async def test_get_pdf_bytes_returns_stored_bytes(doc_store):
    from doc_store import StoredDocument

    doc = StoredDocument(
        document_id="test_42628",
        title="test",
        pdf_bytes=b"%PDF-1.4\nfake content",
        markdown_content="x",
    )
    await doc_store.store_document(doc)

    result = await doc_store.get_pdf_bytes("test_42628")
    assert result == b"%PDF-1.4\nfake content"


@pytest.mark.asyncio
async def test_get_pdf_bytes_none_for_missing(doc_store):
    result = await doc_store.get_pdf_bytes("nonexistent")
    assert result is None
```

- [ ] **Step 2: Test'i çalıştır, fail doğrula**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/test_doc_store.py::test_get_pdf_bytes_returns_stored_bytes -v
```

- [ ] **Step 3: `doc_store.py`'e method ekle**

`DocumentStore` sınıfı içine (uygun bir yere, `store_document`'ın yakınına):

```python
async def get_pdf_bytes(self, document_id: str) -> bytes | None:
    """Return cached PDF bytes for a document, or None if absent."""
    async with self._pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT pdf_bytes FROM documents WHERE document_id = $1",
            document_id,
        )
    if row is None or row["pdf_bytes"] is None:
        return None
    return bytes(row["pdf_bytes"])
```

- [ ] **Step 4: `doc_sync.py` — force=True'da DB'den PDF dene (opsiyonel bandwidth tasarrufu)**

`sync_document` başında, `has_document` kontrolünden hemen sonra, force=True durumunda:

```python
# If we're re-extracting with force=True and the PDF is already cached in DB,
# skip re-downloading (bandwidth saving).
cached_pdf = None
if force and doc_id.startswith("mevzuat_"):
    cached_pdf = await self._store.get_pdf_bytes(doc_id)
```

Sonra `try` bloğunda:

```python
try:
    if cached_pdf:
        content, method, ext = cached_pdf, "cached_pdf", ".pdf"
    elif doc_id.startswith("mevzuat_"):
        content, method, ext = await self._download_mevzuat(doc_id, source_url)
    elif doc_id.isdigit():
        content, method, ext = await self._download_bddk(doc_id)
    else:
        return SyncResult(...)
```

- [ ] **Step 5: Testleri çalıştır**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/ -v --tb=short
```

---

## Task 11: Integration testi (GPU gerekli, CI skip)

**Files:**
- Create: `tests/test_integration_lightocr.py`
- Modify: `pyproject.toml` (pytest mark kaydı)

- [ ] **Step 1: `pyproject.toml`'a `gpu` marker tanımı ekle**

`[tool.pytest.ini_options]` bölümüne:

```toml
markers = [
    "gpu: requires CUDA GPU (skipped in CI)",
]
```

- [ ] **Step 2: `tests/test_integration_lightocr.py` oluştur**

```python
"""GPU integration tests for LightOnOCR on real BDDK documents.

Skipped in CI (no GPU). Run locally: pytest tests/test_integration_lightocr.py -m gpu
"""

import pytest

from ocr_backends import LightOCRBackend


pytestmark = pytest.mark.gpu


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="module")
def backend() -> LightOCRBackend:
    if not _cuda_available():
        pytest.skip("CUDA not available")
    return LightOCRBackend()


@pytest.fixture(scope="module")
def pdf_42628() -> bytes:
    """Fetch the GeneratePdf for mevzuat_42628 once per test module."""
    import httpx

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        # Main page visit for cookies
        client.get(
            "https://www.mevzuat.gov.tr/mevzuat?MevzuatNo=42628&MevzuatTur=7&MevzuatTertip=5"
        )
        resp = client.get(
            "https://www.mevzuat.gov.tr/File/GeneratePdf?mevzuatNo=42628&mevzuatTur=Yonetmelik&mevzuatTertip=5"
        )
    assert resp.status_code == 200
    assert resp.content[:5] == b"%PDF-"
    return resp.content


def test_42628_ek2_formulas_extracted(backend: LightOCRBackend, pdf_42628: bytes):
    """EK-2'de formül sembollerinden en az biri görünmeli."""
    markdown = backend.extract(pdf_42628)
    assert markdown is not None, "LightOCR returned None"
    assert len(markdown) > 5000, f"Output too short: {len(markdown)} chars"

    lower = markdown.lower()
    # At least one formula indicator must appear
    formula_markers = ["$", "\\frac", "t_0", "t0", "paralel yukarı"]
    hits = [m for m in formula_markers if m in lower]
    assert hits, f"No formula markers found in output. First 2000 chars:\n{markdown[:2000]}"


def test_turkish_chars_preserved(backend: LightOCRBackend, pdf_42628: bytes):
    """Türkçe karakterler (ç, ğ, ı, ş, ö, ü) bozulmamalı."""
    markdown = backend.extract(pdf_42628)
    assert markdown is not None
    # Common Turkish words from the document
    assert "Yönetmelik" in markdown or "yönetmelik" in markdown.lower()
    assert any(c in markdown for c in "çğıöşü")
```

- [ ] **Step 3: Testi çalıştır**

```bash
cd /home/cagatay/bddk-mcp && uv run pytest tests/test_integration_lightocr.py -v -m gpu
```
Beklenen: 2 test PASS. İlk çalıştırmada model ~2GB indirir (normal).

---

## Task 12: 42628 manuel doğrulama

**Files:** yok (CLI + DB sorguları)

- [ ] **Step 1: DB'den mevcut içerik backup al**

```bash
cd /home/cagatay/bddk-mcp && uv run python -c "
import asyncio, asyncpg
from config import DATABASE_URL

async def main():
    conn = await asyncpg.connect(DATABASE_URL)
    row = await conn.fetchrow(
        'SELECT markdown_content, extraction_method FROM documents WHERE document_id=\$1',
        'mevzuat_42628',
    )
    if row:
        with open('/tmp/42628_before.md', 'w') as f:
            f.write(row['markdown_content'])
        print(f'Backed up {len(row[\"markdown_content\"])} chars, method={row[\"extraction_method\"]}')
    await conn.close()

asyncio.run(main())
"
```
Beklenen: `/tmp/42628_before.md` oluşur, method muhtemelen `html_parser` veya `mevzuat_htm`.

- [ ] **Step 2: Force sync çalıştır**

```bash
cd /home/cagatay/bddk-mcp && uv run python doc_sync.py sync --doc-id mevzuat_42628 --force
```
Beklenen çıktı: `[OK] mevzuat_42628: mevzuat_generate_pdf+lightocr`

- [ ] **Step 3: Yeni içeriği dışa aktar**

```bash
cd /home/cagatay/bddk-mcp && uv run python -c "
import asyncio, asyncpg
from config import DATABASE_URL

async def main():
    conn = await asyncpg.connect(DATABASE_URL)
    row = await conn.fetchrow(
        'SELECT markdown_content, extraction_method, file_size FROM documents WHERE document_id=\$1',
        'mevzuat_42628',
    )
    with open('/tmp/42628_after.md', 'w') as f:
        f.write(row['markdown_content'])
    print(f'New: {len(row[\"markdown_content\"])} chars, method={row[\"extraction_method\"]}, pdf_size={row[\"file_size\"]}')
    await conn.close()

asyncio.run(main())
"
```
Beklenen: method=`lightocr`, chars yeni ≥ chars eski (formüller eklendi).

- [ ] **Step 4: EK-2 bölümünü elle kontrol et**

```bash
grep -A 20 "EK-2\|EK 2" /tmp/42628_after.md | head -50
```
Beklenen: formül işaretleri (`$`, `\frac`, `t_0`, veya Unicode formül sembolleri) görünür.

- [ ] **Step 5: Regresyon spot check — 10 rastgele mevzuat**

```bash
cd /home/cagatay/bddk-mcp && uv run python -c "
import asyncio, asyncpg
from config import DATABASE_URL

async def main():
    conn = await asyncpg.connect(DATABASE_URL)
    rows = await conn.fetch(
        'SELECT document_id, LENGTH(markdown_content) AS sz, extraction_method'
        ' FROM documents WHERE document_id LIKE \$1 ORDER BY random() LIMIT 10',
        'mevzuat_%',
    )
    for r in rows:
        print(f'{r[\"document_id\"]}: {r[\"sz\"]} chars, method={r[\"extraction_method\"]}')
    await conn.close()

asyncio.run(main())
"
```
Beklenen: hiçbirinde içerik boşalmamış (hepsi > 500 karakter) — veri kaybı koruması çalışmış.

---

## Task 13: Faz 1 toplu geri-dolum

**Files:** yok (CLI)

- [ ] **Step 1: Hedef set sorgusu**

```bash
cd /home/cagatay/bddk-mcp && uv run python -c "
import asyncio, asyncpg
from config import DATABASE_URL

async def main():
    conn = await asyncpg.connect(DATABASE_URL)
    rows = await conn.fetch('''
        SELECT document_id FROM documents
        WHERE document_id LIKE 'mevzuat_%'
          AND extraction_method IN ('html_parser', 'markitdown', 'mevzuat_htm')
        ORDER BY document_id
    ''')
    ids = [r['document_id'] for r in rows]
    print(f'Target count: {len(ids)}')
    with open('/tmp/resync_targets.txt', 'w') as f:
        for i in ids:
            f.write(i + '\n')
    await conn.close()

asyncio.run(main())
"
```
Beklenen: hedef sayısı yazdırılır, `/tmp/resync_targets.txt` doldurulur.

- [ ] **Step 2: İlk 10 belge ile zaman ölçümü**

```bash
cd /home/cagatay/bddk-mcp && head -10 /tmp/resync_targets.txt | while read id; do
    time uv run python doc_sync.py sync --doc-id "$id" --force
done 2>&1 | tee /tmp/resync_timing.log
```
Beklenen: her belge 1-5 dk arası; toplam extrapolasyon ile tüm süre tahmini çıkar.

- [ ] **Step 3: Kalan belgeleri sync et**

```bash
cd /home/cagatay/bddk-mcp && tail -n +11 /tmp/resync_targets.txt | while read id; do
    uv run python doc_sync.py sync --doc-id "$id" --force
done 2>&1 | tee -a /tmp/resync_bulk.log
```
Not: `sync --force` (--doc-id olmadan) tüm cache'i yeniden syncler — bizim ihtiyacımız sadece belirli bir subset, bu yüzden tek tek loop doğru yaklaşım.

- [ ] **Step 4: Sonuç raporu**

```bash
cd /home/cagatay/bddk-mcp && uv run python -c "
import asyncio, asyncpg
from config import DATABASE_URL

async def main():
    conn = await asyncpg.connect(DATABASE_URL)
    stats = await conn.fetch('''
        SELECT extraction_method, COUNT(*) AS cnt, AVG(LENGTH(markdown_content)) AS avg_chars
        FROM documents WHERE document_id LIKE 'mevzuat_%'
        GROUP BY extraction_method ORDER BY cnt DESC
    ''')
    for s in stats:
        print(f'{s[\"extraction_method\"]}: {s[\"cnt\"]} docs, avg {int(s[\"avg_chars\"])} chars')
    await conn.close()

asyncio.run(main())
"
```
Beklenen: çoğunluğu `lightocr`; `html_parser`/`mevzuat_htm` sayısı küçük (sadece PDF alamayan belgeler).

- [ ] **Step 5: Seed export**

```bash
cd /home/cagatay/bddk-mcp && uv run python seed.py export
```
Beklenen: `seed_data/` altındaki JSON dosyaları güncellenir. Git commit edilmez (memory kuralı).

---

## Self-Review Notları

**Spec coverage check:**
- [x] Nougat kaldırma → Task 7 Step 2
- [x] LightOnOCR entegrasyon → Task 4
- [x] PP-StructureV3 fallback → Task 5
- [x] markitdown son çare → Task 2 (name="markitdown_degraded")
- [x] Mevzuat download sırası → Task 9
- [x] Config env var'ları → Task 1 Step 3
- [x] OCRBackend protocol → Task 2 Step 3
- [x] Data-loss koruması (force fail'de delete etme) → Task 8
- [x] get_pdf_bytes helper → Task 10
- [x] Integration testi (42628 formül kontrolü) → Task 11
- [x] Manuel doğrulama adımları → Task 12
- [x] Faz 1 rollout → Task 13
- [x] Railway scope dışı (lokal'de sync) → implicit; CLI zaten lokal'de koşar

**Ambiguity check:**
- MarkitdownBackend.name `"markitdown_degraded"` PDF yolunda; HTML yolunda `"markitdown"` → Task 7 Step 4'te inline `MarkItDown()` kullanılıyor, `extraction_method="markitdown"` dönüyor. Bu şekilde PDF/HTML ayrımı korunuyor.
- `get_pdf_bytes` Task 10'da eklendi ama mevcut `pdf_bytes` kolonu yalnızca PDF'ler için doludur, HTML/DOC için None. `cached_pdf` bu yüzden sadece PDF rotaları için mantıklı — `mevzuat_*` için `force=True` geldiğinde deneniyor.

**Type consistency check:**
- `OCRBackend.extract()` → `str | None` (her üç backend aynı imzaya sahip) ✓
- `ExtractionAttempt.backend` str, `ExtractionResult.method` str (isimler farklı ama uyumlu) ✓
- `DocumentSyncer.__init__` kwarg `ocr_backends` her yerde aynı isimle ✓

**No placeholders:** ✓ Tüm kod blokları gerçek implementasyon.
