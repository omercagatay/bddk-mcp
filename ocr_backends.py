"""
OCR backend implementations for BDDK document extraction.

Pluggable Protocol-based backends in preference order:
    1. LightOCRBackend       (GPU, formula-aware, primary)
    2. MarkitdownBackend     (CPU last resort, no formulas)

Each backend owns its own model lifecycle and availability check.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_EMPTY_FORM_RE = re.compile(r"<form>\s*</form>", re.IGNORECASE)


def _content_ok(text: str, min_len: int) -> tuple[bool, str]:
    """Chain-level quality guard.

    Returns (passes, reason). reason is '' when passes is True.
    Rejects output shorter than min_len or containing empty <form></form>
    blocks (the LightOCR failure signature observed on 2026-04-17).
    """
    if len(text) < min_len:
        return False, f"output too short ({len(text)} < {min_len})"
    if _EMPTY_FORM_RE.search(text):
        return False, "contains empty <form></form> block"
    return True, ""


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


class ExtractionAttempt(BaseModel):
    """Result of a single backend extraction attempt."""

    backend: str
    content: str = ""
    error: str = ""


@runtime_checkable
class OCRBackend(Protocol):
    """Protocol for PDF -> markdown extraction backends."""

    name: str

    def is_available(self) -> bool:
        """Check if backend can run (model loaded, CUDA available, etc.)."""
        ...

    def extract(self, pdf_bytes: bytes) -> str | None:
        """Extract markdown from PDF bytes. Returns None on failure."""
        ...


# --- Markitdown backend (CPU, no formula support) ----------------------------


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


# --- Marker backend (GPU/CPU, formula-aware, primary for Turkish) ------------


class MarkerBackend:
    """Primary extractor: Marker PDF-to-Markdown converter.

    Marker uses a vision transformer for layout detection and text
    recognition. It handles formulas (LaTeX), tables, and mixed-language
    documents better than OCR-centric models like Chandra2.

    Loads models lazily on first extract().
    """

    name = "marker"

    def __init__(self) -> None:
        self._converter = None

    def is_available(self) -> bool:
        try:
            import marker  # noqa: F401

            return True
        except ImportError:
            return False

    def _load_converter(self):
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict

        logger.info("Loading Marker PDF converter...")
        artifact_dict = create_model_dict()
        return PdfConverter(artifact_dict=artifact_dict)

    @staticmethod
    def _is_formula_image(img_name: str, img) -> bool:
        """Heuristic: small, wide images are likely inline/block formulas."""
        w, h = img.size
        if h <= 0:
            return False
        ratio = w / h
        # Inline formulas: short (<60px) and wide (ratio > 2)
        # Block formulas: medium (<150px) and wide (ratio > 1.5)
        if h < 60 and ratio > 2:
            return True
        if h < 150 and ratio > 1.5:
            return True
        return False

    @staticmethod
    def _replace_image_refs(text: str, images: dict) -> str:
        """Replace markdown image refs with [FORMULA IMAGE] or [IMAGE] placeholders."""
        for img_name, img in images.items():
            escaped = re.escape(img_name)
            is_formula = MarkerBackend._is_formula_image(img_name, img)
            placeholder = f"[FORMULA IMAGE: {img_name}]" if is_formula else f"[IMAGE: {img_name}]"
            # Match markdown image syntax: ![alt](path)
            text = re.sub(rf"!\[[^\]]*\]\({escaped}\)", placeholder, text)
            # Also handle bare src references that might appear
            text = re.sub(rf"(?<!\()\b{escaped}\b(?!\))", placeholder, text)
        return text

    def extract(self, pdf_bytes: bytes) -> str | None:
        if not pdf_bytes:
            return None
        import os
        import tempfile

        tmp_path: str | None = None
        try:
            if self._converter is None:
                self._converter = self._load_converter()

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            rendered = self._converter(tmp_path)

            from marker.output import text_from_rendered

            result = text_from_rendered(rendered)
            if isinstance(result, tuple):
                text = result[0]
                images = result[2] if len(result) > 2 else {}
            else:
                text = result
                images = {}

            if images:
                formula_count = sum(1 for name, img in images.items() if self._is_formula_image(name, img))
                if formula_count:
                    logger.info("Marker found %d formula image(s) out of %d total image(s)", formula_count, len(images))
                text = self._replace_image_refs(text, images)

            return text.strip() if text else None
        except Exception as e:
            logger.warning("Marker extraction failed: %s", e)
            return None
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


# --- LightOnOCR-2-1B backend (GPU, fallback) ---------------------------------


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
        device: str = "",
    ) -> None:
        from config import (
            LIGHTOCR_DEVICE,
            LIGHTOCR_MODEL_NAME,
            LIGHTOCR_MODEL_PATH,
        )

        self._model_name = model_name or LIGHTOCR_MODEL_NAME
        self._model_path = model_path or LIGHTOCR_MODEL_PATH
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
        from transformers import AutoProcessor, LightOnOcrForConditionalGeneration

        source = self._model_path or self._model_name
        device = "cuda" if self._device_pref == "auto" and torch.cuda.is_available() else self._device_pref

        logger.info("Loading LightOnOCR from %s onto %s", source, device)
        processor = AutoProcessor.from_pretrained(source, trust_remote_code=True)
        model_obj = LightOnOcrForConditionalGeneration.from_pretrained(
            source,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model_obj = model_obj.to(device)
        # Put PyTorch model in inference mode (train(False) == eval()).
        model_obj.train(False)

        class _Wrapper:
            def __init__(self, model, processor, device):
                self.model = model
                self.processor = processor
                self.device = device

            def generate_markdown(self, pdf_bytes: bytes) -> str | None:
                from pdf2image import convert_from_bytes

                images = convert_from_bytes(pdf_bytes, dpi=200)
                if not images:
                    return None
                pages: list[str] = []
                for image in images:
                    conversation = [{"role": "user", "content": [{"type": "image", "url": image}]}]
                    inputs = self.processor.apply_chat_template(
                        conversation,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                    inputs = {
                        k: (
                            v.to(device=self.device, dtype=torch.bfloat16)
                            if hasattr(v, "is_floating_point") and v.is_floating_point()
                            else v.to(self.device)
                        )
                        for k, v in inputs.items()
                    }
                    with torch.no_grad():
                        output_ids = self.model.generate(**inputs, max_new_tokens=4096)
                    prompt_len = inputs["input_ids"].shape[1]
                    generated = output_ids[0, prompt_len:]
                    text = self.processor.decode(generated, skip_special_tokens=True)
                    if text.strip():
                        pages.append(text.strip())
                return "\n\n".join(pages) if pages else None

        logger.info("LightOnOCR loaded, VRAM cached: %.1f GB", torch.cuda.memory_allocated() / 1e9)
        return _Wrapper(model_obj, processor, device)

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
        ok, reason = _content_ok(result, min_len)
        if not ok:
            errors.append(f"{backend.name}: {reason}")
            continue

        logger.info("backend=%s succeeded, chars=%d", backend.name, len(result))
        return ExtractionAttempt(backend=backend.name, content=result)

    return ExtractionAttempt(
        backend="failed",
        error=f"all backends failed: {'; '.join(errors)}",
    )


def get_default_backends(include_chandra: bool = False) -> list[OCRBackend]:
    """Return backend chain in preference order.

    Default: [marker, lightocr, markitdown_degraded].
    With include_chandra=True: [chandra2, marker, lightocr, markitdown_degraded].
    """
    chain: list[OCRBackend] = [MarkerBackend(), LightOCRBackend(), MarkitdownBackend()]
    if include_chandra:
        from ocr_backends_chandra import ChandraBackend

        chain.insert(0, ChandraBackend())
    return chain
