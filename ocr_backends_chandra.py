"""Chandra2 OCR backend — in-process HuggingFace inference.

Runs Chandra2 via chandra-ocr's HuggingFace backend (InferenceManager).
The model is loaded lazily on the first extract() call (~100s), then
reused for the lifetime of the backend instance.

Safe to import even when torch / chandra-ocr are not installed.
"""

from __future__ import annotations

import logging
import tempfile

logger = logging.getLogger(__name__)


def _cuda_available() -> bool:
    """Isolated for test mocking."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _chandra_available() -> bool:
    """Isolated for test mocking."""
    try:
        import chandra  # noqa: F401

        return True
    except ImportError:
        return False


class ChandraBackend:
    """Primary OCR backend using in-process chandra-ocr HF inference."""

    name = "chandra2"

    def __init__(self) -> None:
        self._manager = None

    def is_available(self) -> bool:
        return _cuda_available() and _chandra_available()

    def _load_manager(self):
        from chandra.model import InferenceManager

        logger.info("Loading chandra InferenceManager(method='hf') — ~100s on first call")
        return InferenceManager(method="hf")

    def extract(self, pdf_bytes: bytes) -> str | None:
        if not pdf_bytes:
            return None
        try:
            from chandra.input import load_file
            from chandra.model.schema import BatchInputItem
        except ImportError as e:
            logger.warning("chandra2: chandra-ocr not installed: %s", e)
            return None

        try:
            if self._manager is None:
                self._manager = self._load_manager()
        except Exception as e:
            logger.warning("chandra2: model load failed: %s", e)
            return None

        with tempfile.NamedTemporaryFile(suffix=".pdf") as tf:
            tf.write(pdf_bytes)
            tf.flush()
            try:
                images = load_file(tf.name, {})
            except Exception as e:
                logger.warning("chandra2: load_file failed: %s", e)
                return None

        if not images:
            return None

        pages: list[str] = []
        for idx, img in enumerate(images, 1):
            batch = [BatchInputItem(image=img, prompt_type="ocr_layout")]
            try:
                outputs = self._manager.generate(batch)
            except Exception as e:
                logger.warning("chandra2: inference failed on page %d: %s", idx, e)
                return None
            if not outputs:
                return None
            result = outputs[0]
            if getattr(result, "error", False):
                logger.warning("chandra2: result.error on page %d", idx)
                return None
            md = (result.markdown or "").strip()
            if md:
                pages.append(md)
        return "\n\n".join(pages) if pages else None
