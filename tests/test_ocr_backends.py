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

    def test_empty_form_block_triggers_fallback(self):
        degraded = "lorem " * 200 + "\n<form>\n</form>\n" + "ipsum " * 200
        assert len(degraded) >= 500  # sanity: would pass length check alone
        b1 = _FakeBackend("first", degraded)
        b2 = _FakeBackend("second", "y" * 600)
        attempt = run_extraction_chain(b"pdf", [b1, b2], min_len=500)
        assert attempt.backend == "second"

    def test_form_guard_accepts_non_empty_form(self):
        good = "lorem " * 200 + "\n<form>E = mc^2</form>\n" + "ipsum " * 200
        b1 = _FakeBackend("first", good)
        b2 = _FakeBackend("second", "y" * 600)
        attempt = run_extraction_chain(b"pdf", [b1, b2], min_len=500)
        assert attempt.backend == "first"


from unittest.mock import MagicMock

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


from ocr_backends import get_default_backends


class TestDefaultBackends:
    def test_default_order_is_lightocr_pp_markitdown(self):
        backends = get_default_backends()
        names = [b.name for b in backends]
        assert names == ["lightocr", "pp_structure", "markitdown_degraded"]

    def test_include_chandra_prepends_chandra2(self):
        backends = get_default_backends(include_chandra=True)
        names = [b.name for b in backends]
        assert names == ["chandra2", "lightocr", "pp_structure", "markitdown_degraded"]

    def test_default_without_chandra_unchanged(self):
        backends = get_default_backends(include_chandra=False)
        names = [b.name for b in backends]
        assert names == ["lightocr", "pp_structure", "markitdown_degraded"]
