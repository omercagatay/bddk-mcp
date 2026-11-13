"""Unit tests for OCR backend implementations."""

from unittest.mock import MagicMock, patch

import pytest

from bddk_mcp.ocr.base import (
    ExtractionAttempt,
    LightOCRBackend,
    LightOCRPageTruncatedError,
    MarkitdownBackend,
    OCRBackend,
    PdftotextBackend,
    _generate_lightocr_markdown,
    _run_pdftotext,
    get_default_backends,
    run_extraction_chain,
)


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
        with patch("bddk_mcp.ocr.base._run_markitdown") as mock:
            mock.return_value = "extracted text"
            result = backend.extract(b"%PDF-1.4\nfake")
            assert result == "extracted text"
            mock.assert_called_once()


class TestPdftotextBackend:
    def test_name_is_pdftotext_degraded(self):
        backend = PdftotextBackend()
        assert backend.name == "pdftotext_degraded"

    def test_is_available_checks_pdftotext_binary(self):
        backend = PdftotextBackend()
        with patch("bddk_mcp.ocr.base.shutil.which", return_value="/usr/bin/pdftotext"):
            assert backend.is_available() is True
        with patch("bddk_mcp.ocr.base.shutil.which", return_value=None):
            assert backend.is_available() is False

    def test_extract_returns_none_on_empty_pdf(self):
        backend = PdftotextBackend()
        assert backend.extract(b"") is None

    def test_extract_wraps_pdftotext(self):
        backend = PdftotextBackend()
        with patch("bddk_mcp.ocr.base._run_pdftotext") as mock:
            mock.return_value = "layout text"
            result = backend.extract(b"%PDF-1.4\nfake")
            assert result == "layout text"
            mock.assert_called_once()

    def test_run_pdftotext_does_not_hold_the_temp_file_open(self):
        """Windows cannot reopen a NamedTemporaryFile that is still open, so the child must see a closed file."""
        seen = {}

        def fake_run(cmd, **kwargs):
            with open(cmd[-2], "rb") as handle:
                seen["bytes"] = handle.read()
            return MagicMock(returncode=0, stdout="layout text\n")

        with (
            patch("bddk_mcp.ocr.base.shutil.which", return_value="pdftotext"),
            patch("bddk_mcp.ocr.base.subprocess.run", side_effect=fake_run),
        ):
            assert _run_pdftotext(b"%PDF-1.4\nfake") == "layout text"
        assert seen["bytes"] == b"%PDF-1.4\nfake"


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


class TestLightOCRBackend:
    def test_name(self):
        backend = LightOCRBackend()
        assert backend.name == "lightocr"

    def test_is_available_false_when_no_cuda(self):
        backend = LightOCRBackend()
        with patch("bddk_mcp.ocr.base._cuda_available", return_value=False):
            assert backend.is_available() is False

    def test_is_available_false_when_transformers_missing(self):
        backend = LightOCRBackend()
        with patch("bddk_mcp.ocr.base._cuda_available", return_value=True):
            with patch("bddk_mcp.ocr.base._transformers_available", return_value=False):
                assert backend.is_available() is False

    def test_is_available_true_when_all_present(self):
        backend = LightOCRBackend()
        with patch("bddk_mcp.ocr.base._cuda_available", return_value=True):
            with patch("bddk_mcp.ocr.base._transformers_available", return_value=True):
                assert backend.is_available() is True

    def test_extract_returns_none_on_empty_pdf(self):
        backend = LightOCRBackend()
        assert backend.extract(b"") is None

    def test_extract_loads_model_lazily(self):
        backend = LightOCRBackend()
        assert backend._model is None  # not loaded at init
        with patch("bddk_mcp.ocr.base._cuda_available", return_value=True):
            with patch("bddk_mcp.ocr.base._transformers_available", return_value=True):
                mock_model = MagicMock()
                mock_model.generate_markdown.return_value = "extracted"
                with patch.object(backend, "_load_model", return_value=mock_model):
                    result = backend.extract(b"%PDF-1.4\nfake")
                    assert result == "extracted"


class TestDefaultBackends:
    def test_default_order_is_lightocr_pdftotext_markitdown(self):
        backends = get_default_backends()
        names = [b.name for b in backends]
        assert names == ["lightocr", "pdftotext_degraded", "markitdown_degraded"]


class _FakeIds(list):
    """A list of token ids with the tensor features the generation helper uses."""

    @property
    def shape(self):
        return (1, len(self))


class _FakeOutput:
    def __init__(self, ids):
        self._ids = ids

    def __getitem__(self, key):
        _row, columns = key
        return self._ids[columns]


class _FakeProcessor:
    def __init__(self, texts):
        self._texts = list(texts)
        self.tokenizer = type("Tok", (), {"eos_token_id": 2})()

    def apply_chat_template(self, conversation, **_kwargs):
        return {"input_ids": _FakeIds([9, 9, 9])}

    def decode(self, generated, skip_special_tokens=True):
        return self._texts.pop(0)


class _FakeModel:
    def __init__(self, generated_per_page):
        self._generated = list(generated_per_page)
        self.calls = 0

    def generate(self, *, input_ids, max_new_tokens):
        self.calls += 1
        return _FakeOutput(list(input_ids) + self._generated.pop(0))


class TestLightOCRGeneration:
    def test_pages_ending_with_eos_are_joined(self):
        model = _FakeModel([[5, 6, 2], [7, 2]])
        processor = _FakeProcessor(["Sayfa bir", "Sayfa iki"])
        assert _generate_lightocr_markdown(model, processor, ["p1", "p2"], max_new_tokens=8) == (
            "Sayfa bir\n\nSayfa iki"
        )

    def test_page_stopped_at_max_new_tokens_without_eos_raises_with_page_number(self, caplog):
        model = _FakeModel([[5, 2], [5, 5, 5, 5]])
        processor = _FakeProcessor(["Sayfa bir", "kesik"])
        with caplog.at_level("WARNING", logger="bddk_mcp.ocr.base"):
            with pytest.raises(LightOCRPageTruncatedError) as error:
                _generate_lightocr_markdown(model, processor, ["p1", "p2", "p3"], max_new_tokens=4)
        assert error.value.page_number == 2
        assert "page 2 of 3" in str(error.value)
        assert "page 2 of 3" in caplog.text
        assert model.calls == 2

    def test_eos_exactly_at_the_limit_is_not_truncation(self):
        model = _FakeModel([[5, 5, 5, 2]])
        assert _generate_lightocr_markdown(model, _FakeProcessor(["tam"]), ["p1"], max_new_tokens=4) == "tam"

    def test_empty_page_is_logged_and_counted_not_failed(self, caplog):
        model = _FakeModel([[5, 2], [2], [6, 2]])
        processor = _FakeProcessor(["bir", "   ", "üç"])
        with caplog.at_level("WARNING", logger="bddk_mcp.ocr.base"):
            result = _generate_lightocr_markdown(model, processor, ["p1", "p2", "p3"], max_new_tokens=8)
        assert result == "bir\n\nüç"
        assert "page 2 of 3 decoded empty" in caplog.text
        assert "1 empty page(s) of 3" in caplog.text

    def test_truncation_makes_the_backend_fail_so_the_chain_falls_through(self):
        backend = LightOCRBackend()
        wrapper = MagicMock()
        wrapper.generate_markdown.side_effect = lambda pdf: _generate_lightocr_markdown(
            _FakeModel([[5, 5]]), _FakeProcessor(["kesik"]), ["p1"], max_new_tokens=2
        )
        fallback = _FakeBackend("pdftotext_degraded", "x" * 200)
        with patch("bddk_mcp.ocr.base._cuda_available", return_value=True):
            with patch("bddk_mcp.ocr.base._transformers_available", return_value=True):
                with patch.object(backend, "_load_model", return_value=wrapper):
                    attempt = run_extraction_chain(b"%PDF", [backend, fallback], min_len=10)
        assert attempt.backend == "pdftotext_degraded"
