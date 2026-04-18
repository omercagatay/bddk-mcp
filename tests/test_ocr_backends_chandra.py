"""Unit tests for ChandraBackend (in-process HF inference)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestChandraBackend:
    def test_name(self):
        from ocr_backends_chandra import ChandraBackend

        assert ChandraBackend().name == "chandra2"

    def test_is_available_false_without_cuda(self):
        from ocr_backends_chandra import ChandraBackend

        backend = ChandraBackend()
        with patch("ocr_backends_chandra._cuda_available", return_value=False):
            with patch("ocr_backends_chandra._chandra_available", return_value=True):
                assert backend.is_available() is False

    def test_is_available_false_without_chandra(self):
        from ocr_backends_chandra import ChandraBackend

        backend = ChandraBackend()
        with patch("ocr_backends_chandra._cuda_available", return_value=True):
            with patch("ocr_backends_chandra._chandra_available", return_value=False):
                assert backend.is_available() is False

    def test_is_available_true(self):
        from ocr_backends_chandra import ChandraBackend

        backend = ChandraBackend()
        with patch("ocr_backends_chandra._cuda_available", return_value=True):
            with patch("ocr_backends_chandra._chandra_available", return_value=True):
                assert backend.is_available() is True

    def test_extract_empty_returns_none(self):
        from ocr_backends_chandra import ChandraBackend

        assert ChandraBackend().extract(b"") is None

    def test_extract_concatenates_per_page_markdown(self):
        from ocr_backends_chandra import ChandraBackend

        backend = ChandraBackend()
        fake_manager = MagicMock()
        fake_manager.generate.side_effect = [
            [SimpleNamespace(markdown="# Page 1", error=False)],
            [SimpleNamespace(markdown="# Page 2", error=False)],
        ]
        backend._manager = fake_manager

        fake_images = [MagicMock(), MagicMock()]
        with patch("chandra.input.load_file", return_value=fake_images):
            out = backend.extract(b"%PDF-fake")
        assert out == "# Page 1\n\n# Page 2"
        assert fake_manager.generate.call_count == 2

    def test_extract_returns_none_when_all_pages_blank(self):
        from ocr_backends_chandra import ChandraBackend

        backend = ChandraBackend()
        fake_manager = MagicMock()
        fake_manager.generate.return_value = [
            SimpleNamespace(markdown="  ", error=False),
        ]
        backend._manager = fake_manager

        with patch("chandra.input.load_file", return_value=[MagicMock()]):
            out = backend.extract(b"%PDF-fake")
        assert out is None

    def test_extract_returns_none_when_load_file_fails(self):
        from ocr_backends_chandra import ChandraBackend

        backend = ChandraBackend()
        backend._manager = MagicMock()

        with patch("chandra.input.load_file", side_effect=RuntimeError("bad pdf")):
            out = backend.extract(b"%PDF-fake")
        assert out is None

    def test_extract_returns_none_on_inference_error(self):
        from ocr_backends_chandra import ChandraBackend

        backend = ChandraBackend()
        fake_manager = MagicMock()
        fake_manager.generate.side_effect = RuntimeError("OOM")
        backend._manager = fake_manager

        with patch("chandra.input.load_file", return_value=[MagicMock()]):
            out = backend.extract(b"%PDF-fake")
        assert out is None

    def test_extract_returns_none_on_result_error_flag(self):
        from ocr_backends_chandra import ChandraBackend

        backend = ChandraBackend()
        fake_manager = MagicMock()
        fake_manager.generate.return_value = [
            SimpleNamespace(markdown="partial", error=True),
        ]
        backend._manager = fake_manager

        with patch("chandra.input.load_file", return_value=[MagicMock()]):
            out = backend.extract(b"%PDF-fake")
        assert out is None

    def test_extract_lazy_loads_manager_once(self):
        from ocr_backends_chandra import ChandraBackend

        backend = ChandraBackend()
        load_calls = {"n": 0}

        def fake_loader():
            load_calls["n"] += 1
            mgr = MagicMock()
            mgr.generate.return_value = [
                SimpleNamespace(markdown="ok", error=False),
            ]
            return mgr

        with patch.object(backend, "_load_manager", side_effect=fake_loader):
            with patch("chandra.input.load_file", return_value=[MagicMock()]):
                backend.extract(b"%PDF-a")
                backend.extract(b"%PDF-b")
        assert load_calls["n"] == 1

    def test_load_manager_propagates_model_name_to_chandra_settings(self, monkeypatch):
        # chandra reads its checkpoint from chandra.settings.MODEL_CHECKPOINT,
        # not a constructor arg — _load_manager must mutate it before
        # InferenceManager() so BDDK_CHANDRA_MODEL is not a no-op.
        from chandra.model import settings as chandra_settings

        import ocr_backends_chandra

        monkeypatch.setattr(chandra_settings, "MODEL_CHECKPOINT", chandra_settings.MODEL_CHECKPOINT)
        monkeypatch.setattr(ocr_backends_chandra, "CHANDRA_MODEL_NAME", "test-org/override-model")

        fake_mgr = MagicMock()
        with patch("chandra.model.InferenceManager", return_value=fake_mgr) as mgr_cls:
            backend = ocr_backends_chandra.ChandraBackend()
            result = backend._load_manager()

        assert chandra_settings.MODEL_CHECKPOINT == "test-org/override-model"
        assert result is fake_mgr
        mgr_cls.assert_called_once_with(method="hf")
