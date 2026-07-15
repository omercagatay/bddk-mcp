"""Sentinel tests for privacy-safe internal operational logging."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from bddk_mcp.ingest.backfill import BackfillCandidate, execute_backfill
from bddk_mcp.ingest.client import BddkApiClient
from bddk_mcp.ingest.doc_sync import DocumentSyncer, ExtractionResult
from bddk_mcp.ocr.base import LightOCRBackend, _run_markitdown, _run_pdftotext, run_extraction_chain
from bddk_mcp.store.doc_store import DocumentStore
from bddk_mcp.store.vector_store import RetrievalProfileError, VectorStore, _load_embedding_tokenizer

PRIVATE_TERM = "PRIVATE_AUDIT_TERM_91f8"
PRIVATE_PATH = f"/bank/private/{PRIVATE_TERM}/source.pdf"
PRIVATE_URL = f"https://internal.example.test/regulation?query={PRIVATE_TERM}&token=secret"
PRIVATE_DSN = f"postgresql://private:password@db/{PRIVATE_TERM}"


def _assert_private_values_absent(caplog: pytest.LogCaptureFixture) -> None:
    rendered = caplog.text
    for value in (PRIVATE_TERM, PRIVATE_PATH, PRIVATE_URL, PRIVATE_DSN, "password@db"):
        assert value not in rendered


def _error_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.levelno >= logging.WARNING]


def test_markitdown_exception_text_is_not_logged(caplog):
    with (
        patch("markitdown.MarkItDown") as markitdown,
        caplog.at_level(logging.WARNING, logger="bddk_mcp.ocr.base"),
    ):
        markitdown.return_value.convert_stream.side_effect = OSError(f"{PRIVATE_PATH} {PRIVATE_URL}")
        assert _run_markitdown(b"%PDF-private") is None

    _assert_private_values_absent(caplog)
    assert _error_records(caplog)[-1].error_type == "OSError"


def test_pdftotext_stderr_is_not_logged(caplog):
    failed = subprocess.CompletedProcess(
        args=["pdftotext"],
        returncode=1,
        stdout="",
        stderr=f"failed reading {PRIVATE_PATH}; source={PRIVATE_URL}",
    )
    with (
        patch("bddk_mcp.ocr.base.shutil.which", return_value="/usr/bin/pdftotext"),
        patch("bddk_mcp.ocr.base.subprocess.run", return_value=failed),
        caplog.at_level(logging.WARNING, logger="bddk_mcp.ocr.base"),
    ):
        assert _run_pdftotext(b"%PDF-private") is None

    _assert_private_values_absent(caplog)
    assert _error_records(caplog)[-1].error_type == "PdftotextProcessError"


def test_lightocr_failure_keeps_exception_type_but_omits_message(caplog):
    backend = LightOCRBackend(model_path=PRIVATE_PATH, device="cpu")
    with (
        patch.object(backend, "_load_model", side_effect=RuntimeError(f"{PRIVATE_DSN} {PRIVATE_TERM}")),
        caplog.at_level(logging.WARNING, logger="bddk_mcp.ocr.base"),
    ):
        assert backend.extract(b"%PDF-private") is None

    _assert_private_values_absent(caplog)
    assert _error_records(caplog)[-1].error_type == "RuntimeError"


def test_lightocr_model_path_is_passed_to_dependency_but_not_logged(caplog, monkeypatch):
    loaded_sources: list[str] = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, source, **_kwargs):
            loaded_sources.append(source)
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, source, **_kwargs):
            loaded_sources.append(source)
            return cls()

        def to(self, _device):
            return self

        def train(self, _enabled):
            return self

    torch_module = ModuleType("torch")
    torch_module.bfloat16 = object()
    torch_module.cuda = SimpleNamespace(is_available=lambda: False, memory_allocated=lambda: 0)
    transformers_module = ModuleType("transformers")
    transformers_module.AutoProcessor = FakeProcessor
    transformers_module.LightOnOcrForConditionalGeneration = FakeModel
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    backend = LightOCRBackend(model_path=PRIVATE_PATH, device="cpu")
    with caplog.at_level(logging.INFO, logger="bddk_mcp.ocr.base"):
        assert backend._load_model() is not None

    assert loaded_sources == [PRIVATE_PATH, PRIVATE_PATH]
    _assert_private_values_absent(caplog)
    assert "local source" in caplog.text


def test_extraction_chain_preserves_user_result_but_sanitizes_failure_log(caplog):
    class FailingBackend:
        name = "private_failure_backend"

        def is_available(self):
            return True

        def extract(self, _pdf_bytes):
            raise RuntimeError(f"{PRIVATE_URL} {PRIVATE_PATH}")

    with caplog.at_level(logging.WARNING, logger="bddk_mcp.ocr.base"):
        result = run_extraction_chain(b"pdf", [FailingBackend()], min_len=1)

    # The task deliberately leaves the caller-visible diagnostic contract
    # unchanged while hardening only the operational log boundary.
    assert PRIVATE_TERM in result.error
    _assert_private_values_absent(caplog)
    assert _error_records(caplog)[-1].error_type == "RuntimeError"


def test_chandra_model_reference_is_used_but_not_logged(caplog, monkeypatch):
    from bddk_mcp.ocr import chandra

    manager = object()
    settings = SimpleNamespace(MODEL_CHECKPOINT="initial")
    model_module = ModuleType("chandra.model")
    model_module.InferenceManager = MagicMock(return_value=manager)
    model_module.settings = settings
    chandra_package = ModuleType("chandra")
    chandra_package.__path__ = []
    monkeypatch.setitem(sys.modules, "chandra", chandra_package)
    monkeypatch.setitem(sys.modules, "chandra.model", model_module)
    monkeypatch.setattr(chandra, "CHANDRA_MODEL_NAME", PRIVATE_PATH)

    with caplog.at_level(logging.INFO, logger="bddk_mcp.ocr.chandra"):
        assert chandra.ChandraBackend()._load_manager() is manager

    assert settings.MODEL_CHECKPOINT == PRIVATE_PATH
    _assert_private_values_absent(caplog)


def test_chandra_input_exception_and_temp_path_are_not_logged(caplog, monkeypatch):
    from bddk_mcp.ocr import chandra

    input_module = ModuleType("chandra.input")
    input_module.load_file = MagicMock(side_effect=OSError(f"{PRIVATE_PATH} {PRIVATE_URL}"))
    schema_module = ModuleType("chandra.model.schema")
    schema_module.BatchInputItem = MagicMock
    chandra_package = ModuleType("chandra")
    chandra_package.__path__ = []
    model_package = ModuleType("chandra.model")
    model_package.__path__ = []
    monkeypatch.setitem(sys.modules, "chandra", chandra_package)
    monkeypatch.setitem(sys.modules, "chandra.input", input_module)
    monkeypatch.setitem(sys.modules, "chandra.model", model_package)
    monkeypatch.setitem(sys.modules, "chandra.model.schema", schema_module)

    backend = chandra.ChandraBackend()
    backend._manager = MagicMock()
    with caplog.at_level(logging.DEBUG, logger="bddk_mcp.ocr.chandra"):
        assert backend.extract(b"%PDF-private") is None

    _assert_private_values_absent(caplog)
    assert _error_records(caplog)[-1].error_type == "OSError"


@pytest.mark.asyncio
async def test_backfill_logs_omit_document_metadata_and_failure_details(caplog):
    candidate = BackfillCandidate(
        document_id=PRIVATE_TERM,
        title=f"Audit title {PRIVATE_TERM}",
        source_url=PRIVATE_URL,
        category="Internal audit",
        decision_date="",
        decision_number="",
        len=1,
        signature="test",
    )
    failure = MagicMock(
        success=False,
        method="",
        size_bytes=0,
        error=f"{PRIVATE_DSN} {PRIVATE_PATH}",
    )
    syncer = MagicMock(sync_document=AsyncMock(return_value=failure))

    with caplog.at_level(logging.INFO, logger="bddk_mcp.ingest.backfill"):
        report = await execute_backfill(syncer, [candidate], inter_request_delay=0)

    assert PRIVATE_TERM in report.failed[0][0]
    assert PRIVATE_DSN in report.failed[0][1]
    _assert_private_values_absent(caplog)
    assert _error_records(caplog)[-1].error_type == "SyncResultFailure"
    assert "item 1/1" in caplog.text.lower()


@pytest.mark.asyncio
async def test_sync_failure_persistence_rejects_raw_error_and_source_url():
    pool = MagicMock(execute=AsyncMock())
    store = DocumentStore(pool)
    await store.record_sync_failure(
        "123",
        f"ConnectError: {PRIVATE_DSN} {PRIVATE_PATH}",
        f"invalid-{PRIVATE_TERM}",
        PRIVATE_URL,
        True,
    )

    persisted = pool.execute.await_args.args
    assert persisted[1:] == ("123", "sync_unknown_failed", "unknown", "", True, persisted[-1])
    for private_value in (PRIVATE_TERM, PRIVATE_DSN, PRIVATE_PATH, PRIVATE_URL):
        assert private_value not in repr(persisted)


@pytest.mark.asyncio
async def test_sync_failure_persistence_preserves_safe_machine_code():
    pool = MagicMock(execute=AsyncMock())
    store = DocumentStore(pool)
    await store.record_sync_failure("123", "reindex_failed", "index", PRIVATE_URL, True)

    persisted = pool.execute.await_args.args
    assert persisted[2:5] == ("reindex_failed", "index", "")


def test_embedding_tokenizer_log_omits_model_path_and_exception(caplog, monkeypatch, tmp_path):
    model_path = tmp_path / PRIVATE_TERM
    model_path.mkdir()
    (model_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    class FailingTokenizer:
        @classmethod
        def from_pretrained(cls, source, **_kwargs):
            assert source == str(model_path)
            raise OSError(f"{PRIVATE_PATH}: {PRIVATE_URL}")

    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = FailingTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setattr("bddk_mcp.store.vector_store.EMBEDDING_CHUNK_MODE", "token")
    monkeypatch.setattr("bddk_mcp.store.vector_store.EMBEDDING_MODEL_PATH", str(model_path))

    with caplog.at_level(logging.WARNING, logger="bddk_mcp.store.vector_store"):
        with pytest.raises(RetrievalProfileError):
            _load_embedding_tokenizer()

    _assert_private_values_absent(caplog)
    assert _error_records(caplog)[-1].error_type == "OSError"
    assert "local source" in caplog.text


def test_embedding_loader_uses_private_path_without_logging_it(caplog, monkeypatch, tmp_path):
    model_path = tmp_path / PRIVATE_TERM
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    loaded_sources: list[str] = []

    class FakeSentenceTransformer:
        def __init__(self, source, **_kwargs):
            loaded_sources.append(source)

        @staticmethod
        def get_sentence_embedding_dimension():
            from bddk_mcp.store.vector_store import EMBEDDING_DIMENSION

            return EMBEDDING_DIMENSION

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    monkeypatch.setattr("bddk_mcp.store.vector_store.EMBEDDING_MODEL_PATH", str(model_path))

    store = VectorStore(pool=MagicMock())
    with caplog.at_level(logging.INFO, logger="bddk_mcp.store.vector_store"):
        store._ensure_embeddings()

    assert loaded_sources == [str(model_path)]
    _assert_private_values_absent(caplog)
    assert "local source" in caplog.text


def test_reranker_loader_uses_private_path_without_logging_it(caplog, monkeypatch, tmp_path):
    model_path = tmp_path / PRIVATE_TERM
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    loaded_sources: list[str] = []

    class FakeCrossEncoder:
        def __init__(self, source, **_kwargs):
            loaded_sources.append(source)

    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)
    monkeypatch.setattr("bddk_mcp.store.vector_store.RERANKER_MODEL_PATH", str(model_path))
    monkeypatch.setattr("bddk_mcp.store.vector_store.RERANKER_ENABLED", True)

    store = VectorStore(pool=MagicMock())
    with caplog.at_level(logging.INFO, logger="bddk_mcp.store.vector_store"):
        store._ensure_reranker()

    assert loaded_sources == [str(model_path)]
    _assert_private_values_absent(caplog)
    assert "local source" in caplog.text


def test_document_sync_extraction_log_omits_structured_error(caplog):
    syncer = DocumentSyncer(
        MagicMock(),
        ocr_backends=[],
        http=MagicMock(spec=httpx.AsyncClient),
        prefer_html_for_mevzuat=False,
    )
    extraction = ExtractionResult(
        method="failed",
        error=f"{PRIVATE_DSN} {PRIVATE_URL} {PRIVATE_PATH}",
        retryable=True,
    )

    with (
        patch.object(syncer, "_extract_structured", return_value=extraction),
        caplog.at_level(logging.WARNING, logger="bddk_mcp.ingest.doc_sync"),
    ):
        content, method = syncer._extract(b"private", ".pdf")

    assert (content, method) == ("", "failed")
    _assert_private_values_absent(caplog)
    assert _error_records(caplog)[-1].error_type == "ExtractionIssue"


@pytest.mark.asyncio
async def test_document_sync_debug_log_omits_url_document_id_and_exception(caplog):
    syncer = DocumentSyncer(
        MagicMock(),
        ocr_backends=[],
        http=MagicMock(spec=httpx.AsyncClient),
        prefer_html_for_mevzuat=False,
    )
    syncer._fetch_trusted_mevzuat = AsyncMock(side_effect=OSError(f"{PRIVATE_PATH} {PRIVATE_DSN}"))
    main_page = f'<iframe src="{PRIVATE_URL}"></iframe>'

    with caplog.at_level(logging.DEBUG, logger="bddk_mcp.ingest.doc_sync"):
        result = await syncer._try_iframe_layer(
            PRIVATE_TERM,
            PRIVATE_TERM,
            main_page,
            httpx.Timeout(1),
            mevzuat_no="123",
        )

    assert result is None
    _assert_private_values_absent(caplog)
    assert caplog.records[-1].error_type == "OSError"


@pytest.mark.asyncio
async def test_legacy_cache_missing_log_omits_configured_path(caplog, monkeypatch):
    syncer = DocumentSyncer(
        MagicMock(),
        ocr_backends=[],
        http=MagicMock(spec=httpx.AsyncClient),
        prefer_html_for_mevzuat=False,
    )
    monkeypatch.setattr("bddk_mcp.ingest.doc_sync.CACHE_FILE", Path(PRIVATE_PATH))

    with caplog.at_level(logging.ERROR, logger="bddk_mcp.ingest.doc_sync"):
        report = await syncer.import_and_sync_from_cache()

    assert report.total == 0
    _assert_private_values_absent(caplog)


@pytest.mark.asyncio
async def test_client_unmapped_upstream_category_is_not_logged(caplog):
    client = BddkApiClient(pool=MagicMock(), http=MagicMock(spec=httpx.AsyncClient))
    html = f"""
    <div class="card">
      <h5>{PRIVATE_TERM}</h5>
      <div class="card-body"><a href="/Mevzuat/DokumanGetir/123">Public title</a></div>
    </div>
    """
    client._fetch_with_retry = AsyncMock(return_value=httpx.Response(200, text=html))

    with caplog.at_level(logging.WARNING, logger="bddk_mcp.ingest.client"):
        decisions = await client._fetch_and_parse_accordion_page(50)

    assert decisions[0].category == PRIVATE_TERM
    _assert_private_values_absent(caplog)
    assert f"chars={len(PRIVATE_TERM)}" in caplog.text


@pytest.mark.asyncio
async def test_client_store_hit_log_omits_document_identifier(caplog):
    page = SimpleNamespace(
        document_id=PRIVATE_TERM,
        markdown_content="public text",
        page_number=1,
        total_pages=2,
    )
    doc_store = MagicMock(get_document_page=AsyncMock(return_value=page))
    client = BddkApiClient(
        pool=MagicMock(),
        http=MagicMock(spec=httpx.AsyncClient),
        doc_store=doc_store,
    )

    with caplog.at_level(logging.INFO, logger="bddk_mcp.ingest.client"):
        result = await client.get_document_markdown(PRIVATE_TERM)

    assert result.document_id == PRIVATE_TERM
    _assert_private_values_absent(caplog)
    assert "page 1/2" in caplog.text
