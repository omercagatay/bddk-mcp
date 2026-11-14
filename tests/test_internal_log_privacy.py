"""Sentinel tests for privacy-safe internal operational logging."""

from __future__ import annotations

import logging
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from bddk_mcp.ingest.client import BddkApiClient
from bddk_mcp.ingest.doc_sync import DocumentSyncer, ExtractionResult
from bddk_mcp.ocr.base import LightOCRBackend, _run_markitdown, _run_pdftotext, run_extraction_chain
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


async def _sync_failure_row(store):
    row = await store.db.run(
        lambda conn: conn.execute("SELECT * FROM sync_failures WHERE document_id = '123'").fetchone()
    )
    return dict(row)


@pytest.mark.asyncio
async def test_sync_failure_persistence_rejects_raw_error_and_source_url(doc_store):
    store = doc_store
    await store.record_sync_failure(
        "123",
        f"ConnectError: {PRIVATE_DSN} {PRIVATE_PATH}",
        f"invalid-{PRIVATE_TERM}",
        PRIVATE_URL,
        True,
    )

    persisted = await _sync_failure_row(store)
    assert (persisted["error"], persisted["error_category"], persisted["source_url"], persisted["retryable"]) == (
        "sync_unknown_failed",
        "unknown",
        "",
        1,
    )
    for private_value in (PRIVATE_TERM, PRIVATE_DSN, PRIVATE_PATH, PRIVATE_URL):
        assert private_value not in repr(persisted)


@pytest.mark.asyncio
async def test_sync_failure_persistence_preserves_safe_machine_code(doc_store):
    await doc_store.record_sync_failure("123", "reindex_failed", "index", PRIVATE_URL, True)

    persisted = await _sync_failure_row(doc_store)
    assert (persisted["error"], persisted["error_category"], persisted["source_url"]) == ("reindex_failed", "index", "")


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

    store = VectorStore(MagicMock())
    with caplog.at_level(logging.INFO, logger="bddk_mcp.store.vector_store"):
        store._ensure_embeddings()

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
async def test_client_unmapped_upstream_category_is_not_logged(caplog):
    client = BddkApiClient(store=MagicMock(), http=MagicMock(spec=httpx.AsyncClient))
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


# -- MCP SDK records -----------------------------------------------------------


@pytest.fixture
def production_log_stream(monkeypatch):
    """Run the packaged JSON logging setup against a buffer and restore the root logger."""
    import io

    from bddk_mcp.core.logging_config import configure_logging

    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    monkeypatch.delenv("BDDK_TOOL_LOG_CONTENT", raising=False)
    stream = io.StringIO()
    try:
        configure_logging(json_output=True)
        root.handlers[0].setStream(stream)
        yield stream
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


@pytest.mark.asyncio
async def test_sdk_logs_do_not_carry_request_content_or_tracebacks(production_log_stream):
    """Requests the SDK rejects must not put caller text or source paths into the logs."""
    import json

    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.http_security import HttpSecurityMiddleware, load_http_security_config
    from bddk_mcp.server import create_mcp

    security = load_http_security_config({"MCP_HOST": "127.0.0.1", "PORT": "8123"})
    deps = Dependencies(db=None, doc_store=MagicMock(), client=MagicMock(), http=None)
    raw_app = create_mcp(deps, http_security=security).streamable_http_app()
    app = HttpSecurityMiddleware(raw_app, security)
    headers = {"accept": "application/json, text/event-stream", "content-type": "application/json"}
    bodies = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": args}})
        for name, args in (("search_document_store", PRIVATE_TERM), (PRIVATE_TERM, {}))
    ]
    bodies.append(json.dumps({"jsonrpc": "2.0", "id": 1, "method": PRIVATE_TERM, "params": {}}))
    bodies.append("[" * 100_000 + "]" * 100_000)  # RecursionError inside the SDK's JSON parsing

    async with raw_app.router.lifespan_context(raw_app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8123", headers=headers
        ) as client:
            for body in bodies:
                await client.post("/mcp", content=body)

    rendered = production_log_stream.getvalue()
    records = [json.loads(line) for line in rendered.splitlines()]
    sdk_problems = [r for r in records if r["level"] in {"WARNING", "ERROR"} and not r["logger"].startswith("bddk_mcp")]
    assert len(sdk_problems) >= 4, "the SDK must still report each rejected request"
    assert PRIVATE_TERM not in rendered
    assert "site-packages" not in rendered
    assert all("exception" not in record for record in records)
    assert any("RecursionError" in record["message"] for record in sdk_problems)


def test_sdk_log_filter_keeps_project_records_and_honours_the_content_opt_in(monkeypatch):
    from bddk_mcp.core.logging_config import SdkLogPrivacyFilter

    def record(name: str, msg: str, *args: object, level: int = logging.WARNING) -> logging.LogRecord:
        return logging.LogRecord(name, level, __file__, 1, msg, args, None)

    privacy_filter = SdkLogPrivacyFilter()
    monkeypatch.delenv("BDDK_TOOL_LOG_CONTENT", raising=False)

    own = record("bddk_mcp.tools.search", f"kept {PRIVATE_TERM}")
    assert privacy_filter.filter(own) and PRIVATE_TERM in own.getMessage()

    request_type = record(
        "mcp.server.lowlevel.server", "Processing request of type %s", "PingRequest", level=logging.INFO
    )
    assert privacy_filter.filter(request_type) and request_type.getMessage() == "Processing request of type PingRequest"

    for sdk in (
        record("root", f"Failed to validate request: input_value='{PRIVATE_TERM}'"),
        record("mcp.server.lowlevel.server", "Tool '%s' not listed, no validation will be performed", PRIVATE_TERM),
        record("mcp.server.lowlevel.server", "Warning: %s: %s", "UserWarning", PRIVATE_TERM, level=logging.INFO),
    ):
        assert privacy_filter.filter(sdk)
        assert PRIVATE_TERM not in sdk.getMessage()
        assert "<withheld>" in sdk.getMessage()

    monkeypatch.setenv("BDDK_TOOL_LOG_CONTENT", "true")
    opted_in = record("root", f"Failed to validate request: input_value='{PRIVATE_TERM}'")
    assert privacy_filter.filter(opted_in) and PRIVATE_TERM in opted_in.getMessage()
