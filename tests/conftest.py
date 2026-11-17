"""Shared test fixtures for BDDK MCP Server tests.

Every storage test runs against a temporary SQLite corpus file; nothing here
needs a database service.  Embeddings come from a deterministic fake so the
vector store is exercised without loading a model.
"""

import hashlib
import importlib.util
import ipaddress
import os
import socket
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import numpy as np
import pytest

from bddk_mcp.store.corpus_db import CorpusDB
from bddk_mcp.store.doc_store import DocumentStore, StoredDocument
from bddk_mcp.store.vector_store import VectorStore


@pytest.fixture
def far_east_timezone():
    """Make a naive (local-time) ``datetime.fromtimestamp`` render UTC+14, on every platform.

    ``time.tzset`` does not exist on Windows, so the host zone cannot be changed portably. The
    production renderers must pass ``tz=UTC``; one that fell back to local time would read UTC+14.
    """
    from datetime import UTC, datetime, timedelta, timezone

    from bddk_mcp.store import doc_store
    from bddk_mcp.tools import structured_outputs

    class _FarEastDatetime(datetime):
        @classmethod
        def fromtimestamp(cls, timestamp, tz=None):
            if tz is None:
                return datetime.fromtimestamp(timestamp, tz=timezone(timedelta(hours=14))).replace(tzinfo=None)
            return datetime.fromtimestamp(timestamp, tz=tz)

    # 2026-04-08 23:30 UTC is already 2026-04-09 in UTC+14.
    assert _FarEastDatetime.fromtimestamp(1775691000.0).date().isoformat() == "2026-04-09"
    assert _FarEastDatetime.fromtimestamp(1775691000.0, tz=UTC).date().isoformat() == "2026-04-08"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(doc_store, "datetime", _FarEastDatetime)
        patch.setattr(structured_outputs, "datetime", _FarEastDatetime)
        yield


@pytest.fixture
def historical_quality_registry(monkeypatch):
    """Exercise fail-closed registry behavior without reflagging repaired seed documents."""
    from bddk_mcp.quality import markdown_quality

    path = Path(__file__).resolve().parents[1] / "tests/fixtures/historical-quality-failures.yml"
    registry = markdown_quality.load_quality_failure_registry(path)
    monkeypatch.setattr(markdown_quality, "_QUALITY_FAILURES", registry)
    return registry


@pytest.fixture(scope="session")
def provisioned_gpu_ocr_lane() -> None:
    """Require a complete, offline GPU/OCR environment only when opted in.

    Merely detecting CUDA must never trigger model downloads or live-source
    access in the ordinary test suite. Once the explicit lane is requested,
    missing dependencies or local model assets are failures rather than skips.
    """
    if os.environ.get("BDDK_REQUIRE_GPU_OCR") != "1":
        pytest.skip("GPU OCR lane not provisioned; set BDDK_REQUIRE_GPU_OCR=1 to require it")

    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        pytest.fail(
            "Required GPU OCR lane must set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1.",
            pytrace=False,
        )

    for variable in ("BDDK_LIGHTOCR_MODEL_PATH",):
        configured_path = os.environ.get(variable, "")
        if not configured_path or not Path(configured_path).is_dir():
            pytest.fail(f"Required GPU OCR lane needs {variable} to name a local model directory.", pytrace=False)

    for package in ("torch", "torchvision", "transformers"):
        if importlib.util.find_spec(package) is None:
            pytest.fail(f"Required GPU OCR lane is missing the optional {package} package.", pytrace=False)

    import torch

    if not torch.cuda.is_available():
        pytest.fail("Required GPU OCR lane cannot access CUDA.", pytrace=False)


# -- Network guard -------------------------------------------------------------


class NetworkAccessBlocked(RuntimeError):
    """Not an OSError on purpose: an HTTP client would turn an OSError into an ordinary, retryable failure."""


def _is_loopback(host) -> bool:
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    if not host or host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


def _is_ip_literal(host) -> bool:
    try:
        ipaddress.ip_address((host.decode("ascii", "replace") if isinstance(host, bytes) else host).split("%", 1)[0])
    except (AttributeError, ValueError):
        return False
    return True


@pytest.fixture(autouse=True)
def network_guard(request, monkeypatch):
    """Fail any test that connects to, or looks up, a host other than loopback, unless it is marked ``network``.

    Guards ``socket.socket.connect``/``connect_ex`` (TCP/UDP only, so pipes and unix sockets pass), ``socket.getaddrinfo``
    (IP literals pass: they need no lookup) and, on Windows, the proactor event loop's ``connect``, which calls
    ConnectEx without going through ``socket.socket.connect``. A blocked attempt also fails the test at teardown, so a
    caller that catches the exception broadly cannot hide it. Yields the list of blocked targets.
    """
    blocked: list[str] = []
    if request.node.get_closest_marker("network"):
        yield blocked
        return

    def check(family, address) -> None:
        if family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address[0]):
            blocked.append(repr(address))
            raise NetworkAccessBlocked(
                f"Network access blocked in a unit test: connect to {address!r}. Mock the transport, or mark the test "
                "@pytest.mark.network if it genuinely needs the network."
            )

    def guard_socket(original):
        def connect(sock, address, *args):
            check(sock.family, address)
            return original(sock, address, *args)

        return connect

    real_getaddrinfo = socket.getaddrinfo

    def getaddrinfo(host, *args, **kwargs):
        if not (_is_loopback(host) or _is_ip_literal(host)):
            blocked.append(repr(host))
            raise NetworkAccessBlocked(
                f"Network access blocked in a unit test: DNS lookup of {host!r}. Mock the transport, or mark the test "
                "@pytest.mark.network if it genuinely needs the network."
            )
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guard_socket(socket.socket.connect))
    monkeypatch.setattr(socket.socket, "connect_ex", guard_socket(socket.socket.connect_ex))
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    if sys.platform == "win32":
        from asyncio.windows_events import IocpProactor

        real_proactor_connect = IocpProactor.connect

        def proactor_connect(proactor, conn, address):
            check(conn.family, address)
            return real_proactor_connect(proactor, conn, address)

        monkeypatch.setattr(IocpProactor, "connect", proactor_connect)
    yield blocked
    if blocked:
        pytest.fail(f"The test attempted network access: {', '.join(blocked)}", pytrace=False)


# -- Temporary corpus ----------------------------------------------------------


def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic bag-of-words unit vectors: shared words give a higher cosine."""
    vectors = []
    for text in texts:
        vector = np.zeros(768, dtype=np.float32)
        for word in text.lower().split():
            seed = int(hashlib.md5(word.encode("utf-8")).hexdigest()[:8], 16)
            vector += np.random.default_rng(seed).standard_normal(768).astype(np.float32)
        norm = float(np.linalg.norm(vector)) or 1.0
        vectors.append(vector / norm)
    return np.stack(vectors)


class FakeTokenizer:
    """Whitespace tokens stand in for the embedding model's tokenizer."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()


@pytest.fixture
def corpus_db(tmp_path):
    """A fresh read-write corpus file."""
    db = CorpusDB.open(tmp_path / "corpus.sqlite")
    yield db
    db.close()


@pytest.fixture
def doc_store(corpus_db):
    return DocumentStore(corpus_db)


@pytest.fixture
def vector_store(corpus_db):
    return VectorStore(corpus_db, embed_fn=fake_embed, tokenizer=FakeTokenizer())


# -- Store double for client tests --------------------------------------------


class MockStore:
    """A DocumentStore double for client tests that only test in-memory behavior."""

    def __init__(self, catalog: list[dict] | None = None) -> None:
        self.catalog = list(catalog or [])
        self.load_catalog = AsyncMock(side_effect=lambda: list(self.catalog))
        self.replace_catalog = AsyncMock(side_effect=self._replace)

    async def _replace(self, items: list[dict], *, cached_at: float | None = None) -> int:
        self.catalog = [dict(item, cached_at=cached_at or 0.0) for item in items]
        return len(items)


# -- HTTP mocking helpers ---------------------------------------------------


def make_http_response(
    text: str = "",
    status_code: int = 200,
    json_data=None,
    content: bytes = b"",
    content_type: str = "text/html",
) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock(spec=httpx.Response)
    resp.text = text
    resp.status_code = status_code
    resp.content = content or text.encode("utf-8")
    resp.headers = {"content-type": content_type}
    resp.raise_for_status = MagicMock()
    if json_data is not None:
        resp.json = MagicMock(return_value=json_data)
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError("Error", request=MagicMock(), response=resp)
    return resp


@pytest.fixture
def mock_http():
    """Create a mock httpx.AsyncClient."""
    client = AsyncMock(spec=httpx.AsyncClient)
    return client


# -- DocumentStore fixtures ------------------------------------------------


@pytest.fixture
def sample_doc():
    """A sample BDDK document for testing."""
    return StoredDocument(
        document_id="1291",
        title="Sermaye Yeterliliği Rehberi",
        category="Rehber",
        decision_date="15.03.2024",
        decision_number="11200",
        source_url="https://www.bddk.org.tr/Mevzuat/DokumanGetir/1291",
        markdown_content=(
            "# Sermaye Yeterliliği\n\n"
            "Bu rehber bankacılık sektöründe sermaye yeterliliği hesaplamalarını düzenler.\n\n"
            "## MADDE 1\n"
            "Kredi riski için asgari sermaye oranı %8 olarak belirlenmiştir."
        ),
        extraction_method="markitdown",
    )


@pytest.fixture
def mevzuat_doc():
    """A sample mevzuat.gov.tr document for testing."""
    return StoredDocument(
        document_id="mevzuat_42628",
        title="Faiz Oranı Riski Yönetmeliği",
        category="Yönetmelik",
        source_url="https://mevzuat.gov.tr/MevzuatMetin/yonetmelik/7.5.42628",
        markdown_content=(
            "# Faiz Oranı Riski\n\n## MADDE 9\nBanka, faiz oranı riskini ölçmek için standart yaklaşım kullanır."
        ),
        extraction_method="nougat",
    )


# -- Sample HTML responses -------------------------------------------------

BDDK_ACCORDION_HTML = """
<div class="card">
  <h5>Yönetmelikler (2)</h5>
  <div class="card-body">
    <ul>
      <li><a href="/Mevzuat/DokumanGetir/1291">Sermaye Yeterliliği Rehberi</a></li>
      <li><a href="https://mevzuat.gov.tr/mevzuat?MevzuatNo=42628&MevzuatTur=7&MevzuatTertip=5">Faiz Riski Yönetmeliği</a></li>
    </ul>
  </div>
</div>
"""

BDDK_DECISION_HTML = """
<a href="/Mevzuat/DokumanGetir/1280">(31.10.2024 - 11000) Ziraat Bankası faaliyet izni</a>
<a href="/Mevzuat/DokumanGetir/1281">(15.06.2023 - 10800) Katılım bankası kuruluş izni</a>
"""
