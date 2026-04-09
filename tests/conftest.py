"""Shared test fixtures for BDDK MCP Server tests."""

import asyncio
import os
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import httpx
import pytest
import pytest_asyncio

from doc_store import DocumentStore, StoredDocument
from models import BddkDecisionSummary

# -- PostgreSQL test database -------------------------------------------------

_TEST_DSN = os.environ.get("BDDK_TEST_DATABASE_URL", "postgresql://bddk:bddk@localhost:5432/bddk_test")


# -- Pool fixture (created once per test, reused via caching) -----------------

_pg_pool_cache: asyncpg.Pool | None = None
_pg_pool_loop_id: int | None = None


@pytest.fixture
async def pg_pool():
    """Provide a PostgreSQL pool. Skips if PG unavailable.

    Caches the pool across tests sharing the same event loop.
    """
    global _pg_pool_cache, _pg_pool_loop_id

    loop = asyncio.get_running_loop()
    loop_id = id(loop)

    # Reuse pool if on same loop
    if _pg_pool_cache is not None and _pg_pool_loop_id == loop_id:
        yield _pg_pool_cache
        return

    # Close stale pool from a different loop
    if _pg_pool_cache is not None:
        try:
            await _pg_pool_cache.close()
        except Exception:
            pass
        _pg_pool_cache = None

    try:
        pool = await asyncpg.create_pool(_TEST_DSN, min_size=1, max_size=5, timeout=5)
        await pool.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await pool.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
        _pg_pool_cache = pool
        _pg_pool_loop_id = loop_id
        yield pool
    except (asyncpg.PostgresError, OSError, asyncio.TimeoutError):
        pytest.skip("PostgreSQL test database not available")


class SingleConnPool:
    """Wraps a single asyncpg connection to behave like a pool.

    Used in tests so each test runs inside a transaction that gets rolled back.
    """

    def __init__(self, conn: asyncpg.Connection):
        self._conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self._conn

    async def fetch(self, query, *args, **kwargs):
        return await self._conn.fetch(query, *args, **kwargs)

    async def fetchrow(self, query, *args, **kwargs):
        return await self._conn.fetchrow(query, *args, **kwargs)

    async def fetchval(self, query, *args, **kwargs):
        return await self._conn.fetchval(query, *args, **kwargs)

    async def execute(self, query, *args, **kwargs):
        return await self._conn.execute(query, *args, **kwargs)


@pytest.fixture
async def doc_store(pg_pool):
    """Create a DocumentStore inside a transaction (rolled back after test)."""
    conn = await pg_pool.acquire()
    tx = conn.transaction()
    await tx.start()

    pool_wrapper = SingleConnPool(conn)
    store = DocumentStore(pool_wrapper)
    await store.initialize()

    # Also create decision_cache table so BddkApiClient queries
    # don't abort the transaction with "relation does not exist"
    from client import _CACHE_SCHEMA_SQL
    await conn.execute(_CACHE_SCHEMA_SQL)

    yield store

    await tx.rollback()
    await pg_pool.release(conn)


# -- Mock pool for tests that don't need real SQL ----------------------------


class MockPool:
    """A no-op pool for client tests that only test in-memory behavior."""

    @asynccontextmanager
    async def acquire(self):
        yield AsyncMock()

    async def fetch(self, *args, **kwargs):
        return []

    async def fetchrow(self, *args, **kwargs):
        return None

    async def fetchval(self, *args, **kwargs):
        return None

    async def execute(self, *args, **kwargs):
        return "SELECT 0"


@pytest.fixture
def mock_pool():
    """A mock pool for client tests that don't touch the database."""
    return MockPool()


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


# -- Sample decision cache -------------------------------------------------


SAMPLE_DECISIONS = [
    BddkDecisionSummary(
        title="Bankaların Kredi İşlemlerine İlişkin Yönetmelik",
        document_id="mevzuat_40520",
        content="Bankaların Kredi İşlemlerine İlişkin Yönetmelik",
        category="Yönetmelik",
        source_url="https://mevzuat.gov.tr/mevzuat?MevzuatNo=40520",
    ),
    BddkDecisionSummary(
        title="2025/1 Sayılı Genelge: Bankacılık Hesapları",
        document_id="1296",
        content="Bankacılık hesapları hakkında genelge",
        category="Genelge",
        source_url="https://www.bddk.org.tr/Mevzuat/DokumanGetir/1296",
    ),
    BddkDecisionSummary(
        title="Ziraat Bankası faaliyet izni",
        document_id="1280",
        content="Ziraat Bankası faaliyet izni",
        decision_date="31.10.2024",
        decision_number="11000",
        category="Kurul Kararı",
        source_url="https://www.bddk.org.tr/Mevzuat/DokumanGetir/1280",
    ),
    BddkDecisionSummary(
        title="5411 Sayılı Bankacılık Kanunu",
        document_id="mevzuat_5411",
        content="Bankacılık Kanunu",
        category="Kanun",
        source_url="https://mevzuat.gov.tr/mevzuat?MevzuatNo=5411&MevzuatTur=1",
    ),
]


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

BDDK_FLAT_HTML = """
<a href="/Mevzuat/DokumanGetir/500">5411 Sayılı Bankacılık Kanunu</a>
<a href="https://mevzuat.gov.tr/mevzuat?MevzuatNo=5411&MevzuatTur=1">Bankacılık Kanunu</a>
"""
