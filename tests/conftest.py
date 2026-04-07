"""Shared test fixtures for BDDK MCP Server tests."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from doc_store import DocumentStore, StoredDocument
from models import BddkDecisionSummary

# -- HTTP mocking helpers -----------------------------------------------


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


# -- DocumentStore fixtures ---------------------------------------------


@pytest.fixture
async def doc_store(tmp_path):
    """Create a temporary DocumentStore."""
    db_path = tmp_path / "test_docs.db"
    s = DocumentStore(db_path=db_path)
    await s.initialize()
    yield s
    await s.close()


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


# -- Sample decision cache ----------------------------------------------


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


# -- Sample HTML responses ----------------------------------------------


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
