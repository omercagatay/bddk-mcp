from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from bddk_mcp.admin.app import create_app
from bddk_mcp.admin.config import AdminConfig
from bddk_mcp.admin.services.documents import DocumentService
from bddk_mcp.store.doc_store import StoredDocument, StoreStats

CONFIG = AdminConfig(bind_host="127.0.0.1", port=8100, database_url="postgresql://x", loopback_only=True)


class StubGovernanceService:
    """The signature panel has its own tests; these only need create_app's
    required collaborator to exist, and fail loudly if it is ever consulted."""

    async def status(self):
        raise AssertionError("governance status must not be fetched by document-view tests")


GOVERNANCE = StubGovernanceService()


class FakeStore:
    """Returns the same shapes the real DocumentStore does, so a field
    rename in StoreStats/StoredDocument breaks these tests instead of
    leaving the admin UI silently rendering blanks."""

    def __init__(self, rows):
        self.rows = rows

    async def list_documents(self, category=None, limit=100, offset=0):
        selected = [r for r in self.rows if category is None or r["category"] == category]
        return selected[offset : offset + limit]

    async def stats(self):
        return StoreStats(categories={"mevzuat": 2}, total_documents=len(self.rows))


@pytest.fixture
def client() -> TestClient:
    rows = [
        {"document_id": "mevzuat_1", "title": "Bankacilik Kanunu", "category": "mevzuat", "total_pages": 3},
        {"document_id": "karar_2", "title": "Kurul Karari", "category": "karar", "total_pages": 1},
    ]
    return TestClient(
        create_app(CONFIG, DocumentService(FakeStore(rows)), GOVERNANCE),
        base_url="http://127.0.0.1",
    )


def test_document_list_renders_every_document(client: TestClient) -> None:
    response = client.get("/documents")
    assert response.status_code == 200
    assert "Bankacilik Kanunu" in response.text
    assert "Kurul Karari" in response.text


def test_document_list_filters_by_category(client: TestClient) -> None:
    response = client.get("/documents?category=karar")
    assert response.status_code == 200
    assert "Kurul Karari" in response.text
    assert "Bankacilik Kanunu" not in response.text


def test_root_redirects_to_documents(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/documents"


def test_invalid_page_number_does_not_error(client: TestClient) -> None:
    response = client.get("/documents?page=not-a-number")
    assert response.status_code == 200


class FakeStoreWithDetail(FakeStore):
    async def get_document(self, doc_id):
        if doc_id != "mevzuat_1":
            return None
        return StoredDocument(
            document_id="mevzuat_1",
            title="Bankacilik Kanunu",
            category="mevzuat",
            decision_date="2005-10-19",
            decision_number="5411",
            source_url="https://example.invalid/5411",
            markdown_content="# Madde 1\nAmac ve kapsam.",
            extraction_method="markitdown",
            total_pages=3,
            file_size=1024,
        )


@pytest.fixture
def detail_client() -> TestClient:
    rows = [{"document_id": "mevzuat_1", "title": "Bankacilik Kanunu", "category": "mevzuat", "total_pages": 3}]
    return TestClient(
        create_app(CONFIG, DocumentService(FakeStoreWithDetail(rows)), GOVERNANCE),
        base_url="http://127.0.0.1",
    )


def test_detail_shows_metadata_and_content(detail_client: TestClient) -> None:
    response = detail_client.get("/documents/mevzuat_1")
    assert response.status_code == 200
    assert "5411" in response.text
    assert "Amac ve kapsam." in response.text


def test_missing_document_returns_404(detail_client: TestClient) -> None:
    response = detail_client.get("/documents/does-not-exist")
    assert response.status_code == 404


class FailingListStore:
    """Every read raises, the way a downed database or a stale migration
    would fail every query on the pool."""

    async def list_documents(self, category=None, limit=100, offset=0):
        raise RuntimeError("connection to server was lost")

    async def stats(self):
        raise RuntimeError("connection to server was lost")

    async def get_document(self, doc_id):
        raise RuntimeError("connection to server was lost")


@pytest.fixture
def failing_client() -> TestClient:
    return TestClient(
        create_app(CONFIG, DocumentService(FailingListStore()), GOVERNANCE),
        base_url="http://127.0.0.1",
    )


def test_list_failure_is_shown_not_swallowed(failing_client: TestClient) -> None:
    response = failing_client.get("/documents")

    # A failed listing must never render as an empty document table.
    assert response.status_code == 200
    assert "connection to server was lost" in response.text
    assert "Kayit bulunamadi" not in response.text


def test_detail_failure_is_shown_not_swallowed_or_treated_as_missing(failing_client: TestClient) -> None:
    response = failing_client.get("/documents/mevzuat_1")

    # A failed lookup must be distinguishable from "document not found":
    # it is neither a 404 nor a blank/"not found" page.
    assert response.status_code == 200
    assert "connection to server was lost" in response.text
    assert "kayitli degil" not in response.text


class PaginatedStore:
    """51 rows so DocumentService.list_page's default page_size=50 leaves
    exactly one row for a Next page, exercising has_next=True end to end."""

    def __init__(self, count: int, category: str = "mevzuat & inceleme") -> None:
        self.rows = [
            {"document_id": f"doc-{i}", "title": f"Belge {i}", "category": category, "total_pages": 1}
            for i in range(count)
        ]

    async def list_documents(self, category=None, limit=100, offset=0):
        selected = [r for r in self.rows if category is None or r["category"] == category]
        return selected[offset : offset + limit]

    async def stats(self):
        return StoreStats(categories={"mevzuat & inceleme": len(self.rows)}, total_documents=len(self.rows))


def test_pagination_link_renders_and_encodes_category_ampersand() -> None:
    store = PaginatedStore(51)
    client = TestClient(
        create_app(CONFIG, DocumentService(store), GOVERNANCE),
        base_url="http://127.0.0.1",
    )

    response = client.get("/documents?category=mevzuat+%26+inceleme")

    assert response.status_code == 200
    assert "Sonraki" in response.text
    # The category contains "&"; it must be percent-encoded inside the
    # href query string, never emitted as a raw "&" that would start a
    # new (bogus) query parameter, nor HTML-escaped to "&amp;" only.
    assert "category=mevzuat%20%26%20inceleme" in response.text
    assert 'category=mevzuat & inceleme"' not in response.text
    assert "category=mevzuat &amp; inceleme" not in response.text


def test_foreign_host_header_is_rejected() -> None:
    client = TestClient(
        create_app(CONFIG, DocumentService(FakeStore([])), GOVERNANCE),
        base_url="http://127.0.0.1",
    )
    response = client.get("/documents", headers={"host": "evil.example"})
    assert response.status_code == 400
