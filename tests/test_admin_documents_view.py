from __future__ import annotations

from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from bddk_mcp.admin.app import create_app
from bddk_mcp.admin.config import AdminConfig
from bddk_mcp.admin.services.documents import DocumentService

CONFIG = AdminConfig(bind_host="127.0.0.1", port=8100, database_url="postgresql://x", loopback_only=True)


class FakeStore:
    def __init__(self, rows):
        self.rows = rows

    async def list_documents(self, category=None, limit=100, offset=0):
        selected = [r for r in self.rows if category is None or r["category"] == category]
        return selected[offset : offset + limit]

    async def stats(self):
        return SimpleNamespace(categories={"mevzuat": 2}, total_documents=len(self.rows))


@pytest.fixture
def client() -> TestClient:
    rows = [
        {"document_id": "mevzuat_1", "title": "Bankacilik Kanunu", "category": "mevzuat", "total_pages": 3},
        {"document_id": "karar_2", "title": "Kurul Karari", "category": "karar", "total_pages": 1},
    ]
    return TestClient(create_app(CONFIG, DocumentService(FakeStore(rows))))


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
        return SimpleNamespace(
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
    return TestClient(create_app(CONFIG, DocumentService(FakeStoreWithDetail(rows))))


def test_detail_shows_metadata_and_content(detail_client: TestClient) -> None:
    response = detail_client.get("/documents/mevzuat_1")
    assert response.status_code == 200
    assert "5411" in response.text
    assert "Amac ve kapsam." in response.text


def test_missing_document_returns_404(detail_client: TestClient) -> None:
    response = detail_client.get("/documents/does-not-exist")
    assert response.status_code == 404
