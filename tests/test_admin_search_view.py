from __future__ import annotations

from starlette.testclient import TestClient

from bddk_mcp.admin.app import create_app
from bddk_mcp.admin.config import AdminConfig
from bddk_mcp.admin.services.documents import DocumentService
from bddk_mcp.store.doc_store import SearchHit, StoreStats

CONFIG = AdminConfig(bind_host="127.0.0.1", port=8100, database_url="postgresql://x", loopback_only=True)


class StubGovernanceService:
    """The signature panel has its own tests; these only need create_app's
    required collaborator to exist, and fail loudly if it is ever consulted."""

    async def status(self):
        raise AssertionError("governance status must not be fetched by search-view tests")


GOVERNANCE = StubGovernanceService()


class SearchStore:
    def __init__(self, hits=None, error: Exception | None = None):
        self.hits = hits or []
        self.error = error

    async def list_documents(self, category=None, limit=100, offset=0):
        return []

    async def stats(self):
        return StoreStats(categories={}, total_documents=0)

    async def search_content(self, query, limit=20, category=None):
        if self.error is not None:
            raise self.error
        return self.hits


def test_search_renders_hits() -> None:
    hit = SearchHit(document_id="mevzuat_1", title="Bankacilik Kanunu", snippet="mevduat toplama")
    client = TestClient(
        create_app(CONFIG, DocumentService(SearchStore(hits=[hit])), GOVERNANCE),
        base_url="http://127.0.0.1",
    )

    response = client.get("/search?q=mevduat")

    assert response.status_code == 200
    assert "Bankacilik Kanunu" in response.text
    assert "mevduat toplama" in response.text


def test_search_failure_is_shown_not_swallowed() -> None:
    store = SearchStore(error=RuntimeError("SEMANTIC_SEARCH_UNAVAILABLE"))
    client = TestClient(
        create_app(CONFIG, DocumentService(store), GOVERNANCE),
        base_url="http://127.0.0.1",
    )

    response = client.get("/search?q=mevduat")

    # A failed search must never render as "no results".
    assert response.status_code == 200
    assert "SEMANTIC_SEARCH_UNAVAILABLE" in response.text
    assert "Sonuc bulunamadi" not in response.text


def test_empty_query_prompts_instead_of_searching() -> None:
    client = TestClient(
        create_app(CONFIG, DocumentService(SearchStore()), GOVERNANCE),
        base_url="http://127.0.0.1",
    )
    response = client.get("/search?q=")
    assert response.status_code == 200
    assert "Arama terimi girin" in response.text
