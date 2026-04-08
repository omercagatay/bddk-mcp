"""Tests for doc_sync.py — document download and extraction pipeline."""

from unittest.mock import AsyncMock

import httpx
import pytest

from doc_store import DocumentStore, StoredDocument
from doc_sync import (
    DocumentSyncer,
    _extract_html_to_markdown,
    _fetch_with_retry,
    _mevzuat_doc_url,
    _mevzuat_pdf_url,
    _parse_mevzuat_params,
)
from tests.conftest import make_http_response

# -- URL helpers -----------------------------------------------------------


class TestMevzuatUrlHelpers:
    def test_mevzuat_pdf_url(self):
        url = _mevzuat_pdf_url("42628", "7", "5")
        assert url == "https://www.mevzuat.gov.tr/MevzuatMetin/yonetmelik/7.5.42628.pdf"

    def test_mevzuat_pdf_url_unknown_tur(self):
        assert _mevzuat_pdf_url("123", "99", "5") is None

    def test_mevzuat_doc_url(self):
        url = _mevzuat_doc_url("42628", "7", "5")
        assert url == "https://www.mevzuat.gov.tr/MevzuatMetin/yonetmelik/7.5.42628.doc"

    def test_parse_mevzuat_params_new_format(self):
        url = "https://mevzuat.gov.tr/mevzuat?MevzuatNo=42628&MevzuatTur=7&MevzuatTertip=5"
        no, tur, tertip = _parse_mevzuat_params(url)
        assert no == "42628"
        assert tur == "7"
        assert tertip == "5"

    def test_parse_mevzuat_params_old_format(self):
        url = "http://www.mevzuat.gov.tr/Metin.Aspx?MevzuatKod=7.5.24788"
        no, tur, tertip = _parse_mevzuat_params(url)
        assert no == "24788"
        assert tur == "7"
        assert tertip == "5"

    def test_parse_mevzuat_params_empty(self):
        no, tur, tertip = _parse_mevzuat_params("https://example.com")
        assert no == ""
        assert tur == "7"  # defaults


# -- Extraction backends ---------------------------------------------------


class TestHtmlToMarkdown:
    def test_basic_html(self):
        html = "<h1>Title</h1><p>Paragraph text.</p><h2>Section</h2><li>Item 1</li>"
        md = _extract_html_to_markdown(html)
        assert "# Title" in md
        assert "Paragraph text." in md
        assert "## Section" in md
        assert "- Item 1" in md

    def test_empty_html(self):
        assert _extract_html_to_markdown("<html></html>") == ""

    def test_strips_scripts(self):
        html = "<p>Real content</p><script>alert('evil')</script>"
        md = _extract_html_to_markdown(html)
        assert "Real content" in md
        assert "alert" not in md

    def test_strips_styles(self):
        html = "<p>Content</p><style>body { color: red; }</style>"
        md = _extract_html_to_markdown(html)
        assert "Content" in md
        assert "color" not in md


class TestFetchWithRetry:
    @pytest.mark.asyncio
    async def test_success(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.get = AsyncMock(return_value=make_http_response("OK"))

        resp = await _fetch_with_retry(http, "https://example.com")
        assert resp.text == "OK"

    @pytest.mark.asyncio
    async def test_retry_then_succeed(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        ok_resp = make_http_response("OK")
        http.get = AsyncMock(side_effect=[httpx.TransportError("fail"), ok_resp])

        resp = await _fetch_with_retry(http, "https://example.com")
        assert resp.text == "OK"

    @pytest.mark.asyncio
    async def test_all_retries_fail(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.get = AsyncMock(side_effect=httpx.TransportError("fail"))

        with pytest.raises(httpx.TransportError):
            await _fetch_with_retry(http, "https://example.com")


# -- DocumentSyncer -------------------------------------------------------


class TestDocumentSyncer:
    @pytest.fixture
    async def store(self, doc_store):
        yield doc_store

    @pytest.mark.asyncio
    async def test_sync_cached_document_skips(self, store):
        # Pre-store a document
        await store.store_document(
            StoredDocument(
                document_id="1291",
                title="Test",
                markdown_content="Some content",
            )
        )

        async with DocumentSyncer(store, prefer_nougat=False) as syncer:
            result = await syncer.sync_document(doc_id="1291")
            assert result.success is True
            assert result.method == "cached"

    @pytest.mark.asyncio
    async def test_sync_bddk_document(self, store):
        async with DocumentSyncer(store, prefer_nougat=False) as syncer:
            # Mock HTTP response
            html_content = "<html><body><h1>Test</h1><p>Content here</p></body></html>"
            syncer._http = AsyncMock(spec=httpx.AsyncClient)
            syncer._http.get = AsyncMock(
                return_value=make_http_response(
                    text=html_content,
                    content_type="text/html",
                )
            )

            result = await syncer.sync_document(
                doc_id="100",
                title="Test Document",
                category="Rehber",
                force=True,
            )
            assert result.success is True
            assert "html_parser" in result.method

            # Verify it's in the store
            assert await store.has_document("100")

    @pytest.mark.asyncio
    async def test_sync_unknown_id_format(self, store):
        async with DocumentSyncer(store, prefer_nougat=False) as syncer:
            result = await syncer.sync_document(doc_id="weird-format-123")
            assert result.success is False
            assert "Unknown" in result.error

    @pytest.mark.asyncio
    async def test_sync_mevzuat_htm_layer(self, store):
        async with DocumentSyncer(store, prefer_nougat=False) as syncer:
            # Content must be >200 bytes to pass the .htm layer check
            html_content = (
                "<html><body><h1>Yönetmelik Başlığı</h1>"
                "<p>Madde 1 — Bu yönetmelik bankacılık sektöründe faiz oranı riskinin "
                "yönetimine ilişkin usul ve esasları düzenler. Banka sermaye yeterliliği "
                "hesaplamalarında kullanılacak yöntemler aşağıda belirtilmiştir.</p>"
                "</body></html>"
            )
            syncer._http = AsyncMock(spec=httpx.AsyncClient)
            syncer._http.get = AsyncMock(
                return_value=make_http_response(
                    text=html_content,
                    content_type="text/html",
                )
            )

            result = await syncer.sync_document(
                doc_id="mevzuat_42628",
                source_url="https://mevzuat.gov.tr/mevzuat?MevzuatNo=42628&MevzuatTur=7&MevzuatTertip=5",
                force=True,
            )
            assert result.success is True

    @pytest.mark.asyncio
    async def test_sync_all_with_concurrency(self, store):
        async with DocumentSyncer(store, prefer_nougat=False) as syncer:
            html = "<html><body><h1>Doc</h1><p>Content</p></body></html>"
            syncer._http = AsyncMock(spec=httpx.AsyncClient)
            syncer._http.get = AsyncMock(return_value=make_http_response(text=html, content_type="text/html"))

            docs = [
                {"document_id": "101", "title": "A", "category": "Rehber", "source_url": ""},
                {"document_id": "102", "title": "B", "category": "Genelge", "source_url": ""},
            ]

            # concurrency=1 because test fixture uses a single-connection pool
            report = await syncer.sync_all(docs, concurrency=1, force=True)
            assert report.total == 2
            assert report.downloaded == 2
            assert report.failed == 0

    @pytest.mark.asyncio
    async def test_sync_all_handles_failures(self, store):
        async with DocumentSyncer(store, prefer_nougat=False) as syncer:
            syncer._http = AsyncMock(spec=httpx.AsyncClient)
            syncer._http.get = AsyncMock(side_effect=httpx.TransportError("network error"))

            docs = [
                {"document_id": "200", "title": "Fail", "category": "", "source_url": ""},
            ]

            report = await syncer.sync_all(docs, concurrency=1, force=True)
            assert report.total == 1
            assert report.failed == 1

    @pytest.mark.asyncio
    async def test_force_redownload(self, store):
        # Pre-store
        await store.store_document(
            StoredDocument(
                document_id="300",
                title="Old",
                markdown_content="Old content",
            )
        )

        async with DocumentSyncer(store, prefer_nougat=False) as syncer:
            html = "<html><body><h1>New</h1><p>New content</p></body></html>"
            syncer._http = AsyncMock(spec=httpx.AsyncClient)
            syncer._http.get = AsyncMock(return_value=make_http_response(text=html, content_type="text/html"))

            result = await syncer.sync_document(doc_id="300", force=True)
            assert result.success is True
            assert result.method != "cached"
