"""Tests for doc_sync.py — document download and extraction pipeline."""

import io
import socket
import sys
import zipfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import bddk_mcp.ingest.doc_sync as doc_sync_module
from bddk_mcp.db_lifecycle import DatabaseNotReadyError
from bddk_mcp.ingest.doc_sync import (
    DocumentSyncer,
    UnsafeUpstreamResourceError,
    _create_pool_and_store,
    _decode_html,
    _extract_annex_zip_markdown,
    _extract_docx_text,
    _extract_html_to_markdown,
    _mevzuat_doc_url,
    _mevzuat_generate_pdf_url,
    _mevzuat_pdf_url,
    _normalize_mevzuat_url,
    _parse_mevzuat_params,
    _validate_office_archive_for_markitdown,
)
from bddk_mcp.ocr.base import MarkitdownBackend
from bddk_mcp.store.doc_store import StoredDocument


@pytest.mark.asyncio
async def test_standalone_cli_store_creation_is_readiness_only():
    pool = MagicMock()
    pool.close = AsyncMock()
    readiness = AsyncMock()
    document_store = MagicMock()
    document_store.initialize = AsyncMock()
    vector_store = MagicMock()
    vector_store.initialize = AsyncMock()
    identity_readiness = AsyncMock()

    with (
        patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)) as create_pool,
        patch("bddk_mcp.db_transport.assert_database_transport", side_effect=lambda value: value),
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=readiness),
        patch("bddk_mcp.db_identity.assert_database_identity", new=identity_readiness),
        patch("bddk_mcp.ingest.doc_sync.DocumentStore", return_value=document_store),
        patch("bddk_mcp.store.vector_store.VectorStore", return_value=vector_store),
    ):
        actual_pool, actual_store, actual_vector_store = await _create_pool_and_store("postgresql://example")

    assert (actual_pool, actual_store, actual_vector_store) == (pool, document_store, vector_store)
    create_pool.assert_awaited_once()
    assert create_pool.await_args.args == ("postgresql://example",)
    assert create_pool.await_args.kwargs["min_size"] == 1
    assert create_pool.await_args.kwargs["max_size"] == 5
    pool_init = create_pool.await_args.kwargs["init"]
    assert pool_init.keywords == {"profile": "ingestion"}
    readiness.assert_awaited_once_with(pool=pool, require_corpus=False)
    identity_readiness.assert_awaited_once_with(pool, "ingestion")
    document_store.initialize.assert_not_awaited()
    vector_store.initialize.assert_not_awaited()
    pool.execute.assert_not_called()
    pool.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_standalone_cli_closes_pool_when_schema_is_not_ready():
    pool = MagicMock()
    pool.close = AsyncMock()
    readiness = AsyncMock(side_effect=DatabaseNotReadyError("migration required"))
    document_store_class = MagicMock()
    vector_store_class = MagicMock()

    with (
        patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)),
        patch("bddk_mcp.db_transport.assert_database_transport", side_effect=lambda value: value),
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=readiness),
        patch("bddk_mcp.ingest.doc_sync.DocumentStore", new=document_store_class),
        patch("bddk_mcp.store.vector_store.VectorStore", new=vector_store_class),
        pytest.raises(DatabaseNotReadyError, match="migration required"),
    ):
        await _create_pool_and_store("postgresql://example")

    pool.close.assert_awaited_once_with()
    document_store_class.assert_not_called()
    vector_store_class.assert_not_called()


# -- URL helpers -----------------------------------------------------------


class TestMevzuatUrlHelpers:
    def test_mevzuat_pdf_url(self):
        url = _mevzuat_pdf_url("42628", "7", "5")
        assert url == "https://www.mevzuat.gov.tr/MevzuatMetin/yonetmelik/7.5.42628.pdf"

    def test_mevzuat_pdf_url_unknown_tur(self):
        assert _mevzuat_pdf_url("123", "99", "5") is None

    def test_mevzuat_generate_pdf_url_yonetmelik(self):
        url = _mevzuat_generate_pdf_url("42628", "7", "5")
        assert (
            url == "https://www.mevzuat.gov.tr/File/GeneratePdf?mevzuatNo=42628&mevzuatTur=Yonetmelik&mevzuatTertip=5"
        )

    def test_mevzuat_generate_pdf_url_teblig(self):
        url = _mevzuat_generate_pdf_url("21196", "9", "5")
        assert url == "https://www.mevzuat.gov.tr/File/GeneratePdf?mevzuatNo=21196&mevzuatTur=Teblig&mevzuatTertip=5"

    def test_mevzuat_generate_pdf_url_kanun(self):
        url = _mevzuat_generate_pdf_url("5411", "1", "5")
        assert url == "https://www.mevzuat.gov.tr/File/GeneratePdf?mevzuatNo=5411&mevzuatTur=Kanun&mevzuatTertip=5"

    def test_mevzuat_generate_pdf_url_unknown_tur(self):
        assert _mevzuat_generate_pdf_url("123", "99", "5") is None

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

    @pytest.mark.parametrize(
        "candidate",
        [
            "http://www.mevzuat.gov.tr/resource",
            "https://www.mevzuat.gov.tr.evil.example/resource",
            "https://user:secret@www.mevzuat.gov.tr/resource",
            "https://www.mevzuat.gov.tr:8443/resource",
            "//127.0.0.1/private",
            "https://www.mevzuat.gov.tr\\@evil.example/resource",
        ],
    )
    def test_normalize_mevzuat_url_rejects_boundary_bypasses(self, candidate):
        with pytest.raises(UnsafeUpstreamResourceError):
            _normalize_mevzuat_url(candidate)

    def test_normalize_mevzuat_url_accepts_relative_path_and_strips_fragment(self):
        assert (
            _normalize_mevzuat_url("/api/Mevzuat/42628/IframeDetay#section")
            == "https://www.mevzuat.gov.tr/api/Mevzuat/42628/IframeDetay"
        )


@pytest.mark.asyncio
async def test_bounded_mevzuat_fetch_revalidates_redirect_target():
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://evil.example/private"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        syncer = DocumentSyncer(object(), http=client, ocr_backends=[MarkitdownBackend()])
        syncer._assert_public_mevzuat_resolution = AsyncMock(return_value=None)
        with pytest.raises(UnsafeUpstreamResourceError, match="approved mevzuat HTTPS boundary"):
            await syncer._fetch_trusted_mevzuat(
                "/redirect",
                timeout=httpx.Timeout(1),
                max_bytes=1024,
            )

    assert requests == ["https://www.mevzuat.gov.tr/redirect"]


@pytest.mark.asyncio
async def test_bounded_mevzuat_fetch_rejects_private_dns_before_request(monkeypatch):
    requested = False

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requested
        requested = True
        return httpx.Response(200, content=b"should not be reached")

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        syncer = DocumentSyncer(object(), http=client, ocr_backends=[MarkitdownBackend()])
        with pytest.raises(UnsafeUpstreamResourceError, match="outside the public network"):
            await syncer._fetch_trusted_mevzuat(
                "/private-dns",
                timeout=httpx.Timeout(1),
                max_bytes=1024,
            )

    assert requested is False


@pytest.mark.asyncio
async def test_bounded_mevzuat_fetch_stops_oversized_response():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 64)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        syncer = DocumentSyncer(object(), http=client, ocr_backends=[MarkitdownBackend()])
        syncer._assert_public_mevzuat_resolution = AsyncMock(return_value=None)
        with pytest.raises(UnsafeUpstreamResourceError, match="download limit"):
            await syncer._fetch_trusted_mevzuat(
                "/oversized",
                timeout=httpx.Timeout(1),
                max_bytes=32,
            )


@pytest.mark.asyncio
async def test_bounded_fetch_stops_chunked_response_without_content_length():
    yielded_chunks = 0

    class ChunkedBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            nonlocal yielded_chunks
            for chunk in (b"a" * 12, b"b" * 12, b"c" * 12, b"d" * 12):
                yielded_chunks += 1
                yield chunk

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=ChunkedBody())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        syncer = DocumentSyncer(object(), http=client, ocr_backends=[MarkitdownBackend()])
        syncer._assert_public_mevzuat_resolution = AsyncMock(return_value=None)
        with pytest.raises(UnsafeUpstreamResourceError, match="download limit"):
            await syncer._fetch_trusted_mevzuat(
                "/chunked-oversized",
                timeout=httpx.Timeout(1),
                max_bytes=30,
            )

    assert yielded_chunks == 3, "stream must be abandoned immediately after crossing the byte limit"


@pytest.mark.asyncio
async def test_bddk_bounded_fetch_retries_transport_429_and_5xx(monkeypatch):
    syncer = DocumentSyncer(object(), ocr_backends=[MarkitdownBackend()])
    fetch = AsyncMock(
        side_effect=[
            httpx.ConnectError("synthetic transport failure"),
            SimpleNamespace(status_code=429, content=b"", headers={}),
            SimpleNamespace(status_code=503, content=b"", headers={}),
            SimpleNamespace(
                status_code=200,
                content=b"%PDF-1.4\nbody",
                headers={"content-type": "application/pdf"},
            ),
        ]
    )
    sleep = AsyncMock()
    monkeypatch.setattr(syncer, "_fetch_bounded_https", fetch)
    monkeypatch.setattr("bddk_mcp.ingest.doc_sync.MAX_RETRIES", 100)
    monkeypatch.setattr("bddk_mcp.ingest.doc_sync.asyncio.sleep", sleep)

    response = await syncer._fetch_trusted_bddk(
        "https://www.bddk.org.tr/Mevzuat/DokumanGetir/12345",
        timeout=httpx.Timeout(1),
        max_bytes=1024,
    )

    assert response.status_code == 200
    assert fetch.await_count == 4
    assert sleep.await_count == 3
    assert fetch.await_args_list[0].kwargs["read_body_statuses"] == frozenset({200})
    await syncer.close()


@pytest.mark.asyncio
async def test_bddk_bounded_fetch_does_not_retry_non_429_4xx(monkeypatch):
    syncer = DocumentSyncer(object(), ocr_backends=[MarkitdownBackend()])
    fetch = AsyncMock(return_value=SimpleNamespace(status_code=404, content=b"", headers={}))
    sleep = AsyncMock()
    monkeypatch.setattr(syncer, "_fetch_bounded_https", fetch)
    monkeypatch.setattr("bddk_mcp.ingest.doc_sync.asyncio.sleep", sleep)

    response = await syncer._fetch_trusted_bddk(
        "https://www.bddk.org.tr/Mevzuat/DokumanGetir/12345",
        timeout=httpx.Timeout(1),
        max_bytes=1024,
    )

    assert response.status_code == 404
    fetch.assert_awaited_once()
    sleep.assert_not_awaited()
    await syncer.close()


@pytest.mark.asyncio
async def test_bddk_bounded_fetch_sanitizes_transport_exhaustion(monkeypatch):
    syncer = DocumentSyncer(object(), ocr_backends=[MarkitdownBackend()])
    fetch = AsyncMock(side_effect=httpx.ConnectError("https://secret.invalid/private"))
    sleep = AsyncMock()
    monkeypatch.setattr(syncer, "_fetch_bounded_https", fetch)
    monkeypatch.setattr("bddk_mcp.ingest.doc_sync.MAX_RETRIES", 2)
    monkeypatch.setattr("bddk_mcp.ingest.doc_sync.asyncio.sleep", sleep)

    with pytest.raises(UnsafeUpstreamResourceError) as exc_info:
        await syncer._fetch_trusted_bddk(
            "https://www.bddk.org.tr/Mevzuat/DokumanGetir/12345",
            timeout=httpx.Timeout(1),
            max_bytes=1024,
        )

    assert "secret" not in str(exc_info.value)
    assert fetch.await_count == 2
    assert sleep.await_count == 1
    await syncer.close()


@pytest.mark.asyncio
async def test_bddk_download_uses_exact_host_bounded_fetch():
    syncer = DocumentSyncer(object(), ocr_backends=[MarkitdownBackend()])
    response = SimpleNamespace(
        status_code=200,
        content=b"%PDF-1.4\nbody",
        headers={"content-type": "application/pdf"},
    )
    syncer._fetch_trusted_bddk = AsyncMock(return_value=response)

    content, method, extension = await syncer._download_bddk("12345")

    assert (content, method, extension) == (response.content, "bddk_direct", ".pdf")
    call = syncer._fetch_trusted_bddk.await_args
    assert call.args == ("https://www.bddk.org.tr/Mevzuat/DokumanGetir/12345",)
    assert call.kwargs["max_bytes"] > 0
    await syncer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            SimpleNamespace(status_code=503, content=b"", headers={"content-type": "text/html"}),
            "successful document response",
        ),
        (
            SimpleNamespace(
                status_code=200,
                content=b"<html><body>not a PDF</body></html>",
                headers={"content-type": "application/pdf"},
            ),
            "PDF response signature",
        ),
        (
            SimpleNamespace(
                status_code=200,
                content=b'{"error":"not a document"}',
                headers={"content-type": "application/json"},
            ),
            "unsupported document media type",
        ),
        (
            SimpleNamespace(
                status_code=200,
                content=b"<html><head><title>Service Unavailable</title></head><body>retry later</body></html>",
                headers={"content-type": "text/html"},
            ),
            "error page instead of a document",
        ),
    ],
)
async def test_bddk_download_rejects_error_media_and_signature(response, message):
    syncer = DocumentSyncer(object(), ocr_backends=[MarkitdownBackend()])
    syncer._fetch_trusted_bddk = AsyncMock(return_value=response)

    with pytest.raises(UnsafeUpstreamResourceError, match=message):
        await syncer._download_bddk("12345")

    await syncer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("document_id", ["../admin", "1?target=other", "", "1" * 21])
async def test_bddk_download_rejects_invalid_document_identifier(document_id):
    syncer = DocumentSyncer(object(), ocr_backends=[MarkitdownBackend()])
    syncer._fetch_trusted_bddk = AsyncMock()

    with pytest.raises(UnsafeUpstreamResourceError, match="identifier is invalid"):
        await syncer._download_bddk(document_id)

    syncer._fetch_trusted_bddk.assert_not_awaited()
    await syncer.close()


@pytest.mark.asyncio
async def test_mevzuat_download_rejects_invalid_source_identifiers_before_network():
    syncer = DocumentSyncer(object(), ocr_backends=[MarkitdownBackend()])
    syncer._fetch_trusted_mevzuat = AsyncMock()

    with pytest.raises(UnsafeUpstreamResourceError, match="document type is invalid"):
        await syncer._download_mevzuat(
            "mevzuat_42628",
            "https://www.mevzuat.gov.tr/mevzuat?MevzuatNo=42628&MevzuatTur=7%26admin%3D1&MevzuatTertip=5",
        )

    syncer._fetch_trusted_mevzuat.assert_not_awaited()
    await syncer.close()


@pytest.mark.asyncio
async def test_mevzuat_download_aborts_after_bounded_fetch_rejection():
    syncer = DocumentSyncer(object(), ocr_backends=[MarkitdownBackend()])
    syncer._fetch_trusted_mevzuat = AsyncMock(
        side_effect=UnsafeUpstreamResourceError("Upstream response exceeds the download limit.")
    )

    with pytest.raises(UnsafeUpstreamResourceError, match="download limit"):
        await syncer._download_mevzuat("mevzuat_42628")

    assert syncer._fetch_trusted_mevzuat.await_count == 1
    await syncer.close()


# -- Extraction backends ---------------------------------------------------


class TestDecodeHtml:
    def test_does_not_silently_return_replacement_chars_in_body(self):
        # Regression for stored mevzuat_* docs. Prior version checked
        # only decoded[:500] for U+FFFD and would return utf-8 decoded
        # content even when body bytes contained literal EF BF BD,
        # baking replacement chars into the document store.
        content = (b"A" * 600) + b"\xef\xbf\xbd test"
        result = _decode_html(content)

        assert "\ufffd" not in result

    def test_maps_c1_en_dash_to_real_en_dash(self):
        # SYSTEMIC-8. Mevzuat HTML sometimes carries a literal U+0096 (C1
        # control) where an en-dash belongs — residue from Word export that
        # stored the Windows-1252 byte 0x96 as its Unicode code point
        # rather than mapping it to U+2013. After decode we remap the
        # whole C1 block back to printable Windows-1252 equivalents so
        # nothing downstream has to look at tofu boxes.
        content = "MADDE 1  (1) Bu Yönetmelik".encode()
        result = _decode_html(content)
        assert "" not in result
        assert "MADDE 1 – (1)" in result

    def test_maps_all_seven_observed_c1_offenders(self):
        # The seven C1 code points actually observed in stored html_parser
        # docs (per error_reports.md SYSTEMIC-8 audit). One assertion over
        # the full roster so the mapping stays auditable.
        raw = "      ".encode()
        assert _decode_html(raw) == "‘ ’ “ ” – — …"


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

    def test_adjacent_inline_tags_get_separated(self):
        html = "<p><span>YÖNETMELİK</span><span>BİRİNCİ BÖLÜM</span></p>"
        md = _extract_html_to_markdown(html)
        assert "YÖNETMELİKBİRİNCİ" not in md
        assert "YÖNETMELİK" in md
        assert "BİRİNCİ BÖLÜM" in md

    def test_mevzuat_heading_pattern(self):
        html = (
            "<p><b>YÖNETMELİK</b></p>"
            "<p><b>BİRİNCİ BÖLÜM</b></p>"
            "<p><b>Başlangıç Hükümleri</b></p>"
            "<p><b>Amaç ve kapsam</b></p>"
        )
        md = _extract_html_to_markdown(html)
        assert "BÖLÜMBaşlangıç" not in md
        assert "HükümleriAmaç" not in md

    def test_nested_block_elements_dont_duplicate(self):
        html = "<table><tr><td><p>Cell Content</p></td></tr></table>"
        md = _extract_html_to_markdown(html)
        assert md.count("Cell Content") == 1

    def test_bold_inline_preserved(self):
        html = "<p><b>Madde 1</b> — Bu yönetmelik düzenler.</p>"
        md = _extract_html_to_markdown(html)
        assert "**Madde 1**" in md

    def test_bold_via_font_weight_style(self):
        html = '<p><span style="font-weight:700">Önemli</span> metin.</p>'
        md = _extract_html_to_markdown(html)
        assert "**Önemli**" in md

    def test_italic_inline_preserved(self):
        html = "<p>Bu <i>italik</i> örnektir.</p>"
        md = _extract_html_to_markdown(html)
        assert "*italik*" in md

    def test_table_rendered_as_gfm(self):
        html = (
            "<table>"
            "<tr><th>Sütun A</th><th>Sütun B</th></tr>"
            "<tr><td>a1</td><td>b1</td></tr>"
            "<tr><td>a2</td><td>b2</td></tr>"
            "</table>"
        )
        md = _extract_html_to_markdown(html)
        assert "| Sütun A | Sütun B |" in md
        assert "|---|---|" in md
        assert "| a1 | b1 |" in md
        assert "| a2 | b2 |" in md

    def test_table_colspan_flattened(self):
        html = "<table><tr><th colspan='2'>Birleşik Başlık</th></tr><tr><td>x</td><td>y</td></tr></table>"
        md = _extract_html_to_markdown(html)
        assert "| Birleşik Başlık |" in md
        assert "| x | y |" in md

    def test_formula_image_preserved(self):
        html = '<p>x = <img src="formul_1.gif" alt="eq"/> + 1</p>'
        md = _extract_html_to_markdown(html)
        assert "![eq](formul_1.gif)" in md

    def test_bolum_heading_promoted(self):
        html = "<p>BİRİNCİ BÖLÜM</p><p>Başlangıç Hükümleri</p>"
        md = _extract_html_to_markdown(html)
        assert "## BİRİNCİ BÖLÜM" in md

    def test_ek_heading_promoted(self):
        html = "<p>EK-1</p><p>Hesaplama Tablosu</p>"
        md = _extract_html_to_markdown(html)
        assert "## EK-1" in md

    def test_paragraph_with_inline_bold_and_plain(self):
        """Bold run inside a paragraph must not swallow surrounding plain text."""
        html = "<p>Öncesi <b>vurgu</b> sonrası metin.</p>"
        md = _extract_html_to_markdown(html)
        assert "Öncesi" in md
        assert "**vurgu**" in md
        assert "sonrası metin." in md

    def test_link_preserves_href(self):
        html = '<p>Bkz. <a href="https://example.com/x">buraya</a> bakın.</p>'
        md = _extract_html_to_markdown(html)
        assert "[buraya](https://example.com/x)" in md

    def test_cross_cell_body_duplication_is_deduped(self):
        # SYSTEMIC-10. A handful of mevzuat pages (seen in mevzuat_24654)
        # duplicate the entire article body across multiple <td> cells —
        # sometimes within one table, sometimes across sibling tables —
        # because of a layout quirk CSS hides in the browser. Without a
        # defense, our extractor faithfully emits every copy and the user
        # sees doubled/tripled prose. Large cell content (≥ the dedup
        # threshold) that matches an earlier cell in the same document
        # should be dropped.
        body = "Uzun madde metni ilave edilmelidir. " * 50  # ≈ 1 850 chars
        html = f"<table><tr><td>{body}</td><td>{body}</td></tr></table><table><tr><td>{body}</td></tr></table>"
        md = _extract_html_to_markdown(html)
        # `body` contains the phrase 50 times internally; emitted once = 50 hits.
        assert md.count("Uzun madde metni") == 50

    def test_short_duplicate_cells_are_kept(self):
        # Labels, numbers, single words repeat legitimately across many
        # tables and must not be stripped by the dedup pass.
        html = "<table><tr><td>1</td><td>2</td><td>1</td></tr><tr><td>A</td><td>B</td><td>A</td></tr></table>"
        md = _extract_html_to_markdown(html)
        assert md.count("| 1 |") >= 1
        assert md.count("| A |") >= 1

    def test_long_but_distinct_cells_both_retained(self):
        # Dedup must not flatten a real multi-column table whose cells
        # happen to be long but carry different content.
        a_block = "Madde hükmü A. " * 80
        b_block = "Madde hükmü B. " * 80
        html = f"<table><tr><td>{a_block}</td><td>{b_block}</td></tr></table>"
        md = _extract_html_to_markdown(html)
        assert "Madde hükmü A." in md
        assert "Madde hükmü B." in md


class TestSanitizeForStorage:
    def test_strips_nul_bytes(self):
        from bddk_mcp.ingest.doc_sync import _sanitize_for_storage

        assert _sanitize_for_storage("hello\x00world") == "helloworld"

    def test_preserves_clean_text(self):
        from bddk_mcp.ingest.doc_sync import _sanitize_for_storage

        clean = "Madde 1 — Bankaların risk yönetimi."
        assert _sanitize_for_storage(clean) is clean

    def test_preserves_other_control_chars(self):
        from bddk_mcp.ingest.doc_sync import _sanitize_for_storage

        assert _sanitize_for_storage("line1\nline2\tcol") == "line1\nline2\tcol"

    def test_empty_is_passed_through(self):
        from bddk_mcp.ingest.doc_sync import _sanitize_for_storage

        assert _sanitize_for_storage("") == ""

    def test_strips_form_feeds(self):
        # SYSTEMIC-3. Markitdown leaves PDF page-break bytes (0x0C) in output.
        # Visual noise with no semantic value — strip in the same pass that
        # removes storage-unsafe NULs.
        from bddk_mcp.ingest.doc_sync import _sanitize_for_storage

        assert _sanitize_for_storage("page1\x0cpage2\x0cpage3") == "page1page2page3"

    def test_replaces_garbled_turkish_capital_i(self):
        # SYSTEMIC-1. Markitdown's PDF path decodes Turkish capital İ (U+0130)
        # as Đ (U+0110) on BDDK legacy PDFs whose embedded font lacks a
        # ToUnicode CMap. Blanket Đ→İ is safe because every Đ observed
        # across 43 affected docs / 235 occurrences is a garbled İ, and Đ
        # (Croatian/Vietnamese) never legitimately appears in Turkish
        # regulatory text — verified by auditing every non-ASCII Turkish
        # character in the document store.
        from bddk_mcp.ingest.doc_sync import _sanitize_for_storage

        assert _sanitize_for_storage("Tevfik BĐLGĐN") == "Tevfik BİLGİN"
        assert _sanitize_for_storage("Đhraççı bankanın") == "İhraççı bankanın"

    def test_all_three_artifacts_in_one_pass(self):
        # Combined fix: one pass handles NUL (storage-unsafe) + form-feed
        # (SYSTEMIC-3) + Đ-garble (SYSTEMIC-1). A markitdown output with
        # all three defects gets cleaned in a single sweep.
        from bddk_mcp.ingest.doc_sync import _sanitize_for_storage

        assert _sanitize_for_storage("BĐLGĐN\x0cĐhraççı\x00") == "BİLGİNİhraççı"

    def test_uses_shared_markdown_quality_storage_rules(self):
        from bddk_mcp.ingest.doc_sync import _sanitize_for_storage

        out = _sanitize_for_storage("A\u00a0B\u200bC\n****\n" + "_" * 80)

        assert "\u00a0" not in out
        assert "\u200b" not in out
        assert "****" not in out
        assert "_" * 40 not in out
        assert "A BC" in out

    def test_strips_cid_and_data_uri_without_other_sentinels(self):
        from bddk_mcp.ingest.doc_sync import _sanitize_for_storage

        out = _sanitize_for_storage("MADDE 1 cid:image001.png@01D12345 ![](data:image/png;base64,AAA=)")

        assert "cid:" not in out
        assert "data:image/" not in out
        assert "MADDE 1" in out


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

        async with DocumentSyncer(store, ocr_backends=[MarkitdownBackend()]) as syncer:
            result = await syncer.sync_document(doc_id="1291")
            assert result.success is True
            assert result.method == "cached"

    @pytest.mark.asyncio
    async def test_sync_bddk_document(self, store):
        async with DocumentSyncer(store, ocr_backends=[MarkitdownBackend()]) as syncer:
            # Mock HTTP response
            html_content = "<html><body><h1>Test</h1><p>Content here</p></body></html>"
            syncer._fetch_trusted_bddk = AsyncMock(
                return_value=SimpleNamespace(
                    status_code=200,
                    content=html_content.encode(),
                    headers={"content-type": "text/html"},
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
        async with DocumentSyncer(store, ocr_backends=[MarkitdownBackend()]) as syncer:
            result = await syncer.sync_document(doc_id="weird-format-123")
            assert result.success is False
            assert "Unknown" in result.error

    @pytest.mark.asyncio
    async def test_sync_mevzuat_htm_layer(self, store):
        async with DocumentSyncer(store, ocr_backends=[MarkitdownBackend()]) as syncer:
            # Content must be >200 bytes to pass the .htm layer check
            html_content = (
                "<html><body><h1>Yönetmelik Başlığı</h1>"
                "<p>Madde 1 — Bu yönetmelik bankacılık sektöründe faiz oranı riskinin "
                "yönetimine ilişkin usul ve esasları düzenler. Banka sermaye yeterliliği "
                "hesaplamalarında kullanılacak yöntemler aşağıda belirtilmiştir.</p>"
                "</body></html>"
            )
            syncer._fetch_trusted_mevzuat = AsyncMock(
                return_value=SimpleNamespace(
                    status_code=200,
                    content=html_content.encode(),
                    headers={"content-type": "text/html"},
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
        async with DocumentSyncer(store, ocr_backends=[MarkitdownBackend()]) as syncer:
            html = "<html><body><h1>Doc</h1><p>Content</p></body></html>"
            syncer._fetch_trusted_bddk = AsyncMock(
                return_value=SimpleNamespace(
                    status_code=200,
                    content=html.encode(),
                    headers={"content-type": "text/html"},
                )
            )

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
        async with DocumentSyncer(store, ocr_backends=[MarkitdownBackend()]) as syncer:
            syncer._fetch_trusted_bddk = AsyncMock(side_effect=httpx.TransportError("network error"))

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

        async with DocumentSyncer(store, ocr_backends=[MarkitdownBackend()]) as syncer:
            html = "<html><body><h1>New</h1><p>New content</p></body></html>"
            syncer._fetch_trusted_bddk = AsyncMock(
                return_value=SimpleNamespace(
                    status_code=200,
                    content=html.encode(),
                    headers={"content-type": "text/html"},
                )
            )

            result = await syncer.sync_document(doc_id="300", force=True)
            assert result.success is True
            assert result.method != "cached"


@pytest.mark.asyncio
async def test_mevzuat_download_tries_pdf_before_htm():
    """With prefer_html_for_mevzuat=False, PDF paths run before HTML to preserve formulas."""
    import httpx as _httpx

    dummy_store = object()  # _download_mevzuat does not touch the store
    async with DocumentSyncer(
        dummy_store,
        ocr_backends=[MarkitdownBackend()],
        prefer_html_for_mevzuat=False,
    ) as syncer:
        call_log: list[str] = []
        main_html = b"<html><body>main page</body></html>"
        pdf_bytes = b"%PDF-1.4\n" + b"x" * 1000

        async def fake_get(url, **_kwargs):
            call_log.append(url)
            if "mevzuat?MevzuatNo=42628" in url and "MevzuatTur=7" in url:
                return SimpleNamespace(status_code=200, content=main_html, headers={"content-type": "text/html"})
            if "GeneratePdf" in url and "mevzuatNo=42628" in url:
                return SimpleNamespace(
                    status_code=200,
                    content=pdf_bytes,
                    headers={"content-type": "application/pdf"},
                )
            # Anything else: 404 so test fails if wrong layer is tried
            return SimpleNamespace(status_code=404, content=b"", headers={})

        syncer._http = AsyncMock(spec=_httpx.AsyncClient)
        syncer._fetch_trusted_mevzuat = AsyncMock(side_effect=fake_get)

        content, method, ext = await syncer._download_mevzuat("mevzuat_42628")

    assert ext == ".pdf"
    assert method == "mevzuat_generate_pdf"
    assert content.startswith(b"%PDF-")
    # Verify .htm was NOT tried before GeneratePdf
    htm_idx = next((i for i, u in enumerate(call_log) if ".htm" in u), -1)
    gen_idx = next((i for i, u in enumerate(call_log) if "GeneratePdf" in u), -1)
    if htm_idx >= 0:
        assert gen_idx < htm_idx, f"GeneratePdf must be tried before .htm; order={call_log}"


@pytest.mark.asyncio
async def test_mevzuat_download_prefers_iframe_when_html_first():
    """When prefer_html_for_mevzuat=True, the iframe layer wins before any PDF call."""
    import httpx as _httpx

    dummy_store = object()
    async with DocumentSyncer(
        dummy_store,
        ocr_backends=[MarkitdownBackend()],
        prefer_html_for_mevzuat=True,
    ) as syncer:
        call_log: list[str] = []
        main_html = (
            '<html><body><iframe id="mevzuatDetayIframe" src="/api/Mevzuat/42628/IframeDetay"></iframe></body></html>'
        )
        iframe_body = "<html><body><p><b>YONETMELIK</b></p><p>Madde 1 -- icerik.</p></body></html>" * 5

        async def fake_get(url, **_kwargs):
            call_log.append(url)
            if "mevzuat?MevzuatNo=42628" in url:
                return SimpleNamespace(status_code=200, content=main_html.encode(), headers={})
            if "IframeDetay" in url:
                return SimpleNamespace(status_code=200, content=iframe_body.encode(), headers={})
            # Anything else must not be reached before iframe success.
            return SimpleNamespace(status_code=404, content=b"", headers={})

        syncer._http = AsyncMock(spec=_httpx.AsyncClient)
        syncer._fetch_trusted_mevzuat = AsyncMock(side_effect=fake_get)

        content, method, ext = await syncer._download_mevzuat("mevzuat_42628")

    assert method == "mevzuat_iframe"
    assert ext == ".html"
    assert b"Madde 1" in content
    # iframe must be fetched and no PDF-generating call should precede it.
    iframe_idx = next((i for i, u in enumerate(call_log) if "IframeDetay" in u), -1)
    assert iframe_idx >= 0, f"iframe was never fetched; order={call_log}"
    gen_idx = next((i for i, u in enumerate(call_log) if "GeneratePdf" in u), -1)
    pdf_idx = next((i for i, u in enumerate(call_log) if u.endswith(".pdf")), -1)
    assert gen_idx == -1, f"GeneratePdf must not be tried before iframe; order={call_log}"
    assert pdf_idx == -1, f"static .pdf must not be tried before iframe; order={call_log}"


@pytest.mark.asyncio
async def test_mevzuat_iframe_download_appends_docx_annex_zip_when_present():
    """HTML-first mevzuat downloads should merge docx annex formulas linked from the iframe."""
    import httpx as _httpx

    dummy_store = object()
    async with DocumentSyncer(
        dummy_store,
        ocr_backends=[MarkitdownBackend()],
        prefer_html_for_mevzuat=True,
    ) as syncer:
        main_html = (
            '<html><body><iframe id="mevzuatDetayIframe" '
            'src="/anasayfa/MevzuatFihristDetayIframe?MevzuatTur=7&MevzuatNo=19498&MevzuatTertip=5">'
            "</iframe></body></html>"
        )
        iframe_body = (
            "<html><body><p>MADDE 9 - Ek-3’te yer alan formül uyarınca hesaplanır.</p>"
            '<a href="7.5.19498-ek.zip">Eki için tıklayınız.</a></body></html>'
        ) * 3
        docx_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>EK:3</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>Yüksek Kaliteli Likit Varlık Stoku = Birinci Kalite + 2A + 2B</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w") as zf:
            zf.writestr("EK-3.docx", _build_minimal_docx(document_xml=docx_xml))

        async def fake_trusted_get(url, **_kwargs):
            if "mevzuat?MevzuatNo=19498" in url:
                return SimpleNamespace(status_code=200, content=main_html.encode(), headers={})
            if "MevzuatFihristDetayIframe" in url:
                return SimpleNamespace(status_code=200, content=iframe_body.encode(), headers={})
            if "MevzuatMetin/yonetmelik/7.5.19498-ek.zip" in url:
                return SimpleNamespace(status_code=200, content=zip_buf.getvalue(), headers={})
            return SimpleNamespace(status_code=404, content=b"", headers={})

        syncer._http = AsyncMock(spec=_httpx.AsyncClient)
        syncer._fetch_trusted_mevzuat = AsyncMock(side_effect=fake_trusted_get)

        content, method, ext = await syncer._download_mevzuat("mevzuat_19498")

    assert method == "mevzuat_iframe+annex_zip"
    assert ext == ".html"
    assert "Yüksek Kaliteli Likit Varlık Stoku".encode() in content


def _build_minimal_docx(*, document_xml: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", document_xml)
    return buf.getvalue()


def test_annex_archive_rejects_excess_member_count(monkeypatch):
    monkeypatch.setattr(doc_sync_module, "_MAX_ARCHIVE_MEMBERS", 1)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("one.txt", "one")
        zf.writestr("two.txt", "two")

    with pytest.raises(UnsafeUpstreamResourceError, match="too many members"):
        _extract_annex_zip_markdown(archive.getvalue())


def test_annex_archive_rejects_oversized_member_metadata(monkeypatch):
    monkeypatch.setattr(doc_sync_module, "_MAX_ARCHIVE_MEMBER_BYTES", 32)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("oversized.docx", b"x" * 64)

    with pytest.raises(UnsafeUpstreamResourceError, match="member exceeds"):
        _extract_annex_zip_markdown(archive.getvalue())


def test_annex_archive_rejects_excessive_expansion_ratio(monkeypatch):
    monkeypatch.setattr(doc_sync_module, "_MAX_ARCHIVE_EXPANSION_RATIO", 2)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("compressed.docx", b"0" * 4096)

    with pytest.raises(UnsafeUpstreamResourceError, match="expansion ratio"):
        _extract_annex_zip_markdown(archive.getvalue())


def test_docx_rejects_oversized_document_xml(monkeypatch):
    monkeypatch.setattr(doc_sync_module, "_MAX_DOCX_XML_BYTES", 32)
    xml = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{'x' * 64}</w:t></w:r></w:p></w:body></w:document>"
    )

    with pytest.raises(UnsafeUpstreamResourceError, match="extraction limit"):
        _extract_docx_text(_build_minimal_docx(document_xml=xml))


def test_docx_rejects_xml_entity_expansion():
    xml = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE w:document [<!ENTITY payload "UNTRUSTED">]>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>&payload;</w:t></w:r></w:p></w:body></w:document>"
    )

    with pytest.raises(UnsafeUpstreamResourceError, match="invalid or unsupported"):
        _extract_docx_text(_build_minimal_docx(document_xml=xml))


def test_office_archive_rejects_entity_expansion_before_markitdown():
    xml = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE w:document [<!ENTITY payload "UNTRUSTED">]>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>&payload;</w:t></w:r></w:p></w:body></w:document>"
    )
    content = _build_minimal_docx(document_xml=xml)
    syncer = DocumentSyncer(object(), ocr_backends=[MarkitdownBackend()])

    result = syncer._extract_structured(content, ".doc")

    assert result.method == "failed"
    assert result.error == "office archive rejected by safety policy"
    assert result.retryable is False


def test_office_archive_rejects_expansion_ratio_before_markitdown(monkeypatch):
    monkeypatch.setattr(doc_sync_module, "_MAX_ARCHIVE_EXPANSION_RATIO", 2)
    xml = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{'0' * 4096}</w:t></w:r></w:p></w:body></w:document>"
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", xml)

    with pytest.raises(UnsafeUpstreamResourceError, match="expansion ratio"):
        _validate_office_archive_for_markitdown(archive.getvalue())


def test_office_archive_rejects_excess_member_count_before_markitdown(monkeypatch):
    monkeypatch.setattr(doc_sync_module, "_MAX_DOCX_MEMBERS", 1)
    xml = '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>'
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", xml)

    with pytest.raises(UnsafeUpstreamResourceError, match="too many members"):
        _validate_office_archive_for_markitdown(archive.getvalue())


@pytest.mark.parametrize("unsafe_name", ["../word/document.xml", "/word/document.xml", "C:/word/document.xml"])
def test_office_archive_rejects_unsafe_member_paths(unsafe_name):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(
            unsafe_name,
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"/>',
        )

    with pytest.raises(UnsafeUpstreamResourceError, match="member layout is unsafe"):
        _validate_office_archive_for_markitdown(archive.getvalue())


def test_validated_zip_office_input_is_routed_to_docx_converter(monkeypatch):
    converted_extensions: list[str] = []

    class FakeMarkItDown:
        def convert_stream(self, _stream, *, file_extension):
            converted_extensions.append(file_extension)
            return SimpleNamespace(text_content="validated office body " * 30)

    monkeypatch.setitem(sys.modules, "markitdown", SimpleNamespace(MarkItDown=FakeMarkItDown))
    xml = (
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body><w:p><w:r><w:t>Validated body</w:t></w:r></w:p></w:body></w:document>"
    )
    syncer = DocumentSyncer(object(), ocr_backends=[MarkitdownBackend()])

    result = syncer._extract_structured(_build_minimal_docx(document_xml=xml), ".doc")

    assert result.method == "markitdown"
    assert converted_extensions == [".docx"]


def test_resolve_html_first_flag_auto_detects_markitdown_only():
    """With only MarkitdownBackend available, auto mode must flip to True."""
    dummy_store = object()
    syncer = DocumentSyncer(dummy_store, ocr_backends=[MarkitdownBackend()])
    assert syncer._prefer_html_for_mevzuat is True


def test_resolve_html_first_flag_explicit_false_overrides_auto():
    dummy_store = object()
    syncer = DocumentSyncer(
        dummy_store,
        ocr_backends=[MarkitdownBackend()],
        prefer_html_for_mevzuat=False,
    )
    assert syncer._prefer_html_for_mevzuat is False


def test_resolve_html_first_flag_formula_capable_backend_flips_auto_false():
    """When a non-markitdown backend reports available, auto should keep PDF-first."""

    class _FakeGPUBackend:
        name = "fake_gpu"

        def is_available(self) -> bool:
            return True

        def extract(self, pdf_bytes: bytes):
            return None

    dummy_store = object()
    syncer = DocumentSyncer(
        dummy_store,
        ocr_backends=[_FakeGPUBackend(), MarkitdownBackend()],
    )
    assert syncer._prefer_html_for_mevzuat is False


@pytest.mark.asyncio
async def test_force_reextract_failure_preserves_old_content(doc_store):
    """When force=True and new extraction fails, old markdown must remain in DB."""
    original = StoredDocument(
        document_id="42628",
        title="Test doc",
        markdown_content="ORIGINAL CONTENT",
        extraction_method="lightocr",
    )
    await doc_store.store_document(original)

    class _AlwaysFailBackend:
        name = "test_fail"

        def is_available(self) -> bool:
            return True

        def extract(self, pdf_bytes: bytes) -> str | None:
            return None

    async with DocumentSyncer(doc_store, ocr_backends=[_AlwaysFailBackend()]) as syncer:
        # Stub HTTP to return a PDF so download succeeds but extraction fails
        fake_pdf = b"%PDF-1.4\n" + b"x" * 1000
        syncer._fetch_trusted_bddk = AsyncMock(
            return_value=SimpleNamespace(
                status_code=200,
                content=fake_pdf,
                headers={"content-type": "application/pdf"},
            )
        )

        result = await syncer.sync_document(doc_id="42628", force=True)

    assert result.success is False
    stored = await doc_store.get_document("42628")
    assert stored is not None
    assert stored.markdown_content == "ORIGINAL CONTENT"


@pytest.mark.asyncio
async def test_extraction_yields_serializes_and_finishes_before_cancellation(monkeypatch):
    import asyncio
    import threading

    store = AsyncMock()
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    calls = []

    def extract(content, ext):
        assert threading.current_thread() is not threading.main_thread()
        calls.append(content)
        loop.call_soon_threadsafe(started.set)
        assert release.wait(3), "event loop failed to release extraction worker"
        return "Extracted document content", "test"

    async with DocumentSyncer(store, http=MagicMock(), ocr_backends=[]) as syncer:
        monkeypatch.setattr(syncer, "_download_bddk", AsyncMock(return_value=(b"pdf", "download", ".pdf")))
        monkeypatch.setattr(syncer, "_extract", extract)
        first = asyncio.create_task(syncer.sync_document("1", force=True))
        second = None
        try:
            await asyncio.wait_for(started.wait(), 1)
            second = asyncio.create_task(syncer.sync_document("2", force=True))
            for _ in range(2):
                first.cancel()
                await asyncio.sleep(0.01)
                assert not first.done()
                assert len(calls) == 1
            store.store_document.assert_not_awaited()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await first
            if second is not None:
                assert (await second).success
        assert len(calls) == 2
        store.store_document.assert_awaited_once()
        assert store.store_document.await_args.args[0].document_id == "2"
