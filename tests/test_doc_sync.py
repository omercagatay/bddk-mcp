"""Tests for doc_sync.py — document download and extraction pipeline."""

from unittest.mock import AsyncMock

import httpx
import pytest

from doc_store import StoredDocument
from doc_sync import (
    DocumentSyncer,
    _decode_html,
    _extract_html_to_markdown,
    _mevzuat_doc_url,
    _mevzuat_generate_pdf_url,
    _mevzuat_pdf_url,
    _parse_mevzuat_params,
)
from ocr_backends import MarkitdownBackend
from tests.conftest import make_http_response
from utils import fetch_with_retry

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


class TestSanitizeForStorage:
    def test_strips_nul_bytes(self):
        from doc_sync import _sanitize_for_storage

        assert _sanitize_for_storage("hello\x00world") == "helloworld"

    def test_preserves_clean_text(self):
        from doc_sync import _sanitize_for_storage

        clean = "Madde 1 — Bankaların risk yönetimi."
        assert _sanitize_for_storage(clean) is clean

    def test_preserves_other_control_chars(self):
        from doc_sync import _sanitize_for_storage

        assert _sanitize_for_storage("line1\nline2\tcol") == "line1\nline2\tcol"

    def test_empty_is_passed_through(self):
        from doc_sync import _sanitize_for_storage

        assert _sanitize_for_storage("") == ""

    def test_strips_form_feeds(self):
        # SYSTEMIC-3. Markitdown leaves PDF page-break bytes (0x0C) in output.
        # Visual noise with no semantic value — strip in the same pass that
        # removes storage-unsafe NULs.
        from doc_sync import _sanitize_for_storage

        assert _sanitize_for_storage("page1\x0cpage2\x0cpage3") == "page1page2page3"

    def test_replaces_garbled_turkish_capital_i(self):
        # SYSTEMIC-1. Markitdown's PDF path decodes Turkish capital İ (U+0130)
        # as Đ (U+0110) on BDDK legacy PDFs whose embedded font lacks a
        # ToUnicode CMap. Blanket Đ→İ is safe because every Đ observed
        # across 43 affected docs / 235 occurrences is a garbled İ, and Đ
        # (Croatian/Vietnamese) never legitimately appears in Turkish
        # regulatory text — verified by auditing every non-ASCII Turkish
        # character in the document store.
        from doc_sync import _sanitize_for_storage

        assert _sanitize_for_storage("Tevfik BĐLGĐN") == "Tevfik BİLGİN"
        assert _sanitize_for_storage("Đhraççı bankanın") == "İhraççı bankanın"

    def test_all_three_artifacts_in_one_pass(self):
        # Combined fix: one pass handles NUL (storage-unsafe) + form-feed
        # (SYSTEMIC-3) + Đ-garble (SYSTEMIC-1). A markitdown output with
        # all three defects gets cleaned in a single sweep.
        from doc_sync import _sanitize_for_storage

        assert _sanitize_for_storage("BĐLGĐN\x0cĐhraççı\x00") == "BİLGİNİhraççı"


class TestFetchWithRetry:
    @pytest.mark.asyncio
    async def test_success(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.get = AsyncMock(return_value=make_http_response("OK"))

        resp = await fetch_with_retry(http, "https://example.com")
        assert resp.text == "OK"

    @pytest.mark.asyncio
    async def test_retry_then_succeed(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        ok_resp = make_http_response("OK")
        http.get = AsyncMock(side_effect=[httpx.TransportError("fail"), ok_resp])

        resp = await fetch_with_retry(http, "https://example.com")
        assert resp.text == "OK"

    @pytest.mark.asyncio
    async def test_all_retries_fail(self):
        http = AsyncMock(spec=httpx.AsyncClient)
        http.get = AsyncMock(side_effect=httpx.TransportError("fail"))

        with pytest.raises(httpx.TransportError):
            await fetch_with_retry(http, "https://example.com")

    @pytest.mark.asyncio
    async def test_no_retry_on_404(self):
        """4xx client errors must not be retried."""
        http = AsyncMock(spec=httpx.AsyncClient)
        http.get = AsyncMock(return_value=make_http_response("Not Found", status_code=404))

        with pytest.raises(httpx.HTTPStatusError):
            await fetch_with_retry(http, "https://example.com")

        assert http.get.call_count == 1  # no retry

    @pytest.mark.asyncio
    async def test_retry_on_500(self):
        """5xx server errors must be retried."""
        http = AsyncMock(spec=httpx.AsyncClient)
        ok_resp = make_http_response("OK", status_code=200)
        http.get = AsyncMock(side_effect=[make_http_response("Server Error", status_code=500), ok_resp])

        resp = await fetch_with_retry(http, "https://example.com")
        assert resp.status_code == 200
        assert http.get.call_count == 2


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
        async with DocumentSyncer(store, ocr_backends=[MarkitdownBackend()]) as syncer:
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
        async with DocumentSyncer(store, ocr_backends=[MarkitdownBackend()]) as syncer:
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

        async with DocumentSyncer(store, ocr_backends=[MarkitdownBackend()]) as syncer:
            html = "<html><body><h1>New</h1><p>New content</p></body></html>"
            syncer._http = AsyncMock(spec=httpx.AsyncClient)
            syncer._http.get = AsyncMock(return_value=make_http_response(text=html, content_type="text/html"))

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

        async def fake_get(url, timeout=None):
            call_log.append(url)
            if "mevzuat?MevzuatNo=42628" in url and "MevzuatTur=7" in url:
                return make_http_response(content=main_html, content_type="text/html")
            if "GeneratePdf" in url and "mevzuatNo=42628" in url:
                return make_http_response(content=pdf_bytes, content_type="application/pdf")
            # Anything else: 404 so test fails if wrong layer is tried
            return make_http_response(status_code=404)

        syncer._http = AsyncMock(spec=_httpx.AsyncClient)
        syncer._http.get = AsyncMock(side_effect=fake_get)

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

        async def fake_get(url, timeout=None):
            call_log.append(url)
            if "mevzuat?MevzuatNo=42628" in url:
                return make_http_response(text=main_html, content_type="text/html")
            if "IframeDetay" in url:
                return make_http_response(text=iframe_body, content_type="text/html")
            # Anything else must not be reached before iframe success.
            return make_http_response(status_code=404)

        syncer._http = AsyncMock(spec=_httpx.AsyncClient)
        syncer._http.get = AsyncMock(side_effect=fake_get)

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
        syncer._http = AsyncMock(spec=httpx.AsyncClient)
        fake_pdf = b"%PDF-1.4\n" + b"x" * 1000
        syncer._http.get = AsyncMock(return_value=make_http_response(content=fake_pdf, content_type="application/pdf"))

        result = await syncer.sync_document(doc_id="42628", force=True)

    assert result.success is False
    stored = await doc_store.get_document("42628")
    assert stored is not None
    assert stored.markdown_content == "ORIGINAL CONTENT"
