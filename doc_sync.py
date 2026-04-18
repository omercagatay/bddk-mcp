"""
Document sync engine for BDDK MCP Server.

Downloads BDDK decisions and mevzuat.gov.tr documents, extracts content
to markdown, and stores them in the PostgreSQL database.

Extraction pipeline (configured in ocr_backends.get_default_backends):
  1. LightOnOCR-2-1B (GPU) — primary, formula-aware
  2. PP-StructureV3 (GPU fallback)
  3. markitdown — CPU last resort, no formulas
  4. HTML parsing — mevzuat.gov.tr HTML fallback

Usage:
    python doc_sync.py sync [--force] [--doc-id DOC_ID] [--concurrency 5]
    python doc_sync.py stats
    python doc_sync.py import-cache
"""

import argparse
import asyncio
import io
import json
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel

from config import (
    BASE_DIR,
    HTTP_CONNECT_TIMEOUT,
    HTTP_POOL_TIMEOUT,
    OCR_MIN_CONTENT_LEN,
    REQUEST_TIMEOUT,
)
from doc_store import DocumentStore, StoredDocument
from ocr_backends import OCRBackend, get_default_backends, run_extraction_chain
from utils import MEVZUAT_TUR_MAP, fetch_with_retry

if TYPE_CHECKING:
    from vector_store import VectorStore

CACHE_FILE = BASE_DIR / ".cache.json"  # legacy path for CLI compat

logger = logging.getLogger(__name__)

_BDDK_DOC_URL = "https://www.bddk.org.tr/Mevzuat/DokumanGetir/{document_id}"


def _categorize_error(error: str) -> tuple[str, bool]:
    """Categorize a sync error and determine if retryable.

    Returns (category, retryable).
    """
    lower = error.lower()
    if "robots" in lower or "403" in lower:
        return "robots_txt", False
    if "timeout" in lower or "timed out" in lower:
        return "timeout", True
    if "extraction failed" in lower or "404" in lower or "error page" in lower:
        return "extraction", False
    if "all download" in lower or "no content" in lower:
        return "download", True
    if "connect" in lower or "connection" in lower:
        return "connection", True
    return "unknown", True


# ── Result models ────────────────────────────────────────────────────────────


class ExtractionResult(BaseModel):
    """Structured result from document extraction attempts."""

    content: str = ""
    method: str = "failed"
    error: str = ""
    retryable: bool = False


class SyncResult(BaseModel):
    document_id: str
    success: bool
    method: str = ""
    error: str = ""
    size_bytes: int = 0


class SyncReport(BaseModel):
    total: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[SyncResult] = []
    elapsed_seconds: float = 0.0


# ── Extraction backends ──────────────────────────────────────────────────────


def _decode_html(content: bytes) -> str:
    """Decode HTML content with encoding detection for Turkish text."""
    for encoding in ("utf-8", "iso-8859-9", "windows-1254"):
        try:
            decoded = content.decode(encoding)
            # Replacement chars anywhere in the body indicate either the
            # wrong encoding or a corrupt source. Either way, do not return
            # silently — try the next encoding.
            if "\ufffd" not in decoded:
                return decoded
        except (UnicodeDecodeError, LookupError):
            continue
    return content.decode("utf-8", errors="replace")


# Known patterns from mevzuat.gov.tr error/navigation pages
_ERROR_PAGE_PATTERNS = [
    "Mevzuat TürüKanunlar",
    "Mevzuat TuruKanunlar",
    "404 - Sayfa Bulunamadı",
    "404 - Sayfa Bulunamadi",
    "Sayfa Bulunamadı",
]


def _is_error_page(content: str) -> bool:
    """Detect 404 pages and navigation-only extractions from mevzuat.gov.tr."""
    import html

    # Decode HTML entities so patterns match raw HTML (e.g. &#x131; → ı)
    decoded = html.unescape(content)
    for pattern in _ERROR_PAGE_PATTERNS:
        if pattern in decoded:
            return True
    return False


def _extract_html_to_markdown(html: str) -> str:
    """Convert HTML content to simple markdown."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove script/style tags
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    # Basic conversion
    lines = []
    for elem in soup.find_all(["h1", "h2", "h3", "h4", "h5", "p", "li", "td", "th"]):
        text = elem.get_text(strip=True)
        if not text:
            continue
        tag = elem.name
        if tag == "h1":
            lines.append(f"# {text}")
        elif tag == "h2":
            lines.append(f"## {text}")
        elif tag == "h3":
            lines.append(f"### {text}")
        elif tag in ("h4", "h5"):
            lines.append(f"#### {text}")
        elif tag == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    return "\n\n".join(lines)


# ── Download helpers ─────────────────────────────────────────────────────────


def _mevzuat_pdf_url(mevzuat_no: str, tur: str = "7", tertip: str = "5") -> str | None:
    """Build mevzuat.gov.tr direct PDF URL."""
    segment = MEVZUAT_TUR_MAP.get(tur)
    if not segment:
        return None
    return f"https://www.mevzuat.gov.tr/MevzuatMetin/{segment}/{tur}.{tertip}.{mevzuat_no}.pdf"


# GeneratePdf API expects these exact mevzuatTur parameter values.
_GENERATE_PDF_TUR_NAME: dict[str, str] = {
    "1": "Kanun",
    "2": "KanunHukmundeKararname",
    "4": "CumhurbaskanligiKararnamesi",
    "5": "Tuzuk",
    "7": "Yonetmelik",
    "9": "Teblig",
    "11": "CumhurbaskanligiKararnamesi",
}


def _mevzuat_generate_pdf_url(mevzuat_no: str, tur: str = "7", tertip: str = "5") -> str | None:
    """Build mevzuat.gov.tr GeneratePdf API URL (server-side PDF generation)."""
    tur_name = _GENERATE_PDF_TUR_NAME.get(tur)
    if not tur_name:
        return None
    return f"https://www.mevzuat.gov.tr/File/GeneratePdf?mevzuatNo={mevzuat_no}&mevzuatTur={tur_name}&mevzuatTertip={tertip}"


def _mevzuat_doc_url(mevzuat_no: str, tur: str = "7", tertip: str = "5") -> str:
    """Build mevzuat.gov.tr Word (.doc) download URL."""
    segment = MEVZUAT_TUR_MAP.get(tur, "yonetmelik")
    return f"https://www.mevzuat.gov.tr/MevzuatMetin/{segment}/{tur}.{tertip}.{mevzuat_no}.doc"


def _parse_mevzuat_params(source_url: str) -> tuple[str, str, str]:
    """Extract mevzuat_no, tur, tertip from a mevzuat.gov.tr URL."""
    parsed = urlparse(source_url)
    params = parse_qs(parsed.query)

    mevzuat_no = params.get("MevzuatNo", [""])[0]
    tur = params.get("MevzuatTur", ["7"])[0]
    tertip = params.get("MevzuatTertip", ["5"])[0]

    if not mevzuat_no:
        kod = params.get("MevzuatKod", [""])[0]
        if kod:
            parts = kod.split(".")
            if len(parts) >= 3:
                tur = parts[0]
                tertip = parts[1]
                mevzuat_no = parts[-1]

    return mevzuat_no, tur, tertip


# ── DocumentSyncer ───────────────────────────────────────────────────────────


class DocumentSyncer:
    """Downloads and extracts BDDK/mevzuat documents into the DocumentStore."""

    def __init__(
        self,
        store: DocumentStore,
        request_timeout: float = REQUEST_TIMEOUT,
        ocr_backends: "list[OCRBackend] | None" = None,
        progress_callback: "Callable[[str, int, int], None] | None" = None,
        http: httpx.AsyncClient | None = None,
        vector_store: "VectorStore | None" = None,
    ) -> None:
        self._store = store
        self._ocr_backends = ocr_backends if ocr_backends is not None else get_default_backends()
        self._progress_callback = progress_callback
        self._vector_store = vector_store
        self._owns_http = http is None
        if http is not None:
            self._http = http
        else:
            self._http = httpx.AsyncClient(
                headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
                timeout=httpx.Timeout(
                    request_timeout,
                    connect=HTTP_CONNECT_TIMEOUT,
                    pool=HTTP_POOL_TIMEOUT,
                ),
                follow_redirects=True,
            )

    async def close(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> "DocumentSyncer":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    # ── Single document sync ─────────────────────────────────────────────

    async def sync_document(
        self,
        doc_id: str,
        title: str = "",
        category: str = "",
        source_url: str = "",
        decision_date: str = "",
        decision_number: str = "",
        force: bool = False,
    ) -> SyncResult:
        """Download and extract a single document."""

        # Skip if already in store and not forced
        if not force and await self._store.has_document(doc_id):
            return SyncResult(document_id=doc_id, success=True, method="cached")

        # If re-extracting with force=True and the PDF is already cached in DB,
        # skip re-downloading (bandwidth saving).
        cached_pdf: bytes | None = None
        if force and doc_id.startswith("mevzuat_"):
            cached_pdf = await self._store.get_pdf_bytes(doc_id)

        try:
            if cached_pdf:
                content, method, ext = cached_pdf, "cached_pdf", ".pdf"
            elif doc_id.startswith("mevzuat_"):
                content, method, ext = await self._download_mevzuat(doc_id, source_url)
            elif doc_id.isdigit():
                content, method, ext = await self._download_bddk(doc_id)
            else:
                return SyncResult(
                    document_id=doc_id,
                    success=False,
                    error=f"Unknown document ID format: {doc_id}",
                )
        except Exception as e:
            error_msg = str(e)
            cat, retryable = _categorize_error(error_msg)
            await self._store.record_sync_failure(doc_id, error_msg, cat, source_url, retryable)
            return SyncResult(
                document_id=doc_id,
                success=False,
                error=error_msg,
            )

        if not content:
            error_msg = "No content downloaded"
            cat, retryable = _categorize_error(error_msg)
            await self._store.record_sync_failure(doc_id, error_msg, cat, source_url, retryable)
            return SyncResult(
                document_id=doc_id,
                success=False,
                error=error_msg,
            )

        # Extract markdown
        markdown, extraction_method = self._extract(content, ext)
        if not markdown:
            error_msg = f"Extraction failed (method={extraction_method})"
            cat, retryable = _categorize_error(error_msg)
            await self._store.record_sync_failure(doc_id, error_msg, cat, source_url, retryable)
            # Preserve old content on failed force re-extract — losing it would
            # erase successful prior extractions when a new backend transiently fails.
            logger.warning("Extraction failed for %s; preserving old content (force=%s)", doc_id, force)
            return SyncResult(
                document_id=doc_id,
                success=False,
                error=error_msg,
            )

        # Store
        doc = StoredDocument(
            document_id=doc_id,
            title=title or doc_id,
            category=category,
            decision_date=decision_date,
            decision_number=decision_number,
            source_url=source_url,
            pdf_bytes=content if ext == ".pdf" else None,
            markdown_content=markdown,
            extraction_method=extraction_method,
            file_size=len(content),
        )
        await self._store.store_document(doc)
        await self._store.clear_sync_failure(doc_id)

        if self._vector_store is not None:
            try:
                await self._vector_store.add_document(
                    doc_id=doc_id,
                    title=title or doc_id,
                    content=markdown,
                    category=category,
                    decision_date=decision_date,
                    decision_number=decision_number,
                    source_url=source_url,
                )
            except Exception as e:
                logger.warning(
                    "Re-index failed for %s after successful sync: %s. "
                    "documents table is fresh; chunks are stale — retry with force=True.",
                    doc_id,
                    e,
                )
                return SyncResult(
                    document_id=doc_id,
                    success=False,
                    method=f"{method}+{extraction_method}",
                    error=f"reindex_failed: {e}",
                    size_bytes=len(content),
                )

        return SyncResult(
            document_id=doc_id,
            success=True,
            method=f"{method}+{extraction_method}",
            size_bytes=len(content),
        )

    # ── Download methods ─────────────────────────────────────────────────

    async def _download_bddk(self, doc_id: str) -> tuple[bytes, str, str]:
        """Download from BDDK DokumanGetir endpoint."""
        url = _BDDK_DOC_URL.format(document_id=doc_id)
        resp = await fetch_with_retry(self._http, url)
        content_type = resp.headers.get("content-type", "").lower()
        ext = ".pdf" if "pdf" in content_type else ".html"
        return resp.content, "bddk_direct", ext

    async def _download_mevzuat(self, doc_id: str, source_url: str = "") -> tuple[bytes, str, str]:
        """
        Download from mevzuat.gov.tr with 5-layer fallback.

        Order optimized for reliability (lightest/fastest first):
        1. Static .htm page — smallest, most reliable
        2. PDF direct download (static file)
        3. Main page → iframe/div content extraction
        3b. GeneratePdf API — requires session cookies from step 3
        4. Word (.doc) download — largest, slowest

        When source_url is not provided and the default tur fails,
        automatically tries all known tur values (tur auto-detection).

        Each layer has its own short timeout to avoid blocking others.
        """
        mevzuat_no = doc_id.removeprefix("mevzuat_")
        tur, tertip = "7", "5"

        if source_url:
            no, t, te = _parse_mevzuat_params(source_url)
            if no:
                mevzuat_no = no
            if t:
                tur = t
            if te:
                tertip = te

        # Build list of tur values to try.
        # Always try the source/default tur first, then fall back to all others.
        # Even when source_url provides tur, it may be stale or wrong (404).
        tur_candidates = [tur] + [t for t in MEVZUAT_TUR_MAP if t != tur]

        for candidate_tur in tur_candidates:
            segment = MEVZUAT_TUR_MAP.get(candidate_tur, "yonetmelik")
            base = f"{candidate_tur}.{tertip}.{mevzuat_no}"

            # Per-layer timeout: short enough so one slow layer doesn't kill the rest
            layer_timeout = httpx.Timeout(30.0, connect=10.0)

            # Layer 1: Main page visit — establishes session cookies for GeneratePdf
            main_url = f"https://www.mevzuat.gov.tr/mevzuat?MevzuatNo={mevzuat_no}&MevzuatTur={candidate_tur}&MevzuatTertip={tertip}"
            main_page_visited = False
            main_page_html = ""
            try:
                resp = await self._http.get(main_url, timeout=layer_timeout)
                if resp.status_code == 200:
                    main_page_visited = True
                    main_page_html = resp.text
            except Exception as e:
                logger.debug("mevzuat %s: main page visit failed (tur=%s): %s", doc_id, candidate_tur, e)

            # Layer 2: GeneratePdf API (preferred — server-rendered PDF with all formulas as images)
            if main_page_visited:
                try:
                    gen_pdf_url = _mevzuat_generate_pdf_url(mevzuat_no, candidate_tur, tertip)
                    if gen_pdf_url:
                        resp = await self._http.get(gen_pdf_url, timeout=layer_timeout)
                        if resp.status_code == 200 and len(resp.content) > 500 and resp.content[:5] == b"%PDF-":
                            logger.info("mevzuat %s: downloaded via GeneratePdf (tur=%s)", doc_id, candidate_tur)
                            return resp.content, "mevzuat_generate_pdf", ".pdf"
                except Exception as e:
                    logger.debug("mevzuat %s: GeneratePdf failed (tur=%s): %s", doc_id, candidate_tur, e)

            # Layer 3: Direct static .pdf
            try:
                pdf_url = _mevzuat_pdf_url(mevzuat_no, candidate_tur, tertip)
                if pdf_url:
                    resp = await self._http.get(pdf_url, timeout=layer_timeout)
                    if resp.status_code == 200 and len(resp.content) > 500 and resp.content[:5] == b"%PDF-":
                        logger.info("mevzuat %s: downloaded via .pdf (tur=%s)", doc_id, candidate_tur)
                        return resp.content, "mevzuat_pdf", ".pdf"
            except Exception as e:
                logger.debug("mevzuat %s: .pdf failed (tur=%s): %s", doc_id, candidate_tur, e)

            # Layer 4: .htm fallback — formulas may be lost (rendered as <img>)
            try:
                htm_url = f"https://www.mevzuat.gov.tr/mevzuatmetin/{segment}/{base}.htm"
                resp = await self._http.get(htm_url, timeout=layer_timeout)
                if resp.status_code == 200 and len(resp.content) > 200 and not _is_error_page(resp.text):
                    logger.warning(
                        "mevzuat %s: falling back to .htm (tur=%s) — formulas may be lost",
                        doc_id,
                        candidate_tur,
                    )
                    return resp.content, "mevzuat_htm", ".html"
            except Exception as e:
                logger.debug("mevzuat %s: .htm failed (tur=%s): %s", doc_id, candidate_tur, e)

            # Layer 5: iframe/div from already-fetched main page
            if main_page_visited and main_page_html:
                try:
                    soup = BeautifulSoup(main_page_html, "html.parser")
                    iframe = soup.find("iframe", src=True)
                    if iframe:
                        iframe_url = iframe["src"]
                        if not iframe_url.startswith("http"):
                            iframe_url = f"https://www.mevzuat.gov.tr{iframe_url}"
                        iframe_resp = await self._http.get(iframe_url, timeout=layer_timeout)
                        if iframe_resp.status_code == 200 and len(iframe_resp.content) > 200:
                            logger.warning("mevzuat %s: falling back to iframe (tur=%s)", doc_id, candidate_tur)
                            return iframe_resp.content, "mevzuat_iframe", ".html"
                    div = soup.find("div", id="divMevzuatMetni")
                    if div and len(div.get_text(strip=True)) > 100:
                        logger.warning("mevzuat %s: falling back to main page div (tur=%s)", doc_id, candidate_tur)
                        return str(div).encode("utf-8"), "mevzuat_div", ".html"
                except Exception as e:
                    logger.debug("mevzuat %s: iframe/div parse failed (tur=%s): %s", doc_id, candidate_tur, e)

            # Layer 6: Word (.doc) — heaviest, slowest (only try for the first/default tur
            # to avoid excessive requests during auto-detection)
            if candidate_tur == tur:
                try:
                    doc_url = _mevzuat_doc_url(mevzuat_no, candidate_tur, tertip)
                    resp = await self._http.get(doc_url, timeout=httpx.Timeout(90.0, connect=15.0))
                    if (
                        resp.status_code == 200
                        and len(resp.content) > 100
                        and resp.content[:4] in (b"\xd0\xcf\x11\xe0", b"PK\x03\x04")
                    ):
                        logger.info("mevzuat %s: downloaded via .doc (tur=%s)", doc_id, candidate_tur)
                        return resp.content, "mevzuat_doc", ".doc"
                except Exception as e:
                    logger.debug("mevzuat %s: .doc failed (tur=%s): %s", doc_id, candidate_tur, e)

            if candidate_tur != tur_candidates[-1]:
                logger.debug("mevzuat %s: tur=%s failed, trying next candidate", doc_id, candidate_tur)

        raise RuntimeError(f"All download methods failed for {doc_id} (tried tur values: {tur_candidates})")

    # ── Extraction ───────────────────────────────────────────────────────

    def _extract(self, content: bytes, ext: str) -> tuple[str, str]:
        """Extract markdown from downloaded content. Returns (markdown, method).

        Uses a structured pipeline and logs detailed failure reasons.
        """
        extraction = self._extract_structured(content, ext)
        if extraction.error:
            logger.warning(
                "Extraction issue: %s (method=%s, retryable=%s)",
                extraction.error,
                extraction.method,
                extraction.retryable,
            )
        return extraction.content, extraction.method

    def _extract_structured(self, content: bytes, ext: str) -> ExtractionResult:
        """Extract markdown via backend chain for PDFs, or HTML/markitdown path for others."""
        if ext == ".pdf":
            attempt = run_extraction_chain(content, self._ocr_backends, min_len=OCR_MIN_CONTENT_LEN)
            if attempt.backend != "failed":
                return ExtractionResult(content=attempt.content, method=attempt.backend)
            return ExtractionResult(method="failed", error=attempt.error, retryable=False)

        if ext in (".html", ".htm"):
            errors: list[str] = []
            html_str = _decode_html(content)
            result = _extract_html_to_markdown(html_str)
            if result and not _is_error_page(result):
                return ExtractionResult(content=result, method="html_parser")
            if result and _is_error_page(result):
                errors.append("html_parser: extracted content is a 404/navigation page")
            else:
                errors.append("html_parser: no meaningful content extracted")

            try:
                from markitdown import MarkItDown

                md = MarkItDown()
                html_result = md.convert_stream(io.BytesIO(content), file_extension=".html").text_content.strip()
                if html_result and not _is_error_page(html_result):
                    return ExtractionResult(content=html_result, method="markitdown")
                errors.append("markitdown: HTML fallback failed or error page")
            except (ValueError, OSError, UnicodeDecodeError) as e:
                errors.append(f"markitdown: {e}")

            retryable = len(content) < 200 or any("404" in e or "navigation" in e for e in errors)
            return ExtractionResult(method="failed", error="; ".join(errors), retryable=retryable)

        if ext in (".doc", ".docx"):
            try:
                from markitdown import MarkItDown

                md = MarkItDown()
                result = md.convert_stream(io.BytesIO(content), file_extension=ext).text_content.strip()
                if result and len(result) >= OCR_MIN_CONTENT_LEN:
                    return ExtractionResult(content=result, method="markitdown")
                return ExtractionResult(method="failed", error="markitdown output too short", retryable=True)
            except (ValueError, OSError, UnicodeDecodeError) as e:
                return ExtractionResult(method="failed", error=f"markitdown: {e}", retryable=True)

        return ExtractionResult(method="failed", error=f"Unsupported extension: {ext}", retryable=False)

    # ── Batch sync ───────────────────────────────────────────────────────

    async def sync_all(
        self,
        documents: list[dict],
        concurrency: int = 5,
        force: bool = False,
    ) -> SyncReport:
        """Sync a batch of documents with concurrency control and progress reporting."""
        start = time.time()
        report = SyncReport(total=len(documents))
        completed = 0

        semaphore = asyncio.Semaphore(concurrency)

        async def _sync_one(doc_info: dict) -> SyncResult:
            nonlocal completed
            async with semaphore:
                result = await self.sync_document(
                    doc_id=doc_info.get("document_id", ""),
                    title=doc_info.get("title", ""),
                    category=doc_info.get("category", ""),
                    source_url=doc_info.get("source_url", ""),
                    decision_date=doc_info.get("decision_date", ""),
                    decision_number=doc_info.get("decision_number", ""),
                    force=force,
                )
                completed += 1
                if self._progress_callback:
                    self._progress_callback(doc_info.get("document_id", ""), completed, len(documents))
                elif completed % 50 == 0 or completed == len(documents):
                    elapsed = time.time() - start
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (len(documents) - completed) / rate if rate > 0 else 0
                    logger.info(
                        "Sync progress: %d/%d (%.0f%%) — %.1f docs/s, ETA %.0fs",
                        completed,
                        len(documents),
                        completed / len(documents) * 100,
                        rate,
                        eta,
                    )
                return result

        tasks = [_sync_one(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                report.failed += 1
                report.errors.append(SyncResult(document_id="unknown", success=False, error=str(r)))
            elif r.success:
                if r.method == "cached":
                    report.skipped += 1
                else:
                    report.downloaded += 1
            else:
                report.failed += 1
                report.errors.append(r)

        report.elapsed_seconds = round(time.time() - start, 2)
        return report

    # ── Cache import helper ──────────────────────────────────────────────

    async def import_and_sync_from_cache(self, force: bool = False, concurrency: int = 5) -> SyncReport:
        """Load documents from .cache.json and sync them all."""
        if not CACHE_FILE.exists():
            logger.error("No cache file found at %s", CACHE_FILE)
            return SyncReport()

        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        items = data.get("items", [])
        if not items:
            logger.warning("Cache file is empty")
            return SyncReport()

        logger.info("Found %d items in cache file", len(items))

        # First import metadata
        await self._store.import_from_cache(items)

        # Then sync content
        return await self.sync_all(items, concurrency=concurrency, force=force)


# ── CLI ──────────────────────────────────────────────────────────────────────


async def _create_pool_and_store(dsn: str | None) -> tuple:
    """Create asyncpg pool, DocumentStore, and VectorStore for CLI usage.

    Returns (pool, store, vector_store). `vector_store` may be None if
    initialization fails — CLI sync then skips re-index.
    """
    import asyncpg as _asyncpg

    from config import require_database_url
    from vector_store import VectorStore

    pool = await _asyncpg.create_pool(dsn or require_database_url(), min_size=1, max_size=5)
    store = DocumentStore(pool)
    await store.initialize()

    vs: VectorStore | None
    try:
        vs = VectorStore(pool)
        await vs.initialize()
    except Exception as e:
        logger.warning("VectorStore init failed (%s) — CLI sync will skip re-index", e)
        vs = None

    return pool, store, vs


async def _cli_sync(args: argparse.Namespace) -> None:
    """CLI: sync documents."""
    pool, store, vs = await _create_pool_and_store(args.db)
    try:
        async with DocumentSyncer(store, vector_store=vs) as syncer:
            if args.doc_id:
                # Look up metadata from decision_cache for source_url/title
                row = await pool.fetchrow(
                    "SELECT source_url, title, category, decision_date, decision_number"
                    " FROM decision_cache WHERE document_id = $1",
                    args.doc_id,
                )
                result = await syncer.sync_document(
                    doc_id=args.doc_id,
                    source_url=row["source_url"] if row else "",
                    title=row["title"] if row else "",
                    category=row["category"] if row else "",
                    decision_date=row["decision_date"] if row else "",
                    decision_number=row["decision_number"] if row else "",
                    force=args.force,
                )
                status = "OK" if result.success else "FAIL"
                print(f"[{status}] {result.document_id}: {result.method or result.error}")
            else:
                report = await syncer.import_and_sync_from_cache(
                    force=args.force,
                    concurrency=args.concurrency,
                )
                print("\nSync Report:")
                print(f"  Total:      {report.total}")
                print(f"  Downloaded: {report.downloaded}")
                print(f"  Skipped:    {report.skipped}")
                print(f"  Failed:     {report.failed}")
                print(f"  Time:       {report.elapsed_seconds}s")
                if report.errors:
                    print("\nErrors:")
                    for e in report.errors[:20]:
                        print(f"  [{e.document_id}] {e.error}")
    finally:
        await pool.close()


async def _cli_stats(args: argparse.Namespace) -> None:
    """CLI: show store stats."""
    pool, store, _vs = await _create_pool_and_store(args.db)
    try:
        st = await store.stats()
        print(f"Documents: {st.total_documents}")
        print(f"Size: {st.total_size_mb} MB")
        print(f"Need refresh: {st.documents_needing_refresh}")
        if st.categories:
            print("\nCategories:")
            for cat, count in st.categories.items():
                print(f"  {cat}: {count}")
        if st.extraction_methods:
            print("\nExtraction methods:")
            for m, count in st.extraction_methods.items():
                print(f"  {m}: {count}")
    finally:
        await pool.close()


async def _cli_import(args: argparse.Namespace) -> None:
    """CLI: import metadata from cache without downloading content."""
    pool, store, _vs = await _create_pool_and_store(args.db)
    try:
        if not CACHE_FILE.exists():
            print(f"No cache file at {CACHE_FILE}")
            return

        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        items = data.get("items", [])
        imported = await store.import_from_cache(items)
        print(f"Imported {imported} new entries from cache ({len(items)} total in cache)")
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="BDDK Document Sync")
    parser.add_argument("--db", help="PostgreSQL DSN (e.g. postgresql://user:pass@host/db)", default=None)
    sub = parser.add_subparsers(dest="command")

    # sync
    sync_p = sub.add_parser("sync", help="Download and extract documents")
    sync_p.add_argument("--force", action="store_true", help="Re-download all")
    sync_p.add_argument("--doc-id", help="Sync a single document by ID")
    sync_p.add_argument("--concurrency", type=int, default=5)

    # stats
    sub.add_parser("stats", help="Show document store statistics")

    # import-cache
    sub.add_parser("import-cache", help="Import metadata from .cache.json (legacy)")

    args = parser.parse_args()

    if args.command == "sync":
        asyncio.run(_cli_sync(args))
    elif args.command == "stats":
        asyncio.run(_cli_stats(args))
    elif args.command == "import-cache":
        asyncio.run(_cli_import(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
