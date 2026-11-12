"""
Document sync engine for BDDK MCP Server.

Downloads BDDK decisions and mevzuat.gov.tr documents, extracts content
to markdown, and stores them in the corpus file.

Extraction pipeline (the PDF chain is built by ocr.base.get_default_backends):
  1. LightOnOCR-2-1B (GPU) — formula-aware
  2. pdftotext (poppler, CPU) — layout text, no formulas or tables
  3. markitdown — CPU last resort, no formulas
  4. HTML parsing — mevzuat.gov.tr iframe HTML (first on hosts where only the
     degraded backends 2 and 3 are available, see BDDK_PREFER_HTML_FOR_MEVZUAT)
The Chandra2 backend (ocr/chandra.py) exists, but no sync path constructs it.

Run it through ``bddk-mcp sync``.
"""

import asyncio
import codecs
import html
import io
import logging
import re
import stat
import time
import zipfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlparse

import anyio
import httpx
from bs4 import BeautifulSoup
from defusedxml import ElementTree
from defusedxml.common import DefusedXmlException
from markitdown import MarkItDownException
from pydantic import BaseModel

from bddk_mcp.core.config import (
    HTTP_CONNECT_TIMEOUT,
    HTTP_POOL_TIMEOUT,
    MAX_RETRIES,
    OCR_MIN_CONTENT_LEN,
    PREFER_HTML_FOR_MEVZUAT,
    REQUEST_TIMEOUT,
)
from bddk_mcp.core.outbound_http import (
    BDDK_HTTPS_HOSTS,
    MEVZUAT_HTTPS_HOSTS,
    OutboundHttpPolicyError,
    assert_public_https_resolution,
    normalize_approved_https_url,
)
from bddk_mcp.ingest.client import MEVZUAT_TUR_MAP
from bddk_mcp.ocr.base import OCRBackend, get_default_backends, run_extraction_chain
from bddk_mcp.quality.markdown_quality import prepare_markdown_for_storage
from bddk_mcp.store.doc_store import DocumentStore, StoredDocument

if TYPE_CHECKING:
    from bddk_mcp.store.vector_store import VectorStore

logger = logging.getLogger(__name__)

_BDDK_DOC_URL = "https://www.bddk.org.tr/Mevzuat/DokumanGetir/{document_id}"

# Untrusted upstream HTML may contain iframe and annex links.  These limits are
# deliberately code-owned safety invariants: a deployment setting must not be
# able to turn an accidental archive or response into unbounded memory/CPU use.
_MAX_UPSTREAM_REDIRECTS = 3
_MAX_IFRAME_BYTES = 8 * 1024 * 1024
_MAX_MAIN_PAGE_BYTES = 8 * 1024 * 1024
_MAX_HTML_DOWNLOAD_BYTES = 16 * 1024 * 1024
_MAX_ANNEX_DOWNLOAD_BYTES = 32 * 1024 * 1024
_MAX_DOC_DOWNLOAD_BYTES = 64 * 1024 * 1024
_MAX_PDF_DOWNLOAD_BYTES = 128 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 128
_MAX_ARCHIVE_MEMBER_BYTES = 16 * 1024 * 1024
_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_MAX_ARCHIVE_EXPANSION_RATIO = 100
_MAX_DOCX_MEMBERS = 256
_MAX_DOCX_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
_MAX_DOCX_XML_BYTES = 16 * 1024 * 1024
_MAX_BDDK_ERROR_BODY_BYTES = 64 * 1024
_MAX_BDDK_RETRY_ATTEMPTS = 4
_MAX_BDDK_RETRY_DELAY_SECONDS = 4
_ZIP_LOCAL_FILE_HEADER = b"PK\x03\x04"
_OLE_COMPOUND_FILE_HEADER = b"\xd0\xcf\x11\xe0"
_APPROVED_PDF_MEDIA_TYPES = frozenset(
    {
        "application/pdf",
        "application/octet-stream",
        "application/download",
        "application/force-download",
        "binary/octet-stream",
    }
)
_APPROVED_HTML_MEDIA_TYPES = frozenset({"text/html", "application/xhtml+xml"})
_SOFT_ERROR_PAGE_MARKERS = (
    "access denied",
    "internal server error",
    "service unavailable",
    "temporarily unavailable",
    "bir hata oluştu",
    "bir hata olustu",
    "erişim engellendi",
    "erisim engellendi",
    "geçici olarak hizmet verememektedir",
    "gecici olarak hizmet verememektedir",
)


class UnsafeUpstreamResourceError(RuntimeError):
    """A privacy-safe rejection of an unsafe URL, redirect, or payload."""


class LegacyDocNotSupportedError(RuntimeError):
    """A legacy OLE ``.doc`` download; the extraction path has no converter for it."""

    def __init__(self) -> None:
        super().__init__("legacy .doc is not supported (OLE Word file)")


@dataclass(frozen=True, slots=True)
class _BoundedHttpResponse:
    status_code: int
    content: bytes
    headers: Mapping[str, str]


def _normalize_approved_https_url(
    candidate: str,
    *,
    base_url: str,
    allowed_hosts: frozenset[str],
    boundary_name: str,
) -> str:
    try:
        return normalize_approved_https_url(
            candidate, base_url=base_url, allowed_hosts=allowed_hosts, boundary_name=boundary_name
        )
    except OutboundHttpPolicyError as error:
        raise UnsafeUpstreamResourceError(str(error)) from None


def _normalize_mevzuat_url(candidate: str, *, base_url: str = "https://www.mevzuat.gov.tr/") -> str:
    """Return one canonical, exact-host HTTPS mevzuat URL."""

    return _normalize_approved_https_url(
        candidate,
        base_url=base_url,
        allowed_hosts=MEVZUAT_HTTPS_HOSTS,
        boundary_name="mevzuat",
    )


def _normalize_bddk_url(candidate: str, *, base_url: str = "https://www.bddk.org.tr/") -> str:
    """Return one canonical, exact-host HTTPS BDDK URL."""

    return _normalize_approved_https_url(
        candidate,
        base_url=base_url,
        allowed_hosts=BDDK_HTTPS_HOSTS,
        boundary_name="BDDK",
    )


def _validate_zip_metadata(
    archive: zipfile.ZipFile,
    *,
    max_members: int,
    max_member_bytes: int,
    max_total_bytes: int,
    max_expansion_ratio: int,
) -> list[zipfile.ZipInfo]:
    """Validate archive metadata before reading a single member body."""

    members = archive.infolist()
    if len(members) > max_members:
        raise UnsafeUpstreamResourceError("Archive contains too many members.")

    total = 0
    for member in members:
        if member.flag_bits & 0x1:
            raise UnsafeUpstreamResourceError("Encrypted archive members are not supported.")
        if member.file_size < 0 or member.compress_size < 0 or member.file_size > max_member_bytes:
            raise UnsafeUpstreamResourceError("Archive member exceeds the extraction limit.")
        total += member.file_size
        if total > max_total_bytes:
            raise UnsafeUpstreamResourceError("Archive exceeds the total extraction limit.")
        if member.file_size:
            if member.compress_size == 0 or member.file_size > member.compress_size * max_expansion_ratio:
                raise UnsafeUpstreamResourceError("Archive expansion ratio exceeds the extraction limit.")
    return members


def _validate_office_archive_for_markitdown(office_bytes: bytes) -> None:
    """Validate a ZIP-based Office document before third-party parsing.

    MarkItDown and its transitive Office parsers receive only archives with a
    bounded compressed body, member count, member size, aggregate expanded
    size, and expansion ratio.  Every XML relationship/body part is parsed by
    defusedxml first so entity declarations cannot be deferred to a downstream
    parser.  No archive member is ever extracted to the filesystem here.

    Legacy OLE ``.doc`` files are not ZIP containers and therefore cannot use
    these structural checks.  They remain download-size bounded, but complete
    CPU/time isolation requires moving the legacy converter to a constrained
    worker process.
    """

    if len(office_bytes) > _MAX_DOC_DOWNLOAD_BYTES:
        raise UnsafeUpstreamResourceError("Office document exceeds the processing limit.")
    if not office_bytes.startswith(_ZIP_LOCAL_FILE_HEADER):
        raise UnsafeUpstreamResourceError("Office archive signature is invalid.")

    try:
        with zipfile.ZipFile(io.BytesIO(office_bytes)) as archive:
            members = _validate_zip_metadata(
                archive,
                max_members=_MAX_DOCX_MEMBERS,
                max_member_bytes=_MAX_ARCHIVE_MEMBER_BYTES,
                max_total_bytes=_MAX_DOCX_UNCOMPRESSED_BYTES,
                max_expansion_ratio=_MAX_ARCHIVE_EXPANSION_RATIO,
            )
            seen_names: set[str] = set()
            document_body_found = False
            for member in members:
                normalized_name = member.filename.replace("\\", "/")
                path_parts = normalized_name.split("/")
                unix_mode = (member.external_attr >> 16) & 0o170000
                if (
                    not normalized_name
                    or normalized_name.startswith("/")
                    or re.match(r"^[A-Za-z]:", normalized_name)
                    or any(ord(character) < 0x20 for character in normalized_name)
                    or any(part == ".." for part in path_parts)
                    or unix_mode == stat.S_IFLNK
                    or normalized_name in seen_names
                ):
                    raise UnsafeUpstreamResourceError("Office archive member layout is unsafe.")
                seen_names.add(normalized_name)
                if normalized_name == "word/document.xml" and not member.is_dir():
                    document_body_found = True

                lower_name = normalized_name.casefold()
                if member.is_dir() or not (lower_name.endswith(".xml") or lower_name.endswith(".rels")):
                    continue
                xml_bytes = archive.read(member)
                if len(xml_bytes) > _MAX_DOCX_XML_BYTES:
                    raise UnsafeUpstreamResourceError("Office XML exceeds the processing limit.")
                ElementTree.fromstring(xml_bytes)

            if not document_body_found:
                raise UnsafeUpstreamResourceError("Office archive has no document body.")
    except UnsafeUpstreamResourceError:
        raise
    except (
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        ElementTree.ParseError,
        DefusedXmlException,
        EOFError,
        NotImplementedError,
        OSError,
        RuntimeError,
    ):
        raise UnsafeUpstreamResourceError("Office archive is invalid or unsupported.") from None


def _categorize_error(error: str, doc_id: str = "") -> tuple[str, bool]:
    """Categorize a sync error and determine if retryable.

    The document id (and its number) is removed first and HTTP status codes
    match only as whole tokens, so an id such as ``mevzuat_14036`` is not read
    as a 403. Returns (category, retryable).
    """
    lower = error.lower()
    for identifier in (doc_id, doc_id.removeprefix("mevzuat_")):
        if identifier:
            lower = re.sub(rf"\b{re.escape(identifier.lower())}\b", " ", lower)
    if "robots" in lower or re.search(r"\b403\b", lower):
        return "robots_txt", False
    if "timeout" in lower or "timed out" in lower:
        return "timeout", True
    if "extraction failed" in lower or re.search(r"\b404\b", lower) or "error page" in lower:
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
    reindexed: int = 0
    failed: int = 0
    errors: list[SyncResult] = []
    elapsed_seconds: float = 0.0


# ── Extraction backends ──────────────────────────────────────────────────────


# Windows-1252 C1-control → printable punctuation remap.
# Mevzuat HTML occasionally carries literal U+0080..U+009F characters where
# Windows-1252 punctuation belongs (en-dashes, smart quotes, ellipsis). This
# is Word-export residue: the exporter stored byte 0x96 as Unicode U+0096
# rather than mapping it to U+2013, so the resulting tofu boxes survive
# every encoding round-trip. None of these code points legitimately appear
# in Turkish regulatory text, so post-decode translation is unambiguous.
# See error_reports.md SYSTEMIC-8 for the affected-doc roster.
_WIN1252_C1_MAP = str.maketrans(
    {
        "": "€",
        "": "‚",
        "": "ƒ",
        "": "„",
        "": "…",
        "": "†",
        "": "‡",
        "": "ˆ",
        "": "‰",
        "": "Š",
        "": "‹",
        "": "Œ",
        "": "Ž",
        "": "‘",
        "": "’",
        "": "“",
        "": "”",
        "": "•",
        "": "–",
        "": "—",
        "": "˜",
        "": "™",
        "": "š",
        "": "›",
        "": "œ",
        "": "ž",
        "": "Ÿ",
    }
)


_META_CHARSET_RE = re.compile(rb"""<meta[^>]*?charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]{1,40})""", re.IGNORECASE)
_HTTP_CHARSET_RE = re.compile(r"""charset\s*=\s*["']?\s*([A-Za-z0-9_.:-]{1,40})""", re.IGNORECASE)


def _http_charset(headers: Mapping[str, str]) -> str | None:
    """Return the charset parameter of a Content-Type header, if any."""

    match = _HTTP_CHARSET_RE.search(headers.get("content-type", ""))
    return match.group(1) if match else None


def _declared_charset(content: bytes, http_charset: str | None) -> str | None:
    """The HTTP charset parameter, else a <meta> charset in the first 2 KB; None if unknown to Python."""

    candidates = [http_charset] if http_charset else []
    match = _META_CHARSET_RE.search(content[:2048])
    if match:
        candidates.append(match.group(1).decode("ascii"))
    for candidate in candidates:
        try:
            return codecs.lookup(candidate).name
        except LookupError:
            continue
    return None


def _decode_utf8_repairing_with_cp1254(content: bytes) -> tuple[str, int]:
    """Decode UTF-8; each invalid byte is decoded as windows-1254. Returns (text, repaired byte count)."""

    parts: list[str] = []
    repaired = 0
    position = 0
    while True:
        try:
            parts.append(content[position:].decode("utf-8"))
            break
        except UnicodeDecodeError as error:
            parts.append(content[position : position + error.start].decode("utf-8"))
            bad = content[position + error.start : position + error.end]
            parts.append(bad.decode("windows-1254", errors="replace"))
            repaired += len(bad)
            position += error.end
    return "".join(parts), repaired


def _decode_html(content: bytes, http_charset: str | None = None) -> str:
    """Decode HTML content for Turkish text.

    The bytes decide before any declaration does, because a single-byte charset
    such as windows-1254 "decodes" every byte string and so proves nothing:

    1. Strict UTF-8. Bytes that are valid UTF-8 are UTF-8 whatever the page or
       the server declares. A literal U+FFFD is kept for the quality scan.
    2. UTF-8 in which each invalid byte is read as windows-1254 (a stray Word
       byte 0x96 becomes an en dash and the Turkish letters around it survive),
       whenever at least one valid multi-byte UTF-8 character decoded: real
       windows-1254 text almost never forms one.
    3. The declared charset (the HTTP ``charset`` parameter, else a ``<meta>``
       charset in the first 2 KB) when the bytes decode strictly in it.
    4. windows-1254.

    After decoding, Windows-1252 C1 controls are remapped to their printable
    equivalents so en-dashes and smart quotes survive intact.
    """
    try:
        return content.decode("utf-8").translate(_WIN1252_C1_MAP)
    except UnicodeDecodeError:
        pass
    repaired_text, repaired = _decode_utf8_repairing_with_cp1254(content)
    valid_multibyte = sum(1 for character in repaired_text if ord(character) > 0x7F) - repaired
    if valid_multibyte > 0:
        return repaired_text.translate(_WIN1252_C1_MAP)
    declared = _declared_charset(content, http_charset)
    if declared is not None:
        try:
            return content.decode(declared).translate(_WIN1252_C1_MAP)
        except UnicodeDecodeError:
            pass
    return content.decode("windows-1254", errors="replace").translate(_WIN1252_C1_MAP)


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


def _response_media_type(headers: Mapping[str, str]) -> str:
    """Return a normalized media type without parameters."""

    return headers.get("content-type", "").split(";", 1)[0].strip().casefold()


def _looks_like_html_document(content: bytes) -> bool:
    """Recognize an HTML envelope without trusting its declared media type."""

    prefix = content[:4096].lstrip(b"\xef\xbb\xbf\x00\t\r\n ").lower()
    return any(marker in prefix for marker in (b"<!doctype html", b"<html", b"<head", b"<body"))


def _looks_like_soft_error_page(content: bytes) -> bool:
    """Reject known navigation and generic upstream error envelopes."""

    decoded = _decode_html(content[:_MAX_BDDK_ERROR_BODY_BYTES])
    if _is_error_page(decoded):
        return True
    folded = html.unescape(decoded).casefold()
    return any(marker in folded for marker in _SOFT_ERROR_PAGE_MARKERS)


def _classify_bddk_document_response(response: _BoundedHttpResponse) -> str:
    """Return the approved extension for one successful BDDK response."""

    if response.status_code != 200:
        raise UnsafeUpstreamResourceError("BDDK upstream did not return a successful document response.")
    if not response.content:
        raise UnsafeUpstreamResourceError("BDDK upstream returned an empty document response.")

    media_type = _response_media_type(response.headers)
    if media_type in _APPROVED_PDF_MEDIA_TYPES:
        if not response.content.lstrip(b"\xef\xbb\xbf\x00\t\r\n ").startswith(b"%PDF-"):
            raise UnsafeUpstreamResourceError("BDDK PDF response signature is invalid.")
        return ".pdf"

    if media_type in _APPROVED_HTML_MEDIA_TYPES:
        if len(response.content) > _MAX_HTML_DOWNLOAD_BYTES:
            raise UnsafeUpstreamResourceError("BDDK HTML response exceeds the processing limit.")
        if not _looks_like_html_document(response.content):
            raise UnsafeUpstreamResourceError("BDDK HTML response signature is invalid.")
        if _looks_like_soft_error_page(response.content):
            raise UnsafeUpstreamResourceError("BDDK upstream returned an error page instead of a document.")
        return ".html"

    raise UnsafeUpstreamResourceError("BDDK upstream returned an unsupported document media type.")


def _sanitize_for_storage(text: str) -> str:
    """Strip storage-unsafe bytes and uniformly-observed extraction artifacts.

    Every extraction path flows through here on the way to the document store.
    DocumentStore.store_document applies the same prepare, so writers cannot
    skip CID / data-URI stripping by avoiding this helper.
    """
    return prepare_markdown_for_storage(text)


def _extract_html_to_markdown(html: str) -> str:
    """Convert HTML content to markdown.

    Delegates to `html_extractor.html_to_markdown` which preserves tables,
    inline bold/italic, formula image refs, and mevzuat BÖLÜM / EK headings.
    """
    from bddk_mcp.ingest.html_extractor import html_to_markdown

    return html_to_markdown(html)


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


def _mevzuat_annex_zip_url(mevzuat_no: str, tur: str = "7", tertip: str = "5") -> str | None:
    """Build mevzuat.gov.tr annex ZIP URL used by `Eki için tıklayınız` links."""
    segment = MEVZUAT_TUR_MAP.get(tur)
    if not segment:
        return None
    return f"https://www.mevzuat.gov.tr/MevzuatMetin/{segment}/{tur}.{tertip}.{mevzuat_no}-ek.zip"


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


def _extract_annex_zip_markdown(zip_bytes: bytes) -> str:
    """Extract supported annex files from a mevzuat `*-ek.zip` archive."""
    if len(zip_bytes) > _MAX_ANNEX_DOWNLOAD_BYTES:
        raise UnsafeUpstreamResourceError("Annex archive exceeds the download limit.")
    sections: list[str] = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            members = _validate_zip_metadata(
                zf,
                max_members=_MAX_ARCHIVE_MEMBERS,
                max_member_bytes=_MAX_ARCHIVE_MEMBER_BYTES,
                max_total_bytes=_MAX_ARCHIVE_UNCOMPRESSED_BYTES,
                max_expansion_ratio=_MAX_ARCHIVE_EXPANSION_RATIO,
            )
            for member in members:
                lower = member.filename.lower()
                if lower.endswith(".docx") and not member.is_dir():
                    text = _extract_docx_text(zf.read(member))
                    if not text:
                        continue
                    title = member.filename.rsplit("/", 1)[-1]
                    sections.append(f"### {title}\n\n{text}")
    except UnsafeUpstreamResourceError:
        raise
    except (zipfile.BadZipFile, zipfile.LargeZipFile, RuntimeError):
        raise UnsafeUpstreamResourceError("Annex archive is invalid or unsupported.") from None
    return "\n\n".join(sections)


def _extract_docx_text(docx_bytes: bytes) -> str:
    """Extract paragraph text from a DOCX without optional markitdown dependencies."""
    if len(docx_bytes) > _MAX_ARCHIVE_MEMBER_BYTES:
        raise UnsafeUpstreamResourceError("DOCX annex exceeds the extraction limit.")
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    try:
        with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
            members = _validate_zip_metadata(
                zf,
                max_members=_MAX_DOCX_MEMBERS,
                max_member_bytes=_MAX_DOCX_XML_BYTES,
                max_total_bytes=_MAX_DOCX_UNCOMPRESSED_BYTES,
                max_expansion_ratio=_MAX_ARCHIVE_EXPANSION_RATIO,
            )
            document = next((member for member in members if member.filename == "word/document.xml"), None)
            if document is None or document.is_dir():
                raise UnsafeUpstreamResourceError("DOCX annex has no document body.")
            xml_bytes = zf.read(document)
        if len(xml_bytes) > _MAX_DOCX_XML_BYTES:
            raise UnsafeUpstreamResourceError("DOCX XML exceeds the extraction limit.")
        root = ElementTree.fromstring(xml_bytes)
    except UnsafeUpstreamResourceError:
        raise
    except (
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        ElementTree.ParseError,
        DefusedXmlException,
        RuntimeError,
    ):
        raise UnsafeUpstreamResourceError("DOCX annex is invalid or unsupported.") from None
    paragraphs: list[str] = []
    for para in root.findall(".//w:p", ns):
        parts = [node.text or "" for node in para.findall(".//w:t", ns)]
        text = "".join(parts).replace("\u00a0", " ").strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


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
        prefer_html_for_mevzuat: bool | None = None,
    ) -> None:
        self._store = store
        # ponytail: serialize this syncer's OCR model; use a worker pool if throughput requires it.
        self._extraction_lock = asyncio.Lock()
        self._ocr_backends = ocr_backends if ocr_backends is not None else get_default_backends()
        self._progress_callback = progress_callback
        self._vector_store = vector_store
        self._prefer_html_for_mevzuat = self._resolve_html_first_flag(prefer_html_for_mevzuat)
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

    async def _assert_public_resolution(self, url: str) -> None:
        try:
            await assert_public_https_resolution(url)
        except OutboundHttpPolicyError as error:
            raise UnsafeUpstreamResourceError(str(error)) from None

    async def _assert_public_mevzuat_resolution(self, url: str) -> None:
        """Compatibility seam for testing and mevzuat-specific policy."""

        await self._assert_public_resolution(url)

    async def _assert_public_bddk_resolution(self, url: str) -> None:
        """Compatibility seam for testing and BDDK-specific policy."""

        await self._assert_public_resolution(url)

    async def _fetch_bounded_https(
        self,
        candidate_url: str,
        *,
        timeout: httpx.Timeout,
        max_bytes: int,
        base_url: str,
        normalize_url: Callable[..., str],
        assert_public_resolution: Callable[[str], Awaitable[None]],
        read_body_statuses: frozenset[int] | None = None,
    ) -> _BoundedHttpResponse:
        """Stream one exact-host HTTPS resource with redirect and size bounds."""

        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        current_url = normalize_url(candidate_url, base_url=base_url)

        for redirect_count in range(_MAX_UPSTREAM_REDIRECTS + 1):
            await assert_public_resolution(current_url)
            async with self._http.stream(
                "GET",
                current_url,
                timeout=timeout,
                follow_redirects=False,
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location", "")
                    if redirect_count >= _MAX_UPSTREAM_REDIRECTS or not location:
                        raise UnsafeUpstreamResourceError("Upstream redirect policy was not satisfied.")
                    current_url = normalize_url(location, base_url=current_url)
                    continue

                # Error bodies are not useful to the BDDK document importer.
                # Returning status and headers without consuming an arbitrarily
                # large error page also keeps each retry cheap.  Mevzuat callers
                # retain the previous behavior by leaving this option unset.
                if read_body_statuses is not None and response.status_code not in read_body_statuses:
                    return _BoundedHttpResponse(
                        status_code=response.status_code,
                        content=b"",
                        headers=dict(response.headers),
                    )

                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        if int(content_length) > max_bytes:
                            raise UnsafeUpstreamResourceError("Upstream response exceeds the download limit.")
                    except ValueError:
                        raise UnsafeUpstreamResourceError("Upstream response length is invalid.") from None

                chunks: list[bytes] = []
                received = 0
                async for chunk in response.aiter_bytes():
                    received += len(chunk)
                    if received > max_bytes:
                        raise UnsafeUpstreamResourceError("Upstream response exceeds the download limit.")
                    chunks.append(chunk)
                return _BoundedHttpResponse(
                    status_code=response.status_code,
                    content=b"".join(chunks),
                    headers=dict(response.headers),
                )

        raise UnsafeUpstreamResourceError("Upstream redirect policy was not satisfied.")

    async def _fetch_trusted_mevzuat(
        self,
        candidate_url: str,
        *,
        timeout: httpx.Timeout,
        max_bytes: int,
        base_url: str = "https://www.mevzuat.gov.tr/",
    ) -> _BoundedHttpResponse:
        """Stream one approved mevzuat resource with redirect and size bounds."""

        return await self._fetch_bounded_https(
            candidate_url,
            timeout=timeout,
            max_bytes=max_bytes,
            base_url=base_url,
            normalize_url=_normalize_mevzuat_url,
            assert_public_resolution=self._assert_public_mevzuat_resolution,
        )

    async def _fetch_trusted_bddk(
        self,
        candidate_url: str,
        *,
        timeout: httpx.Timeout,
        max_bytes: int,
    ) -> _BoundedHttpResponse:
        """Stream one approved BDDK resource with bounded transient retries."""

        attempts = min(max(1, MAX_RETRIES), _MAX_BDDK_RETRY_ATTEMPTS)
        for attempt in range(attempts):
            retry_status: int | None = None
            retry_error_type: str | None = None
            try:
                response = await self._fetch_bounded_https(
                    candidate_url,
                    timeout=timeout,
                    max_bytes=max_bytes,
                    base_url="https://www.bddk.org.tr/",
                    normalize_url=_normalize_bddk_url,
                    assert_public_resolution=self._assert_public_bddk_resolution,
                    read_body_statuses=frozenset({200}),
                )
                retry_status = response.status_code
                retryable = response.status_code == 429 or response.status_code >= 500
                if not retryable or attempt == attempts - 1:
                    return response
            except httpx.TransportError as error:
                retry_error_type = type(error).__name__
                if attempt == attempts - 1:
                    raise UnsafeUpstreamResourceError("BDDK upstream request failed after bounded retries.") from None

            logger.warning(
                "Retrying bounded BDDK document request",
                extra={
                    "attempt": attempt + 1,
                    "max_attempts": attempts,
                    "status_code": retry_status,
                    "error_type": retry_error_type,
                },
            )
            await asyncio.sleep(min(2**attempt, _MAX_BDDK_RETRY_DELAY_SECONDS))

        raise UnsafeUpstreamResourceError("BDDK upstream request failed after bounded retries.")

    def _resolve_html_first_flag(self, explicit: bool | None) -> bool:
        """Pick the effective HTML-first routing flag for mevzuat downloads.

        Precedence: explicit ctor arg → BDDK_PREFER_HTML_FOR_MEVZUAT env var
        (via config.PREFER_HTML_FOR_MEVZUAT). When the env var resolves to
        "auto", flip to True iff no formula-capable OCR backend is available
        — the *_degraded backends (pdftotext, markitdown) lose formulas and
        tables on PDF, so the rich HTML path beats them whenever only they are on hand.
        """
        if explicit is not None:
            return explicit
        setting = PREFER_HTML_FOR_MEVZUAT
        if setting in ("1", "true", "yes"):
            return True
        if setting in ("0", "false", "no"):
            return False
        # "auto": prefer HTML iff every available backend is degraded (*_degraded: no formulas or tables).
        formula_capable = any(
            not getattr(b, "name", "").endswith("_degraded") and b.is_available() for b in self._ocr_backends
        )
        return not formula_capable

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

        # Skip when the text is stored and, if a vector store is attached, its chunks
        # are current. Stored text without current chunks (an earlier reindex_failed,
        # an interrupted build) is re-indexed from the stored text, not downloaded again.
        if not force and await self._store.has_document(doc_id):
            if self._vector_store is None or await self._vector_store.has_document(doc_id):
                return SyncResult(document_id=doc_id, success=True, method="cached")
            stored = await self._store.get_document(doc_id)
            if stored is not None and stored.markdown_content:
                return await self._reindex_stored_document(stored, source_url)

        # If re-extracting with force=True and the PDF is already cached in DB,
        # skip re-downloading (bandwidth saving).
        # Exception: when HTML-first routing is active, the whole point is to
        # route *around* the PDF, so the cached PDF shortcut would sabotage it
        # (re-extracting the same PDF reproduces the degraded markdown).
        cached_pdf: bytes | None = None
        if force and doc_id.startswith("mevzuat_") and not self._prefer_html_for_mevzuat:
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
            cat, retryable = _categorize_error(error_msg, doc_id)
            await self._store.record_sync_failure(doc_id, error_msg, cat, source_url, retryable)
            return SyncResult(
                document_id=doc_id,
                success=False,
                error=error_msg,
            )

        if not content:
            error_msg = "No content downloaded"
            cat, retryable = _categorize_error(error_msg, doc_id)
            await self._store.record_sync_failure(doc_id, error_msg, cat, source_url, retryable)
            return SyncResult(
                document_id=doc_id,
                success=False,
                error=error_msg,
            )

        # Extract markdown
        async with self._extraction_lock:
            worker = asyncio.create_task(asyncio.to_thread(self._extract, content, ext))
            cancellation = None
            # On cancellation, wait for the OCR worker thread to finish so the extraction lock
            # (and the GPU) are not released while it is still running.
            with anyio.CancelScope(shield=True):
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError as error:
                        cancellation = error
                    except Exception:  # the finished worker's error is read below
                        pass
                if cancellation is not None:
                    worker.exception()  # the worker is done; its own error yields to the cancellation
                    raise cancellation
                try:
                    markdown, extraction_method = worker.result()
                except Exception as error:  # CancelledError is a BaseException and still propagates
                    extraction_error: Exception | None = error
                else:
                    extraction_error = None
        if extraction_error is not None:
            error_msg = f"Extraction failed: {extraction_error}"
            cat, retryable = _categorize_error(error_msg, doc_id)
            await self._store.record_sync_failure(doc_id, error_msg, cat, source_url, retryable)
            logger.warning(
                "Document extraction raised; preserving old content (force=%s)",
                force,
                extra={"error_type": type(extraction_error).__name__},
            )
            return SyncResult(document_id=doc_id, success=False, error=error_msg)
        markdown = _sanitize_for_storage(markdown or "")
        if not markdown:
            error_msg = f"Extraction failed (method={extraction_method})"
            cat, retryable = _categorize_error(error_msg, doc_id)
            await self._store.record_sync_failure(doc_id, error_msg, cat, source_url, retryable)
            # Preserve old content on failed force re-extract — losing it would
            # erase successful prior extractions when a new backend transiently fails.
            logger.warning(
                "Document extraction failed; preserving old content (force=%s)",
                force,
                extra={"error_type": "EmptyExtractionResult"},
            )
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

        if self._vector_store is not None:
            try:
                await self._vector_store.add_document(
                    doc_id=doc_id,
                    title=title or doc_id,
                    content=doc.markdown_content,
                    category=category,
                    decision_date=decision_date,
                    decision_number=decision_number,
                    source_url=source_url,
                )
            except Exception as error:
                logger.warning(
                    "Vector index publication failed after document sync; "
                    "hash-gated retrieval will hide stale chunks until the next sync retries it",
                    extra={"error_type": type(error).__name__},
                )
                await self._store.record_sync_failure(
                    doc_id,
                    "reindex_failed",
                    "index",
                    source_url,
                    True,
                )
                return SyncResult(
                    document_id=doc_id,
                    success=False,
                    method=f"{method}+{extraction_method}",
                    error="reindex_failed",
                    size_bytes=len(content),
                )

        await self._store.clear_sync_failure(doc_id)
        return SyncResult(
            document_id=doc_id,
            success=True,
            method=f"{method}+{extraction_method}",
            size_bytes=len(content),
        )

    async def _reindex_stored_document(self, stored: StoredDocument, source_url: str) -> SyncResult:
        """Publish chunks for stored text; the stored text and extraction method stay untouched."""
        assert self._vector_store is not None
        doc_id = stored.document_id
        try:
            await self._vector_store.add_document(
                doc_id=doc_id,
                title=stored.title or doc_id,
                content=stored.markdown_content,
                category=stored.category,
                decision_date=stored.decision_date,
                decision_number=stored.decision_number,
                source_url=stored.source_url,
            )
        except Exception as error:
            logger.warning(
                "Vector index publication failed for stored document text; the next sync retries it",
                extra={"error_type": type(error).__name__},
            )
            await self._store.record_sync_failure(
                doc_id, "reindex_failed", "index", stored.source_url or source_url, True
            )
            return SyncResult(document_id=doc_id, success=False, method="reindexed", error="reindex_failed")
        await self._store.clear_sync_failure(doc_id)
        return SyncResult(
            document_id=doc_id,
            success=True,
            method="reindexed",
            size_bytes=len(stored.markdown_content.encode()),
        )

    # ── Download methods ─────────────────────────────────────────────────

    async def _download_bddk(self, doc_id: str) -> tuple[bytes, str, str]:
        """Download from BDDK DokumanGetir endpoint."""
        if not re.fullmatch(r"[0-9]{1,20}", doc_id):
            raise UnsafeUpstreamResourceError("BDDK document identifier is invalid.")
        url = _BDDK_DOC_URL.format(document_id=doc_id)
        resp = await self._fetch_trusted_bddk(
            url,
            timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=HTTP_CONNECT_TIMEOUT, pool=HTTP_POOL_TIMEOUT),
            max_bytes=_MAX_PDF_DOWNLOAD_BYTES,
        )
        ext = _classify_bddk_document_response(resp)
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

        if not re.fullmatch(r"[0-9]{1,20}", mevzuat_no):
            raise UnsafeUpstreamResourceError("Mevzuat document identifier is invalid.")
        if tur not in MEVZUAT_TUR_MAP:
            raise UnsafeUpstreamResourceError("Mevzuat document type is invalid.")
        if not re.fullmatch(r"[0-9]{1,3}", tertip):
            raise UnsafeUpstreamResourceError("Mevzuat document series is invalid.")

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
                resp = await self._fetch_trusted_mevzuat(
                    main_url,
                    timeout=layer_timeout,
                    max_bytes=_MAX_MAIN_PAGE_BYTES,
                )
                if resp.status_code == 200:
                    main_page_visited = True
                    main_page_html = _decode_html(resp.content, _http_charset(resp.headers))
            except UnsafeUpstreamResourceError:
                raise
            except Exception as error:
                logger.debug(
                    "Mevzuat main-page visit failed",
                    extra={"error_type": type(error).__name__},
                )

            # Layer 1b (HTML-first route): when no formula-capable OCR backend is
            # available, the rich iframe HTML produces a better extraction than
            # markitdown-on-PDF. Try the iframe immediately after the main page.
            if self._prefer_html_for_mevzuat and main_page_visited and main_page_html:
                iframe_result = await self._try_iframe_layer(
                    doc_id,
                    candidate_tur,
                    main_page_html,
                    layer_timeout,
                    mevzuat_no=mevzuat_no,
                    tertip=tertip,
                )
                if iframe_result is not None:
                    return iframe_result

            # Layer 2: GeneratePdf API (preferred — server-rendered PDF with all formulas as images)
            if main_page_visited:
                try:
                    gen_pdf_url = _mevzuat_generate_pdf_url(mevzuat_no, candidate_tur, tertip)
                    if gen_pdf_url:
                        resp = await self._fetch_trusted_mevzuat(
                            gen_pdf_url,
                            timeout=layer_timeout,
                            max_bytes=_MAX_PDF_DOWNLOAD_BYTES,
                        )
                        if resp.status_code == 200 and len(resp.content) > 500 and resp.content[:5] == b"%PDF-":
                            logger.info("Mevzuat document downloaded via GeneratePdf")
                            return resp.content, "mevzuat_generate_pdf", ".pdf"
                except UnsafeUpstreamResourceError:
                    raise
                except Exception as error:
                    logger.debug(
                        "Mevzuat GeneratePdf download failed",
                        extra={"error_type": type(error).__name__},
                    )

            # Layer 3: Direct static .pdf
            try:
                pdf_url = _mevzuat_pdf_url(mevzuat_no, candidate_tur, tertip)
                if pdf_url:
                    resp = await self._fetch_trusted_mevzuat(
                        pdf_url,
                        timeout=layer_timeout,
                        max_bytes=_MAX_PDF_DOWNLOAD_BYTES,
                    )
                    if resp.status_code == 200 and len(resp.content) > 500 and resp.content[:5] == b"%PDF-":
                        logger.info("Mevzuat document downloaded via static PDF")
                        return resp.content, "mevzuat_pdf", ".pdf"
            except UnsafeUpstreamResourceError:
                raise
            except Exception as error:
                logger.debug(
                    "Mevzuat static-PDF download failed",
                    extra={"error_type": type(error).__name__},
                )

            # Layer 4: .htm fallback — formulas may be lost (rendered as <img>)
            try:
                htm_url = f"https://www.mevzuat.gov.tr/mevzuatmetin/{segment}/{base}.htm"
                resp = await self._fetch_trusted_mevzuat(
                    htm_url,
                    timeout=layer_timeout,
                    max_bytes=_MAX_HTML_DOWNLOAD_BYTES,
                )
                if (
                    resp.status_code == 200
                    and len(resp.content) > 200
                    and not _is_error_page(_decode_html(resp.content, _http_charset(resp.headers)))
                ):
                    logger.warning("Mevzuat extraction is falling back to HTML; formulas may be lost")
                    return resp.content, "mevzuat_htm", ".html"
            except UnsafeUpstreamResourceError:
                raise
            except Exception as error:
                logger.debug(
                    "Mevzuat HTML download failed",
                    extra={"error_type": type(error).__name__},
                )

            # Layer 5: iframe/div from already-fetched main page
            if main_page_visited and main_page_html:
                iframe_result = await self._try_iframe_layer(
                    doc_id,
                    candidate_tur,
                    main_page_html,
                    layer_timeout,
                    mevzuat_no=mevzuat_no,
                    tertip=tertip,
                )
                if iframe_result is not None:
                    return iframe_result

            # Layer 6: Word (.doc) — heaviest, slowest (only try for the first/default tur
            # to avoid excessive requests during auto-detection)
            if candidate_tur == tur:
                try:
                    doc_url = _mevzuat_doc_url(mevzuat_no, candidate_tur, tertip)
                    resp = await self._fetch_trusted_mevzuat(
                        doc_url,
                        timeout=httpx.Timeout(90.0, connect=15.0),
                        max_bytes=_MAX_DOC_DOWNLOAD_BYTES,
                    )
                    if (
                        resp.status_code == 200
                        and len(resp.content) > 100
                        and resp.content[:4] in (b"\xd0\xcf\x11\xe0", b"PK\x03\x04")
                    ):
                        logger.info("Mevzuat document downloaded via Word fallback")
                        return resp.content, "mevzuat_doc", ".doc"
                except UnsafeUpstreamResourceError:
                    raise
                except Exception as error:
                    logger.debug(
                        "Mevzuat Word download failed",
                        extra={"error_type": type(error).__name__},
                    )

            if candidate_tur != tur_candidates[-1]:
                logger.debug("Mevzuat download candidate failed; trying next configured candidate")

        raise RuntimeError(f"All download methods failed for {doc_id} (tried tur values: {tur_candidates})")

    async def _try_iframe_layer(
        self,
        doc_id: str,
        candidate_tur: str,
        main_page_html: str,
        layer_timeout: httpx.Timeout,
        mevzuat_no: str = "",
        tertip: str = "5",
    ) -> tuple[bytes, str, str] | None:
        """Fetch mevzuatDetayIframe content referenced by the main page.

        Returns (content, method, ext) or None if the iframe/div can't be
        resolved. Exceptions are caught and logged — the caller falls through
        to other download layers.
        """
        try:
            soup = BeautifulSoup(main_page_html, "html.parser")
            iframe = soup.find("iframe", src=True)
            if iframe:
                iframe_resp = await self._fetch_trusted_mevzuat(
                    str(iframe["src"]),
                    timeout=layer_timeout,
                    max_bytes=_MAX_IFRAME_BYTES,
                )
                if (
                    iframe_resp.status_code == 200
                    and len(iframe_resp.content) > 200
                    and not _is_error_page(_decode_html(iframe_resp.content, _http_charset(iframe_resp.headers)))
                ):
                    logger.info("Mevzuat document fetched from approved iframe layer")
                    content, annex_merged = await self._append_annex_zip_if_present(
                        iframe_resp.content,
                        doc_id=doc_id,
                        mevzuat_no=mevzuat_no or doc_id.removeprefix("mevzuat_"),
                        tur=candidate_tur,
                        tertip=tertip,
                        layer_timeout=layer_timeout,
                    )
                    method = "mevzuat_iframe+annex_zip" if annex_merged else "mevzuat_iframe"
                    return content, method, ".html"
            div = soup.find("div", id="divMevzuatMetni")
            if div and len(div.get_text(strip=True)) > 100 and not _is_error_page(str(div)):
                logger.info("Mevzuat document fetched from approved main-page layer")
                return str(div).encode("utf-8"), "mevzuat_div", ".html"
        except Exception as error:
            logger.debug(
                "Mevzuat iframe/main-page layer failed",
                extra={"error_type": type(error).__name__},
            )
        return None

    async def _append_annex_zip_if_present(
        self,
        content: bytes,
        *,
        doc_id: str,
        mevzuat_no: str,
        tur: str,
        tertip: str,
        layer_timeout: httpx.Timeout,
    ) -> tuple[bytes, bool]:
        """Append docx annex text from mevzuat `*-ek.zip` links when present."""
        html_text = _decode_html(content)
        soup = BeautifulSoup(html_text, "html.parser")
        if not soup.find("a", href=lambda href: bool(href and href.lower().endswith("-ek.zip"))):
            return content, False

        annex_url = _mevzuat_annex_zip_url(mevzuat_no, tur, tertip)
        if not annex_url:
            return content, False
        try:
            resp = await self._fetch_trusted_mevzuat(
                annex_url,
                timeout=layer_timeout,
                max_bytes=_MAX_ANNEX_DOWNLOAD_BYTES,
            )
            if resp.status_code != 200 or len(resp.content) < 100:
                return content, False
            annex_markdown = _extract_annex_zip_markdown(resp.content)
        except Exception as error:
            logger.debug(
                "Mevzuat annex archive fetch or extraction failed",
                extra={"error_type": type(error).__name__},
            )
            return content, False

        if not annex_markdown:
            return content, False
        # ASCII with character references, so the block decodes the same in whatever
        # charset the iframe page uses.
        annex_html = (
            f'\n\n<div id="mevzuat-ekler">\n<h2>Ekler</h2>\n<pre>{html.escape(annex_markdown)}</pre>\n</div>\n'
        ).encode("ascii", "xmlcharrefreplace")
        # The HTML extractor walks only the first <body>: a block after </body></html> would be dropped.
        body_end = content.lower().find(b"</body")
        if body_end == -1:
            return content + annex_html, True
        return content[:body_end] + annex_html + content[body_end:], True

    # ── Extraction ───────────────────────────────────────────────────────

    def _extract(self, content: bytes, ext: str) -> tuple[str, str]:
        """Extract markdown from downloaded content. Returns (markdown, method).

        Uses a structured pipeline and logs detailed failure reasons.
        """
        extraction = self._extract_structured(content, ext)
        if extraction.error:
            logger.warning(
                "Document extraction returned an issue (retryable=%s)",
                extraction.retryable,
                extra={"error_type": "ExtractionIssue"},
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
            try:
                result = _extract_html_to_markdown(html_str)
            except Exception as error:
                # Untrusted upstream HTML: a parser error falls through to markitdown.
                logger.warning(
                    "HTML parser extraction raised; trying markitdown",
                    extra={"error_type": type(error).__name__},
                )
                result = ""
                errors.append(f"html_parser: raised {type(error).__name__}")
            if result and not _is_error_page(result):
                return ExtractionResult(content=result, method="html_parser")
            if result:
                errors.append("html_parser: extracted content is a 404/navigation page")
            elif not errors:
                errors.append("html_parser: no meaningful content extracted")

            try:
                from markitdown import MarkItDown

                md = MarkItDown()
                html_result = md.convert_stream(io.BytesIO(content), file_extension=".html").text_content.strip()
                if html_result and not _is_error_page(html_result):
                    return ExtractionResult(content=html_result, method="markitdown")
                errors.append("markitdown: HTML fallback failed or error page")
            except (MarkItDownException, ValueError, OSError, UnicodeDecodeError) as e:
                errors.append(f"markitdown: {e}")

            retryable = len(content) < 200 or any("404" in e or "navigation" in e for e in errors)
            return ExtractionResult(method="failed", error="; ".join(errors), retryable=retryable)

        if ext in (".doc", ".docx"):
            if len(content) > _MAX_DOC_DOWNLOAD_BYTES:
                return ExtractionResult(
                    method="failed",
                    error="office document exceeds the processing limit",
                    retryable=False,
                )
            markitdown_extension = ext
            if content.startswith(_ZIP_LOCAL_FILE_HEADER):
                try:
                    _validate_office_archive_for_markitdown(content)
                except UnsafeUpstreamResourceError:
                    return ExtractionResult(
                        method="failed",
                        error="office archive rejected by safety policy",
                        retryable=False,
                    )
                # Mevzuat labels both legacy OLE and OOXML downloads as .doc.
                # Route a validated ZIP container through the DOCX converter.
                markitdown_extension = ".docx"
            elif ext == ".docx" or not content.startswith(_OLE_COMPOUND_FILE_HEADER):
                return ExtractionResult(
                    method="failed",
                    error="office document signature is invalid",
                    retryable=False,
                )
            else:
                # markitdown has no converter for legacy OLE Word files; fail with a clear reason.
                raise LegacyDocNotSupportedError()
            try:
                from markitdown import MarkItDown

                md = MarkItDown()
                result = md.convert_stream(
                    io.BytesIO(content),
                    file_extension=markitdown_extension,
                ).text_content.strip()
                if result and len(result) >= OCR_MIN_CONTENT_LEN:
                    return ExtractionResult(content=result, method="markitdown")
                return ExtractionResult(method="failed", error="markitdown output too short", retryable=True)
            except (MarkItDownException, ValueError, OSError, UnicodeDecodeError) as e:
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

        for doc_info, r in zip(documents, results, strict=True):
            if isinstance(r, Exception):
                report.failed += 1
                report.errors.append(
                    SyncResult(document_id=doc_info.get("document_id") or "unknown", success=False, error=str(r))
                )
            elif r.success:
                if r.method == "cached":
                    report.skipped += 1
                elif r.method == "reindexed":
                    report.reindexed += 1
                else:
                    report.downloaded += 1
            else:
                report.failed += 1
                report.errors.append(r)

        report.elapsed_seconds = round(time.time() - start, 2)
        return report
