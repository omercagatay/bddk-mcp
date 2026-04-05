import io
import logging
import math
import re
import time
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup, Tag
from markitdown import MarkItDown

from models import (
    BddkDecisionSummary,
    BddkDocumentMarkdown,
    BddkSearchRequest,
    BddkSearchResult,
)

logger = logging.getLogger(__name__)

_DOCUMENT_URL_TEMPLATE = "https://www.bddk.org.tr/Mevzuat/DokumanGetir/{document_id}"
_BDDK_BASE_URL = "https://www.bddk.org.tr"
_CHUNK_SIZE = 5000
_LIST_PAGE_IDS = [50, 55, 56]
_CACHE_TTL_SECONDS = 3600  # 1 hour

# Maps h5 header text (without count suffix) to singular category name
_PAGE_50_CATEGORY_MAP = {
    "Yönetmelikler": "Yönetmelik",
    "Genelgeler": "Genelge",
    "Tebliğler": "Tebliğ",
    "Rehberler": "Rehber",
    "Bilgi Sistemleri ve İş Süreçlerine İlişkin Düzenlemeler": "Bilgi Sistemleri",
    "Sermaye Yeterliliğine İlişkin Tebliğler ve Rehberler": "Sermaye Yeterliliği",
    "Faizsiz Bankacılığa İlişkin Düzenlemeler": "Faizsiz Bankacılık",
    "Tekdüzen Hesap Planı": "Tekdüzen Hesap Planı",
}

# mevzuat.gov.tr MevzuatTur to path segment mapping
_MEVZUAT_TUR_MAP = {
    "7": "yonetmelik",
    "9": "teblig",
}


def _turkish_lower(text: str) -> str:
    """Turkish-aware lowercase conversion."""
    return (
        text
        .replace("İ", "i")
        .replace("I", "ı")
        .replace("Ş", "ş")
        .replace("Ç", "ç")
        .replace("Ü", "ü")
        .replace("Ö", "ö")
        .replace("Ğ", "ğ")
        .lower()
    )


def _external_url_to_id(url: str) -> Optional[str]:
    """Generate a stable synthetic ID from an external URL.

    For mevzuat.gov.tr URLs, extracts MevzuatNo to produce IDs like 'mevzuat_42628'.
    Handles both new format (?MevzuatNo=42628) and old format (?MevzuatKod=7.5.24788).
    Returns None if the URL cannot be parsed.
    """
    parsed = urlparse(url)
    if "mevzuat.gov.tr" in parsed.netloc:
        params = parse_qs(parsed.query)
        # New format: ?MevzuatNo=42628
        mevzuat_no = params.get("MevzuatNo", [None])[0]
        if mevzuat_no:
            return f"mevzuat_{mevzuat_no}"
        # Old format: ?MevzuatKod=7.5.24788 (last segment is the number)
        mevzuat_kod = params.get("MevzuatKod", [None])[0]
        if mevzuat_kod:
            parts = mevzuat_kod.split(".")
            if len(parts) >= 3:
                return f"mevzuat_{parts[-1]}"
    return None


def _mevzuat_to_pdf_url(mevzuat_no: str, mevzuat_tur: str = "7", mevzuat_tertip: str = "5") -> Optional[str]:
    """Convert mevzuat.gov.tr parameters to a direct PDF download URL."""
    path_segment = _MEVZUAT_TUR_MAP.get(mevzuat_tur)
    if not path_segment:
        return None
    return f"https://www.mevzuat.gov.tr/MevzuatMetin/{path_segment}/{mevzuat_tur}.{mevzuat_tertip}.{mevzuat_no}.pdf"


class BddkApiClient:
    """
    Client for searching and retrieving BDDK decisions.

    Directly scrapes BDDK website and caches decision lists in memory.
    """

    def __init__(self, request_timeout: float = 60.0) -> None:
        self._http = httpx.AsyncClient(
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
            },
            timeout=httpx.Timeout(request_timeout),
            follow_redirects=True,
        )
        self._md = MarkItDown()
        self._cache: List[BddkDecisionSummary] = []
        self._cache_timestamp: float = 0.0

    # -- context manager --------------------------------------------------

    async def __aenter__(self) -> "BddkApiClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        await self._http.aclose()
        logger.info("BddkApiClient session closed")

    # -- cache ------------------------------------------------------------

    def _is_cache_valid(self) -> bool:
        return (
            len(self._cache) > 0
            and (time.time() - self._cache_timestamp) < _CACHE_TTL_SECONDS
        )

    async def _fetch_and_parse_list_page(self, list_id: int) -> List[BddkDecisionSummary]:
        """Fetch a BDDK list page and parse decisions from HTML."""
        url = f"{_BDDK_BASE_URL}/Mevzuat/Liste/{list_id}"
        logger.info("Fetching BDDK list page: %s", url)

        response = await self._http.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        decisions: List[BddkDecisionSummary] = []

        links = soup.find_all("a", href=re.compile(r'/Mevzuat/DokumanGetir/\d+'))

        for link in links:
            href = link.get("href", "")
            doc_id_match = re.search(r'/DokumanGetir/(\d+)', href)
            if not doc_id_match:
                continue

            doc_id = doc_id_match.group(1)
            raw_text = link.get_text(strip=True)
            if not raw_text:
                continue

            date_match = re.match(r'\((\d{2}\.\d{2}\.\d{4})\s*-\s*(\d+)\)\s*(.*)', raw_text)
            if date_match:
                decision_date = date_match.group(1)
                decision_number = date_match.group(2)
                title = date_match.group(3).strip()
            else:
                decision_date = ""
                decision_number = ""
                title = raw_text

            decisions.append(BddkDecisionSummary(
                title=title,
                document_id=doc_id,
                content=title,
                decision_date=decision_date,
                decision_number=decision_number,
                category="Kurul Kararı",
                source_url=_DOCUMENT_URL_TEMPLATE.format(document_id=doc_id),
            ))

        logger.info("Parsed %d decisions from list page %d", len(decisions), list_id)
        return decisions

    async def _fetch_and_parse_page_50(self) -> List[BddkDecisionSummary]:
        """Fetch page 50 and parse regulations from accordion card structure."""
        url = f"{_BDDK_BASE_URL}/Mevzuat/Liste/50"
        logger.info("Fetching BDDK page 50: %s", url)

        response = await self._http.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        decisions: List[BddkDecisionSummary] = []
        seen_urls: set[str] = set()

        # Page uses Bootstrap accordion: div.card > div.card-header(h5) + div.collapse(card-body)
        cards = soup.find_all("div", class_="card")

        for card in cards:
            h5 = card.find("h5")
            if not h5:
                continue

            raw_header = h5.get_text(strip=True)
            header_name = re.sub(r"\s*\(\d+\)\s*$", "", raw_header).strip()
            category = _PAGE_50_CATEGORY_MAP.get(header_name, header_name)

            # Links are inside collapse > card-body
            body = card.find("div", class_="card-body")
            if not body:
                body = card.find("div", class_="collapse")
            if not body:
                continue

            for anchor in body.find_all("a", href=True):
                href = anchor.get("href", "").strip()
                text = anchor.get_text(strip=True)

                # Skip icon-only links (no text) and detail links
                if not text or "/Mevzuat/Detay/" in href:
                    continue

                # Skip duplicates
                if href in seen_urls:
                    continue
                seen_urls.add(href)

                # Internal DokumanGetir link
                doc_id_match = re.search(r"/DokumanGetir/(\d+)", href)
                if doc_id_match:
                    doc_id = doc_id_match.group(1)
                    source_url = f"{_BDDK_BASE_URL}{href}" if href.startswith("/") else href
                    decisions.append(BddkDecisionSummary(
                        title=text,
                        document_id=doc_id,
                        content=text,
                        category=category,
                        source_url=source_url,
                    ))
                    continue

                # External link (mevzuat.gov.tr or other)
                if href.startswith("http"):
                    synthetic_id = _external_url_to_id(href)
                    if synthetic_id:
                        decisions.append(BddkDecisionSummary(
                            title=text,
                            document_id=synthetic_id,
                            content=text,
                            category=category,
                            source_url=href,
                        ))

        logger.info("Parsed %d items from page 50", len(decisions))
        return decisions

    async def _ensure_cache(self) -> None:
        """Ensure the decision cache is populated and valid."""
        if self._is_cache_valid():
            return

        logger.info("Refreshing BDDK cache...")
        all_decisions: List[BddkDecisionSummary] = []

        for list_id in _LIST_PAGE_IDS:
            try:
                if list_id == 50:
                    page_decisions = await self._fetch_and_parse_page_50()
                else:
                    page_decisions = await self._fetch_and_parse_list_page(list_id)
                all_decisions.extend(page_decisions)
            except Exception as e:
                logger.error("Failed to fetch BDDK list page %d: %s", list_id, e)

        self._cache = all_decisions
        self._cache_timestamp = time.time()
        logger.info("BDDK cache refreshed: %d total items", len(self._cache))

    # -- public API -------------------------------------------------------

    async def search_decisions(
        self,
        request: BddkSearchRequest,
    ) -> BddkSearchResult:
        """Search for BDDK decisions/regulations by keyword matching against cached list."""
        await self._ensure_cache()

        keywords_lower = _turkish_lower(request.keywords)
        keyword_parts = keywords_lower.split()

        # Optional category filter
        category_filter = _turkish_lower(request.category) if request.category else None

        matching: List[BddkDecisionSummary] = []
        for dec in self._cache:
            # Apply category filter first
            if category_filter and category_filter not in _turkish_lower(dec.category):
                continue

            text_lower = _turkish_lower(
                f"{dec.title} {dec.decision_date} {dec.decision_number} {dec.category}"
            )
            if all(part in text_lower for part in keyword_parts):
                matching.append(dec)

        total_results = len(matching)
        start_idx = (request.page - 1) * request.page_size
        end_idx = start_idx + request.page_size
        page_results = matching[start_idx:end_idx]

        logger.info(
            "BDDK search for '%s': %d matches, returning page %d (%d items)",
            request.keywords, total_results, request.page, len(page_results),
        )

        return BddkSearchResult(
            decisions=page_results,
            total_results=total_results,
            page=request.page,
            page_size=request.page_size,
        )

    def _resolve_document_url(self, document_id: str) -> str:
        """Resolve a document ID to a fetchable URL."""
        # Standard numeric BDDK document ID
        if document_id.isdigit():
            return _DOCUMENT_URL_TEMPLATE.format(document_id=document_id)

        # Synthetic mevzuat.gov.tr ID (e.g., "mevzuat_42628")
        if document_id.startswith("mevzuat_"):
            mevzuat_no = document_id.removeprefix("mevzuat_")
            # Look up source_url from cache for tur/tertip info
            for dec in self._cache:
                if dec.document_id == document_id and dec.source_url:
                    source = dec.source_url
                    parsed = urlparse(source)
                    params = parse_qs(parsed.query)

                    # New format: ?MevzuatNo=...&MevzuatTur=...
                    tur = params.get("MevzuatTur", [None])[0]
                    tertip = params.get("MevzuatTertip", [None])[0]

                    # Old format: ?MevzuatKod=7.5.24788 (tur.tertip.no)
                    if not tur:
                        kod = params.get("MevzuatKod", [None])[0]
                        if kod:
                            parts = kod.split(".")
                            if len(parts) >= 3:
                                tur = parts[0]
                                tertip = parts[1]

                    pdf_url = _mevzuat_to_pdf_url(mevzuat_no, tur or "7", tertip or "5")
                    if pdf_url:
                        return pdf_url
                    break

            # Fallback: assume yönetmelik (tur=7, tertip=5)
            pdf_url = _mevzuat_to_pdf_url(mevzuat_no)
            if pdf_url:
                return pdf_url

        # Fallback: try as raw BDDK document ID
        return _DOCUMENT_URL_TEMPLATE.format(document_id=document_id)

    async def get_document_markdown(
        self,
        document_id: str,
        page_number: int = 1,
    ) -> BddkDocumentMarkdown:
        """Fetch a BDDK document and return its content as paginated Markdown."""
        url = self._resolve_document_url(document_id)
        logger.info("Fetching BDDK document: %s", url)

        response = await self._http.get(url)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        ext = ".pdf" if "pdf" in content_type else ".html"

        result = self._md.convert_stream(
            io.BytesIO(response.content),
            file_extension=ext,
        )
        markdown = result.text_content.strip()

        total_pages = max(1, math.ceil(len(markdown) / _CHUNK_SIZE))
        start = (page_number - 1) * _CHUNK_SIZE
        page_content = markdown[start : start + _CHUNK_SIZE]

        return BddkDocumentMarkdown(
            document_id=document_id,
            markdown_content=page_content,
            page_number=page_number,
            total_pages=total_pages,
        )
