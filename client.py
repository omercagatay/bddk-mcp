import io
import json
import logging
import math
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup
from markitdown import MarkItDown

from doc_store import DocumentStore
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
_CACHE_TTL_SECONDS = 3600  # 1 hour
_MAX_RETRIES = 3
_CACHE_FILE = Path(__file__).parent / ".cache.json"

# Pages that use accordion card structure (h5 headers with card-body)
_ACCORDION_PAGE_IDS = [50, 51]
# Pages that use flat list with DokumanGetir links and (date - number) format
_DECISION_PAGE_IDS = [55, 56]
# Pages that use flat list with mixed link types (no date format)
_FLAT_PAGE_IDS = [49, 52, 54, 58, 63]

_ALL_PAGE_IDS = _ACCORDION_PAGE_IDS + _DECISION_PAGE_IDS + _FLAT_PAGE_IDS

# Maps h5 header text (without count suffix) to singular category name
_ACCORDION_CATEGORY_MAP = {
    # Page 50
    "Yönetmelikler": "Yönetmelik",
    "Genelgeler": "Genelge",
    "Tebliğler": "Tebliğ",
    "Rehberler": "Rehber",
    "Bilgi Sistemleri ve İş Süreçlerine İlişkin Düzenlemeler": "Bilgi Sistemleri",
    "Sermaye Yeterliliğine İlişkin Tebliğler ve Rehberler": "Sermaye Yeterliliği",
    "Faizsiz Bankacılığa İlişkin Düzenlemeler": "Faizsiz Bankacılık",
    "Tekdüzen Hesap Planı": "Tekdüzen Hesap Planı",
    # Page 51
    "Yönetmelik": "Yönetmelik",
    "Tebliğ": "Tebliğ",
    "Banka Kartları ve Kredi Kartları Kanununa İlişkin Düzenlemeler": "Banka Kartları",
}

# Category assigned to flat-list pages (by page ID)
_FLAT_PAGE_CATEGORY = {
    49: "Kanun",
    52: "Finansal Kiralama ve Faktoring",
    54: "BDDK Düzenlemesi",
    58: "Düzenleme Taslağı",
    63: "Mülga Düzenleme",
}

# mevzuat.gov.tr MevzuatTur to path segment mapping
_MEVZUAT_TUR_MAP = {
    "1": "kanun",
    "2": "kanunhukmundekararname",
    "4": "cumhurbaskanligikararnamesi",
    "5": "tuzuk",
    "7": "yonetmelik",
    "9": "teblig",
    "11": "cumhurbaskanligikararnamesi",
}

# Common Turkish suffixes for basic stemming
_TURKISH_SUFFIXES = [
    "ları",
    "leri",
    "ların",
    "lerin",
    "lara",
    "lere",
    "lardan",
    "lerden",
    "larla",
    "lerle",
    "lar",
    "ler",
    "ının",
    "inin",
    "unun",
    "ünün",
    "ına",
    "ine",
    "una",
    "üne",
    "ında",
    "inde",
    "unda",
    "ünde",
    "ından",
    "inden",
    "undan",
    "ünden",
    "ıyla",
    "iyle",
    "uyla",
    "üyle",
    "nın",
    "nin",
    "nun",
    "nün",
    "dan",
    "den",
    "tan",
    "ten",
    "ya",
    "ye",
    "da",
    "de",
    "ta",
    "te",
    "ın",
    "in",
    "un",
    "ün",
    "na",
    "ne",
    "dır",
    "dir",
    "dur",
    "dür",
    "tır",
    "tir",
    "tur",
    "tür",
]


def _turkish_lower(text: str) -> str:
    """Turkish-aware lowercase conversion."""
    return (
        text.replace("İ", "i")
        .replace("I", "ı")
        .replace("Ş", "ş")
        .replace("Ç", "ç")
        .replace("Ü", "ü")
        .replace("Ö", "ö")
        .replace("Ğ", "ğ")
        .lower()
    )


def _turkish_stem(word: str) -> str:
    """Strip common Turkish suffixes for basic morphological matching."""
    for suffix in _TURKISH_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _parse_date(date_str: str) -> datetime | None:
    """Parse DD.MM.YYYY date string to datetime."""
    try:
        return datetime.strptime(date_str, "%d.%m.%Y")
    except (ValueError, TypeError):
        return None


def _external_url_to_id(url: str) -> str | None:
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


def _mevzuat_to_pdf_url(mevzuat_no: str, mevzuat_tur: str = "7", mevzuat_tertip: str = "5") -> str | None:
    """Convert mevzuat.gov.tr parameters to a direct PDF download URL."""
    path_segment = _MEVZUAT_TUR_MAP.get(mevzuat_tur)
    if not path_segment:
        return None
    return f"https://www.mevzuat.gov.tr/MevzuatMetin/{path_segment}/{mevzuat_tur}.{mevzuat_tertip}.{mevzuat_no}.pdf"


class BddkApiClient:
    """
    Client for searching and retrieving BDDK decisions and regulations.

    Scrapes BDDK website, caches document lists in memory and on disk.
    """

    def __init__(
        self,
        request_timeout: float = 60.0,
        doc_store: DocumentStore | None = None,
    ) -> None:
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
        self._doc_store = doc_store
        self._cache: list[BddkDecisionSummary] = []
        self._cache_timestamp: float = 0.0
        self._page_errors: dict[int, str] = {}

    # -- context manager --------------------------------------------------

    async def __aenter__(self) -> "BddkApiClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        await self._http.aclose()
        logger.info("BddkApiClient session closed")

    # -- HTTP with retry --------------------------------------------------

    async def _fetch_with_retry(self, url: str) -> httpx.Response:
        """Fetch a URL with exponential backoff retry."""
        last_exc = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._http.get(url)
                response.raise_for_status()
                return response
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    wait = 2**attempt
                    logger.warning("Retry %d/%d for %s: %s", attempt + 1, _MAX_RETRIES, url, exc)
                    import asyncio

                    await asyncio.sleep(wait)
        raise last_exc  # type: ignore[misc]

    # -- cache persistence ------------------------------------------------

    def _save_cache_to_disk(self) -> None:
        """Persist cache to JSON file."""
        try:
            data = {
                "timestamp": self._cache_timestamp,
                "items": [d.model_dump() for d in self._cache],
            }
            _CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            logger.info("Cache saved to disk: %d items", len(self._cache))
        except (OSError, TypeError, ValueError) as e:
            logger.warning("Failed to save cache to disk: %s", e)

    def _load_cache_from_disk(self) -> bool:
        """Load cache from JSON file if it exists and is valid."""
        try:
            if not _CACHE_FILE.exists():
                return False
            data = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            ts = data.get("timestamp", 0)
            if (time.time() - ts) >= _CACHE_TTL_SECONDS:
                return False
            self._cache = [BddkDecisionSummary(**item) for item in data["items"]]
            self._cache_timestamp = ts
            logger.info("Cache loaded from disk: %d items", len(self._cache))
            return True
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning("Failed to load cache from disk: %s", e)
            return False

    # -- cache ------------------------------------------------------------

    def _is_cache_valid(self) -> bool:
        return len(self._cache) > 0 and (time.time() - self._cache_timestamp) < _CACHE_TTL_SECONDS

    def cache_status(self) -> dict:
        """Return cache statistics."""
        age = time.time() - self._cache_timestamp if self._cache_timestamp else None
        return {
            "total_items": len(self._cache),
            "cache_age_seconds": round(age) if age else None,
            "cache_valid": self._is_cache_valid(),
            "ttl_seconds": _CACHE_TTL_SECONDS,
            "page_errors": dict(self._page_errors),
            "categories": _count_categories(self._cache),
        }

    # -- parsers ----------------------------------------------------------

    def _extract_links(self, soup: BeautifulSoup, category: str) -> list[BddkDecisionSummary]:
        """Extract document links from a soup fragment, handling both internal and external links."""
        decisions: list[BddkDecisionSummary] = []
        seen: set[str] = set()

        for anchor in soup.find_all("a", href=True):
            href = anchor.get("href", "").strip()
            text = anchor.get_text(strip=True)

            if not text or "/Mevzuat/Detay/" in href:
                continue
            if href in seen:
                continue
            seen.add(href)

            # Internal DokumanGetir link
            doc_id_match = re.search(r"/DokumanGetir/(\d+)", href)
            if doc_id_match:
                doc_id = doc_id_match.group(1)
                source_url = f"{_BDDK_BASE_URL}{href}" if href.startswith("/") else href
                decisions.append(
                    BddkDecisionSummary(
                        title=text,
                        document_id=doc_id,
                        content=text,
                        category=category,
                        source_url=source_url,
                    )
                )
                continue

            # External link (mevzuat.gov.tr or other)
            if href.startswith("http"):
                synthetic_id = _external_url_to_id(href)
                if synthetic_id:
                    decisions.append(
                        BddkDecisionSummary(
                            title=text,
                            document_id=synthetic_id,
                            content=text,
                            category=category,
                            source_url=href,
                        )
                    )

        return decisions

    async def _fetch_and_parse_decision_page(self, list_id: int) -> list[BddkDecisionSummary]:
        """Parse pages 55/56: DokumanGetir links with (date - number) format."""
        url = f"{_BDDK_BASE_URL}/Mevzuat/Liste/{list_id}"
        response = await self._fetch_with_retry(url)

        soup = BeautifulSoup(response.text, "html.parser")
        decisions: list[BddkDecisionSummary] = []

        for link in soup.find_all("a", href=re.compile(r"/Mevzuat/DokumanGetir/\d+")):
            href = link.get("href", "")
            doc_id_match = re.search(r"/DokumanGetir/(\d+)", href)
            if not doc_id_match:
                continue

            doc_id = doc_id_match.group(1)
            raw_text = link.get_text(strip=True)
            if not raw_text:
                continue

            date_match = re.match(r"\((\d{2}\.\d{2}\.\d{4})\s*-\s*(\d+)\)\s*(.*)", raw_text)
            if date_match:
                decision_date = date_match.group(1)
                decision_number = date_match.group(2)
                title = date_match.group(3).strip()
            else:
                decision_date = ""
                decision_number = ""
                title = raw_text

            decisions.append(
                BddkDecisionSummary(
                    title=title,
                    document_id=doc_id,
                    content=title,
                    decision_date=decision_date,
                    decision_number=decision_number,
                    category="Kurul Kararı",
                    source_url=_DOCUMENT_URL_TEMPLATE.format(document_id=doc_id),
                )
            )

        logger.info("Parsed %d decisions from page %d", len(decisions), list_id)
        return decisions

    async def _fetch_and_parse_accordion_page(self, list_id: int) -> list[BddkDecisionSummary]:
        """Parse pages with accordion card structure (e.g., 50, 51)."""
        url = f"{_BDDK_BASE_URL}/Mevzuat/Liste/{list_id}"
        response = await self._fetch_with_retry(url)

        soup = BeautifulSoup(response.text, "html.parser")
        decisions: list[BddkDecisionSummary] = []

        for card in soup.find_all("div", class_="card"):
            h5 = card.find("h5")
            if not h5:
                continue

            raw_header = h5.get_text(strip=True)
            header_name = re.sub(r"\s*\(\d+\)\s*$", "", raw_header).strip()
            category = _ACCORDION_CATEGORY_MAP.get(header_name, header_name)

            body = card.find("div", class_="card-body")
            if not body:
                body = card.find("div", class_="collapse")
            if not body:
                continue

            decisions.extend(self._extract_links(body, category))

        logger.info("Parsed %d items from accordion page %d", len(decisions), list_id)
        return decisions

    async def _fetch_and_parse_flat_page(self, list_id: int) -> list[BddkDecisionSummary]:
        """Parse flat-list pages (e.g., 49, 52, 54, 58, 63)."""
        url = f"{_BDDK_BASE_URL}/Mevzuat/Liste/{list_id}"
        response = await self._fetch_with_retry(url)

        soup = BeautifulSoup(response.text, "html.parser")
        category = _FLAT_PAGE_CATEGORY.get(list_id, f"Sayfa {list_id}")
        decisions = self._extract_links(soup, category)

        logger.info("Parsed %d items from flat page %d", len(decisions), list_id)
        return decisions

    async def _ensure_cache(self) -> None:
        """Ensure the decision cache is populated and valid."""
        if self._is_cache_valid():
            return

        # Try loading from disk first
        if self._load_cache_from_disk():
            return

        logger.info("Refreshing BDDK cache...")
        all_decisions: list[BddkDecisionSummary] = []
        self._page_errors.clear()

        for list_id in _ALL_PAGE_IDS:
            try:
                if list_id in _ACCORDION_PAGE_IDS:
                    page_decisions = await self._fetch_and_parse_accordion_page(list_id)
                elif list_id in _DECISION_PAGE_IDS:
                    page_decisions = await self._fetch_and_parse_decision_page(list_id)
                else:
                    page_decisions = await self._fetch_and_parse_flat_page(list_id)
                all_decisions.extend(page_decisions)
            except (httpx.HTTPError, httpx.TransportError, ValueError, AttributeError) as e:
                self._page_errors[list_id] = str(e)
                logger.error("Failed to fetch BDDK list page %d: %s", list_id, e)

        if not all_decisions and self._page_errors:
            logger.error("All page fetches failed: %s", self._page_errors)

        self._cache = all_decisions
        self._cache_timestamp = time.time()
        self._save_cache_to_disk()
        logger.info("BDDK cache refreshed: %d total items", len(self._cache))

    async def ensure_cache(self) -> None:
        """Public wrapper for _ensure_cache."""
        await self._ensure_cache()

    # -- public API -------------------------------------------------------

    async def search_decisions(
        self,
        request: BddkSearchRequest,
    ) -> BddkSearchResult:
        """Search for BDDK decisions/regulations by keyword matching against cached list."""
        await self._ensure_cache()

        keywords_lower = _turkish_lower(request.keywords)
        keyword_parts = keywords_lower.split()
        keyword_stems = [_turkish_stem(p) for p in keyword_parts]

        # Optional category filter
        category_filter = _turkish_lower(request.category) if request.category else None

        # Optional date range filter
        date_from = _parse_date(request.date_from) if request.date_from else None
        date_to = _parse_date(request.date_to) if request.date_to else None

        matching: list[tuple[int, BddkDecisionSummary]] = []
        for dec in self._cache:
            # Category filter
            if category_filter and category_filter not in _turkish_lower(dec.category):
                continue

            # Date range filter (skip items without dates when filter is active)
            if date_from or date_to:
                if not dec.decision_date:
                    continue
                doc_date = _parse_date(dec.decision_date)
                if not doc_date:
                    continue
                if date_from and doc_date < date_from:
                    continue
                if date_to and doc_date > date_to:
                    continue

            text_lower = _turkish_lower(f"{dec.title} {dec.decision_date} {dec.decision_number} {dec.category}")
            text_words = text_lower.split()
            text_stems = [_turkish_stem(w) for w in text_words]

            # Score: exact match in title > stem match > substring match
            score = 0
            title_lower = _turkish_lower(dec.title)
            all_match = True
            for part, stem in zip(keyword_parts, keyword_stems, strict=True):
                if part in title_lower:
                    score += 3  # exact in title
                elif stem in " ".join(text_stems):
                    score += 2  # stem match
                elif part in text_lower:
                    score += 1  # substring match
                else:
                    all_match = False
                    break

            if all_match and score > 0:
                matching.append((score, dec))

        # Sort by relevance score (descending)
        matching.sort(key=lambda x: x[0], reverse=True)
        results_only = [dec for _, dec in matching]

        total_results = len(results_only)
        start_idx = (request.page - 1) * request.page_size
        end_idx = start_idx + request.page_size
        page_results = results_only[start_idx:end_idx]

        logger.info(
            "BDDK search for '%s': %d matches, returning page %d (%d items)",
            request.keywords,
            total_results,
            request.page,
            len(page_results),
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
        """Fetch a BDDK document and return its content as paginated Markdown.

        Uses store-first strategy: check local DocumentStore before fetching
        from the network. If fetched live, stores result for future use.
        """
        # ── Store-first lookup ──
        if self._doc_store:
            page = await self._doc_store.get_document_page(document_id, page_number)
            if page and page.markdown_content and "Invalid page" not in page.markdown_content:
                logger.info(
                    "Document %s served from store (page %d/%d)", document_id, page.page_number, page.total_pages
                )
                return BddkDocumentMarkdown(
                    document_id=page.document_id,
                    markdown_content=page.markdown_content,
                    page_number=page.page_number,
                    total_pages=page.total_pages,
                )

        # ── Live fetch fallback ──
        url = self._resolve_document_url(document_id)
        logger.info("Fetching BDDK document (live): %s", url)

        try:
            response = await self._fetch_with_retry(url)
        except (httpx.HTTPError, httpx.TransportError) as e:
            return BddkDocumentMarkdown(
                document_id=document_id,
                markdown_content=f"Error fetching document: {e}\nSource URL: {url}",
                page_number=1,
                total_pages=1,
            )

        content_type = response.headers.get("content-type", "").lower()
        ext = ".pdf" if "pdf" in content_type else ".html"

        try:
            result = self._md.convert_stream(
                io.BytesIO(response.content),
                file_extension=ext,
            )
            markdown = result.text_content.strip()
        except (ValueError, OSError, UnicodeDecodeError) as e:
            return BddkDocumentMarkdown(
                document_id=document_id,
                markdown_content=f"Error converting document to Markdown: {e}\nSource URL: {url}",
                page_number=1,
                total_pages=1,
            )

        # ── Store for future use ──
        if self._doc_store and markdown:
            from doc_store import StoredDocument

            try:
                await self._doc_store.store_document(
                    StoredDocument(
                        document_id=document_id,
                        title=document_id,
                        source_url=url,
                        markdown_content=markdown,
                        extraction_method="markitdown",
                        file_size=len(response.content),
                        pdf_bytes=response.content if ext == ".pdf" else None,
                    )
                )
                logger.info("Stored document %s in local store", document_id)
            except (RuntimeError, OSError) as e:
                logger.warning("Failed to store document %s: %s", document_id, e)

        total_pages = max(1, math.ceil(len(markdown) / _CHUNK_SIZE))

        if page_number < 1 or page_number > total_pages:
            return BddkDocumentMarkdown(
                document_id=document_id,
                markdown_content=f"Invalid page number {page_number}. Document has {total_pages} page(s).",
                page_number=page_number,
                total_pages=total_pages,
            )

        start = (page_number - 1) * _CHUNK_SIZE
        page_content = markdown[start : start + _CHUNK_SIZE]

        return BddkDocumentMarkdown(
            document_id=document_id,
            markdown_content=page_content,
            page_number=page_number,
            total_pages=total_pages,
        )


def _count_categories(cache: list[BddkDecisionSummary]) -> dict[str, int]:
    """Count documents per category."""
    counts: dict[str, int] = {}
    for d in cache:
        cat = d.category or "Unknown"
        counts[cat] = counts.get(cat, 0) + 1
    return dict(sorted(counts.items()))
