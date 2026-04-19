import asyncio
import io
import logging
import math
import re
import time
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import asyncpg
import httpx
from bs4 import BeautifulSoup
from markitdown import MarkItDown

from config import (
    CACHE_TTL_SECONDS,
    HTTP_CONNECT_TIMEOUT,
    HTTP_POOL_TIMEOUT,
    PAGE_SIZE,
    REQUEST_TIMEOUT,
    STALE_CACHE_FALLBACK,
    SYNC_CONCURRENCY,
)
from doc_store import DocumentStore
from models import (
    BddkDecisionSummary,
    BddkDocumentMarkdown,
    BddkSearchRequest,
    BddkSearchResult,
)
from utils import MEVZUAT_TUR_MAP, fetch_with_retry

logger = logging.getLogger(__name__)

_DOCUMENT_URL_TEMPLATE = "https://www.bddk.org.tr/Mevzuat/DokumanGetir/{document_id}"
_BDDK_BASE_URL = "https://www.bddk.org.tr"

# Pages that use accordion card structure (h5 headers with card-body)
_ACCORDION_PAGE_IDS = [50, 51]
# Pages that use flat list with DokumanGetir links and (date - number) format
# Page 55 (Kurul Kararları) excluded by default — mostly faaliyet/kuruluş izni decisions; revisit with whitelist if specific kararlar are needed.
_DECISION_PAGE_IDS = [56]
# Pages that use flat list with mixed link types (no date format)
# Page 52 (Finansal Kiralama / Faktoring / Finansman / Tasarruf Finansman — non-bank) excluded — out of bank scope.
_FLAT_PAGE_IDS = [49, 54, 58, 63]

_ALL_PAGE_IDS = _ACCORDION_PAGE_IDS + _DECISION_PAGE_IDS + _FLAT_PAGE_IDS

# Post-scrape filters: drop items not relevant to a conventional commercial bank.
_EXCLUDED_CATEGORIES: set[str] = {"Faizsiz Bankacılık"}
# Substring matched against decision title; case-insensitive.
_EXCLUDED_TITLE_SUBSTRINGS: tuple[str, ...] = (
    "6361 sayılı",  # Finansal Kiralama, Faktoring, Finansman ve Tasarruf Finansman Şirketleri Kanunu
)


def _is_in_scope(dec: BddkDecisionSummary) -> bool:
    """True iff the decision should be kept (not in any exclusion list)."""
    if dec.category in _EXCLUDED_CATEGORIES:
        return False
    title_lower = (dec.title or "").lower()
    return not any(s.lower() in title_lower for s in _EXCLUDED_TITLE_SUBSTRINGS)

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
    54: "BDDK Düzenlemesi",
    58: "Düzenleme Taslağı",
    63: "Mülga Düzenleme",
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
    """Parse DD.MM.YYYY or DD/MM/YYYY date string to datetime."""
    for fmt in ("%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    return None


def _external_url_to_id(url: str) -> str | None:
    """Generate a stable synthetic ID from an external URL."""
    parsed = urlparse(url)
    if "mevzuat.gov.tr" in parsed.netloc:
        params = parse_qs(parsed.query)
        mevzuat_no = params.get("MevzuatNo", [None])[0]
        if mevzuat_no:
            return f"mevzuat_{mevzuat_no}"
        mevzuat_kod = params.get("MevzuatKod", [None])[0]
        if mevzuat_kod:
            parts = mevzuat_kod.split(".")
            if len(parts) >= 3:
                return f"mevzuat_{parts[-1]}"
    return None


def _mevzuat_to_pdf_url(mevzuat_no: str, mevzuat_tur: str = "7", mevzuat_tertip: str = "5") -> str | None:
    """Convert mevzuat.gov.tr parameters to a direct PDF download URL."""
    path_segment = MEVZUAT_TUR_MAP.get(mevzuat_tur)
    if not path_segment:
        return None
    return f"https://www.mevzuat.gov.tr/MevzuatMetin/{path_segment}/{mevzuat_tur}.{mevzuat_tertip}.{mevzuat_no}.pdf"


# -- Decision cache schema (PostgreSQL) ----------------------------------------

_CACHE_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS decision_cache (
    document_id       TEXT PRIMARY KEY,
    title             TEXT NOT NULL DEFAULT '',
    content           TEXT DEFAULT '',
    decision_date     TEXT DEFAULT '',
    decision_number   TEXT DEFAULT '',
    category          TEXT DEFAULT '',
    source_url        TEXT DEFAULT '',
    cached_at         DOUBLE PRECISION NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decision_cache_category ON decision_cache(category);
"""


class BddkApiClient:
    """
    Client for searching and retrieving BDDK decisions and regulations.

    Scrapes BDDK website, caches document lists in PostgreSQL for instant
    startup across server restarts.
    """

    def __init__(
        self,
        pool: asyncpg.Pool,
        request_timeout: float = REQUEST_TIMEOUT,
        doc_store: DocumentStore | None = None,
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self._pool = pool
        self._owns_http = http is None
        if http is not None:
            self._http = http
        else:
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
                timeout=httpx.Timeout(
                    request_timeout,
                    connect=HTTP_CONNECT_TIMEOUT,
                    pool=HTTP_POOL_TIMEOUT,
                ),
                follow_redirects=True,
            )
        self._md = MarkItDown()
        self._doc_store = doc_store
        self._cache: list[BddkDecisionSummary] = []
        self._cache_timestamp: float = 0.0
        self._page_errors: dict[int, str] = {}
        self.known_announcements: set[str] = set()
        # Bound the parallel _scrape_bddk() fan-out so BDDK sees at most
        # SYNC_CONCURRENCY (default 5) in-flight requests at once, matching
        # the rate limit data_sources.py already enforces for the same host.
        self._scrape_sem = asyncio.Semaphore(SYNC_CONCURRENCY)

    async def initialize(self) -> None:
        """Create cache table and load existing cache from PostgreSQL."""
        async with self._pool.acquire() as conn:
            await conn.execute(_CACHE_SCHEMA_SQL)
        # Eagerly load from DB — instant, no network needed
        await self._load_cache_from_db()

    # -- context manager ------------------------------------------------------

    async def __aenter__(self) -> "BddkApiClient":
        await self.initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._owns_http:
            await self._http.aclose()
            logger.info("BddkApiClient session closed")

    # -- HTTP with retry ------------------------------------------------------

    async def _fetch_with_retry(self, url: str) -> httpx.Response:
        """Fetch a URL with exponential backoff retry."""
        return await fetch_with_retry(self._http, url)

    # -- cache persistence (PostgreSQL) ----------------------------------------

    async def _save_cache_to_db(self) -> None:
        """Persist cache to PostgreSQL using upsert (no DELETE ALL)."""
        try:
            async with self._pool.acquire() as conn:
                now = time.time()
                args_list = [
                    (
                        d.document_id,
                        d.title,
                        d.content,
                        d.decision_date,
                        d.decision_number,
                        d.category,
                        d.source_url or "",
                        now,
                    )
                    for d in self._cache
                ]
                await conn.executemany(
                    """
                    INSERT INTO decision_cache
                        (document_id, title, content, decision_date, decision_number,
                         category, source_url, cached_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT(document_id) DO UPDATE SET
                        title=EXCLUDED.title, content=EXCLUDED.content,
                        decision_date=EXCLUDED.decision_date,
                        decision_number=EXCLUDED.decision_number,
                        category=EXCLUDED.category, source_url=EXCLUDED.source_url,
                        cached_at=EXCLUDED.cached_at
                    """,
                    args_list,
                )
                self._cache_timestamp = now
                logger.debug("Cache saved to PostgreSQL: %d items", len(self._cache))
        except (asyncpg.PostgresError, OSError) as e:
            logger.error("Failed to save cache to PostgreSQL: %s", e)

    async def _load_cache_from_db(self) -> bool:
        """Load cache from PostgreSQL. Always loads regardless of TTL."""
        try:
            rows = await self._pool.fetch(
                "SELECT document_id, title, content, decision_date, decision_number, "
                "category, source_url, cached_at FROM decision_cache"
            )
            if not rows:
                return False
            loaded = [
                BddkDecisionSummary(
                    document_id=row["document_id"],
                    title=row["title"],
                    content=row["content"] or "",
                    decision_date=row["decision_date"] or "",
                    decision_number=row["decision_number"] or "",
                    category=row["category"] or "",
                    source_url=row["source_url"] or "",
                )
                for row in rows
            ]
            # Apply scope filter on every load — keeps stale DB items hidden after filter rules change.
            self._cache = [dec for dec in loaded if _is_in_scope(dec)]
            # Use the most recent cached_at as timestamp
            self._cache_timestamp = max(row["cached_at"] for row in rows)
            logger.info("Cache loaded from PostgreSQL: %d items", len(self._cache))
            return True
        except (asyncpg.PostgresError, OSError) as e:
            logger.warning("Failed to load cache from DB: %s", e)
            return False

    # -- cache ----------------------------------------------------------------

    def _is_cache_valid(self) -> bool:
        return len(self._cache) > 0 and (time.time() - self._cache_timestamp) < CACHE_TTL_SECONDS

    def cache_status(self) -> dict:
        """Return cache statistics."""
        age = time.time() - self._cache_timestamp if self._cache_timestamp else None
        return {
            "total_items": len(self._cache),
            "cache_age_seconds": round(age) if age else None,
            "cache_valid": self._is_cache_valid(),
            "ttl_seconds": CACHE_TTL_SECONDS,
            "page_errors": dict(self._page_errors),
            "categories": _count_categories(self._cache),
        }

    def get_cache_items(self) -> list[BddkDecisionSummary]:
        """Return a shallow copy of the cached decision list."""
        return list(self._cache)

    def find_by_id(self, doc_id: str) -> BddkDecisionSummary | None:
        """Find a cached decision by document_id. Returns None if not found."""
        for dec in self._cache:
            if dec.document_id == doc_id:
                return dec
        return None

    def cache_size(self) -> int:
        """Return the number of items in the cache."""
        return len(self._cache)

    # -- parsers --------------------------------------------------------------

    def _extract_links(self, soup: BeautifulSoup, category: str) -> list[BddkDecisionSummary]:
        """Extract document links from a soup fragment."""
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

            date_match = re.match(r"\((\d{2}[./]\d{2}[./]\d{4})\s*[-–—]\s*(\d+)\)\s*(.*)", raw_text)
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
            if header_name not in _ACCORDION_CATEGORY_MAP:
                logger.warning(
                    "Unmapped accordion category '%s' on page %d -- using raw text as category",
                    header_name,
                    list_id,
                )

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
        """Ensure the decision cache is populated.

        Serves from in-memory cache or PostgreSQL without hitting BDDK.
        Only scrapes BDDK if the database is completely empty.
        Use refresh_cache() for an explicit BDDK refresh.
        """
        if self._cache:
            return

        # Try loading from DB — no network needed
        if await self._load_cache_from_db():
            logger.info("Serving %d items from PostgreSQL cache", len(self._cache))
            return

        # DB is empty — must scrape BDDK for initial population
        logger.info("PostgreSQL cache empty — initial scrape from BDDK...")
        await self._scrape_bddk()

    async def _scrape_bddk(self) -> None:
        """Scrape all BDDK pages in parallel and persist to PostgreSQL.

        Previously serialised across _ALL_PAGE_IDS (9 pages × ~1–2s per
        request on a cold cache). asyncio.gather() fans them out so the
        full refresh completes in roughly the slowest single page.
        """
        self._page_errors.clear()

        async def _fetch_page_safe(list_id: int) -> list[BddkDecisionSummary]:
            async with self._scrape_sem:
                try:
                    if list_id in _ACCORDION_PAGE_IDS:
                        return await self._fetch_and_parse_accordion_page(list_id)
                    if list_id in _DECISION_PAGE_IDS:
                        return await self._fetch_and_parse_decision_page(list_id)
                    return await self._fetch_and_parse_flat_page(list_id)
                except (httpx.HTTPError, httpx.TransportError, ValueError, AttributeError) as e:
                    self._page_errors[list_id] = str(e)
                    logger.error("Failed to fetch BDDK list page %d: %s", list_id, e)
                    return []

        page_results = await asyncio.gather(*(_fetch_page_safe(pid) for pid in _ALL_PAGE_IDS))
        scraped: list[BddkDecisionSummary] = [dec for page in page_results for dec in page]
        all_decisions = [dec for dec in scraped if _is_in_scope(dec)]
        dropped = len(scraped) - len(all_decisions)
        if dropped:
            logger.info("Scope filter dropped %d/%d items", dropped, len(scraped))

        if not all_decisions and self._page_errors:
            logger.error("All page fetches failed: %s", self._page_errors)
            if STALE_CACHE_FALLBACK and self._cache:
                logger.warning("Serving stale DB cache (%d items) — BDDK unreachable", len(self._cache))
                return
            return

        self._cache = all_decisions
        self._cache_timestamp = time.time()
        await self._save_cache_to_db()
        logger.info("BDDK cache refreshed: %d total items", len(self._cache))

    async def ensure_cache(self) -> None:
        """Public wrapper for _ensure_cache."""
        await self._ensure_cache()

    async def refresh_cache(self) -> int:
        """Force re-scrape BDDK and update PostgreSQL. Returns new item count."""
        logger.info("Force-refreshing BDDK cache from live site...")
        await self._scrape_bddk()
        return len(self._cache)

    # -- public API -----------------------------------------------------------

    async def search_decisions(
        self,
        request: BddkSearchRequest,
    ) -> BddkSearchResult:
        """Search for BDDK decisions/regulations by keyword matching against cached list."""
        await self._ensure_cache()

        keywords_lower = _turkish_lower(request.keywords)
        keyword_parts = keywords_lower.split()
        keyword_stems = [_turkish_stem(p) for p in keyword_parts]

        category_filter = _turkish_lower(request.category) if request.category else None
        date_from = _parse_date(request.date_from) if request.date_from else None
        date_to = _parse_date(request.date_to) if request.date_to else None

        matching: list[tuple[int, BddkDecisionSummary]] = []
        for dec in self._cache:
            if category_filter and category_filter not in _turkish_lower(dec.category):
                continue

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

            score = 0
            title_lower = _turkish_lower(dec.title)
            all_match = True
            for part, stem in zip(keyword_parts, keyword_stems, strict=True):
                if part in title_lower:
                    score += 3
                elif stem in " ".join(text_stems):
                    score += 2
                elif part in text_lower:
                    score += 1
                else:
                    all_match = False
                    break

            if all_match and score > 0:
                matching.append((score, dec))

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
        if document_id.isdigit():
            return _DOCUMENT_URL_TEMPLATE.format(document_id=document_id)

        if document_id.startswith("mevzuat_"):
            mevzuat_no = document_id.removeprefix("mevzuat_")
            for dec in self._cache:
                if dec.document_id == document_id and dec.source_url:
                    source = dec.source_url
                    parsed = urlparse(source)
                    params = parse_qs(parsed.query)

                    tur = params.get("MevzuatTur", [None])[0]
                    tertip = params.get("MevzuatTertip", [None])[0]

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

            pdf_url = _mevzuat_to_pdf_url(mevzuat_no)
            if pdf_url:
                return pdf_url

        return _DOCUMENT_URL_TEMPLATE.format(document_id=document_id)

    async def get_document_markdown(
        self,
        document_id: str,
        page_number: int = 1,
    ) -> BddkDocumentMarkdown:
        """Fetch a BDDK document and return its content as paginated Markdown.

        Uses store-first strategy: check local DocumentStore before fetching
        from the network.
        """
        # Store-first lookup
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

        # Live fetch fallback
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

        # Store for future use
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

        total_pages = max(1, math.ceil(len(markdown) / PAGE_SIZE))

        if page_number < 1 or page_number > total_pages:
            return BddkDocumentMarkdown(
                document_id=document_id,
                markdown_content=f"Invalid page number {page_number}. Document has {total_pages} page(s).",
                page_number=page_number,
                total_pages=total_pages,
            )

        start = (page_number - 1) * PAGE_SIZE
        page_content = markdown[start : start + PAGE_SIZE]

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
