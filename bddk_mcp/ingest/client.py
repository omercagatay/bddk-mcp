import asyncio
import logging
import re
import sqlite3
import time
from datetime import datetime
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup
from markitdown import MarkItDown

from bddk_mcp.core.config import (
    HTTP_CONNECT_TIMEOUT,
    HTTP_POOL_TIMEOUT,
    REQUEST_TIMEOUT,
)
from bddk_mcp.core.exceptions import BddkStorageError, BddkUpstreamError
from bddk_mcp.core.models import (
    BddkDecisionSummary,
    BddkSearchRequest,
    BddkSearchResult,
)
from bddk_mcp.core.outbound_http import (
    BDDK_HTTPS_HOSTS,
    MEVZUAT_HTTPS_HOSTS,
    OutboundHttpPolicyError,
    assert_public_https_resolution,
    bounded_request_with_retry,
)
from bddk_mcp.store.doc_store import DocumentStore

logger = logging.getLogger(__name__)

MEVZUAT_TUR_MAP: dict[str, str] = {
    "1": "kanun",
    "2": "kanunhukmundekararname",
    "4": "cumhurbaskanligikararnamesi",
    "5": "tuzuk",
    "7": "yonetmelik",
    "9": "teblig",
    "11": "cumhurbaskanligikararnamesi",
}

_DOCUMENT_URL_TEMPLATE = "https://www.bddk.org.tr/Mevzuat/DokumanGetir/{document_id}"
_BDDK_BASE_URL = "https://www.bddk.org.tr"

# Code-owned limits keep an operator-triggered refresh or live fallback from
# buffering an unbounded upstream response. Catalog pages should be small;
# document fallback permits large regulatory PDFs while remaining finite.
_MAX_CATALOG_RESPONSE_BYTES = 8 * 1024 * 1024
# Parallel catalog-page requests BDDK sees at once.
_SCRAPE_CONCURRENCY = 5
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


def _canonicalize_catalog(decisions: list[BddkDecisionSummary]) -> list[BddkDecisionSummary]:
    """Return one deterministic row per upstream document identifier.

    Exact duplicates can legitimately be linked from more than one list page,
    but conflicting metadata for the same identifier is ambiguous and must not
    be silently resolved by a last-write-wins upsert.
    """

    by_id: dict[str, BddkDecisionSummary] = {}
    for decision in decisions:
        existing = by_id.get(decision.document_id)
        if existing is None:
            by_id[decision.document_id] = decision
        elif existing != decision:
            raise BddkUpstreamError(
                "BDDK catalog refresh contained conflicting records for one document; "
                "the last known-good cache was retained."
            )
    return list(by_id.values())


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


# Catalog-query words that carry no title content ("8970 sayılı", "... hakkında").
# They are dropped before the all-token match.
_CATALOG_FUNCTION_WORDS = frozenset(
    {
        "sayılı",
        "sayili",
        "hakkında",
        "hakkinda",
        "hk",
        "ilişkin",
        "iliskin",
        "dair",
        "göre",
        "gore",
        "ile",
        "ve",
        "veya",
        "nolu",
        "no",
        "numaralı",
        "numarali",
        "tarihli",
    }
)


# Document-type words with their case, possessive and plural inflections, with or
# without Turkish diacritics. They never have to occur literally: a type word is
# satisfied by the title or the category and otherwise only affects ranking.
# Only an explicit suffix set in the noun's own vowel harmony is accepted, so
# derived words are content words: "kanuni"/"kanunen" (adjective/adverb),
# "kararname", "kararlaştırılan", "kararlı", "rehberlik", "tebligat".
def _type_noun_pattern(hard: str, soft: str, *, back: bool, rounded: bool = False) -> str:
    """Regex for a document-type noun and its inflections.

    ``hard`` is the bare stem, ``soft`` the stem before a vowel suffix (k/ğ
    alternation); a vowel-final stem passes ``soft=""``. Suffix vowels follow
    the noun's harmony; ``[ıi]`` keeps unaccented spellings ("karari").
    """
    a = "a" if back else "e"
    i = ("u" if rounded else "[ıi]") if back else ("ü" if rounded else "i")
    plural_i = "[ıi]" if back else "i"

    def case_after(vowel: str) -> str:
        # After a possessive vowel: -n (genitive), -nı, -nın, -na, -nda, -ndan, -yla.
        return f"(?:n(?:{vowel}n?|{a}|d{a}n?)?|yl{a})?"

    plural = f"l{a}r(?:{plural_i}{case_after(plural_i)}|{plural_i}n|{a}|d{a}n?|l{a})?"
    if not soft:
        suffixes = f"(?:s{i}{case_after(i)}|n{i}n|y{a}|d{a}n?|y?l{a}|{plural})?"
        return f"{hard}{suffixes}"
    return f"(?:{hard}(?:[dt]{a}n?|l{a}|{plural})?|{soft}(?:{i}{case_after(i)}|{a}))"


_DOCUMENT_TYPE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        _type_noun_pattern("y[öo]netmelik", "y[öo]netmeli[ğg]", back=False),
        _type_noun_pattern("tebli[ğg]", "tebli[ğg]", back=False),
        _type_noun_pattern("genelge", "", back=False),
        _type_noun_pattern("rehber", "rehber", back=False),
        _type_noun_pattern("karar", "karar", back=True),
        _type_noun_pattern("kanun", "kanun", back=True, rounded=True),
        _type_noun_pattern("taslak", "tasla[ğgk]", back=True),
    )
)

# A content token shares a stem with a title word when both are at least this
# long and their common prefix covers all but the last few characters of the
# shorter word (Turkish case/possessive suffixes).
_STEM_MIN_LENGTH = 5
_STEM_MAX_SUFFIX = 4

_TOKEN_EDGE_PUNCTUATION = ".,;:()[]\"'’!?"


def _catalog_tokens(text: str) -> list[str]:
    """Turkish-lowercased whitespace tokens without surrounding punctuation."""
    tokens = (token.strip(_TOKEN_EDGE_PUNCTUATION) for token in _turkish_lower(text).split())
    return [token for token in tokens if token]


def catalog_keyword_tokens(keywords: str) -> list[str]:
    """The tokens search_decisions matches for ``keywords``; empty means browse mode."""
    return _catalog_tokens(keywords)


def _document_type_pattern(token: str) -> re.Pattern[str] | None:
    for pattern in _DOCUMENT_TYPE_PATTERNS:
        if pattern.fullmatch(token):
            return pattern
    return None


def _shares_stem(token: str, word: str) -> bool:
    """Conservative inflection-tolerant comparison of two lowercased words."""
    shorter = min(len(token), len(word))
    if shorter < _STEM_MIN_LENGTH:
        return False
    prefix = 0
    for left, right in zip(token, word, strict=False):
        if left != right:
            break
        prefix += 1
    return prefix >= _STEM_MIN_LENGTH and prefix >= shorter - _STEM_MAX_SUFFIX


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
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").rstrip(".").lower()
    except ValueError:
        return None
    if host in MEVZUAT_HTTPS_HOSTS and parsed.username is None and parsed.password is None:
        params = parse_qs(parsed.query)
        mevzuat_no = params.get("MevzuatNo", [None])[0]
        if mevzuat_no and re.fullmatch(r"[0-9]{1,20}", mevzuat_no):
            return f"mevzuat_{mevzuat_no}"
        mevzuat_kod = params.get("MevzuatKod", [None])[0]
        if mevzuat_kod:
            parts = mevzuat_kod.split(".")
            if len(parts) >= 3 and re.fullmatch(r"[0-9]{1,20}", parts[-1]):
                return f"mevzuat_{parts[-1]}"
    return None


class BddkApiClient:
    """
    Client for searching and retrieving BDDK decisions and regulations.

    Scrapes the BDDK website and keeps the catalog in the corpus file for
    instant startup across server restarts.
    """

    def __init__(
        self,
        store: DocumentStore,
        request_timeout: float = REQUEST_TIMEOUT,
        http: httpx.AsyncClient | None = None,
        allow_live_population: bool = False,
    ) -> None:
        self._store = store
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
                follow_redirects=False,
            )
        self._md = MarkItDown()
        self._allow_live_population = allow_live_population
        self._cache: list[BddkDecisionSummary] = []
        self._cache_timestamp: float = 0.0
        self._page_errors: dict[int, str] = {}
        # Bound the parallel _scrape_bddk() fan-out so BDDK sees at most
        # _SCRAPE_CONCURRENCY in-flight requests at once, matching
        # the rate limit data_sources.py already enforces for the same host.
        self._scrape_sem = asyncio.Semaphore(_SCRAPE_CONCURRENCY)

    # -- context manager ------------------------------------------------------

    async def __aenter__(self) -> "BddkApiClient":
        # A client context may read a prepared cache but must never create or
        # repair its schema implicitly.
        await self.load_cache_read_only(require_nonempty=False)
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._owns_http:
            await self._http.aclose()
            logger.info("BddkApiClient session closed")

    # -- HTTP with retry ------------------------------------------------------

    async def _assert_public_resolution(self, url: str) -> None:
        """Compatibility seam for exact-host public-address validation."""

        await assert_public_https_resolution(url)

    async def _fetch_with_retry(
        self,
        url: str,
        *,
        max_bytes: int = _MAX_CATALOG_RESPONSE_BYTES,
    ) -> httpx.Response:
        """Fetch one approved BDDK/mevzuat URL with bounded safe retries."""

        return await bounded_request_with_retry(
            self._http,
            "GET",
            url,
            base_url=f"{_BDDK_BASE_URL}/",
            allowed_hosts=BDDK_HTTPS_HOSTS | MEVZUAT_HTTPS_HOSTS,
            boundary_name="BDDK or mevzuat",
            max_bytes=max_bytes,
            resolve=self._assert_public_resolution,
        )

    # -- cache persistence (corpus file) ----------------------------------------

    async def _save_cache_to_db(self) -> None:
        """Atomically publish the complete validated upstream catalog membership."""
        if not self._cache:
            raise BddkStorageError("An empty decision catalog cannot replace the last known-good publication.")
        try:
            now = time.time()
            await self._store.replace_catalog([decision.model_dump() for decision in self._cache], cached_at=now)
        except (sqlite3.Error, OSError, RuntimeError) as exc:
            logger.error("Failed to save decision cache", extra={"error_type": type(exc).__name__})
            raise BddkStorageError("Decision cache could not be persisted safely.") from None
        self._cache_timestamp = now
        logger.debug("Cache saved to the corpus: %d items", len(self._cache))

    async def load_cache_read_only(self, *, require_nonempty: bool = True) -> int:
        """Load the decision cache with one SELECT and no repair or live fetch.

        Raises a sanitized :class:`BddkStorageError` when serving prerequisites
        are absent.  The error deliberately excludes driver text, SQL, paths,
        and connection details.
        """
        try:
            rows = await self._store.load_catalog()
        except (sqlite3.Error, OSError, RuntimeError):
            raise BddkStorageError(
                "Decision cache could not be read from the corpus file. Run `bddk-mcp build` before starting "
                "the server."
            ) from None

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
        # Apply scope filtering on every load so stale excluded rows cannot
        # become visible after filtering rules change.
        self._cache = [decision for decision in loaded if _is_in_scope(decision)]
        self._cache_timestamp = max((row["cached_at"] for row in rows), default=0.0)

        if require_nonempty and not self._cache:
            raise BddkStorageError(
                "Decision cache is empty or contains no in-scope documents. Run `bddk-mcp build` with the "
                "reviewed seed corpus before starting the server."
            )

        if self._cache:
            logger.info("Cache loaded from the corpus: %d items", len(self._cache))
        return len(self._cache)

    async def _load_cache_from_db(self) -> bool:
        """Best-effort compatibility wrapper for legacy auto-population paths."""
        try:
            return (await self.load_cache_read_only(require_nonempty=False)) > 0
        except BddkStorageError:
            logger.warning("Decision cache could not be loaded from the corpus")
            return False

    # -- cache ----------------------------------------------------------------

    def get_cache_items(self) -> list[BddkDecisionSummary]:
        """Return a shallow copy of the cached decision list."""
        return list(self._cache)

    def find_by_id(self, doc_id: str) -> BddkDecisionSummary | None:
        """Find a cached decision by document_id. Returns None if not found."""
        for dec in self._cache:
            if dec.document_id == doc_id:
                return dec
        return None

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

            date_match = re.match(r"\((\d{2}[./]\d{2}[./]\d{4})\s*[-–—]\s*(\d+)\s*\)\s*(.*)", raw_text)
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
                    "Unmapped accordion category on page %d; using upstream text as category (chars=%d)",
                    list_id,
                    len(header_name),
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

        Serves from the in-memory cache or the corpus file without hitting BDDK.
        Only scrapes BDDK if the database is completely empty.
        Use refresh_cache() for an explicit BDDK refresh.
        """
        if self._cache:
            return

        # Try loading from DB — no network needed
        if await self._load_cache_from_db():
            logger.info("Serving %d items from the corpus catalog", len(self._cache))
            return

        if not self._allow_live_population:
            raise BddkStorageError(
                "Decision cache became unavailable while the server was running. "
                "Rebuild the corpus with `bddk-mcp build`; serving mode will not populate it from the network."
            )

        # DB is empty — must scrape BDDK for initial population
        logger.info("Corpus catalog empty — initial scrape from BDDK...")
        await self._scrape_bddk()

    async def _scrape_bddk(self) -> int:
        """Atomically refresh all expected BDDK pages and return item count.

        Previously serialised across _ALL_PAGE_IDS (9 pages × ~1–2s per
        request on a cold cache). asyncio.gather() fans them out so the
        full refresh completes in roughly the slowest single page.

        The previous in-memory and persisted catalog remains authoritative
        unless every configured source page is fetched and parsed and the new
        catalog is persisted.  This prevents a partial upstream outage or an
        HTML layout change from being reported as a successful refresh.
        """
        self._page_errors.clear()

        async def _fetch_page_safe(list_id: int) -> list[BddkDecisionSummary]:
            async with self._scrape_sem:
                try:
                    if list_id in _ACCORDION_PAGE_IDS:
                        decisions = await self._fetch_and_parse_accordion_page(list_id)
                    elif list_id in _DECISION_PAGE_IDS:
                        decisions = await self._fetch_and_parse_decision_page(list_id)
                    else:
                        decisions = await self._fetch_and_parse_flat_page(list_id)
                    if not decisions:
                        self._page_errors[list_id] = "empty_page"
                        logger.error("BDDK list page %d returned no parseable items", list_id)
                    return decisions
                except (
                    httpx.HTTPError,
                    httpx.TransportError,
                    OutboundHttpPolicyError,
                    ValueError,
                    AttributeError,
                ) as exc:
                    self._page_errors[list_id] = type(exc).__name__
                    logger.error(
                        "Failed to fetch BDDK list page %d",
                        list_id,
                        extra={"error_type": type(exc).__name__},
                    )
                    return []

        page_results = await asyncio.gather(*(_fetch_page_safe(pid) for pid in _ALL_PAGE_IDS))
        scraped: list[BddkDecisionSummary] = [dec for page in page_results for dec in page]
        in_scope = [dec for dec in scraped if _is_in_scope(dec)]
        all_decisions = _canonicalize_catalog(in_scope)
        scope_dropped = len(scraped) - len(in_scope)
        duplicate_links = len(in_scope) - len(all_decisions)
        if scope_dropped:
            logger.info("Scope filter dropped %d/%d items", scope_dropped, len(scraped))
        if duplicate_links:
            logger.info("Catalog canonicalization collapsed %d duplicate links", duplicate_links)

        if self._page_errors:
            logger.error(
                "BDDK catalog refresh rejected because %d/%d source pages failed validation",
                len(self._page_errors),
                len(_ALL_PAGE_IDS),
            )
            if self._cache:
                logger.warning("Retaining last known-good decision cache (%d items)", len(self._cache))
            raise BddkUpstreamError("BDDK catalog refresh was incomplete; the last known-good cache was retained.")

        if not all_decisions:
            raise BddkUpstreamError("BDDK catalog refresh returned no in-scope documents.")

        previous_cache = self._cache
        previous_timestamp = self._cache_timestamp
        self._cache = all_decisions
        try:
            await self._save_cache_to_db()
        except BddkStorageError:
            self._cache = previous_cache
            self._cache_timestamp = previous_timestamp
            raise
        logger.info("BDDK cache refreshed: %d total items", len(self._cache))
        return len(self._cache)

    async def ensure_cache(self) -> None:
        """Public wrapper for _ensure_cache."""
        await self._ensure_cache()

    async def refresh_cache(self) -> int:
        """Force re-scrape BDDK and update the corpus catalog. Returns new item count."""
        logger.info("Force-refreshing BDDK cache from live site...")
        return await self._scrape_bddk()

    # -- public API -----------------------------------------------------------

    async def search_decisions(
        self,
        request: BddkSearchRequest,
    ) -> BddkSearchResult:
        """Search the corpus catalog by title, category, date and decision number.

        Function words ("sayılı", "hakkında", "ilişkin", ...) are dropped.
        Document-type words ("yönetmeliği", "tebliğ", "kararı", ...) are never
        required literally. Every remaining content token must match the title
        (exactly, or by a shared stem of at least five characters) or the
        category/date/number. Ranking: a numeric token equal to a decision
        number first; then the number of content tokens found literally in the
        title; then the requested document type (named by title and category
        counts most); then the overall content score. Empty keywords list the filtered catalog (browse
        mode), newest decision date first.
        """
        await self._ensure_cache()

        keyword_parts = _catalog_tokens(request.keywords)
        content_tokens: list[str] = []
        type_patterns: list[re.Pattern[str]] = []
        for part in keyword_parts:
            if part in _CATALOG_FUNCTION_WORDS:
                continue
            pattern = _document_type_pattern(part)
            if pattern is not None:
                type_patterns.append(pattern)
            else:
                content_tokens.append(part)
        if keyword_parts and not content_tokens and not type_patterns:
            # Only function words: match them literally rather than everything.
            content_tokens = keyword_parts
        browse = not keyword_parts

        category_filter = _turkish_lower(request.category) if request.category else None
        date_from = _parse_date(request.date_from) if request.date_from else None
        date_to = _parse_date(request.date_to) if request.date_to else None

        matching: list[tuple[tuple[int, int, int, int], BddkDecisionSummary]] = []
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

            if browse:
                matching.append(((0, 0, 0, 0), dec))
                continue

            text_lower = _turkish_lower(f"{dec.title} {dec.decision_date} {dec.decision_number} {dec.category}")
            text_stems = " ".join(_turkish_stem(w) for w in text_lower.split())
            title_lower = _turkish_lower(dec.title)
            title_words = _catalog_tokens(dec.title)
            # Type-word title words ("Kanunu", "Rehberi", "Kararı") are matched only by the type
            # patterns: a derived content word ("kanuni", "rehberlik", "kararname") must not reach
            # them through the shared-stem rule.
            stem_words = [word for word in title_words if _document_type_pattern(word) is None]
            category_lower = _turkish_lower(dec.category)

            content_score = 0
            exact_title_hits = 0
            number_hit = 0
            all_match = True
            for part in content_tokens:
                if part in title_lower:
                    content_score += 3
                    exact_title_hits += 1
                elif any(_shares_stem(part, word) for word in stem_words):
                    content_score += 2
                elif _turkish_stem(part) in text_stems:
                    content_score += 2
                elif part in text_lower:
                    content_score += 1
                else:
                    all_match = False
                    break
                if part.isdigit() and part == dec.decision_number.strip():
                    number_hit = 1
            if not all_match:
                continue

            # A type word named by both the title and the category counts twice.
            category_words = _catalog_tokens(category_lower)
            type_hits = sum(
                any(pattern.fullmatch(word) for word in title_words)
                + any(pattern.fullmatch(word) for word in category_words)
                for pattern in type_patterns
            )
            if content_score == 0 and type_hits == 0:
                continue
            # Decision number first, then content words found literally in the title, then the
            # requested document type, then the overall content score (stem/category matches).
            matching.append(((number_hit, exact_title_hits, type_hits, content_score), dec))

        if browse:
            # Newest decision date first; undated entries keep catalog order at the end.
            dated = [(d, dec) for _, dec in matching if (d := _parse_date(dec.decision_date)) is not None]
            undated = [dec for _, dec in matching if _parse_date(dec.decision_date) is None]
            dated.sort(key=lambda item: item[0], reverse=True)
            results_only = [dec for _, dec in dated] + undated
        else:
            matching.sort(key=lambda x: x[0], reverse=True)
            results_only = [dec for _, dec in matching]

        total_results = len(results_only)
        start_idx = (request.page - 1) * request.page_size
        end_idx = start_idx + request.page_size
        page_results = results_only[start_idx:end_idx]

        logger.info(
            "BDDK search completed: keyword_chars=%d keyword_terms=%d matches=%d page=%d returned=%d",
            len(request.keywords),
            len(keyword_parts),
            total_results,
            request.page,
            len(page_results),
        )

        return BddkSearchResult(
            decisions=page_results,
            total_results=total_results,
            page=request.page,
            page_size=request.page_size,
            catalog_size=len(self._cache),
        )
