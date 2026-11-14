import asyncio
import logging
import re
import sqlite3
import time
from datetime import datetime
from fractions import Fraction
from urllib.parse import parse_qs, urlparse

import httpx
from bs4 import BeautifulSoup
from markitdown import MarkItDown
from pydantic import Field

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
from bddk_mcp.store.doc_store import DocumentStore, normalize_search_text
from bddk_mcp.store.legal_ref import parse_legal_refs

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
# They are dropped before the all-token match. Matching is on folded tokens
# (normalize_search_text), so each word is listed once in its Turkish spelling.
_CATALOG_FUNCTION_WORDS = frozenset(
    normalize_search_text(word)
    for word in (
        "sayılı",
        "sayısı",
        "hakkında",
        "hk",
        "ilişkin",
        "dair",
        "göre",
        "ile",
        "ve",
        "veya",
        "nolu",
        "no",
        "numaralı",
        "tarihli",
        "tarihi",
    )
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
# Short roots ("fon", "dış", "test", "kart", "izin"): a common root of this many characters is enough when
# what follows it on both sides is an inflectional ending (plural, possessive, case); two letters never are.
_STEM_MIN_ROOT = 3
# A remainder longer than this after the common prefix must be an inflectional ending, unless the common
# prefix is longer than the remainder: "yapılandırma" does not reach "yapılması" or "yapılacak" through "yapıl".
_STEM_MAX_FREE_REMAINDER = 5

_TOKEN_EDGE_PUNCTUATION = ".,;:()[]\"'’!?-–—"
_TOKEN_SPLIT_RE = re.compile(r"[\s()\[\]{},;:!?\"“”«»\-–—]+")
_INTERIOR_APOSTROPHE_RE = re.compile(r"['’]")
_SLASH_DATE_RE = re.compile(r"\d{1,2}/\d{1,2}/\d{4}")
# Two words joined by a slash or a dot ("Destek/Danışmanlık", "Pozisyon/Özkaynak", "T.Vakıflar"): a letter, then
# the joiner, then a word of at least three letters. Abbreviations ("A.Ş.", "S.A.L") and numbers stay one token.
_JOINED_WORDS_RE = re.compile(r"(?<=[^\W\d_])[./](?=[^\W\d_]{3})")

# The nominal inflection paradigm on folded text (ı→i, ü→u, ö→o, ç→c), reviewed against every tail between
# two title words of the catalog: [plural] [third-person possessive] [case]. The buffer letters depend on the
# letter before the ending: after a vowel the possessive is -s[ıiuü] and the cases take y/n ("bankaya",
# "bankanın"); after a consonant they start with the vowel ("kanunu", "kanuna"); after a possessive the cases
# take the pronominal n ("fonunu", "fonuna", "fonunda", "fonunca", "kararıyla"), also directly after a vowel-final
# stem, whose last vowel may be that possessive ("ili"/"ilinde"). Cases: accusative, dative, locative (+ relative
# -ki), ablative, genitive, instrumental, equative -ca/-ce. Anything else is a derivation or another word: -siz,
# -lik, -ci, -li, -sel, -ma/-me, -an/-en, -ış ("Faizsiz", "Kuruluş", "Kurmasına").
_PLURAL = r"l[ae]r"
_CASE_AFTER_POSSESSIVE = r"n[iu]n?|n[ae]|nd[ae](?:n|ki)?|nc[ae]|yl[ae]"
_CASE_AFTER_VOWEL = r"y[iu]|y[ae]|n[iu]n|d[ae](?:n|ki)?|y?l[ae]|c[ae]"
_CASE_AFTER_CONSONANT = r"[iu]n?|[ae]|[dt][ae](?:n|ki)?|l[ae]|c[ae]"
_AFTER_CONSONANT = rf"(?:[iu](?:{_CASE_AFTER_POSSESSIVE})?|{_CASE_AFTER_CONSONANT})?"
_INFLECTION_AFTER_CONSONANT_RE = re.compile(rf"(?:{_PLURAL})?{_AFTER_CONSONANT}")
_INFLECTION_AFTER_VOWEL_RE = re.compile(
    rf"{_PLURAL}{_AFTER_CONSONANT}|(?:s[iu](?:{_CASE_AFTER_POSSESSIVE})?|{_CASE_AFTER_VOWEL}|{_CASE_AFTER_POSSESSIVE})?"
)
# Consonant softening before a vowel ending (grup/grubu, hesap/hesabı, kayıt/kaydı, yönetmelik/yönetmeliği);
# ç/c is already one letter after folding.
_SOFTENING_PAIRS = frozenset({("p", "b"), ("b", "p"), ("t", "d"), ("d", "t"), ("k", "g"), ("g", "k")})
_VOICED_SOFT = frozenset("bdg")
_VOWELS = frozenset("aeiou")
# Reviewed roots that drop a vowel before a vowel ending (izin/izni, isim/ismi, kayıt/kaydı, ...), folded:
# elided form -> full root. Applied only when the elided form is followed by a vowel.
_ELIDED_ROOTS: dict[str, str] = {
    "izn": "izin",
    "ism": "isim",
    "kayd": "kayit",
    "akl": "akil",
    "hukm": "hukum",
    "emr": "emir",
    "nakl": "nakil",
    "nakd": "nakit",
    "sehr": "sehir",
    "devr": "devir",
}
# Reviewed lexicalised words, folded, that the paradigm would otherwise read as an inflection of a shorter root:
# "gerekçe" (justification; "Genel Gerekçe" of the draft titles) is not gerek + equative -ce, "malî" (financial;
# "Mali İşler") is not mal + -i. Such a word, or an inflection of it, never matches through a root shorter than
# itself, whichever side it is on; it still matches itself and its own inflections ("gerekçesi").
_LEXICALISED_WORDS = frozenset({"gerekce", "mali"})

# The taslak pattern of _DOCUMENT_TYPE_PATTERNS: a query naming it asks for drafts.
_DRAFT_TYPE_PATTERN = _DOCUMENT_TYPE_PATTERNS[-1]
_DRAFT_CATEGORY = normalize_search_text("Düzenleme Taslağı")
_REPEALED_CATEGORY = normalize_search_text("Mülga Düzenleme")
_REPEALED_WORD = normalize_search_text("mülga")
_PREFIXED_DOCUMENT_ID_RE = re.compile(r"(?:mevzuat|bddk)_(\d+)")
_SAYILI = normalize_search_text("sayılı")
# "Kurul" directly before a document-type word ("Kurul Kararı") is boilerplate in title-side coverage.
_KURUL = normalize_search_text("kurul")


def _catalog_tokens(text: str) -> list[str]:
    """Folded word tokens: normalize_search_text, split on spaces and punctuation, an interior
    apostrophe cut off with what follows it ("Kanunu'nun" -> "kanunu"), DD/MM/YYYY as DD.MM.YYYY, two words
    joined by a slash or a dot split ("pozisyon/ozkaynak" -> "pozisyon", "ozkaynak")."""
    tokens: list[str] = []
    for joined in _TOKEN_SPLIT_RE.split(normalize_search_text(text)):
        for raw in _JOINED_WORDS_RE.split(joined):
            token = _INTERIOR_APOSTROPHE_RE.split(raw.strip(_TOKEN_EDGE_PUNCTUATION), maxsplit=1)[0]
            token = token.strip(_TOKEN_EDGE_PUNCTUATION)
            if _SLASH_DATE_RE.fullmatch(token):
                token = token.replace("/", ".")
            if token:
                tokens.append(token)
    return tokens


class CatalogSearchResult(BddkSearchResult):
    """A catalog search result that also names the section references left out of the required words."""

    section_references_not_searched: list[str] = Field(
        default_factory=list,
        description=(
            "Section references of the keywords (an article, principle, paragraph or annex, e.g. 'madde 9', "
            "'EK-1') that were not searched as catalog title words; empty when none was left out."
        ),
    )


def _without_section_references(keywords: str) -> tuple[str, list[str]]:
    """The keywords without the section references ``parse_legal_refs`` recognises (a kind word with its
    number, an annex), and those references as typed. A catalog title names an instrument, not one of its
    articles, so "madde 9" is never a required title word; a document or decision number and a standard's
    number ("TFRS 9") are not section references and stay. When nothing else would remain to search, the
    keywords are returned unchanged."""
    parts: list[str] = []
    left_out: list[str] = []
    end = 0
    for start, stop, _, _ in parse_legal_refs(keywords).section_spans:
        if start < end:
            continue
        parts.append(keywords[end:start])
        left_out.append(keywords[start:stop].strip())
        end = stop
    if not left_out:
        return keywords, []
    parts.append(keywords[end:])
    rest = " ".join(part.strip() for part in parts if part.strip())
    if not [token for token in _catalog_tokens(rest) if token not in _CATALOG_FUNCTION_WORDS]:
        return keywords, []
    return rest, list(dict.fromkeys(left_out))


def catalog_keyword_tokens(keywords: str) -> list[str]:
    """The tokens search_decisions matches for ``keywords``; empty means browse mode."""
    return _catalog_tokens(keywords)


def _document_type_pattern(token: str) -> re.Pattern[str] | None:
    for pattern in _DOCUMENT_TYPE_PATTERNS:
        if pattern.fullmatch(token):
            return pattern
    return None


def _is_inflection_tail(remainder: str, before: str) -> bool:
    """Whether ``remainder`` is a nominal inflection of a stem whose last letter is ``before``."""
    pattern = _INFLECTION_AFTER_VOWEL_RE if before in _VOWELS else _INFLECTION_AFTER_CONSONANT_RE
    return pattern.fullmatch(remainder) is not None


def _restore_elided_root(word: str) -> str:
    for elided, root in _ELIDED_ROOTS.items():
        if word.startswith(elided) and len(word) > len(elided) and word[len(elided)] in _VOWELS:
            return root + word[len(elided) :]
    return word


def _lexicalised_root_length(word: str) -> int:
    """Length of the reviewed lexicalised word that ``word`` is or inflects, else 0."""
    for lexicalised in _LEXICALISED_WORDS:
        if word.startswith(lexicalised) and _is_inflection_tail(word[len(lexicalised) :], lexicalised[-1]):
            return len(lexicalised)
    return 0


def _common_prefix(left: str, right: str, *, softening: bool = False) -> int:
    """Length of the common prefix; with ``softening`` one p/b, t/d or k/g pair counts as equal when the
    voiced letter is followed by a vowel (a root-final consonant before a vowel ending)."""
    prefix = 0
    softened = False
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a == b:
            prefix += 1
            continue
        if not softening or softened or (a, b) not in _SOFTENING_PAIRS:
            break
        voiced_side = left if a in _VOICED_SOFT else right
        if index + 1 >= len(voiced_side) or voiced_side[index + 1] not in _VOWELS:
            break
        softened = True
        prefix += 1
    return prefix


def _stem_grade(token: str, word: str) -> int:
    """How closely two folded words match: 2 = the same word or an inflection of the same root,
    1 = a shared stem by the general prefix rule, 0 = no match."""
    if token == word:
        return 2
    # Inflection of one root: a root of at least _STEM_MIN_ROOT characters within the common prefix
    # (softening and the reviewed elisions allowed) followed on both sides by an inflectional ending
    # ("riske"/"riski", "orani"/"oranini" through the root "oran").
    restored_token, restored_word = _restore_elided_root(token), _restore_elided_root(word)
    prefix = _common_prefix(restored_token, restored_word, softening=True)
    min_root = max(_STEM_MIN_ROOT, _lexicalised_root_length(restored_token), _lexicalised_root_length(restored_word))
    for root in range(prefix, min_root - 1, -1):
        if _is_inflection_tail(restored_token[root:], restored_token[root - 1]) and _is_inflection_tail(
            restored_word[root:], restored_word[root - 1]
        ):
            return 2
    # General rule: both words at least _STEM_MIN_LENGTH long and a plain common prefix that covers all
    # but _STEM_MAX_SUFFIX characters of the shorter word; a long remainder must be an ending.
    shorter = min(len(token), len(word))
    if shorter < _STEM_MIN_LENGTH:
        return 0
    prefix = _common_prefix(token, word)
    if prefix < _STEM_MIN_LENGTH or prefix < shorter - _STEM_MAX_SUFFIX:
        return 0
    before = token[prefix - 1]
    for remainder in (token[prefix:], word[prefix:]):
        if (
            len(remainder) > _STEM_MAX_FREE_REMAINDER
            and prefix <= len(remainder)
            and not _is_inflection_tail(remainder, before)
        ):
            return 0
    # A query token that extends the whole title word by a derivation ("yapılan" + "dırma") is a different
    # word; a short query still reaches its longer title forms ("banka" -> "Bankacılık").
    token_tail = token[prefix:]
    if prefix == len(word) and len(token_tail) >= _STEM_MAX_SUFFIX and not _is_inflection_tail(token_tail, before):
        return 0
    return 1


def _shares_stem(token: str, word: str) -> bool:
    """Conservative inflection-tolerant comparison of two folded words."""
    return _stem_grade(token, word) > 0


def _title_word_matches(token: str, word: str) -> bool:
    """A content token matches a title word: equal, or (not a number, not a type word) by the stem rule."""
    if token == word:
        return True
    if any(char.isdigit() for char in token) or _document_type_pattern(word) is not None:
        return False
    return _stem_grade(token, word) > 0


def _phrase_and_title_coverage(content_tokens: list[str], title_words: list[str]) -> tuple[int, Fraction]:
    """Phrase hit and title-side coverage of a title for the query's content tokens.

    Phrase hit: 2 when the tokens match adjacent title words in query order at the start of the title (a leading
    document-type word does not count), 1 when they do so later in the title, 0 otherwise. Function words of the
    title are skipped ("Usul ve Esaslar" holds the phrase "usul esaslar"). Title-side coverage: the title's
    content words (neither function nor document-type words) matched by a token, over all of them, as an exact
    fraction. Boilerplate is left out of both sides of that fraction: "kurul" directly before a document-type word
    ("Kurul Kararı") and the number of a leading "<n> sayılı" citation ("5464 sayılı ...").
    """
    words = [word for word in title_words if word not in _CATALOG_FUNCTION_WORDS]
    if not content_tokens or not words:
        return 0, Fraction(0)
    lead = 0
    while lead < len(words) - 1 and _document_type_pattern(words[lead]) is not None:
        lead += 1
    phrase = 0
    for start in range(len(words) - len(content_tokens) + 1):
        if all(_title_word_matches(token, words[start + offset]) for offset, token in enumerate(content_tokens)):
            phrase = 2 if start == lead else 1
            break
    boilerplate = {
        index
        for index, word in enumerate(title_words)
        if (
            word == _KURUL
            and index + 1 < len(title_words)
            and _document_type_pattern(title_words[index + 1]) is not None
        )
        or (index == 0 and len(title_words) > 1 and title_words[1] == _SAYILI and any(c.isdigit() for c in word))
    }
    content_words = [
        word
        for index, word in enumerate(title_words)
        if index not in boilerplate and word not in _CATALOG_FUNCTION_WORDS and _document_type_pattern(word) is None
    ]
    matched = sum(1 for word in content_words if any(_title_word_matches(token, word) for token in content_tokens))
    return phrase, Fraction(matched, len(content_words)) if content_words else Fraction(0)


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

        Titles, catalog fields and keywords are folded alike (normalize_search_text:
        Turkish-aware lower case, diacritics removed), so "siniflandirilmasi",
        "SINIFLANDIRILMASI" and "sınıflandırılması" are one word, and split into
        words; an interior apostrophe ends a word ("Kanunu'nun" -> "kanunu").
        Function words ("sayılı", "hakkında", "ilişkin", ...) are dropped.
        Document-type words ("yönetmeliği", "tebliğ", "kararı", ...) are never
        required literally. Every remaining content token must match a whole
        title word (equal, an inflection of the same root, or a shared stem of at
        least five characters) or a whole word of the category, the decision
        date, the decision number or the document id; a number matches whole
        numbers only. Ranking: a number identifying the entry (its decision
        number or the number of its mevzuat_/bddk_ document id, then a leading
        "<n> sayılı" in its title) first; then a title equal to the query; the
        share of content tokens found in the title; tokens found as the title's
        own words or their inflections; the requested document type (named by
        title and category counts most); in-force text before drafts and repealed
        text unless the query names them; the overall content score; the share of
        the title's content words matched by the query (exact fraction, so the
        instrument whose title is the query's subject comes first); the query's
        content words adjacent and in order in the title, at its start before
        later; then the newest decision date and the shorter title. Empty keywords list the
        filtered catalog (browse mode), newest decision date first.
        A section reference in the keywords ("madde 9", "ilke 5", "EK-1") is not a title word: it is
        left out of the required tokens and named in ``section_references_not_searched``.
        """
        await self._ensure_cache()

        searched_keywords, references_left_out = _without_section_references(request.keywords)
        keyword_parts = _catalog_tokens(searched_keywords)
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
        # A title equal to the query counts the section references too (a title may name an article).
        typed_parts = _catalog_tokens(request.keywords) if references_left_out else keyword_parts
        query_words = [part for part in typed_parts if part not in _CATALOG_FUNCTION_WORDS] or typed_parts
        asks_draft = _DRAFT_TYPE_PATTERN in type_patterns
        asks_repealed = _REPEALED_WORD in keyword_parts

        category_filter = _turkish_lower(request.category) if request.category else None
        date_from = _parse_date(request.date_from) if request.date_from else None
        date_to = _parse_date(request.date_to) if request.date_to else None

        matching: list[tuple[tuple[int | float, ...], BddkDecisionSummary]] = []
        # Entries that pass every other filter but have no parseable decision date: a date bound cannot
        # consider them, so they are counted and reported instead of being dropped silently.
        undated_excluded = 0
        for dec in self._cache:
            if category_filter and category_filter not in _turkish_lower(dec.category):
                continue

            undated = False
            if date_from or date_to:
                doc_date = _parse_date(dec.decision_date) if dec.decision_date else None
                if doc_date is None:
                    undated = True
                elif (date_from and doc_date < date_from) or (date_to and doc_date > date_to):
                    continue

            if browse:
                if undated:
                    undated_excluded += 1
                else:
                    matching.append(((0,), dec))
                continue

            title_words = _catalog_tokens(dec.title)
            # Type-word title words ("Kanunu", "Rehberi", "Kararı") are matched only by the type
            # patterns: a derived content word ("kanuni", "rehberlik", "kararname") must not reach
            # them through the shared-stem rule. Function words ("Dair") are matched only literally:
            # "daire" is not an inflection of "dair".
            stem_words = [
                word
                for word in title_words
                if _document_type_pattern(word) is None and word not in _CATALOG_FUNCTION_WORDS
            ]
            category_words = _catalog_tokens(dec.category)
            category_stem_words = [word for word in category_words if _document_type_pattern(word) is None]
            document_id = normalize_search_text(dec.document_id.strip())
            decision_number = dec.decision_number.strip()
            id_number_match = _PREFIXED_DOCUMENT_ID_RE.fullmatch(document_id)
            id_number = id_number_match.group(1) if id_number_match else ""
            title_number = title_words[0] if len(title_words) > 1 and title_words[1] == _SAYILI else ""
            # Whole words of the other catalog fields: date, decision number, document id.
            field_words = set(_catalog_tokens(f"{dec.decision_date} {decision_number}"))
            if id_number:
                field_words.update({document_id, id_number})

            content_score = 0
            exact_title_hits = 0
            title_found = 0
            number_hit = 0
            all_match = True
            for part in content_tokens:
                numeric = any(char.isdigit() for char in part)
                if numeric:
                    title_grade = 2 if part in title_words else 0
                elif part in title_words:
                    title_grade = 2
                else:
                    title_grade = max((_stem_grade(part, word) for word in stem_words), default=0)
                if title_grade == 2:
                    content_score += 3
                    exact_title_hits += 1
                    title_found += 1
                elif title_grade == 1:
                    content_score += 2
                    title_found += 1
                elif (
                    part in field_words
                    or part in category_words
                    or (not numeric and any(_shares_stem(part, word) for word in category_stem_words))
                ):
                    content_score += 1
                else:
                    all_match = False
                    break
                if part.isdigit():
                    if part in (decision_number, id_number):
                        number_hit = max(number_hit, 2)
                    elif part == title_number:
                        number_hit = max(number_hit, 1)
            if not all_match:
                continue

            # A type word named by both the title and the category counts twice.
            type_hits = sum(
                any(pattern.fullmatch(word) for word in title_words)
                + any(pattern.fullmatch(word) for word in category_words)
                for pattern in type_patterns
            )
            if content_score == 0 and type_hits == 0:
                continue
            if undated:
                undated_excluded += 1
                continue
            folded_category = normalize_search_text(dec.category)
            is_draft = folded_category == _DRAFT_CATEGORY or any(
                _DRAFT_TYPE_PATTERN.fullmatch(word) for word in title_words
            )
            is_repealed = folded_category == _REPEALED_CATEGORY or _REPEALED_WORD in title_words
            in_force_or_asked = int((not is_draft or asks_draft) and (not is_repealed or asks_repealed))
            title_equal = int([w for w in title_words if w not in _CATALOG_FUNCTION_WORDS] == query_words)
            coverage = title_found / len(content_tokens) if content_tokens else 0.0
            phrase, title_coverage = _phrase_and_title_coverage(content_tokens, title_words)
            matching.append(
                (
                    (
                        number_hit,
                        title_equal,
                        coverage,
                        exact_title_hits,
                        type_hits,
                        in_force_or_asked,
                        content_score,
                        title_coverage,
                        phrase,
                    ),
                    dec,
                )
            )

        if browse:
            # Newest decision date first; undated entries keep catalog order at the end.
            dated = [(d, dec) for _, dec in matching if (d := _parse_date(dec.decision_date)) is not None]
            undated = [dec for _, dec in matching if _parse_date(dec.decision_date) is None]
            dated.sort(key=lambda item: item[0], reverse=True)
            results_only = [dec for _, dec in dated] + undated
        else:
            # Score first, then the newest decision date (undated last), then the shorter title; never the
            # catalog's document-id order.
            def _rank(item: tuple[tuple[int | float, ...], BddkDecisionSummary]) -> tuple:
                score, dec = item
                parsed = _parse_date(dec.decision_date) if dec.decision_date else None
                return (
                    tuple(-value for value in score),
                    -(parsed.toordinal() if parsed else 0),
                    len(dec.title),
                    normalize_search_text(dec.title),
                )

            matching.sort(key=_rank)
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

        return CatalogSearchResult(
            decisions=page_results,
            total_results=total_results,
            page=request.page,
            page_size=request.page_size,
            catalog_size=len(self._cache),
            undated_excluded=undated_excluded,
            section_references_not_searched=references_left_out,
        )
