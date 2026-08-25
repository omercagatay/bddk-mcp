"""Reusable, fail-closed input contracts for public MCP tools.

The :class:`~pydantic.BeforeValidator` checks are intentionally duplicated by
JSON Schema constraints.  The schema lets clients prevent invalid calls, while
the validators turn every rejected value into the stable ``INVALID_INPUT`` MCP
error contract instead of exposing Pydantic internals or the supplied value.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Annotated, Literal, Never

from pydantic import BeforeValidator, Field

from bddk_mcp.tools.errors import INVALID_INPUT, tool_error

MAX_QUERY_LENGTH = 500
MAX_HEADING_LENGTH = 300
MAX_DOCUMENT_ID_LENGTH = 64
MAX_PAGE = 10_000
MAX_PAGE_SIZE = 50
MAX_RESULTS = 50
MAX_HISTORY_DAYS = 3_650
MAX_LOOKBACK_WEEKS = 260
MAX_METRIC_ID_LENGTH = 32
MAX_METRIC_IDS = 10
MAX_METRIC_LIST_LENGTH = MAX_METRIC_IDS * MAX_METRIC_ID_LENGTH + (MAX_METRIC_IDS - 1)

DATE_PATTERN = r"^\d{2}\.\d{2}\.\d{4}$"
METRIC_ID_PATTERN = r"^\d+\.\d+\.\d+$"
METRIC_LIST_PATTERN = r"^\d+\.\d+\.\d+(?:\s*,\s*\d+\.\d+\.\d+)*$"
DOCUMENT_ID_PATTERN = r"^[A-Za-z0-9_-]+$"
SECTION_REF_PATTERN = r"^\d+(?:\.\d+)*[A-Za-zÇĞİÖŞÜçğıöşü]?$"
# Dotted outline refs ("2.1", "3.15") come from the numbered-paragraph fallback
# in section_index; the grammar there is \d{1,3}(?:\.\d{1,3})*, so a ref can
# legitimately reach four components. Any ref a tool prints must be accepted
# back as input to that same tool.
SECTION_REF_MAX_LENGTH = 16
INSTRUMENT_ID_PATTERN = r"^inst_sha256_[0-9a-f]{64}$"
ISO_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"

REGULATION_CATEGORIES = (
    "Yönetmelik",
    "Genelge",
    "Tebliğ",
    "Rehber",
    "Bilgi Sistemleri",
    "Sermaye Yeterliliği",
    "Faizsiz Bankacılık",
    "Tekdüzen Hesap Planı",
    "Kurul Kararı",
    "Kanun",
    "Banka Kartları",
    "Finansal Kiralama ve Faktoring",
    "BDDK Düzenlemesi",
    "Düzenleme Taslağı",
    "Mülga Düzenleme",
)
INSTITUTION_TYPES = (
    "Banka",
    "Finansal Kiralama Şirketi",
    "Faktoring Şirketi",
    "Finansman Şirketi",
    "Varlık Yönetim Şirketi",
)
SECTION_TYPES = ("madde", "gecici_madde", "ilke", "paragraf", "ek", "fikra", "bent")
PARTY_CODES = ("10001", "10002", "10003", "10004", "20001", "20002", "20003")


def _invalid(message: str) -> Never:
    tool_error(INVALID_INPUT, message, retryable=False)


def _turkish_casefold(value: str) -> str:
    return value.translate(str.maketrans({"I": "ı", "İ": "i"})).casefold()


def _string(value: object, *, name: str, maximum: int, allow_empty: bool) -> str:
    if not isinstance(value, str):
        _invalid(f"{name} must be a string.")
    normalized = value.strip()
    if not allow_empty and not normalized:
        _invalid(f"{name} must not be empty.")
    if len(normalized) > maximum:
        _invalid(f"{name} exceeds the maximum allowed length.")
    return normalized


def _integer(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        _invalid(f"{name} must be an integer.")
    if value < minimum or value > maximum:
        _invalid(f"{name} is outside the supported range.")
    return value


def _enum(value: object, *, name: str, values: tuple[str, ...], aliases: dict[str, str] | None = None) -> str:
    normalized = _string(value, name=name, maximum=max(len(item) for item in values), allow_empty=False)
    by_folded = {_turkish_casefold(item): item for item in values}
    if aliases:
        by_folded.update({_turkish_casefold(alias): canonical for alias, canonical in aliases.items()})
    canonical = by_folded.get(_turkish_casefold(normalized))
    if canonical is None:
        _invalid(f"{name} is not one of the supported values.")
    return canonical


def _date(value: object, *, name: str, allow_empty: bool) -> str:
    normalized = _string(value, name=name, maximum=10, allow_empty=allow_empty)
    if not normalized and allow_empty:
        return normalized
    if re.fullmatch(DATE_PATTERN, normalized) is None:
        _invalid(f"{name} must use DD.MM.YYYY format.")
    try:
        datetime.strptime(normalized, "%d.%m.%Y")
    except ValueError:
        _invalid(f"{name} must be a real calendar date in DD.MM.YYYY format.")
    return normalized


def _optional_date(value: object, *, name: str) -> str | None:
    if value is None:
        return None
    return _date(value, name=name, allow_empty=False)


def _instrument_id(value: object) -> str:
    if not isinstance(value, str) or value != value.strip():
        _invalid("instrument_id must be an exact canonical identifier without surrounding whitespace.")
    normalized = _string(value, name="instrument_id", maximum=76, allow_empty=False)
    if re.fullmatch(INSTRUMENT_ID_PATTERN, normalized) is None:
        _invalid("instrument_id must be a canonical inst_sha256 identifier.")
    return normalized


def _iso_date(value: object) -> str:
    if not isinstance(value, str) or value != value.strip():
        _invalid("as_of must be an exact ISO date without surrounding whitespace.")
    normalized = _string(value, name="as_of", maximum=10, allow_empty=False)
    if re.fullmatch(ISO_DATE_PATTERN, normalized) is None:
        _invalid("as_of must use ISO YYYY-MM-DD format.")
    try:
        date.fromisoformat(normalized)
    except ValueError:
        _invalid("as_of must be a real calendar date in ISO YYYY-MM-DD format.")
    return normalized


def _metric_id(value: object) -> str:
    normalized = _string(value, name="metric_id", maximum=MAX_METRIC_ID_LENGTH, allow_empty=False)
    if re.fullmatch(METRIC_ID_PATTERN, normalized) is None:
        _invalid("metric_id must use the numeric X.X.X format.")
    return normalized


def parse_metric_ids(value: object) -> list[str]:
    """Validate a bounded comma-separated metric list and return its items."""
    normalized = _string(value, name="metric_ids", maximum=MAX_METRIC_LIST_LENGTH, allow_empty=False)
    raw_ids = normalized.split(",")
    if any(not item.strip() for item in raw_ids):
        _invalid("metric_ids must not contain empty list items.")
    if len(raw_ids) > MAX_METRIC_IDS:
        _invalid("metric_ids contains too many items.")
    return [_metric_id(item) for item in raw_ids]


def _metric_ids(value: object) -> str:
    return ",".join(parse_metric_ids(value))


def _document_id(value: object) -> str:
    normalized = _string(value, name="document_id", maximum=MAX_DOCUMENT_ID_LENGTH, allow_empty=False)
    if re.fullmatch(DOCUMENT_ID_PATTERN, normalized) is None:
        _invalid("document_id contains unsupported characters.")
    return normalized


def _optional_document_id(value: object) -> str | None:
    if value is None:
        return None
    return _document_id(value)


def _section_type(value: object) -> str:
    aliases = {"geçici madde": "gecici_madde", "geçici_madde": "gecici_madde", "fıkra": "fikra"}
    return _enum(value, name="section_type", values=SECTION_TYPES, aliases=aliases)


def _optional_section_type(value: object) -> str | None:
    if value is None:
        return None
    return _section_type(value)


def _section_ref(value: object) -> str | int | None:
    if value is None:
        return None
    if type(value) is int:
        return _integer(value, name="section_ref", minimum=1, maximum=99_999)
    normalized = _string(value, name="section_ref", maximum=SECTION_REF_MAX_LENGTH, allow_empty=False)
    if re.fullmatch(SECTION_REF_PATTERN, normalized) is None:
        _invalid("section_ref must be a positive number or dotted outline, with at most one letter suffix.")
    return normalized.lower()


def _optional_heading(value: object) -> str | None:
    if value is None:
        return None
    return _string(value, name="heading", maximum=MAX_HEADING_LENGTH, allow_empty=False)


def _required_query(value: object) -> str:
    return _string(value, name="query", maximum=MAX_QUERY_LENGTH, allow_empty=False)


def _required_keywords(value: object) -> str:
    return _string(value, name="keywords", maximum=MAX_QUERY_LENGTH, allow_empty=False)


def _optional_keywords(value: object) -> str:
    return _string(value, name="keywords", maximum=MAX_QUERY_LENGTH, allow_empty=True)


def _regulation_category(value: object) -> str:
    aliases = {
        "yonetmelik": "Yönetmelik",
        "teblig": "Tebliğ",
        "bilgi sistemleri": "Bilgi Sistemleri",
        "sermaye yeterliligi": "Sermaye Yeterliliği",
        "faizsiz bankacilik": "Faizsiz Bankacılık",
        "tekduzen hesap plani": "Tekdüzen Hesap Planı",
        "kurul karari": "Kurul Kararı",
        "banka kartlari": "Banka Kartları",
        "bddk duzenlemesi": "BDDK Düzenlemesi",
        "duzenleme taslagi": "Düzenleme Taslağı",
        "mulga duzenleme": "Mülga Düzenleme",
    }
    return _enum(value, name="category", values=REGULATION_CATEGORIES, aliases=aliases)


def _optional_regulation_category(value: object) -> str | None:
    if value is None:
        return None
    return _regulation_category(value)


def normalize_institution_type(value: object) -> str | None:
    if value is None:
        return None
    return _enum(value, name="institution_type", values=INSTITUTION_TYPES)


def normalize_announcement_category(value: object) -> str:
    aliases = {
        "press": "basın",
        "regul": "mevzuat",
        "regulation": "mevzuat",
        "insan": "insan kaynakları",
        "hr": "insan kaynakları",
        "data": "veri",
        "institution": "kuruluş",
        "all": "tümü",
    }
    return _enum(
        value,
        name="category",
        values=("basın", "mevzuat", "insan kaynakları", "veri", "kuruluş", "tümü"),
        aliases=aliases,
    )


def _bool(value: object) -> bool:
    if type(value) is not bool:
        _invalid("active_only must be a boolean.")
    return value


def _weekly_currency(value: object) -> str:
    return _enum(value, name="currency", values=("TRY", "USD"))


def _monthly_currency(value: object) -> str:
    return _enum(value, name="currency", values=("TL", "USD"))


def _column(value: object) -> str:
    return _enum(value, name="column", values=("1", "2", "3"))


def _party_code(value: object) -> str:
    return _enum(value, name="party_code", values=PARTY_CODES)


def _period(value: object) -> str:
    return _enum(value, name="period", values=("day", "week", "month", "quarter"))


def validate_date_order(date_from: str | None, date_to: str | None) -> None:
    """Reject a reversed optional date interval without echoing either value."""
    normalized_from = _optional_date(date_from, name="date_from")
    normalized_to = _optional_date(date_to, name="date_to")
    if normalized_from and normalized_to:
        start = datetime.strptime(normalized_from, "%d.%m.%Y")
        end = datetime.strptime(normalized_to, "%d.%m.%Y")
        if start > end:
            _invalid("date_from must not be later than date_to.")


RegulationKeywords = Annotated[
    str,
    Field(min_length=1, max_length=MAX_QUERY_LENGTH, description="Turkish words required in catalog metadata."),
    BeforeValidator(_required_keywords),
]
OptionalSearchKeywords = Annotated[
    str,
    Field(max_length=MAX_QUERY_LENGTH, description="Optional Turkish name or title substring; empty searches all."),
    BeforeValidator(_optional_keywords),
]
SemanticQuery = Annotated[
    str,
    Field(min_length=1, max_length=MAX_QUERY_LENGTH, description="Non-empty Turkish full-text or semantic query."),
    BeforeValidator(_required_query),
]
SectionQuery = Annotated[
    str,
    Field(min_length=1, max_length=MAX_QUERY_LENGTH, description="Non-empty Turkish legal-section query."),
    BeforeValidator(_required_query),
]
PageNumber = Annotated[
    int,
    Field(ge=1, le=MAX_PAGE, description="One-indexed catalog result page."),
    BeforeValidator(lambda value: _integer(value, name="page", minimum=1, maximum=MAX_PAGE)),
]
PageSize = Annotated[
    int,
    Field(ge=1, le=MAX_PAGE_SIZE, description="Catalog results per page."),
    BeforeValidator(lambda value: _integer(value, name="page_size", minimum=1, maximum=MAX_PAGE_SIZE)),
]
ResultLimit = Annotated[
    int,
    Field(ge=1, le=MAX_RESULTS, description="Maximum number of results to return."),
    BeforeValidator(lambda value: _integer(value, name="limit", minimum=1, maximum=MAX_RESULTS)),
]
SectionResultLimit = Annotated[
    int,
    Field(ge=1, le=20, description="Maximum number of bounded section excerpts to return."),
    BeforeValidator(lambda value: _integer(value, name="limit", minimum=1, maximum=20)),
]
ActiveOnly = Annotated[
    bool,
    Field(description="When true, exclude institutions not marked active."),
    BeforeValidator(_bool),
]

RegulationCategory = Annotated[
    Literal[
        "Yönetmelik",
        "Genelge",
        "Tebliğ",
        "Rehber",
        "Bilgi Sistemleri",
        "Sermaye Yeterliliği",
        "Faizsiz Bankacılık",
        "Tekdüzen Hesap Planı",
        "Kurul Kararı",
        "Kanun",
        "Banka Kartları",
        "Finansal Kiralama ve Faktoring",
        "BDDK Düzenlemesi",
        "Düzenleme Taslağı",
        "Mülga Düzenleme",
    ],
    Field(description="Exact BDDK regulation category; matching is case-insensitive."),
    BeforeValidator(_regulation_category),
]
OptionalRegulationCategory = Annotated[
    RegulationCategory | None,
    Field(description="Optional exact BDDK regulation category."),
    BeforeValidator(_optional_regulation_category),
]
InstitutionType = Annotated[
    Literal[
        "Banka",
        "Finansal Kiralama Şirketi",
        "Faktoring Şirketi",
        "Finansman Şirketi",
        "Varlık Yönetim Şirketi",
    ]
    | None,
    Field(description="Optional exact institution directory type; matching is case-insensitive."),
    BeforeValidator(normalize_institution_type),
]
AnnouncementCategory = Annotated[
    Literal["basın", "mevzuat", "insan kaynakları", "veri", "kuruluş", "tümü"],
    Field(description="Announcement category; use 'tümü' to query every category."),
    BeforeValidator(normalize_announcement_category),
]

DateFrom = Annotated[
    str | None,
    Field(
        max_length=10,
        pattern=DATE_PATTERN,
        description="Optional inclusive start date in DD.MM.YYYY format.",
        json_schema_extra={"format": "date-dd-mm-yyyy"},
    ),
    BeforeValidator(lambda value: _optional_date(value, name="date_from")),
]
DateTo = Annotated[
    str | None,
    Field(
        max_length=10,
        pattern=DATE_PATTERN,
        description="Optional inclusive end date in DD.MM.YYYY format.",
        json_schema_extra={"format": "date-dd-mm-yyyy"},
    ),
    BeforeValidator(lambda value: _optional_date(value, name="date_to")),
]
BulletinDate = Annotated[
    str,
    Field(
        max_length=10,
        pattern=r"^(?:|\d{2}\.\d{2}\.\d{4})$",
        description="Specific DD.MM.YYYY calendar date, or an empty string for the latest data.",
        json_schema_extra={"format": "date-dd-mm-yyyy-or-empty"},
    ),
    BeforeValidator(lambda value: _date(value, name="date", allow_empty=True)),
]

DocumentId = Annotated[
    str,
    Field(
        min_length=1,
        max_length=MAX_DOCUMENT_ID_LENGTH,
        pattern=DOCUMENT_ID_PATTERN,
        description="Stored document ID using letters, digits, underscores, or hyphens.",
    ),
    BeforeValidator(_document_id),
]
OptionalDocumentId = Annotated[
    str | None,
    Field(
        max_length=MAX_DOCUMENT_ID_LENGTH,
        pattern=DOCUMENT_ID_PATTERN,
        description="Optional stored document ID filter.",
    ),
    BeforeValidator(_optional_document_id),
]
InstrumentId = Annotated[
    str,
    Field(
        min_length=76,
        max_length=76,
        pattern=INSTRUMENT_ID_PATTERN,
        description="Exact canonical legal instrument ID in inst_sha256_<64 lowercase hex> form.",
    ),
    BeforeValidator(_instrument_id),
]
AsOfDate = Annotated[
    str,
    Field(
        min_length=10,
        max_length=10,
        pattern=ISO_DATE_PATTERN,
        description="Required inclusive legal-status date in ISO YYYY-MM-DD format; currentness is never inferred.",
        json_schema_extra={"format": "date"},
    ),
    BeforeValidator(_iso_date),
]
SectionType = Annotated[
    Literal["madde", "gecici_madde", "ilke", "paragraf", "ek", "fikra", "bent"] | None,
    Field(description="Optional canonical structural type stored by the section index."),
    BeforeValidator(_optional_section_type),
]
SectionRef = Annotated[
    Annotated[
        str,
        Field(min_length=1, max_length=SECTION_REF_MAX_LENGTH, pattern=SECTION_REF_PATTERN),
    ]
    | Annotated[int, Field(ge=1, le=99_999)]
    | None,
    Field(
        description=(
            "Optional positive section number or dotted outline ref (e.g. 9, 2.1), with at most one letter suffix."
        )
    ),
    BeforeValidator(_section_ref),
]
HeadingFilter = Annotated[
    str | None,
    Field(min_length=1, max_length=MAX_HEADING_LENGTH, description="Optional non-empty heading substring."),
    BeforeValidator(_optional_heading),
]
EdgeDirection = Annotated[
    Literal["both", "incoming", "outgoing"],
    Field(description="Relation edge direction relative to the queried document."),
]
ExpandReferences = Annotated[
    bool,
    Field(
        description=(
            "Follow validated cross-reference edges one hop and append related "
            "section pointers; related content is never inlined."
        )
    ),
]

MetricId = Annotated[
    str,
    Field(
        min_length=5,
        max_length=MAX_METRIC_ID_LENGTH,
        pattern=METRIC_ID_PATTERN,
        description="BDDK bulletin metric identifier in numeric X.X.X format.",
    ),
    BeforeValidator(_metric_id),
]
MetricIdList = Annotated[
    str,
    Field(
        min_length=5,
        max_length=MAX_METRIC_LIST_LENGTH,
        pattern=METRIC_LIST_PATTERN,
        description=f"One to {MAX_METRIC_IDS} comma-separated BDDK metric IDs in X.X.X format.",
    ),
    BeforeValidator(_metric_ids),
]
WeeklyCurrency = Annotated[
    Literal["TRY", "USD"],
    Field(description="Weekly bulletin currency: TRY or USD."),
    BeforeValidator(_weekly_currency),
]
MonthlyCurrency = Annotated[
    Literal["TL", "USD"],
    Field(description="Monthly bulletin currency: TL or USD."),
    BeforeValidator(_monthly_currency),
]
BulletinColumn = Annotated[
    Literal["1", "2", "3"],
    Field(description="Bulletin value column: 1=TP, 2=YP, or 3=total."),
    BeforeValidator(_column),
]
HistoryDays = Annotated[
    int,
    Field(ge=1, le=MAX_HISTORY_DAYS, description="Calendar-day history window."),
    BeforeValidator(lambda value: _integer(value, name="days", minimum=1, maximum=MAX_HISTORY_DAYS)),
]
LookbackWeeks = Annotated[
    int,
    Field(ge=1, le=MAX_LOOKBACK_WEEKS, description="Weekly trend lookback window."),
    BeforeValidator(lambda value: _integer(value, name="lookback_weeks", minimum=1, maximum=MAX_LOOKBACK_WEEKS)),
]
MonthlyTableNumber = Annotated[
    int,
    Field(ge=1, le=17, description="BDDK monthly bulletin table number."),
    BeforeValidator(lambda value: _integer(value, name="table_no", minimum=1, maximum=17)),
]
MonthlyYear = Annotated[
    int,
    Field(ge=2000, le=2100, description="Four-digit bulletin year from 2000 through 2100."),
    BeforeValidator(lambda value: _integer(value, name="year", minimum=2000, maximum=2100)),
]
MonthlyMonth = Annotated[
    int,
    Field(ge=1, le=12, description="Calendar month number from 1 through 12."),
    BeforeValidator(lambda value: _integer(value, name="month", minimum=1, maximum=12)),
]
PartyCode = Annotated[
    Literal["10001", "10002", "10003", "10004", "20001", "20002", "20003"],
    Field(description="BDDK monthly bulletin bank-group party code."),
    BeforeValidator(_party_code),
]
RegulatoryPeriod = Annotated[
    Literal["day", "week", "month", "quarter"],
    Field(description="Digest lookback: day, week, month, or quarter."),
    BeforeValidator(_period),
]
