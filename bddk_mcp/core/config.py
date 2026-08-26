"""Unified configuration for BDDK MCP Server.

All tunable constants in one place. Values can be overridden via environment
variables (prefixed with BDDK_).
"""

import math
import os
import re
from pathlib import Path


def _environment_bool(name: str, *, default: bool) -> bool:
    """Parse one explicit boolean without silently treating typos as false."""

    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise RuntimeError(f"{name} must be one of true, false, 1, 0, yes, or no.")


def _environment_int(
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """Parse a bounded integer and return a stable, value-free error."""

    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        raise RuntimeError(f"{name} must be an integer between {minimum} and {maximum}.") from None
    if not minimum <= value <= maximum:
        raise RuntimeError(f"{name} must be an integer between {minimum} and {maximum}.")
    return value


def _environment_float(
    name: str,
    *,
    default: float,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
) -> float:
    """Parse a finite bounded float and return a stable, value-free error."""

    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        comparator = "between" if minimum_inclusive else "greater than"
        raise RuntimeError(f"{name} must be finite and {comparator} the configured bounds.") from None
    lower_bound_ok = value >= minimum if minimum_inclusive else value > minimum
    if not math.isfinite(value) or not lower_bound_ok or value > maximum:
        comparator = "between" if minimum_inclusive else "greater than"
        raise RuntimeError(f"{name} must be finite and {comparator} the configured bounds.")
    return value


def _environment_choice(name: str, *, default: str, choices: frozenset[str]) -> str:
    """Parse one lowercase closed-set value without reflecting its contents."""

    value = os.environ.get(name, default).strip().lower()
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise RuntimeError(f"{name} must be one of: {allowed}.")
    return value


# -- Paths --------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
# -- PostgreSQL ---------------------------------------------------------------

DATABASE_URL = os.environ.get("BDDK_DATABASE_URL", "")
OPERATOR_DATABASE_URL = os.environ.get("BDDK_OPERATOR_DATABASE_URL", "")
SCHEMA_OWNER_DATABASE_URL = os.environ.get("BDDK_SCHEMA_OWNER_DATABASE_URL", "")
INGESTION_DATABASE_URL = os.environ.get("BDDK_INGESTION_DATABASE_URL", "")
RELEASE_PUBLISHER_DATABASE_URL = os.environ.get("BDDK_RELEASE_PUBLISHER_DATABASE_URL", "")
RELEASE_VERIFIER_DATABASE_URL = os.environ.get("BDDK_RELEASE_VERIFIER_DATABASE_URL", "")
RELEASE_VERIFIER_REVISION_SHA256 = os.environ.get("BDDK_RELEASE_VERIFIER_REVISION_SHA256", "").strip()
RELEASE_VERIFIER_IMAGE_DIGEST = os.environ.get("BDDK_RELEASE_VERIFIER_IMAGE_DIGEST", "").strip()
RELEASE_VERIFICATION_VALIDITY_SECONDS = _environment_int(
    "BDDK_RELEASE_VERIFICATION_VALIDITY_SECONDS",
    default=900,
    minimum=60,
    maximum=3600,
)
EXPECTED_DATABASE_NAME = os.environ.get("BDDK_EXPECTED_DATABASE_NAME", "").strip()
_DATABASE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")


def require_expected_database_name() -> str:
    """Return the independently configured lifecycle target database name."""

    if not EXPECTED_DATABASE_NAME:
        raise RuntimeError(
            "BDDK_EXPECTED_DATABASE_NAME is required for database lifecycle operations. "
            "Set it independently from the connection URL to guard against a wrong-database deployment."
        )
    if not _DATABASE_NAME_RE.fullmatch(EXPECTED_DATABASE_NAME):
        raise RuntimeError(
            "BDDK_EXPECTED_DATABASE_NAME must be 1-63 characters using only letters, digits, '.', '_' or '-'."
        )
    return EXPECTED_DATABASE_NAME


def require_database_url(profile: str = "public") -> str:
    """Return the database URL assigned to one process profile.

    Call this at the start of any entry point (server, seed CLI, doc_sync CLI)
    that needs to open a pool.  Public serving uses ``BDDK_DATABASE_URL``;
    operator serving must use the separately provisioned
    ``BDDK_OPERATOR_DATABASE_URL`` so enabling mutation tools cannot silently
    reuse the public runtime identity.
    """
    normalized = profile.strip().lower()
    if normalized == "public":
        variable = "BDDK_DATABASE_URL"
        dsn = DATABASE_URL
    elif normalized == "operator":
        variable = "BDDK_OPERATOR_DATABASE_URL"
        dsn = OPERATOR_DATABASE_URL
    elif normalized == "schema-owner":
        variable = "BDDK_SCHEMA_OWNER_DATABASE_URL"
        dsn = SCHEMA_OWNER_DATABASE_URL
    elif normalized == "ingestion":
        variable = "BDDK_INGESTION_DATABASE_URL"
        dsn = INGESTION_DATABASE_URL
    elif normalized == "release-publisher":
        variable = "BDDK_RELEASE_PUBLISHER_DATABASE_URL"
        dsn = RELEASE_PUBLISHER_DATABASE_URL
    elif normalized == "release-verifier":
        variable = "BDDK_RELEASE_VERIFIER_DATABASE_URL"
        dsn = RELEASE_VERIFIER_DATABASE_URL
    else:
        raise RuntimeError(
            f"Unknown database profile {profile!r}; expected public, operator, schema-owner, ingestion, "
            "release-verifier, or release-publisher"
        )

    if not dsn:
        raise RuntimeError(
            f"{variable} is not set for the {normalized} process profile. "
            "Provision a PostgreSQL DSN with the least-privileged role for "
            "that profile before starting the server."
        )
    if normalized == "operator" and DATABASE_URL and dsn == DATABASE_URL:
        raise RuntimeError(
            "BDDK_OPERATOR_DATABASE_URL must not reuse BDDK_DATABASE_URL. "
            "Provision a distinct least-privileged operator database identity."
        )
    if normalized in {"schema-owner", "ingestion", "release-verifier", "release-publisher"}:
        other_identities = {
            name: value
            for name, value in {
                "BDDK_DATABASE_URL": DATABASE_URL,
                "BDDK_OPERATOR_DATABASE_URL": OPERATOR_DATABASE_URL,
                "BDDK_SCHEMA_OWNER_DATABASE_URL": SCHEMA_OWNER_DATABASE_URL,
                "BDDK_INGESTION_DATABASE_URL": INGESTION_DATABASE_URL,
                "BDDK_RELEASE_VERIFIER_DATABASE_URL": RELEASE_VERIFIER_DATABASE_URL,
                "BDDK_RELEASE_PUBLISHER_DATABASE_URL": RELEASE_PUBLISHER_DATABASE_URL,
            }.items()
            if value and name != variable
        }
        reused = next((name for name, value in other_identities.items() if value == dsn), None)
        if reused is not None:
            raise RuntimeError(
                f"{variable} must not reuse {reused}. Provision a distinct database identity for each lifecycle plane."
            )
    from bddk_mcp.db_transport import assert_database_transport

    return assert_database_transport(dsn)


# asyncpg pool settings
PG_POOL_MIN = _environment_int("BDDK_PG_POOL_MIN", default=2, minimum=1, maximum=100)
PG_POOL_MAX = _environment_int("BDDK_PG_POOL_MAX", default=10, minimum=1, maximum=1000)
if PG_POOL_MIN > PG_POOL_MAX:
    raise RuntimeError("BDDK_PG_POOL_MIN must not exceed BDDK_PG_POOL_MAX.")


# Local research can explicitly operate without a signed release identity.
# Bank/OpenShift profiles set this true so serving readiness requires the
# strict bootstrap activation recorded by migration v0005.
REQUIRE_ACTIVE_CORPUS_RELEASE = _environment_bool(
    "BDDK_REQUIRE_ACTIVE_CORPUS_RELEASE",
    default=False,
)

# -- Embedding model (offline-first) -----------------------------------------

# Path to a pre-downloaded model directory.  When set, the vector store loads
# from this local path instead of downloading from Hugging Face.
EMBEDDING_MODEL_PATH = os.environ.get("BDDK_EMBEDDING_MODEL_PATH", "")
EMBEDDING_MODEL_NAME = os.environ.get("BDDK_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
_DEFAULT_EMBEDDING_MODEL_REVISION = "d13f1b27baf31030b7fd040960d60d909913633f"
EMBEDDING_MODEL_REVISION = os.environ.get(
    "BDDK_EMBEDDING_MODEL_REVISION",
    _DEFAULT_EMBEDDING_MODEL_REVISION if EMBEDDING_MODEL_NAME == "intfloat/multilingual-e5-base" else "",
)

# -- OCR extraction backends -------------------------------------------------

# LightOnOCR-2-1B model path (offline-first; empty = download from HF)
LIGHTOCR_MODEL_PATH = os.environ.get("BDDK_LIGHTOCR_MODEL_PATH", "")
LIGHTOCR_MODEL_NAME = os.environ.get("BDDK_LIGHTOCR_MODEL", "lightonai/LightOnOCR-2-1B")

# Device: auto | cuda | cpu
LIGHTOCR_DEVICE = _environment_choice(
    "BDDK_LIGHTOCR_DEVICE",
    default="auto",
    choices=frozenset({"auto", "cpu", "cuda"}),
)

# Minimum extracted character count to accept a backend's output
OCR_MIN_CONTENT_LEN = _environment_int(
    "BDDK_OCR_MIN_CONTENT_LEN",
    default=500,
    minimum=1,
    maximum=10_000_000,
)

# -- Chandra2 (primary OCR, backfill-only, in-process HF) --------------------

CHANDRA_MODEL_NAME = os.environ.get("BDDK_CHANDRA_MODEL", "datalab-to/chandra-ocr-2")

# -- pgvector -----------------------------------------------------------------

# The immutable PostgreSQL migration stores public.vector(768), and the default
# E5 model emits exactly 768 values. Reject a misleading override instead of
# allowing ingestion to fail after an expensive embedding run.
EMBEDDING_DIMENSION = 768
_configured_embedding_dimension = os.environ.get("BDDK_EMBEDDING_DIM", str(EMBEDDING_DIMENSION))
try:
    if int(_configured_embedding_dimension) != EMBEDDING_DIMENSION:
        raise RuntimeError("BDDK_EMBEDDING_DIM must be 768 for the current immutable database schema.")
except ValueError:
    raise RuntimeError(
        "BDDK_EMBEDDING_DIM must be the integer 768 for the current immutable database schema."
    ) from None

# -- Document chunking -------------------------------------------------------

# Page size for paginated markdown output (client, doc_store, vector_store)
PAGE_SIZE = _environment_int("BDDK_PAGE_SIZE", default=5000, minimum=1, maximum=1_000_000)

# Embedding chunk size and overlap (vector_store only)
EMBEDDING_CHUNK_SIZE = _environment_int(
    "BDDK_EMBEDDING_CHUNK_SIZE",
    default=1000,
    minimum=1,
    maximum=1_000_000,
)
EMBEDDING_CHUNK_OVERLAP = _environment_int(
    "BDDK_EMBEDDING_CHUNK_OVERLAP",
    default=200,
    minimum=0,
    maximum=999_999,
)
EMBEDDING_CHUNK_MODE = _environment_choice(
    "BDDK_EMBEDDING_CHUNK_MODE",
    default="token",
    choices=frozenset({"character", "token"}),
)
EMBEDDING_CHUNK_TARGET_TOKENS = _environment_int(
    "BDDK_EMBEDDING_CHUNK_TARGET_TOKENS",
    default=400,
    minimum=1,
    maximum=1_000_000,
)
EMBEDDING_CHUNK_TOKEN_OVERLAP = _environment_int(
    "BDDK_EMBEDDING_CHUNK_TOKEN_OVERLAP",
    default=40,
    minimum=0,
    maximum=999_999,
)
if EMBEDDING_CHUNK_OVERLAP >= EMBEDDING_CHUNK_SIZE:
    raise RuntimeError("BDDK_EMBEDDING_CHUNK_OVERLAP must be smaller than BDDK_EMBEDDING_CHUNK_SIZE.")
if EMBEDDING_CHUNK_TOKEN_OVERLAP >= EMBEDDING_CHUNK_TARGET_TOKENS:
    raise RuntimeError("BDDK_EMBEDDING_CHUNK_TOKEN_OVERLAP must be smaller than BDDK_EMBEDDING_CHUNK_TARGET_TOKENS.")

# -- Cache --------------------------------------------------------------------

# Decision list cache TTL (seconds) -- how long before re-scraping BDDK pages
CACHE_TTL_SECONDS = _environment_int(
    "BDDK_CACHE_TTL",
    default=3600,
    minimum=0,
    maximum=31_536_000,
)

# Search result in-memory cache
SEARCH_CACHE_TTL = _environment_int(
    "BDDK_SEARCH_CACHE_TTL",
    default=300,
    minimum=0,
    maximum=31_536_000,
)
SEARCH_CACHE_MAX = _environment_int(
    "BDDK_SEARCH_CACHE_MAX",
    default=200,
    minimum=1,
    maximum=100_000,
)

# When BDDK is unreachable, serve stale DB cache even if TTL expired
STALE_CACHE_FALLBACK = _environment_bool("BDDK_STALE_CACHE_FALLBACK", default=True)

# -- Relevance thresholds (anti-hallucination) --------------------------------

SEMANTIC_RELEVANCE_THRESHOLD = _environment_float(
    "BDDK_SEMANTIC_THRESHOLD",
    default=0.50,
    minimum=-1.0,
    maximum=1.0,
)
FTS_RANK_THRESHOLD = _environment_float(
    "BDDK_FTS_THRESHOLD",
    default=0.01,
    minimum=0.0,
    maximum=1_000_000.0,
)

# -- Hybrid search (dense + sparse fusion) ------------------------------------

HYBRID_SEARCH = _environment_bool("BDDK_HYBRID_SEARCH", default=True)
HYBRID_RRF_K = _environment_int("BDDK_RRF_K", default=60, minimum=1, maximum=1_000_000)

# -- Cross-encoder re-ranking -------------------------------------------------

RERANKER_ENABLED = _environment_bool("BDDK_RERANKER", default=False)
RERANKER_MODEL_NAME = os.environ.get("BDDK_RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
RERANKER_MODEL_PATH = os.environ.get("BDDK_RERANKER_MODEL_PATH", "")
_DEFAULT_RERANKER_MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
RERANKER_MODEL_REVISION = os.environ.get(
    "BDDK_RERANKER_MODEL_REVISION",
    _DEFAULT_RERANKER_MODEL_REVISION if RERANKER_MODEL_NAME == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1" else "",
)
RERANKER_TOP_N = _environment_int("BDDK_RERANKER_TOP_N", default=20, minimum=1, maximum=10_000)


def validate_model_asset_policy(
    *,
    embedding_model_path: str | None = None,
    reranker_enabled: bool | None = None,
    reranker_model_path: str | None = None,
    hub_offline: bool | None = None,
) -> None:
    """Fail at startup, not inside a tool call, when model assets cannot load.

    The delivered container runs with HF_HUB_OFFLINE=1 and pre-baked model
    directories. Without this check, a missing local model path surfaces as a
    Hugging Face download failure at first search time; a misconfigured
    reranker flag does the same. Keyword overrides exist for tests only.
    """
    embedding_path = EMBEDDING_MODEL_PATH if embedding_model_path is None else embedding_model_path
    reranker_on = RERANKER_ENABLED if reranker_enabled is None else reranker_enabled
    reranker_path = RERANKER_MODEL_PATH if reranker_model_path is None else reranker_model_path
    offline = (
        (os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1")
        if hub_offline is None
        else hub_offline
    )

    if embedding_path and not Path(embedding_path).is_dir():
        raise RuntimeError("BDDK_EMBEDDING_MODEL_PATH does not point to an existing model directory.")
    if reranker_path and not Path(reranker_path).is_dir():
        raise RuntimeError("BDDK_RERANKER_MODEL_PATH does not point to an existing model directory.")
    if offline and not embedding_path:
        raise RuntimeError(
            "Offline model mode (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE) requires BDDK_EMBEDDING_MODEL_PATH; "
            "the embedding model cannot be downloaded at serving runtime."
        )
    if reranker_on and not reranker_path:
        raise RuntimeError(
            "BDDK_RERANKER=true requires BDDK_RERANKER_MODEL_PATH: the reranker model is not baked into "
            "the delivered image and cannot be downloaded at serving runtime. Provide a local model "
            "directory or disable the reranker."
        )


# -- HTTP ---------------------------------------------------------------------

REQUEST_TIMEOUT = _environment_float(
    "BDDK_REQUEST_TIMEOUT",
    default=60.0,
    minimum=0.0,
    maximum=3600.0,
    minimum_inclusive=False,
)
HTTP_CONNECT_TIMEOUT = _environment_float(
    "BDDK_HTTP_CONNECT_TIMEOUT",
    default=10.0,
    minimum=0.0,
    maximum=3600.0,
    minimum_inclusive=False,
)
HTTP_POOL_TIMEOUT = _environment_float(
    "BDDK_HTTP_POOL_TIMEOUT",
    default=10.0,
    minimum=0.0,
    maximum=3600.0,
    minimum_inclusive=False,
)
MAX_RETRIES = _environment_int("BDDK_MAX_RETRIES", default=3, minimum=1, maximum=20)

# -- Optional telemetry -------------------------------------------------------

# Disabled by default. When enabled, retrieval tools persist privacy-safe
# call traces for production retrieval debugging and benchmark comparison.
TELEMETRY_ENABLED = _environment_bool("BDDK_TELEMETRY_ENABLED", default=False)
TELEMETRY_DATABASE_URL = os.environ.get("BDDK_TELEMETRY_DATABASE_URL", "")
TELEMETRY_STORE_TEXT = _environment_bool("BDDK_TELEMETRY_STORE_TEXT", default=False)
TELEMETRY_MODEL_ID = os.environ.get("BDDK_TELEMETRY_MODEL_ID", "")
TELEMETRY_SESSION_ID = os.environ.get("BDDK_TELEMETRY_SESSION_ID", "")


def require_telemetry_database_url() -> str:
    """Return a dedicated telemetry-writer DSN when telemetry is enabled."""

    if not TELEMETRY_ENABLED:
        raise RuntimeError("Telemetry is disabled; no telemetry database identity should be opened")
    if not TELEMETRY_DATABASE_URL:
        raise RuntimeError(
            "BDDK_TELEMETRY_DATABASE_URL is required when BDDK_TELEMETRY_ENABLED=true. "
            "Provision a dedicated INSERT-only telemetry database identity."
        )
    if TELEMETRY_DATABASE_URL in {DATABASE_URL, OPERATOR_DATABASE_URL}:
        raise RuntimeError(
            "BDDK_TELEMETRY_DATABASE_URL must not reuse a public or operator database identity. "
            "Provision a dedicated INSERT-only telemetry role."
        )
    from bddk_mcp.db_transport import assert_database_transport

    return assert_database_transport(TELEMETRY_DATABASE_URL)


# -- Sync ---------------------------------------------------------------------

SYNC_CONCURRENCY = _environment_int("BDDK_SYNC_CONCURRENCY", default=5, minimum=1, maximum=100)
OPERATOR_JOB_DRAIN_TIMEOUT = _environment_float(
    "BDDK_OPERATOR_JOB_DRAIN_TIMEOUT",
    default=30.0,
    minimum=0.0,
    maximum=3600.0,
)
OPERATOR_JOB_HISTORY = _environment_int(
    "BDDK_OPERATOR_JOB_HISTORY",
    default=1000,
    minimum=1,
    maximum=1_000_000,
)

# Prefer the iframe/HTML download path over PDF for mevzuat.gov.tr documents.
# Values: "true" | "false" | "auto". "auto" flips to true when no GPU OCR
# backend is available — markitdown-on-PDF produces no formulas / no tables,
# so the rich HTML path is the better CPU-only choice.
PREFER_HTML_FOR_MEVZUAT = _environment_choice(
    "BDDK_PREFER_HTML_FOR_MEVZUAT",
    default="auto",
    choices=frozenset({"auto", "false", "true"}),
)

# -- BDDK announcements -------------------------------------------------------

# BDDK announcement category IDs surfaced on the public site.
# 39=Basın (press), 40=Mevzuat (regulation), 41=İnsan Kaynakları (HR),
# 42=Veri Yayınları (data), 48=Kuruluş (institution).
ANNOUNCEMENT_CATEGORY_IDS: tuple[int, ...] = (39, 40, 41, 42, 48)

# -- Validation helpers -------------------------------------------------------


def validate_metric_id(metric_id: str) -> str:
    """Validate and return a metric ID in X.X.X format.

    Raises ValueError if the format is invalid.
    """
    import re

    if not re.match(r"^\d+\.\d+\.\d+$", metric_id):
        raise ValueError(f"Invalid metric_id '{metric_id}'. Expected format: X.X.X (e.g. '1.0.1')")
    return metric_id


def validate_table_no(table_no: int) -> int:
    """Validate monthly bulletin table number (1-17)."""
    if not 1 <= table_no <= 17:
        raise ValueError(f"Invalid table_no {table_no}. Must be between 1 and 17.")
    return table_no


def validate_month(month: int) -> int:
    """Validate month number (1-12)."""
    if not 1 <= month <= 12:
        raise ValueError(f"Invalid month {month}. Must be between 1 and 12.")
    return month


def validate_year(year: int) -> int:
    """Validate year (reasonable range for BDDK data)."""
    if not 2000 <= year <= 2100:
        raise ValueError(f"Invalid year {year}. Must be between 2000 and 2100.")
    return year


def validate_currency(currency: str, bulletin_type: str = "weekly") -> str:
    """Validate currency parameter."""
    if bulletin_type == "weekly":
        valid = ("TRY", "USD")
    else:
        valid = ("TL", "USD")
    if currency not in valid:
        raise ValueError(f"Invalid currency '{currency}'. Must be one of: {', '.join(valid)}")
    return currency


def validate_column(column: str) -> str:
    """Validate bulletin column parameter."""
    if column not in ("1", "2", "3"):
        raise ValueError(f"Invalid column '{column}'. Must be '1' (TP), '2' (YP), or '3' (Toplam)")
    return column
