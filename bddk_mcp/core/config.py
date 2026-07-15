"""Unified configuration for BDDK MCP Server.

All tunable constants in one place. Values can be overridden via environment
variables (prefixed with BDDK_).
"""

import os
import re
from pathlib import Path

# -- Paths --------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
# -- PostgreSQL ---------------------------------------------------------------

DATABASE_URL = os.environ.get("BDDK_DATABASE_URL", "")
OPERATOR_DATABASE_URL = os.environ.get("BDDK_OPERATOR_DATABASE_URL", "")
SCHEMA_OWNER_DATABASE_URL = os.environ.get("BDDK_SCHEMA_OWNER_DATABASE_URL", "")
INGESTION_DATABASE_URL = os.environ.get("BDDK_INGESTION_DATABASE_URL", "")
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
    else:
        raise RuntimeError(
            f"Unknown database profile {profile!r}; expected public, operator, schema-owner, or ingestion"
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
    if normalized in {"schema-owner", "ingestion"}:
        other_identities = {
            name: value
            for name, value in {
                "BDDK_DATABASE_URL": DATABASE_URL,
                "BDDK_OPERATOR_DATABASE_URL": OPERATOR_DATABASE_URL,
                "BDDK_SCHEMA_OWNER_DATABASE_URL": SCHEMA_OWNER_DATABASE_URL,
                "BDDK_INGESTION_DATABASE_URL": INGESTION_DATABASE_URL,
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
PG_POOL_MIN = int(os.environ.get("BDDK_PG_POOL_MIN", "2"))
PG_POOL_MAX = int(os.environ.get("BDDK_PG_POOL_MAX", "10"))

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
LIGHTOCR_DEVICE = os.environ.get("BDDK_LIGHTOCR_DEVICE", "auto")

# Minimum extracted character count to accept a backend's output
OCR_MIN_CONTENT_LEN = int(os.environ.get("BDDK_OCR_MIN_CONTENT_LEN", "500"))

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
PAGE_SIZE = int(os.environ.get("BDDK_PAGE_SIZE", "5000"))

# Embedding chunk size and overlap (vector_store only)
EMBEDDING_CHUNK_SIZE = int(os.environ.get("BDDK_EMBEDDING_CHUNK_SIZE", "1000"))
EMBEDDING_CHUNK_OVERLAP = int(os.environ.get("BDDK_EMBEDDING_CHUNK_OVERLAP", "200"))
EMBEDDING_CHUNK_MODE = os.environ.get("BDDK_EMBEDDING_CHUNK_MODE", "token").lower()
EMBEDDING_CHUNK_TARGET_TOKENS = int(os.environ.get("BDDK_EMBEDDING_CHUNK_TARGET_TOKENS", "400"))
EMBEDDING_CHUNK_TOKEN_OVERLAP = int(os.environ.get("BDDK_EMBEDDING_CHUNK_TOKEN_OVERLAP", "40"))

# -- Cache --------------------------------------------------------------------

# Decision list cache TTL (seconds) -- how long before re-scraping BDDK pages
CACHE_TTL_SECONDS = int(os.environ.get("BDDK_CACHE_TTL", "3600"))

# Search result in-memory cache
SEARCH_CACHE_TTL = int(os.environ.get("BDDK_SEARCH_CACHE_TTL", "300"))
SEARCH_CACHE_MAX = int(os.environ.get("BDDK_SEARCH_CACHE_MAX", "200"))

# When BDDK is unreachable, serve stale DB cache even if TTL expired
STALE_CACHE_FALLBACK = os.environ.get("BDDK_STALE_CACHE_FALLBACK", "true").lower() in ("1", "true", "yes")

# -- Relevance thresholds (anti-hallucination) --------------------------------

SEMANTIC_RELEVANCE_THRESHOLD = float(os.environ.get("BDDK_SEMANTIC_THRESHOLD", "0.50"))
FTS_RANK_THRESHOLD = float(os.environ.get("BDDK_FTS_THRESHOLD", "0.01"))

# -- Hybrid search (dense + sparse fusion) ------------------------------------

HYBRID_SEARCH = os.environ.get("BDDK_HYBRID_SEARCH", "true").lower() in ("1", "true", "yes")
HYBRID_RRF_K = int(os.environ.get("BDDK_RRF_K", "60"))

# -- Cross-encoder re-ranking -------------------------------------------------

RERANKER_ENABLED = os.environ.get("BDDK_RERANKER", "false").lower() in ("1", "true", "yes")
RERANKER_MODEL_NAME = os.environ.get("BDDK_RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
RERANKER_MODEL_PATH = os.environ.get("BDDK_RERANKER_MODEL_PATH", "")
_DEFAULT_RERANKER_MODEL_REVISION = "1427fd652930e4ba29e8149678df786c240d8825"
RERANKER_MODEL_REVISION = os.environ.get(
    "BDDK_RERANKER_MODEL_REVISION",
    _DEFAULT_RERANKER_MODEL_REVISION if RERANKER_MODEL_NAME == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1" else "",
)
RERANKER_TOP_N = int(os.environ.get("BDDK_RERANKER_TOP_N", "20"))

# -- HTTP ---------------------------------------------------------------------

REQUEST_TIMEOUT = float(os.environ.get("BDDK_REQUEST_TIMEOUT", "60.0"))
HTTP_CONNECT_TIMEOUT = float(os.environ.get("BDDK_HTTP_CONNECT_TIMEOUT", "10.0"))
HTTP_POOL_TIMEOUT = float(os.environ.get("BDDK_HTTP_POOL_TIMEOUT", "10.0"))
MAX_RETRIES = int(os.environ.get("BDDK_MAX_RETRIES", "3"))

# -- Optional telemetry -------------------------------------------------------

# Disabled by default. When enabled, retrieval tools persist privacy-safe
# call traces for production retrieval debugging and benchmark comparison.
TELEMETRY_ENABLED = os.environ.get("BDDK_TELEMETRY_ENABLED", "false").lower() in ("1", "true", "yes")
TELEMETRY_DATABASE_URL = os.environ.get("BDDK_TELEMETRY_DATABASE_URL", "")
TELEMETRY_STORE_TEXT = os.environ.get("BDDK_TELEMETRY_STORE_TEXT", "false").lower() in ("1", "true", "yes")
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

AUTO_SYNC = os.environ.get("BDDK_AUTO_SYNC", "false").lower() in ("1", "true", "yes")
SYNC_CONCURRENCY = int(os.environ.get("BDDK_SYNC_CONCURRENCY", "5"))
OPERATOR_JOB_DRAIN_TIMEOUT = float(os.environ.get("BDDK_OPERATOR_JOB_DRAIN_TIMEOUT", "30"))
OPERATOR_JOB_HISTORY = int(os.environ.get("BDDK_OPERATOR_JOB_HISTORY", "1000"))

# Prefer the iframe/HTML download path over PDF for mevzuat.gov.tr documents.
# Values: "true" | "false" | "auto". "auto" flips to true when no GPU OCR
# backend is available — markitdown-on-PDF produces no formulas / no tables,
# so the rich HTML path is the better CPU-only choice.
PREFER_HTML_FOR_MEVZUAT = os.environ.get("BDDK_PREFER_HTML_FOR_MEVZUAT", "auto").lower()

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
