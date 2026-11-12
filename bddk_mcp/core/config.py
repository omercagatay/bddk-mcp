"""Unified configuration for BDDK MCP Server.

All tunable constants in one place. Values can be overridden via environment
variables (prefixed with BDDK_).
"""

import math
import os
from pathlib import Path


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


# -- Corpus file --------------------------------------------------------------

# The whole corpus (documents, sections, chunks with embeddings, catalog) is one
# SQLite file.  Serving opens it read-only; `bddk-mcp build` and `bddk-mcp sync`
# write it.
CORPUS_PATH = os.environ.get("BDDK_CORPUS_PATH", "").strip() or "corpus/bddk.sqlite"

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

# -- Chandra2 (in-process HF backend; no sync path constructs it today) -------

CHANDRA_MODEL_NAME = os.environ.get("BDDK_CHANDRA_MODEL", "datalab-to/chandra-ocr-2")

# -- Embedding dimension ------------------------------------------------------

# The default E5 model emits exactly 768 values and every corpus file records
# the dimension it was built with. A fixed corpus-format constant, not a setting.
EMBEDDING_DIMENSION = 768

# -- Document chunking -------------------------------------------------------

# Page size for paginated markdown output (client, doc_store, vector_store)
PAGE_SIZE = _environment_int("BDDK_PAGE_SIZE", default=5000, minimum=1, maximum=1_000_000)

# Chunk size and overlap change the corpus format and are recorded nowhere in the
# corpus file, so they are constants: changing them means editing code and
# rebuilding the corpus.
EMBEDDING_CHUNK_TARGET_TOKENS = 400
EMBEDDING_CHUNK_TOKEN_OVERLAP = 40

# -- Cache --------------------------------------------------------------------

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

# -- Relevance thresholds (anti-hallucination) --------------------------------

SEMANTIC_RELEVANCE_THRESHOLD = _environment_float(
    "BDDK_SEMANTIC_THRESHOLD",
    default=0.50,
    minimum=-1.0,
    maximum=1.0,
)


def validate_model_asset_policy(
    *,
    embedding_model_path: str | None = None,
    hub_offline: bool | None = None,
) -> None:
    """Fail at startup, not inside a tool call, when model assets cannot load.

    The delivered container runs with HF_HUB_OFFLINE=1 and pre-baked model
    directories. Without this check, a missing local model path surfaces as a
    Hugging Face download failure at first search time. Keyword overrides exist
    for tests only.
    """
    embedding_path = EMBEDDING_MODEL_PATH if embedding_model_path is None else embedding_model_path
    if hub_offline is None:
        # Hugging Face treats several spellings as true; mirror that rather than
        # matching only "1", or an offline image would slip past this gate.
        truthy = {"1", "true", "yes", "on"}
        offline = any(
            (os.environ.get(name) or "").strip().lower() in truthy
            for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        )
    else:
        offline = hub_offline

    if embedding_path and not Path(embedding_path).is_dir():
        raise RuntimeError("BDDK_EMBEDDING_MODEL_PATH does not point to an existing model directory.")
    if offline and not embedding_path:
        raise RuntimeError(
            "Offline model mode (HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE) requires BDDK_EMBEDDING_MODEL_PATH; "
            "the embedding model cannot be downloaded at serving runtime."
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

# Prefer the iframe/HTML download path over PDF for mevzuat.gov.tr documents.
# Values: "true" | "false" | "auto". "auto" flips to true when every available
# OCR backend is a *_degraded one (pdftotext, markitdown): they produce no
# formulas / no tables, so the rich HTML path is the better CPU-only choice.
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
