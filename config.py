"""Unified configuration for BDDK MCP Server.

All tunable constants in one place. Values can be overridden via environment
variables (prefixed with BDDK_).
"""

import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
DB_PATH = Path(os.environ.get("BDDK_DB_PATH", BASE_DIR / "bddk_docs.db"))
CHROMA_PATH = Path(os.environ.get("BDDK_CHROMA_PATH", BASE_DIR / "chroma_db"))
CACHE_FILE = BASE_DIR / ".cache.json"

# ── Embedding model (offline-first) ─────────────────────────────────────────

# Path to a pre-downloaded model directory.  When set, the vector store loads
# from this local path instead of downloading from Hugging Face.
EMBEDDING_MODEL_PATH = os.environ.get("BDDK_EMBEDDING_MODEL_PATH", "")
EMBEDDING_MODEL_NAME = os.environ.get("BDDK_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
EMBEDDING_MODEL_REVISION = "d4210e50c0"  # v1.0.0 stable

# ── Document chunking ───────────────────────────────────────────────────────

# Page size for paginated markdown output (client, doc_store, vector_store)
PAGE_SIZE = int(os.environ.get("BDDK_PAGE_SIZE", "5000"))

# Embedding chunk size and overlap (vector_store only)
EMBEDDING_CHUNK_SIZE = int(os.environ.get("BDDK_EMBEDDING_CHUNK_SIZE", "1000"))
EMBEDDING_CHUNK_OVERLAP = int(os.environ.get("BDDK_EMBEDDING_CHUNK_OVERLAP", "200"))

# ── Cache ────────────────────────────────────────────────────────────────────

# Decision list cache TTL (seconds) — how long before re-scraping BDDK pages
CACHE_TTL_SECONDS = int(os.environ.get("BDDK_CACHE_TTL", "3600"))

# Search result in-memory cache
SEARCH_CACHE_TTL = int(os.environ.get("BDDK_SEARCH_CACHE_TTL", "300"))
SEARCH_CACHE_MAX = int(os.environ.get("BDDK_SEARCH_CACHE_MAX", "200"))

# When BDDK is unreachable, serve stale disk cache even if TTL expired
STALE_CACHE_FALLBACK = os.environ.get("BDDK_STALE_CACHE_FALLBACK", "true").lower() in ("1", "true", "yes")

# ── HTTP ─────────────────────────────────────────────────────────────────────

REQUEST_TIMEOUT = float(os.environ.get("BDDK_REQUEST_TIMEOUT", "60.0"))
MAX_RETRIES = int(os.environ.get("BDDK_MAX_RETRIES", "3"))

# ── Sync ─────────────────────────────────────────────────────────────────────

AUTO_SYNC = os.environ.get("BDDK_AUTO_SYNC", "").lower() in ("1", "true", "yes")
SYNC_CONCURRENCY = int(os.environ.get("BDDK_SYNC_CONCURRENCY", "5"))
PREFER_NOUGAT = os.environ.get("BDDK_PREFER_NOUGAT", "false").lower() in ("1", "true", "yes")

# ── Validation helpers ───────────────────────────────────────────────────────


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
