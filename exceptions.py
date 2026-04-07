"""Custom exception hierarchy for BDDK MCP Server."""


class BddkError(Exception):
    """Base exception for all BDDK MCP errors."""


class BddkScrapingError(BddkError):
    """Error during web scraping or HTTP requests to BDDK/mevzuat."""


class BddkStorageError(BddkError):
    """Error during SQLite or ChromaDB storage operations."""


class BddkExtractionError(BddkError):
    """Error during document content extraction (PDF/HTML to Markdown)."""


class BddkCacheError(BddkError):
    """Error during cache load/save operations."""


class BddkVectorStoreError(BddkStorageError):
    """Error specific to ChromaDB vector store operations."""
