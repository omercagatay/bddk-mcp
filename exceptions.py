"""Custom exception hierarchy for BDDK MCP Server."""


class BddkError(Exception):
    """Base exception for all BDDK MCP errors."""


class BddkStorageError(BddkError):
    """Error during PostgreSQL storage operations."""


class BddkVectorStoreError(BddkStorageError):
    """Error specific to pgvector operations."""
