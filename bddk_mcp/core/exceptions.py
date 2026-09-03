"""Custom exception hierarchy for BDDK MCP Server."""


class BddkError(Exception):
    """Base exception for all BDDK MCP errors."""


class BddkUpstreamError(BddkError):
    """A required upstream regulatory source could not be read completely."""


class BddkStorageError(BddkError):
    """Error during PostgreSQL storage operations."""
