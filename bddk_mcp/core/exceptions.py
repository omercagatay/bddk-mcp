"""Custom exception hierarchy for BDDK MCP Server."""


class BddkError(Exception):
    """Base exception for all BDDK MCP errors."""


class BddkUpstreamError(BddkError):
    """A required upstream regulatory source could not be read completely."""


class BddkUpstreamUnreachableError(BddkUpstreamError):
    """The upstream host itself could not be reached (transport or egress policy).

    Distinct from a per-resource failure: every resource on that host fails the
    same way, so serial callers abort remaining work instead of repeating a slow
    failure. A blocked bank egress path produces this class.
    """


class BddkStorageError(BddkError):
    """Error during PostgreSQL storage operations."""
