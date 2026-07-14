"""Canonical MCP tool profiles and runtime registration."""

from __future__ import annotations

from enum import StrEnum

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.tools import admin, analytics, bulletin, documents, search, sections, sync


class ToolProfile(StrEnum):
    """Supported runtime tool surfaces."""

    PUBLIC = "public"
    OPERATOR = "operator"


PUBLIC_TOOL_NAMES: tuple[str, ...] = (
    "search_bddk_regulations",
    "search_bddk_institutions",
    "search_bddk_announcements",
    "search_document_store",
    "get_bddk_document",
    "get_document_history",
    "get_document_section",
    "search_document_sections",
    "get_bddk_bulletin",
    "get_bddk_bulletin_snapshot",
    "get_bddk_monthly",
    "analyze_bulletin_trends",
    "get_regulatory_digest",
    "compare_bulletin_metrics",
    "check_bddk_updates",
)

OPERATOR_TOOL_NAMES: tuple[str, ...] = (
    "document_store_stats",
    "bddk_cache_status",
    "refresh_bddk_cache",
    "sync_bddk_documents",
    "trigger_startup_sync",
    "document_health",
    "health_check",
    "bddk_metrics",
    "backfill_degraded_documents",
    "backfill_status",
    "document_quality_report",
)


def expected_tool_names(profile: ToolProfile) -> tuple[str, ...]:
    """Return the reviewed name snapshot for a runtime profile."""
    if profile is ToolProfile.PUBLIC:
        return PUBLIC_TOOL_NAMES
    return PUBLIC_TOOL_NAMES + OPERATOR_TOOL_NAMES


def register_tool_profile(server, deps: Dependencies, profile: ToolProfile) -> None:
    """Register one complete, explicit tool profile on a FastMCP server."""
    include_operator = profile is ToolProfile.OPERATOR
    search.register(server, deps)
    documents.register(server, deps, include_operator=include_operator)
    sections.register(server, deps)
    bulletin.register(server, deps, include_operator=include_operator)
    analytics.register(server, deps)
    if include_operator:
        sync.register(server, deps)
        admin.register(server, deps)


def registered_tool_names(server) -> tuple[str, ...]:
    """Read the names registered by FastMCP without opening runtime dependencies."""
    return tuple(tool.name for tool in server._tool_manager.list_tools())


def assert_tool_profile(server, profile: ToolProfile) -> None:
    """Fail fast if implementation registration drifts from the reviewed profile."""
    actual = set(registered_tool_names(server))
    expected = set(expected_tool_names(profile))
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(f"MCP tool profile {profile.value!r} drifted: missing={missing}, unexpected={unexpected}")
