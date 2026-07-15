"""Canonical MCP tool profiles and runtime registration."""

from __future__ import annotations

from enum import StrEnum

from mcp.types import ToolAnnotations
from pydantic import ConfigDict

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
    "get_operator_job",
    "list_operator_jobs",
    "cancel_operator_job",
    "document_health",
    "health_check",
    "bddk_metrics",
    "backfill_degraded_documents",
    "document_quality_report",
)


def _tool_annotations(
    *,
    read_only: bool,
    destructive: bool,
    idempotent: bool,
    open_world: bool,
) -> ToolAnnotations:
    """Build a complete, explicit MCP risk annotation set."""
    return ToolAnnotations(
        readOnlyHint=read_only,
        destructiveHint=destructive,
        idempotentHint=idempotent,
        openWorldHint=open_world,
    )


_CLOSED_READ = _tool_annotations(
    read_only=True,
    destructive=False,
    idempotent=True,
    open_world=False,
)
_OPEN_READ = _tool_annotations(
    read_only=True,
    destructive=False,
    idempotent=True,
    open_world=True,
)

# These annotations deliberately describe the most privileged behavior a tool
# can perform.  In particular, backfill_degraded_documents remains mutating and
# destructive even though its default invocation is a dry run.
TOOL_ANNOTATIONS: dict[str, ToolAnnotations] = {
    # Closed-world reads over the local catalog, corpus, and server state.
    "search_bddk_regulations": _CLOSED_READ,
    "search_document_store": _CLOSED_READ,
    "get_bddk_document": _CLOSED_READ,
    "get_document_history": _CLOSED_READ,
    "get_document_section": _CLOSED_READ,
    "search_document_sections": _CLOSED_READ,
    "document_store_stats": _CLOSED_READ,
    "bddk_cache_status": _CLOSED_READ,
    "document_health": _CLOSED_READ,
    "health_check": _CLOSED_READ,
    "bddk_metrics": _CLOSED_READ,
    "get_operator_job": _CLOSED_READ,
    "list_operator_jobs": _CLOSED_READ,
    "document_quality_report": _CLOSED_READ,
    # Read-only calls that consult BDDK or mevzuat services.
    "search_bddk_institutions": _OPEN_READ,
    "search_bddk_announcements": _OPEN_READ,
    "get_bddk_bulletin": _OPEN_READ,
    "get_bddk_bulletin_snapshot": _OPEN_READ,
    "get_bddk_monthly": _OPEN_READ,
    "analyze_bulletin_trends": _OPEN_READ,
    "get_regulatory_digest": _OPEN_READ,
    "compare_bulletin_metrics": _OPEN_READ,
    # Stateful monitoring and operator mutations.
    "check_bddk_updates": _tool_annotations(
        read_only=False,
        destructive=False,
        idempotent=False,
        open_world=True,
    ),
    "refresh_bddk_cache": _tool_annotations(
        read_only=False,
        destructive=True,
        idempotent=False,
        open_world=True,
    ),
    "sync_bddk_documents": _tool_annotations(
        read_only=False,
        destructive=True,
        idempotent=False,
        open_world=True,
    ),
    "trigger_startup_sync": _tool_annotations(
        read_only=False,
        destructive=True,
        idempotent=False,
        open_world=True,
    ),
    "backfill_degraded_documents": _tool_annotations(
        read_only=False,
        destructive=True,
        idempotent=False,
        open_world=True,
    ),
    "cancel_operator_job": _tool_annotations(
        read_only=False,
        destructive=True,
        idempotent=True,
        open_world=False,
    ),
}


def expected_tool_names(profile: ToolProfile) -> tuple[str, ...]:
    """Return the reviewed name snapshot for a runtime profile."""
    if profile is ToolProfile.PUBLIC:
        return PUBLIC_TOOL_NAMES
    return PUBLIC_TOOL_NAMES + OPERATOR_TOOL_NAMES


def _apply_tool_contracts(server, profile: ToolProfile) -> None:
    """Apply registry-owned annotations and strict argument-object policy.

    FastMCP builds a private Pydantic model for each function signature.  Its
    default extra-field policy is ``ignore``, which makes misspelled or stale
    client arguments disappear silently.  The MCP contract is fail-closed, so
    rebuild each generated model with ``extra='forbid'`` and refresh the schema
    exposed by ``tools/list``.
    """
    for tool_name in expected_tool_names(profile):
        tool = server._tool_manager.get_tool(tool_name)
        if tool is None:
            raise RuntimeError(f"MCP tool {tool_name!r} was not registered before contract application")

        annotations = TOOL_ANNOTATIONS.get(tool_name)
        if annotations is None:
            raise RuntimeError(f"MCP tool {tool_name!r} has no reviewed risk annotations")
        tool.annotations = annotations.model_copy(deep=True)

        argument_model = tool.fn_metadata.arg_model
        model_config = dict(argument_model.model_config)
        model_config["extra"] = "forbid"
        argument_model.model_config = ConfigDict(**model_config)
        argument_model.model_rebuild(force=True)
        tool.parameters = argument_model.model_json_schema(by_alias=True)


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
    _apply_tool_contracts(server, profile)


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
