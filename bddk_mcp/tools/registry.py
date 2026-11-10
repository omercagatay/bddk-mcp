"""Reviewed MCP tool surface and runtime registration."""

from __future__ import annotations

from mcp.types import ToolAnnotations
from pydantic import ConfigDict

from bddk_mcp.core.deps import Dependencies
from bddk_mcp.tools import analytics, bulletin, documents, search, sections

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

TOOL_ANNOTATIONS: dict[str, ToolAnnotations] = {
    # Closed-world reads over the local catalog and corpus.
    "search_bddk_regulations": _CLOSED_READ,
    "search_document_store": _CLOSED_READ,
    "get_bddk_document": _CLOSED_READ,
    "get_document_history": _CLOSED_READ,
    "get_document_section": _CLOSED_READ,
    "search_document_sections": _CLOSED_READ,
    # Read-only calls that consult BDDK or mevzuat services.
    "search_bddk_institutions": _OPEN_READ,
    "search_bddk_announcements": _OPEN_READ,
    "get_bddk_bulletin": _OPEN_READ,
    "get_bddk_bulletin_snapshot": _OPEN_READ,
    "get_bddk_monthly": _OPEN_READ,
    "analyze_bulletin_trends": _OPEN_READ,
    "get_regulatory_digest": _OPEN_READ,
    "compare_bulletin_metrics": _OPEN_READ,
}


def _apply_tool_contracts(server) -> None:
    """Apply registry-owned annotations and strict argument-object policy.

    FastMCP builds a private Pydantic model for each function signature.  Its
    default extra-field policy is ``ignore``, which makes misspelled or stale
    client arguments disappear silently.  The MCP contract is fail-closed, so
    rebuild each generated model with ``extra='forbid'`` and refresh the schema
    exposed by ``tools/list``.
    """
    for tool_name in PUBLIC_TOOL_NAMES:
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


def register_tools(server, deps: Dependencies) -> None:
    """Register the complete reviewed tool surface on a FastMCP server."""
    search.register(server, deps)
    documents.register(server, deps)
    sections.register(server, deps)
    bulletin.register(server, deps)
    analytics.register(server, deps)
    _apply_tool_contracts(server)


def registered_tool_names(server) -> tuple[str, ...]:
    """Read the names registered by FastMCP without opening runtime dependencies."""
    return tuple(tool.name for tool in server._tool_manager.list_tools())


def assert_tool_surface(server) -> None:
    """Fail fast if implementation registration drifts from the reviewed surface."""
    actual = set(registered_tool_names(server))
    expected = set(PUBLIC_TOOL_NAMES)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(f"MCP tool surface drifted: missing={missing}, unexpected={unexpected}")
