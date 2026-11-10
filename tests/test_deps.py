"""Tests for the Dependencies container."""

from bddk_mcp.core.deps import Dependencies


def test_dependencies_defaults():
    """Dependencies initializes with correct defaults."""
    deps = Dependencies(
        db=None,
        doc_store=None,
        client=None,
        http=None,
    )
    assert deps.vector_store is None


def test_all_tool_modules_importable():
    """All tool modules can be imported and have register()."""
    from bddk_mcp.tools import analytics, bulletin, documents, search, sections

    for mod in [analytics, bulletin, documents, search, sections]:
        assert hasattr(mod, "register"), f"{mod.__name__} missing register()"
        assert callable(mod.register)
