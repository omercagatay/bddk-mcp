"""Thin shim — delegates to bddk_mcp.server for backward-compatible launch commands.

Preserves `uv run python server.py`, `uv run mcp run server.py`, Procfile,
Dockerfile, and .mcp.json entry points without modification.
"""

from bddk_mcp.server import main, mcp  # noqa: F401

if __name__ == "__main__":
    main()
