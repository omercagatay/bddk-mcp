"""Thin shim — delegates to bddk_mcp.ingest.seed for backward-compatible CLI.

Preserves `uv run python seed.py import/export` without modification.
"""

import logging

from bddk_mcp.ingest.seed import main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    main()
