"""Dependency container for BDDK MCP Server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx

    from bddk_mcp.ingest.client import BddkApiClient
    from bddk_mcp.store.corpus_db import CorpusDB
    from bddk_mcp.store.doc_store import DocumentStore
    from bddk_mcp.store.vector_store import VectorStore


@dataclass
class Dependencies:
    """Shared state for all tool modules.

    Created once at startup, injected into tool modules via register().
    Tools access dependencies through closure capture.
    """

    db: CorpusDB | None
    doc_store: DocumentStore | None
    client: BddkApiClient | None
    http: httpx.AsyncClient | None
    vector_store: VectorStore | None = None
