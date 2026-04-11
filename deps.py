"""Dependency container for BDDK MCP Server."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg
    import httpx

    from client import BddkApiClient
    from doc_store import DocumentStore
    from vector_store import VectorStore


@dataclass
class Dependencies:
    """Shared state for all tool modules.

    Created once at startup, injected into tool modules via register().
    Tools access dependencies through closure capture.
    """

    pool: asyncpg.Pool | None
    doc_store: DocumentStore | None
    client: BddkApiClient | None
    http: httpx.AsyncClient | None
    vector_store: VectorStore | None = None
    sync_task: asyncio.Task | None = None
    vector_init_task: asyncio.Task | None = None

    # Health state
    last_sync_time: float | None = None
    last_sync_error: str | None = None
    sync_consecutive_failures: int = 0
    sync_circuit_open: bool = False
    server_start_time: float = field(default_factory=time.time)
