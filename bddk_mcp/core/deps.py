"""Dependency container for BDDK MCP Server."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncpg
    import httpx

    from bddk_mcp.ingest.client import BddkApiClient
    from bddk_mcp.jobs import OperatorJobManager
    from bddk_mcp.store.doc_store import DocumentStore
    from bddk_mcp.store.vector_store import VectorStore


def _initially_idle_event() -> asyncio.Event:
    event = asyncio.Event()
    event.set()
    return event


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
    telemetry_pool: asyncpg.Pool | None = None
    vector_store: VectorStore | None = None
    job_manager: OperatorJobManager | None = None

    # Strict serving state shared by public and operator MCP surfaces that close
    # over this dependency container.  The lock protects short lease-state
    # transitions; same-epoch tool bodies execute concurrently.  A release
    # switch waits for ``active_corpus_idle`` before clearing/reloading caches.
    active_corpus_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    active_corpus_idle: asyncio.Event = field(default_factory=_initially_idle_event, repr=False)
    active_corpus_readers: int = 0
    served_corpus_release_id: str | None = None

    # Health state
    last_sync_time: float | None = None
    last_sync_error: str | None = None
    sync_consecutive_failures: int = 0
    sync_circuit_open: bool = False
    server_start_time: float = field(default_factory=time.time)
