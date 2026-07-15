"""Fail-closed release-epoch guard for local-corpus MCP reads."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from mcp.server.fastmcp.exceptions import ToolError

from bddk_mcp.core.exceptions import BddkStorageError
from bddk_mcp.corpus_publication import CorpusPublicationError, inspect_active_corpus_release
from bddk_mcp.tools.registry import LOCAL_CORPUS_PUBLIC_TOOL_NAMES
from bddk_mcp.tools.search import clear_search_cache

if TYPE_CHECKING:
    from bddk_mcp.core.deps import Dependencies

logger = logging.getLogger(__name__)

_CORPUS_CHECK_TIMEOUT_SECONDS = 5.0
_CORPUS_UNAVAILABLE_ERROR = (
    "[ERROR:CORPUS_RELEASE_UNAVAILABLE] retryable=true\nA verified active regulatory corpus release is not available."
)


class ActiveCorpusGuard:
    """Bind local read results to one verified active corpus release.

    One dependency-scoped lock covers release inspection, any catalog reload,
    the tool body, and a final release inspection.  This intentionally favors
    correctness over local-corpus read concurrency in strict mode.  Calls that
    do not consume the local corpus bypass the lock entirely.
    """

    def __init__(self, deps: Dependencies, *, required: bool) -> None:
        self._deps = deps
        self._required = required

    def _invalidate_process_epoch(self) -> None:
        self._deps.served_corpus_release_id = None
        clear_search_cache()

    async def _active_release_id(self) -> str:
        pool = self._deps.pool
        if pool is None:
            raise CorpusPublicationError("Active corpus release evidence could not be verified.")
        async with asyncio.timeout(_CORPUS_CHECK_TIMEOUT_SECONDS):
            release = await inspect_active_corpus_release(pool)
        if release is None:
            raise CorpusPublicationError("Active corpus release evidence is unavailable.")
        return release.release_id

    def _safe_unavailable(self, error: BaseException) -> ToolError:
        logger.warning(
            "Strict local-corpus tool call rejected",
            extra={"error_type": type(error).__name__},
        )
        return ToolError(_CORPUS_UNAVAILABLE_ERROR)

    async def _prepare_epoch(self) -> str:
        """Verify the release and atomically load its exact database catalog."""
        try:
            release_id = await self._active_release_id()
        except (TimeoutError, CorpusPublicationError, OSError) as error:
            self._invalidate_process_epoch()
            raise self._safe_unavailable(error) from None

        if self._deps.served_corpus_release_id == release_id:
            return release_id

        # An old response must not survive even when the replacement catalog
        # load subsequently fails.  Guarded reads remain unavailable until a
        # complete reload is confirmed against the same active release ID.
        self._invalidate_process_epoch()
        client = self._deps.client
        if client is None:
            raise self._safe_unavailable(RuntimeError("client_unavailable")) from None
        try:
            await client.load_cache_read_only()
            confirmed_release_id = await self._active_release_id()
        except (TimeoutError, CorpusPublicationError, BddkStorageError, OSError) as error:
            self._invalidate_process_epoch()
            raise self._safe_unavailable(error) from None
        if confirmed_release_id != release_id:
            self._invalidate_process_epoch()
            raise self._safe_unavailable(RuntimeError("release_changed_during_reload")) from None

        self._deps.served_corpus_release_id = release_id
        return release_id

    async def _confirm_epoch(self, release_id: str) -> None:
        """Discard a completed result if its release ceased to be active."""
        try:
            confirmed_release_id = await self._active_release_id()
        except (TimeoutError, CorpusPublicationError, OSError) as error:
            self._invalidate_process_epoch()
            raise self._safe_unavailable(error) from None
        if confirmed_release_id != release_id:
            self._invalidate_process_epoch()
            raise self._safe_unavailable(RuntimeError("release_changed_during_tool_call")) from None

    @asynccontextmanager
    async def tool_call(self, tool_name: str) -> AsyncIterator[None]:
        """Guard one MCP tool call when it reads the local corpus."""
        if not self._required or tool_name not in LOCAL_CORPUS_PUBLIC_TOOL_NAMES:
            yield
            return

        async with self._deps.active_corpus_lock:
            release_id = await self._prepare_epoch()
            try:
                yield
            except BaseException:
                raise
            else:
                await self._confirm_epoch(release_id)


__all__ = ("ActiveCorpusGuard",)
