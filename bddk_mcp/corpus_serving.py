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

    A dependency-scoped reader lease lets calls for the same verified epoch run
    concurrently.  Epoch replacement is exclusive: it waits until every prior
    reader has completed before clearing and reloading process-local caches.
    Calls that do not consume the local corpus bypass the guard entirely.
    """

    def __init__(self, deps: Dependencies, *, required: bool) -> None:
        self._deps = deps
        self._required = required

    def _invalidate_process_epoch(self) -> None:
        """Fail closed, deferring cache clear until every reader has drained.

        The caller must hold ``active_corpus_lock``.  A body may still be using
        or writing the shared cache, so clearing it while readers remain would
        allow a late stale write to survive the invalidation.
        """

        self._deps.served_corpus_release_id = None
        if self._deps.active_corpus_readers == 0:
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

    def _start_reader(self) -> None:
        if self._deps.active_corpus_readers == 0:
            self._deps.active_corpus_idle.clear()
        self._deps.active_corpus_readers += 1

    async def _reload_epoch(self, release_id: str) -> None:
        """Load ``release_id`` while holding the state lock with zero readers."""

        if self._deps.active_corpus_readers:
            raise RuntimeError("active corpus reload attempted with live readers")

        # An old response must not survive even when the replacement catalog
        # load subsequently fails.  Guarded reads remain unavailable until a
        # complete reload is confirmed against the same active release ID.
        self._invalidate_process_epoch()
        client = self._deps.client
        if client is None:
            raise self._safe_unavailable(RuntimeError("client_unavailable")) from None
        try:
            await client.load_cache_read_only(require_nonempty=False)
            confirmed_release_id = await self._active_release_id()
        except asyncio.CancelledError:
            self._invalidate_process_epoch()
            raise
        except (TimeoutError, CorpusPublicationError, BddkStorageError, OSError) as error:
            self._invalidate_process_epoch()
            raise self._safe_unavailable(error) from None
        if confirmed_release_id != release_id:
            self._invalidate_process_epoch()
            raise self._safe_unavailable(RuntimeError("release_changed_during_reload")) from None

        self._deps.served_corpus_release_id = release_id

    async def _start_epoch_lease(self) -> str:
        """Verify the active release, switching only after old readers drain."""

        while True:
            wait_for_idle: asyncio.Event | None = None
            async with self._deps.active_corpus_lock:
                try:
                    release_id = await self._active_release_id()
                except (TimeoutError, CorpusPublicationError, OSError) as error:
                    self._invalidate_process_epoch()
                    raise self._safe_unavailable(error) from None

                if self._deps.served_corpus_release_id != release_id:
                    if self._deps.active_corpus_readers:
                        wait_for_idle = self._deps.active_corpus_idle
                    else:
                        await self._reload_epoch(release_id)
                        self._start_reader()
                        return release_id
                else:
                    self._start_reader()
                    return release_id

            # Never hold the state lock while waiting: finishing readers need
            # it to decrement the shared count and signal the event.
            assert wait_for_idle is not None
            await wait_for_idle.wait()

    async def _finish_epoch_lease(self, release_id: str) -> None:
        """Post-check the release and always return one reader lease."""

        confirmation_error: BaseException | None = None
        confirmed_release_id: str | None = None
        try:
            confirmed_release_id = await self._active_release_id()
        except (TimeoutError, CorpusPublicationError, OSError) as error:
            confirmation_error = error

        async with self._deps.active_corpus_lock:
            if self._deps.active_corpus_readers <= 0:
                raise RuntimeError("active corpus reader lease underflow")

            if confirmation_error is not None or confirmed_release_id != release_id:
                self._invalidate_process_epoch()

            self._deps.active_corpus_readers -= 1
            if self._deps.active_corpus_readers == 0:
                # Clear only after the last body can no longer write stale
                # entries, then release any epoch-switch waiter.
                if self._deps.served_corpus_release_id is None:
                    clear_search_cache()
                self._deps.active_corpus_idle.set()

        if confirmation_error is not None:
            raise self._safe_unavailable(confirmation_error) from None
        if confirmed_release_id != release_id:
            raise self._safe_unavailable(RuntimeError("release_changed_during_tool_call")) from None

    @asynccontextmanager
    async def tool_call(self, tool_name: str) -> AsyncIterator[None]:
        """Guard one MCP tool call when it reads the local corpus."""
        if not self._required or tool_name not in LOCAL_CORPUS_PUBLIC_TOOL_NAMES:
            yield
            return

        release_id = await self._start_epoch_lease()
        body_failed = False
        try:
            yield
        except BaseException:
            body_failed = True
            raise
        finally:
            # Shield lease return so ordinary task cancellation cannot strand a
            # reader and permanently block a later epoch switch.
            finish_task = asyncio.create_task(self._finish_epoch_lease(release_id))
            try:
                await asyncio.shield(finish_task)
            except asyncio.CancelledError:
                await asyncio.shield(finish_task)
                raise
            except ToolError:
                if not body_failed:
                    raise


__all__ = ("ActiveCorpusGuard",)
