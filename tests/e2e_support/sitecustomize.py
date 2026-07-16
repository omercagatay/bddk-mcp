"""Child-process dependency stub for MCP transport end-to-end tests.

Python imports ``sitecustomize`` during interpreter startup when this directory
is placed on ``PYTHONPATH``.  The guard keeps the module inert outside the
explicit E2E child environment.  This avoids adding a database-bypass switch to
production code while still exercising the installed ``bddk-mcp`` command.
"""

from __future__ import annotations

import os
from pathlib import Path

if os.environ.get("BDDK_MCP_E2E_STUB") == "1":
    import bddk_mcp.server as server_module
    from bddk_mcp.core.deps import Dependencies

    class FakeDocumentStore:
        """Smallest local dependency needed by the representative tool call."""

        async def get_document_history(self, document_id: str) -> list[dict]:
            return []

    async def create_e2e_deps(_profile=None) -> Dependencies:
        return Dependencies(
            pool=None,
            doc_store=FakeDocumentStore(),
            client=None,
            http=None,
        )

    async def teardown_e2e_deps(_deps: Dependencies) -> None:
        sentinel = os.environ.get("BDDK_MCP_E2E_TEARDOWN_SENTINEL")
        if sentinel:
            Path(sentinel).write_text("closed\n", encoding="utf-8")

    server_module.create_deps = create_e2e_deps
    server_module.teardown_deps = teardown_e2e_deps
