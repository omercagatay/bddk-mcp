"""Tests for admin tool module registration."""

import time
from unittest.mock import MagicMock

from deps import Dependencies
from tools.admin import register


def test_admin_register():
    """admin.register() exposes the expected admin tools."""
    mcp = MagicMock()
    deps = Dependencies(
        pool=None,
        doc_store=None,
        client=None,
        http=None,
        server_start_time=time.time(),
    )
    register(mcp, deps)

    tool_names = {call.args[0].__name__ for call in mcp.tool.return_value.call_args_list}
    assert tool_names == {
        "health_check",
        "bddk_metrics",
        "document_quality_report",
        "backfill_degraded_documents",
        "backfill_status",
    }
