"""Tests for admin tool module registration."""

import time
from unittest.mock import MagicMock

from deps import Dependencies


def test_admin_register():
    """admin.register() adds health_check and bddk_metrics tools."""
    mcp = MagicMock()
    deps = Dependencies(
        pool=None, doc_store=None, client=None, http=None,
        server_start_time=time.time(),
    )
    from tools.admin import register
    register(mcp, deps)
    assert mcp.tool.call_count >= 2
