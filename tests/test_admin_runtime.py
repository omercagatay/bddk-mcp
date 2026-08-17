from __future__ import annotations

import asyncio

import pytest

from bddk_mcp.admin.config import AdminConfigError
from bddk_mcp.admin.runtime import build_app_from_env


def test_build_app_requires_configuration() -> None:
    with pytest.raises(AdminConfigError):
        asyncio.run(build_app_from_env({}))
