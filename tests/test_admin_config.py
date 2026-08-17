from __future__ import annotations

import pytest

from bddk_mcp.admin.config import AdminConfig, AdminConfigError


def test_defaults_to_loopback() -> None:
    config = AdminConfig.from_env({"BDDK_DATABASE_URL": "postgresql://u:p@localhost:5432/bddk"})
    assert config.bind_host == "127.0.0.1"
    assert config.loopback_only is True
    assert config.port == 8100


def test_database_url_is_required() -> None:
    with pytest.raises(AdminConfigError, match="BDDK_DATABASE_URL"):
        AdminConfig.from_env({})


def test_non_loopback_bind_without_authentication_fails_closed() -> None:
    env = {
        "BDDK_DATABASE_URL": "postgresql://u:p@localhost:5432/bddk",
        "BDDK_ADMIN_HOST": "0.0.0.0",
    }
    with pytest.raises(AdminConfigError, match="authenticated or loopback-only"):
        AdminConfig.from_env(env)


def test_error_never_leaks_the_database_url() -> None:
    env = {"BDDK_DATABASE_URL": "postgresql://user:sup3rsecret@host:5432/bddk", "BDDK_ADMIN_HOST": "0.0.0.0"}
    with pytest.raises(AdminConfigError) as excinfo:
        AdminConfig.from_env(env)
    assert "sup3rsecret" not in str(excinfo.value)
