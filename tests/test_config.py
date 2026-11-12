"""Tests for unified config module."""

import os
import subprocess
import sys

import pytest

from bddk_mcp.core.config import (
    validate_column,
    validate_currency,
    validate_metric_id,
    validate_month,
    validate_table_no,
    validate_year,
)


def _import_config_in_subprocess(**overrides: str) -> subprocess.CompletedProcess[str]:
    """Import config under an isolated BDDK environment."""

    environment = {key: value for key, value in os.environ.items() if not key.startswith("BDDK_")}
    environment.update(overrides)
    return subprocess.run(
        [sys.executable, "-c", "from bddk_mcp.core import config; print(config.SEARCH_CACHE_MAX)"],
        cwd=os.getcwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"BDDK_SEARCH_CACHE_MAX": "0"}, "BDDK_SEARCH_CACHE_MAX must be an integer"),
        ({"BDDK_REQUEST_TIMEOUT": "nan"}, "BDDK_REQUEST_TIMEOUT must be finite"),
        ({"BDDK_LIGHTOCR_DEVICE": "remote"}, "BDDK_LIGHTOCR_DEVICE must be one of"),
        ({"BDDK_MAX_RETRIES": "0"}, "BDDK_MAX_RETRIES must be an integer"),
    ],
)
def test_environment_configuration_fails_closed(environment, message):
    result = _import_config_in_subprocess(**environment)

    assert result.returncode != 0
    assert message in result.stderr


def test_environment_configuration_accepts_explicit_boundary_values():
    result = _import_config_in_subprocess(
        BDDK_SEARCH_CACHE_MAX="1",
        BDDK_MAX_RETRIES="1",
        BDDK_SEARCH_CACHE_TTL="0",
    )

    assert result.returncode == 0, result.stderr


class TestValidateMetricId:
    def test_valid_metric_ids(self):
        assert validate_metric_id("1.0.1") == "1.0.1"
        assert validate_metric_id("1.0.10") == "1.0.10"
        assert validate_metric_id("2.3.4") == "2.3.4"

    def test_invalid_metric_ids(self):
        with pytest.raises(ValueError, match="Invalid metric_id"):
            validate_metric_id("bad")
        with pytest.raises(ValueError, match="Invalid metric_id"):
            validate_metric_id("1.0")
        with pytest.raises(ValueError, match="Invalid metric_id"):
            validate_metric_id("1.0.1.2")
        with pytest.raises(ValueError, match="Invalid metric_id"):
            validate_metric_id("")
        with pytest.raises(ValueError, match="Invalid metric_id"):
            validate_metric_id("abc.def.ghi")


class TestValidateTableNo:
    def test_valid_range(self):
        for i in range(1, 18):
            assert validate_table_no(i) == i

    def test_invalid_range(self):
        with pytest.raises(ValueError, match="Invalid table_no"):
            validate_table_no(0)
        with pytest.raises(ValueError, match="Invalid table_no"):
            validate_table_no(18)
        with pytest.raises(ValueError, match="Invalid table_no"):
            validate_table_no(-1)


class TestValidateMonth:
    def test_valid_months(self):
        for m in range(1, 13):
            assert validate_month(m) == m

    def test_invalid_months(self):
        with pytest.raises(ValueError):
            validate_month(0)
        with pytest.raises(ValueError):
            validate_month(13)


class TestValidateYear:
    def test_valid_years(self):
        assert validate_year(2024) == 2024
        assert validate_year(2000) == 2000
        assert validate_year(2100) == 2100

    def test_invalid_years(self):
        with pytest.raises(ValueError):
            validate_year(1999)
        with pytest.raises(ValueError):
            validate_year(2101)


class TestValidateCurrency:
    def test_weekly_currencies(self):
        assert validate_currency("TRY", "weekly") == "TRY"
        assert validate_currency("USD", "weekly") == "USD"

    def test_monthly_currencies(self):
        assert validate_currency("TL", "monthly") == "TL"
        assert validate_currency("USD", "monthly") == "USD"

    def test_invalid_currencies(self):
        with pytest.raises(ValueError):
            validate_currency("EUR", "weekly")
        with pytest.raises(ValueError):
            validate_currency("TRY", "monthly")  # TRY not valid for monthly


class TestValidateColumn:
    def test_valid_columns(self):
        assert validate_column("1") == "1"
        assert validate_column("2") == "2"
        assert validate_column("3") == "3"

    def test_invalid_columns(self):
        with pytest.raises(ValueError):
            validate_column("0")
        with pytest.raises(ValueError):
            validate_column("4")
        with pytest.raises(ValueError):
            validate_column("abc")


class TestChandraConfig:
    def test_chandra_model_name_default(self, monkeypatch):
        monkeypatch.delenv("BDDK_CHANDRA_MODEL", raising=False)
        import importlib

        from bddk_mcp.core import config

        importlib.reload(config)
        assert config.CHANDRA_MODEL_NAME == "datalab-to/chandra-ocr-2"

    def test_chandra_model_name_env_override(self, monkeypatch):
        monkeypatch.setenv("BDDK_CHANDRA_MODEL", "custom/model")
        import importlib

        from bddk_mcp.core import config

        importlib.reload(config)
        assert config.CHANDRA_MODEL_NAME == "custom/model"


# -- Model asset startup policy ----------------------------------------------


def test_model_asset_policy_accepts_online_defaults(tmp_path):
    from bddk_mcp.core.config import validate_model_asset_policy

    validate_model_asset_policy(
        embedding_model_path="",
        hub_offline=False,
    )


def test_model_asset_policy_rejects_offline_without_embedding_path():
    from bddk_mcp.core.config import validate_model_asset_policy

    with pytest.raises(RuntimeError, match="BDDK_EMBEDDING_MODEL_PATH"):
        validate_model_asset_policy(
            embedding_model_path="",
            hub_offline=True,
        )


def test_model_asset_policy_rejects_missing_embedding_directory(tmp_path):
    from bddk_mcp.core.config import validate_model_asset_policy

    with pytest.raises(RuntimeError, match="existing model directory"):
        validate_model_asset_policy(
            embedding_model_path=str(tmp_path / "missing"),
            hub_offline=False,
        )


def test_model_asset_policy_accepts_fully_local_offline_setup(tmp_path):
    from bddk_mcp.core.config import validate_model_asset_policy

    embedding_dir = tmp_path / "embedding"
    embedding_dir.mkdir()
    validate_model_asset_policy(
        embedding_model_path=str(embedding_dir),
        hub_offline=True,
    )


def test_model_asset_policy_reads_hf_offline_env_spellings(tmp_path, monkeypatch):
    """HF accepts several truthy spellings; the gate must not match only "1"."""
    from bddk_mcp.core.config import validate_model_asset_policy

    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("HF_HUB_OFFLINE", value)
        with pytest.raises(RuntimeError, match="BDDK_EMBEDDING_MODEL_PATH"):
            validate_model_asset_policy(
                embedding_model_path="",
            )

    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    validate_model_asset_policy(
        embedding_model_path="",
    )


def test_model_asset_policy_runs_in_the_serving_lifespan():
    """The gate must be wired into startup, not merely importable."""
    import inspect

    from bddk_mcp import server

    source = inspect.getsource(server)
    assert "validate_model_asset_policy()" in source
