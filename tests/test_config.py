"""Tests for unified config module."""

import os
import subprocess
import sys

import pytest

from bddk_mcp.core.config import (
    require_database_url,
    require_expected_database_name,
    require_telemetry_database_url,
    validate_column,
    validate_currency,
    validate_metric_id,
    validate_month,
    validate_table_no,
    validate_year,
)


@pytest.fixture(autouse=True)
def _allow_insecure_database_for_config_selection_tests(monkeypatch):
    monkeypatch.setenv("BDDK_ALLOW_INSECURE_DATABASE", "true")


def test_expected_database_name_is_required_and_bounded(monkeypatch):
    from bddk_mcp.core import config

    monkeypatch.setattr(config, "EXPECTED_DATABASE_NAME", "bddk_prod")
    assert require_expected_database_name() == "bddk_prod"

    monkeypatch.setattr(config, "EXPECTED_DATABASE_NAME", "")
    with pytest.raises(RuntimeError, match="BDDK_EXPECTED_DATABASE_NAME is required"):
        require_expected_database_name()

    monkeypatch.setattr(config, "EXPECTED_DATABASE_NAME", "unsafe database/name")
    with pytest.raises(RuntimeError, match="must be 1-63 characters"):
        require_expected_database_name()


def test_database_url_is_selected_by_process_profile(monkeypatch):
    from bddk_mcp.core import config

    monkeypatch.setattr(config, "DATABASE_URL", "postgresql://public")
    monkeypatch.setattr(config, "OPERATOR_DATABASE_URL", "postgresql://operator")
    monkeypatch.setattr(config, "SCHEMA_OWNER_DATABASE_URL", "postgresql://schema-owner")
    monkeypatch.setattr(config, "INGESTION_DATABASE_URL", "postgresql://ingestion")
    monkeypatch.setattr(config, "RELEASE_VERIFIER_DATABASE_URL", "postgresql://release-verifier")
    monkeypatch.setattr(config, "RELEASE_PUBLISHER_DATABASE_URL", "postgresql://release-publisher")

    assert require_database_url("public") == "postgresql://public"
    assert require_database_url("operator") == "postgresql://operator"
    assert require_database_url("schema-owner") == "postgresql://schema-owner"
    assert require_database_url("ingestion") == "postgresql://ingestion"
    assert require_database_url("release-verifier") == "postgresql://release-verifier"
    assert require_database_url("release-publisher") == "postgresql://release-publisher"


def test_operator_profile_requires_its_own_database_identity(monkeypatch):
    from bddk_mcp.core import config

    monkeypatch.setattr(config, "DATABASE_URL", "postgresql://public")
    monkeypatch.setattr(config, "OPERATOR_DATABASE_URL", "")

    with pytest.raises(RuntimeError, match="BDDK_OPERATOR_DATABASE_URL"):
        require_database_url("operator")


def test_operator_profile_rejects_the_public_database_identity(monkeypatch):
    from bddk_mcp.core import config

    shared_dsn = "postgresql://shared"
    monkeypatch.setattr(config, "DATABASE_URL", shared_dsn)
    monkeypatch.setattr(config, "OPERATOR_DATABASE_URL", shared_dsn)

    with pytest.raises(RuntimeError, match="must not reuse"):
        require_database_url("operator")


def test_unknown_database_profile_is_rejected():
    with pytest.raises(RuntimeError, match="Unknown database profile"):
        require_database_url("combined")


@pytest.mark.parametrize(
    ("profile", "variable"),
    [
        ("schema-owner", "SCHEMA_OWNER_DATABASE_URL"),
        ("ingestion", "INGESTION_DATABASE_URL"),
        ("release-verifier", "RELEASE_VERIFIER_DATABASE_URL"),
        ("release-publisher", "RELEASE_PUBLISHER_DATABASE_URL"),
    ],
)
def test_lifecycle_profiles_require_distinct_database_identities(monkeypatch, profile, variable):
    from bddk_mcp.core import config

    monkeypatch.setattr(config, "DATABASE_URL", "postgresql://public")
    monkeypatch.setattr(config, "OPERATOR_DATABASE_URL", "postgresql://operator")
    monkeypatch.setattr(config, "SCHEMA_OWNER_DATABASE_URL", "postgresql://schema-owner")
    monkeypatch.setattr(config, "INGESTION_DATABASE_URL", "postgresql://ingestion")
    monkeypatch.setattr(config, "RELEASE_VERIFIER_DATABASE_URL", "postgresql://release-verifier")
    monkeypatch.setattr(config, "RELEASE_PUBLISHER_DATABASE_URL", "postgresql://release-publisher")
    monkeypatch.setattr(config, variable, "postgresql://public")

    with pytest.raises(RuntimeError, match="must not reuse"):
        require_database_url(profile)


def test_enabled_telemetry_requires_dedicated_identity(monkeypatch):
    from bddk_mcp.core import config

    monkeypatch.setattr(config, "TELEMETRY_ENABLED", True)
    monkeypatch.setattr(config, "DATABASE_URL", "postgresql://public")
    monkeypatch.setattr(config, "OPERATOR_DATABASE_URL", "postgresql://operator")
    monkeypatch.setattr(config, "TELEMETRY_DATABASE_URL", "postgresql://telemetry")

    assert require_telemetry_database_url() == "postgresql://telemetry"


@pytest.mark.parametrize("dsn", ["", "postgresql://public", "postgresql://operator"])
def test_enabled_telemetry_rejects_missing_or_reused_identity(monkeypatch, dsn):
    from bddk_mcp.core import config

    monkeypatch.setattr(config, "TELEMETRY_ENABLED", True)
    monkeypatch.setattr(config, "DATABASE_URL", "postgresql://public")
    monkeypatch.setattr(config, "OPERATOR_DATABASE_URL", "postgresql://operator")
    monkeypatch.setattr(config, "TELEMETRY_DATABASE_URL", dsn)

    with pytest.raises(RuntimeError, match="BDDK_TELEMETRY_DATABASE_URL|must not reuse"):
        require_telemetry_database_url()


def _import_config_in_subprocess(**overrides: str) -> subprocess.CompletedProcess[str]:
    """Import config under an isolated BDDK environment."""

    environment = {key: value for key, value in os.environ.items() if not key.startswith("BDDK_")}
    environment.update(overrides)
    return subprocess.run(
        [sys.executable, "-c", "from bddk_mcp.core import config; print(config.PG_POOL_MAX)"],
        cwd=os.getcwd(),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ({"BDDK_PG_POOL_MIN": "0"}, "BDDK_PG_POOL_MIN must be an integer"),
        (
            {"BDDK_PG_POOL_MIN": "20", "BDDK_PG_POOL_MAX": "10"},
            "BDDK_PG_POOL_MIN must not exceed BDDK_PG_POOL_MAX",
        ),
        ({"BDDK_REQUEST_TIMEOUT": "nan"}, "BDDK_REQUEST_TIMEOUT must be finite"),
        ({"BDDK_TELEMETRY_ENABLED": "enabled"}, "BDDK_TELEMETRY_ENABLED must be one of"),
        ({"BDDK_LIGHTOCR_DEVICE": "remote"}, "BDDK_LIGHTOCR_DEVICE must be one of"),
        ({"BDDK_MAX_RETRIES": "0"}, "BDDK_MAX_RETRIES must be an integer"),
        (
            {"BDDK_EMBEDDING_CHUNK_SIZE": "100", "BDDK_EMBEDDING_CHUNK_OVERLAP": "100"},
            "BDDK_EMBEDDING_CHUNK_OVERLAP must be smaller",
        ),
        (
            {"BDDK_RELEASE_VERIFICATION_VALIDITY_SECONDS": "30"},
            "BDDK_RELEASE_VERIFICATION_VALIDITY_SECONDS must be an integer",
        ),
    ],
)
def test_environment_configuration_fails_closed(environment, message):
    result = _import_config_in_subprocess(**environment)

    assert result.returncode != 0
    assert message in result.stderr


def test_environment_configuration_accepts_explicit_boundary_values():
    result = _import_config_in_subprocess(
        BDDK_PG_POOL_MIN="1",
        BDDK_PG_POOL_MAX="1",
        BDDK_MAX_RETRIES="1",
        BDDK_SEARCH_CACHE_TTL="0",
        BDDK_OPERATOR_JOB_DRAIN_TIMEOUT="0",
        BDDK_HYBRID_SEARCH="no",
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
