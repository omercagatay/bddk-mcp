"""Tests for unified config module."""

import pytest

from config import (
    validate_column,
    validate_currency,
    validate_metric_id,
    validate_month,
    validate_table_no,
    validate_year,
)


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
