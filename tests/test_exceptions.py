"""Tests for exceptions.py and logging_config.py."""

import json
import logging

from exceptions import (
    BddkCacheError,
    BddkError,
    BddkExtractionError,
    BddkScrapingError,
    BddkStorageError,
    BddkVectorStoreError,
)
from logging_config import (
    HumanFormatter,
    JsonFormatter,
    configure_logging,
    get_correlation_id,
    set_correlation_id,
)


class TestExceptionHierarchy:
    def test_base_exception(self):
        with __import__("pytest").raises(BddkError):
            raise BddkError("base error")

    def test_scraping_error_is_bddk_error(self):
        assert issubclass(BddkScrapingError, BddkError)

    def test_storage_error_is_bddk_error(self):
        assert issubclass(BddkStorageError, BddkError)

    def test_extraction_error_is_bddk_error(self):
        assert issubclass(BddkExtractionError, BddkError)

    def test_cache_error_is_bddk_error(self):
        assert issubclass(BddkCacheError, BddkError)

    def test_vector_store_error_is_storage_error(self):
        assert issubclass(BddkVectorStoreError, BddkStorageError)
        assert issubclass(BddkVectorStoreError, BddkError)

    def test_catch_specific_as_base(self):
        try:
            raise BddkScrapingError("test")
        except BddkError as e:
            assert str(e) == "test"


class TestCorrelationId:
    def test_get_generates_id(self):
        set_correlation_id("")
        cid = get_correlation_id()
        assert len(cid) == 12

    def test_set_and_get(self):
        set_correlation_id("test123")
        assert get_correlation_id() == "test123"
        set_correlation_id("")  # cleanup


class TestJsonFormatter:
    def test_format_output_is_valid_json(self):
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test message",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "test message"
        assert parsed["logger"] == "test"

    def test_format_includes_correlation_id(self):
        set_correlation_id("abc123")
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="msg",
            args=(),
            exc_info=None,
        )
        parsed = json.loads(formatter.format(record))
        assert parsed["correlation_id"] == "abc123"
        set_correlation_id("")


class TestConfigureLogging:
    def test_configure_human(self):
        configure_logging(json_output=False)
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, HumanFormatter)

    def test_configure_json(self):
        configure_logging(json_output=True)
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
        # Restore human for other tests
        configure_logging(json_output=False)
