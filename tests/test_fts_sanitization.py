"""Tests for FTS5 query sanitization in DocumentStore."""

import pytest

from doc_store import DocumentStore, StoredDocument


class TestFtsSanitization:
    """Test the _sanitize_fts5_term static method."""

    def test_normal_term(self):
        assert DocumentStore._sanitize_fts5_term("sermaye") == "sermaye"

    def test_strips_quotes(self):
        assert DocumentStore._sanitize_fts5_term('"test"') == "test"

    def test_strips_asterisk(self):
        assert DocumentStore._sanitize_fts5_term("test*") == "test"

    def test_strips_parentheses(self):
        assert DocumentStore._sanitize_fts5_term("(test)") == "test"

    def test_strips_plus_minus(self):
        assert DocumentStore._sanitize_fts5_term("+test-") == "test"

    def test_rejects_and_operator(self):
        assert DocumentStore._sanitize_fts5_term("AND") == ""

    def test_rejects_or_operator(self):
        assert DocumentStore._sanitize_fts5_term("OR") == ""

    def test_rejects_not_operator(self):
        assert DocumentStore._sanitize_fts5_term("NOT") == ""

    def test_rejects_near_operator(self):
        assert DocumentStore._sanitize_fts5_term("NEAR") == ""

    def test_case_insensitive_operator_rejection(self):
        assert DocumentStore._sanitize_fts5_term("and") == ""
        assert DocumentStore._sanitize_fts5_term("Or") == ""

    def test_normal_words_not_rejected(self):
        # Words containing operator substrings should not be rejected
        assert DocumentStore._sanitize_fts5_term("android") == "android"
        assert DocumentStore._sanitize_fts5_term("notion") == "notion"

    def test_empty_string(self):
        assert DocumentStore._sanitize_fts5_term("") == ""


class TestFtsSearchSanitized:
    """Integration tests: FTS search with potentially dangerous input."""

    @pytest.fixture
    async def store_with_data(self, tmp_path):
        db_path = tmp_path / "fts_test.db"
        s = DocumentStore(db_path=db_path)
        await s.initialize()
        await s.store_document(
            StoredDocument(
                document_id="1",
                title="Sermaye Yeterliliği",
                markdown_content="Bankacılık sermaye yeterliliği hesaplama rehberi.",
            )
        )
        yield s
        await s.close()

    @pytest.mark.asyncio
    async def test_search_with_quotes(self, store_with_data):
        hits = await store_with_data.search_content('"sermaye"')
        assert len(hits) >= 1

    @pytest.mark.asyncio
    async def test_search_with_operators(self, store_with_data):
        # Should not crash, operators get stripped
        hits = await store_with_data.search_content("sermaye AND OR NOT")
        assert isinstance(hits, list)

    @pytest.mark.asyncio
    async def test_search_with_special_chars(self, store_with_data):
        hits = await store_with_data.search_content("sermaye* (test) +risk-")
        assert isinstance(hits, list)

    @pytest.mark.asyncio
    async def test_search_all_operators_returns_empty(self, store_with_data):
        # If all terms are operators, should return empty
        hits = await store_with_data.search_content("AND OR NOT")
        assert hits == []
