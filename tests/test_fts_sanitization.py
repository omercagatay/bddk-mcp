"""Tests for FTS query sanitization in DocumentStore."""

import pytest

from doc_store import DocumentStore, StoredDocument


class TestFtsSanitization:
    """Test the _sanitize_fts_term static method."""

    def test_normal_term(self):
        assert DocumentStore._sanitize_fts_term("sermaye") == "sermaye"

    def test_strips_quotes(self):
        assert DocumentStore._sanitize_fts_term('"test"') == "test"

    def test_strips_asterisk(self):
        assert DocumentStore._sanitize_fts_term("test*") == "test"

    def test_strips_parentheses(self):
        assert DocumentStore._sanitize_fts_term("(test)") == "test"

    def test_strips_plus_minus(self):
        assert DocumentStore._sanitize_fts_term("+test-") == "test"

    def test_rejects_and_operator(self):
        assert DocumentStore._sanitize_fts_term("AND") == ""

    def test_rejects_or_operator(self):
        assert DocumentStore._sanitize_fts_term("OR") == ""

    def test_rejects_not_operator(self):
        assert DocumentStore._sanitize_fts_term("NOT") == ""

    def test_rejects_near_operator(self):
        assert DocumentStore._sanitize_fts_term("NEAR") == ""

    def test_case_insensitive_operator_rejection(self):
        assert DocumentStore._sanitize_fts_term("and") == ""
        assert DocumentStore._sanitize_fts_term("Or") == ""

    def test_normal_words_not_rejected(self):
        assert DocumentStore._sanitize_fts_term("android") == "android"
        assert DocumentStore._sanitize_fts_term("notion") == "notion"

    def test_empty_string(self):
        assert DocumentStore._sanitize_fts_term("") == ""


class TestFtsSearchSanitized:
    """Integration tests: FTS search with potentially dangerous input."""

    @pytest.mark.asyncio
    async def test_search_with_quotes(self, doc_store):
        await doc_store.store_document(
            StoredDocument(
                document_id="1",
                title="Sermaye Yeterliliği",
                markdown_content="Bankacılık sermaye yeterliliği hesaplama rehberi.",
            )
        )
        hits = await doc_store.search_content('"sermaye"')
        assert len(hits) >= 1

    @pytest.mark.asyncio
    async def test_search_with_operators(self, doc_store):
        await doc_store.store_document(
            StoredDocument(
                document_id="1",
                title="Sermaye Yeterliliği",
                markdown_content="Bankacılık sermaye yeterliliği hesaplama rehberi.",
            )
        )
        hits = await doc_store.search_content("sermaye AND OR NOT")
        assert isinstance(hits, list)

    @pytest.mark.asyncio
    async def test_search_with_special_chars(self, doc_store):
        await doc_store.store_document(
            StoredDocument(
                document_id="1",
                title="Sermaye Yeterliliği",
                markdown_content="Bankacılık sermaye yeterliliği hesaplama rehberi.",
            )
        )
        hits = await doc_store.search_content("sermaye* (test) +risk-")
        assert isinstance(hits, list)

    @pytest.mark.asyncio
    async def test_search_all_operators_returns_empty(self, doc_store):
        hits = await doc_store.search_content("AND OR NOT")
        assert hits == []
