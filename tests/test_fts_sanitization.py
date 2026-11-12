"""FTS5 query construction: user text can never inject MATCH operators."""

import logging

import pytest

from bddk_mcp.store.doc_store import StoredDocument, fts_match_query


class TestFtsMatchQuery:
    """Every token is quoted, so FTS5 syntax in user input is matched literally or dropped."""

    def test_normal_term(self):
        assert fts_match_query("sermaye") == '"sermaye"'

    def test_every_token_must_match(self):
        assert fts_match_query("sermaye yeterliliği") == '"sermaye" "yeterliligi"'

    def test_operators_and_syntax_characters_are_neutralized(self):
        assert fts_match_query('"test"') == '"test"'
        assert fts_match_query("test*") == '"test"'
        assert fts_match_query("(test)") == '"test"'
        assert fts_match_query("+test-") == '"test"'
        assert fts_match_query("test:col") == '"test" "col"'

    def test_operator_words_become_plain_quoted_tokens(self):
        assert fts_match_query("sermaye AND risk") == '"sermaye" "and" "risk"'
        assert fts_match_query("NEAR") == '"near"'

    def test_case_and_diacritics_are_folded(self):
        assert fts_match_query("SERMAYE Yeterliliği İlke") == '"sermaye" "yeterliligi" "ilke"'

    def test_empty_or_symbol_only_input(self):
        assert fts_match_query("") is None
        assert fts_match_query("*** ()) --") is None


class TestSectionSearchSanitized:
    """Integration: section search with potentially dangerous input never raises."""

    async def _seed(self, doc_store):
        await doc_store.store_document(
            StoredDocument(
                document_id="1",
                title="Sermaye Yeterliliği",
                markdown_content="MADDE 1\nBankacılık sermaye yeterliliği hesaplama rehberi.",
            )
        )

    @pytest.mark.asyncio
    async def test_search_with_quotes(self, doc_store):
        await self._seed(doc_store)
        hits = await doc_store.search_document_sections('"sermaye"')
        assert [hit.doc_id for hit in hits] == ["1"]

    @pytest.mark.asyncio
    async def test_search_with_operators(self, doc_store):
        await self._seed(doc_store)
        assert await doc_store.search_document_sections("sermaye AND OR NOT") == []
        assert [hit.doc_id for hit in await doc_store.search_document_sections("sermaye OR")] == []

    @pytest.mark.asyncio
    async def test_search_with_special_chars(self, doc_store):
        await self._seed(doc_store)
        hits = await doc_store.search_document_sections("sermaye* (rehberi) +yeterliliği-")
        assert [hit.doc_id for hit in hits] == ["1"]

    @pytest.mark.asyncio
    async def test_search_symbols_only_returns_empty(self, doc_store):
        await self._seed(doc_store)
        assert await doc_store.search_document_sections("*** ()) --") == []


@pytest.mark.asyncio
async def test_section_search_log_does_not_include_raw_query(doc_store, caplog):
    sentinel = "PRIVATE_FTS_QUERY_SENTINEL_4d2a"

    with caplog.at_level(logging.DEBUG, logger="bddk_mcp.store"):
        hits = await doc_store.search_document_sections(sentinel)

    assert hits == []
    assert sentinel not in caplog.text
