"""Tests for VectorStore (pgvector) — chunking, add, search, retrieval."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bddk_mcp.store.doc_store import DocumentStore, StoredDocument
from bddk_mcp.store.vector_store import VectorIndexConsistencyError, VectorStore, _chunk_document, _chunk_text

# VectorStore integration tests require both PostgreSQL and the embedding model.
# They skip if either is unavailable.
_SKIP_REASON = "Embedding model not available or PostgreSQL not reachable"
_VECTOR_TEST_IDS = {
    "a",
    "b",
    "capital",
    "d1",
    "del_me",
    "empty",
    "interest",
    "loans",
    "multi",
    "s1",
    "s2",
    "tampered-chunk",
    "test_1",
    "test_doc1",
    "test_long",
}


async def _store_and_index(
    vector_store: VectorStore,
    pool,
    *,
    doc_id: str,
    title: str,
    content: str,
    category: str = "",
    decision_date: str = "",
) -> int:
    await DocumentStore(pool).store_document(
        StoredDocument(
            document_id=doc_id,
            title=title,
            markdown_content=content,
            category=category,
            decision_date=decision_date,
        )
    )
    return await vector_store.add_document(
        doc_id=doc_id,
        title=title,
        content=content,
        category=category,
        decision_date=decision_date,
    )


async def _clean_vector_test_documents(pool) -> None:
    ids = sorted(_VECTOR_TEST_IDS)
    await pool.execute("DELETE FROM public.document_chunks WHERE doc_id = ANY($1::text[])", ids)
    await pool.execute("DELETE FROM public.document_sections WHERE doc_id = ANY($1::text[])", ids)
    await pool.execute("DELETE FROM public.document_versions WHERE document_id = ANY($1::text[])", ids)
    await pool.execute("DELETE FROM public.documents WHERE document_id = ANY($1::text[])", ids)


class WhitespaceTokenizer:
    def encode(self, text: str, **_kwargs):
        return text.split()


def _token_count(text: str) -> int:
    return len(text.split())


class TestChunkText:
    """Test the text chunking utility (no DB needed)."""

    def test_empty_text(self):
        assert _chunk_text("") == []

    def test_short_text(self):
        text = "Short text"
        chunks = _chunk_text(text, chunk_size=1000)
        assert chunks == [text]

    def test_exact_chunk_size(self):
        text = "A" * 1000
        chunks = _chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_two_chunks_with_overlap(self):
        text = "A" * 1500
        chunks = _chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) == 2
        assert len(chunks[0]) == 1000
        assert len(chunks[1]) == 700

    def test_many_chunks(self):
        text = "A" * 5000
        chunks = _chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) >= 5
        assert len(chunks[0]) == 1000

    def test_whitespace_only_chunks_skipped(self):
        text = "real content" + " " * 2000
        chunks = _chunk_text(text, chunk_size=1000, overlap=200)
        for chunk in chunks:
            assert chunk.strip() != ""

    def test_chunk_document_attaches_section_metadata(self):
        text = (
            "MADDE 9 - TFRS 9 karşılık\nBankalar karşılık ayırır.\n\nMADDE 10 - Diğer hükümler\nBaşka hüküm uygulanır."
        )

        chunks = _chunk_document("mevzuat_22599", text, chunk_size=55, overlap=0)

        assert chunks[0].chunk_text.startswith("MADDE 9")
        assert chunks[0].section_type == "madde"
        assert chunks[0].section_ref == "9"
        assert chunks[0].section_start_char == 0
        assert chunks[0].section_end_char > chunks[0].section_start_char
        assert len(chunks[0].section_content_hash) == 64
        assert any(chunk.section_ref == "10" for chunk in chunks)

    def test_chunk_document_uses_token_budget_when_tokenizer_provided(self):
        text = (
            "MADDE 9 - Karşılık ayrılması\n"
            "bir iki üç dört beş altı yedi sekiz dokuz on onbir oniki\n\n"
            "MADDE 10 - Diğer hükümler\n"
            "kısa hüküm metni"
        )

        chunks = _chunk_document(
            "mevzuat_22599",
            text,
            tokenizer=WhitespaceTokenizer(),
            target_tokens=8,
            token_overlap=2,
        )

        assert len(chunks) > 2
        assert all(_token_count(chunk.chunk_text) <= 8 for chunk in chunks)
        assert {chunk.section_ref for chunk in chunks} >= {"9", "10"}

    def test_token_aware_chunks_do_not_cross_section_boundaries(self):
        text = (
            "MADDE 9 - Karşılık ayrılması\n"
            "bir iki üç dört beş altı yedi sekiz\n\n"
            "MADDE 10 - Diğer hükümler\n"
            "dokuz on onbir oniki"
        )

        chunks = _chunk_document(
            "mevzuat_22599",
            text,
            tokenizer=WhitespaceTokenizer(),
            target_tokens=40,
            token_overlap=0,
        )

        assert not any("MADDE 9" in chunk.chunk_text and "MADDE 10" in chunk.chunk_text for chunk in chunks)

    def test_reconstruct_content_uses_chunk_offsets_for_token_overlap(self):
        vs = VectorStore.__new__(VectorStore)
        rows = [
            {"chunk_text": "alpha beta", "chunk_start_char": 0, "chunk_end_char": 10},
            {"chunk_text": "beta gamma", "chunk_start_char": 6, "chunk_end_char": 16},
        ]

        assert vs._reconstruct_content(rows) == "alpha beta gamma"


class TestModelReproducibility:
    def test_remote_embedding_model_uses_pinned_revision_and_dimension(self):
        calls: list[tuple[str, dict]] = []

        class FakeSentenceTransformer:
            def __init__(self, model_ref: str, **kwargs):
                calls.append((model_ref, kwargs))

            def get_sentence_embedding_dimension(self):
                return 768

        fake_module = SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        with (
            patch.dict("sys.modules", {"sentence_transformers": fake_module}),
            patch("bddk_mcp.store.vector_store.EMBEDDING_MODEL_PATH", ""),
            patch("bddk_mcp.store.vector_store.EMBEDDING_MODEL_REVISION", "a" * 40),
        ):
            store = VectorStore(MagicMock(), embedding_model="reviewed/model")
            store._ensure_embeddings()

        assert calls == [("reviewed/model", {"device": "cuda", "revision": "a" * 40})]

    def test_remote_custom_embedding_model_requires_revision(self):
        with (
            patch("bddk_mcp.store.vector_store.EMBEDDING_MODEL_PATH", ""),
            patch("bddk_mcp.store.vector_store.EMBEDDING_MODEL_REVISION", ""),
            pytest.raises(RuntimeError, match="BDDK_EMBEDDING_MODEL_REVISION"),
        ):
            VectorStore(MagicMock(), embedding_model="custom/model")._ensure_embeddings()

    def test_wrong_embedding_dimension_is_rejected(self):
        class FakeSentenceTransformer:
            def __init__(self, _model_ref: str, **_kwargs):
                pass

            def get_sentence_embedding_dimension(self):
                return 384

        fake_module = SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        with (
            patch.dict("sys.modules", {"sentence_transformers": fake_module}),
            patch("bddk_mcp.store.vector_store.EMBEDDING_MODEL_PATH", "/reviewed/local/model"),
            pytest.raises(RuntimeError, match="dimension must be 768"),
        ):
            VectorStore(MagicMock())._ensure_embeddings()

    def test_remote_reranker_uses_pinned_revision(self):
        calls: list[tuple[str, dict]] = []

        class FakeCrossEncoder:
            def __init__(self, model_ref: str, **kwargs):
                calls.append((model_ref, kwargs))

        fake_module = SimpleNamespace(CrossEncoder=FakeCrossEncoder)
        with (
            patch.dict("sys.modules", {"sentence_transformers": fake_module}),
            patch("bddk_mcp.store.vector_store.RERANKER_MODEL_PATH", ""),
            patch("bddk_mcp.store.vector_store.RERANKER_MODEL_NAME", "reviewed/reranker"),
            patch("bddk_mcp.store.vector_store.RERANKER_MODEL_REVISION", "b" * 40),
        ):
            store = VectorStore(MagicMock())
            store._ensure_reranker()

        assert calls == [("reviewed/reranker", {"device": "cuda", "revision": "b" * 40})]


class TestHybridSearchOrdering:
    """Regression coverage for the RRF-vs-final-relevance sort invariant.

    _rrf_fuse() ranks candidates by rrf_score (dense rank + FTS rank) but the
    `relevance` field surfaced to the user is the final dense+lexical score.
    _hybrid_search must keep output order monotonic in that displayed score.
    """

    def test_rrf_fuse_leaves_fts_only_hits_at_zero_relevance(self):
        """FTS-only hits have no cosine — stay at 0.0 so the
        SEMANTIC_RELEVANCE_THRESHOLD filter drops them downstream."""
        vs = VectorStore.__new__(VectorStore)

        vector_hits = [{"doc_id": "a", "relevance": 0.9, "title": "A", "snippet": "a"}]
        fts_hits = [
            {"doc_id": "a", "fts_rank": 0.5, "title": "A", "snippet": "a"},
            {"doc_id": "b", "fts_rank": 0.3, "title": "B", "snippet": "b"},  # FTS-only
        ]

        fused = vs._rrf_fuse(vector_hits, fts_hits)
        by_id = {r["doc_id"]: r for r in fused}

        assert by_id["a"]["relevance"] == 0.9
        assert by_id["b"]["relevance"] == 0.0
        assert "rrf_score" in by_id["a"]
        assert "rrf_score" in by_id["b"]

    @pytest.mark.asyncio
    async def test_hybrid_search_output_monotonic_in_relevance(self):
        """The RRF winner can have a lower cosine than a doc ranked below it
        when the two signals disagree. _hybrid_search must re-sort so the
        output order matches the displayed `relevance` — otherwise callers
        see rank #1 scoring lower than rank #2.
        """
        vs = VectorStore.__new__(VectorStore)

        # Cosines chosen close together so the score-gap filter keeps
        # all three hits. Vector and FTS rankings disagree: vector ranks
        # a > b > c by cosine, FTS ranks c > a > b. The lexical boost lets
        # c outrank b, but the final output must still be monotonic in the
        # displayed score.
        vs._vector_search = AsyncMock(
            return_value=[
                {"doc_id": "a", "relevance": 0.85, "title": "A", "snippet": "a"},
                {"doc_id": "b", "relevance": 0.82, "title": "B", "snippet": "b"},
                {"doc_id": "c", "relevance": 0.80, "title": "C", "snippet": "c"},
            ]
        )
        vs._fts_search = AsyncMock(
            return_value=[
                {"doc_id": "c", "fts_rank": 0.9, "title": "C", "snippet": "c"},
                {"doc_id": "a", "fts_rank": 0.5, "title": "A", "snippet": "a"},
                {"doc_id": "b", "fts_rank": 0.1, "title": "B", "snippet": "b"},
            ]
        )

        results = await vs._hybrid_search("q", limit=10)

        relevances = [r["relevance"] for r in results]
        assert relevances == sorted(relevances, reverse=True), f"Results not monotonic in relevance: {relevances}"
        assert [r["doc_id"] for r in results] == ["a", "c", "b"]

    @pytest.mark.asyncio
    async def test_hybrid_search_preserves_fts_only_exact_legal_reference(self):
        """Exact legal-reference FTS hits must survive semantic thresholding.

        A query like `Madde 9` can be best answered by an exact lexical hit even
        when dense retrieval misses it or gives it no cosine relevance.
        """
        vs = VectorStore.__new__(VectorStore)

        vs._vector_search = AsyncMock(return_value=[])
        vs._fts_search = AsyncMock(
            return_value=[
                {
                    "doc_id": "mevzuat_22599",
                    "fts_rank": 0.4,
                    "title": "Karşılık Yönetmeliği",
                    "snippet": "MADDE 9 - TFRS 9 kapsamında karşılık ayrılır.",
                }
            ]
        )

        results = await vs._hybrid_search("Karşılık Yönetmeliği Madde 9 TFRS 9", limit=10)

        assert [r["doc_id"] for r in results] == ["mevzuat_22599"]
        assert results[0]["match_type"] == "fts_exact"
        assert results[0]["semantic_relevance"] == 0.0
        assert results[0]["fts_rank"] == 0.4
        assert results[0]["relevance"] == 0.0

    @pytest.mark.asyncio
    async def test_hybrid_search_drops_fts_only_non_legal_query(self):
        """FTS-only bypass is limited to exact legal-reference queries."""
        vs = VectorStore.__new__(VectorStore)

        vs._vector_search = AsyncMock(return_value=[])
        vs._fts_search = AsyncMock(
            return_value=[
                {
                    "doc_id": "general",
                    "fts_rank": 0.4,
                    "title": "General",
                    "snippet": "Bankacılık işlemleri genel hükümler.",
                }
            ]
        )

        results = await vs._hybrid_search("bankacılık işlemleri", limit=10)

        assert results == []


async def _can_load_embedding_model(pg_pool) -> bool:
    """Check whether the configured embedding model can be loaded."""
    try:
        vs = VectorStore(pg_pool)
        vs._ensure_embeddings()
        return True
    except Exception:
        return False


@pytest.fixture
async def _check_model(pg_pool):
    """Check if embedding model is available."""
    try:
        return await _can_load_embedding_model(pg_pool)
    except Exception:
        return False


class TestVectorStoreLifecycle:
    """Test VectorStore initialization and basic operations."""

    @pytest.mark.asyncio
    async def test_initialize_is_select_only_readiness_wrapper(self):
        pool = MagicMock()
        readiness = AsyncMock()

        with patch("bddk_mcp.db_lifecycle.assert_database_ready", new=readiness):
            await VectorStore(pool).initialize()

        readiness.assert_awaited_once_with(pool=pool, require_corpus=False)
        pool.acquire.assert_not_called()
        pool.execute.assert_not_called()

    @pytest.fixture
    async def store(self, pg_pool, _check_model):
        if not _check_model:
            pytest.skip(_SKIP_REASON)
        vs = VectorStore(pg_pool)
        # Clean up any leftover test data
        await _clean_vector_test_documents(pg_pool)
        try:
            yield vs
        finally:
            await _clean_vector_test_documents(pg_pool)

    @pytest.mark.asyncio
    async def test_initialize(self, store):
        stats = await store.stats()
        assert isinstance(stats["total_documents"], int)
        assert isinstance(stats["total_chunks"], int)

    @pytest.mark.asyncio
    async def test_add_document(self, store, pg_pool):
        chunks = await _store_and_index(
            store,
            pg_pool,
            doc_id="test_1",
            title="Test Document",
            content="This is a test document with some content.",
            category="Rehber",
        )
        assert chunks >= 1
        assert await store.has_document("test_1")
        assert not await store.has_document("nonexistent")
        # Cleanup
        await store.delete_document("test_1")

    @pytest.mark.asyncio
    async def test_add_empty_document(self, store):
        chunks = await store.add_document(doc_id="empty", title="Empty", content="", category="")
        assert chunks == 0
        assert not await store.has_document("empty")

    @pytest.mark.asyncio
    async def test_add_document_replaces_existing(self, store, pg_pool):
        await _store_and_index(
            store,
            pg_pool,
            doc_id="d1",
            title="V1",
            content="Version one content",
        )
        await _store_and_index(
            store,
            pg_pool,
            doc_id="d1",
            title="V2",
            content="Version two content updated",
        )

        doc = await store.get_document("d1")
        assert doc is not None
        assert doc["title"] == "V2"
        assert "two" in doc["content"]
        await store.delete_document("d1")

    @pytest.mark.asyncio
    async def test_stale_chunks_are_hidden_until_matching_index_is_published(self, store, pg_pool):
        await _store_and_index(
            store,
            pg_pool,
            doc_id="d1",
            title="V1",
            content="Version one content",
        )
        assert await store.has_document("d1")

        await DocumentStore(pg_pool).store_document(
            StoredDocument(
                document_id="d1",
                title="V2",
                markdown_content="Version two content updated",
            )
        )
        assert not await store.has_document("d1")
        assert await store.get_document("d1") is None

        with pytest.raises(VectorIndexConsistencyError, match="stored document hash does not match"):
            await store.add_document(doc_id="d1", title="V1", content="Version one content")

        await store.add_document(doc_id="d1", title="V2", content="Version two content updated")
        assert await store.has_document("d1")

    @pytest.mark.asyncio
    async def test_publication_refuses_chunk_text_not_regenerated_from_canonical_document(self, store, pg_pool):
        chunks = await _store_and_index(
            store,
            pg_pool,
            doc_id="tampered-chunk",
            title="Canonical",
            content="Canonical regulatory text",
        )
        content_hash = await pg_pool.fetchval(
            "SELECT content_hash FROM public.documents WHERE document_id = $1",
            "tampered-chunk",
        )
        await pg_pool.execute(
            "UPDATE public.document_chunks SET chunk_text = $2 WHERE doc_id = $1 AND chunk_index = 0",
            "tampered-chunk",
            "Unsupported replacement text",
        )

        async with pg_pool.acquire() as connection:
            with pytest.raises(VectorIndexConsistencyError, match="publication integrity validation"):
                await store._publish_document_on_connection(
                    connection,
                    doc_id="tampered-chunk",
                    content_hash=content_hash,
                    expected_chunks=chunks,
                )

        assert not await store.has_document("tampered-chunk")
        await store.delete_document("tampered-chunk")

    @pytest.mark.asyncio
    async def test_get_document(self, store, pg_pool):
        await _store_and_index(
            store,
            pg_pool,
            doc_id="test_doc1",
            title="Capital Adequacy",
            content="Capital adequacy regulation for banks.",
            category="Rehber",
            decision_date="15.03.2024",
        )

        doc = await store.get_document("test_doc1")
        assert doc is not None
        assert doc["doc_id"] == "test_doc1"
        assert doc["title"] == "Capital Adequacy"
        assert doc["category"] == "Rehber"
        assert "Capital adequacy" in doc["content"]
        await store.delete_document("test_doc1")

    @pytest.mark.asyncio
    async def test_get_nonexistent_document(self, store):
        assert await store.get_document("nonexistent") is None

    @pytest.mark.asyncio
    async def test_get_document_page(self, store, pg_pool):
        long_content = "A" * 12000
        await _store_and_index(
            store,
            pg_pool,
            doc_id="test_long",
            title="Long Doc",
            content=long_content,
        )

        page1 = await store.get_document_page("test_long", page=1)
        assert page1 is not None
        assert page1["page_number"] == 1
        assert page1["total_pages"] >= 2

        invalid = await store.get_document_page("test_long", page=999)
        assert invalid is not None
        assert "Invalid page" in invalid["content"]
        await store.delete_document("test_long")

    @pytest.mark.asyncio
    async def test_delete_document(self, store, pg_pool):
        await _store_and_index(
            store,
            pg_pool,
            doc_id="del_me",
            title="Delete Me",
            content="Some content",
        )
        assert await store.has_document("del_me")

        deleted = await store.delete_document("del_me")
        assert deleted is True
        assert not await store.has_document("del_me")

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        deleted = await store.delete_document("never_existed")
        assert deleted is False


class TestVectorStoreSearch:
    """Test semantic search functionality."""

    @pytest.fixture
    async def populated_store(self, pg_pool, _check_model):
        if not _check_model:
            pytest.skip(_SKIP_REASON)
        vs = VectorStore(pg_pool)

        # Clean and populate
        await _clean_vector_test_documents(pg_pool)

        await _store_and_index(
            vs,
            pg_pool,
            doc_id="capital",
            title="Sermaye Yeterliliği Rehberi",
            content="Bu rehber bankacılık sektöründe sermaye yeterliliği hesaplamalarını düzenler.",
            category="Rehber",
        )
        await _store_and_index(
            vs,
            pg_pool,
            doc_id="interest",
            title="Faiz Oranı Riski Yönetmeliği",
            content="Banka faiz oranı riskini ölçmek için standart yaklaşım kullanır.",
            category="Yönetmelik",
        )
        await _store_and_index(
            vs,
            pg_pool,
            doc_id="loans",
            title="Kredi İşlemleri Genelgesi",
            content="Bankaların kredi işlemlerine ilişkin genel kurallar ve prosedürler.",
            category="Genelge",
        )

        try:
            yield vs
        finally:
            await _clean_vector_test_documents(pg_pool)

    @pytest.mark.asyncio
    async def test_search_returns_results(self, populated_store):
        hits = await populated_store.search("sermaye yeterliliği", limit=10)
        assert len(hits) >= 1
        assert hits[0]["doc_id"] == "capital"

    @pytest.mark.asyncio
    async def test_search_relevance_ordering(self, populated_store):
        hits = await populated_store.search("faiz oranı riski", limit=10)
        assert len(hits) >= 1
        assert hits[0]["doc_id"] == "interest"

    @pytest.mark.asyncio
    async def test_search_with_category_filter(self, populated_store):
        hits = await populated_store.search("banka", limit=10, category="Genelge")
        assert all(h["category"] == "Genelge" for h in hits)

    @pytest.mark.asyncio
    async def test_search_no_results(self, populated_store):
        hits = await populated_store.search("quantum physics dark matter", limit=10)
        assert isinstance(hits, list)

    @pytest.mark.asyncio
    async def test_search_deduplicates_by_doc_id(self, populated_store):
        hits = await populated_store.search("banka", limit=10)
        doc_ids = [h["doc_id"] for h in hits]
        assert len(doc_ids) == len(set(doc_ids))

    @pytest.mark.asyncio
    async def test_search_includes_relevance_score(self, populated_store):
        hits = await populated_store.search("sermaye", limit=5)
        for h in hits:
            assert "relevance" in h
            assert 0 <= h["relevance"] <= 1
