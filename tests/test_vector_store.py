"""Tests for VectorStore (pgvector) — chunking, add, search, retrieval."""

from unittest.mock import AsyncMock

import pytest

from vector_store import VectorStore, _chunk_text

# VectorStore integration tests require both PostgreSQL and the embedding model.
# They skip if either is unavailable.
_SKIP_REASON = "Embedding model not available or PostgreSQL not reachable"


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


class TestHybridSearchOrdering:
    """Regression coverage for the RRF-vs-cosine sort inconsistency.

    _rrf_fuse() ranks candidates by rrf_score (dense rank + FTS rank) but the
    `relevance` field surfaced to the user is the raw vector cosine. If the
    final output is left in RRF order, callers observe non-monotonic scores
    (e.g. rank #1 = 87.9%, rank #2 = 89.9%). _hybrid_search now re-sorts by
    `relevance` after the threshold filter.
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

        # Cosines chosen close together so the 0.08 score-gap filter keeps
        # all three hits. Vector and FTS rankings disagree: vector ranks
        # a > b > c by cosine, FTS ranks c > a > b. With RRF sort, the
        # output would be [a, c, b] — relevance [0.85, 0.80, 0.82] — which
        # is non-monotonic. The fix re-sorts to [a, b, c].
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
        assert relevances == sorted(relevances, reverse=True), (
            f"Results not monotonic in relevance: {relevances}"
        )
        assert [r["doc_id"] for r in results] == ["a", "b", "c"]


async def _can_initialize_store(pg_pool) -> bool:
    """Check if VectorStore can initialize with embeddings."""
    try:
        vs = VectorStore(pg_pool)
        await vs.initialize()
        vs._ensure_embeddings()
        return True
    except Exception:
        return False


@pytest.fixture
async def _check_model(pg_pool):
    """Check if embedding model is available."""
    try:
        return await _can_initialize_store(pg_pool)
    except Exception:
        return False


class TestVectorStoreLifecycle:
    """Test VectorStore initialization and basic operations."""

    @pytest.fixture
    async def store(self, pg_pool, _check_model):
        if not _check_model:
            pytest.skip(_SKIP_REASON)
        vs = VectorStore(pg_pool)
        await vs.initialize()
        # Clean up any leftover test data
        await pg_pool.execute(
            "DELETE FROM document_chunks WHERE doc_id LIKE 'test_%' OR doc_id IN ('a','b','d1','del_me','empty','multi','s1','s2')"
        )
        yield vs

    @pytest.mark.asyncio
    async def test_initialize(self, store):
        stats = await store.stats()
        assert isinstance(stats["total_documents"], int)
        assert isinstance(stats["total_chunks"], int)

    @pytest.mark.asyncio
    async def test_add_document(self, store):
        chunks = await store.add_document(
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
    async def test_add_document_replaces_existing(self, store):
        await store.add_document(doc_id="d1", title="V1", content="Version one content")
        await store.add_document(doc_id="d1", title="V2", content="Version two content updated")

        doc = await store.get_document("d1")
        assert doc is not None
        assert doc["title"] == "V2"
        assert "two" in doc["content"]
        await store.delete_document("d1")

    @pytest.mark.asyncio
    async def test_get_document(self, store):
        await store.add_document(
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
    async def test_get_document_page(self, store):
        long_content = "A" * 12000
        await store.add_document(doc_id="test_long", title="Long Doc", content=long_content)

        page1 = await store.get_document_page("test_long", page=1)
        assert page1 is not None
        assert page1["page_number"] == 1
        assert page1["total_pages"] >= 2

        invalid = await store.get_document_page("test_long", page=999)
        assert invalid is not None
        assert "Invalid page" in invalid["content"]
        await store.delete_document("test_long")

    @pytest.mark.asyncio
    async def test_delete_document(self, store):
        await store.add_document(doc_id="del_me", title="Delete Me", content="Some content")
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
        await vs.initialize()

        # Clean and populate
        for did in ("capital", "interest", "loans"):
            await vs.delete_document(did)

        await vs.add_document(
            doc_id="capital",
            title="Sermaye Yeterliliği Rehberi",
            content="Bu rehber bankacılık sektöründe sermaye yeterliliği hesaplamalarını düzenler.",
            category="Rehber",
        )
        await vs.add_document(
            doc_id="interest",
            title="Faiz Oranı Riski Yönetmeliği",
            content="Banka faiz oranı riskini ölçmek için standart yaklaşım kullanır.",
            category="Yönetmelik",
        )
        await vs.add_document(
            doc_id="loans",
            title="Kredi İşlemleri Genelgesi",
            content="Bankaların kredi işlemlerine ilişkin genel kurallar ve prosedürler.",
            category="Genelge",
        )

        yield vs

        # Cleanup
        for did in ("capital", "interest", "loans"):
            await vs.delete_document(did)

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
