"""Tests for VectorStore (ChromaDB) — chunking, add, search, retrieval."""

import pytest

from vector_store import VectorStore, _chunk_text

# Skip all VectorStore integration tests if embedding model is unavailable
_SKIP_REASON = "Embedding model not downloadable in this environment"


def _can_initialize_store(tmp_path):
    """Check if VectorStore can initialize with embeddings (model downloadable)."""
    try:
        vs = VectorStore(db_path=tmp_path / "_probe_chroma")
        vs.initialize()
        vs._ensure_embeddings()  # This triggers the actual model download
        vs.close()
        return True
    except Exception:
        return False


class TestChunkText:
    """Test the text chunking utility."""

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
        # Second chunk starts at 1000-200=800, so len = 1500-800=700
        assert len(chunks[1]) == 700

    def test_many_chunks(self):
        text = "A" * 5000
        chunks = _chunk_text(text, chunk_size=1000, overlap=200)
        assert len(chunks) >= 5
        # First chunk is full size
        assert len(chunks[0]) == 1000

    def test_whitespace_only_chunks_skipped(self):
        text = "real content" + " " * 2000
        chunks = _chunk_text(text, chunk_size=1000, overlap=200)
        # Should not include empty/whitespace chunks
        for chunk in chunks:
            assert chunk.strip() != ""


@pytest.fixture(scope="module")
def _check_model(tmp_path_factory):
    """Module-scoped check if embedding model is available."""
    return _can_initialize_store(tmp_path_factory.mktemp("probe"))


class TestVectorStoreLifecycle:
    """Test VectorStore initialization and basic operations."""

    @pytest.fixture
    def store(self, tmp_path, _check_model):
        """Create a temporary VectorStore."""
        if not _check_model:
            pytest.skip(_SKIP_REASON)
        db_path = tmp_path / "test_chroma"
        vs = VectorStore(db_path=db_path)
        vs.initialize()
        yield vs
        vs.close()

    def test_initialize(self, store):
        assert store._collection is not None
        stats = store.stats()
        assert stats["total_documents"] == 0
        assert stats["total_chunks"] == 0

    def test_add_document(self, store):
        chunks = store.add_document(
            doc_id="test_1",
            title="Test Document",
            content="This is a test document with some content.",
            category="Rehber",
        )
        assert chunks >= 1

        # Verify it exists
        assert store.has_document("test_1")
        assert not store.has_document("nonexistent")

    def test_add_empty_document(self, store):
        chunks = store.add_document(doc_id="empty", title="Empty", content="", category="")
        assert chunks == 0
        assert not store.has_document("empty")

    def test_add_document_replaces_existing(self, store):
        store.add_document(doc_id="d1", title="V1", content="Version one content")
        store.add_document(doc_id="d1", title="V2", content="Version two content updated")

        doc = store.get_document("d1")
        assert doc is not None
        assert doc["title"] == "V2"
        assert "two" in doc["content"]

    def test_get_document(self, store):
        store.add_document(
            doc_id="doc_1",
            title="Capital Adequacy",
            content="Capital adequacy regulation for banks.",
            category="Rehber",
            decision_date="15.03.2024",
        )

        doc = store.get_document("doc_1")
        assert doc is not None
        assert doc["doc_id"] == "doc_1"
        assert doc["title"] == "Capital Adequacy"
        assert doc["category"] == "Rehber"
        assert "Capital adequacy" in doc["content"]

    def test_get_nonexistent_document(self, store):
        assert store.get_document("nonexistent") is None

    def test_get_document_page(self, store):
        # Create a document that spans multiple pages (page size is 5000 chars)
        long_content = "A" * 12000
        store.add_document(doc_id="long", title="Long Doc", content=long_content)

        page1 = store.get_document_page("long", page=1)
        assert page1 is not None
        assert page1["page_number"] == 1
        assert page1["total_pages"] >= 2

        # Invalid page
        invalid = store.get_document_page("long", page=999)
        assert invalid is not None
        assert "Invalid page" in invalid["content"]

    def test_get_document_page_nonexistent(self, store):
        assert store.get_document_page("nope") is None

    def test_delete_document(self, store):
        store.add_document(doc_id="del_me", title="Delete Me", content="Some content")
        assert store.has_document("del_me")

        deleted = store.delete_document("del_me")
        assert deleted is True
        assert not store.has_document("del_me")

    def test_delete_nonexistent(self, store):
        deleted = store.delete_document("never_existed")
        assert deleted is False

    def test_document_count(self, store):
        store.add_document(doc_id="a", title="A", content="Content A")
        store.add_document(doc_id="b", title="B", content="Content B")
        assert store.document_count() == 2

    def test_chunk_count(self, store):
        # Add a document with enough content for multiple chunks
        content = "Word " * 500  # ~2500 chars, should make 2-3 chunks
        store.add_document(doc_id="multi", title="Multi", content=content)
        assert store.chunk_count() >= 2

    def test_stats(self, store):
        store.add_document(doc_id="s1", title="S1", content="Content", category="Rehber")
        store.add_document(doc_id="s2", title="S2", content="Content", category="Genelge")

        stats = store.stats()
        assert stats["total_documents"] == 2
        assert stats["total_chunks"] >= 2
        assert "Rehber" in stats["categories"]
        assert "Genelge" in stats["categories"]


class TestVectorStoreSearch:
    """Test semantic search functionality."""

    @pytest.fixture
    def populated_store(self, tmp_path, _check_model):
        if not _check_model:
            pytest.skip(_SKIP_REASON)
        db_path = tmp_path / "search_chroma"
        vs = VectorStore(db_path=db_path)
        vs.initialize()

        vs.add_document(
            doc_id="capital",
            title="Sermaye Yeterliliği Rehberi",
            content="Bu rehber bankacılık sektöründe sermaye yeterliliği hesaplamalarını düzenler. Kredi riski için asgari sermaye oranı yüzde sekiz olarak belirlenmiştir.",
            category="Rehber",
        )
        vs.add_document(
            doc_id="interest",
            title="Faiz Oranı Riski Yönetmeliği",
            content="Banka faiz oranı riskini ölçmek için standart yaklaşım kullanır. Faiz riski yönetimi bankacılık düzenlemelerinin önemli bir parçasıdır.",
            category="Yönetmelik",
        )
        vs.add_document(
            doc_id="loans",
            title="Kredi İşlemleri Genelgesi",
            content="Bankaların kredi işlemlerine ilişkin genel kurallar ve prosedürler. Tüketici kredileri ve ticari krediler ayrı ayrı düzenlenmektedir.",
            category="Genelge",
        )

        yield vs
        vs.close()

    def test_search_returns_results(self, populated_store):
        hits = populated_store.search("sermaye yeterliliği", limit=10)
        assert len(hits) >= 1
        assert hits[0]["doc_id"] == "capital"

    def test_search_relevance_ordering(self, populated_store):
        hits = populated_store.search("faiz oranı riski", limit=10)
        assert len(hits) >= 1
        # The "interest" document should be most relevant
        assert hits[0]["doc_id"] == "interest"

    def test_search_with_category_filter(self, populated_store):
        hits = populated_store.search("banka", limit=10, category="Genelge")
        # Should only return Genelge documents
        assert all(h["category"] == "Genelge" for h in hits)

    def test_search_no_results(self, populated_store):
        hits = populated_store.search("quantum physics dark matter", limit=10)
        # May return results with low relevance, but should handle gracefully
        assert isinstance(hits, list)

    def test_search_deduplicates_by_doc_id(self, populated_store):
        hits = populated_store.search("banka", limit=10)
        doc_ids = [h["doc_id"] for h in hits]
        assert len(doc_ids) == len(set(doc_ids))

    def test_search_includes_relevance_score(self, populated_store):
        hits = populated_store.search("sermaye", limit=5)
        for h in hits:
            assert "relevance" in h
            assert 0 <= h["relevance"] <= 1

    def test_search_includes_snippet(self, populated_store):
        hits = populated_store.search("kredi", limit=5)
        for h in hits:
            assert "snippet" in h


class TestContentReconstruction:
    """Test that content reconstruction from chunks preserves text."""

    @pytest.fixture
    def store(self, tmp_path, _check_model):
        if not _check_model:
            pytest.skip(_SKIP_REASON)
        db_path = tmp_path / "reconstruct_chroma"
        vs = VectorStore(db_path=db_path)
        vs.initialize()
        yield vs
        vs.close()

    def test_short_document_roundtrip(self, store):
        original = "This is a short document."
        store.add_document(doc_id="short", title="Short", content=original)
        doc = store.get_document("short")
        assert doc["content"] == original

    def test_long_document_roundtrip(self, store):
        # Create a document longer than one chunk
        original = "ABCDEFGHIJ" * 150  # 1500 chars
        store.add_document(doc_id="long", title="Long", content=original)
        doc = store.get_document("long")
        # Due to overlap removal, content should be very close to original
        assert len(doc["content"]) >= len(original) * 0.9
