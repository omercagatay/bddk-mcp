"""Tests for VectorStore — chunking, model loading, add, hybrid search — on a temporary corpus file."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bddk_mcp.core.config import EMBEDDING_MODEL_NAME
from bddk_mcp.quality.markdown_quality import assess_markdown_quality
from bddk_mcp.store.doc_store import DocumentStore, StoredDocument
from bddk_mcp.store.vector_store import (
    SemanticSearchReadinessError,
    SemanticSearchUnavailableError,
    VectorIndexConsistencyError,
    VectorStore,
    _build_query_vocabulary,
    _chunk_document,
    _document_quality_by_ids,
    _quality_metadata,
    displayed_relevance,
    normalize_query,
    relevance_calibration,
    semantic_threshold,
)
from tests.conftest import FakeTokenizer, fake_embed


async def _store_and_index(
    vector_store: VectorStore,
    doc_store: DocumentStore,
    *,
    doc_id: str,
    title: str,
    content: str,
    category: str = "",
    decision_date: str = "",
) -> int:
    stored = StoredDocument(
        document_id=doc_id,
        title=title,
        markdown_content=content,
        category=category,
        decision_date=decision_date,
    )
    await doc_store.store_document(stored)
    return await vector_store.add_document(
        doc_id=doc_id,
        title=title,
        content=stored.markdown_content,
        category=category,
        decision_date=decision_date,
    )


class WhitespaceTokenizer:
    def encode(self, text: str, **_kwargs):
        return text.split()


def _token_count(text: str) -> int:
    return len(text.split())


class TestQualityMetadata:
    def test_quality_metadata_uses_full_document_not_matching_chunk(self):
        matching_chunk = "Bankaların kredi işlemlerine ilişkin genel kurallar ve prosedürler."
        full_document = matching_chunk + "\n\nBu metinde aşağıdaki formül kullanılır."

        chunk_quality = _quality_metadata(matching_chunk, "quality_formula")
        document_quality = _quality_metadata(full_document, "quality_formula")

        assert chunk_quality["quality_label"] == "clean"
        assert document_quality["quality_label"] == "warning"
        assert "formula_ref_without_latex_or_image" in document_quality["quality_flags"]

    @pytest.mark.asyncio
    async def test_document_quality_by_ids_reads_full_stored_markdown(self, corpus_db, doc_store):
        matching_chunk = "Bankaların kredi işlemlerine ilişkin genel kurallar ve prosedürler."
        await doc_store.store_document(
            StoredDocument(
                document_id="quality_formula",
                title="Formül uyarısı belgesi",
                markdown_content=matching_chunk + "\n\nBu metinde aşağıdaki formül kullanılır.",
            )
        )
        quality_by_id = await _document_quality_by_ids(corpus_db, ["quality_formula", "missing"])
        quality = quality_by_id["quality_formula"]
        assert quality["quality_label"] == "warning"
        assert "formula_ref_without_latex_or_image" in quality["quality_flags"]
        assert quality_by_id["missing"]["quality_label"] != "clean" or quality_by_id["missing"]["quality_flags"] == []
        assert _quality_metadata(matching_chunk, "quality_formula")["quality_label"] == "clean"

    @pytest.mark.asyncio
    async def test_search_assesses_quality_once_for_final_hits_off_the_event_loop(self, doc_store, vector_store):
        for index in range(4):
            await _store_and_index(
                vector_store,
                doc_store,
                doc_id=f"quality_{index}",
                title="Sermaye yeterliliği hesaplaması",
                content=f"sermaye yeterliliği hesaplaması {index}",
            )
        calls: list[tuple[str, bool]] = []

        def spy(text, *, document_id):
            calls.append((document_id, threading.current_thread() is threading.main_thread()))
            return assess_markdown_quality(text, document_id=document_id)

        # The fake embedder's cosines are not on the scale of the calibrated model's
        # 0.80 default threshold; this test is about quality assessment, not thresholds.
        with (
            patch("bddk_mcp.store.vector_store.assess_markdown_quality", new=spy),
            patch("bddk_mcp.store.vector_store.SEMANTIC_RELEVANCE_THRESHOLD", 0.5),
        ):
            hits = await vector_store.search("sermaye yeterliliği hesaplaması", limit=1)

        assert len(hits) == 1
        assert hits[0]["quality_label"] == "clean"
        assert hits[0]["quality_flags"] == []
        assert calls == [(hits[0]["doc_id"], False)]


class TestChunkText:
    """Test the text chunking utility (no corpus needed)."""

    def test_chunk_document_attaches_section_metadata(self):
        text = (
            "MADDE 9 - TFRS 9 karşılık\nBankalar karşılık ayırır.\n\nMADDE 10 - Diğer hükümler\nBaşka hüküm uygulanır."
        )

        chunks = _chunk_document(
            "mevzuat_22599",
            text,
            tokenizer=WhitespaceTokenizer(),
            target_tokens=50,
            token_overlap=0,
        )

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

    def test_every_chunk_is_an_exact_slice_of_the_source(self):
        text = "MADDE 1 - Giriş\n" + " ".join(f"kelime{i}" for i in range(300)) + "\n\nMADDE 2 - Son\nkısa."
        chunks = _chunk_document("doc", text, tokenizer=WhitespaceTokenizer(), target_tokens=50, token_overlap=5)
        assert len(chunks) > 3
        for chunk in chunks:
            assert chunk.chunk_text == text[chunk.start_char : chunk.end_char]


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
            patch("bddk_mcp.store.vector_store._torch_device", return_value="cuda"),
        ):
            store = VectorStore(MagicMock(), embedding_model="reviewed/model")
            store._ensure_embeddings()

        assert calls == [
            (
                "reviewed/model",
                {
                    "device": "cuda",
                    "backend": "torch",
                    "local_files_only": False,
                    "trust_remote_code": False,
                    "revision": "a" * 40,
                },
            )
        ]

    def test_embedding_model_is_loaded_once_per_process(self):
        calls: list[str] = []

        class FakeSentenceTransformer:
            def __init__(self, model_ref: str, **kwargs):
                calls.append(model_ref)

            def get_sentence_embedding_dimension(self):
                return 768

        fake_module = SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        with (
            patch.dict("sys.modules", {"sentence_transformers": fake_module}),
            patch("bddk_mcp.store.vector_store.EMBEDDING_MODEL_PATH", ""),
            patch("bddk_mcp.store.vector_store.EMBEDDING_MODEL_REVISION", "a" * 40),
        ):
            first = VectorStore(MagicMock(), embedding_model="reviewed/model")
            second = VectorStore(MagicMock(), embedding_model="reviewed/model")
            first._ensure_embeddings()
            second._ensure_embeddings()

        assert calls == ["reviewed/model"]

    def test_wrong_embedding_dimension_is_rejected(self, tmp_path):
        class FakeSentenceTransformer:
            def __init__(self, _model_ref: str, **_kwargs):
                pass

            def get_sentence_embedding_dimension(self):
                return 384

        fake_module = SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        model_path = tmp_path / "reviewed-local-model"
        model_path.mkdir()
        (model_path / "config.json").write_text("{}", encoding="utf-8")
        with (
            patch.dict("sys.modules", {"sentence_transformers": fake_module}),
            patch("bddk_mcp.store.vector_store.EMBEDDING_MODEL_PATH", str(model_path)),
            pytest.raises(RuntimeError, match="dimension must be 768"),
        ):
            VectorStore(MagicMock())._ensure_embeddings()

    def test_renamed_embedding_dimension_accessor_is_preferred(self, tmp_path):
        class FakeSentenceTransformer:
            def __init__(self, _model_ref: str, **_kwargs):
                pass

            def get_embedding_dimension(self):
                return 384

            def get_sentence_embedding_dimension(self):
                raise AssertionError("the deprecated accessor must not be called when the new one exists")

        fake_module = SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        model_path = tmp_path / "reviewed-local-model"
        model_path.mkdir()
        (model_path / "config.json").write_text("{}", encoding="utf-8")
        with (
            patch.dict("sys.modules", {"sentence_transformers": fake_module}),
            patch("bddk_mcp.store.vector_store.EMBEDDING_MODEL_PATH", str(model_path)),
            pytest.raises(RuntimeError, match="dimension must be 768"),
        ):
            VectorStore(MagicMock())._ensure_embeddings()


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
        vs._embedding_model = EMBEDDING_MODEL_NAME

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
        vs._embedding_model = EMBEDDING_MODEL_NAME

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
        """Exact legal-reference FTS hits must survive semantic thresholding."""
        vs = VectorStore.__new__(VectorStore)
        vs._embedding_model = EMBEDDING_MODEL_NAME

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
    async def test_exact_reference_query_ranks_hybrid_hits_ahead_of_fts_only_matches(self):
        """A hit found by both FTS and dense search must not be cut by FTS-only exact matches."""
        vs = VectorStore.__new__(VectorStore)
        vs._embedding_model = EMBEDDING_MODEL_NAME

        vs._vector_search = AsyncMock(
            return_value=[
                {"doc_id": "good", "relevance": 0.85, "title": "Good", "snippet": "good"},
                {"doc_id": "vec", "relevance": 0.84, "title": "Vec", "snippet": "vec"},
            ]
        )
        vs._fts_search = AsyncMock(
            return_value=[
                {"doc_id": "x1", "fts_rank": 0.9, "title": "X1", "snippet": "MADDE 5 - x1"},
                {"doc_id": "x2", "fts_rank": 0.8, "title": "X2", "snippet": "MADDE 5 - x2"},
                {"doc_id": "good", "fts_rank": 0.5, "title": "Good", "snippet": "good"},
            ]
        )
        query = "sermaye yeterliliği madde 5"

        top_two = await vs._hybrid_search(query, limit=2)
        everything = await vs._hybrid_search(query, limit=10)

        assert [r["doc_id"] for r in top_two] == ["good", "x1"]
        assert top_two[0]["match_type"] == "hybrid"
        assert [r["doc_id"] for r in everything] == ["good", "x1", "x2", "vec"]
        assert [r["match_type"] for r in everything] == ["hybrid", "fts_exact", "fts_exact", "vector"]

    @pytest.mark.asyncio
    async def test_vector_only_hit_follows_fts_exact_hits_even_with_a_higher_score(self):
        """Pins the order that DocumentSearchItem.relevance documents: with fts_exact hits present,
        a vector-only hit comes after them even when it outscores a hybrid hit."""
        from bddk_mcp.tools.structured_outputs import DocumentSearchItem

        vs = VectorStore.__new__(VectorStore)
        vs._embedding_model = EMBEDDING_MODEL_NAME
        vs._vector_search = AsyncMock(
            return_value=[
                {"doc_id": "vec", "relevance": 0.84, "title": "Vec", "snippet": "vec"},
                # Above the calibrated model's 0.80 raw-cosine threshold.
                {"doc_id": "hyb", "relevance": 0.81, "title": "Hyb", "snippet": "hyb"},
            ]
        )
        vs._fts_search = AsyncMock(
            return_value=[
                {"doc_id": "ex", "fts_rank": 0.9, "title": "Ex", "snippet": "MADDE 5 - ex", "fts_relaxed": False},
                {"doc_id": "hyb", "fts_rank": 0.5, "title": "Hyb", "snippet": "hyb", "fts_relaxed": False},
            ]
        )

        results = await vs._hybrid_search("sermaye madde 5", limit=10)

        assert [(r["doc_id"], r["match_type"]) for r in results] == [
            ("hyb", "hybrid"),
            ("ex", "fts_exact"),
            ("vec", "vector"),
        ]
        assert results[2]["relevance"] > results[0]["relevance"]
        description = DocumentSearchItem.model_fields["relevance"].description
        assert "then fts_exact" in description
        assert "then the remaining vector hits" in description
        assert "keyword-only fts hits" in description

    @pytest.mark.asyncio
    async def test_hybrid_search_keeps_a_strict_fts_only_hit_of_a_non_legal_query_as_a_keyword_match(self):
        """A document that contains every query word is not dropped for want of a semantic score."""
        vs = VectorStore.__new__(VectorStore)
        vs._embedding_model = EMBEDDING_MODEL_NAME

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

        assert [(r["doc_id"], r["match_type"]) for r in results] == [("general", "fts")]
        assert results[0]["relevance"] == 0.0
        assert results[0]["semantic_relevance"] == 0.0

    @pytest.mark.asyncio
    async def test_keyword_only_hits_follow_strong_and_moderate_hits_in_bm25_order_and_precede_weak_ones(self):
        vs = VectorStore.__new__(VectorStore)
        vs._embedding_model = EMBEDDING_MODEL_NAME
        vs._vector_search = AsyncMock(
            return_value=[
                {"doc_id": "sem1", "relevance": 0.86, "title": "S1", "snippet": "dense chunk"},
                {"doc_id": "sem2", "relevance": 0.83, "title": "S2", "snippet": "dense chunk"},
                # Below the 0.80 threshold: not a semantic hit, but it contains every query word.
                {"doc_id": "below", "relevance": 0.75, "title": "B", "snippet": "dense chunk of below"},
            ]
        )
        vs._fts_search = AsyncMock(
            return_value=[
                {"doc_id": "kw_best", "fts_rank": 3.0, "title": "K1", "snippet": "k1", "fts_relaxed": False},
                {"doc_id": "below", "fts_rank": 2.0, "title": "B", "snippet": "keyword chunk", "fts_relaxed": False},
                {"doc_id": "sem2", "fts_rank": 1.0, "title": "S2", "snippet": "s2", "fts_relaxed": False},
            ]
        )

        with patch("bddk_mcp.store.vector_store.SEMANTIC_RELEVANCE_THRESHOLD", None):
            everything = await vs._hybrid_search("pandemi kararı", limit=10)
            top_two = await vs._hybrid_search("pandemi kararı", limit=2)

        # sem1 and sem2 are shown as weak (displayed relevance below 0.50), so the documents that
        # contain every query word come before them.
        assert [(r["doc_id"], r["match_type"]) for r in everything] == [
            ("kw_best", "fts"),
            ("below", "fts"),
            ("sem1", "vector"),
            ("sem2", "hybrid"),
        ]
        # The keyword hit shows the chunk that contains the words, not the dense chunk.
        assert everything[1]["snippet"] == "keyword chunk"
        assert all(r["relevance"] == 0.0 for r in everything[:2])
        assert [r["doc_id"] for r in top_two] == ["kw_best", "below"]

    @pytest.mark.asyncio
    async def test_relaxed_or_hits_are_never_appended_as_keyword_matches(self):
        vs = VectorStore.__new__(VectorStore)
        vs._embedding_model = EMBEDDING_MODEL_NAME
        vs._vector_search = AsyncMock(return_value=[])
        vs._fts_search = AsyncMock(
            return_value=[{"doc_id": "loose", "fts_rank": 0.5, "title": "L", "snippet": "l", "fts_relaxed": True}]
        )

        assert await vs._hybrid_search("kredi kartı asgari ödeme tutarı", limit=10) == []


class TestHitsOfTheOldCutKeepTheirPlace:
    """Hits the old score-gap cut kept come first, in their old order; hits it would have cut follow them."""

    @staticmethod
    def _store(cosines: dict[str, float], exact: list[str]):
        vs = VectorStore.__new__(VectorStore)
        vs._embedding_model = EMBEDDING_MODEL_NAME
        vs._vector_search = AsyncMock(
            return_value=[{"doc_id": d, "relevance": c, "title": d, "snippet": "x"} for d, c in cosines.items()]
        )
        vs._fts_search = AsyncMock(
            return_value=[
                {"doc_id": d, "fts_rank": 1.0, "title": d, "snippet": "x", "fts_relaxed": False} for d in cosines
            ]
            + [
                {"doc_id": d, "fts_rank": 1.0, "title": d, "snippet": "MADDE 5 - metin", "fts_relaxed": False}
                for d in exact
            ]
        )
        return vs

    @pytest.mark.asyncio
    async def test_exact_hits_stay_ahead_of_hybrid_hits_the_old_cut_would_have_dropped(self):
        cosines = {"h1": 0.95, "h2": 0.94, "h3": 0.86, "h4": 0.85, "h5": 0.84, "h6": 0.83}
        vs = self._store(cosines, ["ex1", "ex2"])

        with patch("bddk_mcp.store.vector_store.SEMANTIC_RELEVANCE_THRESHOLD", None):
            top = await vs._hybrid_search("sermaye madde 5", limit=4)
            everything = await vs._hybrid_search("sermaye madde 5", limit=10)

        # h1/h2 are within 0.051 of the top ordering score (the old cut kept them); h3-h6 are not.
        assert [r["doc_id"] for r in top] == ["h1", "h2", "ex1", "ex2"]
        assert [r["doc_id"] for r in everything] == ["h1", "h2", "ex1", "ex2", "h3", "h4", "h5", "h6"]
        assert [r["match_type"] for r in everything[2:4]] == ["fts_exact", "fts_exact"]

    @pytest.mark.asyncio
    async def test_without_exact_hits_the_order_is_the_ordering_score_order(self):
        vs = self._store({"a": 0.95, "b": 0.86, "c": 0.90}, [])

        with patch("bddk_mcp.store.vector_store.SEMANTIC_RELEVANCE_THRESHOLD", None):
            results = await vs._hybrid_search("sermaye", limit=10)

        assert [r["doc_id"] for r in results] == ["a", "c", "b"]


class TestKeywordOnlyHitsComeBeforeWeakHits:
    @pytest.mark.asyncio
    async def test_order_is_strong_and_moderate_then_keyword_only_then_weak(self):
        vs = VectorStore.__new__(VectorStore)
        vs._embedding_model = EMBEDDING_MODEL_NAME
        vs._vector_search = AsyncMock(
            return_value=[
                {"doc_id": "strong", "relevance": 0.95, "title": "S", "snippet": "s"},
                {"doc_id": "moderate", "relevance": 0.90, "title": "M", "snippet": "m"},
                {"doc_id": "weak", "relevance": 0.85, "title": "W", "snippet": "w"},
            ]
        )
        vs._fts_search = AsyncMock(
            return_value=[
                {"doc_id": "kw2", "fts_rank": 1.0, "title": "K2", "snippet": "k", "fts_relaxed": False},
                {"doc_id": "kw1", "fts_rank": 2.0, "title": "K1", "snippet": "k", "fts_relaxed": False},
            ]
        )

        with patch("bddk_mcp.store.vector_store.SEMANTIC_RELEVANCE_THRESHOLD", None):
            results = await vs._hybrid_search("deprem", limit=10)
            top = await vs._hybrid_search("deprem", limit=3)

        assert [(r["doc_id"], r["confidence"]) for r in results] == [
            ("strong", "high"),
            ("moderate", "medium"),
            ("kw1", "low"),
            ("kw2", "low"),
            ("weak", "low"),
        ]
        assert [r["doc_id"] for r in top] == ["strong", "moderate", "kw1"]


class TestKeywordOnlyHitsAreCountedByDocument:
    @pytest.mark.asyncio
    async def test_one_document_owning_more_matching_chunks_than_the_fetch_limit_does_not_hide_the_others(
        self, vector_store, doc_store
    ):
        articles = "\n\n".join(f"MADDE {n} – 5411 5411 5411 kısa." for n in range(1, 61))
        await _store_and_index(vector_store, doc_store, doc_id="kanun", title="Kanun", content=articles)
        filler = " ".join(f"kelime{i}" for i in range(150))
        for doc_id in ("other1", "other2", "other3"):
            await _store_and_index(
                vector_store, doc_store, doc_id=doc_id, title="Karar", content=f"{filler} 5411 sayılı Kanun {filler}"
            )
        vector_store._vector_search = AsyncMock(return_value=[])

        chunk_hits = await vector_store._fts_search("5411", limit=50)
        results = await vector_store._hybrid_search("5411", limit=10)

        assert [hit["doc_id"] for hit in chunk_hits] == ["kanun"]
        assert [hit["doc_id"] for hit in results][0] == "kanun"
        assert {hit["doc_id"] for hit in results} == {"kanun", "other1", "other2", "other3"}
        assert all(hit["match_type"] == "fts" for hit in results)


class TestKeywordLabelOnLegalReferences:
    """fts_exact ("keyword match on the legal reference in the query") only where the chunk holds that reference."""

    @staticmethod
    def _store(fts_hit):
        vs = VectorStore.__new__(VectorStore)
        vs._embedding_model = EMBEDDING_MODEL_NAME
        vs._vector_search = AsyncMock(return_value=[])
        vs._fts_search = AsyncMock(return_value=[{"fts_rank": 1.0, "title": "T", "fts_relaxed": False, **fts_hit}])
        return vs

    @pytest.mark.asyncio
    async def test_article_number_inside_a_date_is_not_the_article(self):
        vs = self._store({"doc_id": "802", "snippet": "MADDE 1 – 20/9/2017 tarihli kararla değişik."})

        [hit] = await vs._hybrid_search("madde 9", limit=10)

        assert hit["match_type"] == "fts"

    @pytest.mark.asyncio
    async def test_date_tokens_scattered_over_the_chunk_are_not_the_date(self):
        vs = self._store({"doc_id": "1184", "snippet": "11/01/2019 tarihli karar, 31 banka ve 12 kuruluş."})

        [hit] = await vs._hybrid_search("31.12.2019", limit=10)

        assert hit["match_type"] == "fts"

    @pytest.mark.asyncio
    async def test_the_article_itself_and_the_date_itself_stay_fts_exact(self):
        article = self._store({"doc_id": "a", "snippet": "MADDE 9 – (1) Bankalar karşılık ayırır."})
        date = self._store({"doc_id": "d", "snippet": "31/12/2019 tarihi itibarıyla uygulanır."})
        decision = self._store({"doc_id": "k", "snippet": "Kurulun 8948 sayılı kararı ile belirlenmiştir."})

        assert (await article._hybrid_search("madde 9", limit=10))[0]["match_type"] == "fts_exact"
        assert (await date._hybrid_search("31.12.2019", limit=10))[0]["match_type"] == "fts_exact"
        assert (await decision._hybrid_search("8948 sayılı kurul kararı", limit=10))[0]["match_type"] == "fts_exact"

    @pytest.mark.asyncio
    async def test_the_section_binding_of_the_chunk_counts_as_the_reference(self):
        vs = self._store(
            {"doc_id": "a", "snippet": "(3) devamı olan metin.", "section_type": "madde", "section_ref": "66/a"}
        )

        [hit] = await vs._hybrid_search("madde 66/A", limit=10)

        assert hit["match_type"] == "fts_exact"

    @pytest.mark.asyncio
    async def test_the_reference_is_checked_on_the_whole_chunk_not_on_the_snippet(self):
        vs = self._store({"doc_id": "a", "snippet": "giriş metni", "chunk_text": "giriş metni ... MADDE 9 – metin"})

        [hit] = await vs._hybrid_search("madde 9", limit=10)

        assert hit["match_type"] == "fts_exact"
        assert "chunk_text" not in hit


class TestRelevanceCalibration:
    """Displayed relevance is a per-model mapping of the ordering score; ranking and raw cosine are untouched."""

    E5 = "intfloat/multilingual-e5-base"

    def test_e5_mapping_has_floor_ceiling_and_clamps(self):
        calibration = relevance_calibration(self.E5)

        assert (calibration.floor, calibration.ceiling) == (0.82, 0.96)
        assert displayed_relevance(0.82, calibration) == 0.0
        assert displayed_relevance(0.96, calibration) == 1.0
        assert displayed_relevance(0.89, calibration) == pytest.approx(0.5)
        assert displayed_relevance(0.70, calibration) == 0.0
        assert displayed_relevance(1.0, calibration) == 1.0

    def test_strong_band_means_ordering_score_at_least_0_918(self):
        calibration = relevance_calibration(self.E5)

        assert displayed_relevance(0.918, calibration) >= 0.70
        assert displayed_relevance(0.917, calibration) < 0.70

    def test_unknown_model_uses_identity_mapping_and_old_threshold(self):
        calibration = relevance_calibration("other/model")

        assert (calibration.floor, calibration.ceiling) == (0.0, 1.0)
        assert displayed_relevance(0.4321, calibration) == pytest.approx(0.4321)
        assert displayed_relevance(-0.2, calibration) == 0.0
        with patch("bddk_mcp.store.vector_store.SEMANTIC_RELEVANCE_THRESHOLD", None):
            assert semantic_threshold("other/model") == 0.50

    def test_default_threshold_per_model_and_environment_override(self):
        with patch("bddk_mcp.store.vector_store.SEMANTIC_RELEVANCE_THRESHOLD", None):
            assert semantic_threshold(self.E5) == 0.80
        with patch("bddk_mcp.store.vector_store.SEMANTIC_RELEVANCE_THRESHOLD", 0.3):
            assert semantic_threshold(self.E5) == 0.3
            assert semantic_threshold("other/model") == 0.3

    @staticmethod
    def _store(vector_hits, fts_hits, model=E5):
        vs = VectorStore.__new__(VectorStore)
        vs._embedding_model = model
        vs._vector_search = AsyncMock(return_value=vector_hits)
        vs._fts_search = AsyncMock(return_value=fts_hits)
        return vs

    @pytest.mark.asyncio
    async def test_gate_penalises_displayed_relevance_not_the_raw_cosine(self):
        vs = self._store(
            [
                {"doc_id": "a", "relevance": 0.86, "title": "A", "snippet": "a"},
                {"doc_id": "low", "relevance": 0.79, "title": "Low", "snippet": "low"},
            ],
            [],
        )

        with patch("bddk_mcp.store.vector_store.SEMANTIC_RELEVANCE_THRESHOLD", None):
            results = await vs._hybrid_search("capital adequacy", limit=10)

        # The 0.86 hit would fall under a 0.80 threshold if the gate (x0.65) hit the raw cosine.
        assert [r["doc_id"] for r in results] == ["a"]
        assert results[0]["semantic_relevance"] == 0.86
        assert results[0]["ordering_score"] == 0.86
        assert results[0]["relevance"] == round((0.86 - 0.82) / 0.14 * 0.65, 4)
        assert results[0]["confidence"] == "low"

    @pytest.mark.asyncio
    async def test_hybrid_hit_shows_calibrated_relevance_and_raw_semantic_score(self):
        vs = self._store(
            [{"doc_id": "a", "relevance": 0.90, "title": "A", "snippet": "a"}],
            [{"doc_id": "a", "fts_rank": 2.0, "title": "A", "snippet": "a"}],
        )

        with patch("bddk_mcp.store.vector_store.SEMANTIC_RELEVANCE_THRESHOLD", None):
            [hit] = await vs._hybrid_search("zzz", limit=10)

        assert hit["semantic_relevance"] == 0.90
        assert hit["ordering_score"] == round(0.90 + 0.045, 4)
        assert hit["relevance"] == round((0.945 - 0.82) / 0.14, 4)
        assert hit["confidence"] == "high"

    @pytest.mark.asyncio
    async def test_ties_in_displayed_relevance_keep_ordering_score_order(self):
        vs = self._store(
            [
                {"doc_id": "first_by_rrf", "relevance": 0.97, "title": "A", "snippet": "a"},
                {"doc_id": "higher_score", "relevance": 0.99, "title": "B", "snippet": "b"},
            ],
            [],
        )

        results = await vs._hybrid_search("q", limit=10)

        assert [r["relevance"] for r in results] == [0.65, 0.65]
        assert [r["doc_id"] for r in results] == ["higher_score", "first_by_rrf"]

    @pytest.mark.asyncio
    async def test_a_hit_far_below_the_top_is_kept_after_it_when_it_passes_the_threshold(self):
        # No score-gap cut: b is 0.06 below a on the ordering score (the FTS gate is open, so the
        # scores are not scaled), passes the 0.80 raw-cosine threshold and is listed after a.
        vs = self._store(
            [
                {"doc_id": "a", "relevance": 0.90, "title": "A", "snippet": "a"},
                {"doc_id": "b", "relevance": 0.84, "title": "B", "snippet": "b"},
                {"doc_id": "low", "relevance": 0.79, "title": "Low", "snippet": "low"},
            ],
            [{"doc_id": "z", "fts_rank": 1.0, "title": "Z", "snippet": "z", "fts_relaxed": True}],
        )

        with patch("bddk_mcp.store.vector_store.SEMANTIC_RELEVANCE_THRESHOLD", None):
            results = await vs._hybrid_search("q", limit=10)

        assert [r["doc_id"] for r in results] == ["a", "b"]
        assert results[0]["ordering_score"] - results[1]["ordering_score"] == pytest.approx(0.06)

    @pytest.mark.asyncio
    async def test_unknown_model_displays_the_ordering_score(self):
        vs = self._store(
            [{"doc_id": "a", "relevance": 0.6, "title": "A", "snippet": "a"}],
            [{"doc_id": "a", "fts_rank": 1.0, "title": "A", "snippet": "a"}],
            model="other/model",
        )

        with patch("bddk_mcp.store.vector_store.SEMANTIC_RELEVANCE_THRESHOLD", None):
            [hit] = await vs._hybrid_search("zzz", limit=10)

        assert hit["relevance"] == hit["ordering_score"] == round(0.6 + 0.045, 4)


class TestRelaxedFtsFallbackIsNotExact:
    """The relaxed OR fallback matches only some query words, so it never earns fts_exact."""

    @pytest.fixture
    async def article_store(self, vector_store, doc_store):
        await _store_and_index(
            vector_store,
            doc_store,
            doc_id="infosys",
            title="Bilgi Sistemleri Yönetmeliği",
            content="MADDE 12 - Bilgi sistemleri yönetimi bankanın yönetim kurulu tarafından gözetilir.",
        )
        await _store_and_index(
            vector_store,
            doc_store,
            doc_id="governance",
            title="Kurumsal Yönetim Yönetmeliği",
            content="MADDE 3 - Yönetim kurulu kurumsal yönetim ilkelerini uygular.",
        )
        # No dense candidates: simulates unrelated chunks falling outside the dense top-k.
        vector_store._vector_search = AsyncMock(return_value=[])
        return vector_store

    @pytest.mark.asyncio
    async def test_relaxed_or_hits_are_flagged_and_not_labelled_fts_exact(self, article_store):
        query = "madde 5 kredi kartı asgari ödeme tutarı"

        fts_hits = await article_store._fts_search(query, limit=50)
        results = await article_store._hybrid_search(query, limit=10)

        assert {hit["doc_id"] for hit in fts_hits} == {"infosys", "governance"}
        assert all(hit["fts_relaxed"] for hit in fts_hits)
        assert all(hit["match_type"] != "fts_exact" for hit in results)
        # Relaxed FTS-only hits have no semantic score and fall under the normal threshold.
        assert results == []

    @pytest.mark.asyncio
    async def test_strict_and_hit_of_a_legal_reference_query_stays_fts_exact(self, article_store):
        query = "madde 12 bilgi sistemleri yönetimi"

        fts_hits = await article_store._fts_search(query, limit=50)
        results = await article_store._hybrid_search(query, limit=10)

        assert [hit["doc_id"] for hit in fts_hits] == ["infosys"]
        assert fts_hits[0]["fts_relaxed"] is False
        assert [(hit["doc_id"], hit["match_type"]) for hit in results] == [("infosys", "fts_exact")]

    @pytest.mark.asyncio
    async def test_hybrid_search_does_not_grant_fts_exact_to_mocked_relaxed_hits(self):
        vs = VectorStore.__new__(VectorStore)
        vs._embedding_model = EMBEDDING_MODEL_NAME
        vs._vector_search = AsyncMock(return_value=[])
        vs._fts_search = AsyncMock(
            return_value=[
                {"doc_id": "strict", "fts_rank": 0.9, "title": "S", "snippet": "MADDE 5 - s", "fts_relaxed": False},
                {"doc_id": "loose", "fts_rank": 0.5, "title": "L", "snippet": "l", "fts_relaxed": True},
            ]
        )

        results = await vs._hybrid_search("sermaye yeterliliği madde 5", limit=10)

        assert [(r["doc_id"], r["match_type"]) for r in results] == [("strict", "fts_exact")]


class TestSearchHitDocumentMetadata:
    @pytest.mark.asyncio
    async def test_hits_carry_source_url_decision_number_and_chunk_section(self, vector_store, doc_store):
        stored = StoredDocument(
            document_id="kurul",
            title="Kurul Kararı",
            markdown_content="MADDE 4 - Sermaye yeterliliği hesaplamalarında standart yaklaşım kullanılır.",
            decision_date="01.02.2024",
            decision_number="10812",
            source_url="https://www.bddk.org.tr/Mevzuat/DokumanGetir/1",
        )
        await doc_store.store_document(stored)
        await vector_store.add_document(doc_id="kurul", title=stored.title, content=stored.markdown_content)

        dense = await vector_store._vector_search("sermaye yeterliliği hesaplamalarında", limit=5)
        lexical = await vector_store._fts_search("sermaye yeterliliği hesaplamalarında", limit=5)

        for hit in (dense[0], lexical[0]):
            assert hit["source_url"] == "https://www.bddk.org.tr/Mevzuat/DokumanGetir/1"
            assert hit["decision_number"] == "10812"
            assert hit["decision_date"] == "01.02.2024"
            assert (hit["section_type"], hit["section_ref"]) == ("madde", "4")


class TestVectorStoreLifecycle:
    """Readiness probe, embedding circuit breaker, add/replace/delete on a corpus file."""

    @pytest.mark.asyncio
    async def test_semantic_readiness_probe_accepts_empty_results_and_is_cached(self):
        store = VectorStore(MagicMock())
        search = AsyncMock(return_value=[])

        with patch.object(store, "search", new=search):
            await store.assert_semantic_search_ready()
            await store.assert_semantic_search_ready()

        search.assert_awaited_once_with("bankacılık düzenlemeleri", limit=1)
        assert store.semantic_search_ready is True

    @pytest.mark.asyncio
    async def test_semantic_readiness_probe_returns_stable_error(self):
        store = VectorStore(MagicMock())
        search = AsyncMock(side_effect=OSError("private runtime detail"))

        with (
            patch.object(store, "search", new=search),
            pytest.raises(SemanticSearchReadinessError) as exc_info,
        ):
            await store.assert_semantic_search_ready()

        assert str(exc_info.value) == "Semantic search readiness probe failed."
        assert "private runtime detail" not in str(exc_info.value)
        assert store.semantic_search_ready is False

    @pytest.mark.asyncio
    async def test_semantic_readiness_probe_propagates_cancellation(self):
        store = VectorStore(MagicMock())
        search = AsyncMock(side_effect=asyncio.CancelledError())

        with patch.object(store, "search", new=search), pytest.raises(asyncio.CancelledError):
            await store.assert_semantic_search_ready()

        assert store.semantic_search_ready is False

    @pytest.mark.asyncio
    async def test_embedding_failure_opens_non_retryable_runtime_circuit(self):
        store = VectorStore(MagicMock())
        load = MagicMock(side_effect=OSError("private model detail"))

        with (
            patch.object(store, "_ensure_embeddings", new=load),
            pytest.raises(SemanticSearchUnavailableError) as first_error,
        ):
            await store._embed(["test"], prefix="query")

        with pytest.raises(SemanticSearchUnavailableError) as second_error:
            await store._embed(["test"], prefix="query")

        assert str(first_error.value) == "Semantic search is unavailable in this runtime."
        assert str(second_error.value) == "Semantic search is unavailable in this runtime."
        assert "private model detail" not in str(first_error.value)
        assert load.call_count == 1
        assert store.semantic_search_ready is False

    @pytest.mark.asyncio
    async def test_add_document(self, vector_store, doc_store):
        chunks = await _store_and_index(
            vector_store,
            doc_store,
            doc_id="test_1",
            title="Test Document",
            content="This is a test document with some content.",
            category="Rehber",
        )
        assert chunks >= 1
        assert await vector_store.has_document("test_1")
        assert not await vector_store.has_document("nonexistent")
        assert await vector_store.chunk_count() == chunks

    @pytest.mark.asyncio
    async def test_add_empty_document(self, vector_store):
        chunks = await vector_store.add_document(doc_id="empty", title="Empty", content="", category="")
        assert chunks == 0
        assert not await vector_store.has_document("empty")

    @pytest.mark.asyncio
    async def test_add_document_replaces_existing(self, vector_store, doc_store, corpus_db):
        await _store_and_index(vector_store, doc_store, doc_id="d1", title="V1", content="Version one content")
        await _store_and_index(vector_store, doc_store, doc_id="d1", title="V2", content="Version two content updated")

        rows = await corpus_db.run(
            lambda conn: conn.execute("SELECT title, chunk_text FROM chunks WHERE doc_id = 'd1'").fetchall()
        )
        assert {row["title"] for row in rows} == {"V2"}
        assert all("two" in row["chunk_text"] for row in rows)

    @pytest.mark.asyncio
    async def test_stale_chunks_are_hidden_until_matching_index_is_published(self, vector_store, doc_store):
        await _store_and_index(vector_store, doc_store, doc_id="d1", title="V1", content="Version one content")
        assert await vector_store.has_document("d1")

        await doc_store.store_document(
            StoredDocument(document_id="d1", title="V2", markdown_content="Version two content updated")
        )
        assert not await vector_store.has_document("d1")
        assert await vector_store.chunk_count() == 0
        assert await vector_store._vector_search("version content", limit=5) == []

        with pytest.raises(VectorIndexConsistencyError, match="stored document hash does not match"):
            await vector_store.add_document(doc_id="d1", title="V1", content="Version one content")

        await vector_store.add_document(doc_id="d1", title="V2", content="Version two content updated")
        assert await vector_store.has_document("d1")

    @pytest.mark.asyncio
    async def test_corpus_records_the_embedding_model_and_rejects_another(self, vector_store, doc_store, corpus_db):
        await _store_and_index(vector_store, doc_store, doc_id="d1", title="V1", content="Version one content")
        readiness = corpus_db.readiness_sync(require_corpus=False)
        assert readiness.issues == []
        assert readiness.embedding_model

        other = VectorStore(corpus_db, embedding_model="other/model", embed_fn=fake_embed, tokenizer=FakeTokenizer())
        with pytest.raises(RuntimeError, match="different embedding model"):
            await other.add_document(doc_id="d1", title="V1", content="Version one content")

    @pytest.mark.asyncio
    async def test_delete_document(self, vector_store, doc_store):
        await _store_and_index(vector_store, doc_store, doc_id="del_me", title="Delete Me", content="Some content")
        assert await vector_store.has_document("del_me")

        assert await vector_store.delete_document("del_me") is True
        assert not await vector_store.has_document("del_me")
        assert await vector_store.delete_document("never_existed") is False


class TestVectorStoreSearch:
    """Semantic search on a small corpus with the deterministic fake embedder."""

    @pytest.fixture
    async def populated_store(self, vector_store, doc_store):
        with patch("bddk_mcp.store.vector_store.SEMANTIC_RELEVANCE_THRESHOLD", 0.3):
            await _store_and_index(
                vector_store,
                doc_store,
                doc_id="capital",
                title="Sermaye Yeterliliği Rehberi",
                content="Bu rehber bankacılık sektöründe sermaye yeterliliği hesaplamalarını düzenler.",
                category="Rehber",
            )
            await _store_and_index(
                vector_store,
                doc_store,
                doc_id="interest",
                title="Faiz Oranı Riski Yönetmeliği",
                content="Banka faiz oranı riskini ölçmek için standart yaklaşım kullanır.",
                category="Yönetmelik",
            )
            await _store_and_index(
                vector_store,
                doc_store,
                doc_id="loans",
                title="Kredi İşlemleri Genelgesi",
                content="Bankaların kredi işlemlerine ilişkin genel kurallar ve prosedürler.",
                category="Genelge",
            )
            yield vector_store

    @pytest.mark.asyncio
    async def test_search_returns_results(self, populated_store):
        hits = await populated_store.search("sermaye yeterliliği hesaplamalarını", limit=10)
        assert len(hits) >= 1
        assert hits[0]["doc_id"] == "capital"
        assert hits[0]["match_type"] == "hybrid"

    @pytest.mark.asyncio
    async def test_search_relevance_ordering(self, populated_store):
        hits = await populated_store.search("faiz oranı riskini ölçmek", limit=10)
        assert len(hits) >= 1
        assert hits[0]["doc_id"] == "interest"

    @pytest.mark.asyncio
    async def test_search_with_category_filter(self, populated_store):
        hits = await populated_store.search("bankaların kredi işlemlerine ilişkin", limit=10, category="Genelge")
        assert hits and all(h["category"] == "Genelge" for h in hits)
        assert await populated_store.search("bankaların kredi işlemlerine ilişkin", limit=10, category="Kanun") == []

    @pytest.mark.asyncio
    async def test_search_without_keyword_overlap_is_gated(self, populated_store):
        hits = await populated_store.search("quantum physics dark matter", limit=10)
        assert hits == []

    @pytest.mark.asyncio
    async def test_search_deduplicates_by_doc_id(self, populated_store):
        hits = await populated_store.search("bankaların kredi işlemlerine ilişkin genel kurallar", limit=10)
        doc_ids = [h["doc_id"] for h in hits]
        assert len(doc_ids) == len(set(doc_ids))

    @pytest.mark.asyncio
    async def test_search_includes_relevance_score(self, populated_store):
        hits = await populated_store.search("sermaye yeterliliği hesaplamalarını", limit=5)
        assert hits
        for h in hits:
            assert 0 <= h["relevance"] <= 1
            assert h["confidence"] in {"high", "medium", "low"}

    @pytest.mark.asyncio
    async def test_fts_matches_ignore_case_and_diacritics(self, populated_store):
        hits = await populated_store._fts_search("SERMAYE YETERLILIGI", limit=5)
        assert [h["doc_id"] for h in hits] == ["capital"]

    @pytest.mark.asyncio
    async def test_search_quality_uses_full_document_markdown(self, populated_store, doc_store):
        matching_chunk = "Bankaların kredi işlemlerine ilişkin genel kurallar ve prosedürler. "
        await _store_and_index(
            populated_store,
            doc_store,
            doc_id="quality_formula",
            title="Formül uyarısı belgesi",
            content=matching_chunk * 80 + "\n\nBu metinde aşağıdaki formül kullanılır.",
            category="Genelge",
        )
        hits = await populated_store.search("bankaların kredi işlemlerine ilişkin genel kurallar", limit=10)
        quality_hit = next(hit for hit in hits if hit["doc_id"] == "quality_formula")
        assert quality_hit["quality_label"] == "warning"
        assert "formula_ref_without_latex_or_image" in quality_hit["quality_flags"]
        assert "aşağıdaki formül" not in quality_hit["snippet"]


def test_caption_stretch_opens_the_chunk_of_the_following_section():
    text = (
        "MADDE 9 - (1) Bankalar karşılık ayırır.\n\n"
        "İKİNCİ BÖLÜM\nGenel Esaslar\n\n"
        "MADDE 10 - (1) Başka hüküm uygulanır.\n"
    )

    chunks = _chunk_document("mevzuat_22599", text, tokenizer=WhitespaceTokenizer(), target_tokens=50, token_overlap=0)

    assert all(chunk.section_type for chunk in chunks)
    assert all(chunk.chunk_text == text[chunk.start_char : chunk.end_char] for chunk in chunks)
    caption = next(chunk for chunk in chunks if "İKİNCİ BÖLÜM" in chunk.chunk_text)
    assert caption.chunk_text.startswith("İKİNCİ BÖLÜM\nGenel Esaslar\n\nMADDE 10 -")
    assert (caption.section_type, caption.section_ref) == ("madde", "10")
    assert caption.section_start_char == text.index("MADDE 10")
    assert caption.start_char < caption.section_start_char


def test_text_between_sections_that_is_no_caption_stays_its_own_chunk():
    text = "MADDE 1 - (1) Birinci.\n\nİMZA\n\nADI-SOYADI\n\nMADDE 2 - (1) İkinci.\n\nBu metin bir paragraftır.\n"
    signature = text.index("İMZA")

    chunks = _chunk_document("mevzuat_16071", text, tokenizer=WhitespaceTokenizer(), target_tokens=50, token_overlap=0)

    standalone = [chunk for chunk in chunks if not chunk.section_type]
    assert [chunk.start_char for chunk in standalone] == [signature]
    assert standalone[0].chunk_text.strip() == "İMZA\n\nADI-SOYADI"
    assert next(chunk for chunk in chunks if chunk.section_ref == "2").chunk_text.startswith("MADDE 2 -")


def test_deletion_note_is_no_caption_and_is_not_merged_into_the_next_section():
    # kgk_tms_12: "69–70 [Silinmiştir]" records deleted paragraphs; it must not be bound to paragraf 71.
    text = "MADDE 1 - (1) Birinci.\n\nSunum\n\n69–70 [Silinmiştir]\n\nMahsup\n\nMADDE 2 - (1) İkinci.\n"
    note = text.index("Sunum")

    chunks = _chunk_document("kgk_tms_12", text, tokenizer=WhitespaceTokenizer(), target_tokens=50, token_overlap=0)

    standalone = [chunk for chunk in chunks if not chunk.section_type]
    assert [chunk.start_char for chunk in standalone] == [note]
    assert standalone[0].chunk_text.strip() == "Sunum\n\n69–70 [Silinmiştir]"
    assert not any("[Silinmiştir]" in chunk.chunk_text for chunk in chunks if chunk.section_type)


def test_seed_deletion_note_is_not_merged_into_the_chunk_of_the_next_paragraph():
    import json
    from pathlib import Path

    from bddk_mcp.store.section_index import extract_document_sections
    from bddk_mcp.store.vector_store import _chunk_spans

    seed = Path(__file__).resolve().parents[1] / "seed_data" / "documents.json"
    text = next(
        d["markdown_content"] for d in json.loads(seed.read_text(encoding="utf-8")) if d["document_id"] == "kgk_tms_12"
    )
    note = text.index("69–70 [Silinmiştir]")
    spans = _chunk_spans(text, extract_document_sections("kgk_tms_12", text))

    covering = [span for span in spans if span.start_char <= note < span.end_char]
    assert len(covering) == 1 and covering[0].section is None
    paragraph_71 = next(span for span in spans if span.section is not None and span.section.section_ref == "71")
    assert paragraph_71.start_char == paragraph_71.section.start_char > note


def test_seed_has_no_standalone_caption_chunk_except_a_document_tail():
    import json
    from pathlib import Path

    from bddk_mcp.store.section_index import extract_document_sections
    from bddk_mcp.store.vector_store import _chunk_spans, _is_caption_stretch

    seed = Path(__file__).resolve().parents[1] / "seed_data" / "documents.json"
    tails = []
    standalone = 0
    for doc in json.loads(seed.read_text(encoding="utf-8")):
        text = doc["markdown_content"] or ""
        sections = extract_document_sections(doc["document_id"], text)
        spans = _chunk_spans(text, sections)
        for index, span in enumerate(spans):
            if span.section is not None:
                # A span bound to a section starts at the section or at the caption stretch above it.
                assert span.start_char <= span.section.start_char < span.end_char
                assert span.start_char == span.section.start_char or _is_caption_stretch(
                    text[span.start_char : span.section.start_char], opens_document=span.start_char == 0
                )
                continue
            if any(s.start_char <= span.start_char < s.end_char for s in sections):
                continue
            standalone += 1
            if _is_caption_stretch(text[span.start_char : span.end_char], opens_document=span.start_char == 0):
                assert index == len(spans) - 1, (doc["document_id"], span.start_char)
                tails.append(doc["document_id"])
    # Measured: 1041 "45. Ödenmemiş borçlar", and the "Eki/Ekleri için tıklayınız" link of 21425 and 40520.
    assert sorted(tails) == ["1041", "mevzuat_21425", "mevzuat_40520"]
    assert standalone < 30


class TestQueryNormalisation:
    """Case and missing Turkish letters are folded on the keyword side; the embedded query is normalised to match."""

    @staticmethod
    def _vocabulary():
        texts = ["Müşteri şikayetleri ve sızma testi raporu."] * 25 + ["musteri kaydı"] * 1
        return _build_query_vocabulary(texts)

    def test_shouted_queries_are_lower_cased_with_the_matching_rules(self):
        vocabulary = self._vocabulary()

        assert normalize_query("SERMAYE KORUMA TAMPONU", vocabulary) == "sermaye koruma tamponu"
        assert normalize_query("MÜŞTERİ ŞİKAYETLERİ", vocabulary) == "müşteri şikayetleri"
        assert normalize_query("İÇ DENETİM BİRİMİ", vocabulary) == "iç denetim birimi"

    def test_short_all_caps_queries_are_codes_and_references_and_stay_as_typed(self):
        vocabulary = self._vocabulary()

        for query in ("TFRS 9", "IFRS 9", "TMS 39", "MADDE 5", "BDDK", "SYR", "LKO", "İÇ DENETİM"):
            assert normalize_query(query, vocabulary) == query

    def test_a_shouted_sentence_is_lower_cased_but_keeps_its_abbreviations(self):
        vocabulary = _build_query_vocabulary(
            ["Bankalar BDDK ve TFRS hükümlerine göre hesaplama yapar."] * 25 + ["hesaplaması nasıl yapılır"] * 25
        )

        assert normalize_query("LKO HESAPLAMASI NASIL YAPILIR", vocabulary) == "LKO hesaplaması nasıl yapılır"
        assert normalize_query("BDDK TFRS HESAPLAMASI NASIL", vocabulary) == "BDDK TFRS hesaplaması nasıl"

    def test_a_query_without_turkish_letters_gets_them_back_from_the_corpus_vocabulary(self):
        vocabulary = self._vocabulary()

        assert normalize_query("musteri sikayetleri", vocabulary) == "müşteri şikayetleri"
        assert normalize_query("BILGI SISTEMLERI SIZMA TESTI", vocabulary) == "bilgi sistemleri sızma testi"
        assert normalize_query("Musteri sikayetleri", vocabulary) == "Müşteri şikayetleri"

    def test_a_folded_form_with_two_common_turkish_spellings_is_left_as_typed(self):
        vocabulary = _build_query_vocabulary(
            ["kısmı"] * 60 + ["kısmi ödeme"] * 30 + ["aynı"] * 100 + ["aynî sermaye"] * 25 + ["ödeme sermaye"] * 30
        )

        assert normalize_query("kismi odeme", vocabulary) == "kismi ödeme"
        assert normalize_query("ayni sermaye", vocabulary) == "ayni sermaye"

    def test_a_query_whose_words_are_mostly_not_in_the_corpus_is_not_restored(self):
        vocabulary = _build_query_vocabulary(["iş ve işlem"] * 50 + ["is"] * 1)

        assert normalize_query("what is NPL", vocabulary) == "what is NPL"
        assert normalize_query("how is the loan classified", vocabulary) == "how is the loan classified"

    def test_a_circumflex_is_a_turkish_letter_and_the_query_passes_unchanged(self):
        vocabulary = _build_query_vocabulary(["sermaye ödenir"] * 50 + ["kâr payı"] * 50)

        assert normalize_query("aynî sermaye odenir", vocabulary) == "aynî sermaye odenir"
        assert normalize_query("kâr payi odenir", vocabulary) == "kâr payi odenir"

    def test_restoration_needs_a_frequent_corpus_form_and_a_rare_typed_form(self):
        rare = _build_query_vocabulary(["müşteri"] * 19)
        typed_common = _build_query_vocabulary(["müşteri"] * 25 + ["musteri"] * 2)

        assert normalize_query("musteri", rare) == "musteri"
        assert normalize_query("musteri", typed_common) == "musteri"

    def test_a_query_with_a_turkish_letter_and_abbreviations_in_a_mixed_query_are_left_alone(self):
        vocabulary = self._vocabulary()

        assert normalize_query("müşteri sikayetleri", vocabulary) == "müşteri sikayetleri"
        assert normalize_query("LKO musteri sikayetleri", vocabulary) == "LKO müşteri şikayetleri"
        assert normalize_query("Bankacılık KANUNU madde 5", vocabulary) == "Bankacılık KANUNU madde 5"

    def test_the_query_is_nfc_normalised(self):
        decomposed = "müşteri"

        assert normalize_query(decomposed, self._vocabulary()) == "müşteri"

    @pytest.mark.asyncio
    async def test_search_embeds_the_normalised_query_and_says_what_was_searched(self, corpus_db, doc_store):
        embedded: list[str] = []

        def recording_embed(texts):
            embedded.extend(texts)
            return fake_embed(texts)

        store = VectorStore(corpus_db, embed_fn=recording_embed, tokenizer=FakeTokenizer())
        await _store_and_index(
            store, doc_store, doc_id="d1", title="Şikayet", content="Müşteri şikayetleri kayda alınır. " * 25
        )
        embedded.clear()

        hits = await store.search("MUSTERI SIKAYETLERI", limit=5)
        unchanged = await store.search("müşteri şikayetleri", limit=5)

        assert "query: müşteri şikayetleri" in embedded
        assert "query: MUSTERI SIKAYETLERI" not in embedded
        assert hits.searched_as == "müşteri şikayetleri"
        assert unchanged.searched_as is None


class TestAbbreviationExpansion:
    """A glossary abbreviation in the query is also searched as its expansion, and the answer says so."""

    @pytest.fixture
    async def recording_store(self, corpus_db, doc_store):
        embedded: list[str] = []

        def recording_embed(texts):
            embedded.extend(texts)
            return fake_embed(texts)

        store = VectorStore(corpus_db, embed_fn=recording_embed, tokenizer=FakeTokenizer())
        await _store_and_index(
            store,
            doc_store,
            doc_id="lko_reg",
            title="Yönetmelik",
            content="MADDE 4 - Likidite karşılama oranı yüzde yüzden az olamaz.",
        )
        await _store_and_index(
            store, doc_store, doc_id="lko_kk", title="Kurul Kararı", content="Bankalar LKO hesaplamasından muaftır."
        )
        embedded.clear()
        return store, embedded

    @pytest.mark.asyncio
    async def test_expansion_is_embedded_searched_by_keyword_and_stated(self, recording_store):
        store, embedded = recording_store

        hits = await store.search("LKO", limit=10)

        assert embedded == ["query: LKO likidite karşılama oranı"]
        assert {hit["doc_id"] for hit in hits} == {"lko_reg", "lko_kk"}
        assert hits.expansions == [("LKO", "likidite karşılama oranı")]

    @pytest.mark.asyncio
    async def test_the_expansion_is_not_appended_when_the_query_already_contains_it(self, recording_store):
        store, embedded = recording_store

        hits = await store.search("LKO likidite karşılama oranı", limit=10)

        assert embedded == ["query: LKO likidite karşılama oranı"]
        assert hits.expansions == []

    @pytest.mark.asyncio
    async def test_an_unknown_abbreviation_changes_nothing(self, recording_store):
        store, embedded = recording_store

        hits = await store.search("XYZQ hesaplaması", limit=10)

        assert embedded == ["query: XYZQ hesaplaması"]
        assert hits.expansions == []
        assert hits.searched_as is None


class TestSearchHonesty:
    @pytest.mark.asyncio
    async def test_a_hit_at_the_calibration_ceiling_is_marked_clamped(self):
        vs = VectorStore.__new__(VectorStore)
        vs._embedding_model = "intfloat/multilingual-e5-base"
        vs._vector_search = AsyncMock(
            return_value=[
                {"doc_id": "top", "relevance": 0.95, "title": "T", "snippet": "t"},
                {"doc_id": "below", "relevance": 0.86, "title": "B", "snippet": "b"},
            ]
        )
        vs._fts_search = AsyncMock(
            return_value=[{"doc_id": "top", "fts_rank": 2.0, "title": "T", "snippet": "t", "fts_relaxed": False}]
        )

        with patch("bddk_mcp.store.vector_store.SEMANTIC_RELEVANCE_THRESHOLD", None):
            top, below = await vs._hybrid_search("zzz", limit=10)

        assert top["relevance"] == 1.0 and top["relevance_clamped"] is True
        assert top["semantic_relevance"] == 0.95
        assert below["relevance_clamped"] is False

    @pytest.mark.asyncio
    async def test_a_query_of_function_words_only_is_not_searched(self, vector_store, doc_store):
        await _store_and_index(vector_store, doc_store, doc_id="d1", title="T", content="ve ile bir bu da de için")
        vector_store._hybrid_search = AsyncMock(return_value=[])

        hits = await vector_store.search("ve ile bir bu da de için", limit=10)
        await vector_store.search("ve sermaye", limit=10)

        single_letter = await vector_store.search("a", limit=10)

        assert hits == [] and hits.no_content_words is True
        assert single_letter == [] and single_letter.no_content_words is True
        vector_store._hybrid_search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_bare_decision_number_returns_that_decision_first(self, vector_store, doc_store):
        await doc_store.store_document(
            StoredDocument(
                document_id="775",
                title="Kurul Kararı",
                markdown_content="Kurul kararı metni: kredilerin sınıflandırılması.",
                category="Kurul Kararı",
                decision_number="8948",
            )
        )
        await vector_store.add_document(
            doc_id="775", title="Kurul Kararı", content="Kurul kararı metni: kredilerin sınıflandırılması."
        )
        await _store_and_index(vector_store, doc_store, doc_id="other", title="T", content="8948 tutarında 8948 işlem")

        for query in ("8948", "8948 sayılı karar", "8948 sayılı Kurul Kararı"):
            hits = await vector_store.search(query, limit=10)
            assert hits[0]["doc_id"] == "775", query
            assert hits[0]["decision_number_match"] is True
            assert hits[0]["match_type"] == "decision_number"
            assert [hit["doc_id"] for hit in hits].count("775") == 1

        assert all(not hit.get("decision_number_match") for hit in await vector_store.search("9999", limit=10))
        assert await vector_store.search("8948", limit=10, category="Yönetmelik") == []
