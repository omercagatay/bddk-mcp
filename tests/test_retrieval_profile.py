"""Focused reproducibility and fail-closed tests for the retrieval profile."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from types import MappingProxyType, ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from bddk_mcp.quality import markdown_quality
from bddk_mcp.quality.markdown_quality import QualityFailure, quality_retrieval_profile_descriptor
from bddk_mcp.store import vector_store
from bddk_mcp.store.vector_store import (
    RetrievalProfileError,
    VectorStore,
    _chunk_document,
    _clear_local_model_identity_cache,
    _load_embedding_tokenizer,
    retrieval_profile_descriptor,
    retrieval_profile_hash,
)


@pytest.fixture(autouse=True)
def stable_remote_profile(monkeypatch):
    """Keep determinant tests independent of the executing workstation."""

    monkeypatch.setattr(vector_store, "EMBEDDING_MODEL_PATH", "")
    monkeypatch.setattr(vector_store, "EMBEDDING_MODEL_NAME", "reviewed/embedding")
    monkeypatch.setattr(vector_store, "EMBEDDING_MODEL_REVISION", "a" * 40)
    monkeypatch.setattr(vector_store, "RERANKER_ENABLED", False)
    monkeypatch.setattr(vector_store, "RERANKER_MODEL_PATH", "")
    monkeypatch.setattr(vector_store, "RERANKER_MODEL_NAME", "reviewed/reranker")
    monkeypatch.setattr(vector_store, "RERANKER_MODEL_REVISION", "b" * 40)
    monkeypatch.setattr(
        vector_store,
        "_installed_retrieval_package_versions",
        lambda: {"sentence-transformers": "test-version", "transformers": "test-version"},
    )
    monkeypatch.setattr(
        vector_store,
        "quality_retrieval_profile_descriptor",
        lambda: {
            "assessment_policy_version": "test-policy",
            "registry_format_version": 1,
            "registry_sha256": "c" * 64,
            "failure_count": 1,
        },
    )
    _clear_local_model_identity_cache()
    yield
    _clear_local_model_identity_cache()


def _write_model_tree(root: Path, content: str = "weights-v1") -> None:
    (root / "tokenizer").mkdir(parents=True)
    (root / "empty-directory").mkdir()
    (root / "config.json").write_text('{"model_type":"test"}', encoding="utf-8")
    (root / "tokenizer" / "model.bin").write_text(content, encoding="utf-8")


def test_descriptor_binds_end_to_end_retrieval_determinants():
    descriptor = retrieval_profile_descriptor()

    assert descriptor["schema_version"] == 2
    assert descriptor["scope"] == "published_end_to_end_document_retrieval"
    assert descriptor["embedding"]["model"] == {
        "source": "remote_repository_commit",
        "repository": "reviewed/embedding",
        "revision": "a" * 40,
    }
    assert descriptor["embedding"]["dimension"] == vector_store.EMBEDDING_DIMENSION
    assert descriptor["embedding"]["passage_prefix"] == "passage: "
    assert descriptor["embedding"]["query_prefix"] == "query: "
    assert descriptor["embedding"]["normalize_embeddings"] is True
    assert descriptor["publication_verification"] == {
        "version": vector_store.PUBLICATION_EMBEDDING_VERIFICATION_VERSION,
        "regenerate_every_chunk_embedding": True,
        "stored_dimension": vector_store.EMBEDDING_DIMENSION,
        "require_finite_components": True,
        "require_nonzero_l2_norm": True,
        "maximum_absolute_component_error": 0.001,
        "minimum_cosine_similarity": 0.99999,
        "hardware_calibration": "representative_cpu_gpu_acceptance_gate_required",
    }
    assert descriptor["chunking"]["version"] == vector_store.CHUNKER_PROFILE_VERSION
    assert descriptor["chunking"]["tokenizer_model"] == descriptor["embedding"]["model"]
    assert descriptor["section_parser"]["version"] == vector_store.SECTION_PARSER_PROFILE_VERSION
    assert descriptor["pagination"]["page_size_chars"] == vector_store.PAGE_SIZE
    assert descriptor["retrieval"]["fusion"]["rrf_k"] == vector_store.HYBRID_RRF_K
    assert descriptor["retrieval"]["fusion"]["semantic_signal_weight"] == 1.0
    assert descriptor["retrieval"]["fusion"]["keyword_signal_weight"] == 1.0
    assert descriptor["retrieval"]["scoring"]["normalized_keyword_rank_boost"] == 0.045
    assert descriptor["retrieval"]["fts"]["version"] == vector_store.FTS_PROFILE_VERSION
    assert descriptor["retrieval"]["document_store"]["rank_threshold"] == vector_store.FTS_RANK_THRESHOLD
    assert descriptor["retrieval"]["document_store"]["version"] == vector_store.DOCUMENT_STORE_SEARCH_PROFILE_VERSION
    assert descriptor["retrieval"]["section_search"]["version"] == vector_store.SECTION_SEARCH_PROFILE_VERSION
    assert descriptor["reranker"]["enabled"] is False
    assert descriptor["reranker"]["model"] == {
        "source": "remote_repository_commit",
        "repository": "reviewed/reranker",
        "revision": "b" * 40,
    }
    assert descriptor["quality_signals"]["registry_sha256"] == "c" * 64
    assert descriptor["runtime"]["packages"]["transformers"] == "test-version"
    assert re.fullmatch(r"[0-9a-f]{64}", retrieval_profile_hash())


@pytest.mark.parametrize(
    ("setting", "replacement"),
    [
        ("CHUNKER_PROFILE_VERSION", "changed-chunker"),
        ("SECTION_PARSER_PROFILE_VERSION", "changed-section-parser"),
        ("EMBEDDING_MODEL_REVISION", "d" * 40),
        ("EMBEDDING_DIMENSION", 1024),
        ("EMBEDDING_CHUNK_TARGET_TOKENS", 399),
        ("PAGE_SIZE", 4_999),
        ("FTS_RANK_THRESHOLD", 0.02),
        ("HYBRID_RRF_K", 61),
        ("HYBRID_SEARCH", False),
        ("SEMANTIC_RELEVANCE_THRESHOLD", 0.51),
        ("_LEXICAL_RELEVANCE_BOOST", 0.046),
        ("FTS_PROFILE_VERSION", "changed-fts-policy"),
        ("DOCUMENT_STORE_SEARCH_PROFILE_VERSION", "changed-document-store-policy"),
        ("SECTION_SEARCH_PROFILE_VERSION", "changed-section-search-policy"),
        ("RERANKER_MODEL_REVISION", "e" * 40),
        ("RERANKER_TOP_N", 21),
    ],
)
def test_hash_changes_when_a_bound_determinant_changes(monkeypatch, setting, replacement):
    baseline = retrieval_profile_hash()

    monkeypatch.setattr(vector_store, setting, replacement)

    assert retrieval_profile_hash() != baseline


def test_hash_changes_with_package_or_quality_registry_identity(monkeypatch):
    baseline = retrieval_profile_hash()
    monkeypatch.setattr(
        vector_store,
        "_installed_retrieval_package_versions",
        lambda: {"sentence-transformers": "changed", "transformers": "test-version"},
    )
    package_hash = retrieval_profile_hash()
    monkeypatch.setattr(
        vector_store,
        "quality_retrieval_profile_descriptor",
        lambda: {
            "assessment_policy_version": "test-policy",
            "registry_format_version": 1,
            "registry_sha256": "e" * 64,
            "failure_count": 1,
        },
    )

    assert package_hash != baseline
    assert retrieval_profile_hash() != package_hash


def test_quality_registry_profile_is_canonical_and_content_bound(monkeypatch):
    baseline = quality_retrieval_profile_descriptor()
    failures = dict(markdown_quality._QUALITY_FAILURES)
    failures["profile-test"] = QualityFailure(
        document_id="profile-test",
        reason="test-reason",
        preferred_backfill="test-backfill",
    )
    monkeypatch.setattr(markdown_quality, "_QUALITY_FAILURES", MappingProxyType(failures))

    changed = quality_retrieval_profile_descriptor()

    assert re.fullmatch(r"[0-9a-f]{64}", baseline["registry_sha256"])
    assert changed["registry_sha256"] != baseline["registry_sha256"]
    assert changed["failure_count"] == baseline["failure_count"] + 1


def test_local_identity_is_path_free_but_changes_with_tree_content(tmp_path, monkeypatch):
    first = tmp_path / "first-location"
    second = tmp_path / "second-location"
    _write_model_tree(first)
    _write_model_tree(second)

    monkeypatch.setattr(vector_store, "EMBEDDING_MODEL_PATH", str(first))
    first_descriptor = retrieval_profile_descriptor()
    first_hash = retrieval_profile_hash()
    monkeypatch.setattr(vector_store, "EMBEDDING_MODEL_PATH", str(second))
    second_descriptor = retrieval_profile_descriptor()
    second_hash = retrieval_profile_hash()

    assert first_descriptor["embedding"]["model"] == second_descriptor["embedding"]["model"]
    assert first_hash == second_hash
    rendered = json.dumps(second_descriptor, sort_keys=True)
    assert str(first) not in rendered
    assert str(second) not in rendered

    (second / "tokenizer" / "model.bin").write_text("weights-v2", encoding="utf-8")
    assert retrieval_profile_hash() != second_hash


def test_local_model_root_and_nested_symlinks_are_rejected(tmp_path, monkeypatch):
    real_model = tmp_path / "real-model"
    _write_model_tree(real_model)
    root_link = tmp_path / "model-link"
    root_link.symlink_to(real_model, target_is_directory=True)
    monkeypatch.setattr(vector_store, "EMBEDDING_MODEL_PATH", str(root_link))

    with pytest.raises(RetrievalProfileError, match="symlink"):
        retrieval_profile_descriptor()

    nested_model = tmp_path / "nested-model"
    nested_model.mkdir()
    (nested_model / "config.json").write_text("{}", encoding="utf-8")
    (nested_model / "linked.bin").symlink_to(real_model / "tokenizer" / "model.bin")
    monkeypatch.setattr(vector_store, "EMBEDDING_MODEL_PATH", str(nested_model))

    with pytest.raises(RetrievalProfileError, match="symlink"):
        retrieval_profile_descriptor()


@pytest.mark.parametrize("kind", ["missing", "regular-file", "empty-directory"])
def test_invalid_local_model_assets_fail_closed(tmp_path, monkeypatch, kind):
    model_path = tmp_path / kind
    if kind == "regular-file":
        model_path.write_text("not a model directory", encoding="utf-8")
    elif kind == "empty-directory":
        model_path.mkdir()
    monkeypatch.setattr(vector_store, "EMBEDDING_MODEL_PATH", str(model_path))

    with pytest.raises(RetrievalProfileError):
        retrieval_profile_descriptor()


def test_special_file_and_malformed_local_path_fail_closed(tmp_path, monkeypatch):
    model_path = tmp_path / "special-file-model"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    os.mkfifo(model_path / "weights.fifo")
    monkeypatch.setattr(vector_store, "EMBEDDING_MODEL_PATH", str(model_path))

    with pytest.raises(RetrievalProfileError, match="regular files"):
        retrieval_profile_descriptor()

    monkeypatch.setattr(vector_store, "EMBEDDING_MODEL_PATH", f"{tmp_path}\0invalid")
    with pytest.raises(RetrievalProfileError):
        retrieval_profile_descriptor()


def test_remote_tokenizer_uses_exact_pinned_revision(monkeypatch):
    calls: list[tuple[str, dict]] = []

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append((source, kwargs))
            return object()

    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setattr(vector_store, "EMBEDDING_CHUNK_MODE", "token")

    assert _load_embedding_tokenizer() is not None
    assert calls == [
        (
            "reviewed/embedding",
            {
                "use_fast": True,
                "trust_remote_code": False,
                "local_files_only": False,
                "revision": "a" * 40,
            },
        )
    ]


def test_tokenizer_load_failure_and_missing_tokenizer_fail_closed(monkeypatch):
    class FailingAutoTokenizer:
        @classmethod
        def from_pretrained(cls, _source, **_kwargs):
            raise OSError("unavailable")

    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = FailingAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setattr(vector_store, "EMBEDDING_CHUNK_MODE", "token")

    with pytest.raises(RetrievalProfileError, match="could not be loaded"):
        _load_embedding_tokenizer()
    with pytest.raises(RetrievalProfileError, match="tokenizer is unavailable"):
        _chunk_document("doc", "regulatory text", tokenizer=None, use_token_chunking=True)

    store = VectorStore(MagicMock())
    store._embed_fn = object()
    with pytest.raises(RetrievalProfileError, match="could not be loaded"):
        store._chunk_tokenizer()


def test_store_chunking_uses_the_explicit_profile_bound_fast_tokenizer(monkeypatch):
    sentinel = object()
    store = VectorStore(MagicMock(), embedding_model="reviewed/embedding")
    ensure_embeddings = MagicMock()
    monkeypatch.setattr(store, "_ensure_embeddings", ensure_embeddings)
    load_tokenizer = MagicMock(return_value=sentinel)
    monkeypatch.setattr(vector_store, "_load_embedding_tokenizer", load_tokenizer)
    monkeypatch.setattr(vector_store, "EMBEDDING_CHUNK_MODE", "token")

    assert store._chunk_tokenizer() is sentinel
    assert store._chunk_tokenizer() is sentinel
    ensure_embeddings.assert_called_once_with()
    load_tokenizer.assert_called_once_with("reviewed/embedding")


def test_tokenizer_refuses_local_asset_mutation_after_embedding_load(tmp_path, monkeypatch):
    model_path = tmp_path / "embedding-model"
    _write_model_tree(model_path)

    class FakeSentenceTransformer:
        @staticmethod
        def get_sentence_embedding_dimension():
            return vector_store.EMBEDDING_DIMENSION

    module = SimpleNamespace(SentenceTransformer=lambda *_args, **_kwargs: FakeSentenceTransformer())
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(vector_store, "EMBEDDING_MODEL_PATH", str(model_path))
    tokenizer_loader = MagicMock(return_value=object())
    monkeypatch.setattr(vector_store, "_load_embedding_tokenizer", tokenizer_loader)
    store = VectorStore(MagicMock())

    store._ensure_embeddings()
    (model_path / "tokenizer" / "model.bin").write_text("weights-v2", encoding="utf-8")

    with pytest.raises(RetrievalProfileError, match="changed after this store initialized"):
        store._chunk_tokenizer()

    tokenizer_loader.assert_not_called()
    assert store._chunk_tokenizer_fn is None


def test_embedding_loader_pins_kwargs_and_detects_local_asset_mutation(tmp_path, monkeypatch):
    model_path = tmp_path / "embedding-model"
    _write_model_tree(model_path)
    calls: list[tuple[str, dict]] = []

    class MutatingSentenceTransformer:
        def __init__(self, source, **kwargs):
            calls.append((source, kwargs))
            (model_path / "tokenizer" / "model.bin").write_text("weights-v2", encoding="utf-8")

        @staticmethod
        def get_sentence_embedding_dimension():
            return vector_store.EMBEDDING_DIMENSION

    module = SimpleNamespace(SentenceTransformer=MutatingSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(vector_store, "EMBEDDING_MODEL_PATH", str(model_path))
    store = VectorStore(MagicMock())

    with pytest.raises(RetrievalProfileError, match="changed while a model was loading"):
        store._ensure_embeddings()

    assert store._embed_fn is None
    assert calls == [
        (
            str(model_path),
            {
                "device": "cuda",
                "backend": "torch",
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )
    ]


def test_remote_embedding_and_reranker_loads_are_pinned(monkeypatch):
    embedding_calls: list[tuple[str, dict]] = []
    reranker_calls: list[tuple[str, dict]] = []

    class FakeSentenceTransformer:
        def __init__(self, source, **kwargs):
            embedding_calls.append((source, kwargs))

        @staticmethod
        def get_sentence_embedding_dimension():
            return vector_store.EMBEDDING_DIMENSION

    class FakeCrossEncoder:
        def __init__(self, source, **kwargs):
            reranker_calls.append((source, kwargs))

    module = SimpleNamespace(
        SentenceTransformer=FakeSentenceTransformer,
        CrossEncoder=FakeCrossEncoder,
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(vector_store, "RERANKER_ENABLED", True)
    store = VectorStore(MagicMock(), embedding_model="reviewed/embedding")

    store._ensure_embeddings()
    store._ensure_reranker()

    common = {
        "device": "cuda",
        "backend": "torch",
        "local_files_only": False,
        "trust_remote_code": False,
    }
    assert embedding_calls == [
        ("reviewed/embedding", {**common, "revision": "a" * 40}),
    ]
    assert reranker_calls == [
        ("reviewed/reranker", {**common, "revision": "b" * 40}),
    ]
    assert store._verified_retrieval_profile_descriptor is not None


@pytest.mark.asyncio
async def test_embedding_execution_matches_profile_options():
    calls: list[tuple[list[str], dict]] = []

    class FakeEmbeddings(list):
        def tolist(self):
            return list(self)

    class FakeModel:
        @staticmethod
        def encode(texts, **kwargs):
            calls.append((texts, kwargs))
            return FakeEmbeddings([[0.1, 0.2]])

    store = VectorStore(MagicMock())
    store._embed_fn = FakeModel()

    assert await store._embed(["sermaye"], prefix="query") == [[0.1, 0.2]]
    assert calls == [
        (
            ["query: sermaye"],
            {
                "batch_size": 32,
                "show_progress_bar": False,
                "output_value": "sentence_embedding",
                "precision": "float32",
                "convert_to_numpy": True,
                "convert_to_tensor": False,
                "normalize_embeddings": True,
            },
        )
    ]
    with pytest.raises(RetrievalProfileError, match="prefix"):
        await store._embed(["sermaye"], prefix="unbound")


@pytest.mark.asyncio
async def test_reranker_execution_matches_profile_options():
    calls: list[tuple[list[tuple[str, str]], dict]] = []

    class FakeReranker:
        @staticmethod
        def predict(pairs, **kwargs):
            calls.append((pairs, kwargs))
            return [0.0]

    store = VectorStore(MagicMock())
    store._rerank_fn = FakeReranker()
    candidates = [{"snippet": "hüküm", "doc_id": "1"}]

    result = await store._rerank("sermaye", candidates)

    assert result[0]["relevance"] == 0.5
    assert calls == [
        (
            [("sermaye", "hüküm")],
            {
                "batch_size": 32,
                "show_progress_bar": False,
                "activation_fn": None,
                "apply_softmax": False,
                "convert_to_numpy": True,
                "convert_to_tensor": False,
            },
        )
    ]


def test_reranker_loader_detects_local_asset_mutation(tmp_path, monkeypatch):
    model_path = tmp_path / "reranker-model"
    _write_model_tree(model_path)

    class MutatingCrossEncoder:
        def __init__(self, _source, **_kwargs):
            (model_path / "tokenizer" / "model.bin").write_text("weights-v2", encoding="utf-8")

    module = SimpleNamespace(CrossEncoder=MutatingCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(vector_store, "RERANKER_ENABLED", True)
    monkeypatch.setattr(vector_store, "RERANKER_MODEL_PATH", str(model_path))
    store = VectorStore(MagicMock())

    with pytest.raises(RetrievalProfileError, match="changed while a model was loading"):
        store._ensure_reranker()

    assert store._rerank_fn is None


def test_remote_revision_must_be_an_immutable_lowercase_sha(monkeypatch):
    monkeypatch.setattr(vector_store, "EMBEDDING_MODEL_REVISION", "main")

    with pytest.raises(RetrievalProfileError, match="40-character lowercase commit SHA"):
        retrieval_profile_descriptor()
