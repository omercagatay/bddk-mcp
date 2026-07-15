"""
pgvector-based vector store for BDDK regulatory documents.

Provides semantic search across all BDDK decisions, regulations, and
guidelines using PostgreSQL + pgvector extension.

Architecture:
  - Table "document_chunks": chunks with vector embeddings + tsvector FTS
  - Embedding model: multilingual-e5-base (best for Turkish legal text)
  - Hybrid search: dense (cosine) + sparse (BM25/tsvector) via RRF fusion
  - Optional cross-encoder re-ranking for precision
  - HNSW index for fast approximate nearest neighbor search
  - Offline-first: supports pre-downloaded model via BDDK_EMBEDDING_MODEL_PATH
"""

import asyncio
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import re
import stat
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import asyncpg

from bddk_mcp.core.config import (
    EMBEDDING_CHUNK_MODE,
    EMBEDDING_CHUNK_OVERLAP,
    EMBEDDING_CHUNK_SIZE,
    EMBEDDING_CHUNK_TARGET_TOKENS,
    EMBEDDING_CHUNK_TOKEN_OVERLAP,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_MODEL_REVISION,
    FTS_RANK_THRESHOLD,
    HYBRID_RRF_K,
    HYBRID_SEARCH,
    PAGE_SIZE,
    RERANKER_ENABLED,
    RERANKER_MODEL_NAME,
    RERANKER_MODEL_PATH,
    RERANKER_MODEL_REVISION,
    RERANKER_TOP_N,
    SEMANTIC_RELEVANCE_THRESHOLD,
)
from bddk_mcp.corpus_coordination import acquire_corpus_mutation_lock
from bddk_mcp.quality.markdown_quality import assess_markdown_quality, quality_retrieval_profile_descriptor
from bddk_mcp.store.bulk_write import (
    insert_document_chunk_rows,
    upsert_document_retrieval_publication_rows,
)
from bddk_mcp.store.doc_store import DOCUMENT_STORE_SEARCH_PROFILE_VERSION
from bddk_mcp.store.legal_ref import parse_legal_refs
from bddk_mcp.store.section_index import (
    MAX_SECTION_CHARS,
    SECTION_PARSER_PROFILE_VERSION,
    SECTION_SEARCH_PROFILE_VERSION,
    DocumentSection,
    extract_document_sections,
)

logger = logging.getLogger(__name__)


class VectorIndexConsistencyError(RuntimeError):
    """The vector index cannot be published for a missing or different source."""


class RetrievalProfileError(RuntimeError):
    """The persisted retrieval profile cannot be identified reproducibly."""


RETRIEVAL_PROFILE_SCHEMA_VERSION = 2
PERSISTED_VECTOR_PIPELINE_VERSION = "sentence-transformers-e5-v2"
CHUNKER_PROFILE_VERSION = "section-aware-token-budget-v2"
TOKEN_COUNTER_PROFILE_VERSION = "hf-encode-no-special-tokens-v1"
RETRIEVAL_SCORING_PROFILE_VERSION = "hybrid-pgvector-fts-scoring-v3"
FTS_PROFILE_VERSION = "postgres-simple-unaccent-tsrankcd-v2"
LEGAL_REFERENCE_MATCH_PROFILE_VERSION = "turkish-legal-reference-bypass-v1"
PHRASE_MATCH_PROFILE_VERSION = "turkish-normalized-phrase-boost-v1"
PUBLICATION_EMBEDDING_VERIFICATION_VERSION = "regenerate-all-vectors-tolerance-v1"
PUBLICATION_EMBEDDING_MAX_ABS_ERROR = 0.001
PUBLICATION_EMBEDDING_MIN_COSINE_SIMILARITY = 0.99999
_PASSAGE_PREFIX = "passage"
_QUERY_PREFIX = "query"
_PREFIX_SEPARATOR = ": "
_EMBEDDING_ENCODE_BATCH_SIZE = 32
_RERANKER_PREDICT_BATCH_SIZE = 32
_HYBRID_VECTOR_CANDIDATE_LIMIT = 50
_HYBRID_VECTOR_FETCH_LIMIT = 100
_HYBRID_FTS_CANDIDATE_LIMIT = 50
_SEARCH_SNIPPET_CHARS = 800
_SCORE_DECIMAL_PLACES = 4
_RRF_SCORE_DECIMAL_PLACES = 6
_FTS_GATE_PENALTY = 0.65
_SCORE_GAP_THRESHOLD = 0.051
_HIGH_CONFIDENCE_THRESHOLD = 0.70
_MEDIUM_CONFIDENCE_THRESHOLD = 0.50
_REMOTE_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_REMOTE_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}/[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RETRIEVAL_PACKAGE_DISTRIBUTIONS = (
    "PyYAML",
    "asyncpg",
    "huggingface-hub",
    "numpy",
    "pydantic",
    "safetensors",
    "scikit-learn",
    "scipy",
    "sentence-transformers",
    "tokenizers",
    "torch",
    "transformers",
)
_LOCAL_MODEL_TREE_FORMAT = "bddk-local-model-tree-sha256-v1"
_LOCAL_MODEL_MAX_FILES = 100_000
_LOCAL_MODEL_MAX_BYTES = 64 * 1024 * 1024 * 1024
_LOCAL_MODEL_HASH_CHUNK_BYTES = 4 * 1024 * 1024
_LOCAL_MODEL_IDENTITY_CACHE_LIMIT = 8


@dataclass(frozen=True)
class _LocalAssetFile:
    relative_path: str
    absolute_path: Path
    size: int
    stat_identity: tuple[int, int, int, int, int]


_LOCAL_MODEL_IDENTITY_CACHE: dict[
    str,
    tuple[tuple[tuple[str, int, int, int, int, int], ...], dict[str, int | str]],
] = {}


def _installed_retrieval_package_versions() -> dict[str, str]:
    """Return exact versions of packages that can affect chunks or vectors."""

    versions: dict[str, str] = {}
    for distribution in _RETRIEVAL_PACKAGE_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            raise RetrievalProfileError(
                "A required retrieval package is not installed; the persisted profile cannot be identified."
            ) from None
    return versions


def _absolute_path_without_symlink_resolution(configured_path: str) -> Path:
    if not isinstance(configured_path, str) or not configured_path.strip():
        raise RetrievalProfileError("The local embedding model path is empty or invalid.")
    try:
        return Path(os.path.abspath(os.path.expanduser(configured_path)))
    except (OSError, ValueError, RuntimeError):
        raise RetrievalProfileError("The local embedding model path is empty or invalid.") from None


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    try:
        for part in path.parts[1:]:
            current /= part
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise RetrievalProfileError("Local model paths and their parents must not be symlinks.")
    except RetrievalProfileError:
        raise
    except (OSError, ValueError):
        raise RetrievalProfileError("The local model asset tree is unavailable.") from None


def _scan_local_model_tree(
    configured_path: str,
) -> tuple[
    Path,
    tuple[_LocalAssetFile, ...],
    tuple[str, ...],
    tuple[tuple[str, int, int, int, int, int], ...],
    int,
]:
    """Inventory a local model without following links or accepting special files."""

    root = _absolute_path_without_symlink_resolution(configured_path)
    _reject_symlink_components(root)
    try:
        root_stat = os.lstat(root)
    except (OSError, ValueError):
        raise RetrievalProfileError("The local model asset tree is unavailable.") from None
    if not stat.S_ISDIR(root_stat.st_mode):
        raise RetrievalProfileError("The local model asset must be a directory.")

    files: list[_LocalAssetFile] = []
    directories: list[tuple[str, os.stat_result]] = []
    total_bytes = 0
    pending: list[tuple[Path, tuple[str, ...]]] = [(root, ())]
    try:
        while pending:
            directory, relative_parts = pending.pop()
            directory_before = os.lstat(directory)
            if not stat.S_ISDIR(directory_before.st_mode):
                raise RetrievalProfileError("Local model assets must contain real directories only.")
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda entry: entry.name)
            child_directories: list[tuple[Path, tuple[str, ...]]] = []
            for entry in entries:
                entry_stat = entry.stat(follow_symlinks=False)
                child_parts = (*relative_parts, entry.name)
                if stat.S_ISLNK(entry_stat.st_mode):
                    raise RetrievalProfileError("Local model assets must not contain symlinks.")
                if stat.S_ISDIR(entry_stat.st_mode):
                    directories.append(("/".join(child_parts), entry_stat))
                    child_directories.append((Path(entry.path), child_parts))
                    continue
                if not stat.S_ISREG(entry_stat.st_mode):
                    raise RetrievalProfileError("Local model assets must contain regular files only.")
                total_bytes += entry_stat.st_size
                if len(files) >= _LOCAL_MODEL_MAX_FILES or total_bytes > _LOCAL_MODEL_MAX_BYTES:
                    raise RetrievalProfileError("The local model asset tree exceeds the verification limit.")
                files.append(
                    _LocalAssetFile(
                        relative_path="/".join(child_parts),
                        absolute_path=Path(entry.path),
                        size=entry_stat.st_size,
                        stat_identity=(
                            entry_stat.st_dev,
                            entry_stat.st_ino,
                            entry_stat.st_size,
                            entry_stat.st_mtime_ns,
                            entry_stat.st_ctime_ns,
                        ),
                    )
                )
            directory_after = os.lstat(directory)
            before_identity = (
                directory_before.st_dev,
                directory_before.st_ino,
                directory_before.st_mtime_ns,
                directory_before.st_ctime_ns,
            )
            after_identity = (
                directory_after.st_dev,
                directory_after.st_ino,
                directory_after.st_mtime_ns,
                directory_after.st_ctime_ns,
            )
            if before_identity != after_identity or not stat.S_ISDIR(directory_after.st_mode):
                raise RetrievalProfileError("The local model asset tree changed during verification.")
            pending.extend(reversed(child_directories))
    except RetrievalProfileError:
        raise
    except (OSError, ValueError):
        raise RetrievalProfileError("The local model asset tree could not be inspected safely.") from None

    files.sort(key=lambda item: item.relative_path)
    directories.sort(key=lambda item: item[0])
    if not files:
        raise RetrievalProfileError("The local model asset tree is empty.")
    inventory = tuple(
        sorted(
            [
                (
                    f"file:{item.relative_path}",
                    item.size,
                    item.stat_identity[0],
                    item.stat_identity[1],
                    item.stat_identity[3],
                    item.stat_identity[4],
                )
                for item in files
            ]
            + [
                (
                    f"directory:{relative_path}",
                    0,
                    directory_stat.st_dev,
                    directory_stat.st_ino,
                    directory_stat.st_mtime_ns,
                    directory_stat.st_ctime_ns,
                )
                for relative_path, directory_stat in directories
            ]
        )
    )
    return root, tuple(files), tuple(item[0] for item in directories), inventory, total_bytes


def _hash_local_asset_file(item: _LocalAssetFile) -> str:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(item.absolute_path, flags)
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            before = os.fstat(stream.fileno())
            before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            if not stat.S_ISREG(before.st_mode) or before_identity != item.stat_identity:
                raise RetrievalProfileError("A local model asset changed during verification.")
            digest = hashlib.sha256()
            observed_size = 0
            while chunk := stream.read(_LOCAL_MODEL_HASH_CHUNK_BYTES):
                observed_size += len(chunk)
                digest.update(chunk)
            after = os.fstat(stream.fileno())
        path_after = os.lstat(item.absolute_path)
    except RetrievalProfileError:
        raise
    except (OSError, ValueError):
        raise RetrievalProfileError("A local model asset could not be verified.") from None

    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    path_identity = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if (
        observed_size != item.size
        or after_identity != item.stat_identity
        or path_identity != item.stat_identity
        or not stat.S_ISREG(path_after.st_mode)
    ):
        raise RetrievalProfileError("A local model asset changed during verification.")
    return digest.hexdigest()


def _update_digest_frame(digest, value: str) -> None:
    try:
        encoded = value.encode("utf-8")
    except UnicodeError:
        raise RetrievalProfileError("Local model asset names must be valid UTF-8 text.") from None
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _verified_local_model_identity(configured_path: str) -> dict[str, int | str]:
    """Hash the exact regular-file tree; paths never enter the returned identity."""

    root, files, directories, inventory, total_bytes = _scan_local_model_tree(configured_path)
    cache_key = str(root)
    cached = _LOCAL_MODEL_IDENTITY_CACHE.get(cache_key)
    if cached is not None and cached[0] == inventory:
        return dict(cached[1])

    tree_digest = hashlib.sha256()
    _update_digest_frame(tree_digest, _LOCAL_MODEL_TREE_FORMAT)
    for relative_path in directories:
        _update_digest_frame(tree_digest, "directory")
        _update_digest_frame(tree_digest, relative_path)
    for item in files:
        _update_digest_frame(tree_digest, "file")
        _update_digest_frame(tree_digest, item.relative_path)
        _update_digest_frame(tree_digest, str(item.size))
        _update_digest_frame(tree_digest, _hash_local_asset_file(item))

    _, _, _, inventory_after, total_bytes_after = _scan_local_model_tree(configured_path)
    if inventory_after != inventory or total_bytes_after != total_bytes:
        raise RetrievalProfileError("The local model asset tree changed during verification.")

    identity: dict[str, int | str] = {
        "format": _LOCAL_MODEL_TREE_FORMAT,
        "sha256": tree_digest.hexdigest(),
        "file_count": len(files),
        "directory_count": len(directories),
        "total_bytes": total_bytes,
    }
    if len(_LOCAL_MODEL_IDENTITY_CACHE) >= _LOCAL_MODEL_IDENTITY_CACHE_LIMIT:
        _LOCAL_MODEL_IDENTITY_CACHE.pop(next(iter(_LOCAL_MODEL_IDENTITY_CACHE)))
    _LOCAL_MODEL_IDENTITY_CACHE[cache_key] = (inventory, dict(identity))
    return identity


def _clear_local_model_identity_cache() -> None:
    """Test seam for deterministic local-asset mutation checks."""

    _LOCAL_MODEL_IDENTITY_CACHE.clear()


def _model_profile_identity(*, name: str, path: str, revision: str, role: str) -> dict:
    """Identify one model by immutable remote revision or verified local bytes."""

    if path:
        return {
            "source": "verified_local_tree",
            "asset": _verified_local_model_identity(path),
        }
    if not isinstance(name, str) or _REMOTE_MODEL_ID_RE.fullmatch(name) is None:
        raise RetrievalProfileError(f"The remote {role} model must be an exact repository identifier.")
    if not isinstance(revision, str) or _REMOTE_REVISION_RE.fullmatch(revision) is None:
        raise RetrievalProfileError(
            f"The remote {role} model revision must be an immutable 40-character lowercase commit SHA."
        )
    return {
        "source": "remote_repository_commit",
        "repository": name,
        "revision": revision,
    }


def retrieval_profile_descriptor(embedding_model: str | None = None) -> dict:
    """Describe the end-to-end retrieval behavior bound to corpus publication."""

    selected_model = embedding_model if embedding_model is not None else EMBEDDING_MODEL_NAME
    if not isinstance(selected_model, str) or not selected_model.strip():
        raise RetrievalProfileError("The embedding model identifier is empty or invalid.")
    if EMBEDDING_CHUNK_MODE not in {"character", "token"}:
        raise RetrievalProfileError("BDDK_EMBEDDING_CHUNK_MODE must be 'character' or 'token'.")
    if EMBEDDING_CHUNK_SIZE < 1 or not 0 <= EMBEDDING_CHUNK_OVERLAP < EMBEDDING_CHUNK_SIZE:
        raise RetrievalProfileError("Character chunk settings are invalid.")
    if EMBEDDING_CHUNK_TARGET_TOKENS < 1 or not 0 <= EMBEDDING_CHUNK_TOKEN_OVERLAP < EMBEDDING_CHUNK_TARGET_TOKENS:
        raise RetrievalProfileError("Token chunk settings are invalid.")
    if EMBEDDING_DIMENSION < 1:
        raise RetrievalProfileError("The embedding dimension must be positive.")
    if HYBRID_RRF_K < 1:
        raise RetrievalProfileError("BDDK_RRF_K must be positive.")
    if not -1.0 <= SEMANTIC_RELEVANCE_THRESHOLD <= 1.0:
        raise RetrievalProfileError("BDDK_SEMANTIC_THRESHOLD must be between -1 and 1.")
    if RERANKER_TOP_N < 1:
        raise RetrievalProfileError("BDDK_RERANKER_TOP_N must be positive.")
    if PAGE_SIZE < 1:
        raise RetrievalProfileError("BDDK_PAGE_SIZE must be positive.")
    if FTS_RANK_THRESHOLD < 0:
        raise RetrievalProfileError("BDDK_FTS_THRESHOLD must not be negative.")

    embedding_identity = _model_profile_identity(
        name=selected_model,
        path=EMBEDDING_MODEL_PATH,
        revision=EMBEDDING_MODEL_REVISION,
        role="embedding",
    )
    reranker_identity = _model_profile_identity(
        name=RERANKER_MODEL_NAME,
        path=RERANKER_MODEL_PATH,
        revision=RERANKER_MODEL_REVISION,
        role="reranker",
    )

    return {
        "schema_version": RETRIEVAL_PROFILE_SCHEMA_VERSION,
        "scope": "published_end_to_end_document_retrieval",
        "pipeline_version": PERSISTED_VECTOR_PIPELINE_VERSION,
        "embedding": {
            "model": embedding_identity,
            "dimension": EMBEDDING_DIMENSION,
            "passage_prefix": f"{_PASSAGE_PREFIX}{_PREFIX_SEPARATOR}",
            "query_prefix": f"{_QUERY_PREFIX}{_PREFIX_SEPARATOR}",
            "backend": "torch",
            "device_selection_policy": "attempt_cuda_then_cpu_for_supported_load_failures",
            "local_files_only": bool(EMBEDDING_MODEL_PATH),
            "batch_size": _EMBEDDING_ENCODE_BATCH_SIZE,
            "output_value": "sentence_embedding",
            "normalize_embeddings": True,
            "precision": "float32",
            "convert_to_numpy": True,
            "convert_to_tensor": False,
            "show_progress_bar": False,
            "trust_remote_code": False,
            "text_encoding": "utf-8",
        },
        "publication_verification": {
            "version": PUBLICATION_EMBEDDING_VERIFICATION_VERSION,
            "regenerate_every_chunk_embedding": True,
            "stored_dimension": EMBEDDING_DIMENSION,
            "require_finite_components": True,
            "require_nonzero_l2_norm": True,
            "maximum_absolute_component_error": PUBLICATION_EMBEDDING_MAX_ABS_ERROR,
            "minimum_cosine_similarity": PUBLICATION_EMBEDDING_MIN_COSINE_SIMILARITY,
            "hardware_calibration": "representative_cpu_gpu_acceptance_gate_required",
        },
        "chunking": {
            "version": CHUNKER_PROFILE_VERSION,
            "mode": EMBEDDING_CHUNK_MODE,
            "character_size": EMBEDDING_CHUNK_SIZE,
            "character_overlap": EMBEDDING_CHUNK_OVERLAP,
            "target_tokens": EMBEDDING_CHUNK_TARGET_TOKENS,
            "token_overlap": EMBEDDING_CHUNK_TOKEN_OVERLAP,
            "token_counter_version": TOKEN_COUNTER_PROFILE_VERSION,
            "tokenizer_model": embedding_identity,
            "tokenizer_use_fast": True,
            "tokenizer_add_special_tokens": False,
            "tokenizer_trust_remote_code": False,
            "tokenizer_local_files_only": bool(EMBEDDING_MODEL_PATH),
            "word_unit_pattern": _WORD_UNIT_RE.pattern,
        },
        "section_parser": {
            "version": SECTION_PARSER_PROFILE_VERSION,
            "max_section_chars": MAX_SECTION_CHARS,
            "content_hash": "sha256-utf8",
        },
        "pagination": {
            "page_size_chars": PAGE_SIZE,
            "total_pages": "max_one_ceiling_character_length_over_page_size",
            "page_slice": "zero_based_python_character_slice",
        },
        "retrieval": {
            "version": RETRIEVAL_SCORING_PROFILE_VERSION,
            "hybrid_enabled": HYBRID_SEARCH,
            "vector": {
                "distance": "pgvector_cosine",
                "candidate_limit": _HYBRID_VECTOR_CANDIDATE_LIMIT,
                "fetch_limit": _HYBRID_VECTOR_FETCH_LIMIT,
                "standalone_fetch_limit": "min_requested_limit_times_five_or_100",
                "deduplication": "best_distance_per_document",
                "score": "one_minus_cosine_distance",
            },
            "fts": {
                "version": FTS_PROFILE_VERSION,
                "stored_vector": "simple_unaccent_title_plus_chunk_text",
                "query_config": "simple",
                "primary_query_function": "plainto_tsquery",
                "fallback_query_function": "to_tsquery_or_prefix",
                "rank_function": "ts_rank_cd",
                "candidate_limit": _HYBRID_FTS_CANDIDATE_LIMIT,
                "relaxed_token_pattern": _FTS_TOKEN_RE.pattern,
                "relaxed_token_pattern_flags": _FTS_TOKEN_RE.flags,
                "relaxed_stopwords": sorted(_FTS_RELAXED_STOPWORDS),
                "relaxed_min_terms": _FTS_RELAXED_MIN_TERMS,
                "relaxed_max_terms": _FTS_RELAXED_MAX_TERMS,
                "deduplication": "best_rank_per_document",
            },
            "document_store": {
                "version": DOCUMENT_STORE_SEARCH_PROFILE_VERSION,
                "query_config": "simple",
                "query_function": "plainto_tsquery_unaccent",
                "rank_function": "ts_rank_cd_normalization_0",
                "rank_threshold": FTS_RANK_THRESHOLD,
                "term_sanitization": "bounded_operator_character_removal_and_boolean_stopwords",
                "headline": {
                    "start_marker": ">>>",
                    "stop_marker": "<<<",
                    "max_words": 40,
                    "min_words": 20,
                },
            },
            "section_search": {
                "version": SECTION_SEARCH_PROFILE_VERSION,
                "query_config": "simple",
                "query_function": "plainto_tsquery_unaccent",
                "rank_function": "ts_rank_cd",
                "rank_normalization": 1,
                "tie_break": "section_start_char_ascending",
            },
            "fusion": {
                "algorithm": "reciprocal_rank_fusion",
                "rrf_k": HYBRID_RRF_K,
                "semantic_signal_weight": 1.0,
                "keyword_signal_weight": 1.0,
                "rrf_score_decimal_places": _RRF_SCORE_DECIMAL_PLACES,
            },
            "scoring": {
                "semantic_relevance_threshold": SEMANTIC_RELEVANCE_THRESHOLD,
                "threshold_basis": "pre_rerank_semantic_relevance",
                "fts_exact_legal_reference_bypass": True,
                "legal_reference_match_version": LEGAL_REFERENCE_MATCH_PROFILE_VERSION,
                "fts_empty_semantic_penalty": _FTS_GATE_PENALTY,
                "normalized_keyword_rank_boost": _LEXICAL_RELEVANCE_BOOST,
                "phrase_match_version": PHRASE_MATCH_PROFILE_VERSION,
                "phrase_match_boost": _PHRASE_RELEVANCE_BOOST,
                "phrase_match_max_boost": _PHRASE_RELEVANCE_MAX,
                "phrase_stem_suffixes": list(_TURKISH_STEM_SUFFIXES),
                "score_gap_threshold": _SCORE_GAP_THRESHOLD,
                "score_decimal_places": _SCORE_DECIMAL_PLACES,
                "high_confidence_threshold": _HIGH_CONFIDENCE_THRESHOLD,
                "medium_confidence_threshold": _MEDIUM_CONFIDENCE_THRESHOLD,
            },
            "snippet_chars": _SEARCH_SNIPPET_CHARS,
        },
        "reranker": {
            "enabled": RERANKER_ENABLED,
            "model": reranker_identity,
            "top_n": RERANKER_TOP_N,
            "backend": "torch",
            "device_selection_policy": "attempt_cuda_then_cpu_for_supported_load_failures",
            "batch_size": _RERANKER_PREDICT_BATCH_SIZE,
            "local_files_only": bool(RERANKER_MODEL_PATH),
            "show_progress_bar": False,
            "activation": "model_default",
            "apply_softmax": False,
            "convert_to_numpy": True,
            "convert_to_tensor": False,
            "displayed_score_transform": "sigmoid",
            "trust_remote_code": False,
        },
        "quality_signals": quality_retrieval_profile_descriptor(),
        "runtime": {
            "python_implementation": sys.implementation.name,
            "python_version": ".".join(str(value) for value in sys.version_info[:3]),
            "python_cache_tag": sys.implementation.cache_tag,
            "unicode_version": unicodedata.unidata_version,
            "packages": _installed_retrieval_package_versions(),
        },
    }


def _retrieval_profile_hash_from_descriptor(descriptor: dict) -> str:
    encoded = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def retrieval_profile_hash(embedding_model: str | None = None) -> str:
    """Fingerprint the deterministic persisted retrieval-profile descriptor."""

    return _retrieval_profile_hash_from_descriptor(retrieval_profile_descriptor(embedding_model))


_FTS_TOKEN_RE = re.compile(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü]{3,}")
_FTS_RELAXED_STOPWORDS = {
    "acaba",
    "ancak",
    "bir",
    "buna",
    "göre",
    "gore",
    "hangi",
    "için",
    "icin",
    "ile",
    "kaç",
    "kac",
    "kadar",
    "nasıl",
    "nasil",
    "nedir",
    "nelerdir",
    "olan",
    "olarak",
    "olabilir",
    "olmalıdır",
    "olmalidir",
    "veya",
}
_FTS_RELAXED_MIN_TERMS = 3
_FTS_RELAXED_MAX_TERMS = 12
_LEXICAL_RELEVANCE_BOOST = 0.045
_PHRASE_RELEVANCE_BOOST = 0.03
_PHRASE_RELEVANCE_MAX = 0.06
_TURKISH_STEM_SUFFIXES = ("leri", "lari", "inin", "unun", "nin", "nun", "ler", "lar", "si", "su", "i", "u")


@dataclass(frozen=True)
class DocumentChunk:
    """A text chunk plus optional legal section metadata."""

    chunk_text: str
    start_char: int
    end_char: int
    section_type: str = ""
    section_ref: str = ""
    section_start_char: int | None = None
    section_end_char: int | None = None
    section_content_hash: str = ""


@dataclass(frozen=True)
class _ChunkSpan:
    start_char: int
    end_char: int
    section: DocumentSection | None = None


@dataclass(frozen=True)
class _TextUnit:
    start_char: int
    end_char: int
    text: str
    token_count: int


_WORD_UNIT_RE = re.compile(r"\S+\s*", re.MULTILINE)


def _chunk_text(text: str, chunk_size: int = EMBEDDING_CHUNK_SIZE, overlap: int = EMBEDDING_CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += step

    return chunks


def _chunk_document(
    doc_id: str,
    text: str,
    chunk_size: int = EMBEDDING_CHUNK_SIZE,
    overlap: int = EMBEDDING_CHUNK_OVERLAP,
    tokenizer=None,
    target_tokens: int = EMBEDDING_CHUNK_TARGET_TOKENS,
    token_overlap: int = EMBEDDING_CHUNK_TOKEN_OVERLAP,
    use_token_chunking: bool | None = None,
) -> list[DocumentChunk]:
    """Split text into chunks and attach best-effort legal section metadata."""
    if not text:
        return []
    sections = extract_document_sections(doc_id, text)
    if use_token_chunking is None:
        use_token_chunking = EMBEDDING_CHUNK_MODE == "token"
    if use_token_chunking:
        if tokenizer is None:
            raise RetrievalProfileError(
                "Token chunking is configured but the profile-bound embedding tokenizer is unavailable."
            )
        return _chunk_document_by_tokens(doc_id, text, sections, tokenizer, target_tokens, token_overlap)
    return _chunk_document_by_chars(doc_id, text, sections, chunk_size, overlap)


def _chunk_document_by_chars(
    doc_id: str,
    text: str,
    sections: list[DocumentSection],
    chunk_size: int = EMBEDDING_CHUNK_SIZE,
    overlap: int = EMBEDDING_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        if chunk.strip():
            section = _section_for_chunk(start, end, sections)
            chunks.append(
                DocumentChunk(
                    chunk_text=chunk,
                    start_char=start,
                    end_char=end,
                    section_type=section.section_type if section else "",
                    section_ref=section.section_ref if section else "",
                    section_start_char=section.start_char if section else None,
                    section_end_char=section.end_char if section else None,
                    section_content_hash=section.content_hash if section else "",
                )
            )
        start += step
    return chunks


def _chunk_document_by_tokens(
    doc_id: str,
    text: str,
    sections: list[DocumentSection],
    tokenizer,
    target_tokens: int,
    token_overlap: int,
) -> list[DocumentChunk]:
    target_tokens = max(1, target_tokens)
    token_overlap = max(0, min(token_overlap, target_tokens - 1))
    chunks: list[DocumentChunk] = []
    for span in _chunk_spans(text, sections):
        for start_char, end_char in _token_budget_ranges(
            text=text,
            span=span,
            tokenizer=tokenizer,
            target_tokens=target_tokens,
            token_overlap=token_overlap,
        ):
            chunk_text = text[start_char:end_char]
            if not chunk_text.strip():
                continue
            section = span.section or _section_for_chunk(start_char, end_char, sections)
            chunks.append(
                DocumentChunk(
                    chunk_text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    section_type=section.section_type if section else "",
                    section_ref=section.section_ref if section else "",
                    section_start_char=section.start_char if section else None,
                    section_end_char=section.end_char if section else None,
                    section_content_hash=section.content_hash if section else "",
                )
            )
    return chunks


def _chunk_spans(text: str, sections: list[DocumentSection]) -> list[_ChunkSpan]:
    if not sections:
        return [_ChunkSpan(start_char=0, end_char=len(text))] if text.strip() else []

    spans: list[_ChunkSpan] = []
    sorted_sections = sorted(sections, key=lambda section: (section.start_char, section.end_char))
    cursor = 0
    for index, section in enumerate(sorted_sections):
        if cursor < section.start_char and text[cursor : section.start_char].strip():
            spans.append(_ChunkSpan(start_char=cursor, end_char=section.start_char))

        later_starts = [
            later.start_char
            for later in sorted_sections[index + 1 :]
            if section.start_char < later.start_char < section.end_char
        ]
        end_char = min(section.end_char, later_starts[0]) if later_starts else section.end_char
        if section.start_char < end_char and text[section.start_char : end_char].strip():
            spans.append(_ChunkSpan(start_char=section.start_char, end_char=end_char, section=section))
        cursor = max(cursor, end_char)

    if cursor < len(text) and text[cursor:].strip():
        spans.append(_ChunkSpan(start_char=cursor, end_char=len(text)))
    return spans


def _token_budget_ranges(
    text: str,
    span: _ChunkSpan,
    tokenizer,
    target_tokens: int,
    token_overlap: int,
) -> list[tuple[int, int]]:
    units = _text_units(text[span.start_char : span.end_char], span.start_char, tokenizer, target_tokens)
    ranges: list[tuple[int, int]] = []
    current: list[_TextUnit] = []
    current_tokens = 0

    for unit in units:
        if current and current_tokens + unit.token_count > target_tokens:
            ranges.append((current[0].start_char, current[-1].end_char))
            current = _overlap_units(current, token_overlap)
            current_tokens = sum(item.token_count for item in current)
            if current and current_tokens + unit.token_count > target_tokens:
                current = []
                current_tokens = 0

        current.append(unit)
        current_tokens += unit.token_count

    if current:
        ranges.append((current[0].start_char, current[-1].end_char))
    return ranges


def _text_units(text: str, absolute_start: int, tokenizer, target_tokens: int) -> list[_TextUnit]:
    units: list[_TextUnit] = []
    for match in _WORD_UNIT_RE.finditer(text):
        unit_text = match.group(0)
        start_char = absolute_start + match.start()
        token_count = max(1, _count_tokens(unit_text, tokenizer))
        if token_count <= target_tokens:
            units.append(
                _TextUnit(
                    start_char=start_char,
                    end_char=absolute_start + match.end(),
                    text=unit_text,
                    token_count=token_count,
                )
            )
        else:
            units.extend(_split_oversized_unit(unit_text, start_char, tokenizer, target_tokens))
    return units


def _split_oversized_unit(text: str, absolute_start: int, tokenizer, target_tokens: int) -> list[_TextUnit]:
    units: list[_TextUnit] = []
    cursor = 0
    while cursor < len(text):
        lo = 1
        hi = len(text) - cursor
        best = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = text[cursor : cursor + mid]
            if _count_tokens(candidate, tokenizer) <= target_tokens or mid == 1:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        chunk_text = text[cursor : cursor + best]
        units.append(
            _TextUnit(
                start_char=absolute_start + cursor,
                end_char=absolute_start + cursor + best,
                text=chunk_text,
                token_count=max(1, _count_tokens(chunk_text, tokenizer)),
            )
        )
        cursor += best
    return units


def _overlap_units(units: list[_TextUnit], token_overlap: int) -> list[_TextUnit]:
    if token_overlap <= 0:
        return []
    kept: list[_TextUnit] = []
    total = 0
    for unit in reversed(units):
        if total + unit.token_count > token_overlap:
            break
        kept.append(unit)
        total += unit.token_count
    return list(reversed(kept))


def _count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _load_embedding_tokenizer(embedding_model: str | None = None):
    if EMBEDDING_CHUNK_MODE != "token":
        return None
    selected_model = embedding_model if embedding_model is not None else EMBEDDING_MODEL_NAME
    model_ref = EMBEDDING_MODEL_PATH if EMBEDDING_MODEL_PATH else selected_model
    model_source = "local" if EMBEDDING_MODEL_PATH else "remote"
    profile_before = retrieval_profile_descriptor(selected_model)
    model_kwargs = {
        "use_fast": True,
        "trust_remote_code": False,
        "local_files_only": bool(EMBEDDING_MODEL_PATH),
    }
    if not EMBEDDING_MODEL_PATH:
        model_kwargs["revision"] = EMBEDDING_MODEL_REVISION
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_ref, **model_kwargs)
    except Exception as exc:
        logger.error(
            "Profile-bound embedding tokenizer load failed (%s source)",
            model_source,
            extra={"error_type": type(exc).__name__},
        )
        raise RetrievalProfileError("The profile-bound embedding tokenizer could not be loaded.") from None
    if retrieval_profile_descriptor(selected_model) != profile_before:
        raise RetrievalProfileError("The embedding model assets changed while the tokenizer was loading.")
    return tokenizer


def _section_for_chunk(start_char: int, end_char: int, sections: list[DocumentSection]) -> DocumentSection | None:
    for section in sections:
        if section.start_char <= start_char < section.end_char:
            return section
    overlapping = [
        section for section in sections if max(start_char, section.start_char) < min(end_char, section.end_char)
    ]
    if not overlapping:
        return None
    return max(
        overlapping,
        key=lambda section: min(end_char, section.end_char) - max(start_char, section.start_char),
    )


def _has_exact_legal_reference(query: str) -> bool:
    refs = parse_legal_refs(query)
    return bool(refs.sections or refs.decision_numbers or refs.dates)


def _relaxed_fts_query(query: str) -> str | None:
    terms: list[str] = []
    seen: set[str] = set()
    for match in _FTS_TOKEN_RE.finditer(query):
        term = match.group(0).lower()
        if term in _FTS_RELAXED_STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= _FTS_RELAXED_MAX_TERMS:
            break

    if len(terms) < _FTS_RELAXED_MIN_TERMS:
        return None
    return " | ".join(f"{term}:*" for term in terms)


def _normalize_search_text(text: str) -> str:
    text = text.translate(str.maketrans({"ı": "i", "İ": "i"})).lower()
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _stem_search_term(term: str) -> str:
    for suffix in _TURKISH_STEM_SUFFIXES:
        if len(term) > len(suffix) + 3 and term.endswith(suffix):
            return term[: -len(suffix)]
    return term


def _phrase_match_count(query: str, text: str) -> int:
    normalized_query = _normalize_search_text(query)
    normalized_text_terms = [
        _stem_search_term(match.group(0)) for match in re.finditer(r"[0-9a-z]{3,}", _normalize_search_text(text))
    ]
    normalized_text = " ".join(normalized_text_terms)
    stopwords = {_normalize_search_text(term) for term in _FTS_RELAXED_STOPWORDS}
    terms = [
        _stem_search_term(match.group(0))
        for match in re.finditer(r"[0-9a-z]{3,}", normalized_query)
        if match.group(0) not in stopwords
    ]

    phrases: set[str] = set()
    for size in (3, 2):
        for index in range(0, len(terms) - size + 1):
            phrases.add(" ".join(terms[index : index + size]))

    return sum(1 for phrase in phrases if phrase in normalized_text)


def _quality_metadata(text: str, doc_id: str) -> dict:
    quality = assess_markdown_quality(text or "", document_id=doc_id)
    return {"quality_label": quality.label, "quality_flags": quality.flags}


def _section_metadata_from_row(row) -> dict:
    return {
        "section_type": row["section_type"] or "",
        "section_ref": row["section_ref"] or "",
        "section_start_char": row["section_start_char"],
        "section_end_char": row["section_end_char"],
        "section_content_hash": row["section_content_hash"] or "",
    }


def _row_get(row, key: str, default=None):
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


class VectorStore:
    """
    pgvector-backed vector store for BDDK documents.

    Supports three search modes:
      - Vector-only: cosine similarity via pgvector
      - Hybrid: vector + FTS combined via Reciprocal Rank Fusion (RRF)
      - Hybrid + re-ranking: cross-encoder re-scores top candidates

    Usage::

        store = VectorStore(pool)
        await store.add_document(doc_id="1291", title="...", content="...", metadata={...})
        results = await store.search("sermaye yeterliliği hesaplama", limit=10)
        doc = await store.get_document("1291")
    """

    def __init__(self, pool: asyncpg.Pool, embedding_model: str = EMBEDDING_MODEL_NAME) -> None:
        self._pool = pool
        self._embedding_model = embedding_model
        self._embed_fn = None
        self._chunk_tokenizer_fn = None
        self._rerank_fn = None
        self._verified_retrieval_profile_descriptor: dict | None = None

    @property
    def retrieval_profile_hash(self) -> str:
        """Return the exact profile accepted by this runtime."""

        descriptor = self._verified_retrieval_profile_descriptor
        if descriptor is None:
            descriptor = retrieval_profile_descriptor(self._embedding_model)
        return _retrieval_profile_hash_from_descriptor(descriptor)

    async def initialize(self) -> None:
        """Deprecated SELECT-only compatibility readiness check.

        Schema migrations and data backfills are explicit operator actions;
        this method never mutates database state.
        """
        from bddk_mcp.db_lifecycle import assert_database_ready

        await assert_database_ready(pool=self._pool, require_corpus=False)

    async def close(self) -> None:
        """No-op — pool lifecycle is managed externally."""
        logger.info("VectorStore closed")

    # -- Model loading -----------------------------------------------------------

    def _pin_verified_profile(self, descriptor_before: dict, descriptor_after: dict) -> None:
        if descriptor_after != descriptor_before:
            raise RetrievalProfileError("The retrieval profile changed while a model was loading.")
        if (
            self._verified_retrieval_profile_descriptor is not None
            and self._verified_retrieval_profile_descriptor != descriptor_after
        ):
            raise RetrievalProfileError("The retrieval profile changed after this store initialized its models.")
        self._verified_retrieval_profile_descriptor = descriptor_after

    def _ensure_embeddings(self) -> None:
        """Lazy-load the embedding model on first search/add."""
        if self._embed_fn is not None:
            return

        from sentence_transformers import SentenceTransformer

        profile_before = retrieval_profile_descriptor(self._embedding_model)
        if (
            self._verified_retrieval_profile_descriptor is not None
            and self._verified_retrieval_profile_descriptor != profile_before
        ):
            raise RetrievalProfileError("The retrieval profile changed after this store initialized its models.")
        model_ref = EMBEDDING_MODEL_PATH if EMBEDDING_MODEL_PATH else self._embedding_model
        model_kwargs = {
            "backend": "torch",
            "local_files_only": bool(EMBEDDING_MODEL_PATH),
            "trust_remote_code": False,
        }
        if EMBEDDING_MODEL_PATH:
            model_source = "local"
        else:
            model_kwargs["revision"] = EMBEDDING_MODEL_REVISION
            model_source = "remote"

        logger.info("Loading embedding model (%s source)", model_source)

        try:
            loaded_model = SentenceTransformer(model_ref, device="cuda", **model_kwargs)
            logger.info("Loaded embedding model (GPU execution, %s source)", model_source)
        except (RuntimeError, ValueError, AssertionError):
            # CPU-only torch raises AssertionError on CUDA probe, not RuntimeError.
            loaded_model = SentenceTransformer(model_ref, device="cpu", **model_kwargs)
            logger.info("Loaded embedding model (CPU execution, %s source)", model_source)
        dimension = loaded_model.get_sentence_embedding_dimension()
        if dimension != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Embedding model dimension must be {EMBEDDING_DIMENSION} for the current database schema."
            )
        profile_after = retrieval_profile_descriptor(self._embedding_model)
        self._pin_verified_profile(profile_before, profile_after)
        self._embed_fn = loaded_model

    def _chunk_tokenizer(self):
        if EMBEDDING_CHUNK_MODE != "token":
            return None
        if self._chunk_tokenizer_fn is not None:
            return self._chunk_tokenizer_fn
        # Keep model loading and dimension/profile verification on the same
        # path used before embedding, but use the explicitly profile-bound
        # fast AutoTokenizer for chunk boundaries. SentenceTransformer may
        # expose a slow or wrapper-specific tokenizer that does not implement
        # the descriptor's ``tokenizer_use_fast=True`` contract.
        self._ensure_embeddings()
        profile_before = retrieval_profile_descriptor(self._embedding_model)
        if (
            self._verified_retrieval_profile_descriptor is not None
            and self._verified_retrieval_profile_descriptor != profile_before
        ):
            raise RetrievalProfileError("The retrieval profile changed after this store initialized its models.")
        tokenizer = _load_embedding_tokenizer(self._embedding_model)
        profile_after = retrieval_profile_descriptor(self._embedding_model)
        self._pin_verified_profile(profile_before, profile_after)
        self._chunk_tokenizer_fn = tokenizer
        return tokenizer

    def _ensure_reranker(self) -> None:
        """Lazy-load the cross-encoder re-ranking model."""
        if self._rerank_fn is not None:
            return

        from sentence_transformers import CrossEncoder

        profile_before = retrieval_profile_descriptor(self._embedding_model)
        if (
            self._verified_retrieval_profile_descriptor is not None
            and self._verified_retrieval_profile_descriptor != profile_before
        ):
            raise RetrievalProfileError("The retrieval profile changed after this store initialized its models.")
        model_ref = RERANKER_MODEL_PATH if RERANKER_MODEL_PATH else RERANKER_MODEL_NAME
        model_kwargs = {
            "backend": "torch",
            "local_files_only": bool(RERANKER_MODEL_PATH),
            "trust_remote_code": False,
        }
        if RERANKER_MODEL_PATH:
            model_source = "local"
        else:
            model_source = "remote"
            model_kwargs["revision"] = RERANKER_MODEL_REVISION
        logger.info("Loading cross-encoder reranker (%s source)", model_source)

        try:
            loaded_model = CrossEncoder(model_ref, device="cuda", **model_kwargs)
            logger.info("Loaded cross-encoder reranker (GPU execution, %s source)", model_source)
        except (RuntimeError, ValueError, AssertionError):
            # CPU-only torch raises AssertionError on CUDA probe, not RuntimeError.
            loaded_model = CrossEncoder(model_ref, device="cpu", **model_kwargs)
            logger.info("Loaded cross-encoder reranker (CPU execution, %s source)", model_source)
        profile_after = retrieval_profile_descriptor(self._embedding_model)
        self._pin_verified_profile(profile_before, profile_after)
        self._rerank_fn = loaded_model

    async def _embed(self, texts: list[str], prefix: str = _PASSAGE_PREFIX) -> list[list[float]]:
        """Generate embeddings in a thread to avoid blocking the event loop."""
        if prefix not in {_PASSAGE_PREFIX, _QUERY_PREFIX}:
            raise RetrievalProfileError("The embedding prefix is not part of the retrieval profile.")
        self._ensure_embeddings()
        prefixed = [f"{prefix}{_PREFIX_SEPARATOR}{text}" for text in texts]
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._embed_fn.encode(
                prefixed,
                batch_size=_EMBEDDING_ENCODE_BATCH_SIZE,
                show_progress_bar=False,
                output_value="sentence_embedding",
                precision="float32",
                convert_to_numpy=True,
                convert_to_tensor=False,
                normalize_embeddings=True,
            ),
        )
        return embeddings.tolist()

    # -- Add documents --------------------------------------------------------

    async def add_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        category: str = "",
        decision_date: str = "",
        decision_number: str = "",
        source_url: str = "",
    ) -> int:
        """Add a document to the vector store. Returns number of chunks created."""
        if not content.strip():
            return 0

        chunks = _chunk_document(doc_id, content, tokenizer=self._chunk_tokenizer())
        if not chunks:
            return 0

        total_pages = max(1, math.ceil(len(content) / PAGE_SIZE))
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        # Generate embeddings
        embeddings = await self._embed([chunk.chunk_text for chunk in chunks])

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await acquire_corpus_mutation_lock(conn)
                stored_hash = await conn.fetchval(
                    "SELECT content_hash FROM public.documents WHERE document_id = $1 FOR SHARE",
                    doc_id,
                )
                if stored_hash != content_hash:
                    raise VectorIndexConsistencyError(
                        "Vector chunks were not published because the stored document hash does not match."
                    )
                # Delete old chunks
                await conn.execute("DELETE FROM public.document_chunks WHERE doc_id = $1", doc_id)

                # Bulk insert new chunks with embeddings (tsv auto-populated by trigger)
                args_list = []
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True)):
                    vec_str = "[" + ",".join(str(v) for v in emb) + "]"
                    args_list.append(
                        (
                            doc_id,
                            i,
                            title,
                            category,
                            decision_date,
                            decision_number,
                            source_url,
                            len(chunks),
                            total_pages,
                            content_hash,
                            chunk.start_char,
                            chunk.end_char,
                            chunk.section_type,
                            chunk.section_ref,
                            chunk.section_start_char,
                            chunk.section_end_char,
                            chunk.section_content_hash,
                            chunk.chunk_text,
                            vec_str,
                        )
                    )

                await insert_document_chunk_rows(conn, args_list)
                await self._publish_document_on_connection(
                    conn,
                    doc_id=doc_id,
                    content_hash=content_hash,
                    expected_chunks=len(chunks),
                )

        logger.debug("Added document to vector index (chunks=%d)", len(chunks))
        return len(chunks)

    async def _publish_document_on_connection(
        self,
        conn,
        *,
        doc_id: str,
        content_hash: str,
        expected_chunks: int,
    ) -> None:
        """Publish one complete, embedded chunk set inside its write transaction."""

        publication = await self._validate_document_publication_on_connection(
            conn,
            doc_id=doc_id,
            content_hash=content_hash,
            expected_chunks=expected_chunks,
        )
        await upsert_document_retrieval_publication_rows(conn, [publication])

    async def _publish_documents_on_connection(
        self,
        conn,
        documents: list[tuple[str, str, int]],
    ) -> None:
        """Validate many chunk sets, then publish their memberships set-wise."""

        publications = [
            await self._validate_document_publication_on_connection(
                conn,
                doc_id=doc_id,
                content_hash=content_hash,
                expected_chunks=expected_chunks,
            )
            for doc_id, content_hash, expected_chunks in documents
        ]
        await upsert_document_retrieval_publication_rows(conn, publications)

    async def _validate_document_publication_on_connection(
        self,
        conn,
        *,
        doc_id: str,
        content_hash: str,
        expected_chunks: int,
    ) -> tuple[str, str, str, int]:
        """Prove one chunk set is complete and return its publication row."""

        if not re.fullmatch(r"[0-9a-f]{64}", content_hash) or expected_chunks < 1:
            raise VectorIndexConsistencyError("Vector chunks failed publication integrity validation.")
        source = await conn.fetchrow(
            "SELECT markdown_content, content_hash FROM public.documents WHERE document_id = $1",
            doc_id,
        )
        if source is None or source["content_hash"] != content_hash:
            raise VectorIndexConsistencyError("Vector chunks failed publication integrity validation.")
        expected = _chunk_document(
            doc_id,
            source["markdown_content"] or "",
            tokenizer=self._chunk_tokenizer(),
        )
        actual = await conn.fetch(
            """
            SELECT chunk_index, chunk_text, chunk_start_char, chunk_end_char,
                   section_type, section_ref, section_start_char, section_end_char,
                   section_content_hash, content_hash, total_chunks
            FROM public.document_chunks
            WHERE doc_id = $1
            ORDER BY chunk_index
            """,
            doc_id,
        )
        if len(expected) != expected_chunks or len(actual) != expected_chunks:
            raise VectorIndexConsistencyError("Vector chunks failed publication integrity validation.")
        for index, (expected_chunk, actual_chunk) in enumerate(zip(expected, actual, strict=True)):
            expected_values = (
                index,
                expected_chunk.chunk_text,
                expected_chunk.start_char,
                expected_chunk.end_char,
                expected_chunk.section_type,
                expected_chunk.section_ref,
                expected_chunk.section_start_char,
                expected_chunk.section_end_char,
                expected_chunk.section_content_hash,
                content_hash,
                expected_chunks,
            )
            actual_values = (
                actual_chunk["chunk_index"],
                actual_chunk["chunk_text"],
                actual_chunk["chunk_start_char"],
                actual_chunk["chunk_end_char"],
                actual_chunk["section_type"] or "",
                actual_chunk["section_ref"] or "",
                actual_chunk["section_start_char"],
                actual_chunk["section_end_char"],
                actual_chunk["section_content_hash"] or "",
                actual_chunk["content_hash"] or "",
                actual_chunk["total_chunks"],
            )
            if actual_values != expected_values:
                raise VectorIndexConsistencyError("Vector chunks failed publication integrity validation.")
        integrity = await conn.fetchrow(
            """
            SELECT document.content_hash AS document_hash,
                   COUNT(chunk.id)::pg_catalog.int4 AS chunk_count,
                   MIN(chunk.chunk_index)::pg_catalog.int4 AS first_index,
                   MAX(chunk.chunk_index)::pg_catalog.int4 AS last_index,
                   pg_catalog.bool_and(chunk.content_hash = $2) AS hashes_match,
                   pg_catalog.bool_and(chunk.embedding IS NOT NULL) AS embeddings_complete,
                   pg_catalog.bool_and(chunk.total_chunks = $3) AS totals_match
            FROM public.documents AS document
            LEFT JOIN public.document_chunks AS chunk
              ON chunk.doc_id = document.document_id
            WHERE document.document_id = $1
            GROUP BY document.content_hash
            """,
            doc_id,
            content_hash,
            expected_chunks,
        )
        if (
            integrity is None
            or integrity["document_hash"] != content_hash
            or integrity["chunk_count"] != expected_chunks
            or integrity["first_index"] != 0
            or integrity["last_index"] != expected_chunks - 1
            or not integrity["hashes_match"]
            or not integrity["embeddings_complete"]
            or not integrity["totals_match"]
        ):
            raise VectorIndexConsistencyError("Vector chunks failed publication integrity validation.")
        return doc_id, content_hash, self.retrieval_profile_hash, expected_chunks

    # -- Retrieve by ID -------------------------------------------------------

    async def get_document(self, doc_id: str) -> dict | None:
        """Retrieve a full document by ID. Reconstructs from chunks."""
        rows = await self._pool.fetch(
            "SELECT chunk.chunk_index, chunk.chunk_text, document.title, document.category, "
            "document.decision_date, document.decision_number, document.source_url, "
            "publication.expected_chunks AS total_chunks, document.total_pages, "
            "chunk.chunk_start_char, chunk.chunk_end_char "
            "FROM public.document_chunks AS chunk "
            "JOIN public.documents AS document "
            "ON document.document_id = chunk.doc_id AND document.content_hash = chunk.content_hash "
            "JOIN public.document_retrieval_publications AS publication "
            "ON publication.doc_id = chunk.doc_id AND publication.content_hash = chunk.content_hash "
            "WHERE chunk.doc_id = $1 AND publication.retrieval_profile_hash = $2 "
            "ORDER BY chunk.chunk_index",
            doc_id,
            self.retrieval_profile_hash,
        )
        if not rows:
            return None

        full_content = self._reconstruct_content(rows)
        meta = rows[0]

        return {
            "doc_id": doc_id,
            "title": meta["title"] or "",
            "content": full_content,
            "category": meta["category"] or "",
            "decision_date": meta["decision_date"] or "",
            "decision_number": meta["decision_number"] or "",
            "source_url": meta["source_url"] or "",
            "total_chunks": meta["total_chunks"] or 1,
            "total_pages": meta["total_pages"] or 1,
        }

    async def get_document_page(self, doc_id: str, page: int = 1) -> dict | None:
        """Retrieve a paginated page by fetching only the overlapping chunks."""
        # Get document metadata (total_pages, total_chunks, title)
        meta = await self._pool.fetchrow(
            "SELECT document.title, document.total_pages, publication.expected_chunks AS total_chunks, "
            "document.category "
            "FROM public.document_chunks AS chunk "
            "JOIN public.documents AS document "
            "ON document.document_id = chunk.doc_id AND document.content_hash = chunk.content_hash "
            "JOIN public.document_retrieval_publications AS publication "
            "ON publication.doc_id = chunk.doc_id AND publication.content_hash = chunk.content_hash "
            "WHERE chunk.doc_id = $1 AND publication.retrieval_profile_hash = $2 LIMIT 1",
            doc_id,
            self.retrieval_profile_hash,
        )
        if not meta:
            return None

        total_pages = meta["total_pages"] or 1
        if page < 1 or page > total_pages:
            return {
                "doc_id": doc_id,
                "title": meta["title"] or "",
                "content": f"Invalid page {page}. Document has {total_pages} page(s).",
                "page_number": page,
                "total_pages": total_pages,
            }

        start_char = (page - 1) * PAGE_SIZE
        end_char = page * PAGE_SIZE
        rows = await self._pool.fetch(
            "SELECT chunk.chunk_index, chunk.chunk_text, chunk.chunk_start_char, chunk.chunk_end_char "
            "FROM public.document_chunks AS chunk "
            "JOIN public.documents AS document "
            "ON document.document_id = chunk.doc_id AND document.content_hash = chunk.content_hash "
            "JOIN public.document_retrieval_publications AS publication "
            "ON publication.doc_id = chunk.doc_id AND publication.content_hash = chunk.content_hash "
            "WHERE chunk.doc_id = $1 AND publication.retrieval_profile_hash = $2 "
            "AND chunk.chunk_start_char IS NOT NULL "
            "AND chunk.chunk_end_char IS NOT NULL AND chunk.chunk_end_char > $3 "
            "AND chunk.chunk_start_char < $4 ORDER BY chunk.chunk_start_char, chunk.chunk_index",
            doc_id,
            self.retrieval_profile_hash,
            start_char,
            end_char,
        )
        used_offsets = bool(rows)

        if not rows:
            # Fallback for legacy rows without chunk offsets.
            step = max(1, EMBEDDING_CHUNK_SIZE - EMBEDDING_CHUNK_OVERLAP)
            first_chunk = max(0, start_char // step)
            last_chunk = end_char // step + 1  # +1 for safety margin
            rows = await self._pool.fetch(
                "SELECT chunk.chunk_index, chunk.chunk_text, chunk.chunk_start_char, chunk.chunk_end_char "
                "FROM public.document_chunks AS chunk "
                "JOIN public.documents AS document "
                "ON document.document_id = chunk.doc_id AND document.content_hash = chunk.content_hash "
                "JOIN public.document_retrieval_publications AS publication "
                "ON publication.doc_id = chunk.doc_id AND publication.content_hash = chunk.content_hash "
                "WHERE chunk.doc_id = $1 AND publication.retrieval_profile_hash = $2 "
                "AND chunk.chunk_index >= $3 AND chunk.chunk_index <= $4 "
                "ORDER BY chunk.chunk_index",
                doc_id,
                self.retrieval_profile_hash,
                first_chunk,
                last_chunk,
            )

        if not rows:
            # Fallback: fetch all chunks
            doc = await self.get_document(doc_id)
            if not doc:
                return None
            content = doc["content"]
            chunk = content[start_char:end_char]
            return {
                "doc_id": doc_id,
                "title": doc["title"],
                "content": chunk,
                "page_number": page,
                "total_pages": total_pages,
            }

        # Reconstruct just the needed slice
        content = self._reconstruct_content(rows)
        if used_offsets:
            first_start = min(_row_get(row, "chunk_start_char", start_char) for row in rows)
            local_start = start_char - first_start
        else:
            step = max(1, EMBEDDING_CHUNK_SIZE - EMBEDDING_CHUNK_OVERLAP)
            first_chunk = rows[0]["chunk_index"]
            local_start = start_char - first_chunk * step
        local_start = max(0, local_start)
        chunk = content[local_start : local_start + PAGE_SIZE]

        return {
            "doc_id": doc_id,
            "title": meta["title"] or "",
            "content": chunk,
            "page_number": page,
            "total_pages": total_pages,
            "category": meta["category"] or "",
        }

    def _reconstruct_content(self, rows: list[asyncpg.Record]) -> str:
        """Reconstruct full document from overlapping chunks."""
        if not rows:
            return ""
        if len(rows) == 1:
            return rows[0]["chunk_text"]

        if all(_row_get(row, "chunk_start_char") is not None for row in rows) and all(
            _row_get(row, "chunk_end_char") is not None for row in rows
        ):
            parts = []
            cursor: int | None = None
            for row in sorted(
                rows,
                key=lambda item: (_row_get(item, "chunk_start_char", 0), _row_get(item, "chunk_index", 0)),
            ):
                text = row["chunk_text"]
                start_char = _row_get(row, "chunk_start_char", 0)
                end_char = _row_get(row, "chunk_end_char", start_char + len(text))
                if cursor is None:
                    parts.append(text)
                elif start_char < cursor:
                    trim = min(len(text), cursor - start_char)
                    parts.append(text[trim:])
                else:
                    parts.append(text)
                cursor = max(cursor or end_char, end_char)
            return "".join(parts)

        chunk_size = EMBEDDING_CHUNK_SIZE
        overlap = EMBEDDING_CHUNK_OVERLAP
        step = max(1, chunk_size - overlap)

        parts = []
        for i, row in enumerate(rows):
            text = row["chunk_text"]
            if i == 0:
                parts.append(text)
            else:
                expected_start = i * step
                prev_text = rows[i - 1]["chunk_text"]
                already_covered = (i - 1) * step + len(prev_text)
                trim = max(0, already_covered - expected_start)
                if trim < len(text):
                    parts.append(text[trim:])

        return "".join(parts)

    # -- Search: public API ----------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
    ) -> list[dict]:
        """Search documents. Uses hybrid search when enabled, else vector-only."""
        if HYBRID_SEARCH:
            return await self._hybrid_search(query, limit, category)
        return await self._vector_search(query, limit, category)

    # -- Vector-only search (dense retrieval) ----------------------------------

    async def _vector_search(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
        fetch_limit: int | None = None,
    ) -> list[dict]:
        """Cosine similarity search via pgvector HNSW index."""
        self._ensure_embeddings()
        query_embedding = (await self._embed([query], prefix=_QUERY_PREFIX))[0]
        vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        where_parts = ["publication.retrieval_profile_hash = $2"]
        params: list = [vec_str, self.retrieval_profile_hash]
        if category:
            where_parts.append(f"document.category = ${len(params) + 1}")
            params.append(category)
        where_clause = "WHERE " + " AND ".join(where_parts)

        if fetch_limit is None:
            fetch_limit = min(limit * 5, 100)
        sql = f"""
            SELECT chunk.doc_id, document.title, document.category, document.decision_date, chunk.chunk_text,
                   chunk.section_type, chunk.section_ref, chunk.section_start_char, chunk.section_end_char,
                   chunk.section_content_hash,
                   chunk.embedding <=> $1::vector AS distance
            FROM public.document_chunks AS chunk
            JOIN public.documents AS document
              ON document.document_id = chunk.doc_id
             AND document.content_hash = chunk.content_hash
            JOIN public.document_retrieval_publications AS publication
              ON publication.doc_id = chunk.doc_id
             AND publication.content_hash = chunk.content_hash
            {where_clause}
            ORDER BY chunk.embedding <=> $1::vector
            LIMIT ${len(params) + 1}
        """
        params.append(fetch_limit)

        rows = await self._pool.fetch(sql, *params)

        # Deduplicate by doc_id, keep best score
        seen: dict[str, dict] = {}
        for row in rows:
            did = row["doc_id"]
            distance = row["distance"]
            if did not in seen or distance < seen[did]["distance"]:
                seen[did] = {
                    "doc_id": did,
                    "title": row["title"] or "",
                    "category": row["category"] or "",
                    "decision_date": row["decision_date"] or "",
                    "snippet": (row["chunk_text"] or "")[:_SEARCH_SNIPPET_CHARS],
                    "distance": distance,
                    "relevance": round(1 - distance, _SCORE_DECIMAL_PLACES),
                    "semantic_relevance": round(1 - distance, _SCORE_DECIMAL_PLACES),
                    "fts_rank": 0.0,
                    "match_type": "vector",
                    **_section_metadata_from_row(row),
                    **_quality_metadata(row["chunk_text"] or "", did),
                }

        hits = sorted(seen.values(), key=lambda x: x["distance"])
        return hits[:limit]

    # -- FTS search (sparse retrieval) -----------------------------------------

    async def _fts_search(
        self,
        query: str,
        limit: int = 50,
        category: str | None = None,
    ) -> list[dict]:
        """Full-text search on chunk tsvector with ts_rank_cd scoring."""
        rows = await self._fts_rows(query, limit=limit, category=category, relaxed=False)
        if not rows and (relaxed_query := _relaxed_fts_query(query)):
            rows = await self._fts_rows(relaxed_query, limit=limit, category=category, relaxed=True)

        # Deduplicate by doc_id, keep best FTS rank
        seen: dict[str, dict] = {}
        for row in rows:
            did = row["doc_id"]
            rank = float(row["fts_rank"])
            if did not in seen or rank > seen[did]["fts_rank"]:
                seen[did] = {
                    "doc_id": did,
                    "title": row["title"] or "",
                    "category": row["category"] or "",
                    "decision_date": row["decision_date"] or "",
                    "snippet": (row["chunk_text"] or "")[:_SEARCH_SNIPPET_CHARS],
                    "fts_rank": rank,
                    "semantic_relevance": 0.0,
                    "match_type": "fts",
                    **_section_metadata_from_row(row),
                    **_quality_metadata(row["chunk_text"] or "", did),
                }

        return sorted(seen.values(), key=lambda x: x["fts_rank"], reverse=True)

    async def _fts_rows(
        self,
        query: str,
        *,
        limit: int,
        category: str | None,
        relaxed: bool,
    ) -> list[asyncpg.Record]:
        tsquery = (
            "to_tsquery('simple', public.immutable_unaccent($1))"
            if relaxed
            else "plainto_tsquery('simple', public.immutable_unaccent($1))"
        )
        where_parts = [f"chunk.tsv @@ {tsquery}", "publication.retrieval_profile_hash = $2"]
        params: list = [query, self.retrieval_profile_hash]

        if category:
            where_parts.append(f"document.category = ${len(params) + 1}")
            params.append(category)

        where_clause = " AND ".join(where_parts)
        params.append(limit)

        sql = f"""
            SELECT chunk.doc_id, document.title, document.category, document.decision_date, chunk.chunk_text,
                   chunk.section_type, chunk.section_ref, chunk.section_start_char, chunk.section_end_char,
                   chunk.section_content_hash,
                   ts_rank_cd(chunk.tsv, {tsquery}) AS fts_rank
            FROM public.document_chunks AS chunk
            JOIN public.documents AS document
              ON document.document_id = chunk.doc_id
             AND document.content_hash = chunk.content_hash
            JOIN public.document_retrieval_publications AS publication
              ON publication.doc_id = chunk.doc_id
             AND publication.content_hash = chunk.content_hash
            WHERE {where_clause}
            ORDER BY fts_rank DESC
            LIMIT ${len(params)}
        """
        return await self._pool.fetch(sql, *params)

    # -- Hybrid search (RRF fusion) -------------------------------------------

    async def _hybrid_search(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
    ) -> list[dict]:
        """Hybrid search: dense + sparse retrieval fused with RRF, optionally re-ranked.

        Key anti-hallucination features:
          - FTS gate: if FTS finds 0 results, apply penalty to vector scores
          - Score gap filtering: drop results that are far below the top hit
        """
        # Step 1: Parallel retrieval from both systems
        vector_hits, fts_hits = await asyncio.gather(
            self._vector_search(
                query,
                limit=_HYBRID_VECTOR_CANDIDATE_LIMIT,
                category=category,
                fetch_limit=_HYBRID_VECTOR_FETCH_LIMIT,
            ),
            self._fts_search(query, limit=_HYBRID_FTS_CANDIDATE_LIMIT, category=category),
        )
        exact_legal_query = _has_exact_legal_reference(query)
        vector_by_doc = {hit["doc_id"]: hit for hit in vector_hits}
        fts_by_doc = {hit["doc_id"]: hit for hit in fts_hits}

        # Step 2: FTS gate — if FTS returns nothing, the query likely has no
        # keyword overlap with any document. Penalize vector-only scores heavily
        # to prevent returning unrelated results with misleadingly high cosine sim.
        fts_gate_active = len(fts_hits) == 0
        if fts_gate_active:
            for hit in vector_hits:
                hit["relevance"] = round(
                    hit.get("relevance", 0) * _FTS_GATE_PENALTY,
                    _SCORE_DECIMAL_PLACES,
                )
            logger.debug(
                "FTS gate: 0 keyword matches, applying %.0f%% penalty to vector scores", (1 - _FTS_GATE_PENALTY) * 100
            )

        # Step 3: RRF fusion
        fused = self._rrf_fuse(vector_hits, fts_hits)
        for hit in fused:
            did = hit["doc_id"]
            hit["semantic_relevance"] = round(hit.get("relevance", 0.0), _SCORE_DECIMAL_PLACES)
            if did in fts_by_doc:
                hit["fts_rank"] = fts_by_doc[did].get("fts_rank", 0.0)
            else:
                hit.setdefault("fts_rank", 0.0)

            if did in vector_by_doc and did in fts_by_doc:
                hit["match_type"] = "hybrid"
            elif did in fts_by_doc and exact_legal_query:
                hit["match_type"] = "fts_exact"
            elif did in fts_by_doc:
                hit["match_type"] = "fts"
            else:
                hit["match_type"] = "vector"

        # Step 4: Cross-encoder re-ranking (optional)
        if RERANKER_ENABLED and fused:
            top_n = min(RERANKER_TOP_N, len(fused))
            fused[:top_n] = await self._rerank(query, fused[:top_n])
        elif fts_hits:
            max_fts_rank = max((hit.get("fts_rank", 0.0) for hit in fts_hits), default=0.0)
            if max_fts_rank > 0:
                for hit in fused:
                    lexical_relevance = float(hit.get("fts_rank", 0.0)) / max_fts_rank
                    hit["lexical_relevance"] = round(lexical_relevance, _SCORE_DECIMAL_PLACES)
                    if hit.get("semantic_relevance", 0.0) > 0 and lexical_relevance > 0:
                        hit["relevance"] = round(
                            min(1.0, hit.get("relevance", 0.0) + (lexical_relevance * _LEXICAL_RELEVANCE_BOOST)),
                            _SCORE_DECIMAL_PLACES,
                        )
                    phrase_matches = _phrase_match_count(
                        query,
                        f"{hit.get('title', '')} {hit.get('snippet', '')}",
                    )
                    hit["phrase_match_count"] = phrase_matches
                    if hit.get("semantic_relevance", 0.0) > 0 and phrase_matches > 0:
                        hit["relevance"] = round(
                            min(
                                1.0,
                                hit.get("relevance", 0.0)
                                + min(_PHRASE_RELEVANCE_MAX, phrase_matches * _PHRASE_RELEVANCE_BOOST),
                            ),
                            _SCORE_DECIMAL_PLACES,
                        )

        # Step 5: Apply threshold
        for hit in fused:
            if "relevance" not in hit:
                hit["relevance"] = 0.0
            hit["relevance"] = round(hit["relevance"], _SCORE_DECIMAL_PLACES)

        fused = [
            h
            for h in fused
            if h["semantic_relevance"] >= SEMANTIC_RELEVANCE_THRESHOLD or h["match_type"] == "fts_exact"
        ]

        # Step 5b: Re-sort so output order matches the displayed `relevance`.
        # _rrf_fuse() ranks by rrf_score (dense rank + FTS rank), but the
        # number surfaced to the user is the vector cosine. When the two
        # signals disagree, the output can be non-monotonic in the displayed
        # score (e.g. rank #1 = 87.9%, rank #2 = 89.9%). Sorting by
        # `relevance` here keeps RRF's value as a membership filter — FTS
        # can still surface docs the vector search missed — while the final
        # ordering matches what each row says. Idempotent for the reranker
        # path, where `relevance` = sigmoid(rerank_score) is already the
        # sort key in _rerank().
        exact_hits = [h for h in fused if h["match_type"] == "fts_exact"]
        semantic_hits = [h for h in fused if h["match_type"] != "fts_exact"]
        semantic_hits.sort(key=lambda h: h["relevance"], reverse=True)

        # Step 6: Score gap filtering — if there's a large gap between top-1 and
        # the rest, only keep results within a reasonable band of the best score.
        # This prevents returning 10 results when only 1-2 are truly relevant.
        if len(semantic_hits) > 1:
            top_score = semantic_hits[0]["relevance"]
            semantic_hits = [h for h in semantic_hits if (top_score - h["relevance"]) <= _SCORE_GAP_THRESHOLD]

        fused = exact_hits + semantic_hits

        # Step 7: Add confidence labels
        for h in fused:
            if h["relevance"] >= _HIGH_CONFIDENCE_THRESHOLD:
                h["confidence"] = "high"
            elif h["relevance"] >= _MEDIUM_CONFIDENCE_THRESHOLD:
                h["confidence"] = "medium"
            else:
                h["confidence"] = "low"

        return fused[:limit]

    def _rrf_fuse(self, vector_hits: list[dict], fts_hits: list[dict], k: int = HYBRID_RRF_K) -> list[dict]:
        """Reciprocal Rank Fusion: combine two ranked lists into one.

        RRF_score(d) = sum(1 / (k + rank_i(d))) for each system i.
        Higher score = better. k=60 is the standard constant from the RRF paper.
        """
        doc_data: dict[str, dict] = {}
        rrf_scores: dict[str, float] = {}

        # Score from vector search (rank 1 = best)
        for rank, hit in enumerate(vector_hits, 1):
            did = hit["doc_id"]
            rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (k + rank)
            if did not in doc_data:
                doc_data[did] = hit.copy()

        # Score from FTS (rank 1 = best)
        for rank, hit in enumerate(fts_hits, 1):
            did = hit["doc_id"]
            rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (k + rank)
            if did not in doc_data:
                doc_data[did] = hit.copy()

        # Sort by RRF score descending
        ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        results = []
        for did in ranked_ids:
            entry = doc_data[did]
            entry["rrf_score"] = round(rrf_scores[did], _RRF_SCORE_DECIMAL_PLACES)
            # FTS-only hits (not seen in vector_hits) have no cosine — leave
            # `relevance` at 0.0 so the downstream SEMANTIC_RELEVANCE_THRESHOLD
            # filter drops them rather than ranking them as top results with
            # a fake score.
            entry.setdefault("relevance", 0.0)
            results.append(entry)

        return results

    # -- Cross-encoder re-ranking ---------------------------------------------

    async def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Re-rank candidates using a cross-encoder model in a thread."""
        if not candidates:
            return candidates
        self._ensure_reranker()
        pairs = [(query, c["snippet"]) for c in candidates]
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self._rerank_fn.predict(
                pairs,
                batch_size=_RERANKER_PREDICT_BATCH_SIZE,
                show_progress_bar=False,
                activation_fn=None,
                apply_softmax=False,
                convert_to_numpy=True,
                convert_to_tensor=False,
            ),
        )
        for candidate, score in zip(candidates, scores, strict=True):
            candidate["rerank_score"] = float(score)
            candidate["relevance"] = round(
                1.0 / (1.0 + math.exp(-float(score))),
                _SCORE_DECIMAL_PLACES,
            )
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    # -- Bulk operations ------------------------------------------------------

    async def has_document(self, doc_id: str) -> bool:
        """Check if a document exists in the store."""
        row = await self._pool.fetchval(
            "SELECT 1 FROM public.document_chunks AS chunk "
            "JOIN public.documents AS document "
            "ON document.document_id = chunk.doc_id AND document.content_hash = chunk.content_hash "
            "JOIN public.document_retrieval_publications AS publication "
            "ON publication.doc_id = chunk.doc_id AND publication.content_hash = chunk.content_hash "
            "WHERE chunk.doc_id = $1 AND publication.retrieval_profile_hash = $2 LIMIT 1",
            doc_id,
            self.retrieval_profile_hash,
        )
        return row is not None

    async def document_count(self) -> int:
        """Return number of unique documents (not chunks)."""
        return await self._pool.fetchval(
            "SELECT COUNT(DISTINCT chunk.doc_id) FROM public.document_chunks AS chunk "
            "JOIN public.documents AS document "
            "ON document.document_id = chunk.doc_id AND document.content_hash = chunk.content_hash "
            "JOIN public.document_retrieval_publications AS publication "
            "ON publication.doc_id = chunk.doc_id AND publication.content_hash = chunk.content_hash "
            "WHERE publication.retrieval_profile_hash = $1",
            self.retrieval_profile_hash,
        )

    async def chunk_count(self) -> int:
        """Return total number of chunks."""
        return await self._pool.fetchval(
            "SELECT COUNT(*) FROM public.document_chunks AS chunk "
            "JOIN public.documents AS document "
            "ON document.document_id = chunk.doc_id AND document.content_hash = chunk.content_hash "
            "JOIN public.document_retrieval_publications AS publication "
            "ON publication.doc_id = chunk.doc_id AND publication.content_hash = chunk.content_hash "
            "WHERE publication.retrieval_profile_hash = $1",
            self.retrieval_profile_hash,
        )

    async def stats(self) -> dict:
        """Return store statistics."""
        doc_count = await self.document_count()
        chunks = await self.chunk_count()

        categories: dict[str, int] = {}
        rows = await self._pool.fetch(
            "SELECT document.category, COUNT(DISTINCT chunk.doc_id) AS cnt "
            "FROM public.document_chunks AS chunk JOIN public.documents AS document "
            "ON document.document_id = chunk.doc_id AND document.content_hash = chunk.content_hash "
            "JOIN public.document_retrieval_publications AS publication "
            "ON publication.doc_id = chunk.doc_id AND publication.content_hash = chunk.content_hash "
            "WHERE publication.retrieval_profile_hash = $1 "
            "GROUP BY document.category ORDER BY document.category",
            self.retrieval_profile_hash,
        )
        for r in rows:
            categories[r["category"] or "Unknown"] = r["cnt"]

        return {
            "total_documents": doc_count,
            "total_chunks": chunks,
            "categories": categories,
            "embedding_model": self._embedding_model,
            "embedding_model_revision": "local" if EMBEDDING_MODEL_PATH else EMBEDDING_MODEL_REVISION,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "retrieval_profile_hash": self.retrieval_profile_hash,
            "hybrid_search": HYBRID_SEARCH,
            "reranker_enabled": RERANKER_ENABLED,
            "reranker_model_revision": (
                "local" if RERANKER_MODEL_PATH else RERANKER_MODEL_REVISION if RERANKER_ENABLED else None
            ),
        }

    async def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a document."""
        async with self._pool.acquire() as conn, conn.transaction():
            await acquire_corpus_mutation_lock(conn)
            result = await conn.execute("DELETE FROM public.document_chunks WHERE doc_id = $1", doc_id)
            return result != "DELETE 0"
