"""Vector store over the SQLite corpus: chunking, embeddings, hybrid search.

Architecture:
  - Table ``chunks``: section-aware token chunks with float16 embeddings and
    an FTS5 index over normalized text (``chunks_fts``)
  - Embedding model: multilingual-e5-base (offline-first via BDDK_EMBEDDING_MODEL_PATH)
  - Dense retrieval: exact cosine similarity with numpy over all chunk vectors
    (``ponytail``: brute force is exact and takes milliseconds up to a few
    hundred thousand chunks; add an ANN index such as hnswlib beyond that)
  - Hybrid search: dense + FTS5 BM25 fused with Reciprocal Rank Fusion
  - Optional cross-encoder re-ranking
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
import sqlite3
import threading
from dataclasses import dataclass

import numpy as np

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
    HYBRID_RRF_K,
    HYBRID_SEARCH,
    RERANKER_ENABLED,
    RERANKER_MODEL_NAME,
    RERANKER_MODEL_PATH,
    RERANKER_MODEL_REVISION,
    RERANKER_TOP_N,
    SEMANTIC_RELEVANCE_THRESHOLD,
)
from bddk_mcp.quality.markdown_quality import assess_markdown_quality
from bddk_mcp.store.corpus_db import CorpusDB
from bddk_mcp.store.doc_store import fts_match_query, normalize_search_text
from bddk_mcp.store.legal_ref import parse_legal_refs
from bddk_mcp.store.section_index import DocumentSection, extract_document_sections

logger = logging.getLogger(__name__)


class VectorIndexConsistencyError(RuntimeError):
    """The vector index cannot be published for a missing or different source."""


class RetrievalProfileError(RuntimeError):
    """The corpus was embedded with a different model or chunking than this runtime."""


class SemanticSearchReadinessError(RuntimeError):
    """The serving runtime cannot execute the advertised semantic-search path."""


class SemanticSearchUnavailableError(RuntimeError):
    """The embedding runtime failed deterministically and requires operator action."""


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
_EMBEDDING_DTYPE = np.float16

# Loaded embedding models shared by every VectorStore in this process, keyed by
# (loader class, model reference, load kwargs).
_LOADED_EMBEDDING_MODELS: dict[tuple[object, str, tuple[tuple[str, object], ...]], object] = {}
_LOADED_EMBEDDING_MODELS_LOCK = threading.Lock()
_LOADED_EMBEDDING_MODELS_LIMIT = 4


# -- Chunking -----------------------------------------------------------------


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
            raise RetrievalProfileError("Token chunking is configured but the embedding tokenizer is unavailable.")
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


def _model_reference(embedding_model: str = EMBEDDING_MODEL_NAME) -> tuple[str, dict]:
    """Return the embedding model reference and the offline-first load kwargs."""
    model_ref = EMBEDDING_MODEL_PATH if EMBEDDING_MODEL_PATH else embedding_model
    kwargs: dict = {"local_files_only": bool(EMBEDDING_MODEL_PATH), "trust_remote_code": False}
    if not EMBEDDING_MODEL_PATH and EMBEDDING_MODEL_REVISION:
        kwargs["revision"] = EMBEDDING_MODEL_REVISION
    return model_ref, kwargs


def _torch_device() -> str:
    """Pick the device once instead of loading the weights twice on CPU-only hosts."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch missing or a broken CUDA probe
        return "cpu"


def _load_torch_model(loader, model_ref: str, **model_kwargs):
    """Load a sentence-transformers model on the selected device, falling back to CPU."""
    device = _torch_device()
    try:
        return loader(model_ref, device=device, **model_kwargs), device
    except (RuntimeError, ValueError, AssertionError):
        if device == "cpu":
            raise
        return loader(model_ref, device="cpu", **model_kwargs), "cpu"


def _load_embedding_tokenizer(embedding_model: str = EMBEDDING_MODEL_NAME):
    if EMBEDDING_CHUNK_MODE != "token":
        return None
    model_ref, kwargs = _model_reference(embedding_model)
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_ref, use_fast=True, **kwargs)
    except Exception as exc:
        logger.error(
            "Embedding tokenizer load failed (%s source)",
            "local" if EMBEDDING_MODEL_PATH else "remote",
            extra={"error_type": type(exc).__name__},
        )
        raise RetrievalProfileError("The embedding tokenizer could not be loaded.") from None


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


# -- Query helpers ------------------------------------------------------------


def _has_exact_legal_reference(query: str) -> bool:
    refs = parse_legal_refs(query)
    return bool(refs.sections or refs.decision_numbers or refs.dates)


def _relaxed_fts_query(query: str) -> str | None:
    """Loose fallback: OR of prefix-matched content words when the strict AND query is empty."""
    terms: list[str] = []
    seen: set[str] = set()
    for match in _FTS_TOKEN_RE.finditer(query):
        term = normalize_search_text(match.group(0))
        if match.group(0).lower() in _FTS_RELAXED_STOPWORDS or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= _FTS_RELAXED_MAX_TERMS:
            break

    if len(terms) < _FTS_RELAXED_MIN_TERMS:
        return None
    return " OR ".join(f'"{term}"*' for term in terms)


def _normalize_search_text(text: str) -> str:
    return normalize_search_text(text)


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


async def _document_quality_by_ids(db: CorpusDB, doc_ids: list[str]) -> dict[str, dict]:
    """Assess quality from full stored documents, not the matching chunk."""
    unique_ids = list(dict.fromkeys(doc_ids))
    if not unique_ids:
        return {}
    placeholders = ",".join("?" for _ in unique_ids)
    rows = await db.run(
        lambda conn: conn.execute(
            f"SELECT document_id, markdown_content FROM documents WHERE document_id IN ({placeholders})",
            unique_ids,
        ).fetchall()
    )
    quality_by_id = {
        row["document_id"]: _quality_metadata(row["markdown_content"] or "", row["document_id"]) for row in rows
    }
    for doc_id in unique_ids:
        quality_by_id.setdefault(doc_id, _quality_metadata("", doc_id))
    return quality_by_id


def _section_metadata_from_row(row: sqlite3.Row) -> dict:
    return {
        "section_type": row["section_type"] or "",
        "section_ref": row["section_ref"] or "",
        "section_start_char": row["section_start_char"],
        "section_end_char": row["section_end_char"],
        "section_content_hash": row["section_content_hash"] or "",
    }


def _embedding_blob(vector) -> bytes:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    if array.shape[0] != EMBEDDING_DIMENSION or not np.all(np.isfinite(array)) or not np.any(array):
        raise VectorIndexConsistencyError("An embedding vector is malformed.")
    return array.astype(_EMBEDDING_DTYPE).tobytes()


def chunk_digest(texts: list[str]) -> str:
    """Identity of a chunk list: exported embeddings are reused only for exactly these texts."""
    digest = hashlib.sha256()
    for text in texts:
        digest.update(f"{len(text)}:".encode())
        digest.update(text.encode())
    return digest.hexdigest()


@dataclass
class _VectorIndex:
    """Every current chunk vector in memory, aligned with its row identity."""

    token: tuple[int, int]
    rowids: np.ndarray
    doc_ids: list[str]
    categories: np.ndarray
    matrix: np.ndarray

    @classmethod
    def load(cls, conn: sqlite3.Connection, token: tuple[int, int]) -> _VectorIndex:
        rows = conn.execute(
            "SELECT c.rowid AS rowid, c.doc_id, d.category, c.embedding FROM chunks AS c "
            "JOIN documents AS d ON d.document_id = c.doc_id AND d.content_hash = c.content_hash "
            "ORDER BY c.rowid"
        ).fetchall()
        width = EMBEDDING_DIMENSION * np.dtype(_EMBEDDING_DTYPE).itemsize
        if any(len(row["embedding"]) != width for row in rows):
            raise VectorIndexConsistencyError("A stored embedding has the wrong size for this runtime.")
        if rows:
            matrix = np.frombuffer(b"".join(row["embedding"] for row in rows), dtype=_EMBEDDING_DTYPE)
            matrix = matrix.reshape(len(rows), EMBEDDING_DIMENSION).astype(np.float32)
        else:
            matrix = np.zeros((0, EMBEDDING_DIMENSION), dtype=np.float32)
        return cls(
            token=token,
            rowids=np.array([row["rowid"] for row in rows], dtype=np.int64),
            doc_ids=[row["doc_id"] for row in rows],
            categories=np.array([row["category"] or "" for row in rows], dtype=object),
            matrix=matrix,
        )


class VectorStore:
    """Chunk index with exact cosine search, FTS5 keyword search and RRF fusion.

    Usage::

        store = VectorStore(db)
        await store.add_document(doc_id="1291", title="...", content="...")
        results = await store.search("sermaye yeterliliği hesaplama", limit=10)
    """

    def __init__(
        self,
        db: CorpusDB,
        embedding_model: str = EMBEDDING_MODEL_NAME,
        *,
        embed_fn=None,
        tokenizer=None,
    ) -> None:
        self._db = db
        self._embedding_model = embedding_model
        # Test seams: ``embed_fn`` (texts -> unit vectors) replaces the model and
        # ``tokenizer`` (``encode(text, add_special_tokens=False)``) the chunk tokenizer.
        self._embed_fn = embed_fn
        self._chunk_tokenizer_fn = tokenizer
        self._rerank_fn = None
        self._embedding_load_lock = threading.Lock()
        self._semantic_search_ready = False
        self._semantic_search_failure_type: str | None = None
        self._index: _VectorIndex | None = None
        self._index_lock = threading.Lock()

    @property
    def semantic_search_ready(self) -> bool:
        """Whether the complete read-only semantic-search path passed its startup probe."""
        return self._semantic_search_ready and self._semantic_search_failure_type is None

    def _raise_if_semantic_search_unavailable(self) -> None:
        if self._semantic_search_failure_type is not None:
            raise SemanticSearchUnavailableError("Semantic search is unavailable in this runtime.")

    async def close(self) -> None:
        logger.info("VectorStore closed")

    # -- Model loading -----------------------------------------------------------

    def _ensure_embeddings(self) -> None:
        """Lazy-load the embedding model on first search/add."""
        if self._embed_fn is not None:
            return
        with self._embedding_load_lock:
            if self._embed_fn is not None:
                return
            from sentence_transformers import SentenceTransformer

            model_ref, model_kwargs = _model_reference(self._embedding_model)
            model_kwargs["backend"] = "torch"
            model_source = "local" if EMBEDDING_MODEL_PATH else "remote"
            cache_key = (SentenceTransformer, model_ref, tuple(sorted(model_kwargs.items())))
            with _LOADED_EMBEDDING_MODELS_LOCK:
                loaded_model = _LOADED_EMBEDDING_MODELS.get(cache_key)
                if loaded_model is None:
                    logger.info("Loading embedding model (%s source)", model_source)
                    loaded_model, device = _load_torch_model(SentenceTransformer, model_ref, **model_kwargs)
                    logger.info("Loaded embedding model (%s execution, %s source)", device, model_source)
                    if len(_LOADED_EMBEDDING_MODELS) >= _LOADED_EMBEDDING_MODELS_LIMIT:
                        _LOADED_EMBEDDING_MODELS.clear()
                    _LOADED_EMBEDDING_MODELS[cache_key] = loaded_model
            dimension = loaded_model.get_sentence_embedding_dimension()
            if dimension != EMBEDDING_DIMENSION:
                raise RuntimeError(f"Embedding model dimension must be {EMBEDDING_DIMENSION} for the corpus.")

            def encode(texts: list[str]):
                return loaded_model.encode(
                    texts,
                    batch_size=_EMBEDDING_ENCODE_BATCH_SIZE,
                    show_progress_bar=False,
                    output_value="sentence_embedding",
                    precision="float32",
                    convert_to_numpy=True,
                    convert_to_tensor=False,
                    normalize_embeddings=True,
                )

            self._embed_fn = encode

    def _chunk_tokenizer(self):
        if EMBEDDING_CHUNK_MODE != "token":
            return None
        if self._chunk_tokenizer_fn is None:
            self._chunk_tokenizer_fn = _load_embedding_tokenizer(self._embedding_model)
        return self._chunk_tokenizer_fn

    def _ensure_reranker(self) -> None:
        """Lazy-load the cross-encoder re-ranking model."""
        if self._rerank_fn is not None:
            return
        from sentence_transformers import CrossEncoder

        model_ref = RERANKER_MODEL_PATH if RERANKER_MODEL_PATH else RERANKER_MODEL_NAME
        model_kwargs = {
            "backend": "torch",
            "local_files_only": bool(RERANKER_MODEL_PATH),
            "trust_remote_code": False,
        }
        model_source = "local" if RERANKER_MODEL_PATH else "remote"
        if not RERANKER_MODEL_PATH and RERANKER_MODEL_REVISION:
            model_kwargs["revision"] = RERANKER_MODEL_REVISION
        logger.info("Loading cross-encoder reranker (%s source)", model_source)
        loaded_model, device = _load_torch_model(CrossEncoder, model_ref, **model_kwargs)
        logger.info("Loaded cross-encoder reranker (%s execution, %s source)", device, model_source)
        self._rerank_fn = loaded_model

    async def _embed(self, texts: list[str], prefix: str = _PASSAGE_PREFIX) -> np.ndarray:
        """Embed texts (unit vectors, float32) on a worker thread."""
        if prefix not in {_PASSAGE_PREFIX, _QUERY_PREFIX}:
            raise RetrievalProfileError("Unknown embedding prefix.")
        self._raise_if_semantic_search_unavailable()
        prefixed = [f"{prefix}{_PREFIX_SEPARATOR}{text}" for text in texts]
        try:
            await asyncio.to_thread(self._ensure_embeddings)
            embeddings = await asyncio.to_thread(self._embed_fn, prefixed)
        except Exception as error:
            self._semantic_search_ready = False
            self._semantic_search_failure_type = type(error).__name__
            logger.exception("Semantic embedding path failed (error_type=%s)", self._semantic_search_failure_type)
            raise SemanticSearchUnavailableError("Semantic search is unavailable in this runtime.") from None
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2 or array.shape != (len(texts), EMBEDDING_DIMENSION):
            raise VectorIndexConsistencyError("The embedding output has the wrong shape.")
        return array

    # -- Add documents -----------------------------------------------------------

    async def add_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        category: str = "",
        decision_date: str = "",
        decision_number: str = "",
        source_url: str = "",
        *,
        cached_embeddings=None,
    ) -> int:
        """Embed and publish a document's chunks. Returns the number of chunks.

        ``cached_embeddings(digest, count)`` may return the vectors exported earlier
        for exactly these chunk texts (see ``chunk_digest``); a seed whose chunks and
        model are unchanged then builds without loading the embedding model.
        """
        if not content.strip():
            return 0
        tokenizer = await asyncio.to_thread(self._chunk_tokenizer)
        chunks = _chunk_document(doc_id, content, tokenizer=tokenizer)
        if not chunks:
            return 0
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        texts = [chunk.chunk_text for chunk in chunks]
        embeddings = cached_embeddings(chunk_digest(texts), len(texts)) if cached_embeddings else None
        if embeddings is None or embeddings.shape != (len(texts), EMBEDDING_DIMENSION):
            embeddings = await self._embed(texts)
        rows = [
            (
                doc_id,
                index,
                title,
                chunk.chunk_text,
                chunk.start_char,
                chunk.end_char,
                chunk.section_type,
                chunk.section_ref,
                chunk.section_start_char,
                chunk.section_end_char,
                chunk.section_content_hash,
                content_hash,
                _embedding_blob(vector),
                normalize_search_text(title),
                normalize_search_text(chunk.chunk_text),
            )
            for index, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=True))
        ]

        def _publish(conn: sqlite3.Connection) -> int:
            stored = conn.execute("SELECT content_hash FROM documents WHERE document_id = ?", (doc_id,)).fetchone()
            if stored is None or stored["content_hash"] != content_hash:
                raise VectorIndexConsistencyError(
                    "Vector chunks were not published because the stored document hash does not match."
                )
            meta = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM meta")}
            expected = {"embedding_model": self._embedding_model, "embedding_dimension": str(EMBEDDING_DIMENSION)}
            for key, value in expected.items():
                if key in meta and meta[key] != value:
                    raise RetrievalProfileError("The corpus was built with a different embedding model.")
                conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)", (key, value))
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.executemany(
                "INSERT INTO chunks (doc_id, chunk_index, title, chunk_text, start_char, end_char, section_type, "
                "section_ref, section_start_char, section_end_char, section_content_hash, content_hash, embedding, "
                "title_norm, chunk_text_norm) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            return len(rows)

        written = await self._db.transaction(_publish)
        logger.debug("Added document to vector index (chunks=%d)", written)
        return written

    # -- Index --------------------------------------------------------------------

    def _current_index(self) -> _VectorIndex:
        """Return the in-memory vector index, reloading it after any corpus write."""
        token = self._db.change_token()
        with self._index_lock:
            if self._index is None or self._index.token != token:
                self._index = self._db.run_sync(lambda conn: _VectorIndex.load(conn, token))
                logger.info("Vector index loaded (chunks=%d)", len(self._index.doc_ids))
            return self._index

    # -- Search: public API ----------------------------------------------------

    async def search(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
    ) -> list[dict]:
        """Search documents. Uses hybrid search when enabled, else vector-only."""
        self._raise_if_semantic_search_unavailable()
        if HYBRID_SEARCH:
            return await self._hybrid_search(query, limit, category)
        return await self._vector_search(query, limit, category)

    async def assert_semantic_search_ready(self) -> None:
        """Fail closed unless model loading, encoding and corpus retrieval all work.

        An empty result is healthy: this probe verifies the executable retrieval
        path, not the relevance of a particular corpus snapshot.
        """
        if self._semantic_search_ready:
            return
        try:
            await self.search("bankacılık düzenlemeleri", limit=1)
        except SemanticSearchUnavailableError:
            raise
        except Exception as error:
            logger.exception("Semantic search readiness probe failed (error_type=%s)", type(error).__name__)
            raise SemanticSearchReadinessError("Semantic search readiness probe failed.") from None
        self._semantic_search_ready = True

    # -- Dense retrieval --------------------------------------------------------

    async def _vector_search(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
        fetch_limit: int | None = None,
    ) -> list[dict]:
        """Exact cosine similarity over every current chunk vector."""
        query_embedding = (await self._embed([query], prefix=_QUERY_PREFIX))[0]
        if fetch_limit is None:
            fetch_limit = min(limit * 5, 100)
        index = await asyncio.to_thread(self._current_index)
        if not index.doc_ids:
            return []
        scores = index.matrix @ query_embedding
        if category:
            scores = np.where(index.categories == category, scores, -np.inf)
        count = min(fetch_limit, scores.shape[0])
        candidate_positions = np.argpartition(-scores, count - 1)[:count]
        candidate_positions = candidate_positions[np.argsort(-scores[candidate_positions], kind="stable")]
        candidate_positions = [int(position) for position in candidate_positions if np.isfinite(scores[position])]
        if not candidate_positions:
            return []
        rowids = [int(index.rowids[position]) for position in candidate_positions]
        placeholders = ",".join("?" for _ in rowids)
        rows = await self._db.run(
            lambda conn: conn.execute(
                f"SELECT c.rowid AS rowid, c.doc_id, d.title, d.category, d.decision_date, c.chunk_text, "
                f"c.section_type, c.section_ref, c.section_start_char, c.section_end_char, c.section_content_hash "
                f"FROM chunks AS c JOIN documents AS d ON d.document_id = c.doc_id "
                f"WHERE c.rowid IN ({placeholders})",
                rowids,
            ).fetchall()
        )
        row_by_id = {row["rowid"]: row for row in rows}

        # Deduplicate by doc_id, keep best score
        seen: dict[str, dict] = {}
        for position, rowid in zip(candidate_positions, rowids, strict=True):
            row = row_by_id.get(rowid)
            if row is None:
                continue
            did = row["doc_id"]
            distance = float(1.0 - scores[position])
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
                }

        quality_by_id = await _document_quality_by_ids(self._db, list(seen))
        for did, hit in seen.items():
            hit.update(quality_by_id[did])

        hits = sorted(seen.values(), key=lambda x: x["distance"])
        return hits[:limit]

    # -- Keyword retrieval -----------------------------------------------------

    async def _fts_search(
        self,
        query: str,
        limit: int = 50,
        category: str | None = None,
    ) -> list[dict]:
        """FTS5 search on chunk text (plus title) with BM25 scoring."""
        rows = await self._fts_rows(fts_match_query(query), limit=limit, category=category)
        if not rows and (relaxed_query := _relaxed_fts_query(query)):
            rows = await self._fts_rows(relaxed_query, limit=limit, category=category)

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
                }

        quality_by_id = await _document_quality_by_ids(self._db, list(seen))
        for did, hit in seen.items():
            hit.update(quality_by_id[did])

        return sorted(seen.values(), key=lambda x: x["fts_rank"], reverse=True)

    async def _fts_rows(self, match: str | None, *, limit: int, category: str | None) -> list[sqlite3.Row]:
        if match is None:
            return []
        where = ["chunks_fts MATCH ?"]
        params: list = [match]
        if category:
            where.append("d.category = ?")
            params.append(category)
        params.append(limit)
        sql = f"""
            SELECT c.doc_id, d.title, d.category, d.decision_date, c.chunk_text,
                   c.section_type, c.section_ref, c.section_start_char, c.section_end_char,
                   c.section_content_hash, -bm25(chunks_fts, 2.0, 1.0) AS fts_rank
            FROM chunks_fts
            JOIN chunks AS c ON c.rowid = chunks_fts.rowid
            JOIN documents AS d ON d.document_id = c.doc_id AND d.content_hash = c.content_hash
            WHERE {" AND ".join(where)}
            ORDER BY fts_rank DESC
            LIMIT ?
        """
        return await self._db.run(lambda conn: conn.execute(sql, params).fetchall())

    # -- Hybrid search (RRF fusion) -------------------------------------------

    async def _hybrid_search(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
    ) -> list[dict]:
        """Hybrid search: dense + sparse retrieval fused with RRF, optionally re-ranked.

        Anti-hallucination features: an FTS gate penalises vector-only scores
        when no keyword matches at all, and score-gap filtering drops results
        far below the top hit.
        """
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

        if len(fts_hits) == 0:
            for hit in vector_hits:
                hit["relevance"] = round(hit.get("relevance", 0) * _FTS_GATE_PENALTY, _SCORE_DECIMAL_PLACES)
            logger.debug(
                "FTS gate: 0 keyword matches, applying %.0f%% penalty to vector scores", (1 - _FTS_GATE_PENALTY) * 100
            )

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
                    phrase_matches = _phrase_match_count(query, f"{hit.get('title', '')} {hit.get('snippet', '')}")
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

        for hit in fused:
            hit["relevance"] = round(hit.get("relevance", 0.0), _SCORE_DECIMAL_PLACES)

        fused = [
            h
            for h in fused
            if h["semantic_relevance"] >= SEMANTIC_RELEVANCE_THRESHOLD or h["match_type"] == "fts_exact"
        ]

        # Output order follows the displayed relevance; RRF only decides membership.
        exact_hits = [h for h in fused if h["match_type"] == "fts_exact"]
        semantic_hits = [h for h in fused if h["match_type"] != "fts_exact"]
        semantic_hits.sort(key=lambda h: h["relevance"], reverse=True)

        if len(semantic_hits) > 1:
            top_score = semantic_hits[0]["relevance"]
            semantic_hits = [h for h in semantic_hits if (top_score - h["relevance"]) <= _SCORE_GAP_THRESHOLD]

        fused = exact_hits + semantic_hits

        for h in fused:
            if h["relevance"] >= _HIGH_CONFIDENCE_THRESHOLD:
                h["confidence"] = "high"
            elif h["relevance"] >= _MEDIUM_CONFIDENCE_THRESHOLD:
                h["confidence"] = "medium"
            else:
                h["confidence"] = "low"

        return fused[:limit]

    def _rrf_fuse(self, vector_hits: list[dict], fts_hits: list[dict], k: int = HYBRID_RRF_K) -> list[dict]:
        """Reciprocal Rank Fusion: RRF(d) = sum over systems of 1 / (k + rank)."""
        doc_data: dict[str, dict] = {}
        rrf_scores: dict[str, float] = {}

        for rank, hit in enumerate(vector_hits, 1):
            did = hit["doc_id"]
            rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (k + rank)
            if did not in doc_data:
                doc_data[did] = hit.copy()

        for rank, hit in enumerate(fts_hits, 1):
            did = hit["doc_id"]
            rrf_scores[did] = rrf_scores.get(did, 0.0) + 1.0 / (k + rank)
            if did not in doc_data:
                doc_data[did] = hit.copy()

        ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        results = []
        for did in ranked_ids:
            entry = doc_data[did]
            entry["rrf_score"] = round(rrf_scores[did], _RRF_SCORE_DECIMAL_PLACES)
            # FTS-only hits have no cosine: leave relevance at 0.0 so the
            # semantic threshold drops them instead of faking a score.
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
        scores = await asyncio.to_thread(
            lambda: self._rerank_fn.predict(
                pairs,
                batch_size=_RERANKER_PREDICT_BATCH_SIZE,
                show_progress_bar=False,
                activation_fn=None,
                apply_softmax=False,
                convert_to_numpy=True,
                convert_to_tensor=False,
            )
        )
        for candidate, score in zip(candidates, scores, strict=True):
            candidate["rerank_score"] = float(score)
            candidate["relevance"] = round(1.0 / (1.0 + math.exp(-float(score))), _SCORE_DECIMAL_PLACES)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    # -- Bookkeeping ------------------------------------------------------------

    async def has_document(self, doc_id: str) -> bool:
        """Whether current (hash-matching) chunks exist for the document."""
        row = await self._db.run(
            lambda conn: conn.execute(
                "SELECT 1 FROM chunks AS c JOIN documents AS d "
                "ON d.document_id = c.doc_id AND d.content_hash = c.content_hash WHERE c.doc_id = ? LIMIT 1",
                (doc_id,),
            ).fetchone()
        )
        return row is not None

    async def chunk_count(self) -> int:
        return await self._db.run(
            lambda conn: conn.execute(
                "SELECT COUNT(*) FROM chunks AS c JOIN documents AS d "
                "ON d.document_id = c.doc_id AND d.content_hash = c.content_hash"
            ).fetchone()[0]
        )

    async def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a document."""
        deleted = await self._db.transaction(
            lambda conn: conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,)).rowcount
        )
        return deleted > 0
