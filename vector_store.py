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
import logging
import math
import re
from dataclasses import dataclass

import asyncpg

from config import (
    EMBEDDING_CHUNK_MODE,
    EMBEDDING_CHUNK_OVERLAP,
    EMBEDDING_CHUNK_SIZE,
    EMBEDDING_CHUNK_TARGET_TOKENS,
    EMBEDDING_CHUNK_TOKEN_OVERLAP,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_PATH,
    HYBRID_RRF_K,
    HYBRID_SEARCH,
    PAGE_SIZE,
    RERANKER_ENABLED,
    RERANKER_MODEL_NAME,
    RERANKER_MODEL_PATH,
    RERANKER_TOP_N,
    SEMANTIC_RELEVANCE_THRESHOLD,
)
from legal_ref import parse_legal_refs
from markdown_quality import assess_markdown_quality
from section_index import DocumentSection, extract_document_sections

logger = logging.getLogger(__name__)

_SCHEMA_SQL = f"""\
CREATE TABLE IF NOT EXISTS document_chunks (
    id              SERIAL PRIMARY KEY,
    doc_id          TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    title           TEXT DEFAULT '',
    category        TEXT DEFAULT '',
    decision_date   TEXT DEFAULT '',
    decision_number TEXT DEFAULT '',
    source_url      TEXT DEFAULT '',
    total_chunks    INTEGER DEFAULT 1,
    total_pages     INTEGER DEFAULT 1,
    content_hash    TEXT DEFAULT '',
    chunk_start_char INTEGER,
    chunk_end_char   INTEGER,
    section_type    TEXT DEFAULT '',
    section_ref     TEXT DEFAULT '',
    section_start_char INTEGER,
    section_end_char   INTEGER,
    section_content_hash TEXT DEFAULT '',
    chunk_text      TEXT NOT NULL,
    embedding       vector({EMBEDDING_DIMENSION}),
    tsv             tsvector,
    UNIQUE(doc_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON document_chunks USING gin(tsv);
"""

# HNSW index created separately (expensive, only once)
_HNSW_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
ON document_chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
"""

# Trigger to auto-populate tsvector on insert/update
_FTS_TRIGGER_SQL = """\
CREATE OR REPLACE FUNCTION chunks_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.tsv := to_tsvector('simple', immutable_unaccent(coalesce(NEW.title, '')))
            || to_tsvector('simple', immutable_unaccent(coalesce(NEW.chunk_text, '')));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS chunks_tsv_update ON document_chunks;
CREATE TRIGGER chunks_tsv_update BEFORE INSERT OR UPDATE
ON document_chunks FOR EACH ROW EXECUTE FUNCTION chunks_tsv_trigger();
"""

# Migration for existing installations without tsv column
_MIGRATION_SQL = """\
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'document_chunks' AND column_name = 'tsv'
    ) THEN
        ALTER TABLE document_chunks ADD COLUMN tsv tsvector;
        CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON document_chunks USING gin(tsv);
    END IF;
END $$;
"""

_SECTION_METADATA_MIGRATION_SQL = """\
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'document_chunks' AND column_name = 'chunk_start_char'
    ) THEN
        ALTER TABLE document_chunks ADD COLUMN chunk_start_char INTEGER;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'document_chunks' AND column_name = 'chunk_end_char'
    ) THEN
        ALTER TABLE document_chunks ADD COLUMN chunk_end_char INTEGER;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'document_chunks' AND column_name = 'section_type'
    ) THEN
        ALTER TABLE document_chunks ADD COLUMN section_type TEXT DEFAULT '';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'document_chunks' AND column_name = 'section_ref'
    ) THEN
        ALTER TABLE document_chunks ADD COLUMN section_ref TEXT DEFAULT '';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'document_chunks' AND column_name = 'section_start_char'
    ) THEN
        ALTER TABLE document_chunks ADD COLUMN section_start_char INTEGER;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'document_chunks' AND column_name = 'section_end_char'
    ) THEN
        ALTER TABLE document_chunks ADD COLUMN section_end_char INTEGER;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'document_chunks' AND column_name = 'section_content_hash'
    ) THEN
        ALTER TABLE document_chunks ADD COLUMN section_content_hash TEXT DEFAULT '';
    END IF;
END $$;
CREATE INDEX IF NOT EXISTS idx_chunks_section_ref ON document_chunks(section_type, section_ref);
"""


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
    if use_token_chunking and tokenizer is not None:
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
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return len(tokenizer.encode(text))


def _tokenizer_from_model(model):
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None:
        return tokenizer
    try:
        first_module = model[0]
    except (TypeError, KeyError, IndexError):
        return None
    return getattr(first_module, "tokenizer", None)


def _load_embedding_tokenizer():
    if EMBEDDING_CHUNK_MODE != "token":
        return None
    model_ref = EMBEDDING_MODEL_PATH if EMBEDDING_MODEL_PATH else EMBEDDING_MODEL_NAME
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_ref)
    except Exception as exc:
        logger.warning("Falling back to character chunking; tokenizer load failed for %s: %s", model_ref, exc)
        return None


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
        await store.initialize()
        await store.add_document(doc_id="1291", title="...", content="...", metadata={...})
        results = await store.search("sermaye yeterliliği hesaplama", limit=10)
        doc = await store.get_document("1291")
    """

    def __init__(self, pool: asyncpg.Pool, embedding_model: str = EMBEDDING_MODEL_NAME) -> None:
        self._pool = pool
        self._embedding_model = embedding_model
        self._embed_fn = None
        self._rerank_fn = None

    async def initialize(self) -> None:
        """Create schema, indexes, FTS trigger, and run migrations."""
        async with self._pool.acquire() as conn:
            # Extensions and helper function first
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
            await conn.execute("""
                CREATE OR REPLACE FUNCTION immutable_unaccent(text) RETURNS text AS $$
                    SELECT unaccent($1)
                $$ LANGUAGE sql IMMUTABLE;
            """)
            await conn.execute(_SCHEMA_SQL)
            # Migration adds tsv column to tables created before FTS was added
            await conn.execute(_MIGRATION_SQL)
            await conn.execute(_SECTION_METADATA_MIGRATION_SQL)
            await conn.execute(_FTS_TRIGGER_SQL)
            await conn.execute(_HNSW_INDEX_SQL)

        # Backfill tsvector for existing chunks that don't have it
        null_count = await self._pool.fetchval("SELECT COUNT(*) FROM document_chunks WHERE tsv IS NULL")
        if null_count and null_count > 0:
            logger.info("Backfilling tsvector for %d chunks...", null_count)
            await self._pool.execute("UPDATE document_chunks SET chunk_text = chunk_text WHERE tsv IS NULL")
            logger.info("tsvector backfill complete")

        logger.info("VectorStore initialized (pgvector + FTS hybrid)")

    async def close(self) -> None:
        """No-op — pool lifecycle is managed externally."""
        logger.info("VectorStore closed")

    # -- Model loading -----------------------------------------------------------

    def _ensure_embeddings(self) -> None:
        """Lazy-load the embedding model on first search/add."""
        if self._embed_fn is not None:
            return

        from sentence_transformers import SentenceTransformer

        model_ref = EMBEDDING_MODEL_PATH if EMBEDDING_MODEL_PATH else self._embedding_model
        if EMBEDDING_MODEL_PATH:
            logger.info("Loading embeddings from local path: %s", EMBEDDING_MODEL_PATH)
        else:
            logger.info("Loading embeddings from model name: %s (may download)", self._embedding_model)

        try:
            self._embed_fn = SentenceTransformer(model_ref, device="cuda")
            logger.info("Loaded GPU-accelerated embeddings: %s", model_ref)
        except (RuntimeError, ValueError, AssertionError):
            # CPU-only torch raises AssertionError on CUDA probe, not RuntimeError.
            self._embed_fn = SentenceTransformer(model_ref, device="cpu")
            logger.info("Loaded CPU embeddings: %s", model_ref)

    def _chunk_tokenizer(self):
        if EMBEDDING_CHUNK_MODE != "token":
            return None
        self._ensure_embeddings()
        tokenizer = _tokenizer_from_model(self._embed_fn)
        if tokenizer is None:
            logger.warning("Embedding model did not expose a tokenizer; falling back to character chunking")
        return tokenizer

    def _ensure_reranker(self) -> None:
        """Lazy-load the cross-encoder re-ranking model."""
        if self._rerank_fn is not None:
            return

        from sentence_transformers import CrossEncoder

        model_ref = RERANKER_MODEL_PATH if RERANKER_MODEL_PATH else RERANKER_MODEL_NAME
        logger.info("Loading cross-encoder reranker: %s", model_ref)

        try:
            self._rerank_fn = CrossEncoder(model_ref, device="cuda")
            logger.info("Loaded GPU-accelerated reranker: %s", model_ref)
        except (RuntimeError, ValueError, AssertionError):
            # CPU-only torch raises AssertionError on CUDA probe, not RuntimeError.
            self._rerank_fn = CrossEncoder(model_ref, device="cpu")
            logger.info("Loaded CPU reranker: %s", model_ref)

    async def _embed(self, texts: list[str], prefix: str = "passage") -> list[list[float]]:
        """Generate embeddings in a thread to avoid blocking the event loop."""
        self._ensure_embeddings()
        prefixed = [f"{prefix}: {t}" for t in texts]
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._embed_fn.encode(prefixed, normalize_embeddings=True),
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
                # Delete old chunks
                await conn.execute("DELETE FROM document_chunks WHERE doc_id = $1", doc_id)

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

                await conn.executemany(
                    """
                    INSERT INTO document_chunks (
                        doc_id, chunk_index, title, category, decision_date,
                        decision_number, source_url, total_chunks, total_pages,
                        content_hash, chunk_start_char, chunk_end_char,
                        section_type, section_ref, section_start_char,
                        section_end_char, section_content_hash, chunk_text, embedding
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19::vector)
                    """,
                    args_list,
                )

        logger.debug("Added %s: %d chunks", doc_id, len(chunks))
        return len(chunks)

    # -- Retrieve by ID -------------------------------------------------------

    async def get_document(self, doc_id: str) -> dict | None:
        """Retrieve a full document by ID. Reconstructs from chunks."""
        rows = await self._pool.fetch(
            "SELECT chunk_index, chunk_text, title, category, decision_date, "
            "decision_number, source_url, total_chunks, total_pages, "
            "chunk_start_char, chunk_end_char "
            "FROM document_chunks WHERE doc_id = $1 ORDER BY chunk_index",
            doc_id,
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
            "SELECT title, total_pages, total_chunks, category FROM document_chunks WHERE doc_id = $1 LIMIT 1",
            doc_id,
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
            "SELECT chunk_index, chunk_text, chunk_start_char, chunk_end_char FROM document_chunks "
            "WHERE doc_id = $1 AND chunk_start_char IS NOT NULL AND chunk_end_char IS NOT NULL "
            "AND chunk_end_char > $2 AND chunk_start_char < $3 "
            "ORDER BY chunk_start_char, chunk_index",
            doc_id,
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
                "SELECT chunk_index, chunk_text, chunk_start_char, chunk_end_char FROM document_chunks "
                "WHERE doc_id = $1 AND chunk_index >= $2 AND chunk_index <= $3 "
                "ORDER BY chunk_index",
                doc_id,
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
        query_embedding = (await self._embed([query], prefix="query"))[0]
        vec_str = "[" + ",".join(str(v) for v in query_embedding) + "]"

        where_clause = ""
        params: list = [vec_str]
        if category:
            where_clause = "WHERE category = $2"
            params.append(category)

        if fetch_limit is None:
            fetch_limit = min(limit * 5, 100)
        sql = f"""
            SELECT doc_id, title, category, decision_date, chunk_text,
                   section_type, section_ref, section_start_char, section_end_char, section_content_hash,
                   embedding <=> $1::vector AS distance
            FROM document_chunks
            {where_clause}
            ORDER BY embedding <=> $1::vector
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
                    "snippet": (row["chunk_text"] or "")[:800],
                    "distance": distance,
                    "relevance": round(1 - distance, 4),
                    "semantic_relevance": round(1 - distance, 4),
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
        where_parts = ["tsv @@ plainto_tsquery('simple', immutable_unaccent($1))"]
        params: list = [query]

        if category:
            where_parts.append(f"category = ${len(params) + 1}")
            params.append(category)

        where_clause = " AND ".join(where_parts)
        params.append(limit)

        sql = f"""
            SELECT doc_id, title, category, decision_date, chunk_text,
                   section_type, section_ref, section_start_char, section_end_char, section_content_hash,
                   ts_rank_cd(tsv, plainto_tsquery('simple', immutable_unaccent($1))) AS fts_rank
            FROM document_chunks
            WHERE {where_clause}
            ORDER BY fts_rank DESC
            LIMIT ${len(params)}
        """

        rows = await self._pool.fetch(sql, *params)

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
                    "snippet": (row["chunk_text"] or "")[:800],
                    "fts_rank": rank,
                    "semantic_relevance": 0.0,
                    "match_type": "fts",
                    **_section_metadata_from_row(row),
                    **_quality_metadata(row["chunk_text"] or "", did),
                }

        return sorted(seen.values(), key=lambda x: x["fts_rank"], reverse=True)

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
            self._vector_search(query, limit=50, category=category, fetch_limit=100),
            self._fts_search(query, limit=50, category=category),
        )
        exact_legal_query = _has_exact_legal_reference(query)
        vector_by_doc = {hit["doc_id"]: hit for hit in vector_hits}
        fts_by_doc = {hit["doc_id"]: hit for hit in fts_hits}

        # Step 2: FTS gate — if FTS returns nothing, the query likely has no
        # keyword overlap with any document. Penalize vector-only scores heavily
        # to prevent returning unrelated results with misleadingly high cosine sim.
        fts_gate_active = len(fts_hits) == 0
        if fts_gate_active:
            _FTS_GATE_PENALTY = 0.65
            for hit in vector_hits:
                hit["relevance"] = round(hit.get("relevance", 0) * _FTS_GATE_PENALTY, 4)
            logger.debug(
                "FTS gate: 0 keyword matches, applying %.0f%% penalty to vector scores", (1 - _FTS_GATE_PENALTY) * 100
            )

        # Step 3: RRF fusion
        fused = self._rrf_fuse(vector_hits, fts_hits)
        for hit in fused:
            did = hit["doc_id"]
            hit["semantic_relevance"] = round(hit.get("relevance", 0.0), 4)
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

        # Step 5: Apply threshold
        for hit in fused:
            if "relevance" not in hit:
                hit["relevance"] = 0.0
            hit["relevance"] = round(hit["relevance"], 4)

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
            _SCORE_GAP_THRESHOLD = 0.08  # drop results >8% below top hit
            top_score = semantic_hits[0]["relevance"]
            semantic_hits = [h for h in semantic_hits if (top_score - h["relevance"]) <= _SCORE_GAP_THRESHOLD]

        fused = exact_hits + semantic_hits

        # Step 7: Add confidence labels
        for h in fused:
            if h["relevance"] >= 0.70:
                h["confidence"] = "high"
            elif h["relevance"] >= 0.50:
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
            entry["rrf_score"] = round(rrf_scores[did], 6)
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
        scores = await loop.run_in_executor(None, self._rerank_fn.predict, pairs)
        for candidate, score in zip(candidates, scores, strict=True):
            candidate["rerank_score"] = float(score)
            candidate["relevance"] = round(1.0 / (1.0 + math.exp(-float(score))), 4)
        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

    # -- Bulk operations ------------------------------------------------------

    async def has_document(self, doc_id: str) -> bool:
        """Check if a document exists in the store."""
        row = await self._pool.fetchval(
            "SELECT 1 FROM document_chunks WHERE doc_id = $1 LIMIT 1",
            doc_id,
        )
        return row is not None

    async def document_count(self) -> int:
        """Return number of unique documents (not chunks)."""
        return await self._pool.fetchval("SELECT COUNT(DISTINCT doc_id) FROM document_chunks")

    async def chunk_count(self) -> int:
        """Return total number of chunks."""
        return await self._pool.fetchval("SELECT COUNT(*) FROM document_chunks")

    async def stats(self) -> dict:
        """Return store statistics."""
        doc_count = await self.document_count()
        chunks = await self.chunk_count()

        categories: dict[str, int] = {}
        rows = await self._pool.fetch(
            "SELECT category, COUNT(DISTINCT doc_id) AS cnt FROM document_chunks GROUP BY category ORDER BY category"
        )
        for r in rows:
            categories[r["category"] or "Unknown"] = r["cnt"]

        return {
            "total_documents": doc_count,
            "total_chunks": chunks,
            "categories": categories,
            "embedding_model": self._embedding_model,
            "hybrid_search": HYBRID_SEARCH,
            "reranker_enabled": RERANKER_ENABLED,
        }

    async def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a document."""
        result = await self._pool.execute("DELETE FROM document_chunks WHERE doc_id = $1", doc_id)
        return result != "DELETE 0"
