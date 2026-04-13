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

import asyncpg

from config import (
    EMBEDDING_CHUNK_OVERLAP,
    EMBEDDING_CHUNK_SIZE,
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


def _chunk_text(text: str, chunk_size: int = EMBEDDING_CHUNK_SIZE, overlap: int = EMBEDDING_CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks for embedding."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


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
        except (RuntimeError, ValueError):
            self._embed_fn = SentenceTransformer(model_ref, device="cpu")
            logger.info("Loaded CPU embeddings: %s", model_ref)

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
        except (RuntimeError, ValueError):
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

        chunks = _chunk_text(content)
        if not chunks:
            return 0

        total_pages = max(1, math.ceil(len(content) / PAGE_SIZE))
        content_hash = hashlib.md5(content.encode()).hexdigest()

        # Generate embeddings
        embeddings = await self._embed(chunks)

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
                            chunk,
                            vec_str,
                        )
                    )

                await conn.executemany(
                    """
                    INSERT INTO document_chunks (
                        doc_id, chunk_index, title, category, decision_date,
                        decision_number, source_url, total_chunks, total_pages,
                        content_hash, chunk_text, embedding
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12::vector)
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
            "decision_number, source_url, total_chunks, total_pages "
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

        # Calculate which chunks overlap with the requested page
        step = EMBEDDING_CHUNK_SIZE - EMBEDDING_CHUNK_OVERLAP
        start_char = (page - 1) * PAGE_SIZE
        end_char = page * PAGE_SIZE
        first_chunk = max(0, start_char // step)
        last_chunk = end_char // step + 1  # +1 for safety margin

        rows = await self._pool.fetch(
            "SELECT chunk_index, chunk_text FROM document_chunks "
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

        chunk_size = EMBEDDING_CHUNK_SIZE
        overlap = EMBEDDING_CHUNK_OVERLAP
        step = chunk_size - overlap

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

        # Step 4: Cross-encoder re-ranking (optional)
        if RERANKER_ENABLED and fused:
            top_n = min(RERANKER_TOP_N, len(fused))
            fused[:top_n] = await self._rerank(query, fused[:top_n])

        # Step 5: Apply threshold
        for hit in fused:
            if "relevance" not in hit:
                hit["relevance"] = 0.0
            hit["relevance"] = round(hit["relevance"], 4)

        fused = [h for h in fused if h["relevance"] >= SEMANTIC_RELEVANCE_THRESHOLD]

        # Step 6: Score gap filtering — if there's a large gap between top-1 and
        # the rest, only keep results within a reasonable band of the best score.
        # This prevents returning 10 results when only 1-2 are truly relevant.
        if len(fused) > 1:
            _SCORE_GAP_THRESHOLD = 0.08  # drop results >8% below top hit
            top_score = fused[0]["relevance"]
            fused = [h for h in fused if (top_score - h["relevance"]) <= _SCORE_GAP_THRESHOLD]

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
            # Preserve vector relevance if available, else estimate from RRF position
            if "relevance" not in entry or entry["relevance"] == 0.0:
                entry["relevance"] = entry.get("relevance", 0.0)
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
        for candidate, score in zip(candidates, scores, strict=False):
            candidate["rerank_score"] = float(score)
            import math as _math

            candidate["relevance"] = round(1.0 / (1.0 + _math.exp(-float(score))), 4)
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
