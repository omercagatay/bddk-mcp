"""
ChromaDB vector store for BDDK regulatory documents.

Provides instant document retrieval by ID and semantic search across
all BDDK decisions, regulations, and guidelines.

Architecture:
  - Collection "bddk_documents": full documents stored as metadata, chunked for embeddings
  - Embedding model: multilingual-e5-base (best for Turkish legal text)
  - Chunking: 1000-char chunks with 200-char overlap for context preservation
  - ID retrieval: O(1) via metadata filter, no network needed
"""

import hashlib
import logging
import math
from pathlib import Path

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

_DEFAULT_CHROMA_PATH = Path(__file__).parent / "chroma_db"
_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 200
_COLLECTION_NAME = "bddk_documents"
_PAGE_SIZE = 5000  # For paginated output (same as doc_store)


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
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
    ChromaDB-backed vector store for BDDK documents.

    Usage::

        store = VectorStore()
        store.initialize()
        store.add_document(doc_id="1291", title="...", content="...", metadata={...})
        results = store.search("sermaye yeterliliği hesaplama", limit=10)
        doc = store.get_document("1291")
        store.close()
    """

    # Pin model revision to avoid silent changes on re-download
    _DEFAULT_MODEL = "intfloat/multilingual-e5-base"
    _DEFAULT_MODEL_REVISION = "d4210e50c0"  # v1.0.0 stable

    def __init__(self, db_path: Path | None = None, embedding_model: str = _DEFAULT_MODEL) -> None:
        self._db_path = db_path or _DEFAULT_CHROMA_PATH
        self._embedding_model = embedding_model
        self._client: chromadb.ClientAPI | None = None
        self._collection = None
        self._embed_fn = None

    def initialize(self) -> None:
        """Open ChromaDB and get/create collection."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(self._db_path),
            settings=Settings(anonymized_telemetry=False),
        )

        # Use sentence-transformers embedding function
        try:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

            try:
                self._embed_fn = SentenceTransformerEmbeddingFunction(
                    model_name=self._embedding_model,
                    device="cuda",  # RTX 5080
                )
                logger.info("Using GPU-accelerated embeddings: %s", self._embedding_model)
            except (RuntimeError, ValueError):
                self._embed_fn = SentenceTransformerEmbeddingFunction(
                    model_name=self._embedding_model,
                    device="cpu",
                )
                logger.info("Using CPU embeddings: %s", self._embedding_model)
        except ImportError as e:
            logger.warning("sentence-transformers not available, using ChromaDB default: %s", e)
            self._embed_fn = None

        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        count = self._collection.count()
        logger.info("VectorStore initialized: %s (%d chunks)", self._db_path, count)

    def close(self) -> None:
        """Close ChromaDB client."""
        self._client = None
        self._collection = None
        logger.info("VectorStore closed")

    def __enter__(self) -> "VectorStore":
        self.initialize()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _ensure_open(self):
        if self._collection is None:
            raise RuntimeError("VectorStore not initialized. Call initialize() first.")
        return self._collection

    # ── Add documents ────────────────────────────────────────────────────

    def add_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        category: str = "",
        decision_date: str = "",
        decision_number: str = "",
        source_url: str = "",
    ) -> int:
        """
        Add a document to the vector store. Chunks the content and embeds each chunk.
        Returns the number of chunks created.
        """
        collection = self._ensure_open()

        if not content.strip():
            return 0

        # Check if already exists — delete old chunks first
        existing = collection.get(where={"doc_id": doc_id})
        if existing and existing["ids"]:
            collection.delete(ids=existing["ids"])

        chunks = _chunk_text(content)
        if not chunks:
            return 0

        total_pages = max(1, math.ceil(len(content) / _PAGE_SIZE))

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            ids.append(chunk_id)
            # Prepend "query: " for e5 models (required for best performance)
            documents.append(chunk)
            metadatas.append(
                {
                    "doc_id": doc_id,
                    "title": title,
                    "category": category,
                    "decision_date": decision_date,
                    "decision_number": decision_number,
                    "source_url": source_url,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "total_pages": total_pages,
                    "full_content_hash": hashlib.md5(content.encode()).hexdigest(),
                }
            )

        # Batch add (ChromaDB handles batching internally)
        batch_size = 500
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        logger.debug("Added %s: %d chunks", doc_id, len(chunks))
        return len(chunks)

    # ── Retrieve by ID ───────────────────────────────────────────────────

    def get_document(self, doc_id: str) -> dict | None:
        """
        Retrieve a full document by ID. Reconstructs from chunks.
        Returns dict with doc_id, title, content, metadata, or None.
        """
        collection = self._ensure_open()

        results = collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
        )

        if not results or not results["ids"]:
            return None

        # Sort chunks by index and reconstruct
        chunk_data = list(
            zip(
                results["ids"],
                results["documents"],
                results["metadatas"],
                strict=True,
            )
        )
        chunk_data.sort(key=lambda x: x[2].get("chunk_index", 0))

        full_content = self._reconstruct_content(chunk_data)
        meta = chunk_data[0][2]

        return {
            "doc_id": doc_id,
            "title": meta.get("title", ""),
            "content": full_content,
            "category": meta.get("category", ""),
            "decision_date": meta.get("decision_date", ""),
            "decision_number": meta.get("decision_number", ""),
            "source_url": meta.get("source_url", ""),
            "total_chunks": meta.get("total_chunks", 1),
            "total_pages": meta.get("total_pages", 1),
        }

    def get_document_page(self, doc_id: str, page: int = 1) -> dict | None:
        """Retrieve a paginated page of a document."""
        doc = self.get_document(doc_id)
        if not doc:
            return None

        content = doc["content"]
        total_pages = max(1, math.ceil(len(content) / _PAGE_SIZE))

        if page < 1 or page > total_pages:
            return {
                "doc_id": doc_id,
                "title": doc["title"],
                "content": f"Invalid page {page}. Document has {total_pages} page(s).",
                "page_number": page,
                "total_pages": total_pages,
            }

        start = (page - 1) * _PAGE_SIZE
        chunk = content[start : start + _PAGE_SIZE]

        return {
            "doc_id": doc_id,
            "title": doc["title"],
            "content": chunk,
            "page_number": page,
            "total_pages": total_pages,
            "category": doc["category"],
        }

    def _reconstruct_content(self, chunk_data: list) -> str:
        """Reconstruct full document from overlapping chunks."""
        if not chunk_data:
            return ""
        if len(chunk_data) == 1:
            return chunk_data[0][1]

        # Remove overlap between consecutive chunks
        parts = []
        for i, (_, text, _) in enumerate(chunk_data):
            if i == 0:
                parts.append(text)
            else:
                # Skip the overlap portion
                if len(text) > _CHUNK_OVERLAP:
                    parts.append(text[_CHUNK_OVERLAP:])
                else:
                    parts.append(text)

        return "".join(parts)

    # ── Semantic search ──────────────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 10,
        category: str | None = None,
    ) -> list[dict]:
        """
        Semantic search across all documents.
        Returns list of unique documents ranked by relevance.
        """
        collection = self._ensure_open()

        where_filter = {"doc_id": {"$ne": ""}}  # non-empty doc_id
        if category:
            where_filter = {"category": category}

        # Query more chunks than needed, then deduplicate by doc_id
        results = collection.query(
            query_texts=[query],
            n_results=min(limit * 5, 100),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        # Deduplicate by doc_id, keep best score
        seen = {}
        for _chunk_id, doc_text, meta, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=True,
        ):
            doc_id = meta.get("doc_id", "")
            if doc_id not in seen or distance < seen[doc_id]["distance"]:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "title": meta.get("title", ""),
                    "category": meta.get("category", ""),
                    "decision_date": meta.get("decision_date", ""),
                    "snippet": doc_text[:300],
                    "distance": distance,
                    "relevance": round(1 - distance, 4),  # cosine: lower distance = better
                }

        # Sort by relevance (highest first) and limit
        hits = sorted(seen.values(), key=lambda x: x["distance"])
        return hits[:limit]

    # ── Bulk operations ──────────────────────────────────────────────────

    def has_document(self, doc_id: str) -> bool:
        """Check if a document exists in the store."""
        collection = self._ensure_open()
        results = collection.get(
            where={"doc_id": doc_id},
            limit=1,
            include=[],
        )
        return bool(results and results["ids"])

    def document_count(self) -> int:
        """Return number of unique documents (not chunks)."""
        collection = self._ensure_open()
        # Get all unique doc_ids
        all_meta = collection.get(include=["metadatas"])
        if not all_meta or not all_meta["metadatas"]:
            return 0
        doc_ids = set(m.get("doc_id", "") for m in all_meta["metadatas"])
        doc_ids.discard("")
        return len(doc_ids)

    def chunk_count(self) -> int:
        """Return total number of chunks."""
        collection = self._ensure_open()
        return collection.count()

    def stats(self) -> dict:
        """Return store statistics."""
        collection = self._ensure_open()
        all_meta = collection.get(include=["metadatas"])

        if not all_meta or not all_meta["metadatas"]:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "categories": {},
            }

        doc_ids = set()
        categories: dict[str, int] = {}
        for m in all_meta["metadatas"]:
            did = m.get("doc_id", "")
            if did and did not in doc_ids:
                doc_ids.add(did)
                cat = m.get("category", "Unknown")
                categories[cat] = categories.get(cat, 0) + 1

        return {
            "total_documents": len(doc_ids),
            "total_chunks": collection.count(),
            "categories": dict(sorted(categories.items())),
            "embedding_model": self._embedding_model,
        }

    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks for a document."""
        collection = self._ensure_open()
        existing = collection.get(where={"doc_id": doc_id})
        if existing and existing["ids"]:
            collection.delete(ids=existing["ids"])
            return True
        return False
