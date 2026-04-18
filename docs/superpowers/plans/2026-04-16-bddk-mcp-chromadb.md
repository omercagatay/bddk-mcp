# BDDK MCP ChromaDB Rebuild — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the BDDK MCP server from scratch using ChromaDB (embedded mode) instead of PostgreSQL + pgvector.

**Architecture:** Flat module design — `server.py`, `store.py`, `scraper.py`, `embeddings.py`, `seed.py`, `models.py`, `config.py`. Two ChromaDB collections (`decisions`, `documents`). Six core MCP tools. Seed data for initial load, live scraper for updates.

**Tech Stack:** Python 3.12, FastMCP, ChromaDB, sentence-transformers (multilingual-e5-base), httpx, BeautifulSoup, Pydantic, pytest

**Spec:** `docs/superpowers/specs/2026-04-16-bddk-mcp-chromadb-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata, dependencies |
| `config.py` | Dataclass config from env vars |
| `embeddings.py` | Custom ChromaDB embedding function wrapping multilingual-e5-base |
| `store.py` | ChromaDB client, collections, add/search/get/count + chunking |
| `models.py` | Pydantic schemas for tool I/O |
| `scraper.py` | BDDK website scraping (decisions + documents) |
| `seed.py` | Load seed JSON files into ChromaDB |
| `server.py` | FastMCP entry point, tool definitions, startup |
| `tests/conftest.py` | Shared fixtures: in-memory ChromaDB, sample data |
| `tests/test_config.py` | Config loading tests |
| `tests/test_embeddings.py` | Embedding function tests |
| `tests/test_store.py` | Store operations tests |
| `tests/test_scraper.py` | Scraper parsing tests (mocked HTTP) |
| `tests/test_seed.py` | Seed loading tests |
| `tests/test_tools.py` | MCP tool integration tests |

---

### Task 1: Project Setup and Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `config.py`
- Create: `tests/test_config.py`

> **Important:** This is a from-scratch rebuild. Before starting, move the existing source files out of the way. Keep `seed_data/` and `docs/` — delete or archive everything else (`.py` files, `tools/`, `Dockerfile`, `docker-compose.yml`, etc.). The new project starts clean.

- [ ] **Step 1: Archive existing source files**

```bash
cd /home/cagatay/bddk-mcp
git checkout -b chromadb-rebuild
# Remove old source files (keep seed_data/, docs/, tests dir structure, .git, .claude, .github)
rm -f client.py doc_store.py vector_store.py doc_sync.py data_sources.py analytics.py deps.py exceptions.py metrics.py logging_config.py utils.py models.py config.py seed.py server.py
rm -rf tools/
rm -f Dockerfile Dockerfile.spaces docker-compose.yml Procfile railway.toml
rm -f pyproject.toml uv.lock
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "bddk-mcp"
version = "6.0.0"
description = "BDDK MCP Server — Turkish banking regulatory intelligence (ChromaDB backend)"
requires-python = ">=3.12,<3.14"
dependencies = [
    "mcp[cli]>=1.0.0",
    "fastmcp>=2.0",
    "chromadb>=1.0",
    "sentence-transformers>=5.0",
    "httpx>=0.27",
    "beautifulsoup4>=4.12",
    "pydantic>=2.0",
]

[dependency-groups]
dev = [
    "pytest>=9.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.8",
]

[tool.ruff]
target-version = "py312"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 3: Write failing test for config**

```python
# tests/test_config.py
import os
from config import Config


def test_default_config():
    cfg = Config()
    assert cfg.chroma_path == "./chroma_data"
    assert cfg.embedding_model == "intfloat/multilingual-e5-base"
    assert cfg.seed_dir == "./seed_data"
    assert cfg.chunk_size == 512
    assert cfg.chunk_overlap == 64
    assert cfg.scrape_delay == 1.0
    assert cfg.mcp_transport == "stdio"
    assert cfg.mcp_host == "0.0.0.0"
    assert cfg.mcp_port == 8000


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("BDDK_CHROMA_PATH", "/tmp/test_chroma")
    monkeypatch.setenv("BDDK_CHUNK_SIZE", "1024")
    monkeypatch.setenv("MCP_TRANSPORT", "streamable-http")
    monkeypatch.setenv("MCP_PORT", "9000")
    cfg = Config()
    assert cfg.chroma_path == "/tmp/test_chroma"
    assert cfg.chunk_size == 1024
    assert cfg.mcp_transport == "streamable-http"
    assert cfg.mcp_port == 9000
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd /home/cagatay/bddk-mcp && uv sync --dev && uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 5: Implement `config.py`**

```python
# config.py
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    chroma_path: str = "./chroma_data"
    embedding_model: str = "intfloat/multilingual-e5-base"
    seed_dir: str = "./seed_data"
    chunk_size: int = 512
    chunk_overlap: int = 64
    scrape_delay: float = 1.0
    mcp_transport: str = "stdio"
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 8000

    def __init__(self) -> None:
        self.chroma_path = os.environ.get("BDDK_CHROMA_PATH", "./chroma_data")
        self.embedding_model = os.environ.get("BDDK_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
        self.seed_dir = os.environ.get("BDDK_SEED_DIR", "./seed_data")
        self.chunk_size = int(os.environ.get("BDDK_CHUNK_SIZE", "512"))
        self.chunk_overlap = int(os.environ.get("BDDK_CHUNK_OVERLAP", "64"))
        self.scrape_delay = float(os.environ.get("BDDK_SCRAPE_DELAY", "1.0"))
        self.mcp_transport = os.environ.get("MCP_TRANSPORT", "stdio")
        self.mcp_host = os.environ.get("MCP_HOST", "0.0.0.0")
        self.mcp_port = int(os.environ.get("MCP_PORT", "8000"))
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: 2 PASSED

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml config.py tests/test_config.py
git commit -m "feat: project setup with config module (ChromaDB rebuild)"
```

---

### Task 2: Embedding Function

**Files:**
- Create: `embeddings.py`
- Create: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_embeddings.py
from embeddings import E5EmbeddingFunction


def test_embedding_function_returns_list_of_lists():
    ef = E5EmbeddingFunction("intfloat/multilingual-e5-base")
    results = ef(["BDDK bankacılık düzenlemesi"])
    assert len(results) == 1
    assert isinstance(results[0], list)
    assert len(results[0]) == 768  # e5-base dimension


def test_embedding_function_multiple_inputs():
    ef = E5EmbeddingFunction("intfloat/multilingual-e5-base")
    results = ef(["birinci metin", "ikinci metin", "üçüncü metin"])
    assert len(results) == 3
    for vec in results:
        assert len(vec) == 768


def test_embedding_function_query_prefix():
    """E5 models expect 'query: ' prefix for queries and 'passage: ' for documents.
    The embedding function adds 'passage: ' by default since ChromaDB calls it for documents."""
    ef = E5EmbeddingFunction("intfloat/multilingual-e5-base")
    doc_result = ef(["bankacılık düzenlemesi"])
    # Just verify it runs and returns correct shape — prefix correctness
    # is verified by search quality in integration tests
    assert len(doc_result[0]) == 768
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'embeddings'`

- [ ] **Step 3: Implement `embeddings.py`**

```python
# embeddings.py
from __future__ import annotations

from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer


class E5EmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB embedding function using multilingual-e5-base.

    E5 models expect prefixed inputs:
    - 'passage: ' for documents (used when ChromaDB embeds stored documents)
    - 'query: ' for search queries (used explicitly via embed_query)
    """

    def __init__(self, model_name: str) -> None:
        self._model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        prefixed = [f"passage: {text}" for text in input]
        embeddings = self._model.encode(prefixed, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embeddings = self._model.encode([f"query: {query}"], normalize_embeddings=True)
        return embeddings[0].tolist()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_embeddings.py -v`
Expected: 3 PASSED (first run downloads model — may take a minute)

- [ ] **Step 5: Commit**

```bash
git add embeddings.py tests/test_embeddings.py
git commit -m "feat: E5 embedding function for ChromaDB"
```

---

### Task 3: ChromaDB Store

**Files:**
- Create: `store.py`
- Create: `tests/conftest.py`
- Create: `tests/test_store.py`

- [ ] **Step 1: Create shared test fixtures**

```python
# tests/conftest.py
import chromadb
import pytest

from config import Config
from embeddings import E5EmbeddingFunction
from store import Store


@pytest.fixture(scope="session")
def embedding_function():
    return E5EmbeddingFunction("intfloat/multilingual-e5-base")


@pytest.fixture
def store(tmp_path, embedding_function):
    cfg = Config()
    cfg.chroma_path = str(tmp_path / "chroma_data")
    return Store(cfg, embedding_function)


SAMPLE_DECISIONS = [
    {
        "id": "310",
        "document": "Bankacılık sektörüne yönelik faiz oranları kararı",
        "metadata": {
            "decision_date": "15.01.2025",
            "decision_number": "10987",
            "category": "Kurul Kararı",
            "institution": "Ziraat Bankası",
            "source_url": "https://www.bddk.org.tr/Mevzuat/DokumanGetir/310",
        },
    },
    {
        "id": "311",
        "document": "Kredi kartı taksitlendirme sınırlaması hakkında karar",
        "metadata": {
            "decision_date": "20.02.2025",
            "decision_number": "10988",
            "category": "Kurul Kararı",
            "institution": "Garanti BBVA",
            "source_url": "https://www.bddk.org.tr/Mevzuat/DokumanGetir/311",
        },
    },
    {
        "id": "312",
        "document": "Sermaye yeterliliği oranı değişikliği tebliği",
        "metadata": {
            "decision_date": "10.03.2025",
            "decision_number": "10989",
            "category": "Kurul Kararı",
            "institution": "",
            "source_url": "https://www.bddk.org.tr/Mevzuat/DokumanGetir/312",
        },
    },
]

SAMPLE_CHUNKS = [
    {
        "id": "doc1_chunk_0",
        "document": "Bankaların likidite karşılama oranı yüzde yüz olmalıdır. Likidite tamponu yüksek kaliteli likit varlıklardan oluşmalıdır.",
        "metadata": {
            "doc_id": "doc1",
            "title": "Likidite Yönetmeliği",
            "doc_type": "Yönetmelik",
            "publish_date": "01.01.2024",
            "chunk_index": 0,
            "total_chunks": 2,
            "source_url": "https://www.bddk.org.tr/Mevzuat/DokumanGetir/100",
        },
    },
    {
        "id": "doc1_chunk_1",
        "document": "Net istikrarlı fonlama oranı hesaplamasında mevduat ve krediler dikkate alınır.",
        "metadata": {
            "doc_id": "doc1",
            "title": "Likidite Yönetmeliği",
            "doc_type": "Yönetmelik",
            "publish_date": "01.01.2024",
            "chunk_index": 1,
            "total_chunks": 2,
            "source_url": "https://www.bddk.org.tr/Mevzuat/DokumanGetir/100",
        },
    },
    {
        "id": "doc2_chunk_0",
        "document": "Sermaye yeterliliği standart oranı asgari yüzde sekiz olarak belirlenmiştir.",
        "metadata": {
            "doc_id": "doc2",
            "title": "Sermaye Yeterliliği Tebliği",
            "doc_type": "Tebliğ",
            "publish_date": "15.06.2024",
            "chunk_index": 0,
            "total_chunks": 1,
            "source_url": "https://www.bddk.org.tr/Mevzuat/DokumanGetir/200",
        },
    },
]
```

- [ ] **Step 2: Write failing tests for store**

```python
# tests/test_store.py
from conftest import SAMPLE_DECISIONS, SAMPLE_CHUNKS


def test_add_and_count_decisions(store):
    store.add_decisions(SAMPLE_DECISIONS)
    assert store.count("decisions") == 3


def test_add_and_count_document_chunks(store):
    store.add_document_chunks(SAMPLE_CHUNKS)
    assert store.count("documents") == 3


def test_get_decision_by_id(store):
    store.add_decisions(SAMPLE_DECISIONS)
    result = store.get("decisions", ["310"])
    assert len(result["ids"]) == 1
    assert result["ids"][0] == "310"
    assert "faiz oranları" in result["documents"][0]


def test_get_document_chunks_by_id(store):
    store.add_document_chunks(SAMPLE_CHUNKS)
    result = store.get("documents", ["doc1_chunk_0", "doc1_chunk_1"])
    assert len(result["ids"]) == 2


def test_search_decisions(store):
    store.add_decisions(SAMPLE_DECISIONS)
    results = store.search("decisions", "faiz oranları bankacılık", n=2)
    assert len(results["ids"][0]) <= 2
    assert len(results["ids"][0]) > 0


def test_search_documents(store):
    store.add_document_chunks(SAMPLE_CHUNKS)
    results = store.search("documents", "likidite karşılama oranı", n=2)
    assert len(results["ids"][0]) <= 2
    assert len(results["ids"][0]) > 0


def test_search_with_metadata_filter(store):
    store.add_decisions(SAMPLE_DECISIONS)
    results = store.search(
        "decisions",
        "karar",
        n=10,
        filters={"institution": "Garanti BBVA"},
    )
    ids = results["ids"][0]
    assert "311" in ids
    assert "310" not in ids


def test_upsert_does_not_duplicate(store):
    store.add_decisions(SAMPLE_DECISIONS)
    store.add_decisions(SAMPLE_DECISIONS)  # add same data again
    assert store.count("decisions") == 3


def test_chunk_text(store):
    text = "A" * 1000
    chunks = store.chunk_text(text, chunk_size=512, overlap=64)
    assert len(chunks) >= 2
    # First chunk is 512 chars
    assert len(chunks[0]) == 512
    # Overlap: second chunk starts 448 chars in (512 - 64)
    assert chunks[0][-64:] == chunks[1][:64]


def test_chunk_text_short_input(store):
    text = "Short text"
    chunks = store.chunk_text(text, chunk_size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0] == "Short text"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'store'`

- [ ] **Step 4: Implement `store.py`**

```python
# store.py
from __future__ import annotations

import chromadb
from chromadb.api.models.Collection import Collection

from config import Config
from embeddings import E5EmbeddingFunction


class Store:
    def __init__(self, config: Config, embedding_function: E5EmbeddingFunction) -> None:
        self._client = chromadb.PersistentClient(path=config.chroma_path)
        self._ef = embedding_function
        self._config = config

        self.decisions: Collection = self._client.get_or_create_collection(
            name="decisions",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        self.documents: Collection = self._client.get_or_create_collection(
            name="documents",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def _get_collection(self, name: str) -> Collection:
        if name == "decisions":
            return self.decisions
        if name == "documents":
            return self.documents
        raise ValueError(f"Unknown collection: {name}")

    def add_decisions(self, decisions: list[dict]) -> None:
        if not decisions:
            return
        self.decisions.upsert(
            ids=[d["id"] for d in decisions],
            documents=[d["document"] for d in decisions],
            metadatas=[d["metadata"] for d in decisions],
        )

    def add_document_chunks(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        self.documents.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["document"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )

    def search(
        self,
        collection_name: str,
        query: str,
        n: int = 10,
        filters: dict | None = None,
    ) -> dict:
        collection = self._get_collection(collection_name)
        query_embedding = self._ef.embed_query(query)
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": min(n, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters
        return collection.query(**kwargs)

    def get(self, collection_name: str, ids: list[str]) -> dict:
        collection = self._get_collection(collection_name)
        return collection.get(ids=ids, include=["documents", "metadatas"])

    def count(self, collection_name: str) -> int:
        return self._get_collection(collection_name).count()

    def chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
        if len(text) <= chunk_size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            # Try to break at whitespace if not at the end of text
            if end < len(text):
                last_space = chunk.rfind(" ")
                if last_space > chunk_size // 2:
                    chunk = chunk[:last_space]
                    end = start + last_space
            chunks.append(chunk)
            start = end - overlap
        return chunks
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: 10 PASSED

> **Note:** The `chunk_text` tests use exact character boundaries without whitespace. The real implementation tries to break at whitespace, so the exact overlap test may need adjustment. If `test_chunk_text` fails because of whitespace-aware splitting, update the test to use text without spaces (e.g., `"A" * 1000`) which has no whitespace to split on, making the behavior deterministic.

- [ ] **Step 6: Commit**

```bash
git add store.py tests/conftest.py tests/test_store.py
git commit -m "feat: ChromaDB store with add, search, get, count, chunk_text"
```

---

### Task 4: Pydantic Models

**Files:**
- Create: `models.py`

- [ ] **Step 1: Create `models.py`**

These models define tool input/output schemas. No complex logic to test — Pydantic handles validation.

```python
# models.py
from __future__ import annotations

from pydantic import BaseModel, Field


class DecisionResult(BaseModel):
    id: str
    text: str
    decision_date: str = ""
    decision_number: str = ""
    category: str = ""
    institution: str = ""
    source_url: str = ""
    score: float = 0.0


class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    doc_id: str
    title: str = ""
    doc_type: str = ""
    publish_date: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    source_url: str = ""
    score: float = 0.0


class DocumentResult(BaseModel):
    doc_id: str
    title: str = ""
    doc_type: str = ""
    publish_date: str = ""
    source_url: str = ""
    full_text: str = ""
    total_chunks: int = 0


class SyncResult(BaseModel):
    upserted: int = 0
    collection: str = ""
```

- [ ] **Step 2: Commit**

```bash
git add models.py
git commit -m "feat: Pydantic models for tool I/O"
```

---

### Task 5: Scraper

**Files:**
- Create: `scraper.py`
- Create: `tests/test_scraper.py`
- Create: `tests/fixtures/decisions_page.html`
- Create: `tests/fixtures/accordion_page.html`

The scraper reuses the parsing logic from the existing `client.py`. The key URLs and patterns are:

- **Decision pages** (IDs 55, 56): Links matching `/Mevzuat/DokumanGetir/\d+`, text format `(DD.MM.YYYY - NUMBER) Title`
- **Accordion pages** (IDs 50, 51): `div.card` > `h5` header + `div.card-body` with links
- **Flat pages** (IDs 49, 52, 54, 58, 63): Simple link lists
- **Document download**: `https://www.bddk.org.tr/Mevzuat/DokumanGetir/{id}`

- [ ] **Step 1: Create test HTML fixtures**

```html
<!-- tests/fixtures/decisions_page.html -->
<html><body>
<div class="content">
  <a href="/Mevzuat/DokumanGetir/310">(15.01.2025 - 10987) Faiz oranları kararı</a>
  <a href="/Mevzuat/DokumanGetir/311">(20.02.2025 - 10988) Kredi kartı taksitlendirme</a>
  <a href="/Mevzuat/Detay/999">Bu link atlanmalı</a>
</div>
</body></html>
```

```html
<!-- tests/fixtures/accordion_page.html -->
<html><body>
<div class="card">
  <h5>Yönetmelikler (3)</h5>
  <div class="card-body">
    <a href="/Mevzuat/DokumanGetir/100">Likidite Yönetmeliği</a>
    <a href="/Mevzuat/DokumanGetir/101">Sermaye Yeterliliği Yönetmeliği</a>
  </div>
</div>
<div class="card">
  <h5>Tebliğler (1)</h5>
  <div class="card-body">
    <a href="/Mevzuat/DokumanGetir/200">Açıklama Tebliği</a>
  </div>
</div>
</body></html>
```

- [ ] **Step 2: Write failing scraper tests**

```python
# tests/test_scraper.py
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from scraper import parse_decision_page, parse_accordion_page, scrape_decisions, scrape_document_content

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_decision_page():
    html = (FIXTURES / "decisions_page.html").read_text()
    decisions = parse_decision_page(html)
    assert len(decisions) == 2
    assert decisions[0]["id"] == "310"
    assert decisions[0]["metadata"]["decision_date"] == "15.01.2025"
    assert decisions[0]["metadata"]["decision_number"] == "10987"
    assert "Faiz oranları" in decisions[0]["document"]
    assert decisions[0]["metadata"]["category"] == "Kurul Kararı"


def test_parse_decision_page_skips_detay_links():
    html = (FIXTURES / "decisions_page.html").read_text()
    decisions = parse_decision_page(html)
    ids = [d["id"] for d in decisions]
    assert "999" not in ids


def test_parse_accordion_page():
    html = (FIXTURES / "accordion_page.html").read_text()
    docs = parse_accordion_page(html)
    assert len(docs) == 3
    categories = [d["category"] for d in docs]
    assert "Yönetmelik" in categories
    assert "Tebliğ" in categories


@pytest.mark.asyncio
async def test_scrape_decisions_calls_correct_url():
    mock_response = httpx.Response(
        200,
        text=(FIXTURES / "decisions_page.html").read_text(),
        request=httpx.Request("GET", "https://www.bddk.org.tr/Mevzuat/Liste/55"),
    )
    with patch("scraper.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        decisions = await scrape_decisions(page_ids=[55])
        mock_client.get.assert_called_once()
        call_url = str(mock_client.get.call_args[0][0])
        assert "55" in call_url
        assert len(decisions) == 2
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_scraper.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scraper'`

- [ ] **Step 4: Implement `scraper.py`**

```python
# scraper.py
from __future__ import annotations

import asyncio
import re

import httpx
from bs4 import BeautifulSoup

BASE_URL = "https://www.bddk.org.tr"

# Page IDs by type
DECISION_PAGES = [55, 56]
ACCORDION_PAGES = [50, 51]
FLAT_PAGES = [49, 52, 54, 58, 63]

FLAT_PAGE_CATEGORIES = {
    49: "Kanun",
    52: "Finansal Kiralama ve Faktoring",
    54: "BDDK Düzenlemesi",
    58: "Düzenleme Taslağı",
    63: "Mülga Düzenleme",
}

ACCORDION_CATEGORY_MAP = {
    "Yönetmelikler": "Yönetmelik",
    "Tebliğler": "Tebliğ",
    "Genelgeler": "Genelge",
    "Yönetmelik": "Yönetmelik",
    "Tebliğ": "Tebliğ",
    "Genelge": "Genelge",
}

_DECISION_PATTERN = re.compile(r"\((\d{2}[./]\d{2}[./]\d{4})\s*[-\u2013\u2014]\s*(\d+)\)\s*(.*)")
_DOC_ID_PATTERN = re.compile(r"/Mevzuat/DokumanGetir/(\d+)")
_HEADER_COUNT_PATTERN = re.compile(r"\s*\(\d+\)\s*$")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "tr-TR,tr;q=0.9",
}


def parse_decision_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    decisions = []
    seen = set()
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/Mevzuat/Detay/" in href:
            continue
        match = _DOC_ID_PATTERN.search(href)
        if not match:
            continue
        doc_id = match.group(1)
        if doc_id in seen:
            continue
        seen.add(doc_id)

        text = link.get_text(strip=True)
        decision_date = ""
        decision_number = ""
        title = text

        m = _DECISION_PATTERN.match(text)
        if m:
            decision_date = m.group(1).replace("/", ".")
            decision_number = m.group(2)
            title = m.group(3).strip()

        decisions.append({
            "id": doc_id,
            "document": title,
            "metadata": {
                "decision_date": decision_date,
                "decision_number": decision_number,
                "category": "Kurul Kararı",
                "institution": "",
                "source_url": f"{BASE_URL}/Mevzuat/DokumanGetir/{doc_id}",
            },
        })
    return decisions


def parse_accordion_page(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    docs = []
    seen = set()
    for card in soup.find_all("div", class_="card"):
        header = card.find("h5")
        if not header:
            continue
        header_text = _HEADER_COUNT_PATTERN.sub("", header.get_text(strip=True))
        category = ACCORDION_CATEGORY_MAP.get(header_text, header_text)

        body = card.find("div", class_="card-body") or card.find("div", class_="collapse")
        if not body:
            continue

        for link in body.find_all("a", href=True):
            href = link["href"]
            match = _DOC_ID_PATTERN.search(href)
            if not match:
                continue
            doc_id = match.group(1)
            if doc_id in seen:
                continue
            seen.add(doc_id)
            title = link.get_text(strip=True)
            docs.append({
                "id": doc_id,
                "title": title,
                "category": category,
                "source_url": f"{BASE_URL}/Mevzuat/DokumanGetir/{doc_id}",
            })
    return docs


def parse_flat_page(html: str, category: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    docs = []
    seen = set()
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "/Mevzuat/Detay/" in href:
            continue
        match = _DOC_ID_PATTERN.search(href)
        if not match:
            continue
        doc_id = match.group(1)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        title = link.get_text(strip=True)
        if not title:
            continue
        docs.append({
            "id": doc_id,
            "title": title,
            "category": category,
            "source_url": f"{BASE_URL}/Mevzuat/DokumanGetir/{doc_id}",
        })
    return docs


async def scrape_decisions(
    page_ids: list[int] | None = None,
    scrape_delay: float = 1.0,
) -> list[dict]:
    page_ids = page_ids or DECISION_PAGES
    all_decisions = []
    async with httpx.AsyncClient(headers=HEADERS, timeout=60.0) as client:
        for page_id in page_ids:
            resp = await client.get(f"{BASE_URL}/Mevzuat/Liste/{page_id}")
            resp.raise_for_status()
            all_decisions.extend(parse_decision_page(resp.text))
            await asyncio.sleep(scrape_delay)
    return all_decisions


async def scrape_documents(
    page_ids: list[int] | None = None,
    scrape_delay: float = 1.0,
) -> list[dict]:
    all_docs = []
    accordion_ids = [p for p in (page_ids or ACCORDION_PAGES) if p in ACCORDION_PAGES]
    flat_ids = [p for p in (page_ids or FLAT_PAGES) if p in FLAT_PAGES]

    async with httpx.AsyncClient(headers=HEADERS, timeout=60.0) as client:
        for page_id in accordion_ids:
            resp = await client.get(f"{BASE_URL}/Mevzuat/Liste/{page_id}")
            resp.raise_for_status()
            all_docs.extend(parse_accordion_page(resp.text))
            await asyncio.sleep(scrape_delay)

        for page_id in flat_ids:
            resp = await client.get(f"{BASE_URL}/Mevzuat/Liste/{page_id}")
            resp.raise_for_status()
            category = FLAT_PAGE_CATEGORIES.get(page_id, "Diğer")
            all_docs.extend(parse_flat_page(resp.text, category))
            await asyncio.sleep(scrape_delay)
    return all_docs


async def scrape_document_content(doc_id: str, scrape_delay: float = 1.0) -> str:
    url = f"{BASE_URL}/Mevzuat/DokumanGetir/{doc_id}"
    async with httpx.AsyncClient(headers=HEADERS, timeout=60.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "pdf" in content_type:
            return ""  # PDF extraction out of scope for core; return empty
        soup = BeautifulSoup(resp.text, "html.parser")
        return soup.get_text(separator="\n", strip=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_scraper.py -v`
Expected: 4 PASSED

- [ ] **Step 6: Commit**

```bash
git add scraper.py tests/test_scraper.py tests/fixtures/
git commit -m "feat: BDDK website scraper with decision and document parsing"
```

---

### Task 6: Seed Data Loading

**Files:**
- Create: `seed.py`
- Create: `tests/test_seed.py`

The existing seed files use the old schema. The seed loader must transform them into the format expected by `store.add_decisions()` and `store.add_document_chunks()`.

**Seed data schemas (from existing files):**

`decision_cache.json`:
```json
{"document_id": "310", "title": "...", "content": "...", "decision_date": "...", "decision_number": "...", "category": "...", "source_url": "..."}
```

`chunks.json`:
```json
{"doc_id": "1225", "chunk_index": 0, "title": "...", "category": "...", "decision_date": "", "decision_number": "", "source_url": "...", "total_chunks": 5, "total_pages": 1, "content_hash": "...", "chunk_text": "..."}
```

- [ ] **Step 1: Write failing test**

```python
# tests/test_seed.py
import json
from pathlib import Path

import pytest

from seed import load_seed_data


@pytest.fixture
def seed_dir(tmp_path):
    decisions = [
        {
            "document_id": "310",
            "title": "Faiz kararı",
            "content": "Faiz oranları hakkında karar",
            "decision_date": "15.01.2025",
            "decision_number": "10987",
            "category": "Kurul Kararı",
            "source_url": "https://www.bddk.org.tr/Mevzuat/DokumanGetir/310",
        },
        {
            "document_id": "311",
            "title": "Taksit kararı",
            "content": "Kredi kartı taksitlendirme",
            "decision_date": "20.02.2025",
            "decision_number": "10988",
            "category": "Kurul Kararı",
            "source_url": "https://www.bddk.org.tr/Mevzuat/DokumanGetir/311",
        },
    ]
    chunks = [
        {
            "doc_id": "100",
            "chunk_index": 0,
            "title": "Likidite Yönetmeliği",
            "category": "Yönetmelik",
            "decision_date": "",
            "decision_number": "",
            "source_url": "https://www.bddk.org.tr/Mevzuat/DokumanGetir/100",
            "total_chunks": 2,
            "total_pages": 1,
            "content_hash": "abc123",
            "chunk_text": "Likidite karşılama oranı hesaplaması",
        },
        {
            "doc_id": "100",
            "chunk_index": 1,
            "title": "Likidite Yönetmeliği",
            "category": "Yönetmelik",
            "decision_date": "",
            "decision_number": "",
            "source_url": "https://www.bddk.org.tr/Mevzuat/DokumanGetir/100",
            "total_chunks": 2,
            "total_pages": 1,
            "content_hash": "def456",
            "chunk_text": "Net istikrarlı fonlama oranı",
        },
    ]
    (tmp_path / "decision_cache.json").write_text(json.dumps(decisions, ensure_ascii=False))
    (tmp_path / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    return tmp_path


def test_load_seed_data(store, seed_dir):
    result = load_seed_data(str(seed_dir), store)
    assert result["decisions"] == 2
    assert result["chunks"] == 2
    assert store.count("decisions") == 2
    assert store.count("documents") == 2


def test_load_seed_data_skips_if_not_empty(store, seed_dir):
    load_seed_data(str(seed_dir), store)
    result = load_seed_data(str(seed_dir), store)
    assert result["skipped"] is True


def test_load_seed_data_missing_files(store, tmp_path):
    result = load_seed_data(str(tmp_path), store)
    assert result["decisions"] == 0
    assert result["chunks"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_seed.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'seed'`

- [ ] **Step 3: Implement `seed.py`**

```python
# seed.py
from __future__ import annotations

import json
from pathlib import Path

from store import Store


def load_seed_data(seed_dir: str, store: Store) -> dict:
    """Load seed JSON files into ChromaDB. Skips if collections already have data."""
    result = {"decisions": 0, "chunks": 0, "skipped": False}

    if store.count("decisions") > 0 or store.count("documents") > 0:
        result["skipped"] = True
        return result

    seed_path = Path(seed_dir)

    # Load decisions
    decisions_file = seed_path / "decision_cache.json"
    if decisions_file.exists():
        raw_decisions = json.loads(decisions_file.read_text(encoding="utf-8"))
        decisions = [
            {
                "id": d["document_id"],
                "document": d.get("content") or d.get("title", ""),
                "metadata": {
                    "decision_date": d.get("decision_date", ""),
                    "decision_number": d.get("decision_number", ""),
                    "category": d.get("category", ""),
                    "institution": "",
                    "source_url": d.get("source_url", ""),
                },
            }
            for d in raw_decisions
        ]
        # ChromaDB has batch size limits; upsert in chunks of 5000
        for i in range(0, len(decisions), 5000):
            store.add_decisions(decisions[i : i + 5000])
        result["decisions"] = len(decisions)

    # Load document chunks
    chunks_file = seed_path / "chunks.json"
    if chunks_file.exists():
        raw_chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
        chunks = [
            {
                "id": f"{c['doc_id']}_chunk_{c['chunk_index']}",
                "document": c["chunk_text"],
                "metadata": {
                    "doc_id": c["doc_id"],
                    "title": c.get("title", ""),
                    "doc_type": c.get("category", ""),
                    "publish_date": c.get("decision_date", ""),
                    "chunk_index": c["chunk_index"],
                    "total_chunks": c.get("total_chunks", 1),
                    "source_url": c.get("source_url", ""),
                },
            }
            for c in raw_chunks
        ]
        for i in range(0, len(chunks), 5000):
            store.add_document_chunks(chunks[i : i + 5000])
        result["chunks"] = len(chunks)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_seed.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add seed.py tests/test_seed.py
git commit -m "feat: seed data loader (transforms old JSON schema to ChromaDB format)"
```

---

### Task 7: MCP Server and Tools

**Files:**
- Create: `server.py`
- Create: `tests/test_tools.py`

- [ ] **Step 1: Write failing tool tests**

```python
# tests/test_tools.py
import json

import pytest
from fastmcp import Client

from conftest import SAMPLE_DECISIONS, SAMPLE_CHUNKS


@pytest.fixture
def loaded_store(store):
    store.add_decisions(SAMPLE_DECISIONS)
    store.add_document_chunks(SAMPLE_CHUNKS)
    return store


@pytest.fixture
def mcp_server(loaded_store):
    from server import create_server
    return create_server(loaded_store)


@pytest.mark.asyncio
async def test_search_decisions(mcp_server):
    async with Client(mcp_server) as client:
        result = await client.call_tool("search_decisions", {"query": "faiz oranları"})
        text = result[0].text
        data = json.loads(text)
        assert len(data) > 0
        assert any("310" == d["id"] for d in data)


@pytest.mark.asyncio
async def test_search_decisions_with_filter(mcp_server):
    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "search_decisions",
            {"query": "karar", "institution": "Garanti BBVA"},
        )
        text = result[0].text
        data = json.loads(text)
        assert all(d["institution"] == "Garanti BBVA" for d in data)


@pytest.mark.asyncio
async def test_search_documents(mcp_server):
    async with Client(mcp_server) as client:
        result = await client.call_tool("search_documents", {"query": "likidite oranı"})
        text = result[0].text
        data = json.loads(text)
        assert len(data) > 0


@pytest.mark.asyncio
async def test_get_decision(mcp_server):
    async with Client(mcp_server) as client:
        result = await client.call_tool("get_decision", {"decision_id": "310"})
        text = result[0].text
        data = json.loads(text)
        assert data["id"] == "310"
        assert "faiz" in data["text"].lower()


@pytest.mark.asyncio
async def test_get_decision_not_found(mcp_server):
    async with Client(mcp_server) as client:
        result = await client.call_tool("get_decision", {"decision_id": "99999"})
        text = result[0].text
        data = json.loads(text)
        assert data.get("error") is not None


@pytest.mark.asyncio
async def test_get_document(mcp_server):
    async with Client(mcp_server) as client:
        result = await client.call_tool("get_document", {"doc_id": "doc1"})
        text = result[0].text
        data = json.loads(text)
        assert data["doc_id"] == "doc1"
        assert data["total_chunks"] == 2
        assert "likidite" in data["full_text"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools.py -v`
Expected: FAIL — `ImportError: cannot import name 'create_server' from 'server'`

- [ ] **Step 3: Implement `server.py`**

```python
# server.py
from __future__ import annotations

import json
import sys

from fastmcp import FastMCP

from config import Config
from embeddings import E5EmbeddingFunction
from models import DecisionResult, ChunkResult, DocumentResult, SyncResult
from store import Store


def create_server(store: Store) -> FastMCP:
    mcp = FastMCP(
        "BDDK MCP Server",
        instructions=(
            "Search and retrieve BDDK (Turkish Banking Regulation) decisions and regulatory documents.\n\n"
            "GROUNDING RULES:\n"
            "1. ONLY use information returned by tool calls. Never supplement with your own knowledge.\n"
            "2. If a search returns no results, say so explicitly. Do NOT guess or invent decisions.\n"
            "3. Always include id, decision_date, and decision_number when available.\n"
            "4. Never fabricate karar numarası, tarih, or legal conclusions.\n"
            "5. Quote only text that appears verbatim in tool output."
        ),
    )

    @mcp.tool()
    def search_decisions(
        query: str,
        date_from: str = "",
        date_to: str = "",
        institution: str = "",
        category: str = "",
        limit: int = 10,
    ) -> str:
        """Search BDDK decisions by query text with optional filters."""
        filters = {}
        if institution:
            filters["institution"] = institution
        if category:
            filters["category"] = category

        results = store.search("decisions", query, n=limit, filters=filters or None)

        decisions = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i] if results.get("distances") else 0
            score = 1 - distance  # cosine distance to similarity

            # Date range filtering (ChromaDB where clauses don't support range ops on strings)
            if date_from or date_to:
                d = meta.get("decision_date", "")
                if d:
                    # Convert DD.MM.YYYY to comparable format
                    parts = d.split(".")
                    if len(parts) == 3:
                        sortable = f"{parts[2]}{parts[1]}{parts[0]}"
                        if date_from:
                            from_parts = date_from.split(".")
                            if len(from_parts) == 3 and sortable < f"{from_parts[2]}{from_parts[1]}{from_parts[0]}":
                                continue
                        if date_to:
                            to_parts = date_to.split(".")
                            if len(to_parts) == 3 and sortable > f"{to_parts[2]}{to_parts[1]}{to_parts[0]}":
                                continue

            decisions.append(DecisionResult(
                id=doc_id,
                text=results["documents"][0][i],
                decision_date=meta.get("decision_date", ""),
                decision_number=meta.get("decision_number", ""),
                category=meta.get("category", ""),
                institution=meta.get("institution", ""),
                source_url=meta.get("source_url", ""),
                score=round(score, 4),
            ))
        return json.dumps([d.model_dump() for d in decisions], ensure_ascii=False)

    @mcp.tool()
    def search_documents(
        query: str,
        doc_type: str = "",
        limit: int = 10,
    ) -> str:
        """Search regulatory documents by query text with optional doc_type filter."""
        filters = {}
        if doc_type:
            filters["doc_type"] = doc_type

        results = store.search("documents", query, n=limit, filters=filters or None)

        chunks = []
        for i, chunk_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i] if results.get("distances") else 0
            chunks.append(ChunkResult(
                chunk_id=chunk_id,
                text=results["documents"][0][i],
                doc_id=meta.get("doc_id", ""),
                title=meta.get("title", ""),
                doc_type=meta.get("doc_type", ""),
                publish_date=meta.get("publish_date", ""),
                chunk_index=meta.get("chunk_index", 0),
                total_chunks=meta.get("total_chunks", 0),
                source_url=meta.get("source_url", ""),
                score=round(1 - distance, 4),
            ))
        return json.dumps([c.model_dump() for c in chunks], ensure_ascii=False)

    @mcp.tool()
    def get_decision(decision_id: str) -> str:
        """Retrieve a single BDDK decision by ID."""
        result = store.get("decisions", [decision_id])
        if not result["ids"]:
            return json.dumps({"error": f"Decision {decision_id} not found"})
        meta = result["metadatas"][0]
        return json.dumps(
            DecisionResult(
                id=decision_id,
                text=result["documents"][0],
                decision_date=meta.get("decision_date", ""),
                decision_number=meta.get("decision_number", ""),
                category=meta.get("category", ""),
                institution=meta.get("institution", ""),
                source_url=meta.get("source_url", ""),
            ).model_dump(),
            ensure_ascii=False,
        )

    @mcp.tool()
    def get_document(doc_id: str) -> str:
        """Retrieve a full regulatory document by ID (all chunks reassembled)."""
        # Get all chunks for this document
        results = store.documents.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
        )
        if not results["ids"]:
            return json.dumps({"error": f"Document {doc_id} not found"})

        # Sort by chunk_index
        indexed = sorted(
            zip(results["ids"], results["documents"], results["metadatas"]),
            key=lambda x: x[2].get("chunk_index", 0),
        )

        full_text = "\n".join(doc for _, doc, _ in indexed)
        meta = indexed[0][2]
        return json.dumps(
            DocumentResult(
                doc_id=doc_id,
                title=meta.get("title", ""),
                doc_type=meta.get("doc_type", ""),
                publish_date=meta.get("publish_date", ""),
                source_url=meta.get("source_url", ""),
                full_text=full_text,
                total_chunks=len(indexed),
            ).model_dump(),
            ensure_ascii=False,
        )

    @mcp.tool()
    async def sync_decisions(pages: int = 1) -> str:
        """Scrape latest decisions from BDDK website and upsert into ChromaDB."""
        from scraper import scrape_decisions as _scrape

        decisions = await _scrape(page_ids=[55, 56][:pages])
        store.add_decisions(decisions)
        return json.dumps(SyncResult(upserted=len(decisions), collection="decisions").model_dump())

    @mcp.tool()
    async def sync_documents(doc_type: str = "") -> str:
        """Download and process new/updated regulatory documents from BDDK."""
        from scraper import scrape_documents as _scrape, scrape_document_content

        docs = await _scrape()
        if doc_type:
            docs = [d for d in docs if d.get("category", "").lower() == doc_type.lower()]

        chunks_added = 0
        for doc in docs:
            content = await scrape_document_content(doc["id"])
            if not content:
                continue
            text_chunks = store.chunk_text(
                content,
                chunk_size=store._config.chunk_size,
                overlap=store._config.chunk_overlap,
            )
            chunk_records = [
                {
                    "id": f"{doc['id']}_chunk_{i}",
                    "document": chunk,
                    "metadata": {
                        "doc_id": doc["id"],
                        "title": doc.get("title", ""),
                        "doc_type": doc.get("category", ""),
                        "publish_date": "",
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                        "source_url": doc.get("source_url", ""),
                    },
                }
                for i, chunk in enumerate(text_chunks)
            ]
            store.add_document_chunks(chunk_records)
            chunks_added += len(chunk_records)

        return json.dumps(SyncResult(upserted=chunks_added, collection="documents").model_dump())

    return mcp


def main() -> None:
    config = Config()
    ef = E5EmbeddingFunction(config.embedding_model)
    store = Store(config, ef)

    # Seed if empty
    if store.count("decisions") == 0 and store.count("documents") == 0:
        from seed import load_seed_data
        result = load_seed_data(config.seed_dir, store)
        if not result["skipped"]:
            print(
                f"Seeded: {result['decisions']} decisions, {result['chunks']} chunks",
                file=sys.stderr,
            )

    mcp = create_server(store)

    if config.mcp_transport == "streamable-http":
        mcp.run(transport="streamable-http", host=config.mcp_host, port=config.mcp_port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools.py -v`
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_tools.py
git commit -m "feat: MCP server with 6 core tools (search, get, sync)"
```

---

### Task 8: Integration Test and Cleanup

**Files:**
- Modify: `.gitignore`
- Modify: `CLAUDE.md`
- Modify: `.mcp.json`

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 2: Run linting**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: No issues. If issues found, fix with `uv run ruff check --fix . && uv run ruff format .`

- [ ] **Step 3: Update `.gitignore`**

Add `chroma_data/` to `.gitignore`:

```
chroma_data/
```

- [ ] **Step 4: Update `CLAUDE.md`**

Replace the entire file with:

```markdown
# BDDK MCP Server

MCP server for Turkish banking regulatory intelligence (BDDK) — search decisions, regulations, and documents. ChromaDB backend with offline-first embeddings.

## Commands

\`\`\`bash
uv sync --dev                              # Install dependencies
uv run python server.py                    # Run MCP server
uv run pytest tests/ -v --tb=short         # Run all tests
uv run ruff check .                        # Lint
uv run ruff format .                       # Format
\`\`\`

## Architecture

- **Entry point**: `server.py` — FastMCP server with 6 core tools
- **Tools**: `search_decisions`, `search_documents`, `get_decision`, `get_document`, `sync_decisions`, `sync_documents`
- **Core modules**:
  - `store.py` — ChromaDB wrapper (add, search, get, count, chunk_text)
  - `embeddings.py` — Custom E5 embedding function for ChromaDB
  - `scraper.py` — BDDK website scraping (httpx, BeautifulSoup)
  - `seed.py` — Load seed JSON files into ChromaDB
  - `config.py` — All configuration via env vars
  - `models.py` — Pydantic schemas

## Conventions

- Python 3.12+
- Pydantic models for tool output schemas
- ChromaDB embedded mode with persistent storage in `chroma_data/`
- Embedding model: `intfloat/multilingual-e5-base` (offline-first)
- Tests use in-memory ChromaDB (never mocked)
- Ruff for linting and formatting (line length 120)
- Config via `BDDK_*` env vars and `MCP_*` env vars

## Important Rules

- Never hardcode paths — use config env vars
- Embedding model loads once at startup, reused by ChromaDB collections
- Seed data loads automatically on first run if collections are empty
- `chroma_data/` is gitignored — seed from `seed_data/` for fresh installs
```

- [ ] **Step 5: Update `.mcp.json`**

```json
{
  "mcpServers": {
    "bddk": {
      "command": "uv",
      "args": ["run", "--directory", "/home/cagatay/bddk-mcp", "--python", "3.12", "python", "server.py"],
      "env": {
        "MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

- [ ] **Step 6: Smoke test the server**

Run: `cd /home/cagatay/bddk-mcp && echo '{"jsonrpc":"2.0","method":"initialize","params":{"capabilities":{}},"id":1}' | uv run python server.py`
Expected: JSON-RPC response with server capabilities and tool list

- [ ] **Step 7: Commit**

```bash
git add .gitignore CLAUDE.md .mcp.json
git commit -m "chore: update project config for ChromaDB rebuild"
```

---

### Task 9: Final Verification

- [ ] **Step 1: Run full test suite one more time**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 2: Verify file structure**

Run: `ls -la *.py tests/ seed_data/`
Expected:
```
config.py
embeddings.py
models.py
scraper.py
seed.py
server.py
store.py
tests/conftest.py
tests/test_config.py
tests/test_embeddings.py
tests/test_store.py
tests/test_scraper.py
tests/test_seed.py
tests/test_tools.py
tests/fixtures/decisions_page.html
tests/fixtures/accordion_page.html
seed_data/decision_cache.json
seed_data/documents.json
seed_data/chunks.json
```

- [ ] **Step 3: Verify no old files remain**

Run: `ls client.py doc_store.py vector_store.py deps.py tools/ 2>/dev/null && echo "OLD FILES FOUND" || echo "Clean"`
Expected: `Clean`
