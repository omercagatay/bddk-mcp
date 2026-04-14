# Foundation First: Making bddk-mcp Better

**Date:** 2026-04-14
**Status:** Approved
**Goal:** Improve search quality, sync reliability, and data coverage for internal audit workflows.

## Motivation

bddk-mcp is a mature MCP server (v5.0.0, 21 tools, 238 tests) for Turkish banking regulatory intelligence. The system works but three areas need improvement:

1. **Search doesn't feel smart enough** — a mix of morphology gaps, structure-blind chunking, and no query understanding
2. **Sync failures erode trust** — error pages sneak through, failed documents stay failed, silently-updated regulations go stale
3. **Coverage gaps** — no TCMB statistical data, no cross-reference graph between regulations

The approach is **foundation first**: fix data quality (Phase 1), then upgrade search intelligence on clean data (Phase 2), then expand coverage (Phase 3).

---

## Phase 1: Bulletproof Sync

### 1A. Post-Extraction Content Validator

**Problem:** Extracted content is stored without quality checks. A document that extracts to 50 characters of garbage or an HTML navigation page gets stored and embedded as if it were a real regulation.

**Where it fits:** New validation step in `doc_sync.py` between extraction and storage (between `self._extract()` and `self._store.store_document()`).

**Validation checks:**

| Check | Rule | Rationale |
|-------|------|-----------|
| Minimum length | Content >= 200 chars (configurable via `BDDK_MIN_CONTENT_LENGTH`) | Real regulations are never shorter than this |
| Turkish character ratio | >= 5% of alpha chars are Turkish-specific (ş, ç, ğ, ü, ö, ı, İ, Ş, Ç, Ğ, Ü, Ö) | Catches cases where HTML/JS/navigation was extracted instead of Turkish legal text |
| Structure signal | At least one of: "Madde", "madde", "MADDE", "Fıkra", "Karar", section headers, numbered lists | Helps distinguish real regulatory content from boilerplate. This is a soft check — failure logs a warning but doesn't reject the document, since not all document types (e.g., announcements) have article structure |
| Duplicate detection | SHA-256 hash of content compared against existing `content_hash` in `document_chunks` | If identical, skip re-embedding (save compute). If different, re-embed (content was updated) |

**Implementation:**

```python
# New in doc_sync.py
class ContentValidation(BaseModel):
    valid: bool
    warnings: list[str] = []
    content_hash: str = ""

def _validate_content(content: str, doc_id: str) -> ContentValidation:
    """Validate extracted content before storage."""
    warnings = []
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    # Minimum length
    if len(content) < MIN_CONTENT_LENGTH:
        return ContentValidation(valid=False, warnings=[f"Too short: {len(content)} chars"])

    # Turkish character ratio
    alpha_chars = [c for c in content if c.isalpha()]
    if alpha_chars:
        turkish_chars = sum(1 for c in alpha_chars if c in "şçğüöıİŞÇĞÜÖ")
        ratio = turkish_chars / len(alpha_chars)
        if ratio < 0.05:
            return ContentValidation(valid=False, warnings=[f"Low Turkish char ratio: {ratio:.2%}"])

    # Structure signal (soft check)
    structure_patterns = ["Madde", "madde", "MADDE", "Fıkra", "Karar", "GEÇİCİ MADDE"]
    has_structure = any(p in content for p in structure_patterns)
    if not has_structure:
        warnings.append("No article structure detected — may be a non-regulation document")

    return ContentValidation(valid=True, warnings=warnings, content_hash=content_hash)
```

**Integration point in `sync_document()`:** After `self._extract()` succeeds and before `self._store.store_document()`, call `_validate_content()`. If `valid=False`, record as sync failure with category `"validation"` and `retryable=False`. If valid with warnings, log warnings but proceed with storage.

### 1B. Automatic Retry Queue

**Problem:** `sync_failures` table records failures with a `retryable` flag, but nothing ever reads it. Failed documents stay failed until someone manually runs `--force`.

**Design:**

- New async function `retry_failed_documents()` in `doc_sync.py`
- Reads `sync_failures WHERE retryable = true AND retry_count < 4`
- Applies exponential backoff: only retry if `now - last_attempt > backoff_interval`
  - Attempt 1: after 1 hour
  - Attempt 2: after 4 hours
  - Attempt 3: after 24 hours
  - Attempt 4: after 7 days
- After 4 failures, sets `retryable = false` (requires manual review)
- Increments `retry_count` and updates `last_attempt` on each try

**Schema additions to `sync_failures`:**

```sql
ALTER TABLE sync_failures ADD COLUMN IF NOT EXISTS retry_count INTEGER DEFAULT 0;
ALTER TABLE sync_failures ADD COLUMN IF NOT EXISTS last_attempt DOUBLE PRECISION;
```

**Trigger points:**
- Runs automatically during `trigger_startup_sync` (existing tool)
- Available as standalone via new MCP tool `retry_failed_documents` in `tools/sync.py`
- Reports: how many retried, how many succeeded, how many permanently failed

### 1C. Content Freshness Tracking

**Problem:** Once a document is downloaded, it's never re-checked. If BDDK silently updates a regulation (which happens — corrections, amendments), the stale version stays in the system indefinitely.

**Design:**

The `sync_metadata` table already exists. Extend it for freshness tracking:

```sql
ALTER TABLE sync_metadata ADD COLUMN IF NOT EXISTS etag TEXT DEFAULT '';
ALTER TABLE sync_metadata ADD COLUMN IF NOT EXISTS last_modified TEXT DEFAULT '';
ALTER TABLE sync_metadata ADD COLUMN IF NOT EXISTS last_verified_at DOUBLE PRECISION;
ALTER TABLE sync_metadata ADD COLUMN IF NOT EXISTS content_hash TEXT DEFAULT '';
```

**Freshness check flow:**

1. New function `check_document_freshness(doc_ids)` in `doc_sync.py`
2. For each document, issue an HTTP HEAD request to the source URL
3. Compare ETag/Last-Modified against stored values
4. If changed (or no stored ETag): re-download, re-extract, compare content hash
5. If content hash differs: update document + re-embed chunks
6. Update `last_verified_at` regardless

**When it runs:**
- New MCP tool `check_freshness` in `tools/sync.py` — manual trigger with optional `--category` filter
- Can be integrated into a scheduled job (cron or Railway cron) for weekly freshness sweeps
- HEAD requests are cheap (~100ms each) — checking 1000 documents takes ~2 minutes with concurrency 10

**Config:**
- `BDDK_FRESHNESS_CHECK_INTERVAL` — minimum hours between freshness checks per document (default: 168 = 1 week)
- `BDDK_FRESHNESS_CONCURRENCY` — parallel HEAD requests (default: 10)

---

## Phase 2: Search Intelligence

### 2A. Turkish Morphological Analysis with Zeyrek

**Problem:** The current stemmer (`_turkish_stem` in `client.py`) strips ~40 fixed suffixes. Turkish is agglutinative — "bankalarımızdaki" has 4 suffixes that the current stemmer can't decompose. The PostgreSQL FTS uses the `simple` dictionary (no stemming at all).

**Solution:** Integrate `zeyrek`, a Python wrapper around the Zemberek morphological analyzer, for proper Turkish lemmatization.

**Dependency:** `zeyrek >= 0.1.1` (pure Python, no native deps, ~50MB model download on first use — can be pre-downloaded for offline deployment like the embedding model).

**Integration points:**

1. **Chunk pre-processing (embedding quality):**
   - Before embedding a chunk, lemmatize key terms to reduce vocabulary sparsity
   - Store both original and lemmatized text: original for display, lemmatized for embedding
   - This means the embedding captures "banka" even when the text says "bankalarımızdaki"

2. **FTS tsvector enrichment:**
   - Modify the `chunks_tsv_trigger` function to also include lemmatized forms
   - New approach: store a `lemmatized_text` column alongside `chunk_text`
   - tsvector is built from both: `to_tsvector('simple', original) || to_tsvector('simple', lemmatized)`
   - This way, both exact matches and root-form matches work

3. **Query expansion:**
   - Before search, lemmatize the query terms
   - Search with both original query AND lemmatized query
   - For FTS: `plainto_tsquery(original) || plainto_tsquery(lemmatized)`
   - For vector: embed the lemmatized query (better semantic match)

4. **Backward compatibility with keyword search in `client.py`:**
   - Replace `_turkish_stem()` with a `_turkish_lemmatize()` that uses zeyrek
   - Fall back to current suffix stripping if zeyrek fails on a word (graceful degradation)

**Schema additions:**

```sql
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS lemmatized_text TEXT DEFAULT '';
```

**Performance:** Zeyrek lemmatization is CPU-bound (~1ms per word). For a 1500-char chunk with ~200 words, that's ~200ms. Run in thread executor to avoid blocking the event loop. Batch processing during sync is fine; real-time query lemmatization is fast (5-10 words = 5-10ms).

### 2B. Structure-Aware Chunking

**Problem:** `_chunk_text()` in `vector_store.py` splits on fixed character boundaries (1500 chars, 200 overlap). A regulation's Madde 5 might be split across two chunks, and the overlap might cut through mid-sentence. This degrades embedding quality.

**Design:**

New chunking strategy with two modes:

**Mode 1: Article-boundary chunking (for regulations)**

Regex patterns to detect article boundaries:
```python
ARTICLE_PATTERNS = [
    r"(?:^|\n)\s*(?:GEÇİCİ\s+)?(?:MADDE|Madde)\s+\d+",   # Madde 1, GEÇİCİ MADDE 1
    r"(?:^|\n)\s*(?:EK\s+)?MADDE\s+\d+",                    # EK MADDE 1
    r"(?:^|\n)\s*(?:BÖLÜM|Bölüm)\s+\w+",                   # BÖLÜM headings
    r"(?:^|\n)\s*(?:KISIM|Kısım)\s+\w+",                    # KISIM headings
]
```

Logic:
1. Split document at article boundaries
2. Each article becomes one chunk (with metadata: `madde_no`, `bolum`, `kisim`)
3. If an article exceeds `chunk_size`, split at paragraph boundaries within the article
4. If a group of consecutive short articles (< 300 chars each) exists, merge them into one chunk up to `chunk_size`
5. Overlap is applied at article boundaries: include the last paragraph of the previous article as context

**Mode 2: Paragraph-boundary chunking (fallback)**

For documents without article structure (announcements, circulars, tables):
1. Split at double-newline (`\n\n`) paragraph boundaries
2. Merge consecutive short paragraphs up to `chunk_size`
3. If a single paragraph exceeds `chunk_size`, split at sentence boundaries (`. `)
4. Apply standard overlap at paragraph boundaries

**Chunk metadata:**

Each chunk carries structured metadata stored in new columns:

```sql
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS madde_no TEXT DEFAULT '';
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS bolum TEXT DEFAULT '';
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS chunk_type TEXT DEFAULT 'character';
-- chunk_type: 'article', 'paragraph', 'character' (legacy)
```

**Migration:** Existing chunks keep `chunk_type = 'character'`. A re-chunking migration tool processes all documents through the new chunker and updates chunks. This can run as a one-time migration or incrementally during sync.

**Where it lives:** New function `_chunk_document()` in `vector_store.py` (replaces `_chunk_text()`). The function signature changes from `_chunk_text(text, chunk_size, overlap)` to `_chunk_document(text, chunk_size, overlap)` returning `list[ChunkResult]` where `ChunkResult` includes text + metadata.

### 2C. Query Router

**Problem:** Every query goes through the same hybrid search pipeline. But "5411 sayılı Kanun Madde 43" needs exact lookup, while "kredi riski yönetimi prensipleri" needs semantic search.

**Design:**

A lightweight heuristic router that classifies queries before search:

**Query types:**

| Type | Detection | Search strategy |
|------|-----------|----------------|
| `reference` | Regex: `\d{4,5}\s*sayılı`, `Madde\s+\d+`, document ID patterns | FTS-first, exact match on madde_no/doc_id, then hybrid as fallback |
| `date_scoped` | Regex: `\d{4}\s*yılı`, date patterns (`DD.MM.YYYY`) | Extract date filter, pass to existing search with date params |
| `concept` | Default — anything that doesn't match above patterns | Full hybrid search (current behavior, unchanged) |

**Implementation:**

```python
# New file: query_router.py
class QueryType(Enum):
    REFERENCE = "reference"
    DATE_SCOPED = "date_scoped"
    CONCEPT = "concept"

class ParsedQuery(BaseModel):
    original: str
    query_type: QueryType
    search_query: str          # cleaned query for search
    date_from: str | None = None
    date_to: str | None = None
    madde_no: str | None = None
    kanun_no: str | None = None

def parse_query(raw_query: str) -> ParsedQuery:
    """Classify and parse a search query."""
    ...
```

**Integration:** `search_document_store` tool in `tools/search.py` calls `parse_query()` before dispatching to `VectorStore.search()`. For reference queries, it first tries a direct lookup (FTS on `madde_no` + `doc_id`), and only falls back to hybrid if the direct lookup returns nothing.

**This is deliberately simple.** No ML-based intent classification — just regex patterns for well-known Turkish legal citation formats. Easy to extend by adding patterns.

---

## Phase 3: Coverage Expansion

### 3B. TCMB EVDS API Integration

**Problem:** Auditors need macroeconomic context — policy rates, exchange rates, banking sector aggregates — alongside regulatory data. Currently they look this up manually on TCMB's website.

**Design:**

**New module: `tcmb_client.py`**

- Uses TCMB EVDS REST API (public, requires free API key via `BDDK_TCMB_API_KEY` env var)
- No scraping — proper API with JSON responses
- No local storage needed — EVDS responses are fast (<500ms) and data is time-series (always queried by date range)
- In-memory LRU cache for repeated queries within a session (same pattern as existing `BddkApiClient`)

**Key data series for banking audit:**

| Series | EVDS Code | Use case |
|--------|-----------|----------|
| Policy rate | TP.PF.ON | IRRBB analysis baseline |
| Weighted avg. funding cost | TP.PF.AOFO | Funding cost trends |
| USD/TRY | TP.DK.USD.S.YTL | FX risk exposure context |
| EUR/TRY | TP.DK.EUR.S.YTL | FX risk exposure context |
| CPI (annual) | TP.FG.J0 | Real rate calculations |
| Banking sector total assets | TP.BKM.B001 | Sector-level context |
| NPL ratio | TP.BKMS.NFO | Credit risk benchmarking |

**New tools in `tools/tcmb.py`:**

1. `search_tcmb_data(keywords, date_from, date_to)` — search EVDS series by Turkish keyword, return matching series metadata + recent values
2. `get_tcmb_series(series_code, date_from, date_to, frequency)` — fetch specific time series with date range and frequency (daily/weekly/monthly)

**Config:**
- `BDDK_TCMB_API_KEY` — EVDS API key (required for TCMB tools, optional for the rest of the system)
- `BDDK_TCMB_CACHE_TTL` — LRU cache TTL in seconds (default: 3600 = 1 hour, since most series update daily)

### 3C. Cross-Reference Graph

**Problem:** Regulations frequently reference each other ("5411 sayılı Kanunun 43 üncü maddesi", "Bu Yönetmeliğin 5 inci maddesinde belirtilen..."). These connections are invisible to the current search system.

**Design:**

**Schema:**

```sql
CREATE TABLE IF NOT EXISTS document_references (
    id              SERIAL PRIMARY KEY,
    from_doc_id     TEXT NOT NULL,
    to_doc_id       TEXT,          -- NULL if we can't resolve the reference
    reference_text  TEXT NOT NULL,  -- raw text of the reference as found
    from_madde      TEXT DEFAULT '',
    to_madde        TEXT DEFAULT '',
    to_kanun_no     TEXT DEFAULT '', -- e.g., "5411" for 5411 sayılı Kanun
    to_regulation   TEXT DEFAULT '', -- e.g., "Yönetmelik", "Tebliğ"
    confidence      TEXT DEFAULT 'high',  -- high/medium/low based on parse quality
    UNIQUE(from_doc_id, reference_text)
);

CREATE INDEX IF NOT EXISTS idx_refs_from ON document_references(from_doc_id);
CREATE INDEX IF NOT EXISTS idx_refs_to ON document_references(to_doc_id);
CREATE INDEX IF NOT EXISTS idx_refs_kanun ON document_references(to_kanun_no);
```

**Reference parsing:**

During structure-aware chunking (Phase 2B), scan each chunk for cross-reference patterns:

```python
REFERENCE_PATTERNS = [
    # "5411 sayılı Kanunun 43 üncü maddesi"
    r"(\d{4,5})\s*sayılı\s+(Kanun|Yönetmelik|Tebliğ|Kararname)(?:un|ün|ın|in|nın|nin)?\s*(?:(\d+).*?maddesi)?",
    # "Bu Yönetmeliğin X inci/üncü maddesi"
    r"[Bb]u\s+(Yönetmelik|Tebliğ|Kanun)(?:in|ın|ün|un)\s+(\d+).*?maddesi",
    # "Ek-1" / "Ek Madde 1"
    r"(?:EK|Ek)\s*[-\s]?\s*(\d+|MADDE\s+\d+)",
]
```

**Resolution:** When a reference mentions a kanun_no (e.g., "5411"), try to resolve it to a `doc_id` by checking `decision_cache` for documents whose title contains that number. If unresolvable, store with `to_doc_id = NULL` and `to_kanun_no = "5411"` — the reference is still useful for search even without full resolution.

**Search integration:**

1. When `search_document_store` returns results, enrich each hit with a `related_documents` field by querying `document_references` for outgoing and incoming edges.
2. New MCP tool: `get_document_references(doc_id)` — returns all documents that reference or are referenced by the given document. Useful for compliance mapping ("what regulations depend on this one?").

---

## Implementation Order

```
Phase 1 (foundation)           Phase 2 (search)           Phase 3 (expansion)
├─ 1A Content validator         ├─ 2A Zeyrek morphology     ├─ 3B TCMB EVDS
├─ 1B Retry queue               ├─ 2B Structure chunking    └─ 3C Cross-ref graph
└─ 1C Freshness tracking        └─ 2C Query router

         ───────────────────────────────────────────────>
         Items within each phase are independent (parallelizable).
         Phase 2 should start after Phase 1 is complete.
         Phase 3 can overlap with late Phase 2 work.
```

## Dependencies

| New dependency | Version | Purpose | Size |
|---------------|---------|---------|------|
| `zeyrek` | >= 0.1.1 | Turkish morphological analysis | ~50MB model |

No other new dependencies. TCMB EVDS uses the existing `httpx` client.

## Config Additions

| Variable | Default | Phase |
|----------|---------|-------|
| `BDDK_MIN_CONTENT_LENGTH` | 200 | 1A |
| `BDDK_FRESHNESS_CHECK_INTERVAL` | 168 (hours) | 1C |
| `BDDK_FRESHNESS_CONCURRENCY` | 10 | 1C |
| `BDDK_TCMB_API_KEY` | (none) | 3B |
| `BDDK_TCMB_CACHE_TTL` | 3600 | 3B |

## Schema Additions

| Table | Column/Change | Phase |
|-------|--------------|-------|
| `sync_failures` | `retry_count INTEGER`, `last_attempt DOUBLE PRECISION` | 1B |
| `sync_metadata` | `etag TEXT`, `last_modified TEXT`, `last_verified_at DOUBLE PRECISION`, `content_hash TEXT` | 1C |
| `document_chunks` | `lemmatized_text TEXT`, `madde_no TEXT`, `bolum TEXT`, `chunk_type TEXT` | 2A, 2B |
| `document_references` | New table | 3C |

## New Files

| File | Purpose | Phase |
|------|---------|-------|
| `query_router.py` | Query classification and parsing | 2C |
| `tcmb_client.py` | TCMB EVDS API client | 3B |
| `tools/tcmb.py` | TCMB MCP tools | 3B |

## New MCP Tools

| Tool | Module | Phase |
|------|--------|-------|
| `retry_failed_documents` | `tools/sync.py` | 1B |
| `check_freshness` | `tools/sync.py` | 1C |
| `search_tcmb_data` | `tools/tcmb.py` | 3B |
| `get_tcmb_series` | `tools/tcmb.py` | 3B |
| `get_document_references` | `tools/documents.py` | 3C |
