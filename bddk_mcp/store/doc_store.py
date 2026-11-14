"""Document store over the SQLite corpus: documents, versions, sections, catalog.

Section full-text search runs on FTS5 over normalized shadow columns (see
:mod:`bddk_mcp.store.corpus_db`).  Every method is ``async`` so the tools can
keep awaiting them; the SQL itself runs on a worker thread.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import sqlite3
import time
import unicodedata
from datetime import UTC, datetime

from pydantic import BaseModel

from bddk_mcp.core.config import PAGE_SIZE
from bddk_mcp.quality.markdown_quality import (
    QualityAssessment,
    QualityFailure,
    assess_markdown_quality,
    get_configured_quality_failure,
    prepare_markdown_for_storage,
)
from bddk_mcp.store.corpus_db import CorpusDB
from bddk_mcp.store.section_index import (
    GOVDE_HEADING,
    GOVDE_SECTION_TYPE,
    TRUNCATION_NOTICE_PREFIX,
    DocumentSection,
    extract_document_sections,
    split_section_truncation_notice,
    truncation_notice_original_length,
)

logger = logging.getLogger(__name__)
_SAFE_SYNC_FAILURE_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_DOCUMENT_QUALITY_CACHE_LIMIT = 4096
_FTS_TOKEN_RE = re.compile(r"\w+")
_EXCLUDED_SECTION_TYPES = ("fikra", "bent", "govde")
_NESTED_SECTION_TYPES = ("fikra", "bent")
_ARTICLE_SECTION_TYPES = ("madde", "gecici_madde", "ek_madde")
# The provisions that can contain a fıkra/bent row: articles, and in guides and standards paragraphs and principles.
_PARENT_SECTION_TYPES = (*_ARTICLE_SECTION_TYPES, "paragraf", "ilke")
# True when the section's document version has no section other than govde/fikra/bent,
# i.e. the parser found no article structure and govde windows carry the whole body.
_STRUCTURELESS_SQL = """NOT EXISTS (
    SELECT 1 FROM document_sections AS x
    WHERE x.doc_id = s.doc_id AND x.source_content_hash = s.source_content_hash
      AND x.section_type NOT IN ('fikra', 'bent', 'govde')
)"""


def _named_sections_only_sql(types: str) -> str:
    """True when the section's document version has named sections and every one is of ``types``.

    ``types`` is a SQL list of literal section types; govde/fikra/bent are not named sections.
    """
    return f"""(EXISTS (
    SELECT 1 FROM document_sections AS y
    WHERE y.doc_id = s.doc_id AND y.source_content_hash = s.source_content_hash AND y.section_type IN ({types})
) AND NOT EXISTS (
    SELECT 1 FROM document_sections AS x
    WHERE x.doc_id = s.doc_id AND x.source_content_hash = s.source_content_hash
      AND x.section_type NOT IN ('fikra', 'bent', 'govde', {types})
))"""


# True for a govde row of a Kurul Kararı whose only named sections are annexes (ek) when the row starts
# before the first annex: the leading window is the decision text itself, not remainder.
_DECISION_WINDOW_SQL = f"""(s.section_type = 'govde' AND d.category = 'Kurul Kararı'
    AND {_named_sections_only_sql("'ek'")}
    AND s.start_char < (
        SELECT MIN(z.start_char) FROM document_sections AS z
        WHERE z.doc_id = s.doc_id AND z.source_content_hash = s.source_content_hash AND z.section_type = 'ek'
    ))"""
# True when the section's document version has named sections and they are only annexes (ek) and/or
# paragraphs (paragraf): its govde rows are body text outside those named sections, not article remainder.
_ANNEX_OR_PARAGRAPH_ONLY_SQL = _named_sections_only_sql("'ek', 'paragraf'")


# Typographic apostrophes and look-alikes that survive normalize_search_text; the heading filter reads each
# as the ASCII apostrophe.
_APOSTROPHE_VARIANTS = ("’", "‘", "ʼ", "`")


def _apostrophes_folded_sql(column: str) -> str:
    """SQL for ``column`` with every apostrophe variant replaced by the ASCII apostrophe."""
    for variant in _APOSTROPHE_VARIANTS:
        column = f"replace({column}, '{variant}', '''')"
    return column


_HEADING_NORM_FOLDED_SQL = _apostrophes_folded_sql("s.heading_norm")


def _fold_apostrophes(text: str) -> str:
    return text.translate(str.maketrans(dict.fromkeys(_APOSTROPHE_VARIANTS, "'")))


def normalize_search_text(text: str) -> str:
    """Lowercase with Turkish dotted/dotless-i rules and strip diacritics."""
    text = text.translate(str.maketrans({"ı": "i", "İ": "i"})).lower()
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def fts_match_query(query: str) -> str | None:
    """Turn free text into an FTS5 MATCH expression: every token must match.

    Tokens are quoted so FTS5 operators in user text are never interpreted.
    """
    tokens = _FTS_TOKEN_RE.findall(normalize_search_text(query))
    if not tokens:
        return None
    return " ".join(f'"{token}"' for token in tokens)


# Prefix-stem matching: a normalized token of at least _PREFIX_STEM_MIN characters loses one known
# Turkish ending (_STEM_SUFFIXES, longest first) when at least _PREFIX_STEM_MIN characters of root
# remain, and the rest is an FTS5 prefix: "denetlenmesi" -> "denet"*, which also finds "denetimi".
# A token with no such ending is not cut: "birincil" stays "birincil"*, never "birin"* (birinci).
_PREFIX_STEM_MIN = 5
# Normalized forms (diacritics stripped, ı -> i): verbal nouns, passives and participles, aorist verb
# endings ("seçilir", "oluşturulur", "hesaplanır"), plural and case endings, possessives. Reviewed list; one ending is removed, never a chain.
_STEM_SUFFIXES = tuple(
    sorted(
        {
            "lenmesi",
            "lanmasi",
            "ilmesi",
            "ilmasi",
            "ulmesi",
            "ulmasi",
            "nmesi",
            "nmasi",
            "meleri",
            "malari",
            "mesi",
            "masi",
            "me",
            "ma",
            "ilmeli",
            "ilmali",
            "ulmeli",
            "ulmali",
            "meli",
            "mali",
            "ilmis",
            "ulmis",
            "ilen",
            "ilan",
            "ulen",
            "ulan",
            "lerinin",
            "larinin",
            "lerinde",
            "larinda",
            "lerine",
            "larina",
            "lerini",
            "larini",
            "lerin",
            "larin",
            "leri",
            "lari",
            "ler",
            "lar",
            "sinin",
            "sinde",
            "sine",
            "sini",
            "si",
            "inin",
            "imi",
            "ilir",
            "ulur",
            "ir",
            "ur",
            "ar",
            "er",
        },
        key=lambda suffix: (-len(suffix), suffix),
    )
)
# Short alphabetic tokens ("on", "gün", "faiz") are matched as prefixes in the stem pass, so that
# "yüzde on" also finds "yüzde onunun"; a single letter stays exact.
_PREFIX_SHORT_MIN = 2


def _prefix_stem(token: str) -> str:
    """Return the FTS prefix of a normalized token: its root without one known ending, else the token."""
    for suffix in _STEM_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= _PREFIX_STEM_MIN:
            return token[: -len(suffix)]
    return token


def fts_prefix_stem_query(query: str) -> str | None:
    """Turn free text into an inflection-tolerant FTS5 MATCH expression.

    Tokens of five or more characters become a quoted prefix query of their root: one known
    Turkish ending is removed when at least five characters remain (``"suresinin"`` ->
    ``"sures"*``, ``"yedeklenmesi"`` -> ``"yedek"*``), otherwise the whole token is the prefix
    (``"birincil"*``). Alphabetic tokens of two to four characters become a prefix of
    themselves (``"gun"*``); numbers and single letters stay exact. Every token stays quoted,
    so FTS5 operators in user text are never interpreted. Returns None when the expression
    would equal :func:`fts_match_query`.
    """
    tokens = _FTS_TOKEN_RE.findall(normalize_search_text(query))
    terms: list[str] = []
    widened = False
    for token in tokens:
        if len(token) >= _PREFIX_STEM_MIN and not token.isdigit():
            terms.append(f'"{_prefix_stem(token)}"*')
            widened = True
        elif len(token) >= _PREFIX_SHORT_MIN and token.isalpha():
            terms.append(f'"{token}"*')
            widened = True
        else:
            terms.append(f'"{token}"')
    return " ".join(terms) if widened else None


def _safe_sync_failure_fields(error: str, category: str) -> tuple[str, str]:
    """Return bounded codes suitable for durable operational metadata."""

    safe_category = category if _SAFE_SYNC_FAILURE_TOKEN_RE.fullmatch(category) else "unknown"
    if _SAFE_SYNC_FAILURE_TOKEN_RE.fullmatch(error):
        return error, safe_category
    return f"sync_{safe_category}_failed", safe_category


# -- Pydantic models ----------------------------------------------------------


class StoredDocument(BaseModel):
    """A document stored in the corpus."""

    document_id: str
    title: str
    category: str = ""
    decision_date: str = ""
    decision_number: str = ""
    source_url: str = ""
    pdf_bytes: bytes | None = None
    markdown_content: str = ""
    content_hash: str = ""
    extraction_method: str = "markitdown"
    total_pages: int = 1
    file_size: int = 0
    downloaded_at: float | None = None
    """Carried from seed data by the build; the writer stamps the current time when absent."""
    extracted_at: float | None = None

    model_config = {"arbitrary_types_allowed": True}


# The website shows an instrument's annexes as a link and the stored text keeps only that link text, on a
# line of its own: "Ekleri için tıklayınız.", "Eki için tıklayınız", "Yönetmeliğin eklerini görmek için
# tıklayınız". Prose that merely mentions "ekleri" is not on a line of its own and never matches.
_ANNEX_PLACEHOLDER_RE = re.compile(
    r"^[ \t*_]*(?:[A-ZÇĞİÖŞÜ][a-zçğıöşü]+[ \t]+)?[Ee]k(?:i|ini|leri|lerini)[ \t]+(?:görmek[ \t]+)?için[ \t]+"
    r"tıklayınız[ \t.*_]*$",
    re.MULTILINE,
)


class DocumentPage(BaseModel):
    """A single page of a paginated document."""

    document_id: str
    title: str
    markdown_content: str
    page_number: int = 1
    total_pages: int = 1
    extraction_method: str = ""
    category: str = ""
    content_hash: str = ""
    """Hash of the whole stored body, not of this page."""
    downloaded_at: float | None = None
    starts_inside_table_row: bool = False
    """The previous page ends inside a line that starts with '|', so this page opens with the rest of that row."""
    annex_placeholder: str = ""
    """The link-text line that stands for the instrument's annexes when the stored text has no annex ('' otherwise)."""
    starts_inside_formula: bool = False
    """An odd number of $$ markers precedes this page, so it opens inside a formula."""
    ends_inside_table_row: bool = False
    """This page's last line starts with '|' and the row is cut there (not a complete row before a line break)."""
    ends_inside_formula: bool = False
    """An odd number of $$ markers lies up to the end of this page, so it ends inside a formula."""


class StoredDocumentSection(BaseModel):
    """A structural section persisted for a document."""

    doc_id: str
    section_type: str
    section_ref: str
    heading: str = ""
    start_char: int
    end_char: int
    content: str
    content_hash: str
    page_start: int | None = None
    page_end: int | None = None
    normalized_source_range: str = ""
    source_content_hash: str = ""
    document_title: str = ""
    document_source_url: str = ""
    document_category: str = ""
    document_extraction_method: str = ""
    document_decision_date: str = ""
    document_decision_number: str = ""
    document_downloaded_at: float | None = None
    """Unix time the document body was downloaded from its source, as stored."""
    document_quality: QualityAssessment | None = None
    document_structureless: bool = False
    """True when the document version has no section other than govde/fikra/bent (no article structure)."""
    document_decision_window: bool = False
    """True for a govde row of a Kurul Kararı whose only named sections are annexes, starting before the
    first annex: the decision text itself."""
    document_annex_or_paragraph_only: bool = False
    """True when the document version's only named sections are annexes (ek) and/or paragraphs (paragraf)."""
    rank: float | None = None
    """FTS match rank (higher is better). Only set by search paths; comparable
    within one query's result set, not across queries."""


def _utc_minute(timestamp: float) -> str:
    """Render a stored Unix time in UTC, independent of the host time zone."""
    return datetime.fromtimestamp(float(timestamp), tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def _content_hash(content: str) -> str:
    """SHA-256 hash of document content for change detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _section_from_row(row: sqlite3.Row, document_quality: QualityAssessment | None = None) -> StoredDocumentSection:
    keys = row.keys()
    return StoredDocumentSection(
        doc_id=row["doc_id"],
        section_type=row["section_type"],
        section_ref=row["section_ref"],
        heading=row["heading"] or "",
        start_char=row["start_char"],
        end_char=row["end_char"],
        content=row["content"] or "",
        content_hash=row["content_hash"] or "",
        page_start=row["page_start"],
        page_end=row["page_end"],
        normalized_source_range=(row["normalized_source_range"] or "") if "normalized_source_range" in keys else "",
        source_content_hash=row["source_content_hash"] or "",
        document_title=row["document_title"] or "",
        document_source_url=row["document_source_url"] or "",
        document_category=row["document_category"] or "",
        document_extraction_method=row["document_extraction_method"] or "",
        document_decision_date=(row["document_decision_date"] or "") if "document_decision_date" in keys else "",
        document_decision_number=(row["document_decision_number"] or "") if "document_decision_number" in keys else "",
        document_downloaded_at=row["document_downloaded_at"] if "document_downloaded_at" in keys else None,
        document_quality=document_quality,
        document_structureless=bool(row["document_structureless"]) if "document_structureless" in keys else False,
        document_decision_window=bool(row["document_decision_window"]) if "document_decision_window" in keys else False,
        document_annex_or_paragraph_only=(
            bool(row["document_annex_or_paragraph_only"]) if "document_annex_or_paragraph_only" in keys else False
        ),
        rank=row["rank"] if "rank" in keys else None,
    )


_SECTION_COLUMNS = (
    """
    s.doc_id, s.section_type, s.section_ref, s.heading, s.start_char, s.end_char,
    s.content, s.content_hash, s.page_start, s.page_end,
    d.content_hash AS source_content_hash, d.title AS document_title,
    d.source_url AS document_source_url, d.category AS document_category,
    d.extraction_method AS document_extraction_method,
    d.decision_date AS document_decision_date, d.decision_number AS document_decision_number,
    d.downloaded_at AS document_downloaded_at,
    """
    + _STRUCTURELESS_SQL
    + " AS document_structureless, "
    + _DECISION_WINDOW_SQL
    + " AS document_decision_window, "
    + _ANNEX_OR_PARAGRAPH_ONLY_SQL
    + """ AS document_annex_or_paragraph_only
"""
)
_SOURCE_RANGE_COLUMN = (
    ", substr(d.markdown_content, s.start_char + 1, s.end_char - s.start_char) AS normalized_source_range"
)


def _section_rows(sections: list[DocumentSection], doc_id: str, source_content_hash: str) -> list[tuple]:
    if any(getattr(section, "doc_id", doc_id) != doc_id for section in sections):
        raise ValueError("section document identity does not match the publication target")
    return [
        (
            doc_id,
            section.section_type,
            section.section_ref,
            section.heading,
            section.start_char,
            section.end_char,
            section.content,
            section.content_hash,
            section.page_start,
            section.page_end,
            source_content_hash,
            # The generic govde heading is a display placeholder, not a heading of the text: its words
            # must not make every govde row match (the stored heading column keeps it).
            "" if section.heading == GOVDE_HEADING else normalize_search_text(section.heading or ""),
            # Index the legal text only: the truncation notice of a capped section is
            # storage metadata, so its words must not make the section searchable.
            normalize_search_text(split_section_truncation_notice(section.content or "")[0]),
        )
        for section in sections
    ]


def replace_sections_sync(
    conn: sqlite3.Connection,
    doc_id: str,
    sections: list[DocumentSection],
    *,
    source_content_hash: str,
) -> int:
    """Replace a document's sections inside the caller's transaction."""
    rows = _section_rows(sections, doc_id, source_content_hash)
    conn.execute("DELETE FROM document_sections WHERE doc_id = ?", (doc_id,))
    conn.executemany(
        "INSERT INTO document_sections (doc_id, section_type, section_ref, heading, start_char, end_char, "
        "content, content_hash, page_start, page_end, source_content_hash, heading_norm, content_norm) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def upsert_document_sync(
    conn: sqlite3.Connection, doc: StoredDocument, *, now: float, prepare: bool = True
) -> tuple[str, int]:
    """Archive the previous version when content changed, then upsert the row.

    Returns the content hash and the number of sections written.  Runs inside
    the caller's transaction; ``store_document`` and the corpus build share it.
    ``prepare=False`` stores reviewed seed text verbatim so its hash is kept.
    The hash is SHA-256 of the stored text, an empty body included, so it agrees
    with the seed validator; "no content" is ``markdown_content == ''``.
    """
    if doc.markdown_content and prepare:
        doc.markdown_content = prepare_markdown_for_storage(doc.markdown_content)
    content_hash = _content_hash(doc.markdown_content)
    total_pages = max(1, math.ceil(len(doc.markdown_content) / PAGE_SIZE)) if doc.markdown_content else 1
    sections = extract_document_sections(doc.document_id, doc.markdown_content) if doc.markdown_content else []

    existing = conn.execute(
        "SELECT content_hash, markdown_content FROM documents WHERE document_id = ?", (doc.document_id,)
    ).fetchone()
    if existing and existing["markdown_content"] and existing["content_hash"] != content_hash:
        max_version = conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM document_versions WHERE document_id = ?", (doc.document_id,)
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO document_versions (document_id, version, content_hash, markdown_content, synced_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (doc.document_id, max_version + 1, existing["content_hash"], existing["markdown_content"], now),
        )
    conn.execute(
        """
        INSERT INTO documents (
            document_id, title, category, decision_date, decision_number, source_url, pdf_blob,
            markdown_content, content_hash, downloaded_at, extracted_at, extraction_method, total_pages, file_size
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(document_id) DO UPDATE SET
            title = excluded.title,
            category = excluded.category,
            decision_date = excluded.decision_date,
            decision_number = excluded.decision_number,
            source_url = excluded.source_url,
            pdf_blob = COALESCE(excluded.pdf_blob, documents.pdf_blob),
            markdown_content = excluded.markdown_content,
            content_hash = excluded.content_hash,
            downloaded_at = excluded.downloaded_at,
            extracted_at = excluded.extracted_at,
            extraction_method = excluded.extraction_method,
            total_pages = excluded.total_pages,
            file_size = excluded.file_size
        """,
        (
            doc.document_id,
            doc.title,
            doc.category,
            doc.decision_date,
            doc.decision_number,
            doc.source_url,
            doc.pdf_bytes,
            doc.markdown_content,
            content_hash,
            doc.downloaded_at if doc.downloaded_at is not None else now,
            doc.extracted_at if doc.extracted_at is not None else (now if doc.markdown_content else None),
            doc.extraction_method,
            total_pages,
            doc.file_size or (len(doc.pdf_bytes) if doc.pdf_bytes else 0),
        ),
    )
    written = replace_sections_sync(conn, doc.document_id, sections, source_content_hash=content_hash)
    return content_hash, written


# -- DocumentStore ------------------------------------------------------------


class DocumentStore:
    """Document, section, version, catalog and sync-failure access over one corpus file."""

    def __init__(self, db: CorpusDB) -> None:
        self._db = db
        self._document_quality_cache: dict[tuple[str, str, QualityFailure | None], QualityAssessment] = {}

    @property
    def db(self) -> CorpusDB:
        return self._db

    # -- documents ----------------------------------------------------------------

    async def store_document(self, doc: StoredDocument) -> None:
        """Atomically replace a document and its derived structural sections."""
        now = time.time()
        _hash, sections = await self._db.transaction(lambda conn: upsert_document_sync(conn, doc, now=now))
        logger.debug("Stored document (content_chars=%d, sections=%d)", len(doc.markdown_content), sections)

    async def get_document(self, doc_id: str) -> StoredDocument | None:
        row = await self._db.run(
            lambda conn: conn.execute("SELECT * FROM documents WHERE document_id = ?", (doc_id,)).fetchone()
        )
        if not row:
            return None
        return StoredDocument(
            document_id=row["document_id"],
            title=row["title"],
            category=row["category"] or "",
            decision_date=row["decision_date"] or "",
            decision_number=row["decision_number"] or "",
            source_url=row["source_url"] or "",
            pdf_bytes=bytes(row["pdf_blob"]) if row["pdf_blob"] is not None else None,
            markdown_content=row["markdown_content"] or "",
            content_hash=row["content_hash"] or "",
            extraction_method=row["extraction_method"] or "markitdown",
            total_pages=row["total_pages"] or 1,
            file_size=row["file_size"] or 0,
        )

    async def get_pdf_bytes(self, document_id: str) -> bytes | None:
        row = await self._db.run(
            lambda conn: conn.execute("SELECT pdf_blob FROM documents WHERE document_id = ?", (document_id,)).fetchone()
        )
        if row is None or row["pdf_blob"] is None:
            return None
        return bytes(row["pdf_blob"])

    async def get_document_quality(self, document_id: str) -> QualityAssessment | None:
        """Assess the canonical body, never a page fragment or cached PDF blob."""
        row = await self._db.run(
            lambda conn: conn.execute(
                "SELECT markdown_content FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        )
        if row is None:
            return None
        return assess_markdown_quality(row["markdown_content"] or "", document_id=document_id)

    async def get_document_page(self, doc_id: str, page: int = 1) -> DocumentPage | None:
        """Retrieve one page of a document's markdown content (PAGE_SIZE characters)."""
        row = await self._db.run(
            lambda conn: conn.execute(
                "SELECT document_id, title, markdown_content, extraction_method, category, content_hash, "
                "downloaded_at FROM documents WHERE document_id = ?",
                (doc_id,),
            ).fetchone()
        )
        if not row:
            return None
        md = row["markdown_content"] or ""
        starts_inside_table_row = starts_inside_formula = False
        ends_inside_table_row = ends_inside_formula = False
        total_pages = max(1, math.ceil(len(md) / PAGE_SIZE))
        if page < 1 or page > total_pages:
            content = f"Invalid page {page}. Document has {total_pages} page(s)."
        else:
            start = (page - 1) * PAGE_SIZE
            content = md[start : start + PAGE_SIZE]
            end = start + len(content)
            # A row that ends exactly at the page edge, with the line break opening the next page, is complete.
            if start > 0:
                previous = md[start - PAGE_SIZE : start]
                starts_inside_table_row = (
                    not previous.endswith("\n")
                    and previous.rsplit("\n", 1)[-1].startswith("|")
                    and md[start] not in "\r\n"
                )
                starts_inside_formula = md.count("$$", 0, start) % 2 == 1
            ends_inside_table_row = (
                end < len(md)
                and not content.endswith("\n")
                and content.rsplit("\n", 1)[-1].startswith("|")
                and md[end] not in "\r\n"
            )
            ends_inside_formula = md.count("$$", 0, end) % 2 == 1
        annex_placeholder = ""
        if match := _ANNEX_PLACEHOLDER_RE.search(md):
            has_annex_section = await self._db.run(
                lambda conn: conn.execute(
                    "SELECT 1 FROM document_sections WHERE doc_id = ? AND section_type = 'ek' LIMIT 1", (doc_id,)
                ).fetchone()
            )
            if has_annex_section is None:
                annex_placeholder = match.group(0).strip(" \t*_")
        return DocumentPage(
            document_id=doc_id,
            title=row["title"],
            markdown_content=content,
            page_number=page,
            total_pages=total_pages,
            extraction_method=row["extraction_method"] or "",
            category=row["category"] or "",
            content_hash=row["content_hash"] or "",
            downloaded_at=row["downloaded_at"],
            starts_inside_table_row=starts_inside_table_row,
            starts_inside_formula=starts_inside_formula,
            ends_inside_table_row=ends_inside_table_row,
            ends_inside_formula=ends_inside_formula,
            annex_placeholder=annex_placeholder,
        )

    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document with its sections, chunks, versions and failure record."""

        def _delete(conn: sqlite3.Connection) -> bool:
            deleted = conn.execute("DELETE FROM documents WHERE document_id = ?", (doc_id,)).rowcount
            for table, column in (
                ("document_sections", "doc_id"),
                ("chunks", "doc_id"),
                ("document_versions", "document_id"),
                ("sync_failures", "document_id"),
            ):
                conn.execute(f"DELETE FROM {table} WHERE {column} = ?", (doc_id,))
            return deleted > 0

        return await self._db.transaction(_delete)

    async def update_document_metadata(
        self,
        doc_id: str,
        *,
        title: str = "",
        category: str = "",
        source_url: str = "",
        decision_date: str = "",
        decision_number: str = "",
    ) -> list[str]:
        """Bring the catalog fields of a stored document up to date without touching its body.

        Only non-empty values that differ from the stored row are written (empty arguments never blank
        stored metadata). Body, content hash, sections, chunks and download date stay as they are. Returns
        the names of the columns that changed; an unknown document changes nothing.
        """
        wanted = {
            "title": title,
            "category": category,
            "source_url": source_url,
            "decision_date": decision_date,
            "decision_number": decision_number,
        }

        def _update(conn: sqlite3.Connection) -> list[str]:
            row = conn.execute(
                "SELECT title, category, source_url, decision_date, decision_number FROM documents "
                "WHERE document_id = ?",
                (doc_id,),
            ).fetchone()
            if row is None:
                return []
            changed = [column for column, value in wanted.items() if value and (row[column] or "") != value]
            if changed:
                assignments = ", ".join(f"{column} = ?" for column in changed)
                conn.execute(
                    f"UPDATE documents SET {assignments} WHERE document_id = ?",  # noqa: S608 - fixed column names
                    (*(wanted[column] for column in changed), doc_id),
                )
            return changed

        return await self._db.transaction(_update)

    async def has_document(self, doc_id: str) -> bool:
        """Check if a document exists in the store (with content)."""
        row = await self._db.run(
            lambda conn: conn.execute(
                "SELECT 1 FROM documents WHERE document_id = ? AND markdown_content <> ''", (doc_id,)
            ).fetchone()
        )
        return row is not None

    # -- sections -----------------------------------------------------------------

    async def replace_document_sections(
        self,
        doc_id: str,
        sections: list,
        *,
        source_content_hash: str,
    ) -> int:
        """Replace sections only when they were parsed from the current body."""

        def _replace(conn: sqlite3.Connection) -> int:
            row = conn.execute("SELECT content_hash FROM documents WHERE document_id = ?", (doc_id,)).fetchone()
            if row is None or row["content_hash"] != source_content_hash:
                raise ValueError("sections were not published because the source document changed")
            return replace_sections_sync(conn, doc_id, sections, source_content_hash=source_content_hash)

        return await self._db.transaction(_replace)

    async def _document_quality_by_doc(self, rows: list[sqlite3.Row]) -> dict[str, QualityAssessment]:
        """Assess each distinct document once per content hash, cached on that key."""
        keys = {
            row["doc_id"]: (row["doc_id"], row["source_content_hash"], get_configured_quality_failure(row["doc_id"]))
            for row in rows
        }
        missing = sorted(doc_id for doc_id, key in keys.items() if key not in self._document_quality_cache)
        if missing:
            if len(self._document_quality_cache) >= _DOCUMENT_QUALITY_CACHE_LIMIT:
                self._document_quality_cache.clear()
            placeholders = ",".join("?" for _ in missing)
            bodies = await self._db.run(
                lambda conn: conn.execute(
                    f"SELECT document_id, content_hash, markdown_content FROM documents "
                    f"WHERE document_id IN ({placeholders})",
                    missing,
                ).fetchall()
            )
            for body in bodies:
                doc_id = body["document_id"]
                key = (doc_id, body["content_hash"], get_configured_quality_failure(doc_id))
                self._document_quality_cache[key] = assess_markdown_quality(
                    body["markdown_content"] or "", document_id=doc_id
                )
        return {
            doc_id: self._document_quality_cache[key]
            for doc_id, key in keys.items()
            if key in self._document_quality_cache
        }

    async def _sections_from_rows(self, rows: list[sqlite3.Row]) -> list[StoredDocumentSection]:
        quality_by_doc = await self._document_quality_by_doc(rows) if rows else {}
        sections = []
        for row in rows:
            quality = quality_by_doc.get(row["doc_id"])
            sections.append(
                _section_from_row(row, document_quality=None if quality is None else quality.model_copy(deep=True))
            )
        return sections

    async def get_document_section(
        self,
        doc_id: str,
        *,
        section_type: str | None = None,
        section_ref: str | None = None,
        heading: str | None = None,
        limit: int | None = None,
        start_char: int | None = None,
    ) -> list[StoredDocumentSection]:
        """Fetch structural sections by document ID and optional exact refs and start offset."""
        if limit is not None and (isinstance(limit, bool) or not 1 <= limit <= 1_000):
            raise ValueError("section limit must be between 1 and 1000")
        where = ["s.doc_id = ?"]
        params: list = [doc_id]
        if section_type:
            where.append("s.section_type = ?")
            params.append(section_type)
        if section_ref:
            where.append("s.section_ref = ?")
            params.append(section_ref)
        if heading:
            # Apostrophe variants are one character on both sides: "BSD'nin" finds "BSD’nin". The stored
            # heading_norm keeps the typographic form, so the column is folded here, not at build time.
            where.append(f"{_HEADING_NORM_FOLDED_SQL} LIKE ? ESCAPE '\\'")
            escaped = (
                _fold_apostrophes(normalize_search_text(heading))
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            params.append(f"%{escaped}%")
        if start_char is not None:
            where.append("s.start_char = ?")
            params.append(start_char)
        limit_clause = ""
        if limit is not None:
            limit_clause = "LIMIT ?"
            params.append(limit)
        sql = f"""
            SELECT {_SECTION_COLUMNS} {_SOURCE_RANGE_COLUMN}
            FROM document_sections AS s
            JOIN documents AS d ON d.document_id = s.doc_id AND d.content_hash = s.source_content_hash
            WHERE {" AND ".join(where)}
            ORDER BY s.start_char
            {limit_clause}
        """
        rows = await self._db.run(lambda conn: conn.execute(sql, params).fetchall())
        return await self._sections_from_rows(rows)

    async def search_document_sections(
        self,
        query: str,
        *,
        document_id: str | None = None,
        section_type: str | None = None,
        limit: int = 10,
        include_source_range: bool = True,
        prefix_stem: bool = False,
    ) -> list[StoredDocumentSection]:
        """Full-text search over structural sections (every query token must match).

        ``include_source_range=False`` skips slicing the document body per row;
        candidate retrieval uses it and hydrates only the final hits via
        ``hydrate_source_ranges``. ``prefix_stem=True`` matches the tokens with
        :func:`fts_prefix_stem_query` instead (inflection-tolerant); it returns
        nothing when that expression would equal the strict one.
        """
        match = fts_prefix_stem_query(query) if prefix_stem else fts_match_query(query)
        if match is None:
            return []
        where = ["sections_fts MATCH ?"]
        params: list = [match]
        if document_id:
            where.append("s.doc_id = ?")
            params.append(document_id)
        if section_type:
            where.append("s.section_type = ?")
            params.append(section_type)
        else:
            # Nested fıkra/bent rows duplicate a parent madde and stay opt-in via section_type.
            # Every govde row is searched: a full-text window, the continuation of a capped
            # provision, decision or body text outside annexes/paragraphs, or remainder; the
            # flags on each row let the caller label it.
            where.append("s.section_type NOT IN (?, ?)")
            params.extend(_NESTED_SECTION_TYPES)
        params.append(limit)
        sql = f"""
            SELECT {_SECTION_COLUMNS} {_SOURCE_RANGE_COLUMN if include_source_range else ""},
                   -bm25(sections_fts, 2.0, 1.0) AS rank
            FROM sections_fts
            JOIN document_sections AS s ON s.rowid = sections_fts.rowid
            JOIN documents AS d ON d.document_id = s.doc_id AND d.content_hash = s.source_content_hash
            WHERE {" AND ".join(where)}
            ORDER BY rank DESC, s.start_char
            LIMIT ?
        """
        rows = await self._db.run(lambda conn: conn.execute(sql, params).fetchall())
        return await self._sections_from_rows(rows)

    async def find_capped_continuations(
        self, sections: list[StoredDocumentSection]
    ) -> dict[tuple[str, str, int], tuple[str, str, int, bool]]:
        """Map each govde row that continues a capped named section to (type, ref, true end, runs past).

        A named section longer than ``MAX_SECTION_CHARS`` is stored as a capped window; the rest of
        its span is indexed as govde windows. A govde row is such a continuation when a capped named
        section of the same document and content hash has an uncapped span (``start_char`` plus the
        length recorded in its truncation notice) that contains the row's ``start_char``. When several
        nested capped sections contain the row, the innermost (latest-starting) one is chosen. Keys are
        ``(doc_id, source_content_hash, start_char)`` of the govde rows passed in. The value's third
        element is the capped section's true end (exclusive code point); the fourth is True when the
        row has non-whitespace document text at or after that end, i.e. part of the row lies outside
        the provision.
        """
        govde = [section for section in sections if section.section_type == GOVDE_SECTION_TYPE]
        if not govde:
            return {}
        doc_ids = sorted({section.doc_id for section in govde})
        placeholders = ",".join("?" for _ in doc_ids)
        rows = await self._db.run(
            lambda conn: conn.execute(
                f"SELECT doc_id, section_type, section_ref, start_char, content, source_content_hash "
                f"FROM document_sections WHERE doc_id IN ({placeholders}) "
                f"AND section_type NOT IN (?, ?, ?) AND instr(content, ?) > 0 "
                f"ORDER BY doc_id, start_char",
                [*doc_ids, *_EXCLUDED_SECTION_TYPES, TRUNCATION_NOTICE_PREFIX],
            ).fetchall()
        )
        capped: dict[tuple[str, str], list[tuple[int, int, str, str]]] = {}
        for row in rows:
            original_length = truncation_notice_original_length(split_section_truncation_notice(row["content"])[1])
            if original_length is None:
                continue
            capped.setdefault((row["doc_id"], row["source_content_hash"]), []).append(
                (row["start_char"], row["start_char"] + original_length, row["section_type"], row["section_ref"])
            )
        continuations: dict[tuple[str, str, int], tuple[str, str, int, bool]] = {}
        overruns: list[tuple[tuple[str, str, int], int, int]] = []
        for section in govde:
            parents = [
                (start, end, section_type, section_ref)
                for start, end, section_type, section_ref in capped.get(
                    (section.doc_id, section.source_content_hash), []
                )
                if start < section.start_char < end
            ]
            if parents:
                # Innermost span first: the latest-starting capped section that contains the row.
                _start, end, section_type, section_ref = max(parents)
                key = (section.doc_id, section.source_content_hash, section.start_char)
                continuations[key] = (section_type, section_ref, end, False)
                if section.end_char > end:
                    overruns.append((key, end, section.end_char))
        if overruns:

            def tails(conn: sqlite3.Connection) -> list[tuple[tuple[str, str, int], str]]:
                return [
                    (
                        key,
                        conn.execute(
                            "SELECT substr(markdown_content, ? + 1, ? - ?) FROM documents "
                            "WHERE document_id = ? AND content_hash = ?",
                            [end, row_end, end, key[0], key[1]],
                        ).fetchone()[0]
                        or "",
                    )
                    for key, end, row_end in overruns
                ]

            for key, tail in await self._db.run(tails):
                if tail.strip():
                    section_type, section_ref, end, _runs_past = continuations[key]
                    continuations[key] = (section_type, section_ref, end, True)
        return continuations

    async def structureless_documents(self, doc_ids: list[str]) -> set[str]:
        """Return the ids among ``doc_ids`` whose current body has govde sections and no article structure.

        Same rule as ``document_structureless`` on section rows: the sections of the current
        content hash are only govde/fikra/bent rows, so the govde windows carry the whole body.
        """
        unique = sorted(set(doc_ids))
        if not unique:
            return set()
        placeholders = ",".join("?" for _ in unique)
        rows = await self._db.run(
            lambda conn: conn.execute(
                f"SELECT d.document_id FROM documents AS d WHERE d.document_id IN ({placeholders}) "
                "AND EXISTS (SELECT 1 FROM document_sections AS g WHERE g.doc_id = d.document_id "
                "AND g.source_content_hash = d.content_hash AND g.section_type = 'govde') "
                "AND NOT EXISTS (SELECT 1 FROM document_sections AS x WHERE x.doc_id = d.document_id "
                "AND x.source_content_hash = d.content_hash AND x.section_type NOT IN ('fikra', 'bent', 'govde'))",
                unique,
            ).fetchall()
        )
        return {row["document_id"] for row in rows}

    async def find_parent_articles(
        self, sections: list[StoredDocumentSection]
    ) -> dict[tuple[str, str, int], tuple[str, str]]:
        """Map each fıkra/bent row to the (type, ref) of the provision containing it.

        The provision is a madde, geçici madde or ek madde, or (in guides and standards) a paragraf
        or ilke. Computed at query time from the stored sections of the same document and content hash:
        the innermost stored provision window ``[start_char, end_char)`` that contains the row's
        ``start_char``. Keys are ``(doc_id, source_content_hash, start_char)`` of the rows passed in;
        a row without such a stored window has no entry.
        """
        nested = [section for section in sections if section.section_type in _NESTED_SECTION_TYPES]
        if not nested:
            return {}
        doc_ids = sorted({section.doc_id for section in nested})
        placeholders = ",".join("?" for _ in doc_ids)
        rows = await self._db.run(
            lambda conn: conn.execute(
                f"SELECT doc_id, section_type, section_ref, start_char, end_char, source_content_hash "
                f"FROM document_sections WHERE doc_id IN ({placeholders}) "
                f"AND section_type IN ({','.join('?' for _ in _PARENT_SECTION_TYPES)}) ORDER BY doc_id, start_char",
                [*doc_ids, *_PARENT_SECTION_TYPES],
            ).fetchall()
        )
        articles: dict[tuple[str, str], list[tuple[int, int, str, str]]] = {}
        for row in rows:
            articles.setdefault((row["doc_id"], row["source_content_hash"]), []).append(
                (row["start_char"], row["end_char"], row["section_type"], row["section_ref"])
            )
        parents: dict[tuple[str, str, int], tuple[str, str]] = {}
        for section in nested:
            containing = [
                (start, section_type, section_ref)
                for start, end, section_type, section_ref in articles.get(
                    (section.doc_id, section.source_content_hash), []
                )
                if start <= section.start_char < end
            ]
            if containing:
                _start, section_type, section_ref = max(containing)
                parents[(section.doc_id, section.source_content_hash, section.start_char)] = (section_type, section_ref)
        return parents

    async def hydrate_source_ranges(self, sections: list[StoredDocumentSection]) -> list[StoredDocumentSection]:
        """Fill ``normalized_source_range`` for sections retrieved without it."""
        pending = [index for index, section in enumerate(sections) if not section.normalized_source_range]
        if not pending:
            return sections
        doc_ids = sorted({sections[index].doc_id for index in pending})
        placeholders = ",".join("?" for _ in doc_ids)
        bodies = await self._db.run(
            lambda conn: conn.execute(
                f"SELECT document_id, content_hash, markdown_content FROM documents "
                f"WHERE document_id IN ({placeholders})",
                doc_ids,
            ).fetchall()
        )
        body_by_doc = {(row["document_id"], row["content_hash"]): row["markdown_content"] or "" for row in bodies}
        hydrated = list(sections)
        for index in pending:
            section = sections[index]
            body = body_by_doc.get((section.doc_id, section.source_content_hash))
            if body is not None:
                hydrated[index] = section.model_copy(
                    update={"normalized_source_range": body[section.start_char : section.end_char]}
                )
        return hydrated

    # -- sync failures ------------------------------------------------------------

    async def record_sync_failure(
        self,
        doc_id: str,
        error: str,
        category: str,
        source_url: str = "",
        retryable: bool = True,
    ) -> None:
        """Record a privacy-safe failure code without exception text or URLs."""
        now = time.time()
        safe_error, safe_category = _safe_sync_failure_fields(error, category)
        await self._db.transaction(
            lambda conn: conn.execute(
                """
                INSERT INTO sync_failures (document_id, error, error_category, source_url, retryable, attempts,
                                           first_failed_at, last_failed_at)
                VALUES (?, ?, ?, '', ?, 1, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    error = excluded.error,
                    error_category = excluded.error_category,
                    retryable = excluded.retryable,
                    attempts = sync_failures.attempts + 1,
                    last_failed_at = excluded.last_failed_at
                """,
                (doc_id, safe_error, safe_category, int(retryable), now, now),
            )
        )

    async def clear_sync_failure(self, doc_id: str) -> None:
        await self._db.transaction(
            lambda conn: conn.execute("DELETE FROM sync_failures WHERE document_id = ?", (doc_id,))
        )

    # -- versions -----------------------------------------------------------------

    async def get_current_body(self, doc_id: str) -> dict | None:
        """Return the stored body's hash, length, download time and extraction method, or None."""
        row = await self._db.run(
            lambda conn: conn.execute(
                "SELECT content_hash, LENGTH(markdown_content) AS content_length, downloaded_at, extraction_method "
                "FROM documents WHERE document_id = ? AND markdown_content <> ''",
                (doc_id,),
            ).fetchone()
        )
        if row is None:
            return None
        return {
            "content_hash": row["content_hash"] or "",
            "content_length": row["content_length"] or 0,
            "downloaded_at": row["downloaded_at"],
            "extraction_method": row["extraction_method"] or "",
        }

    async def get_document_history(self, doc_id: str) -> list[dict]:
        """List archived bodies, newest first; ``synced_at`` is when each was replaced (UTC)."""
        rows = await self._db.run(
            lambda conn: conn.execute(
                "SELECT version, content_hash, LENGTH(markdown_content) AS content_length, synced_at "
                "FROM document_versions WHERE document_id = ? ORDER BY version DESC",
                (doc_id,),
            ).fetchall()
        )
        return [
            {
                "version": row["version"],
                "content_hash": row["content_hash"],
                "synced_at": _utc_minute(row["synced_at"]),
                "content_length": row["content_length"],
            }
            for row in rows
        ]

    async def get_version_counts(self, doc_ids: list[str]) -> dict[str, tuple[int, str | None]]:
        """Batch count of archived bodies. Returns {doc_id: (count, latest replacement time in UTC)}."""
        if not doc_ids:
            return {}
        placeholders = ",".join("?" for _ in doc_ids)
        rows = await self._db.run(
            lambda conn: conn.execute(
                f"SELECT document_id, COUNT(*) AS cnt, MAX(synced_at) AS latest FROM document_versions "
                f"WHERE document_id IN ({placeholders}) GROUP BY document_id",
                list(doc_ids),
            ).fetchall()
        )
        result = {}
        for row in rows:
            latest = _utc_minute(row["latest"]) if row["latest"] else None
            result[row["document_id"]] = (row["cnt"], latest)
        return result

    # -- catalog (decision_cache) -------------------------------------------------

    async def load_catalog(self) -> list[dict]:
        rows = await self._db.run(
            lambda conn: conn.execute(
                "SELECT document_id, title, content, decision_date, decision_number, category, source_url, "
                "cached_at FROM decision_cache ORDER BY document_id"
            ).fetchall()
        )
        return [dict(row) for row in rows]

    async def replace_catalog(self, items: list[dict], *, cached_at: float | None = None) -> int:
        """Replace the catalog with exactly ``items`` (document_id, title, content, ...)."""
        now = cached_at if cached_at is not None else time.time()
        rows = [
            (
                item["document_id"],
                item.get("title", ""),
                item.get("content", ""),
                item.get("decision_date", ""),
                item.get("decision_number", ""),
                item.get("category", ""),
                item.get("source_url", "") or "",
                now,
            )
            for item in items
        ]

        def _replace(conn: sqlite3.Connection) -> int:
            conn.execute("DELETE FROM decision_cache")
            conn.executemany(
                "INSERT INTO decision_cache (document_id, title, content, decision_date, decision_number, "
                "category, source_url, cached_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            return len(rows)

        return await self._db.transaction(_replace)
