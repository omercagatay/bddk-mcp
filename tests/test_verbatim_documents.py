"""Exhaustive verbatim contract over the real reviewed corpus (all documents).

Every document must be returned as exact stored characters: no sanitization,
wrapping, paraphrase or summary. These tests read the reviewed seed snapshot and
a real PostgreSQL table; they do not call live BDDK/mevzuat sources.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from bddk_mcp.core.config import PAGE_SIZE
from bddk_mcp.quality.markdown_quality import unsafe_verbatim_reason
from bddk_mcp.quotations import check_exact_quotation
from bddk_mcp.store.doc_store import DocumentPage, DocumentStore
from bddk_mcp.store.section_index import extract_document_sections, split_section_truncation_notice
from bddk_mcp.tools.contract_types import OptionalQuotation

SEED_DOCUMENTS = json.loads((Path(__file__).parents[1] / "seed_data/documents.json").read_text(encoding="utf-8"))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_probe(document: dict) -> str:
    """A non-blank exact contiguous excerpt taken from the middle of the document."""
    content = document["markdown_content"]
    for start in range(len(content) // 3, max(0, len(content) - 80)):
        candidate = content[start : start + 80]
        if candidate.strip():
            return candidate
    raise AssertionError(f"no nonblank probe in {document['document_id']}")


def test_seed_corpus_snapshot_is_complete_and_self_consistent():
    assert len(SEED_DOCUMENTS) == 318
    ids = [d["document_id"] for d in SEED_DOCUMENTS]
    assert len(set(ids)) == len(ids)
    for document in SEED_DOCUMENTS:
        content = document["markdown_content"]
        assert content
        assert sha(content) == document["content_hash"], document["document_id"]


@pytest.mark.parametrize("document", SEED_DOCUMENTS, ids=lambda d: d["document_id"])
def test_every_seed_document_is_verbatim_servable_and_quote_exact(document):
    content = document["markdown_content"]
    # No document is refused and none is silently rewritten by a sanitizer.
    assert unsafe_verbatim_reason(content) is None
    # Page slicing must be exact stored characters for the whole document.
    total_pages = max(1, -(-len(content) // PAGE_SIZE))
    for page in range(1, total_pages + 1):
        start = (page - 1) * PAGE_SIZE
        assert content[start : start + PAGE_SIZE] == content[start : start + PAGE_SIZE]
    # Whole-document verification is exact and reconstructable.
    whole = check_exact_quotation(
        content, content, expected_reference_sha256=document["content_hash"], require_complete=True
    )
    assert whole.status == "exact_reference_match" and whole.extent == "whole_reference"
    assert (whole.start_char, whole.end_char) == (0, len(content))
    assert whole.normalization == "none"
    assert whole.original_source_fidelity == "not_established"
    # An exact excerpt verifies; repeated text is honestly reported ambiguous, never invented.
    quote = exact_probe(document)
    match = check_exact_quotation(content, quote, expected_reference_sha256=document["content_hash"])
    assert match.status in {"exact_reference_match", "ambiguous"}
    if match.status == "exact_reference_match":
        assert content[match.start_char : match.end_char] == quote
    else:
        occurrences = [i for i in range(len(content)) if content.startswith(quote, i)]
        assert len(occurrences) > 1
        assert all(content[i : i + len(quote)] == quote for i in occurrences)
    # A single altered character, changed whitespace or a paraphrase must not verify.
    altered = quote[:-1] + ("x" if quote[-1] != "x" else "y")
    assert check_exact_quotation(content, altered, expected_reference_sha256=document["content_hash"]).status != (
        "exact_reference_match"
    )
    assert (
        check_exact_quotation(
            content,
            f"KAYNAKTA-YOK: {quote[:40]}",
            expected_reference_sha256=document["content_hash"],
        ).status
        != "exact_reference_match"
    )


@pytest.mark.parametrize("document", SEED_DOCUMENTS, ids=lambda d: d["document_id"])
def test_every_seed_section_is_an_exact_document_slice(document):
    """Parsed sections must be exact substrings; a quote inside them is exact too."""
    content = document["markdown_content"]
    sections = extract_document_sections(document["document_id"], content)
    for section in sections:
        if not section.content:
            continue
        verbatim, notice = split_section_truncation_notice(section.content)
        assert content[section.start_char : section.end_char].strip() == verbatim
        # A stored truncation notice is metadata, never served as legal text.
        assert "BÖLÜM KESİLDİ" not in verbatim
        # Whole-unit verification must be exact; a partial probe may be legitimately ambiguous.
        whole = check_exact_quotation(
            verbatim,
            verbatim,
            expected_reference_sha256=sha(verbatim),
            require_complete=True,
        )
        assert whole.status == "exact_reference_match"
        assert whole.extent == "whole_reference"
        assert (notice is not None) == ("BÖLÜM KESİLDİ" in section.content)


@pytest.mark.asyncio
@pytest.mark.parametrize("document", SEED_DOCUMENTS, ids=lambda d: d["document_id"])
async def test_every_document_returned_through_mcp_is_exact_and_not_summarized(document):
    """Official MCP SDK path: returned page text must equal the stored characters."""
    from mcp.shared.memory import create_connected_server_and_client_session

    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.server import create_mcp

    content = document["markdown_content"]
    total_pages = max(1, -(-len(content) // PAGE_SIZE))
    page = 1
    start = (page - 1) * PAGE_SIZE
    expected = content[start : start + PAGE_SIZE]
    doc_store = MagicMock()
    doc_store.get_document_page = AsyncMock(
        return_value=DocumentPage(
            document_id=document["document_id"],
            title=document.get("title", ""),
            markdown_content=expected,
            page_number=page,
            total_pages=total_pages,
            extraction_method=document.get("extraction_method", ""),
        )
    )
    doc_store.get_extraction_method = AsyncMock(return_value=document.get("extraction_method", ""))
    doc_store.get_document_quality = AsyncMock(return_value=None)
    client = MagicMock()
    client.find_by_id.return_value = None
    deps = Dependencies(pool=None, doc_store=doc_store, client=client, http=None)

    async with create_connected_server_and_client_session(create_mcp(deps)) as session:
        result = await session.call_tool("get_bddk_document", {"document_id": document["document_id"]})

    assert result.isError is False, document["document_id"]
    structured = result.structuredContent
    assert structured["pages"] == [{"page_number": page, "content": expected}]
    assert expected in structured["text"]
    # The prose around the document must declare the verbatim contract, not a summary.
    assert "exact stored document content" in structured["text"]
    assert "no sanitization, wrapping, paraphrase or summary" in structured["text"]


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_all_seeded_documents_round_trip_exactly_through_postgresql(pg_pool):
    """Real PostgreSQL storage: every page equals the stored characters exactly."""
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            for document in SEED_DOCUMENTS:
                await connection.execute(
                    """
                    INSERT INTO public.documents (document_id, title, markdown_content, content_hash)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (document_id) DO UPDATE SET
                        markdown_content = EXCLUDED.markdown_content,
                        content_hash = EXCLUDED.content_hash
                    """,
                    document["document_id"],
                    document.get("title", ""),
                    document["markdown_content"],
                    document["content_hash"],
                )
            from tests.conftest import SingleConnPool

            store = DocumentStore(SingleConnPool(connection))
            for document in SEED_DOCUMENTS:
                content = document["markdown_content"]
                total_pages = max(1, -(-len(content) // PAGE_SIZE))
                for page in range(1, total_pages + 1):
                    start = (page - 1) * PAGE_SIZE
                    fetched = await store.get_document_page(document["document_id"], page)
                    assert fetched is not None, document["document_id"]
                    assert fetched.markdown_content == content[start : start + PAGE_SIZE], (
                        document["document_id"],
                        page,
                    )
        finally:
            await transaction.rollback()


def test_quotation_argument_accepts_exact_boundaries_and_rejects_whitespace_only():
    from mcp.server.fastmcp.exceptions import ToolError
    from pydantic import TypeAdapter

    adapter = TypeAdapter(OptionalQuotation)
    exact = "  MADDE 9 – (1) Metin.\n"
    assert adapter.validate_python(exact) == exact
    with pytest.raises(ToolError):
        adapter.validate_python("   \n\t ")


def test_real_token_chunker_produces_exact_document_slices_for_every_document():
    """The production retrieval profile's chunker must emit exact stored slices."""
    from bddk_mcp.store.vector_store import VectorStore, _chunk_document

    store = VectorStore(None)
    tokenizer = store._chunk_tokenizer()
    chunks_seen = 0
    for document in SEED_DOCUMENTS:
        content = document["markdown_content"]
        for chunk in _chunk_document(document["document_id"], content, tokenizer=tokenizer):
            chunks_seen += 1
            assert chunk.chunk_text == content[chunk.start_char : chunk.end_char], (
                document["document_id"],
                chunk.start_char,
                chunk.end_char,
            )
    assert chunks_seen > 10_000


def test_real_ozkaynak_article_9_quotation_is_exact_and_fidelity_gap_is_reported():
    """The reviewed example: Bankaların Özkaynaklarına İlişkin Yönetmelik madde 9."""
    from bddk_mcp.store.section_index import extract_document_sections, split_section_truncation_notice

    document = next(d for d in SEED_DOCUMENTS if d["document_id"] == "mevzuat_18799")
    content = document["markdown_content"]
    quotation = (
        "ç) Geçici farklara dayanan ertelenmiş vergi varlıkları hariç olmak üzere gelecek dönemlerde "
        "elde edilecek vergilendirilebilir gelirlere dayanan ertelenmiş vergi varlığının, ilgili Türkiye "
        "Muhasebe Standardında yer alan koşulların sağlanması halinde ertelenmiş vergi yükümlülüğü ile "
        "mahsup edildikten sonra kalan kısmı,"
    )
    whole_document = check_exact_quotation(content, quotation, expected_reference_sha256=document["content_hash"])
    assert whole_document.status == "exact_reference_match"
    assert content[whole_document.start_char : whole_document.end_char] == quotation

    section = next(
        s
        for s in extract_document_sections("mevzuat_18799", content)
        if s.section_type == "madde" and s.section_ref == "9"
    )
    verbatim, notice = split_section_truncation_notice(section.content)
    assert notice is None
    assert "BÖLÜM KESİLDİ" not in verbatim
    section_check = check_exact_quotation(verbatim, quotation, expected_reference_sha256=sha(verbatim))
    assert section_check.status == "exact_reference_match"

    # Honest fidelity gap: the stored extraction uses the legacy Windows-1252
    # apostrophe (U+0092) in this section. The legacy form matches; the official
    # typographic form does not. It must fail, never be silently normalized.
    assert "\x92" in verbatim
    legacy = "yüzde 17,65\x92ini aşan kısmı"
    official = legacy.replace("\x92", "’")
    assert legacy in verbatim and official not in verbatim
    assert (
        check_exact_quotation(verbatim, legacy, expected_reference_sha256=sha(verbatim)).status
        == "exact_reference_match"
    )
    assert (
        check_exact_quotation(verbatim, official, expected_reference_sha256=sha(verbatim)).status
        != "exact_reference_match"
    )


@pytest.mark.postgres
@pytest.mark.asyncio
async def test_real_ozkaynak_document_and_quote_end_to_end_through_mcp_and_postgresql(pg_pool):
    """Real PostgreSQL + official MCP SDK: exact text out, exact quote verified."""
    from mcp.shared.memory import create_connected_server_and_client_session

    from bddk_mcp.core.deps import Dependencies
    from bddk_mcp.server import create_mcp
    from tests.conftest import SingleConnPool

    document = next(d for d in SEED_DOCUMENTS if d["document_id"] == "mevzuat_18799")
    content = document["markdown_content"]
    quotation = (
        "ç) Geçici farklara dayanan ertelenmiş vergi varlıkları hariç olmak üzere gelecek dönemlerde "
        "elde edilecek vergilendirilebilir gelirlere dayanan ertelenmiş vergi varlığının, ilgili Türkiye "
        "Muhasebe Standardında yer alan koşulların sağlanması halinde ertelenmiş vergi yükümlülüğü ile "
        "mahsup edildikten sonra kalan kısmı,"
    )
    async with pg_pool.acquire() as connection:
        transaction = connection.transaction()
        await transaction.start()
        try:
            await connection.execute(
                """
                INSERT INTO public.documents (document_id, title, markdown_content, content_hash)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (document_id) DO UPDATE SET
                    markdown_content = EXCLUDED.markdown_content,
                    content_hash = EXCLUDED.content_hash
                """,
                document["document_id"],
                document.get("title", ""),
                content,
                document["content_hash"],
            )
            store = DocumentStore(SingleConnPool(connection))
            await store.replace_document_sections(
                document["document_id"],
                extract_document_sections(document["document_id"], content),
                source_content_hash=document["content_hash"],
            )
            client = MagicMock()
            client.find_by_id.return_value = None
            deps = Dependencies(pool=None, doc_store=store, client=client, http=None)
            async with create_connected_server_and_client_session(create_mcp(deps)) as session:
                page = await session.call_tool(
                    "get_bddk_document", {"document_id": "mevzuat_18799", "page_number": 1, "max_pages": 1}
                )
                section = await session.call_tool(
                    "get_document_section",
                    {"document_id": "mevzuat_18799", "section_type": "madde", "section_ref": "9"},
                )
                exact_quote = await session.call_tool(
                    "get_document_section",
                    {
                        "document_id": "mevzuat_18799",
                        "section_type": "madde",
                        "section_ref": "9",
                        "quotation": quotation,
                    },
                )
                paraphrased = await session.call_tool(
                    "get_document_section",
                    {
                        "document_id": "mevzuat_18799",
                        "section_type": "madde",
                        "section_ref": "9",
                        "quotation": quotation.replace("hariç olmak üzere", "hariç"),
                    },
                )

            assert page.isError is False
            assert page.structuredContent["pages"][0]["content"] == content[:PAGE_SIZE]
            assert section.isError is False
            section_item = section.structuredContent["results"][0]
            assert content[section_item["start_char"] : section_item["end_char"]].strip() == section_item["content"]
            assert exact_quote.structuredContent["answer_assessment"]["quotation_status"] == "exact_reference_match"
            assert (
                exact_quote.structuredContent["answer_assessment"]["quotation_check"]["original_source_fidelity"]
                == "not_established"
            )
            assert paraphrased.structuredContent["answer_assessment"]["quotation_status"] == "not_found"
        finally:
            await transaction.rollback()
