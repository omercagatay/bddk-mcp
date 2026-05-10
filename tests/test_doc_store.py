"""Tests for DocumentStore (PostgreSQL + tsvector)."""

import pytest

from doc_store import _SCHEMA_SQL, StoredDocument, StoredDocumentSection
from section_index import extract_document_sections


# Uses doc_store, sample_doc, mevzuat_doc fixtures from conftest.py
# Alias doc_store → store for shorter test signatures
@pytest.fixture
async def store(doc_store):
    yield doc_store


def test_document_sections_schema_is_declared():
    assert "CREATE TABLE IF NOT EXISTS document_sections" in _SCHEMA_SQL
    assert "UNIQUE(doc_id, section_type, section_ref, content_hash)" in _SCHEMA_SQL
    assert "idx_document_sections_tsv" in _SCHEMA_SQL


def test_tool_call_trace_schema_is_declared():
    assert "CREATE TABLE IF NOT EXISTS tool_call_traces" in _SCHEMA_SQL
    assert "args_summary   JSONB" in _SCHEMA_SQL
    assert "quality_labels JSONB" in _SCHEMA_SQL
    assert "idx_tool_call_traces_doc_ids" in _SCHEMA_SQL


def test_stored_document_section_model():
    section = StoredDocumentSection(
        doc_id="943",
        section_type="ilke",
        section_ref="5",
        heading="Model validasyonu",
        start_char=0,
        end_char=42,
        content="İlke 5 - Model validasyonu",
        content_hash="abc",
    )

    assert section.doc_id == "943"
    assert section.section_type == "ilke"


async def test_store_and_retrieve(store, sample_doc):
    await store.store_document(sample_doc)
    doc = await store.get_document("1291")
    assert doc is not None
    assert doc.title == "Sermaye Yeterliliği Rehberi"
    assert doc.category == "Rehber"
    assert "sermaye yeterliliği" in doc.markdown_content.lower()


async def test_get_nonexistent(store):
    doc = await store.get_document("nonexistent")
    assert doc is None


async def test_pagination(store):
    # Create a long document
    long_content = "A" * 15000
    doc = StoredDocument(
        document_id="long_doc",
        title="Long Document",
        markdown_content=long_content,
    )
    await store.store_document(doc)

    page1 = await store.get_document_page("long_doc", page=1)
    assert page1 is not None
    assert page1.total_pages == 3
    assert len(page1.markdown_content) == 5000

    page3 = await store.get_document_page("long_doc", page=3)
    assert page3 is not None
    assert len(page3.markdown_content) == 5000

    invalid = await store.get_document_page("long_doc", page=99)
    assert invalid is not None
    assert "Invalid page" in invalid.markdown_content


async def test_fts_search(store, sample_doc, mevzuat_doc):
    await store.store_document(sample_doc)
    await store.store_document(mevzuat_doc)

    hits = await store.search_content("sermaye")
    assert len(hits) >= 1
    assert any(h.document_id == "1291" for h in hits)

    hits = await store.search_content("faiz oranı")
    assert len(hits) >= 1
    assert any(h.document_id == "mevzuat_42628" for h in hits)


async def test_search_with_category_filter(store, sample_doc, mevzuat_doc):
    await store.store_document(sample_doc)
    await store.store_document(mevzuat_doc)

    hits = await store.search_content("sermaye", category="Rehber")
    assert all(h.category == "Rehber" for h in hits)


async def test_needs_refresh(store, sample_doc):
    assert await store.needs_refresh("1291") is True  # not in store
    await store.store_document(sample_doc)
    assert await store.needs_refresh("1291") is False  # just stored


async def test_has_document(store, sample_doc):
    assert await store.has_document("1291") is False
    await store.store_document(sample_doc)
    assert await store.has_document("1291") is True


async def test_delete_document(store, sample_doc):
    await store.store_document(sample_doc)
    assert await store.has_document("1291") is True
    deleted = await store.delete_document("1291")
    assert deleted is True
    assert await store.has_document("1291") is False


async def test_replace_and_get_document_sections(store):
    text = "İlke 5 - Model validasyonu\nBankalar modeli doğrular.\n\nİlke 6\nSonraki ilke."
    sections = extract_document_sections("943", text)

    await store.replace_document_sections("943", sections)
    found = await store.get_document_section("943", section_type="ilke", section_ref="5")

    assert len(found) == 1
    assert found[0].doc_id == "943"
    assert found[0].section_type == "ilke"
    assert found[0].section_ref == "5"
    assert "Bankalar modeli doğrular." in found[0].content


async def test_replace_document_sections_deletes_stale_rows(store):
    first = extract_document_sections("943", "İlke 5\nEski içerik.\n\nİlke 6\nSilinecek.")
    second = extract_document_sections("943", "İlke 5\nYeni içerik.")

    await store.replace_document_sections("943", first)
    await store.replace_document_sections("943", second)

    found = await store.get_document_section("943", section_type="ilke")
    assert len(found) == 1
    assert found[0].section_ref == "5"
    assert "Yeni içerik." in found[0].content


async def test_store_document_auto_indexes_sections(store):
    doc = StoredDocument(
        document_id="auto_sections",
        title="Auto Section Doc",
        markdown_content="MADDE 9 - TFRS 9 karşılık\nBankalar karşılık ayırır.\n\nMADDE 10\nBaşka hüküm.",
    )

    await store.store_document(doc)
    found = await store.get_document_section("auto_sections", section_type="madde", section_ref="9")

    assert len(found) == 1
    assert "Bankalar karşılık ayırır." in found[0].content


async def test_store_document_replaces_stale_auto_indexed_sections(store):
    first = StoredDocument(
        document_id="auto_sections", title="Doc", markdown_content="İlke 5\nEski içerik.\n\nİlke 6\nSilinecek."
    )
    second = StoredDocument(document_id="auto_sections", title="Doc", markdown_content="İlke 5\nYeni içerik.")

    await store.store_document(first)
    await store.store_document(second)

    found = await store.get_document_section("auto_sections", section_type="ilke")
    assert len(found) == 1
    assert found[0].section_ref == "5"
    assert "Yeni içerik." in found[0].content


async def test_store_document_empty_content_clears_sections(store):
    first = StoredDocument(document_id="auto_sections", title="Doc", markdown_content="İlke 5\nEski içerik.")
    empty = StoredDocument(document_id="auto_sections", title="Doc", markdown_content="")

    await store.store_document(first)
    await store.store_document(empty)

    assert await store.get_document_section("auto_sections") == []


async def test_search_document_sections(store):
    text = "MADDE 9 - TFRS 9 karşılık\nBankalar karşılık ayırır.\n\nMADDE 10\nBaşka hüküm."
    await store.replace_document_sections("mevzuat_22599", extract_document_sections("mevzuat_22599", text))

    hits = await store.search_document_sections("TFRS 9 karşılık", document_id="mevzuat_22599")

    assert len(hits) >= 1
    assert hits[0].doc_id == "mevzuat_22599"
    assert hits[0].section_type == "madde"
    assert hits[0].section_ref == "9"


async def test_upsert(store, sample_doc):
    await store.store_document(sample_doc)
    updated = sample_doc.model_copy(update={"title": "Güncellenmiş Başlık"})
    await store.store_document(updated)
    doc = await store.get_document("1291")
    assert doc.title == "Güncellenmiş Başlık"


async def test_import_from_cache(store):
    cache_items = [
        {"document_id": "100", "title": "Doc A", "category": "Genelge"},
        {"document_id": "200", "title": "Doc B", "category": "Tebliğ"},
    ]
    imported = await store.import_from_cache(cache_items)
    assert imported == 2

    # Second import should skip existing
    imported2 = await store.import_from_cache(cache_items)
    assert imported2 == 0


async def test_stats(store, sample_doc, mevzuat_doc):
    await store.store_document(sample_doc)
    await store.store_document(mevzuat_doc)

    st = await store.stats()
    assert st.total_documents == 2
    assert "Rehber" in st.categories
    assert "Yönetmelik" in st.categories


async def test_list_documents(store, sample_doc, mevzuat_doc):
    await store.store_document(sample_doc)
    await store.store_document(mevzuat_doc)

    docs = await store.list_documents()
    assert len(docs) == 2

    docs_filtered = await store.list_documents(category="Rehber")
    assert len(docs_filtered) == 1
    assert docs_filtered[0]["document_id"] == "1291"
