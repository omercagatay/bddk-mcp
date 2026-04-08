"""Tests for DocumentStore (PostgreSQL + tsvector)."""

import pytest

from doc_store import DocumentStore, StoredDocument


# Uses doc_store, sample_doc, mevzuat_doc fixtures from conftest.py
# Alias doc_store → store for shorter test signatures
@pytest.fixture
async def store(doc_store):
    yield doc_store


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
