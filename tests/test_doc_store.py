"""Tests for DocumentStore (SQLite + FTS5)."""

import pytest

from doc_store import DocumentStore, StoredDocument


@pytest.fixture
async def store(tmp_path):
    """Create a temporary DocumentStore."""
    db_path = tmp_path / "test_docs.db"
    s = DocumentStore(db_path=db_path)
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
def sample_doc():
    return StoredDocument(
        document_id="1291",
        title="Sermaye Yeterliliği Rehberi",
        category="Rehber",
        decision_date="15.03.2024",
        decision_number="11200",
        source_url="https://www.bddk.org.tr/Mevzuat/DokumanGetir/1291",
        markdown_content="# Sermaye Yeterliliği\n\nBu rehber bankacılık sektöründe sermaye yeterliliği hesaplamalarını düzenler.\n\n## MADDE 1\nKredi riski için asgari sermaye oranı %8 olarak belirlenmiştir.",
        extraction_method="markitdown",
    )


@pytest.fixture
def mevzuat_doc():
    return StoredDocument(
        document_id="mevzuat_42628",
        title="Faiz Oranı Riski Yönetmeliği",
        category="Yönetmelik",
        source_url="https://mevzuat.gov.tr/MevzuatMetin/yonetmelik/7.5.42628",
        markdown_content="# Faiz Oranı Riski\n\nİskonto formülü: $İO_{0,p}(t_o) = \\exp(-F_{0,p}(t_o) \\cdot t_o)$\n\n## MADDE 9\nBanka, faiz oranı riskini ölçmek için standart yaklaşım kullanır.",
        extraction_method="nougat",
    )


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
