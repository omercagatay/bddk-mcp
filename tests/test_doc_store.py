"""Tests for DocumentStore (SQLite corpus file + FTS5)."""

import hashlib
from unittest.mock import MagicMock

import pytest

from bddk_mcp.quality.markdown_quality import assess_markdown_quality
from bddk_mcp.store.doc_store import StoredDocument, StoredDocumentSection
from bddk_mcp.store.section_index import extract_document_sections


# Uses doc_store, sample_doc, mevzuat_doc fixtures from conftest.py
# Alias doc_store → store for shorter test signatures
@pytest.fixture
async def store(doc_store):
    yield doc_store


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


async def test_store_document_sanitizes_cid_and_data_uri(store):
    await store.store_document(
        StoredDocument(
            document_id="1291",
            title="Kirli çıkarım",
            markdown_content=(
                "MADDE 1 cid:image001.png@01D12345 hukuki metin.\n"
                "![](data:image/x-wmf;base64,AQAJAAADcgIAAAIAHAAAAAAABQAAAA==) devam."
            ),
        )
    )
    doc = await store.get_document("1291")
    assert doc is not None
    assert "cid:" not in doc.markdown_content
    assert "data:image/" not in doc.markdown_content
    assert "hukuki metin" in doc.markdown_content


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
    source_hash = hashlib.sha256(text.encode()).hexdigest()
    await store.store_document(StoredDocument(document_id="943", title="Test", markdown_content=text))

    await store.replace_document_sections("943", sections, source_content_hash=source_hash)
    found = await store.get_document_section("943", section_type="ilke", section_ref="5")

    assert len(found) == 1
    assert found[0].doc_id == "943"
    assert found[0].section_type == "ilke"
    assert found[0].section_ref == "5"
    assert "Bankalar modeli doğrular." in found[0].content


async def test_replace_document_sections_deletes_stale_rows(store):
    first_text = "İlke 5\nEski içerik.\n\nİlke 6\nSilinecek."
    second_text = "İlke 5\nYeni içerik."
    first = extract_document_sections("943", first_text)
    second = extract_document_sections("943", second_text)

    await store.store_document(StoredDocument(document_id="943", title="Test", markdown_content=first_text))
    await store.replace_document_sections(
        "943",
        first,
        source_content_hash=hashlib.sha256(first_text.encode()).hexdigest(),
    )
    await store.store_document(StoredDocument(document_id="943", title="Test", markdown_content=second_text))
    await store.replace_document_sections(
        "943",
        second,
        source_content_hash=hashlib.sha256(second_text.encode()).hexdigest(),
    )

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


async def test_document_and_sections_publish_in_one_transaction(store, monkeypatch):
    original = StoredDocument(
        document_id="atomic_sections",
        title="Original",
        markdown_content="İlke 5\nEski içerik.",
    )
    await store.store_document(original)
    section_failure = MagicMock(side_effect=RuntimeError("synthetic section failure"))
    monkeypatch.setattr("bddk_mcp.store.doc_store.replace_sections_sync", section_failure)

    with pytest.raises(RuntimeError, match="synthetic section failure"):
        await store.store_document(
            StoredDocument(
                document_id="atomic_sections",
                title="Replacement",
                markdown_content="İlke 5\nYeni içerik.",
            )
        )

    stored = await store.get_document("atomic_sections")
    assert stored is not None
    assert stored.title == "Original"
    assert stored.markdown_content == original.markdown_content


async def test_store_document_empty_content_clears_sections(store):
    first = StoredDocument(document_id="auto_sections", title="Doc", markdown_content="İlke 5\nEski içerik.")
    empty = StoredDocument(document_id="auto_sections", title="Doc", markdown_content="")

    await store.store_document(first)
    await store.store_document(empty)

    assert await store.get_document_section("auto_sections") == []


async def test_filling_an_empty_body_archives_no_version(store):
    await store.store_document(StoredDocument(document_id="x1", title="Doc", markdown_content=""))
    await store.store_document(StoredDocument(document_id="x1", title="Doc", markdown_content="MADDE 1 – Metin."))

    assert await store.get_document_history("x1") == []
    assert await store.get_version_counts(["x1"]) == {}


async def test_search_document_sections(store):
    text = "MADDE 9 - TFRS 9 karşılık\nBankalar karşılık ayırır.\n\nMADDE 10\nBaşka hüküm."
    await store.store_document(StoredDocument(document_id="mevzuat_22599", title="Test", markdown_content=text))
    await store.replace_document_sections(
        "mevzuat_22599",
        extract_document_sections("mevzuat_22599", text),
        source_content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )

    hits = await store.search_document_sections("TFRS 9 karşılık", document_id="mevzuat_22599")

    assert len(hits) >= 1
    assert hits[0].doc_id == "mevzuat_22599"
    assert hits[0].section_type == "madde"
    assert hits[0].section_ref == "9"
    # Document-level quality comes from the whole body, assessed once per content hash.
    expected_quality = assess_markdown_quality(text, document_id="mevzuat_22599")
    assert hits[0].document_quality == expected_quality
    assert ("mevzuat_22599", hashlib.sha256(text.encode()).hexdigest(), None) in store._document_quality_cache
    again = await store.search_document_sections("TFRS 9 karşılık", document_id="mevzuat_22599")
    assert again[0].document_quality == expected_quality
    assert again[0].document_quality is not hits[0].document_quality
    # Candidate retrieval skips the body substring; hydration restores the exact range.
    candidates = await store.search_document_sections(
        "TFRS 9 karşılık", document_id="mevzuat_22599", include_source_range=False
    )
    assert candidates[0].normalized_source_range == ""
    hydrated = await store.hydrate_source_ranges(candidates)
    assert hits[0].normalized_source_range != ""
    assert hydrated[0].normalized_source_range == hits[0].normalized_source_range


async def test_search_document_sections_skips_nested_fikra_unless_requested(store):
    text = "MADDE 9 - TFRS 9 karşılık\n(1) Bankalar karşılık ayırır.\n\nMADDE 10\nBaşka hüküm."
    await store.store_document(StoredDocument(document_id="mevzuat_22599", title="Test", markdown_content=text))
    await store.replace_document_sections(
        "mevzuat_22599",
        extract_document_sections("mevzuat_22599", text),
        source_content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )

    hits = await store.search_document_sections("karşılık ayırır", document_id="mevzuat_22599")
    assert hits
    assert all(hit.section_type != "fikra" for hit in hits)

    fikra_hits = await store.search_document_sections(
        "karşılık ayırır", document_id="mevzuat_22599", section_type="fikra"
    )
    assert any(hit.section_type == "fikra" for hit in fikra_hits)


async def test_search_document_sections_includes_govde_by_default_and_flags_its_kind(store):
    preamble = "Önsöz likidite yeterlilik kalanı. " * 8
    text = preamble + "\n\nMADDE 13 - Asgari likidite yeterlilik oranı yüzde yüzden az olamaz.\n"
    await store.store_document(StoredDocument(document_id="mevzuat_10749", title="Test", markdown_content=text))
    await store.replace_document_sections(
        "mevzuat_10749",
        extract_document_sections("mevzuat_10749", text),
        source_content_hash=hashlib.sha256(text.encode()).hexdigest(),
    )

    hits = await store.search_document_sections("likidite yeterlilik", document_id="mevzuat_10749")
    assert hits
    # Every govde row is searched by default; the preamble of an article-structured document is
    # remainder, so none of the kind flags is set on it.
    [preamble] = [hit for hit in hits if hit.section_type == "govde"]
    assert (
        preamble.document_structureless,
        preamble.document_decision_window,
        preamble.document_annex_or_paragraph_only,
    ) == (False, False, False)
    assert any(hit.section_type == "madde" and hit.section_ref == "13" for hit in hits)

    govde_hits = await store.search_document_sections(
        "likidite yeterlilik", document_id="mevzuat_10749", section_type="govde"
    )
    assert any(hit.section_type == "govde" for hit in govde_hits)


async def test_upsert(store, sample_doc):
    await store.store_document(sample_doc)
    updated = sample_doc.model_copy(update={"title": "Güncellenmiş Başlık"})
    await store.store_document(updated)
    doc = await store.get_document("1291")
    assert doc.title == "Güncellenmiş Başlık"


async def test_truncation_notice_words_are_not_searchable_but_the_stored_content_keeps_them(store):
    body = "Söz konusu yükümlülük yerine getirilir ve kayıt altına alınır. " * 500
    text = f"MADDE 1 - Yükümlülük\n{body}\n\nMADDE 2 - Yürürlük\nBu yönetmelik yayımı tarihinde yürürlüğe girer.\n"
    await store.store_document(StoredDocument(document_id="mevzuat_9", title="Uzun", markdown_content=text))
    [capped] = await store.get_document_section("mevzuat_9", section_type="madde", section_ref="1")
    assert "[BÖLÜM KESİLDİ:" in capped.content
    assert capped.content_hash == hashlib.sha256(capped.content.encode("utf-8")).hexdigest()

    assert await store.search_document_sections("karakterden kısaltıldı", document_id="mevzuat_9") == []
    assert await store.search_document_sections("get_bddk_document", document_id="mevzuat_9") == []
    hits = await store.search_document_sections("yükümlülük kayıt altına", document_id="mevzuat_9")
    # The capped madde and the govde windows that continue it; none contains the notice words.
    assert ("madde", "1") in [(hit.section_type, hit.section_ref) for hit in hits]
    assert {hit.section_type for hit in hits} == {"madde", "govde"}


def test_prefix_stem_query_keeps_every_token_quoted_and_a_five_character_floor():
    from bddk_mcp.store.doc_store import fts_prefix_stem_query

    assert fts_prefix_stem_query("Gecikme süresi") == '"gecik"* "suresi"*'
    assert fts_prefix_stem_query("sınıflandırılması") == '"siniflandir"*'
    assert fts_prefix_stem_query("geçici") == '"gecici"*'
    # Short alphabetic tokens are prefixes, numbers stay exact; operators in user text are only words.
    assert fts_prefix_stem_query("grup 12345 NEAR OR süresinin") == '"grup"* "12345" "near"* "or"* "sures"*'
    assert fts_prefix_stem_query('"kredilerin" -faiz* (AND)') == '"kredi"* "faiz"* "and"*'
    assert fts_prefix_stem_query("yüzde on 90 gün") == '"yuzde"* "on"* "90" "gun"*'
    # Nothing to widen: the strict expression would be the same.
    assert fts_prefix_stem_query("2024 a 15") is None
    assert fts_prefix_stem_query("") is None


def test_prefix_stem_removes_one_turkish_ending_above_a_five_character_root():
    from bddk_mcp.store.doc_store import fts_prefix_stem_query

    cases = {
        "denetlenmesi": '"denet"*',
        "yedeklenmesi": '"yedek"*',
        "bulundurulması": '"bulundur"*',
        "saklanması": '"sakla"*',
        "denetimi": '"denet"*',
        "sistemlerinin": '"sistem"*',
        "yapılandırılmış": '"yapilandir"*',
        "uzatılması": '"uzatil"*',
        # No known ending, or the root would be shorter than five characters: the token is not cut.
        "birincil": '"birincil"*',
        "ikincil": '"ikincil"*',
        "testleri": '"testleri"*',
        "süresi": '"suresi"*',
    }
    assert {word: fts_prefix_stem_query(word) for word in cases} == cases


@pytest.mark.asyncio
async def test_prefix_stem_finds_a_root_form_and_keeps_birincil_apart_from_birinci(doc_store):
    from bddk_mcp.store.doc_store import StoredDocument

    await doc_store.store_document(
        StoredDocument(
            document_id="mevzuat_1",
            title="T",
            markdown_content=(
                "MADDE 9 - Destek hizmeti kuruluşlarının denetimi\nKurum denetim yapar.\n\n"
                "MADDE 10 - Birinci grup\nBirinci ve ikinci grup krediler.\n"
            ),
        )
    )

    hits = await doc_store.search_document_sections("kuruluşlarının denetlenmesi", prefix_stem=True)
    assert [(hit.section_type, hit.section_ref) for hit in hits] == [("madde", "9")]
    assert await doc_store.search_document_sections("birincil ikincil", prefix_stem=True) == []


@pytest.mark.asyncio
async def test_prefix_stem_section_search_matches_inflected_forms(doc_store):
    from bddk_mcp.store.doc_store import StoredDocument

    await doc_store.store_document(
        StoredDocument(document_id="1", title="T", markdown_content="MADDE 1 - Süre\nGecikme süresinin uzatılması.\n")
    )

    assert await doc_store.search_document_sections("gecikme süresi") == []
    hits = await doc_store.search_document_sections("gecikme süresi", prefix_stem=True)
    assert [(h.doc_id, h.section_type, h.section_ref) for h in hits] == [("1", "madde", "1")]
    assert await doc_store.search_document_sections("faiz", prefix_stem=True) == []


def test_prefix_stem_removes_an_aorist_verb_ending_above_a_five_character_root():
    from bddk_mcp.store.doc_store import fts_prefix_stem_query

    cases = {
        "seçilir": '"secil"*',
        "hesaplanır": '"hesaplan"*',
        "oluşturulur": '"olustur"*',
        "değerlenir": '"degerlen"*',
        "belirlenir": '"belirlen"*',
        # The root floor still holds: a short word is not cut.
        "takdir": '"takdir"*',
        "karar": '"karar"*',
    }
    for token, expected in cases.items():
        assert fts_prefix_stem_query(token) == expected, token


async def test_the_generic_govde_heading_is_not_indexed_but_govde_content_is(store):
    from bddk_mcp.store.section_index import GOVDE_HEADING

    text = "Kurul kararı ile kredi kartı borçlarının yeniden yapılandırılmasına karar verilmiştir.\n"
    await store.store_document(StoredDocument(document_id="1301", title="Kurul Kararı", markdown_content=text))
    [govde] = await store.get_document_section("1301", section_type="govde")
    assert govde.heading == GOVDE_HEADING

    for words in ("yapısal başlık yok", "dipnot", "gövde kalanı", "başlık"):
        assert await store.search_document_sections(words) == []
    heading_norms = await store._db.run(
        lambda conn: conn.execute("SELECT heading_norm FROM document_sections WHERE doc_id = '1301'").fetchall()
    )
    assert [row["heading_norm"] for row in heading_norms] == [""]
    [hit] = await store.search_document_sections("yeniden yapılandırılmasına")
    assert (hit.doc_id, hit.section_type, hit.heading) == ("1301", "govde", GOVDE_HEADING)


@pytest.mark.parametrize("typed", ["BSD'nin kapsamı", "BSD’nin kapsamı", "bsd‘nin KAPSAMI", "BSDʼnin kapsamı"])
async def test_heading_filter_reads_apostrophe_variants_as_one_character(store, typed):
    text = (
        "MADDE 21 - BSD’nin tanımı\nBirinci metin.\n\nMADDE 22 - BSD’nin kapsamı\nİkinci metin.\n\n"
        "MADDE 23 - Banka'nın yükümlülüğü\nÜçüncü metin.\n"
    )
    await store.store_document(StoredDocument(document_id="mevzuat_39257", title="Test", markdown_content=text))

    [section] = await store.get_document_section("mevzuat_39257", heading=typed)
    assert (section.section_type, section.section_ref, section.heading) == ("madde", "22", "BSD’nin kapsamı")
    [ascii_stored] = await store.get_document_section("mevzuat_39257", heading="Banka’nın")
    assert ascii_stored.section_ref == "23"
    assert await store.get_document_section("mevzuat_39257", heading="BSD nin kapsamı") == []
