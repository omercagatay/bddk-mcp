"""Bridge between document_sections and regulatory_provisions."""

from __future__ import annotations

import pytest

from bddk_mcp.regulatory.bridge import (
    canonical_provision_path,
    ensure_section_provision_map,
    refresh_section_provision_map,
    sql_canonical_provision_path,
)
from bddk_mcp.regulatory.repository import import_legal_version_bundle
from bddk_mcp.store.doc_store import DocumentStore, StoredDocument
from tests.test_regulatory_legal_versions import make_fixture_bundle
from tests.test_regulatory_repository import regulatory_pool  # noqa: F401


def _sample_doc_943() -> StoredDocument:
    """Document whose section index yields an (ilke, 5) row for doc_id 943."""
    return StoredDocument(
        document_id="943",
        title="TFRS 9 Uygulama Rehberi",
        category="Rehber",
        source_url="https://www.bddk.org.tr/example/943.pdf",
        markdown_content=(
            "# TFRS 9 Uygulama Rehberi\n\n"
            "İlke 5 — Model validasyonu\n"
            "Banka, beklenen kredi zararı modellerini bağımsız olarak valide eder.\n"
        ),
        extraction_method="markitdown",
    )


def test_canonical_path_normalization():
    assert canonical_provision_path("madde", "9") == "madde/9"
    assert canonical_provision_path("Madde", "9/A") == "madde/9-a"
    assert canonical_provision_path("İlke", "5") == "ilke/5"
    assert canonical_provision_path("ILKE", "5") == "ilke/5"  # Turkish dotless-I fold
    assert canonical_provision_path("ek", "2") == "ek/2"


def test_canonical_path_matches_section_index_stored_form():
    """section_index stores lettered refs fused ("MADDE 9A" → ref "9a", no slash);
    the fused form must land on the same path as the producer spelling "9/A"."""
    assert canonical_provision_path("madde", "9a") == "madde/9-a"
    assert canonical_provision_path("madde", "9a") == canonical_provision_path("Madde", "9/A")
    # Kinds stored by the section index are ASCII ("fikra", not "fıkra");
    # Turkish spellings must land on the stored ASCII form.
    assert canonical_provision_path("Fıkra", "3") == "fikra/3"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("section_type", "section_ref"),
    [
        ("madde", "9"),
        ("madde", "9a"),  # fused stored form
        ("madde", "9s"),  # discriminator: '[/\\s]+' misread as literal would eat the 's'
        ("Madde", "9/A"),
        ("İlke", "5"),
        ("ILKE", "5"),
        ("Fıkra", "3"),
        ("madde", "9 a"),
    ],
)
async def test_sql_expression_matches_python_normalization(regulatory_pool, section_type, section_ref):  # noqa: F811
    """The view's join expression and canonical_provision_path are one normalization."""
    sql_value = await regulatory_pool.fetchval(
        f"SELECT {sql_canonical_provision_path('$1::text', '$2::text')}",
        section_type,
        section_ref,
    )
    assert sql_value == canonical_provision_path(section_type, section_ref)


@pytest.mark.asyncio
async def test_map_links_section_to_provision(regulatory_pool, doc_store_factory):  # noqa: F811
    """A stored section resolves to the fixture bundle's provision through the map."""
    store = await doc_store_factory(regulatory_pool)
    await store.store_document(_sample_doc_943())

    # Sanity: the section row the map joins against must actually exist.
    section_row = await regulatory_pool.fetchrow(
        "SELECT section_type, section_ref FROM document_sections"
        " WHERE doc_id = '943' AND section_type = 'ilke' AND section_ref = '5'"
    )
    assert section_row is not None, "expected document_sections row (943, ilke, 5)"

    bundle = make_fixture_bundle()  # artifact.repository_document_id == "943", provision ilke/5
    await import_legal_version_bundle(regulatory_pool, bundle, imported_by="test-suite", allow_fixture=True)

    await ensure_section_provision_map(regulatory_pool)
    await refresh_section_provision_map(regulatory_pool)

    rows = await regulatory_pool.fetch(
        "SELECT doc_id, section_type, section_ref, provision_id, instrument_id"
        " FROM regulatory_section_provision_map WHERE doc_id = '943'"
    )
    match = [dict(row) for row in rows if row["provision_id"] == "prov-943-ilke-5"]
    assert match, f"expected ilke/5 mapping, got {rows}"
    assert match[0]["instrument_id"] == "inst-tfrs9"


@pytest.mark.asyncio
async def test_unmapped_document_has_no_rows(regulatory_pool):  # noqa: F811
    # The view joins document_sections; make sure the doc tables exist even
    # on a fresh database before creating the materialized view.
    await DocumentStore(regulatory_pool).initialize()

    await ensure_section_provision_map(regulatory_pool)
    await refresh_section_provision_map(regulatory_pool)
    count = await regulatory_pool.fetchval(
        "SELECT count(*) FROM regulatory_section_provision_map WHERE doc_id = 'no-such-doc'"
    )
    assert count == 0
