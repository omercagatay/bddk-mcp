"""expand_references flag on search_document_sections."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest

from bddk_mcp.regulatory.bridge import ensure_section_provision_map, refresh_section_provision_map
from bddk_mcp.regulatory.legal_versions import (
    LegalVersionBundle,
    Provision,
    ProvisionOccurrence,
    canonical_bundle_sha256,
)
from bddk_mcp.regulatory.relations import RegulatoryRelation, import_relations
from bddk_mcp.regulatory.repository import import_legal_version_bundle
from bddk_mcp.store.doc_store import StoredDocument
from bddk_mcp.tools import sections as sections_module
from tests.test_graph_queries import _HUMAN_OK
from tests.test_regulatory_legal_versions import _evidence, make_fixture_bundle
from tests.test_regulatory_repository import regulatory_pool  # noqa: F401

pytestmark = pytest.mark.asyncio


def _capture_tools(deps):
    captured = {}

    class _StubMCP:
        def tool(self, *args, **kwargs):
            def decorator(fn):
                captured[fn.__name__] = fn
                return fn

            return decorator

    sections_module.register(_StubMCP(), deps)
    return captured


def _doc_943_two_sections() -> StoredDocument:
    """Doc 943 whose section index yields (ilke, 5) AND (ilke, 6) rows."""
    return StoredDocument(
        document_id="943",
        title="TFRS 9 Uygulama Rehberi",
        category="Rehber",
        source_url="https://www.bddk.org.tr/example/943.pdf",
        markdown_content=(
            "# TFRS 9 Uygulama Rehberi\n\n"
            "İlke 5 — Model validasyonu\n"
            "Banka, beklenen kredi zararı modellerini bağımsız olarak valide eder.\n\n"
            "İlke 6 — Model izleme\n"
            "Banka, model performansını düzenli olarak izler ve raporlar.\n"
        ),
        extraction_method="markitdown",
    )


def _bundle_with_ilke6() -> LegalVersionBundle:
    """Shared fixture bundle extended with a second mapped provision (ilke/6).

    Built on make_fixture_bundle() via dataclasses.replace so the shared
    fixture (and the counts earlier tests assert on) stays untouched.
    """
    base = make_fixture_bundle()
    ilke6 = Provision(
        provision_id="prov-943-ilke-6",
        instrument_id="inst-tfrs9",
        kind="ilke",
        canonical_path="ilke/6",
    )
    v1 = base.versions[0]
    occurrence = ProvisionOccurrence(
        legal_version_id=v1.legal_version_id,
        provision_id="prov-943-ilke-6",
        normalized_text_sha256="9" * 64,
        evidence=_evidence("ev-ilke-6"),
    )
    v1 = dataclasses.replace(v1, provisions=(*v1.provisions, occurrence))
    draft = dataclasses.replace(
        base,
        versions=(v1, *base.versions[1:]),
        provisions=(*base.provisions, ilke6),
    )
    return dataclasses.replace(draft, bundle_sha256=canonical_bundle_sha256(draft))


def _amends_edge() -> RegulatoryRelation:
    return RegulatoryRelation(
        relation_type="amends",
        source_instrument_id="inst-tfrs9",
        source_provision_id="prov-943-ilke-6",
        target_instrument_id="inst-tfrs9",
        target_provision_id="prov-943-ilke-5",
        target_external_ref=None,
        evidence=_evidence("ev-edge-expand-1"),
        extraction_method="manual",
        confidence=1.0,
        validation=_HUMAN_OK,
    )


def _deps(pool, store) -> MagicMock:
    deps = MagicMock()
    deps.pool = pool
    deps.doc_store = store
    return deps


@pytest.fixture
async def seeded_section_deps(regulatory_pool, doc_store_factory):  # noqa: F811
    """Store doc 943 (ilke 5 + 6), import bundle + validated edge, refresh map."""
    store = await doc_store_factory(regulatory_pool)
    await store.store_document(_doc_943_two_sections())
    await import_legal_version_bundle(
        regulatory_pool, _bundle_with_ilke6(), imported_by="test-suite", allow_fixture=True
    )
    await import_relations(regulatory_pool, [_amends_edge()], imported_by="test-suite")
    await ensure_section_provision_map(regulatory_pool)
    await refresh_section_provision_map(regulatory_pool)
    return _deps(regulatory_pool, store)


@pytest.fixture
async def unmapped_section_deps(regulatory_pool, doc_store_factory):  # noqa: F811
    """Same store shape, but no regulatory rows at all (map refreshed empty)."""
    store = await doc_store_factory(regulatory_pool)
    await store.store_document(
        StoredDocument(
            document_id="mevzuat_22599",
            title="Karşılıklar Yönetmeliği",
            category="Yönetmelik",
            source_url="https://mevzuat.gov.tr/example/22599",
            markdown_content=(
                "# Karşılıklar Yönetmeliği\n\nMADDE 9 – Karşılıklar\nBanka, TFRS 9 kapsamında karşılık ayırır.\n"
            ),
            extraction_method="markitdown",
        )
    )
    await ensure_section_provision_map(regulatory_pool)
    await refresh_section_provision_map(regulatory_pool)
    return _deps(regulatory_pool, store)


async def test_expand_off_is_unchanged(regulatory_pool, seeded_section_deps):  # noqa: F811
    tools = _capture_tools(seeded_section_deps)
    plain = await tools["search_document_sections"](query="İlke 5 model validasyonu")
    assert "İlişkili bölümler" not in plain


async def test_expand_appends_labeled_neighbors(regulatory_pool, seeded_section_deps):  # noqa: F811
    tools = _capture_tools(seeded_section_deps)
    expanded = await tools["search_document_sections"](query="İlke 5 model validasyonu", expand_references=True)
    assert "İlişkili bölümler (doğrulanmış kenarlar)" in expanded
    assert "kenar: amends" in expanded


async def test_expand_flag_off_output_is_byte_identical(regulatory_pool, seeded_section_deps):  # noqa: F811
    tools = _capture_tools(seeded_section_deps)
    plain = await tools["search_document_sections"](query="İlke 5 model validasyonu")
    explicit_off = await tools["search_document_sections"](query="İlke 5 model validasyonu", expand_references=False)
    assert explicit_off == plain


async def test_expand_never_inlines_neighbor_content(regulatory_pool, seeded_section_deps):  # noqa: F811
    tools = _capture_tools(seeded_section_deps)
    expanded = await tools["search_document_sections"](
        query="İlke 5 model validasyonu", document_id="943", section_type="ilke", expand_references=True
    )
    # Pointer lines only: neighbor label appears, neighbor body text does not
    # get inlined by the expansion (it may appear only if ilke 6 is itself a hit).
    assert "(kenar: amends)" in expanded
    for line in expanded.splitlines():
        if line.startswith("- 943 — ilke"):
            assert "izler" not in line and "valide" not in line


async def test_expand_is_noop_without_graph_coverage(regulatory_pool, unmapped_section_deps):  # noqa: F811
    tools = _capture_tools(unmapped_section_deps)
    expanded = await tools["search_document_sections"](query="madde 9", expand_references=True)
    assert "İlişkili bölümler" not in expanded  # silent no-op, search output unchanged
    assert "mevzuat_22599" in expanded


async def test_expand_degrades_to_plain_search_on_failure(regulatory_pool, seeded_section_deps):  # noqa: F811
    class _BrokenPool:
        async def fetch(self, *args, **kwargs):
            raise RuntimeError("graph query exploded")

        async def execute(self, *args, **kwargs):
            return "SELECT 0"

    deps = _deps(_BrokenPool(), seeded_section_deps.doc_store)
    tools = _capture_tools(deps)
    out = await tools["search_document_sections"](query="İlke 5 model validasyonu", expand_references=True)
    assert "İlişkili bölümler" not in out
    assert "943" in out
