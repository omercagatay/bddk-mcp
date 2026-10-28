"""Rule-based proposal extraction: fixture text, the real mevzuat_22599 text, and the committed file."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from bddk_mcp.ontology.proposals import (
    ProposalFileError,
    ProposalFlag,
    ProposalKind,
    ProposalSource,
    TargetKind,
    dump_proposal_file,
    load_proposal_file,
    sha256_text,
)
from bddk_mcp.ontology.rule_extract import (
    RULE_ARTICLE_CLAUSE,
    RULE_INLINE_TAG,
    RULE_RG_TABLE,
    RuleExtractionError,
    article_spans,
    build_proposal_file,
    extract_inline_tags,
    extract_rg_table,
)
from bddk_mcp.ontology.spec import DEFAULT_SPEC_PATH, load_ontology_spec

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_22599_link_proposals.py"
COMMITTED = REPO_ROOT / "docs" / "evidence" / "legal-versions" / "22599.proposals.json"
SPEC_SHA = hashlib.sha256(DEFAULT_SPEC_PATH.read_bytes()).hexdigest()

FIXTURE = """Örnek Yönetmelik

Amaç

MADDE 1 – (1) Bu Yönetmeliğin amacı örnek metin sağlamaktır.

Sınıflandırma

MADDE 2 – (1) (Ek:RG-6/7/2021-31533)$^{(2)}$ Bankalar kredilerini izler.

(2) (Mülga:RG-15/8/2018-30510)

Teminatlar

MADDE 3 – (Değişik ibare:RG-20/1/2016-294599) Teminatlar dikkate alınır.

Bilinmeyen değişiklik

MADDE 4 – (1) (Ek:RG-1/1/2020-99999) Tabloda bulunmayan bir değişiklik.

Yürürlükten kaldırılan yönetmelik

MADDE 5 – (1) 1/11/2006 tarihli ve 26333 sayılı Resmî Gazete’de yayımlanan Eski Örnek Yönetmelik yürürlükten kaldırılmıştır.

Yürürlük

MADDE 6 – (Değişik:RG-14/12/2016-29918)$^{(1)}$

(1) Bu Yönetmelik 1/1/2018 tarihinde yürürlüğe girer.

Geçiş hükmü

GEÇİCİ MADDE 1 – (Başlığı ile Birlikte Değişik:RG-14/12/2016-29918) Geçiş süresi tanınır.

_______

$^{(1)}$ Bu değişiklik 1/1/2017 tarihinde yürürlüğe girer.

|  | Yönetmeliğin Yayımlandığı Resmî Gazete’nin | Yönetmeliğin Yayımlandığı Resmî Gazete’nin |
|---|---|---|
|  | Tarihi | Sayısı |
|  | 22/6/2016 | 29750 |
|  | Yönetmelikte Değişiklik Yapan Yönetmeliklerin Yayımlandığı Resmî Gazetelerin | Yönetmelikte Değişiklik Yapan Yönetmeliklerin Yayımlandığı Resmî Gazetelerin |
|  | Tarihi | Sayısı |
| 1 | 14/12/2016 | 29918 |
| 2 | 15/8/2018 | 30510 |
| 3 | 6/7/2021 | 31533 |
"""
FIXTURE_ID = "fixture_doc"


def _build_fixture():
    return build_proposal_file(
        document_id=FIXTURE_ID,
        content_hash=sha256_text(FIXTURE),
        title="Örnek Yönetmelik",
        text=FIXTURE,
        ontology_spec_sha256=SPEC_SHA,
    )


def _seed_document() -> dict:
    documents = json.loads((REPO_ROOT / "seed_data" / "documents.json").read_text(encoding="utf-8"))
    return next(item for item in documents if item["document_id"] == "mevzuat_22599")


def _load_script():
    spec = importlib.util.spec_from_file_location("build_22599_link_proposals", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --- fixture text ------------------------------------------------------------


def test_rg_table_parses_original_and_numbered_amendments():
    table = extract_rg_table(FIXTURE)
    assert table.original.rg.key() == "2016-06-22-29750"
    assert [row.rg.number for row in table.amendments] == ["29918", "30510", "31533"]
    assert [row.ordinal for row in table.amendments] == [1, 2, 3]
    for row in (table.original, *table.amendments):
        assert FIXTURE[row.evidence.start_char : row.evidence.end_char] == row.evidence.quote
        assert row.evidence.quote.startswith("|")


def test_inline_tags_keep_the_footnote_marker_outside_the_quote():
    tags = extract_inline_tags(FIXTURE)
    kinds = [tag.kind for tag in tags]
    assert kinds == ["Ek", "Mülga", "Değişik ibare", "Ek", "Değişik", "Başlığı ile Birlikte Değişik"]
    for tag in tags:
        quote = FIXTURE[tag.start : tag.end]
        assert quote.startswith("(") and quote.endswith(")")
        assert "$" not in quote


def test_article_spans_use_the_section_parser_paths():
    paths = [span.path for span in article_spans(FIXTURE_ID, FIXTURE)]
    assert paths == ["madde/1", "madde/2", "madde/3", "madde/4", "madde/5", "madde/6", "gecici_madde/1"]


def test_fixture_proposals_cover_table_tags_repeal_and_events():
    file = _build_fixture()
    by_rule = file.counts.by_rule
    assert by_rule == {RULE_ARTICLE_CLAUSE: 2, RULE_INLINE_TAG: 6, RULE_RG_TABLE: 4}
    assert file.counts.total == 12
    assert file.counts.links == 10 and file.counts.events == 2
    assert all(item.source is ProposalSource.RULE for item in file.proposals)


def test_tags_are_cross_confirmed_against_the_table_and_malformed_numbers_are_flagged():
    file = _build_fixture()
    tags = {item.evidence.quote: item for item in file.proposals if item.rule_id == RULE_INLINE_TAG}
    confirmed = tags["(Ek:RG-6/7/2021-31533)"]
    assert ProposalFlag.CROSS_CONFIRMED in confirmed.flags
    assert confirmed.confidence >= 0.9
    assert confirmed.subject_provision == "madde/2"
    assert confirmed.change_kind == "Ek"
    assert confirmed.target.kind is TargetKind.RG_INSTRUMENT and confirmed.target.rg.number == "31533"

    malformed = tags["(Değişik ibare:RG-20/1/2016-294599)"]
    assert ProposalFlag.RG_NUMBER_MALFORMED in malformed.flags
    assert ProposalFlag.NOT_IN_RG_TABLE in malformed.flags
    assert malformed.confidence <= 0.3

    unknown = tags["(Ek:RG-1/1/2020-99999)"]
    assert ProposalFlag.NOT_IN_RG_TABLE in unknown.flags
    assert ProposalFlag.CROSS_CONFIRMED not in unknown.flags
    assert unknown.confidence < confirmed.confidence
    assert unknown.subject_provision == "madde/4"

    transitional = tags["(Başlığı ile Birlikte Değişik:RG-14/12/2016-29918)"]
    assert transitional.subject_provision == "gecici_madde/1"


def test_table_rows_echoed_by_tags_are_cross_confirmed():
    file = _build_fixture()
    rows = {
        item.target.rg.number: item
        for item in file.proposals
        if item.rule_id == RULE_RG_TABLE and item.kind is ProposalKind.LINK
    }
    assert set(rows) == {"29918", "30510", "31533"}
    assert all(ProposalFlag.CROSS_CONFIRMED in item.flags for item in rows.values())
    assert all(item.subject_provision is None and item.link_type == "amends" for item in rows.values())


def test_repeal_and_entry_into_force_clauses_need_review():
    file = _build_fixture()
    clauses = [item for item in file.proposals if item.rule_id == RULE_ARTICLE_CLAUSE]
    repeal = next(item for item in clauses if item.kind is ProposalKind.LINK)
    effective = next(item for item in clauses if item.kind is ProposalKind.EVENT)
    assert repeal.link_type == "repeals"
    assert repeal.subject_provision == "madde/5"
    assert repeal.target.kind is TargetKind.EXTERNAL_REF
    assert (
        repeal.target.external_ref
        == "1/11/2006 tarihli ve 26333 sayılı Resmî Gazete'de yayımlanan Eski Örnek Yönetmelik"
    )
    assert repeal.evidence.quote.endswith("yürürlükten kaldırılmıştır.")
    assert ProposalFlag.NEEDS_REVIEW in repeal.flags
    assert effective.event_type == "effective"
    assert effective.target.value_date.isoformat() == "2018-01-01"
    assert effective.subject_provision == "madde/6"
    assert ProposalFlag.NEEDS_REVIEW in effective.flags
    publication = next(item for item in file.proposals if item.event_type == "publication")
    assert publication.target.value_date.isoformat() == "2016-06-22"
    assert ProposalFlag.NEEDS_REVIEW in publication.flags
    assert publication.subject_provision is None


def test_every_quote_is_an_exact_slice_and_ids_are_deterministic_and_unique():
    first = _build_fixture()
    second = _build_fixture()
    assert dump_proposal_file(first) == dump_proposal_file(second)
    ids = [item.proposal_id for item in first.proposals]
    assert len(set(ids)) == len(ids)
    for item in first.proposals:
        assert FIXTURE[item.evidence.start_char : item.evidence.end_char] == item.evidence.quote
        assert item.evidence.statement_sha256 == sha256_text(item.evidence.quote)
        assert "\n" not in item.evidence.quote  # a table row quote must not swallow its line break
    positions = [item.evidence.start_char for item in first.proposals]
    assert positions == sorted(positions)


def test_proposals_use_only_spec_vocabulary():
    spec = load_ontology_spec()
    file = _build_fixture()
    assert {item.source.value for item in file.proposals} <= set(spec.link_sources)
    link_types = {item.link_type for item in file.proposals if item.link_type}
    assert link_types <= spec.stored_relation_types()
    event_types = {item.event_type for item in file.proposals if item.event_type}
    assert event_types <= set(spec.object_types["legal_version"].lifecycle_events)
    assert all(item.rule_id.startswith(("rg_table", "inline_tag", "article_clause")) for item in file.proposals)


def test_extraction_refuses_a_text_that_does_not_match_its_hash_or_lacks_the_table():
    with pytest.raises(RuleExtractionError):
        build_proposal_file(
            document_id=FIXTURE_ID, content_hash="0" * 64, title="x", text=FIXTURE, ontology_spec_sha256=SPEC_SHA
        )
    no_table = FIXTURE.split("|  | Yönetmeliğin")[0]
    with pytest.raises(RuleExtractionError):
        build_proposal_file(
            document_id=FIXTURE_ID,
            content_hash=sha256_text(no_table),
            title="x",
            text=no_table,
            ontology_spec_sha256=SPEC_SHA,
        )


def test_file_round_trips_and_tampering_is_rejected(tmp_path):
    file = _build_fixture()
    path = tmp_path / "proposals.json"
    path.write_text(dump_proposal_file(file), encoding="utf-8")
    assert load_proposal_file(path) == file

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["proposals"][0]["evidence"]["quote"] = tampered["proposals"][0]["evidence"]["quote"] + " "
    path.write_text(json.dumps(tampered, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ProposalFileError):
        load_proposal_file(path)

    reflagged = json.loads(dump_proposal_file(file))
    reflagged["proposals"][0]["confidence"] = 0.1
    reflagged["proposals"][0]["flags"] = ["needs_review"]
    path.write_text(json.dumps(reflagged, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ProposalFileError):
        load_proposal_file(path)  # counts no longer match the proposals


# --- the real mevzuat_22599 text ---------------------------------------------


def test_real_text_lists_seven_amendments_and_every_tagged_article():
    document = _seed_document()
    file = build_proposal_file(
        document_id="mevzuat_22599",
        content_hash=document["content_hash"],
        title=document["title"],
        text=document["markdown_content"],
        ontology_spec_sha256=SPEC_SHA,
    )
    assert file.rg_table.original.rg.key() == "2016-06-22-29750"
    assert [row.rg.number for row in file.rg_table.amendments] == [
        "29918",
        "30373",
        "30510",
        "30569",
        "30961",
        "31533",
        "31845",
    ]
    assert file.counts.by_rule == {RULE_ARTICLE_CLAUSE: 2, RULE_INLINE_TAG: 24, RULE_RG_TABLE: 8}
    assert file.counts.total == 34
    tags = [item for item in file.proposals if item.rule_id == RULE_INLINE_TAG]
    assert all(ProposalFlag.CROSS_CONFIRMED in item.flags for item in tags)
    assert file.counts.by_flag.get(ProposalFlag.RG_NUMBER_MALFORMED.value, 0) == 0
    tagged_articles = {item.subject_provision for item in tags}
    assert tagged_articles == {
        "madde/5",
        "madde/6",
        "madde/7",
        "madde/8",
        "madde/13",
        "madde/16",
        "madde/18",
        "madde/19",
        "gecici_madde/1",
        "madde/23",
    }
    repeal = next(item for item in file.proposals if item.link_type == "repeals")
    assert repeal.subject_provision == "madde/22"
    assert "26333" in repeal.target.external_ref
    effective = next(item for item in file.proposals if item.event_type == "effective")
    assert effective.subject_provision == "madde/23"
    assert effective.target.value_date.isoformat() == "2018-01-01"


def test_committed_proposals_file_matches_the_stored_text_and_rules():
    module = _load_script()
    assert dump_proposal_file(module.build()) == COMMITTED.read_text(encoding="utf-8")
    loaded = load_proposal_file(COMMITTED)
    assert loaded.document_id == "mevzuat_22599"
    assert loaded.content_hash == _seed_document()["content_hash"]
    assert loaded.ontology_spec_sha256 == SPEC_SHA
    assert loaded.counts.total == 34


def test_extraction_path_never_touches_a_database():
    sources = [
        (REPO_ROOT / "bddk_mcp" / "ontology" / "proposals.py").read_text(encoding="utf-8"),
        (REPO_ROOT / "bddk_mcp" / "ontology" / "rule_extract.py").read_text(encoding="utf-8"),
        SCRIPT.read_text(encoding="utf-8"),
    ]
    for source in sources:
        assert "asyncpg" not in source
        assert "DATABASE_URL" not in source
        assert "bddk_mcp.regulatory.repository" not in source
        assert "resolve_and_import" not in source
