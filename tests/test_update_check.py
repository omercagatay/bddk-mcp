"""Tests for the read-only update check and the explicit add path (no network: httpx.MockTransport)."""

from __future__ import annotations

import hashlib
import json
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from bddk_mcp import cli
from bddk_mcp.core.models import BddkDecisionSummary
from bddk_mcp.ingest import update_check
from bddk_mcp.ingest.client import BddkApiClient
from bddk_mcp.ingest.update_check import (
    EXIT_ERROR,
    EXIT_NOTHING_TO_REPORT,
    EXIT_SOMETHING_TO_REPORT,
    check_updates,
    compare_catalog,
    entries_to_add,
    format_report,
    load_accepted_findings,
    text_dates,
)
from bddk_mcp.store.corpus_db import CorpusDB
from bddk_mcp.store.doc_store import DocumentStore, StoredDocument


def _entry(document_id: str, category: str = "Kurul Kararı", title: str = "") -> BddkDecisionSummary:
    return BddkDecisionSummary(document_id=document_id, title=title or f"Belge {document_id}", category=category)


def _held(*ids: str) -> list[dict]:
    return [{"document_id": document_id, "title": f"Belge {document_id}", "category": "Rehber"} for document_id in ids]


class TestCompareCatalog:
    def test_selection_pages_report_every_entry_the_catalog_lacks_whatever_its_id(self):
        pages = [(51, [_entry("16", "Banka Kartları"), _entry("mevzuat_11180", "Banka Kartları")], ""), (58, [], "")]
        result = compare_catalog(pages, _held("mevzuat_11180", "1349"))
        assert [item["document_id"] for item in result["new"]] == ["16"]
        assert result["new"][0]["page"] == 51

    def test_decision_pages_report_only_ids_above_the_newest_catalogued_bddk_id(self):
        pages = [
            (56, [_entry("1351"), _entry("1264"), _entry("1349")], ""),
            (55, [_entry("1350"), _entry("1176"), _entry("1333")], ""),
        ]
        result = compare_catalog(pages, _held("1349", "1176", "mevzuat_99999"))
        assert result["newest_catalogued_bddk_id"] == 1349
        assert [item["document_id"] for item in result["new"]] == ["1351"]
        assert [item["document_id"] for item in result["new_on_page_55"]] == ["1350"]
        assert result["older_not_catalogued"] == {"55": 1, "56": 1}

    def test_out_of_scope_entries_and_duplicates_are_reported_once_or_not_at_all(self):
        pages = [
            (50, [_entry("20", "Faizsiz Bankacılık"), _entry("21", "Rehber")], ""),
            (54, [_entry("21", "Rehber")], ""),
            (56, [_entry("30")], ""),
            (55, [_entry("30")], ""),
        ]
        result = compare_catalog(pages, _held("10"))
        assert [(item["document_id"], item["page"]) for item in result["new"]] == [("21", 50), ("30", 56)]
        assert result["new_on_page_55"] == []

    def test_entries_listed_on_no_page_are_named_and_incomplete_when_a_page_failed(self):
        complete = compare_catalog([(50, [_entry("1", "Rehber")], "")], _held("1", "kgk_tms_12"))
        assert [item["document_id"] for item in complete["not_listed"]] == ["kgk_tms_12"]
        assert complete["not_listed_complete"] is True
        partial = compare_catalog([(50, [_entry("1", "Rehber")], ""), (63, [], "TransportError")], _held("1", "2"))
        assert partial["not_listed_complete"] is False


class TestTextDates:
    def test_reads_inline_notes_footnotes_and_amendment_tables(self):
        text = (
            "MADDE 6 – (Değişik:RG-20/8/2015-29451) (1) Metin. (Ek fıkra: 25/10/2017-7051/1 md.) "
            "( İptal ikinci cümle: Anayasa Mahkemesi’nin 14/11/2013 tarihli ve E.: 2013/24 sayılı Kararı ile.)\n"
            "[1] 01.04.2026 tarihli ve 33210 sayılı Resmî Gazete’de yayımlanan Yönetmelik ile değiştirilmiştir.\n"
            "| 2 | 4/2/2022 | 31740 |"
        )
        assert text_dates(text) == {"2015-08-20", "2017-10-25", "2013-11-14", "2026-04-01", "2022-02-04"}

    def test_numbers_that_are_not_dates_are_ignored(self):
        assert text_dates("E.: 2013/24, K.: 2013/133; 7051/1 md.; 12/3/20255") == set()


# -- a fake BDDK and mevzuat site ---------------------------------------------------------------------------------

_PAGE_50 = (
    '<div class="card"><h5>Rehberler (1)</h5><div class="card-body">'
    '<a href="/Mevzuat/DokumanGetir/943">Rehber</a></div></div>'
)
_FLAT = {
    49: '<a href="http://www.mevzuat.gov.tr/Metin1.Aspx?MevzuatKod=1.5.5411">5411 sayılı Bankacılık Kanunu</a>',
    51: '<a href="/Mevzuat/DokumanGetir/16">Kredi Kartı Taksit Sınırları ve Yasakları</a>',
    54: '<a href="/Mevzuat/DokumanGetir/54">BDDK düzenlemesi</a>',
    58: '<a href="/Mevzuat/DokumanGetir/1336">Tebliğ Taslağı</a>',
    63: '<a href="/Mevzuat/DokumanGetir/63">Mülga</a>',
}
_DECISIONS = {
    56: '<a href="/Mevzuat/DokumanGetir/1349">(01.10.2026 - 11581) Kurul Kararı</a>',
    55: '<a href="/Mevzuat/DokumanGetir/1350">(04.11.2026 - 11593) Faaliyet izni Kurul Kararı</a>',
}
_IFRAME = "<html><body><p>MADDE 1 – (Değişik:RG-1/2/2020-31000) Metin.</p>{extra}</body></html>"


def _site(*, sizes: dict[str, int], failing_page: int | None = None, mevzuat_extra: str = "", mevzuat_page: str = ""):
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = urlparse(str(request.url))
        calls.append((request.method, url.path))
        if url.path.startswith("/Mevzuat/Liste/"):
            page = int(url.path.rsplit("/", 1)[1])
            if page == failing_page:
                raise httpx.ConnectError("down", request=request)
            body = _PAGE_50 if page == 50 else _FLAT.get(page) or _DECISIONS[page]
            return httpx.Response(200, text=f"<html><body>{body}</body></html>", request=request)
        if url.path.startswith("/Mevzuat/DokumanGetir/"):
            document_id = url.path.rsplit("/", 1)[1]
            if document_id not in sizes:
                return httpx.Response(404, request=request)
            return httpx.Response(200, headers={"content-length": str(sizes[document_id])}, request=request)
        if url.path == "/anasayfa/MevzuatFihristDetayIframe":
            assert parse_qs(url.query)["MevzuatNo"] == ["5411"]
            return httpx.Response(200, text=mevzuat_page or _IFRAME.format(extra=mevzuat_extra), request=request)
        raise AssertionError(f"unexpected request {request.method} {url.path}")

    return httpx.MockTransport(handler), calls


async def _corpus(doc_store: DocumentStore) -> None:
    for document_id, source_url, size, text in (
        ("943", "https://www.bddk.org.tr/Mevzuat/DokumanGetir/943", 1000, "Rehber metni."),
        ("1349", "https://www.bddk.org.tr/Mevzuat/DokumanGetir/1349", 2000, "Karar metni."),
        (
            "mevzuat_5411",
            "https://www.mevzuat.gov.tr/mevzuat?MevzuatNo=5411&MevzuatTur=1&MevzuatTertip=5",
            3000,
            "MADDE 1 – (Değişik:RG-1/2/2020-31000) Metin.",
        ),
        ("kgk_tms_12", "https://www.kgk.gov.tr/TMS_12.pdf", 4000, "TMS 12"),
    ):
        await doc_store.store_document(
            StoredDocument(
                document_id=document_id,
                title=f"Belge {document_id}",
                source_url=source_url,
                markdown_content=text,
                file_size=size,
            )
        )
    await doc_store.replace_catalog(
        [{"document_id": d, "title": f"Belge {d}"} for d in ("943", "1349", "mevzuat_5411", "54", "63", "kgk_tms_12")]
    )


def _client(transport: httpx.MockTransport, store: DocumentStore) -> BddkApiClient:
    client = BddkApiClient(store=store, http=httpx.AsyncClient(transport=transport))
    client._assert_public_resolution = AsyncMock()
    return client


class TestCheckUpdates:
    async def test_reports_new_entries_changed_sources_and_unlisted_entries(self, doc_store):
        await _corpus(doc_store)
        transport, calls = _site(sizes={"943": 1000, "1349": 2100}, mevzuat_extra="<p>(Ek:RG-5/6/2026-32000)</p>")
        client = _client(transport, doc_store)

        report = await check_updates(doc_store, client, accepted={})

        assert [item["document_id"] for item in report["new"]] == ["16", "1336"]
        assert [item["document_id"] for item in report["new_on_page_55"]] == ["1350"]
        assert [item["document_id"] for item in report["not_listed"]] == ["kgk_tms_12"]
        changed = {item["document_id"]: item["reason"] for item in report["changed"]}
        assert set(changed) == {"1349", "mevzuat_5411"}
        assert "2100" in changed["1349"] and "2000" in changed["1349"]
        assert "2026-06-05" in changed["mevzuat_5411"]
        assert report["not_checked"] == [{"document_id": "kgk_tms_12", "reason": "source host is not on the allowlist"}]
        assert report["unchanged"] == {"bddk": 1, "mevzuat": 0}
        assert report["status"] == "something_to_report"
        assert update_check.exit_code(report) == EXIT_SOMETHING_TO_REPORT
        assert ("HEAD", "/Mevzuat/DokumanGetir/943") in calls
        assert not any(method != "HEAD" and path.startswith("/Mevzuat/DokumanGetir/") for method, path in calls)
        text = format_report(report)
        assert "16 [Banka Kartları] page 51" in text and "1350" in text

    async def test_one_failing_page_does_not_hide_the_others_and_is_an_error(self, doc_store):
        await _corpus(doc_store)
        transport, _ = _site(sizes={"943": 1000, "1349": 2000}, failing_page=58)
        report = await check_updates(doc_store, _client(transport, doc_store), accepted={})

        pages = {page["page"]: page for page in report["pages"]}
        assert pages[58]["error"] == "ConnectError" and pages[51]["entries"] == 1
        assert [item["document_id"] for item in report["new"]] == ["16"]
        assert report["not_listed_complete"] is False
        assert report["status"] == "error"
        assert update_check.exit_code(report) == EXIT_ERROR
        assert "page 58: failed (ConnectError)" in format_report(report)

    async def test_a_source_that_answers_404_is_reported_as_changed(self, doc_store):
        await _corpus(doc_store)
        transport, _ = _site(sizes={"943": 1000})
        report = await check_updates(doc_store, _client(transport, doc_store), accepted={})
        assert {item["document_id"]: item["reason"] for item in report["changed"]}["1349"] == (
            "source answers HTTP 404"
        )

    async def test_an_answer_that_is_not_the_document_is_an_error_not_unchanged(self, doc_store):
        """A maintenance or bot page answers 200 without the text: no shared date, so it cannot count as unchanged."""
        await _corpus(doc_store)
        page = "<html><body><p>Sistem bakımda, lütfen daha sonra tekrar deneyiniz.</p></body></html>"
        transport, _ = _site(sizes={"943": 1000, "1349": 2000}, mevzuat_page=page)
        report = await check_updates(doc_store, _client(transport, doc_store), accepted={})
        assert [item["document_id"] for item in report["check_errors"]] == ["mevzuat_5411"]
        assert report["unchanged"] == {"bddk": 2, "mevzuat": 0}
        assert report["status"] == "error"

    async def test_nothing_to_report(self, doc_store):
        await doc_store.store_document(
            StoredDocument(
                document_id="943",
                title="Rehber",
                source_url="https://www.bddk.org.tr/Mevzuat/DokumanGetir/943",
                markdown_content="Rehber metni.",
                file_size=1000,
            )
        )
        listed = ["943", "mevzuat_5411", "16", "54", "1336", "63", "1349", "1350"]
        await doc_store.replace_catalog([{"document_id": d, "title": d} for d in listed])
        transport, _ = _site(sizes={"943": 1000})
        report = await check_updates(doc_store, _client(transport, doc_store), accepted={})
        assert report["status"] == "nothing_to_report"
        assert update_check.exit_code(report) == EXIT_NOTHING_TO_REPORT
        assert json.loads(json.dumps(report)) == report


def _registry(tmp_path, entries: str) -> dict:
    path = tmp_path / "accepted.yml"
    path.write_text(f"format_version: 1\naccepted:\n{entries}", encoding="utf-8")
    return load_accepted_findings(path)


class TestAcceptedFindings:
    def test_the_shipped_registry_loads_with_the_reviewed_entries(self):
        registry = load_accepted_findings()
        assert set(registry) == {
            ("kgk_tms_12", "not_listed"),
            ("mevzuat_6362", "not_listed"),
            ("rg_32202_20230526_6", "not_listed"),
            ("1350", "new_on_page_55"),
        }
        assert all(entry.reviewed == "2026-11-15" and entry.reason for entry in registry.values())

    @pytest.mark.parametrize(
        ("entries", "message"),
        [
            ("  - {document_id: '1', finding: moved, reason: r, reviewed: 2026-11-15}\n", "finding"),
            ("  - {document_id: '1', finding: new, reviewed: 2026-11-15}\n", "reason"),
            ("  - {document_id: '1', finding: new, reason: r, reviewed: 15.11.2026}\n", "reviewed"),
            ("  - {document_id: '1', finding: new, reason: r, reviewed: 2026-11-15, observed: '9'}\n", "observed"),
            ("  - {document_id: '1', finding: changed, reason: r, reviewed: 2026-11-15}\n", "observed"),
            ("  - {document_id: '1', finding: new, reason: r, reviewed: 2026-11-15, note: x}\n", "unknown"),
            ("  - {document_id: '../1', finding: new, reason: r, reviewed: 2026-11-15}\n", "document_id"),
            (
                "  - {document_id: '1', finding: new, reason: r, reviewed: 2026-11-15}\n"
                "  - {document_id: '1', finding: new, reason: s, reviewed: 2026-11-15}\n",
                "duplicate",
            ),
        ],
    )
    def test_an_invalid_registry_is_refused(self, tmp_path, entries, message):
        with pytest.raises(RuntimeError, match=message):
            _registry(tmp_path, entries)

    async def test_accepted_findings_do_not_count_and_a_vanished_one_is_stale(self, doc_store, tmp_path):
        await _corpus(doc_store)
        transport, _ = _site(sizes={"943": 1000, "1349": 2000})
        registry = _registry(
            tmp_path,
            "  - {document_id: kgk_tms_12, finding: not_listed, reason: by hand, reviewed: 2026-11-15}\n"
            "  - {document_id: '1350', finding: new_on_page_55, reason: not selected, reviewed: 2026-11-15}\n"
            "  - {document_id: '16', finding: new, reason: held back, reviewed: 2026-11-15}\n"
            "  - {document_id: '1336', finding: new, reason: held back, reviewed: 2026-11-15}\n"
            "  - {document_id: '943', finding: not_listed, reason: gone once, reviewed: 2026-11-01}\n",
        )

        report = await check_updates(doc_store, _client(transport, doc_store), accepted=registry)

        assert report["new"] == report["new_on_page_55"] == report["not_listed"] == report["changed"] == []
        assert [(item["document_id"], item["finding"]) for item in report["accepted"]] == [
            ("16", "new"),
            ("1336", "new"),
            ("1350", "new_on_page_55"),
            ("kgk_tms_12", "not_listed"),
        ]
        assert report["accepted"][0]["accepted_reason"] == "held back"
        assert report["stale_acceptances"] == [
            {"document_id": "943", "finding": "not_listed", "reason": "gone once", "reviewed": "2026-11-01"}
        ]
        assert report["status"] == "nothing_to_report"
        assert update_check.exit_code(report) == EXIT_NOTHING_TO_REPORT
        text = format_report(report)
        assert "Accepted (reviewed, not counted): 4" in text and "Stale acceptances" in text

    async def test_a_changed_acceptance_holds_only_for_the_observed_value(self, doc_store, tmp_path):
        await _corpus(doc_store)
        registry = _registry(
            tmp_path,
            "  - {document_id: '1349', finding: changed, observed: '2100', reason: r, reviewed: 2026-11-15}\n"
            "  - {document_id: mevzuat_5411, finding: changed, observed: '2026-06-05', reason: r, reviewed: 2026-11-15}\n",
        )
        same, _ = _site(sizes={"943": 1000, "1349": 2100}, mevzuat_extra="<p>(Ek:RG-5/6/2026-32000)</p>")
        report = await check_updates(doc_store, _client(same, doc_store), accepted=registry)
        assert report["changed"] == [] and report["stale_acceptances"] == []
        assert {item["observed"] for item in report["accepted"]} == {"2100", "2026-06-05"}

        later, _ = _site(
            sizes={"943": 1000, "1349": 2200},
            mevzuat_extra="<p>(Ek:RG-5/6/2026-32000)</p><p>(Değişik:RG-1/9/2026-32100)</p>",
        )
        report = await check_updates(doc_store, _client(later, doc_store), accepted=registry)
        assert {(item["document_id"], item["observed"]) for item in report["changed"]} == {
            ("1349", "2200"),
            ("mevzuat_5411", "2026-06-05, 2026-09-01"),
        }
        assert {item["document_id"] for item in report["stale_acceptances"]} == {"1349", "mevzuat_5411"}
        assert report["status"] == "something_to_report"

    async def test_an_entry_the_run_could_not_evaluate_is_not_called_stale(self, doc_store, tmp_path):
        await _corpus(doc_store)
        transport, _ = _site(sizes={"943": 1000, "1349": 2000}, failing_page=58)
        registry = _registry(
            tmp_path, "  - {document_id: '1336', finding: new, reason: held back, reviewed: 2026-11-15}\n"
        )
        report = await check_updates(doc_store, _client(transport, doc_store), accepted=registry)
        assert report["stale_acceptances"] == []


class TestEntriesToAdd:
    async def test_takes_the_listed_entry_and_refuses_held_unlisted_or_out_of_scope_ids(self, doc_store):
        await _corpus(doc_store)
        transport, _ = _site(sizes={})
        client = _client(transport, doc_store)

        (entry,) = await entries_to_add(doc_store, client, ["16"])
        assert (entry.document_id, entry.category, entry.title) == (
            "16",
            "Banka Kartları",
            "Kredi Kartı Taksit Sınırları ve Yasakları",
        )
        with pytest.raises(RuntimeError, match="already in the catalog: 943"):
            await entries_to_add(doc_store, client, ["943"])
        with pytest.raises(RuntimeError, match="not listed on the catalog pages: 77777"):
            await entries_to_add(doc_store, client, ["77777"])

    async def test_refuses_when_the_page_that_could_list_it_failed(self, doc_store):
        await _corpus(doc_store)
        transport, _ = _site(sizes={}, failing_page=51)
        with pytest.raises(RuntimeError, match=r"pages that could not be read: 51"):
            await entries_to_add(doc_store, _client(transport, doc_store), ["16"])


@pytest.mark.parametrize("value", ["16", "mevzuat_11180"])
def test_add_accepts_bddk_and_mevzuat_ids(value):
    assert cli.build_parser().parse_args(["sync", "--add", value]).add == [value]


@pytest.mark.parametrize("value", ["../16", "16 ", "kgk_tms_12", "mevzuat_", "1" * 21, "16;drop"])
def test_add_rejects_anything_but_a_document_id(value):
    with pytest.raises(SystemExit) as exited:
        cli.build_parser().parse_args(["sync", "--add", value])
    assert exited.value.code == 2


def test_check_updates_opens_the_corpus_read_only_and_leaves_it_unchanged(tmp_path, monkeypatch, capsys):
    path = tmp_path / "c.sqlite"
    db = CorpusDB.open(path)
    store = DocumentStore(db)
    import asyncio

    asyncio.run(_corpus(store))
    db.close()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    transport, _ = _site(sizes={"943": 1000, "1349": 2000})
    opened = []
    real_open = CorpusDB.open

    def spy_open(*args, **kwargs):
        opened.append(kwargs)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(CorpusDB, "open", spy_open)
    monkeypatch.setattr("bddk_mcp.ingest.client.BddkApiClient._assert_public_resolution", AsyncMock(return_value=None))
    real_init = BddkApiClient.__init__

    def init(self, store, **kwargs):
        real_init(self, store, http=httpx.AsyncClient(transport=transport))

    monkeypatch.setattr(BddkApiClient, "__init__", init)

    with pytest.raises(SystemExit) as exited:
        cli.main(["check-updates", "--corpus", str(path), "--json"])

    assert exited.value.code == EXIT_SOMETHING_TO_REPORT
    assert opened == [{"readonly": True}]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    report = json.loads(capsys.readouterr().out)
    assert [item["document_id"] for item in report["new"]] == ["16", "1336"]


def _patched_client(monkeypatch, transport: httpx.MockTransport) -> None:
    monkeypatch.setattr("bddk_mcp.ingest.client.BddkApiClient._assert_public_resolution", AsyncMock(return_value=None))
    real_init = BddkApiClient.__init__

    def init(self, store, **kwargs):
        real_init(self, store, http=httpx.AsyncClient(transport=transport), **kwargs)

    monkeypatch.setattr(BddkApiClient, "__init__", init)


def test_sync_add_inserts_only_the_named_entry_and_syncs_only_it(tmp_path, monkeypatch, capsys):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    path = tmp_path / "c.sqlite"
    db = CorpusDB.open(path)
    asyncio.run(_corpus(DocumentStore(db)))
    before = asyncio.run(DocumentStore(db).load_catalog())
    db.close()
    transport, _ = _site(sizes={})
    _patched_client(monkeypatch, transport)
    syncer = MagicMock()
    syncer.__aenter__ = AsyncMock(return_value=syncer)
    syncer.__aexit__ = AsyncMock(return_value=None)
    syncer.sync_all = AsyncMock(
        return_value=SimpleNamespace(
            total=1, downloaded=1, reindexed=0, skipped=0, failed=0, elapsed_seconds=0.1, errors=[]
        )
    )
    monkeypatch.setattr("bddk_mcp.ingest.doc_sync.DocumentSyncer", MagicMock(return_value=syncer))
    monkeypatch.setattr("bddk_mcp.store.vector_store.VectorStore", MagicMock())

    cli.main(["sync", "--add", "16", "--corpus", str(path)])

    db = CorpusDB.open(path, readonly=True)
    after = asyncio.run(DocumentStore(db).load_catalog())
    db.close()
    assert [row for row in after if row["document_id"] != "16"] == before
    (added,) = [row for row in after if row["document_id"] == "16"]
    assert (added["category"], added["source_url"]) == (
        "Banka Kartları",
        "https://www.bddk.org.tr/Mevzuat/DokumanGetir/16",
    )
    documents = syncer.sync_all.await_args.args[0]
    assert [document["document_id"] for document in documents] == ["16"]
    assert "Added to the catalog: 16" in capsys.readouterr().out


def test_sync_add_refuses_a_catalogued_id_without_writing(tmp_path, monkeypatch, capsys):
    import asyncio

    path = tmp_path / "c.sqlite"
    db = CorpusDB.open(path)
    asyncio.run(_corpus(DocumentStore(db)))
    before = asyncio.run(DocumentStore(db).load_catalog())
    db.close()
    transport, calls = _site(sizes={})
    _patched_client(monkeypatch, transport)

    with pytest.raises(SystemExit) as exited:
        cli.main(["sync", "--add", "16", "--add", "943", "--corpus", str(path)])

    assert exited.value.code == 1
    assert "already in the catalog: 943" in capsys.readouterr().err
    assert calls == []
    db = CorpusDB.open(path, readonly=True)
    assert asyncio.run(DocumentStore(db).load_catalog()) == before
    db.close()


def test_sync_add_cannot_be_combined_with_a_refresh_or_one_document(capsys):
    for extra in (["--refresh-catalog"], ["--doc-id", "943"]):
        with pytest.raises(SystemExit) as exited:
            cli.main(["sync", "--add", "16", *extra])
        assert exited.value.code == 1
        assert "--add cannot be combined" in capsys.readouterr().err
