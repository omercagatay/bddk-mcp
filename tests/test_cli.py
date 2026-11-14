"""Tests for the packaged ``bddk-mcp`` command-line entry point."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bddk_mcp import __version__, cli


def test_parser_exposes_explicit_runtime_commands():
    parser = cli.build_parser()

    serve = parser.parse_args(["serve", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8443"])
    assert (serve.command, serve.transport, serve.host, serve.port) == ("serve", "streamable-http", "0.0.0.0", 8443)

    build = parser.parse_args(["build", "--corpus", "c.sqlite", "--seed-dir", "seed", "--force"])
    assert (build.command, build.corpus, build.seed_dir, build.force) == ("build", "c.sqlite", "seed", True)
    assert parser.parse_args(["build"]).corpus is None

    export = parser.parse_args(["export", "--corpus", "c.sqlite"])
    assert (export.command, export.corpus, export.seed_dir) == ("export", "c.sqlite", None)

    sync = parser.parse_args(["sync", "--doc-id", "943", "--force", "--concurrency", "2", "--refresh-catalog"])
    assert (sync.command, sync.doc_id, sync.force, sync.concurrency) == ("sync", "943", True, 2)
    assert sync.refresh_catalog is True
    defaults = parser.parse_args(["sync"])
    assert (defaults.concurrency, defaults.refresh_catalog, defaults.doc_id, defaults.corpus) == (5, False, None, None)

    info = parser.parse_args(["info", "--corpus", "c.sqlite"])
    assert (info.command, info.corpus) == ("info", "c.sqlite")
    assert parser.parse_args(["info"]).corpus is None


def test_version_does_not_import_runtime(capsys):
    result = subprocess.run(
        [sys.executable, "-c", "import sys; from bddk_mcp import cli; cli.main(['--version'])"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "bddk-mcp" in result.stdout


def test_build_command_reports_counts_and_skips(capsys):
    built = {"decision_cache": 1, "documents": 2, "sections": 3, "chunks": 4, "chunks_cached": 3, "skipped": False}
    with patch("bddk_mcp.ingest.seed.build_corpus", new=AsyncMock(return_value=built)) as build:
        cli.main(["build", "--corpus", "c.sqlite", "--force"])
    build.assert_awaited_once_with(corpus_path="c.sqlite", seed_dir=None, force=True)
    assert "2 documents, 3 sections, 4 chunks (1 embedded, 3 from the seed cache)" in capsys.readouterr().out

    skipped = dict(built, skipped=True)
    with patch("bddk_mcp.ingest.seed.build_corpus", new=AsyncMock(return_value=skipped)):
        cli.main(["build"])
    assert "already matches" in capsys.readouterr().out


def test_export_command_reports_counts(capsys):
    with patch("bddk_mcp.ingest.seed.export_corpus", new=AsyncMock(return_value=(319, 300, 318))) as export:
        cli.main(["export", "--seed-dir", "out"])
    export.assert_awaited_once_with(corpus_path=None, seed_dir="out")
    assert "Exported 319 documents, 300 catalog entries and embeddings for 318 documents." in capsys.readouterr().out


def test_runtime_errors_print_one_line_and_exit_1(capsys):
    with (
        patch("bddk_mcp.ingest.seed.build_corpus", new=AsyncMock(side_effect=RuntimeError("corpus is not ready"))),
        pytest.raises(SystemExit) as exited,
    ):
        cli.main(["build"])

    assert exited.value.code == 1
    err = capsys.readouterr().err
    assert err == "bddk-mcp build: error: corpus is not ready\n"


def test_build_of_an_empty_or_missing_seed_fails(tmp_path, capsys):
    with pytest.raises(SystemExit) as exited:
        cli.main(["build", "--seed-dir", str(tmp_path / "typo"), "--corpus", str(tmp_path / "c.sqlite")])

    assert exited.value.code == 1
    assert f"no seed data in {tmp_path / 'typo'} " in capsys.readouterr().err
    assert not (tmp_path / "c.sqlite").exists()


def test_build_without_seed_dir_names_the_resolved_default_seed_directory(tmp_path, monkeypatch, capsys):
    from bddk_mcp.ingest import seed

    monkeypatch.setattr(seed, "SEED_DIR", tmp_path / "packaged" / "seed_data")
    with pytest.raises(SystemExit) as exited:
        cli.main(["build", "--corpus", str(tmp_path / "c.sqlite")])

    assert exited.value.code == 1
    assert f"no seed data in {tmp_path / 'packaged' / 'seed_data'} " in capsys.readouterr().err


@pytest.mark.parametrize("value", ["0", "-1", "x"])
def test_concurrency_must_be_a_positive_integer(value):
    with pytest.raises(SystemExit) as exited:
        cli.build_parser().parse_args(["sync", "--concurrency", value])
    assert exited.value.code == 2


def test_serve_applies_overrides_before_import(monkeypatch):
    monkeypatch.delenv("MCP_TRANSPORT", raising=False)
    monkeypatch.delenv("MCP_HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    observed: dict[str, str | None] = {}

    def fake_main() -> None:
        import os

        observed.update(
            transport=os.environ.get("MCP_TRANSPORT"),
            host=os.environ.get("MCP_HOST"),
            port=os.environ.get("PORT"),
        )

    with patch("bddk_mcp.server.main", new=fake_main):
        cli.main(["serve", "--transport", "streamable-http", "--host", "127.0.0.1", "--port", "8123"])

    assert observed == {"transport": "streamable-http", "host": "127.0.0.1", "port": "8123"}


def test_port_must_be_in_tcp_range():
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--port", "70000"])


class _Catalog:
    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        self.title = f"Title {document_id}"
        self.category = "Rehber"
        self.source_url = f"https://www.bddk.org.tr/Mevzuat/DokumanGetir/{document_id}"
        self.decision_date = ""
        self.decision_number = ""


def _sync_doubles(catalog, failed: int = 0):
    db = MagicMock()
    client = MagicMock()
    client.refresh_cache = AsyncMock(return_value=len(catalog))
    client.ensure_cache = AsyncMock()
    client.get_cache_items.return_value = catalog
    client.close = AsyncMock()
    syncer = MagicMock()
    syncer.__aenter__ = AsyncMock(return_value=syncer)
    syncer.__aexit__ = AsyncMock(return_value=None)
    syncer.sync_all = AsyncMock(
        return_value=SimpleNamespace(
            total=len(catalog),
            downloaded=1,
            reindexed=1,
            skipped=0,
            failed=failed,
            elapsed_seconds=0.1,
            errors=[],
        )
    )
    patches = (
        patch("bddk_mcp.store.corpus_db.CorpusDB.open", return_value=db),
        patch("bddk_mcp.ingest.client.BddkApiClient", return_value=client),
        patch("bddk_mcp.ingest.doc_sync.DocumentSyncer", return_value=syncer),
        patch("bddk_mcp.store.doc_store.DocumentStore"),
        patch("bddk_mcp.store.vector_store.VectorStore"),
    )
    return db, client, syncer, patches


@pytest.mark.asyncio
async def test_sync_reads_the_catalog_from_the_corpus_and_syncs_every_document(capsys):
    catalog = [_Catalog("943"), _Catalog("1296")]
    db, client, syncer, patches = _sync_doubles(catalog)
    args = cli.build_parser().parse_args(["sync", "--concurrency", "2", "--corpus", "c.sqlite"])

    with ExitStack() as stack:
        opener = stack.enter_context(patches[0])
        for item in patches[1:]:
            stack.enter_context(item)
        await cli._sync(args)

    opener.assert_called_once_with("c.sqlite")
    client.ensure_cache.assert_awaited_once_with()
    client.refresh_cache.assert_not_awaited()
    documents = syncer.sync_all.await_args.args[0]
    assert [document["document_id"] for document in documents] == ["943", "1296"]
    assert documents[0]["source_url"] == catalog[0].source_url
    assert syncer.sync_all.await_args.kwargs == {"concurrency": 2, "force": False}
    client.close.assert_awaited_once_with()
    db.close.assert_called_once_with()
    assert "Sync: 2 total, 1 downloaded, 1 reindexed, 0 skipped, 0 failed" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_sync_can_refresh_the_catalog_and_target_one_document(capsys):
    catalog = [_Catalog("943"), _Catalog("1296")]
    _db, client, syncer, patches = _sync_doubles(catalog)
    args = cli.build_parser().parse_args(["sync", "--refresh-catalog", "--doc-id", "1296", "--force"])

    with ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        await cli._sync(args)

    client.refresh_cache.assert_awaited_once_with()
    client.ensure_cache.assert_not_awaited()
    documents = syncer.sync_all.await_args.args[0]
    assert [document["document_id"] for document in documents] == ["1296"]
    assert syncer.sync_all.await_args.kwargs == {"concurrency": 5, "force": True}
    assert "Catalog refreshed from BDDK: 2 items" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_sync_refuses_an_uncatalogued_document_id():
    db, client, _syncer, patches = _sync_doubles([_Catalog("943")])
    args = cli.build_parser().parse_args(["sync", "--doc-id", "99999"])

    with ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        with pytest.raises(RuntimeError, match="not in the catalog"):
            await cli._sync(args)

    client.close.assert_awaited_once_with()
    db.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_sync_exits_1_when_a_document_failed(capsys):
    _db, _client, _syncer, patches = _sync_doubles([_Catalog("943")], failed=1)
    args = cli.build_parser().parse_args(["sync"])

    with ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        with pytest.raises(SystemExit) as exited:
            await cli._sync(args)

    assert exited.value.code == 1
    assert "1 failed" in capsys.readouterr().out


_INFO_CONTENT = "MADDE 1 - Amaç\nBankalar karşılık ayırır.\n"


def _ready_corpus(tmp_path) -> Path:
    from bddk_mcp.ingest import seed
    from bddk_mcp.store.vector_store import VectorStore
    from tests.conftest import FakeTokenizer, fake_embed

    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    content = _INFO_CONTENT
    document = {
        "document_id": "mevzuat_1",
        "title": "Yönetmelik",
        "markdown_content": content,
        "content_hash": hashlib.sha256(content.encode()).hexdigest(),
    }
    (seed_dir / "documents.json").write_text(json.dumps([document], ensure_ascii=False), encoding="utf-8")
    (seed_dir / "decision_cache.json").write_text(
        json.dumps([{"document_id": "mevzuat_1", "title": "Yönetmelik"}], ensure_ascii=False), encoding="utf-8"
    )
    corpus = tmp_path / "c.sqlite"
    asyncio.run(
        seed.build_corpus(
            corpus_path=corpus,
            seed_dir=seed_dir,
            vector_store_factory=lambda db: VectorStore(db, embed_fn=fake_embed, tokenizer=FakeTokenizer()),
        )
    )
    return corpus


def test_info_describes_a_ready_corpus_read_only(tmp_path, capsys):
    from bddk_mcp.core.config import EMBEDDING_MODEL_NAME
    from bddk_mcp.store.corpus_db import SCHEMA_VERSION, corpus_digest

    corpus = _ready_corpus(tmp_path)
    before = corpus.read_bytes()
    capsys.readouterr()

    cli.main(["info", "--corpus", str(corpus)])

    out = capsys.readouterr().out
    expected_digest = corpus_digest([("mevzuat_1", hashlib.sha256(_INFO_CONTENT.encode()).hexdigest())])
    lines = out.splitlines()
    assert lines[0] == f"bddk-mcp version: {__version__}"
    assert lines[1] == f"corpus digest: {expected_digest}"
    assert "documents: 1" in lines and "catalog entries: 1" in lines
    assert any(line.startswith("chunks: ") and int(line.split(": ")[1]) >= 1 for line in lines)
    assert f"embedding model: {EMBEDDING_MODEL_NAME}" in lines
    assert f"schema version: {SCHEMA_VERSION}" in lines
    assert f"file size: {corpus.stat().st_size} bytes" in lines
    assert "journal mode: delete" in lines
    assert lines[-1] == "status: ready"
    assert str(tmp_path) not in out
    assert corpus.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["c.sqlite", "seed"]


def test_info_lists_the_issues_and_exits_1_for_a_corpus_that_is_not_ready(tmp_path, capsys):
    from bddk_mcp.store.corpus_db import CorpusDB

    corpus = tmp_path / "empty.sqlite"
    CorpusDB.open(corpus).close()

    with pytest.raises(SystemExit) as exited:
        cli.main(["info", "--corpus", str(corpus)])

    assert exited.value.code == 1
    out = capsys.readouterr().out
    assert "status: not ready" in out
    assert "  issue: the corpus holds no retrieval chunks" in out.splitlines()


def test_info_of_a_missing_or_foreign_file_prints_one_line_and_exits_1(tmp_path, capsys):
    with pytest.raises(SystemExit) as exited:
        cli.main(["info", "--corpus", str(tmp_path / "missing.sqlite")])
    assert exited.value.code == 1
    assert capsys.readouterr().err.startswith("bddk-mcp info: error: Corpus file not found")

    foreign = tmp_path / "foreign.sqlite"
    foreign.write_bytes(b"not a database, " * 64)
    with pytest.raises(SystemExit) as exited:
        cli.main(["info", "--corpus", str(foreign)])
    assert exited.value.code == 1
    assert capsys.readouterr().err.startswith("bddk-mcp info: error: Corpus is not ready: the file is not a readable")


def test_info_refuses_a_stale_wal_beside_a_replaced_file_in_one_line(tmp_path, capsys):
    from tests.test_corpus_db import _replaced_after_a_killed_writer

    corpus = _replaced_after_a_killed_writer(tmp_path)
    before = corpus.read_bytes()

    with pytest.raises(SystemExit) as exited:
        cli.main(["info", "--corpus", str(corpus)])

    assert exited.value.code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("bddk-mcp info: error: ")
    assert "bddk.sqlite-wal" in captured.err and "Traceback" not in captured.err
    assert len(captured.err.strip().splitlines()) == 1
    assert corpus.read_bytes() == before


def test_info_of_a_file_it_may_not_open_prints_the_permission_message_without_a_traceback(
    tmp_path, capsys, monkeypatch
):
    import sqlite3

    from bddk_mcp.store.corpus_db import UNREADABLE_MESSAGE, CorpusDB
    from tests.test_corpus_db import _refusing_connect

    corpus = tmp_path / "c.sqlite"
    CorpusDB.open(corpus).close()
    monkeypatch.setattr(sqlite3, "connect", _refusing_connect(sqlite3.connect))

    with pytest.raises(SystemExit) as exited:
        cli.main(["info", "--corpus", str(corpus)])

    assert exited.value.code == 1
    assert capsys.readouterr().err == f"bddk-mcp info: error: {UNREADABLE_MESSAGE}\n"


def test_info_reports_sync_failures_by_category(tmp_path, capsys):
    from bddk_mcp.store.corpus_db import CorpusDB
    from bddk_mcp.store.doc_store import DocumentStore

    corpus = _ready_corpus(tmp_path)
    cli.main(["info", "--corpus", str(corpus)])
    assert "sync failures: none" in capsys.readouterr().out.splitlines()

    db = CorpusDB.open(corpus)
    store = DocumentStore(db)
    for doc_id, category in (("a", "download"), ("b", "download"), ("c", "extraction")):
        asyncio.run(store.record_sync_failure(doc_id, "timeout", category))
    db.close()

    cli.main(["info", "--corpus", str(corpus)])
    lines = capsys.readouterr().out.splitlines()
    assert "sync failures: 3 (download: 2, extraction: 1)" in lines
    assert lines[-1] == "status: ready"


def _set_fingerprint(corpus, change) -> None:
    from bddk_mcp.store import fingerprint
    from bddk_mcp.store.corpus_db import CorpusDB

    db = CorpusDB.open(corpus)

    def write(conn):
        key = fingerprint.FINGERPRINT_META_KEY
        stored = fingerprint.decode(conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()[0])
        updated = change(stored)
        if updated is None:
            conn.execute("DELETE FROM meta WHERE key = ?", (key,))
        else:
            conn.execute("UPDATE meta SET value = ? WHERE key = ?", (fingerprint.encode(updated), key))

    db.transaction_sync(write)
    db.close()


def test_info_reports_a_current_stale_or_absent_build_fingerprint_without_changing_the_exit_code(tmp_path, capsys):
    corpus = _ready_corpus(tmp_path)
    capsys.readouterr()
    cli.main(["info", "--corpus", str(corpus)])
    lines = capsys.readouterr().out.splitlines()
    assert "build fingerprint: current" in lines and lines[-1] == "status: ready"

    _set_fingerprint(corpus, lambda stored: {**stored, "parser": "1" * 16, "chunker": "2" * 16, "seed": "3" * 16})
    cli.main(["info", "--corpus", str(corpus)])
    lines = capsys.readouterr().out.splitlines()
    assert "build fingerprint: stale (parser, chunker) — run bddk-mcp build" in lines
    assert lines[-1] == "status: ready"

    # A model-revision difference cannot be fixed by a plain build (build refuses it), so info says so.
    _set_fingerprint(corpus, lambda stored: {**stored, "model": "4" * 16})
    cli.main(["info", "--corpus", str(corpus)])
    lines = capsys.readouterr().out.splitlines()
    expected = (
        "build fingerprint: stale (parser, chunker, model) — delete the corpus file and run bddk-mcp build --force"
    )
    assert expected in lines
    assert lines[-1] == "status: ready"

    _set_fingerprint(corpus, lambda stored: None)
    cli.main(["info", "--corpus", str(corpus)])
    lines = capsys.readouterr().out.splitlines()
    assert "build fingerprint: absent (file predates fingerprinting) — run bddk-mcp build" in lines
    assert lines[-1] == "status: ready"
