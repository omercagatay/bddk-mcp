"""Tests for the packaged ``bddk-mcp`` command-line entry point."""

from __future__ import annotations

import subprocess
import sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bddk_mcp import cli


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


def test_runtime_errors_become_clean_parser_errors(capsys):
    with (
        patch("bddk_mcp.ingest.seed.build_corpus", new=AsyncMock(side_effect=RuntimeError("corpus is not ready"))),
        pytest.raises(SystemExit),
    ):
        cli.main(["build"])

    assert "corpus is not ready" in capsys.readouterr().err


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


def _sync_doubles(catalog):
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
            total=len(catalog), downloaded=1, skipped=0, failed=0, elapsed_seconds=0.1, errors=[]
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
    assert "Sync: 2 total" in capsys.readouterr().out


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
