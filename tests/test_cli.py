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

    migrate = parser.parse_args(["migrate", "--db", "postgresql://example", "--allow-retrieval-publication-backfill"])
    assert migrate.command == "migrate"
    assert migrate.db == "postgresql://example"
    assert migrate.allow_retrieval_publication_backfill is True

    sync = parser.parse_args(["sync", "--doc-id", "943", "--force", "--concurrency", "2", "--refresh-catalog"])
    assert (sync.command, sync.doc_id, sync.force, sync.concurrency) == ("sync", "943", True, 2)
    assert sync.refresh_catalog is True
    defaults = parser.parse_args(["sync"])
    assert (defaults.concurrency, defaults.refresh_catalog, defaults.doc_id) == (5, False, None)


def test_version_does_not_import_runtime(capsys):
    result = subprocess.run(
        [sys.executable, "-c", "import sys; from bddk_mcp import cli; cli.main(['--version'])"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "bddk-mcp" in result.stdout


def test_migrate_command_forwards_only_explicit_maintenance_approvals(capsys):
    with patch("bddk_mcp.cli._migrate", new=AsyncMock()) as migrate:
        cli.main(["migrate", "--db", "postgresql://example"])
        cli.main(["migrate", "--allow-retrieval-publication-backfill"])

    assert [call.args[0] for call in migrate.await_args_list] == ["postgresql://example", None]
    assert [call.kwargs["allow_retrieval_publication_backfill"] for call in migrate.await_args_list] == [False, True]
    assert capsys.readouterr().out.count("Database schema is ready.") == 2


def test_runtime_errors_become_clean_parser_errors(capsys):
    with (
        patch("bddk_mcp.cli._migrate", new=AsyncMock(side_effect=RuntimeError("database is not ready"))),
        pytest.raises(SystemExit),
    ):
        cli.main(["migrate"])

    assert "database is not ready" in capsys.readouterr().err


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
    pool = MagicMock()
    pool.close = AsyncMock()
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
        patch("asyncpg.create_pool", new=AsyncMock(return_value=pool)),
        patch("bddk_mcp.core.config.require_database_url", return_value="postgresql://ingestion"),
        patch("bddk_mcp.db_lifecycle.assert_database_ready", new=AsyncMock()),
        patch("bddk_mcp.ingest.client.BddkApiClient", return_value=client),
        patch("bddk_mcp.ingest.doc_sync.DocumentSyncer", return_value=syncer),
        patch("bddk_mcp.store.doc_store.DocumentStore"),
        patch("bddk_mcp.store.vector_store.VectorStore"),
    )
    return pool, client, syncer, patches


@pytest.mark.asyncio
async def test_sync_reads_the_catalog_from_the_database_and_syncs_every_document(capsys):
    catalog = [_Catalog("943"), _Catalog("1296")]
    pool, client, syncer, patches = _sync_doubles(catalog)
    args = cli.build_parser().parse_args(["sync", "--concurrency", "2"])

    with ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        await cli._sync(args)

    client.ensure_cache.assert_awaited_once_with()
    client.refresh_cache.assert_not_awaited()
    documents = syncer.sync_all.await_args.args[0]
    assert [document["document_id"] for document in documents] == ["943", "1296"]
    assert documents[0]["source_url"] == catalog[0].source_url
    assert syncer.sync_all.await_args.kwargs == {"concurrency": 2, "force": False}
    client.close.assert_awaited_once_with()
    pool.close.assert_awaited_once_with()
    assert "Sync: 2 total" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_sync_can_refresh_the_catalog_and_target_one_document(capsys):
    catalog = [_Catalog("943"), _Catalog("1296")]
    _pool, client, syncer, patches = _sync_doubles(catalog)
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
    pool, client, _syncer, patches = _sync_doubles([_Catalog("943")])
    args = cli.build_parser().parse_args(["sync", "--doc-id", "99999"])

    with ExitStack() as stack:
        for item in patches:
            stack.enter_context(item)
        with pytest.raises(RuntimeError, match="not in the catalog"):
            await cli._sync(args)

    client.close.assert_awaited_once_with()
    pool.close.assert_awaited_once_with()
