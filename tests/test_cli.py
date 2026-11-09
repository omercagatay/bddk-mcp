"""Tests for the packaged ``bddk-mcp`` command-line entry point."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import AsyncMock, patch

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

    sync = parser.parse_args(["sync", "--doc-id", "943", "--force", "--concurrency", "2"])
    assert (sync.command, sync.doc_id, sync.force, sync.concurrency) == ("sync", "943", True, 2)
    assert parser.parse_args(["sync"]).concurrency == 5


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


def test_sync_command_delegates_to_the_document_syncer():
    with patch("bddk_mcp.ingest.doc_sync._cli_sync", new=AsyncMock()) as sync:
        cli.main(["sync", "--doc-id", "943"])

    args = sync.await_args.args[0]
    assert (args.doc_id, args.force, args.concurrency, args.db) == ("943", False, 5, None)


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
