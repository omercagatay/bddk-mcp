"""Tests for the packaged bddk-mcp command surface."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from bddk_mcp import __version__, cli


def test_parser_exposes_explicit_runtime_commands():
    parser = cli.build_parser()

    assert parser.parse_args(["serve", "--transport", "stdio"]).command == "serve"
    assert parser.parse_args(["serve", "--profile", "operator"]).profile == "operator"
    assert parser.parse_args(["migrate"]).command == "migrate"
    assert parser.parse_args(["migrate"]).allow_retrieval_publication_backfill is False
    assert (
        parser.parse_args(["migrate", "--allow-retrieval-publication-backfill"]).allow_retrieval_publication_backfill
        is True
    )
    assert parser.parse_args(["bootstrap", "--seed-dir", "/corpus"]).seed_dir == Path("/corpus")
    assert parser.parse_args(["bootstrap", "--reindex-existing"]).reindex_existing is True


def test_version_does_not_import_runtime(capsys):
    with pytest.raises(SystemExit, match="0"):
        cli.main(["--version"])

    assert capsys.readouterr().out.strip() == f"bddk-mcp {__version__}"


def test_migrate_command_forwards_only_explicit_maintenance_approvals(capsys):
    with (
        patch("bddk_mcp.cli._migrate") as migrate,
        patch("bddk_mcp.cli.asyncio.run") as run,
    ):
        cli.main(["migrate", "--adopt-legacy", "--allow-retrieval-publication-backfill"])

    migrate.assert_called_once_with(
        None,
        adopt_legacy=True,
        allow_retrieval_publication_backfill=True,
    )
    run.assert_called_once()
    run.call_args.args[0].close()
    assert capsys.readouterr().out.strip() == "Database schema is ready."


def test_serve_applies_overrides_before_import():
    args = Namespace(profile="operator", transport="streamable-http", host="127.0.0.2", port=8123)

    with (
        patch.dict(cli.os.environ, {}, clear=False),
        patch("bddk_mcp.server.main") as server_main,
    ):
        cli._run_serve(args)

        server_main.assert_called_once_with()
        assert cli.os.environ["MCP_TRANSPORT"] == "streamable-http"
        assert cli.os.environ["BDDK_TOOL_PROFILE"] == "operator"
        assert cli.os.environ["MCP_HOST"] == "127.0.0.2"
        assert cli.os.environ["PORT"] == "8123"


def test_port_must_be_in_tcp_range():
    parser = cli.build_parser()

    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(["serve", "--port", "0"])
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(["serve", "--port", "65536"])
