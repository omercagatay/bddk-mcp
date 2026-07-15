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
    bootstrap = parser.parse_args(
        [
            "bootstrap",
            "--reindex-existing",
            "--require-quantified-freshness",
            "--require-measured-freshness",
            "--require-verified-signature",
            "--trusted-signing-key",
            "/trust/corpus.pem",
        ]
    )
    assert bootstrap.reindex_existing is True
    assert bootstrap.require_quantified_freshness is True
    assert bootstrap.require_measured_freshness is True
    assert bootstrap.require_verified_signature is True
    assert bootstrap.trusted_signing_key == Path("/trust/corpus.pem")
    verify = parser.parse_args(
        [
            "verify-corpus",
            "--require-quantified-freshness",
            "--require-measured-freshness",
            "--require-verified-signature",
        ]
    )
    assert verify.command == "verify-corpus"
    assert verify.require_quantified_freshness is True
    assert verify.require_measured_freshness is True
    assert verify.require_verified_signature is True
    assert verify.trusted_signing_key is None


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


def test_bootstrap_reports_the_path_free_manifest_identity(capsys):
    result = {
        "decision_cache": 1,
        "documents": 2,
        "sections": 3,
        "chunks": 4,
        "embedded": 4,
        "reindex_published": 0,
        "corpus_manifest_id": "reviewed-corpus-v1",
        "corpus_manifest_sha256": "a" * 64,
    }
    with (
        patch("bddk_mcp.cli._bootstrap") as bootstrap,
        patch("bddk_mcp.cli.asyncio.run", return_value=result) as run,
    ):
        cli.main(["bootstrap"])

    bootstrap.assert_called_once_with(
        None,
        None,
        False,
        reindex_existing=False,
        require_quantified_freshness=False,
        require_measured_freshness=False,
        require_verified_signature=False,
        trusted_signing_key=None,
    )
    run.assert_called_once()
    run.call_args.args[0].close()
    output = capsys.readouterr().out
    assert "Corpus manifest used: id=reviewed-corpus-v1" in output
    assert "sha256=" + "a" * 64 in output
    assert "seed_data" not in output


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


def test_verify_corpus_checks_tracked_artifacts_and_reports_safe_identity(capsys):
    cli.main(["verify-corpus"])

    output = capsys.readouterr().out
    assert "id=bddk-job-corpus-2026-07-15" in output
    assert "artifacts=3 exhaustive=false" in output
    assert "WARNING: This corpus is a job-specific selection" in output
    assert "markdown_content" not in output


def test_verify_corpus_production_requirements_fail_until_owner_policies_exist(capsys):
    with pytest.raises(SystemExit, match="2"):
        cli.main(["verify-corpus", "--require-quantified-freshness"])

    assert "freshness objectives are not quantified" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="2"):
        cli.main(["verify-corpus", "--require-measured-freshness"])

    assert "SLO compliance is not measured" in capsys.readouterr().err
