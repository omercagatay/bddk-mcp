"""Tests for the executable expert-evaluation release preflight."""

from __future__ import annotations

import json

import benchmark.release_preflight as release_preflight
from benchmark.release_preflight import main


def test_tracked_draft_fails_preflight_without_exposing_case_content(capsys) -> None:
    result = main(["--now", "2026-07-16T00:00:00+00:00"])

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert result == 1
    assert captured.out == ""
    assert payload == {
        "schema_version": 1,
        "status": "release_preflight_failed",
        "error_code": "EXPERT_EVALUATION_NOT_RELEASE_READY",
        "model_scores_authorized": False,
    }
    assert "Bugün itibarıyla" not in captured.err
    assert "markdown_content" not in captured.err


def test_unexpected_preflight_failure_is_path_and_content_free(capsys, monkeypatch) -> None:
    def fail_without_leaking(*args, **kwargs):
        raise OSError("/private/bank/source.pdf: secret page text")

    monkeypatch.setattr(release_preflight, "run_release_preflight", fail_without_leaking)

    result = main([])

    captured = capsys.readouterr()
    assert result == 3
    assert captured.out == ""
    assert json.loads(captured.err) == {
        "schema_version": 1,
        "status": "release_preflight_failed",
        "error_code": "RELEASE_PREFLIGHT_INTERNAL_ERROR",
        "model_scores_authorized": False,
    }
    assert "/private/bank/source.pdf" not in captured.err
    assert "secret page text" not in captured.err
